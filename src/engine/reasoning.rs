// src/engine/reasoning.rs — `--reasoning-budget`：限制思考长度（P6.2）
//
// 蓝本：llama.cpp `common/reasoning-budget.cpp` 的状态机
//   Idle → Counting(remaining=N) → WaitingUtf8 → Forcing → Done
//
// 语义：思考模式下，模型会先写一段推理再输出思考结束标签（本模型是 `</think>`）。
// 这段推理可能很长 —— 实测同一个"写 2000 字恐怖小说"的 prompt，模型写了 **2232 个
// token 的英文规划**才进入正文（而正文本身只有 1259 token）。于是 `-n` 的预算基本
// 被推理吃掉，用户看到的就是"没写出东西"或者被截断。
//
// `--reasoning-budget N` 的做法与 llama.cpp 一致：**不注入 prompt**，而是在推理写满
// N 个 token 后，用"强制下一个 token"的方式把思考结束标签（必要时先补一段提示语）
// 输出出去，让模型立刻转入正文。
//
// 与 llama.cpp 的两点差异（都是简化，语义不变）：
//   - 它把强制实现成 logits 掩码再走采样器；我们直接覆盖这一步的采样结果。
//     被强制的序列是确定的一小段 token，两者结果相同。
//   - 它需要 `WaitingUtf8` 状态避免把多字节字符截断 —— 那是为"强制提示语是任意文本"
//     准备的；我们按**整 token** 强制（提示语整体 tokenize 后再逐个强制），
//     不可能出现半个字符，所以不需要这个状态。

use std::collections::VecDeque;

/// 思考预算的配置（由 CLI / server 组装，token id 从模板推导，不硬编码）。
#[derive(Clone, Debug, Default)]
pub struct ReasoningConfig {
    /// 思考模式是否开启（= 渲染 prompt 时的 `enable_thinking`）。关掉时预算无意义。
    pub enabled: bool,
    /// 预算：允许的思考 token 数。
    /// `< 0` = 不限制（默认，保持不干预）；`0` = 一个思考 token 都不给、立刻结束思考。
    pub budget: i64,
    /// 思考结束标签的 token 序列（本模型 `</think>` = 单个 token）。
    /// 由调用方从**模板源码**里的标签推导（见 `main.rs` 的 helper），空 = 无法强制。
    pub end_ids: Vec<u32>,
    /// 强制结束前先输出的提示语（`--reasoning-budget-message`），可为空。
    pub message_ids: Vec<u32>,
}

/// 思考预算状态机。
#[derive(Clone, Debug)]
pub struct ReasoningBudget {
    cfg: ReasoningConfig,
    /// 还能思考多少 token
    remaining: usize,
    /// 待强制的 token（提示语 ++ 结束标签），只在 `remaining == 0` 后使用
    pending: VecDeque<u32>,
    /// 计数阶段：模型自己已经在输出结束标签的第几个 token
    matched: usize,
    /// 已经结束（模型自己收了，或我们强制完了）
    done: bool,
}

impl ReasoningBudget {
    pub fn new(cfg: ReasoningConfig) -> Self {
        // 只有在"思考开着 + 确实要限制（budget >= 0）+ 拿得到结束标签"时才生效。
        // budget < 0 = 不限制（默认）。
        let active = cfg.enabled && cfg.budget >= 0 && !cfg.end_ids.is_empty();
        let remaining = if active { cfg.budget as usize } else { 0 };
        let mut pending = VecDeque::new();
        if active && remaining == 0 {
            // 预算为 0：一个思考 token 都不给
            pending.extend(cfg.message_ids.iter().copied());
            pending.extend(cfg.end_ids.iter().copied());
        }
        Self {
            cfg: ReasoningConfig { budget: if active { cfg.budget } else { -1 }, ..cfg },
            remaining,
            pending,
            matched: 0,
            done: !active,
        }
    }

    /// 是否在强制输出（诊断用）。
    pub fn is_forcing(&self) -> bool {
        !self.done && self.pending.front().is_some() && self.remaining == 0
    }

    /// 按**本次请求**的 `max_tokens` 压预算：保证正文至少还有 `reserve` 个 token 的余地。
    ///
    /// 为什么需要：预算是**启动时**定的（`--reasoning-budget`），而 `max_tokens` 是
    /// **每请求**的。两者一旦撞上（例如客户端给 2048、预算也是 2048，或预算不限制），
    /// 思考会正好吃光预算 → `finish_reason=length` 且 `content` 为空，用户看到的就是
    /// 「回答了但什么都没有」。这不是模型的问题，是预算分配的问题。
    ///
    /// 只在「思考开着 + 拿得到结束标签」时生效；`--reasoning-budget` 显式给的更小值优先
    /// （只会往下压，不会放大）。
    pub fn ensure_answer_room(&mut self, max_tokens: usize, reserve: usize) {
        if !self.cfg.enabled || self.cfg.end_ids.is_empty() {
            return;
        }
        let ceil = max_tokens.saturating_sub(reserve);
        // `done` 且不限预算 → remaining 是 0，要重新激活；已经更小就保持原样
        if self.done && self.cfg.budget < 0 {
            self.done = false;
            self.remaining = ceil;
        } else if self.remaining > ceil {
            self.remaining = ceil;
        } else {
            return;
        }
        self.pending.clear();
        self.matched = 0;
        if self.remaining == 0 {
            // 预算被压到 0：立刻强制「提示语 + 结束标签」，把剩下的 token 全留给正文
            self.pending.extend(self.cfg.message_ids.iter().copied());
            self.pending.extend(self.cfg.end_ids.iter().copied());
        }
    }

    /// 取样前调用：返回本步**必须**输出的 token（`None` = 正常采样）。
    pub fn next_forced(&self) -> Option<u32> {
        if self.done || self.remaining > 0 {
            return None;
        }
        self.pending.front().copied()
    }

    /// 生成一个 token 之后调用，推进状态。
    pub fn accept(&mut self, tok: u32) {
        if self.done {
            return;
        }
        // 强制阶段：按顺序吃掉强制序列
        if self.remaining == 0 {
            debug_assert_eq!(self.pending.front(), Some(&tok), "强制序列与实际输出不一致");
            if self.pending.front() == Some(&tok) {
                self.pending.pop_front();
            } else {
                // 调用方没有按强制序列输出（不应发生）：直接结束，避免卡死
                self.pending.clear();
            }
            if self.pending.is_empty() {
                self.done = true;
            }
            return;
        }

        // 计数阶段：模型可能自己提前结束思考（结束标签可能是多 token）
        if self.cfg.end_ids.get(self.matched) == Some(&tok) {
            self.matched += 1;
            if self.matched == self.cfg.end_ids.len() {
                self.done = true;
                return;
            }
        } else {
            self.matched = usize::from(self.cfg.end_ids.first() == Some(&tok));
        }

        self.remaining -= 1;
        if self.remaining == 0 && !self.cfg.message_ids.is_empty() {
            // 先补提示语，再补结束标签
            self.pending.extend(self.cfg.message_ids.iter().copied());
            self.pending.extend(self.cfg.end_ids.iter().copied());
        } else if self.remaining == 0 {
            self.pending.extend(self.cfg.end_ids.iter().copied());
        }
    }

    /// 换一段新序列时重置（`Engine::generate` 开头调用）。
    pub fn restart(&mut self) {
        *self = Self::new(self.cfg.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(enabled: bool, budget: i64, end: &[u32], msg: &[u32]) -> ReasoningConfig {
        ReasoningConfig {
            enabled,
            budget,
            end_ids: end.to_vec(),
            message_ids: msg.to_vec(),
        }
    }

    /// 预算用完后强制输出结束标签（本模型的 `</think>` 是单 token）。
    #[test]
    fn forces_end_tag_after_budget() {
        let mut b = ReasoningBudget::new(cfg(true, 3, &[248069], &[]));
        for t in [10, 11, 12] {
            assert_eq!(b.next_forced(), None);
            b.accept(t);
        }
        // 第 4 步开始强制
        assert_eq!(b.next_forced(), Some(248069));
        b.accept(248069);
        assert!(b.done);
        assert_eq!(b.next_forced(), None, "结束后不再强制");
    }

    /// 模型自己提前结束思考 → 不再强制。
    #[test]
    fn stops_when_model_ends_early() {
        let mut b = ReasoningBudget::new(cfg(true, 100, &[248069], &[]));
        b.accept(10);
        b.accept(248069);
        assert!(b.done);
        assert_eq!(b.next_forced(), None);
    }

    /// `--reasoning-budget 0`：一个思考 token 都不给，立刻强制结束。
    #[test]
    fn zero_budget_forces_immediately() {
        let mut b = ReasoningBudget::new(cfg(true, 0, &[248069], &[]));
        assert_eq!(b.next_forced(), Some(248069));
        b.accept(248069);
        assert!(b.done);
    }

    /// 提示语按整 token 先输出，再输出结束标签。
    #[test]
    fn forces_message_then_end_tag() {
        let mut b = ReasoningBudget::new(cfg(true, 1, &[99], &[50, 51]));
        b.accept(10);
        assert_eq!(b.next_forced(), Some(50));
        b.accept(50);
        assert_eq!(b.next_forced(), Some(51));
        b.accept(51);
        assert_eq!(b.next_forced(), Some(99));
        b.accept(99);
        assert!(b.done);
    }

    /// 多 token 的结束标签：模型自己分几步输出也能识别。
    #[test]
    fn detects_multi_token_end_tag() {
        let mut b = ReasoningBudget::new(cfg(true, 50, &[7, 8], &[]));
        b.accept(7);
        assert!(!b.done);
        b.accept(8);
        assert!(b.done);
    }

    /// 关掉思考、或 budget<=0、或拿不到标签 → 完全不干预（默认行为）。
    #[test]
    fn inactive_configs_are_noop() {
        for c in [
            cfg(false, 10, &[248069], &[]),
            cfg(true, 0 - 1, &[248069], &[]),
            cfg(true, 10, &[], &[]),
        ] {
            let mut b = ReasoningBudget::new(c);
            for _ in 0..20 {
                assert_eq!(b.next_forced(), None);
                b.accept(1);
            }
            assert!(!b.is_forcing());
        }
    }

    #[test]
    fn restart_resets_state() {
        let mut b = ReasoningBudget::new(cfg(true, 1, &[248069], &[]));
        b.accept(1);
        assert_eq!(b.next_forced(), Some(248069));
        b.restart();
        assert_eq!(b.next_forced(), None);
        assert!(!b.done);
    }
}
