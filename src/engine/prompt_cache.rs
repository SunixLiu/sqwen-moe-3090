// src/engine/prompt_cache.rs — `--cache-prompt`：跨请求复用 prompt 前缀（P6.3）
//
// 目标：多轮对话里，新请求的 prompt 通常 = 上一次的 prompt + 上次的回答 + 新的用户消息。
// 这一大段前缀已经算过一次了，没必要重算 —— 复用能把 TTFT 从 O(历史长度) 降到
// O(新增 token)，10 万 token 的历史从 ~60 秒降到 ~1 秒。
//
// ---------------------------------------------------------------------------
// 本模型的特殊之处：GDN 递推状态不能"丢 KV 行"回滚
// ---------------------------------------------------------------------------
// 30 层 GDN 是**递推**的：状态是"到目前为止所有 token 的压缩摘要"，不像注意力的 KV
// 那样按位置寻址。所以哪怕只差一个 token，也不能通过删 KV 行回到中间某个位置。
// llama.cpp 的做法（`server-context.cpp:3360-3396`）是保存/恢复**状态检查点**：
// 有合适的检查点就恢复，没有就打印 "forcing full prompt re-processing due to lack of
// cache data (likely due to SWA or hybrid/recurrent memory)" 并整段重算。
//
// 我们采用同一套思路，但更直接：
//   - `PromptCache` 保存「上一次序列处理过的 token 列表」+ 若干**检查点**
//     （位置 + 该位置的 GDN 递推状态快照）；
//   - 新请求先求与缓存 token 列表的**最长公共前缀** `lcp`；
//   - 取「位置 ≤ min(lcp, 新prompt长度-1)」的**最新**检查点 `p`：
//     - 恢复该检查点的 GDN 状态；
//     - KV 游标回退到 `p`（`[0,p)` 那些行的数据**原地不动** —— 前缀 token 相同，
//       它们本来就是对的；`[p, ...)` 会被本次 prefill 覆盖）；
//     - 只 prefill 新 prompt 的 `[p, len)`。
//   - 至少要重算 1 个 token（否则拿不到 logits），与 llama.cpp `n_past--` 同理。
//
// 为什么不需要 llama.cpp 的 `--cache-reuse`（KV 平移）：那套是给"前缀中间出现相同
// 片段、但位置整体后移"的场景做 KV 搬移的；我们直接重新 prefill `[p, len)` 这一段，
// 检查点把 `p` 顶到离分歧点很近的地方，重算量本来就只有几十~几千 token。
//
// 失效规则：KV 行与递推状态都只在「token 前缀没变」时有效。一旦新序列在 `lcp` 处
// 分叉，位置 > lcp 的检查点就必须丢掉（它们的 KV 行会被新 prefill 覆盖）——
// 对应 llama.cpp 的 "erase any checkpoints with pos_max > pos_next"。

/// 一个状态检查点：位置 + 该位置的 GDN 递推状态快照。
#[derive(Clone, Debug)]
pub struct Checkpoint {
    /// 该状态对应的「已处理 token 数」（= 下一个 token 的位置）。
    pub pos: usize,
    /// GDN 各层的 `state` + `conv_state` 原始字节（布局见 `EngineState::snapshot_gdn`）。
    pub data: Vec<u8>,
}

impl Checkpoint {
    pub fn bytes(&self) -> usize {
        self.data.len()
    }
}

/// 复用计划（`PromptCache::plan` 的返回值）。
pub struct ReusePlan {
    /// 从哪个 token 位置开始 prefill（之前的都复用）
    pub start_pos: usize,
    /// 用了哪个检查点（下标，诊断用）
    pub checkpoint_idx: usize,
}

/// 前缀复用缓存（单序列；`-np 1` 时的完整实现）。
#[derive(Debug)]
pub struct PromptCache {
    enabled: bool,
    /// 上一次序列**已处理**的 token（prompt + 生成，不含最后那个"采出来但没喂进去"的 token）
    tokens: Vec<u32>,
    /// 检查点，按 `pos` 升序
    checkpoints: Vec<Checkpoint>,
    /// 检查点总字节上限（超过就丢最旧的）
    max_bytes: usize,
    /// 统计（诊断/验证用）
    pub hits: u64,
    pub misses: u64,
    /// 累计复用掉的 token 数（= 省下的 prefill 量）
    pub reused_tokens: u64,
}

impl PromptCache {
    /// `max_bytes`：检查点快照的总预算。一个快照 ≈ 「GDN 状态字节数」
    /// （本模型 30 层 × (2 MiB state + conv) ≈ 63 MiB），所以默认 512 MiB 大约留 8 个。
    pub fn new(enabled: bool, max_bytes: usize) -> Self {
        Self {
            enabled,
            tokens: Vec::new(),
            checkpoints: Vec::new(),
            max_bytes,
            hits: 0,
            misses: 0,
            reused_tokens: 0,
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// 缓存的 token 序列（诊断用）。
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// 检查点总字节数（诊断用）。
    pub fn bytes(&self) -> usize {
        self.checkpoints.iter().map(|c| c.bytes()).sum()
    }

    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    /// 最长公共前缀长度。
    pub fn lcp(&self, tokens: &[u32]) -> usize {
        self.tokens
            .iter()
            .zip(tokens.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// 找到「位置 ≤ `up_to`」的最新检查点下标。
    pub fn best_checkpoint(&self, up_to: usize) -> Option<usize> {
        self.checkpoints
            .iter()
            .rposition(|c| c.pos <= up_to)
    }

    /// 给出复用计划；`None` = 老实整段 prefill。
    ///
    /// 复用门槛：省下的 token 占比 ≥ `min_keep_frac`（llama.cpp 的 `f_keep ≥ 0.25`），
    /// 且至少省下 `MIN_REUSE_TOKENS` 个（太短就不值得走一遍恢复状态的 H2D）。
    pub fn plan(&self, tokens: &[u32], min_keep_frac: f32) -> Option<ReusePlan> {
        if !self.enabled || tokens.is_empty() || self.tokens.is_empty() {
            return None;
        }
        let lcp = self.lcp(tokens);
        if lcp == 0 {
            return None;
        }
        // 至少要重算 1 个 token 才能拿到 logits（llama.cpp:414-417 的 n_past-- 同理）
        let limit = lcp.min(tokens.len().saturating_sub(1));
        let idx = self.best_checkpoint(limit)?;
        let start_pos = self.checkpoints[idx].pos;
        if start_pos < MIN_REUSE_TOKENS || start_pos == 0 || start_pos >= tokens.len() {
            return None;
        }
        if (start_pos as f32) < tokens.len() as f32 * min_keep_frac {
            return None;
        }
        Some(ReusePlan { start_pos, checkpoint_idx: idx })
    }

    /// 取某个检查点的数据（`plan` 之后调用）。
    pub fn checkpoint_data(&self, idx: usize) -> &[u8] {
        &self.checkpoints[idx].data
    }

    /// 记录一次命中。
    pub fn note_hit(&mut self, start_pos: usize) {
        self.hits += 1;
        self.reused_tokens += start_pos as u64;
    }

    /// 丢掉位置 > `lcp` 的检查点（前缀一旦分叉，它们就失效了）。
    pub fn truncate_to(&mut self, lcp: usize) {
        self.checkpoints.retain(|c| c.pos <= lcp);
    }

    /// 整段重算：清空缓存并记为一次 miss。
    pub fn note_miss(&mut self) {
        self.misses += 1;
        self.tokens.clear();
        self.checkpoints.clear();
    }

    /// 保存一个检查点。
    ///
    /// 同一个位置只会留一份（后写覆盖先写）；超出 `max_bytes` 时从**最旧**的开始丢
    /// —— 前缀复用最常命中的是靠近序列末尾的那些检查点。
    pub fn push_checkpoint(&mut self, pos: usize, data: Vec<u8>) {
        if !self.enabled || data.is_empty() {
            return;
        }
        if let Some(c) = self.checkpoints.iter_mut().find(|c| c.pos == pos) {
            *c = Checkpoint { pos, data };
        } else {
            let at = self
                .checkpoints
                .iter()
                .position(|c| c.pos > pos)
                .unwrap_or(self.checkpoints.len());
            self.checkpoints.insert(at, Checkpoint { pos, data });
        }
        self.evict();
    }

    fn evict(&mut self) {
        while self.bytes() > self.max_bytes && self.checkpoints.len() > 1 {
            self.checkpoints.remove(0); // 丢最旧的
        }
    }

    /// 序列跑完：把「已处理 token 列表」记下来，供下一次请求求 lcp。
    pub fn finish(&mut self, tokens: Vec<u32>) {
        if !self.enabled {
            return;
        }
        self.tokens = tokens;
    }

    /// 关掉缓存并清空（`--no-cache-prompt` / 换模型时用）。
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.tokens.clear();
            self.checkpoints.clear();
        }
    }
}

/// 至少复用这么多 token 才值得（恢复状态要走一次 H2D，太短反而亏）。
pub const MIN_REUSE_TOKENS: usize = 32;

/// 默认的检查点快照预算（512 MiB）。本模型一个快照 ≈ 63 MiB → 约 8 个。
pub const DEFAULT_CACHE_BYTES: usize = 512 * 1024 * 1024;

#[cfg(test)]
mod tests {
    use super::*;

    /// tokens = [1, 2, ..., n]
    fn seq(n: usize) -> Vec<u32> {
        (1..=n as u32).collect()
    }

    fn cache_with(n: usize, ckpts: &[(usize, u8)]) -> PromptCache {
        let mut c = PromptCache::new(true, DEFAULT_CACHE_BYTES);
        for (pos, tag) in ckpts {
            c.push_checkpoint(*pos, vec![*tag; 8]);
        }
        c.finish(seq(n));
        c
    }

    #[test]
    fn lcp_basics() {
        let c = cache_with(5, &[]);
        assert_eq!(c.lcp(&[1, 2, 3, 9]), 3);
        assert_eq!(c.lcp(&[1, 2, 3, 4, 5, 6]), 5);
        assert_eq!(c.lcp(&[9]), 0);
        assert_eq!(c.lcp(&[]), 0);
    }

    /// 新 prompt 完全延伸了旧序列 → 用末尾检查点，只重算 1 个 token。
    #[test]
    fn plan_extends_cached_sequence() {
        let c = cache_with(200, &[(100, 1), (200, 2)]);
        let mut prompt = seq(200);
        prompt.extend_from_slice(&[201, 202]);
        let p = c.plan(&prompt, 0.25).expect("应当命中");
        assert_eq!(p.start_pos, 200, "用末尾检查点，只重算后面 2 个 token");
    }

    /// 中间分叉（lcp = 150）→ 只能用位置 ≤ 150 的检查点，重算 [100, len)。
    #[test]
    fn plan_uses_checkpoint_at_or_before_divergence() {
        let c = cache_with(200, &[(100, 1), (200, 2)]);
        let mut prompt = seq(150);
        prompt.extend_from_slice(&[999, 998, 997, 996]);
        let p = c.plan(&prompt, 0.25).expect("应当命中");
        assert_eq!(p.start_pos, 100);
        assert_eq!(c.checkpoint_data(p.checkpoint_idx), &[1u8; 8]);
    }

    /// 新 prompt 是旧序列的前缀 → 必须重算最后 1 个 token（拿 logits），
    /// 所以末尾那个位置的检查点**不能用**。
    #[test]
    fn plan_leaves_one_token_to_evaluate() {
        let c = cache_with(200, &[(100, 1), (199, 3), (200, 2)]);
        let p = c.plan(&seq(200), 0.0).expect("应当命中");
        assert_eq!(p.start_pos, 199, "位置 200 的检查点不可用（至少要重算 1 个 token）");
    }

    #[test]
    fn no_plan_when_too_little_is_reused() {
        // 只省下 5 个 token（低于 MIN_REUSE_TOKENS）→ 不复用
        let c = cache_with(200, &[(5, 2)]);
        let mut prompt = seq(200);
        prompt.push(300);
        assert!(c.plan(&prompt, 0.0).is_none());

        // 占比太低 → 不复用（llama.cpp 的 f_keep ≥ 0.25）
        let c = cache_with(200, &[(100, 1)]);
        let mut prompt = seq(100);
        prompt.extend(1000..1300);
        assert!(c.plan(&prompt, 0.9).is_none(), "f_keep = 100/400 = 0.25，低于 0.9");
    }

    #[test]
    fn truncate_drops_invalidated_checkpoints() {
        let mut c = cache_with(200, &[(100, 1), (200, 2)]);
        c.truncate_to(150);
        assert_eq!(c.checkpoint_count(), 1);
        assert_eq!(c.checkpoints[0].pos, 100);
    }

    #[test]
    fn evicts_oldest_when_over_budget() {
        let mut c = PromptCache::new(true, 20); // 每个快照 8 字节 → 只放得下 2 个
        for pos in [100, 200, 300, 400] {
            c.push_checkpoint(pos, vec![0; 8]);
        }
        assert_eq!(c.checkpoint_count(), 2);
        assert_eq!(c.checkpoints[0].pos, 300, "最旧的被丢掉");
    }

    #[test]
    fn disabled_cache_never_plans() {
        let mut c = PromptCache::new(false, DEFAULT_CACHE_BYTES);
        c.push_checkpoint(64, vec![0; 8]);
        c.finish(seq(200));
        assert!(c.plan(&seq(200), 0.0).is_none());
    }
}
