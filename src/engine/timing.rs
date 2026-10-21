// src/engine/timing.rs — 推理计量的统一结构
//
// 字段与 llama.cpp 的 `timings` 对齐，单位统一为毫秒 / token 数；
// `prompt_*` 是 prefill（prompt evaluation），`gen_*` 是 decode（token generation），
// `ttft_ms` 是「从开始到第一个 token 出来」的时间。
//
// 这里只做「存数据 + 算速率」，不持有计时器 —— 计时由 runner 用 `Instant` 完成，
// 这样 Timing 是纯数据，方便直接塞进 API 响应序列化。

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct Timing {
    /// 首 token 延迟（毫秒）
    pub ttft_ms: f64,
    /// prefill 处理的 token 数
    pub prompt_tokens: u32,
    /// 其中**直接复用**（`--cache-prompt` 命中）的 token 数，不计入 `prompt_ms` 的分子
    /// 对应的实际计算量。`prompt_tokens - prompt_cached` 才是本次真正 prefill 的量
    #[serde(default)]
    pub prompt_cached: u32,
    /// prefill 耗时（毫秒）
    pub prompt_ms: f64,
    /// 生成的 token 数
    pub gen_tokens: u32,
    /// 生成耗时（毫秒）
    pub gen_ms: f64,
    /// **稳态窗口**的 token 数与耗时：从「生成开始后第 `STEADY_SKIP_MS` 毫秒」
    /// 那一刻起算，跳过冷启动段；窗口内不足 `MIN_STEADY_TOKENS` 个 token 时是 0。
    ///
    /// **为什么需要它**：`eval_rate()` 是**整段累计**的，会把开头那段 GPU 时钟爬升
    /// （空闲 P8/270MHz → 负载 P2/1900MHz，要几十秒）也平均进去。实测同一段生成：
    /// 累计 133.97 t/s，而最近 3 秒窗口是 143 t/s —— **累计值低约 6%**。
    ///
    /// llama.cpp 的 server 同时输出 `tg`（累计）与 `tg_3s`（3 秒窗口），而我们的
    /// `Eval rate` 一直是累计口径，拿去和它的 `tg_3s` 比就系统性吃亏（也会让
    /// 两台机器/两次运行的对比被各自的冷启动状态污染）。这两个字段让口径能对齐。
    pub gen_tokens_steady: u32,
    pub gen_ms_steady: f64,
}

impl Timing {
    /// prefill 速率（tok/s）。
    pub fn prompt_rate(&self) -> f64 {
        rate(self.prompt_tokens, self.prompt_ms)
    }

    /// decode 速率（tok/s）—— **累计**口径，含冷启动段。
    pub fn eval_rate(&self) -> f64 {
        rate(self.gen_tokens, self.gen_ms)
    }

    /// decode 速率（tok/s）—— **稳态**口径，跳过冷启动段。
    ///
    /// 与 llama.cpp `tg_3s` 的差别：它是「最近 3 秒」的滑动窗口，我们是
    /// 「第 3 秒之后到结束」的固定窗口。稳态下两者收敛；生成太短（< 3 秒或窗口内
    /// token 太少）时我们返回 0，调用方应据此不显示。
    pub fn eval_rate_steady(&self) -> f64 {
        rate(self.gen_tokens_steady, self.gen_ms_steady)
    }
}

/// `tokens / ms * 1000`，即 tok/s。
///
/// `ms <= 0`（还没计时、或者时钟粒度导致 0）时返回 0 而不是 inf/NaN ——
/// 这个结构会直接序列化进 API 响应，JSON 里出现 Infinity 是非法的。
fn rate(tokens: u32, ms: f64) -> f64 {
    if ms <= 0.0 {
        0.0
    } else {
        tokens as f64 / ms * 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_finite() {
        let t = Timing::default();
        assert_eq!(t.prompt_rate(), 0.0);
        assert_eq!(t.eval_rate(), 0.0);
        assert!(t.prompt_rate().is_finite() && t.eval_rate().is_finite());
    }

    #[test]
    fn negative_ms_is_finite() {
        let t = Timing {
            prompt_tokens: 100,
            prompt_ms: -5.0,
            ..Default::default()
        };
        assert_eq!(t.prompt_rate(), 0.0);
    }
}
