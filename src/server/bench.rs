// src/server/bench.rs — Bench 模式（Task 13 / Phase 0）
//
// 协议对齐 llama-bench，这样两边的数字才可比：
//   - **warmup**：第一次测量丢弃（避开 CUDA graph capture、权重 repack、页表预热）；
//   - **重复**：默认 3 次，报告均值（llama-bench 报 mean ± std，这里用 `txt` 输出给脚本解析）；
//   - **PP / TG 分开**：PP = prefill 速率（tok/s），TG = 逐 token 生成速率；
//   - **同一个 prompt**：可以喂 `--prompt-ids-file`（与 llama.cpp 用完全相同的 token id），
//     否则用 `--prompt-tokens N` 造一个确定性的假 prompt。
//
// 输出两种格式：
//   `txt`（默认）人类可读
//   `csv`        一行一个指标，便于脚本收集（sqwen_bench / llama_bench 对比）

use crate::engine::runner::Engine;
use crate::engine::timing::Timing;

/// Bench 配置。
pub struct BenchConfig {
    /// 假 prompt 的 token 数（`prompt_ids` 为 Some 时忽略）
    pub prompt_tokens: usize,
    /// 生成 token 数
    pub gen_tokens: usize,
    /// 重复次数（第一次作为 warmup 丢弃；0 = 只跑一次不丢弃）
    pub repetitions: usize,
    /// 显式的 prompt token id（与 llama.cpp 对齐用）
    pub prompt_ids: Option<Vec<u32>>,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            prompt_tokens: 1024,
            gen_tokens: 128,
            repetitions: 3,
            prompt_ids: None,
        }
    }
}

/// 一次 bench 的结果（多次重复的均值）。
#[derive(Clone, Copy, Debug, Default)]
pub struct BenchReport {
    pub n_prompt: usize,
    pub n_gen: usize,
    pub runs: usize,
    pub ttft_ms: f64,
    pub prompt_rate: f64,
    pub eval_rate: f64,
    /// 单次 TG 耗时的标准差（ms/token），用于判断测量是否稳定
    pub eval_std_ms: f64,
}

impl BenchReport {
    pub fn to_txt(&self) -> String {
        format!(
            "pp{}: {:.1} t/s | tg{}: {:.1} t/s | TTFT {:.1} ms | pp_ms {:.1} | tg_ms/tok {:.2} (std {:.2}) | runs {}\n",
            self.n_prompt,
            self.prompt_rate,
            self.n_gen,
            self.eval_rate,
            self.ttft_ms,
            self.n_prompt as f64 / self.prompt_rate * 1000.0,
            if self.eval_rate > 0.0 { 1000.0 / self.eval_rate } else { 0.0 },
            self.eval_std_ms,
            self.runs
        )
    }

    pub fn to_csv(&self) -> String {
        let mut s = String::from("n_prompt,n_gen,runs,ttft_ms,pp_tok_s,tg_tok_s,tg_ms_per_tok,tg_ms_std\n");
        s.push_str(&format!(
            "{},{},{},{:.3},{:.3},{:.3},{:.4},{:.4}\n",
            self.n_prompt,
            self.n_gen,
            self.runs,
            self.ttft_ms,
            self.prompt_rate,
            self.eval_rate,
            if self.eval_rate > 0.0 { 1000.0 / self.eval_rate } else { 0.0 },
            self.eval_std_ms
        ));
        s
    }
}

/// 跑一轮（不做 warmup / 不重复），返回 Timing。
///
/// 与 llama-bench 一致：**临时清空 EOS**，固定生成 `gen_tokens` 个 token。
/// 否则模型提前吐 EOS 会让"每秒 token 数"变成"每秒 token 数 ÷ 实际长度"，两边不可比。
pub fn run_once(eng: &mut Engine, prompt: &[u32], gen_tokens: usize) -> Timing {
    let saved_eos = eng.eos_tokens().to_vec();
    eng.set_eos(Vec::new());
    let (tokens, timing) = eng.generate(prompt, gen_tokens, &mut |_| {}).expect("bench generate 失败");
    eng.set_eos(saved_eos);
    assert_eq!(
        tokens.len(),
        gen_tokens,
        "bench 生成数量与请求不符（清空 EOS 后应当刚好生成 {} 个）",
        gen_tokens
    );
    timing
}

/// 按 llama-bench 的协议跑：warmup（可选）+ `repetitions` 次取均值。
pub fn run_bench(eng: &mut Engine, cfg: BenchConfig) -> BenchReport {
    let prompt: Vec<u32> = match &cfg.prompt_ids {
        Some(ids) => ids.clone(),
        // 确定性的假 prompt：token id 在词表内且不撞 special
        None => (0..cfg.prompt_tokens).map(|i| (i % 1000) as u32).collect(),
    };
    if prompt.is_empty() {
        panic!("bench: prompt 不能为空");
    }

    let reps = if cfg.repetitions == 0 { 1 } else { cfg.repetitions };
    let warmup = cfg.repetitions > 0;

    // warmup：结果直接丢弃
    if warmup {
        let _ = run_once(eng, &prompt, cfg.gen_tokens);
    }

    let mut ttft = 0.0f64;
    let mut pp = 0.0f64;
    let mut tg = 0.0f64;
    let mut per_run_ms = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = run_once(eng, &prompt, cfg.gen_tokens);
        ttft += t.ttft_ms;
        pp += t.prompt_rate();
        tg += t.eval_rate();
        per_run_ms.push(if t.gen_tokens > 0 {
            t.gen_ms / t.gen_tokens as f64
        } else {
            0.0
        });
    }
    let n = reps as f64;
    let tg_mean_ms = per_run_ms.iter().sum::<f64>() / n;
    let var = per_run_ms
        .iter()
        .map(|x| (x - tg_mean_ms).powi(2))
        .sum::<f64>()
        / n;
    BenchReport {
        n_prompt: prompt.len(),
        n_gen: cfg.gen_tokens,
        runs: reps,
        ttft_ms: ttft / n,
        prompt_rate: pp / n,
        eval_rate: tg / n,
        eval_std_ms: var.sqrt(),
    }
}

/// 人类可读输出（`sqwen bench` 的默认格式）。
pub fn run_bench_detailed(eng: &mut Engine, cfg: BenchConfig) -> String {
    let r = run_bench(eng, cfg);
    r.to_txt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_config_defaults() {
        let cfg = BenchConfig::default();
        assert_eq!(cfg.prompt_tokens, 1024);
        assert_eq!(cfg.gen_tokens, 128);
        assert_eq!(cfg.repetitions, 3);
    }

    #[test]
    fn report_formats() {
        let r = BenchReport {
            n_prompt: 512,
            n_gen: 64,
            runs: 3,
            ttft_ms: 100.0,
            prompt_rate: 2000.0,
            eval_rate: 100.0,
            eval_std_ms: 0.5,
        };
        assert!(r.to_txt().contains("pp512"));
        assert!(r.to_txt().contains("tg64"));
        let csv = r.to_csv();
        assert!(csv.starts_with("n_prompt,"));
        assert!(csv.lines().count() == 2);
        // tg_ms_per_tok = 1000/100 = 10
        assert!(csv.contains(",10.0000,"), "{csv}");
    }
}
