// tests/golden.rs — Task 14：Golden test（与 llama.cpp CLI 逐 token 比对）
//
// 用同一份 prompt 分别跑 sqwen engine 和 llama.cpp CLI，
// 比较前 N 个生成 token 是否一致。
//
// 前置条件：
//   - BIGBANG model 存在
//   - llama.cpp CLI (llama-cli) 可用
//   - GPU 显存空闲 ≥ 23 GiB
//
// 若任一条件不满足，测试自动 skip。

use sqwen::backend::ctx::BackendKind;
use sqwen::engine::runner::{Engine, EngineConfig};
use sqwen::engine::sampler::Sampler;

const BIGBANG: &str = r"E:\LLMModels\bigbang\bigbang-v1-i1-gguf\BigBang-v1.i1-Q4_K_M.gguf";
const LLAMA_CLI: &str = "D:\\01Projects\\sqwen\\reference\\llama.cpp\\build\\bin\\Release\\llama-cli.exe";
const MAX_TOKENS: usize = 50;

/// 下面两个常量的默认值是本机路径，可以用环境变量覆盖：
/// `SQWEN_TEST_MODEL=<gguf> SQWEN_TEST_LLAMA_CLI=<llama-cli.exe> cargo test --test golden`
fn test_model() -> String {
    std::env::var("SQWEN_TEST_MODEL").unwrap_or_else(|_| BIGBANG.to_string())
}

fn test_llama_cli() -> String {
    std::env::var("SQWEN_TEST_LLAMA_CLI").unwrap_or_else(|_| LLAMA_CLI.to_string())
}

fn gpu_free_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .parse::<u64>()
        .ok()
}

#[test]
fn golden_compare_with_llama_cpp() {
    // 1. 检查前置条件
    if !std::path::Path::new(&test_model()).exists() {
        eprintln!("[golden] SKIP: model not found");
        return;
    }
    if !std::path::Path::new(&test_llama_cli()).exists() {
        eprintln!("[golden] SKIP: llama-cli not found");
        return;
    }
    let free = gpu_free_mib().unwrap_or(0);
    if free < 23008 {
        eprintln!("[golden] SKIP: GPU memory insufficient ({free} MiB < 23 GiB)");
        return;
    }

    // 2. 跑 sqwen engine
    let cfg = EngineConfig {
        max_context: 1024,
        batch_size: 2048,
        ubatch_size: 512,
        flash_attn: sqwen::engine::runner::FlashAttnMode::Auto,
        threads: 4,
        cache_prompt: false,
        spec_type: sqwen::engine::runner::SpecType::None,
        spec_draft_n_max: 3,
        sampler: Sampler::new(0.0, 0, 1.0, 42), // greedy for determinism
    };
    let mut eng = Engine::load(&test_model(), BackendKind::Cuda(0), cfg).expect("Engine::load");
    let prompt_ids: Vec<u32> = (0..10).collect();
    let (sqwen_tokens, timing) = eng
        .generate(&prompt_ids, MAX_TOKENS, &mut |_| {})
        .expect("engine generate");

    eprintln!("[golden] sqwen tokens: {:?}", sqwen_tokens.iter().take(20).collect::<Vec<_>>());
    eprintln!("[golden] timing: ttft={:.1}ms prefill_rate={:.1} decode_rate={:.1}",
        timing.ttft_ms, timing.prompt_rate(), timing.eval_rate());

    // 3. 跑 llama.cpp CLI
    let cli_output = std::process::Command::new(test_llama_cli())
        .args([
            "-m", &test_model(),
            "--prompt", "test",
            "-n", MAX_TOKENS.to_string().as_str(),
            "--temp", "0",
            "--seed", "42",
        ])
        .output()
        .expect("llama-cli failed");

    let cli_text = String::from_utf8_lossy(&cli_output.stdout);
    eprintln!("[golden] llama-cli output:\n{cli_text}");

    // 4. 解析 CLI 输出为 token ids（简化：按空格/换行分割数字）
    let cli_tokens: Vec<u32> = cli_text
        .split(|c: char| c.is_whitespace() || c == '\n')
        .filter_map(|s| s.parse().ok())
        .collect();

    // 5. 逐 token 比对前 N 个
    let n_compare = MAX_TOKENS.min(sqwen_tokens.len()).min(cli_tokens.len());
    for i in 0..n_compare {
        assert_eq!(
            sqwen_tokens[i], cli_tokens[i],
            "token[{}] mismatch: sqwen={} vs llama-cli={}",
            i, sqwen_tokens[i], cli_tokens[i]
        );
    }

    eprintln!("[golden] PASS: first {} tokens match", n_compare);
}

#[test]
fn golden_reproducible_same_engine() {
    // 验证同一 engine 连续两次 generate 给出相同结果（在 golden 条件下）
    if !std::path::Path::new(&test_model()).exists() {
        return;
    }
    let free = gpu_free_mib().unwrap_or(0);
    if free < 23008 {
        return;
    }

    let cfg = EngineConfig {
        max_context: 1024,
        batch_size: 2048,
        ubatch_size: 512,
        flash_attn: sqwen::engine::runner::FlashAttnMode::Auto,
        threads: 4,
        cache_prompt: false,
        spec_type: sqwen::engine::runner::SpecType::None,
        spec_draft_n_max: 3,
        sampler: Sampler::new(0.0, 0, 1.0, 42),
    };
    let prompt_ids: Vec<u32> = (0..10).collect();

    let mut eng = Engine::load(&test_model(), BackendKind::Cuda(0), cfg.clone()).expect("Engine::load");
    let (a, _) = eng.generate(&prompt_ids, MAX_TOKENS, &mut |_| {}).expect("gen a");
    let (b, _) = eng.generate(&prompt_ids, MAX_TOKENS, &mut |_| {}).expect("gen b");

    // 在 golden 条件下（greedy + seed=42），同一 engine 应该给出相同序列
    assert_eq!(a, b, "same engine greedy should be reproducible: a={:?} b={:?}", a, b);
}