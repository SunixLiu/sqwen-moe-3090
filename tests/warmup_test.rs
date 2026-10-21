// tests/warmup_test.rs — 启动预热（`Engine::warmup`）的**等价性**测试
//
// 单独一个文件是有意的：`runner_test.rs` 里几个测试各持一个 20.2 GiB 的引擎，
// 靠 WDDM 分页勉强能同时活两个，再加一个就 OOM。cargo 对**不同测试二进制是串行**
// 执行的，所以把这条放这里能保证「同一时刻只有一个引擎」，同时不改动已有测试。
//
// 断言的是：**预热必须等价于"什么都没做过"**。dummy 前向会真实地写 KV cache、
// GDN 递推状态与 MTP 的 `h_buf`；复位只要漏掉一处，第一个真实请求就会接着 dummy
// 序列往下走 —— 输出悄悄跑偏，而性能数字看起来一切正常。

use sqwen::backend::ctx::BackendKind;
use sqwen::engine::runner::{Engine, EngineConfig};
use sqwen::engine::sampler::Sampler;

const BIGBANG: &str = r"E:\LLMModels\bigbang\bigbang-v1-i1-gguf\BigBang-v1.i1-Q4_K_M.gguf";
const MAX_CONTEXT: i64 = 1024;

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

fn base_cfg(spec: sqwen::engine::runner::SpecType) -> EngineConfig {
    EngineConfig {
        max_context: MAX_CONTEXT,
        batch_size: 2048,
        ubatch_size: 512,
        flash_attn: sqwen::engine::runner::FlashAttnMode::Auto,
        threads: 4,
        cache_prompt: false,
        spec_type: spec,
        spec_draft_n_max: 2,
        spec_adaptive: true,
        sampler: Sampler::new(0.0, 0, 1.0, 7), // greedy，保证确定性
    }
}

/// 预热后生成 == 不预热直接生成。开投机时也成立（MTP 的图同样要复位）。
#[test]
fn warmup_does_not_change_output() {
    if !std::path::Path::new(BIGBANG).exists() {
        eprintln!("[warmup] SKIP: 模型文件不存在");
        return;
    }
    if gpu_free_mib().unwrap_or(0) < 21504 {
        eprintln!("[warmup] SKIP: 显存不足 21 GiB");
        return;
    }

    for spec in [
        sqwen::engine::runner::SpecType::None,
        sqwen::engine::runner::SpecType::DraftMtp,
    ] {
        let prompt: Vec<u32> = vec![11, 22, 33, 44, 55];

        // A：先 warmup 再生成。作用域很关键 —— 取到结果就把引擎丢掉，再加载 B
        // （同一时刻只能有一个引擎）。
        let (a, warm_ms) = {
            let mut warm = Engine::load(BIGBANG, BackendKind::Cuda(0), base_cfg(spec))
                .expect("Engine::load A");
            let t0 = std::time::Instant::now();
            warm.warmup().expect("warmup");
            let ms = t0.elapsed().as_millis();
            let (out, _) = warm.generate(&prompt, 8, &mut |_| {}).expect("generate A");
            (out, ms)
        };

        // B：全新引擎，不预热
        let mut cold = Engine::load(BIGBANG, BackendKind::Cuda(0), base_cfg(spec))
            .expect("Engine::load B");
        let (b, _) = cold.generate(&prompt, 8, &mut |_| {}).expect("generate B");

        assert_eq!(a, b, "预热改变了输出（spec={spec:?}，说明复位不干净）");
        eprintln!("[warmup] spec={spec:?} warmup {warm_ms} ms；预热前后逐 token 一致：{a:?}");
    }
}
