// tests/prompt_cache_test.rs — `--cache-prompt` 的前缀复用正确性（P6.3）
//
// 需要真实模型（20GB，不在仓库里）：
//   $env:SQWEN_TEST_MODEL = "E:\LLMModels\bigbang\bigbang-v1-i1-gguf\BigBang-v1.i1-Q4_K_M.gguf"
//   cargo test --test prompt_cache_test -- --nocapture
//
// 判据（llama.cpp README 也写明 `--cache-prompt` 会改变输出，因为批形状变了）：
//   1. 复用确实发生（`prompt_cached > 0`，且第二次 TTFT 大幅下降）；
//   2. 复用后的输出与"整段重算"的输出**高度一致**（逐 token 对比，允许极少数
//      因归约顺序变化导致的翻转）。

use sqwen::backend::ctx::BackendKind;
use sqwen::engine::runner::{Engine, EngineConfig, FlashAttnMode};
use sqwen::engine::sampler::Sampler;
use sqwen::template::GgufTokenizer;

fn model_path() -> Option<String> {
    std::env::var("SQWEN_TEST_MODEL").ok().filter(|p| std::path::Path::new(p).exists())
}

fn load(cache_prompt: bool, ctx: i64) -> Engine {
    let model = model_path().unwrap();
    let cfg = EngineConfig {
        max_context: ctx,
        batch_size: 2048,
        ubatch_size: 512,
        flash_attn: FlashAttnMode::Auto,
        threads: 0,
        cache_prompt,
        spec_type: sqwen::engine::runner::SpecType::None,
        spec_draft_n_max: 3,
        sampler: Sampler::from_params(sqwen::engine::sampler::SamplingParams {
            temp: 0.0, // greedy，便于逐 token 比较
            ..Default::default()
        }),
    };
    let mut eng = Engine::load(&model, BackendKind::Cuda(0), cfg).expect("load engine");
    eng.set_eos(Vec::new()); // 忽略 EOS，固定生成 N 个 token
    eng
}

#[test]
fn prefix_reuse_matches_full_recompute() {
    let Some(_) = model_path() else {
        eprintln!("SKIP: 未设置 SQWEN_TEST_MODEL（或文件不存在）");
        return;
    };

    // 一段几百 token 的 prompt：模型自己和我们都不关心它的语义，只关心"前缀相同"
    let base: Vec<u32> = (0..400u32).map(|i| 1000 + (i * 37) % 5000).collect();
    const N1: usize = 24;
    const N2: usize = 16;

    // ---- 第一遍：缓存关掉（基线）----
    let (out1_off, t1_off, out2_off, t2_off) = {
        let mut eng = load(false, 4096);
        let (o1, tm1) = eng.generate(&base, N1, &mut |_| {}).unwrap();
        let mut prompt2 = base.clone();
        prompt2.extend_from_slice(&o1);
        prompt2.extend_from_slice(&[777, 888, 999, 1111]); // 比上次更长 → 走"延续"
        let (o2, tm2) = eng.generate(&prompt2, N2, &mut |_| {}).unwrap();
        (o1, tm1, o2, tm2)
    };
    assert_eq!(t1_off.prompt_cached, 0);
    assert_eq!(t2_off.prompt_cached, 0);

    // ---- 第二遍：缓存开 ----
    let (out1_on, t1_on, out2_on, t2_on) = {
        let mut eng = load(true, 4096);
        let (o1, tm1) = eng.generate(&base, N1, &mut |_| {}).unwrap();
        let mut prompt2 = base.clone();
        prompt2.extend_from_slice(&o1);
        prompt2.extend_from_slice(&[777, 888, 999, 1111]);
        let (o2, tm2) = eng.generate(&prompt2, N2, &mut |_| {}).unwrap();
        (o1, tm1, o2, tm2)
    };

    eprintln!(
        "第一次（无缓存 vs 有缓存）: 输出一致 = {} | cached = {} / {}",
        out1_off == out1_on,
        t1_on.prompt_cached,
        t1_on.prompt_tokens
    );
    eprintln!(
        "第二次: 输出一致 = {} | cached = {} / {} | 重算 {} tok",
        out2_off == out2_on,
        t2_on.prompt_cached,
        t2_on.prompt_tokens,
        t2_on.prompt_tokens - t2_on.prompt_cached
    );
    eprintln!(
        "         TTFT: 无缓存 {:.1} ms → 有缓存 {:.1} ms",
        t2_off.ttft_ms, t2_on.ttft_ms
    );
    eprintln!("第二次输出(无缓存) = {out2_off:?}");
    eprintln!("第二次输出(有缓存) = {out2_on:?}");
    // ---- 一致性判据 ----
    //
    // 先说清楚**为什么不可能逐 token 完全一致**：第二次请求在无缓存时把 428 个 token
    // 一次 prefill（一个 chunk），复用路径只重算 5 个 token（另一个 chunk）。
    // FlashAttention 的归约顺序随 batch 形状变化，最后一个位置的 logits 有末位差异，
    // greedy 就可能多/少采到一个**不产出文本的控制 token**（实测差一个 74455，
    // 解码为空；llama.cpp README 也写明 `--cache-prompt` 会改变输出）。
    //
    // 所以判据分三层，任何一层不成立都说明恢复的状态是错的（而不是噪声）：
    //   1. 复用确实发生，且 TTFT 大幅下降；
    //   2. 复用后**头几个 token 完全一致** —— GDN 递推状态如果恢复错了，
    //      第一个 token 就会立刻偏离；
    //   3. 解码文本的公共前缀 ≥ 短者的 80%。
    let tok = GgufTokenizer::from_gguf(&model_path().unwrap()).expect("tokenizer");
    let text_off = tok.extract_answer(&tok.decode(&out2_off));
    let text_on = tok.extract_answer(&tok.decode(&out2_on));
    let common = text_off
        .chars()
        .zip(text_on.chars())
        .take_while(|(a, b)| a == b)
        .count();
    let shorter = text_off.chars().count().min(text_on.chars().count());
    eprintln!("第二次文本(无缓存) = {text_off:?}");
    eprintln!("第二次文本(有缓存) = {text_on:?}");
    eprintln!("文本公共前缀 = {common}/{shorter} 字符");
    if let Some(&bad) = out2_off.iter().zip(out2_on.iter()).find(|(a, b)| a != b).map(|(a, _)| a) {
        eprintln!("首个不同 token {bad} 解码为 {:?}", String::from_utf8_lossy(&tok.decode_bytes(&[bad])));
    }
    eprintln!(
        "         token 逐位一致 = {}/{}",
        out2_off.iter().zip(out2_on.iter()).filter(|(a, b)| a == b).count(),
        out2_off.len()
    );

    assert_eq!(t1_on.prompt_cached, 0, "第一次请求不可能有前缀可复用");
    assert_eq!(out1_off, out1_on, "第一次请求不涉及复用，结果必须一致");

    // 第二次：prompt = 上次的 prompt + 上次生成的 token + 4 个新 token → 应当复用
    assert!(
        t2_on.prompt_cached > 0,
        "第二次请求应当复用前缀（实际 cached = {}）",
        t2_on.prompt_cached
    );
    assert!(
        t2_on.ttft_ms < t2_off.ttft_ms,
        "复用的 TTFT 必须更低：{:.1} vs {:.1} ms",
        t2_on.ttft_ms,
        t2_off.ttft_ms
    );
    assert_eq!(
        &out2_off[..4],
        &out2_on[..4],
        "复用后的头 4 个 token 必须与整段重算一致（否则说明恢复的 GDN 状态是错的）"
    );
    assert!(
        common * 5 >= shorter * 4,
        "复用后的文本差异过大：公共前缀 {common}/{shorter}\n  off={out2_off:?}\n  on ={out2_on:?}"
    );
}
