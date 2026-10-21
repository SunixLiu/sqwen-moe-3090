// tests/sampler_test.rs — Task 9：采样器 + timing 计量
//
// 覆盖三类东西：
//   1. 语义：greedy == argmax、top_k 只保留前 k 大、top_p 核采样、
//      权重确实按 softmax 概率（统计检验，不是「跑通就行」）；
//   2. 确定性：同 seed 同序列、不同 seed 不同序列（后者证明 RNG 没被忽略）；
//   3. 边界：单元素、全相等、NaN、top_k 超出长度、top_p <= 0 / >= 1、ms 为 0。

use sqwen::engine::sampler::{Sampler, SamplingParams};
use sqwen::engine::timing::Timing;

/// 采样 `n` 次，返回每个 token 的出现次数（下标即 token id）。
fn histogram(s: &mut Sampler, logits: &[f32], n: usize) -> Vec<usize> {
    let mut h = vec![0usize; logits.len()];
    for _ in 0..n {
        h[s.sample(logits) as usize] += 1;
    }
    h
}

// ---------------------------------------------------------------------------
// greedy
// ---------------------------------------------------------------------------

#[test]
fn greedy_is_argmax() {
    let mut s = Sampler::new(0.0, 0, 1.0, 42);
    let logits = [0.1f32, 5.0, 2.0, 9.0];
    assert_eq!(s.sample(&logits), 3);
}

/// temp=0 时 argmax 与 seed 无关，且相同最大值取最小下标（可复现的 tie-break）。
#[test]
fn greedy_ignores_seed_and_breaks_ties_by_index() {
    for seed in [0u64, 1, 42, u64::MAX] {
        let mut s = Sampler::new(0.0, 0, 1.0, seed);
        let logits = [3.0f32, 7.0, 7.0, 1.0];
        assert_eq!(s.sample(&logits), 1, "seed={seed}");
    }
    // 负数 logits 也要正确（不能把「第一个」当最大值）
    let mut s = Sampler::new(0.0, 0, 1.0, 1);
    assert_eq!(s.sample(&[-5.0f32, -1.0, -3.0]), 1);
}

/// 负温度/NaN 温度按 greedy 处理（文档约定），不能返回垃圾。
#[test]
fn non_positive_temperature_is_greedy() {
    for t in [-1.0f32, -0.0, f32::NAN] {
        let mut s = Sampler::new(t, 0, 1.0, 3);
        assert_eq!(s.sample(&[1.0f32, 4.0, 2.0]), 1, "temp={t}");
    }
}

/// NaN logits 视为 -inf：greedy 绝不能选到 NaN 位置。
///
/// 这不是纸上谈兵 —— 引擎真出过 NaN logits（GGUF 偏移 bug），采样器至少不该
/// 把 NaN 当成「最大的数」而把垃圾 token 选出来。
#[test]
fn nan_logits_are_treated_as_negative_infinity() {
    let mut s = Sampler::new(0.0, 0, 1.0, 5);
    assert_eq!(s.sample(&[f32::NAN, 1.0, 2.0]), 2);
    assert_eq!(s.sample(&[f32::NAN, f32::NAN, -1.0]), 2);
    // 全 NaN：退化为下标 0（处处相等），不能 panic
    assert_eq!(s.sample(&[f32::NAN, f32::NAN]), 0);
    // 随机采样时也不该选到 NaN 位置
    let mut r = Sampler::new(1.0, 0, 1.0, 9);
    for _ in 0..200 {
        assert_ne!(r.sample(&[f32::NAN, 0.0, 0.0]), 0);
    }
}

#[test]
fn single_token_always_wins() {
    let mut g = Sampler::new(0.0, 0, 1.0, 1);
    let mut r = Sampler::new(2.0, 0, 1.0, 1);
    assert_eq!(g.sample(&[0.5]), 0);
    assert_eq!(r.sample(&[0.5]), 0);
}

// ---------------------------------------------------------------------------
// 确定性
// ---------------------------------------------------------------------------

#[test]
fn deterministic_same_seed() {
    let mut a = Sampler::new(1.0, 10, 0.9, 123);
    let mut b = Sampler::new(1.0, 10, 0.9, 123);
    let logits: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1).collect();
    for _ in 0..50 {
        assert_eq!(a.sample(&logits), b.sample(&logits));
    }
}

/// 换个 seed 必须换出不同序列 —— 否则说明 seed 根本没参与（比如 RNG 被写死）。
#[test]
fn different_seed_gives_different_sequence() {
    let logits = [0.0f32; 32]; // 全相等：结果完全由 RNG 决定
    let draws = |seed: u64| -> Vec<u32> {
        let mut s = Sampler::new(1.0, 0, 1.0, seed);
        (0..64).map(|_| s.sample(&logits)).collect()
    };
    let a = draws(1);
    let b = draws(2);
    let diff = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    assert!(diff > 20, "不同 seed 的序列几乎相同（{diff}/64 不同），RNG 可能没用 seed");
    // 全相等 logits 下也不能退化成常量
    let uniq: std::collections::HashSet<_> = a.iter().collect();
    assert!(uniq.len() > 8, "全相等 logits 下只采到 {} 种 token，分布不对", uniq.len());
}

// ---------------------------------------------------------------------------
// top_k
// ---------------------------------------------------------------------------

#[test]
fn top_k_respects_bound() {
    let mut s = Sampler::new(1.0, 2, 1.0, 7);
    let logits = [0.0f32; 50];
    for _ in 0..100 {
        assert!(s.sample(&logits) < 2);
    }
}

/// top_k 必须保留**最大的 k 个**，而不只是「前 k 个下标」。
/// 故意把大 logit 放在高位下标，用下标顺序实现就会失败。
#[test]
fn top_k_keeps_the_largest_values() {
    let mut logits = [0.0f32; 32];
    logits[20] = 5.0;
    logits[7] = 4.0;
    logits[31] = 3.0;

    let mut s = Sampler::new(1.0, 2, 1.0, 11);
    let h = histogram(&mut s, &logits, 500);
    assert_eq!(h[20] + h[7], 500, "top_k=2 应只采到下标 20 与 7，实际 {h:?}");
    assert!(h[20] > 0 && h[7] > 0, "两个候选都应出现: {h:?}");
    // 概率 exp(5)/(exp(5)+exp(4)) = 0.731
    let p = h[20] as f64 / 500.0;
    assert!((p - 0.731).abs() < 0.08, "top_k 内权重比例不对: p={p}");
}

/// top_k 大于候选数、top_k=1 退化等边界。
#[test]
fn top_k_edge_cases() {
    // top_k 超过长度：不 panic，且全体都可被采到
    let mut s = Sampler::new(1.0, 999, 1.0, 3);
    let h = histogram(&mut s, &[0.0f32; 8], 800);
    assert_eq!(h.iter().sum::<usize>(), 800);
    assert!(h.iter().filter(|c| **c > 0).count() >= 6, "top_k 超长时不应截断: {h:?}");

    // top_k=1 等价 greedy
    let mut one = Sampler::new(1.0, 1, 1.0, 3);
    for _ in 0..50 {
        assert_eq!(one.sample(&[1.0f32, 9.0, 2.0]), 1);
    }
}

// ---------------------------------------------------------------------------
// top_p
// ---------------------------------------------------------------------------

/// 概率 0.5/0.3/0.15/0.05（logits = ln p）：
/// top_p=0.75 的核应恰好是 {0, 1}（0.5 < 0.75，0.8 >= 0.75 停）。
#[test]
fn top_p_nucleus_is_exact() {
    let logits: Vec<f32> = [0.5f64, 0.3, 0.15, 0.05]
        .iter()
        .map(|p| (p.ln()) as f32)
        .collect();

    let mut s = Sampler::new(1.0, 0, 0.75, 17);
    let h = histogram(&mut s, &logits, 600);
    assert_eq!(h[2] + h[3], 0, "top_p=0.75 不应采到核外 token: {h:?}");
    assert_eq!(h[0] + h[1], 600);
    assert!(h[0] > 0 && h[1] > 0, "核内两个 token 都应出现: {h:?}");

    // top_p=0.4 的核只有 {0}（0.5 >= 0.4 立刻停）
    let mut s2 = Sampler::new(1.0, 0, 0.4, 17);
    for _ in 0..300 {
        assert_eq!(s2.sample(&logits), 0);
    }
}

/// top_p >= 1 表示不截断（含「概率几乎全在第一个 token」的情形，
/// 此时朴素实现会因为前缀累积 >= 1.0 提前 break 而丢掉尾巴）。
#[test]
fn top_p_one_disables_truncation() {
    // 两个等概率 token：不截断就两个都该出现
    let mut eq = Sampler::new(1.0, 0, 1.0, 23);
    let h = histogram(&mut eq, &[0.0f32, 0.0], 200);
    assert!(h[0] > 0 && h[1] > 0, "top_p=1.0 下两个等概率 token 都应出现: {h:?}");

    // 长尾分布：尾巴（下标 8..16）概率约 3.6e-4，2 万次采样应出现若干次
    let logits: Vec<f32> = (0..16).map(|i| -(i as f32)).collect();
    let mut s = Sampler::new(1.0, 0, 1.0, 23);
    let h = histogram(&mut s, &logits, 20_000);
    assert!(h[8..].iter().sum::<usize>() > 0, "top_p=1.0 不应截断: {h:?}");
    // 头部仍然占绝大多数，说明不是「权重算错变成了均匀分布」
    assert!(h[0] > 8000, "头部概率异常: {h:?}");
}

/// top_p <= 0 退化为 top-1（保留第一个，即 argmax）。
#[test]
fn top_p_zero_keeps_only_argmax() {
    for top_p in [0.0f32, -0.5] {
        let mut s = Sampler::new(1.0, 0, top_p, 4);
        for _ in 0..100 {
            assert_eq!(s.sample(&[1.0f32, 3.0, 2.0]), 1, "top_p={top_p}");
        }
    }
}

// ---------------------------------------------------------------------------
// softmax 权重正确性（统计）
// ---------------------------------------------------------------------------

/// logits = [0,1,2,3]，softmax 概率 = [0.0321, 0.0871, 0.2369, 0.6439]。
/// 4000 次采样的频率必须落在 ~4σ 内 —— 只有权重真的按 softmax 算才能通过。
#[test]
fn weights_follow_softmax() {
    let logits = [0.0f32, 1.0, 2.0, 3.0];
    let mut s = Sampler::new(1.0, 0, 1.0, 2024);
    let n = 4000usize;
    let h = histogram(&mut s, &logits, n);

    let e: Vec<f64> = logits.iter().map(|x| (*x as f64).exp()).collect();
    let total: f64 = e.iter().sum();
    for (i, ei) in e.iter().enumerate() {
        let p = ei / total;
        let expect = p * n as f64;
        let sigma = (n as f64 * p * (1.0 - p)).sqrt();
        let d = (h[i] as f64 - expect).abs();
        assert!(
            d < 4.0 * sigma + 1.0,
            "token {i}: 实测 {} 次，期望 {expect:.1}（4σ={:.1}）",
            h[i],
            4.0 * sigma
        );
    }
}

/// 温度必须真的改变分布陡峭程度：低温更集中、高温更平。
#[test]
fn temperature_changes_concentration() {
    let logits = [0.0f32, 1.0, 2.0, 3.0];
    let cold = histogram(&mut Sampler::new(0.3, 0, 1.0, 77), &logits, 2000);
    let hot = histogram(&mut Sampler::new(3.0, 0, 1.0, 77), &logits, 2000);
    // 低温时最后一个 token 占比更高
    assert!(
        cold[3] > hot[3],
        "低温应更集中: cold[3]={} hot[3]={}",
        cold[3],
        hot[3]
    );
    assert!(cold[3] > 1900, "temp=0.3 时 argmax 占比应 >95%: {}", cold[3]);
    // 高温时尾部 token 也要出现
    assert!(hot[0] > 0, "temp=3.0 时最低 logit 也应出现");
}

// ---------------------------------------------------------------------------
// min_p（logit 阈值，语义对齐 llama.cpp：min_logit = max_logit + ln(min_p)）
// ---------------------------------------------------------------------------

/// min_p=0.05、p_max=0.9 → 阈值概率 0.045 → 只保留 p >= 0.045 的候选。
#[test]
fn min_p_filters_low_probability_tokens() {
    let probs = [0.9f64, 0.09, 0.009, 0.0009, 0.0001];
    let logits: Vec<f32> = probs.iter().map(|p| p.ln() as f32).collect();
    let p = SamplingParams {
        temp: 1.0,
        min_p: 0.05,
        ..Default::default()
    };
    let mut s = Sampler::from_params(p);
    let h = histogram(&mut s, &logits, 500);
    assert_eq!(h[2] + h[3] + h[4], 0, "min_p 之外的 token 不应出现: {h:?}");
    assert!(h[0] > 0 && h[1] > 0, "阈值内的两个 token 都应出现: {h:?}");
}

/// min_keep 保证集合不会被压到比它更小（llama.cpp "first token always matches"）。
#[test]
fn min_p_respects_min_keep() {
    // 阈值 = 5.0 + ln(0.9) = 4.895 → 只有 token 0 过线；min_keep=3 把 1、2 也留下。
    let logits = [5.0f32, 4.9, 4.8, 4.7];
    let p = SamplingParams {
        temp: 1.0,
        min_p: 0.9,
        min_keep: 3,
        ..Default::default()
    };
    let mut s = Sampler::from_params(p);
    let h = histogram(&mut s, &logits, 1200);
    assert!(
        h[0] > 0 && h[1] > 0 && h[2] > 0,
        "min_keep=3 应保留前 3 个: {h:?}"
    );
    assert_eq!(h[3], 0, "第 4 个应在阈值之外: {h:?}");
}

// ---------------------------------------------------------------------------
// 惩罚（repeat / presence / frequency）
// ---------------------------------------------------------------------------

/// 默认参数（repeat=1.0、presence=0、freq=0）下惩罚必须完全不生效。
#[test]
fn penalties_default_are_noop() {
    let logits: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05 - 1.0).collect();
    let mut a = Sampler::new(1.0, 10, 0.9, 5);
    let mut b = Sampler::from_params(SamplingParams {
        temp: 1.0,
        top_k: 10,
        top_p: 0.9,
        repeat_last_n: 64, // 窗口开着，但惩罚值都是中性
        seed: 5,
        ..Default::default()
    });
    for _ in 0..50 {
        let x = a.sample(&logits);
        a.accept(x);
        let y = b.sample(&logits);
        b.accept(y);
        assert_eq!(x, y);
    }
}

/// repeat_penalty：正 logit 除以它，非正 logit 乘以它（llama.cpp 同款修正）。
#[test]
fn repeat_penalty_discourages_repeats() {
    let logits = [3.0f32, 3.5];
    let p = SamplingParams {
        temp: 0.0, // greedy：结果完全确定
        repeat_penalty: 2.0,
        repeat_last_n: 64,
        ..Default::default()
    };
    let mut s = Sampler::from_params(p);
    assert_eq!(s.sample(&logits), 1, "无历史时选最大 logit");
    s.accept(1);
    assert_eq!(s.sample(&logits), 0, "token 1 被罚（3.5/2=1.75 < 3.0）后应改选 0");

    // 负数 logit 必须变小（乘 penalty），不能因为除法而变大
    let neg = [-5.0f32, -5.5];
    let mut g = Sampler::from_params(SamplingParams {
        temp: 0.0,
        repeat_penalty: 2.0,
        repeat_last_n: 64,
        ..Default::default()
    });
    g.accept(0); // token 0: -5.0 * 2 = -10.0
    assert_eq!(g.sample(&neg), 1, "被罚后 token 0 应比 token 1 更差");
}

/// presence 惩罚：出现过就减一个常数（与出现次数无关）。
#[test]
fn presence_penalty_subtracts_once() {
    let logits = [1.0f32, 1.0, 0.0];
    let mut s = Sampler::from_params(SamplingParams {
        temp: 0.0,
        presence_penalty: 1.5,
        repeat_last_n: 64,
        ..Default::default()
    });
    assert_eq!(s.sample(&logits), 0);
    s.accept(0);
    s.accept(0); // 出现两次，presence 也只扣一次
    assert_eq!(s.sample(&logits), 1, "token 0 被 presence 罚掉后应选 1");
}

/// frequency 惩罚：按出现次数线性扣分。
#[test]
fn frequency_penalty_scales_with_count() {
    let logits = [1.0f32, 1.0];
    let p = SamplingParams {
        temp: 0.0,
        frequency_penalty: 0.3,
        repeat_last_n: 64,
        ..Default::default()
    };
    let mut s = Sampler::from_params(p);
    s.accept(0);
    assert_eq!(s.sample(&logits), 1, "一次出现扣 0.3：1.0-0.3 < 1.0，应选 1");

    // 只出现一次时不该被扣两次
    let mut t = Sampler::from_params(p);
    t.accept(0);
    t.accept(1);
    assert_eq!(t.sample(&logits), 0, "两者都被扣一次后 token 0 与 token 1 同分，取小下标");
}

/// repeat_last_n 是滑动窗口：老 token 移出窗口后惩罚消失。
#[test]
fn repeat_last_n_evicts_old_tokens() {
    let logits = [3.0f32, 3.5];
    let p = SamplingParams {
        temp: 0.0,
        repeat_penalty: 2.0,
        repeat_last_n: 1,
        ..Default::default()
    };
    let mut s = Sampler::from_params(p);
    s.accept(1); // 窗口 = {1}
    assert_eq!(s.sample(&logits), 0, "token 1 在窗口内，被罚");
    s.accept(0); // 窗口 = {0}，1 被挤出
    assert_eq!(s.sample(&logits), 1, "token 1 已移出窗口，不再被罚");
}

/// restart() 必须清掉惩罚历史（新序列不能受上一段生成影响）。
#[test]
fn restart_clears_penalty_history() {
    let logits = [3.0f32, 3.5];
    let p = SamplingParams {
        temp: 0.0,
        repeat_penalty: 2.0,
        repeat_last_n: 64,
        ..Default::default()
    };
    let mut s = Sampler::from_params(p);
    s.accept(1);
    assert_eq!(s.sample(&logits), 0);
    s.restart();
    assert_eq!(s.sample(&logits), 1, "restart 后历史应清空");
}

// ---------------------------------------------------------------------------
// 组合与性能相关的语义
// ---------------------------------------------------------------------------

/// top_k + top_p + min_p 同时开启时结果必须落在 top_k 集合内。
#[test]
fn combined_truncation_stays_within_top_k() {
    let logits: Vec<f32> = (0..256).map(|i| 10.0 - (i as f32) * 0.1).collect();
    let p = SamplingParams {
        temp: 0.8,
        top_k: 20,
        top_p: 0.95,
        min_p: 0.02,
        seed: 7,
        ..Default::default()
    };
    let mut s = Sampler::from_params(p);
    for _ in 0..2000 {
        let t = s.sample(&logits);
        assert!(t < 20, "top_k=20 之外的 token 被采到: {t}");
    }
}

/// 惩罚开启时，top_k 作用在**惩罚后**的 logits 上（与 llama.cpp 的 chain 顺序一致）——
/// 所以被反复采到的 token 会被罚下去，让原本在 top_k 之外的 token 进来。
/// 这里只要求：不 panic、结果始终是合法下标。
#[test]
fn combined_params_with_penalties_are_consistent() {
    let logits: Vec<f32> = (0..256).map(|i| 10.0 - (i as f32) * 0.1).collect();
    let p = SamplingParams {
        temp: 0.8,
        top_k: 20,
        top_p: 0.95,
        min_p: 0.02,
        repeat_penalty: 1.1,
        presence_penalty: 0.1,
        frequency_penalty: 0.05,
        repeat_last_n: 16,
        seed: 7,
        ..Default::default()
    };
    let mut s = Sampler::from_params(p);
    for _ in 0..2000 {
        let t = s.sample(&logits);
        assert!((t as usize) < logits.len(), "越界下标: {t}");
        s.accept(t);
    }
}

/// 大词表（248320）下的两条路径：全量 softmax 与 top_k/top_p 截断。
#[test]
fn large_vocab_sampling_is_sane() {
    // 背景 logit 取 -30：每个背景 token 概率 ~1e-13，合计可以忽略，
    // 于是两个高 logit token 应占几乎全部概率质量。
    let mut logits = vec![-30.0f32; 248_320];
    logits[123_456] = 5.0;
    logits[7] = 4.0;

    // (a) 完全不截断：走一次 O(n) softmax 累积
    let mut s = Sampler::new(1.0, 0, 1.0, 1);
    let mut hits = 0;
    for _ in 0..400 {
        let t = s.sample(&logits);
        assert!((t as usize) < logits.len());
        if t == 123_456 || t == 7 {
            hits += 1;
        }
    }
    assert!(hits > 380, "两个高 logit token 应占绝大多数，实际 {hits}/400");

    // (b) top_k=40 + top_p=0.95：必须只落在那两个 token 上
    let p = SamplingParams {
        temp: 1.0,
        top_k: 40,
        top_p: 0.95,
        ..Default::default()
    };
    let mut t = Sampler::from_params(p);
    for _ in 0..400 {
        let id = t.sample(&logits);
        assert!(id == 123_456 || id == 7, "top_k/top_p 之外被采到: {id}");
    }
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

#[test]
fn timing_rates() {
    let t = Timing {
        ttft_ms: 100.0,
        prompt_tokens: 100,
        prompt_cached: 0,
        prompt_ms: 200.0,
        gen_tokens: 50,
        gen_ms: 1000.0,
        ..Default::default()
    };
    assert!((t.prompt_rate() - 500.0).abs() < 1e-6);
    assert!((t.eval_rate() - 50.0).abs() < 1e-6);
}

/// 未产生 token / 未计时时不能出 inf/NaN。
#[test]
fn timing_rates_guard_division_by_zero() {
    let t = Timing {
        ttft_ms: 0.0,
        prompt_tokens: 0,
        prompt_cached: 0,
        prompt_ms: 0.0,
        gen_tokens: 0,
        gen_ms: 0.0,
        ..Default::default()
    };
    assert!(t.prompt_rate().is_finite(), "prompt_rate={}", t.prompt_rate());
    assert!(t.eval_rate().is_finite(), "eval_rate={}", t.eval_rate());
    assert_eq!(t.prompt_rate(), 0.0);
    assert_eq!(t.eval_rate(), 0.0);

    // 只有分母为 0（有 token 但没耗时）也不能是 inf
    let t2 = Timing {
        ttft_ms: 0.0,
        prompt_tokens: 10,
        prompt_cached: 0,
        prompt_ms: 0.0,
        gen_tokens: 10,
        gen_ms: 0.0,
        ..Default::default()
    };
    assert!(t2.prompt_rate().is_finite() && t2.eval_rate().is_finite());
}

/// rate 与 tokens/ms 的定义自洽，并且是 tok/s 而不是 tok/ms。
#[test]
fn timing_rate_units() {
    let t = Timing {
        ttft_ms: 12.5,
        prompt_tokens: 512,
        prompt_cached: 0,
        prompt_ms: 250.0,
        gen_tokens: 128,
        gen_ms: 4000.0,
        ..Default::default()
    };
    assert!((t.prompt_rate() - 2048.0).abs() < 1e-9, "{}", t.prompt_rate());
    assert!((t.eval_rate() - 32.0).abs() < 1e-9, "{}", t.eval_rate());
    // 反推：rate * ms / 1000 == tokens
    assert!(((t.prompt_rate() * t.prompt_ms / 1000.0) - 512.0).abs() < 1e-9);
    assert!(((t.eval_rate() * t.gen_ms / 1000.0) - 128.0).abs() < 1e-9);
}

/// Timing 需要被序列化进 API 响应（Task 12），字段名必须稳定。
#[test]
fn timing_serializes_with_stable_field_names() {
    let t = Timing {
        ttft_ms: 1.0,
        prompt_tokens: 2,
        prompt_cached: 0,
        prompt_ms: 3.0,
        gen_tokens: 4,
        gen_ms: 5.0,
        ..Default::default()
    };
    let json = serde_json::to_string(&t).unwrap();
    for k in [
        "ttft_ms",
        "prompt_tokens",
        "prompt_ms",
        "gen_tokens",
        "gen_ms",
    ] {
        assert!(json.contains(&format!("\"{k}\"")), "缺少字段 {k}: {json}");
    }
}
