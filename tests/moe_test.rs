// tests/moe_test.rs — Task 5: MoE FFN graph verification
//
// 验证三件事：
//   1. build_moe_ffn 的数值正确性 —— 与一份独立的 Rust 参考实现（逐 token
//      逐专家直接算，不经过 ggml 图）比对；
//   2. CPU 与 CUDA 后端在同一份输入上的一致性；
//   3. 图结构 / 输出 shape。
//
// 关键点：CPU 与 CUDA 必须喂入**完全相同**的字节。早先的实现每次 run_moe 都用
// `rand::thread_rng()` 重新生成权重和输入，两个后端算的其实是两个不同的随机模型，
// 那个 "CPU vs CUDA max diff 17" 的失败与后端无关。这里改成一次性生成、按 backend 复制。

use sqwen::backend::ctx::{BackendBuffer, BackendKind, GgmlCtx, GraphBuilder};
use sqwen::backend::ffi::GgmlType;
use sqwen::model::moe::build_moe_ffn;
use sqwen::model::weights::{LayerWeights, TensorRef};

/// 与 llama.cpp `build_moe_ffn` 里 clamp 用的同一个常量（F16 最小正规数）。
const EPSILON: f32 = 6.103515625e-5;

struct Dims {
    n_embd: i64,
    n_ff: i64,
    n_expert: i64,
    n_tokens: i64,
    top_k: i64,
}

fn dims(n_tokens: i64) -> Dims {
    Dims {
        n_embd: 32,
        n_ff: 16,
        n_expert: 8,
        n_tokens,
        top_k: 2,
    }
}

// ===========================================================================
// 确定性数据
// ===========================================================================

/// 确定性伪随机（LCG），取值域 `[-1, 1)`。
///
/// 用它而不是 `thread_rng()`，是为了让 CPU / CUDA / 参考实现看到同一份数据，
/// 也让失败可以复现。
fn pseudo(len: i64, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32) / 8388608.0 - 1.0
        })
        .collect()
}

/// 一次实验用到的全部输入与权重（按 ggml 的行优先布局拍平）。
struct MoeData {
    /// [n_embd, n_tokens]
    h: Vec<f32>,
    /// [n_embd, n_expert]
    gate_inp: Vec<f32>,
    /// [n_embd, n_ff, n_expert]
    up_exps: Vec<f32>,
    /// [n_embd, n_ff, n_expert]
    gate_exps: Vec<f32>,
    /// [n_ff, n_embd, n_expert]
    down_exps: Vec<f32>,
}

fn make_data_seeded(d: &Dims, seed: u32) -> MoeData {
    let (ne, nf, nx, nt) = (d.n_embd, d.n_ff, d.n_expert, d.n_tokens);
    MoeData {
        h: pseudo(ne * nt, seed ^ 0x11),
        gate_inp: pseudo(ne * nx, seed ^ 0x22),
        up_exps: pseudo(ne * nf * nx, seed ^ 0x33),
        gate_exps: pseudo(ne * nf * nx, seed ^ 0x44),
        down_exps: pseudo(nf * ne * nx, seed ^ 0x55),
    }
}

/// 每个 token 的专家概率：`logits = gate_inpᵀ h`，再逐 token softmax。
///
/// 与图里的 softmax 门控（`ggml_soft_max`，减最大值后归一化）一致。
fn expert_probs(data: &MoeData, d: &Dims) -> Vec<Vec<f32>> {
    let (ne, nx, nt) = (
        d.n_embd as usize,
        d.n_expert as usize,
        d.n_tokens as usize,
    );
    (0..nt)
        .map(|t| {
            let mut p: Vec<f32> = (0..nx)
                .map(|e| {
                    (0..ne)
                        .map(|c| data.gate_inp[c + e * ne] * data.h[c + t * ne])
                        .sum()
                })
                .collect();
            let mx = p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            for v in p.iter_mut() {
                *v = (*v - mx).exp();
                denom += *v;
            }
            for v in p.iter_mut() {
                *v /= denom;
            }
            p
        })
        .collect()
}

/// 所有 token 中「第 k 名与第 k+1 名概率」的最小间隔。
fn topk_min_gap(probs: &[Vec<f32>], d: &Dims) -> f32 {
    let top_k = d.top_k as usize;
    probs
        .iter()
        .map(|p| {
            let mut order: Vec<usize> = (0..p.len()).collect();
            order.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap());
            p[order[top_k - 1]] - p[order[top_k]]
        })
        .fold(f32::INFINITY, f32::min)
}

/// 生成合成数据，并保证 top-k 边界足够清晰。
///
/// 如果第 k 名和第 k+1 名的概率几乎相等（在 64 token × 8 专家下很容易撞上，
/// 实测遇到过 1.1e-7），f32 的舍入抖动就可能让参考实现和 ggml 选中不同的专家，
/// 于是比对失败变成一个与实现无关的偶发问题。这里按 seed 搜索，直到间隔 > 1e-4。
fn make_data(d: &Dims) -> MoeData {
    for attempt in 0..512u32 {
        let data = make_data_seeded(d, 0x9e37_79b9u32.wrapping_mul(attempt + 1));
        if topk_min_gap(&expert_probs(&data, d), d) > 1e-4 {
            return data;
        }
    }
    panic!("未能在 512 次尝试内找到 top-k 间隔 > 1e-4 的合成数据");
}

// ===========================================================================
// 图构建与执行
// ===========================================================================

fn make_synthetic_weights(ctx: &GgmlCtx, d: &Dims) -> LayerWeights {
    let (ne, nf, nx) = (d.n_embd, d.n_ff, d.n_expert);
    let ffn_gate_inp = unsafe { TensorRef::from_ptr(ctx.new_tensor_2d(GgmlType::F32, ne, nx).ptr()) };
    // up/gate_exps: [K=n_embd, R=n_ff, NE=n_expert]
    let ffn_up_exps = unsafe { TensorRef::from_ptr(ctx.new_tensor_3d(GgmlType::F32, ne, nf, nx).ptr()) };
    let ffn_gate_exps = unsafe { TensorRef::from_ptr(ctx.new_tensor_3d(GgmlType::F32, ne, nf, nx).ptr()) };
    // down_exps: [K=n_ff, R=n_embd, NE=n_expert]（ne[0] = n_ff）
    let ffn_down_exps = unsafe { TensorRef::from_ptr(ctx.new_tensor_3d(GgmlType::F32, nf, ne, nx).ptr()) };

    LayerWeights {
        ffn_gate_inp,
        ffn_up_exps,
        ffn_gate_exps,
        ffn_down_exps,
        ..LayerWeights::EMPTY
    }
}

/// 在指定后端上跑一次 build_moe_ffn，返回 [n_embd, n_tokens] 的 f32。
fn run_moe(kind: BackendKind, data: &MoeData, d: &Dims) -> Vec<f32> {
    let ctx = GgmlCtx::new_no_alloc().expect("GgmlCtx::new_no_alloc");
    let w = make_synthetic_weights(&ctx, d);
    let h = ctx.new_tensor_2d(GgmlType::F32, d.n_embd, d.n_tokens);

    let mut g = GraphBuilder::new(&ctx);
    let moe_out = build_moe_ffn(
        &ctx, &mut g, &w, &h, d.n_expert, d.top_k, d.n_ff, d.n_embd, d.n_tokens,
    );

    let buf = BackendBuffer::from_ctx(&ctx, kind).expect("BackendBuffer::from_ctx");

    // 必须走 backend API 上传/回读：CUDA 上 tensor->data 是设备地址，直接 memcpy
    // 得不到正确结果。
    buf.set_tensor_data(&h, bytemuck::cast_slice(&data.h));
    buf.set_tensor_data(&ctx.wrap(w.ffn_gate_inp.as_ptr()), bytemuck::cast_slice(&data.gate_inp));
    buf.set_tensor_data(&ctx.wrap(w.ffn_up_exps.as_ptr()), bytemuck::cast_slice(&data.up_exps));
    buf.set_tensor_data(&ctx.wrap(w.ffn_gate_exps.as_ptr()), bytemuck::cast_slice(&data.gate_exps));
    buf.set_tensor_data(&ctx.wrap(w.ffn_down_exps.as_ptr()), bytemuck::cast_slice(&data.down_exps));

    g.expand(&moe_out);
    g.compute(&buf).expect("graph compute");

    let bytes = buf.get_tensor_data(&moe_out);
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

// ===========================================================================
// 独立参考实现
// ===========================================================================

/// 逐 token、逐专家直接计算 MoE FFN，不经过任何 ggml 算子。
///
/// 与 llama.cpp `build_moe_ffn`（qwen35moe 的参数：softmax 门控、norm_w = true、
/// expert_weights_scale = 0 即不缩放、SILU 激活）语义一致：
///
///     probs = softmax(gate_inpᵀ h)
///     sel   = top-k(probs)
///     w     = probs[sel] / max(Σ probs[sel], eps)
///     每个专家: act = silu(gate_expsᵀ h) * (up_expsᵀ h)
///               out += w * (down_expsᵀ act)
fn moe_reference(data: &MoeData, probs: &[Vec<f32>], d: &Dims) -> Vec<f32> {
    let (ne, nf, nt) = (
        d.n_embd as usize,
        d.n_ff as usize,
        d.n_tokens as usize,
    );
    let top_k = d.top_k as usize;

    let mut out = vec![0.0f32; ne * nt];

    for t in 0..nt {
        let p = &probs[t];
        let mut order: Vec<usize> = (0..p.len()).collect();
        order.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap());
        let sel = &order[..top_k];

        let wsum = sel.iter().map(|&e| p[e]).sum::<f32>().max(EPSILON);

        for &e in sel {
            let w = p[e] / wsum;

            let mut act = vec![0.0f32; nf];
            for r in 0..nf {
                let mut up = 0.0f32;
                let mut gt = 0.0f32;
                for c in 0..ne {
                    let base = c + r * ne + e * ne * nf;
                    up += data.up_exps[base] * data.h[c + t * ne];
                    gt += data.gate_exps[base] * data.h[c + t * ne];
                }
                let silu = gt / (1.0 + (-gt).exp());
                act[r] = silu * up;
            }

            for c2 in 0..ne {
                let mut acc = 0.0f32;
                for r in 0..nf {
                    acc += data.down_exps[r + c2 * nf + e * nf * ne] * act[r];
                }
                out[c2 + t * ne] += w * acc;
            }
        }
    }

    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn gpu_free_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next()?.trim().parse::<u64>().ok()
}

fn gpu_available() -> bool {
    use std::sync::OnceLock;
    static AVAIL: OnceLock<bool> = OnceLock::new();
    *AVAIL.get_or_init(|| match gpu_free_mib() {
        Some(m) if m >= 2048 => true,
        Some(m) => {
            eprintln!("[moe_test] SKIP GPU: 仅 {} MiB 空闲 (< 2048 MiB)", m);
            false
        }
        None => {
            eprintln!("[moe_test] SKIP GPU: nvidia-smi 不可用");
            false
        }
    })
}

// ===========================================================================
// 测试用例
// ===========================================================================

/// CPU 图 vs 独立参考实现 —— 验证 build_moe_ffn 的数值本身对不对。
#[test]
fn moe_matches_reference_cpu() {
    let d = dims(4);
    let data = make_data(&d);
    let probs = expert_probs(&data, &d);

    let got = run_moe(BackendKind::Cpu, &data, &d);
    let want = moe_reference(&data, &probs, &d);

    assert_eq!(got.len(), (d.n_embd * d.n_tokens) as usize);
    for &v in &got {
        assert!(v.is_finite(), "CPU MoE 输出非有限值: {}", v);
    }
    let diff = max_abs_diff(&got, &want);
    eprintln!("[moe_test] CPU vs reference: max diff {diff:e}");
    assert!(diff < 1e-4, "CPU 与参考实现不一致: max diff {diff} >= 1e-4");
}

/// 单 token：CPU / CUDA / 参考实现三者一致。
#[test]
fn moe_single_token_cpu_cuda_match() {
    let d = dims(1);
    let data = make_data(&d);
    let probs = expert_probs(&data, &d);
    let want = moe_reference(&data, &probs, &d);

    let cpu = run_moe(BackendKind::Cpu, &data, &d);
    assert_eq!(cpu.len(), (d.n_embd * d.n_tokens) as usize);
    for &v in &cpu {
        assert!(v.is_finite(), "CPU MoE 输出非有限值: {}", v);
    }
    let diff_ref = max_abs_diff(&cpu, &want);
    eprintln!("[moe_test] single token CPU vs reference: max diff {diff_ref:e}");
    assert!(diff_ref < 1e-4, "CPU 与参考实现不一致: {diff_ref}");

    if !gpu_available() {
        return;
    }
    let cuda = run_moe(BackendKind::Cuda(0), &data, &d);
    let diff = max_abs_diff(&cpu, &cuda);
    eprintln!("[moe_test] max diff CPU vs CUDA: {diff:e}");
    assert!(diff < 1e-3, "CPU/CUDA mismatch: max diff {diff} >= 1e-3");
    let diff_ref = max_abs_diff(&cuda, &want);
    eprintln!("[moe_test] single token CUDA vs reference: max diff {diff_ref:e}");
    assert!(diff_ref < 1e-4, "CUDA 与参考实现不一致: {diff_ref}");
}

/// 64 token：同一个失败用例，但这次两端吃的是同一份数据。
#[test]
fn moe_chunk_64_cpu_cuda_match() {
    let d = dims(64);
    let data = make_data(&d);
    let probs = expert_probs(&data, &d);
    let want = moe_reference(&data, &probs, &d);

    let cpu = run_moe(BackendKind::Cpu, &data, &d);
    assert_eq!(cpu.len(), (d.n_embd * d.n_tokens) as usize);
    for &v in &cpu {
        assert!(v.is_finite(), "CPU MoE 输出非有限值: {}", v);
    }
    let diff_ref = max_abs_diff(&cpu, &want);
    eprintln!("[moe_test] chunk64 CPU vs reference: max diff {diff_ref:e}");
    assert!(diff_ref < 1e-4, "CPU 与参考实现不一致: {diff_ref}");

    if !gpu_available() {
        return;
    }
    let cuda = run_moe(BackendKind::Cuda(0), &data, &d);
    let diff = max_abs_diff(&cpu, &cuda);
    eprintln!("[moe_test] chunk64 max diff CPU vs CUDA: {diff:e}");
    assert!(diff < 1e-3, "CPU/CUDA mismatch: max diff {diff} >= 1e-3");
    let diff_ref = max_abs_diff(&cuda, &want);
    eprintln!("[moe_test] chunk64 CUDA vs reference: max diff {diff_ref:e}");
    assert!(diff_ref < 1e-4, "CUDA 与参考实现不一致: {diff_ref}");
}

/// 输出 shape：[n_embd, n_tokens]。
#[test]
fn moe_output_shape_correct() {
    let d = dims(4);
    let ctx = GgmlCtx::new_no_alloc().expect("GgmlCtx::new_no_alloc");
    let w = make_synthetic_weights(&ctx, &d);
    let h = ctx.new_tensor_2d(GgmlType::F32, d.n_embd, d.n_tokens);

    let mut g = GraphBuilder::new(&ctx);
    let moe_out = build_moe_ffn(
        &ctx, &mut g, &w, &h, d.n_expert, d.top_k, d.n_ff, d.n_embd, d.n_tokens,
    );

    assert_eq!(moe_out.ne()[0], d.n_embd);
    assert_eq!(moe_out.ne()[1], d.n_tokens);
}

/// 同一份输入跑两次，输出必须逐字节相同。
#[test]
fn moe_deterministic_same_input() {
    let d = dims(4);
    let data = make_data(&d);

    // 早先这个用例从不调用 compute()，比较的是两块没算过的缓冲区，恒等成立。
    let a = run_moe(BackendKind::Cpu, &data, &d);
    let b = run_moe(BackendKind::Cpu, &data, &d);
    assert_eq!(a, b, "同一输入两次运行结果不同");

    // 顺带确认它确实是非平凡的（不是全 0 / 常量）
    assert!(a.iter().any(|v| v.abs() > 1e-6), "输出恒为零，用例无意义");
}
