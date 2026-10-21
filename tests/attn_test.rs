// tests/attn_test.rs — Task 7: 全注意力层 + Q4_0 KV cache
//
// 验证三件事：
//   1. build_attn_layer 的数值正确性 —— 与一份独立的 Rust 参考实现比对
//      （逐 token 手算投影 / RMS norm / IMROPE / softmax / GQA / 门控 / 输出投影，
//       不经过任何 ggml 算子；K cache 的量化用 ggml_quantize_chunk 复现）；
//   2. CPU 与 CUDA 后端在同一份输入上的一致性；
//   3. q4_0 量化往返的误差上界。
//
// 合成配置（对应计划里的 n_head=4 / n_head_kv=1 / head_dim=32）：
//   n_embd = 128, n_head = 4, n_head_kv = 1, head_dim = 32,
//   n_rot = 16, sections = [3,3,2,0]（和 = n_rot/2 = 8）, kv_cap = 32

use sqwen::backend::ctx::{BackendBuffer, BackendKind, GgmlCtx, GraphBuilder, Tensor};
use sqwen::backend::ffi::{self, GgmlType};
use sqwen::engine::kv::KvCache;
use sqwen::model::arch::Qwen35MoeArch;
use sqwen::model::attn::{build_attn_layer, RopeParams};
use sqwen::model::weights::{LayerWeights, TensorRef};

// FA 只支持 head_dim ∈ {40,64,72,80,96,112,128,192,256,...}（ggml-cuda/fattn.cu 的 switch），
    // 所以这里用 64（生产模型是 256）。32 会让 FA 直接 GGML_ABORT。
    const HEAD_DIM: i64 = 64;
const N_HEAD: i64 = 4;
const N_HEAD_KV: i64 = 1;
const N_EMBD: i64 = HEAD_DIM * N_HEAD; // 128
const N_ROT: i32 = 16;
const SECTIONS: [i32; 4] = [3, 3, 2, 0];
const FREQ_BASE: f32 = 1e7;
const KV_CAP: usize = 32;
const NORM_EPS: f32 = 1e-6;

fn arch_test() -> Qwen35MoeArch {
    Qwen35MoeArch {
        n_embd: N_EMBD as u32,
        ssm_d_inner: 32,
        ssm_d_state: 8,
        ssm_n_group: 2,
        ssm_dt_rank: 4,
        ssm_d_conv: 4,
        full_attn_interval: 1,
        rope_dim_count: N_ROT as u32,
        rope_freq_base: FREQ_BASE,
        rope_sections: SECTIONS,
        norm_rms_eps: NORM_EPS,
        expert_ffn_len: 0,
        shared_expert_ffn_len: 0,
        n_layer_nextn: 0,
        n_ctx_train: 262144,
        head_dim: HEAD_DIM as u32,
        n_head: N_HEAD as u32,
        n_head_kv: N_HEAD_KV as u32,
        n_expert: 0,
        n_expert_used: 0,
        n_layer: 1,
    }
}

fn rope_params() -> RopeParams {
    RopeParams::text(N_ROT, SECTIONS, FREQ_BASE, 4096)
}

// ===========================================================================
// 确定性数据
// ===========================================================================

fn pseudo(len: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32) / 8388608.0 - 1.0
        })
        .collect()
}

struct Weights {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

fn make_weights() -> Weights {
    Weights {
        q: pseudo((N_EMBD * 2 * HEAD_DIM * N_HEAD) as usize, 1),
        k: pseudo((N_EMBD * HEAD_DIM * N_HEAD_KV) as usize, 2),
        v: pseudo((N_EMBD * HEAD_DIM * N_HEAD_KV) as usize, 3),
        o: pseudo((N_HEAD * HEAD_DIM * N_EMBD) as usize, 4),
        q_norm: pseudo(HEAD_DIM as usize, 5),
        k_norm: pseudo(HEAD_DIM as usize, 6),
    }
}

fn make_layer_weights(ctx: &GgmlCtx) -> LayerWeights {
    let q = ctx.new_tensor_2d(GgmlType::F32, N_EMBD, 2 * HEAD_DIM * N_HEAD);
    let k = ctx.new_tensor_2d(GgmlType::F32, N_EMBD, HEAD_DIM * N_HEAD_KV);
    let v = ctx.new_tensor_2d(GgmlType::F32, N_EMBD, HEAD_DIM * N_HEAD_KV);
    let o = ctx.new_tensor_2d(GgmlType::F32, N_HEAD * HEAD_DIM, N_EMBD);
    let q_norm = ctx.new_tensor(GgmlType::F32, &[HEAD_DIM]);
    let k_norm = ctx.new_tensor(GgmlType::F32, &[HEAD_DIM]);
    LayerWeights {
        attn_q: unsafe { TensorRef::from_ptr(q.ptr()) },
        attn_k: unsafe { TensorRef::from_ptr(k.ptr()) },
        attn_v: unsafe { TensorRef::from_ptr(v.ptr()) },
        attn_output: unsafe { TensorRef::from_ptr(o.ptr()) },
        attn_q_norm: unsafe { TensorRef::from_ptr(q_norm.ptr()) },
        attn_k_norm: unsafe { TensorRef::from_ptr(k_norm.ptr()) },
        ..LayerWeights::EMPTY
    }
}

fn upload_weights(buf: &BackendBuffer, ctx: &GgmlCtx, w: &LayerWeights, d: &Weights) {
    buf.set_tensor_data(&w.attn_q.get(ctx), bytemuck::cast_slice(&d.q));
    buf.set_tensor_data(&w.attn_k.get(ctx), bytemuck::cast_slice(&d.k));
    buf.set_tensor_data(&w.attn_v.get(ctx), bytemuck::cast_slice(&d.v));
    buf.set_tensor_data(&w.attn_output.get(ctx), bytemuck::cast_slice(&d.o));
    buf.set_tensor_data(&w.attn_q_norm.get(ctx), bytemuck::cast_slice(&d.q_norm));
    buf.set_tensor_data(&w.attn_k_norm.get(ctx), bytemuck::cast_slice(&d.k_norm));
}

fn pos_tensor(ctx: &GgmlCtx, n_tokens: i64) -> Tensor<'_> {
    ctx.new_tensor_1d_i32(4 * n_tokens)
}

/// IMROPE 位置：4 段（t/h/w/e）取同一个 token 位置。
fn pos_bytes(start: i64, n_tokens: i64) -> Vec<u8> {
    let mut v = Vec::with_capacity((4 * n_tokens) as usize * 4);
    for _ in 0..4 {
        for t in 0..n_tokens {
            v.extend_from_slice(&((start + t) as i32).to_ne_bytes());
        }
    }
    v
}

fn get_f32(buf: &BackendBuffer, t: &Tensor<'_>) -> Vec<f32> {
    let b = buf.get_tensor_data(t);
    bytemuck::cast_slice::<u8, f32>(&b).to_vec()
}

// ===========================================================================
// 独立参考实现
// ===========================================================================

/// IMROPE 参考（逐行对照 ggml-cpu/ops.cpp ggml_mrope_cache_init + rotate_pairs）。
///
/// `a` 的布局是 [head_dim, n_head, n_tokens]（dim0 连续）。只有前 `n_rot` 个维度
/// 参与旋转，旋转对是 (i, i + n_rot/2)。
#[allow(clippy::too_many_arguments)]
fn rope_ref(
    a: &[f32],
    head_dim: usize,
    n_head: usize,
    n_tokens: usize,
    pos: &[i32],
    sections: [i32; 4],
    freq_base: f32,
    n_rot: usize,
) -> Vec<f32> {
    let n_dims = n_rot;
    let mut out = a.to_vec();
    let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
    let sect_dims: i32 = sections.iter().sum();

    for t in 0..n_tokens {
        let p_t = pos[t] as f32;
        let p_h = pos[t + n_tokens] as f32;
        let p_w = pos[t + n_tokens * 2] as f32;
        let p_e = pos[t + n_tokens * 3] as f32;
        let (mut th_t, mut th_h, mut th_w, mut th_e) = (p_t, p_h, p_w, p_e);

        let mut cache = vec![0f32; n_dims];
        for i0 in (0..n_dims).step_by(2) {
            let sector = ((i0 / 2) as i32) % sect_dims;
            let theta = if sector % 3 == 1 && sector < 3 * sections[1] {
                th_h
            } else if sector % 3 == 2 && sector < 3 * sections[2] {
                th_w
            } else if sector % 3 == 0 && sector < 3 * sections[0] {
                th_t
            } else {
                th_e
            };
            cache[i0] = theta.cos();
            cache[i0 + 1] = theta.sin();
            th_t *= theta_scale;
            th_w *= theta_scale;
            th_h *= theta_scale;
            th_e *= theta_scale;
        }

        for h in 0..n_head {
            let base = h * head_dim + t * head_dim * n_head;
            for i0 in (0..n_dims).step_by(2) {
                let ic = i0 / 2;
                let (ct, st) = (cache[i0], cache[i0 + 1]);
                let x0 = a[base + ic];
                let x1 = a[base + ic + n_dims / 2];
                out[base + ic] = x0 * ct - x1 * st;
                out[base + ic + n_dims / 2] = x0 * st + x1 * ct;
            }
        }
    }
    out
}

/// 用 ggml 自己的量化器把 f32 压成 q4_0，再逐块反量化（模拟 K cache 的读写）。
fn quantize_dequant_q4_0(data: &[f32], n_per_row: usize) -> Vec<f32> {
    assert_eq!(n_per_row % 32, 0);
    let nrows = data.len() / n_per_row;
    let row_size = unsafe { ffi::ggml_row_size(GgmlType::Q4_0 as i32, n_per_row as i64) };
    let mut bytes = vec![0u8; row_size * nrows];
    let n = unsafe {
        ffi::ggml_quantize_chunk(
            GgmlType::Q4_0 as i32,
            data.as_ptr(),
            bytes.as_mut_ptr() as *mut std::ffi::c_void,
            0,
            nrows as i64,
            n_per_row as i64,
            std::ptr::null(),
        )
    };
    assert_eq!(n, bytes.len());
    dequant_q4_0(&bytes, n_per_row, nrows)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let man = (bits & 0x3ff) as u32;
    let f = if exp == 0 {
        if man == 0 {
            0.0
        } else {
            (man as f32) * 2f32.powi(-24)
        }
    } else if exp == 0x1f {
        if man == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + man as f32 / 1024.0) * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 {
        -f
    } else {
        f
    }
}

/// q4_0 反量化（块布局：f16 d + 16 字节 nibble）。
fn dequant_q4_0(bytes: &[u8], n_per_row: usize, nrows: usize) -> Vec<f32> {
    let blocks = n_per_row / 32;
    let mut out = vec![0f32; n_per_row * nrows];
    let mut off = 0usize;
    for r in 0..nrows {
        for b in 0..blocks {
            let d = f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]));
            let qs = &bytes[off + 2..off + 18];
            for j in 0..16 {
                let lo = (qs[j] & 0x0f) as i32 - 8;
                let hi = (qs[j] >> 4) as i32 - 8;
                out[r * n_per_row + b * 32 + j] = lo as f32 * d;
                out[r * n_per_row + b * 32 + j + 16] = hi as f32 * d;
            }
            off += 18;
        }
    }
    out
}

fn rms_norm_row(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n;
    let denom = (ms + eps).sqrt();
    x.iter().map(|v| v / denom).collect()
}

/// 一次注意力层的完整参考计算。
///
/// `k_cache` / `v_cache` 是**已经写入 cache 的** K/V（布局 [head_dim, n_head_kv, n_kv]，
/// 其中 K 已按 q4_0 往返），代表图的读路径。
#[allow(clippy::too_many_arguments)]
fn attn_ref(
    w: &Weights,
    n_tokens: usize,
    q_rope: &[f32],      // [head_dim, n_head, n_tokens]
    gate: &[f32],        // [head_dim, n_head, n_tokens]
    k_cache: &[f32],     // [head_dim, n_head_kv, n_kv]
    v_cache: &[f32],     // [head_dim, n_head_kv, n_kv]
    n_kv: usize,
    start: usize,        // 本步第一个 query 的绝对位置
    output_scale: f32,
) -> Vec<f32> {
    let hd = HEAD_DIM as usize;
    let nh = N_HEAD as usize;
    let nkv = N_HEAD_KV as usize;
    let ne = N_EMBD as usize;

    // scores[i, t, h] = Σ_d k_cache[d, h % nkv, i] * q_rope[d, h, t] * scale
    // 注意 ggml 的 GQA 广播是 block 重复：kv_head = h / (nh/nkv)；n_head_kv = 1 时恒为 0。
    let group = nh / nkv;
    let mut out = vec![0f32; ne * n_tokens];
    for t in 0..n_tokens {
        for head in 0..nh {
            let kv_head = head / group;
            let mut scores = vec![0f32; n_kv];
            for i in 0..n_kv {
                let mut acc = 0f32;
                for d in 0..hd {
                    let kq = k_cache[d + hd * (kv_head + nkv * i)];
                    let qq = q_rope[d + hd * (head + nh * t)];
                    acc += kq * qq;
                }
                let allowed = i <= start + t;
                scores[i] = if allowed { acc * output_scale } else { f32::NEG_INFINITY };
            }
            // softmax
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0f32;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                denom += *s;
            }
            for s in scores.iter_mut() {
                *s /= denom;
            }
            // out[d, head, t] = Σ_i v_cache[d, kv_head, i] * p[i]
            for d in 0..hd {
                let mut acc = 0f32;
                for i in 0..n_kv {
                    acc += v_cache[d + hd * (kv_head + nkv * i)] * scores[i];
                }
                // 门控：cur[c, t] = out[d, head, t] * sigmoid(gate[d, head, t])，c = d + head*hd
                let c = d + head * hd;
                out[c + ne * t] = acc * (1.0 / (1.0 + (-gate[d + hd * (head + nh * t)]).exp()));
            }
        }
    }

    // 输出投影 y[e, t] = Σ_c o[c, e] * gated[c, t]
    let mut y = vec![0f32; ne * n_tokens];
    for t in 0..n_tokens {
        for e in 0..ne {
            let mut acc = 0f32;
            for c in 0..ne {
                acc += w.o[c + ne * e] * out[c + ne * t];
            }
            y[e + ne * t] = acc;
        }
    }
    y
}

/// 参考的投影 + 逐头 RMS norm + RoPE，返回 (q_rope, gate, k_rope, v)。
#[allow(clippy::type_complexity)]
fn proj_ref(
    h: &[f32],
    w: &Weights,
    n_tokens: usize,
    pos: &[i32],
    start: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let hd = HEAD_DIM as usize;
    let nh = N_HEAD as usize;
    let nkv = N_HEAD_KV as usize;
    let ne = N_EMBD as usize;

    // q_full[e2, t] = Σ_e w.q[e + e2*ne] * h[e + t*ne]
    let q_full_len = 2 * hd * nh;
    let k_len = hd * nkv;
    let mut q = vec![0f32; hd * nh * n_tokens];
    let mut gate = vec![0f32; hd * nh * n_tokens];
    let mut k = vec![0f32; hd * nkv * n_tokens];
    let mut v = vec![0f32; hd * nkv * n_tokens];

    for t in 0..n_tokens {
        let mut qf = vec![0f32; q_full_len];
        for e2 in 0..q_full_len {
            let mut acc = 0f32;
            for e in 0..ne {
                acc += w.q[e + e2 * ne] * h[e + t * ne];
            }
            qf[e2] = acc;
        }
        // 拆 Q / gate：每 head 一段 [q(hd), gate(hd)]
        for head in 0..nh {
            for d in 0..hd {
                q[d + hd * (head + nh * t)] = qf[head * 2 * hd + d];
                gate[d + hd * (head + nh * t)] = qf[head * 2 * hd + hd + d];
            }
        }
        for e2 in 0..k_len {
            let mut acc = 0f32;
            for e in 0..ne {
                acc += w.k[e + e2 * ne] * h[e + t * ne];
            }
            k[e2 + k_len * t] = acc;
        }
        for e2 in 0..k_len {
            let mut acc = 0f32;
            for e in 0..ne {
                acc += w.v[e + e2 * ne] * h[e + t * ne];
            }
            v[e2 + k_len * t] = acc;
        }
    }

    // 逐头 RMS norm（每个 (head, token) 一条 head_dim 向量）
    let mut qn = q.clone();
    for t in 0..n_tokens {
        for head in 0..nh {
            let row: Vec<f32> = (0..hd).map(|d| q[d + hd * (head + nh * t)]).collect();
            let n = rms_norm_row(&row, NORM_EPS);
            for d in 0..hd {
                qn[d + hd * (head + nh * t)] = n[d] * w.q_norm[d];
            }
        }
    }
    let mut kn = k.clone();
    for t in 0..n_tokens {
        for head in 0..nkv {
            let row: Vec<f32> = (0..hd).map(|d| k[d + hd * (head + nkv * t)]).collect();
            let n = rms_norm_row(&row, NORM_EPS);
            for d in 0..hd {
                kn[d + hd * (head + nkv * t)] = n[d] * w.k_norm[d];
            }
        }
    }

    let q_rope = rope_ref(
        &qn,
        hd,
        nh,
        n_tokens,
        pos,
        SECTIONS,
        FREQ_BASE,
        N_ROT as usize,
    );
    let k_rope = rope_ref(
        &kn,
        hd,
        nkv,
        n_tokens,
        pos,
        SECTIONS,
        FREQ_BASE,
        N_ROT as usize,
    );
    let _ = start;
    (q_rope, gate, k_rope, v)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len(), "长度不一致 {} vs {}", a.len(), b.len());
    let mut worst = 0f32;
    let mut at = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
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
        Some(m) if m >= 1024 => true,
        Some(m) => {
            eprintln!("[attn_test] SKIP GPU: 仅 {} MiB 空闲 (< 1024 MiB)", m);
            false
        }
        None => {
            eprintln!("[attn_test] SKIP GPU: nvidia-smi 不可用");
            false
        }
    })
}

// ===========================================================================
// 图执行封装
// ===========================================================================

/// 在同一 KvCache 上依次跑 `steps`（每项是 (起点, token 数)），返回**最后一步**
/// 的输出，以及与之一致的参考值。
///
/// 只 expand 最后一步的输出：mask 张量是所有步共用的，只能保留一步的形状
/// （见 `KvCache::fill_mask` 的说明），所以一次图里只算一步的注意力。
fn run_steps(kind: BackendKind, wd: &Weights, steps: &[(usize, usize)]) -> (Vec<f32>, Vec<f32>) {
    run_steps_full(kind, wd, steps, false, GgmlType::F32)
}

/// `v_q4` 与 `flash_attn` 分开传，是为了能测"FA + f32 V"（排除 V 量化的影响）。
fn run_steps_full(
    kind: BackendKind,
    wd: &Weights,
    steps: &[(usize, usize)],
    flash_attn: bool,
    v_type: GgmlType,
) -> (Vec<f32>, Vec<f32>) {
    let arch = arch_test();
    let rope = rope_params();
    let ctx = GgmlCtx::new_no_alloc().expect("ctx");
    let w = make_layer_weights(&ctx);
    let mut kv = KvCache::new(&ctx, 1, HEAD_DIM, N_HEAD_KV, KV_CAP, v_type);

    let mut hs = Vec::new();
    let mut poss = Vec::new();
    let mut masks = Vec::new();
    let mut idxss = Vec::new();
    let mut outs = Vec::new();
    let mut kv_writes: Vec<Tensor> = Vec::new();
    for &(_start, n) in steps {
        let h = ctx.new_tensor_2d(GgmlType::F32, N_EMBD, n as i64);
        let p = pos_tensor(&ctx, n as i64);
        // mask 的形状只跟本步有关：keys = 对齐后的 n_kv，queries = n
        kv.begin_step(n).expect("begin_step");
        let mask_type = if flash_attn {
            GgmlType::F16
        } else {
            GgmlType::F32
        };
        let m = ctx.new_tensor_2d(mask_type, kv.n_kv_padded() as i64, n as i64);
        // KV 写入行号（I64）：本步各 token 的绝对 KV 位置
        let ix = ctx.new_tensor(GgmlType::I64, &[n as i64]);
        let (out, writes) = build_attn_layer(
            &ctx, &w, &h, &mut kv, 0, &p, &m, &ix, n as i64, &arch, &rope, flash_attn,
        );
        kv_writes.extend(writes);
        hs.push(h);
        poss.push(p);
        masks.push(m);
        idxss.push(ix);
        outs.push(out);
    }

    let buf = BackendBuffer::from_ctx(&ctx, kind).expect("buffer");
    // 与引擎一致：n_kv 对齐到 256 后，[n_used, n_kv_padded) 的行会被注意力读到，
    // 必须是零（否则未初始化显存里的 NaN/Inf 会污染 softmax）。必须在 buffer 分配之后。
    kv.zero_caches();
    for (i, &(start, n)) in steps.iter().enumerate() {
        buf.set_tensor_data(&hs[i], bytemuck::cast_slice(&h_tokens(start, n)));
        buf.set_tensor_data(&poss[i], &pos_bytes(start as i64, n as i64));
        write_causal_mask(&masks[i], start, n);
        // set_rows 的行号：本步第 j 个 token 写到绝对位置 start + j
        let idx_bytes: Vec<u8> = (0..n)
            .flat_map(|j| ((start + j) as i64).to_ne_bytes())
            .collect();
        buf.set_tensor_data(&idxss[i], &idx_bytes);
    }
    upload_weights(&buf, &ctx, &w, wd);

    let mut gb = GraphBuilder::new(&ctx);
    // 写 cache 的节点必须先 expand（图节点顺序 = expand 顺序）
    for node in &kv_writes {
        gb.expand(node);
    }
    gb.expand(outs.last().unwrap());
    gb.compute(&buf).expect("compute");

    let got = get_f32(&buf, &outs[outs.len() - 1]);
    let expect = expect_steps(wd, steps);
    (got, expect)
}

/// 写因果 mask：`mask[key, query] = 0`（key <= query 绝对位置）否则 -inf。
///
/// 行数用 mask 自身的形状（调用方已按 `KvCache::n_kv_padded()` 建好）：
/// 对齐多出来的行自然是 -inf。
fn write_causal_mask(mask: &Tensor<'_>, start: usize, n: usize) {
    let n_kv = mask.ne()[0] as usize;
    assert!(n_kv >= start + n, "mask 的 keys 不能少于 start+n");
    assert_eq!(mask.ne()[1], n as i64);
    // FA 的 mask 是 F16（0x0000 = 0.0，0xFC00 = -inf），手工路径是 F32
    let f16_mask = mask.r#type() == GgmlType::F16 as i32;
    let esize = if f16_mask { 2 } else { 4 };
    let zero: [u8; 4] = if f16_mask { [0x00, 0x00, 0, 0] } else { 0.0f32.to_ne_bytes() };
    let inf: [u8; 4] = if f16_mask {
        [0x00, 0xFC, 0, 0]
    } else {
        f32::NEG_INFINITY.to_ne_bytes()
    };
    let mut row = vec![0u8; n_kv * esize];
    for i in 0..n {
        let limit = start + i;
        for j in 0..n_kv {
            let v = if j <= limit { zero } else { inf };
            row[j * esize..j * esize + esize].copy_from_slice(&v[..esize]);
        }
        mask.write_bytes_at(i * n_kv * esize, &row);
    }
}

/// 合成输入：`h[e, t] = 0.1 * sin((e + start) * 0.017)`。
fn h_tokens(start: usize, n: usize) -> Vec<f32> {
    (0..(N_EMBD as usize * n))
        .map(|k| 0.1 * ((k as f32 + start as f32) * 0.017).sin())
        .collect()
}

/// 多步的参考输出：把各步的 K/V 依次拼进 cache，最后一步按 [head_dim, n_head_kv, n_kv]
/// 的完整 cache 做注意力（K 经 q4_0 往返）。
fn expect_steps(wd: &Weights, steps: &[(usize, usize)]) -> Vec<f32> {
    let hd = HEAD_DIM as usize;
    let nkv = N_HEAD_KV as usize;
    let n_kv: usize = steps.iter().map(|&(_, n)| n).sum();
    let mut k_all = vec![0f32; hd * nkv * n_kv];
    let mut v_all = vec![0f32; hd * nkv * n_kv];

    let mut off = 0usize;
    let mut last = None;
    for &(start, n) in steps {
        let h_data = h_tokens(start, n);
        let pos: Vec<i32> = (0..4)
            .flat_map(|_| (0..n).map(|t| (start + t) as i32))
            .collect();
        let (q_rope, gate, k_rope, v) = proj_ref(&h_data, wd, n, &pos, start);
        for d in 0..hd {
            for head in 0..nkv {
                for t in 0..n {
                    k_all[d + hd * (head + nkv * (off + t))] = k_rope[d + hd * (head + nkv * t)];
                    v_all[d + hd * (head + nkv * (off + t))] = v[d + hd * (head + nkv * t)];
                }
            }
        }
        off += n;
        last = Some((start, n, h_data, q_rope, gate));
    }

    let (start, n, _h_data, q_rope, gate) = last.unwrap();
    let k_q = quantize_dequant_q4_0(&k_all, hd * nkv);
    attn_ref(
        wd,
        n,
        &q_rope,
        &gate,
        &k_q,
        &v_all,
        n_kv,
        start,
        1.0 / (HEAD_DIM as f32).sqrt(),
    )
}

/// 单步执行（起点 `start`，共 `n_tokens` 个 token），返回 (图输出, 参考输出)。
fn run_single(kind: BackendKind, n_tokens: usize, start: usize) -> (Vec<f32>, Vec<f32>) {
    run_steps(kind, &make_weights(), &[(start, n_tokens)])
}

// ===========================================================================
// 测试用例
// ===========================================================================

/// CPU 图 vs 独立参考实现（prefill）：验证投影 / Q-gate 拆分 / 逐头 RMS norm /
/// IMROPE / q4_0 cache 写入 / 因果 softmax / GQA / 门控 / 输出投影。
#[test]
fn attn_matches_reference_cpu() {
    for &n in &[1usize, 4, 8, 16] {
        let (got, expect) = run_single(BackendKind::Cpu, n, 0);
        assert_eq!(got.len(), (N_EMBD as usize * n));
        for &v in &got {
            assert!(v.is_finite(), "n={n}: 输出非有限值 {v}");
        }
        let (worst, at) = max_abs_diff(&got, &expect);
        let scale = expect.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        eprintln!(
            "[attn_test] n={n}: CPU vs 参考 max={worst:.3e} (相对 {:.4}%)",
            100.0 * worst / scale
        );
        // 参考里 K 已经过 q4_0 往返，但图里的量化 mul_mat 还会把激活压成 int8，
        // 所以容差按相对 2% 给（与 kernel_verify 的量化用例同一策略）。
        assert!(
            worst <= 0.02 * scale,
            "n={n}: CPU 与参考不一致 idx {at}: diff={worst:e} 相对 {:.4}% > 2%",
            100.0 * worst / scale
        );
    }
}

/// 16 token prefill，CPU vs CUDA。
#[test]
fn attn_prefill_with_kv_append_matches_cpu() {
    const N: usize = 16;

    let (cpu_got, expect) = run_single(BackendKind::Cpu, N, 0);
    assert_eq!(cpu_got.len(), (N_EMBD as usize) * N);
    for &v in &cpu_got {
        assert!(v.is_finite(), "CPU 输出非有限值 {v}");
    }
    let (worst, at) = max_abs_diff(&cpu_got, &expect);
    let scale = expect.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    eprintln!(
        "[attn_test] prefill16: CPU vs 参考 max={worst:.3e} (相对 {:.4}%)",
        100.0 * worst / scale
    );
    assert!(
        worst <= 0.02 * scale,
        "prefill16: CPU 与参考不一致 idx {at}: diff={worst:e} 相对 {:.4}%",
        100.0 * worst / scale
    );

    if !gpu_available() {
        return;
    }
    let (cuda_got, _) = run_single(BackendKind::Cuda(0), N, 0);
    let (diff, at) = max_abs_diff(&cpu_got, &cuda_got);
    eprintln!(
        "[attn_test] prefill16: CPU vs CUDA max={diff:.3e} (相对 {:.4}%)",
        100.0 * diff / scale
    );
    // KV cache 是量化的，两后端的量化/反量化路径不同，容差按计划给的 5e-2。
    assert!(
        diff <= 0.05 * scale,
        "prefill16: CPU/CUDA 不一致 idx {at}: diff={diff:e} 相对 {:.4}% > 5%",
        100.0 * diff / scale
    );
}

/// prefill 16 → decode 1：decode 那一步要对 cache 里全部 17 个位置做因果注意力。
#[test]
fn attn_decode_after_prefill_matches_cpu() {
    let steps = [(0usize, 16usize), (16, 1)];
    let wd = make_weights();

    let (got, want) = run_steps(BackendKind::Cpu, &wd, &steps);
    assert_eq!(got.len(), N_EMBD as usize);
    for &x in &got {
        assert!(x.is_finite(), "decode 输出非有限值 {x}");
    }
    let (worst, at) = max_abs_diff(&got, &want);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    eprintln!(
        "[attn_test] decode(17 kv): CPU vs 参考 max={worst:.3e} (相对 {:.4}%)",
        100.0 * worst / scale
    );
    assert!(
        worst <= 0.02 * scale,
        "decode: CPU 与参考不一致 idx {at}: diff={worst:e} 相对 {:.4}% > 2%",
        100.0 * worst / scale
    );

    if !gpu_available() {
        return;
    }
    let (cuda, _) = run_steps(BackendKind::Cuda(0), &wd, &steps);
    let (diff, at) = max_abs_diff(&got, &cuda);
    eprintln!(
        "[attn_test] decode: CPU vs CUDA max={diff:.3e} (相对 {:.4}%)",
        100.0 * diff / scale
    );
    assert!(
        diff <= 0.05 * scale,
        "decode: CPU/CUDA 不一致 idx {at}: diff={diff:e} 相对 {:.4}% > 5%",
        100.0 * diff / scale
    );
}

/// f32 → q4_0 → 反量化的单行往返误差上界。
///
/// q4_0 每块 32 个值共用一个 scale `d = max / -8`（`max` 是块内幅值最大的**带符号**
/// 值），可表示的值是 `d * (q - 8)`，`q ∈ [0, 15]`，也就是 `[-8d, 7d]`。
///
/// 注意这个区间是**不对称**的：它恰好覆盖 `max` 那一侧到 ±amax，而另一侧只能到
/// `7/8 · amax`。所以最坏情况下（块内出现与 max 反号、幅值接近 amax 的元素）
/// 误差可达一整个 `|d|`，而不是对称量化常见的 `0.5|d|`。实测约 0.93|d|。
#[test]
fn kv_quantize_roundtrip_error_bounded() {
    const N_PER_ROW: usize = 32;
    const NROWS: usize = 8;

    let data = pseudo(N_PER_ROW * NROWS, 7);
    let deq = quantize_dequant_q4_0(&data, N_PER_ROW);

    let mut worst_rel = 0f32;
    let mut worst_abs = 0f32;
    let mut sum_rel = 0f32;
    let mut n = 0usize;
    for r in 0..NROWS {
        for b in 0..(N_PER_ROW / 32) {
            let base = r * N_PER_ROW + b * 32;
            // 块内最大幅值 = 8·|d|
            let amax = (0..32).fold(0f32, |m, j| m.max(deq[base + j].abs()));
            let d = amax / 8.0;
            if d == 0.0 {
                continue;
            }
            for j in 0..32 {
                let e = (deq[base + j] - data[base + j]).abs();
                worst_abs = worst_abs.max(e);
                worst_rel = worst_rel.max(e / d);
                sum_rel += e / d;
                n += 1;
            }
        }
    }
    let mean_rel = sum_rel / n as f32;
    eprintln!(
        "[attn_test] q4_0 往返: max abs err={worst_abs:.3e}, max err/d={worst_rel:.4}, mean err/d={mean_rel:.4}"
    );
    // 每个元素都落在 [-8d, 7d] 的栅格上，误差不超过一格
    assert!(
        worst_rel <= 1.0 + 1e-3,
        "q4_0 往返误差超出理论上界 |d|: {worst_rel}"
    );
    // 平均误差应当接近半格（0.25 左右），过小说明量化没生效
    assert!(
        mean_rel > 0.1 && mean_rel < 0.4,
        "平均误差 {mean_rel} 不合理，量化可能没生效"
    );
}

// ---------------------------------------------------------------------------
// FlashAttention（P4）
// ---------------------------------------------------------------------------

/// FA 与手工注意力在**同样输入**下必须数值一致。
///
/// 用 **f16 V** 让两条路径的输入完全相同（手工路径不能吃 q4_0 V：Vᵀ 视图的量化行
/// 不是 32 元素对齐的，见 kv.rs 的说明）；FA+f32 V 在本机 DLL 上会出 NaN（f32 需要
/// 内部转 f16 的路径），生产也不用这个组合。
#[test]
fn attn_flash_attn_matches_manual() {
    if !gpu_available() {
        eprintln!("[attn_test] 跳过 FA 对比：无可用 GPU");
        return;
    }
    let wd = make_weights();
    let steps = [(0usize, 16usize), (16, 1)];

    // (1) 同输入（f16 V）：FA vs 手工，差异只应来自 kernel 内部精度
    let (manual, _) = run_steps_full(BackendKind::Cuda(0), &wd, &steps, false, GgmlType::F16);
    let (fa, _) = run_steps_full(BackendKind::Cuda(0), &wd, &steps, true, GgmlType::F16);
    eprintln!(
        "[attn_test] f16V manual[0..3]={:?} fa[0..3]={:?}",
        &manual[..3],
        &fa[..3]
    );
    let (diff, rel, cos) = compare(&manual, &fa);
    eprintln!("[attn_test] FA(f16 V) vs 手工(f16 V): cos={cos:.6}, 最大差 {diff:.3e}（相对 {rel:.3}%）");
    for (i, v) in fa.iter().enumerate() {
        assert!(v.is_finite(), "FA 输出非有限值 idx {i}: {v}");
    }
    assert!(cos > 0.999, "FA 与手工路径一致性过低: cos={cos}");
    // 1~2% 的差来自 FA kernel 内部的 f16 精度（MMA 路径的 QK^T 在 f16 上算），
    // 与 llama.cpp 自己 FA/非 FA 的差异同量级；greedy 输出对这点差异通常不敏感。
    assert!(rel < 3.0, "FA 与手工路径相对差过大: {rel}%");

    // (2) 生产配置（FA + q4_0 V）：只要求有限且与手工(f16 V)大方向一致
    let (fa_q4, _) = run_steps_full(BackendKind::Cuda(0), &wd, &steps, true, GgmlType::Q4_0);
    for (i, v) in fa_q4.iter().enumerate() {
        assert!(v.is_finite(), "FA(q4_0 V) 输出非有限值 idx {i}: {v}");
    }
    let (_, rel_q4, cos_q4) = compare(&manual, &fa_q4);
    eprintln!("[attn_test] FA(q4_0 V) vs 手工(f16 V): cos={cos_q4:.6}, 相对 {rel_q4:.3}%（V 量化误差）");
    assert!(cos_q4 > 0.99, "FA(q4_0 V) 与手工路径偏差过大: cos={cos_q4}");
}

/// 比较两段输出，返回 (最大绝对差, 相对差 %, 余弦相似度)。
fn compare(a: &[f32], b: &[f32]) -> (f32, f32, f64) {
    assert_eq!(a.len(), b.len());
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut worst = 0f32;
    let mut scale = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
        worst = worst.max((x - y).abs());
        scale = scale.max(x.abs());
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
    let rel = 100.0 * worst / scale.max(1e-6);
    (worst, rel, cos)
}