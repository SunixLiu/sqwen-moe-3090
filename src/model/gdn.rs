// src/model/gdn.rs — Gated DeltaNet (GDN) 循环注意力层
//
// 蓝本（逐行对照）：
//   reference/llama.cpp/src/models/qwen35moe.cpp      build_layer_attn_linear (L359-492)
//   reference/llama.cpp/src/models/delta-net-base.cpp build_delta_net_fused   (L373-423)
//
// 递推数学由 ggml 内建融合算子 `ggml_gated_delta_net`（GGML_OP_GATED_DELTA_NET）
// 完成，也就是 llama.cpp 在 `fused_gdn_ar` / `fused_gdn_ch` 打开时走的
// `build_delta_net_fused` 路径。算子内部按 token 递推：
//
//     S_t = S_{t-1} * exp(g_t)            // 标量门控（g->ne[0] == 1）
//     Δ_t = (v_t - S_tᵀ k_t) ⊙ β_t
//     S_t = S_t + k_t Δ_tᵀ
//     o_t = (S_tᵀ q_t) / sqrt(S_v)
//
// 因此本层只需要把 q/k/v/gate/beta/state 组织成算子要求的形状：
//
//   q, k  : [S_k, H_k, n_tokens, n_seqs]   （S_k == S_v；H_k <= H_v，tiled 广播）
//   v     : [S_v, H_v, n_tokens, n_seqs]
//   gate  : [1,   H_v, n_tokens, n_seqs]   （标量门控；[S_v, H_v, ...] 为 KDA 变体）
//   beta  : [1,   H_v, n_tokens, n_seqs]
//   state : [S_v, S_v, H_v, n_seqs]
//
// 注意：`1/sqrt(S_v)` 的缩放由算子内部完成，调用方不要再对 q 做 scale。

use super::super::backend::ctx::{GgmlCtx, Tensor};
use super::super::backend::ffi;
use crate::backend::ctx::StateTensor;

use super::arch::Qwen35MoeArch;

/// 造一个形状为 `ne`、内容全 0 的 F32 张量（消融用）。
fn zeros<'a>(g: &'a GgmlCtx, ne: &[i64]) -> Tensor<'a> {
    let t = g.new_tensor(ffi::GgmlType::F32, ne);
    g.wrap(unsafe { ffi::ggml_fill(g.as_ptr(), t.ptr(), 0.0) })
}

/// 造一个与 `x` **同形状**的全 0 张量。
///
/// 消融时用法：先按原样把算子**建**出来（ggml 是惰性的，不执行），只为拿到它的
/// `ne`；然后下游改用零张量 —— 这样原算子永远不会被 `expand` 到图里，等于跳过。
fn zeros_like<'a>(g: &'a GgmlCtx, x: &Tensor<'a>) -> Tensor<'a> {
    zeros(g, &x.ne())
}
use super::weights::LayerWeights;

/// f32 元素字节数。
const F32_SIZE: usize = 4;

/// 单个 GDN 层的循环状态。
///
/// 两个字段都指向 context 中的状态张量：首次调用时是调用方预分配的常驻张量，
/// 之后会被 [`build_gdn_layer`] 重绑定为本步算出的图内张量（视图），由调用方
/// 负责把新状态写回常驻张量（见 `graph.rs::gdn_step` 的 cpy）。
///
/// 用 [`StateTensor`] 而不是 `Tensor<'a>`：整个 `EngineState` 要被 `Engine` 持有，
/// 而 `Engine` 同时拥有 `GgmlCtx`，借用版句柄会构成自引用结构。不变式是
/// 「ctx 比状态张量活得久」。
///
/// # 回退快照（`n_rs`）—— 投机解码的部分接受
///
/// GDN 是**递推**状态：把最后几个 token 的 KV 丢掉并不能回到中间位置。而投机解码
/// 的校验批（`[id_last, d0..d_{K-1}]`）里可能有后面的草稿被拒绝，只有前缀被接受 ——
/// 这时必须把递推状态退回"接受前缀之后"。
///
/// 做法与 llama.cpp `n_rs_seq` 一致：常驻张量**加宽成 `1 + n_rs` 个平面**，
/// `ggml_gated_delta_net` 传 `K = n_rs + 1`，一次前向就顺便写出**每个 token 边界**的
/// 状态（槽 0 = 最终状态，槽 s = 往回 s 个 token 的状态）；回退时只要把"读哪个平面"
/// 的索引改成 s 即可（`build_gdn_layer` 的 `sel` 参数，走 `get_rows`，零拷贝）。
/// `n_rs = 0` 时行为与从前完全一致（多一个平面的加宽都没有）。
pub struct GdnState {
    /// DeltaNet 递推状态 `[head_v_dim, head_v_dim, num_v_heads, 1 + n_rs]`
    pub state: StateTensor,
    /// 短卷积历史 `[conv_kernel_size - 1, conv_channels, 1 + n_rs]`
    pub conv_state: StateTensor,
    /// 每个序列保留的回退快照数（0 = 关闭，投机解码时 = `--spec-draft-n-max`）
    pub n_rs: usize,
    /// 本步融合算子实际写出的快照数（`= min(n_tokens, n_rs + 1)`），由
    /// [`build_gdn_layer`] 设定；调用方据此决定写回几个平面。
    pub n_snap: usize,
}

impl GdnState {
    /// 按架构维度在 `ctx` 里预分配常驻状态张量。
    ///
    /// `n_rs`：回退快照数（0 = 不保留；投机解码时给 `--spec-draft-n-max`）。
    pub fn new(ctx: &GgmlCtx, arch: &Qwen35MoeArch, n_rs: usize) -> Self {
        let head_v_dim = (arch.ssm_d_inner / arch.ssm_dt_rank) as i64;
        let n_v_heads = arch.ssm_dt_rank as i64;
        let conv_channels = (arch.ssm_d_inner + 2 * arch.ssm_n_group * arch.ssm_d_state) as i64;
        let n_planes = (1 + n_rs) as i64;
        Self {
            state: StateTensor::new(ctx.new_tensor(
                ffi::GgmlType::F32,
                &[head_v_dim, head_v_dim, n_v_heads, n_planes],
            )),
            conv_state: StateTensor::new(ctx.new_tensor_3d(
                ffi::GgmlType::F32,
                arch.ssm_d_conv as i64 - 1,
                conv_channels,
                n_planes,
            )),
            n_rs,
            n_snap: 1,
        }
    }

    /// 平面 0 的字节数（递推状态）。检查点/快照只保存平面 0 —— 回退快照只对
    /// "当前这一步的校验批"有意义，跨请求复用时无需保存。
    pub fn state_plane_bytes(&self) -> usize {
        let ne = self.state.ne();
        (ne[0] * ne[1] * ne[2]) as usize * F32_SIZE
    }

    /// 平面 0 的字节数（卷积历史）。
    pub fn conv_plane_bytes(&self) -> usize {
        let ne = self.conv_state.ne();
        (ne[0] * ne[1]) as usize * F32_SIZE
    }
}

// ---------------------------------------------------------------------------
// 辅助算子
// ---------------------------------------------------------------------------

/// GDN 的 L2 归一化，等价于 llama.cpp `build_gdn_l2_norm`：
///
/// ```text
/// scale(rms_norm(x, eps/n), 1/sqrt(n))    // n = x->ne[0]
/// ```
///
/// 因为 `rms(x) = ||x|| / sqrt(n)`，所以该组合正是 `x / ||x||`。
fn l2_norm<'a>(
    ctx: *mut ffi::GgmlContext,
    g: &'a GgmlCtx,
    x: &Tensor<'a>,
    eps: f32,
) -> Tensor<'a> {
    let n = x.ne()[0] as f32;
    let normalized = g.wrap(unsafe { ffi::ggml_rms_norm(ctx, x.ptr(), eps / n) });
    g.wrap(unsafe { ffi::ggml_scale(ctx, normalized.ptr(), 1.0 / n.sqrt()) })
}

/// 门控归一化，等价于 llama.cpp `build_norm_gated`：
/// `rms_norm(x) * weight` 再逐元素乘以 `silu(gate)`。
///
/// `weight` 的广播维度是 dim0（每个 value head 的通道）：
/// - 真实权重形状 `[head_v_dim]`（llama.cpp `create_tensor(..., { head_v_dim })`）
/// - 合成测试中给的是 `[head_v_dim * num_v_heads]`，同样按 dim0 展开
fn norm_gated<'a>(
    ctx: *mut ffi::GgmlContext,
    g: &'a GgmlCtx,
    x: &Tensor<'a>,      // [head_v_dim, H_v, n_tokens, n_seqs]
    weight: &Tensor<'a>, // [head_v_dim] 或 [head_v_dim * H_v]
    gate: &Tensor<'a>,   // [head_v_dim, H_v, n_tokens, n_seqs]
    eps: f32,
) -> Tensor<'a> {
    let s_v = x.ne()[0];
    let h_v = x.ne()[1];

    let normalized = g.wrap(unsafe { ffi::ggml_rms_norm(ctx, x.ptr(), eps) });

    // 权重拍平成 [ne0, ne1, 1, 1]，只沿 dim0/dim1 广播。
    // 权重常驻且连续 → **不需要 cont**（cont 会白发一次拷贝 kernel）。
    let n_weight = weight.ne()[0];
    let weight = g.cont_if_needed(weight);
    let weight = if n_weight == s_v {
        g.wrap(unsafe { ffi::ggml_reshape_4d(ctx, weight.ptr(), s_v, 1, 1, 1) })
    } else {
        assert_eq!(
            n_weight,
            s_v * h_v,
            "ssm_norm 元素数必须等于 head_v_dim 或 head_v_dim*num_v_heads"
        );
        g.wrap(unsafe { ffi::ggml_reshape_4d(ctx, weight.ptr(), s_v, h_v, 1, 1) })
    };

    let scaled = g.wrap(unsafe { ffi::ggml_mul(ctx, normalized.ptr(), weight.ptr()) });
    let gated = g.wrap(unsafe { ffi::ggml_silu(ctx, gate.ptr()) });
    g.wrap(unsafe { ffi::ggml_mul(ctx, scaled.ptr(), gated.ptr()) })
}

// ---------------------------------------------------------------------------
// build_gdn_layer — 移植 build_layer_attn_linear（fused GDN 路径）
// ---------------------------------------------------------------------------

/// 构建一个 GDN 层的前向图，返回 `[n_embd, n_tokens]`。
///
/// `h` 为输入 `[n_embd, n_tokens]`；`n_tokens` 同时是 chunk 长度（prefill）或 1
/// （decode），融合算子对两者使用同一套逐 token 递推，因此这里不区分路径。
///
/// `sel`：递推状态的**平面选择**输入（I32 `[1]`）。`Some` 时读 `state[.., sel]`
/// （即"往回 sel 个 token"的快照），`None` 时读平面 0。只有 `n_rs > 0`（投机解码）
/// 才需要它；此时图里会多一个 `get_rows`（一次 `head_v_dim²·H_v` 的拷贝）。
///
/// 调用后 `st.state` / `st.conv_state` 被重绑定为**本步图内的张量**：
///   - `st.state` = 融合算子写出的快照块 `[S_v, S_v, H_v, n_snap]`（`n_snap` 见字段说明）
///   - `st.conv_state` = `n_rs == 0` 时是卷积尾部 `[K-1, C, 1]`，否则是整段
///     `conv_input` `[K-1+n_t, C, 1]`（快照要从不同的行偏移切出来）
pub fn build_gdn_layer<'a>(
    g: &'a GgmlCtx,
    w: &LayerWeights,
    h: &Tensor<'a>,
    st: &mut GdnState,
    n_tokens: i64,
    arch: &Qwen35MoeArch,
    sel: Option<&Tensor<'a>>,
) -> Tensor<'a> {
    let ctx = g.as_ptr();

    // n_seqs 固定为 1：与 ggml_concat（要求除 dim0 外形状一致）以及引擎的
    // 单序列推理路径一致。
    let n_seqs: i64 = 1;
    let n_t = n_tokens;

    let n_embd = arch.n_embd as i64;
    let head_k_dim = arch.ssm_d_state as i64;
    let num_k_heads = arch.ssm_n_group as i64;
    let num_v_heads = arch.ssm_dt_rank as i64;
    let head_v_dim = (arch.ssm_d_inner as i64) / num_v_heads;

    assert_eq!(
        head_k_dim, head_v_dim,
        "GDN 融合算子要求 S_k == S_v（head_k_dim == head_v_dim）"
    );
    assert_eq!(
        num_v_heads % num_k_heads,
        0,
        "num_v_heads 必须是 num_k_heads 的整数倍"
    );
    assert_eq!(
        h.ne()[1], n_tokens,
        "n_tokens 与输入张量 h 的 token 维不一致"
    );

    let attn_qkv = w.attn_qkv.get(g);
    let attn_gate = w.attn_gate.get(g);
    let ssm_beta = w.ssm_beta.get(g);
    let ssm_alpha = w.ssm_alpha.get(g);
    let ssm_dt = w.ssm_dt.get(g);
    let ssm_a = w.ssm_a.get(g);
    let ssm_conv1d = w.ssm_conv1d.get(g);
    let ssm_norm = w.ssm_norm.get(g);
    let ssm_out = w.ssm_out.get(g);

    // 卷积核长度 / 通道数以权重实际 shape 为准（llama.cpp 同）。
    let conv_kernel_size = ssm_conv1d.ne()[0];
    let conv_channels = ssm_conv1d.ne()[1];
    let qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    assert_eq!(qkv_dim, conv_channels, "QKV 投影维度与 conv 通道数不一致");

    // ---- 1. 输入投影 ----------------------------------------------------
    // 消融 `gdn_proj`：跳过四个投影 matmul，直接给全 0（下游客积很小，可忽略）。
    let ab = crate::model::graph::Ablate::cached();
    let (qkv_mixed, z, beta, gate) = if ab.gdn_proj {
        (
            zeros(g, &[conv_channels, n_t]),
            zeros(g, &[head_v_dim * num_v_heads, n_t]),
            zeros(g, &[1, num_v_heads, n_t, n_seqs]),
            zeros(g, &[1, num_v_heads, n_t, n_seqs]),
        )
    } else {
        // qkv_mixed: [conv_channels, n_tokens]
        let qkv_mixed = g.wrap(unsafe { ffi::ggml_mul_mat(ctx, attn_qkv.ptr(), h.ptr()) });
        // z: [head_v_dim * num_v_heads, n_tokens]
        let z = g.wrap(unsafe { ffi::ggml_mul_mat(ctx, attn_gate.ptr(), h.ptr()) });

        // ---- 2. beta = sigmoid(W_beta · h) → [1, H_v, T, n_seqs] -------------
        let beta = g.wrap(unsafe { ffi::ggml_mul_mat(ctx, ssm_beta.ptr(), h.ptr()) });
        let beta = g.wrap(unsafe {
            ffi::ggml_reshape_4d(ctx, beta.ptr(), 1, num_v_heads, n_t, n_seqs)
        });
        let beta = g.wrap(unsafe { ffi::ggml_sigmoid(ctx, beta.ptr()) });

        // ---- 3. gate = softplus(W_alpha · h + dt) * a → [1, H_v, T, n_seqs] --
        let alpha = g.wrap(unsafe { ffi::ggml_mul_mat(ctx, ssm_alpha.ptr(), h.ptr()) });
        let alpha = g.wrap(unsafe {
            ffi::ggml_reshape_3d(ctx, alpha.ptr(), num_v_heads, n_t, n_seqs)
        });
        // ssm_dt / ssm_a 均为 [H_v]，沿 dim1 广播
        let alpha_biased = g.wrap(unsafe { ffi::ggml_add(ctx, alpha.ptr(), ssm_dt.ptr()) });
        let alpha_softplus = g.wrap(unsafe { ffi::ggml_softplus(ctx, alpha_biased.ptr()) });
        let gate = g.wrap(unsafe { ffi::ggml_mul(ctx, alpha_softplus.ptr(), ssm_a.ptr()) });
        let gate = g.wrap(unsafe {
            ffi::ggml_reshape_4d(ctx, gate.ptr(), 1, num_v_heads, n_t, n_seqs)
        });
        (qkv_mixed, z, beta, gate)
    };

    // ---- 4. 卷积状态拼接 ------------------------------------------------
    // 常驻 conv_state 是 [K-1, C, 1+n_rs]（加宽）；这里先选出要读的平面。
    let n_planes = (1 + st.n_rs) as i64;
    let row_count = (conv_kernel_size - 1) * conv_channels;
    let conv_plane = match sel {
        Some(sel) => {
            let wide_conv = st.conv_state.get(g);
            let wide = g.wrap(unsafe {
                ffi::ggml_view_2d(
                    ctx,
                    wide_conv.ptr(),
                    row_count,
                    n_planes,
                    (row_count as usize) * F32_SIZE,
                    0,
                )
            });
            let row = g.wrap(unsafe { ffi::ggml_get_rows(ctx, wide.ptr(), sel.ptr()) });
            g.wrap(unsafe {
                ffi::ggml_reshape_3d(ctx, row.ptr(), conv_kernel_size - 1, conv_channels, n_seqs)
            })
        }
        None => {
            // 平面 0 的视图（加宽时不能直接 reshape 整个张量 —— 元素数对不上）
            let wide_conv = st.conv_state.get(g);
            g.wrap(unsafe {
                ffi::ggml_view_3d(
                    ctx,
                    wide_conv.ptr(),
                    conv_kernel_size - 1,
                    conv_channels,
                    n_seqs,
                    wide_conv.nb()[1],
                    wide_conv.nb()[2],
                    0,
                )
            })
        }
    };

    // conv_state 期望形状 [K-1, C, n_seqs]；若已是该形状则直接复用（可能是
    // 上一步留下的 strided 视图，不能再 reshape）。
    let cs_ne = conv_plane.ne();
    let conv_states = if cs_ne[0] == conv_kernel_size - 1
        && cs_ne[1] == conv_channels
        && cs_ne[2] == n_seqs
    {
        conv_plane
    } else {
        g.wrap(unsafe {
            ffi::ggml_reshape_3d(
                ctx,
                conv_plane.ptr(),
                conv_kernel_size - 1,
                conv_channels,
                n_seqs,
            )
        })
    };

    // qkv_mixed 转置成 [n_tokens, conv_channels] 以便沿 dim0 拼接
    let qkv_t = g.wrap(unsafe { ffi::ggml_transpose(ctx, qkv_mixed.ptr()) });
    let qkv_t = g.wrap(unsafe { ffi::ggml_cont(ctx, qkv_t.ptr()) });

    // conv_input: [K-1 + T, conv_channels, n_seqs]
    let conv_input = g.wrap(unsafe {
        ffi::ggml_concat(ctx, conv_states.ptr(), qkv_t.ptr(), 0)
    });

    // ---- 5. 短卷积 + silu ----------------------------------------------
    let conv_out = g.wrap(unsafe { ffi::ggml_ssm_conv(ctx, conv_input.ptr(), ssm_conv1d.ptr()) });
    let conv_out = g.wrap(unsafe { ffi::ggml_silu(ctx, conv_out.ptr()) });

    // 新的卷积历史 = conv_input 的最后 (K-1) 个 conv 位置（所有通道）。
    // 与 llama.cpp build_conv_state 的 view 完全一致：offset = row_size(f32, T)。
    let conv_tail = g.wrap(unsafe {
        ffi::ggml_view_3d(
            ctx,
            conv_input.ptr(),
            conv_kernel_size - 1,
            conv_channels,
            n_seqs,
            conv_input.nb()[1],
            conv_input.nb()[2],
            (n_t as u64) * F32_SIZE as u64,
        )
    });

    // 消融 `gdn_conv`：卷积产物与写回都用零顶替 —— 上面的 transpose/cont/concat/
    // ssm_conv 因为没人再引用，不会被 expand 进图（等于整段跳过）。
    let (conv_out, conv_tail) = if ab.gdn_conv {
        (zeros_like(g, &conv_out), zeros_like(g, &conv_tail))
    } else {
        (conv_out, conv_tail)
    };

    // ---- 6. 切出 Q / K / V（view，零拷贝）-------------------------------
    // conv_out 的内存布局是 [conv_channels, n_tokens, n_seqs]，因此
    // dim0 = conv 通道（连续），dim1 = token，dim2 = 序列。
    let row_qkv = (conv_channels as usize) * F32_SIZE;
    let nb1 = (head_k_dim as usize) * F32_SIZE;

    let q_conv = g.wrap(unsafe {
        ffi::ggml_view_4d(
            ctx,
            conv_out.ptr(),
            head_k_dim,
            num_k_heads,
            n_t,
            n_seqs,
            nb1,
            row_qkv,
            row_qkv * (n_t as usize),
            0,
        )
    });
    let k_off = (head_k_dim * num_k_heads) as usize * F32_SIZE;
    let k_conv = g.wrap(unsafe {
        ffi::ggml_view_4d(
            ctx,
            conv_out.ptr(),
            head_k_dim,
            num_k_heads,
            n_t,
            n_seqs,
            nb1,
            row_qkv,
            row_qkv * (n_t as usize),
            k_off,
        )
    });
    let v_off = (head_k_dim * num_k_heads * 2) as usize * F32_SIZE;
    let v_conv = g.wrap(unsafe {
        ffi::ggml_view_4d(
            ctx,
            conv_out.ptr(),
            head_v_dim,
            num_v_heads,
            n_t,
            n_seqs,
            (head_v_dim as usize) * F32_SIZE,
            row_qkv,
            row_qkv * (n_t as usize),
            v_off,
        )
    });

    // ---- 7. Q / K 的 L2 归一化 -----------------------------------------
    // 归一化后是全新的连续张量 [S_k, H_k, T, n_seqs]；H_k < H_v 时由融合算子
    // 做 tiled 广播（iv1 % H_k），无需显式 repeat。
    let q_conv = l2_norm(ctx, g, &q_conv, arch.norm_rms_eps);
    let k_conv = l2_norm(ctx, g, &k_conv, arch.norm_rms_eps);

    // ---- 8. Gated DeltaNet 递推（融合算子）-----------------------------
    // 输入状态：选中平面（n_rs == 0 时就是常驻张量的平面 0）。
    let state_in = match sel {
        Some(sel) => {
            // [S_v, S_v, H_v, n_planes] → [S_v·S_v·H_v, n_planes] → 取一行
            let d_state = head_v_dim * head_v_dim * num_v_heads;
            let wide = g.wrap(unsafe {
                ffi::ggml_view_2d(
                    ctx,
                    st.state.ptr(),
                    d_state,
                    n_planes,
                    (d_state as usize) * F32_SIZE,
                    0,
                )
            });
            let row = g.wrap(unsafe { ffi::ggml_get_rows(ctx, wide.ptr(), sel.ptr()) });
            g.wrap(unsafe {
                ffi::ggml_reshape_4d(ctx, row.ptr(), head_v_dim, head_v_dim, num_v_heads, n_seqs)
            })
        }
        None => {
            // 平面 0 的视图（加宽时直接当输入用会读到后面的平面）
            let wide_state = st.state.get(g);
            g.wrap(unsafe {
                ffi::ggml_view_4d(
                    ctx,
                    wide_state.ptr(),
                    head_v_dim,
                    head_v_dim,
                    num_v_heads,
                    n_seqs,
                    (head_v_dim as usize) * F32_SIZE,
                    (head_v_dim * head_v_dim) as usize * F32_SIZE,
                    (head_v_dim * head_v_dim * num_v_heads) as usize * F32_SIZE,
                    0,
                )
            })
        }
    };

    // K = n_rs + 1：让算子顺便写出每个 token 边界的递推状态（槽 0 = 最终状态）。
    let k_snap = (st.n_rs + 1) as i64;
    let n_snap = n_t.min(k_snap);
    st.n_snap = n_snap as usize;

    let result = g.wrap(unsafe {
        ffi::ggml_gated_delta_net(
            ctx,
            q_conv.ptr(),
            k_conv.ptr(),
            v_conv.ptr(),
            gate.ptr(),
            beta.ptr(),
            state_in.ptr(),
            k_snap,
        )
    });

    // 消融 `gdn_core`：递推核（含状态写回）整体跳过 —— 用零顶替它的输出。
    // 下游的 attn / new_state 都是 result 的视图，`st.state` 指向零张量，
    // 于是写回变成"写零"（数值不可用，但形状与内核数结构一致）。
    let result = if ab.gdn_core { zeros_like(g, &result) } else { result };

    // attn scores: [S_v, H_v, T, n_seqs]
    let attn = g.wrap(unsafe {
        ffi::ggml_view_4d(
            ctx,
            result.ptr(),
            head_v_dim,
            num_v_heads,
            n_t,
            n_seqs,
            (head_v_dim as usize) * F32_SIZE,
            (head_v_dim * num_v_heads) as usize * F32_SIZE,
            (head_v_dim * num_v_heads * n_t) as usize * F32_SIZE,
            0,
        )
    });

    // new state: [S_v, S_v, H_v, n_snap]（紧随 attn scores 之后，槽 0 = 最终状态，
    // 槽 s = 往回 s 个 token 的状态）
    let new_state = g.wrap(unsafe {
        ffi::ggml_view_4d(
            ctx,
            result.ptr(),
            head_v_dim,
            head_v_dim,
            num_v_heads,
            n_snap,
            (head_v_dim as usize) * F32_SIZE,
            (head_v_dim * head_v_dim) as usize * F32_SIZE,
            (head_v_dim * head_v_dim * num_v_heads) as usize * F32_SIZE,
            (head_v_dim * num_v_heads * n_t * n_seqs) as usize * F32_SIZE,
        )
    });

    // 状态写回：把本步算出的新状态重新绑定到 `GdnState`。
    //
    // `new_state` 是融合算子输出的视图，`conv_tail` 是 `conv_input` 尾部的视图，
    // 两者都留在同一个 ggml context 里，因此下一次调用在构建图时能通过数据依赖
    // 读到本步结果 —— 这就是分块 prefill（chunk A → chunk B）状态接力的机制，
    // 与 llama.cpp 中 `st.state` / `st.conv_state` 是图内张量的语义一致。
    //
    // 需要跨 ggml context 复用（例如逐 token decode 时每步新建 context）的场景，
    // 由引擎按 llama.cpp `build_rs` 的方式用 `ggml_cpy` 做 gather/scatter。
    st.state = StateTensor::new(new_state);
    st.conv_state = StateTensor::new(if st.n_rs == 0 { conv_tail } else { conv_input });

    // ---- 9. 门控归一化 --------------------------------------------------
    // z: [head_v_dim * H_v, T] → [head_v_dim, H_v, T, n_seqs]
    let z = g.wrap(unsafe { ffi::ggml_cont(ctx, z.ptr()) });
    let z_2d = g.wrap(unsafe {
        ffi::ggml_reshape_4d(ctx, z.ptr(), head_v_dim, num_v_heads, n_t, n_seqs)
    });

    let attn_out_norm = norm_gated(ctx, g, &attn, &ssm_norm, &z_2d, arch.norm_rms_eps);

    // [S_v, H_v, T, n_seqs] → [S_v * H_v, T, n_seqs]
    let final_output = g.wrap(unsafe {
        ffi::ggml_reshape_3d(
            ctx,
            attn_out_norm.ptr(),
            head_v_dim * num_v_heads,
            n_t,
            n_seqs,
        )
    });

    // ---- 10. 输出投影 ---------------------------------------------------
    // 消融 `gdn_out`：门控 norm + out 投影整段跳过（它们的产物只喂这里）。
    if ab.gdn_out {
        return zeros(g, &[n_embd, n_t]);
    }
    let out = g.wrap(unsafe { ffi::ggml_mul_mat(ctx, ssm_out.ptr(), final_output.ptr()) });
    g.wrap(unsafe { ffi::ggml_reshape_2d(ctx, out.ptr(), n_embd, n_t) })
}
