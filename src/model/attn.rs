// src/model/attn.rs — 全注意力层（GQA + 分节 M-RoPE + 量化 KV cache）
//
// 蓝本：llama.cpp `src/models/qwen35moe.cpp` L278-357 `build_layer_attn`
//       + `src/llama-graph.cpp` L2611-2746 `build_attn_mha`（非 flash 路径）。
//
// 图结构：
//   1. QG 联合投影  q_full = W_q h      → [head_dim*2*n_head, n_tokens]
//   2. 拆出 Q 与 gate（同一个投影的前后半，按 head 交错）
//   3. Q/K 逐头 RMS norm（attn_q_norm / attn_k_norm，长度 head_dim）
//   4. IMROPE（rope_multi，分节 sections）
//   5. K/V 追加进 KvCache（K 以 q4_0 写入）
//   6. scores = (K_cacheᵀ q) / sqrt(head_dim) + causal_mask → softmax
//   7. attn_out = V_cacheᵀ · scores
//   8. out = attn_out * sigmoid(gate)
//   9. W_o 输出投影 → [n_embd, n_tokens]

use crate::backend::ctx::{GgmlCtx, Tensor};
use crate::backend::ffi;
use crate::engine::kv::KvCache;

use super::arch::Qwen35MoeArch;
use super::weights::LayerWeights;

/// qwen35moe 的 RoPE 参数。
///
/// 只有 `sections` 目前不在 `Qwen35MoeArch` 里（GGUF 键
/// `qwen35moe.rope.dimension_sections`），所以由调用方传入；YaRN 的
/// `freq_scale/ext_factor/...` 在文本-only 场景保持中性值。
#[derive(Clone, Copy, Debug)]
pub struct RopeParams {
    /// 旋转的维度数（对应 `rope.dimension_count`）
    pub n_rot: i32,
    /// M-RoPE 分节；四段之和必须等于 `n_rot / 2`
    pub sections: [i32; 4],
    pub freq_base: f32,
    pub n_ctx_orig: i32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

impl RopeParams {
    /// 文本-only 默认值（未启用 YaRN）。
    pub fn text(n_rot: i32, sections: [i32; 4], freq_base: f32, n_ctx_orig: i32) -> Self {
        let sum: i32 = sections.iter().sum();
        assert_eq!(
            sum,
            n_rot / 2,
            "M-RoPE 四段之和必须等于 n_rot/2（sections={sections:?}, n_rot={n_rot}）"
        );
        Self {
            n_rot,
            sections,
            freq_base,
            n_ctx_orig,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }
}

/// 逐头 RMS 归一化：`rms_norm(x)` 再乘长度 `head_dim` 的权重（沿 dim0 广播）。
fn head_rms_norm<'a>(
    ctx: *mut ffi::GgmlContext,
    g: &'a GgmlCtx,
    x: &Tensor<'a>,
    weight: &Tensor<'a>,
    eps: f32,
) -> Tensor<'a> {
    let head_dim = x.ne()[0];
    assert_eq!(weight.ne()[0], head_dim, "attn_[qk]_norm 长度应等于 head_dim");

    let normed = g.wrap(unsafe { ffi::ggml_rms_norm(ctx, x.ptr(), eps) });
    // [head_dim] → [head_dim, 1, 1, 1]，只沿 dim0 广播。
    // 权重是常驻的 1D 连续张量 → **不需要 cont**（cont 会白发一次拷贝 kernel）。
    let w = g.cont_if_needed(weight);
    let w = g.wrap(unsafe { ffi::ggml_reshape_4d(ctx, w.ptr(), head_dim, 1, 1, 1) });
    g.wrap(unsafe { ffi::ggml_mul(ctx, normed.ptr(), w.ptr()) })
}

/// 构建一个全注意力层的前向图，返回 `[n_embd, n_tokens]`。
///
/// `pos` 是 IMROPE 的位置张量：一维 i32，长度 `4 * n_tokens`，布局为
/// `[t0..tn, h0..hn, w0..wn, e0..en]`（文本场景四段取同一个位置）。
///
/// `mask` 是因果 mask，形状 `[n_kv_padded, n_tokens]`（key × query）：允许的位置为 0，
/// 屏蔽为 -inf。由调用方创建并填值，这样它的尺寸只跟本步有关。
///
/// `kv_idxs` 是 I64 `[n_tokens]` 的 KV 写入行号（值 = 各 token 的绝对位置），
/// 供 `KvCache::append` 用 `ggml_set_rows` 写入（见那里的注释：为了 CUDA graph 重放）。
#[allow(clippy::too_many_arguments)]
pub fn build_attn_layer<'a>(
    g: &'a GgmlCtx,
    w: &LayerWeights,
    h: &Tensor<'a>,      // [n_embd, n_tokens]
    kv: &mut KvCache,
    slot: usize,         // 该层在 KvCache 中的槽位（全注意力层的序号）
    pos: &Tensor<'a>,    // i32 [4 * n_tokens]
    mask: &Tensor<'a>,   // f32 [n_kv_padded, n_tokens]
    kv_idxs: &Tensor<'a>, // i64 [n_tokens]
    n_tokens: i64,
    arch: &Qwen35MoeArch,
    rope: &RopeParams,
    // 是否走 FlashAttention（P4）：要求 K/V 已按 n_kv_padded 对齐，且开启时 V cache 必须是 q4_0
    flash_attn: bool,
) -> (Tensor<'a>, [Tensor<'a>; 2]) {
    let ctx = g.as_ptr();

    let head_dim = arch.head_dim as i64;
    let n_head = arch.n_head as i64;
    let n_head_kv = arch.n_head_kv as i64;
    assert!(head_dim > 0 && n_head > 0 && n_head_kv > 0);
    assert_eq!(n_head % n_head_kv, 0, "n_head 必须是 n_head_kv 的整数倍（GQA）");

    let attn_q = w.attn_q.get(g);
    let attn_k = w.attn_k.get(g);
    let attn_v = w.attn_v.get(g);
    let attn_output = w.attn_output.get(g);

    // ---- 1. 投影 --------------------------------------------------------
    // W_q 输出 Q 与 gate 交错（每个 head 一段 [q(head_dim), gate(head_dim)]）
    let q_full = g.wrap(unsafe { ffi::ggml_mul_mat(ctx, attn_q.ptr(), h.ptr()) });
    let k_cur = g.wrap(unsafe { ffi::ggml_mul_mat(ctx, attn_k.ptr(), h.ptr()) });
    let v_cur = g.wrap(unsafe { ffi::ggml_mul_mat(ctx, attn_v.ptr(), h.ptr()) });

    // ---- 2. 拆 Q / gate -------------------------------------------------
    // q_full 的元素布局：(head, {q, gate}, dim)，head 跨步 2*head_dim，gate 偏移 head_dim
    let nb_h = (2 * head_dim as usize) * 4;
    let nb_t = nb_h * n_head as usize;
    let q_view = g.wrap(unsafe {
        ffi::ggml_view_3d(
            ctx,
            q_full.ptr(),
            head_dim,
            n_head,
            n_tokens,
            nb_h as u64,
            nb_t as u64,
            0,
        )
    });
    let gate_view = g.wrap(unsafe {
        ffi::ggml_view_3d(
            ctx,
            q_full.ptr(),
            head_dim,
            n_head,
            n_tokens,
            nb_h as u64,
            nb_t as u64,
            (head_dim as u64) * 4,
        )
    });
    let q = g.wrap(unsafe { ffi::ggml_cont(ctx, q_view.ptr()) }); // [head_dim, n_head, n_tokens]
    let gate3 = g.wrap(unsafe { ffi::ggml_cont(ctx, gate_view.ptr()) });
    // [head_dim, n_head, n_tokens] → [head_dim*n_head, n_tokens]（布局 (d,h,t) 与注意力输出一致）
    let gate = g.wrap(unsafe {
        ffi::ggml_reshape_2d(ctx, gate3.ptr(), head_dim * n_head, n_tokens)
    });

    // ---- 3. Q/K 逐头 RMS norm -------------------------------------------
    let q_norm = w.attn_q_norm.get(g);
    let k_norm = w.attn_k_norm.get(g);
    let q = head_rms_norm(ctx, g, &q, &q_norm, arch.norm_rms_eps);

    let k_cur = g.wrap(unsafe { ffi::ggml_cont(ctx, k_cur.ptr()) });
    let k3 = g.wrap(unsafe {
        ffi::ggml_reshape_3d(ctx, k_cur.ptr(), head_dim, n_head_kv, n_tokens)
    });
    let k3 = head_rms_norm(ctx, g, &k3, &k_norm, arch.norm_rms_eps);

    let v3 = g.wrap(unsafe { ffi::ggml_cont(ctx, v_cur.ptr()) });
    let v3 = g.wrap(unsafe {
        ffi::ggml_reshape_3d(ctx, v3.ptr(), head_dim, n_head_kv, n_tokens)
    });

    // ---- 4. IMROPE ------------------------------------------------------
    let rope_apply = |x: &Tensor<'a>| -> Tensor<'a> {
        g.wrap(unsafe {
            ffi::ggml_rope_multi(
                ctx,
                x.ptr(),
                pos.ptr(),
                std::ptr::null_mut(),
                rope.n_rot,
                rope.sections.as_ptr(),
                ffi::GGML_ROPE_TYPE_IMROPE,
                rope.n_ctx_orig,
                rope.freq_base,
                rope.freq_scale,
                rope.ext_factor,
                rope.attn_factor,
                rope.beta_fast,
                rope.beta_slow,
            )
        })
    };
    let q = rope_apply(&q);
    let k3 = rope_apply(&k3);

    // ---- 5. 写入 KV cache 并取回读视图 ----------------------------------
    // 注意：append 产生的 cpy 节点需要调用方在 expand 层输出之前 expand。
    let (k_cache, v_cache, kv_writes) = kv.append(g, slot, &k3, &v3, n_tokens, kv_idxs);
    // mask / K / V 都用对齐后的 n_kv（KvCache::n_kv_padded），三者必须一致
    debug_assert_eq!(mask.ne()[0], kv.n_kv_padded() as i64);
    debug_assert_eq!(k_cache.ne()[2], kv.n_kv_padded() as i64);
    debug_assert_eq!(mask.ne()[1], n_tokens);

    // ---- 6. 注意力 ------------------------------------------------------
    let scale = 1.0 / (head_dim as f32).sqrt();

    // 6a. FlashAttention 路径（P4）
    //
    // 要求（ggml.h `ggml_flash_attn_ext`）：
    //   q    [n_embd_k, n_tokens, n_head,    1]  f32（**contiguous**）
    //   k/v  [n_embd_k, n_kv,     n_head_kv, 1]  **不转置**（我们的 cache 是
    //        [head_dim, n_head_kv, n_kv]，交换 dim1/dim2 即可，permute 是零拷贝视图）
    //   mask [n_kv, n_tokens, 1, 1] **F16**，且 `mask->ne[0]` 必须严格等于 `k->ne[1]`
    //        我们两边都是 `n_kv_padded`，正好相等（P2 的对齐是这里的前提）
    //   结果 [n_embd_v, n_head, n_tokens, 1]，布局与手工路径的 [head_dim, n_head, n_tokens] 相同
    //
    // 数值上它把 scale 直接传给 kernel（不再单独 mul），mask 也由 kernel 内部处理。
    let attn_out = if flash_attn {
        // 维度顺序：FA 要 q [head_dim, n_tokens, n_head, 1]（ggml.h 写成
        // `[n_embd_k, n_batch, n_head, ne3]`），而我们的 q 是 [head_dim, n_head, n_tokens]
        // → **必须 permute(0,2,1,3)**。少了这一步 FA 会按错误的 stride 读 Q，输出是 NaN。
        let q4d = g.wrap(unsafe { ffi::ggml_permute(ctx, q.ptr(), 0, 2, 1, 3) });
        let k4d = g.wrap(unsafe { ffi::ggml_permute(ctx, k_cache.ptr(), 0, 2, 1, 3) });
        let v4d = g.wrap(unsafe { ffi::ggml_permute(ctx, v_cache.ptr(), 0, 2, 1, 3) });
        // mask 在 FA 路径下**已经是 F16**（由 forward 直接建成 F16，同 llama.cpp；不要 cast）
        debug_assert_eq!(mask.r#type(), ffi::GgmlType::F16 as i32);
        debug_assert_eq!(mask.ne()[0], k4d.ne()[1]);
        let fa = g.wrap(unsafe {
            ffi::ggml_flash_attn_ext(
                ctx,
                q4d.ptr(),
                k4d.ptr(),
                v4d.ptr(),
                mask.ptr(),
                scale,
                0.0,
                0.0,
            )
        });
        // [head_dim, n_head, n_tokens, 1]：contiguous，直接 reshape 成 2D（d 最快、head 次之）
        g.wrap(unsafe { ffi::ggml_reshape_2d(ctx, fa.ptr(), head_dim * n_head, n_tokens) })
    } else {
        // 6b. 手工路径（非 FA 回退；V 必须是 f32，见 KvCache 的注释）
        // mul_mat(A, B) = AᵀB：A = K cache [head_dim, n_kv, n_head_kv]，
        // B = q [head_dim, n_tokens, n_head]（n_head_kv 广播到 n_head）
        let k_perm = g.wrap(unsafe { ffi::ggml_permute(ctx, k_cache.ptr(), 0, 2, 1, 3) });
        let q_perm = g.wrap(unsafe { ffi::ggml_permute(ctx, q.ptr(), 0, 2, 1, 3) });
        let kq = g.wrap(unsafe { ffi::ggml_mul_mat(ctx, k_perm.ptr(), q_perm.ptr()) });
        // [n_kv, n_tokens, n_head]

        let kq = g.wrap(unsafe { ffi::ggml_scale(ctx, kq.ptr(), scale) });

        let mask_add = g.wrap(unsafe { ffi::ggml_add(ctx, kq.ptr(), mask.ptr()) });
        let scores = g.wrap(unsafe { ffi::ggml_soft_max(ctx, mask_add.ptr()) });
        // [n_kv, n_tokens, n_head]

        // 加权求和：需要 Vᵀscores：把 V 从 [head_dim, n_head_kv, n_kv] 变成
        // [n_kv, head_dim, n_head_kv]。注意 ggml_permute(a, ax0..ax3) 的语义是
        // 「结果的第 ax_i 维取 a 的第 i 维」，所以把 a 的 dim2 放到结果 dim0 要写 axis2=0。
        let v_perm = g.wrap(unsafe { ffi::ggml_permute(ctx, v_cache.ptr(), 1, 2, 0, 3) });
        let v_t = g.wrap(unsafe { ffi::ggml_cont(ctx, v_perm.ptr()) });
        let kqv = g.wrap(unsafe { ffi::ggml_mul_mat(ctx, v_t.ptr(), scores.ptr()) });
        // [head_dim, n_tokens, n_head]

        // [head_dim, n_tokens, n_head] → [head_dim, n_head, n_tokens] → [head_dim*n_head, n_tokens]
        let kqv = g.wrap(unsafe { ffi::ggml_permute(ctx, kqv.ptr(), 0, 2, 1, 3) });
        let kqv = g.wrap(unsafe { ffi::ggml_cont(ctx, kqv.ptr()) });
        g.wrap(unsafe { ffi::ggml_reshape_2d(ctx, kqv.ptr(), head_dim * n_head, n_tokens) })
    };

    // ---- 7. 门控 + 输出投影 ---------------------------------------------
    let gate_sig = g.wrap(unsafe { ffi::ggml_sigmoid(ctx, gate.ptr()) });
    let gated = g.wrap(unsafe { ffi::ggml_mul(ctx, attn_out.ptr(), gate_sig.ptr()) });

    let out = g.wrap(unsafe { ffi::ggml_mul_mat(ctx, attn_output.ptr(), gated.ptr()) });
    // [n_embd, n_tokens]
    (out, kv_writes)
}
