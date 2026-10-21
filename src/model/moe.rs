// src/model/moe.rs — Task 5: qwen35moe MoE FFN graph construction
//
// 移植 llama.cpp `build_moe_ffn`（qwen35moe.cpp L494-548）到我们的 FFI 层。
//
// 计算图流程（对应 build_moe_ffn）：
//   1. logits = mul_mat(gate_inp, h)              → [n_expert, n_tokens]
//   2. probs  = softmax(logits)                   → [n_expert, n_tokens]
//   3. sel    = argsort_top_k(probs, n_expert_used) → [n_expert_used, n_tokens]
//   4. w      = get_rows(reshape_3d(probs, 1, n_expert, n_tokens), sel) → [1, n_expert_used, n_tokens]
//   5. w_norm = div(w, clamp(sum_rows(w), eps, inf))  （归一化专家权重）
//      注意 sum_rows 固定对 dim0 求和，而 w 的 dim0 是 1（见第 4 步的形状），
//      所以必须先 reshape_2d 成 [n_expert_used, n_tokens] 再求和，否则归一化是空操作。
//   6. h3     = reshape_3d(h, n_embd, 1, n_tokens)
//   7. up/gate = mul_mat_id(experts, h3, sel)       → [n_ff, n_expert_used, n_tokens]
//   8. act     = swiglu(gate, up)（GLU 算子，触发 ggml-cuda 的 matmul+GLU 融合）                    → [n_ff, n_expert_used, n_tokens]
//   9. experts = mul_mat_id(down_exps, act, sel)    → [n_embd, n_expert_used, n_tokens]
//  10. moe_out = sum over experts (weighted by w_norm) → [n_embd, n_tokens]
//      第 i 个专家是 [n_embd, n_tokens] 的 view_2d（行跨步 nb[2]，偏移 i*nb[1]），
//      逐个 ggml_add 相加（与 llama.cpp 相同）。
//
// qwen35moe 传入的门控参数：SOFTMAX 门控、norm_w = true、
// expert_weights_scale = 0（即 llama.cpp 里不缩放）、SILU 激活。
//
// 共享专家（build_layer_ffn 中 ffn_up_shexp != nullptr 时）：
//   shexp = standard FFN(h)                          → [n_embd, n_tokens]
//   sg    = sigmoid(mul_mat(ffn_gate_inp_shexp, h))  → [1, n_tokens]
//   shexp = shexp * sg
//   out   = moe_out + shexp
//
// 注意：ggml 的 mul_mat_id(as_, b, ids) 语义为：
//   output[r, e, t] = sum_c as[c, r, ids[e,t]] * b[c, e, t]
// 因此 expert 权重张量必须具有形状 [K=n_embd, R=n_ff, NE=n_expert]（ne[0]=K 与 b 的 ne[0] 匹配）。

use super::weights::LayerWeights;
use crate::backend::ffi::{self};
use crate::backend::ctx::{GgmlCtx, GraphBuilder, Tensor};

/// `enum ggml_glu_op` 里的 SWIGLU（REGLU=0, GEGLU=1, SWIGLU=2）。
///
/// 只有用 `GGML_OP_GLU` 才会触发 ggml-cuda 的 `{MUL_MAT(_ID), MUL_MAT(_ID), GLU}`
/// 融合：gate/up 两个 matmul 与 SwiGLU 合成**一个 kernel**。写成 `silu(gate) * up`
/// （UNARY + MUL）时这条融合完全不生效 —— 40 层 × 2 处 = 80 个多余 kernel（P8）。
pub const GLU_OP_SWIGLU: i32 = 2;

const EPSILON: f32 = 6.103515625e-5; // F16 最小正规数，llama.cpp 用同一常量防除零

/// 在 ggml 上下文中构建 MoE FFN 计算图。
///
/// # 参数
/// - `g`: GgmlCtx，用于创建张量
/// - `w`: &LayerWeights，包含当前层的专家权重
/// - `h`: &Tensor，形状 [n_embd, n_tokens]（或 [n_tokens, n_embd]，会自动 reshape）
/// - `n_expert`: 专家总数
/// - `n_expert_used`: 每 token 选中的专家数 (top-k)
/// - `n_ff`: 每个专家的 FFN 隐藏层大小
/// - `n_embd`: 嵌入维度
/// - `n_tokens`: 当前 token 数
///
/// # 返回
/// Tensor，形状 [n_embd, n_tokens]，即 MoE FFN 输出（含共享专家）。
pub fn build_moe_ffn<'a>(
    g: &'a GgmlCtx,
    builder: &mut GraphBuilder<'a>,
    w: &LayerWeights,
    h: &Tensor<'a>,
    n_expert: i64,
    n_expert_used: i64,
    _n_ff: i64,
    n_embd: i64,
    n_tokens: i64,
) -> Tensor<'a> {
    let ctx = g.as_ptr();

    // --- Step 1: logits = mul_mat(gate_inp, h) → [n_expert, n_tokens] ---
    // gate_inp shape: [n_embd, n_expert] (2D)
    // h shape: [n_embd, n_tokens] (2D)
    // ggml_mul_mat(a, b) computes a^T @ b, so result is [n_expert, n_tokens]
    let logits = unsafe { ffi::ggml_mul_mat(ctx, w.ffn_gate_inp.as_ptr(), h.ptr()) };
    let logits = g.wrap(logits);

    // --- Step 2: probs = softmax(logits) → [n_expert, n_tokens] ---
    let probs = unsafe { ffi::ggml_soft_max(ctx, logits.ptr()) };
    let probs = g.wrap(probs);

    // --- Step 3: sel = argsort_top_k(probs, n_expert_used) → [n_expert_used, n_tokens] ---
    // ggml_argsort_top_k returns indices of top-k values along dim 0 (experts)
    let sel = unsafe { ffi::ggml_argsort_top_k(ctx, probs.ptr(), n_expert_used as i32) };
    let sel = g.wrap(sel);

    // --- Step 4: w = get_rows(reshape_3d(probs, 1, n_expert, n_tokens), sel) ---
    // reshape probs to [1, n_expert, n_tokens] so get_rows can gather along the expert dimension
    let probs_3d = unsafe { ffi::ggml_reshape_3d(ctx, probs.ptr(), 1, n_expert, n_tokens) };
    let probs_3d = g.wrap(probs_3d);
    // ggml_get_rows(a, b) 返回 [a->ne[0], b->ne[0], b->ne[1], ...]，
    // 因此这里得到的是 [1, n_expert_used, n_tokens]（不是 [n_expert_used, 1, n_tokens]）。
    let weights = unsafe { ffi::ggml_get_rows(ctx, probs_3d.ptr(), sel.ptr()) };
    let weights = g.wrap(weights);

    // --- Step 5: 归一化专家权重 → [1, n_expert_used, n_tokens] ---
    // 对齐 llama.cpp build_moe_ffn：先 reshape 成 [E, n_tokens] 再 sum_rows，让求和
    // 落在专家维上。ggml_sum_rows 固定对 dim0 求和，而 weights 的 dim0 是 1
    // （见上一步的形状），所以这次 reshape 不能省 —— 否则求和对长度为 1 的维做，
    // w_norm 恒等于 1，top-k 权重被整个丢掉。
    let weights_2d = unsafe {
        ffi::ggml_reshape_2d(ctx, weights.ptr(), n_expert_used, n_tokens)
    };
    let weights_2d = g.wrap(weights_2d);

    let w_sum = unsafe { ffi::ggml_sum_rows(ctx, weights_2d.ptr()) }; // [1, n_tokens]
    let w_sum = g.wrap(w_sum);
    // clamp 到 [eps, inf) 防止除零（eps 取 F16 最小正规数，与 llama.cpp 一致）
    let w_sum_clamped = unsafe { ffi::ggml_clamp(ctx, w_sum.ptr(), EPSILON, f32::INFINITY) };
    let w_sum_clamped = g.wrap(w_sum_clamped);

    let w_norm_2d = unsafe { ffi::ggml_div(ctx, weights_2d.ptr(), w_sum_clamped.ptr()) };
    let w_norm_2d = g.wrap(w_norm_2d);

    // 回到 [1, n_expert_used, n_tokens]，以便和 experts 沿 dim0 广播相乘
    let w_norm = unsafe {
        ffi::ggml_reshape_3d(ctx, w_norm_2d.ptr(), 1, n_expert_used, n_tokens)
    };
    let w_norm = g.wrap(w_norm);

    // ---- 关键：把路由子图**提前**展开 -------------------------------------
    //
    // 对齐 llama.cpp `llama-graph.cpp:2163-2164` 的
    //     //call early so that topk-moe can be used
    //     ggml_build_forward_expand(gf, weights);
    //
    // ggml-cuda 的 `ggml_cuda_topk_moe_fusion`（`ggml-cuda.cu:2813`）要求节点序列
    // **严格相邻**：
    //     SOFT_MAX → RESHAPE → ARGSORT → VIEW → GET_ROWS
    //              → RESHAPE → SUM_ROWS → CLAMP → DIV → RESHAPE
    // 而 `cgraph->nodes` 的顺序由**展开时的后序 DFS** 决定，不是源码顺序。
    //
    // 不提前展开会怎样（实测）：DFS 从最终输出出发，先经
    // `weighted = mul(experts, w_norm)` 的 src[0]（experts → mul_mat_id）摸到 `sel`，
    // 于是 ARGSORT/VIEW 落在 RESHAPE **之前**；等 DFS 走到 w_norm 分支时，GET_ROWS
    // 已经和它们隔了 4 个 MoE 节点 → 模式失配 → 整条路由退化成 ~6 个 kernel
    // （SOFT_MAX / ARGSORT / GET_ROWS / SUM_ROWS / CLAMP / DIV 各一个）。
    //
    // 提前展开让这段连续落盘，融合命中后 40 层的路由合计只剩 40 个 kernel。
    builder.expand(&w_norm);

    // --- Step 6: reshape h to [n_embd, 1, n_tokens] for mul_mat_id ---
    let h3 = unsafe { ffi::ggml_reshape_3d(ctx, h.ptr(), n_embd, 1, n_tokens) };
    let h3 = g.wrap(h3);

    // --- Steps 7-8: compute up and gate via mul_mat_id ---
    // For qwen35moe, the experts are separate (not merged):
    //   up_exps shape: [K=n_embd, R=n_ff, NE=n_expert] (ne[0]=K, ne[1]=R, ne[2]=NE)
    //   gate_exps shape: [K=n_embd, R=n_ff, NE=n_expert]
    //   down_exps shape: [K=n_ff, R=n_embd, NE=n_expert] (ne[0]=K=n_ff, ne[1]=R=n_embd, ne[2]=NE)
    // mul_mat_id(up_exps, h3, sel) → [n_ff, n_expert_used, n_tokens]
    // mul_mat_id(gate_exps, h3, sel) → [n_ff, n_expert_used, n_tokens]

    let up = unsafe { ffi::ggml_mul_mat_id(ctx, w.ffn_up_exps.as_ptr(), h3.ptr(), sel.ptr()) };
    let up = g.wrap(up);

    let gate = unsafe { ffi::ggml_mul_mat_id(ctx, w.ffn_gate_exps.as_ptr(), h3.ptr(), sel.ptr()) };
    let gate = g.wrap(gate);

    // --- Step 8: activation = swiglu(gate, up) → [n_ff, n_expert_used, n_tokens] ---
    //
    // 用 GLU 算子而不是 silu+mul：ggml-cuda 的 `{MUL_MAT_ID, MUL_MAT_ID, GLU}` 融合
    // 会把 gate/up 两个 matmul 与 SwiGLU **合成一个 kernel**（P8 实测：我们之前写成
    // silu*mul，这条融合完全没生效，MoE 每层多 2 个 kernel，40 层就是 80 个）。
    let act = unsafe { ffi::ggml_glu_split(ctx, gate.ptr(), up.ptr(), GLU_OP_SWIGLU) };
    let act = g.wrap(act);

    // --- Step 9: experts = mul_mat_id(down_exps, act, sel) → [n_embd, n_expert_used, n_tokens] ---
    let experts = unsafe { ffi::ggml_mul_mat_id(ctx, w.ffn_down_exps.as_ptr(), act.ptr(), sel.ptr()) };
    let experts = g.wrap(experts);

    // --- Step 10: moe_out = sum over experts (weighted by w_norm) → [n_embd, n_tokens] ---
    // experts: [n_embd, E, n_tokens]，w_norm: [1, E, n_tokens] → 广播到每个专家。
    // 随后按 llama.cpp 的做法逐专家切片相加：第 i 个专家在内存中是 [n_embd, n_tokens]
    // 的视图，行跨步为 nb[2]，偏移为 i*nb[1]。
    //
    // 不能改用 reshape_2d([n_embd, E, T] → [E, n_embd*T]) + sum_rows：那个 reshape
    // 把 flat index 重新切成 (i0 + i1*n_embd) mod E，会混掉 embedding 与 expert 两个
    // 维度，求和出来的分组没有意义。
    let weighted = unsafe { ffi::ggml_mul(ctx, experts.ptr(), w_norm.ptr()) };
    let weighted = g.wrap(weighted);

    let mut moe_out = g.wrap(unsafe {
        ffi::ggml_view_2d(ctx, weighted.ptr(), n_embd, n_tokens, weighted.nb()[2] as usize, 0)
    });
    for i in 1..n_expert_used {
        let expert_i = g.wrap(unsafe {
            ffi::ggml_view_2d(
                ctx,
                weighted.ptr(),
                n_embd,
                n_tokens,
                weighted.nb()[2] as usize,
                i as usize * weighted.nb()[1] as usize,
            )
        });
        moe_out = g.wrap(unsafe { ffi::ggml_add(ctx, moe_out.ptr(), expert_i.ptr()) });
    }

    if n_expert_used == 1 {
        // 避免把非连续张量返回给上层
        moe_out = g.wrap(unsafe { ffi::ggml_cont(ctx, moe_out.ptr()) });
    }

    // --- Shared expert (if present) ---
    // In qwen35moe, the shared expert is present when ffn_up_shexp is not NULL.
    if !w.ffn_up_shexp.is_null() && !w.ffn_gate_shexp.is_null() && !w.ffn_down_shexp.is_null() {
        // Standard FFN: down(swiglu(gate(h), up(h)))（GLU 算子触发 matmul+GLU 融合）
        let sh_gate = unsafe { ffi::ggml_mul_mat(ctx, w.ffn_gate_shexp.as_ptr(), h.ptr()) };
        let sh_gate = g.wrap(sh_gate);

        let sh_up = unsafe { ffi::ggml_mul_mat(ctx, w.ffn_up_shexp.as_ptr(), h.ptr()) };
        let sh_up = g.wrap(sh_up);

        let sh_act = g.wrap(unsafe {
            ffi::ggml_glu_split(ctx, sh_gate.ptr(), sh_up.ptr(), GLU_OP_SWIGLU)
        });

        let sh_expert_out = unsafe { ffi::ggml_mul_mat(ctx, w.ffn_down_shexp.as_ptr(), sh_act.ptr()) };
        let sh_expert_out = g.wrap(sh_expert_out);

        // Shared expert gating: sigmoid(mul_mat(ffn_gate_inp_shexp, h)) → [1, n_tokens]
        let shared_gate = unsafe { ffi::ggml_mul_mat(ctx, w.ffn_gate_inp_shexp.as_ptr(), h.ptr()) };
        let shared_gate = g.wrap(shared_gate);
        let shared_gate = unsafe { ffi::ggml_sigmoid(ctx, shared_gate.ptr()) };
        let shared_gate = g.wrap(shared_gate);

        // Apply gate: sh_expert_out * shared_gate
        let sh_gated = unsafe { ffi::ggml_mul(ctx, sh_expert_out.ptr(), shared_gate.ptr()) };
        let sh_gated = g.wrap(sh_gated);

        // Final output: moe_out + sh_gated
        let out = unsafe { ffi::ggml_add(ctx, moe_out.ptr(), sh_gated.ptr()) };
        return g.wrap(out);
    }

    moe_out
}

/// 构建标准 FFN（用于共享专家），对应 llama.cpp `build_ffn` with SILU + PAR.
///
/// ffn(h) = down(swiglu(gate(h), up(h)))（GLU 算子触发 matmul+GLU 融合）
pub fn build_standard_ffn<'a>(
    g: &'a GgmlCtx,
    w: &LayerWeights,
    h: &Tensor<'a>,
    _n_ff: i64,
    _n_embd: i64,
) -> Tensor<'a> {
    let ctx = g.as_ptr();

    let gate = unsafe { ffi::ggml_mul_mat(ctx, w.ffn_gate_shexp.as_ptr(), h.ptr()) };
    let gate = g.wrap(gate);

    let up = unsafe { ffi::ggml_mul_mat(ctx, w.ffn_up_shexp.as_ptr(), h.ptr()) };
    let up = g.wrap(up);

    let act = g.wrap(unsafe { ffi::ggml_glu_split(ctx, gate.ptr(), up.ptr(), GLU_OP_SWIGLU) });

    let out = unsafe { ffi::ggml_mul_mat(ctx, w.ffn_down_shexp.as_ptr(), act.ptr()) };
    g.wrap(out)
}