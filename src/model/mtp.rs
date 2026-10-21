// src/model/mtp.rs — MTP / NextN 头（投机解码的草稿模型）
//
// 蓝本：llama.cpp `src/models/qwen35moe.cpp` L550-741 `graph_mtp`（`LLM_GRAPH_TYPE_DECODER_MTP`）。
//
// MTP 头预测的是"**再下一个** token"：把主干在位置 `p-1` 的最终隐藏态 `h_{p-1}`
// （经 `output_norm` 之后那一份）与位置 `p` 的 token embedding 拼起来，投回 `n_embd`
// 维，再过一个完整的全注意力 + MoE 块，最后过 `shared_head_norm` 与共享的 lm_head：
//
//   e = enorm(emb(x_p))                 // [n_embd]
//   h = hnorm(h_{p-1})                  // [n_embd]
//   c = concat(e, h, dim=0)             // [2*n_embd]
//   cur = eh_proj · c                   // [n_embd]     ← 残差基准（inpSA）
//   cur = attn_norm(cur)
//   cur = attn(cur) * sigmoid(gate)     // sigmoid 门就是 Q 投影里融合的那一半
//   cur = wo(cur) + inpSA               // 残差
//   res = cur
//   cur = moe(attn_post_norm(cur))      // 路由专家 + 门控共享专家
//   cur = cur + res
//   h_nextn = shared_head_norm(cur)     // 这一份要被**链式**喂回下一步的 h
//   logits  = lm_head(h_nextn)
//
// 三个容易搞错的点（都与 llama.cpp 逐行对齐）：
//   1. 残差是 **post-eh_proj** 那一份，不是原始 h；
//   2. `h_{p-1}` 是**主干**的隐藏态（经 output_norm），不是上一步 MTP 的输出 ——
//      只有草稿链式推进时才用 MTP 自己的输出（那部分 KV 只在本轮内有效，见 runner）；
//   3. 注意力门控在 `attn_q` 投影的后半（与主干 `build_attn_layer` 完全一样），
//      所以这里直接复用 `build_attn_layer`。
//
// MTP 块是**全注意力**（`arch.is_full_attn(40) == true`），因此需要一份**独立的
// KV cache**（主干只存 0..39 层）。KV 的位置与主干序列一致。
//
// # 隐藏态的接力（两种模式）
//
// 所有 h 都借 `MtpState::h_buf` 走（`[n_embd, cap+1]`）：
//
//   **主干图**  写行 `1..=n`：就是本批各位置的最终隐藏态（经 output_norm）。
//   **Catchup** 读行 `0..n-1`（`h_buf[j]` 配 token `j`，即 teacher forcing），
//               结束后写行 0 ← 本批"最后一个已提交位置"那一行（`h_sel`）。
//   **Draft**   先 `h_buf[0] → h_in`，跑完把 `h_nextn` 写回 `h_buf[0]`（链的下一跳）。
//
// llama.cpp 在主机侧做同样的右移（把 `h_nextn` 的 D2H 再 H2D 移位），我们全程留在
// device 上，只多两条 8 KiB 的 cpy。

use crate::backend::ctx::{GgmlCtx, GraphBuilder, StateTensor, Tensor};
use crate::backend::ffi;
use crate::engine::kv::KvCache;

use super::arch::Qwen35MoeArch;
use super::attn::{build_attn_layer, RopeParams};
use super::moe::build_moe_ffn;
use super::weights::{ModelWeights, MtpWeights};

/// MTP 头的常驻状态（跨步存活，分配在 persistent buffer 里）。
pub struct MtpState {
    /// MTP 块的独立 KV cache（1 个全注意力层）
    pub kv: KvCache,
    /// 主干隐藏态暂存 `[n_embd, cap + 1]`（行 0 是"上一个已提交位置"）
    pub h_buf: StateTensor,
    /// 单行草稿步的输入 h `[n_embd, 1]`（由 `h_buf` 行 0 接力而来）
    pub h_in: StateTensor,
    /// `h_buf` 能容纳的处理长度（不含行 0）
    pub cap: usize,
}

impl MtpState {
    /// 预分配（必须在 `BackendBuffer::from_ctx` 之前调用）。
    ///
    /// `cap` = 一次前向最多处理多少 token（`max(ubatch, n_max+1)`）。
    pub fn new(
        ctx: &GgmlCtx,
        arch: &Qwen35MoeArch,
        kv_cap: usize,
        v_type: ffi::GgmlType,
        cap: usize,
    ) -> Self {
        let n_embd = arch.n_embd as i64;
        Self {
            kv: KvCache::new(ctx, 1, arch.head_dim as i64, arch.n_head_kv as i64, kv_cap, v_type),
            h_buf: StateTensor::new(ctx.new_tensor_2d(
                ffi::GgmlType::F32,
                n_embd,
                cap as i64 + 1,
            )),
            h_in: StateTensor::new(ctx.new_tensor(ffi::GgmlType::F32, &[n_embd])),
            cap,
        }
    }

    /// `h_buf` 行 `[0, n)` 的视图：Catchup 的 `h` 输入。
    pub fn h_src_view<'a>(&self, ctx: &'a GgmlCtx, n: i64) -> Tensor<'a> {
        let n_embd = self.h_buf.ne()[0];
        let row = (n_embd as usize) * 4;
        ctx.wrap(unsafe { ffi::ggml_view_2d(ctx.as_ptr(), self.h_buf.ptr(), n_embd, n, row, 0) })
    }

    /// `h_buf` 行 `[1, n+1)` 的视图：主干图写 h 的目标；Catchup 也从这里选新的行 0。
    pub fn h_dst_view<'a>(&self, ctx: &'a GgmlCtx, n: i64) -> Tensor<'a> {
        let n_embd = self.h_buf.ne()[0];
        let row = (n_embd as usize) * 4;
        ctx.wrap(unsafe { ffi::ggml_view_2d(ctx.as_ptr(), self.h_buf.ptr(), n_embd, n, row, row) })
    }

    /// `h_buf` 行 0 的字节数（检查点里保存的只有这一行）。
    pub fn h_row0_view_bytes(&self) -> usize {
        (self.h_buf.ne()[0] as usize) * 4
    }

    /// `h_buf` 行 0 的视图（单元素）。
    pub fn h_row0_view<'a>(&self, ctx: &'a GgmlCtx) -> Tensor<'a> {
        let n_embd = self.h_buf.ne()[0];
        let row = (n_embd as usize) * 4;
        ctx.wrap(unsafe { ffi::ggml_view_2d(ctx.as_ptr(), self.h_buf.ptr(), n_embd, 1, row, 0) })
    }
}

/// 一次 MTP 前向要做什么。
#[derive(Clone, Copy, Debug)]
pub enum MtpMode {
    /// **KV 重建 / teacher forcing**：跑完本批（`n` 个 token）并把 MTP KV 写到
    /// 位置 `[s, s+n)`，然后 `h_buf[0] ← h_buf[h_sel + 1]`（本批最后一个**已提交**
    /// 位置的主干 h，由 `h_sel_idx` 这个 I32[1] 图输入给出）。不算 logits
    /// （省掉一次 ~300 MB 的 lm_head 读）。
    Catchup,
    /// **单行草稿 / 种子**：`h_in ← h_buf[0]`，跑 1 个 token，输出 logits，
    /// 并把 `h_nextn` 写回 `h_buf[0]`（草稿链的下一跳）。
    Draft,
}

/// 一次 MTP 前向的产物。
pub struct MtpGraph<'a> {
    /// `[n_embd, n]`：本批各位置的 MTP 隐藏态（`shared_head_norm` 之后）
    pub h_nextn: Tensor<'a>,
    /// 只有 [`MtpMode::Draft`] 有：`[n_vocab, 1]`
    pub logits: Option<Tensor<'a>>,
    /// KV 写入节点
    pub kv_writes: [Tensor<'a>; 2],
}

/// 构建 MTP 头的前向图（一次一图，不做跨步复用 —— MTP 图只有 ~70 个节点，
/// 重建 + 分配的开销远小于主干图的 1.4 ms）。
///
/// 调用方必须先 `mtp.kv.begin_step(n)` 定好本步的 KV 写入区间。
#[allow(clippy::too_many_arguments)]
pub fn build_mtp<'a>(
    g: &'a GgmlCtx,
    builder: &mut GraphBuilder<'a>,
    w: &MtpWeights,
    m: &ModelWeights,
    arch: &Qwen35MoeArch,
    st: &mut MtpState,
    mode: MtpMode,
    tokens: &Tensor<'a>,
    pos: &Tensor<'a>,
    mask: &Tensor<'a>,
    kv_idxs: &Tensor<'a>,
    n: i64,
    rope: &RopeParams,
    flash_attn: bool,
    // `Catchup` 用：`h_buf[0]` 要换成主干 h 的第几行（本批里最后一个已提交位置）。
    // 因为 MTP 图每步重建，这里直接烙进 view 偏移，不需要动态索引张量。
    h_sel: usize,
) -> Result<MtpGraph<'a>, String> {
    let ctx = g.as_ptr();
    let n_embd = arch.n_embd as i64;
    let eps = arch.norm_rms_eps;

    // ---- 0. 隐藏态接力（必须在读之前进入图：同一张图里还会写 h_buf[0]）----
    let h_in = match mode {
        MtpMode::Draft => {
            let dst = st.h_in.get(g);
            let src = st.h_row0_view(g);
            let cp = g.wrap(unsafe { ffi::ggml_cpy(ctx, src.ptr(), dst.ptr()) });
            builder.expand(&cp);
            dst
        }
        MtpMode::Catchup { .. } => st.h_src_view(g, n),
    };

    // ---- 1. embedding 与两个 norm，再 concat ------------------------------
    let emb_w = if w.embed_tokens.is_null() { m.token_embd } else { w.embed_tokens };
    let tok_embd = g.wrap(unsafe { ffi::ggml_get_rows(ctx, emb_w.get(g).ptr(), tokens.ptr()) });

    let h_norm = rms_norm(g, &h_in, &w.hnorm.get(g), n_embd, n, eps);
    let e_norm = rms_norm(g, &tok_embd, &w.enorm.get(g), n_embd, n, eps);
    let concat = g.wrap(unsafe { ffi::ggml_concat(ctx, e_norm.ptr(), h_norm.ptr(), 0) });

    let cur = g.wrap(unsafe { ffi::ggml_mul_mat(ctx, w.eh_proj.get(g).ptr(), concat.ptr()) });
    // 注意：`Tensor` 不是 Copy，残差基准用裸指针存着（数据流上是同一个张量）
    let inp_sa = cur.ptr();

    // ---- 2. 注意力块（复用主干的全注意力层：含 sigmoid 门 + IMROPE + 独立 KV）----
    let normed = rms_norm(g, &cur, &w.block.attn_norm.get(g), n_embd, n, eps);
    let (attn_out, kv_writes) = build_attn_layer(
        g,
        &w.block,
        &normed,
        &mut st.kv,
        0,
        pos,
        mask,
        kv_idxs,
        n,
        arch,
        rope,
        flash_attn,
    );
    // KV 写必须排在注意力读之前进入图（ggml 按 expand 顺序执行节点）
    for wr in &kv_writes {
        builder.expand(wr);
    }

    let cur = g.wrap(unsafe { ffi::ggml_add(ctx, attn_out.ptr(), inp_sa) });
    let ffn_residual = cur.ptr();

    // ---- 3. MoE（含门控共享专家，与主干同一套实现）------------------------
    let post = rms_norm(g, &cur, &w.block.attn_post_norm.get(g), n_embd, n, eps);
    let ffn = build_moe_ffn(
        g,
        builder,
        &w.block,
        &post,
        arch.n_expert as i64,
        arch.n_expert_used as i64,
        arch.expert_ffn_len as i64,
        n_embd,
        n,
    );
    let cur = g.wrap(unsafe { ffi::ggml_add(ctx, ffn.ptr(), ffn_residual) });

    // ---- 4. shared_head_norm → h_nextn -----------------------------------
    let head_norm = if w.shared_head_norm.is_null() { m.output_norm } else { w.shared_head_norm };
    let h_nextn = rms_norm(g, &cur, &head_norm.get(g), n_embd, n, eps);
    // 让整条链进入图（这些节点的"输出"不在依赖链上时会被漏掉）
    builder.expand(&h_nextn);

    let logits = match mode {
        MtpMode::Catchup => {
            // h_buf 行 0 ← 本批第 h_sel 行（= 最后一个已提交位置的主干 h）
            assert!((h_sel as i64) < n, "Catchup: h_sel={h_sel} 超出一批 {n} 个 token");
            let rows = st.h_dst_view(g, n);
            let row = g.wrap(unsafe {
                ffi::ggml_view_2d(
                    ctx,
                    rows.ptr(),
                    n_embd,
                    1,
                    rows.nb()[1] as usize,
                    h_sel * (n_embd as usize) * 4,
                )
            });
            let dst = st.h_row0_view(g);
            let cp = g.wrap(unsafe { ffi::ggml_cpy(ctx, row.ptr(), dst.ptr()) });
            builder.expand(&cp);
            None
        }
        MtpMode::Draft => {
            // 当前步只有 1 行，直接取第 0 行
            let row = g.wrap(unsafe {
                ffi::ggml_view_2d(ctx, h_nextn.ptr(), n_embd, 1, (n_embd as usize) * 4, 0)
            });
            let dst = st.h_row0_view(g);
            let cp = g.wrap(unsafe { ffi::ggml_cpy(ctx, row.ptr(), dst.ptr()) });
            builder.expand(&cp);

            let head_w = if w.shared_head_head.is_null() { m.output } else { w.shared_head_head };
            let lg = g.wrap(unsafe { ffi::ggml_mul_mat(ctx, head_w.get(g).ptr(), row.ptr()) });
            builder.expand(&lg);
            Some(lg)
        }
    };

    Ok(MtpGraph { h_nextn, logits, kv_writes })
}

/// RMS 归一化 + 逐元素乘权重（与 `graph.rs::rms_norm` 同）。
fn rms_norm<'a>(
    g: &'a GgmlCtx,
    x: &Tensor<'a>,
    weight: &Tensor<'a>,
    n_embd: i64,
    n_tokens: i64,
    eps: f32,
) -> Tensor<'a> {
    let ctx = g.as_ptr();
    assert_eq!(x.ne()[0], n_embd);
    assert_eq!(x.ne()[1], n_tokens);
    assert_eq!(weight.ne()[0], n_embd);
    let normed = g.wrap(unsafe { ffi::ggml_rms_norm(ctx, x.ptr(), eps) });
    g.wrap(unsafe { ffi::ggml_mul(ctx, normed.ptr(), weight.ptr()) })
}
