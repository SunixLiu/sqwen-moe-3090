// src/model/graph.rs — 整模型前向图组装（Task 8）
//
// 蓝本：llama.cpp `src/models/qwen35moe.cpp` L178-240 主循环
//
//   embd = get_rows(tok_embd, tokens)
//   for il in 0..n_layer:
//       normed   = rms_norm(h, attn_norm)
//       attn_out = full attention(il) | gated delta net(il)
//       h        = attn_out + h                      // 残差
//       post     = rms_norm(h, attn_post_norm)
//       ffn      = moe_ffn(post)                     // 含共享专家
//       h        = ffn + h                           // 残差
//   logits = mul_mat(output, rms_norm(h, output_norm))
//
// 装配顺序（`Model::from_gguf` 的三阶段 + 引擎状态）：
//
//   1. ctx = GgmlCtx::new_no_alloc()
//      weights = ModelWeights::create(&ctx, &gguf, &arch)
//      state   = EngineState::new(&ctx, &arch, kv_cap)      // 常驻状态张量
//   2. persistent = BackendBuffer::from_ctx(&ctx, kind)     // 权重 + 状态一次分配
//      weights.load_data(&ctx, &gguf)
//      state.zero_states()                                  // 状态清零
//   3. 每一步：
//        g = GraphBuilder::new_with_size(&ctx, NODES)
//        logits = forward(&ctx, &mut g, &weights, &arch, &mut state, &tokens, start)
//        step_buf = BackendBuffer::from_ctx(&ctx, kind)      // 只分配本步新张量
//        state.fill_inputs(&tokens)                          // 写 tokens/pos/mask
//        g.compute(&step_buf)
//
// 第 2、3 步要分别调一次 `from_ctx`：`ggml_backend_alloc_ctx_tensors` 只处理
// `data == NULL` 的张量，所以第二次调用只会给本步新建的图节点分配内存，权重与
// 状态仍留在常驻 buffer 里。常驻状态要跨 step 存活，因此两步必须分开。
//
// 注意 `KvCache::pending_writes()` 与 GDN 的状态写回都是「不在输出依赖链上」的
// cpy 节点：`forward` 会先把它们 expand 进图，再 expand 输出，这样它们的执行顺序
// 必然早于读取状态/缓存的注意力节点（ggml 按 expand 顺序执行节点）。

use crate::backend::ctx::{GgmlCtx, GraphBuilder, StateTensor, Tensor};
use crate::backend::ffi;
use crate::engine::kv::KvCache;
use crate::model::attn::{build_attn_layer, RopeParams};
use crate::model::gdn::{build_gdn_layer, GdnState};
use crate::model::moe::build_moe_ffn;
use crate::model::mtp::MtpState;
use crate::model::weights::ModelWeights;

use super::arch::Qwen35MoeArch;

/// 一次前向需要的图节点上限（40 层 × 每层约 40 个节点）。
pub const GRAPH_NODE_BUDGET: usize = 16384;

/// 本步的图输入张量（由 `forward` 创建，调用方分配后填值）。
struct StepTensors {
    tokens: StateTensor,
    pos: StateTensor,
    mask: StateTensor,
    /// KV 写入的行号（I64 [n_tokens]）：内容是本步各 token 的绝对 KV 位置。
    ///
    /// 写入走 `ggml_set_rows(cache, rows, idxs)` —— dst 是整块 cache（指针固定），
    /// 位置只在**数据**里。这样每步的图节点属性逐字节一致，ggml-cuda 的 CUDA graph
    /// 才能重放（用 view 偏移做 cpy 会让指针每步都变，graph 永远退回 eager）。
    kv_idxs: StateTensor,
    /// 需要输出 logits 的行号（I32 [n_out]）；`None` = 所有行都输出。
    ///
    /// 生成时只有**最后一个位置**的 logits 有用，但旧实现会对全部 n 行算 lm_head
    /// 并把整块 [n_vocab, n] 拷回主机 —— 512-token 的 chunk 就是
    /// 248320×512×4 ≈ **508 MB** 的一次 D2H，实测占掉 prefill 的 317 ms（PP 的最大浪费）。
    /// 现在按 llama.cpp 的做法（`cur = ggml_get_rows(cur, inp_out_ids)`）在最后一层
    /// 先取行再算输出投影。
    out_ids: Option<StateTensor>,
    /// GDN 递推状态的**平面选择**（I32 [1]）；`None` = 不走 `get_rows`（`n_rs == 0`）。
    ///
    /// 投机解码部分接受时用它把递推状态退回"接受前缀之后"（见 `GdnState` 的说明）。
    sel: Option<StateTensor>,
}

/// 引擎的常驻状态：每层的 GDN 状态 + 全注意力层的 KV cache。
///
/// 不带 ctx 生命周期（用 [`StateTensor`]）：`Engine` 要同时拥有 `GgmlCtx` 和它，
/// 借用版会构成自引用结构。不变式见 `StateTensor` 的文档。
pub struct EngineState {
    /// 按主干层索引；只有递归层是 `Some`。常驻张量跨步保留。
    pub gdn: Vec<Option<GdnState>>,
    pub kv: KvCache,
    rope: RopeParams,
    /// 全注意力层 → KV cache 槽位
    kv_slot: Vec<Option<usize>>,
    /// 是否启用 FlashAttention（P4）：决定 attn 走 FA 还是手工路径。
    flash_attn: bool,
    /// 主机侧并行度（`-t/--threads`）：目前用于并行填充因果 mask。
    /// 大上下文下 mask 每块上百 MB，这一步是纯 CPU 访存负载（见 `fill_mask_rows`）。
    n_threads: usize,
    /// 常驻的 mask 字节缓冲（**复用**，避免每步重新分配 → 缺页中断）。
    ///
    /// 大上下文下 mask 每块上百 MB（ctx=172032、ubatch=512、F16 ≈ 176 MB）。
    /// 每步 `vec![0u8; N]` 会让 OS 逐页发新页，填充时每个 4KB 页一次 page fault
    /// （17.6 万次），实测要 87 ms 且**几乎无法用多线程改善**（缺页在内核里串行：
    /// t=8 只降到 68 ms）。复用同一块缓冲后只剩 memcpy/memset，多线程才真正吃带宽。
    mask_buf: Vec<u8>,
    step: Option<StepTensors>,
    /// 本步需要输出 logits 的行号（`None` = 全部行）；由 `forward` 设定，
    /// `fill_inputs` 据此写入 `out_ids` 张量的内容。
    out_rows: Option<Vec<i32>>,
    step_n: usize,
    step_start: usize,
    /// 已写入 KV cache 的 token 数（= start_pos 的期望值）
    pub n_past: usize,
    /// GDN 每个序列保留的回退快照数（0 = 关闭；投机解码 = `--spec-draft-n-max`）
    pub n_rs: usize,
    /// 下一步读哪个递推状态平面（仅在投机解码的校验批前设置一次，用完归零）
    sel_value: usize,
    /// MTP 头的常驻状态；`None` = 未启用投机解码
    pub mtp: Option<MtpState>,
}

impl EngineState {
    /// 预分配 GDN 状态与 KV cache。
    ///
    /// `kv_cap` 是 KV cache 的容量，必须 ≥ 整个序列长度。因果 mask 是**按步**创建
    /// 的 `[n_kv, n_tokens]`（见 `forward`），不再按 `kv_cap²` 预分配。
    pub fn new(
        ctx: &GgmlCtx,
        arch: &Qwen35MoeArch,
        kv_cap: usize,
        v_type: ffi::GgmlType,
        flash_attn: bool,
        n_threads: usize,
        n_rs: usize,
        mtp_cap: usize,
    ) -> Result<Self, String> {
        let n_layer = arch.n_layer_trunk() as usize;

        let head_dim = arch.head_dim as i64;
        let mut gdn = Vec::with_capacity(n_layer);
        let mut kv_slot = Vec::with_capacity(n_layer);
        let mut n_full = 0usize;
        for il in 0..n_layer {
            if arch.is_full_attn(il as u32) {
                gdn.push(None);
                kv_slot.push(Some(n_full));
                n_full += 1;
            } else {
                gdn.push(Some(GdnState::new(ctx, arch, n_rs)));
                kv_slot.push(None);
            }
        }

        let kv = KvCache::new(ctx, n_full, head_dim, arch.n_head_kv as i64, kv_cap, v_type);
        // MTP 头的常驻状态：独立 KV（1 个全注意力层）+ h 暂存缓冲
        let mtp = if n_rs > 0 && arch.n_layer_nextn > 0 {
            Some(MtpState::new(ctx, arch, kv_cap, v_type, mtp_cap.max(1)))
        } else {
            None
        };

        Ok(Self {
            gdn,
            kv,
            flash_attn,
            n_threads: n_threads.max(1),
            mask_buf: Vec::new(),
            rope: RopeParams::text(
                arch.rope_dim_count as i32,
                arch.rope_sections,
                arch.rope_freq_base,
                // `n_ctx_orig` = **模型训练时的上下文长度**（YaRN 缩放用），不是我们的
                // kv 容量。llama.cpp 同样用 `hparams.n_ctx_orig_yarn = n_ctx_train`。
                // 缺失时退回 kv_cap。
                if arch.n_ctx_train > 0 {
                    arch.n_ctx_train as i32
                } else {
                    kv_cap as i32
                },
            ),
            kv_slot,
            step: None,
            out_rows: None,
            step_n: 0,
            step_start: 0,
            n_past: 0,
            n_rs,
            sel_value: 0,
            mtp,
        })
    }

    /// 设置递推状态的读取平面（投机解码：`n_rs - m`，m = 本轮接受的草稿数）。
    ///
    /// **一次性**：调用方在读它的那一步之后必须归零，否则下一步会读到过期的平面。
    pub fn set_recurrent_sel(&mut self, sel: usize) {
        self.sel_value = sel.min(self.n_rs);
    }

    pub fn recurrent_sel(&self) -> usize {
        self.sel_value
    }

    /// 把主干与 MTP 的 KV 游标一起归零（同一 context 内换序列时用）。
    pub fn reset_caches(&mut self) {
        self.kv.reset();
        if let Some(mtp) = self.mtp.as_mut() {
            mtp.kv.reset();
        }
        self.step_start = 0;
        self.step_n = 0;
        self.n_past = 0;
    }

    /// 常驻状态清零（分配之后、第一次前向之前调用一次）。
    ///
    /// backend buffer 来自对齐 malloc / cudaMalloc，不做清零；GDN 的递推状态与
    /// 卷积历史必须先归零。每个 GDN 层的 state/conv_state 大小可能不同，所以
    /// **逐层按自身尺寸**写零，不能共用一块最大尺寸的缓冲区。
    pub fn zero_states(&self) {
        for st in self.gdn.iter().flatten() {
            let n_state = (st.state.ne()[..4].iter().product::<i64>() as usize) * 4;
            let n_conv = (st.conv_state.ne()[..3].iter().product::<i64>() as usize) * 4;
            st.state.write_bytes_at(0, &vec![0u8; n_state]);
            st.conv_state.write_bytes_at(0, &vec![0u8; n_conv]);
        }
        // MTP 的 h 暂存也要清零：`h_buf` 行 0 表示"位置 -1 的主干 h"，
        // 第一个 chunk 的正确值就是 0（llama.cpp 的 `pending_h` 同样初始化为 0）。
        if let Some(mtp) = self.mtp.as_ref() {
            mtp.h_buf.write_bytes_at(0, &vec![0u8; mtp.h_buf.nbytes()]);
            mtp.h_in.write_bytes_at(0, &vec![0u8; mtp.h_in.nbytes()]);
        }
    }

    /// 检查点载荷：GDN 平面 0 快照 + （MTP 启用时）`h_buf` 行 0。
    ///
    /// **为什么 MTP 只需要存 h_buf 行 0**：MTP 的 KV 是普通注意力 KV，它的内容只
    /// 取决于 token 前缀（每个位置是 `(x_p, h_{p-1})`，而 `h_{p-1}` 只依赖 `[0, p)`）。
    /// 前缀复用的前提正是"新 prompt 的 `[0, p)` 与上一次完全相同"，所以 MTP 的 KV
    /// **本来就还在显存里且是对的**，只把游标退回去就行。唯一丢掉的派生量是行 0
    /// （`h_{p-1}`），8 KiB。
    pub fn snapshot_state(&self) -> Vec<u8> {
        let mut out = self.snapshot_gdn();
        if let Some(mtp) = self.mtp.as_ref() {
            let n = mtp.h_row0_view_bytes();
            out.extend_from_slice(&mtp.h_buf.read_bytes_at(0, n));
        }
        out
    }

    /// 从 [`EngineState::snapshot_state`] 的缓冲恢复（GDN 平面 0 + MTP `h_buf` 行 0）。
    pub fn restore_state(&self, data: &[u8]) -> Result<(), String> {
        let gdn_len = self.gdn_bytes() + 8;
        if data.len() < gdn_len {
            return Err(format!(
                "检查点载荷太短：{} < {gdn_len}（GDN 部分）",
                data.len()
            ));
        }
        self.restore_gdn(&data[..gdn_len])?;
        if let Some(mtp) = self.mtp.as_ref() {
            let n = mtp.h_row0_view_bytes();
            if data.len() < gdn_len + n {
                return Err(format!(
                    "检查点载荷太短：{} < {}（含 MTP h_buf 行 0）",
                    data.len(),
                    gdn_len + n
                ));
            }
            mtp.h_buf.write_bytes_at(0, &data[gdn_len..gdn_len + n]);
        }
        Ok(())
    }

    /// 复用上一张图时推进一步：只做 `forward` 里的 KV 位置簿记，不建图（P2c）。
    ///
    /// `forward` 在新图里会做四件与"位置"有关的事：`kv.begin_step`、`step_start`/
    /// `step_n`/`n_past`、`out_rows`。图本身（`st.step` 里的输入张量）已经在缓存里，
    /// 这里只更新这些随步变化的标量/内容，随后由 `fill_inputs` 写进张量。
    ///
    /// **调用方必须先确认 `n_kv_padded_after(n_tokens)` 与建图时一致** —— 那才是图结构，
    /// 这里不检查（缓存签名负责）。
    pub fn reuse_step(
        &mut self,
        start_pos: usize,
        n_tokens: usize,
        out_rows: Option<Vec<i32>>,
    ) -> Result<(), String> {
        self.kv.begin_step(n_tokens)?;
        self.step_start = start_pos;
        self.step_n = n_tokens;
        self.out_rows = out_rows;
        self.n_past = start_pos + n_tokens;
        Ok(())
    }

    /// 本步处理后的总序列长度。
    pub fn n_kv(&self) -> usize {
        self.step_start + self.step_n
    }

    /// 所有递归层 GDN 递推状态的字节数（`state` + `conv_state` 之和，**只看平面 0**）。
    ///
    /// 这是 `--cache-prompt` 每个检查点的快照大小（本模型约 63 MiB），
    /// 缓存预算就是按它算的。回退快照（平面 1..n_rs）只对"当前这一步的校验批"
    /// 有意义，跨请求复用时不需要保存。
    pub fn gdn_bytes(&self) -> usize {
        self.gdn
            .iter()
            .flatten()
            .map(|st| st.state_plane_bytes() + st.conv_plane_bytes())
            .sum()
    }

    /// 把 GDN 递推状态（平面 0）读回主机（检查点快照）。
    ///
    /// 布局：`[u64 总字节数][逐层: state 原始字节 ++ conv_state 原始字节]`（小端）。
    /// 带上长度头是为了 `restore_gdn` 能给出清楚的错误，而不是悄悄错位。
    ///
    /// 为什么必须有它：GDN 是**递推**状态，不能像注意力的 KV 那样"丢掉几行"回到
    /// 中间位置；要复用前缀就只能整份保存/恢复（见 `engine/prompt_cache.rs`）。
    pub fn snapshot_gdn(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.gdn_bytes() + 8);
        out.extend_from_slice(&(self.gdn_bytes() as u64).to_le_bytes());
        for st in self.gdn.iter().flatten() {
            out.extend_from_slice(&st.state.read_bytes_at(0, st.state_plane_bytes()));
            out.extend_from_slice(&st.conv_state.read_bytes_at(0, st.conv_plane_bytes()));
        }
        out
    }

    /// 从 [`EngineState::snapshot_gdn`] 的缓冲恢复 GDN 递推状态（写回平面 0）。
    pub fn restore_gdn(&self, data: &[u8]) -> Result<(), String> {
        let want = self.gdn_bytes();
        if data.len() < 8 {
            return Err(format!("GDN 快照太短: {} 字节", data.len()));
        }
        let mut len = [0u8; 8];
        len.copy_from_slice(&data[..8]);
        let declared = u64::from_le_bytes(len) as usize;
        if declared != want || data.len() != want + 8 {
            return Err(format!(
                "GDN 快照长度不符: 声明 {declared} + 头，实际 {}，期望 {want} + 头",
                data.len()
            ));
        }
        let mut off = 8;
        for st in self.gdn.iter().flatten() {
            let (n_state, n_conv) = (st.state_plane_bytes(), st.conv_plane_bytes());
            st.state.write_bytes_at(0, &data[off..off + n_state]);
            off += n_state;
            st.conv_state.write_bytes_at(0, &data[off..off + n_conv]);
            off += n_conv;
        }
        Ok(())
    }

    /// 把「本步的 KV 写入位置」回退到 `pos`（`--cache-prompt` 复用前缀 / 投机接受用）。
    ///
    /// 只改游标，**不动 cache 里的数据**：`[0, pos)` 那些行的 token 与新的序列
    /// 前缀相同，数据本来就是对的；`[pos, ...)` 会在本次前向里被覆盖。
    ///
    /// **只回退主干的 KV**：MTP 的独立 KV 由 `runner::mtp_forward` 全权管理
    /// （它每次都先 `rewind_to(start)` 再 `begin_step(n)`），因为投机解码里 MTP 的
    /// 写入节奏与主干不同步（起草期会先写到更远的位置）。
    pub fn rewind_to(&mut self, pos: usize) -> Result<(), String> {
        self.kv.rewind_to(pos)?;
        self.step_start = pos;
        self.step_n = 0;
        self.n_past = pos;
        Ok(())
    }

    /// 本步的 query 起始位置。
    pub fn step_start(&self) -> usize {
        self.step_start
    }

    /// RoPE 参数（MTP 图也要用同一套，见 `runner::mtp_forward`）。
    pub fn rope(&self) -> &RopeParams {
        &self.rope
    }

    /// 是否走 FlashAttention（决定 MTP 图的 mask 类型）。
    pub fn flash_attn(&self) -> bool {
        self.flash_attn
    }

    /// 把图输入（token ids / IMROPE positions / 因果 mask）写入本步张量。
    ///
    /// 必须在 `galloc.alloc()` **之后**调用：此时图内张量的 backend buffer 已分配，
    /// 写操作才真正生效到可执行内存。
    ///
    /// - tokens: i32 [n_tokens]
    /// - pos:    i32 [4 * n_tokens]，布局 [t0..tn, h0..hn, w0..wn, e0..en]
    ///           （文本场景四段取同一个位置值）
    /// - mask:   f32 [n_kv_padded, n_tokens]，mask[key, query] = 0 允许 / -inf 屏蔽
    /// - kv_idxs: i64 [n_tokens]，本步各 token 的绝对 KV 位置（set_rows 的行号）
    /// - out_ids: i32 [n_out]，需要输出 logits 的行号（`None` 表示不裁剪）
    pub fn fill_inputs(&mut self, ids: &[i32]) -> Result<(), String> {
        let st = self.step.as_mut().ok_or("fill_inputs: 尚未调用 forward 建图")?;
        let n_kv = self.kv.n_kv_padded();
        write_step_inputs(
            &mut self.mask_buf,
            &st.tokens,
            &st.pos,
            &st.mask,
            &st.kv_idxs,
            ids,
            self.step_start,
            n_kv,
            self.flash_attn,
            self.n_threads,
        );

        // 输出行号（只影响最后一层的取行；内容每步写一次即可）
        // `SQWEN_ABLATE=lm_head` 时没人读它 → 未分配 → 跳过。
        if let (Some(dst), Some(rows)) = (st.out_ids, self.out_rows.as_ref()) {
            if dst.is_allocated() {
                let bytes: Vec<u8> = rows.iter().flat_map(|&x| x.to_ne_bytes()).collect();
                dst.write_bytes_at(0, &bytes);
            }
        }

        // GDN 递推状态的平面选择（投机解码用；`n_rs == 0` 时没有这个张量）
        // `SQWEN_ABLATE=gdn` 时 GDN 层被整层跳过 → 没人读它 → 未分配 → 跳过。
        if let Some(sel) = st.sel {
            if sel.is_allocated() {
                sel.write_bytes_at(0, &(self.sel_value as i32).to_ne_bytes());
            }
        }

        Ok(())
    }
}

/// 写入一步图输入里与"位置"有关的那几项：tokens / pos / mask / kv_idxs。
///
/// 抽出来是为了让 MTP 图（在 `runner.rs` 里每步新建、有自己的 KV cache 与
/// scratch ctx）能复用同一套填值逻辑，避免两份 mask 构造代码走偏。
#[allow(clippy::too_many_arguments)]
pub fn write_step_inputs(
    mask_buf: &mut Vec<u8>,
    tokens: &StateTensor,
    pos_t: &StateTensor,
    mask_t: &StateTensor,
    kv_idxs: &StateTensor,
    ids: &[i32],
    step_start: usize,
    n_kv: usize,
    flash_attn: bool,
    n_threads: usize,
) {
    let n = ids.len();

    // tokens：i32 → bytes（小端）
    // 极端消融（attn+gdn+moe+lm_head 全跳）时连 embedding 都不可达 → 未分配 → 跳过。
    if tokens.is_allocated() {
        let token_bytes: Vec<u8> = ids.iter().flat_map(|&x| x.to_ne_bytes()).collect();
        tokens.write_bytes_at(0, &token_bytes);
    }

    // positions：四段各占 n 个 i32（文本场景四段取同一个位置值）。
    //
    // **必须是绝对位置**：RoPE 的相位在写入 K cache 时就烙进了 K 的值里，query 只能用
    // 自己的绝对位置才能还原出正确的相对距离。旧实现写的是批内相对下标（`0..n`）：
    // 单块 prefill 恰好等于绝对位置因而看不出来，但 decode 步（n=1）永远按位置 0
    // 旋转 —— 生成时 query 与全部历史 K 的相位关系都是错的。修正后投机解码的草稿
    // 命中率明显上升，greedy 输出也与 llama.cpp 对齐。
    let pos_data: Vec<i32> = (0..n)
        .map(|i| (step_start + i) as i32)
        .flat_map(|i| [i, i, i, i])
        .collect();
    let pos_bytes: Vec<u8> = pos_data.iter().flat_map(|&x| x.to_ne_bytes()).collect();
    // pos 只被注意力层的 RoPE 读；`SQWEN_ABLATE=attn` 时没人引用 → 未分配 → 跳过。
    if pos_t.is_allocated() {
        pos_t.write_bytes_at(0, &pos_bytes);
    }

    // KV 写入行号：本步各 token 的绝对位置（set_rows 的行号）
    //
    // `SQWEN_ABLATE=attn` 把注意力整层跳过时，mask / kv_idxs 没有任何节点引用 →
    // gallocr 不会给它们分配内存 → 这里必须跳过（否则写指针会 panic）。
    let have_kv_inputs = kv_idxs.is_allocated() && mask_t.is_allocated();
    if have_kv_inputs {
        let idx_bytes: Vec<u8> = (0..n)
            .flat_map(|i| ((step_start + i) as i64).to_ne_bytes())
            .collect();
        kv_idxs.write_bytes_at(0, &idx_bytes);
    }

    // causal mask：[n_kv_padded, n_tokens]，mask[key, query] = 0 if key <= query else -inf
    //
    // 行长度用**对齐后**的 n_kv（与 mask 张量、K/V 视图一致）：被读到的
    // [n_used, n_kv_padded) 这几行是零初始化的 K/V + -inf mask，贡献严格为 0。
    //
    // 整块一次性构造成字节缓冲再一次 `write_bytes_at`（一次 H2D）：
    // 相邻 query 的行只差**一个**元素（第 limit+1 个从 -inf 变 0），所以逐行 memcpy
    // 再改一个元素即可。旧实现每行都 clone 一个 Vec 再 flat_map 成字节，
    // 512-token chunk 实测要 16.6~47 ms —— 比一次 memcpy 慢两个数量级。
    //
    // 只有两种取值：0.0（允许）与 -inf（屏蔽）。F16 的编码是固定的：
    //   0.0  → 0x0000，-inf → 0xFC00（小端字节序）
    let (inf_val, esize): ([u8; 4], usize) = if flash_attn {
        ([0x00, 0xFC, 0x00, 0x00], 2)
    } else {
        (f32::NEG_INFINITY.to_ne_bytes(), 4)
    };
    let row_bytes = n_kv * esize;
    // 复用常驻缓冲（只在需要更大时增长一次），避免每步 page fault
    let needed = row_bytes * n;
    if mask_buf.len() < needed {
        mask_buf.resize(needed, 0);
    }
    if !have_kv_inputs {
        return;
    }
    let mask_bytes = &mut mask_buf[..needed];
    fill_mask_rows(
        mask_bytes,
        n,
        n_kv,
        esize,
        step_start,
        &inf_val,
        n_threads,
    );
    mask_t.write_bytes_at(0, mask_bytes);
}

/// 构造因果 mask 的字节缓冲（可并行）。
///
/// 布局：`mask_bytes[q * row_bytes + k * esize]` = 第 q 个 query 对第 k 个 key 的允许位。
/// 值只有两个：`0.0`（允许，全 0 字节）与 `-inf`（屏蔽，`inf_val`）。
/// 第 q 行 = 前 `step_start + q + 1` 个 key 允许，其余 `-inf`。
///
/// 为什么值得并行：ctx=172032、ubatch=512 时每块要填 172032×512×2B ≈ **176 MB**，
/// 单线程实测 85~90 ms/块（占该块 ~16%）。这里按行分块给 N 个线程，各线程
/// memcpy 一整行模板后再把前缀清零 —— 这是纯访存负载，线程数到 8 左右就饱和。
fn fill_mask_rows(
    mask_bytes: &mut [u8],
    n_rows: usize,
    n_kv: usize,
    esize: usize,
    step_start: usize,
    inf_val: &[u8; 4],
    n_threads: usize,
) {
    if n_rows == 0 || n_kv == 0 {
        return;
    }
    let row_bytes = n_kv * esize;
    // 一整行的 -inf 模板（只有这一种字节模式）
    let mut template = vec![0u8; row_bytes];
    for chunk in template.chunks_exact_mut(esize) {
        chunk.copy_from_slice(&inf_val[..esize]);
    }

    // 每个线程处理连续的一段行：写模板 + 把 [0, step_start+q] 前缀清零
    let build = |rows: &mut [u8], first_row: usize| {
        for (i, row) in rows.chunks_exact_mut(row_bytes).enumerate() {
            row.copy_from_slice(&template);
            let q = first_row + i;
            let allowed = (step_start + q + 1).min(n_kv); // 允许的 key 数
            row[..allowed * esize].fill(0);
        }
    };

    let threads = n_threads.clamp(1, n_rows);
    if threads <= 1 {
        build(mask_bytes, 0);
        return;
    }
    // 每线程一段（最后一段收到余数）
    let per = n_rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (t, chunk) in mask_bytes.chunks_mut(per * row_bytes).enumerate() {
            let first = t * per;
            scope.spawn(move || build(chunk, first));
        }
    });
}

/// 构建一次前向（prefill 或单步 decode），返回 logits。
///
/// `out_ids`：需要 logits 的行号（相对本步的 query 位置）。`Some(&[n-1])` 表示只要
/// 最后一个位置 —— 生成时唯一需要的那个。`None` = 所有行（测试/打分场景）。
/// 只有需要的行会走输出投影与 D2H 回读（见 `StepTensors::out_ids` 的说明）。
///
/// 调用方拿到返回值后：填图输入 → 分配本步 buffer → `compute`。
pub fn forward<'a>(
    ctx: &'a GgmlCtx,
    g: &mut GraphBuilder<'a>,
    m: &ModelWeights,
    arch: &Qwen35MoeArch,
    st: &mut EngineState,
    token_ids: &[i32],
    start_pos: usize,
    out_ids: Option<&[i32]>,
) -> Result<Tensor<'a>, String> {
    let c = ctx.as_ptr();
    let n = token_ids.len() as i64;
    if n == 0 {
        return Err("forward: token_ids 为空".to_string());
    }
    let n_layer = arch.n_layer_trunk() as usize;

    // ---- 图输入 --------------------------------------------------------
    // 本步的 KV 位置区间一次定好（所有全注意力层共用），再据此建 mask
    st.kv
        .begin_step(token_ids.len())
        .map_err(|e| format!("forward: {e}"))?;
    // mask / K / V 全部按**对齐后**的 n_kv 建（见 KvCache::n_kv_padded）：
    // 每 256 步内图拓扑完全一致，ggml-cuda 的 CUDA graph 才能真正 capture/replay。
    let n_kv = st.kv.n_kv_padded();
    let tokens = ctx.new_tensor_1d_i32(n);
    let pos = ctx.new_tensor_1d_i32(4 * n);
    // mask：FA 路径必须是 **F16**（llama.cpp 也是直接把 mask 建成 F16 再填
// `ggml_fp16_t`，而不是用 ggml_cast 转 —— F32→F16 的 cpy 在 FA 前会出问题）。
    let mask_type = if st.flash_attn {
        ffi::GgmlType::F16
    } else {
        ffi::GgmlType::F32
    };
    let mask = ctx.new_tensor_2d(mask_type, n_kv as i64, n);
    // KV 写入行号：I64 [n]，内容由 fill_inputs 填成 [step_start, step_start+n)
    let kv_idxs = ctx.new_tensor(ffi::GgmlType::I64, &[n]);
    // 需要输出 logits 的行号：I32 [n_out]（生成时只有最后一行）
    let out_ids_t = out_ids.map(|ids| ctx.new_tensor_i32(&[ids.len() as i64]));
    // GDN 递推状态的平面选择（n_rs > 0 才有）
    let sel_t = (st.n_rs > 0).then(|| ctx.new_tensor_i32(&[1]));

    // ---- embedding：get_rows(tok_embd, tokens) → [n_embd, n] -----------
    let tok_embd = m.token_embd.get(ctx);
    let mut h = ctx.wrap(unsafe { ffi::ggml_get_rows(c, tok_embd.ptr(), tokens.ptr()) });
    let n_embd = arch.n_embd as i64;
    let eps = arch.norm_rms_eps;

    // 本步 KV cache 的写回节点**必须在本层注意力节点进入图之前** expand。
    //
    // ggml 按节点进入图的顺序执行；而 `ggml_build_forward_expand` 是「后序遍历 +
    // 已访问集合」——只有 cpy 自己的依赖（K/V 投影子树）会排在它前面。若把 cpy 攒到
    // 循环结束再 expand，后面那些 GDN 层的状态写回会先把 h 链上的**注意力读节点**
    // （scores/softmax）拉进图，等再 expand cpy 时它只能被追加到节点表末尾：
    // 本步的 K/V 写排在读之后，注意力读到的是上一步（首次 prefill 则是空 cache，
    // V=0 让 attn 输出恒等于 0）。所以这里**逐层立即 expand**，与 `gdn_step` 一致。
    for il in 0..n_layer {
        let w = &m.layers[il];
        let ab = Ablate::cached();

        // attn_norm → 注意力（全注意力 / GDN）
        let normed = rms_norm(ctx, &h, &w.attn_norm.get(ctx), n_embd, n, eps);
        let attn_out = if let Some(slot) = st.kv_slot[il] {
            if ab.attn {
                // 消融：跳过整层全注意力（投影 + FA + o_proj），残差恒等
                zero_like(ctx, c, &normed)
            } else {
                let (out, writes) = build_attn_layer(
                    ctx,
                    w,
                    &normed,
                    &mut st.kv,
                    slot,
                    &pos,
                    &mask,
                    &kv_idxs,
                    n,
                    arch,
                    &st.rope,
                    st.flash_attn,
                );
                for wr in &writes {
                    g.expand(wr);
                }
                out
            }
        } else if ab.gdn {
            // 消融：跳过 GDN 层（qkv/conv/递推/门控全跳），残差恒等。
            // 注意状态不再更新（数值本就不可用）。
            zero_like(ctx, c, &normed)
        } else {
            let gst = st.gdn[il].as_mut().expect("递归层应有 GDN 状态");
            gdn_step(ctx, g, w, &normed, gst, n, arch, sel_t.as_ref())?
        };

        // 残差
        h = ctx.wrap(unsafe { ffi::ggml_add(c, attn_out.ptr(), h.ptr()) });

        // post_attention_norm → MoE FFN
        let post = rms_norm(ctx, &h, &w.attn_post_norm.get(ctx), n_embd, n, eps);
        let ffn = if ab.moe {
            // 消融：跳过 MoE（路由 + 专家 + 共享专家），残差恒等
            zero_like(ctx, c, &post)
        } else {
            build_moe_ffn(
                ctx,
                g,
                w,
                &post,
                arch.n_expert as i64,
                // P8 诊断：`SQWEN_EXPERT_USED=n` 临时覆盖 top-k 专家数（结果不对，只用于量测
                // MoE 在 compute 里的占比：若 compute 近似线性下降 → 瓶颈是专家权重的
                // 显存带宽；若几乎不变 → 瓶颈是每节点固定开销）。
                std::env::var("SQWEN_EXPERT_USED")
                    .ok()
                    .and_then(|v| v.parse::<i64>().ok())
                    .filter(|n| *n > 0 && (*n as u32) <= arch.n_expert)
                    .unwrap_or(arch.n_expert_used as i64),
                arch.expert_ffn_len as i64,
                n_embd,
                n,
            )
        };
        h = ctx.wrap(unsafe { ffi::ggml_add(c, ffn.ptr(), h.ptr()) });
    }

    // ---- 输出 ----------------------------------------------------------
    // 只需要部分行的 logits 时，先 `get_rows` 把隐藏状态裁到那几行再算输出投影
    // （llama.cpp qwen3.cpp: `cur = ggml_get_rows(ctx0, cur, inp_out_ids)`）。
    // 这一步同时省掉 GPU 上的 lm_head matmul 与主机侧的 508 MB 回读。
    let out_norm = rms_norm(ctx, &h, &m.output_norm.get(ctx), n_embd, n, eps);

    // ---- MTP：把全行 h 导出到常驻缓冲（行 1..n+1）-------------------------
    //
    // `h_buf` 行 0 是"上一个已提交位置的主干 h"，由 MTP 的 Catchup 图在本批之后
    // 更新（它要从本批里挑一行）。这里只按原样写行 1..n+1，因此与 `h_buf[0]`
    // 不冲突。这个 cpy 在依赖链上与 logits 无关（纯写节点），必须显式 expand。
    if let Some(mtp) = st.mtp.as_ref() {
        let dst = mtp.h_dst_view(ctx, n);
        let src = ctx.wrap(unsafe {
            ffi::ggml_view_2d(c, out_norm.ptr(), n_embd, n, (n_embd as usize) * 4, 0)
        });
        let cp = ctx.wrap(unsafe { ffi::ggml_cpy(c, src.ptr(), dst.ptr()) });
        g.expand(&cp);
    }

    let out_norm = match &out_ids_t {
        Some(ids) => ctx.wrap(unsafe { ffi::ggml_get_rows(c, out_norm.ptr(), ids.ptr()) }),
        None => out_norm,
    };
    let logits = if Ablate::cached().lm_head {
        // 消融：跳过 output_norm / get_rows / lm_head matmul，直接给零 logits。
        // 形状必须与真实 logits 一致（[n_vocab, rows]），否则回读路径会对不上。
        let n_vocab = m.output.get(ctx).ne()[1];
        let rows = out_ids.map_or(n, |ids| ids.len() as i64);
        let t = ctx.new_tensor_2d(ffi::GgmlType::F32, n_vocab, rows);
        ctx.wrap(unsafe { ffi::ggml_fill(c, t.ptr(), 0.0) })
    } else {
        let output = m.output.get(ctx);
        ctx.wrap(unsafe { ffi::ggml_mul_mat(c, output.ptr(), out_norm.ptr()) })
    };

    // ---- 输出展开进图 ---------------------------------------------------
    // 写回节点（KV cache / GDN 状态）已在上面逐层 expand 过：它们必须早于读到它们
    // 的注意力节点进入图，否则顺序会被后序遍历打乱（详见循环里的注释）。
    g.expand(&logits);

    st.step = Some(StepTensors {
        tokens: StateTensor::new(tokens),
        pos: StateTensor::new(pos),
        mask: StateTensor::new(mask),
        kv_idxs: StateTensor::new(kv_idxs),
        out_ids: out_ids_t.map(StateTensor::new),
        sel: sel_t.map(StateTensor::new),
    });
    st.out_rows = out_ids.map(|ids| ids.to_vec());
    st.step_n = token_ids.len();
    st.step_start = start_pos;
    st.n_past = start_pos + token_ids.len();

    Ok(logits)
}

/// `SQWEN_ABLATE`：把某个模块替换成恒等/零，用来量它在 decode compute 里占多少。
///
/// **只用于量测，输出是错的**（数值完全不可用）。逗号分隔，可组合：
/// `attn`（10 个全注意力层）、`gdn`（30 个递推层）、`moe`（40 层 MoE FFN）、
/// `lm_head`（输出投影）、`norm`（所有 rms_norm）。
///
/// 用法见 `plans/2026-10-08-perf-and-parity.md` 里「剩下那部分 decode 差距」一节 ——
/// 之前只能靠猜，这个开关是唯一能给出「每个模块各占多少」的实验手段。
#[derive(Clone, Copy, Debug, Default)]
pub struct Ablate {
    pub attn: bool,
    pub gdn: bool,
    pub moe: bool,
    pub lm_head: bool,
    pub norm: bool,
    /// GDN 内部四段（只在该层**没有**被 `gdn` 整层跳过时才有意义）：
    /// 输入投影（qkv / gate / beta / alpha）、短卷积（conv_state + ssm_conv）、
    /// 递推核（`gated_delta_net` + 状态写回）、收尾（门控 norm + out 投影）。
    pub gdn_proj: bool,
    pub gdn_conv: bool,
    pub gdn_core: bool,
    pub gdn_out: bool,
}

impl Ablate {
    fn from_env() -> Self {
        let mut a = Self::default();
        let raw = std::env::var("SQWEN_ABLATE").unwrap_or_default();
        for part in raw.split(',') {
            match part.trim() {
                "attn" => a.attn = true,
                "gdn" => a.gdn = true,
                "moe" => a.moe = true,
                "lm_head" => a.lm_head = true,
                "norm" => a.norm = true,
                "gdn_proj" => a.gdn_proj = true,
                "gdn_conv" => a.gdn_conv = true,
                "gdn_core" => a.gdn_core = true,
                "gdn_out" => a.gdn_out = true,
                "" => {}
                other => eprintln!(
                    "[ablate] 未知模块名 `{other}`（可用：attn,gdn,moe,lm_head,norm,gdn_proj,gdn_conv,gdn_core,gdn_out）"
                ),
            }
        }
        if a.attn || a.gdn || a.moe || a.lm_head || a.norm
            || a.gdn_proj || a.gdn_conv || a.gdn_core || a.gdn_out
        {
            eprintln!("[ablate] 已启用 {a:?} —— 数值不可用，只测耗时");
        }
        a
    }

    /// 本进程的消融设置（读一次环境变量）。
    ///
    /// 每步都要问，所以用 `OnceLock` 缓存；测量本来就是「一个开关一个进程」。
    pub(crate) fn cached() -> Self {
        static ONCE: std::sync::OnceLock<Ablate> = std::sync::OnceLock::new();
        *ONCE.get_or_init(Ablate::from_env)
    }
}

/// 造一个与 `x` 同形状的**零**张量（消融用：让残差恒等于 `h`）。
fn zero_like<'a>(ctx: &'a GgmlCtx, c: *mut ffi::GgmlContext, x: &Tensor<'a>) -> Tensor<'a> {
    ctx.wrap(unsafe { ffi::ggml_scale(c, x.ptr(), 0.0) })
}

/// RMS 归一化 + 逐元素乘权重（`attn_norm` / `attn_post_norm` / `output_norm`）。
fn rms_norm<'a>(
    ctx: &'a GgmlCtx,
    x: &Tensor<'a>,
    weight: &Tensor<'a>,
    n_embd: i64,
    n_tokens: i64,
    eps: f32,
) -> Tensor<'a> {
    assert_eq!(x.ne()[0], n_embd);
    assert_eq!(x.ne()[1], n_tokens);
    assert_eq!(weight.ne()[0], n_embd);
    if Ablate::cached().norm {
        return ctx.wrap(x.ptr());
    }
    let c = ctx.as_ptr();
    let normed = ctx.wrap(unsafe { ffi::ggml_rms_norm(c, x.ptr(), eps) });
    ctx.wrap(unsafe { ffi::ggml_mul(c, normed.ptr(), weight.ptr()) })
}

/// 跑一个 GDN 层，并把新的递推状态 / 卷积历史写回常驻 buffer。
///
/// `build_gdn_layer` 会把 `st.state` / `st.conv_state` 重绑定为本步的图内张量；
/// 这里额外插一对 `ggml_cpy` 写回常驻张量并 expand，然后把 `st` 恢复成常驻张量，
/// 这样状态才能跨 step 存活（常驻张量在常驻 buffer 里，不会被逐步释放）。
///
/// **回退快照**（`n_rs > 0`）：常驻张量有 `1 + n_rs` 个平面，本步要写回 `n_snap` 个
/// （槽 0 = 最终状态，槽 s = 往回 s 个 token）。递推状态一次 `cpy` 就够（算子已把
/// 快照排成连续的一块）；卷积历史要按行偏移逐槽切（每槽的"尾部"起点不同）。
/// `n_rs == 0` 时两条路径都退化成原来的一次 `cpy`。
fn gdn_step<'a>(
    ctx: &'a GgmlCtx,
    g: &mut GraphBuilder<'a>,
    w: &super::weights::LayerWeights,
    h: &Tensor<'a>,
    st: &mut GdnState,
    n_tokens: i64,
    arch: &Qwen35MoeArch,
    sel: Option<&Tensor<'a>>,
) -> Result<Tensor<'a>, String> {
    let c = ctx.as_ptr();
    let resident_state = ctx.wrap(st.state.ptr());
    let resident_conv = ctx.wrap(st.conv_state.ptr());

    let out = build_gdn_layer(ctx, w, h, st, n_tokens, arch, sel);

    let n_rs = st.n_rs;
    let n_snap = st.n_snap as i64;

    // ---- 递推状态：一次 cpy 写 n_snap 个平面 ------------------------------
    // st.state 已是 [S_v, S_v, H_v, n_snap]；常驻张量是 [S_v, S_v, H_v, 1+n_rs]，
    // 取前 n_snap 个平面作为目标。
    //
    // 目标**必须建成 3D `[D, 1, n_snap]`（D = S_v*S_v*H_v）**，不能建成
    // `[S_v, S_v, H_v, n_snap]` —— 新版 ggml-cuda 的 `ggml_cuda_try_gdn_cache_fusion`
    // （`ggml-cuda.cu:2897`）要求 dst 恰好是这个形状且 `nb[1] == row_size(F32, D)`，
    // 命中后 GDN kernel 直接把快照写进 cache，省掉这次 CPY（连同它的读+写）。
    // 两者的元素数与内存布局完全一致（`ggml_cpy` 只校验 nelements），数值不变。
    {
        let ne = resident_state.ne();
        let (s_v, h_v) = (ne[0], ne[2]);
        let d = s_v * s_v * h_v; // 一个快照平面的元素数 = 融合要求的 D
        let nb1 = (d as u64) * 4; // [D] 行跨度，同时也就是平面跨度
        let dst = ctx.wrap(unsafe {
            ffi::ggml_view_3d(
                c,
                resident_state.ptr(),
                d,
                1, // n_seqs（单序列）
                n_snap,
                nb1,
                nb1,
                0,
            )
        });
        let wb = ctx.wrap(unsafe { ffi::ggml_cpy(c, st.state.ptr(), dst.ptr()) });
        g.expand(&wb);
    }

    // ---- 卷积历史：逐槽写 ------------------------------------------------
    // `conv_input` 是 [K-1+n_tokens, C, 1]；槽 s 的尾部是它的第 (n_tokens - s) 行起、
    // 长 K-1 的那一段（见 llama.cpp delta-net-base.cpp L504-521，对方按同样偏移逐槽 cpy）。
    if n_rs == 0 {
        let wb = ctx.wrap(unsafe { ffi::ggml_cpy(c, st.conv_state.ptr(), resident_conv.ptr()) });
        g.expand(&wb);
    } else {
        let ne = resident_conv.ne();
        let (ck, cc) = (ne[0], ne[1]);
        // 常驻卷积张量是连续的 [ck, cc, 1+n_rs]：
        //   nb1 = ck*4（相邻通道），nb2 = ck*cc*4（相邻"平面"）
        let nb1 = resident_conv.nb()[1];
        let nb2 = resident_conv.nb()[2];
        let conv_input = ctx.wrap(st.conv_state.ptr()); // [K-1+n_t, C, 1]
        let in_nb1 = conv_input.nb()[1];
        let in_nb2 = conv_input.nb()[2];
        let k_snap = n_rs as i64 + 1;
        for t in 1..=k_snap {
            // 槽 s = k_snap - t 对应"往回 s 个 token"，起点行 = n_tokens - s
            let s_idx = (n_tokens - k_snap + t).max(0);
            let src = ctx.wrap(unsafe {
                ffi::ggml_view_3d(
                    c,
                    conv_input.ptr(),
                    ck,
                    cc,
                    1,
                    in_nb1,
                    in_nb2,
                    (s_idx as u64) * 4,
                )
            });
            let dst = ctx.wrap(unsafe {
                ffi::ggml_view_3d(
                    c,
                    resident_conv.ptr(),
                    ck,
                    cc,
                    1,
                    nb1,
                    nb2,
                    ((k_snap - t) as u64) * nb2,
                )
            });
            let wb = ctx.wrap(unsafe { ffi::ggml_cpy(c, src.ptr(), dst.ptr()) });
            g.expand(&wb);
        }
    }

    // 恢复指向常驻张量，下一步继续从它读
    st.state = StateTensor::new(resident_state);
    st.conv_state = StateTensor::new(resident_conv);

    Ok(out)
}
