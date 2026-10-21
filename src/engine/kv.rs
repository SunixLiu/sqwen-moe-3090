// src/engine/kv.rs — Q4_0 KV cache（单序列、追加式）
//
// 蓝本：llama.cpp `src/llama-kv-cache.cpp` 的 `get_k/get_v/cpy_k/cpy_v` +
//       `llama-graph.cpp` 的 `build_attn_mha`（非 flash 路径）。
//
// 布局（kv_cap = 逻辑容量，kv_cap_alloc = 对齐后的分配容量，n_kv = 已用长度）：
//   k[slot] : q4_0  [n_embd_k_gqa, kv_cap_alloc]        n_embd_k_gqa = head_dim * n_head_kv
//   v[slot] : f32   [head_dim, n_head_kv, kv_cap_alloc]
//
// 因果 mask 不放在这里：它是每步的图输入，形状 [n_kv_padded, n_tokens]，由调用方创建
// 并填值（见 `graph.rs`）。若按 kv_cap² 预分配，16k 上下文就是 1 GB。
//
// K 用 Q4_0：mul_mat 的 A 侧允许量化类型，且行长度 = n_embd_k_gqa（head_dim 的倍数）
// 一定是 32 的整数倍，所以量化后的 cache 可以直接当作矩阵乘的操作数 —— 这正是
// llama.cpp 的做法（`ggml_mul_mat(ctx0, k, q)`，k 就是 cache）。
//
// V 暂时存 f32，原因值得记下来：要让 V 也能被 mul_mat 直接消费，量化必须沿 KV 轴
// 分块（llama.cpp 的 `v_trans = !flash_attn` 路径，`get_v` 把 cache 看成
// [n_kv, n_head_kv, head_dim]，行长为 kv_size）。而「追加 n 个新 token」只写每行的
// 前 n 个位置，属于对一个量化行的**部分写入**，`ggml_cpy` 表达不了；llama.cpp 因此
// 改用一个元素一个 index 的 `set_rows`（`v_idxs` 长度是 n_tokens*n_embd_v_gqa，
// 行长为 1）。我们的 cache 是单序列追加式的，没必要引入那套 scatter，先让 V 走
// f32 自然布局（连续 cpy 写入即可）。
// **V 改成 q4_0 与 FlashAttention 一起落地**（FA 的 q4_0-q4_0 vec kernel 已编译，
// 见 plans/2026-10-08-perf-and-parity.md 的 Phase 3）。
//
// n_kv 对齐到 KV_PAD：见 `n_kv_padded()` 的注释（CUDA graph 重放 + FA 要求）。

use crate::backend::ctx::{GgmlCtx, StateTensor, Tensor};
use crate::backend::ffi::{self, GgmlType};

/// q4_0 的 block 大小：量化行的元素数必须是它的整数倍。
pub const QK4_0: i64 = 32;

/// n_kv 对齐粒度：llama.cpp 用 `pad(n_kv, 256)`（`llama_kv_cache::get_n_kv` 与
/// FA 的 vec kernel 步进一致）。256 同时满足 `% 128 == 0`。
pub const KV_PAD: usize = 256;

/// KV cache 的 K 量化类型。
pub const KV_TYPE_K: GgmlType = GgmlType::Q4_0;

/// V cache 的默认类型（P4）。
///
/// - **FlashAttention 开启**（默认，前提是后端支持）：V 用 **q4_0**。
///   本机 ggml-cuda 编译了原生 `q4_0-q4_0` FA vec kernel，所以不需要任何转换；
///   KV 从 22.8 KiB/token 降到 **5.6 KiB/token**（262k 上下文 5.84 → 1.47 GiB）。
/// - **FA 关闭**：V 必须是 f32。原因见文件头注释：非 FA 路径要把 V 当 `mul_mat` 的
///   A 侧（转置布局），而量化后的行只能整块写、无法"按 token 追加部分行"。
pub const KV_TYPE_V_FA: GgmlType = GgmlType::Q4_0;
/// FA 关闭时的 V 类型。
pub const KV_TYPE_V_NO_FA: GgmlType = GgmlType::F32;

const F32_SIZE: usize = 4;

/// 单序列追加式 KV cache。
///
/// **不带 ctx 生命周期**：cache 张量跨步常驻（用 [`StateTensor`]），而本步产生的
/// 写回节点由调用方持有 —— `append` 会把它们作为返回值交出去。这样
/// `EngineState` 可以被 `Engine` 连 ctx 一起拥有，同时调用方仍然完全掌控
/// 「写 cache 的节点必须先于读它的注意力节点」这一展开顺序：
///
///   1. `begin_step(n)` 定好本步的写入区间；
///   2. 逐层 `build_attn_layer(...)` 建图，收集返回的写回节点；
///   3. `BackendBuffer::from_ctx` 分配；
///   4. **先** expand 收集到的写回节点，**再** expand 输出 —— 图节点按 expand 顺序
///      排列，因此写一定排在对应的读之前。
pub struct KvCache {
    /// 每个全注意力层一份 K cache: q4_0 [n_embd_k_gqa, kv_cap_alloc]
    k: Vec<StateTensor>,
    /// 每个全注意力层一份 V cache: q4_0（FA）或 f32（非 FA）[head_dim, n_head_kv, kv_cap_alloc]
    v: Vec<StateTensor>,
    /// V cache 的类型（决定 FA / 非 FA 路径）
    v_type: GgmlType,
    head_dim: i64,
    n_head_kv: i64,
    /// 逻辑容量（调用方要的上下文长度）
    kv_cap: usize,
    /// 实际分配容量（向上对齐到 [`KV_PAD`]）
    kv_cap_alloc: usize,
    /// 已写入的 KV 位置数。**每步推进一次**（所有层共用同一段位置），
    /// 不是每层推进一次。
    n_used: usize,
    /// 本步的写入起点（`begin_step` 设定）
    step_start: usize,
}

impl KvCache {
    /// 为 `n_layers` 个全注意力层预分配 cache。
    ///
    /// `head_dim` 必须是 32 的整数倍（q4_0 的行长要求）。
    /// `v_type`：[`KV_TYPE_V_FA`]（q4_0，配合 FlashAttention）或 [`KV_TYPE_V_NO_FA`]（f32）。
    ///
    /// 分配容量会向上对齐到 [`KV_PAD`]（256），这样 `n_kv_padded()` 永远不会越过
    /// 分配边界；逻辑容量仍是调用方传入的 `kv_cap`。
    pub fn new(
        g: &GgmlCtx,
        n_layers: usize,
        head_dim: i64,
        n_head_kv: i64,
        kv_cap: usize,
        v_type: GgmlType,
    ) -> Self {
        assert_eq!(
            head_dim % QK4_0,
            0,
            "head_dim 必须是 {QK4_0} 的整数倍（q4_0 行长）"
        );
        let n_embd_k_gqa = head_dim * n_head_kv;
        assert_eq!(
            n_embd_k_gqa % QK4_0,
            0,
            "n_embd_k_gqa 必须是 {QK4_0} 的整数倍（q4_0 行长）"
        );
        assert!(kv_cap > 0);
        assert!(
            matches!(v_type, GgmlType::F32 | GgmlType::F16 | GgmlType::Q4_0),
            "V cache 只支持 f32（非 FA）/ f16 / q4_0（FA）"
        );
        let kv_cap_alloc = kv_cap.div_ceil(KV_PAD) * KV_PAD;

        let mut k = Vec::with_capacity(n_layers);
        let mut v = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k.push(StateTensor::new(g.new_tensor_2d(
                KV_TYPE_K,
                n_embd_k_gqa,
                kv_cap_alloc as i64,
            )));
            v.push(StateTensor::new(g.new_tensor_3d(
                v_type,
                head_dim,
                n_head_kv,
                kv_cap_alloc as i64,
            )));
        }

        Self {
            k,
            v,
            v_type,
            head_dim,
            n_head_kv,
            kv_cap,
            kv_cap_alloc,
            n_used: 0,
            step_start: 0,
        }
    }

    /// 把整块 K/V cache 清零（**一次性**，在 `Engine::load` 里调用）。
    ///
    /// 为什么必须做：`n_kv_padded()` 会把 `[n_used, n_kv_padded)` 这几行交给注意力读，
    /// mask 虽然是 -inf，但如果那几行是未初始化显存里的 NaN/Inf，`scores + (-inf)` 仍是
    /// NaN，会把整行 softmax 污染成 NaN（V 侧的 `0 * NaN` 同理）。
    /// 清零后：K=0 → scores=0 → +(-inf) = -inf → exp = 0；V=0 → 贡献 0，数值不受影响。
    ///
    /// 之后不再需要重复清零：写入只会落在 `[step_start, n_used)` ⊂ 已初始化的前缀里。
    ///
    /// **调用时机**：必须在 backend buffer 分配之后（否则张量既没有 buffer 也没有 data）。
    pub fn zero_caches(&self) {
        let mut buf: Vec<u8> = Vec::new();
        for slot in 0..self.k.len() {
            for t in [self.k[slot], self.v[slot]] {
                let n = t.nbytes();
                buf.clear();
                buf.resize(n, 0);
                t.write_bytes_at(0, &buf);
            }
        }
    }

    pub fn kv_cap(&self) -> usize {
        self.kv_cap
    }

    /// 每个 token 的 KV 字节数（所有全注意力层合计），用于启动时估算显存。
    ///
    /// K 恒为 q4_0；V 按 `v_type`（FA 时 q4_0，否则 f32）。
    pub fn bytes_per_token(&self) -> usize {
        let n_embd_k_gqa = (self.head_dim * self.n_head_kv) as usize;
        // q4_0 block = 2 字节 scale + 16 字节 qs（32 个元素）
        let k_row = n_embd_k_gqa / QK4_0 as usize * 18;
        let v_row = match self.v_type {
            GgmlType::Q4_0 => n_embd_k_gqa / QK4_0 as usize * 18,
            GgmlType::F16 => n_embd_k_gqa * 2,
            _ => n_embd_k_gqa * F32_SIZE,
        };
        (k_row + v_row) * self.k.len()
    }

    /// V cache 的类型。
    pub fn v_type(&self) -> GgmlType {
        self.v_type
    }

    /// 按当前容量（逻辑容量）估算 KV cache 总字节数。
    pub fn bytes_total(&self) -> usize {
        self.bytes_per_token() * self.kv_cap
    }

    /// 全注意力层数（KV cache 的份数）。
    pub fn n_layers(&self) -> usize {
        self.k.len()
    }

    /// 已写入 cache 的 token 数（跨步累积）。
    pub fn n_used(&self) -> usize {
        self.n_used
    }

    /// 对齐到 256 之后、供图使用的 KV 长度（`ceil(n_used / 256) * 256`）。
    ///
    /// 为什么要对齐（P2，见 plans/2026-10-08-perf-and-parity.md）：
    ///   - ggml-cuda 的 CUDA graph 只在**图拓扑不变**时才能 capture/replay。mask 形状
    ///     之前每步 +1 → 每步拓扑都变 → warmup 状态机永远回到 eager launch，
    ///     gallocr 也每步重新规划（实测 1.3~1.8 ms/token 白花）；
    ///   - 对齐后每 256 步内所有张量形状完全一致，graph 才能真正重放；
    ///   - 附带满足 FA（flash-attn）vec kernel 对 `K->ne[1] % 128 == 0` 的硬要求。
    ///
    /// 多出来的那几行 K/V 是**从未写过**的零（见 `zero_caches`），对应 mask 为 -inf，
    /// 对注意力的贡献严格为 0 —— 结果不变，但 softmax 的归约长度变了，末位舍入会有
    /// 1e-7 量级差别（同任何 batch-shape 变化一样，近并列的 token 可能翻转，
    /// 见 tests/graph_test.rs 的标定注释）。
    pub fn n_kv_padded(&self) -> usize {
        self.n_used.div_ceil(KV_PAD) * KV_PAD
    }

    /// 若现在推进 `n_tokens`，对齐后的 n_kv 会是多少（**不改变任何状态**）。
    ///
    /// P2c 用它来判定"本步的图是否与上一步同构"：形状只在跨 256 边界时才变。
    pub fn n_kv_padded_after(&self, n_tokens: usize) -> usize {
        (self.n_used + n_tokens).div_ceil(KV_PAD) * KV_PAD
    }

    /// 开始新的一步：确定本步的写入区间并一次性推进 `n_used`。
    ///
    /// 必须在所有层建图之前调用一次；所有全注意力层都把本步的 K/V 写到
    /// `[step_start, step_start + n_tokens)` 这同一段位置上（各自的 cache 张量里）。
    pub fn begin_step(&mut self, n_tokens: usize) -> Result<(), String> {
        if self.n_used + n_tokens > self.kv_cap {
            return Err(format!(
                "KV cache 溢出：n_used={} + n_tokens={} > kv_cap={}",
                self.n_used, n_tokens, self.kv_cap
            ));
        }
        debug_assert!(
            (self.n_used + n_tokens).div_ceil(KV_PAD) * KV_PAD <= self.kv_cap_alloc,
            "对齐后的 n_kv 越过分配容量"
        );
        self.step_start = self.n_used;
        self.n_used += n_tokens;
        Ok(())
    }

    /// 某个全注意力层的 K cache 张量（诊断/测试用）。
    ///
    /// 存在的意义之一：验证「后续步不会改写已经写好的位置」。早期实现把每步的
    /// 写回节点**累积**在 cache 里、并在下一步重新 expand，结果第 1 步的整条子图
    /// 被重跑（此时常驻 GDN 状态已经变了），把前面位置的 K/V 覆盖成错的值。
    pub fn k_state(&self, slot: usize) -> StateTensor {
        self.k[slot]
    }

    /// 同上，V cache。
    pub fn v_state(&self, slot: usize) -> StateTensor {
        self.v[slot]
    }

    /// 本步的写入起点。
    pub fn step_start(&self) -> usize {
        self.step_start
    }

    /// 重置为「空 cache」（同一 context 内换序列时使用）。
    pub fn reset(&mut self) {
        self.n_used = 0;
        self.step_start = 0;
    }

    /// 把写入游标回退到 `pos`（`--cache-prompt` 复用前缀时用）。
    ///
    /// **只改游标，不动数据**：`[0, pos)` 那几行的 K/V 保留原样（前缀 token 没变，
    /// 它们仍然是对的），`[pos, ...)` 会在随后的 prefill 里被覆盖。
    pub fn rewind_to(&mut self, pos: usize) -> Result<(), String> {
        if pos > self.n_used {
            return Err(format!(
                "KV 回退越界：pos={pos} > 已写入 {n_used}",
                n_used = self.n_used
            ));
        }
        if pos > self.kv_cap {
            return Err(format!("KV 回退越界：pos={pos} > 容量 {}", self.kv_cap));
        }
        self.n_used = pos;
        self.step_start = pos;
        Ok(())
    }

    /// 图内把本步新 token 的 K/V 写到 `[step_start, step_start + n_tokens)`。
    ///
    /// **写入用 `ggml_set_rows`，目标张量是整块 cache（指针固定）**，写到哪一行由
    /// `idxs` 的**内容**决定：
    ///
    ///   - 旧实现用 `view(cache, offset = step_start * nb1)` 做 cpy，view 的 `data`
    ///     指针每步都不同 → ggml-cuda 的 `cuda_graph_update_required` 每步都判定
    ///     "属性变了" → `warmup_complete = false` → 永远退回 eager 执行。
    ///     实测（临时把偏移写死为 0 验证拓扑假设）：TG 从 62 → **82 t/s**，
    ///     即 CUDA graph 重放值 ~23%。
    ///   - `ggml_set_rows` 的 dst 是固定的 cache 指针，只有 I64 索引张量的内容变，
    ///     节点属性逐字节一致 → graph 可以真正重放。
    ///
    /// 返回该层用于注意力计算的 K/V 视图（覆盖 `[0, n_kv_padded)`）。
    pub fn append<'a>(
        &mut self,
        g: &'a GgmlCtx,
        slot: usize,
        k_cur: &Tensor<'a>, // [head_dim, n_head_kv, n_tokens] f32（已做 RoPE）
        v_cur: &Tensor<'a>, // [head_dim, n_head_kv, n_tokens] f32
        n_tokens: i64,
        idxs: &Tensor<'a>, // I64 [n_tokens]，值为本步各 token 的绝对 KV 位置
    ) -> (Tensor<'a>, Tensor<'a>, [Tensor<'a>; 2]) {
        let ctx = g.as_ptr();
        let n_embd_k_gqa = self.head_dim * self.n_head_kv;

        assert_eq!(k_cur.ne()[0], self.head_dim);
        assert_eq!(k_cur.ne()[1], self.n_head_kv);
        assert_eq!(k_cur.ne()[2], n_tokens);
        assert_eq!(v_cur.ne()[0], self.head_dim);
        assert_eq!(v_cur.ne()[1], self.n_head_kv);
        assert_eq!(v_cur.ne()[2], n_tokens);
        assert_eq!(k_cur.nb()[1], (self.head_dim as u64) * F32_SIZE as u64);
        // 每个 token 的 K/V 必须是连续的一段（set_rows 按行搬运的前提）
        assert_eq!(
            k_cur.nb()[2],
            (n_embd_k_gqa as u64) * F32_SIZE as u64,
            "K 的每 token 步长必须是 n_embd_k_gqa 个 f32"
        );
        assert_eq!(
            v_cur.nb()[2],
            (n_embd_k_gqa as u64) * F32_SIZE as u64,
            "V 的每 token 步长必须是 n_embd_k_gqa 个 f32"
        );
        assert_eq!(idxs.ne()[0], n_tokens);
        assert_eq!(idxs.r#type(), GgmlType::I64 as i32);
        assert_eq!(
            self.step_start + n_tokens as usize,
            self.n_used,
            "append 的 n_tokens 必须与 begin_step 一致"
        );

        let k_cache = self.k[slot].get(g);
        let v_cache = self.v[slot].get(g);

        // K cache [n_embd_k_gqa, kv_cap_alloc]：每个新 token 正好是一整条量化行。
        // src 视图 [n_embd_k_gqa, n_tokens]（行步长 = 每 token 的字节数）。
        let k_src = g.wrap(unsafe {
            ffi::ggml_view_2d(
                ctx,
                k_cur.ptr(),
                n_embd_k_gqa,
                n_tokens,
                k_cur.nb()[2] as usize,
                0,
            )
        });
        // dst 是**整块 cache**（不带偏移）→ 图节点指针稳定
        let k_write = g.wrap(unsafe {
            ffi::ggml_set_rows(ctx, k_cache.ptr(), k_src.ptr(), idxs.ptr())
        });

        // V cache [head_dim, n_head_kv, kv_cap_alloc]：一个 token 的 V 是 dim2 上的
        // 一段（head_dim * n_head_kv 个连续 f32）。把它看成 2D [n_embd_k_gqa, cap]、
        // 行步长 = cache 的 nb2，就能用同一套按行写入。
        let v_src = g.wrap(unsafe {
            ffi::ggml_view_2d(
                ctx,
                v_cur.ptr(),
                n_embd_k_gqa,
                n_tokens,
                v_cur.nb()[2] as usize,
                0,
            )
        });
        let v_dst = g.wrap(unsafe {
            ffi::ggml_view_2d(
                ctx,
                v_cache.ptr(),
                n_embd_k_gqa,
                self.kv_cap_alloc as i64,
                v_cache.nb()[2] as usize,
                0,
            )
        });
        let v_write = g.wrap(unsafe {
            ffi::ggml_set_rows(ctx, v_dst.ptr(), v_src.ptr(), idxs.ptr())
        });

        (self.k_view(g, slot), self.v_view(g, slot), [k_write, v_write])
    }

    /// K cache 的 `[head_dim, n_head_kv, n_kv_padded]` 视图（q4_0，可直接做 mul_mat 的 A 侧）。
    ///
    /// 注意用的是**对齐后**的长度（见 `n_kv_padded`）：graph 形状在 256 步内保持不变，
    /// CUDA graph 才能重放。对齐多出来的行是零、mask 为 -inf，数值上无影响。
    pub fn k_view<'a>(&self, g: &'a GgmlCtx, slot: usize) -> Tensor<'a> {
        let ctx = g.as_ptr();
        let k = self.k[slot].get(g);
        let n_kv = self.n_kv_padded() as i64;
        let row_head = unsafe { ffi::ggml_row_size(k.r#type(), self.head_dim) };
        let row_gqa =
            unsafe { ffi::ggml_row_size(k.r#type(), self.head_dim * self.n_head_kv) };
        g.wrap(unsafe {
            ffi::ggml_view_4d(
                ctx,
                k.ptr(),
                self.head_dim,
                self.n_head_kv,
                n_kv,
                1,
                row_head,
                row_gqa,
                row_gqa * self.kv_cap_alloc,
                0,
            )
        })
    }

    /// V cache 的 `[head_dim, n_head_kv, n_kv_padded]` 视图（f32）。
    pub fn v_view<'a>(&self, g: &'a GgmlCtx, slot: usize) -> Tensor<'a> {
        let ctx = g.as_ptr();
        let v = self.v[slot].get(g);
        let n_kv = self.n_kv_padded() as i64;
        g.wrap(unsafe {
            ffi::ggml_view_4d(
                ctx,
                v.ptr(),
                self.head_dim,
                self.n_head_kv,
                n_kv,
                1,
                v.nb()[1] as usize,
                v.nb()[2] as usize,
                v.nb()[3] as usize,
                0,
            )
        })
    }
}
