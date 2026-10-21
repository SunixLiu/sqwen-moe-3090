// src/backend/ffi.rs — ggml / gguf FFI declarations (hand-written,对照 references/llama.cpp @ ed7ac35e1)
//
// 所有枚举值、结构体字段均从 ggml/include/ggml.h、ggml/include/ggml-backend.h、
// ggml/include/ggml-cuda.h、ggml/include/gguf.h 抄录，禁止猜测。

use std::ffi::{c_char, c_void};

// ---------------------------------------------------------------------------
// Opaque handle types (对应 ggml.h 中的 forward declarations)
// ---------------------------------------------------------------------------
pub type GgmlContext = c_void;
pub type GgmlCgraph = c_void;            // compute graph (plan 中称 ggml_compute_graph)
pub type GgmlBackend = c_void;
pub type GgmlBackendBuffer = c_void;
pub type GgmlBackendBufferType = c_void;
pub type GgmlGallocr = c_void;
pub type GgufContext = c_void;

// ---------------------------------------------------------------------------
// ggml_tensor — 对照 ggml.h L685-717 struct ggml_tensor
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ggml_tensor {
    pub r#type: i32,                            // enum ggml_type
    pub buffer: *mut GgmlBackendBuffer,         // struct ggml_backend_buffer *
    pub ne: [i64; 4],                           // int64_t ne[GGML_MAX_DIMS]
    pub nb: [u64; 4],                           // size_t nb[GGML_MAX_DIMS]
    pub op: i32,                                // enum ggml_op
    pub op_params: [i32; 16],                   // int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)]
    pub flags: i32,                             // int32_t flags
    pub src: [*mut ggml_tensor; 10],            // struct ggml_tensor * src[GGML_MAX_SRC]
    pub view_src: *mut ggml_tensor,             // struct ggml_tensor * view_src
    pub view_offs: u64,                         // size_t view_offs
    pub data: *mut u8,                          // void * data
    pub name: [c_char; 64],                     // char name[GGML_MAX_NAME]
    pub extra: *mut c_void,                     // void * extra
    pub padding: [u8; 8],                       // char padding[8]
}

// ---------------------------------------------------------------------------
// Enumerations — 对照 ggml.h GGML_TYPE_* / GGML_OP_*
// ---------------------------------------------------------------------------
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2K     = 10,
    Q3K     = 11,
    Q4K     = 12,
    Q5K     = 13,
    Q6K     = 14,
    Q8K     = 15,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    BF16    = 30,
    Q1_0    = 41,
    Q2_0    = 42,
}

impl GgmlType {
    /// 从 GGUF 里的 `ggml_type` 数值安全转换。
    /// 用显式 match 而不是 transmute —— 非法判别值不会变成 UB。
    pub fn from_i32(v: i32) -> Option<Self> {
        Some(match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            30 => Self::BF16,
            41 => Self::Q1_0,
            42 => Self::Q2_0,
            _ => return None,
        })
    }
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlOp {
    None               = 0,
    Dup                = 1,
    Add                = 2,
    AddId              = 3,
    Add1               = 4,
    Acc                = 5,
    Sub                = 6,
    Mul                = 7,
    Div                = 8,
    Sqr                = 9,
    Sqrt               = 10,
    Log                = 11,
    Sin                = 12,
    Cos                = 13,
    Sum                = 14,
    SumRows            = 15,
    Cumsum             = 16,
    Mean               = 17,
    Argmax             = 18,
    CountEqual         = 19,
    Repeat             = 20,
    RepeatBack         = 21,
    Concat             = 22,
    SiluBack           = 23,
    Norm               = 24,
    RmsNorm            = 25,
    RmsNormBack        = 26,
    GroupNorm          = 27,
    L2Norm             = 28,
    MulMat             = 29,
    MulMatId           = 30,
    OutProd            = 31,
    Scale              = 32,
    Set                = 33,
    Cpy                = 34,
    Cont               = 35,
    Reshape            = 36,
    View               = 37,
    Permute            = 38,
    Transpose          = 39,
    GetRows            = 40,
    GetRowsBack        = 41,
    SetRows            = 42,
    Diag               = 43,
    DiagMaskInf        = 44,
    DiagMaskZero       = 45,
    SoftMax            = 46,
    SoftMaxBack        = 47,
    Rope               = 48,
    RopeBack           = 49,
    Clamp              = 50,
    ConvTranspose1D    = 51,
    Im2Col             = 52,
    Im2ColBack         = 53,
    Im2Col3D           = 54,
    Col2Im1D           = 55,
    Conv2D             = 56,
    Conv3D             = 57,
    Conv2DDw           = 58,
    ConvTranspose2D    = 59,
    Pool1D             = 60,
    Pool2D             = 61,
    Pool2DBack         = 62,
    Upscale            = 63,
    Pad                = 64,
    PadReflect1D       = 65,
    Roll               = 66,
    Arange             = 67,
    TimestepEmbedding  = 68,
    Argsort            = 69,
    TopK               = 70,
    LeakyRelu         = 71,
    Tri                = 72,
    Fill               = 73,
    FlashAttnExt       = 74,
    FlashAttnBack      = 75,
    SsmConv            = 76,
    SsmScan            = 77,
    WinPart            = 78,
    WinUnpart          = 79,
    GetRelPos          = 80,
    AddRelPos          = 81,
    RwkvWkv6           = 82,
    GatedLinearAttn    = 83,
    RwkvWkv7           = 84,
    SolveTri           = 85,
    GatedDeltaNet      = 86,
    LightningIndexer   = 87,
    Dsv4HcComb         = 88,
    Dsv4HcPre          = 89,
    Dsv4HcPost         = 90,
    Unary              = 91,
    MapCustom1         = 92,
    MapCustom2         = 93,
    MapCustom3         = 94,
    Custom             = 95,
    CrossEntropyLoss   = 96,
    CrossEntropyLossBack = 97,
    OptStepAdamw       = 98,
    OptStepSgd         = 99,
    Glu                = 100,
}

// ---------------------------------------------------------------------------
// ggml_init_params — 对照 ggml.h L677-682
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct GgmlInitParams {
    pub mem_size: usize,          // size_t mem_size;  bytes
    pub mem_buffer: *mut c_void,  // void * mem_buffer (NULL → internal alloc)
    pub no_alloc: bool,           // bool no_alloc
}

// ---------------------------------------------------------------------------
// gguf_init_params — 对照 gguf.h L72-77
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct GgufInitParams {
    pub no_alloc: bool,                           // bool no_alloc
    pub ctx: *mut *mut GgmlContext,               // struct ggml_context ** ctx (optional)
}

// ---------------------------------------------------------------------------
// extern "C" declarations — 全部从 ggml.h / ggml-backend.h / ggml-cuda.h / gguf.h 抄录
// ---------------------------------------------------------------------------
extern "C" {
    // ---- Context (ggml.h L823) ----
    pub fn ggml_init(params: GgmlInitParams) -> *mut GgmlContext;
    /// 清空上下文的对象表（**不释放 arena**）：下一次分配会复用同样的地址。
    ///
    /// 用途（P2）：每步重建图时，如果每次都新建 ggml context，图节点的地址会漂移，
    /// 而 ggml-cuda 用 `cgraph->nodes[0]` 当 CUDA graph 的 key、并逐字段比较节点属性
    /// （含 `data` 指针）—— 地址一变就永远重放不了。复用同一个 ctx + `ggml_reset`
    /// 让每步的地址完全确定，于是 warmup 一次后就能持续重放。
    pub fn ggml_reset(ctx: *mut GgmlContext);
    pub fn ggml_free(ctx: *mut GgmlContext);

    // ---- Tensor creation (ggml.h L836-868) ----
    pub fn ggml_new_tensor(
        ctx: *mut GgmlContext,
        r#type: i32,
        n_dims: i32,
        ne: *const i64,
    ) -> *mut ggml_tensor;
    pub fn ggml_new_buffer(ctx: *mut GgmlContext, nbytes: usize) -> *mut c_void;

    // ---- View / reshape (ggml.h L1686-1706) ----
    pub fn ggml_view_3d(
        ctx: *mut GgmlContext,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        nb1: u64,
        nb2: u64,
        offset: u64,
    ) -> *mut ggml_tensor;

    // ---- Graph (ggml.h L2889-2894, 2906) ----
    pub fn ggml_new_graph(ctx: *mut GgmlContext) -> *mut GgmlCgraph;
    /// 指定节点上限建图（默认 `GGML_DEFAULT_GRAPH_SIZE` = 2048，40 层模型不够）。
    pub fn ggml_new_graph_custom(ctx: *mut GgmlContext, size: usize, grads: bool) -> *mut GgmlCgraph;
    pub fn ggml_build_forward_expand(cgraph: *mut GgmlCgraph, tensor: *mut ggml_tensor);
    pub fn ggml_graph_clear(cgraph: *mut GgmlCgraph);
    /// 图里的节点数（诊断/性能归因用）。
    pub fn ggml_graph_n_nodes(cgraph: *mut GgmlCgraph) -> i32;
    /// 取第 i 个节点。
    pub fn ggml_graph_node(cgraph: *mut GgmlCgraph, i: i32) -> *mut ggml_tensor;
    /// 算子名（如 "MUL_MAT" / "MUL_MAT_ID" / "FLASH_ATTN_EXT"）。
    pub fn ggml_op_name(op: i32) -> *const c_char;
    pub fn ggml_graph_get_tensor(cgraph: *const GgmlCgraph, name: *const c_char) -> *mut ggml_tensor;

    // ---- Backend / buffer (ggml-backend.h L79-82, 56, 86-95, 428) ----
    pub fn ggml_backend_cpu_init() -> *mut GgmlBackend;
    pub fn ggml_backend_cuda_init(device: i32) -> *mut GgmlBackend;
    pub fn ggml_backend_alloc_buffer(backend: *mut GgmlBackend, size: usize) -> *mut GgmlBackendBuffer;
    pub fn ggml_backend_buffer_free(buffer: *mut GgmlBackendBuffer);
    pub fn ggml_backend_buffer_init_tensor(buffer: *mut GgmlBackendBuffer, tensor: *mut ggml_tensor) -> i32;
    pub fn ggml_backend_tensor_alloc(buffer: *mut GgmlBackendBuffer, tensor: *mut ggml_tensor, addr: *mut c_void) -> i32;
    pub fn ggml_backend_graph_compute(backend: *mut GgmlBackend, cgraph: *mut GgmlCgraph) -> i32;
    pub fn ggml_backend_tensor_set(tensor: *mut ggml_tensor, data: *const c_void, offset: usize, size: usize);
    pub fn ggml_backend_tensor_get(tensor: *const ggml_tensor, data: *mut c_void, offset: usize, size: usize);
    pub fn ggml_backend_free(backend: *mut GgmlBackend);
    // ggml-alloc.h L82 — 要求 ctx 为 no_alloc；为本 ctx 内所有未分配 tensor 建 buffer
    pub fn ggml_backend_alloc_ctx_tensors(ctx: *mut GgmlContext, backend: *mut GgmlBackend) -> *mut GgmlBackendBuffer;
    pub fn ggml_backend_get_default_buffer_type(backend: *mut GgmlBackend) -> *mut GgmlBackendBufferType;

    // ---- graph allocator：跨步复用同一块 buffer（地址稳定、不每步新建后端）----
    pub fn ggml_gallocr_new(buft: *mut GgmlBackendBufferType) -> *mut GgmlGallocr;
    pub fn ggml_gallocr_alloc_graph(galloc: *mut GgmlGallocr, graph: *mut GgmlCgraph) -> bool;
    pub fn ggml_gallocr_free(galloc: *mut GgmlGallocr);

    // ---- Tensor shape / nbytes (ggml.h L761, 838) ----
    pub fn ggml_row_size(ty: i32, ne: i64) -> usize;
    pub fn ggml_nbytes(tensor: *const ggml_tensor) -> usize;

    // ---- 1D-4D tensor creation (ggml.h L842-866) ----
    pub fn ggml_new_tensor_1d(ctx: *mut GgmlContext, ty: i32, ne0: i64) -> *mut ggml_tensor;
    pub fn ggml_new_tensor_2d(ctx: *mut GgmlContext, ty: i32, ne0: i64, ne1: i64) -> *mut ggml_tensor;
    pub fn ggml_new_tensor_3d(ctx: *mut GgmlContext, ty: i32, ne0: i64, ne1: i64, ne2: i64) -> *mut ggml_tensor;
    pub fn ggml_new_tensor_4d(ctx: *mut GgmlContext, ty: i32, ne0: i64, ne1: i64, ne2: i64, ne3: i64) -> *mut ggml_tensor;

    // ---- Elementwise / reduce (ggml.h L902-1111, 1207, 1389-1402) ----
    pub fn ggml_dup(ctx: *mut GgmlContext, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_add(ctx: *mut GgmlContext, a: *mut ggml_tensor, b: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_mul(ctx: *mut GgmlContext, a: *mut ggml_tensor, b: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_sqr(ctx: *mut GgmlContext, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_sqrt(ctx: *mut GgmlContext, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_softplus(ctx: *mut GgmlContext, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_silu(ctx: *mut GgmlContext, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_cumsum(ctx: *mut GgmlContext, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_sum_rows(ctx: *mut GgmlContext, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_repeat(ctx: *mut GgmlContext, a: *mut ggml_tensor, b: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_concat(ctx: *mut GgmlContext, a: *mut ggml_tensor, b: *mut ggml_tensor, dim: i32) -> *mut ggml_tensor;
    pub fn ggml_norm(ctx: *mut GgmlContext, a: *mut ggml_tensor, eps: f32) -> *mut ggml_tensor;
    pub fn ggml_rms_norm(ctx: *mut GgmlContext, a: *mut ggml_tensor, eps: f32) -> *mut ggml_tensor;
    pub fn ggml_clamp(ctx: *mut GgmlContext, a: *mut ggml_tensor, min: f32, max: f32) -> *mut ggml_tensor;
    pub fn ggml_div(ctx: *mut GgmlContext, a: *mut ggml_tensor, b: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_scale(ctx: *mut GgmlContext, a: *mut ggml_tensor, s: f32) -> *mut ggml_tensor;
    pub fn ggml_sigmoid(ctx: *mut GgmlContext, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_tanh(ctx: *mut GgmlContext, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_neg(ctx: *mut GgmlContext, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_exp(ctx: *mut GgmlContext, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_log(ctx: *mut GgmlContext, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_repeat_4d(ctx: *mut GgmlContext, a: *mut ggml_tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) -> *mut ggml_tensor;

    // ---- Gated Delta Net (ggml.h L2639-2660) ----
    // 融合的 Gated DeltaNet 递推算子：内部完成 exp(g) 衰减、delta 规则更新与
    // (Sᵀq)/sqrt(S_v) 输出，是 llama.cpp `fused_gdn_*` 路径所用的 op。
    pub fn ggml_gated_delta_net(ctx: *mut GgmlContext, q: *mut ggml_tensor, k: *mut ggml_tensor, v: *mut ggml_tensor, g: *mut ggml_tensor, beta: *mut ggml_tensor, state: *mut ggml_tensor, K: i64) -> *mut ggml_tensor;

    // ---- Matmul (ggml.h L1482-1504) ----
    pub fn ggml_mul_mat(ctx: *mut GgmlContext, a: *mut ggml_tensor, b: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_mul_mat_id(ctx: *mut GgmlContext, as_: *mut ggml_tensor, b: *mut ggml_tensor, ids: *mut ggml_tensor) -> *mut ggml_tensor;

    // ---- Shape ops / views (ggml.h L1603-1729, 2359, 2427, 2452-2474, 2524) ----
    pub fn ggml_cont(ctx: *mut GgmlContext, a: *mut ggml_tensor) -> *mut ggml_tensor;
    /// 张量是否连续（ggml.h L793）。用于避免对**已经连续**的张量调 `ggml_cont` ——
    /// `ggml_cont` 走 `ggml_dup`，对连续张量也是**实打实的一次拷贝 kernel**。
    pub fn ggml_is_contiguous(a: *const ggml_tensor) -> bool;
    /// 门控线性单元（`GGML_OP_GLU`）：`a` = gate，`b` = up，`op` = ggml_glu_op。
    ///
    /// 关键点（P8）：ggml-cuda 有 `{MUL_MAT_ID, MUL_MAT_ID, GLU}` 的融合
    /// （见 ggml-cuda.cu 的 `mul_mat_id_glu_ops`），gate/up 两个 matmul + SwiGLU
    /// 会**合成一个 kernel**。但只有用 `GLU` 算子才触发 —— 写成 `silu(gate) * up`
    /// （UNARY + MUL）融合就完全不生效，MoE 每层多 2 个 kernel。
    pub fn ggml_glu_split(
        ctx: *mut GgmlContext,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        op: i32,
    ) -> *mut ggml_tensor;
    /// f32 → f16：FlashAttention 的 mask 必须是 F16。
    pub fn ggml_cast(ctx: *mut GgmlContext, a: *mut ggml_tensor, type_: i32) -> *mut ggml_tensor;
    /// 后端是否支持某个算子（P4：用 FA 之前必须问它，否则 ggml-cuda 会直接 ABORT）。
    pub fn ggml_backend_supports_op(backend: *mut GgmlBackend, op: *const ggml_tensor) -> bool;
    /// FlashAttention（ggml.h: q `[n_embd_k, n_batch, n_head, ne3]`、
    /// k `[n_embd_k, n_kv, n_head_kv, ne3]`（**不转置**）、
    /// v `[n_embd_v, n_kv, n_head_kv, ne3]`（**不转置**）、
    /// mask `[n_kv, n_batch, ne32, ne33]`（F16，`ne[0]` 必须严格等于 `k->ne[1]`）、
    /// 结果 `[n_embd_v, n_head, n_batch, ne3]`）。
    pub fn ggml_flash_attn_ext(
        ctx: *mut GgmlContext,
        q: *mut ggml_tensor,
        k: *mut ggml_tensor,
        v: *mut ggml_tensor,
        mask: *mut ggml_tensor,
        scale: f32,
        max_bias: f32,
        logit_softcap: f32,
    ) -> *mut ggml_tensor;
    /// `a -> b`：把 `a` 写入 `b`，返回 `b` 的视图（ggml.h ggml_cpy）。
    pub fn ggml_cpy(ctx: *mut GgmlContext, a: *mut ggml_tensor, b: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_reshape_2d(ctx: *mut GgmlContext, a: *mut ggml_tensor, ne0: i64, ne1: i64) -> *mut ggml_tensor;
    pub fn ggml_reshape_3d(ctx: *mut GgmlContext, a: *mut ggml_tensor, ne0: i64, ne1: i64, ne2: i64) -> *mut ggml_tensor;
    pub fn ggml_reshape_4d(ctx: *mut GgmlContext, a: *mut ggml_tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) -> *mut ggml_tensor;
    pub fn ggml_view_1d(ctx: *mut GgmlContext, a: *mut ggml_tensor, ne0: i64, offset: usize) -> *mut ggml_tensor;
    pub fn ggml_view_2d(ctx: *mut GgmlContext, a: *mut ggml_tensor, ne0: i64, ne1: i64, nb1: usize, offset: usize) -> *mut ggml_tensor;
    pub fn ggml_view_4d(
        ctx: *mut GgmlContext,
        a: *mut ggml_tensor,
        ne0: i64, ne1: i64, ne2: i64, ne3: i64,
        nb1: usize, nb2: usize, nb3: usize, offset: usize,
    ) -> *mut ggml_tensor;
    pub fn ggml_permute(ctx: *mut GgmlContext, a: *mut ggml_tensor, axis0: i32, axis1: i32, axis2: i32, axis3: i32) -> *mut ggml_tensor;
    pub fn ggml_transpose(ctx: *mut GgmlContext, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_get_rows(ctx: *mut GgmlContext, a: *mut ggml_tensor, b: *mut ggml_tensor) -> *mut ggml_tensor;
    /// `a[n_embd, ne1] ← b[n_embd, n_rows]`，行号取自 `c`（I64 `[n_rows]`，值 ∈ [0, ne1)）。
    ///
    /// 用途（P2）：KV cache 的写入必须让**目标张量指针固定**（整个 cache 张量，不带
    /// 逐步变化的 view 偏移），否则 ggml-cuda 的 `cuda_graph_update_required` 每步都
    /// 认为属性变了 → 永远退回 eager 执行；而 set_rows 把"写到哪一行"放进 `c` 的
    /// **内容**里，图的形状与指针就完全不变了（llama.cpp 的 `cpy_k/cpy_v` 同款做法）。
    pub fn ggml_set_rows(
        ctx: *mut GgmlContext,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    pub fn ggml_diag(ctx: *mut GgmlContext, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_soft_max(ctx: *mut GgmlContext, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_pad(ctx: *mut GgmlContext, a: *mut ggml_tensor, p0: i32, p1: i32, p2: i32, p3: i32) -> *mut ggml_tensor;
    pub fn ggml_fill(ctx: *mut GgmlContext, a: *mut ggml_tensor, c: f32) -> *mut ggml_tensor;
    pub fn ggml_argsort(ctx: *mut GgmlContext, a: *mut ggml_tensor, order: i32) -> *mut ggml_tensor;
    pub fn ggml_argsort_top_k(ctx: *mut GgmlContext, a: *mut ggml_tensor, k: i32) -> *mut ggml_tensor;
    pub fn ggml_top_k(ctx: *mut GgmlContext, a: *mut ggml_tensor, k: i32) -> *mut ggml_tensor;
    pub fn ggml_arange(ctx: *mut GgmlContext, start: f32, stop: f32, step: f32) -> *mut ggml_tensor;
    pub fn ggml_ssm_conv(ctx: *mut GgmlContext, sx: *mut ggml_tensor, c: *mut ggml_tensor) -> *mut ggml_tensor;

    // ---- RoPE (ggml.h L1937-1951) ----
    pub fn ggml_rope_multi(
        ctx: *mut GgmlContext,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        n_dims: i32,
        sections: *const i32,
        mode: i32,
        n_ctx_orig: i32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> *mut ggml_tensor;

    // ---- Quantization (ggml.h L2946-2953) ----
    pub fn ggml_quantize_chunk(
        ty: i32,
        src: *const f32,
        dst: *mut c_void,
        start: i64,
        nrows: i64,
        n_per_row: i64,
        imatrix: *const f32,
    ) -> usize;

    // ---- GGUF (gguf.h L83-90, 98-134) ----
    pub fn gguf_init_from_file(fname: *const c_char, params: GgufInitParams) -> *mut GgufContext;
    pub fn gguf_free(ctx: *mut GgufContext);
    pub fn gguf_get_n_kv(ctx: *const GgufContext) -> i64;
    pub fn gguf_find_key(ctx: *const GgufContext, key: *const c_char) -> i64;
    pub fn gguf_get_key(ctx: *const GgufContext, key_id: i64) -> *const c_char;
    pub fn gguf_get_kv_type(ctx: *const GgufContext, key_id: i64) -> i32;
    pub fn gguf_get_val_u8(ctx: *const GgufContext, key_id: i64) -> u8;
    pub fn gguf_get_val_i8(ctx: *const GgufContext, key_id: i64) -> i8;
    pub fn gguf_get_val_u16(ctx: *const GgufContext, key_id: i64) -> u16;
    pub fn gguf_get_val_i16(ctx: *const GgufContext, key_id: i64) -> i16;
    pub fn gguf_get_val_u32(ctx: *const GgufContext, key_id: i64) -> u32;
    pub fn gguf_get_val_i32(ctx: *const GgufContext, key_id: i64) -> i32;
    pub fn gguf_get_val_f32(ctx: *const GgufContext, key_id: i64) -> f32;
    pub fn gguf_get_val_u64(ctx: *const GgufContext, key_id: i64) -> u64;
    pub fn gguf_get_val_i64(ctx: *const GgufContext, key_id: i64) -> i64;
    pub fn gguf_get_val_f64(ctx: *const GgufContext, key_id: i64) -> f64;
    pub fn gguf_get_val_bool(ctx: *const GgufContext, key_id: i64) -> bool;
    pub fn gguf_get_val_str(ctx: *const GgufContext, key_id: i64) -> *const c_char;
    pub fn gguf_get_arr_type(ctx: *const GgufContext, key_id: i64) -> i32;
    pub fn gguf_get_arr_n(ctx: *const GgufContext, key_id: i64) -> usize;
    pub fn gguf_get_arr_data(ctx: *const GgufContext, key_id: i64) -> *const c_void;
    /// 读取字符串数组中第 i 个元素（`GGUF_TYPE_ARRAY` of `GGUF_TYPE_STRING`）。
    pub fn gguf_get_arr_str(ctx: *const GgufContext, key_id: i64, i: usize) -> *const c_char;
    pub fn gguf_get_n_tensors(ctx: *const GgufContext) -> i64;
    pub fn gguf_find_tensor(ctx: *const GgufContext, name: *const c_char) -> i64;
    /// 张量数据区在文件中的起始偏移。
    ///
    /// 注意 `gguf_get_tensor_offset` 返回的是**相对于数据区**的偏移；绝对文件偏移
    /// 必须是两者之和（对照 llama-model-loader.h L46）。
    pub fn gguf_get_data_offset(ctx: *const GgufContext) -> usize;
    pub fn gguf_get_tensor_offset(ctx: *const GgufContext, tensor_id: i64) -> usize;
    pub fn gguf_get_tensor_name(ctx: *const GgufContext, tensor_id: i64) -> *const c_char;
    pub fn gguf_get_tensor_ne(ctx: *const GgufContext, tensor_id: i64) -> *const i64;
    pub fn gguf_get_tensor_type(ctx: *const GgufContext, tensor_id: i64) -> i32;
    pub fn gguf_get_tensor_size(ctx: *const GgufContext, tensor_id: i64) -> usize;
}

// ---------------------------------------------------------------------------
// Helper: convert a C string to Rust &str (unsafe but safe for ggml's ASCII names)
// ---------------------------------------------------------------------------
pub unsafe fn c_str_to_str<'a>(ptr: *const c_char) -> &'a str {
    use std::ffi::CStr;
    CStr::from_ptr(ptr).to_str().unwrap()
}

// ---------------------------------------------------------------------------
// 常量 — 对照 ggml.h（禁止猜测）
// ---------------------------------------------------------------------------
// RoPE mode, ggml.h L250-254
pub const GGML_ROPE_TYPE_NORMAL: i32 = 0;
pub const GGML_ROPE_TYPE_NEOX: i32 = 2;
pub const GGML_ROPE_TYPE_MROPE: i32 = 8;
pub const GGML_ROPE_TYPE_VISION: i32 = 24;
pub const GGML_ROPE_TYPE_IMROPE: i32 = 40;

// enum ggml_sort_order, ggml.h L2447-2450
pub const GGML_SORT_ORDER_ASC: i32 = 0;
pub const GGML_SORT_ORDER_DESC: i32 = 1;

// GGML_MROPE_SECTIONS, ggml.h L256
pub const GGML_MROPE_SECTIONS: usize = 4;
