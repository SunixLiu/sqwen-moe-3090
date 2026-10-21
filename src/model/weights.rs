// src/model/weights.rs — GGUF 元数据读取 + 张量按名映射 + 权重加载
//
// 加载流程严格遵循 llama.cpp 的两阶段模式（`llama_model_loader` +
// `ggml_backend_alloc_ctx_tensors`）：
//
//   阶段 1  `ModelWeights::create(&ctx, &gguf, &arch)`
//           —— 在 no_alloc 的 ggml_context 里，按 GGUF 元数据（名字 / shape /
//              量化类型）创建全部需要的张量，并建立 名字→字段 映射。
//              tensor->data 仍为 NULL。
//   阶段 2  `BackendBuffer::from_ctx(&ctx, kind)`
//           —— 一次性为 ctx 内所有张量分配后端内存（CUDA 显存 / 主机内存）。
//   阶段 3  `weights.load_data(&ctx, &gguf)`
//           —— 单文件句柄、分块把每个张量的数据从 GGUF 数据区搬进已分配内存。
//
// 之所以必须分成三步：`ggml_backend_alloc_ctx_tensors` 要求 ctx 是 no_alloc，
// 且只有当 tensor->buffer 已就绪后才能用 `ggml_backend_tensor_set` 写数据。

use std::ffi::CString;
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ptr;
use std::time::Instant;

use crate::backend::ctx::{BackendBuffer, BackendKind, GgmlCtx, Tensor};
use crate::backend::ffi::{self, GgmlType};
use crate::model::arch::Qwen35MoeArch;

// ---------------------------------------------------------------------------
// GgufFile — 只读 GGUF 元数据包装（no_alloc，不加载张量数据）
// ---------------------------------------------------------------------------
pub struct GgufFile {
    ctx: *mut ffi::GgufContext,
    path: String,
}

unsafe impl Send for GgufFile {}
unsafe impl Sync for GgufFile {}

impl Drop for GgufFile {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe {
                ffi::gguf_free(self.ctx);
            }
        }
    }
}

#[derive(Debug)]
pub enum GgufError {
    OpenFailed,
    ArchMismatch(String),
    MissingKey(String),
    TensorNotFound(String),
    LoadFailed(String),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::OpenFailed => write!(f, "无法打开 GGUF 文件"),
            GgufError::ArchMismatch(s) => write!(f, "架构不匹配: {s}"),
            GgufError::MissingKey(k) => write!(f, "缺少 GGUF 键: {k}"),
            GgufError::TensorNotFound(n) => write!(f, "缺少张量: {n}"),
            GgufError::LoadFailed(s) => write!(f, "加载失败: {s}"),
        }
    }
}

impl std::error::Error for GgufError {}

impl GgufFile {
    pub fn open(path: &str) -> Result<Self, GgufError> {
        let c_path = match CString::new(path) {
            Ok(c) => c,
            Err(_) => return Err(GgufError::OpenFailed),
        };

        unsafe {
            let params = ffi::GgufInitParams { no_alloc: true, ctx: ptr::null_mut() };
            let ctx = ffi::gguf_init_from_file(c_path.as_ptr(), params);
            if ctx.is_null() {
                return Err(GgufError::OpenFailed);
            }
            Ok(GgufFile { ctx, path: path.to_string() })
        }
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn tensor_count(&self) -> u64 {
        unsafe { ffi::gguf_get_n_tensors(self.ctx) as u64 }
    }

    pub fn tensor_name(&self, i: u64) -> Option<String> {
        unsafe {
            let name = ffi::gguf_get_tensor_name(self.ctx, i as i64);
            if name.is_null() {
                return None;
            }
            Some(ffi::c_str_to_str(name).to_string())
        }
    }

    pub fn find_tensor(&self, name: &str) -> Option<i64> {
        unsafe {
            let c_name = CString::new(name).ok()?;
            let id = ffi::gguf_find_tensor(self.ctx, c_name.as_ptr());
            if id < 0 {
                return None;
            }
            Some(id)
        }
    }

    /// 张量数据区在文件中的起始偏移（`gguf_get_data_offset`）。
    pub fn data_offset(&self) -> usize {
        unsafe { ffi::gguf_get_data_offset(self.ctx) }
    }

    /// 张量在文件中的**绝对**字节偏移。
    ///
    /// `gguf_get_tensor_offset` 给的是相对于数据区的偏移，必须加上数据区起点
    /// （`gguf_get_data_offset`）。少了这一项会读到文件头/metadata 区域 —— 权重全是
    /// 垃圾，而且「加载后的字节 == read_raw 的字节」这种自比对测试还会通过。
    /// 对照 llama.cpp `llama-model-loader.h` L46：
    /// `offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx)`
    pub fn tensor_offset(&self, tensor_id: i64) -> usize {
        unsafe {
            ffi::gguf_get_data_offset(self.ctx) + ffi::gguf_get_tensor_offset(self.ctx, tensor_id)
        }
    }

    /// 张量数据字节数。
    pub fn tensor_size(&self, tensor_id: i64) -> usize {
        unsafe { ffi::gguf_get_tensor_size(self.ctx, tensor_id) }
    }

    /// 获取 GGUF 张量的 ne 数组（形状）。
    pub fn tensor_ne(&self, tensor_id: i64) -> [i64; 4] {
        unsafe {
            let ne_ptr = ffi::gguf_get_tensor_ne(self.ctx, tensor_id);
            if ne_ptr.is_null() {
                return [1, 1, 1, 1];
            }
            ptr::read(ne_ptr as *const [i64; 4])
        }
    }

    /// 获取 GGUF 张量的 ggml_type（原始数值）。
    pub fn tensor_type(&self, tensor_id: i64) -> i32 {
        unsafe { ffi::gguf_get_tensor_type(self.ctx, tensor_id) }
    }

    // ---- KV 访问 ----
    //
    // 所有访问器都先用 `gguf_get_kv_type` 校验值的类型再取。这不是洁癖：
    // ggml 的 `gguf_get_val_*` 内部是 `GGML_ASSERT(type_to_gguf_type<T>::value == type)`，
    // 类型不匹配时是 **abort 整个进程**（不是可捕获的 panic）。踩过一次：
    // 拿 `kv_get_u32` 去读一个 i32/u64 型的 key，直接 STATUS_STACK_BUFFER_OVERRUN。

    fn key_id(&self, key: &str) -> Option<i64> {
        unsafe {
            let c_key = CString::new(key).ok()?;
            let id = ffi::gguf_find_key(self.ctx, c_key.as_ptr());
            if id < 0 {
                None
            } else {
                Some(id)
            }
        }
    }

    /// KV 的 gguf 值类型编号（见 `GGUF_TYPE_*`）；key 不存在返回 None。
    pub fn kv_type(&self, key: &str) -> Option<i32> {
        let id = self.key_id(key)?;
        Some(unsafe { ffi::gguf_get_kv_type(self.ctx, id) })
    }

    /// 只在该 key 的类型属于 `want` 时返回 key id（否则 None，绝不硬取）。
    fn typed_id(&self, key: &str, want: &[i32]) -> Option<i64> {
        let id = self.key_id(key)?;
        let t = unsafe { ffi::gguf_get_kv_type(self.ctx, id) };
        if want.contains(&t) {
            Some(id)
        } else {
            None
        }
    }

    pub fn kv_get_str(&self, key: &str) -> Option<String> {
        let id = self.typed_id(key, &[GGUF_TYPE_STRING])?;
        unsafe {
            let val = ffi::gguf_get_val_str(self.ctx, id);
            if val.is_null() {
                return None;
            }
            Some(ffi::c_str_to_str(val).to_string())
        }
    }

    pub fn kv_get_u32(&self, key: &str) -> Option<u32> {
        let id = self.typed_id(key, &[GGUF_TYPE_U32])?;
        unsafe { Some(ffi::gguf_get_val_u32(self.ctx, id)) }
    }

    /// 读一个整数并转成 i64：`GGUF` 里同一语义的 key 在不同模型上可能是
    /// i32 / u32 / i64 / u64，这里统一收口。
    pub fn kv_get_i64(&self, key: &str) -> Option<i64> {
        let id = self.typed_id(
            key,
            &[GGUF_TYPE_I32, GGUF_TYPE_U32, GGUF_TYPE_I64, GGUF_TYPE_U64],
        )?;
        unsafe {
            Some(match self.kv_type(key)? {
                GGUF_TYPE_I32 => ffi::gguf_get_val_i32(self.ctx, id) as i64,
                GGUF_TYPE_U32 => ffi::gguf_get_val_u32(self.ctx, id) as i64,
                GGUF_TYPE_U64 => ffi::gguf_get_val_u64(self.ctx, id) as i64,
                _ => ffi::gguf_get_val_i64(self.ctx, id),
            })
        }
    }

    /// 只读 i32（需要区分 i32 与 i64 时用）。
    pub fn kv_get_i32(&self, key: &str) -> Option<i32> {
        let id = self.typed_id(key, &[GGUF_TYPE_I32])?;
        unsafe { Some(ffi::gguf_get_val_i32(self.ctx, id)) }
    }

    pub fn kv_get_f32(&self, key: &str) -> Option<f32> {
        let id = self.typed_id(key, &[GGUF_TYPE_F32])?;
        unsafe { Some(ffi::gguf_get_val_f32(self.ctx, id)) }
    }

    pub fn kv_get_bool(&self, key: &str) -> Option<bool> {
        let id = self.typed_id(key, &[GGUF_TYPE_BOOL])?;
        unsafe { Some(ffi::gguf_get_val_bool(self.ctx, id)) }
    }

    pub fn kv_get_u64(&self, key: &str) -> Option<u64> {
        let id = self.typed_id(key, &[GGUF_TYPE_U64])?;
        unsafe { Some(ffi::gguf_get_val_u64(self.ctx, id)) }
    }

    pub fn kv_get_f64(&self, key: &str) -> Option<f64> {
        let id = self.typed_id(key, &[GGUF_TYPE_F64, GGUF_TYPE_F32])?;
        unsafe {
            Some(match self.kv_type(key)? {
                GGUF_TYPE_F32 => ffi::gguf_get_val_f32(self.ctx, id) as f64,
                _ => ffi::gguf_get_val_f64(self.ctx, id),
            })
        }
    }

    /// 读取 i32 数组型 KV（例如 `rope.dimension_sections`）。
    pub fn kv_get_i32_array(&self, key: &str, out: &mut [i32]) -> Option<usize> {
        let id = self.typed_id(key, &[GGUF_TYPE_ARRAY])?;
        unsafe {
            if ffi::gguf_get_arr_type(self.ctx, id) != GGUF_TYPE_I32 {
                return None;
            }
            let n = ffi::gguf_get_arr_n(self.ctx, id);
            let data = ffi::gguf_get_arr_data(self.ctx, id) as *const i32;
            if data.is_null() {
                return None;
            }
            let take = (n as usize).min(out.len());
            ptr::copy_nonoverlapping(data, out.as_mut_ptr(), take);
            Some(n as usize)
        }
    }

    /// 读取一个 KV f32 数组（`GGUF_TYPE_ARRAY` of `GGUF_TYPE_F32`）。
    pub fn kv_get_f32_array(&self, key: &str) -> Option<Vec<f32>> {
        let id = self.typed_id(key, &[GGUF_TYPE_ARRAY])?;
        unsafe {
            if ffi::gguf_get_arr_type(self.ctx, id) != GGUF_TYPE_F32 {
                return None;
            }
            let n = ffi::gguf_get_arr_n(self.ctx, id);
            let data = ffi::gguf_get_arr_data(self.ctx, id) as *const f32;
            if data.is_null() {
                return None;
            }
            Some(std::slice::from_raw_parts(data, n as usize).to_vec())
        }
    }

    /// 读取 i32 数组型 KV（`GGUF_TYPE_ARRAY` of `GGUF_TYPE_I32`），长度未知时用。
    pub fn kv_get_i32_array_vec(&self, key: &str) -> Option<Vec<i32>> {
        let id = self.typed_id(key, &[GGUF_TYPE_ARRAY])?;
        unsafe {
            if ffi::gguf_get_arr_type(self.ctx, id) != GGUF_TYPE_I32 {
                return None;
            }
            let n = ffi::gguf_get_arr_n(self.ctx, id);
            let data = ffi::gguf_get_arr_data(self.ctx, id) as *const i32;
            if data.is_null() {
                return None;
            }
            Some(std::slice::from_raw_parts(data, n as usize).to_vec())
        }
    }

    /// 读取一个 KV 字符串数组（`GGUF_TYPE_ARRAY` of `GGUF_TYPE_STRING`）。
    pub fn kv_get_str_array(&self, key: &str) -> Option<Vec<String>> {
        let id = self.typed_id(key, &[GGUF_TYPE_ARRAY])?;
        unsafe {
            if ffi::gguf_get_arr_type(self.ctx, id) != GGUF_TYPE_STRING {
                return None;
            }
            let n = ffi::gguf_get_arr_n(self.ctx, id);
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let ptr = ffi::gguf_get_arr_str(self.ctx, id, i);
                if ptr.is_null() {
                    continue;
                }
                if let Ok(s) = std::ffi::CStr::from_ptr(ptr).to_str() {
                    out.push(s.to_string());
                }
            }
            Some(out)
        }
    }
}

/// gguf 值类型编号（与 ggml `enum gguf_type` 一致）。
pub const GGUF_TYPE_U8: i32 = 0;
pub const GGUF_TYPE_I8: i32 = 1;
pub const GGUF_TYPE_U16: i32 = 2;
pub const GGUF_TYPE_I16: i32 = 3;
pub const GGUF_TYPE_U32: i32 = 4;
pub const GGUF_TYPE_I32: i32 = 5;
pub const GGUF_TYPE_F32: i32 = 6;
pub const GGUF_TYPE_BOOL: i32 = 7;
pub const GGUF_TYPE_STRING: i32 = 8;
pub const GGUF_TYPE_ARRAY: i32 = 9;
pub const GGUF_TYPE_U64: i32 = 10;
pub const GGUF_TYPE_I64: i32 = 11;
pub const GGUF_TYPE_F64: i32 = 12;

// ---------------------------------------------------------------------------
// 张量引用
// ---------------------------------------------------------------------------

/// 指向 ggml_context 内某个张量的裸指针句柄。
///
/// 安全性约定：`GgmlCtx` 只是 `*mut ggml_context` 的所有权封装，移动或借用它
/// 都不会移动 ggml 的内存池，所以把裸指针保存进由同一 owner 持有的结构是安全的
/// （owner 必须先构造 ctx、后析构 —— `Model` 的字段顺序保证了这一点）。
/// 所有解引用都必须传入仍然存活的 `&GgmlCtx`。
#[derive(Clone, Copy)]
pub struct TensorRef(*mut ffi::ggml_tensor);

unsafe impl Send for TensorRef {}
unsafe impl Sync for TensorRef {}

impl TensorRef {
    pub const NULL: TensorRef = TensorRef(ptr::null_mut());

    pub fn is_null(self) -> bool {
        self.0.is_null()
    }

    pub fn as_ptr(self) -> *mut ffi::ggml_tensor {
        self.0
    }

    /// 从原始指针创建（用于测试和外部构造）
    pub unsafe fn from_ptr(ptr: *mut ffi::ggml_tensor) -> Self {
        Self(ptr)
    }

    /// 借用为类型安全的 `Tensor`。
    pub fn get<'a>(self, ctx: &'a GgmlCtx) -> Tensor<'a> {
        assert!(!self.0.is_null(), "TensorRef::get on NULL");
        ctx.wrap(self.0)
    }
}

impl fmt::Debug for TensorRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_null() {
            return write!(f, "TensorRef(NULL)");
        }
        unsafe {
            let name = ffi::c_str_to_str((*self.0).name.as_ptr());
            write!(
                f,
                "TensorRef({} ne=[{},{},{},{}] type={})",
                name, (*self.0).ne[0], (*self.0).ne[1], (*self.0).ne[2], (*self.0).ne[3], (*self.0).r#type
            )
        }
    }
}

// ---------------------------------------------------------------------------
// 分层权重（字段命名与 GGUF 实际张量名一一对应）
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct LayerWeights {
    // ---- 每层都有 ----
    pub attn_norm: TensorRef,        // blk.{i}.attn_norm.weight
    pub attn_post_norm: TensorRef,   // blk.{i}.post_attention_norm.weight
    pub ffn_gate_inp: TensorRef,     // blk.{i}.ffn_gate_inp.weight

    // 共享专家
    pub ffn_gate_inp_shexp: TensorRef, // blk.{i}.ffn_gate_inp_shexp.weight
    pub ffn_gate_shexp: TensorRef,     // blk.{i}.ffn_gate_shexp.weight
    pub ffn_up_shexp: TensorRef,       // blk.{i}.ffn_up_shexp.weight
    pub ffn_down_shexp: TensorRef,     // blk.{i}.ffn_down_shexp.weight

    // 路由专家：fused 或分离布局，二者取其一
    pub ffn_gate_up_exps: TensorRef, // blk.{i}.ffn_gate_up_exps.weight（本模型为 NULL）
    pub ffn_gate_exps: TensorRef,    // blk.{i}.ffn_gate_exps.weight
    pub ffn_up_exps: TensorRef,      // blk.{i}.ffn_up_exps.weight
    pub ffn_down_exps: TensorRef,    // blk.{i}.ffn_down_exps.weight

    // ---- 循环层（gated delta net）专有 ----
    pub attn_qkv: TensorRef,     // blk.{i}.attn_qkv.weight     [n_embd, conv_dim]
    pub attn_gate: TensorRef,    // blk.{i}.attn_gate.weight    [n_embd, value_dim]
    pub ssm_conv1d: TensorRef,   // blk.{i}.ssm_conv1d.weight   [d_conv, conv_dim]
    pub ssm_dt: TensorRef,       // blk.{i}.ssm_dt.bias         [dt_rank]
    pub ssm_a: TensorRef,        // blk.{i}.ssm_a               [dt_rank]
    pub ssm_beta: TensorRef,     // blk.{i}.ssm_beta.weight     [n_embd, n_v_heads]
    pub ssm_alpha: TensorRef,    // blk.{i}.ssm_alpha.weight    [n_embd, n_v_heads]
    pub ssm_norm: TensorRef,     // blk.{i}.ssm_norm.weight     [head_v_dim]
    pub ssm_out: TensorRef,      // blk.{i}.ssm_out.weight      [value_dim, n_embd]

    // ---- 全注意力层专有 ----
    pub attn_q: TensorRef,       // blk.{i}.attn_q.weight        [n_embd, n_head*head_dim*2]（Q+gate 融合）
    pub attn_k: TensorRef,       // blk.{i}.attn_k.weight        [n_embd, n_head_kv*head_dim]
    pub attn_v: TensorRef,       // blk.{i}.attn_v.weight        [n_embd, n_head_kv*head_dim]
    pub attn_output: TensorRef,  // blk.{i}.attn_output.weight   [n_head*head_dim, n_embd]
    pub attn_q_norm: TensorRef,  // blk.{i}.attn_q_norm.weight   [head_dim]
    pub attn_k_norm: TensorRef,  // blk.{i}.attn_k_norm.weight   [head_dim]
}

impl LayerWeights {
    pub const EMPTY: LayerWeights = LayerWeights {
        attn_norm: TensorRef::NULL,
        attn_post_norm: TensorRef::NULL,
        ffn_gate_inp: TensorRef::NULL,
        ffn_gate_inp_shexp: TensorRef::NULL,
        ffn_gate_shexp: TensorRef::NULL,
        ffn_up_shexp: TensorRef::NULL,
        ffn_down_shexp: TensorRef::NULL,
        ffn_gate_up_exps: TensorRef::NULL,
        ffn_gate_exps: TensorRef::NULL,
        ffn_up_exps: TensorRef::NULL,
        ffn_down_exps: TensorRef::NULL,
        attn_qkv: TensorRef::NULL,
        attn_gate: TensorRef::NULL,
        ssm_conv1d: TensorRef::NULL,
        ssm_dt: TensorRef::NULL,
        ssm_a: TensorRef::NULL,
        ssm_beta: TensorRef::NULL,
        ssm_alpha: TensorRef::NULL,
        ssm_norm: TensorRef::NULL,
        ssm_out: TensorRef::NULL,
        attn_q: TensorRef::NULL,
        attn_k: TensorRef::NULL,
        attn_v: TensorRef::NULL,
        attn_output: TensorRef::NULL,
        attn_q_norm: TensorRef::NULL,
        attn_k_norm: TensorRef::NULL,
    };
}

// ---------------------------------------------------------------------------
// MTP / NextN 头（投机解码的草稿模型）
// ---------------------------------------------------------------------------
/// `blk.{n_layer_trunk}` 的 MTP（NextN）头。
///
/// 蓝本：llama.cpp `src/models/qwen35moe.cpp` 的 `load_block_mtp` / `graph_mtp`。
/// MTP 头 = 一个**完整的高斯全注意力块**（attn + MoE + 共享专家）加上四个投影：
///
/// ```text
///   concat(enorm(emb(x_p)), hnorm(h_{p-1}))  →  eh_proj  →  注意力块  →  MoE  →  shared_head_norm  →  lm_head
/// ```
///
/// 因此 `block` 复用了主干那套 [`LayerWeights`]（第 40 层也是全注意力 + MoE）。
/// 本模型没有 `nextn.embed_tokens` / `nextn.shared_head_head`，按 llama.cpp 的
/// 回退规则用 `token_embd.weight` / `output.weight`。
#[derive(Clone, Copy, Debug)]
pub struct MtpWeights {
    pub block: LayerWeights,
    /// `blk.{n}.nextn.eh_proj.weight` `[2*n_embd, n_embd]`
    pub eh_proj: TensorRef,
    /// `blk.{n}.nextn.enorm.weight` `[n_embd]`
    pub enorm: TensorRef,
    /// `blk.{n}.nextn.hnorm.weight` `[n_embd]`
    pub hnorm: TensorRef,
    /// `blk.{n}.nextn.shared_head_norm.weight` `[n_embd]`（缺失时回退 `output_norm`）
    pub shared_head_norm: TensorRef,
    /// `blk.{n}.nextn.shared_head_head.weight`（缺失时为 NULL，调用方回退 `output`）
    pub shared_head_head: TensorRef,
    /// `blk.{n}.nextn.embed_tokens.weight`（缺失时为 NULL，调用方回退 `token_embd`）
    pub embed_tokens: TensorRef,
}

// ---------------------------------------------------------------------------
// ModelWeights
// ---------------------------------------------------------------------------
pub struct ModelWeights {
    pub token_embd: TensorRef,
    pub output_norm: TensorRef,
    pub output: TensorRef,
    /// 主干层（`arch.n_layer_trunk()` 个，不含 MTP 层）
    pub layers: Vec<LayerWeights>,
    /// MTP / NextN 头；`None` = 未加载（`load_mtp = false`，与 llama.cpp 一致：
    /// 整块跳过并省下约 0.5 GiB 显存）
    pub mtp: Option<MtpWeights>,
    /// 已创建的全部张量：(张量, GGUF tensor_id)，顺序即创建顺序
    entries: Vec<(TensorRef, i64)>,
    /// 与 entries 一一对应的张量名（用于报错与加载日志）
    names: Vec<String>,
    total_bytes: usize,
}

/// 一次 `load_data` 的统计。
#[derive(Debug, Clone, Copy)]
pub struct LoadReport {
    pub tensors: usize,
    pub bytes: usize,
    pub elapsed_ms: u128,
}

impl ModelWeights {
    /// 阶段 1：按 GGUF 元数据在 no_alloc context 中创建全部张量并建立映射。
    ///
    /// 不加载 MTP 头的便捷入口（等价于 `create_with_mtp(..., false)`）。
    pub fn create(ctx: &GgmlCtx, gguf: &GgufFile, arch: &Qwen35MoeArch) -> Result<Self, GgufError> {
        Self::create_with_mtp(ctx, gguf, arch, false)
    }

    /// 同上，但可以要求加载 MTP / NextN 头（`--spec-type draft-mtp`）。
    ///
    /// 只读元数据（不碰数据区），因此很快。`load_mtp = false` 时 MTP / NextN 层的
    /// 张量全部跳过（同 llama.cpp 的默认：不加 `--spec-type draft-mtp` 就不加载）。
    pub fn create_with_mtp(
        ctx: &GgmlCtx,
        gguf: &GgufFile,
        arch: &Qwen35MoeArch,
        load_mtp: bool,
    ) -> Result<Self, GgufError> {
        let mut f = TensorFactory::new(ctx, gguf);

        let token_embd = f.make("token_embd.weight", true)?;
        let output_norm = f.make("output_norm.weight", true)?;
        // output.weight 缺失时复用 token_embd（tied embedding），与 llama.cpp 一致
        let output = {
            let o = f.make("output.weight", false)?;
            if o.is_null() {
                token_embd
            } else {
                o
            }
        };

        let n_trunk = arch.n_layer_trunk();
        let mut layers = Vec::with_capacity(n_trunk as usize);
        for il in 0..n_trunk {
            let mut w = LayerWeights::EMPTY;
            let p = |s: &str| format!("blk.{il}.{s}");

            w.attn_norm = f.make(&p("attn_norm.weight"), true)?;
            w.attn_post_norm = f.make(&p("post_attention_norm.weight"), true)?;

            if arch.is_recurrent_attn(il) {
                w.attn_qkv = f.make(&p("attn_qkv.weight"), true)?;
                w.attn_gate = f.make(&p("attn_gate.weight"), true)?;
                w.ssm_conv1d = f.make(&p("ssm_conv1d.weight"), true)?;
                w.ssm_dt = f.make(&p("ssm_dt.bias"), true)?;
                w.ssm_a = f.make(&p("ssm_a"), true)?;
                w.ssm_beta = f.make(&p("ssm_beta.weight"), true)?;
                w.ssm_alpha = f.make(&p("ssm_alpha.weight"), true)?;
                w.ssm_norm = f.make(&p("ssm_norm.weight"), true)?;
                w.ssm_out = f.make(&p("ssm_out.weight"), true)?;
            } else {
                w.attn_q = f.make(&p("attn_q.weight"), true)?;
                w.attn_k = f.make(&p("attn_k.weight"), true)?;
                w.attn_v = f.make(&p("attn_v.weight"), true)?;
                w.attn_output = f.make(&p("attn_output.weight"), true)?;
                w.attn_q_norm = f.make(&p("attn_q_norm.weight"), true)?;
                w.attn_k_norm = f.make(&p("attn_k_norm.weight"), true)?;
            }

            // 路由 MoE（每层都有）
            w.ffn_gate_inp = f.make(&p("ffn_gate_inp.weight"), true)?;
            w.ffn_down_exps = f.make(&p("ffn_down_exps.weight"), true)?;
            w.ffn_gate_up_exps = f.make(&p("ffn_gate_up_exps.weight"), false)?;
            if w.ffn_gate_up_exps.is_null() {
                w.ffn_gate_exps = f.make(&p("ffn_gate_exps.weight"), true)?;
                w.ffn_up_exps = f.make(&p("ffn_up_exps.weight"), true)?;
            } else {
                // fused 布局：gate/up 由 view 切出，此处只记录 fused 张量
                return Err(GgufError::LoadFailed(format!(
                    "{}: fused ffn_gate_up_exps 布局尚未支持（本引擎只支持分离的 ffn_gate_exps/ffn_up_exps）",
                    p("ffn_gate_up_exps.weight")
                )));
            }

            // 共享专家（每层都有）
            w.ffn_gate_inp_shexp = f.make(&p("ffn_gate_inp_shexp.weight"), true)?;
            w.ffn_gate_shexp = f.make(&p("ffn_gate_shexp.weight"), true)?;
            w.ffn_up_shexp = f.make(&p("ffn_up_shexp.weight"), true)?;
            w.ffn_down_shexp = f.make(&p("ffn_down_shexp.weight"), true)?;

            layers.push(w);
        }

        // MTP / NextN 头（`blk.{n_trunk}` 那一块）。默认不加载。
        let mtp = if load_mtp {
            let il = n_trunk;
            let p = |s: &str| format!("blk.{il}.{s}");
            let mut w = LayerWeights::EMPTY;
            // MTP 块是一个"全注意力 + MoE"的完整解码块（`arch.is_full_attn(il)` 为真）
            w.attn_norm = f.make(&p("attn_norm.weight"), true)?;
            w.attn_post_norm = f.make(&p("post_attention_norm.weight"), true)?;
            w.attn_q = f.make(&p("attn_q.weight"), true)?;
            w.attn_k = f.make(&p("attn_k.weight"), true)?;
            w.attn_v = f.make(&p("attn_v.weight"), true)?;
            w.attn_output = f.make(&p("attn_output.weight"), true)?;
            w.attn_q_norm = f.make(&p("attn_q_norm.weight"), true)?;
            w.attn_k_norm = f.make(&p("attn_k_norm.weight"), true)?;
            w.ffn_gate_inp = f.make(&p("ffn_gate_inp.weight"), true)?;
            w.ffn_down_exps = f.make(&p("ffn_down_exps.weight"), true)?;
            let fused = f.make(&p("ffn_gate_up_exps.weight"), false)?;
            if fused.is_null() {
                w.ffn_gate_exps = f.make(&p("ffn_gate_exps.weight"), true)?;
                w.ffn_up_exps = f.make(&p("ffn_up_exps.weight"), true)?;
            } else {
                return Err(GgufError::LoadFailed(format!(
                    "{}: fused ffn_gate_up_exps 布局尚未支持",
                    p("ffn_gate_up_exps.weight")
                )));
            }
            w.ffn_gate_inp_shexp = f.make(&p("ffn_gate_inp_shexp.weight"), true)?;
            w.ffn_gate_shexp = f.make(&p("ffn_gate_shexp.weight"), true)?;
            w.ffn_up_shexp = f.make(&p("ffn_up_shexp.weight"), true)?;
            w.ffn_down_shexp = f.make(&p("ffn_down_shexp.weight"), true)?;

            let n = |s: &str| format!("blk.{il}.nextn.{s}");
            Some(MtpWeights {
                block: w,
                eh_proj: f.make(&n("eh_proj.weight"), true)?,
                enorm: f.make(&n("enorm.weight"), true)?,
                hnorm: f.make(&n("hnorm.weight"), true)?,
                shared_head_norm: f.make(&n("shared_head_norm.weight"), false)?,
                shared_head_head: f.make(&n("shared_head_head.weight"), false)?,
                embed_tokens: f.make(&n("embed_tokens.weight"), false)?,
            })
        } else {
            // TODO: 未加载时按 llama.cpp 打印 "unused tensor" 提示
            None
        };

        let (entries, names, total_bytes) = f.into_parts();
        Ok(ModelWeights { token_embd, output_norm, output, layers, mtp, entries, names, total_bytes })
    }

    /// 创建出的张量数量（= 需要从 GGUF 加载的张量数）。
    pub fn n_tensors(&self) -> usize {
        self.entries.len()
    }

    /// 全部张量的字节数合计（≈ 显存占用）。
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// 已创建张量清单：(张量, GGUF tensor_id)，顺序为创建顺序。
    pub fn entries(&self) -> &[(TensorRef, i64)] {
        &self.entries
    }

    /// MTP / NextN 头的字节数合计（用于启动提示；未加载时为 0）。
    pub fn mtp_weight_bytes(&self, _ctx: &GgmlCtx) -> Option<usize> {
        let mtp = self.mtp.as_ref()?;
        let mut names: Vec<&str> = Vec::new();
        let _ = &mut names;
        let mut total = 0usize;
        let mut add = |t: TensorRef| {
            if !t.is_null() {
                total += t.get(_ctx).nbytes();
            }
        };
        add(mtp.eh_proj);
        add(mtp.enorm);
        add(mtp.hnorm);
        add(mtp.shared_head_norm);
        add(mtp.shared_head_head);
        add(mtp.embed_tokens);
        let b = &mtp.block;
        for t in [
            b.attn_norm,
            b.attn_post_norm,
            b.attn_q,
            b.attn_k,
            b.attn_v,
            b.attn_output,
            b.attn_q_norm,
            b.attn_k_norm,
            b.ffn_gate_inp,
            b.ffn_gate_exps,
            b.ffn_up_exps,
            b.ffn_down_exps,
            b.ffn_gate_inp_shexp,
            b.ffn_gate_shexp,
            b.ffn_up_shexp,
            b.ffn_down_shexp,
        ] {
            add(t);
        }
        Some(total)
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// 阶段 3：把全部张量数据从 GGUF 读入已分配的后端内存。
    pub fn load_data(&self, ctx: &GgmlCtx, gguf: &GgufFile) -> Result<LoadReport, GgufError> {
        self.load_data_budget(ctx, gguf, usize::MAX)
    }

    /// 只加载预算字节内的张量（按创建顺序）。
    ///
    /// 用于低显存/快速自检；`budget = usize::MAX` 表示全部加载。
    /// 至少会加载第一个张量，以便能报出「buffer 未分配」这类错误。
    pub fn load_data_budget(
        &self,
        ctx: &GgmlCtx,
        gguf: &GgufFile,
        budget: usize,
    ) -> Result<LoadReport, GgufError> {
        // 前置检查：必须已经调用过 BackendBuffer::from_ctx
        if let Some((first, _)) = self.entries.first() {
            if !first.is_null() && !first.get(ctx).has_buffer() {
                return Err(GgufError::LoadFailed(
                    "张量尚未分配 backend buffer：请先调用 BackendBuffer::from_ctx(&ctx, kind)".into(),
                ));
            }
        }

        let mut reader = GgufReader::new(gguf)?;
        let start = Instant::now();
        let mut tensors = 0usize;
        let mut bytes = 0usize;

        for (r, id) in &self.entries {
            let size = gguf.tensor_size(*id);
            if tensors > 0 && bytes + size > budget {
                break;
            }
            reader.load_into(&r.get(ctx), *id)?;
            bytes += size;
            tensors += 1;
        }

        Ok(LoadReport { tensors, bytes, elapsed_ms: start.elapsed().as_millis() })
    }
}

// ---------------------------------------------------------------------------
// 张量工厂：按 GGUF 元数据创建张量并登记
// ---------------------------------------------------------------------------
struct TensorFactory<'a> {
    ctx: &'a GgmlCtx,
    gguf: &'a GgufFile,
    entries: Vec<(TensorRef, i64)>,
    names: Vec<String>,
    total_bytes: usize,
}

impl<'a> TensorFactory<'a> {
    fn new(ctx: &'a GgmlCtx, gguf: &'a GgufFile) -> Self {
        Self { ctx, gguf, entries: Vec::new(), names: Vec::new(), total_bytes: 0 }
    }

    /// 按名字创建张量；`required=false` 时缺失返回 `TensorRef::NULL`。
    fn make(&mut self, name: &str, required: bool) -> Result<TensorRef, GgufError> {
        // 先把引用取出来，避免后续 &mut self 与 Tensor 的生命周期纠缠
        let ctx = self.ctx;
        let gguf = self.gguf;

        let Some(id) = gguf.find_tensor(name) else {
            if required {
                return Err(GgufError::TensorNotFound(name.to_string()));
            }
            return Ok(TensorRef::NULL);
        };

        let ne = gguf.tensor_ne(id);
        let ty_raw = gguf.tensor_type(id);
        let ty = GgmlType::from_i32(ty_raw).ok_or_else(|| {
            GgufError::LoadFailed(format!("{name}: 不支持的 ggml_type = {ty_raw}"))
        })?;

        let n_dims = ne_dims(&ne);
        let t = ctx.new_tensor(ty, &ne[..n_dims]);

        // 尺寸自检：ggml 侧算出的字节数必须与 GGUF 记录一致
        let expect = gguf.tensor_size(id);
        let got = t.nbytes();
        if got != expect {
            return Err(GgufError::LoadFailed(format!(
                "{name}: 字节数不符（ggml 计算 {got}，GGUF 记录 {expect}）；shape={:?} type={ty:?}",
                &ne[..n_dims]
            )));
        }

        let r = TensorRef(t.ptr());
        self.entries.push((r, id));
        self.names.push(name.to_string());
        self.total_bytes += got;
        Ok(r)
    }

    fn into_parts(self) -> (Vec<(TensorRef, i64)>, Vec<String>, usize) {
        (self.entries, self.names, self.total_bytes)
    }
}

/// 由 ne 数组推断维度数（ggml 约定：未使用的维度为 1）。
fn ne_dims(ne: &[i64; 4]) -> usize {
    let mut n = 1usize;
    for i in 0..4 {
        if ne[i] != 1 {
            n = i + 1;
        }
    }
    n
}

// ---------------------------------------------------------------------------
// GGUF 数据区读取器：单文件句柄 + 分块拷贝
// ---------------------------------------------------------------------------
pub struct GgufReader<'a> {
    gguf: &'a GgufFile,
    file: File,
    chunk: Vec<u8>,
}

impl<'a> GgufReader<'a> {
    pub const DEFAULT_CHUNK: usize = 8 << 20; // 8 MiB

    pub fn new(gguf: &'a GgufFile) -> Result<Self, GgufError> {
        let file = File::open(gguf.path())
            .map_err(|e| GgufError::LoadFailed(format!("打开 {} 失败: {e}", gguf.path())))?;
        Ok(Self { gguf, file, chunk: vec![0u8; Self::DEFAULT_CHUNK] })
    }

    /// 把 GGUF 中 `tensor_id` 的数据分块写入目标张量，返回写入字节数。
    pub fn load_into(&mut self, tensor: &Tensor<'_>, tensor_id: i64) -> Result<usize, GgufError> {
        let offset = self.gguf.tensor_offset(tensor_id);
        let size = self.gguf.tensor_size(tensor_id);

        self.file
            .seek(SeekFrom::Start(offset as u64))
            .map_err(|e| GgufError::LoadFailed(format!("seek {offset} 失败: {e}")))?;

        let mut done = 0usize;
        while done < size {
            let n = self.chunk.len().min(size - done);
            self.file
                .read_exact(&mut self.chunk[..n])
                .map_err(|e| GgufError::LoadFailed(format!("读取张量 {tensor_id} @{done} 失败: {e}")))?;
            tensor.write_bytes_at(done, &self.chunk[..n]);
            done += n;
        }
        Ok(size)
    }

    /// 把数据读进一个新建的 `Vec<u8>`（测试用：逐字节比对文件内容）。
    pub fn read_raw(&mut self, tensor_id: i64) -> Result<Vec<u8>, GgufError> {
        let offset = self.gguf.tensor_offset(tensor_id);
        let size = self.gguf.tensor_size(tensor_id);
        self.file
            .seek(SeekFrom::Start(offset as u64))
            .map_err(|e| GgufError::LoadFailed(format!("seek {offset} 失败: {e}")))?;
        let mut out = vec![0u8; size];
        self.file
            .read_exact(&mut out)
            .map_err(|e| GgufError::LoadFailed(format!("读取张量 {tensor_id} 失败: {e}")))?;
        Ok(out)
    }
}

/// 便捷入口：单张量加载（内部使用 `GgufReader`，会重新打开文件）。
pub fn load_gguf_tensor(
    tensor: &Tensor<'_>,
    gguf: &GgufFile,
    tensor_id: i64,
) -> Result<usize, GgufError> {
    let mut reader = GgufReader::new(gguf)?;
    reader.load_into(tensor, tensor_id)
}

// ---------------------------------------------------------------------------
// Model — 拥有 ggml context、backend buffer 与全部权重的完整模型
// ---------------------------------------------------------------------------
pub struct Model {
    // 字段顺序 = 析构顺序：先释放 backend buffer，再释放 ggml context（对齐 llama.cpp）
    buffer: BackendBuffer,
    ctx: GgmlCtx,
    pub arch: Qwen35MoeArch,
    pub weights: ModelWeights,
    path: String,
    pub load_report: LoadReport,
}

impl Model {
    /// 从 GGUF 文件完整加载模型到指定后端。
    pub fn open(path: &str, kind: BackendKind) -> Result<Self, GgufError> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf, kind, path)
    }

    /// 复用已打开的 `GgufFile` 加载模型。
    pub fn from_gguf(gguf: &GgufFile, kind: BackendKind, path: &str) -> Result<Self, GgufError> {
        let arch = Qwen35MoeArch::from_gguf(gguf)?;

        // 阶段 1：创建张量（no_alloc）
        let ctx = GgmlCtx::new_no_alloc().map_err(|e| GgufError::LoadFailed(e.to_string()))?;
        let weights = ModelWeights::create(&ctx, gguf, &arch)?;

        // 阶段 2：一次性在后端分配全部张量内存
        let buffer = BackendBuffer::from_ctx(&ctx, kind).map_err(|e| GgufError::LoadFailed(e.to_string()))?;

        // 阶段 3：流式读入权重数据
        let report = weights.load_data(&ctx, gguf)?;

        Ok(Model {
            buffer,
            ctx,
            arch,
            weights,
            path: path.to_string(),
            load_report: report,
        })
    }

    pub fn ctx(&self) -> &GgmlCtx {
        &self.ctx
    }

    pub fn buffer(&self) -> &BackendBuffer {
        &self.buffer
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    /// 取某个主干层的规范张量引用。
    pub fn layer(&self, il: usize) -> &LayerWeights {
        &self.weights.layers[il]
    }
}
