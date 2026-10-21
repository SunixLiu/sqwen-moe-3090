// src/backend/ctx.rs — 安全封装：GgmlCtx / BackendBuffer / Tensor / GraphBuilder
//
// 对应 plan §核心接口约定。所有 unsafe 操作都有明确注释。

use super::ffi;
use std::ptr;

// ---------------------------------------------------------------------------
// GgmlCtx — 拥有 ggml_context*，负责分配/释放内存池
// ---------------------------------------------------------------------------
pub struct GgmlCtx {
    ptr: *mut ffi::GgmlContext,
}

impl GgmlCtx {
    /// 创建新的 ggml 上下文（内部分配内存池）。
    pub fn new() -> Result<Self, &'static str> {
        let params = ffi::GgmlInitParams {
            mem_size: 1 << 30,          // 1 GB default pool
            mem_buffer: ptr::null_mut(),
            no_alloc: false,
        };
        let ctx = unsafe { ffi::ggml_init(params) };
        if ctx.is_null() {
            return Err("ggml_init returned null");
        }
        Ok(Self { ptr: ctx })
    }

    /// 创建 no_alloc 上下文：只分配 tensor 元数据，数据指针留空，
    /// 由 `BackendBuffer::from_ctx`（ggml_backend_alloc_ctx_tensors）在
    /// 指定后端上统一分配。这是 llama.cpp 模型加载的标准流程。
    pub fn new_no_alloc() -> Result<Self, &'static str> {
        let params = ffi::GgmlInitParams {
            mem_size: 1 << 26,          // 64 MB metadata pool
            mem_buffer: ptr::null_mut(),
            no_alloc: true,
        };
        let ctx = unsafe { ffi::ggml_init(params) };
        if ctx.is_null() {
            return Err("ggml_init(no_alloc) returned null");
        }
        Ok(Self { ptr: ctx })
    }

    /// 底层 context 指针（供需要直接调用 ggml_* 的封装使用）。
    pub fn as_ptr(&self) -> *mut ffi::GgmlContext { self.ptr }

    /// 清空对象表，让下一次分配复用同样的地址（arena 不释放）。
    ///
    /// **调用后本 ctx 里此前创建的所有 `Tensor`/`StateTensor` 都失效**（内存会被复用），
    /// 只应在"每步重建图"的场景里、且旧图已经不再被引用时调用。
    ///
    /// 为什么需要（见 plans/2026-10-08-perf-and-parity.md 的 P2）：
    /// ggml-cuda 用 `cgraph->nodes[0]` 作为 CUDA graph 的 key，并逐字段比较节点属性
    /// （含每个节点的 `data` 指针）。每步新建 ctx 会让地址漂移 → 每步都是"新图" →
    /// 永远进不了重放。复用同一个 ctx + reset 后地址确定，warmup 一次即可持续重放。
    pub fn reset(&self) {
        unsafe { ffi::ggml_reset(self.ptr) }
    }

    /// 创建指定类型和形状的 F32 tensor（在 context 的内存池中分配）。
    pub fn new_tensor_f32(&self, ne: &[i64]) -> Tensor<'_> {
        assert!(ne.len() <= 4, "max 4 dims");
        let n_dims = ne.len() as i32;
        unsafe {
            let t = ffi::ggml_new_tensor(self.ptr, ffi::GgmlType::F32 as i32, n_dims, ne.as_ptr());
            debug_assert!(!t.is_null(), "ggml_new_tensor returned null");
            Tensor { ptr: t, _ctx: self }
        }
    }

    /// 创建指定 ggml_type 和形状的 tensor。
    pub fn new_tensor(&self, ty: ffi::GgmlType, ne: &[i64]) -> Tensor<'_> {
        assert!(ne.len() <= 4, "max 4 dims");
        let n_dims = ne.len() as i32;
        unsafe {
            let t = ffi::ggml_new_tensor(self.ptr, ty as i32, n_dims, ne.as_ptr());
            debug_assert!(!t.is_null(), "ggml_new_tensor returned null");
            Tensor { ptr: t, _ctx: self }
        }
    }

    /// 创建 I32 tensor（索引 / 位置张量）。
    pub fn new_tensor_i32(&self, ne: &[i64]) -> Tensor<'_> {
        self.new_tensor(ffi::GgmlType::I32, ne)
    }

    /// 创建 1D I32 tensor。
    pub fn new_tensor_1d_i32(&self, ne0: i64) -> Tensor<'_> {
        self.new_tensor_i32(&[ne0])
    }

    /// 创建指定量化/浮点类型的 2D tensor（ne0 = 行内列数, ne1 = 行数）。
    pub fn new_tensor_2d(&self, ty: ffi::GgmlType, ne0: i64, ne1: i64) -> Tensor<'_> {
        unsafe {
            let t = ffi::ggml_new_tensor_2d(self.ptr, ty as i32, ne0, ne1);
            debug_assert!(!t.is_null(), "ggml_new_tensor_2d returned null");
            Tensor { ptr: t, _ctx: self }
        }
    }

    /// 创建 3D tensor。
    pub fn new_tensor_3d(&self, ty: ffi::GgmlType, ne0: i64, ne1: i64, ne2: i64) -> Tensor<'_> {
        unsafe {
            let t = ffi::ggml_new_tensor_3d(self.ptr, ty as i32, ne0, ne1, ne2);
            debug_assert!(!t.is_null(), "ggml_new_tensor_3d returned null");
            Tensor { ptr: t, _ctx: self }
        }
    }

    /// 创建一个 view（不复制数据），用于切片/reshape。
    pub fn view_3d(&self, src: &Tensor<'_>, ne0: i64, ne1: i64, ne2: i64, nb1: u64, nb2: u64, offset: u64) -> Tensor<'_> {
        unsafe {
            let t = ffi::ggml_view_3d(
                self.ptr, src.ptr, ne0, ne1, ne2, nb1, nb2, offset,
            );
            debug_assert!(!t.is_null(), "ggml_view_3d returned null");
            Tensor { ptr: t, _ctx: self }
        }
    }

    /// 分配一块原始内存（用于 KV cache 等）。
    pub fn new_buffer(&self, nbytes: usize) -> *mut std::ffi::c_void {
        unsafe { ffi::ggml_new_buffer(self.ptr, nbytes) }
    }

    /// 把一个裸 `ggml_tensor*`（例如 `ggml_mul_mat` 等 op 的返回值）
    /// 包装成 `Tensor`，生命周期绑定到本 context。
    pub fn wrap(&self, t: *mut ffi::ggml_tensor) -> Tensor<'_> {
        assert!(!t.is_null(), "wrap: tensor pointer is null");
        Tensor { ptr: t, _ctx: self }
    }

    /// 只在**需要时**做 `ggml_cont`。
    ///
    /// `ggml_cont` 走 `ggml_dup`：即使源张量已经连续，也会**真的发一次拷贝 kernel**
    /// （只有 VIEW/RESHAPE/PERMUTE 才是零成本元数据）。decode 每 token 约 7.5 ms，
    /// 几十个空拷贝就是几十微秒起。连续时原样返回（零节点）。
    ///
    /// ⚠️ **只用在「常驻权重 / 本就连续的算子输出」上**。对**视图**（融合投影切出来的
    /// Q/K/V 等）改用它会改变 ggml-cuda 的 kernel 选择，进而让投机路径的末位数值变化
    /// （实测会让「spec 与 nospec 的 greedy 输出完全一致」这条不变式失效）。
    pub fn cont_if_needed<'a>(&self, t: &Tensor<'a>) -> Tensor<'a> {
        if unsafe { ffi::ggml_is_contiguous(t.ptr) } {
            return Tensor { ptr: t.ptr, _ctx: t._ctx };
        }
        let r = unsafe { ffi::ggml_cont(self.ptr, t.ptr) };
        assert!(!r.is_null(), "ggml_cont returned null");
        Tensor { ptr: r, _ctx: t._ctx }
    }
}

impl Drop for GgmlCtx {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::ggml_free(self.ptr) };
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor — 不拥有数据，生命周期绑定到 GgmlCtx / BackendBuffer
// ---------------------------------------------------------------------------
pub struct Tensor<'ctx> {
    ptr: *mut ffi::ggml_tensor,
    _ctx: &'ctx GgmlCtx,  // lifetime anchor; not read directly
}

impl<'ctx> Tensor<'ctx> {
    pub fn ptr(&self) -> *mut ffi::ggml_tensor { self.ptr }

    /// 获取 ne[0..n_dims]（只读）。
    pub fn ne(&self) -> [i64; 4] {
        unsafe { std::ptr::read(&(*self.ptr).ne) }
    }

    /// 获取 nb[0..n_dims]（字节步长，只读）。view 张量可能不连续。
    pub fn nb(&self) -> [u64; 4] {
        unsafe { std::ptr::read(&(*self.ptr).nb) }
    }

    /// 获取 type 字段。
    pub fn r#type(&self) -> i32 {
        unsafe { std::ptr::read(&(*self.ptr).r#type) }
    }

    /// 数据字节数（ggml_nbytes，量化类型也正确）。
    pub fn nbytes(&self) -> usize {
        unsafe { ffi::ggml_nbytes(self.ptr) }
    }

    /// 设置 op 字段（用于构建计算图）。
    pub fn set_op(&mut self, op: ffi::GgmlOp) {
        unsafe { std::ptr::write(&mut (*self.ptr).op, op as i32) };
    }

    /// 设置 src[0]（第一个输入源）。
    pub fn set_src0(&mut self, src: &Tensor<'ctx>) {
        unsafe { std::ptr::write(&mut (*self.ptr).src[0], src.ptr) };
    }

    /// 设置 src[1]（第二个输入源）。
    pub fn set_src1(&mut self, src: &Tensor<'ctx>) {
        unsafe { std::ptr::write(&mut (*self.ptr).src[1], src.ptr) };
    }

    /// 获取 op_params 的 mutable slice。
    pub fn op_params_mut(&mut self) -> &mut [i32; 16] {
        unsafe { &mut (*self.ptr).op_params }
    }

    /// 将 f32 数据拷贝到 tensor（通过 backend buffer）。
    pub fn set_data_f32(&self, data: &[f32]) {
        let size = data.len() * std::mem::size_of::<f32>();
        unsafe {
            ffi::ggml_backend_tensor_set(
                self.ptr,
                data.as_ptr() as *const std::ffi::c_void,
                0,
                size,
            );
        }
    }

    /// 将任意字节数据拷贝到 tensor（量化权重 / i32 索引等）。
    pub fn set_data_bytes(&self, data: &[u8]) {
        self.write_bytes_at(0, data);
    }

    /// 从指定字节偏移写入一段数据（分块加载大张量时使用）。
    ///
    /// 有 backend buffer 时走 `ggml_backend_tensor_set`（支持 CUDA 显存）；
    /// 否则直接写 ggml context 内存池中的主机指针（`no_alloc=false` 的 ctx）。
    pub fn write_bytes_at(&self, offset: usize, data: &[u8]) {
        unsafe { write_ptr_bytes(self.ptr, offset, data) }
    }

    /// 从指定字节偏移读取一段数据（对应 `write_bytes_at`）。
    pub fn read_bytes_at(&self, offset: usize, len: usize) -> Vec<u8> {
        unsafe { read_ptr_bytes(self.ptr, offset, len) }
    }

    /// 从指定字节偏移直接读成 `Vec<f32>`（见 [`StateTensor::read_f32_at`] 的说明）。
    pub fn read_f32_at(&self, offset: usize, n: usize) -> Vec<f32> {
        let mut out: Vec<f32> = vec![0.0; n];
        unsafe { read_ptr_into(self.ptr, offset, out.as_mut_ptr() as *mut u8, n * 4) };
        out
    }

    /// tensor 是否已被分配到某个 backend buffer（`BackendBuffer::from_ctx` 之后为真）。
    pub fn has_buffer(&self) -> bool {
        unsafe { !(*self.ptr).buffer.is_null() }
    }

    /// 将 i32 数据拷贝到 tensor（I32 类型）。
    pub fn set_data_i32(&self, data: &[i32]) {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        self.set_data_bytes(bytes);
    }

    /// 读取 tensor 的原始字节（长度由调用方按类型给出）。
    pub fn get_data_bytes(&self, len: usize) -> Vec<u8> {
        self.read_bytes_at(0, len)
    }

    /// 从指定字节偏移读取一段数据（用于非连续 view，按 nb 步长逐行取）。
    pub fn get_data_bytes_at(&self, offset: usize, len: usize) -> Vec<u8> {
        self.read_bytes_at(offset, len)
    }

    /// 读取 I32 tensor 的数据。
    pub fn get_data_i32(&self) -> Vec<i32> {
        let ne = self.ne();
        let n = ne.iter().filter(|&&x| x > 0).product::<i64>() as usize;
        let bytes = self.get_data_bytes(n * 4);
        let mut out = Vec::with_capacity(n);
        for c in bytes.chunks_exact(4) {
            out.push(i32::from_ne_bytes([c[0], c[1], c[2], c[3]]));
        }
        out
    }

    /// 从 tensor 读取 f32 数据。
    pub fn get_data_f32(&self) -> Vec<f32> {
        let ne = self.ne();
        let n = ne.iter().filter(|&&x| x > 0).product::<i64>() as usize;
        let mut out: Vec<f32> = Vec::with_capacity(n);
        unsafe {
            ffi::ggml_backend_tensor_get(
                self.ptr,
                out.as_mut_ptr() as *mut std::ffi::c_void,
                0,
                n * std::mem::size_of::<f32>(),
            );
        }
        unsafe { out.set_len(n) };
        out
    }

    /// 设置 name。
    pub fn set_name(&mut self, name: &str) {
        unsafe {
            let dst = &mut (*self.ptr).name;
            std::ptr::write_bytes(dst.as_mut_ptr(), 0, dst.len());
            for (i, c) in name.bytes().enumerate() {
                if i < dst.len() - 1 {
                    dst[i] = c as std::ffi::c_char;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// StateTensor — 不借用 ctx 的张量句柄（给跨步存活的状态容器用）
// ---------------------------------------------------------------------------

/// 与 [`Tensor`] 指向同一个 `ggml_tensor`，但**不携带 ctx 生命周期**。
///
/// 存在的理由：`EngineState`（GDN 递推状态 + KV cache + 每步图输入）必须能被
/// `Engine` 持有，而 `Engine` 同时拥有 `GgmlCtx` —— 用借用版 `Tensor<'a>` 会构成
/// 自引用结构，Rust 表达不了。所以这里把「谁活得更久」从类型系统挪进不变式：
///
/// **不变式：`GgmlCtx` 以及张量所在的后端 buffer，必须比 `StateTensor` 活得久。**
///
/// 解引用要换回带生命周期的 `Tensor`，借用期就是安全期：
///
/// ```ignore
/// let t = st.state.get(ctx);            // 与 ctx 绑定，之后按 Tensor 用
/// st.state = StateTensor::new(new_t);   // 存回去时不带生命周期
/// ```
///
/// 图算子的输入输出仍然全部走 `Tensor<'a>`，生命周期检查只在状态容器这一处放宽。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateTensor {
    ptr: *mut ffi::ggml_tensor,
}

impl StateTensor {
    /// 从带生命周期的 `Tensor` 造一个（`ctx` 必须比它活得久）。
    pub fn new(t: Tensor<'_>) -> Self {
        Self { ptr: t.ptr() }
    }

    /// 换回借用 ctx 的 `Tensor`。
    pub fn get<'a>(&self, ctx: &'a GgmlCtx) -> Tensor<'a> {
        ctx.wrap(self.ptr)
    }

    pub fn ptr(&self) -> *mut ffi::ggml_tensor {
        self.ptr
    }

    /// 形状 `ne[0..4]`。
    pub fn ne(&self) -> [i64; 4] {
        unsafe { std::ptr::read(&(*self.ptr).ne) }
    }

    /// ggml 类型（`ggml_type` 枚举值）。
    pub fn r#type(&self) -> i32 {
        unsafe { (*self.ptr).r#type }
    }

    /// 数据字节数（量化类型也正确）。
    pub fn nbytes(&self) -> usize {
        unsafe { ffi::ggml_nbytes(self.ptr) }
    }

    /// 这个张量是否真的拿到了内存（backend buffer 或主机 `data` 指针）。
    ///
    /// 为什么需要：图里**没有任何节点引用**的张量不会被 gallocr 分配。正常图里
    /// 每个输入都会被用到，但 `SQWEN_ABLATE` 把某个模块整个跳过时（例如跳过注意力），
    /// mask / kv_idxs 就没人用了 → 未分配 → 填值时不能硬写。
    pub fn is_allocated(&self) -> bool {
        unsafe { !(*self.ptr).buffer.is_null() || !(*self.ptr).data.is_null() }
    }

    /// 写数据（分配之后填图输入用）。不需要 ctx。
    pub fn write_bytes_at(&self, offset: usize, data: &[u8]) {
        unsafe { write_ptr_bytes(self.ptr, offset, data) }
    }

    /// 读数据（跨步复用图时回读 logits 用）。不需要 ctx。
    pub fn read_bytes_at(&self, offset: usize, len: usize) -> Vec<u8> {
        unsafe { read_ptr_bytes(self.ptr, offset, len) }
    }

    /// 直接读成 `Vec<f32>`（decode 回读 logits 用）。
    ///
    /// 与 `read_bytes_at` + 逐 4 字节重组的区别：**D2H 的字节直接落到 `Vec<f32>` 的缓冲里**，
    /// 省掉一次同样大小的临时 `Vec<u8>` 和一次逐元素的 `from_ne_bytes`（投机校验批 n=4 时
    /// 是 4 MB + 103 万个元素，每轮都要付一次）。
    pub fn read_f32_at(&self, offset: usize, n: usize) -> Vec<f32> {
        let mut out: Vec<f32> = vec![0.0; n];
        unsafe { read_ptr_into(self.ptr, offset, out.as_mut_ptr() as *mut u8, n * 4) };
        out
    }
}

/// 往任意 `ggml_tensor*` 写字节：有 backend buffer 走 `tensor_set`，否则直接写 `data`。
///
/// 单独抽出来是因为它只需要裸指针，不需要 ctx（`Tensor` 与 [`StateTensor`] 共用）。
unsafe fn write_ptr_bytes(ptr: *mut ffi::ggml_tensor, offset: usize, data: &[u8]) {
    if !(*ptr).buffer.is_null() {
        ffi::ggml_backend_tensor_set(
            ptr,
            data.as_ptr() as *const std::ffi::c_void,
            offset,
            data.len(),
        );
    } else {
        let base = (*ptr).data;
        assert!(
            !base.is_null(),
            "tensor 既没有 backend buffer 也没有 data 指针（name={}）",
            unsafe { ffi::c_str_to_str((*ptr).name.as_ptr()) }
        );
        std::ptr::copy_nonoverlapping(data.as_ptr(), base.add(offset), data.len());
    }
}

/// 从任意 `ggml_tensor*` 读字节（同样不需要 ctx）。
unsafe fn read_ptr_bytes(ptr: *mut ffi::ggml_tensor, offset: usize, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    read_ptr_into(ptr, offset, out.as_mut_ptr(), len);
    out
}

/// 把一个 tensor 的一段字节读到**调用方给的缓冲**里（不额外分配）。
///
/// 有 backend buffer 时走 `ggml_backend_tensor_get`（CUDA 的同步 D2H），
/// 否则直接从 `data` 拷。`read_ptr_bytes` 与 `read_f32_at` 共用这一条路径。
unsafe fn read_ptr_into(ptr: *mut ffi::ggml_tensor, offset: usize, dst: *mut u8, len: usize) {
    if !(*ptr).buffer.is_null() {
        ffi::ggml_backend_tensor_get(ptr, dst as *mut std::ffi::c_void, offset, len);
    } else {
        let base = (*ptr).data;
        assert!(!base.is_null(), "tensor 既没有 backend buffer 也没有 data 指针");
        std::ptr::copy_nonoverlapping(base.add(offset), dst, len);
    }
}

// ---------------------------------------------------------------------------
// BackendBuffer — 拥有 ggml_backend_buffer*，负责分配/释放后端缓冲区
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Cuda(i32),  // device index
}

pub struct BackendBuffer {
    backend: *mut ffi::GgmlBackend,
    buffer: *mut ffi::GgmlBackendBuffer,
}

impl BackendBuffer {
    /// 创建 CPU / CUDA 后端，并按 size 分配一块空缓冲区。
    pub fn new(kind: BackendKind, size: usize) -> Result<Self, &'static str> {
        let backend = match kind {
            BackendKind::Cpu => unsafe { ffi::ggml_backend_cpu_init() },
            BackendKind::Cuda(device) => unsafe { ffi::ggml_backend_cuda_init(device) },
        };
        if backend.is_null() {
            return Err("backend init returned null");
        }
        let buffer = unsafe { ffi::ggml_backend_alloc_buffer(backend, size) };
        if buffer.is_null() {
            unsafe { ffi::ggml_backend_free(backend) };
            return Err("alloc_buffer returned null");
        }
        Ok(Self { backend, buffer })
    }

    /// 在给定后端上，为 no_alloc 上下文中的所有 tensor 分配内存。
    ///
    /// 对应 llama.cpp 的 `ggml_backend_alloc_ctx_tensors`：要求 `ctx` 由
    /// `GgmlCtx::new_no_alloc()` 创建（tensor->data == NULL），分配后
    /// 每个 tensor 的 `buffer`/`data` 都被设置好，可直接 compute。
    pub fn from_ctx(ctx: &GgmlCtx, kind: BackendKind) -> Result<Self, &'static str> {
        let backend = match kind {
            BackendKind::Cpu => unsafe { ffi::ggml_backend_cpu_init() },
            BackendKind::Cuda(device) => unsafe { ffi::ggml_backend_cuda_init(device) },
        };
        if backend.is_null() {
            return Err("backend init returned null");
        }
        let buffer = unsafe { ffi::ggml_backend_alloc_ctx_tensors(ctx.as_ptr(), backend) };
        if buffer.is_null() {
            unsafe { ffi::ggml_backend_free(backend) };
            return Err("ggml_backend_alloc_ctx_tensors returned null");
        }
        Ok(Self { backend, buffer })
    }

    /// 底层 backend 指针（用于 ggml_backend_graph_compute）。
    pub fn backend(&self) -> *mut ffi::GgmlBackend { self.backend }

    pub fn raw_buffer(&self) -> *mut ffi::GgmlBackendBuffer { self.buffer }

    /// 在 buffer 中为 tensor 分配/初始化内存（设置 tensor.buffer 指针）。
    pub fn init_tensor(&self, tensor: &Tensor<'_>) -> Result<(), &'static str> {
        let status = unsafe { ffi::ggml_backend_buffer_init_tensor(self.buffer, tensor.ptr) };
        if status == 0 { Ok(()) } else { Err("buffer_init_tensor failed") }
    }

    /// 将数据写入 tensor。
    ///
    /// - tensor 属于某个 backend buffer（CUDA 显存 / backend 管理的主机内存）→
    ///   走 `ggml_backend_tensor_set`；
    /// - tensor 数据由 ggml context 内存池直接分配 → 直接写 `tensor->data`。
    pub fn set_tensor_data(&self, tensor: &Tensor<'_>, data: &[u8]) {
        unsafe {
            let buffer = (*tensor.ptr).buffer;
            if !buffer.is_null() {
                ffi::ggml_backend_tensor_set(
                    tensor.ptr,
                    data.as_ptr() as *const std::ffi::c_void,
                    0,
                    data.len(),
                );
                return;
            }
            let data_ptr = (*tensor.ptr).data;
            assert!(!data_ptr.is_null(), "tensor data pointer is null");
            std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, data.len());
        }
    }

    /// 读取 tensor 的全部字节（长度取自 ggml_nbytes，量化类型也正确）。
    pub fn get_tensor_data_raw(&self, tensor: &Tensor<'_>) -> Vec<u8> {
        unsafe {
            let size = ffi::ggml_nbytes(tensor.ptr);
            let mut out = vec![0u8; size];
            let buffer = (*tensor.ptr).buffer;
            if !buffer.is_null() {
                ffi::ggml_backend_tensor_get(
                    tensor.ptr,
                    out.as_mut_ptr() as *mut std::ffi::c_void,
                    0,
                    size,
                );
                return out;
            }
            let data_ptr = (*tensor.ptr).data;
            assert!(!data_ptr.is_null(), "tensor data pointer is null");
            std::ptr::copy_nonoverlapping(data_ptr, out.as_mut_ptr(), size);
            out
        }
    }

    /// 从 tensor 获取数据（等价于 `get_tensor_data_raw`）。
    pub fn get_tensor_data(&self, tensor: &Tensor<'_>) -> Vec<u8> {
        self.get_tensor_data_raw(tensor)
    }
}

impl Drop for BackendBuffer {
    fn drop(&mut self) {
        if !self.buffer.is_null() {
            unsafe { ffi::ggml_backend_buffer_free(self.buffer) };
        }
        if !self.backend.is_null() {
            unsafe { ffi::ggml_backend_free(self.backend) };
        }
    }
}

// ---------------------------------------------------------------------------
// GraphAllocator — 跨步复用同一块 buffer 的图分配器（ggml_gallocr）
// ---------------------------------------------------------------------------

/// 用 `ggml_gallocr` 给「每步新建的图」分配张量。
///
/// 为什么需要它：只用 `BackendBuffer::from_ctx` 的话每步都要新建后端 + 分配一块
/// buffer，于是**张量地址每步都不一样**。ggml-cuda 的部分 kernel 会按地址对齐选择
/// 向量化路径，累加顺序随之改变 —— 实测同一个 engine 连续三次 generate 的 prefill
/// logits 之和是 -678261.9 / -711227.4 / -702538.1（新建 engine 则永远复现第一个值）。
/// 复用同一块 buffer 后地址稳定，结果可复现；顺带省掉每步新建后端的开销，并且
/// ctx 里的图元数据可以随步释放。
pub struct GraphAllocator {
    backend: *mut ffi::GgmlBackend,
    gallocr: *mut ffi::GgmlGallocr,
}

impl GraphAllocator {
    pub fn new(kind: BackendKind) -> Result<Self, &'static str> {
        let backend = unsafe {
            match kind {
                BackendKind::Cpu => ffi::ggml_backend_cpu_init(),
                BackendKind::Cuda(device) => ffi::ggml_backend_cuda_init(device),
            }
        };
        if backend.is_null() {
            return Err("backend init returned null");
        }
        let buft = unsafe { ffi::ggml_backend_get_default_buffer_type(backend) };
        if buft.is_null() {
            unsafe { ffi::ggml_backend_free(backend) };
            return Err("ggml_backend_get_default_buffer_type returned null");
        }
        let gallocr = unsafe { ffi::ggml_gallocr_new(buft) };
        if gallocr.is_null() {
            unsafe { ffi::ggml_backend_free(backend) };
            return Err("ggml_gallocr_new returned null");
        }
        Ok(Self { backend, gallocr })
    }

    /// 为图里的张量分配 buffer（尽量复用之前那块）。
    pub fn alloc(&self, g: &GraphBuilder<'_>) -> Result<(), &'static str> {
        let ok = unsafe { ffi::ggml_gallocr_alloc_graph(self.gallocr, g.raw()) };
        if ok {
            Ok(())
        } else {
            Err("ggml_gallocr_alloc_graph failed")
        }
    }

    /// 在本分配器的后端上执行图。
    pub fn compute(&self, g: &GraphBuilder<'_>) -> Result<(), &'static str> {
        self.compute_graph(g.as_raw())
    }

    /// 对不带 ctx 生命周期的图句柄执行（P2c：跨步复用的图）。
    pub fn compute_graph(&self, g: RawGraph) -> Result<(), &'static str> {
        let status = unsafe { ffi::ggml_backend_graph_compute(self.backend, g.raw()) };
        if status == 0 {
            Ok(())
        } else {
            Err("ggml_backend_graph_compute failed")
        }
    }

    pub fn backend(&self) -> *mut ffi::GgmlBackend {
        self.backend
    }
}

impl Drop for GraphAllocator {
    fn drop(&mut self) {
        unsafe {
            ffi::ggml_gallocr_free(self.gallocr);
            ffi::ggml_backend_free(self.backend);
        }
    }
}

// ---------------------------------------------------------------------------
// GraphBuilder — 拥有 ggml_cgraph*，负责构建和计算前向图
// ---------------------------------------------------------------------------
pub struct GraphBuilder<'ctx> {
    ptr: *mut ffi::GgmlCgraph,
    _ctx: &'ctx GgmlCtx,
}

impl<'ctx> GraphBuilder<'ctx> {
    pub fn new(ctx: &'ctx GgmlCtx) -> Self {
        let ptr = unsafe { ffi::ggml_new_graph(ctx.ptr) };
        debug_assert!(!ptr.is_null(), "ggml_new_graph returned null");
        Self { ptr, _ctx: ctx }
    }

    /// 以指定节点上限建图。
    ///
    /// `ggml_new_graph` 用 `GGML_DEFAULT_GRAPH_SIZE`（2048）；整模型 40 层、每层
    /// 几十个节点会超出这个上限，因此引擎侧要显式给一个足够大的值。
    pub fn new_with_size(ctx: &'ctx GgmlCtx, size: usize) -> Self {
        let ptr = unsafe { ffi::ggml_new_graph_custom(ctx.ptr, size, false) };
        assert!(!ptr.is_null(), "ggml_new_graph_custom returned null");
        Self { ptr, _ctx: ctx }
    }

    /// 将 tensor 及其依赖加入前向图。
    pub fn expand(&mut self, tensor: &Tensor<'_>) {
        unsafe { ffi::ggml_build_forward_expand(self.ptr, tensor.ptr) };
    }

    /// 清空图（保留节点但清除计算状态）。
    pub fn clear(&mut self) {
        unsafe { ffi::ggml_graph_clear(self.ptr) };
    }

    /// 在给定 backend（由 `BackendBuffer` 持有）上执行计算。
    ///
    /// `ggml_backend_graph_compute` 返回 `enum ggml_status`：
    /// 0 = GGML_STATUS_SUCCESS，非 0 表示失败。
    pub fn compute(&self, buf: &BackendBuffer) -> Result<(), &'static str> {
        let status = unsafe { ffi::ggml_backend_graph_compute(buf.backend(), self.ptr) };
        if status == 0 {
            Ok(())
        } else {
            Err("ggml_backend_graph_compute failed")
        }
    }

    /// 使用裸 backend 指针执行计算（兼容底层用法）。
    pub fn compute_raw(&self, backend: *mut ffi::GgmlBackend) -> i32 {
        unsafe { ffi::ggml_backend_graph_compute(backend, self.ptr) }
    }

    /// 裸 cgraph 指针（给 gallocr / 后端计算用）。
    pub fn raw(&self) -> *mut ffi::GgmlCgraph {
        self.ptr
    }

    /// 交出不带 ctx 生命周期的图句柄（P2c 的跨步复用）。
    ///
    /// 交出后仍可交给 [`GraphAllocator::compute_graph`] 执行，但调用方必须保证该 ctx
    /// **不被 `reset()`** —— 图节点分配在 ctx 的 arena 里，reset 会把那块内存交还给
    /// 后续分配。图对象本身仍归 ctx 所有（`Drop` 是空的）。
    pub fn as_raw(&self) -> RawGraph {
        RawGraph { ptr: self.ptr }
    }

    /// 从图中获取 named tensor。
    pub fn get_tensor(&self, name: &str) -> Option<Tensor<'ctx>> {
        let c_name = std::ffi::CString::new(name).ok()?;
        let t = unsafe { ffi::ggml_graph_get_tensor(self.ptr, c_name.as_ptr()) };
        if t.is_null() { None } else { Some(Tensor { ptr: t, _ctx: self._ctx }) }
    }

    /// 图里的节点总数。
    pub fn n_nodes(&self) -> usize {
        graph_n_nodes(self.ptr)
    }

    /// 节点算子名序列（按图执行顺序）。
    pub fn op_sequence(&self) -> Vec<String> {
        graph_op_sequence(self.ptr)
    }

    /// 按算子统计节点数：`(总数, [(算子名, 数量)])`，按数量降序。
    ///
    /// 用途（P8 性能归因）：一次 decode 的 compute 是 7.4 ms，而 llama.cpp 整 token
    /// 只要 6.8 ms。节点的**数量与构成**决定 kernel 启动次数（每个非 view 节点至少一次
    /// launch），所以先把这张直方图打出来，而不是靠猜。
    pub fn op_histogram(&self) -> (usize, Vec<(String, usize)>) {
        graph_op_histogram(self.ptr)
    }
}

/// 不借用 ctx 的图句柄（P2c：把一张 decode 图跨步复用）。
///
/// 与 [`StateTensor`] 同一套不变式：**`GgmlCtx` 必须比它活得久，且期间不能被 `reset()`**。
/// 存在的理由同样是自引用：`Engine` 要同时拥有 `scratch: GgmlCtx` 和建在它里面的图。
#[derive(Clone, Copy, Debug)]
pub struct RawGraph {
    ptr: *mut ffi::GgmlCgraph,
}

impl RawGraph {
    pub fn raw(&self) -> *mut ffi::GgmlCgraph {
        self.ptr
    }

    /// 图里的节点总数。
    pub fn n_nodes(&self) -> usize {
        graph_n_nodes(self.ptr)
    }

    /// 节点算子名序列（按图执行顺序）。
    pub fn op_sequence(&self) -> Vec<String> {
        graph_op_sequence(self.ptr)
    }

    /// 按算子统计节点数（见 `GraphBuilder::op_histogram`）。
    pub fn op_histogram(&self) -> (usize, Vec<(String, usize)>) {
        graph_op_histogram(self.ptr)
    }
}

fn graph_n_nodes(g: *mut ffi::GgmlCgraph) -> usize {
    unsafe { ffi::ggml_graph_n_nodes(g).max(0) as usize }
}

fn graph_op_sequence(g: *mut ffi::GgmlCgraph) -> Vec<String> {
    let n = graph_n_nodes(g);
    (0..n as i32)
        .map(|i| {
            let node = unsafe { ffi::ggml_graph_node(g, i) };
            if node.is_null() {
                String::new()
            } else {
                op_name(unsafe { (*node).op })
            }
        })
        .collect()
}

fn graph_op_histogram(g: *mut ffi::GgmlCgraph) -> (usize, Vec<(String, usize)>) {
    use std::collections::HashMap;
    let n = graph_n_nodes(g);
    let mut m: HashMap<i32, usize> = HashMap::new();
    for i in 0..n as i32 {
        let node = unsafe { ffi::ggml_graph_node(g, i) };
        if node.is_null() {
            continue;
        }
        let op = unsafe { (*node).op };
        *m.entry(op).or_insert(0) += 1;
    }
    let mut v: Vec<(String, usize)> = m.into_iter().map(|(op, c)| (op_name(op), c)).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    (n, v)
}

/// `ggml_op` 枚举值 → 名字（拿不到时退回 `op#N`）。
fn op_name(op: i32) -> String {
    unsafe {
        let p = ffi::ggml_op_name(op);
        if p.is_null() {
            format!("op#{op}")
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

impl Drop for GraphBuilder<'_> {
    fn drop(&mut self) {
        // ggml_cgraph is owned by the context; no explicit free needed
    }
}