// tests/ffi_smoke.rs — Task 2: FFI + safe wrapper smoke tests
//
// 验证：
//   1. ggml CUDA 库链接 + 版本报告（Task 1）
//   2. CUDA backend 符号可解析（Task 2，仅检查符号，不调用）
//   3. GgmlCtx / BackendBuffer / Tensor roundtrip（Task 2 Step 3）

#[link(name = "ggml-base")]
#[link(name = "ggml-cpu")]
#[link(name = "ggml-cuda")]
#[link(name = "ggml")]
#[link(name = "cudart")]
extern "C" {}

#[test]
fn links_and_reports_version() {
    extern "C" {
        #[link_name = "ggml_version"]
        fn ggml_version() -> *const i8;
    }
    unsafe {
        let p = ggml_version();
        assert!(!p.is_null(), "ggml_version returned null pointer");
        let cstr = std::ffi::CStr::from_ptr(p);
        let v = cstr.to_str().unwrap();
        println!("ggml version: {}", v);
        assert!(v.len() > 0, "version string is empty");
    }
}

#[test]
fn cuda_backend_symbol_resolved() {
    extern "C" {
        fn ggml_backend_cuda_init() -> *mut std::ffi::c_void;
    }
    // 仅验证函数符号可解析；不实际调用（需要 GPU）
    let ptr = ggml_backend_cuda_init as *const () as usize;
    assert_ne!(ptr, 0, "ggml_backend_cuda_init symbol not resolved");
}

#[test]
fn ctx_and_tensor_roundtrip() {
    use sqwen::backend::ctx::{GgmlCtx, BackendBuffer, BackendKind};

    let ctx = GgmlCtx::new().unwrap();
    let buf = BackendBuffer::new(BackendKind::Cpu, 1 << 20).unwrap();

    // 创建 [4, 3] F32 tensor (ne[0]=4, ne[1]=3)
    let t = ctx.new_tensor_f32(&[4, 3]);
    let data: [f32; 12] = (0..12).map(|i| i as f32).collect::<Vec<_>>().try_into().unwrap();
    buf.set_tensor_data(&t, bytemuck_bytes(&data));

    let back = buf.get_tensor_data_raw(&t);
    assert_eq!(back.len(), 48); // 12 * sizeof(f32)
    let back_f32: &[f32] = unsafe { std::slice::from_raw_parts(back.as_ptr() as *const f32, 12) };
    for i in 0..12 {
        assert!((back_f32[i] - data[i]).abs() < 1e-6, "mismatch at {}: {} vs {}", i, back_f32[i], data[i]);
    }
}

/// bytemuck_bytes: 将 &[f32; N] 转为 &[u8]（不依赖 bytemuck crate）。
fn bytemuck_bytes<T: Sized>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of::<T>() * slice.len()) }
}
