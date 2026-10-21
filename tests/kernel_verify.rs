// tests/kernel_verify.rs — Task 4: GPU vs CPU 差分验证套件
//
// 目的：验证 qwen35moe 计算图所需的 ggml 算子在 CUDA 后端上的数值正确性。
// 方法：
//   1. 同一张图分别在 CPU 与 CUDA 后端上以**逐字节相同**的输入计算；
//   2. 断言两后端输出一致（差分测试，这是最终裁判）；
//   3. 对可独立推导的算子，再用 Rust 侧参考实现交叉校验（防止两端同错）。
//
// 张量分配遵循 llama.cpp 标准流程：
//   GgmlCtx::new_no_alloc() → 建图（tensor->data == NULL）
//   → BackendBuffer::from_ctx()（ggml_backend_alloc_ctx_tensors 统一分配）
//   → set 输入 → GraphBuilder::compute → get 输出
//
// 覆盖算子：mul_mat(Q4_0/Q4_K/F32)、mul_mat_id、argsort_top_k、top_k、
//           ssm_conv、rope_multi(IMROPE)、cumsum、softplus、silu、soft_max、
//           get_rows、repeat、transpose

use sqwen::backend::ctx::{BackendBuffer, BackendKind, GgmlCtx, GraphBuilder, Tensor};
use sqwen::backend::ffi::{self, GgmlType};

// ===========================================================================
// 测试框架
// ===========================================================================

/// 一次后端运行所需的图：输入张量（含要写入的字节）+ 输出张量。
struct Case<'c> {
    inputs: Vec<(Tensor<'c>, Vec<u8>)>,
    output: Tensor<'c>,
}

fn gpu_free_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next()?.trim().parse::<u64>().ok()
}

/// 用户约束：GPU 测试前先查显存，不足则跳过。
fn gpu_available() -> bool {
    use std::sync::OnceLock;
    static AVAIL: OnceLock<bool> = OnceLock::new();
    *AVAIL.get_or_init(|| match gpu_free_mib() {
        Some(m) if m >= 1024 => true,
        Some(m) => {
            eprintln!("[kernel_verify] SKIP GPU: 仅 {} MiB 空闲 (< 1024 MiB)", m);
            false
        }
        None => {
            eprintln!("[kernel_verify] SKIP GPU: nvidia-smi 不可用");
            false
        }
    })
}

/// 在指定后端上跑一遍图，并用 `extract` 抽取结果。
fn run_case_with<F, R>(kind: BackendKind, build: &F, extract: impl Fn(&Tensor) -> R) -> R
where
    F: for<'c> Fn(&'c GgmlCtx) -> Case<'c>,
{
    let ctx = GgmlCtx::new_no_alloc().expect("GgmlCtx::new_no_alloc");
    let case = build(&ctx);
    let buf = BackendBuffer::from_ctx(&ctx, kind).expect("BackendBuffer::from_ctx");
    for (t, bytes) in &case.inputs {
        t.set_data_bytes(bytes);
    }
    let mut g = GraphBuilder::new(&ctx);
    g.expand(&case.output);
    g.compute(&buf).expect("graph compute");
    extract(&case.output)
}

/// 在指定后端上跑一遍图，返回输出张量的原始字节。
fn run_case<F>(kind: BackendKind, build: &F) -> Vec<u8>
where
    F: for<'c> Fn(&'c GgmlCtx) -> Case<'c>,
{
    run_case_with(kind, build, |t| t.get_data_bytes(t.nbytes()))
}

/// 按 nb 步长逐行读取 I32 张量（view 结果可能不连续，如 argsort_top_k）。
fn read_i32_strided(t: &Tensor) -> Vec<i32> {
    let ne = t.ne();
    let nb = t.nb();
    let n0 = ne[0].max(0) as usize;
    let n1 = ne[1].max(0) as usize;
    let n2 = ne[2].max(0) as usize;
    let n3 = ne[3].max(0) as usize;
    let rows = n1 * n2 * n3;
    let mut out = Vec::with_capacity(n0 * rows);
    for r in 0..rows {
        // 只在 dim1/2/3 上有步长差异；dim0 连续
        let i3 = r / (n1 * n2);
        let i2 = (r / n1) % n2;
        let i1 = r % n1;
        let off = i1 as u64 * nb[1] + i2 as u64 * nb[2] + i3 as u64 * nb[3];
        let bytes = t.get_data_bytes_at(off as usize, n0 * 4);
        for c in bytes.chunks_exact(4) {
            out.push(i32::from_ne_bytes([c[0], c[1], c[2], c[3]]));
        }
    }
    out
}

fn run_i32_strided<F>(kind: BackendKind, build: &F) -> Vec<i32>
where
    F: for<'c> Fn(&'c GgmlCtx) -> Case<'c>,
{
    run_case_with(kind, build, read_i32_strided)
}

fn to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn as_bytes_f32(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_ne_bytes());
    }
    out
}

fn as_bytes_i32(v: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_ne_bytes());
    }
    out
}

/// 差值报告
fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
}

/// 核心断言：CPU 结果 vs Rust 参考、CPU vs CUDA。
fn check_f32<F>(name: &str, build: F, expect: &[f32], ref_tol: f32, cross_tol: f32)
where
    F: for<'c> Fn(&'c GgmlCtx) -> Case<'c>,
{
    let cpu = to_f32(&run_case(BackendKind::Cpu, &build));
    assert_eq!(cpu.len(), expect.len(), "{name}: CPU 输出长度不符");
    let (worst, at) = max_abs_diff(&cpu, expect);
    assert!(
        worst <= ref_tol,
        "{name}: CPU vs 参考 超差 idx {at}: cpu={} ref={} diff={worst:e} tol={ref_tol:e}",
        cpu[at],
        expect[at]
    );
    eprintln!("[kernel_verify] {name}: CPU vs ref  max={worst:.3e} (tol {ref_tol:.1e})");

    if !gpu_available() {
        eprintln!("[kernel_verify] {name}: GPU 跳过");
        return;
    }
    let gpu = to_f32(&run_case(BackendKind::Cuda(0), &build));
    assert_eq!(gpu.len(), expect.len(), "{name}: CUDA 输出长度不符");
    let (worst, at) = max_abs_diff(&cpu, &gpu);
    assert!(
        worst <= cross_tol,
        "{name}: CPU vs CUDA 超差 idx {at}: cpu={} gpu={} diff={worst:e} tol={cross_tol:e}",
        cpu[at],
        gpu[at]
    );
    let (worst_ref, _) = max_abs_diff(&gpu, expect);
    eprintln!(
        "[kernel_verify] {name}: CPU vs CUDA max={worst:.3e} (tol {cross_tol:.1e}); CUDA vs ref max={worst_ref:.3e}"
    );
}

/// i32 输出（索引类算子）：要求逐元素完全相等。按 nb 步长读取，兼容 view 结果。
fn check_i32<F>(name: &str, build: F, expect: &[i32])
where
    F: for<'c> Fn(&'c GgmlCtx) -> Case<'c>,
{
    let cpu = run_i32_strided(BackendKind::Cpu, &build);
    assert_eq!(
        cpu, expect,
        "{name}: CPU vs 参考不一致\n cpu={cpu:?}\n ref={expect:?}"
    );
    if !gpu_available() {
        eprintln!("[kernel_verify] {name}: GPU 跳过");
        return;
    }
    let gpu = run_i32_strided(BackendKind::Cuda(0), &build);
    assert_eq!(
        cpu, gpu,
        "{name}: CPU vs CUDA 不一致\n cpu={cpu:?}\n gpu={gpu:?}"
    );
    eprintln!("[kernel_verify] {name}: CPU == CUDA ({} elems)", cpu.len());
}

/// 同时取回 CPU 与（可选的）CUDA 结果，供需要自定义比较方式的用例使用。
fn run_both_i32<F>(name: &str, build: F) -> (Vec<i32>, Option<Vec<i32>>)
where
    F: for<'c> Fn(&'c GgmlCtx) -> Case<'c>,
{
    let cpu = run_i32_strided(BackendKind::Cpu, &build);
    if !gpu_available() {
        eprintln!("[kernel_verify] {name}: GPU 跳过");
        return (cpu, None);
    }
    let gpu = run_i32_strided(BackendKind::Cuda(0), &build);
    (cpu, Some(gpu))
}

// ===========================================================================
// 工具：确定性伪随机、f16 解码、Q4_0 反量化、量化
// ===========================================================================

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// [lo, hi) 的均匀分布
    fn f32(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u32() >> 8) as f32 / ((1u32 << 24) as f32);
        lo + (hi - lo) * u
    }
    /// 保证互不相等的取值（用于排序类算子避免并列）
    fn distinct_f32(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u32() as f64) / (u32::MAX as f64);
        (lo as f64 + (hi - lo) as f64 * u) as f32
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = if (h >> 15) & 1 == 1 { -1.0f32 } else { 1.0f32 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    if e == 0 {
        sign * m * 2f32.powi(-24)
    } else if e == 31 {
        if m == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    }
}

/// Q4_0 反量化（对照 ggml quantize_row_q4_0_ref / dequantize_row_q4_0）
/// 布局：每 32 个权重一个 block = { f16 d; u8 qs[16] }，
/// 元素 j = low_nibble(qs[j]) - 8，元素 j+16 = high_nibble(qs[j]) - 8，均乘 d。
fn dequant_q4_0(bytes: &[u8], n_per_row: usize, nrows: usize) -> Vec<f32> {
    assert_eq!(n_per_row % 32, 0);
    let blocks_per_row = n_per_row / 32;
    let mut out = vec![0f32; n_per_row * nrows];
    let mut off = 0usize;
    for r in 0..nrows {
        for b in 0..blocks_per_row {
            let d = f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]));
            let qs = &bytes[off + 2..off + 18];
            for j in 0..16 {
                let lo = (qs[j] & 0x0f) as i32 - 8;
                let hi = (qs[j] >> 4) as i32 - 8;
                out[r * n_per_row + b * 32 + j] = lo as f32 * d;
                out[r * n_per_row + b * 32 + j + 16] = hi as f32 * d;
            }
            off += 18;
        }
    }
    out
}

/// 用 ggml 自己的量化器把 f32 权重压成目标类型（CPU 与 CUDA 共用同一份字节）。
fn quantize(ty: GgmlType, data: &[f32], n_per_row: i64, nrows: i64) -> Vec<u8> {
    let row_size = unsafe { ffi::ggml_row_size(ty as i32, n_per_row) };
    let mut out = vec![0u8; row_size * nrows as usize];
    let n = unsafe {
        ffi::ggml_quantize_chunk(
            ty as i32,
            data.as_ptr(),
            out.as_mut_ptr() as *mut std::ffi::c_void,
            0,
            nrows,
            n_per_row,
            std::ptr::null(),
        )
    };
    assert_eq!(n, out.len(), "ggml_quantize_chunk 返回长度不符");
    out
}

// ===========================================================================
// 参考实现（Rust 侧）
// ===========================================================================

/// C = A^T-ish: A [k, n], B [k, m] → C [n, m]，C[j,i] = sum_c A[c,j]*B[c,i]
fn matmul_ref(a: &[f32], k: usize, n: usize, b: &[f32], m: usize) -> Vec<f32> {
    let mut c = vec![0f32; n * m];
    for j in 0..n {
        for i in 0..m {
            let mut s = 0f32;
            for cc in 0..k {
                s += a[cc + k * j] * b[cc + k * i];
            }
            c[j + n * i] = s;
        }
    }
    c
}

/// 量化矩阵乘的误差上界（逐个输出元素推导）：
///
///   out[j,i] = Σ_c A[c,j]·B[c,i]
///   |Δout| ≤ Σ_c |ΔA[c,j]|·|B[c,i]| + Σ_c |A[c,j]|·|ΔB[c,i]|
///          ≤ e_a·Σ_c|B[c,i]| + e_b·Σ_c|A[c,j]|
///
/// 其中 e_a = max|A| / 2^bits_a（4bit 权重量化步长的一半），
///     e_b = max|B| / 2^8（ggml 会先把 f32 激活量化成 int8 再点积）。
fn quant_matmul_bound(a: &[f32], b: &[f32], k: usize, n: usize, m: usize, bits_a: u32) -> f32 {
    let max_a = a.iter().fold(0f32, |x, y| x.max(y.abs()));
    let max_b = b.iter().fold(0f32, |x, y| x.max(y.abs()));
    let e_a = max_a / (1u64 << bits_a) as f32;
    let e_b = max_b / 256.0;

    let mut worst = 0f32;
    for j in 0..n {
        let mut sum_a = 0f32;
        for c in 0..k {
            sum_a += a[c + k * j].abs();
        }
        for i in 0..m {
            let mut sum_b = 0f32;
            for c in 0..k {
                sum_b += b[c + k * i].abs();
            }
            worst = worst.max(e_a * sum_b + e_b * sum_a);
        }
    }
    worst
}

/// 量化算子的断言策略：
///  - CPU vs 参考：绝对容差 = 误差理论上界（`abs_tol`）**且** 相对容差 `rel_tol`；
///  - CPU vs CUDA：相对容差 `rel_tol`（两后端激活量化路径不同：CPU q8_0/q8_K vs CUDA MMQ）。
///
/// 相对容差是真正有效的回归护栏（实测均在 0.2%~0.5%，取 2% 留 4 倍余量）；
/// 绝对容差来自公式推导，防止在数值被放大时相对判据失真。
fn check_quant_f32<F>(name: &str, build: F, expect: &[f32], abs_tol: f32, rel_tol: f32)
where
    F: for<'c> Fn(&'c GgmlCtx) -> Case<'c>,
{
    let scale = expect
        .iter()
        .fold(0f32, |x, y| x.max(y.abs()))
        .max(1e-6);
    let rel_limit = rel_tol * scale;

    let cpu = to_f32(&run_case(BackendKind::Cpu, &build));
    assert_eq!(cpu.len(), expect.len(), "{name}: CPU 输出长度不符");
    let (worst, at) = max_abs_diff(&cpu, expect);
    assert!(
        worst <= abs_tol,
        "{name}: CPU vs 参考 超出理论上界 idx {at}: diff={worst:e} > {abs_tol:e}"
    );
    assert!(
        worst <= rel_limit,
        "{name}: CPU vs 参考 相对超差 idx {at}: cpu={} ref={} diff={worst:e} 相对 {:.4}% > {:.2}%",
        cpu[at],
        expect[at],
        100.0 * worst / scale,
        100.0 * rel_tol
    );
    eprintln!(
        "[kernel_verify] {name}: CPU vs ref max={worst:.3e} (相对 {:.4}%, 上界 {abs_tol:.1e})",
        100.0 * worst / scale
    );

    if !gpu_available() {
        eprintln!("[kernel_verify] {name}: GPU 跳过");
        return;
    }
    let gpu = to_f32(&run_case(BackendKind::Cuda(0), &build));
    assert_eq!(gpu.len(), expect.len(), "{name}: CUDA 输出长度不符");
    let (worst, at) = max_abs_diff(&cpu, &gpu);
    assert!(
        worst <= rel_limit,
        "{name}: CPU vs CUDA 超差 idx {at}: cpu={} gpu={} diff={worst:e} 相对 {:.4}% > {:.2}%",
        cpu[at],
        gpu[at],
        100.0 * worst / scale,
        100.0 * rel_tol
    );
    eprintln!(
        "[kernel_verify] {name}: CPU vs CUDA max={worst:.3e} (相对 {:.4}%)",
        100.0 * worst / scale
    );
}

/// IMROPE 参考实现（逐行对照 ggml-cpu/ops.cpp ggml_mrope_cache_init + rotate_pairs）
fn rope_imrope_ref(
    a: &[f32],
    head_dim: usize,
    n_head: usize,
    n_tokens: usize,
    pos: &[i32],
    sections: [i32; 4],
    freq_base: f32,
) -> Vec<f32> {
    let n_dims = head_dim;
    let mut out = a.to_vec();
    let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
    let sect_dims: i32 = sections.iter().sum();
    let sec_w = sections[1] + sections[0];

    for t in 0..n_tokens {
        let p_t = pos[t] as f32;
        let p_h = pos[t + n_tokens] as f32;
        let p_w = pos[t + n_tokens * 2] as f32;
        let p_e = pos[t + n_tokens * 3] as f32;
        let (mut th_t, mut th_h, mut th_w, mut th_e) = (p_t, p_h, p_w, p_e);

        // cache[i0] = cos(theta_i), cache[i0+1] = sin(theta_i)
        let mut cache = vec![0f32; n_dims];
        for i0 in (0..n_dims).step_by(2) {
            let sector = ((i0 / 2) as i32) % sect_dims;
            let theta = if sector % 3 == 1 && sector < 3 * sections[1] {
                th_h
            } else if sector % 3 == 2 && sector < 3 * sections[2] {
                th_w
            } else if sector % 3 == 0 && sector < 3 * sections[0] {
                th_t
            } else {
                th_e
            };
            cache[i0] = theta.cos();
            cache[i0 + 1] = theta.sin();
            th_t *= theta_scale;
            th_w *= theta_scale;
            th_h *= theta_scale;
            th_e *= theta_scale;
        }
        let _ = sec_w; // sec_w 仅非 imrope 分支使用

        for h in 0..n_head {
            let base = h * head_dim + t * head_dim * n_head;
            for i0 in (0..n_dims).step_by(2) {
                let ic = i0 / 2;
                let (ct, st) = (cache[i0], cache[i0 + 1]);
                let x0 = a[base + ic];
                let x1 = a[base + ic + n_dims / 2];
                out[base + ic] = x0 * ct - x1 * st;
                out[base + ic + n_dims / 2] = x0 * st + x1 * ct;
            }
        }
    }
    out
}

// ===========================================================================
// 用例 1：mul_mat（F32 × F32）
// ===========================================================================

#[test]
fn mul_mat_f32() {
    const K: usize = 8;
    const N: usize = 3; // 输出行数（A 的行数）
    const M: usize = 2; // token 数（B 的行数）

    let mut rng = Rng::new(1);
    let a: Vec<f32> = (0..K * N).map(|_| rng.f32(-1.0, 1.0)).collect();
    let b: Vec<f32> = (0..K * M).map(|_| rng.f32(-1.0, 1.0)).collect();
    let expect = matmul_ref(&a, K, N, &b, M);

    let a2 = a.clone();
    let b2 = b.clone();
    check_f32(
        "mul_mat_f32",
        move |ctx: &GgmlCtx| {
            let ta = ctx.new_tensor_2d(GgmlType::F32, K as i64, N as i64);
            let tb = ctx.new_tensor_2d(GgmlType::F32, K as i64, M as i64);
            let out = ctx.wrap(unsafe { ffi::ggml_mul_mat(ctx.as_ptr(), ta.ptr(), tb.ptr()) });
            Case {
                inputs: vec![(ta, as_bytes_f32(&a2)), (tb, as_bytes_f32(&b2))],
                output: out,
            }
        },
        &expect,
        1e-5,
        1e-5,
    );
}

// ===========================================================================
// 用例 2：mul_mat 的 B 侧为 Q4_0（量化权重路径）
// ===========================================================================

#[test]
fn mul_mat_q4_0_b() {
    const K: usize = 32;
    const N: usize = 8;
    const M: usize = 4;

    let mut rng = Rng::new(2);
    let a: Vec<f32> = (0..K * N).map(|_| rng.f32(-1.0, 1.0)).collect();
    let b: Vec<f32> = (0..K * M).map(|_| rng.f32(-1.0, 1.0)).collect();

    let q = quantize(GgmlType::Q4_0, &a, K as i64, N as i64);
    // 独立参考：用 ggml 量化出的字节自行反量化后做矩阵乘。
    // 注意 ggml 的量化 mul_mat 会先把 f32 激活压成 int8（Q8_0/Q8_1），
    // 因此这里与「纯 f32 累加」必然存在 int8 量化误差，容差按上界给。
    let a_deq = dequant_q4_0(&q, K, N);
    let expect = matmul_ref(&a_deq, K, N, &b, M);
    let tol = quant_matmul_bound(&a_deq, &b, K, N, M, 4);

    let q2 = q.clone();
    let b2 = b.clone();
    check_quant_f32(
        "mul_mat_q4_0_b",
        move |ctx: &GgmlCtx| {
            let ta = ctx.new_tensor_2d(GgmlType::Q4_0, K as i64, N as i64);
            let tb = ctx.new_tensor_2d(GgmlType::F32, K as i64, M as i64);
            let out = ctx.wrap(unsafe { ffi::ggml_mul_mat(ctx.as_ptr(), ta.ptr(), tb.ptr()) });
            Case {
                inputs: vec![(ta, q2.clone()), (tb, as_bytes_f32(&b2))],
                output: out,
            }
        },
        &expect,
        tol,
        0.02,
    );
}

// ===========================================================================
// 用例 3：mul_mat 的 B 侧为 Q4_K（模型主力量化类型）
// ===========================================================================

#[test]
fn mul_mat_q4k_b() {
    const K: usize = 256; // Q4_K 要求 256 的倍数
    const N: usize = 8;
    const M: usize = 4;

    let mut rng = Rng::new(3);
    let a: Vec<f32> = (0..K * N).map(|_| rng.f32(-1.0, 1.0)).collect();
    let b: Vec<f32> = (0..K * M).map(|_| rng.f32(-1.0, 1.0)).collect();

    let q = quantize(GgmlType::Q4K, &a, K as i64, N as i64);
    // Q4_K 反量化较复杂，改用「原始 f32 权重」做参考，容差按量化误差上界给
    let expect = matmul_ref(&a, K, N, &b, M);
    let tol = quant_matmul_bound(&a, &b, K, N, M, 4);

    let q2 = q.clone();
    let b2 = b.clone();
    check_quant_f32(
        "mul_mat_q4k_b",
        move |ctx: &GgmlCtx| {
            let ta = ctx.new_tensor_2d(GgmlType::Q4K, K as i64, N as i64);
            let tb = ctx.new_tensor_2d(GgmlType::F32, K as i64, M as i64);
            let out = ctx.wrap(unsafe { ffi::ggml_mul_mat(ctx.as_ptr(), ta.ptr(), tb.ptr()) });
            Case {
                inputs: vec![(ta, q2.clone()), (tb, as_bytes_f32(&b2))],
                output: out,
            }
        },
        &expect,
        tol,
        0.02,
    );
}

// ===========================================================================
// 用例 4：mul_mat_id（MoE 专家 gather-GEMM，F32 专家）
// ===========================================================================

#[test]
fn mul_mat_id_f32() {
    const K: usize = 8; // cols
    const R: usize = 4; // rows（每个专家的输出维度）
    const NE: usize = 3; // 专家数
    const NU: usize = 2; // n_expert_used
    const NT: usize = 2; // token 数

    let mut rng = Rng::new(4);
    // as [K, R, NE]
    let asp: Vec<f32> = (0..K * R * NE).map(|_| rng.f32(-1.0, 1.0)).collect();
    // b [K, NU, NT]
    let b: Vec<f32> = (0..K * NU * NT).map(|_| rng.f32(-1.0, 1.0)).collect();
    // ids [NU, NT]
    let ids: Vec<i32> = vec![2, 0, 1, 2];

    // c[r, e, t] = sum_c as[c, r, ids[e,t]] * b[c, e, t]
    let mut expect = vec![0f32; R * NU * NT];
    for t in 0..NT {
        for e in 0..NU {
            let id = ids[e + NU * t] as usize;
            for r in 0..R {
                let mut s = 0f32;
                for c in 0..K {
                    s += asp[c + K * r + K * R * id] * b[c + K * e + K * NU * t];
                }
                expect[r + R * e + R * NU * t] = s;
            }
        }
    }

    let asp2 = asp.clone();
    let b2 = b.clone();
    let ids2 = ids.clone();
    check_f32(
        "mul_mat_id_f32",
        move |ctx: &GgmlCtx| {
            let ta = ctx.new_tensor_3d(GgmlType::F32, K as i64, R as i64, NE as i64);
            let tb = ctx.new_tensor_3d(GgmlType::F32, K as i64, NU as i64, NT as i64);
            let ti = ctx.new_tensor_2d(GgmlType::I32, NU as i64, NT as i64);
            let out = ctx.wrap(unsafe {
                ffi::ggml_mul_mat_id(ctx.as_ptr(), ta.ptr(), tb.ptr(), ti.ptr())
            });
            Case {
                inputs: vec![
                    (ta, as_bytes_f32(&asp2)),
                    (tb, as_bytes_f32(&b2)),
                    (ti, as_bytes_i32(&ids2)),
                ],
                output: out,
            }
        },
        &expect,
        1e-5,
        1e-5,
    );
}

// ===========================================================================
// 用例 5：mul_mat_id 的专家权重为 Q4_0（真实 MoE 路径）
// ===========================================================================

#[test]
fn mul_mat_id_q4_0_experts() {
    const K: usize = 32;
    const R: usize = 8;
    const NE: usize = 4;
    const NU: usize = 2;
    const NT: usize = 3;

    let mut rng = Rng::new(5);
    let asp: Vec<f32> = (0..K * R * NE).map(|_| rng.f32(-1.0, 1.0)).collect();
    let b: Vec<f32> = (0..K * NU * NT).map(|_| rng.f32(-1.0, 1.0)).collect();
    let ids: Vec<i32> = vec![3, 0, 1, 2, 2, 1];

    // 量化 as（[K, R*NE] 行主序，共 R*NE 行、每行 K 个元素）
    let q = quantize(GgmlType::Q4_0, &asp, K as i64, (R * NE) as i64);
    let asp_deq = dequant_q4_0(&q, K, R * NE);

    let mut expect = vec![0f32; R * NU * NT];
    for t in 0..NT {
        for e in 0..NU {
            let id = ids[e + NU * t] as usize;
            for r in 0..R {
                let mut s = 0f32;
                for c in 0..K {
                    s += asp_deq[c + K * r + K * R * id] * b[c + K * e + K * NU * t];
                }
                expect[r + R * e + R * NU * t] = s;
            }
        }
    }

    let q2 = q.clone();
    let b2 = b.clone();
    let ids2 = ids.clone();
    // 参考用的是精确反量化权重，故只剩 B 的 int8 激活量化误差项
    let max_b = b.iter().fold(0f32, |x, y| x.max(y.abs()));
    let mut tol = 0f32;
    for t in 0..NT {
        for e in 0..NU {
            let id = ids[e + NU * t] as usize;
            for r in 0..R {
                let mut sum_a = 0f32;
                for c in 0..K {
                    sum_a += asp_deq[c + K * r + K * R * id].abs();
                }
                tol = tol.max(max_b / 256.0 * sum_a);
            }
        }
    }
    check_quant_f32(
        "mul_mat_id_q4_0_experts",
        move |ctx: &GgmlCtx| {
            let ta = ctx.new_tensor_3d(GgmlType::Q4_0, K as i64, R as i64, NE as i64);
            let tb = ctx.new_tensor_3d(GgmlType::F32, K as i64, NU as i64, NT as i64);
            let ti = ctx.new_tensor_2d(GgmlType::I32, NU as i64, NT as i64);
            let out = ctx.wrap(unsafe {
                ffi::ggml_mul_mat_id(ctx.as_ptr(), ta.ptr(), tb.ptr(), ti.ptr())
            });
            Case {
                inputs: vec![
                    (ta, q2.clone()),
                    (tb, as_bytes_f32(&b2)),
                    (ti, as_bytes_i32(&ids2)),
                ],
                output: out,
            }
        },
        &expect,
        tol,
        0.02,
    );
}

// ===========================================================================
// 用例 6：argsort_top_k（MoE 路由）
// ===========================================================================

#[test]
fn argsort_top_k() {
    const N: usize = 16;
    const ROWS: usize = 2;
    const K: usize = 4;

    let mut rng = Rng::new(6);
    let a: Vec<f32> = (0..N * ROWS).map(|_| rng.distinct_f32(-1.0, 1.0)).collect();

    let mut expect = Vec::with_capacity(K * ROWS);
    for r in 0..ROWS {
        let mut idx: Vec<usize> = (0..N).collect();
        idx.sort_by(|&x, &y| a[x + N * r].partial_cmp(&a[y + N * r]).unwrap().reverse());
        for &i in idx.iter().take(K) {
            expect.push(i as i32);
        }
    }

    let a2 = a.clone();
    check_i32(
        "argsort_top_k",
        move |ctx: &GgmlCtx| {
            let ta = ctx.new_tensor_2d(GgmlType::F32, N as i64, ROWS as i64);
            let out =
                ctx.wrap(unsafe { ffi::ggml_argsort_top_k(ctx.as_ptr(), ta.ptr(), K as i32) });
            Case {
                inputs: vec![(ta, as_bytes_f32(&a2))],
                output: out,
            }
        },
        &expect,
    );
}

// ===========================================================================
// 用例 7：top_k（结果无序，比较索引集合）
// ===========================================================================

#[test]
fn top_k_indices_set() {
    const N: usize = 16;
    const ROWS: usize = 2;
    const K: usize = 4;

    let mut rng = Rng::new(7);
    let a: Vec<f32> = (0..N * ROWS).map(|_| rng.distinct_f32(-1.0, 1.0)).collect();

    // 参考：每行 top-k 索引的升序集合
    let mut expect_sets: Vec<Vec<i32>> = Vec::new();
    for r in 0..ROWS {
        let mut idx: Vec<usize> = (0..N).collect();
        idx.sort_by(|&x, &y| a[x + N * r].partial_cmp(&a[y + N * r]).unwrap().reverse());
        let mut s: Vec<i32> = idx.iter().take(K).map(|&i| i as i32).collect();
        s.sort_unstable();
        expect_sets.push(s);
    }

    let a2 = a.clone();
    let (cpu, gpu) = run_both_i32("top_k_indices_set", move |ctx: &GgmlCtx| -> Case<'_> {
        let ta = ctx.new_tensor_2d(GgmlType::F32, N as i64, ROWS as i64);
        let out = ctx.wrap(unsafe { ffi::ggml_top_k(ctx.as_ptr(), ta.ptr(), K as i32) });
        Case {
            inputs: vec![(ta, as_bytes_f32(&a2))],
            output: out,
        }
    });

    let mut cpu_sets: Vec<Vec<i32>> = Vec::new();
    for r in 0..ROWS {
        let mut s = cpu[r * K..(r + 1) * K].to_vec();
        s.sort_unstable();
        cpu_sets.push(s);
    }
    assert_eq!(cpu_sets, expect_sets, "top_k: CPU 索引集合与参考不一致");

    let Some(gpu) = gpu else { return };
    let mut gpu_sets: Vec<Vec<i32>> = Vec::new();
    for r in 0..ROWS {
        let mut s = gpu[r * K..(r + 1) * K].to_vec();
        s.sort_unstable();
        gpu_sets.push(s);
    }
    assert_eq!(cpu_sets, gpu_sets, "top_k: CPU 与 CUDA 索引集合不一致");
    eprintln!("[kernel_verify] top_k_indices_set: CPU == CUDA (集合比较)");
}

// ===========================================================================
// 用例 8：ssm_conv（gated delta-net 的短卷积）
// ===========================================================================

#[test]
fn ssm_conv() {
    // CUDA kernel 约束：ggml-cuda/ssm-conv.cu:132 `GGML_ASSERT(nr % threads == 0)`，
    // threads = 128，故 d_inner（真实模型 ssm_d_inner=4096）必须是 128 的倍数。
    const D_CONV: usize = 4;
    const N_T: usize = 3;
    const D_INNER: usize = 128;
    const N_S: usize = 1;
    let ncs = D_CONV - 1 + N_T; // sx 的时间维

    let mut rng = Rng::new(8);
    let sx: Vec<f32> = (0..ncs * D_INNER * N_S).map(|_| rng.f32(-1.0, 1.0)).collect();
    let c: Vec<f32> = (0..D_CONV * D_INNER).map(|_| rng.f32(-1.0, 1.0)).collect();

    // sx [ncs, d_inner, n_s], c [d_conv, d_inner] → out [d_inner, n_t, n_s]
    // out[i1, i2, i3] = sum_{i0<d_conv} sx[i0+i2, i1, i3] * c[i0, i1]
    let mut expect = vec![0f32; D_INNER * N_T * N_S];
    for i3 in 0..N_S {
        for i2 in 0..N_T {
            for i1 in 0..D_INNER {
                let mut s = 0f32;
                for i0 in 0..D_CONV {
                    s += sx[(i0 + i2) + ncs * i1 + ncs * D_INNER * i3] * c[i0 + D_CONV * i1];
                }
                expect[i1 + D_INNER * i2 + D_INNER * N_T * i3] = s;
            }
        }
    }

    let sx2 = sx.clone();
    let c2 = c.clone();
    check_f32(
        "ssm_conv",
        move |ctx: &GgmlCtx| {
            let tsx = ctx.new_tensor_3d(GgmlType::F32, ncs as i64, D_INNER as i64, N_S as i64);
            let tc = ctx.new_tensor_2d(GgmlType::F32, D_CONV as i64, D_INNER as i64);
            let out = ctx.wrap(unsafe { ffi::ggml_ssm_conv(ctx.as_ptr(), tsx.ptr(), tc.ptr()) });
            Case {
                inputs: vec![(tsx, as_bytes_f32(&sx2)), (tc, as_bytes_f32(&c2))],
                output: out,
            }
        },
        &expect,
        1e-5,
        1e-5,
    );
}

// ===========================================================================
// 用例 9：rope_multi（IMROPE，qwen35moe 使用的 M-RoPE 变体）
// ===========================================================================

#[test]
fn rope_multi_imrope() {
    const HEAD_DIM: usize = 64;
    const N_HEAD: usize = 2;
    const N_TOK: usize = 3;
    const SECTIONS: [i32; 4] = [11, 11, 10, 0];
    const FREQ_BASE: f32 = 1e7;

    let mut rng = Rng::new(9);
    let a: Vec<f32> = (0..HEAD_DIM * N_HEAD * N_TOK)
        .map(|_| rng.f32(-1.0, 1.0))
        .collect();

    // pos 布局：[t0..tn, h0..hn, w0..wn, e0..en]（CPU/CUDA 均按 pos[i2 + ne2*k] 取值）
    let mut pos = vec![0i32; 4 * N_TOK];
    for t in 0..N_TOK {
        pos[t] = t as i32;
        pos[N_TOK + t] = 100 + t as i32;
        pos[2 * N_TOK + t] = 200 + t as i32;
        pos[3 * N_TOK + t] = 0;
    }

    let expect = rope_imrope_ref(&a, HEAD_DIM, N_HEAD, N_TOK, &pos, SECTIONS, FREQ_BASE);

    let a2 = a.clone();
    let pos2 = pos.clone();
    check_f32(
        "rope_multi_imrope",
        move |ctx: &GgmlCtx| {
            let ta = ctx.new_tensor_3d(GgmlType::F32, HEAD_DIM as i64, N_HEAD as i64, N_TOK as i64);
            let tp = ctx.new_tensor_1d_i32((4 * N_TOK) as i64);
            let out = ctx.wrap(unsafe {
                ffi::ggml_rope_multi(
                    ctx.as_ptr(),
                    ta.ptr(),
                    tp.ptr(),
                    std::ptr::null_mut(),
                    HEAD_DIM as i32,
                    SECTIONS.as_ptr(),
                    ffi::GGML_ROPE_TYPE_IMROPE,
                    4096,
                    FREQ_BASE,
                    1.0,
                    0.0,
                    1.0,
                    32.0,
                    1.0,
                )
            });
            Case {
                inputs: vec![(ta, as_bytes_f32(&a2)), (tp, as_bytes_i32(&pos2))],
                output: out,
            }
        },
        &expect,
        1e-5,
        1e-4,
    );
}

// ===========================================================================
// 用例 10：cumsum（gated delta-net 的累积和）
// ===========================================================================

#[test]
fn cumsum() {
    const N: usize = 8;
    const ROWS: usize = 3;

    let mut rng = Rng::new(10);
    let a: Vec<f32> = (0..N * ROWS).map(|_| rng.f32(-1.0, 1.0)).collect();

    let mut expect = vec![0f32; N * ROWS];
    for r in 0..ROWS {
        let mut acc = 0f32;
        for i in 0..N {
            acc += a[i + N * r];
            expect[i + N * r] = acc;
        }
    }

    let a2 = a.clone();
    check_f32(
        "cumsum",
        move |ctx: &GgmlCtx| {
            let ta = ctx.new_tensor_2d(GgmlType::F32, N as i64, ROWS as i64);
            let out = ctx.wrap(unsafe { ffi::ggml_cumsum(ctx.as_ptr(), ta.ptr()) });
            Case {
                inputs: vec![(ta, as_bytes_f32(&a2))],
                output: out,
            }
        },
        &expect,
        1e-5,
        1e-4,
    );
}

// ===========================================================================
// 用例 11：softplus / silu（delta-net 门控激活）
// ===========================================================================

#[test]
fn softplus_and_silu() {
    const N: usize = 8;
    const ROWS: usize = 4;

    let mut rng = Rng::new(11);
    let a: Vec<f32> = (0..N * ROWS).map(|_| rng.f32(-4.0, 4.0)).collect();

    let expect_softplus: Vec<f32> = a.iter().map(|&x| (1.0 + x.exp()).ln()).collect();
    let expect_silu: Vec<f32> = a.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

    let a2 = a.clone();
    check_f32(
        "softplus",
        move |ctx: &GgmlCtx| {
            let ta = ctx.new_tensor_2d(GgmlType::F32, N as i64, ROWS as i64);
            let out = ctx.wrap(unsafe { ffi::ggml_softplus(ctx.as_ptr(), ta.ptr()) });
            Case {
                inputs: vec![(ta, as_bytes_f32(&a2))],
                output: out,
            }
        },
        &expect_softplus,
        1e-5,
        1e-4,
    );

    let a3 = a.clone();
    check_f32(
        "silu",
        move |ctx: &GgmlCtx| {
            let ta = ctx.new_tensor_2d(GgmlType::F32, N as i64, ROWS as i64);
            let out = ctx.wrap(unsafe { ffi::ggml_silu(ctx.as_ptr(), ta.ptr()) });
            Case {
                inputs: vec![(ta, as_bytes_f32(&a3))],
                output: out,
            }
        },
        &expect_silu,
        1e-5,
        1e-4,
    );
}

// ===========================================================================
// 用例 12：soft_max
// ===========================================================================

#[test]
fn soft_max() {
    const N: usize = 8;
    const ROWS: usize = 3;

    let mut rng = Rng::new(12);
    let a: Vec<f32> = (0..N * ROWS).map(|_| rng.f32(-3.0, 3.0)).collect();

    let mut expect = vec![0f32; N * ROWS];
    for r in 0..ROWS {
        let row = &a[r * N..(r + 1) * N];
        let m = row.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        let mut sum = 0f32;
        for &x in row {
            sum += (x - m).exp();
        }
        for (i, &x) in row.iter().enumerate() {
            expect[r * N + i] = (x - m).exp() / sum;
        }
    }

    let a2 = a.clone();
    check_f32(
        "soft_max",
        move |ctx: &GgmlCtx| {
            let ta = ctx.new_tensor_2d(GgmlType::F32, N as i64, ROWS as i64);
            let out = ctx.wrap(unsafe { ffi::ggml_soft_max(ctx.as_ptr(), ta.ptr()) });
            Case {
                inputs: vec![(ta, as_bytes_f32(&a2))],
                output: out,
            }
        },
        &expect,
        1e-5,
        1e-5,
    );
}

// ===========================================================================
// 用例 13：get_rows（token embedding 查表）
// ===========================================================================

#[test]
fn get_rows() {
    const N_EMBD: usize = 4;
    const N_VOCAB: usize = 5;
    const N_ROWS: usize = 3;

    let mut rng = Rng::new(13);
    let a: Vec<f32> = (0..N_EMBD * N_VOCAB).map(|_| rng.f32(-1.0, 1.0)).collect();
    let idx: Vec<i32> = vec![3, 0, 4];

    let mut expect = vec![0f32; N_EMBD * N_ROWS];
    for r in 0..N_ROWS {
        let src = idx[r] as usize;
        for c in 0..N_EMBD {
            expect[c + N_EMBD * r] = a[c + N_EMBD * src];
        }
    }

    let a2 = a.clone();
    let idx2 = idx.clone();
    check_f32(
        "get_rows",
        move |ctx: &GgmlCtx| {
            let ta = ctx.new_tensor_2d(GgmlType::F32, N_EMBD as i64, N_VOCAB as i64);
            let ti = ctx.new_tensor_1d_i32(N_ROWS as i64);
            let out = ctx.wrap(unsafe { ffi::ggml_get_rows(ctx.as_ptr(), ta.ptr(), ti.ptr()) });
            Case {
                inputs: vec![(ta, as_bytes_f32(&a2)), (ti, as_bytes_i32(&idx2))],
                output: out,
            }
        },
        &expect,
        1e-6,
        1e-6,
    );
}

// ===========================================================================
// 用例 14：repeat（广播）
// ===========================================================================

#[test]
fn repeat() {
    const A0: usize = 2;
    const A1: usize = 3;
    const B0: usize = 4;
    const B1: usize = 6;

    let mut rng = Rng::new(14);
    let a: Vec<f32> = (0..A0 * A1).map(|_| rng.f32(-1.0, 1.0)).collect();

    // dst[i0, i1] = a[i0 % A0, i1 % A1]
    let mut expect = vec![0f32; B0 * B1];
    for i1 in 0..B1 {
        for i0 in 0..B0 {
            expect[i0 + B0 * i1] = a[(i0 % A0) + A0 * (i1 % A1)];
        }
    }

    let a2 = a.clone();
    check_f32(
        "repeat",
        move |ctx: &GgmlCtx| {
            let ta = ctx.new_tensor_2d(GgmlType::F32, A0 as i64, A1 as i64);
            let tb = ctx.new_tensor_2d(GgmlType::F32, B0 as i64, B1 as i64);
            let out = ctx.wrap(unsafe { ffi::ggml_repeat(ctx.as_ptr(), ta.ptr(), tb.ptr()) });
            Case {
                inputs: vec![(ta, as_bytes_f32(&a2))],
                output: out,
            }
        },
        &expect,
        1e-6,
        1e-6,
    );
}

// ===========================================================================
// 用例 15：transpose
// ===========================================================================

#[test]
fn transpose() {
    const N0: usize = 3;
    const N1: usize = 4;

    let mut rng = Rng::new(15);
    let a: Vec<f32> = (0..N0 * N1).map(|_| rng.f32(-1.0, 1.0)).collect();

    let mut expect = vec![0f32; N0 * N1];
    for i1 in 0..N1 {
        for i0 in 0..N0 {
            expect[i1 + N1 * i0] = a[i0 + N0 * i1];
        }
    }

    let a2 = a.clone();
    check_f32(
        "transpose",
        move |ctx: &GgmlCtx| {
            let ta = ctx.new_tensor_2d(GgmlType::F32, N0 as i64, N1 as i64);
            let tc = ctx.wrap(unsafe { ffi::ggml_cont(ctx.as_ptr(), ta.ptr()) });
            let out = ctx.wrap(unsafe { ffi::ggml_transpose(ctx.as_ptr(), tc.ptr()) });
            let out = ctx.wrap(unsafe { ffi::ggml_cont(ctx.as_ptr(), out.ptr()) });
            Case {
                inputs: vec![(ta, as_bytes_f32(&a2))],
                output: out,
            }
        },
        &expect,
        1e-6,
        1e-6,
    );
}
