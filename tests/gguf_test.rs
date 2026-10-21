const BIGBANG: &str = r"E:\LLMModels\bigbang\bigbang-v1-i1-gguf\BigBang-v1.i1-Q4_K_M.gguf";
const HERMES: &str = r"E:\LLMModels\unsloth\qwen3.6-35B-A3B-Q4_K_m\Hermes3.6-35B-A3B-Uncensored-Genesis-V7-APEX-Compact.gguf";

#[test]
fn bigbang_arch_validates() {
    let gguf = sqwen::model::weights::GgufFile::open(BIGBANG).unwrap();
    let arch = sqwen::model::arch::Qwen35MoeArch::from_gguf(&gguf).unwrap();
    assert_eq!(arch.n_layer, 41); // block_count 含 MTP 块
    assert_eq!(arch.n_layer_nextn, 1);
    assert_eq!(arch.n_layer_trunk(), 40);
    assert_eq!(arch.n_embd, 2048);
    assert_eq!(arch.n_expert, 256);
    assert_eq!(arch.n_expert_used, 8);
    assert_eq!(arch.ssm_d_inner, 4096);
    assert_eq!(arch.full_attn_interval, 4);
    // llama.cpp qwen35moe.cpp L23-25 的 is_recr 规则
    assert!(arch.is_full_attn(3) && !arch.is_full_attn(0));
    // 主干内：10 层全注意力 + 30 层循环注意力
    assert_eq!(arch.n_full_attn_layers(), 10);
    assert_eq!(arch.n_recurrent_layers(), 30);
    // MTP 块（第 40 层）不在主干内，且被标记为全注意力
    assert!(arch.is_mtp_layer(40));
    assert!(!arch.is_mtp_layer(39));
    assert!(arch.is_full_attn(40));
    assert!(!arch.is_recurrent_attn(40));
}

#[test]
fn hermes_arch_validates() {
    let gguf = sqwen::model::weights::GgufFile::open(HERMES).unwrap();
    let arch = sqwen::model::arch::Qwen35MoeArch::from_gguf(&gguf).unwrap();
    assert_eq!(arch.n_layer, 40);
    assert_eq!(arch.n_layer_nextn, 0);
    assert_eq!(arch.n_layer_trunk(), 40);
    assert_eq!(arch.n_full_attn_layers(), 10);
    assert_eq!(arch.n_recurrent_layers(), 30);
}

#[test]
fn wrong_arch_rejected() {
    // 用 llama.cpp 自带小模型（非 qwen35moe）验证拒绝路径
    let p = r"D:\01Projects\llama.cpp\models\ggml-vocab-bert-bge.gguf";
    if std::path::Path::new(p).exists() {
        match sqwen::model::weights::GgufFile::open(p)
            .and_then(|g| sqwen::model::arch::Qwen35MoeArch::from_gguf(&g)) {
            Ok(_) => panic!("should reject non-qwen35moe"),
            Err(ref e) => println!("rejected as expected: {:?}", e),
        }
    }
}

#[test]
fn load_small_tensor_from_gguf() {
    use sqwen::backend::ctx::{BackendBuffer, BackendKind, GgmlCtx};
    use sqwen::model::weights::{load_gguf_tensor, GgufFile};

    let gguf = GgufFile::open(BIGBANG).unwrap();

    // 找到 blk.0.ssm_a（已知很小，type=F32, size=128）
    let tensor_id = match gguf.find_tensor("blk.0.ssm_a") {
        Some(id) => id,
        None => panic!("blk.0.ssm_a not found"),
    };

    let ne = gguf.tensor_ne(tensor_id);
    let typ = gguf.tensor_type(tensor_id);
    let expected_size = gguf.tensor_size(tensor_id);

    // 创建 GgmlCtx 和 BackendBuffer，分配同类型/形状的 tensor
    let ctx = GgmlCtx::new().unwrap();
    let ggml_type = unsafe { std::mem::transmute::<i32, sqwen::backend::ffi::GgmlType>(typ) };
    let ne_slice: Vec<i64> = ne.iter().take_while(|&&x| x > 1).map(|x| *x as i64).collect();
    let ne_slice = if ne_slice.is_empty() { vec![ne[0]] } else { ne_slice };
    let tensor = ctx.new_tensor(ggml_type, &ne_slice[..]);

    // 从 GGUF 文件加载数据到 tensor 的 data 指针
    let written = load_gguf_tensor(&tensor, &gguf, tensor_id).unwrap();
    assert_eq!(written, expected_size);

    // 验证：通过 backend buffer 读取 loaded 数据，大小与预期一致
    let buf = BackendBuffer::new(BackendKind::Cpu, expected_size + 1024).unwrap();
    let data = buf.get_tensor_data_raw(&tensor);
    assert_eq!(data.len(), expected_size, "loaded data size mismatch");

    // **数值**校验（不是「自己和自己比」）：`ssm_a` 存的是 `-A_log.exp()`，必然全为负、
    // 且量级有限。只比字节数/自比对的检查抓不到「偏移算错读到了 metadata 区」这类
    // 错误 —— 2026-02 就出现过一次：`gguf_get_tensor_offset` 返回的是相对数据区的
    // 偏移，当时漏加了 `gguf_get_data_offset`，整个模型加载到的都是垃圾（都是大
    // 数量级的随机 float），而所有自比对的字节检查照样通过。
    let vals: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
    assert!(vals.iter().all(|v| v.is_finite()), "ssm_a 含非有限值");
    assert!(vals.iter().all(|v| *v < 0.0), "ssm_a 应为负值（-A_log.exp()）: {:?}", &vals[..4]);
    let mx = vals.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(mx < 1e3, "ssm_a 量级异常（max|x|={mx}），偏移可能算错");
}

/// 张量的绝对偏移必须落在数据区之内（回归护栏：曾经漏加 data_offset）。
#[test]
fn bigbang_tensor_offsets_are_absolute() {
    use sqwen::model::weights::GgufFile;

    if !std::path::Path::new(BIGBANG).exists() {
        eprintln!("SKIP: 模型不存在");
        return;
    }
    let gguf = GgufFile::open(BIGBANG).unwrap();
    let data_off = gguf.data_offset();
    let n = gguf.tensor_count();
    let mut min_off = usize::MAX;
    for id in 0..n as i64 {
        let off = gguf.tensor_offset(id);
        assert!(
            off >= data_off,
            "张量 {} 的偏移 {off} 小于数据区起点 {data_off}（忘了加 data_offset？）",
            gguf.tensor_name(id as u64).unwrap_or_else(|| id.to_string())
        );
        min_off = min_off.min(off);
    }
    // 第一个张量的相对偏移是 0，所以最小绝对偏移必须正好等于数据区起点
    assert_eq!(min_off, data_off, "最小张量偏移与数据区起点不一致");
    eprintln!(
        "[gguf_test] {n} 个张量偏移全部 >= data_offset({data_off})，最小 {min_off}"
    );
}

// ===========================================================================
// Task 3 补齐：ModelWeights 名字映射 + 完整权重加载
// ===========================================================================

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

fn gpu_available_at_least(need_mib: u64, tag: &str) -> bool {
    match gpu_free_mib() {
        Some(m) if m >= need_mib => true,
        Some(m) => {
            eprintln!("[gguf_test] SKIP {tag}: 仅 {m} MiB 空闲 (< {need_mib} MiB)");
            false
        }
        None => {
            eprintln!("[gguf_test] SKIP {tag}: nvidia-smi 不可用");
            false
        }
    }
}

fn gpu_available() -> bool {
    use std::sync::OnceLock;
    static AVAIL: OnceLock<bool> = OnceLock::new();
    *AVAIL.get_or_init(|| gpu_available_at_least(1024, "GPU"))
}

/// 按层类型校验每个字段是否齐全，并确认「另一分支」的张量确实为空。
fn validate_layers(w: &sqwen::model::weights::ModelWeights, arch: &sqwen::model::arch::Qwen35MoeArch) {
    assert_eq!(
        w.layers.len(),
        arch.n_layer_trunk() as usize,
        "layers 数量应为主干层数（不含 MTP）"
    );

    let mut n_recr = 0;
    let mut n_full = 0;
    for (il, lw) in w.layers.iter().enumerate() {
        let need = |r: sqwen::model::weights::TensorRef, name: &str| {
            assert!(!r.is_null(), "blk.{il}.{name} 缺失");
        };
        need(lw.attn_norm, "attn_norm.weight");
        need(lw.attn_post_norm, "post_attention_norm.weight");
        need(lw.ffn_gate_inp, "ffn_gate_inp.weight");
        need(lw.ffn_down_exps, "ffn_down_exps.weight");
        need(lw.ffn_gate_inp_shexp, "ffn_gate_inp_shexp.weight");
        need(lw.ffn_gate_shexp, "ffn_gate_shexp.weight");
        need(lw.ffn_up_shexp, "ffn_up_shexp.weight");
        need(lw.ffn_down_shexp, "ffn_down_shexp.weight");

        // 路由专家：本模型为分离布局
        assert!(
            lw.ffn_gate_up_exps.is_null(),
            "blk.{il}: 本模型应使用分离的 gate/up 布局"
        );
        need(lw.ffn_gate_exps, "ffn_gate_exps.weight");
        need(lw.ffn_up_exps, "ffn_up_exps.weight");

        if arch.is_recurrent_attn(il as u32) {
            n_recr += 1;
            need(lw.attn_qkv, "attn_qkv.weight");
            need(lw.attn_gate, "attn_gate.weight");
            need(lw.ssm_conv1d, "ssm_conv1d.weight");
            need(lw.ssm_dt, "ssm_dt.bias");
            need(lw.ssm_a, "ssm_a");
            need(lw.ssm_beta, "ssm_beta.weight");
            need(lw.ssm_alpha, "ssm_alpha.weight");
            need(lw.ssm_norm, "ssm_norm.weight");
            need(lw.ssm_out, "ssm_out.weight");
            assert!(lw.attn_q.is_null(), "blk.{il}: 循环层不应有 attn_q");
            assert!(lw.attn_k.is_null(), "blk.{il}: 循环层不应有 attn_k");
        } else {
            n_full += 1;
            need(lw.attn_q, "attn_q.weight");
            need(lw.attn_k, "attn_k.weight");
            need(lw.attn_v, "attn_v.weight");
            need(lw.attn_output, "attn_output.weight");
            need(lw.attn_q_norm, "attn_q_norm.weight");
            need(lw.attn_k_norm, "attn_k_norm.weight");
            assert!(lw.attn_qkv.is_null(), "blk.{il}: 全注意力层不应有 attn_qkv");
            assert!(lw.ssm_out.is_null(), "blk.{il}: 全注意力层不应有 ssm_out");
        }
    }
    assert_eq!(n_recr, arch.n_recurrent_layers() as usize);
    assert_eq!(n_full, arch.n_full_attn_layers() as usize);
}

fn weight_map_smoke(path: &str, expect_trunk: u32, expect_nextn: u32) {
    use sqwen::backend::ctx::GgmlCtx;
    use sqwen::model::arch::Qwen35MoeArch;
    use sqwen::model::weights::{GgufFile, ModelWeights};

    let gguf = GgufFile::open(path).unwrap();
    let arch = Qwen35MoeArch::from_gguf(&gguf).unwrap();
    assert_eq!(arch.n_layer_trunk(), expect_trunk);
    assert_eq!(arch.n_layer_nextn, expect_nextn);

    let ctx = GgmlCtx::new_no_alloc().unwrap();
    let w = ModelWeights::create(&ctx, &gguf, &arch).unwrap();
    validate_layers(&w, &arch);

    // 不应把 MTP / nextn 张量也建出来
    assert!(
        !w.names().iter().any(|n| n.contains("nextn")),
        "MTP/nextn 张量应被跳过"
    );
    // 也不应创建 MTP 层（block.40）的普通张量
    if expect_nextn > 0 {
        let mtp = format!("blk.{}.attn_norm.weight", expect_trunk);
        assert!(!w.names().iter().any(|n| n == &mtp), "MTP 层 {mtp} 应被跳过");
    }

    println!(
        "[Task3] {}: trunk={} nextn={} created_tensors={} total={:.2} GiB",
        std::path::Path::new(path).file_name().unwrap().to_string_lossy(),
        arch.n_layer_trunk(),
        arch.n_layer_nextn,
        w.n_tensors(),
        w.total_bytes() as f64 / (1u64 << 30) as f64,
    );
}

#[test]
fn bigbang_weight_map_complete() {
    // BigBang: block_count=41、nextn_predict_layers=1 → 主干 40 层，第 40 层是 MTP
    weight_map_smoke(BIGBANG, 40, 1);
}

#[test]
fn hermes_weight_map_complete() {
    // Hermes: block_count=40、无 nextn → 主干 40 层
    weight_map_smoke(HERMES, 40, 0);
}

/// 完整链路的低成本验证：create → from_ctx → load_data（只加载预算内张量），
/// 并逐字节比对文件内容。
fn partial_load_and_verify(kind_name: &str, kind: sqwen::backend::ctx::BackendKind, budget: usize) {
    use sqwen::backend::ctx::{BackendBuffer, GgmlCtx};
    use sqwen::model::arch::Qwen35MoeArch;
    use sqwen::model::weights::{GgufFile, GgufReader, ModelWeights};

    let gguf = GgufFile::open(BIGBANG).unwrap();
    let arch = Qwen35MoeArch::from_gguf(&gguf).unwrap();

    let ctx = GgmlCtx::new_no_alloc().unwrap();
    let w = ModelWeights::create(&ctx, &gguf, &arch).unwrap();

    // 未分配 buffer 时 load_data 必须给出明确错误，而不是段错误
    match w.load_data(&ctx, &gguf) {
        Err(e) => {
            let msg = e.to_string();
            assert!(msg.contains("backend buffer"), "错误信息应提示需要先分配 buffer，实际: {msg}");
        }
        Ok(_) => panic!("未分配 buffer 时不应加载成功"),
    }

    let buf = BackendBuffer::from_ctx(&ctx, kind).unwrap();
    let report = w.load_data_budget(&ctx, &gguf, budget).unwrap();
    assert!(report.tensors > 0, "至少应加载一个张量");
    assert!(report.bytes <= budget.max(report.bytes));
    println!(
        "[Task3] {kind_name} 部分加载: {} 个张量 / {:.1} MiB / {} ms",
        report.tensors,
        report.bytes as f64 / (1u64 << 20) as f64,
        report.elapsed_ms
    );

    // 逐字节比对：已加载的张量内容必须与文件数据完全一致
    let mut reader = GgufReader::new(&gguf).unwrap();
    for (i, (r, id)) in w.entries().iter().enumerate() {
        if i >= report.tensors {
            break;
        }
        let expect = reader.read_raw(*id).unwrap();
        let got = r.get(&ctx).get_data_bytes(expect.len());
        assert_eq!(got.len(), expect.len());
        assert!(
            got == expect,
            "张量 {} ({:?}) 数据与文件不一致",
            w.names()[i],
            r
        );
    }
    println!("[Task3] {kind_name}: {} 个张量逐字节校验通过", report.tensors);

    let _ = buf;
}

#[test]
fn partial_load_bytes_exact_cpu() {
    // 第一个张量是 token_embd（Q4_K，286MB），预算 300MiB 会加载它 + output_norm
    partial_load_and_verify("CPU", sqwen::backend::ctx::BackendKind::Cpu, 300 << 20);
}

#[test]
fn partial_load_bytes_exact_cuda() {
    if !gpu_available() {
        return;
    }
    partial_load_and_verify(
        "CUDA",
        sqwen::backend::ctx::BackendKind::Cuda(0),
        300 << 20,
    );
}

/// 完整链路端到端：Model::open（create → CUDA 分配 → 全量加载），
/// 并抽查若干张量内容与文件逐字节一致。
///
/// BigBang 主干权重实测 19.70 GiB，加上 CUDA context 开销，要求 ≥ 21 GiB 空闲。
#[test]
fn full_load_bigbang_to_cuda() {
    use sqwen::backend::ctx::BackendKind;
    use sqwen::model::weights::{GgufFile, GgufReader, Model};

    if !gpu_available_at_least(21 * 1024, "full_load_bigbang_to_cuda") {
        return;
    }

    let t0 = std::time::Instant::now();
    let m = Model::open(BIGBANG, BackendKind::Cuda(0)).unwrap();
    let elapsed = t0.elapsed();

    assert_eq!(m.arch.n_layer_trunk(), 40);
    assert_eq!(m.weights.n_tensors(), 733);
    assert_eq!(m.weights.layers.len(), 40);
    println!(
        "[Task3] 全量加载完成: {} 张量 / {:.2} GiB / 加载 {:.1}s（含分配与读盘）",
        m.load_report.tensors,
        m.load_report.bytes as f64 / (1u64 << 30) as f64,
        elapsed.as_secs_f64()
    );

    // 抽查：全局张量 + 循环层 + 全注意力层 + 末层
    let gguf = GgufFile::open(BIGBANG).unwrap();
    let mut reader = GgufReader::new(&gguf).unwrap();
    let ctx = m.ctx();

    let mut check = |name: &str| {
        let id = gguf.find_tensor(name).unwrap_or_else(|| panic!("{name} 不存在"));
        let expect = reader.read_raw(id).unwrap();
        let got = ctx
            .wrap(gguf_tensor_ptr(&m, name))
            .get_data_bytes(expect.len());
        assert!(got == expect, "{name} 内容与文件不一致");
        println!("[Task3]   校验通过 {name} ({} KiB)", expect.len() / 1024);
    };

    for name in [
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.ssm_conv1d.weight",
        "blk.0.ssm_a",
        "blk.3.attn_q.weight",
        "blk.3.attn_v.weight",
        "blk.39.ffn_down_exps.weight",
        "blk.39.ffn_gate_shexp.weight",
    ] {
        check(name);
    }
}

/// 在已加载的 Model 里按名字找到张量指针（测试辅助）。
fn gguf_tensor_ptr(m: &sqwen::model::weights::Model, name: &str) -> *mut sqwen::backend::ffi::ggml_tensor {
    use sqwen::model::weights::TensorRef;
    let w = &m.weights;
    let direct: Option<TensorRef> = match name {
        "token_embd.weight" => Some(w.token_embd),
        "output.weight" => Some(w.output),
        "output_norm.weight" => Some(w.output_norm),
        _ => None,
    };
    if let Some(r) = direct {
        return r.as_ptr();
    }

    let rest = name.strip_prefix("blk.").expect("blk.N. 前缀");
    let (il_s, field) = rest.split_once('.').expect("blk.N.field");
    let il: usize = il_s.parse().unwrap();
    let l = &w.layers[il];
    let r = match field {
        "attn_norm.weight" => l.attn_norm,
        "post_attention_norm.weight" => l.attn_post_norm,
        "ffn_gate_inp.weight" => l.ffn_gate_inp,
        "ffn_gate_inp_shexp.weight" => l.ffn_gate_inp_shexp,
        "ffn_gate_shexp.weight" => l.ffn_gate_shexp,
        "ffn_up_shexp.weight" => l.ffn_up_shexp,
        "ffn_down_shexp.weight" => l.ffn_down_shexp,
        "ffn_gate_exps.weight" => l.ffn_gate_exps,
        "ffn_up_exps.weight" => l.ffn_up_exps,
        "ffn_down_exps.weight" => l.ffn_down_exps,
        "attn_qkv.weight" => l.attn_qkv,
        "attn_gate.weight" => l.attn_gate,
        "ssm_conv1d.weight" => l.ssm_conv1d,
        "ssm_dt.bias" => l.ssm_dt,
        "ssm_a" => l.ssm_a,
        "ssm_beta.weight" => l.ssm_beta,
        "ssm_alpha.weight" => l.ssm_alpha,
        "ssm_norm.weight" => l.ssm_norm,
        "ssm_out.weight" => l.ssm_out,
        "attn_q.weight" => l.attn_q,
        "attn_k.weight" => l.attn_k,
        "attn_v.weight" => l.attn_v,
        "attn_output.weight" => l.attn_output,
        "attn_q_norm.weight" => l.attn_q_norm,
        "attn_k_norm.weight" => l.attn_k_norm,
        other => panic!("未知字段 {other}"),
    };
    r.as_ptr()
}
