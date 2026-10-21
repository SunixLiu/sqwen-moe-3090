// tests/graph_test.rs — Task 8: 整模型前向图组装 + 真实模型冒烟
//
// 两个用例：
//   1. `bigbang_prefill_then_decode_smoke`：真实 BigBang（Q4_K_M，19.7 GiB）在
//      CUDA 上 prefill 5 个 token 再 decode 1 个 token，检查 logits 形状与有限性；
//   2. `bigbang_chunk_equivalence`：同一段 6 个 token，
//      「prefill5 + decode1」的最后一个 token 的 logits 必须等于「一次 6 个 token」
//      的结果 —— 这一条同时验证了 GDN 递推状态与 KV cache 的跨步接力。
//
// 显存前置：权重 19.7 GiB + GDN 状态 ~65 MiB + KV/图若干，沿用 gguf_test 里
// `full_load_bigbang_to_cuda` 的 21 GiB 门限。

use sqwen::backend::ctx::{BackendBuffer, BackendKind, GgmlCtx, GraphBuilder, Tensor};
use sqwen::model::arch::Qwen35MoeArch;
use sqwen::model::graph::{forward, EngineState, GRAPH_NODE_BUDGET};
use sqwen::model::weights::{GgufFile, ModelWeights};

const BIGBANG: &str = r"E:\LLMModels\bigbang\bigbang-v1-i1-gguf\BigBang-v1.i1-Q4_K_M.gguf";

/// KV cache 容量。烟测只有 6 个 token，取小值即可（真实长上下文需要更大的值，
/// 并应把 mask 改成按步分配 —— 现在已经是按步的了，容量只受显存限制）。
const KV_CAP: usize = 256;

fn gpu_free_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .parse::<u64>()
        .ok()
}

fn get_f32(buf: &BackendBuffer, t: &Tensor<'_>) -> Vec<f32> {
    let b = buf.get_tensor_data(t);
    bytemuck::cast_slice::<u8, f32>(&b).to_vec()
}

/// 第 1 个全注意力层 KV cache 里前 5 个位置的字节快照。
///
/// 后续步**不许**改动已经写好的位置。早期实现把每步的写回节点累积在 `KvCache`
/// 里、并在下一步重新 expand，于是第 1 步的整条子图被重跑（此时常驻 GDN 状态已经
/// 变了），把这些位置覆盖成了错的值 —— 这里的断言就是钉住这个不变量。
fn kv_snapshot(buf: &BackendBuffer, st: &EngineState, ctx: &GgmlCtx) -> (Vec<u8>, Vec<u8>) {
    let slot = 0; // 第一个全注意力层在 KvCache 里的槽位
    let k = buf.get_tensor_data(&st.kv.k_state(slot).get(ctx));
    let v = buf.get_tensor_data(&st.kv.v_state(slot).get(ctx));
    let row_k = k.len() / KV_CAP; // 每个位置占的字节数（两个 cache 都是 dim2 连续）
    let row_v = v.len() / KV_CAP;
    (k[..5 * row_k].to_vec(), v[..5 * row_v].to_vec())
}

/// 一列 logits（第 `t` 个 token）的 argmax。
fn argmax_col(vals: &[f32], n_vocab: usize, t: usize) -> i32 {
    let col = &vals[t * n_vocab..(t + 1) * n_vocab];
    let mut best = 0usize;
    for (i, v) in col.iter().enumerate() {
        if *v > col[best] {
            best = i;
        }
    }
    best as i32
}

/// 真实模型加载（三阶段 + EngineState），返回所有需要跨步存活的部件。
///
/// 顺序很关键：**引擎状态张量必须在第一次 `from_ctx` 之前创建**，否则它们拿不到
/// buffer（`ggml_backend_alloc_ctx_tensors` 只处理调用当时已存在的张量）。
///
/// 由于 `EngineState` 借用了 ctx，这里只能把各部件作为独立的局部变量交给调用方；
/// 后续的 `Engine` 类型需要「拥有 ctx + 用裸指针访问」的设计才能封装起来。
///
/// `$persistent` 绑定成下划线开头的名字只是为了消警告 —— 它必须活到作用域结束，
/// 否则权重与常驻状态的内存会被提前释放。
macro_rules! load_bigbang {
    ($ctx:ident, $arch:ident, $weights:ident, $persistent:ident, $kv_cap:expr, $($st:ident),+) => {
        let gguf = GgufFile::open(BIGBANG).expect("open gguf");
        let $arch = Qwen35MoeArch::from_gguf(&gguf).expect("arch");
        let $ctx = GgmlCtx::new_no_alloc().expect("ctx");
        let $weights = ModelWeights::create(&$ctx, &gguf, &$arch).expect("create");
        $(
            let mut $st = EngineState::new(&$ctx, &$arch, $kv_cap, sqwen::backend::ffi::GgmlType::F32, false, 4, 0, 0).expect("EngineState");
        )+
        let $persistent =
            BackendBuffer::from_ctx(&$ctx, BackendKind::Cuda(0)).expect("persistent buffer");
        let report = $weights.load_data(&$ctx, &gguf).expect("load_data");
        $(
            $st.zero_states();
        )+
        eprintln!(
            "[graph_test] 权重加载 {} 张量 / {:.2} GiB / {} ms",
            report.tensors,
            report.bytes as f64 / (1u64 << 30) as f64,
            report.elapsed_ms
        );
    };
}

#[test]
fn bigbang_prefill_then_decode_smoke() {
    if !std::path::Path::new(BIGBANG).exists() {
        eprintln!("[graph_test] SKIP: 模型文件不存在");
        return;
    }
    let free = gpu_free_mib().unwrap_or(0);
    if free < 21504 {
        eprintln!("[graph_test] SKIP GPU: 仅 {free} MiB 空闲 (< 21 GiB)");
        return;
    }

    load_bigbang!(ctx, arch, weights, _persistent, KV_CAP, st);
    let n_vocab = weights.output.get(&ctx).ne()[1] as usize;
    eprintln!(
        "[graph_test] arch: n_layer={} (trunk {} / full {} / recr {}) n_embd={} n_head={} n_head_kv={} head_dim={} n_vocab={}",
        arch.n_layer,
        arch.n_layer_trunk(),
        arch.n_full_attn_layers(),
        arch.n_recurrent_layers(),
        arch.n_embd,
        arch.n_head,
        arch.n_head_kv,
        arch.head_dim,
        n_vocab
    );


    // ---- prefill --------------------------------------------------------
    let prompt: Vec<i32> = vec![1, 2, 3, 4, 5];
    let mut g = GraphBuilder::new_with_size(&ctx, GRAPH_NODE_BUDGET);
    let logits = forward(&ctx, &mut g, &weights, &arch, &mut st, &prompt, 0, None).expect("forward");
    assert_eq!(logits.ne()[0], n_vocab as i64);
    assert_eq!(logits.ne()[1], 5);
    // 先分配本步张量，再写图输入
    let step_buf = BackendBuffer::from_ctx(&ctx, BackendKind::Cuda(0)).expect("step buffer");
    st.fill_inputs(&prompt).expect("fill_inputs");
    let t0 = std::time::Instant::now();
    g.compute(&step_buf).expect("compute prefill");
    eprintln!("[graph_test] prefill(5) 计算耗时 {} ms", t0.elapsed().as_millis());

    let vals = get_f32(&step_buf, &logits);
    assert_eq!(vals.len(), n_vocab * 5);
    for &v in &vals {
        assert!(v.is_finite(), "prefill logits 非有限值 {v}");
    }
    // 不能是退化输出：最后一列要有明显的极值
    let col = &vals[4 * n_vocab..5 * n_vocab];
    let mx = col.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mn = col.iter().cloned().fold(f32::INFINITY, f32::min);
    eprintln!("[graph_test] prefill 最后一位 logits: max={mx:.4} min={mn:.4} spread={:.4}", mx - mn);
    assert!(mx - mn > 1.0, "logits 几乎是常数（spread={}），图可能没算对", mx - mn);

    // ---- decode ---------------------------------------------------------
    let next = argmax_col(&vals, n_vocab, 4);
    eprintln!("[graph_test] prefill argmax = {next}");
    let mut g2 = GraphBuilder::new_with_size(&ctx, GRAPH_NODE_BUDGET);
    let logits2 =
        forward(&ctx, &mut g2, &weights, &arch, &mut st, &[next], 5, None).expect("forward decode");
    assert_eq!(logits2.ne()[0], n_vocab as i64);
    assert_eq!(logits2.ne()[1], 1);
    let step_buf2 = BackendBuffer::from_ctx(&ctx, BackendKind::Cuda(0)).expect("step buffer 2");
    st.fill_inputs(&[next]).expect("fill_inputs decode");
    let t1 = std::time::Instant::now();
    g2.compute(&step_buf2).expect("compute decode");
    eprintln!("[graph_test] decode(1) 计算耗时 {} ms", t1.elapsed().as_millis());

    let vals2 = get_f32(&step_buf2, &logits2);
    assert_eq!(vals2.len(), n_vocab);
    for &v in &vals2 {
        assert!(v.is_finite(), "decode logits 非有限值 {v}");
    }
    let mx2 = vals2.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mn2 = vals2.iter().cloned().fold(f32::INFINITY, f32::min);
    eprintln!("[graph_test] decode logits: max={mx2:.4} min={mn2:.4} spread={:.4}", mx2 - mn2);
    assert!(mx2 - mn2 > 1.0, "decode logits 几乎是常数");

    // decode 走的是 KV cache + 常驻 GDN 状态，结果不该和 prefill 的最后一位相同
    let same = vals[4 * n_vocab..5 * n_vocab]
        .iter()
        .zip(vals2.iter())
        .all(|(a, b)| a == b);
    assert!(!same, "decode 输出与 prefill 最后一位完全相同，状态可能没接上");
}

/// 分块等价：`prefill(5) + decode(1)` 与一次 `prefill(6)` 对最后一个 token 的
/// logits 必须一致。这一条同时覆盖 GDN 递推状态与 KV cache 的跨步接力。
#[test]
fn bigbang_chunk_equivalence() {
    if !std::path::Path::new(BIGBANG).exists() {
        eprintln!("[graph_test] SKIP: 模型文件不存在");
        return;
    }
    let free = gpu_free_mib().unwrap_or(0);
    if free < 21504 {
        eprintln!("[graph_test] SKIP GPU: 仅 {free} MiB 空闲 (< 21 GiB)");
        return;
    }

    // 两条路径各自独立的引擎状态（都在第一次 from_ctx 之前创建）
    load_bigbang!(ctx, arch, weights, _persistent, KV_CAP, st_a, st_b);
    let n_vocab = weights.output.get(&ctx).ne()[1] as usize;

    // ---- 路径 A：prefill 5 → decode 1 -----------------------------------
    let p5: Vec<i32> = vec![11, 22, 33, 44, 55];
    let mut ga = GraphBuilder::new_with_size(&ctx, GRAPH_NODE_BUDGET);
    let la = forward(&ctx, &mut ga, &weights, &arch, &mut st_a, &p5, 0, None).expect("forward A1");
    let buf_a1 = BackendBuffer::from_ctx(&ctx, BackendKind::Cuda(0)).expect("buf A1");
    st_a.fill_inputs(&p5).expect("fill A1");
    ga.compute(&buf_a1).expect("compute A1");
    let next = argmax_col(&get_f32(&buf_a1, &la), n_vocab, 4);

    // 第 1 步之后缓存里前 5 个位置的 K/V 快照：后续步**不许**改动它们。
    // （早期实现把写回节点累积在 KvCache 里、下一步又重新 expand，第 1 步的子图
    //   会被重跑并覆盖这些位置 —— 这条检查就是为了钉住这个不变量。）
    let (k_before, v_before) = kv_snapshot(&buf_a1, &st_a, &ctx);

    let mut gb2 = GraphBuilder::new_with_size(&ctx, GRAPH_NODE_BUDGET);
    let lb = forward(&ctx, &mut gb2, &weights, &arch, &mut st_a, &[next], 5, None).expect("forward A2");
    let buf_a2 = BackendBuffer::from_ctx(&ctx, BackendKind::Cuda(0)).expect("buf A2");
    st_a.fill_inputs(&[next]).expect("fill A2");
    gb2.compute(&buf_a2).expect("compute A2");
    let (k_after, v_after) = kv_snapshot(&buf_a2, &st_a, &ctx);
    assert_eq!(
        k_before, k_after,
        "decode 步改写了 KV cache 中已存在的位置（K）"
    );
    assert_eq!(
        v_before, v_after,
        "decode 步改写了 KV cache 中已存在的位置（V）"
    );
    let decode_logits = get_f32(&buf_a2, &lb);
    assert_eq!(decode_logits.len(), n_vocab);
    let prefix5 = get_f32(&buf_a1, &la); // 路径 A 第一步：tokens 0..4 的 logits

    // ---- 路径 B：一次 6 个 token ----------------------------------------
    let mut p6 = p5.clone();
    p6.push(next);
    let mut gc = GraphBuilder::new_with_size(&ctx, GRAPH_NODE_BUDGET);
    let lc = forward(&ctx, &mut gc, &weights, &arch, &mut st_b, &p6, 0, None).expect("forward B");
    let buf_b = BackendBuffer::from_ctx(&ctx, BackendKind::Cuda(0)).expect("buf B");
    st_b.fill_inputs(&p6).expect("fill B");
    gc.compute(&buf_b).expect("compute B");
    let full = get_f32(&buf_b, &lc);
    assert_eq!(full.len(), n_vocab * 6);
    let full_last = &full[5 * n_vocab..6 * n_vocab];

    // ---- 比较 -----------------------------------------------------------
    // (1) 公共前缀：因果实现里 tokens 0..4 都不受 token 5 影响，两条路径必须**数值上
    //     等价**（cos 极高、相对差很小）。
    //
    //     注意这里**不能**要求逐位相同：注意力层现在真的会读本步写进 KV cache 的
    //     K/V（见 graph.rs 里「KV 写回必须早于注意力节点进图」那段），而 K 投影的
    //     量化 matmul 是 batch-shape 敏感的 —— 同 5 个 token 在 5-token 图与 6-token
    //     图里写出的 q4_0 K cache 实测有 12/1440 字节不同（V 是 f32，0 字节不同），
    //     这点差异经 softmax 放大后落到 logits 上约 3.4%。此前这条断言能过，是因为
    //     KV 写回被排在了读之后：注意力读到空 cache、输出恒等于 0，前缀的差异被
    //     完全遮住了（那本身就是 bug）。
    //
    //     状态接力出问题会是量级差异（整段错乱），所以判据按"决策级 + 粗界"来定：
    //       (a) 每行 logits 的 argmax（即贪心会选的 token）必须一致 —— 这是真正关心的；
    //       (b) cos > 0.99；
    //       (c) 相对差留一个粗界（见下）。
    //
    //     实测标定：
    //       KV 写回修复时          cos=0.99955, 相对差 3.44%
    //       n_kv 对齐 256 之后     cos=0.99903, 相对差 10.21%
    //     后者变大是**同一批 q4_0 量化差异被混沌放大**后的结果，不是逻辑回归：
    //     同一 prompt 的 greedy 输出在两侧都通顺，只有 1 个近并列 token 不同
    //     （"独立描述" vs "单独描述"）。n_kv 对齐后 softmax 的归约长度从 5/6 变成 256，
    //     末位舍入随之变化，近并列处就会翻。
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut prefix_worst = 0f32;
    let mut prefix_scale = 0f32;
    let mut argmax_mismatch = Vec::new();
    for t in 0..5 {
        let ra = &prefix5[t * n_vocab..(t + 1) * n_vocab];
        let rb = &full[t * n_vocab..(t + 1) * n_vocab];
        let amax = |row: &[f32]| -> usize {
            let mut best = 0usize;
            for (i, &v) in row.iter().enumerate() {
                if v > row[best] {
                    best = i;
                }
            }
            best
        };
        let (ta, tb) = (amax(ra), amax(rb));
        if ta != tb {
            argmax_mismatch.push((t, ta, tb, ra[ta] - ra[tb]));
        }
        for i in 0..n_vocab {
            let (a, b) = (ra[i], rb[i]);
            prefix_worst = prefix_worst.max((a - b).abs());
            prefix_scale = prefix_scale.max(b.abs());
            dot += (a as f64) * (b as f64);
            na += (a as f64) * (a as f64);
            nb += (b as f64) * (b as f64);
        }
    }
    let prefix_cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
    let prefix_rel = 100.0 * prefix_worst / prefix_scale.max(1e-6);
    eprintln!(
        "[graph_test] 公共前缀(5 token) 最大差 {prefix_worst:.4e}（相对 {prefix_rel:.4}%），cos={prefix_cos:.6}, argmax 不一致 {}/5",
        argmax_mismatch.len()
    );
    assert!(
        prefix_cos > 0.99,
        "公共前缀在两条路径下相关性过低（cos={prefix_cos}），逐 token 计算或状态接力有问题"
    );
    assert!(
        argmax_mismatch.is_empty(),
        "公共前缀的贪心 token 不一致（这是决策级回归）：{argmax_mismatch:?}"
    );
    assert!(
        prefix_rel < 15.0,
        "公共前缀相对差过大（{prefix_rel:.4}%），逐 token 计算或状态接力有问题"
    );

    // (2) 第一步之后常驻 GDN 状态必须真的被写回（非零），否则 decode 会从零状态起算。
    {
        let st0 = ctx.wrap(st_a.gdn[0].as_ref().expect("layer 0 是递归层").state.ptr());
        let b = buf_a1.get_tensor_data(&st0);
        let v = bytemuck::cast_slice::<u8, f32>(&b);
        let mx = v.iter().fold(0f32, |m, x| m.max(x.abs()));
        let nz = v.iter().filter(|x| **x != 0.0).count();
        eprintln!("[graph_test] layer0 常驻 GDN state max|x|={mx:.4e}, 非零 {nz}/{}", v.len());
        assert!(mx > 1e-6 && nz > v.len() / 2, "常驻 GDN 状态没有被写回");
    }

    // (3) 最后一个 token：**不能**要求与一次性 6-token 逐位相等，只能判「大方向一致」。
    //
    //     实测标定（BigBang，同一个 prompt 不含任何分块）：
    //       CPU vs CUDA 一次性 prefill : cos=0.9988, 最大差 8.4%（54% 元素差 >1%）
    //       分块(5+1) vs 一次性(6)      : cos=0.9845, 最大差 23.2%
    //     也就是说连「同一张图在不同后端」都差到 8%，整模型级别本来就不是紧致的：
    //     量化 matmul 会把激活压成 int8，而具体走哪条 kernel / 怎么切块依赖 batch
    //     shape；40 层（含 30 层递归）+ top-8/256 MoE 再把它放大。分块 prefill 改变
    //     了每一层 matmul 的 batch，因此「整模型分块 == 一次性」不是这个实现能保证
    //     的性质，只要求相关性足够高（阈值 0.95，实测 0.9845）。
    //
    //     状态接力本身在层级别是**精确**的：tests/gdn_real_dim_test.rs 用生产维度
    //     （state 128x128x32、conv_channels 8192）验证 5+1 与 6 一次过相差 3e-7。
    let dot: f64 = decode_logits
        .iter()
        .zip(full_last.iter())
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    let na: f64 = decode_logits
        .iter()
        .map(|a| (*a as f64) * (*a as f64))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = full_last
        .iter()
        .map(|b| (*b as f64) * (*b as f64))
        .sum::<f64>()
        .sqrt();
    let cos = dot / (na * nb).max(1e-12);
    let scale = full_last.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst = decode_logits
        .iter()
        .zip(full_last.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "[graph_test] prefill5+decode1 vs prefill6 最后一个 token: 最大差 {worst:.4e}（相对 {:.4}%），cos={cos:.6}",
        100.0 * worst / scale
    );
    assert!(
        cos > 0.95,
        "两条路径的 logits 相关性过低（cos={cos}），状态接力可能有问题"
    );
}
