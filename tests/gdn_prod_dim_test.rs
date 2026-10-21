// tests/gdn_prod_dim_test.rs — 生产维度下的 GDN 分块等价性
//
// `tests/gdn_test.rs` 用的是小规模合成配置（head_v_dim=8、n_v_heads=4、
// conv_channels=64），而且它的 `ssm_norm` 长度是 `head_v_dim*n_v_heads`，
// 走的是 `norm_gated` 里的另一条分支。这里用 **BigBang 的真实维度**
// （head_v_dim=128、n_v_heads=32、n_k_heads=16、conv_channels=8192、conv=4、
// ssm_norm 长度 = head_v_dim = 128）再验一次「5+1 两次调用」与「6 一次调用」等价。
//
// 这是 GDN 分块正确性的核心不变式，也是引擎跨步状态接力（图内 cpy 写回常驻
// buffer）正确性的依据。CPU 后端即可，不需要 GPU。

use sqwen::backend::ctx::{BackendBuffer, BackendKind, GgmlCtx, GraphBuilder, StateTensor, Tensor};
use sqwen::backend::ffi::{self, GgmlType};
use sqwen::model::arch::Qwen35MoeArch;
use sqwen::model::gdn::{build_gdn_layer, GdnState};
use sqwen::model::weights::{LayerWeights, TensorRef};

const N_EMBD: i64 = 2048;
const HEAD_K: i64 = 128; // ssm_d_state：head_k_dim == head_v_dim
const N_K: i64 = 16; // ssm_n_group
const N_V: i64 = 32; // ssm_dt_rank
const D_INNER: i64 = 4096;
const CONV: i64 = 4; // ssm_d_conv = 卷积核长度
const CONV_CHANNELS: i64 = D_INNER + 2 * N_K * HEAD_K; // 8192

fn arch() -> Qwen35MoeArch {
    Qwen35MoeArch {
        n_embd: N_EMBD as u32,
        ssm_d_inner: D_INNER as u32,
        ssm_d_state: HEAD_K as u32,
        ssm_n_group: N_K as u32,
        ssm_dt_rank: N_V as u32,
        ssm_d_conv: CONV as u32,
        full_attn_interval: 4,
        rope_dim_count: 64,
        rope_freq_base: 1e7,
        rope_sections: [11, 11, 10, 0],
        norm_rms_eps: 1e-6,
        expert_ffn_len: 0,
        shared_expert_ffn_len: 0,
        n_layer_nextn: 0,
        n_ctx_train: 262144,
        head_dim: 256,
        n_head: 16,
        n_head_kv: 2,
        n_expert: 0,
        n_expert_used: 0,
        n_layer: 1,
    }
}

fn pseudo(len: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32) / 8388608.0 - 1.0
        })
        .collect()
}

fn weights(ctx: &GgmlCtx) -> LayerWeights {
    let qkv = ctx.new_tensor_2d(GgmlType::F32, N_EMBD, CONV_CHANNELS);
    let gate = ctx.new_tensor_2d(GgmlType::F32, N_EMBD, D_INNER);
    let beta = ctx.new_tensor_2d(GgmlType::F32, N_EMBD, N_V);
    let alpha = ctx.new_tensor_2d(GgmlType::F32, N_EMBD, N_V);
    let dt = ctx.new_tensor(GgmlType::F32, &[N_V]);
    let sa = ctx.new_tensor(GgmlType::F32, &[N_V]);
    let conv = ctx.new_tensor_2d(GgmlType::F32, CONV, CONV_CHANNELS);
    // 真实模型的 ssm_norm 长度是 head_v_dim（不是 head_v_dim*n_v_heads）
    let norm = ctx.new_tensor(GgmlType::F32, &[HEAD_K]);
    let out = ctx.new_tensor_2d(GgmlType::F32, D_INNER, N_EMBD);
    LayerWeights {
        attn_qkv: unsafe { TensorRef::from_ptr(qkv.ptr()) },
        attn_gate: unsafe { TensorRef::from_ptr(gate.ptr()) },
        ssm_beta: unsafe { TensorRef::from_ptr(beta.ptr()) },
        ssm_alpha: unsafe { TensorRef::from_ptr(alpha.ptr()) },
        ssm_dt: unsafe { TensorRef::from_ptr(dt.ptr()) },
        ssm_a: unsafe { TensorRef::from_ptr(sa.ptr()) },
        ssm_conv1d: unsafe { TensorRef::from_ptr(conv.ptr()) },
        ssm_norm: unsafe { TensorRef::from_ptr(norm.ptr()) },
        ssm_out: unsafe { TensorRef::from_ptr(out.ptr()) },
        ..LayerWeights::EMPTY
    }
}

fn upload(buf: &BackendBuffer, ctx: &GgmlCtx, w: &LayerWeights) {
    let n = |len: i64, seed: u32| -> Vec<f32> {
        pseudo(len as usize, seed).iter().map(|x| x * 0.2).collect()
    };
    buf.set_tensor_data(&w.attn_qkv.get(ctx), bytemuck::cast_slice(&n(N_EMBD * CONV_CHANNELS, 1)));
    buf.set_tensor_data(&w.attn_gate.get(ctx), bytemuck::cast_slice(&n(N_EMBD * D_INNER, 2)));
    buf.set_tensor_data(&w.ssm_beta.get(ctx), bytemuck::cast_slice(&n(N_EMBD * N_V, 3)));
    buf.set_tensor_data(&w.ssm_alpha.get(ctx), bytemuck::cast_slice(&n(N_EMBD * N_V, 4)));
    buf.set_tensor_data(&w.ssm_dt.get(ctx), bytemuck::cast_slice(&n(N_V, 5)));
    // ssm_a = -A_log.exp()，必须为负（gate <= 0 → exp(gate) <= 1）
    let a: Vec<f32> = pseudo(N_V as usize, 6).iter().map(|x| -x.abs() - 0.1).collect();
    buf.set_tensor_data(&w.ssm_a.get(ctx), bytemuck::cast_slice(&a));
    buf.set_tensor_data(&w.ssm_conv1d.get(ctx), bytemuck::cast_slice(&n(CONV * CONV_CHANNELS, 7)));
    buf.set_tensor_data(&w.ssm_norm.get(ctx), bytemuck::cast_slice(&n(HEAD_K, 8)));
    buf.set_tensor_data(&w.ssm_out.get(ctx), bytemuck::cast_slice(&n(D_INNER * N_EMBD, 9)));
}

/// 第 `tok` 个 token、第 `e` 个通道的输入值。
///
/// 按 **token** 而不是按元素下标偏移：分块路径的第 2 步只喂 1 个 token，
/// 它必须与一次性 6 个 token 的第 6 列逐元素相同，否则比较的就不是同一件事。
fn h_value(tok: usize, e: usize) -> f32 {
    0.1 * (((tok * N_EMBD as usize + e) as f32) * 0.017).sin()
}

fn fill_h(buf: &BackendBuffer, t: &Tensor<'_>, start_tok: usize, n: usize) {
    let mut v = Vec::with_capacity(N_EMBD as usize * n);
    for ti in 0..n {
        for e in 0..N_EMBD as usize {
            v.push(h_value(start_tok + ti, e));
        }
    }
    buf.set_tensor_data(t, bytemuck::cast_slice(&v));
}

/// 跑一条路径（元素是 (起始 token, token 数)），每步把新状态 cpy 回常驻张量。
fn run(steps: &[(usize, usize)]) -> Vec<f32> {
    let a = arch();
    let ctx = GgmlCtx::new_no_alloc().unwrap();
    let w = weights(&ctx);
    let mut st = GdnState {
        state: StateTensor::new(ctx.new_tensor(GgmlType::F32, &[HEAD_K, HEAD_K, N_V, 1])),
        n_rs: 0,
        n_snap: 1,
        conv_state: StateTensor::new(ctx.new_tensor_3d(GgmlType::F32, CONV - 1, CONV_CHANNELS, 1)),
    };

    let mut hs = Vec::new();
    let mut outs = Vec::new();
    let mut writes: Vec<Tensor> = Vec::new();
    for &(start, n) in steps {
        let h = ctx.new_tensor_2d(GgmlType::F32, N_EMBD, n as i64);
        hs.push((ctx.wrap(h.ptr()), start, n));

        let resident_state = ctx.wrap(st.state.ptr());
        let resident_conv = ctx.wrap(st.conv_state.ptr());
        let out = build_gdn_layer(&ctx, &w, &h, &mut st, n as i64, &a, None);
        let c = ctx.as_ptr();
        writes.push(ctx.wrap(unsafe {
            ffi::ggml_cpy(c, st.state.ptr(), resident_state.ptr())
        }));
        writes.push(ctx.wrap(unsafe {
            ffi::ggml_cpy(c, st.conv_state.ptr(), resident_conv.ptr())
        }));
        st.state = StateTensor::new(resident_state);
        st.conv_state = StateTensor::new(resident_conv);
        outs.push(out);
    }
    let st_res = ctx.wrap(st.state.ptr());
    let conv_res = ctx.wrap(st.conv_state.ptr());

    let buf = BackendBuffer::from_ctx(&ctx, BackendKind::Cpu).unwrap();
    let zeros = |t: &Tensor| vec![0u8; t.nbytes()];
    buf.set_tensor_data(&st_res, &zeros(&st_res));
    buf.set_tensor_data(&conv_res, &zeros(&conv_res));
    for (h, start, n) in &hs {
        fill_h(&buf, h, *start, *n);
    }
    upload(&buf, &ctx, &w);

    let mut g = GraphBuilder::new(&ctx);
    for wr in &writes {
        g.expand(wr);
    }
    g.expand(outs.last().unwrap());
    g.compute(&buf).unwrap();

    let b = buf.get_tensor_data(outs.last().unwrap());
    bytemuck::cast_slice::<u8, f32>(&b).to_vec()
}

/// 跑一批 `n` 个 token，返回 `(快照块 [S_v,S_v,H_v,n_written], conv_input [K-1+n, C, 1])`
/// 的**主机侧**数据。`n_rs` 决定 K。
///
/// 不做任何写回（不 cpy 到常驻张量）：快照本来就是图算子的输出，直接在 buffer 里读。
fn run_snapshots(n: usize, n_rs: usize) -> (Vec<f32>, Vec<f32>) {
    let a = arch();
    let ctx = GgmlCtx::new_no_alloc().unwrap();
    let w = weights(&ctx);
    let mut st = GdnState {
        state: StateTensor::new(ctx.new_tensor(
            GgmlType::F32,
            &[HEAD_K, HEAD_K, N_V, (1 + n_rs) as i64],
        )),
        conv_state: StateTensor::new(ctx.new_tensor_3d(
            GgmlType::F32,
            CONV - 1,
            CONV_CHANNELS,
            (1 + n_rs) as i64,
        )),
        n_rs,
        n_snap: 1,
    };
    let h = ctx.new_tensor_2d(GgmlType::F32, N_EMBD, n as i64);
    // 常驻（图输入）张量的句柄：build_gdn_layer 会把 st.state/st.conv_state
    // 重绑定为图内视图，所以要在调用前留一份。
    let resident_state = ctx.wrap(st.state.ptr());
    let resident_conv = ctx.wrap(st.conv_state.ptr());
    let out = build_gdn_layer(&ctx, &w, &h, &mut st, n as i64, &a, None);
    let snapshot = st.state.get(&ctx);
    let conv_input = st.conv_state.get(&ctx);

    let buf = BackendBuffer::from_ctx(&ctx, BackendKind::Cpu).unwrap();
    buf.set_tensor_data(&resident_state, &vec![0u8; resident_state.nbytes()]);
    buf.set_tensor_data(&resident_conv, &vec![0u8; resident_conv.nbytes()]);
    fill_h(&buf, &h, 0, n);
    upload(&buf, &ctx, &w);
    let mut g = GraphBuilder::new(&ctx);
    g.expand(&out);
    g.compute(&buf).unwrap();

    let st_v = bytemuck::cast_slice::<u8, f32>(&buf.get_tensor_data(&snapshot)).to_vec();
    let ci_v = bytemuck::cast_slice::<u8, f32>(&buf.get_tensor_data(&conv_input)).to_vec();
    (st_v, ci_v)
}

/// 读递推快照的槽 `s`（`[S_v,S_v,H_v]`，行优先）。
fn state_slot(v: &[f32], s: usize) -> Vec<f32> {
    let plane = (HEAD_K * HEAD_K * N_V) as usize;
    v[s * plane..(s + 1) * plane].to_vec()
}

/// 读卷积历史的槽 `s`：`conv_input` 的第 `(n_tokens - s)` 行起 `CONV-1` 行，
/// 按 `(i, c)` 逻辑顺序展开（与递推状态的槽比较时都不看物理布局）。
fn conv_slot(v: &[f32], n_tokens: usize, s: usize) -> Vec<f32> {
    let cc = CONV_CHANNELS as usize;
    let ck = (CONV - 1) as usize;
    let stride = ck + n_tokens;
    let off = n_tokens - s;
    let mut out = Vec::with_capacity(ck * cc);
    for c in 0..cc {
        for i in 0..ck {
            out.push(v[(off + i) + c * stride]);
        }
    }
    out
}

/// **投机解码回退快照的核心不变式**：一次 `n` token 的前向（`K = n_rs+1`）写出的
/// 槽 `s`，必须等于"只跑前 `n - s` 个 token"得到的最终状态。
///
/// 这条不变式一旦破了，部分接受之后的递推状态就是错的 —— 表现为投机解码"能跑但
/// 命中率明显偏低"（目标模型的 logits 被错误状态污染），是极难从输出上看出来的 bug。
#[test]
fn gdn_rollback_snapshots_match_shorter_runs() {
    const NRS: usize = 3;
    const N: usize = 4; // K = 4 → 写 4 个槽
    let (st_all, ci_all) = run_snapshots(N, NRS);

    for s in 0..N {
        let want_n = N - s; // 槽 s = "往回 s 个 token" = 跑 want_n 个 token 的最终状态
        let (st_ref, ci_ref) = run_snapshots(want_n, NRS);

        let got = state_slot(&st_all, s);
        let want = state_slot(&st_ref, 0);
        let mut worst = 0f32;
        for (a, b) in got.iter().zip(want.iter()) {
            worst = worst.max((a - b).abs());
        }
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        eprintln!(
            "[gdn_rollback] 状态槽 s={s}（= 跑 {want_n} token）: 最大差 {worst:.3e}（相对 {:.5}%）",
            100.0 * worst / scale
        );
        assert!(
            worst <= 1e-5 * scale,
            "递推状态快照槽 {s} 与跑 {want_n} 个 token 的结果不一致"
        );

        let got_c = conv_slot(&ci_all, N, s);
        let want_c = conv_slot(&ci_ref, want_n, 0);
        let mut worst_c = 0f32;
        for (a, b) in got_c.iter().zip(want_c.iter()) {
            worst_c = worst_c.max((a - b).abs());
        }
        let scale_c = want_c.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        eprintln!(
            "[gdn_rollback] 卷积槽 s={s}: 最大差 {worst_c:.3e}（相对 {:.5}%）",
            100.0 * worst_c / scale_c
        );
        assert!(
            worst_c <= 1e-5 * scale_c,
            "卷积历史快照槽 {s} 与跑 {want_n} 个 token 的结果不一致"
        );
    }
}

#[test]
fn gdn_chunk_equivalence_real_dims() {
    const T: usize = 6;
    let split = run(&[(0, 5), (5, 1)]);
    let single = run(&[(0, T)]);

    assert_eq!(split.len(), N_EMBD as usize);
    assert_eq!(single.len(), N_EMBD as usize * T);

    // split 的输出只有 1 列；single 的第 T-1 列是同一个 token
    let mut worst = 0f32;
    let mut at = 0usize;
    for e in 0..N_EMBD as usize {
        let d = (split[e] - single[(T - 1) * N_EMBD as usize + e]).abs();
        if d > worst {
            worst = d;
            at = e;
        }
    }
    let scale = single[(T - 1) * N_EMBD as usize..]
        .iter()
        .fold(0f32, |m, v| m.max(v.abs()))
        .max(1e-6);
    eprintln!(
        "[gdn_prod_dim] 5+1 vs 6 最后一个 token: 最大差 {worst:.4e}（相对 {:.7}%）idx {at}",
        100.0 * worst / scale
    );
    assert!(
        worst <= 1e-4 * scale,
        "生产维度下 GDN 分块不等价：最大差 {worst:e} 相对 {:.5}% > 1e-4",
        100.0 * worst / scale
    );
}
