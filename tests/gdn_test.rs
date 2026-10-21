// tests/gdn_test.rs — Task 6: GDN layer CPU/CUDA differential tests
use sqwen::backend::ctx::{GgmlCtx, BackendBuffer, BackendKind, GraphBuilder, Tensor};
use sqwen::backend::ffi;
use sqwen::model::arch::Qwen35MoeArch;
use sqwen::model::weights::{LayerWeights, TensorRef};
use sqwen::model::gdn::{GdnState, build_gdn_layer};

fn arch_test() -> Qwen35MoeArch {
    Qwen35MoeArch {
        n_embd: 64, ssm_d_inner: 32, ssm_d_state: 8,
        ssm_n_group: 2, ssm_dt_rank: 4, ssm_d_conv: 4,
        full_attn_interval: 4, rope_dim_count: 0, rope_freq_base: 0.0,
        rope_sections: [0; 4], norm_rms_eps: 1e-6,
        expert_ffn_len: 0, shared_expert_ffn_len: 0, n_layer_nextn: 0,
        n_ctx_train: 262144, head_dim: 0,
        n_head: 0, n_head_kv: 0, n_expert: 0, n_expert_used: 0, n_layer: 1,
    }
}

/// 由 arch 推出 conv 相关维度，避免测试里手写常量与配置漂移。
fn conv_dims(arch: &Qwen35MoeArch) -> (i64, i64) {
    let conv_kernel_size = (arch.ssm_d_conv + 1) as i64;
    let conv_channels =
        (arch.ssm_d_inner + 2 * arch.ssm_n_group * arch.ssm_d_state) as i64;
    (conv_kernel_size, conv_channels)
}

fn make_synthetic_weights_in_ctx(arch: &Qwen35MoeArch, ctx: &GgmlCtx) -> LayerWeights {
    let n_embd = arch.n_embd as i64;
    let d_inner = arch.ssm_d_inner as i64;
    let num_v_heads = arch.ssm_dt_rank as i64;
    let head_v_dim = d_inner / num_v_heads;
    let (conv_kernel_size, conv_channels) = conv_dims(arch);

    // ggml_mul_mat(a, b) computes a^T * b. Constraint: a.ne[0] == b.ne[0].
    // Weight tensors must be [in_features, out_features] so mul_mat(weight, input) → [out, batch].
    let w_qkvz = ctx.new_tensor_2d(sqwen::backend::ffi::GgmlType::F32, n_embd, conv_channels);
    let w_attn_gate = ctx.new_tensor_2d(sqwen::backend::ffi::GgmlType::F32, n_embd, head_v_dim * num_v_heads);
    let w_ssm_beta = ctx.new_tensor_2d(sqwen::backend::ffi::GgmlType::F32, n_embd, num_v_heads);
    let w_ssm_alpha = ctx.new_tensor_2d(sqwen::backend::ffi::GgmlType::F32, n_embd, num_v_heads);
    let w_ssm_dt = ctx.new_tensor(sqwen::backend::ffi::GgmlType::F32, &[num_v_heads]);
    let w_ssm_a = ctx.new_tensor(sqwen::backend::ffi::GgmlType::F32, &[num_v_heads]);
    let w_ssm_conv1d = ctx.new_tensor_2d(sqwen::backend::ffi::GgmlType::F32, conv_kernel_size, conv_channels);
    let w_ssm_norm = ctx.new_tensor(sqwen::backend::ffi::GgmlType::F32, &[head_v_dim * num_v_heads]);
    let w_ssm_out = ctx.new_tensor_2d(sqwen::backend::ffi::GgmlType::F32, head_v_dim * num_v_heads, n_embd);

    LayerWeights {
        attn_qkv: unsafe { TensorRef::from_ptr(w_qkvz.ptr()) },
        attn_gate: unsafe { TensorRef::from_ptr(w_attn_gate.ptr()) },
        ssm_beta: unsafe { TensorRef::from_ptr(w_ssm_beta.ptr()) },
        ssm_alpha: unsafe { TensorRef::from_ptr(w_ssm_alpha.ptr()) },
        ssm_dt: unsafe { TensorRef::from_ptr(w_ssm_dt.ptr()) },
        ssm_a: unsafe { TensorRef::from_ptr(w_ssm_a.ptr()) },
        ssm_conv1d: unsafe { TensorRef::from_ptr(w_ssm_conv1d.ptr()) },
        ssm_norm: unsafe { TensorRef::from_ptr(w_ssm_norm.ptr()) },
        ssm_out: unsafe { TensorRef::from_ptr(w_ssm_out.ptr()) },
        ..LayerWeights::EMPTY
    }
}

fn make_synthetic_state(arch: &Qwen35MoeArch, ctx: &GgmlCtx) -> GdnState {
    let d_inner = arch.ssm_d_inner as i64;
    let num_v_heads = arch.ssm_dt_rank as i64;
    let head_v_dim = d_inner / num_v_heads;
    let (conv_kernel_size, conv_channels) = conv_dims(arch);

    let state = ctx.new_tensor(sqwen::backend::ffi::GgmlType::F32, &[head_v_dim, head_v_dim, num_v_heads, 1]);
    let conv_state = ctx.new_tensor_3d(sqwen::backend::ffi::GgmlType::F32, (conv_kernel_size - 1) as i64, conv_channels, 1);

    GdnState {
        state: sqwen::backend::ctx::StateTensor::new(state),
        n_rs: 0,
        n_snap: 1,
        conv_state: sqwen::backend::ctx::StateTensor::new(conv_state),
    }
}

fn get_f32(buf: &BackendBuffer, tensor: &Tensor<'_>) -> Vec<f32> {
    let data = buf.get_tensor_data(tensor);
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4) }.to_vec()
}

/// 所有测试共用的确定性权重/输入填充。
fn ramp(len: i64, lo: f32, hi: f32) -> Vec<f32> {
    (0..len)
        .map(|i| lo + (hi - lo) * (i as f32 + 1.0) / (len as f32 + 1.0))
        .collect()
}

/// 确定性伪随机（简单 LCG），取值域 `[-0.5, 0.5)`。
///
/// 刻意不用线性 ramp：全局相关性太强的数据会让 DeltaNet 状态的方向与单 token
/// 贡献的方向趋同，从而掩盖「状态是否真的接力」。
fn pseudo(len: i64, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32) / 8388608.0 - 1.0
        })
        .map(|v| v * 0.5)
        .collect()
}

fn fill_weights(buf: &BackendBuffer, ctx: &GgmlCtx, w: &LayerWeights, arch: &Qwen35MoeArch) {
    let (conv_kernel_size, conv_channels) = conv_dims(arch);
    let n_embd = arch.n_embd as i64;
    let head_v_dim = (arch.ssm_d_inner / arch.ssm_dt_rank) as i64;
    let num_v_heads = arch.ssm_dt_rank as i64;

    buf.set_tensor_data(&w.attn_qkv.get(ctx), bytemuck::cast_slice(&pseudo(conv_channels * n_embd, 1)));
    buf.set_tensor_data(&w.attn_gate.get(ctx), bytemuck::cast_slice(&pseudo(head_v_dim * num_v_heads * n_embd, 2)));
    buf.set_tensor_data(&w.ssm_beta.get(ctx), bytemuck::cast_slice(&pseudo(num_v_heads * n_embd, 3)));
    buf.set_tensor_data(&w.ssm_alpha.get(ctx), bytemuck::cast_slice(&pseudo(num_v_heads * n_embd, 4)));
    buf.set_tensor_data(&w.ssm_dt.get(ctx), bytemuck::cast_slice(&pseudo(num_v_heads, 5)));
    // ssm_a 是 -A_log.exp()，必须为负：gate <= 0 => exp(gate) <= 1，状态不发散
    let a: Vec<f32> = pseudo(num_v_heads, 6).iter().map(|v| -v.abs() - 0.05).collect();
    buf.set_tensor_data(&w.ssm_a.get(ctx), bytemuck::cast_slice(&a));
    buf.set_tensor_data(&w.ssm_conv1d.get(ctx), bytemuck::cast_slice(&pseudo(conv_kernel_size * conv_channels, 7)));
    buf.set_tensor_data(&w.ssm_norm.get(ctx), bytemuck::cast_slice(&pseudo(head_v_dim * num_v_heads, 8)));
    buf.set_tensor_data(&w.ssm_out.get(ctx), bytemuck::cast_slice(&pseudo(head_v_dim * num_v_heads * n_embd, 9)));
}

/// 把状态张量清零。
///
/// ggml 的 backend buffer 来自对齐 malloc、不做清零，而 `build_gdn_layer` 会把
/// `st.state` / `st.conv_state` 重绑定为图内视图，所以初值必须写在调用方最初
/// 创建的那两个常驻张量上（图节点读的就是它们）。
fn zero_state(buf: &BackendBuffer, state: &Tensor<'_>, conv_state: &Tensor<'_>) {
    let n_state: usize = state.ne()[..4].iter().product::<i64>() as usize;
    buf.set_tensor_data(state, bytemuck::cast_slice(&vec![0.0f32; n_state]));
    let n_conv: usize = conv_state.ne()[..3].iter().product::<i64>() as usize;
    buf.set_tensor_data(conv_state, bytemuck::cast_slice(&vec![0.0f32; n_conv]));
}

#[test]
fn gdn_prefill_chunk_32_matches_cpu() {
    let arch = arch_test();
    let ctx1 = GgmlCtx::new_no_alloc().unwrap();
    let w = make_synthetic_weights_in_ctx(&arch, &ctx1);
    let mut st = make_synthetic_state(&arch, &ctx1);
    let h1 = ctx1.new_tensor_f32(&[64, 32]);

    let state0 = ctx1.wrap(st.state.ptr());
    let conv0 = ctx1.wrap(st.conv_state.ptr());
    let out1 = build_gdn_layer(&ctx1, &w, &h1, &mut st, 32, &arch, None);
    let buf1 = BackendBuffer::from_ctx(&ctx1, BackendKind::Cpu).expect("BackendBuffer::from_ctx");

    zero_state(&buf1, &state0, &conv0);
    fill_weights(&buf1, &ctx1, &w, &arch);
    buf1.set_tensor_data(&h1, bytemuck::cast_slice(&pseudo(64 * 32, 42)));

    let mut g1 = GraphBuilder::new(&ctx1);
    g1.expand(&out1);
    g1.compute(&buf1).unwrap();

    let vals1 = get_f32(&buf1, &out1);
    assert!(vals1.iter().all(|v| v.is_finite()), "CPU output not finite");
}

#[test]
fn gdn_decode_step_matches_cpu() {
    let arch = arch_test();
    let ctx1 = GgmlCtx::new_no_alloc().unwrap();
    let w = make_synthetic_weights_in_ctx(&arch, &ctx1);
    let mut st = make_synthetic_state(&arch, &ctx1);
    let h1 = ctx1.new_tensor_f32(&[64, 1]);

    let state0 = ctx1.wrap(st.state.ptr());
    let conv0 = ctx1.wrap(st.conv_state.ptr());
    let out1 = build_gdn_layer(&ctx1, &w, &h1, &mut st, 1, &arch, None);
    let buf1 = BackendBuffer::from_ctx(&ctx1, BackendKind::Cpu).expect("BackendBuffer::from_ctx");

    zero_state(&buf1, &state0, &conv0);
    fill_weights(&buf1, &ctx1, &w, &arch);
    buf1.set_tensor_data(&h1, bytemuck::cast_slice(&pseudo(64 * 1, 42)));

    let mut g1 = GraphBuilder::new(&ctx1);
    g1.expand(&out1);
    g1.compute(&buf1).unwrap();

    let vals1 = get_f32(&buf1, &out1);
    assert!(vals1.iter().all(|v| v.is_finite()), "CPU decode output not finite");
}

/// chunk A(16) → chunk B(16) 的输出，必须等于一次性 32-token prefill 的后半段。
///
/// 这是 chunking 正确性的核心断言：它同时验证 DeltaNet 递推状态与短卷积历史
/// 在两次调用之间正确接力。
#[test]
fn gdn_state_carry_across_chunks() {
    let arch = arch_test();
    let ctx = GgmlCtx::new_no_alloc().unwrap();
    let w = make_synthetic_weights_in_ctx(&arch, &ctx);

    // --- 一次性 32 token ---
    let mut st_full = make_synthetic_state(&arch, &ctx);
    let h_full = ctx.new_tensor_f32(&[64, 32]);
    let s_full0 = ctx.wrap(st_full.state.ptr());
    let c_full0 = ctx.wrap(st_full.conv_state.ptr());
    let out_full = build_gdn_layer(&ctx, &w, &h_full, &mut st_full, 32, &arch, None);

    // --- 两块 16 token：第二块复用第一块写回的状态 ---
    let mut st_chunk = make_synthetic_state(&arch, &ctx);
    let h_a = ctx.new_tensor_f32(&[64, 16]);
    let h_b = ctx.new_tensor_f32(&[64, 16]);
    let s_chunk0 = ctx.wrap(st_chunk.state.ptr());
    let c_chunk0 = ctx.wrap(st_chunk.conv_state.ptr());
    let _out_a = build_gdn_layer(&ctx, &w, &h_a, &mut st_chunk, 16, &arch, None);
    // 第二次调用会通过 st_chunk 读到第一块的 new_state / conv 尾部
    let out_b = build_gdn_layer(&ctx, &w, &h_b, &mut st_chunk, 16, &arch, None);

    let buf = BackendBuffer::from_ctx(&ctx, BackendKind::Cpu).expect("BackendBuffer::from_ctx");
    zero_state(&buf, &s_full0, &c_full0);
    zero_state(&buf, &s_chunk0, &c_chunk0);
    fill_weights(&buf, &ctx, &w, &arch);
    let h_data = pseudo(64 * 32, 42);
    buf.set_tensor_data(&h_full, bytemuck::cast_slice(&h_data));
    buf.set_tensor_data(&h_a, bytemuck::cast_slice(&h_data[..64 * 16]));
    buf.set_tensor_data(&h_b, bytemuck::cast_slice(&h_data[64 * 16..]));

    let mut graph = GraphBuilder::new(&ctx);
    graph.expand(&out_full);
    graph.expand(&out_b);
    graph.compute(&buf).unwrap();

    let full = get_f32(&buf, &out_full);
    let chunk = get_f32(&buf, &out_b);
    assert_eq!(full.len(), 64 * 32);
    assert_eq!(chunk.len(), 64 * 16);

    // out 形状是 [n_embd, n_tokens]，token 为最外层维度：
    // chunk 的第 i 个元素对应 full 的 i + 16*n_embd。
    let off = 16 * 64;
    let mut max_rel = 0.0f32;
    for (i, &x) in chunk.iter().enumerate() {
        let y = full[off + i];
        let d = (x - y).abs() / y.abs().max(1.0);
        if d > max_rel {
            max_rel = d;
        }
    }
    assert!(
        max_rel <= 1e-6,
        "分块 prefill 与整段 prefill 不一致：最大相对误差 {max_rel}"
    );
}

// ---------------------------------------------------------------------------
// 融合 GDN 算子的接线契约验证
// ---------------------------------------------------------------------------

/// 沿 dim2（token 维）切出从 `off_tok` 开始、长度 `ne_t` 的视图。
///
/// 这里刻意使用与 `build_gdn_layer` 相同的步长写法，因此该测试同时验证了层
/// 实现里 state / attn 视图的偏移与步长是否正确。
fn tok_view<'a>(
    ctx: &'a GgmlCtx,
    x: &Tensor<'a>,
    ne0: i64,
    ne1: i64,
    ne_t: i64,
    off_tok: i64,
) -> Tensor<'a> {
    const ELEM: usize = 4;
    ctx.wrap(unsafe {
        ffi::ggml_view_4d(
            ctx.as_ptr(),
            x.ptr(),
            ne0,
            ne1,
            ne_t,
            1,
            (ne0 as usize) * ELEM,
            (ne0 * ne1) as usize * ELEM,
            (ne0 * ne1 * ne_t) as usize * ELEM,
            (off_tok * ne0 * ne1) as usize * ELEM,
        )
    })
}

/// 直接对融合算子做分块 / 整段等价性验证，独立于层实现的其余部分：
///   * `H_k < H_v` 时的 tiled 头广播
///   * 从 op 输出切出 `new_state` 并作为下一次调用的 `state` 输入
#[test]
fn gdn_fused_chunk_equivalence() {
    let s_v: i64 = 8; // head_v_dim
    let h_v: i64 = 4; // num_v_heads
    let h_k: i64 = 2; // num_k_heads（tiled 广播到 h_v）
    let n_tok: i64 = 32;
    let cs: i64 = 16;

    let ctx = GgmlCtx::new_no_alloc().unwrap();
    let q = ctx.new_tensor_f32(&[s_v, h_k, n_tok]);
    let k = ctx.new_tensor_f32(&[s_v, h_k, n_tok]);
    let v = ctx.new_tensor_f32(&[s_v, h_v, n_tok]);
    let g = ctx.new_tensor_f32(&[1, h_v, n_tok]);
    let b = ctx.new_tensor_f32(&[1, h_v, n_tok]);
    let s0 = ctx.new_tensor_f32(&[s_v, s_v, h_v]);

    let res_full = ctx.wrap(unsafe {
        ffi::ggml_gated_delta_net(
            ctx.as_ptr(), q.ptr(), k.ptr(), v.ptr(), g.ptr(), b.ptr(), s0.ptr(), 1,
        )
    });
    let attn_full = tok_view(&ctx, &res_full, s_v, h_v, n_tok, 0);

    let qa = tok_view(&ctx, &q, s_v, h_k, cs, 0);
    let ka = tok_view(&ctx, &k, s_v, h_k, cs, 0);
    let va = tok_view(&ctx, &v, s_v, h_v, cs, 0);
    let ga = tok_view(&ctx, &g, 1, h_v, cs, 0);
    let ba = tok_view(&ctx, &b, 1, h_v, cs, 0);
    let res_a = ctx.wrap(unsafe {
        ffi::ggml_gated_delta_net(
            ctx.as_ptr(), qa.ptr(), ka.ptr(), va.ptr(), ga.ptr(), ba.ptr(), s0.ptr(), 1,
        )
    });

    let state_a = ctx.wrap(unsafe {
        ffi::ggml_view_4d(
            ctx.as_ptr(),
            res_a.ptr(),
            s_v,
            s_v,
            h_v,
            1,
            (s_v as usize) * 4,
            (s_v * s_v) as usize * 4,
            (s_v * s_v * h_v) as usize * 4,
            (s_v * h_v * cs) as usize * 4,
        )
    });

    let qb = tok_view(&ctx, &q, s_v, h_k, cs, cs);
    let kb = tok_view(&ctx, &k, s_v, h_k, cs, cs);
    let vb = tok_view(&ctx, &v, s_v, h_v, cs, cs);
    let gb = tok_view(&ctx, &g, 1, h_v, cs, cs);
    let bb = tok_view(&ctx, &b, 1, h_v, cs, cs);
    let res_b = ctx.wrap(unsafe {
        ffi::ggml_gated_delta_net(
            ctx.as_ptr(), qb.ptr(), kb.ptr(), vb.ptr(), gb.ptr(), bb.ptr(), state_a.ptr(), 1,
        )
    });
    let attn_b = tok_view(&ctx, &res_b, s_v, h_v, cs, 0);

    let buf = BackendBuffer::from_ctx(&ctx, BackendKind::Cpu).unwrap();
    buf.set_tensor_data(&q, bytemuck::cast_slice(&ramp(s_v * h_k * n_tok, -0.5, 0.5)));
    buf.set_tensor_data(&k, bytemuck::cast_slice(&ramp(s_v * h_k * n_tok, 0.5, -0.5)));
    buf.set_tensor_data(&v, bytemuck::cast_slice(&ramp(s_v * h_v * n_tok, -1.0, 1.0)));
    // gate 是 log 衰减，取非正值保证 exp(g) <= 1
    buf.set_tensor_data(&g, bytemuck::cast_slice(&ramp(h_v * n_tok, -0.5, 0.0)));
    buf.set_tensor_data(&b, bytemuck::cast_slice(&ramp(h_v * n_tok, 0.1, 0.9)));
    buf.set_tensor_data(&s0, bytemuck::cast_slice(&vec![0.0f32; (s_v * s_v * h_v) as usize]));

    let mut graph = GraphBuilder::new(&ctx);
    graph.expand(&attn_full);
    graph.expand(&attn_b);
    graph.compute(&buf).unwrap();

    let full = get_f32(&buf, &attn_full);
    let chunk = get_f32(&buf, &attn_b);
    assert_eq!(full.len(), (s_v * h_v * n_tok) as usize);
    assert_eq!(chunk.len(), (s_v * h_v * cs) as usize);

    let off = (cs * s_v * h_v) as usize;
    let mut max_rel = 0.0f32;
    for (i, &x) in chunk.iter().enumerate() {
        let y = full[off + i];
        let d = (x - y).abs() / y.abs().max(1.0);
        if d > max_rel {
            max_rel = d;
        }
    }
    assert!(
        max_rel <= 1e-5,
        "分块递推与整段递推不一致：最大相对误差 {max_rel}"
    );
}
