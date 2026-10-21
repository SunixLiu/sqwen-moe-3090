// src/model/arch.rs — qwen35moe 架构校验
use crate::model::weights::{GgufFile, GgufError};

#[derive(Debug)]
pub struct Qwen35MoeArch {
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub ssm_d_conv: u32,
    pub ssm_d_state: u32,
    pub ssm_n_group: u32,
    pub ssm_dt_rank: u32,
    pub ssm_d_inner: u32,
    pub full_attn_interval: u32,
    pub rope_dim_count: u32,
    pub rope_freq_base: f32,
    /// IMROPE 分节（`qwen35moe.rope.dimension_sections`），四段之和 = `rope_dim_count / 2`
    pub rope_sections: [i32; 4],
    /// RMS norm 的 eps（`attention.layer_norm_rms_epsilon`）
    pub norm_rms_eps: f32,
    pub expert_ffn_len: u32,
    pub shared_expert_ffn_len: u32,
    pub n_layer_nextn: u32,
    /// 模型训练时的上下文长度（`qwen35moe.context_length`，本模型 = 262144）。
    /// `-c/--ctx-size 0` 表示用它；同时是 RoPE 的 `n_ctx_orig`（YaRN 缩放用）。
    pub n_ctx_train: u32,
    /// 全注意力头维度（attention.key_length，本架构要求 = value_length = 256）
    pub head_dim: u32,
}

impl Qwen35MoeArch {
    /// 主干层数（不含 MTP / NextN 层）。
    ///
    /// 对应 llama.cpp `llama_hparams::n_layer()`（= `n_layer_all - n_layer_nextn`）：
    /// GGUF 的 `block_count` 把 MTP 块也算进去了，例如 BigBang `block_count=41`、
    /// `nextn_predict_layers=1` → 主干 40 层（0..39），第 40 层是 MTP。
    pub fn n_layer_trunk(&self) -> u32 {
        self.n_layer.saturating_sub(self.n_layer_nextn)
    }

    /// 是否为 MTP / NextN 层（本次实现不执行，仅用于跳过其权重）。
    pub fn is_mtp_layer(&self, il: u32) -> bool {
        il >= self.n_layer_trunk()
    }

    /// 判断第 il 层是否使用全注意力（非循环/滑动窗口注意力）
    ///
    /// 对应 llama.cpp qwen35moe.cpp L23-25：
    ///   `is_recr[i] = (i < n_layer()) && ((i + 1) % full_attn_interval != 0)`
    /// 注意 MTP 块（`i >= n_layer()`）被显式标记为非循环，即全注意力。
    pub fn is_full_attn(&self, il: u32) -> bool {
        if self.is_mtp_layer(il) {
            return true;
        }
        if self.full_attn_interval == 0 {
            return false;
        }
        (il + 1) % self.full_attn_interval == 0
    }

    /// 判断第 il 层是否使用循环/滑动窗口注意力（gated delta net）。
    pub fn is_recurrent_attn(&self, il: u32) -> bool {
        !self.is_full_attn(il)
    }

    /// 全注意力头的维度（= attention.key_length = value_length）。
    pub fn head_dim(&self) -> u32 {
        self.head_dim
    }

    /// 主干中全注意力层数。
    pub fn n_full_attn_layers(&self) -> u32 {
        (0..self.n_layer_trunk())
            .filter(|&il| self.is_full_attn(il))
            .count() as u32
    }

    /// 主干中循环（GDN）注意力层数。
    pub fn n_recurrent_layers(&self) -> u32 {
        self.n_layer_trunk() - self.n_full_attn_layers()
    }
}

impl Qwen35MoeArch {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, GgufError> {
        // 1. 检查架构
        let arch = gguf.kv_get_str("general.architecture").ok_or(GgufError::MissingKey(
            "general.architecture".to_string()))?;
        if arch != "qwen35moe" {
            return Err(GgufError::ArchMismatch(format!(
                "Expected qwen35moe, got {}", arch)));
        }

        // 2. 读取核心 KV 字段
        let block_count = gguf.kv_get_u32("qwen35moe.block_count")
            .ok_or(GgufError::MissingKey("qwen35moe.block_count".to_string()))?;
        let embedding_length = gguf.kv_get_u32("qwen35moe.embedding_length")
            .ok_or(GgufError::MissingKey("qwen35moe.embedding_length".to_string()))?;
        let head_count = gguf.kv_get_u32("qwen35moe.attention.head_count")
            .ok_or(GgufError::MissingKey("qwen35moe.attention.head_count".to_string()))?;
        let head_count_kv = gguf.kv_get_u32("qwen35moe.attention.head_count_kv")
            .ok_or(GgufError::MissingKey("qwen35moe.attention.head_count_kv".to_string()))?;
        let expert_count = gguf.kv_get_u32("qwen35moe.expert_count")
            .ok_or(GgufError::MissingKey("qwen35moe.expert_count".to_string()))?;
        let expert_used_count = gguf.kv_get_u32("qwen35moe.expert_used_count")
            .ok_or(GgufError::MissingKey("qwen35moe.expert_used_count".to_string()))?;
        let key_length = gguf.kv_get_u32("qwen35moe.attention.key_length")
            .ok_or(GgufError::MissingKey("qwen35moe.attention.key_length".to_string()))?;
        let value_length = gguf.kv_get_u32("qwen35moe.attention.value_length")
            .ok_or(GgufError::MissingKey("qwen35moe.attention.value_length".to_string()))?;
        let expert_ffn_len = gguf.kv_get_u32("qwen35moe.expert_feed_forward_length")
            .ok_or(GgufError::MissingKey("qwen35moe.expert_feed_forward_length".to_string()))?;
        let shared_expert_ffn_len = gguf.kv_get_u32("qwen35moe.expert_shared_feed_forward_length")
            .ok_or(GgufError::MissingKey("qwen35moe.expert_shared_feed_forward_length".to_string()))?;
        let ssm_d_conv = gguf.kv_get_u32("qwen35moe.ssm.conv_kernel")
            .ok_or(GgufError::MissingKey("qwen35moe.ssm.conv_kernel".to_string()))?;
        let ssm_d_state = gguf.kv_get_u32("qwen35moe.ssm.state_size")
            .ok_or(GgufError::MissingKey("qwen35moe.ssm.state_size".to_string()))?;
        let ssm_n_group = gguf.kv_get_u32("qwen35moe.ssm.group_count")
            .ok_or(GgufError::MissingKey("qwen35moe.ssm.group_count".to_string()))?;
        let ssm_dt_rank = gguf.kv_get_u32("qwen35moe.ssm.time_step_rank")
            .ok_or(GgufError::MissingKey("qwen35moe.ssm.time_step_rank".to_string()))?;
        let ssm_d_inner = gguf.kv_get_u32("qwen35moe.ssm.inner_size")
            .ok_or(GgufError::MissingKey("qwen35moe.ssm.inner_size".to_string()))?;
        let full_attn_interval = gguf.kv_get_u32("qwen35moe.full_attention_interval")
            .ok_or(GgufError::MissingKey("qwen35moe.full_attention_interval".to_string()))?;
        let rope_dim_count = gguf.kv_get_u32("qwen35moe.rope.dimension_count")
            .ok_or(GgufError::MissingKey("qwen35moe.rope.dimension_count".to_string()))?;
        let rope_freq_base = gguf.kv_get_f32("qwen35moe.rope.freq_base")
            .ok_or(GgufError::MissingKey("qwen35moe.rope.freq_base".to_string()))?;
        // IMROPE 分节（四段之和 = rope_dim_count / 2）
        let mut rope_sections = [0i32; 4];
        if gguf
            .kv_get_i32_array("qwen35moe.rope.dimension_sections", &mut rope_sections)
            .is_none()
        {
            return Err(GgufError::MissingKey(
                "qwen35moe.rope.dimension_sections".to_string(),
            ));
        }
        let norm_rms_eps = gguf
            .kv_get_f32("qwen35moe.attention.layer_norm_rms_epsilon")
            .ok_or(GgufError::MissingKey(
                "qwen35moe.attention.layer_norm_rms_epsilon".to_string(),
            ))?;
        let n_layer_nextn = gguf.kv_get_u32("qwen35moe.nextn_predict_layers").unwrap_or(0);
        // 模型训练时的上下文长度（`qwen35moe.context_length`，本模型 = 262144）。
        // 用途：① `-c/--ctx-size 0` 表示"用训练上下文"（与 llama.cpp 的 n_ctx=0 一致）；
        //       ② RoPE 的 `n_ctx_orig`（YaRN 缩放用，必须填训练上下文而不是我们的 kv 容量）。
        // 缺失时退回 0，由调用方决定默认值。
        let n_ctx_train = gguf.kv_get_u32("qwen35moe.context_length").unwrap_or(0);

        // 3. 校验 head_dim == key_length == value_length == 256
        if key_length != 256 || value_length != 256 {
            return Err(GgufError::ArchMismatch(format!(
                "Expected key_length=value_length=256, got {}={}", key_length, value_length)));
        }

        // 4. 校验 MTP 层数不超过总层数
        if n_layer_nextn > block_count {
            return Err(GgufError::ArchMismatch(format!(
                "nextn_predict_layers={} > block_count={}", n_layer_nextn, block_count)));
        }

        Ok(Qwen35MoeArch {
            n_layer: block_count,
            n_embd: embedding_length,
            n_head: head_count,
            n_head_kv: head_count_kv,
            n_expert: expert_count,
            n_expert_used: expert_used_count,
            ssm_d_conv,
            ssm_d_state,
            ssm_n_group,
            ssm_dt_rank,
            ssm_d_inner,
            full_attn_interval,
            rope_dim_count,
            rope_freq_base,
            rope_sections,
            norm_rms_eps,
            expert_ffn_len,
            shared_expert_ffn_len,
            n_layer_nextn,
            n_ctx_train,
            head_dim: key_length,
        })
    }
}
