// src/main.rs — CLI 入口（Task 15）
//
// 子命令：
//   sqwen chat <model> --prompt "..." [--max-tokens N] [--stream] [--seed N]
//   sqwen bench <model> --prompt-tokens N --gen-tokens N
//   sqwen server <model> --port 8000 --queue-capacity N
//
// 使用 clap derive macro。

use std::sync::{Arc, Mutex};

use anyhow::Result;
use clap::{Parser, Subcommand};

use sqwen::backend::ctx::BackendKind;
use sqwen::engine::runner::{Engine, EngineConfig};
use sqwen::engine::sampler::{Sampler, SamplingParams};
use sqwen::server::api::{Generate, RealGenerate, ServerConfig, ServerState, build_router};
use sqwen::template::GgufTokenizer;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// 采样参数（各子命令共用；server 里作为请求未指定时的默认值）。
///
/// **默认值与 llama.cpp `common_params_sampling` 对齐**（`common/common.h:225-243`）：
/// temp=0.80、top_k=40、top_p=0.95、min_p=0.05、repeat_last_n=64，惩罚项中性。
/// bench 例外：不显式传 `--temp` 时用 greedy（0.0），与 llama-bench 协议一致。
#[derive(clap::Args, Clone, Debug)]
struct SamplingArgs {
    /// 采样温度；0 = greedy（默认 0.80，同 llama.cpp；bench 默认 0 = greedy）
    #[arg(long)]
    temp: Option<f32>,
    /// top-k 截断；0 = 关闭（默认 40，同 llama.cpp）
    #[arg(long = "top-k", default_value_t = sqwen::engine::sampler::DEFAULT_TOP_K)]
    top_k: u32,
    /// top-p（核采样）；1.0 = 关闭（默认 0.95，同 llama.cpp）
    #[arg(long = "top-p", default_value_t = sqwen::engine::sampler::DEFAULT_TOP_P)]
    top_p: f32,
    /// min-p；0 = 关闭（默认 0.05，同 llama.cpp）
    #[arg(long = "min-p", default_value_t = sqwen::engine::sampler::DEFAULT_MIN_P)]
    min_p: f32,
    /// 重复惩罚；1.0 = 关闭（默认 1.0，同 llama.cpp）
    #[arg(long = "repeat-penalty", default_value_t = 1.0)]
    repeat_penalty: f32,
    /// presence 惩罚；0 = 关闭
    #[arg(long = "presence-penalty", default_value_t = 0.0)]
    presence_penalty: f32,
    /// frequency 惩罚；0 = 关闭
    #[arg(long = "frequency-penalty", default_value_t = 0.0)]
    frequency_penalty: f32,
    /// 惩罚回溯窗口；0 = 关闭；-1 = 全部已生成 token（默认 64，同 llama.cpp）
    #[arg(long = "repeat-last-n", default_value_t = sqwen::engine::sampler::DEFAULT_REPEAT_LAST_N)]
    repeat_last_n: i32,
    /// 随机种子（默认 42；固定 seed + 固定参数 = 可复现输出）
    #[arg(long)]
    seed: Option<u64>,
}

impl SamplingArgs {
    /// `default_temp`：未显式传 `--temp` 时用的温度
    /// （chat/server = `DEFAULT_TEMP` 即 0.80；bench = 0.0 greedy）。
    fn to_params(&self, default_temp: f32) -> SamplingParams {
        SamplingParams {
            temp: self.temp.unwrap_or(default_temp),
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            repeat_last_n: self.repeat_last_n,
            seed: self.seed.unwrap_or(42),
            ..Default::default()
        }
    }
}

/// 批大小参数（对齐 llama.cpp 的 `-b` / `-ub`）。
///
/// - `batch_size`（逻辑）：一次提交给引擎的最大 token 数；目前用于约束 ubatch，
///   将来 server 的连续批处理也用它。
/// - `ubatch_size`（物理）：**一次前向**最多处理多少 token，prefill 按它分块建图。
/// FlashAttention 开关（P4）。
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum FlashAttnArg {
    /// 后端支持就用（默认）
    Auto,
    /// 必须用；不支持则报错
    On,
    /// 禁用（手工注意力，V cache 退回 f32）
    Off,
}

impl FlashAttnArg {
    fn to_mode(self) -> sqwen::engine::runner::FlashAttnMode {
        use sqwen::engine::runner::FlashAttnMode;
        match self {
            FlashAttnArg::Auto => FlashAttnMode::Auto,
            FlashAttnArg::On => FlashAttnMode::On,
            FlashAttnArg::Off => FlashAttnMode::Off,
        }
    }
}

/// 思考 / 推理参数（对齐 llama.cpp 的 `--reasoning-*`，P6.2）。
///
/// 这些开关**不改图、不改 kernel**，只影响两件事：
///   1. 渲染 prompt 时传给模板的变量（是否先思考、思考强度、历史里的 think 块留不留）；
///   2. 生成结果的切分（思考内容与回答分开）。
///
/// 默认值与 llama.cpp 逐条对齐：`enable_thinking=true`（`common/chat.h:261`）、
/// `reasoning_preserve=true`（`--reasoning-preserve` 的默认）。注意我们曾经硬编码
/// `enable_thinking=false`，那会让模型走"直接作答"分支 —— 与 llama.cpp 渲染出的
/// prompt 不同，属于真实的行为偏差。
#[derive(clap::Args, Clone, Debug)]
struct ReasoningArgs {
    /// 是否让模型先思考（模板变量 enable_thinking）。off = 直接作答（更快，输出更短）
    #[arg(long = "reasoning", value_enum, default_value_t = ReasoningArg::On)]
    reasoning: ReasoningArg,
    /// 思考强度（模板变量 reasoning_effort）：low | medium | xhigh
    #[arg(long = "reasoning-effort", default_value = "medium")]
    reasoning_effort: String,
    /// 渲染历史时保留上一轮 assistant 的 think 块（默认开，同 llama.cpp）
    #[arg(long = "reasoning-preserve", action = clap::ArgAction::SetTrue)]
    reasoning_preserve: bool,
    /// 关掉上面的保留（与 `--reasoning-preserve` 互斥，后者优先）
    #[arg(long = "no-reasoning-preserve", action = clap::ArgAction::SetTrue)]
    no_reasoning_preserve: bool,
    /// 思考预算（token 数）：思考写满 N 个 token 就强制输出思考结束标签转入正文。
    /// 负数 = 不限制；0 = 立刻结束思考。同 llama.cpp 的 `--reasoning-budget`
    #[arg(long = "reasoning-budget", default_value_t = -1)]
    reasoning_budget: i64,
    /// 强制结束思考前先输出的提示语（同 llama.cpp 的 `--reasoning-budget-message`）
    #[arg(long = "reasoning-budget-message", default_value = "")]
    reasoning_budget_message: String,
    /// 【server】推理内容怎么放进响应：answer=只给回答 | deepseek=推理进 reasoning_content |
    /// none=原样留在 content 里（同 llama.cpp 的 none）。CLI 的 chat 把推理写到 stderr
    #[arg(long = "reasoning-format", value_enum, default_value_t = ReasoningFormatArg::Answer)]
    reasoning_format: ReasoningFormatArg,
}

/// `--reasoning-format` 的 CLI 侧取值（映射到 `server::api::ReasoningFormat`）。
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum ReasoningFormatArg {
    Answer,
    Deepseek,
    None,
}

impl ReasoningFormatArg {
    fn to_format(self) -> sqwen::server::api::ReasoningFormat {
        use sqwen::server::api::ReasoningFormat;
        match self {
            ReasoningFormatArg::Answer => ReasoningFormat::Answer,
            ReasoningFormatArg::Deepseek => ReasoningFormat::Deepseek,
            ReasoningFormatArg::None => ReasoningFormat::None,
        }
    }
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum ReasoningArg {
    On,
    Off,
}

impl ReasoningArgs {
    fn thinking(&self) -> bool {
        self.reasoning == ReasoningArg::On
    }

    fn preserve(&self) -> bool {
        // 默认开（同 llama.cpp）；`--reasoning-preserve` 是显式确认，优先级高于 `--no-`
        if self.reasoning_preserve {
            true
        } else {
            !self.no_reasoning_preserve
        }
    }

    fn to_vars(&self) -> sqwen::template::TemplateVars {
        sqwen::template::TemplateVars {
            enable_thinking: self.thinking(),
            reasoning_effort: self.reasoning_effort.clone(),
            preserve_reasoning: self.preserve(),
            add_generation_prompt: true,
            // CLI 不走工具调用（server 才需要，按请求填）
            ..Default::default()
        }
    }
}

/// 投机解码参数（对齐 llama.cpp 的 `--spec-type` / `--spec-draft-n-max`，P5）。
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum SpecTypeArg {
    /// 关闭（默认，同 llama.cpp）
    None,
    /// 用模型自带的 MTP / NextN 头起草（`blk.{n}.nextn.*`）
    DraftMtp,
}

impl SpecTypeArg {
    fn to_type(self) -> sqwen::engine::runner::SpecType {
        use sqwen::engine::runner::SpecType;
        match self {
            SpecTypeArg::None => SpecType::None,
            SpecTypeArg::DraftMtp => SpecType::DraftMtp,
        }
    }
}

#[derive(clap::Args, Clone, Debug)]
struct SpecArgs {
    /// 投机解码类型：none = 关闭（默认）| draft-mtp = 用模型自带的 MTP 头起草。
    /// 打开后每轮起草 n_max 个 token 再让主干一次校验，接受的最长前缀直接产出
    #[arg(long = "spec-type", value_enum, default_value_t = SpecTypeArg::None)]
    spec_type: SpecTypeArg,
    /// 每轮最多起草几个 token（默认 1，见下方实测说明）。
    /// 也是递推状态回退快照的数量：每加 1 就多 `1/(n+1)` 的 GDN 状态显存。
    ///
    /// 实测（3090 / Q4_K_M / bench tg600 / ABBA）：起草 1 个 = 145~147 t/s，
    /// 2 个 = 125 t/s，3 个 = 106 t/s，不投机 = 125~130 t/s —— 每多一个草稿 token，
    /// 校验批就要多读一遍它激活的专家权重，边际成本约等于一次完整 decode 的 60%。
    #[arg(long = "spec-draft-n-max", default_value_t = 1)]
    spec_draft_n_max: usize,
    /// 深度自适应（默认开）：按实测吞吐在线决定每轮起草几个 token —— 校验批里每个
    /// 草稿 token 都要多读一遍它激活的专家权重，低接受率内容上「多起草」是净亏。
    #[arg(long = "spec-adaptive", action = clap::ArgAction::SetTrue)]
    spec_adaptive: bool,
    /// 关掉深度自适应，固定用 `--spec-draft-n-max` 的深度（旧行为，便于 A/B）
    #[arg(long = "no-spec-adaptive", action = clap::ArgAction::SetTrue)]
    no_spec_adaptive: bool,
}

impl SpecArgs {
    /// 默认开。
    fn spec_adaptive_enabled(&self) -> bool {
        self.spec_adaptive || !self.no_spec_adaptive
    }
}

#[derive(clap::Args, Clone, Debug)]
struct BatchArgs {
    /// 逻辑 batch 上限（默认 2048，同 llama.cpp）
    #[arg(short = 'b', long = "batch-size", default_value_t = 2048)]
    batch_size: i64,
    /// 物理 batch / prefill 分块大小（默认 512，同 llama.cpp）
    #[arg(short = 'u', long = "ubatch-size", default_value_t = 512)]
    ubatch_size: i64,
    /// FlashAttention（默认 auto：后端支持就开）。开时 KV 的 V 用 q4_0，KV 显存降到 1/4
    #[arg(long = "flash-attn", value_enum, default_value_t = FlashAttnArg::Auto)]
    flash_attn: FlashAttnArg,
    /// 主机侧线程数（默认 0 = 自动，取 min(逻辑核,8)）。全 GPU 模式下它不影响 GPU
    /// 计算，只影响主机侧工作 —— 目前是因果 mask 填充（大上下文下每块上百 MB）。
    #[arg(short = 't', long = "threads", default_value_t = 0)]
    threads: i64,
    /// prompt 前缀复用（跨请求记住已算过的前缀 + GDN 状态检查点）。默认开，同 llama.cpp
    #[arg(long = "cache-prompt", action = clap::ArgAction::SetTrue)]
    cache_prompt: bool,
    /// 关掉 prompt 前缀复用（与 `--cache-prompt` 互斥，后者优先）
    #[arg(long = "no-cache-prompt", action = clap::ArgAction::SetTrue)]
    no_cache_prompt: bool,
    /// 关掉启动预热。默认开（同 llama.cpp 的 `--no-warmup`）：加载完先跑一次 dummy
    /// 前向，把 CUDA 模块惰性加载 / CUDA graph capture 这些一次性成本（实测 ~340 ms）
    /// 从首字延迟里挪到加载阶段
    #[arg(long = "no-warmup", action = clap::ArgAction::SetTrue)]
    no_warmup: bool,
}

impl BatchArgs {
    /// 默认开（同 llama.cpp 的 `cache_prompt = true`）。
    fn cache_prompt_enabled(&self) -> bool {
        if self.cache_prompt {
            true
        } else {
            !self.no_cache_prompt
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// 聊天生成（单轮）
    Chat {
        model: String,
        #[arg(short, long)]
        prompt: Option<String>,
        /// 生成 token 数上限；-1 = 不限制（跑到 EOS 或把上下文写满），同 llama.cpp 默认
        #[arg(short = 'n', long = "max-tokens", default_value = "-1")]
        max_tokens: i64,
        /// 上下文长度（KV cache 容量）；0 = 用模型训练上下文（本模型 262144，注意显存）
        #[arg(short = 'c', long = "ctx-size", default_value_t = 4096)]
        ctx_size: i64,
        #[arg(long)]
        stream: bool,
        /// 自定义 chat template (Jinja2)；不指定则用 GGUF 内嵌模板
        #[arg(long = "template")]
        template_path: Option<String>,
        #[command(flatten)]
        batch: BatchArgs,
        #[command(flatten)]
        sampling: SamplingArgs,
        #[command(flatten)]
        spec: SpecArgs,
        #[command(flatten)]
        reasoning: ReasoningArgs,
    },
    /// Bench 模式：测量 TTFT / eval rate
    Bench {
        model: String,
        #[arg(long = "prompt-tokens", default_value = "1024")]
        prompt_tokens: usize,
        #[arg(long = "gen-tokens", default_value = "128")]
        gen_tokens: usize,
        /// 上下文长度；默认 0 = 只分配测试所需（`prompt + gen + 64`，与 llama-bench 一致）
        #[arg(short = 'c', long = "ctx-size", default_value_t = 0)]
        ctx_size: i64,
        /// 重复次数（第 1 次作为 warmup 丢弃；0 = 不丢弃、只跑一次）
        #[arg(short = 'r', long = "repetitions", default_value_t = 3)]
        repetitions: usize,
        /// 从文件读 prompt 的 token id（每行一个），便于与 llama.cpp 用同一 prompt 对比
        #[arg(long = "prompt-ids-file")]
        prompt_ids_file: Option<String>,
        #[command(flatten)]
        batch: BatchArgs,
        #[command(flatten)]
        sampling: SamplingArgs,
        #[command(flatten)]
        spec: SpecArgs,
    },
    /// 启动 OpenAI 兼容 API 服务器
    Server {
        model: String,
        #[arg(short, long, default_value_t = 8000)]
        port: u16,
        /// 上下文长度（KV cache 容量）；0 = 用模型训练上下文（注意显存）
        #[arg(short = 'c', long = "ctx-size", default_value_t = 4096)]
        ctx_size: i64,
        #[arg(long = "queue-capacity", default_value_t = 4)]
        queue_capacity: usize,
        /// 自定义 chat template (Jinja2)；不指定则用 GGUF 内嵌模板
        #[arg(long = "template")]
        template_path: Option<String>,
        #[command(flatten)]
        batch: BatchArgs,
        #[command(flatten)]
        sampling: SamplingArgs,
        #[command(flatten)]
        spec: SpecArgs,
        #[command(flatten)]
        reasoning: ReasoningArgs,
    },
}

fn load_engine(
    model: &str,
    params: SamplingParams,
    max_context: i64,
    batch: &BatchArgs,
    spec: &SpecArgs,
) -> Result<Engine> {
    let cfg = EngineConfig {
        max_context,
        batch_size: batch.batch_size,
        ubatch_size: batch.ubatch_size,
        flash_attn: batch.flash_attn.to_mode(),
        threads: batch.threads,
        cache_prompt: batch.cache_prompt_enabled(),
        spec_type: spec.spec_type.to_type(),
        spec_draft_n_max: spec.spec_draft_n_max,
        spec_adaptive: spec.spec_adaptive_enabled(),
        sampler: Sampler::from_params(params),
    };
    let mut eng = Engine::load(model, BackendKind::Cuda(0), cfg)?;
    // 启动预热（默认开，`--no-warmup` 关闭）。见 `Engine::warmup` 的注释：
    // 把 CUDA 模块惰性加载 / CUDA graph capture 这些一次性成本从 TTFT 挪到加载阶段。
    if !batch.no_warmup {
        eprintln!("[engine] 预热中（dummy 前向；--no-warmup 可跳过）...");
        eng.warmup()?;
    }
    Ok(eng)
}

/// 流式输出的**限流缓冲**。
///
/// # 为什么必须要它
///
/// `generate` 的 `on_token` 回调是在**生成循环内部同步调用**的，所以回调里的
/// `write_all` + `flush` 会直接串进每 token 的墙钟时间。而控制台的 flush 一点都不便宜：
/// Windows 控制台（尤其经 ConPTY 的 Windows Terminal）每次都要把数据序列化到终端
/// 进程并触发重绘。一个 token 一次 flush，足以让吞吐掉一半。
///
/// 实测：同一条命令（temp 0.1、`--spec-draft-n-max 2`）把 stdout 重定向到文件是
/// **129 t/s**，直接打到控制台只有 **~55 t/s** —— 而引擎侧的分阶段数据完全一样。
/// 之前没发现是因为所有性能测量都走管道/文件 sink。
///
/// # 策略
///
/// 攒着，满足任一条件才真正写出去：
/// - **出现了换行** —— 让整行尽快可见（代码场景基本是逐行刷新，观感不受影响）；
/// - **距上次 flush ≥ `FLUSH_MS`** —— 保证长句/无换行输出也保持"流式"的观感；
/// - **缓冲超过 `MAX_BUF`** —— 防极端情况。
///
/// 效果是把 flush 次数从「每 token 一次」降到「每行一次或每秒 ~16 次」。
struct StreamBuf {
    buf: String,
    saw_nl: bool,
    last: std::time::Instant,
}

impl StreamBuf {
    /// 无换行输出时最长攒多久（ms）。16 次/秒足够"流式"的观感，又比逐 token 少一个量级。
    const FLUSH_MS: u64 = 60;
    /// 缓冲上限（字节）。
    const MAX_BUF: usize = 8192;

    fn new() -> Self {
        Self {
            buf: String::new(),
            saw_nl: false,
            last: std::time::Instant::now(),
        }
    }

    fn push(&mut self, s: &str) {
        if s.contains('\n') {
            self.saw_nl = true;
        }
        self.buf.push_str(s);
    }

    fn due(&self) -> bool {
        self.saw_nl
            || self.buf.len() >= Self::MAX_BUF
            || self.last.elapsed() >= std::time::Duration::from_millis(Self::FLUSH_MS)
    }

    fn flush<W: std::io::Write>(&mut self, w: &mut W) {
        if !self.buf.is_empty() {
            let _ = w.write_all(self.buf.as_bytes());
            let _ = w.flush();
            self.buf.clear();
        }
        self.saw_nl = false;
        self.last = std::time::Instant::now();
    }
}

/// 统计生成结果里"思考"占了多少 token（不含结束标签本身）。
///
/// 用途：思考模式下推理会吃掉预算（实测同一 prompt 能写掉 2232 个 token），
/// 这个数字是调 `-n` 与 `--reasoning-budget` 的直接依据。走一遍 token 流即可，
/// 开销可忽略。结束标签（`</think>`）那 1~2 个 token 不计入思考 —— 它属于"收尾"。
fn count_reasoning_tokens(tok: &GgufTokenizer, tokens: &[u32], thinking: bool) -> usize {
    if !thinking {
        return 0;
    }
    let mut dec = sqwen::template::tokenizer::StreamDecoder::new();
    let mut split = sqwen::template::tokenizer::ReasoningSplitter::new(true);
    let mut counted = 0usize;
    let mut tag_tokens = 0usize;
    for &t in tokens {
        let text = dec.push(tok, t);
        if text.is_empty() {
            continue;
        }
        let was = split.in_think();
        let (_, answer) = split.push(&text);
        if was {
            counted += 1;
            if !split.in_think() {
                tag_tokens += 1; // 这个 token 完成了结束标签
            }
        } else if !answer.is_empty() {
            break;
        }
    }
    counted.saturating_sub(tag_tokens)
}

/// 组装引擎侧的思考预算配置（CLI / server 共用）。
fn reasoning_config(
    reasoning: &ReasoningArgs,
    thinking: bool,
    tmpl_src: &str,
    tok: &GgufTokenizer,
) -> sqwen::engine::reasoning::ReasoningConfig {
    let end_ids = sqwen::template::thinking_end_tag_ids(tmpl_src, tok).unwrap_or_default();
    if thinking && reasoning.reasoning_budget >= 0 && end_ids.is_empty() {
        eprintln!(
            "warning: 模板里找不到 <think>/</think> 标签，--reasoning-budget 不生效"
        );
    }
    let message_ids = if reasoning.reasoning_budget_message.is_empty() {
        Vec::new()
    } else {
        tok.encode(&reasoning.reasoning_budget_message)
    };
    sqwen::engine::reasoning::ReasoningConfig {
        enabled: thinking,
        budget: reasoning.reasoning_budget,
        end_ids,
        message_ids,
    }
}

/// 读一个"每行一个 token id"的文件（便于和 llama.cpp 跑同一个 prompt）。
///
/// 支持 `#` 注释、空行、以及逗号/空格分隔（文件里一行多个 id 也能解析）。
fn read_prompt_ids(path: &str) -> Result<Vec<u32>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("read prompt ids {}: {}", path, e))?;
    let mut ids = Vec::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("");
        for tok in line.split(|c: char| c == ',' || c.is_whitespace()) {
            if tok.is_empty() {
                continue;
            }
            let id: u32 = tok
                .parse()
                .map_err(|e| anyhow::anyhow!("prompt ids 文件 {:?} 里的 {:?} 不是合法 token id: {}", path, tok, e))?;
            ids.push(id);
        }
    }
    if ids.is_empty() {
        return Err(anyhow::anyhow!("prompt ids 文件 {} 是空的", path));
    }
    Ok(ids)
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Chat { model, prompt, max_tokens, ctx_size, stream, template_path, batch, sampling, spec, reasoning } => {
            let mut eng = load_engine(
                &model,
                sampling.to_params(sqwen::engine::sampler::DEFAULT_TEMP),
                ctx_size,
                &batch,
                &spec,
            )?;
            // 从 GGUF 加载 tokenizer + chat template
            let tokenizer = GgufTokenizer::from_gguf(&model)
                .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {}", e))?;
            eprintln!("Loaded tokenizer with {} tokens", tokenizer.len());

            let prompt_text = prompt.expect("需要 --prompt");
            // 渲染模板：优先用 --template 文件，否则用 GGUF 内嵌模板
            let msgs = vec![sqwen::template::ChatMessage::new("user", &prompt_text)];
            let tmpl_src: String = match template_path {
                Some(path) => std::fs::read_to_string(&path)
                    .map_err(|e| anyhow::anyhow!("read template {}: {}", path, e))?,
                None => tokenizer.chat_template()
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "{{ messages[0].content }}".to_string()),
            };
            // 渲染模板：优先用 --template 文件，否则用 GGUF 内嵌模板。
            // 模板变量（是否先思考等）与 llama.cpp 的默认值逐条对齐 —— 见 ReasoningArgs。
            let vars = reasoning.to_vars();
            let rendered =
                sqwen::template::render_chat_template_with(&tmpl_src, &msgs, &vars)?;
            let prompt_ids = tokenizer.encode(&rendered);
            eprintln!(
                "Prompt: \"{}\" ({} chars) -> rendered {} chars -> {} tokens | reasoning={} effort={} preserve={}",
                prompt_text,
                prompt_text.chars().count(),
                rendered.chars().count(),
                prompt_ids.len(),
                if vars.enable_thinking { "on" } else { "off" },
                vars.reasoning_effort,
                vars.preserve_reasoning,
            );
            if vars.enable_thinking && reasoning.reasoning_budget >= 0 {
                eprintln!("[reasoning] budget={} token", reasoning.reasoning_budget);
            }

            // 思考预算（--reasoning-budget）：命令在拿到模板与 tokenizer 之后才能装配
            let rcfg = reasoning_config(&reasoning, vars.enable_thinking, &tmpl_src, &tokenizer);
            eng.set_reasoning(rcfg);

            // `-n -1`（默认）＝不限制：生成到 EOS，或把上下文写满为止。
            // 与 llama.cpp 的 `-n -1` 一致 —— 之前默认 128 会让"写 2000 字"这类
            // 长输出在句子中间被硬截断，且完全没有任何提示。
            let limit = if max_tokens < 0 {
                eng.max_context().saturating_sub(prompt_ids.len())
            } else {
                max_tokens as usize
            };

            let mut output_tokens = Vec::new();
            // 流式：字节级 BPE 下一个汉字可能跨多个 token，所以先过 StreamDecoder 拿到
            // 完整的 UTF-8 片段，再过 ReasoningSplitter 把思考与回答分开
            // （思考走 stderr，回答走 stdout —— stdout 保持"只有回答"的契约）。
            let mut dec = sqwen::template::tokenizer::StreamDecoder::new();
            let mut split = sqwen::template::tokenizer::ReasoningSplitter::new(vars.enable_thinking);
            // 流式输出走限流缓冲：**逐 token flush 会让吞吐被终端速度绑死**
            // （回调在生成循环里同步执行）。详见 `StreamBuf` 的注释。
            let mut out_buf = StreamBuf::new();
            let mut err_buf = StreamBuf::new();
            let (tokens, timing) = eng.generate(&prompt_ids, limit, &mut |tok| {
                output_tokens.push(tok);
                if stream {
                    let text = dec.push(&tokenizer, tok);
                    if !text.is_empty() {
                        let (reasoning, answer) = split.push(&text);
                        if !reasoning.is_empty() {
                            err_buf.push(&reasoning);
                            if err_buf.due() {
                                err_buf.flush(&mut std::io::stderr());
                            }
                        }
                        if !answer.is_empty() {
                            out_buf.push(&answer);
                            if out_buf.due() {
                                out_buf.flush(&mut std::io::stdout());
                            }
                        }
                    }
                }
            })?;
            // 收尾：必须把攒下的都写出去，否则最后一个不满一行/不到时间的片段会丢
            if stream {
                out_buf.flush(&mut std::io::stdout());
                err_buf.flush(&mut std::io::stderr());
            }

            if !stream {
                // 非流式：解码后按「思考 / 回答」切两路（与流式共用同一个切分器）。
                // 思考走 stderr、回答走 stdout；思考没结束（-n 太小）时**不会**把整段
                // 推理当成回答打到 stdout，而是给出明确告警。
                let raw = sqwen::template::tokenizer::strip_template_markers(&tokenizer.decode(&tokens));
                let mut split =
                    sqwen::template::tokenizer::ReasoningSplitter::new(vars.enable_thinking);
                let (mut reasoning, mut answer) = split.push(&raw);
                let (r2, a2) = split.finish();
                reasoning.push_str(&r2);
                answer.push_str(&a2);
                if !reasoning.trim().is_empty() {
                    eprintln!("[推理]\n{}", reasoning.trim_end());
                }
                if answer.trim().is_empty() && vars.enable_thinking {
                    eprintln!(
                        "warning: 思考没有在给定预算内结束（没出现 </think>），没有产出回答；\
                         加大 -n 或去掉思考（--reasoning off）"
                    );
                } else {
                    println!("{}", answer.trim());
                }
            } else {
                let (reasoning, answer) = split.finish();
                let (reasoning, answer) = {
                    // finish 之后 decoder 里可能还剩半个字符
                    let rest = dec.finish();
                    let (r2, a2) = split.push(&rest);
                    (
                        format!("{reasoning}{r2}"),
                        format!("{answer}{a2}"),
                    )
                };
                if !reasoning.is_empty() {
                    eprint!("{}", reasoning);
                }
                if !answer.is_empty() {
                    print!("{}", answer);
                }
                println!();
            }

            println!("TTFT: {:.1} ms", timing.ttft_ms);
            println!("Prompt eval rate: {:.1} tok/s", timing.prompt_rate());
            if timing.prompt_cached > 0 {
                println!(
                    "Prompt cached: {} / {} tokens（--cache-prompt 命中的前缀）",
                    timing.prompt_cached, timing.prompt_tokens
                );
            }
            println!("Eval rate: {:.1} tok/s", timing.eval_rate());
            // 稳态口径（跳过冷启动段）。累计值会被开头几十秒的 GPU 时钟爬升拉低约 6%，
            // 而且两台机器/两次运行的冷启动状态不同 —— 做对比时要看这一行。
            // 生成太短（<3 秒或窗口内 token 太少）时不显示。
            if timing.gen_tokens_steady > 0 {
                println!(
                    "Eval rate (steady, 跳过冷启动后 {} tok): {:.1} tok/s",
                    timing.gen_tokens_steady,
                    timing.eval_rate_steady()
                );
            }
            println!("Generated: {} tokens", timing.gen_tokens);
            if vars.enable_thinking {
                let n_think = count_reasoning_tokens(&tokenizer, &tokens, true);
                println!(
                    "Tokens: {} 推理 + {} 回答",
                    n_think,
                    timing.gen_tokens as usize - n_think
                );
            }
            // 被 `-n` 上限截断时明确提示（否则用户只会看到句子中途断掉，无从判断原因）
            if max_tokens >= 0 && timing.gen_tokens as usize >= limit {
                eprintln!(
                    "warning: 已达到 -n {} 的上限，输出被截断（用 -n -1 或更大的值可继续）",
                    max_tokens
                );
            }
        }
        Commands::Bench {
            model,
            prompt_tokens,
            gen_tokens,
            ctx_size,
            repetitions,
            prompt_ids_file,
            batch,
            sampling,
            spec,
        } => {
            // bench 默认 greedy（temp=0），与 llama-bench 协议一致；显式 --temp 才随机采样
            let params = sampling.to_params(0.0);
            // 默认只分配测试所需上下文（llama-bench 也是 `n_ctx = n_prompt + n_gen`），
            // 这样显存对比才有意义；要测长上下文就用 -c 显式指定。
            let ctx = if ctx_size > 0 {
                ctx_size
            } else {
                (prompt_tokens + gen_tokens + 64) as i64
            };
            let mut eng = load_engine(&model, params, ctx, &batch, &spec)?;
            // bench 必须测**冷** prefill：默认开着 --cache-prompt 的话，第 2 次起会命中
            // 上一次的同一条 prompt（实测 pp8192 从 2328 虚高到 4676 t/s），数字就不可比了。
            // 与 llama-bench 的口径一致：每次都从头算。
            eng.set_cache_prompt(false);
            use sqwen::server::bench::{BenchConfig, run_bench_detailed};
            let prompt_ids = match &prompt_ids_file {
                Some(path) => Some(read_prompt_ids(path)?),
                None => None,
            };
            let cfg = BenchConfig {
                prompt_tokens,
                gen_tokens,
                repetitions,
                prompt_ids,
            };
            print!("{}", run_bench_detailed(&mut eng, cfg));
        }
        Commands::Server { model, port, ctx_size, queue_capacity, template_path, batch, sampling, spec, reasoning } => {
            let default_sampling = sampling.to_params(sqwen::engine::sampler::DEFAULT_TEMP);
            let vars = reasoning.to_vars();
            // 从 GGUF 加载 tokenizer（词表内嵌在模型文件中）。
            // 顺序上**先 tokenizer 后引擎**：思考预算的强制序列要从模板 + 词表推导出来。
            let mut tokenizer = GgufTokenizer::from_gguf(&model)
                .unwrap_or_else(|e| {
                    eprintln!("WARNING: failed to load tokenizer from GGUF: {}", e);
                    panic!("tokenizer is required for the server");
                });
            eprintln!("Loaded tokenizer with {} tokens", tokenizer.len());
            // 如果指定了 --template，覆盖 GGUF 内嵌模板
            if let Some(path) = &template_path {
                let src = std::fs::read_to_string(path)
                    .map_err(|e| anyhow::anyhow!("read template {}: {}", path, e))?;
                tokenizer.set_chat_template(src);
                eprintln!("Using custom chat template from {}", path);
            }
            let tmpl_src = tokenizer.chat_template().unwrap_or("").to_string();
            // 这条提示是有代价换来的：`--reasoning-format answer`（默认）会把推理从响应里
            // 丢掉，流式客户端在**整个思考阶段收不到一个内容字节**（实测思考能吃 4~30 秒），
            // 外部应用看起来就像"服务没有响应"。要让思考可见就用 deepseek。
            if vars.enable_thinking
                && reasoning.reasoning_format == ReasoningFormatArg::Answer
            {
                eprintln!(
                    "[server] 提示：--reasoning-format answer 会把推理从响应里丢掉 —— \
                     流式客户端在思考阶段收不到任何内容（只有 5 秒一次的 keep-alive 注释行）。\
                     想让思考跟着流出来：--reasoning-format deepseek"
                );
            }
            let mut eng = load_engine(&model, default_sampling, ctx_size, &batch, &spec)?;
            eng.set_reasoning(reasoning_config(&reasoning, vars.enable_thinking, &tmpl_src, &tokenizer));
            let eng = Arc::new(Mutex::new(eng));
            let gen: Arc<dyn Generate + Send + Sync> = Arc::new(RealGenerate(eng.clone()));
            let state = ServerState {
                gen,
                tokenizer,
                default_sampling,
                template_vars: vars,
                reasoning_format: reasoning.reasoning_format.to_format(),
            };
            let cfg = ServerConfig { port, queue_capacity };
            let router = build_router(state.clone(), cfg);
            let addr = format!("0.0.0.0:{port}");
            eprintln!("Listening on http://127.0.0.1:{port}");
            tokio::runtime::Runtime::new()?
                .block_on(async move {
                    let listener = tokio::net::TcpListener::bind(&addr)
                        .await
                        .expect("bind failed");
                    axum::serve(listener, router.with_state(state))
                        .await
                        .expect("serve ended with error");
                });
        }
    }

    Ok(())
}