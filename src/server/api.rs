// src/server/api.rs — OpenAI 兼容 chat completions API（Task 12）
//
// 请求体：OpenAI chat completions 字段子集（model, messages, temperature, top_k,
// top_p, max_tokens, stream, seed）；未知字段忽略。
//
// 流程：messages → render_chat_template → tokenize → 排队（Semaphore）→
// engine.generate（stream 时**每个 token 采出来就推 SSE**；非 stream 收集全部）→ detokenize → 响应
//
// 响应形状严格按 OpenAI —— 这是「外部应用调不通」的关键，别改回去：
//
//   非流式：{id, object:"chat.completion", created, model,
//            choices:[{index, message:{role,content[,reasoning_content]}, finish_reason}],
//            附加 prompt_eval_rate / eval_rate / eval_rate_steady / ttft_ms / cached_tokens}
//     `role`/`content` **必须**嵌在 `choices[i].message` 下：OpenAI 的客户端读的是
//     `choices[0].message.content`，把 role/content 平铺在 choice 上会被严格校验的
//     SDK（pydantic）判为非法响应 —— 应用侧表现就是"完全没有响应"。
//
//   流式：每个分片 object="chat.completion.chunk"，带 id/created/index，
//        最后一个分片给 finish_reason，然后是 `data: [DONE]`。
//        `data: ` 前缀由 axum 的 `Event` 负责，**我们自己不要再加**（加了就是
//        `data: data: {...}`：客户端 JSON 解析失败，`[DONE]` 也永远匹配不上，
//        于是流永远不会结束 —— 又一个"完全没有响应"）。

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use axum::routing::{get, post};
use axum::Json;
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::engine::runner::Engine;
use crate::engine::sampler::SamplingParams;
use crate::engine::timing::Timing;
use crate::template::{ChatMessage, GgufTokenizer, render_chat_template_with};

// ---------------------------------------------------------------------------
// 请求 / 响应类型
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<serde_json::Value>,
    pub temperature: Option<f32>,
    #[serde(alias = "top_k")]
    pub top_k: Option<u32>,
    #[serde(alias = "top_p")]
    pub top_p: Option<f32>,
    #[serde(alias = "min_p")]
    pub min_p: Option<f32>,
    #[serde(alias = "repeat_penalty")]
    pub repeat_penalty: Option<f32>,
    #[serde(alias = "presence_penalty")]
    pub presence_penalty: Option<f32>,
    #[serde(alias = "frequency_penalty")]
    pub frequency_penalty: Option<f32>,
    #[serde(alias = "repeat_last_n")]
    pub repeat_last_n: Option<i32>,
    #[serde(alias = "max_tokens")]
    pub max_tokens: Option<u32>,
    pub stream: Option<bool>,
    pub seed: Option<u64>,
    /// 工具定义（OpenAI `tools`）。会原样渲染进模板的 `tools` 变量。
    pub tools: Option<Vec<serde_json::Value>>,
    /// `auto` / `none` / `required` / 具体函数。目前 `none` 会被尊重（不渲染工具）。
    pub tool_choice: Option<serde_json::Value>,
}

impl ChatCompletionRequest {
    /// 用请求里的字段覆盖服务器默认采样参数（未提供的字段保持默认）。
    pub fn sampling_params(&self, defaults: &SamplingParams) -> SamplingParams {
        let mut p = *defaults;
        if let Some(v) = self.temperature {
            p.temp = v;
        }
        if let Some(v) = self.top_k {
            p.top_k = v;
        }
        if let Some(v) = self.top_p {
            p.top_p = v;
        }
        if let Some(v) = self.min_p {
            p.min_p = v;
        }
        if let Some(v) = self.repeat_penalty {
            p.repeat_penalty = v;
        }
        if let Some(v) = self.presence_penalty {
            p.presence_penalty = v;
        }
        if let Some(v) = self.frequency_penalty {
            p.frequency_penalty = v;
        }
        if let Some(v) = self.repeat_last_n {
            p.repeat_last_n = v;
        }
        if let Some(v) = self.seed {
            p.seed = v;
        }
        p
    }
}

/// 推理内容怎么放进响应（`--reasoning-format`，P6.2）。
///
/// 取值与语义对齐 llama.cpp 的 `--reasoning-format`，但**默认值不同**：
/// llama.cpp 默认 `none`（推理原样留在 content 里），我们默认 `answer`
/// —— 因为我们的思考模式默认开启，而本模型的推理可以写掉 2000+ token，
/// 默认把推理倒进 content 会让所有 OpenAI 客户端看到满屏英文规划。
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ReasoningFormat {
    /// 只给回答：推理丢弃（默认；与 `chat` 子命令的行为一致）
    #[default]
    Answer,
    /// 推理放进 `reasoning_content`，`content` 只留回答（= llama.cpp 的 `deepseek`）
    Deepseek,
    /// 原样返回：推理连 `<think>` 标签一起留在 `content` 里（= llama.cpp 的 `none`）
    None,
}

/// 进程内自增序号：保证同一纳秒/同一毫秒内的多个请求 `id` 不撞。
static RESPONSE_SEQ: AtomicU64 = AtomicU64::new(0);

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// OpenAI 的响应身份字段。严格校验的客户端把 `id`/`created` 当**必填**，
/// 缺了直接抛错 —— 应用侧看到的就是"调不通"。
fn new_response_id() -> String {
    let seq = RESPONSE_SEQ.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("chatcmpl-{nanos:x}{seq:x}")
}

/// `finish_reason`：正好用满 `max_tokens` 就是 `length`（客户端据此决定要不要续写），
/// 其余（EOS / 上下文写满）记 `stop`。
fn finish_reason(gen_tokens: usize, max_tokens: Option<usize>) -> &'static str {
    match max_tokens {
        Some(n) if n > 0 && gen_tokens >= n => "length",
        _ => "stop",
    }
}

/// 非流式响应里的 `message`。
#[derive(Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// 工具调用（OpenAI `message.tool_calls`）。有它时 `finish_reason` 是 `tool_calls`。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    /// `stop` / `length`，见 `finish_reason()`
    pub finish_reason: &'static str,
}

/// 流式分片里的 `delta`。
#[derive(Serialize)]
pub struct Delta {
    /// 常驻字段（OpenAI 的分片里 `content` 一直有；空串对客户端是 falsy，无副作用）
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// 工具调用增量。**整块发**（一个调用一片），不做参数级增量 —— 客户端按
    /// `index` 合并即可；比逐字符增量简单得多，且 SDK 都支持。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: Delta,
    /// 只有收尾分片非空，其余是 `null`
    pub finish_reason: Option<&'static str>,
}

/// 一个 SSE 分片（OpenAI `chat.completion.chunk`）。
#[derive(Serialize)]
pub struct Chunk<'a> {
    pub id: &'a str,
    pub object: &'static str,
    pub created: u64,
    pub model: &'static str,
    pub choices: Vec<StreamChoice>,
    /// 只出现在**收尾分片**（OpenAI 的 `stream_options.include_usage` 是同一个位置）。
    ///
    /// 之前这里写的是「流式响应体里塞不下 timing，只能写 stderr」—— 那是错的：
    /// 多带几个字段不会破坏任何客户端（不认识的字段直接忽略），但少带它们，
    /// WebUI 这类客户端就只能显示 TTFT 和总耗时、算不出速率。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_rate_steady: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
}

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: &'static str,
    pub choices: Vec<Choice>,
    pub prompt_eval_rate: f64,
    pub eval_rate: f64,
    /// 稳态口径的 decode 速率（跳过冷启动段）；窗口内 token 太少时为 0。
    /// 与 llama.cpp server 的 `timings.predicted_per_second` 对比时应该看这个还是
    /// `eval_rate`，见 `engine::timing::Timing::eval_rate_steady` 的注释。
    pub eval_rate_steady: f64,
    pub ttft_ms: f64,
    /// 本次请求里被 `--cache-prompt` 直接复用（没重算）的 prompt token 数
    pub cached_tokens: u32,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// OpenAI 的 `usage` 有这三个字段；**缺 `total_tokens` 时 SDK 会解析成 None**
    /// （实测 openai-python 2.x 的 `CompletionUsage.total_tokens` 直接是 None）。
    pub total_tokens: u32,
}

#[derive(Serialize)]
pub struct ServerInfo {
    pub id: &'static str,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

// ---------------------------------------------------------------------------
// Generate trait + 实现
// ---------------------------------------------------------------------------

/// 流式 token 出口（引擎侧「采出来一个就送一个」）。
///
/// **unbounded 是有意的**：引擎的回调是在**生成循环内部同步执行**的，有界 channel
/// 会把"客户端读得慢"反馈成"模型停下来等"，等于把 CLI 侧踩过的坑（retro §2.5）
/// 搬到 server 上。慢客户端的代价由解码任务侧的（有界）SSE channel 承担。
pub type TokenSink = tokio::sync::mpsc::UnboundedSender<u32>;

/// 生成接口（测试用 StubEngine 注入）。
#[async_trait::async_trait]
pub trait Generate: Send + Sync {
    /// `max_tokens = None` 表示不限制（生成到 EOS 或上下文写满，同 llama.cpp 的 `-n -1`）。
    ///
    /// `prompt_boundaries`：prompt 里**稳定历史的末尾**（token 位置，升序）。
    /// 引擎会在这些位置存 GDN 状态检查点，下一个请求的历史只要没变就能从那儿接上
    /// （`--cache-prompt`，P6.3）。空 = 只按引擎内部的固定间隔存检查点。
    ///
    /// `token_sink`：`Some` = 流式，每个 token **采出来就送**（回调在生成循环内同步调用，
    /// 所以只能做不阻塞的事）；`None` = 一次性返回全部。
    async fn generate(
        &self,
        prompt_ids: Vec<u32>,
        max_tokens: Option<usize>,
        sampling: SamplingParams,
        prompt_boundaries: Vec<usize>,
        token_sink: Option<TokenSink>,
    ) -> Result<GenerationResult>;
}

#[derive(Clone)]
pub struct GenerationResult {
    pub tokens: Vec<u32>,
    pub timing: Timing,
}

/// 真实引擎包装。
#[derive(Clone)]
pub struct RealGenerate(pub Arc<std::sync::Mutex<Engine>>);

#[async_trait::async_trait]
impl Generate for RealGenerate {
    async fn generate(
        &self,
        prompt_ids: Vec<u32>,
        max_tokens: Option<usize>,
        sampling: SamplingParams,
        prompt_boundaries: Vec<usize>,
        token_sink: Option<TokenSink>,
    ) -> Result<GenerationResult> {
        // 克隆 Arc 而不是捕获 &self，避免 Send/Sync 问题
        let eng = self.0.clone();
        // 在后台线程执行同步的 Engine::generate，避免阻塞 tokio runtime
        let (out, timing) = tokio::task::spawn_blocking(move || {
            let mut eng = eng.lock().unwrap();
            // 每个请求用请求自带的采样参数（引擎是单实例串行执行，直接覆盖即可）
            eng.sampler_mut().set_params(sampling);
            // 历史边界（`--cache-prompt` 的检查点位置）必须与本次 generate 在同一把锁内
            // 设置，否则并发请求会互相覆盖
            eng.set_prompt_boundaries(prompt_boundaries);
            // 请求没给 max_tokens 时不限制：剩下的上下文空间就是上限
            let limit = match max_tokens {
                Some(n) => n,
                None => eng.max_context().saturating_sub(prompt_ids.len()),
            };
            // 回调在生成循环内部同步执行，所以这里只做"塞进 channel"这一件不阻塞的事
            let mut on_token = |t: u32| {
                if let Some(tx) = &token_sink {
                    let _ = tx.send(t);
                }
            };
            eng.generate(&prompt_ids, limit, &mut on_token)
        })
        .await
        .map_err(|e| anyhow::anyhow!("engine generate panicked: {e}"))??;
        Ok(GenerationResult { tokens: out, timing })
    }
}

// ---------------------------------------------------------------------------
// Server config + router
// ---------------------------------------------------------------------------

pub struct ServerConfig {
    pub port: u16,
    pub queue_capacity: usize,
}

/// 服务器状态：生成器 + tokenizer + 默认采样参数（请求未指定时用它）。
#[derive(Clone)]
pub struct ServerState {
    pub gen: Arc<dyn Generate + Send + Sync>,
    pub tokenizer: GgufTokenizer,
    pub default_sampling: SamplingParams,
    /// 渲染模板用的变量（是否先思考等），由 `--reasoning-*` 决定；P6.2。
    pub template_vars: crate::template::TemplateVars,
    /// 推理内容在响应里的呈现方式（`--reasoning-format`）。
    pub reasoning_format: ReasoningFormat,
}

/// 用给定的 Generate 实现和 tokenizer 构建 axum Router。
pub fn build_router(
    state: ServerState,
    cfg: ServerConfig,
) -> axum::Router<ServerState> {
    let queue = crate::server::queue::RequestQueue::new(cfg.queue_capacity);

    axum::Router::new()
        // OpenAI 兼容：POST /v1/chat/completions
        .route(
            "/v1/chat/completions",
            post(move |req| chat_completions(req, state.clone(), queue.clone())),
        )
        // GET /v1/models（很多应用**先列模型再聊天**，字段缺了会直接不给用）
        .route("/v1/models", get(|| async { Json(serde_json::json!({
            "object": "list",
            "data": [ServerInfo {
                id: "sqwen",
                object: "model",
                created: now_secs(),
                owned_by: "sqwen",
            }]
        }))}))
        // GET /health
        .route("/health", get(|| async { (StatusCode::OK, "ok") }))
        // WebUI：GET /
        .route("/", get(webui_root))
        // Bench：GET/POST /v1/bench
        .route("/v1/bench", post(bench_endpoint))
}

// ---------------------------------------------------------------------------
// chat_completions 处理
// ---------------------------------------------------------------------------

/// 按 `--reasoning-format` 把生成文本切成 `(reasoning_content, content)`。
///
/// - `Answer`：只要回答（推理丢弃）→ `(None, answer)`
/// - `Deepseek`：推理单独给 → `(Some(reasoning), answer)`
/// - `None`：原样（含 `<think>` 标签）→ `(None, raw)`
fn split_for_response(
    raw: &str,
    thinking: bool,
    format: ReasoningFormat,
) -> (Option<String>, String) {
    if format == ReasoningFormat::None {
        return (None, raw.to_string());
    }
    let mut split = crate::template::tokenizer::ReasoningSplitter::new(thinking);
    let (mut reasoning, mut answer) = split.push(raw);
    let (r2, a2) = split.finish();
    reasoning.push_str(&r2);
    answer.push_str(&a2);
    let answer = answer.trim().to_string();
    if format == ReasoningFormat::Deepseek {
        let reasoning = reasoning.trim().to_string();
        if reasoning.is_empty() {
            (None, answer)
        } else {
            (Some(reasoning), answer)
        }
    } else {
        (None, answer)
    }
}

/// 思考没在 `max_tokens` 内结束 → 回答是空的。CLI 侧有同样的告警（`main.rs`），
/// server 侧以前是**静默**返回 `"content": ""` —— 外部应用看到的就是"完全没响应"。
fn warn_if_answer_empty(content: &str, thinking: bool, gen_tokens: usize) {
    if thinking && content.trim().is_empty() {
        eprintln!(
            "warning: 回答为空：思考没有在 {gen_tokens} 个 token 内结束（没出现 </think>）；\
             客户端应加大 max_tokens，或用 --reasoning off / --reasoning-budget N 启动 server"
        );
    }
}

/// 流式分流器：把 `StreamDecoder` 吐出的文本按 `--reasoning-format` 切成 (推理, 回答)。
///
/// 为什么不能直接用 `ReasoningSplitter`：它为了识别 `<think>` / `</think>` 会**押住**
/// 可能是标签前缀的尾巴，所以 `finish()` 的 flush 不能省 —— 少了它最后几个字会丢。
struct StreamSplit {
    format: ReasoningFormat,
    split: crate::template::tokenizer::ReasoningSplitter,
}

impl StreamSplit {
    fn new(thinking: bool, format: ReasoningFormat) -> Self {
        Self {
            format,
            split: crate::template::tokenizer::ReasoningSplitter::new(thinking),
        }
    }

    /// 推进一段文本，返回 `(reasoning, content)`；`none` 不切分（原样进 content）。
    fn push(&mut self, text: &str) -> (String, String) {
        if self.format == ReasoningFormat::None {
            (String::new(), text.to_string())
        } else {
            self.split.push(text)
        }
    }

    /// 收尾 flush（`none` 没有可 flush 的东西）。
    fn finish(&mut self) -> (String, String) {
        if self.format == ReasoningFormat::None {
            (String::new(), String::new())
        } else {
            self.split.finish()
        }
    }

    /// 按格式决定这一段要不要发、推理要不要单列。空则返回 `None`（不发空分片）。
    fn line(&self, reasoning: String, content: String) -> Option<(String, String)> {
        if self.format == ReasoningFormat::Deepseek {
            (!reasoning.is_empty() || !content.is_empty()).then_some((reasoning, content))
        } else {
            // answer：推理丢掉（与 `split_for_response` 一致）；
            // none：上面 push 时已把原文塞进 content，这里同样原样发
            (!content.is_empty()).then_some((String::new(), content))
        }
    }
}

/// 一个 SSE 分片的 **JSON 本体**。
///
/// `data: ` 前缀由 axum 的 `Event` 加 —— 这里**不要**自己拼 `data: `，加了就是
/// `data: data: {...}`：客户端 JSON 解析失败、`[DONE]` 也永远匹配不上，
/// 流永远不会结束（"外部应用完全没有响应"的真正原因之一）。
fn chunk_json(
    id: &str,
    created: u64,
    reasoning: String,
    content: String,
    finish_reason: Option<&'static str>,
) -> String {
    let chunk = Chunk {
        id,
        object: "chat.completion.chunk",
        created,
        model: "sqwen",
        choices: vec![StreamChoice {
            index: 0,
            delta: Delta {
                content,
                reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
                tool_calls: None,
            },
            finish_reason,
        }],
        usage: None,
        prompt_eval_rate: None,
        eval_rate: None,
        eval_rate_steady: None,
        ttft_ms: None,
    };
    serde_json::to_string(&chunk).expect("chunk 可序列化")
}

/// 收尾分片：带 `finish_reason` + `usage` + 三项速率。
///
/// 为什么放在收尾分片而不是单独发一片：多一片就要多一次 `[DONE]` 判定，
/// 而收尾分片本来就要发，挂在这里对客户端是零成本的（不认识就忽略）。
fn final_chunk_json(
    id: &str,
    created: u64,
    reason: &'static str,
    timing: Option<&StreamTiming>,
    tool_calls: Option<serde_json::Value>,
) -> String {
    let chunk = Chunk {
        id,
        object: "chat.completion.chunk",
        created,
        model: "sqwen",
        choices: vec![StreamChoice {
            index: 0,
            delta: Delta {
                content: String::new(),
                reasoning_content: None,
                tool_calls,
            },
            finish_reason: Some(reason),
        }],
        usage: timing.map(|t| Usage {
            prompt_tokens: t.prompt_tokens,
            completion_tokens: t.gen_tokens,
            total_tokens: t.prompt_tokens + t.gen_tokens,
        }),
        prompt_eval_rate: timing.map(|t| t.prompt_eval_rate),
        eval_rate: timing.map(|t| t.eval_rate),
        eval_rate_steady: timing.map(|t| t.eval_rate_steady),
        ttft_ms: timing.map(|t| t.ttft_ms),
    };
    serde_json::to_string(&chunk).expect("chunk 可序列化")
}

/// 流式收尾要回传给客户端的计时（`GenerateResult::timing` 的投影）。
struct StreamTiming {
    prompt_tokens: u32,
    gen_tokens: u32,
    prompt_eval_rate: f64,
    eval_rate: f64,
    eval_rate_steady: f64,
    ttft_ms: f64,
}

async fn chat_completions(
    req: axum::extract::Json<ChatCompletionRequest>,
    state: ServerState,
    queue: crate::server::queue::RequestQueue,
) -> Response {
    let msgs = match extract_messages(&req.0.messages) {
        Ok(m) => m,
        Err(e) => return (StatusCode::BAD_REQUEST, e).into_response(),
    };

    // 模板渲染 + tokenization（在后台线程做，避免阻塞）
    // 优先使用 GGUF 内嵌的 chat template；回退到 assets/chat_template.jinja
    let tokenizer = state.tokenizer.clone();
    let tmpl_src: String = match tokenizer.chat_template() {
        Some(t) => t.to_string(),
        None => match std::fs::read_to_string("assets/chat_template.jinja") {
            Ok(s) => s,
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("template load failed: {}", e)).into_response(),
        },
    };
    let mut vars = state.template_vars.clone();
    // 工具定义：`tool_choice: "none"` 时不渲染（模板就不会进工具分支）。
    let tools_forbidden = matches!(
        req.tool_choice.as_ref().and_then(|v| v.as_str()),
        Some("none")
    );
    if let Some(tools) = &req.tools {
        if !tools.is_empty() && !tools_forbidden {
            let mapped = oai_tools_to_template(tools);
            if !mapped.is_empty() {
                vars.tools = Some(mapped);
            }
        }
    }
    // 渲染两遍：一遍带 generation prompt（真正的 prompt），一遍不带（= 稳定历史的末尾）。
    // 这两段 token 只有前者是后者的**延长**，所以历史长度就是 `--cache-prompt` 最想要的
    // 检查点位置 —— 下一个请求只要历史的渲染结果没变，就能从这里直接接上。
    let (prompt_ids, history_len) = match tokio::task::spawn_blocking(move || {
        let full = render_chat_template_with(&tmpl_src, &msgs, &vars)?;
        let history = render_chat_template_with(
            &tmpl_src,
            &msgs,
            &crate::template::TemplateVars { add_generation_prompt: false, ..vars },
        )?;
        let ids = tokenizer.encode(&full);
        let hist = tokenizer.encode(&history);
        // 只有"历史确实是 prompt 的前缀"时这个边界才有效
        let boundary = if !hist.is_empty() && ids.len() > hist.len() && ids[..hist.len()] == hist[..] {
            vec![hist.len()]
        } else {
            Vec::new()
        };
        Ok::<_, anyhow::Error>((ids, boundary))
    })
    .await
    {
        Ok(Ok(v)) => v,
        // 别把具体错误吞掉：模板渲染失败最常见的原因是「请求里的 messages / tools
        // 形状不对」，只说 "template/tokenizer error" 根本没法排查。
        Ok(Err(e)) => {
            eprintln!("[server] 模板渲染失败: {e:#}");
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("template error: {e}")).into_response();
        }
        Err(join) => {
            eprintln!("[server] 模板渲染任务失败: {join}");
            return (StatusCode::INTERNAL_SERVER_ERROR, "template task failed").into_response();
        }
    };

    // 请求没给 max_tokens 时不限制（同 llama.cpp server 的 n_predict=-1）
    let max_tokens = req.max_tokens.map(|n| n as usize);
    let stream = req.stream.unwrap_or(false);
    // 请求里的采样参数覆盖服务器默认值（temperature / top_k / top_p / min_p /
    // repeat_penalty / presence_penalty / frequency_penalty / repeat_last_n / seed）
    let sampling = req.sampling_params(&state.default_sampling);
    let thinking = state.template_vars.enable_thinking;
    let format = state.reasoning_format;

    // 排队：满则 429
    let queue_guard = match queue.try_acquire() {
        Ok(g) => g,
        Err(qf) => return qf.into_response(),
    };

    if !stream {
        let res = state
            .gen
            .generate(prompt_ids, max_tokens, sampling, history_len, None)
            .await;
        match res {
            Ok(gr) => {
                let raw = crate::template::tokenizer::strip_template_markers(
                    &state.tokenizer.decode(&gr.tokens),
                );
                let (reasoning_opt, content) = split_for_response(&raw, thinking, format);
                // 工具调用：从正文里摘出 `<tool_call>…</tool_call>` 块。
                //
                // **思考流里也要摘**：预算到点被强制 `</think>` 时，模型有可能把调用
                // 写在思考块内（实测 DSH 场景就是这样 —— 只在 content 里找，整块会留在
                // reasoning 里被客户端当正文显示出来）。
                let (reasoning, calls_r) = match reasoning_opt {
                    Some(r) => {
                        let (r2, calls) = crate::server::toolcall::extract(&r);
                        let r2 = if r2.trim().is_empty() { None } else { Some(r2) };
                        (r2, calls)
                    }
                    None => (None, Vec::new()),
                };
                let (content, calls_c) = crate::server::toolcall::extract(&content);
                let mut calls = calls_r;
                calls.extend(calls_c);
                if std::env::var_os("SQWEN_TOOL_DEBUG").is_some() {
                    eprintln!(
                        "[tool] raw={} 字 reasoning={} 字 content={} 字 calls={}",
                        raw.chars().count(),
                        reasoning.as_ref().map_or(0, |r| r.chars().count()),
                        content.chars().count(),
                        calls.len()
                    );
                }
                let tool_calls = (!calls.is_empty()).then(|| {
                    serde_json::Value::Array(calls.iter().map(|c| c.to_json()).collect())
                });
                if tool_calls.is_none() {
                    warn_if_answer_empty(&content, thinking, gr.tokens.len());
                }
                let resp = ChatCompletionResponse {
                    id: new_response_id(),
                    object: "chat.completion",
                    created: now_secs(),
                    model: "sqwen",
                    choices: vec![Choice {
                        index: 0,
                        message: Message {
                            role: "assistant".into(),
                            content,
                            reasoning_content: reasoning,
                            tool_calls,
                        },
                        finish_reason: if calls.is_empty() {
                            finish_reason(gr.tokens.len(), max_tokens)
                        } else {
                            "tool_calls"
                        },
                    }],
                    prompt_eval_rate: gr.timing.prompt_rate(),
                    eval_rate: gr.timing.eval_rate(),
                    eval_rate_steady: gr.timing.eval_rate_steady(),
                    ttft_ms: gr.timing.ttft_ms,
                    cached_tokens: gr.timing.prompt_cached,
                    usage: Usage {
                        prompt_tokens: gr.timing.prompt_tokens,
                        completion_tokens: gr.timing.gen_tokens,
                        total_tokens: gr.timing.prompt_tokens + gr.timing.gen_tokens,
                    },
                };
                Json(resp).into_response()
            }
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        }
    } else {
        // ---- 流式：**真**流式（每个 token 采出来就往外走）----
        //
        // 两条流水线，中间用 channel 解耦：
        //     引擎 --(token id, **unbounded**)--> 解码任务 --(SSE 行, 有界)--> HTTP
        //
        // 1) token 通道必须 unbounded：引擎侧回调是在**生成循环内部同步执行**的，
        //    有界 channel 会把"客户端读得慢"反馈成模型阻塞（CLI 侧踩过同一个坑，
        //    见 `main.rs` 的 StreamBuf）。慢客户端的代价放在下面那条有界 channel 上。
        // 2) 解码 / 分流（StreamDecoder / ReasoningSplitter）放在**生成循环之外**：
        //    它每个 token 都要干活，塞进回调就等于把输出端的开销算进吞吐（retro §2.5）。
        //
        // 旧写法是 `await` 整段生成完、再遍历 tokens 发 SSE —— 客户端在生成**全部**
        // 结束前收不到一个字节（1000 token ≈ 8 秒，外加 prefill），外部应用看到的就是
        // "完全没有响应"。**别改回去。**
        let (tok_tx, mut tok_rx) = tokio::sync::mpsc::unbounded_channel::<u32>();
        let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);
        let gen2 = state.gen.clone();
        let tokenizer2 = state.tokenizer.clone();

        tokio::spawn(async move {
            // 生成任务：`generate` 返回时 sink 被 drop → `tok_rx` 结束 → 解码任务收尾。
            // 队列 permit 跟着**生成**走：流式响应在 `chat_completions` 返回之后还在
            // 继续生成，permit 若在函数返回时就释放，`--queue-capacity` 对流式请求形同虚设。
            let gen_task = tokio::spawn(async move {
                let _guard = queue_guard;
                gen2.generate(prompt_ids, max_tokens, sampling, history_len, Some(tok_tx))
                    .await
            });

            let id = new_response_id();
            let created = now_secs();
            let mut dec = crate::template::tokenizer::StreamDecoder::new();
            let mut split = StreamSplit::new(thinking, format);
            // 工具调用：正文与**思考**都要先过切分器 —— 预算强制结束后，模型可能把
            // 调用写在思考块内；只过滤 content 的话整块会留在 reasoning 里被客户端显示。
            let mut tc_split = crate::server::toolcall::ToolCallSplitter::new();
            let mut tc_split_r = crate::server::toolcall::ToolCallSplitter::new();
            // 客户端断开后继续把 channel 抽干（否则 unbounded 会一直涨），但不再发送
            let mut alive = true;
            // 有没有真的往外发过**回答**（`deepseek` 下推理分片不算）
            let mut sent_content = false;
            while let Some(tok) = tok_rx.recv().await {
                let text = dec.push(&tokenizer2, tok);
                if text.is_empty() {
                    continue;
                }
                let (r, a) = split.push(&text);
                if let Some((r, a)) = split.line(r, a) {
                    let vis_r = tc_split_r.push(&r);
                    let vis_a = tc_split.push(&a);
                    sent_content |= !vis_a.is_empty();
                    if alive && (!vis_r.is_empty() || !vis_a.is_empty()) {
                        alive = tx.send(chunk_json(&id, created, vis_r, vis_a, None)).await.is_ok();
                    }
                }
            }

            // 收尾 flush：解码器可能还押着半个字符，分流器可能押着标签前缀，
            // 两者都**必须**无条件 flush（押着不发就是丢字）。
            let tail = dec.finish();
            let mut reasoning = String::new();
            let mut content = String::new();
            if !tail.is_empty() {
                let (r, a) = split.push(&tail);
                reasoning.push_str(&r);
                content.push_str(&a);
            }
            let (r, a) = split.finish();
            reasoning.push_str(&r);
            content.push_str(&a);
            let flushed = (reasoning.chars().count(), content.chars().count());
            if let Some((r, a)) = split.line(reasoning, content) {
                let vis_r = tc_split_r.push(&r);
                let vis_a = tc_split.push(&a);
                sent_content |= !vis_a.is_empty();
                if alive && (!vis_r.is_empty() || !vis_a.is_empty()) {
                    alive = tx.send(chunk_json(&id, created, vis_r, vis_a, None)).await.is_ok();
                }
            }

            // 工具调用收尾：两个切分器里可能都还押着块内内容（含未闭合的情况）
            let (tc_tail, mut calls) = tc_split.finish();
            let (tc_tail_r, calls_r) = tc_split_r.finish();
            calls.extend(calls_r);
            if alive && !tc_tail_r.is_empty() {
                alive = tx.send(chunk_json(&id, created, tc_tail_r, String::new(), None)).await.is_ok();
            }
            if alive && !tc_tail.is_empty() {
                alive = tx.send(chunk_json(&id, created, String::new(), tc_tail, None)).await.is_ok();
            }
            let tool_calls = (!calls.is_empty()).then(|| {
                serde_json::Value::Array(calls.iter().map(|c| c.to_json()).collect())
            });
            if std::env::var_os("SQWEN_TOOL_DEBUG").is_some() {
                eprintln!(
                    "[tool] (stream) reasoning={} 字 content={} 字 calls={}",
                    flushed.0,
                    flushed.1,
                    calls.len()
                );
            }

            // 生成早已结束（`tok_rx` 关闭就是因为 sink 被 drop），这里只是取结果
            let mut err: Option<String> = None;
            let mut reason = "stop";
            let mut timing: Option<StreamTiming> = None;
            match gen_task.await {
                Ok(Ok(gr)) => {
                    reason = if tool_calls.is_some() {
                        // 有工具调用就报 tool_calls（OpenAI 的约定），而不是 stop
                        "tool_calls"
                    } else {
                        finish_reason(gr.tokens.len(), max_tokens)
                    };
                    timing = Some(StreamTiming {
                        prompt_tokens: gr.timing.prompt_tokens,
                        gen_tokens: gr.tokens.len() as u32,
                        prompt_eval_rate: gr.timing.prompt_rate(),
                        eval_rate: gr.timing.eval_rate(),
                        eval_rate_steady: gr.timing.eval_rate_steady(),
                        ttft_ms: gr.timing.ttft_ms,
                    });
                    // 思考吃掉了整个 max_tokens → 回答一个字都没有。以前这里**静默**，
                    // 客户端只会看到"没有任何内容"，看起来就是服务没响应。
                    // 有工具调用时不算「空回答」。
                    if !sent_content && tool_calls.is_none() {
                        warn_if_answer_empty("", thinking, gr.tokens.len());
                    }
                    if std::env::var_os("SQWEN_QUIET").is_none() {
                        // 流式响应体里塞不下 timing（OpenAI 的分片结构），只能写 stderr
                        eprintln!(
                            "[stream] {} tok | prompt {} tok @ {:.1} t/s | eval {:.1} t/s | steady {:.1} t/s | ttft {:.0} ms",
                            gr.tokens.len(),
                            gr.timing.prompt_tokens,
                            gr.timing.prompt_rate(),
                            gr.timing.eval_rate(),
                            gr.timing.eval_rate_steady(),
                            gr.timing.ttft_ms,
                        );
                    }
                }
                Ok(Err(e)) => err = Some(e.to_string()),
                Err(join) => err = Some(format!("generate task failed: {join}")),
            }
            if alive {
                match &err {
                    // 头已经发出去了（HTTP 200），状态码改不了 —— 按 OpenAI 的做法发一个
                    // error 对象再收尾，比直接掐断连接信息量大得多
                    Some(msg) => {
                        let body = serde_json::json!({
                            "error": { "message": msg, "type": "server_error" }
                        });
                        let _ = tx.send(body.to_string()).await;
                    }
                    None => {
                        let _ = tx
                            .send(final_chunk_json(
                                &id,
                                created,
                                reason,
                                timing.as_ref(),
                                tool_calls.clone(),
                            ))
                            .await;
                    }
                }
                let _ = tx.send("[DONE]".to_string()).await;
            }
        });

        let sse = Sse::new(
            ReceiverStream::new(rx)
                .map(|data| Ok::<_, std::convert::Infallible>(Event::default().data(data))),
        )
        // 默认每 15 秒一个空注释行；这里缩到 5 秒 —— `--reasoning-format answer` 下
        // 思考阶段一个内容字节都没有（实测思考可以吃掉几秒到几十秒），注释行是唯一
        // 能挡住"客户端读取超时"的东西（按 SSE 规范，注释行必须被客户端忽略）。
        .keep_alive(KeepAlive::default().interval(Duration::from_secs(5)));
        sse.into_response()
    }
}

pub fn extract_messages(values: &[serde_json::Value]) -> Result<Vec<ChatMessage>, String> {
    let mut msgs = Vec::new();
    for v in values {
        let role = v.get("role").and_then(|x| x.as_str()).ok_or("missing role")?;
        // `content` 允许为 null（带 tool_calls 的 assistant 消息就是这样），也允许缺省。
        let content = match v.get("content") {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Null) | None => String::new(),
            // 少数客户端把 content 发成数组（多段），这里拼成纯文本
            Some(serde_json::Value::Array(parts)) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join(""),
            Some(other) => other.to_string(),
        };
        let mut m = ChatMessage::new(role, content);
        if let Some(tc) = v.get("tool_calls") {
            if !tc.is_null() {
                m.tool_calls = Some(tc.clone());
            }
        }
        if let Some(id) = v.get("tool_call_id").and_then(|x| x.as_str()) {
            m.tool_call_id = Some(id.to_string());
        }
        msgs.push(m);
    }
    Ok(msgs)
}

/// 把 OpenAI 的 `tools` 转成模板要的形状：`{name, description, parameters}`。
///
/// 为什么要转：模板是 `tool | tojson`（`chat_template_v22.5.jinja:175`），
/// llama.cpp 传的就是这三个字段（`common_chat_tool`）。直接塞 OpenAI 的
/// `{type:"function", function:{...}}` 外壳会让模型看到另一套字段名。
pub fn oai_tools_to_template(tools: &[serde_json::Value]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .filter_map(|t| {
            let f = t.get("function").unwrap_or(t);
            let name = f.get("name").and_then(|x| x.as_str())?;
            let mut out = serde_json::json!({ "name": name });
            if let Some(d) = f.get("description").and_then(|x| x.as_str()) {
                out["description"] = serde_json::json!(d);
            }
            // parameters 在模板里是 JSON Schema 对象；llama.cpp 那边是字符串，但
            // `tojson` 之后两者渲染出来一样，这里保留对象更直观。
            if let Some(p) = f.get("parameters") {
                out["parameters"] = p.clone();
            }
            Some(out)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// WebUI + Bench（占位，Task 13 补充）
// ---------------------------------------------------------------------------

async fn webui_root() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("webui.html"))
}

async fn bench_endpoint(
    axum::extract::State(state): axum::extract::State<ServerState>,
    params: axum::extract::Query<serde_json::Value>,
) -> Response {
    let prompt_tokens = params.get("prompt_tokens").and_then(|x| x.as_u64()).unwrap_or(1024) as u32;
    let gen_tokens = params.get("gen_tokens").and_then(|x| x.as_u64()).unwrap_or(128) as u32;
    // bench 走 greedy，避免采样开销混进吞吐数字
    let sampling = SamplingParams { temp: 0.0, ..state.default_sampling };
    match state
        .gen
        .generate(
            vec![0u32; prompt_tokens as usize],
            Some(gen_tokens as usize),
            sampling,
            Vec::new(),
            None,
        )
        .await
    {
        Ok(gr) => Json(serde_json::json!({
            "prompt_eval_rate": gr.timing.prompt_rate(),
            "eval_rate": gr.timing.eval_rate(),
            "eval_rate_steady": gr.timing.eval_rate_steady(),
            "ttft_ms": gr.timing.ttft_ms,
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

#[cfg(test)]
mod response_format_tests {
    use super::*;

    /// 思考开启时：`deepseek` 把推理分出去，`answer` 丢掉推理，`none` 原样保留。
    #[test]
    fn split_for_response_covers_all_formats() {
        let raw = "先想一下\n</think>\n\n答案在此";
        let (r, c) = split_for_response(raw, true, ReasoningFormat::Deepseek);
        assert_eq!(c, "答案在此");
        assert_eq!(r.as_deref(), Some("先想一下"));

        let (r, c) = split_for_response(raw, true, ReasoningFormat::Answer);
        assert_eq!(c, "答案在此");
        assert!(r.is_none());

        let (r, c) = split_for_response(raw, true, ReasoningFormat::None);
        assert_eq!(c, raw, "none = 原样返回（含标签）");
        assert!(r.is_none());
    }

    /// 没开思考时没有 `</think>`，整段都是回答。
    #[test]
    fn split_for_response_without_thinking() {
        let raw = "# 镜中客\n\n林默搬进这栋老式公寓…";
        let (r, c) = split_for_response(raw, false, ReasoningFormat::Deepseek);
        assert_eq!(c, raw);
        assert!(r.is_none(), "没有推理时不应出现 reasoning_content 字段");
    }

    /// 思考没结束时（预算被 -n 截断）不能把推理塞进 content。
    #[test]
    fn split_for_response_with_unclosed_thinking() {
        let (r, c) = split_for_response("还在想\n没想完", true, ReasoningFormat::Deepseek);
        assert_eq!(c, "", "没有回答");
        assert_eq!(r.as_deref(), Some("还在想\n没想完"));
    }

    #[test]
    fn chunk_json_carries_reasoning_only_when_present() {
        let d = chunk_json("chatcmpl-x", 1, "想".into(), "答".into(), None);
        assert!(d.contains(r#""reasoning_content":"想""#), "{d}");
        assert!(d.contains(r#""content":"答""#), "{d}");
        let d = chunk_json("chatcmpl-x", 1, String::new(), "答".into(), None);
        assert!(!d.contains("reasoning_content"), "空推理不应出现该字段：{d}");
    }

    /// SSE 分片的**帧**由 axum 的 `Event` 加，我们只给 JSON 本体。
    /// 以前这里拼了 `data: `，线上就成了 `data: data: {...}`：客户端 JSON 解析失败、
    /// `[DONE]` 永远匹配不上 —— 这是"外部应用完全没有响应"的复现根因。
    #[test]
    fn chunk_json_is_bare_json_without_sse_prefix() {
        let d = chunk_json("chatcmpl-x", 1, String::new(), "答".into(), None);
        assert!(!d.contains("data:"), "分片本体不能自带 data: 前缀：{d}");
        let v: serde_json::Value = serde_json::from_str(&d).expect("分片必须是合法 JSON");
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["id"], "chatcmpl-x");
        assert_eq!(v["choices"][0]["index"], 0);
        assert!(v["choices"][0]["finish_reason"].is_null(), "内容分片没有 finish_reason");
    }

    /// 收尾分片：`finish_reason` 非空、delta 为空。
    #[test]
    fn chunk_json_finish_reason() {
        let d = chunk_json("chatcmpl-x", 1, String::new(), String::new(), Some("length"));
        let v: serde_json::Value = serde_json::from_str(&d).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "length");
    }

    /// `finish_reason`：用满 max_tokens 是 `length`，其余是 `stop`。
    #[test]
    fn finish_reason_maps_max_tokens() {
        assert_eq!(finish_reason(10, Some(10)), "length");
        assert_eq!(finish_reason(9, Some(10)), "stop");
        assert_eq!(finish_reason(100, None), "stop", "不限制 max_tokens 时不该报 length");
        assert_eq!(finish_reason(0, Some(0)), "stop");
    }

    /// 非流式响应必须是 OpenAI 的形状：`choices[0].message.content`（**嵌套**），
    /// 外加 id / created / index / finish_reason。平铺 role/content 的老写法会被
    /// 严格校验的 SDK 直接判为非法响应。
    #[test]
    fn non_stream_response_shape_is_openai() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-x".into(),
            object: "chat.completion",
            created: 42,
            model: "sqwen",
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".into(),
                    content: "答".into(),
                    reasoning_content: Some("想".into()),
                    tool_calls: None,
                },
                finish_reason: "stop",
            }],
            prompt_eval_rate: 1.0,
            eval_rate: 2.0,
            eval_rate_steady: 3.0,
            ttft_ms: 4.0,
            cached_tokens: 0,
            usage: Usage { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        };
        let v: serde_json::Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        assert_eq!(v["choices"][0]["message"]["content"], "答");
        assert_eq!(v["choices"][0]["message"]["reasoning_content"], "想");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["choices"][0]["index"], 0);
        assert!(v["created"].is_u64());
        assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
        // 附加的计时口径不能被弄丢（对比 llama.cpp 时要看它们）
        assert_eq!(v["eval_rate_steady"], 3.0);
    }

    /// 流式分流：`answer` 丢推理、`deepseek` 单列推理、`none` 原样；
    /// 并且 `finish()` 必须把分流器押住的尾巴吐出来。
    #[test]
    fn stream_split_formats_and_flush() {
        // deepseek：推理进 reasoning
        let mut s = StreamSplit::new(true, ReasoningFormat::Deepseek);
        let (r, a) = s.push("先想一下\n</think>\n\n答案");
        let (r, a) = {
            let (r2, a2) = s.finish();
            (format!("{r}{r2}"), format!("{a}{a2}"))
        };
        assert_eq!(r.trim(), "先想一下");
        assert_eq!(a.trim(), "答案");
        assert!(s.line(r, a).is_some());

        // answer：推理在 `line()` 里被丢掉（`push` 只是原样转出切分结果）
        let mut s = StreamSplit::new(true, ReasoningFormat::Answer);
        let (r, a) = s.push("先想一下\n</think>\n\n答案");
        let (r, a) = s.line(r, a).expect("有回答就该发");
        assert!(r.is_empty(), "answer 格式不该给 reasoning：{r}");
        assert!(a.contains("答案"), "{a}");
        assert!(s.line(String::new(), String::new()).is_none(), "空分片不发");

        // none：不切分，原样进 content
        let mut s = StreamSplit::new(true, ReasoningFormat::None);
        let (r, a) = s.push("<think>原文</think>");
        assert!(r.is_empty());
        assert_eq!(a, "<think>原文</think>");

        // 尾巴 flush：切分器为了识别标签会押住可能是前缀的字符
        let mut s = StreamSplit::new(true, ReasoningFormat::Answer);
        let (r, a) = s.push("答</thi");
        let (r2, a2) = s.finish();
        let text = format!("{r}{a}{r2}{a2}");
        assert!(text.contains("答"), "尾巴丢了：{text}");
    }
}
