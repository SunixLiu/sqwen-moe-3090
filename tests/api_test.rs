// tests/api_test.rs — Task 12：OpenAI 兼容 API
use sqwen::server::api::{ChatCompletionRequest, extract_messages};
#[test]
fn extract_messages_valid() {
    let values = vec![
        serde_json::json!({"role": "user", "content": "Hello"}),
        serde_json::json!({"role": "assistant", "content": "Hi there!"}),
    ];
    let msgs = extract_messages(&values).unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0].role, "user");
    assert_eq!(msgs[1].content, "Hi there!");
}

#[test]
fn extract_messages_missing_field() {
    // 缺 `role` 仍然报错
    let values = vec![serde_json::json!({"content": "hi"})];
    assert!(extract_messages(&values).is_err());
    // 缺 `content` 现在**允许**：带 tool_calls 的 assistant 消息本来就是 content=null
    let values = vec![serde_json::json!({
        "role": "assistant",
        "tool_calls": [{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]
    })];
    let msgs = extract_messages(&values).expect("带 tool_calls 的 assistant 消息应当合法");
    assert_eq!(msgs.len(), 1);
    assert!(msgs[0].tool_calls.is_some());
    assert_eq!(msgs[0].content, "");
}

#[test]
fn request_deserialize_alias() {
    // 验证 top_k / max_tokens 别名解析
    let json = r#"{
        "model": "sqwen",
        "messages": [{"role": "user", "content": "hi"}],
        "top_k": 50,
        "max_tokens": 128,
        "stream": true,
        "seed": 42
    }"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.top_k, Some(50));
    assert_eq!(req.max_tokens, Some(128));
    assert!(req.stream.unwrap());
    assert_eq!(req.seed, Some(42));
}

#[test]
fn queue_full_returns_429() {
    use sqwen::server::queue::RequestQueue;
    let q = RequestQueue::new(1);
    // 拿满 permit
    let _g1 = q.try_acquire().unwrap();
    // 再拿应该失败
    assert!(q.try_acquire().is_err());
    drop(_g1);
    // 释放后再次可用
    assert!(q.try_acquire().is_ok());
}

// ---------------------------------------------------------------------------
// 端到端回归：真流式 + SSE 帧格式
//
// 起因（retro 会话复盘（内部资料，未随仓库发布） 的 ⛳）：
// 外部应用调 sqwen 的 server「完全没有响应」。根因两条，都是这里钉住的：
//   1. `stream: true` 是**假流式**（`await` 整段生成完再发），客户端在生成结束前
//      收不到一个字节；
//   2. 分片自己又拼了一次 `data: `，线上成了 `data: data: {...}` —— 客户端 JSON
//      解析失败，`[DONE]` 也永远匹配不上，流永远不会结束。
//
// 这个测试用「桩引擎发完第一个 token 就停住、等测试收到这一帧才继续」把第 1 条
// 变成**确定性**断言（假流式会一直等到超时），不依赖任何计时余量。
// 只用到词表（不加载权重），所以不需要 20 GiB 显存。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use sqwen::engine::sampler::SamplingParams;
use sqwen::engine::timing::Timing;
use sqwen::server::api::{
    Generate, GenerationResult, ReasoningFormat, ServerConfig, ServerState, TokenSink, build_router,
};
use sqwen::template::{GgufTokenizer, TemplateVars};

const BIGBANG: &str = r"E:\LLMModels\bigbang\bigbang-v1-i1-gguf\BigBang-v1.i1-Q4_K_M.gguf";

/// 取一段**能解出可见文本**的 token id。
///
/// 必须是可见文本：特殊 token（BOS 之类）经 `StreamDecoder` 会被丢掉，一个字都产不出来，
/// 桩引擎的闸门就永远等不到第一帧。
fn visible_ids(tok: &GgufTokenizer) -> Vec<u32> {
    use sqwen::template::tokenizer::StreamDecoder;
    let mut dec = StreamDecoder::new();
    let mut out = Vec::new();
    for id in tok.encode("Hello world") {
        if !dec.push(tok, id).is_empty() {
            out.push(id);
        }
    }
    assert!(out.len() >= 2, "没能从词表里取到两个可见 token");
    out
}

/// 桩引擎：送出第一个 token 后**卡住**，直到测试确认收到第一帧。
struct GateEngine {
    released: Arc<AtomicBool>,
    ids: Vec<u32>,
}

#[async_trait]
impl Generate for GateEngine {
    async fn generate(
        &self,
        _prompt_ids: Vec<u32>,
        _max_tokens: Option<usize>,
        _sampling: SamplingParams,
        _prompt_boundaries: Vec<usize>,
        token_sink: Option<TokenSink>,
    ) -> anyhow::Result<GenerationResult> {
        let timing = Timing {
            gen_tokens: self.ids.len() as u32,
            gen_ms: 1.0,
            ..Default::default()
        };
        // 非流式：没有 sink，一次性把 tokens 交出去
        let Some(tx) = token_sink else {
            return Ok(GenerationResult { tokens: self.ids.clone(), timing });
        };
        tx.send(self.ids[0]).expect("sink 开着");
        while !self.released.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        for t in &self.ids[1..] {
            tx.send(*t).expect("sink 开着");
        }
        Ok(GenerationResult { tokens: self.ids.clone(), timing })
    }
}

/// 读 socket 直到 `pred` 成立（或超时）。`what` 只用于报错信息。
async fn read_until(
    sock: &mut tokio::net::TcpStream,
    buf: &mut String,
    what: &str,
    pred: impl Fn(&str) -> bool,
) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    loop {
        if pred(buf) {
            return;
        }
        let mut chunk = [0u8; 4096];
        let n = tokio::time::timeout_at(deadline, sock.read(&mut chunk))
            .await
            .unwrap_or_else(|_| {
                panic!("等 {what} 超时（20 秒）。收到的内容：\n{buf}");
            })
            .expect("socket read");
        if n == 0 {
            return; // EOF
        }
        buf.push_str(&String::from_utf8_lossy(&chunk[..n]));
    }
}

/// 读到连接结束（或 20 秒超时）—— 非流式响应是一次性 body，不需要等某个标志。
async fn read_to_end(sock: &mut tokio::net::TcpStream, buf: &mut String) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    loop {
        let mut chunk = [0u8; 4096];
        match tokio::time::timeout_at(deadline, sock.read(&mut chunk)).await {
            Ok(Ok(0)) | Err(_) => return,
            Ok(Ok(n)) => buf.push_str(&String::from_utf8_lossy(&chunk[..n])),
            Ok(Err(e)) => panic!("socket read: {e}"),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_is_incremental_and_openai_framed() {
    if !std::path::Path::new(BIGBANG).exists() {
        eprintln!("[api] SKIP stream_is_incremental_and_openai_framed: 模型文件不存在");
        return;
    }
    let tokenizer = GgufTokenizer::from_gguf(BIGBANG).expect("加载词表");
    // 用真实词表取两个能解出可见文本的 token（桩引擎只发 id，靠 server 侧解码）
    let ids = visible_ids(&tokenizer);

    let released = Arc::new(AtomicBool::new(false));
    let state = ServerState {
        gen: Arc::new(GateEngine { released: released.clone(), ids: ids.clone() }),
        tokenizer,
        default_sampling: SamplingParams::default(),
        // 关掉思考：桩引擎的输出里没有 think 标签，开着思考的话整段都会被当成推理
        template_vars: TemplateVars { enable_thinking: false, ..Default::default() },
        reasoning_format: ReasoningFormat::Answer,
    };
    let router = build_router(state.clone(), ServerConfig { port: 0, queue_capacity: 4 })
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });

    let body = serde_json::json!({
        "model": "sqwen",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
        "max_tokens": 8,
    })
    .to_string();
    let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    sock.write_all(req.as_bytes()).await.unwrap();

    // ---- 第一帧必须在生成结束**之前**到达 ----
    // 桩引擎此刻正卡在第一个 token 之后不动：假流式的话这里永远等不到东西。
    let mut got = String::new();
    read_until(&mut sock, &mut got, "第一个 SSE 分片", |s| {
        s.contains(r#""finish_reason":null"#)
    })
    .await;
    assert!(
        got.contains(r#"data: {"id":"chatcmpl-"#),
        "分片没有按 OpenAI 的 SSE 形状发出：\n{got}"
    );
    assert!(
        !got.contains("data: data:"),
        "SSE 分片被加了两次 `data: ` 前缀（客户端会 JSON 解析失败、[DONE] 也匹配不上）：\n{got}"
    );
    assert!(
        !got.contains(r#""content":"""#),
        "第一个分片不该是空内容（空内容分片应该被丢弃）：\n{got}"
    );
    assert!(
        got.contains(r#""object":"chat.completion.chunk""#) && got.contains(r#""index":0"#),
        "分片缺少 OpenAI 的 object/index 字段：\n{got}"
    );

    // 放行，收剩下的
    released.store(true, Ordering::SeqCst);
    let mut rest = String::new();
    read_until(&mut sock, &mut rest, "流结束([DONE])", |s| s.contains("[DONE]")).await;
    let all = format!("{got}{rest}");
    assert!(all.contains("data: [DONE]"), "缺少 `data: [DONE]` 终止符：\n{all}");
    assert!(
        all.contains(r#""finish_reason":"stop""#),
        "缺少收尾的 finish_reason 分片：\n{all}"
    );
    assert_eq!(
        all.matches("data: [DONE]").count(),
        1,
        "[DONE] 只能出现一次：\n{all}"
    );
}

/// 非流式响应必须能被 OpenAI 的客户端读出来：`choices[0].message.content`。
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_stream_response_is_openai_shaped() {
    if !std::path::Path::new(BIGBANG).exists() {
        eprintln!("[api] SKIP non_stream_response_is_openai_shaped: 模型文件不存在");
        return;
    }
    let tokenizer = GgufTokenizer::from_gguf(BIGBANG).expect("加载词表");
    let ids = visible_ids(&tokenizer);
    let state = ServerState {
        gen: Arc::new(GateEngine { released: Arc::new(AtomicBool::new(true)), ids }),
        tokenizer,
        default_sampling: SamplingParams::default(),
        template_vars: TemplateVars { enable_thinking: false, ..Default::default() },
        reasoning_format: ReasoningFormat::Answer,
    };
    let router = build_router(state.clone(), ServerConfig { port: 0, queue_capacity: 4 })
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });

    let body = serde_json::json!({
        "model": "sqwen",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": false,
    })
    .to_string();
    let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    sock.write_all(req.as_bytes()).await.unwrap();

    let mut raw = String::new();
    read_to_end(&mut sock, &mut raw).await;
    let json_start = raw.find("\r\n\r\n").expect("HTTP 头") + 4;
    let payload: serde_json::Value =
        serde_json::from_str(raw[json_start..].trim()).expect("非流式响应必须是 JSON");
    assert_eq!(payload["object"], "chat.completion");
    assert!(
        payload["id"].as_str().unwrap_or("").starts_with("chatcmpl-"),
        "缺 id：{payload}"
    );
    assert!(payload["created"].is_u64(), "缺 created：{payload}");
    assert_eq!(payload["choices"][0]["index"], 0);
    assert_eq!(payload["choices"][0]["message"]["role"], "assistant");
    assert!(
        !payload["choices"][0]["message"]["content"].as_str().unwrap_or("").is_empty(),
        "回答为空（OpenAI 客户端会以为没响应）：{payload}"
    );
    assert_eq!(payload["choices"][0]["finish_reason"], "stop");
    // 平铺在 choice 上的老写法（没有 message 嵌套）必须不复存在
    assert!(payload["choices"][0].get("content").is_none(), "{payload}");
}
