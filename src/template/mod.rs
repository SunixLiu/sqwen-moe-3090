// src/template/mod.rs — Chat template rendering + tokenizer (Task 11)
//
// 使用 minijinja 渲染 Qwen3.6-35B-A3B 的 chat_template_v22.5.jinja，
// 用 tokenizers crate 加载 HF tokenizer.json 做 BPE encode/decode。
//
// 对应 plan Task 11：
//   - render_chat_template(template_path, messages) -> String
//   - tokenize(text, tokenizer_path) -> Vec<u32>
//   - detokenize(ids, tokenizer_path) -> String

pub mod tokenizer;
pub use tokenizer::GgufTokenizer;

/// 一条聊天消息。
#[derive(Clone, Debug, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// 助手消息里的工具调用（OpenAI `tool_calls` 原样保存）。
    ///
    /// 模板按 `message.tool_calls` 渲染（`chat_template_v22.5.jinja:337`），
    /// 同时也接受扁平的 `{name, arguments}` 与 OpenAI 的 `{function:{...}}` 两种形状。
    pub tool_calls: Option<serde_json::Value>,
    /// `role == "tool"` 时对应的调用 id（模板用它配对结果，见 `:412` 的分支）。
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

/// 渲染模板时传给 Jinja 的变量（对齐 llama.cpp 的 `common_chat_templates_inputs`）。
///
/// 为什么要有这个结构：模板里这些开关都有**缺省值**，而缺省值与 llama.cpp 的默认行为
/// 必须逐条对齐，否则同一个 prompt 在两边会渲染出不同的东西（模型行为随之不同）。
/// 实测过的一次偏差：我们曾硬编码 `enable_thinking=false`，而模板自身的缺省是 `true`
/// （`chat_template_v22.5.jinja:6`）、llama.cpp 也默认传 `true`（`common/chat.h:261`）
/// —— 于是 llama.cpp 的 generation prompt 以 `<|im_start|>assistant\n<think>\n` 结尾
/// （模型先思考），我们的以 `<think>\n\n</think>\n\n` 结尾（直接作答）。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemplateVars {
    /// 是否让模型先思考（模板变量 `enable_thinking`）。默认 **true**（同 llama.cpp）。
    pub enable_thinking: bool,
    /// 思考强度（模板变量 `reasoning_effort`）：`low` / `medium` / `xhigh`。
    /// 模板自己会把其它取值折叠到这三档，所以这里原样传字符串。
    pub reasoning_effort: String,
    /// 渲染历史时是否保留上一轮 assistant 的 think 块（模板变量 `preserve_reasoning`）。
    /// 默认 **true**（同 llama.cpp `--reasoning-preserve`）。
    pub preserve_reasoning: bool,
    /// 是否追加 generation prompt（模板以 `<|im_start|>assistant\n` 收尾）。
    /// llama.cpp 渲染生成用 prompt 时固定为 true。
    pub add_generation_prompt: bool,
    /// 工具定义（OpenAI `tools` 数组，**原样**传进模板的 `tools` 变量）。
    ///
    /// 模板用 `tool | tojson` 渲染（`chat_template_v22.5.jinja:175`），所以这里要传
    /// llama.cpp 同款形状 `{name, description, parameters}`，而不是 OpenAI 的
    /// `{type:"function", function:{...}}` 外壳 —— 否则模型看到的是另一套字段名。
    pub tools: Option<Vec<serde_json::Value>>,
    /// 工具调用格式（模板变量 `tool_call_format`）：`json` 或 `xml`（模板缺省 `xml`）。
    ///
    /// 两者都是 `<tool_call>…</tool_call>`，区别在内层：`json` 是
    /// `{"name":…,"arguments":{…}}`，`xml` 是 `<function=…><parameter=…>`。
    /// 我们按 `json` 传 —— 解析简单、和 OpenAI 结构一一对应。
    pub tool_call_format: String,
}

impl Default for TemplateVars {
    /// 与 llama.cpp 的默认值逐条一致（`common/chat.h` 的 `inputs` 默认 + 模板缺省）。
    fn default() -> Self {
        Self {
            enable_thinking: true,
            reasoning_effort: "medium".to_string(),
            preserve_reasoning: true,
            add_generation_prompt: true,
            tools: None,
            tool_call_format: "json".to_string(),
        }
    }
}

/// 渲染 chat template（Jinja2 语法），从文件路径加载模板。
pub fn render_chat_template(
    template_path: &str,
    messages: &[ChatMessage],
) -> Result<String, anyhow::Error> {
    let src = std::fs::read_to_string(template_path)?;
    render_chat_template_from_str(&src, messages)
}

/// 渲染 chat template（Jinja2 语法），直接使用模板源码字符串。
pub fn render_chat_template_from_str(
    template_src: &str,
    messages: &[ChatMessage],
) -> Result<String, anyhow::Error> {
    render_chat_template_with(template_src, messages, &TemplateVars::default())
}

/// 同 [`render_chat_template_from_str`]，但显式给出模板变量。
pub fn render_chat_template_with(
    template_src: &str,
    messages: &[ChatMessage],
    vars: &TemplateVars,
) -> Result<String, anyhow::Error> {
    let mut env = minijinja::Environment::new();

    // Jinja2 的 `tojson` 过滤器（minijinja 默认没有）。
    //
    // 模板用它把工具定义渲染成 JSON（`chat_template_v22.5.jinja:175` 的
    // `tool | tojson`，以及 `:356` 的 `arguments | tojson`）。**没有工具时走不到这些
    // 分支**，所以以前一直没暴露 —— 第一次发带 `tools` 的请求才炸。
    env.add_filter("tojson", |v: minijinja::value::Value| -> Result<String, minijinja::Error> {
        serde_json::to_string(&v).map_err(|e| {
            minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, format!("tojson: {e}"))
        })
    });

    // BigBang / Qwen 等模型的 chat template 用 Python 风格方法调用（.startswith 等）
    env.set_unknown_method_callback(|_state, value, method, args| {
        use minijinja::value::{Value, ValueKind};
        use minijinja::Error;
        if value.kind() == ValueKind::String {
            let s = value.as_str().unwrap_or("");
            match method {
                "startswith" => {
                    let prefix = args.first().and_then(|v| v.as_str()).unwrap_or("");
                    Ok(Value::from(s.starts_with(prefix)))
                }
                "endswith" => {
                    let suffix = args.first().and_then(|v| v.as_str()).unwrap_or("");
                    Ok(Value::from(s.ends_with(suffix)))
                }
                "strip" | "trim" => Ok(Value::from(s.trim().to_string())),
                "lower" => Ok(Value::from(s.to_lowercase())),
                "upper" => Ok(Value::from(s.to_uppercase())),
                "replace" => {
                    let old = args.first().and_then(|v| v.as_str()).unwrap_or("");
                    let new = args.get(1).and_then(|v| v.as_str()).unwrap_or("");
                    Ok(Value::from(s.replace(old, new)))
                }
                "split" => {
                    let sep = args.first().and_then(|v| v.as_str()).unwrap_or(" ");
                    let parts: Vec<Value> = s.split(sep).map(|p| Value::from(p.to_string())).collect();
                    Ok(Value::from(parts))
                }
                _ => Err(Error::from(minijinja::ErrorKind::UnknownMethod)),
            }
        } else {
            Err(Error::from(minijinja::ErrorKind::UnknownMethod))
        }
    });

    let msgs: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| {
            let mut v = serde_json::json!({"role": m.role, "content": m.content});
            if let Some(tc) = &m.tool_calls {
                v["tool_calls"] = tc.clone();
            }
            if let Some(id) = &m.tool_call_id {
                v["tool_call_id"] = serde_json::json!(id);
            }
            v
        })
        .collect();
    let mut ctx = serde_json::json!({
        "messages": msgs,
        // 逐条对齐 llama.cpp：这些变量**总要传**（模板里都是 "if defined" 判定，
        // 不传就退回模板缺省，两边可能不一致；显式传值才能保证同源）。
        "enable_thinking": vars.enable_thinking,
        "reasoning_effort": vars.reasoning_effort,
        "preserve_reasoning": vars.preserve_reasoning,
        // 关键：llama.cpp 渲染生成用 prompt 时固定传 add_generation_prompt=true。
        // 少了这个变量，Qwen 模板不会输出结尾的 "<|im_start|>assistant\n<think>..."，
        // 模型看到的是一个「没轮到它说话」的 prompt，输出会完全跑飞。
        "add_generation_prompt": vars.add_generation_prompt,
    });
    // 有工具才传 `tools`：模板用 `_has_tools` 判定，传了空数组也会走工具分支。
    if let Some(tools) = &vars.tools {
        if !tools.is_empty() {
            ctx["tools"] = serde_json::json!(tools);
            ctx["tool_call_format"] = serde_json::json!(vars.tool_call_format);
        }
    }
    let tmpl = env.template_from_named_str("chat", template_src)?;
    Ok(tmpl.render(&ctx)?)
}

/// 从**模板源码**推导思考结束标签（`</think>`）的 token id —— 不硬编码 token id。
///
/// 为什么扫模板而不写死：`--reasoning-budget` 要在预算耗尽时**强制**模型输出思考结束
/// 标签，而标签长什么样是模板决定的。模板的 generation prompt 分支里同时出现了
/// `<think>` 与 `</think>` 两个字面量（`chat_template_v22.5.jinja:453-459`），
/// 扫一遍模板源就能拿到它 —— 换模板/换模型都能跟上，也不会和词表脱节。
///
/// 返回 `None` 表示这个模板没有成对的思考标签 → 调用方应当**关掉**预算并说明原因，
/// 而不是套一个猜出来的标签（那会把 `</think>` 塞进不该出现的地方）。
pub fn thinking_end_tag_ids(tmpl_src: &str, tok: &GgufTokenizer) -> Option<Vec<u32>> {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    if !tmpl_src.contains(OPEN) || !tmpl_src.contains(CLOSE) {
        return None;
    }
    let ids = tok.encode(CLOSE);
    if ids.is_empty() {
        None
    } else {
        Some(ids)
    }
}

/// 用 HF tokenizer.json 把文本编码成 token id。
pub fn tokenize(text: &str, tokenizer_path: &str) -> Result<Vec<u32>, anyhow::Error> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {}", e))?;
    match tok.encode(text, false) {
        Ok(enc) => Ok(enc.get_ids().to_vec()),
        Err(e) => Err(anyhow::anyhow!("tokenize failed: {}", e)),
    }
}

/// 把 token id 解码回文本（用于 API 响应中 content 的还原）。
pub fn detokenize(ids: &[u32], tokenizer_path: &str) -> Result<String, anyhow::Error> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {}", e))?;
    tok.decode(ids, false).map_err(|e| anyhow::anyhow!("detokenize failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_simple_template() {
        // 用一段极简的 jinja template 验证渲染管线
        let template = r#"system
You are a helpful assistant.
user
{{ messages[0].content }}
assistant
"#;
        let tmp = std::env::temp_dir().join("test_chat_tmpl.jinja");
        std::fs::write(&tmp, template).unwrap();
        let msgs = vec![ChatMessage::new("user", "Hello!")];
        let out = render_chat_template(tmp.to_str().unwrap(), &msgs).unwrap();
        assert!(out.contains("Hello!"));
    }

    #[test]
    fn tokenize_detokenize_roundtrip() {
        // tokenizer.json 可能不存在或格式不完整，加载失败时跳过
        let tok_path = "assets/tokenizer.json";
        if !std::path::Path::new(tok_path).exists() {
            return;
        }
        match tokenize("Hello, world!", tok_path) {
            Ok(ids) => {
                assert!(!ids.is_empty());
                let text = detokenize(&ids, tok_path).unwrap_or_default();
                assert_eq!(text, "Hello, world!");
            }
            Err(e) => {
                eprintln!("tokenizer.json 加载失败（{}），跳过 roundtrip", e);
            }
        }
    }

    /// 用一段仿 Qwen 的极简模板钉住"模板变量确实被传进去"这件事：
    /// 思考开时 generation prompt 以 `<think>` 结尾（模型先思考），关时是成对的空块。
    #[test]
    fn thinking_var_controls_generation_prompt() {
        let tmpl = r#"{%- set enable_thinking = enable_thinking if enable_thinking is defined else true %}
{%- for m in messages %}<|im_start|>{{ m.role }}
{{ m.content }}<|im_end|>
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}
{%- if not enable_thinking %}{{- '<think>\n\n</think>\n\n' }}{%- else %}{{- '<think>\n' }}{%- endif %}
{%- endif %}"#;
        let msgs = vec![ChatMessage::new("user", "写一篇2000字的恐怖小说")];

        let on = render_chat_template_with(tmpl, &msgs, &TemplateVars::default()).unwrap();
        assert!(on.ends_with("<|im_start|>assistant\n<think>\n"), "思考开: {on:?}");
        assert!(!on.contains("</think>"));

        let off = render_chat_template_with(
            tmpl,
            &msgs,
            &TemplateVars { enable_thinking: false, ..Default::default() },
        )
        .unwrap();
        assert!(off.ends_with("<think>\n\n</think>\n\n"), "思考关: {off:?}");

        // 不带 generation prompt（打分场景）
        let plain = render_chat_template_with(
            tmpl,
            &msgs,
            &TemplateVars { add_generation_prompt: false, ..Default::default() },
        )
        .unwrap();
        assert!(!plain.contains("assistant"));
    }
}
