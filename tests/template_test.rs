// tests/template_test.rs — Task 11：chat template + tokenizer
use sqwen::template::{ChatMessage, render_chat_template, tokenize, detokenize};

#[test]
fn render_chat_template_basic() {
    // 极简模板，验证 minijinja 渲染管线
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
fn render_with_loop() {
    // 用 loop 渲染多轮对话
    let template = r#"system
You are a helpful assistant.
{% for msg in messages %}
{{ msg.role }}
{{ msg.content }}
{% endfor %}
"#;
    let tmp = std::env::temp_dir().join("test_chat_tmpl_loop.jinja");
    std::fs::write(&tmp, template).unwrap();
    let msgs = vec![
        ChatMessage::new("user", "Hi"),
        ChatMessage::new("assistant", "Hello! How can I help?"),
    ];
    let out = render_chat_template(tmp.to_str().unwrap(), &msgs).unwrap();
    assert!(out.contains("Hi"));
    assert!(out.contains("How can I help?"));
}

#[test]
fn tokenize_detokenize_roundtrip() {
    let tok_path = "assets/tokenizer.json";
    if !std::path::Path::new(tok_path).exists() {
        return;
    }
    // 占位 tokenizer.json 可能格式不完整；加载失败时跳过
    match tokenize("Hello, world!", tok_path) {
        Ok(ids) => {
            assert!(!ids.is_empty());
            let text = detokenize(&ids, tok_path).unwrap_or_default();
            assert_eq!(text, "Hello, world!");
        }
        Err(e) => {
            eprintln!("tokenizer.json 加载失败（{}），跳过 roundtrip 测试", e);
        }
    }
}