// tests/reasoning_tags.rs — 思考标签的假设（`--reasoning-budget` 的强制序列依赖它）
//
// 需要真实模型文件（20GB，不在仓库里），用环境变量指定：
//   $env:SQWEN_TEST_MODEL = "E:\LLMModels\bigbang\bigbang-v1-i1-gguf\BigBang-v1.i1-Q4_K_M.gguf"
//   cargo test --test reasoning_tags -- --nocapture
//
// 为什么值得单独钉住：预算耗尽时我们会**直接输出** `</think>` 这个 token（覆盖采样结果）。
// 如果它在词表里不是"一个独立 token"（或者解码不是这个字符串），强制就会写出乱码。
// 实测本模型：`</think>` = 单个 token 248069（user-defined），`<think>` = 248068。

use sqwen::template::GgufTokenizer;

fn model_path() -> Option<String> {
    std::env::var("SQWEN_TEST_MODEL").ok().filter(|p| std::path::Path::new(p).exists())
}

#[test]
fn think_tags_are_single_tokens() {
    let Some(model) = model_path() else {
        eprintln!("SKIP: 未设置 SQWEN_TEST_MODEL（或文件不存在）");
        return;
    };
    let tok = GgufTokenizer::from_gguf(&model).expect("load gguf tokenizer");

    for s in ["<think>", "</think>"] {
        let ids = tok.encode(s);
        assert_eq!(ids.len(), 1, "{s} 应当是单个 token，实际 {ids:?}");
        assert_eq!(
            String::from_utf8_lossy(&tok.decode_bytes(&ids)),
            s,
            "{s} 的 token 解码回来必须还是它自己"
        );
    }

    // 模板推导：模板源里有成对标签时能拿到结束标签的 id
    let tmpl = "{%- if not enable_thinking %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}";
    let end = sqwen::template::thinking_end_tag_ids(tmpl, &tok).expect("模板里有 </think>");
    assert_eq!(end, tok.encode("</think>"));
    // 没有标签的模板 → None（调用方据此关掉预算）
    assert!(sqwen::template::thinking_end_tag_ids("{{ messages[0].content }}", &tok).is_none());
}
