// tests/tokenizer_parity.rs — 与 llama.cpp 的 tokenizer 逐 id 对齐测试
//
// 需要真实模型文件（20GB，不在仓库里），用环境变量指定：
//   $env:SQWEN_TEST_MODEL = "E:\LLMModels\bigbang\bigbang-v1-i1-gguf\BigBang-v1.i1-Q4_K_M.gguf"
//   cargo test --test tokenizer_parity -- --nocapture
//
// 基线 `tests/data/tokenizer_ground_truth.json` 由 llama.cpp 的 server
// (`POST /tokenize`) 生成，是唯一的“标准答案”。

use sqwen::template::GgufTokenizer;

fn model_path() -> Option<String> {
    std::env::var("SQWEN_TEST_MODEL").ok().filter(|p| std::path::Path::new(p).exists())
}

#[test]
fn tokenizer_matches_llama_cpp() {
    let Some(model) = model_path() else {
        eprintln!("SKIP: 未设置 SQWEN_TEST_MODEL（或文件不存在），跳过与 llama.cpp 的对齐测试");
        return;
    };
    let tok = GgufTokenizer::from_gguf(&model).expect("load gguf tokenizer");
    assert_eq!(tok.len(), 248320, "vocab size 应与 GGUF tokenizer.ggml.tokens 一致");

    let fixture = std::fs::read_to_string("tests/data/tokenizer_ground_truth.json").unwrap();
    let v: serde_json::Value = serde_json::from_str(&fixture).unwrap();
    let cases = v["cases"].as_array().unwrap();

    let mut failed = 0;
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let text = case["text"].as_str().unwrap();
        let add_special = case["add_special"].as_bool().unwrap();
        let parse_special = case["parse_special"].as_bool().unwrap();
        let want: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();

        let got = tok.encode_ex(text, add_special, parse_special);
        if got != want {
            failed += 1;
            eprintln!("MISMATCH [{name}]");
            eprintln!("  text       : {text:?}");
            eprintln!("  llama.cpp  : {want:?}");
            eprintln!("  sqwen      : {got:?}");
            eprintln!("  llama pieces: {:?}", want.iter().map(|i| tok.decode(&[*i])).collect::<Vec<_>>());
            eprintln!("  sqwen pieces: {:?}", got.iter().map(|i| tok.decode(&[*i])).collect::<Vec<_>>());
        } else {
            println!("OK   [{name}] {} tokens", got.len());
        }
    }
    assert_eq!(failed, 0, "{failed} / {} 个用例与 llama.cpp 不一致", cases.len());
}

/// 解码必须能还原原文（byte-level 往返）
#[test]
fn decode_roundtrip() {
    let Some(model) = model_path() else {
        eprintln!("SKIP: 未设置 SQWEN_TEST_MODEL，跳过 decode 往返测试");
        return;
    };
    let tok = GgufTokenizer::from_gguf(&model).unwrap();
    for text in [
        "给我讲一个笑话",
        "Hello, world! How are you?",
        "Hello 世界! 1234 abc\n\nnext line",
        "def f(x):\n    return x ** 2  # comment\n",
        "😀🚀 test ✅",
        "春眠不觉晓，处处闻啼鸟。",
    ] {
        let ids = tok.encode(text);
        let back = tok.decode(&ids);
        assert_eq!(back, text, "decode(encode(text)) != text");
    }
}
