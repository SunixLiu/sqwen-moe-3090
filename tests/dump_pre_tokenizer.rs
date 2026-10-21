//! Try various key names to find the pre-tokenizer.
#[test]
fn dump_keys() {
    let path = r"E:\LLMModels\bigbang\bigbang-v1-i1-gguf\BigBang-v1.i1-Q4_K_M.gguf";
    let gguf = sqwen::model::weights::GgufFile::open(path).unwrap();
    
    let candidates = [
        "tokenizer.ggml.pre_tokenizer",
        "tokenizer.pre_tokenizer",
        "pre_tokenizer",
        "tokenizer.ggml.model",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.scores",
        "tokenizer.ggml.token_type",
        "tokenizer.chat_template",
    ];
    for key in &candidates {
        match gguf.kv_get_str(key) {
            Some(s) => {
                if s.len() > 300 {
                    println!("FOUND: {} ({} chars): {}...", key, s.len(), &s[..300]);
                } else {
                    println!("FOUND: {}: {}", key, s);
                }
            }
            None => println!("MISSING: {}", key),
        }
    }
}
