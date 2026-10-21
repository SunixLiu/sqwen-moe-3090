// src/template/tokenizer.rs — llama.cpp 兼容的 GGUF tokenizer（GPT-2 字节级 BPE）
//
// 为什么不能用「贪心最长匹配」：
//   Qwen 系列 GGUF 的 `tokenizer.ggml.model` 是 `gpt2`（BPE），词表里的 token 是
//   **GPT-2 字节编码**后的字符串（空格 → `Ġ` U+0120，换行 → `Ċ` U+010A，非 ASCII
//   文本被拆成“每个 UTF-8 字节一个字符”），合并顺序由 `tokenizer.ggml.merges`
//   的 rank 唯一决定。另外 `<|im_start|>` 这类控制 token 必须先在原文本里切出来
//   整体输出（`tokenizer_st_partition`），否则会被拆成 `<` `|` `i` `m` ...
//
// 参考实现（D:\01Projects\llama.cpp\src）：
//   * `llama-vocab.cpp`
//       - `llm_tokenizer_bpe_session::tokenize`       预分词 + 按 rank 合并
//       - `llama_vocab::impl::tokenizer_st_partition` 特殊 token 切分
//       - `llama_vocab::impl::load`                   merges / token_type / 特殊 token 缓存
//       - `token_to_piece` / `detokenize`             解码（字节还原）
//   * `unicode.cpp`
//       - `unicode_byte_to_utf8_map` / `unicode_byte_encoding_process`  GPT-2 字节映射
//       - `unicode_regex_split_custom_qwen35`                           预分词器（自定义实现）
//
// 复刻程度：同一段文本产生的 token id 与 llama.cpp **逐 id 一致**
//           （见 tests/tokenizer_parity.rs）。

use std::collections::{BinaryHeap, HashMap};
use std::sync::OnceLock;

use regex::Regex;

// ---------------------------------------------------------------------------
// 常量：与 include/llama.h 的 enum 对齐
// ---------------------------------------------------------------------------

/// `llama_token_type`
pub const TT_UNDEFINED: i32 = 0;
pub const TT_NORMAL: i32 = 1;
pub const TT_UNKNOWN: i32 = 2;
pub const TT_CONTROL: i32 = 3;
pub const TT_USER_DEFINED: i32 = 4;
pub const TT_UNUSED: i32 = 5;
pub const TT_BYTE: i32 = 6;

/// `llama_token_attr`
pub const ATTR_UNKNOWN: u16 = 1 << 0;
pub const ATTR_UNUSED: u16 = 1 << 1;
pub const ATTR_NORMAL: u16 = 1 << 2;
pub const ATTR_CONTROL: u16 = 1 << 3;
pub const ATTR_USER_DEFINED: u16 = 1 << 4;
pub const ATTR_BYTE: u16 = 1 << 5;
pub const ATTR_LSTRIP: u16 = 1 << 7;
pub const ATTR_RSTRIP: u16 = 1 << 8;

fn toktype_to_attr(t: i32) -> u16 {
    match t {
        TT_UNKNOWN => ATTR_UNKNOWN,
        TT_UNUSED => ATTR_UNUSED,
        TT_NORMAL => ATTR_NORMAL,
        TT_CONTROL => ATTR_CONTROL,
        TT_USER_DEFINED => ATTR_USER_DEFINED,
        TT_BYTE => ATTR_BYTE,
        // TT_UNDEFINED / 未知：llama.cpp 给 LLAMA_TOKEN_ATTR_UNDEFINED（= 0，无属性）
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// GPT-2 字节编码表（unicode.cpp: unicode_byte_to_utf8_map）
// ---------------------------------------------------------------------------

/// byte → char。cp932 之外的所有字节都能映射成一个合法 Unicode 标量。
fn byte_to_char_table() -> &'static [char; 256] {
    static TABLE: OnceLock<[char; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut map = ['\0'; 256];
        let mut used = [false; 256];
        let mut mark = |b: u32| {
            // b 一定 < 256
            map[b as usize] = char::from_u32(b).unwrap();
            used[b as usize] = true;
        };
        for b in 0x21..=0x7E {
            mark(b);
        }
        for b in 0xA1..=0xAC {
            mark(b);
        }
        for b in 0xAE..=0xFF {
            mark(b);
        }
        let mut n: u32 = 0;
        for b in 0..256usize {
            if !used[b] {
                map[b] = char::from_u32(0x100 + n).unwrap();
                n += 1;
            }
        }
        map
    })
}

/// char → byte（逆表）
fn char_to_byte_table() -> &'static HashMap<char, u8> {
    static TABLE: OnceLock<HashMap<char, u8>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let fwd = byte_to_char_table();
        let mut rev = HashMap::with_capacity(256);
        for (b, c) in fwd.iter().enumerate() {
            rev.insert(*c, b as u8);
        }
        rev
    })
}

/// `unicode_byte_encoding_process`：把整段文本按 UTF-8 字节逐个映射。
fn byte_encode_str(s: &str) -> String {
    let table = byte_to_char_table();
    let mut out = String::with_capacity(s.len());
    for b in s.as_bytes() {
        out.push(table[*b as usize]);
    }
    out
}

/// `llama_decode_text`：把字节编码后的 token 文本还原成原始字节。
fn byte_decode_str(s: &str) -> Vec<u8> {
    let table = char_to_byte_table();
    let mut out = Vec::with_capacity(s.len());
    for c in s.chars() {
        match table.get(&c) {
            Some(b) => out.push(*b),
            None => {
                // llama.cpp 遇到表外字符会拼接 "[UNK_BYTE_0x..]" 调试串；
                // 正常词表不会出现，这里退化成 UTF-8 原字节，保证不会 panic。
                let mut buf = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Unicode 字符类别（对应 llama.cpp 的 unicode_cpt_flags）
// ---------------------------------------------------------------------------

fn re_letter() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\p{L}").unwrap())
}
fn re_number() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\p{N}").unwrap())
}
fn re_accent() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\p{M}").unwrap())
}

fn is_letter(c: char) -> bool {
    re_letter().is_match(c.encode_utf8(&mut [0u8; 4]))
}
fn is_number(c: char) -> bool {
    re_number().is_match(c.encode_utf8(&mut [0u8; 4]))
}
fn is_accent(c: char) -> bool {
    re_accent().is_match(c.encode_utf8(&mut [0u8; 4]))
}

// ---------------------------------------------------------------------------
// 预分词器类型（对应 llama.cpp 的 llama_vocab_pre_type）
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreType {
    /// `qwen35`：`\p{L}\p{M}`，数字单字
    Qwen35,
    /// `qwen2` / `deepseek-r1-qwen` / `kormo` / `f2llmv2`
    Qwen2,
    /// `llama3` / `llama-bpe` 等：数字 1..3 个一组
    Llama3,
    /// 其它（未实现精确预分词器，退化为 Qwen2 规则并告警）
    Unsupported,
}

impl PreType {
    pub fn from_gguf(pre: &str) -> Self {
        match pre {
            "qwen35" => PreType::Qwen35,
            "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" => PreType::Qwen2,
            "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "pixtral"
            | "midm-2.0" | "lfm2" | "jina-v5-nano" | "dbrx" | "smaug-bpe" | "seed-coder"
            | "grok-2" | "ufakezeka" | "laguna" | "bailingmoe" => PreType::Llama3,
            _ => PreType::Unsupported,
        }
    }

    /// 字母类里是否带 `\p{M}`（combining mark）
    fn accent_marks(self) -> bool {
        matches!(self, PreType::Qwen35)
    }

    /// 数字一次吃几个：0 = `\p{N}`（单个），3 = `\p{N}{1,3}`
    fn digit_run(self) -> usize {
        match self {
            PreType::Llama3 => 3,
            _ => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// GgufTokenizer
// ---------------------------------------------------------------------------

/// 一条被切出来的片段：普通文本 or 直接命中的特殊 token
enum Frag {
    Raw(String),
    Tok(u32),
}

#[derive(Debug, Clone)]
struct Bigram {
    rank: i32,
    left: usize,
    right: usize,
    text: String,
}

impl PartialEq for Bigram {
    fn eq(&self, o: &Self) -> bool {
        self.rank == o.rank && self.left == o.left
    }
}
impl Eq for Bigram {}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Bigram {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        // llama.cpp `llm_bigram_bpe::comparator`：rank 小的先合并；rank 相同靠左的先合并。
        // BinaryHeap 是最大堆，所以这里把顺序反过来。
        o.rank.cmp(&self.rank).then_with(|| o.left.cmp(&self.left))
    }
}

/// 从 GGUF 文件构建的 tokenizer（GPT-2 字节级 BPE + 特殊 token 切分）。
#[derive(Clone)]
pub struct GgufTokenizer {
    /// token 文本 → id
    vocab: HashMap<String, u32>,
    /// id → token 文本（与 GGUF 数组下标严格一致，不能跳项）
    ids: Vec<String>,
    /// id → llama_token_attr
    attrs: Vec<u16>,
    /// (left, right) → rank
    ranks: HashMap<String, HashMap<String, i32>>,
    /// 特殊 token（按文本长度降序，与 llama.cpp cache_special_tokens 相同）
    special: Vec<(String, u32)>,
    pre: PreType,
    /// 是否使用 GPT-2 字节编码（gpt2 BPE 为 true）
    byte_encode: bool,
    add_bos: bool,
    add_eos: bool,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    /// chat template（Jinja2 语法，从 GGUF `tokenizer.chat_template` 提取）
    chat_template: Option<String>,
}

impl GgufTokenizer {
    /// 从 GGUF 文件加载词表 / merges / 特殊 token / chat template。
    pub fn from_gguf(gguf_path: &str) -> Result<Self, anyhow::Error> {
        use crate::model::weights::GgufFile;
        let gguf = GgufFile::open(gguf_path)
            .map_err(|e| anyhow::anyhow!("failed to open GGUF: {}", e))?;

        let tokens = gguf
            .kv_get_str_array("tokenizer.ggml.tokens")
            .ok_or_else(|| anyhow::anyhow!("GGUF missing tokenizer.ggml.tokens"))?;
        if tokens.is_empty() {
            return Err(anyhow::anyhow!("GGUF tokenizer vocabulary is empty"));
        }

        let mut vocab = HashMap::with_capacity(tokens.len());
        for (i, tok) in tokens.iter().enumerate() {
            vocab.insert(tok.clone(), i as u32);
        }

        // token_type → attr；缺失时 llama.cpp 全部按 NORMAL 处理
        let types = gguf.kv_get_i32_array_vec("tokenizer.ggml.token_type");
        let attrs: Vec<u16> = (0..tokens.len())
            .map(|i| match &types {
                Some(t) if i < t.len() => toktype_to_attr(t[i]),
                _ => ATTR_NORMAL,
            })
            .collect();

        // merges：GGUF 里每项形如 "left right"，从下标 1 开始找第一个空格
        // （llama-vocab.cpp: tokenizer_model == "gpt2" 分支）
        let merge_strs = gguf.kv_get_str_array("tokenizer.ggml.merges").unwrap_or_default();
        let mut ranks: HashMap<String, HashMap<String, i32>> = HashMap::new();
        for (i, m) in merge_strs.iter().enumerate() {
            let bytes = m.as_bytes();
            if bytes.len() < 3 {
                continue;
            }
            if let Some(rel) = bytes[1..].iter().position(|&b| b == b' ') {
                let pos = rel + 1;
                let first = &m[..pos];
                let second = &m[pos + 1..];
                ranks
                    .entry(first.to_string())
                    .or_default()
                    .insert(second.to_string(), i as i32);
            }
        }

        // 特殊 token 缓存：attr 含 CONTROL / USER_DEFINED / UNKNOWN 的 token，
        // 按文本长度降序（长 token 先切，避免被短 token 抢走前缀）
        let mut special: Vec<(String, u32)> = attrs
            .iter()
            .enumerate()
            .filter(|(_, a)| **a & (ATTR_CONTROL | ATTR_USER_DEFINED | ATTR_UNKNOWN) != 0)
            .map(|(i, _)| (tokens[i].clone(), i as u32))
            .collect();
        special.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let model = gguf.kv_get_str("tokenizer.ggml.model").unwrap_or_default();
        let pre_str = gguf.kv_get_str("tokenizer.ggml.pre").unwrap_or_default();
        let pre = PreType::from_gguf(&pre_str);
        let is_bpe = model == "gpt2" || model == "whitespace" || model == "hybriddna";
        if !is_bpe {
            eprintln!(
                "WARNING: tokenizer.ggml.model='{}' 不是 gpt2 BPE，当前实现按字节级 BPE 处理",
                model
            );
        }
        if pre == PreType::Unsupported {
            eprintln!(
                "WARNING: 未实现 chat template 对应的 pre-tokenizer '{}'，回退到 qwen2 规则（token 可能与 llama.cpp 不一致）",
                pre_str
            );
        }

        let chat_template = gguf.kv_get_str("tokenizer.chat_template");
        if let Some(ref tmpl) = chat_template {
            eprintln!("Loaded chat template from GGUF ({} chars)", tmpl.chars().count());
        } else {
            eprintln!("WARNING: no tokenizer.chat_template in GGUF, will use fallback");
        }

        let add_bos = gguf.kv_get_bool("tokenizer.ggml.add_bos_token").unwrap_or(false);
        let add_eos = gguf.kv_get_bool("tokenizer.ggml.add_eos_token").unwrap_or(false);
        let bos_id = gguf.kv_get_u32("tokenizer.ggml.bos_token_id");
        let eos_id = gguf.kv_get_u32("tokenizer.ggml.eos_token_id");

        Ok(GgufTokenizer {
            vocab,
            ids: tokens,
            attrs,
            ranks,
            special,
            pre,
            byte_encode: is_bpe,
            add_bos,
            add_eos,
            bos_id,
            eos_id,
            chat_template,
        })
    }

    /// 词表大小。
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// 返回 GGUF 内嵌的 chat template（Jinja2），如果存在。
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    /// 覆盖 chat template（用于 --template 命令行参数）。
    pub fn set_chat_template(&mut self, src: String) {
        self.chat_template = Some(src);
    }

    pub fn pre_type(&self) -> PreType {
        self.pre
    }

    /// 该 id 是否是“特殊 token”（CONTROL / USER_DEFINED / UNKNOWN）。
    pub fn is_special(&self, id: u32) -> bool {
        self.attrs
            .get(id as usize)
            .map(|a| a & (ATTR_CONTROL | ATTR_USER_DEFINED | ATTR_UNKNOWN) != 0)
            .unwrap_or(false)
    }

    // -- 编码 ----------------------------------------------------------------

    /// 编码。等价于 llama.cpp `common_tokenize(text, add_special=true, parse_special=true)`。
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_ex(text, true, true)
    }

    /// 编码（可控制是否加 BOS/EOS、是否解析特殊 token）。
    pub fn encode_ex(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<u32> {
        let frags = self.partition(text, parse_special);
        let mut out = Vec::new();

        if add_special && self.add_bos {
            if let Some(bos) = self.bos_id {
                out.push(bos);
            }
        }
        for frag in frags {
            match frag {
                Frag::Tok(id) => out.push(id),
                Frag::Raw(s) => self.encode_raw(&s, &mut out),
            }
        }
        if add_special && self.add_eos {
            if let Some(eos) = self.eos_id {
                out.push(eos);
            }
        }
        out
    }

    /// `tokenizer_st_partition`：把原文本按特殊 token 切成 Raw / Tok 片段。
    fn partition(&self, text: &str, parse_special: bool) -> Vec<Frag> {
        let mut frags: Vec<Frag> = Vec::new();
        if text.is_empty() {
            return frags;
        }
        frags.push(Frag::Raw(text.to_string()));

        for (tok_text, tok_id) in &self.special {
            if tok_text.is_empty() {
                continue;
            }
            let attr = self.attrs[*tok_id as usize];
            // parse_special == false 时跳过 CONTROL/UNKNOWN；USER_DEFINED 始终参与切分
            if !parse_special && (attr & (ATTR_CONTROL | ATTR_UNKNOWN)) != 0 {
                continue;
            }

            let mut next_frags: Vec<Frag> = Vec::with_capacity(frags.len() + 2);
            for frag in frags.drain(..) {
                let s = match frag {
                    Frag::Tok(id) => {
                        next_frags.push(Frag::Tok(id));
                        continue;
                    }
                    Frag::Raw(s) => s,
                };

                let mut rest: &str = &s;
                loop {
                    let pos = match rest.find(tok_text.as_str()) {
                        Some(p) => p,
                        None => break,
                    };
                    let mut left: &str = &rest[..pos];
                    let mut right: &str = &rest[pos + tok_text.len()..];
                    if attr & ATTR_LSTRIP != 0 {
                        left = left.trim_end_matches(char::is_whitespace);
                    }
                    if attr & ATTR_RSTRIP != 0 {
                        right = right.trim_start_matches(char::is_whitespace);
                    }
                    if !left.is_empty() {
                        next_frags.push(Frag::Raw(left.to_string()));
                    }
                    next_frags.push(Frag::Tok(*tok_id));
                    rest = right;
                }
                if !rest.is_empty() {
                    next_frags.push(Frag::Raw(rest.to_string()));
                }
            }
            frags = next_frags;
        }
        frags
    }

    /// 预分词 + 字节编码 + BPE 合并。
    fn encode_raw(&self, text: &str, out: &mut Vec<u32>) {
        let cpts: Vec<char> = text.chars().collect();
        if cpts.is_empty() {
            return;
        }
        let n = cpts.len();

        // 逐字符类别（对应 unicode_cpt_flags）
        struct Flags {
            letter: bool,
            number: bool,
            accent: bool,
            space: bool,
        }
        let accent_marks = self.pre.accent_marks();
        let flags: Vec<Flags> = cpts
            .iter()
            .map(|&c| Flags {
                letter: is_letter(c),
                number: is_number(c),
                accent: accent_marks && is_accent(c),
                space: c.is_whitespace(),
            })
            .collect();
        let digit_run = self.pre.digit_run();

        // 预分词结果：码点区间 [start, end)
        let mut pieces: Vec<(usize, usize)> = Vec::new();
        let mut last_end: usize = 0;
        let push = |pieces: &mut Vec<(usize, usize)>, last_end: &mut usize, end: usize| {
            if end > *last_end {
                pieces.push((*last_end, end));
            }
            *last_end = end;
        };

        let mut pos: usize = 0;
        while pos < n {
            let cpt = cpts[pos];
            let f = &flags[pos];

            // ---- regex: (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
            if cpt == '\'' && pos + 1 < n {
                let c1 = cpts[pos + 1].to_ascii_lowercase();
                if matches!(c1, 's' | 't' | 'm' | 'd') {
                    pos += 2;
                    push(&mut pieces, &mut last_end, pos);
                    continue;
                }
                if pos + 2 < n {
                    let c2 = cpts[pos + 2].to_ascii_lowercase();
                    if (c1 == 'r' && c2 == 'e')
                        || (c1 == 'v' && c2 == 'e')
                        || (c1 == 'l' && c2 == 'l')
                    {
                        pos += 3;
                        push(&mut pieces, &mut last_end, pos);
                        continue;
                    }
                }
            }

            // ---- regex: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
            if !(cpt == '\r' || cpt == '\n' || f.number) {
                let next_ok = flags
                    .get(pos + 1)
                    .map(|g| g.accent || g.letter)
                    .unwrap_or(false);
                if f.letter || f.accent || next_ok {
                    pos += 1;
                    while pos < n && (flags[pos].letter || flags[pos].accent) {
                        pos += 1;
                    }
                    push(&mut pieces, &mut last_end, pos);
                    continue;
                }
            }

            // ---- regex: \p{N} / \p{N}{1,3}
            if f.number {
                let mut taken = 0usize;
                while taken < digit_run && pos < n && flags[pos].number {
                    pos += 1;
                    taken += 1;
                }
                push(&mut pieces, &mut last_end, pos);
                continue;
            }

            // ---- regex: <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
            let f2 = if cpt == ' ' {
                flags.get(pos + 1)
            } else {
                Some(f)
            };
            let (f2_ws, f2_letter, f2_accent, f2_number) = match f2 {
                Some(g) => (g.space, g.letter, g.accent, g.number),
                None => (false, false, false, false),
            };
            let f2_defined = if cpt == ' ' { pos + 1 < n } else { true };
            if !(f2_ws || f2_letter || f2_accent || f2_number) && f2_defined {
                if cpt == ' ' {
                    pos += 1;
                }
                loop {
                    let g = match flags.get(pos) {
                        Some(g) => g,
                        None => break,
                    };
                    if g.space || g.letter || g.accent || g.number {
                        break;
                    }
                    pos += 1;
                }
                while matches!(cpts.get(pos), Some('\r') | Some('\n')) {
                    pos += 1;
                }
                push(&mut pieces, &mut last_end, pos);
                continue;
            }

            // ---- regex: \s*[\r\n]+   /   \s+(?!\S)   /   \s+
            let mut num_ws = 0usize;
            let mut last_end_r_or_n = 0usize;
            while pos + num_ws < n && flags[pos + num_ws].space {
                let c2 = cpts[pos + num_ws];
                if c2 == '\r' || c2 == '\n' {
                    last_end_r_or_n = pos + num_ws + 1;
                }
                num_ws += 1;
            }
            if last_end_r_or_n > 0 {
                pos = last_end_r_or_n;
                push(&mut pieces, &mut last_end, pos);
                continue;
            }
            if num_ws > 1 && pos + num_ws < n {
                pos += num_ws - 1;
                push(&mut pieces, &mut last_end, pos);
                continue;
            }
            if num_ws > 0 {
                pos += num_ws;
                push(&mut pieces, &mut last_end, pos);
                continue;
            }

            // 没有命中任何规则
            pos += 1;
            push(&mut pieces, &mut last_end, pos);
        }

        // 每个词做字节编码 + BPE
        for (start, end) in pieces {
            let word: String = cpts[start..end].iter().collect();
            self.bpe_word(&word, out);
        }
    }

    /// 单个词的 BPE（`llm_tokenizer_bpe_session::tokenize` 的内层循环）。
    fn bpe_word(&self, word: &str, out: &mut Vec<u32>) {
        let encoded = if self.byte_encode {
            byte_encode_str(word)
        } else {
            word.to_string()
        };
        let chars: Vec<char> = encoded.chars().collect();
        let n = chars.len();
        if n == 0 {
            return;
        }

        let mut text: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
        let mut prev: Vec<isize> = (0..n as isize).map(|i| i - 1).collect();
        let mut next: Vec<isize> = (0..n as isize)
            .map(|i| if i + 1 == n as isize { -1 } else { i + 1 })
            .collect();
        let mut alive = vec![true; n];
        let mut heap: BinaryHeap<Bigram> = BinaryHeap::new();

        for i in 1..n {
            self.add_bigram(&text, i - 1, i, &mut heap);
        }

        while let Some(b) = heap.pop() {
            let (l, r) = (b.left, b.right);
            if !alive[l] || !alive[r] {
                continue;
            }
            if format!("{}{}", text[l], text[r]) != b.text {
                continue; // 过期的 bigram
            }
            let rt = std::mem::take(&mut text[r]);
            text[l].push_str(&rt);
            alive[r] = false;
            next[l] = next[r];
            if next[r] >= 0 {
                prev[next[r] as usize] = l as isize;
            }
            if prev[l] >= 0 {
                self.add_bigram(&text, prev[l] as usize, l, &mut heap);
            }
            if next[l] >= 0 {
                self.add_bigram(&text, l, next[l] as usize, &mut heap);
            }
        }

        // 走一遍链表输出
        let mut i: isize = 0;
        while i != -1 {
            let idx = i as usize;
            if alive[idx] {
                let s = &text[idx];
                match self.vocab.get(s.as_str()) {
                    Some(&id) => out.push(id),
                    None => {
                        // 词表里没有这个合并结果 → 退化成逐字节查表
                        // （llama.cpp：tokenizer.byte_encode 时用单字节字符串查词表）
                        for b in s.as_bytes() {
                            if *b < 0x80 {
                                let key = (*b as char).to_string();
                                if let Some(&id) = self.vocab.get(&key) {
                                    out.push(id);
                                }
                            }
                        }
                    }
                }
            }
            i = next[idx];
        }
    }

    fn add_bigram(
        &self,
        text: &[String],
        left: usize,
        right: usize,
        heap: &mut BinaryHeap<Bigram>,
    ) {
        if text[left].is_empty() || text[right].is_empty() {
            return;
        }
        let rank = self
            .ranks
            .get(text[left].as_str())
            .and_then(|m| m.get(text[right].as_str()))
            .copied();
        let rank = match rank {
            Some(r) => r,
            None => return,
        };
        heap.push(Bigram {
            rank,
            left,
            right,
            text: format!("{}{}", text[left], text[right]),
        });
    }

    // -- 解码 ----------------------------------------------------------------

    /// 解码：byte-level 还原（`token_to_piece` + `detokenize`）。
    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    /// 解码成原始字节。字节级 BPE 下，一个汉字的 UTF-8 序列可能跨多个 token，
    /// 流式输出时必须按字节流拼接，否则每个 token 单独 decode 会得到 U+FFFD。
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if let Some(p) = self.token_piece_bytes(id) {
                bytes.extend_from_slice(&p);
            }
        }
        bytes
    }

    /// 单个 token 的 piece 字节（special=false 语义：控制/未使用 token 输出空）。
    ///
    /// 与 llama.cpp `token_to_piece`（LLAMA_VOCAB_TYPE_BPE 分支）逐条对应。
    pub fn token_piece_bytes(&self, id: u32) -> Option<Vec<u8>> {
        let text = self.ids.get(id as usize)?;
        let attr = self.attrs.get(id as usize).copied().unwrap_or(ATTR_NORMAL);
        if attr & (ATTR_UNKNOWN | ATTR_CONTROL) != 0 {
            return None; // 控制 token 在 special=false 时不输出
        }
        if attr & ATTR_USER_DEFINED != 0 {
            return Some(text.as_bytes().to_vec());
        }
        if attr & ATTR_NORMAL != 0 {
            return Some(if self.byte_encode {
                byte_decode_str(text)
            } else {
                text.replace('\u{2581}', " ").into_bytes()
            });
        }
        if attr & ATTR_BYTE != 0 {
            return Some(byte_decode_str(text));
        }
        None // UNUSED / UNDEFINED：什么都不输出
    }

    /// 从模型输出中提取纯回答（见 [`extract_answer_from_text`]）。
    pub fn extract_answer(&self, text: &str) -> String {
        extract_answer_from_text(text)
    }
}

/// 去掉模型输出里的模板标记（`<|im_start|>` 等）。
pub fn strip_template_markers(text: &str) -> String {
    text.replace("<|im_start|>", "")
        .replace("<|im_end|>", "")
        .replace("<|user|>", "")
        .replace("<|assistant|>", "")
}

/// 从模型输出中提取纯回答：去掉模板标记，剥掉 `<think>...</think>` 思考块。
///
/// 有两种形态都要处理（思考模式开时是第 2 种，光看 `<think>` 会漏掉）：
///   1. 生成文本里成对出现 `<think>...</think>`；
///   2. **prompt 里已经带了 `<think>\n`**（模板的 generation prompt 就是这样结尾的，
///      见 `chat_template_v22.5.jinja:453-459`），于是生成文本里只剩 `</think>` ——
///      它之前的内容全是思考。
///
/// 注意：**思考没结束时**（`-n` 太小，还没输出 `</think>`）这里会把整段思考当成回答返回。
/// 需要区分"思考/回答"两路时用 [`ReasoningSplitter`]（CLI 走的正是那条路）。
pub fn extract_answer_from_text(text: &str) -> String {
    let cleaned = strip_template_markers(text);

    let mut s = cleaned.as_str();
    if let Some(p) = s.find("<think>") {
        let after = &s[p + "<think>".len()..];
        if let Some(q) = after.find("</think>") {
            s = &after[q + "</think>".len()..];
        } else {
            s = after;
        }
    } else if let Some(q) = s.find("</think>") {
        s = &s[q + "</think>".len()..];
    }
    s.trim().to_string()
}

/// 流式解码器：字节级 BPE 的一个汉字可能跨多个 token，必须按字节累积，
/// 只把「已经完整的 UTF-8 序列」交给上层（否则会出现 U+FFFD 乱码）。
#[derive(Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self { pending: Vec::new() }
    }

    /// 追加一个 token，返回本次可以安全输出的文本（可能为空）。
    pub fn push(&mut self, tok: &GgufTokenizer, id: u32) -> String {
        if let Some(p) = tok.token_piece_bytes(id) {
            self.pending.extend_from_slice(&p);
        }
        self.take_valid()
    }

    /// 序列结束时把余下字节按 lossy 方式吐出。
    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }

    fn take_valid(&mut self) -> String {
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_string();
                self.pending.clear();
                out
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid == 0 {
                    return String::new();
                }
                let out = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                self.pending.drain(..valid);
                out
            }
        }
    }
}

/// 思考块收尾标签（`think_end` tag）。
const THINK_END: &[u8] = b"</think>";

/// 把流式文本切成「思考」与「回答」两路。
///
/// 为什么需要：思考模式开启时，**模板的 generation prompt 以 `<think>\n` 结尾**
/// （开标签在 prompt 里，不在生成文本里），所以生成文本里只有收尾的 `</think>`：
/// 它之前全是思考，之后才是回答。非流式路径由 `extract_answer` 处理同一件事；
/// 流式路径必须增量判定，并且不能把半个 `</think>` 当成回答吐出去（`tail` 缓冲）。
///
/// 只处理**开头**那一段思考：切成回答之后不再重新进入思考态。
#[derive(Debug)]
pub struct ReasoningSplitter {
    in_think: bool,
    /// 可能是 `</think>` 前缀的尾巴，留到下一段再判定。
    tail: Vec<u8>,
}

impl ReasoningSplitter {
    /// `in_think` = 本轮流是否从思考开始（= 渲染时 enable_thinking）。
    pub fn new(in_think: bool) -> Self {
        Self { in_think, tail: Vec::new() }
    }

    /// 当前是否还在思考段（诊断/计数用）。
    pub fn in_think(&self) -> bool {
        self.in_think
    }

    /// 追加一段文本，返回 `(思考, 回答)`（各自可能为空）。
    pub fn push(&mut self, text: &str) -> (String, String) {
        let mut reasoning = String::new();
        let mut answer = String::new();
        self.tail.extend_from_slice(text.as_bytes());
        if self.in_think {
            if let Some(pos) = find_subslice(&self.tail, THINK_END) {
                reasoning.push_str(&String::from_utf8_lossy(&self.tail[..pos]));
                self.tail.drain(..pos + THINK_END.len());
                self.in_think = false;
                // 收尾标签之后的部分全部是回答
                answer.push_str(&String::from_utf8_lossy(&self.tail));
            } else {
                // 没有完整标签：只吐出确定不是标签前缀的部分
                let keep = max_tag_prefix_len(&self.tail, THINK_END);
                let emit = self.tail.len() - keep;
                reasoning.push_str(&String::from_utf8_lossy(&self.tail[..emit]));
                self.tail.drain(..emit);
            }
        } else {
            answer.push_str(&String::from_utf8_lossy(&self.tail));
        }
        if !self.in_think {
            self.tail.clear();
        }
        (reasoning, answer)
    }

    /// 序列结束：把残留按当前状态吐出。
    pub fn finish(&mut self) -> (String, String) {
        let s = String::from_utf8_lossy(&self.tail).into_owned();
        self.tail.clear();
        if self.in_think {
            (s, String::new())
        } else {
            (String::new(), s)
        }
    }
}

/// 在字节串里找子串，返回下标。
fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    (0..=hay.len() - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
}

/// `buf` 末尾与 `tag` 前缀匹配的最长长度（要求 < tag.len()，否则就是完整匹配了）。
fn max_tag_prefix_len(buf: &[u8], tag: &[u8]) -> usize {
    let max = tag.len().saturating_sub(1).min(buf.len());
    (1..=max)
        .rev()
        .find(|&k| &buf[buf.len() - k..] == &tag[..k])
        .unwrap_or(0)
}

#[cfg(test)]
mod reasoning_tests {
    use super::*;

    #[test]
    fn extract_strips_paired_think_block() {
        let out = extract_answer_from_text("<|im_start|>assistant\n<think>\n想一下\n</think>\n答案<|im_end|>");
        assert_eq!(out, "答案");
    }

    /// 思考模式下 prompt 里已经带了 `<think>`，生成文本里只剩收尾标签 ——
    /// 这个形态曾经会被原样输出（推理内容混进回答）。
    #[test]
    fn extract_strips_dangling_end_tag() {
        let out = extract_answer_from_text("用户想要 2000 字\n先列个提纲\n</think>\n\n# 镜中客\n\n林默搬进…");
        assert_eq!(out, "# 镜中客\n\n林默搬进…");
    }

    #[test]
    fn extract_keeps_answer_without_tags() {
        assert_eq!(extract_answer_from_text("  直接回答  "), "直接回答");
    }

    /// 流式切分的两个关键性质：思考与回答分流正确；绝不把半个 `</think>` 当回答吐出。
    #[test]
    fn splitter_splits_at_end_tag() {
        let mut s = ReasoningSplitter::new(true);
        let (r, a) = s.push("思考中");
        assert_eq!((r.as_str(), a.as_str()), ("思考中", ""));
        let (r, a) = s.push("</think>\n回答");
        assert_eq!((r.as_str(), a.as_str()), ("", "\n回答"));
        let (r, a) = s.finish();
        assert_eq!((r.as_str(), a.as_str()), ("", ""));
    }

    #[test]
    fn splitter_never_emits_partial_tag() {
        let mut s = ReasoningSplitter::new(true);
        let (r, a) = s.push("abc</thi");
        assert_eq!((r.as_str(), a.as_str()), ("abc", ""));
        let (r, a) = s.push("nk>OK");
        assert_eq!((r.as_str(), a.as_str()), ("", "OK"));
    }

    #[test]
    fn splitter_without_thinking_passes_everything() {
        let mut s = ReasoningSplitter::new(false);
        let (r, a) = s.push("</think>其实不是思考");
        assert_eq!((r.as_str(), a.as_str()), ("", "</think>其实不是思考"));
    }
}
