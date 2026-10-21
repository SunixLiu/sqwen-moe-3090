//! 工具调用的**文本侧**：把生成文本里的 `<tool_call>…</tool_call>` 解析成结构化调用，
//! 以及流式场景下的增量切分（标签不能漏进 `content`，否则客户端会当成普通文本）。
//!
//! 为什么走这条路而不是约束解码：llama.cpp 也是这么做的 —— 工具定义由 chat template
//! 渲染进 prompt（`common/chat.cpp`），生成后再用解析器把调用抽出来
//! （`common/chat-peg-parser.cpp`），而且显式禁止 `grammar` 与 `tools` 混用
//! （`chat.cpp:1379` "Cannot specify grammar with tools"）。
//!
//! 我们比它简单：只有一个模型、一种格式。模型模板
//! （`chat_template_v22.5.jinja:167-193`）已经规定了内层是
//! `{"name": "...", "arguments": {...}}`（`tool_call_format='json'`），所以这里只需要
//! 一个固定格式的解析器，不需要从模板自动推导文法。

use serde_json::Value;

pub const OPEN: &str = "<tool_call>";
pub const CLOSE: &str = "</tool_call>";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// JSON **字符串**（OpenAI 的 `function.arguments` 是字符串，不是对象）
    pub arguments: String,
}

impl ToolCall {
    /// OpenAI 工具调用形状（响应体与流式 delta 都用它）。
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "id": self.id,
            "type": "function",
            "function": { "name": self.name, "arguments": self.arguments }
        })
    }
}

fn new_id(seq: usize) -> String {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("call_{ms}_{seq}")
}

/// 解析一个 `<tool_call>` 块的内层。
///
/// 宽容三种写法（模型实际会给的）：
/// - `{"name": "x", "arguments": {...}}`
/// - `{"name": "x", "arguments": "{...}"}`（arguments 是字符串）
/// - **JSON 里有非法转义/裸控制字符/少了收尾括号** —— 见 [`parse_json_lenient`]
fn parse_block(body: &str) -> Option<(String, String)> {
    let body = body.trim();
    // 有些输出会在块里再套一层 ```json … ```
    let body = body.strip_prefix("```json").or_else(|| body.strip_prefix("```")).unwrap_or(body);
    let body = body.strip_suffix("```").unwrap_or(body).trim();
    let v: Value = parse_json_lenient(body)?;
    let name = v.get("name").and_then(|x| x.as_str())?.to_string();
    if name.is_empty() {
        return None;
    }
    let arguments = match v.get("arguments") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => "{}".to_string(),
    };
    Some((name, arguments))
}

/// 修「字符串里没转义的引号」。
///
/// 真实案例：模型写 Python 文件时，docstring 开头写成 `\"\"\"`（正确），结尾写成了
/// 裸的 `"""` —— JSON 字符串在结尾那三个引号处**提前终止**，后面整段代码变成非法 JSON，
/// 于是整块工具调用降级成正文。
///
/// 判定规则：字符串内遇到 `"`，往后跳过空白看第一个字符 ——
/// 是 `,` `}` `]` `:` 或输入结束，才是真的收尾；否则把它当字面引号，补成 `\"`。
/// 这样 `"""` 这种三连引号会被全部转义，字符串继续往下走，直到真正的 `"}`。
fn repair_bare_quotes(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len() + 16);
    let mut in_str = false;
    /// 当前字符串是**键**（`{`/`,` 之后、`:` 之前）还是**值**。
    let mut in_key = false;
    /// 容器栈：`{` 或 `[`，用来判断 `,` 之后是键还是元素。
    let mut stack: Vec<char> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if !in_str {
            match c {
                '{' => {
                    stack.push('{');
                    in_key = true;
                }
                '[' => {
                    stack.push('[');
                    in_key = false;
                }
                '}' | ']' => {
                    stack.pop();
                }
                // 对象里的 `,` 后面是下一个键；数组里是下一个元素
                ',' => in_key = stack.last() == Some(&'{'),
                ':' => in_key = false,
                // 开字符串：`in_key` 此时已经由上面几个分支定好
                '"' => in_str = true,
                _ => {}
            }
            out.push(c);
            i += 1;
            continue;
        }
        if c == '\\' {
            // 转义序列原样保留（`\"` `\\` `\n` …）
            out.push(c);
            if i + 1 < chars.len() {
                out.push(chars[i + 1]);
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        if c == '"' {
            // 键：第一个引号就是收尾；值：要判断像不像 JSON 结构
            if in_key || is_string_end(&chars, i) {
                in_str = false;
                out.push(c);
            } else {
                out.push_str("\\\"");
            }
            i += 1;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// 判断 `chars[i]`（一个引号）是不是**字符串的收尾**。
///
/// 难点：字符串里可能有没转义的引号（模型把 Python 代码塞进 JSON 时极常见）。
/// 光看「后面是不是 `,` `}` `]` `:`」不够 —— 真实的代码里：
/// - `if x == "text":` 收尾引号后面就是 `:`（`:` 永远不是值的收尾，只在键后面出现）
/// - `["a", "b"]` 的收尾引号后面就是 `,`
///
/// 所以判据换成「**像不像 JSON 的结构**」：
/// - 输入结束 → 收尾
/// - 后面是 `,` → 再往后必须是一个**键**（`"ident" :`）才算收尾
/// - 后面是 `}` 或 `]` → 跳过它，再往后必须是 `,` `}` `]` 或输入结束
/// - 后面是 `:` → **不是**收尾（`:` 只出现在键之后）
fn is_string_end(chars: &[char], i: usize) -> bool {
    let mut j = i + 1;
    while j < chars.len() && chars[j].is_whitespace() {
        j += 1;
    }
    match chars.get(j) {
        None => true,
        Some(',') => looks_like_json_key(chars, j + 1),
        Some('}') | Some(']') => {
            let mut k = j + 1;
            while k < chars.len() && chars[k].is_whitespace() {
                k += 1;
            }
            matches!(chars.get(k), None | Some(',') | Some('}') | Some(']') | Some('"'))
        }
        _ => false,
    }
}

/// `chars[j..]` 是否长得像 `"identifier" :`（JSON 的下一个键）。
fn looks_like_json_key(chars: &[char], mut j: usize) -> bool {
    while j < chars.len() && chars[j].is_whitespace() {
        j += 1;
    }
    if chars.get(j) != Some(&'"') {
        return false;
    }
    j += 1;
    let start = j;
    while j < chars.len() && chars[j] != '"' {
        // 键里不会有反斜杠（我们的工具名/参数名都是标识符）
        if chars[j] == '\\' {
            return false;
        }
        j += 1;
    }
    if j == start || chars.get(j) != Some(&'"') {
        return false;
    }
    j += 1;
    while j < chars.len() && chars[j].is_whitespace() {
        j += 1;
    }
    chars.get(j) == Some(&':')
}

/// 宽容地解析 JSON：先严格来一遍，不行就**修一次再试**。
///
/// 为什么必须修：模型写 Windows 路径时几乎必然给出 `"D:\01Projects\..."` ——
/// `\0` 不是合法 JSON 转义，严格解析会整块失败。失败后按「原样当正文吐出去」处理，
/// 用户看到的就是一大块 `<tool_call>` 文本（文件其实已经被别处改掉了，但界面很难看）。
///
/// 修三件事（都是模型的高频错误）：
/// 1. 非法转义 → 补成 `\\`（`D:\01` → `D:\\01`）；合法转义（`\" \\ \/ \b \f \n \r \t \u`）不动；
/// 2. 字符串里的裸控制字符（真换行/真 tab）→ 转义；
/// 3. 少收尾括号（被截断）→ 按栈补齐；顺带吃掉 `}` 前的多余逗号。
fn parse_json_lenient(s: &str) -> Option<Value> {
    // 第一道门槛：**块体不像 JSON 就根本不试**。
    //
    // 实测：模型写文档时正文里会**字面提到** `<tool_call>` 这个词（比如在总结调试过程），
    // 切分器会把它当成调用开始，块体是一大段中文散文。那种情况按设计保留成正文就好，
    // 不该尝试解析、更不该打错误日志（以前每次都会刷一行「第 1 列 expected value」）。
    if !s.trim_start().starts_with('{') {
        if std::env::var_os("SQWEN_TOOL_DEBUG").is_some() {
            let snip: String = s.chars().take(60).collect();
            eprintln!("[toolcall] 块体不像 JSON（正文里提到 <tool_call>？），按正文保留: {snip}");
        }
        return None;
    }
    let mut last_err: Option<serde_json::Error> = None;
    let mut attempt = 0usize;
    let mut try_parse = |text: &str| -> Option<Value> {
        attempt += 1;
        match serde_json::from_str::<Value>(text) {
            Ok(v) => {
                if std::env::var_os("SQWEN_TOOL_DEBUG").is_some() {
                    eprintln!("[toolcall] JSON 解析成功（第 {attempt} 个候选：1=原始 2=修转义 3=补引号 4=补括号）");
                }
                Some(v)
            }
            Err(e) => {
                last_err = Some(e);
                None
            }
        }
    };
    if let Some(v) = try_parse(s) {
        return Some(v);
    }
    let mut fixed = repair_json(s);
    if let Some(v) = try_parse(&fixed) {
        return Some(v);
    }
    // 截断的常见形态：`..., "x": "y"}` 少一层 `}`，或结尾多一个逗号
    fixed = fixed.trim_end().trim_end_matches(',').to_string();
    // 再修「字符串里没转义的引号」（模型写长文本/代码时的典型失误）
    let quoted = repair_bare_quotes(&fixed);
    if let Some(v) = try_parse(&quoted) {
        return Some(v);
    }
    let closed = close_open_containers(&quoted);
    for cand in [
        closed.clone(),
        strip_trailing_commas(&closed),
        close_open_containers(&fixed),
    ] {
        if let Some(v) = try_parse(&cand) {
            return Some(v);
        }
    }
    // 走到这里说明**块体确实像 JSON**（以 `{` 开头）但修不回来 —— 这才是真问题，值得报。
    if std::env::var_os("SQWEN_QUIET").is_none() {
        // **把出错位置打出来**：JSON 参数动辄上千字符（写文件的 content 就是整个文件），
        // 只打前缀没法定位。serde 的 line/column 指向出问题的那个字符，前后各截 40 字。
        if let Some(e) = last_err {
            let col = e.column().saturating_sub(1);
            let chars: Vec<char> = s.chars().collect();
            let lo = col.saturating_sub(40);
            let hi = (col + 40).min(chars.len());
            let ctx: String = chars[lo..hi].iter().collect();
            eprintln!(
                "[toolcall] JSON 解析失败（第 {} 行第 {} 列，共 {} 字符）: {} —— 出错处上下文: …{ctx}…",
                e.line(),
                e.column(),
                chars.len(),
                e
            );
        } else {
            let snip: String = s.chars().take(160).collect();
            eprintln!("[toolcall] 工具调用参数不是合法 JSON，按正文输出。前 160 字: {snip}");
        }
    }
    None
}

/// 修字符串里的非法转义与裸控制字符（只改字符串内部，结构字符原样保留）。
fn repair_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    let mut in_str = false;
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if !in_str {
            if c == '"' {
                in_str = true;
            }
            out.push(c);
            continue;
        }
        match c {
            '"' => {
                in_str = false;
                out.push(c);
            }
            // 转义：**必须一次吃掉两个字符**。只 push 一个反斜杠的话，下一轮会把
            // `\\` 的第二个反斜杠当新的转义符再判一次（`\\0` → `\\\0`），把本来是
            // 合法 JSON 的内容改坏 —— 这个 bug 会让每个**正确转义**的 Windows 路径解析失败。
            '\\' => match it.next() {
                Some(n) if matches!(n, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u') => {
                    out.push('\\');
                    out.push(n);
                }
                // 非法转义（`\0`、`\'`、Windows 路径里的单个反斜杠）→ 补成字面反斜杠
                Some(n) => {
                    out.push_str("\\\\");
                    out.push(n);
                }
                None => out.push_str("\\\\"),
            },
            c if (c as u32) < 0x20 => {
                for e in c.escape_default() {
                    out.push(e);
                }
            }
            c => out.push(c),
        }
    }
    out
}

/// 按括号栈补齐收尾（只补不删），最多补 8 层，避免把垃圾补成"合法"。
fn close_open_containers(s: &str) -> String {
    let mut stack: Vec<char> = Vec::new();
    let mut in_str = false;
    let mut esc = false;
    for c in s.chars() {
        if in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' | '[' => stack.push(c),
            '}' | ']' => {
                stack.pop();
            }
            _ => {}
        }
    }
    let mut out = s.to_string();
    if in_str {
        out.push('"');
    }
    for open in stack.iter().rev().take(8) {
        out.push(if *open == '{' { '}' } else { ']' });
    }
    out
}

/// 去掉 `}` / `]` 前多余的逗号（模型常见的尾逗号）。
fn strip_trailing_commas(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut in_str = false;
    let mut esc = false;
    for (i, c) in chars.iter().enumerate() {
        if in_str {
            if esc {
                esc = false;
            } else if *c == '\\' {
                esc = true;
            } else if *c == '"' {
                in_str = false;
            }
            out.push(*c);
            continue;
        }
        if *c == '"' {
            in_str = true;
            out.push(*c);
            continue;
        }
        if *c == ',' {
            // 往后看第一个非空白字符是不是收尾括号
            if let Some(next) = chars[i + 1..].iter().find(|x| !x.is_whitespace()) {
                if *next == '}' || *next == ']' {
                    continue; // 丢掉这个逗号
                }
            }
        }
        out.push(*c);
    }
    out
}

/// 从**完整**文本里抽出所有调用，返回「去掉调用块的正文」+「调用列表」。
///
/// 非流式路径直接用它；流式路径用 [`ToolCallSplitter`]（需要增量切分）。
pub fn extract(text: &str) -> (String, Vec<ToolCall>) {
    let mut sp = ToolCallSplitter::new();
    let mut content = sp.push(text);
    let (tail, calls) = sp.finish();
    content.push_str(&tail);
    (content, calls)
}

/// 流式切分器：喂进来的正文出去时保证 `<tool_call>…</tool_call>` 已经被摘掉。
///
/// 关键点是**尾巴要 hold**：`"<tool"` 这种半截标签必须先留住，否则它会先作为正文
/// 发给客户端、后面再补上标签就收不回来了。
pub struct ToolCallSplitter {
    /// 还没决定归属的尾巴（可能是 `<tool_call>` 的前缀）
    pending: String,
    /// 已经进入块内、还没等到 `</tool_call>` 的内容
    in_call: Option<String>,
    calls: Vec<ToolCall>,
}

impl ToolCallSplitter {
    pub fn new() -> Self {
        Self { pending: String::new(), in_call: None, calls: Vec::new() }
    }

    /// 喂一段正文，返回**可以立刻发出去**的正文（调用块被吞掉）。
    pub fn push(&mut self, s: &str) -> String {
        // 已经在块内：新文本必须进块，不能进 `pending`（否则块里会漏字符）
        if let Some(call) = self.in_call.as_mut() {
            call.push_str(s);
        } else {
            self.pending.push_str(s);
        }
        self.drain(false)
    }

    /// 收尾：返回（剩余正文，全部调用）。
    pub fn finish(mut self) -> (String, Vec<ToolCall>) {
        let tail = self.drain(true);
        (tail, self.calls)
    }

    fn drain(&mut self, flush: bool) -> String {
        let mut out = String::new();
        loop {
            // ---- 在块内：等 `</tool_call>` ----
            if let Some(call) = self.in_call.take() {
                if let Some(p) = call.find(CLOSE) {
                    let body = &call[..p];
                    let rest = call[p + CLOSE.len()..].to_string();
                    match parse_block(body) {
                        Some((name, arguments)) => {
                            let id = new_id(self.calls.len());
                            self.calls.push(ToolCall { id, name, arguments });
                        }
                        None => {
                            // 解析不了就原样吐回去，别把模型的话吞掉
                            out.push_str(OPEN);
                            out.push_str(body);
                            out.push_str(CLOSE);
                        }
                    }
                    self.pending.push_str(&rest);
                    continue;
                }
                if flush {
                    // 没有闭合标签：尽量当成一次调用，失败就原样吐出
                    match parse_block(&call) {
                        Some((name, arguments)) => {
                            let id = new_id(self.calls.len());
                            self.calls.push(ToolCall { id, name, arguments });
                        }
                        None => {
                            out.push_str(OPEN);
                            out.push_str(&call);
                        }
                    }
                    return out;
                }
                self.in_call = Some(call);
                return out;
            }

            // ---- 不在块内：找开始标签 ----
            if let Some(p) = self.pending.find(OPEN) {
                out.push_str(&self.pending[..p]);
                let rest = self.pending[p + OPEN.len()..].to_string();
                self.pending.clear();
                self.in_call = Some(rest);
                continue;
            }

            // 没有开始标签：把「可能是标签前缀」的尾巴留住
            let hold = if flush { 0 } else { suffix_matching_prefix(&self.pending, OPEN) };
            let emit = self.pending.len() - hold;
            out.push_str(&self.pending[..emit]);
            let tail = self.pending[emit..].to_string();
            self.pending = tail;
            return out;
        }
    }
}

/// `s` 的最长后缀长度，使得该后缀是 `pat` 的前缀（且短于 `pat`）。
fn suffix_matching_prefix(s: &str, pat: &str) -> usize {
    let max = s.len().min(pat.len() - 1);
    for k in (1..=max).rev() {
        // 只在字符边界上比较，避免切坏 UTF-8
        if !s.is_char_boundary(s.len() - k) {
            continue;
        }
        if pat.as_bytes().starts_with(&s.as_bytes()[s.len() - k..]) {
            return k;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_single_call() {
        let text = "我来查一下。\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.txt\"}}\n</tool_call>\n";
        let (content, calls) = extract(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments, "{\"path\":\"a.txt\"}");
        assert!(content.contains("我来查一下"));
        assert!(!content.contains("<tool_call>"), "标签不能留在正文里: {content}");
    }

    #[test]
    fn extracts_two_calls_and_keeps_text() {
        let text = "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>中间<tool_call>{\"name\":\"b\",\"arguments\":{\"x\":1}}</tool_call>尾巴";
        let (content, calls) = extract(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].name, "b");
        assert_eq!(content, "中间尾巴");
    }

    #[test]
    fn arguments_as_string_is_kept() {
        let text = r#"<tool_call>{"name":"x","arguments":"{\"a\":1}"}</tool_call>"#;
        let (_, calls) = extract(text);
        assert_eq!(calls[0].arguments, "{\"a\":1}");
    }

    #[test]
    fn unterminated_block_still_parses() {
        let text = "<tool_call>{\"name\":\"a\",\"arguments\":{}}";
        let (content, calls) = extract(text);
        assert_eq!(calls.len(), 1);
        assert!(content.is_empty());
    }

    #[test]
    fn bad_json_is_preserved_as_text() {
        let text = "<tool_call>not json</tool_call>";
        let (content, calls) = extract(text);
        assert!(calls.is_empty());
        assert!(content.contains("not json"));
    }

    /// 真实踩到的坑：模型把 Windows 路径原样写进 JSON（`D:\01…`），`\0` 不是合法转义。
    #[test]
    fn windows_path_with_invalid_escape_is_repaired() {
        let text = r#"<tool_call>{"name":"edit","arguments":{"file_path":"D:\01Projects\pygame\audio.py","old_string":"a"}}</tool_call>"#;
        let (content, calls) = extract(text);
        assert_eq!(calls.len(), 1, "非法转义不该让整块降级成正文: {content}");
        assert_eq!(calls[0].name, "edit");
        let args: serde_json::Value = serde_json::from_str(&calls[0].arguments).expect("arguments 必须是合法 JSON");
        assert_eq!(args["file_path"], "D:\\01Projects\\pygame\\audio.py");
        assert!(content.trim().is_empty());
    }

    /// 合法转义不能被改动（`\n` 仍是换行，不是字面 `\n`）。
    #[test]
    fn valid_escapes_are_left_alone() {
        let text = r#"<tool_call>{"name":"edit","arguments":{"new_string":"a\nb\tc\"d"}}</tool_call>"#;
        let (_, calls) = extract(text);
        let args: serde_json::Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["new_string"], "a\nb\tc\"d");
    }

    /// 被截断的 JSON（少收尾括号）也要能救回来。
    #[test]
    fn truncated_json_is_closed() {
        let text = r#"<tool_call>{"name":"edit","arguments":{"file_path":"a.py","new_string":"x"}</tool_call>"#;
        let (_, calls) = extract(text);
        assert_eq!(calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["file_path"], "a.py");
    }

    /// 尾逗号
    #[test]
    fn trailing_comma_is_tolerated() {
        let text = r#"<tool_call>{"name":"f","arguments":{"a":1,}}</tool_call>"#;
        let (_, calls) = extract(text);
        assert_eq!(calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["a"], 1);
    }

    /// 实测踩到：模型写 Python 时 docstring 开头转义了、结尾没转义。
    #[test]
    fn unescaped_triple_quote_inside_content_is_repaired() {
        let text = concat!(
            r#"<tool_call>{"name":"write","arguments":{"file_path":"D:\\01Projects\\a.py","#,
            r#""content":"\"\"\"docstring。""" \nimport x\nclass A:\n    pass\n"}}</tool_call>"#
        );
        let (content, calls) = extract(text);
        assert_eq!(calls.len(), 1, "漏转义的引号不该让整块降级: {content}");
        let args: serde_json::Value = serde_json::from_str(&calls[0].arguments).expect("arguments 必须合法");
        let body = args["content"].as_str().unwrap();
        assert!(body.contains("docstring"), "content 里应该保留正文: {body}");
        assert!(body.contains("import x"), "content 应该延续到后面的代码: {body}");
    }

    /// 实测踩到第二种：Python f-string 里的引号没转义。
    #[test]
    fn unescaped_quotes_inside_python_fstring_are_repaired() {
        let text = concat!(
            r#"<tool_call>{"name":"write","arguments":{"file_path":"D:\\01Projects\\a.py","content":"#,
            r#""\"\"\"doc.\"\"\"\nimport sys\nprint(\"Hello, world!\")\nprint(f"Python path: {sys.executable}")\n"}}</tool_call>"#
        );
        let (content, calls) = extract(text);
        assert_eq!(calls.len(), 1, "f-string 里的裸引号不该让整块降级: {content}");
        let args: serde_json::Value = serde_json::from_str(&calls[0].arguments).expect("arguments 必须合法");
        let body = args["content"].as_str().unwrap();
        assert!(body.contains("f\"Python path"), "f-string 应该完整保留: {body}");
        assert!(body.contains("import sys"));
    }

    /// 正常的多个字段不能被误判成「漏转义」。
    #[test]
    fn normal_multiple_fields_still_parse() {
        let text = r#"<tool_call>{"name":"f","arguments":{"a":"x","b":"y","c":1}}</tool_call>"#;
        let (_, calls) = extract(text);
        let args: serde_json::Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["a"], "x");
        assert_eq!(args["b"], "y");
        assert_eq!(args["c"], 1);
    }

    #[test]
    fn stream_holds_back_partial_tag() {
        let mut sp = ToolCallSplitter::new();
        // 逐字符喂：任何时刻都不能把 `<tool_call>` 的前缀当成正文发出去
        let text = "hi<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>";
        let mut out = String::new();
        for ch in text.chars() {
            let piece = sp.push(&ch.to_string());
            assert!(
                !piece.contains("tool") && !piece.contains("<t"),
                "半截标签漏出去了: {piece:?}"
            );
            out.push_str(&piece);
        }
        let (tail, calls) = sp.finish();
        out.push_str(&tail);
        assert_eq!(calls.len(), 1);
        assert_eq!(out, "hi");
    }

    #[test]
    fn plain_text_passes_through() {
        let mut sp = ToolCallSplitter::new();
        let mut out = String::new();
        out.push_str(&sp.push("你好"));
        out.push_str(&sp.push("，世界"));
        let (tail, calls) = sp.finish();
        out.push_str(&tail);
        assert_eq!(out, "你好，世界");
        assert!(calls.is_empty());
    }

    /// Python 语句里的收尾引号后面跟 `:`，不能当成 JSON 字符串结束。
    #[test]
    fn python_annotation_quote_is_not_string_end() {
        let text = concat!(
            "<tool_call>{\"name\":\"edit\",\"arguments\":{\"file_path\":\"D:\\\\a.py\",",
            "\"old_string\":\"if x == \"text\":\\n    pass\\n\"}}",
            "</tool_call>"
        );
        let (content, calls) = extract(text);
        assert_eq!(calls.len(), 1, "代码里的引号不该让整块降级: {content}");
        let args: serde_json::Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert!(args["old_string"].as_str().unwrap().contains("if x == \"text\":"));
    }

    /// 数组字面量里的 `", "` 也不能当成「下一个键」。
    #[test]
    fn python_list_quote_is_not_string_end() {
        let text = concat!(
            "<tool_call>{\"name\":\"edit\",\"arguments\":{\"file_path\":\"D:\\\\a.py\",",
            "\"old_string\":\"xs = [\"a\", \"b\"]\\n\"}}",
            "</tool_call>"
        );
        let (content, calls) = extract(text);
        assert_eq!(calls.len(), 1, "数组字面量不该让整块降级: {content}");
        let args: serde_json::Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert!(args["old_string"].as_str().unwrap().contains("xs = [\"a\", \"b\"]"));
    }
}
