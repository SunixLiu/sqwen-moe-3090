# sqwen

A local inference engine for **Qwen3.6-35B-A3B** (the `qwen35moe` architecture: gated-delta-net
linear attention + full attention + MoE) that runs entirely on a **single 24 GB GPU** and serves an
**OpenAI-compatible HTTP API**.

> 中文文档：[README.zh-CN.md](README.zh-CN.md)

## Why this exists

Running a 35B-A3B MoE locally is easy. Running it with a **168K–200K token context on one RTX 3090**
— and serving it to agent frameworks that need streaming and tool calling — is not.

`sqwen` targets exactly that: a purpose-built engine for one architecture family, tuned to squeeze a
24 GB card to the limit instead of being general.

| | This setup |
|---|---|
| Model | Qwen3.6-35B-A3B, Q4_K_M, 20.2 GiB |
| Hardware | 1 × RTX 3090 (24 GB), everything resident on GPU |
| Context | `-c 172032` → 23.7 GiB used, ~530 MiB free · `-c 200000` → ~380 MiB free |
| KV cache | **5.6 KiB / token** (q4_0 K and V, 10 full-attention layers) |

## How it works

`sqwen` is **not** a wrapper around llama.cpp. It links the `ggml` / `ggml-cuda` C libraries
(built from a llama.cpp checkout) and drives them through hand-written FFI, but everything above
that is ours: the compute graph, KV cache, sampler, prompt cache, speculative decoding and HTTP server.

```
Rust (this repo)
 ├─ model/    per-architecture graph builder: GDN layers, full attention, MoE routing, MTP head
 ├─ engine/   KV cache (q4_0), sampler, prompt cache, reasoning budget, timing
 ├─ server/   axum: /v1/chat/completions (stream + non-stream), tool calling, WebUI, /v1/bench
 └─ backend/  FFI to ggml / ggml-cuda (no llama.cpp runtime dependency)
```

Highlights:

- **Long context on a small card.** q4_0 KV + FlashAttention keep per-token cost flat; context length
  costs VRAM, not speed (172K → 200K is −0.4% decode).
- **MTP speculative decoding** using the model's built-in NextN head, with GDN state snapshots so a
  partially-accepted draft can be rolled back correctly.
- **Cost-aware draft depth.** A draft token's marginal cost is ~60% of a full decode step
  (verification has to read the union of the experts it activates) while acceptance decays with depth,
  so the optimum is *one* draft per step — deeper drafting is a net loss on this model. The engine
  measures both modes online and locks the winner (`--spec-adaptive`) instead of trusting a flag.
- **Prompt cache** — prefix reuse plus GDN state checkpoints, so multi-turn TTFT is ~11× better than
  re-prefilling.
- **Tool calling** via chat-template injection + text parsing (same approach as llama.cpp), including
  streaming and multi-turn `role: "tool"` results.
- **Built-in diagnostics**: per-phase profiling, graph node histogram, and module-level ablation.

## Performance

Same machine, same session, **identical `ggml-cuda.dll` on both sides**, ABBA ordering, both engines
served over HTTP, greedy, 1200 tokens generated:

| Config | sqwen | llama.cpp | ratio |
|---|---:|---:|---:|
| Decode, no speculation | 128 t/s | 143 t/s | 1.12× faster |
| Decode, MTP (1 draft / token) | **137 t/s** | 151 t/s | 1.10× faster |
| Prefill `pp512` | ~3 000 t/s | ~3 200 t/s | ~5% faster |
| Startup (load only) | 12.3 s | 19.2 s | — |
| VRAM at 172K ctx | 23 676 MiB | 24 083 MiB | — |

Honest summary: **prefill is at parity, memory is better, decode is ~10–12% behind**, and
speculation does not widen it: both engines gain ~5–7% from it (ours slightly more). Speculative decoding used to be a
*net loss* here (−13% on `n_max=2`); it is now a win, because the optimum turned out to be one draft
token per step (see below), not the three that llama.cpp's flag suggests.

> Measurement discipline matters: absolute numbers drift ±5% between sessions. Only same-session,
> same-DLL A/B comparisons are trustworthy.

## Recent changes

| Date | Change | Effect |
|---|---|---|
| 2026-10-26 | **Cost-aware adaptive speculation**: chooses between 0 and 1 draft token per step; probes 5 rounds per candidate *after* 32 warm-up rounds (early drafts are poor) and compares **median** per-round throughput; `--spec-draft-n-max` default 3 → 1, new `--spec-adaptive` (on by default) | decode **+6.8%** (HTTP, 1200 tokens: 128.3 → 137.0 t/s); bench tg600: 148 vs 131 t/s (+13%); gap to llama.cpp 1.36× → 1.10× |
| 2026-10-23 | **Upgraded the ggml DLLs to llama.cpp 10-23** (includes the `flash_attn_ext_vec` shared-memory race fix), resynced the `ggml_op` enum and added an ABI guard test | decode **+3–4%** |

## Requirements

- Windows 10/11 (the build script and toolchain are PowerShell + MSVC; other platforms are untested)
- Visual Studio 2022 (MSVC), CUDA 12.5 (nvcc), CMake
- A 24 GB NVIDIA GPU (developed on RTX 3090 / sm_86)
- **A llama.cpp checkout with a built CUDA `ggml`** — the ggml libraries are *not* vendored here
- A GGUF model matching `qwen35moe` (Qwen3.6-35B-A3B, optionally with a `nextn.*` MTP head)

## Build

```powershell
# 1. tell the build script where your llama.cpp checkout is
$env:SQWEN_LLAMA_CPP = "D:\path\to\llama.cpp"
.\build_ggml.ps1            # copies ggml .lib/.dll into .\build-ggml\

# 2. build
cargo build --release       # produces target\release\sqwen.exe
```

Notes:

- `build.rs` copies the ggml DLLs next to the binary **on every build**. If you replace the DLLs
  manually, put them in `build-ggml\bin\Release\` — otherwise the next `cargo build` overwrites them.
- The ggml ABI must match `src/backend/ffi.rs`. The FFI was verified against llama.cpp
  `0.7.0-dev` (build 1218).

## Run

Serve an OpenAI-compatible endpoint (this is the configuration used in production here):

```powershell
.\target\release\sqwen.exe server "E:\models\BigBang-v1.i1-Q4_K_M.gguf" `
  --port 8100 -c 172032 -b 2048 -u 512 -t 8 `
  --template "E:\models\chat_template_v22.5.jinja" `
  --spec-type draft-mtp `
  --temp 0.1 --top-k 40 --top-p 0.95 --min-p 0.05 `
  --repeat-penalty 1.08 --repeat-last-n 128 `
  --reasoning on --reasoning-effort medium --reasoning-budget 2048 `
  --reasoning-format deepseek --queue-capacity 8
```

- `--repeat-penalty 1.08 --repeat-last-n 128` is **not optional** for agent use: with the default
  `1.0` (disabled) and a low temperature, a model that starts repeating never stops.
- `--reasoning-format deepseek` streams the model's thinking into `reasoning_content` so `content`
  stays clean; `--reasoning-budget` caps thinking, and 256 tokens are always reserved for the answer.
- `-c 172032` is the daily driver; `-c 200000` still fits but leaves ~380 MiB, so any other GPU app
  will trigger WDDM paging.

Other modes:

```powershell
.\target\release\sqwen.exe chat  <model> -p "hello" -n 256 --temp 0.1
.\target\release\sqwen.exe bench <model> --prompt-tokens 512 --gen-tokens 128 -r 2
```

## HTTP API

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible; streaming and non-streaming |
| `GET /v1/models`, `GET /health` | discovery / readiness (the port is bound only after the model is warm) |
| `GET /` | built-in WebUI |
| `POST /v1/bench` | quick benchmark endpoint |

**Tool calling** works by rendering the tool definitions into the chat template and parsing
`<tool_call>{...}</tool_call>` out of the generated text — the same route llama.cpp takes. Streaming
tool calls are delivered whole in the final chunk.

Known limits (deliberate — see the docs):

- Requests are **serialized** (single engine); beyond `--queue-capacity` you get `429`.
- `tool_choice: "required"` and named-function forcing are not enforced; only `"none"` is honored.
- No `stop`, `n>1`, `logprobs`, `response_format`, or JSON-schema constrained decoding.
- Tool-call JSON occasionally arrives with unescaped quotes; the server repairs the common cases and
  otherwise falls back to plain text rather than emitting a broken call.

## Diagnostics

| Env var | Effect |
|---|---|
| `SQWEN_PROFILE=1` | per-step `build / alloc / fill / compute / read` timings |
| `SQWEN_DUMP_GRAPH=1` | graph node histogram + fusion-pattern hit counts |
| `SQWEN_ABLATE=<mods>` | replace modules with zeros to measure their cost (`attn,gdn,moe,lm_head,norm,gdn_proj,gdn_conv,gdn_core,gdn_out`) |
| `SQWEN_TOOL_DEBUG=1` | per-request tool-call parse summary |
| `SQWEN_NO_LOOP_GUARD=1` | disable the repetition-loop guard |
| `SQWEN_EXPERT_USED=n` | override top-k experts (measurement only) |

## Project layout

```
src/
  main.rs        CLI: chat / bench / server (clap)
  lib.rs
  backend/       ggml + ggml-cuda FFI, tensor/context wrappers
  model/         arch, weights, graph builder, attn, gdn, moe, mtp
  engine/        runner (prefill/decode/spec), kv, sampler, prompt_cache, reasoning, timing
  server/        api, queue, bench, toolcall, webui.html
  template/      minijinja chat-template rendering, tokenizer
assets/          fallback chat template + tokenizer
scripts/         ABBA benchmark harness, SSE probe, OpenAI SDK check
tests/           graph / attn / gdn / moe / sampler / server tests
```

## Status

Early. One architecture family, Windows-only build, single sequence at a time. It is fast enough to
be useful today (this project was developed against it with an agent framework), but it is not a
drop-in llama.cpp replacement.

## License

MIT — see [LICENSE](LICENSE).
