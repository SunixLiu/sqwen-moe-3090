# sqwen

面向 **Qwen3.6-35B-A3B**（`qwen35moe` 架构：Gated DeltaNet 线性注意力 + 全注意力 + MoE）的
本地推理引擎，整模型跑在**单张 24 GB 显卡**上，并对外提供 **OpenAI 兼容的 HTTP API**。

> English docs: [README.md](README.md)（默认语言）

## 为什么做这个

本地跑起一个 35B-A3B 的 MoE 不难。难的是**在单张 RTX 3090 上开出 168K–200K 的上下文**，
并且能直接喂给需要流式和工具调用的 agent 框架。

`sqwen` 就是为这一件事做的：不追求通用，只针对一个架构族，把 24 GB 显存榨到位。

| 项目 | 本方案 |
|---|---|
| 模型 | Qwen3.6-35B-A3B，Q4_K_M，20.2 GiB |
| 硬件 | 1 × RTX 3090（24 GB），权重全量常驻显存 |
| 上下文 | `-c 172032` → 占 23.7 GiB、余 ~530 MiB · `-c 200000` → 余 ~380 MiB |
| KV cache | **5.6 KiB / token**（K/V 都是 q4_0，10 个全注意力层） |

## 实现方式

`sqwen` **不是** llama.cpp 的封装。它链接 `ggml` / `ggml-cuda` 两个 C 库（由 llama.cpp 的源码构建），
通过手写 FFI 调用；**上面的一切都是自己实现的**：计算图、KV cache、采样器、prompt cache、
投机解码、HTTP 服务。

```
Rust（本仓库）
 ├─ model/    按架构搭图：GDN 层、全注意力、MoE 路由、MTP 头
 ├─ engine/   KV cache（q4_0）、采样器、prompt cache、思考预算、计时
 ├─ server/   axum：/v1/chat/completions（流式 + 非流式）、tool calling、WebUI、/v1/bench
 └─ backend/  对 ggml / ggml-cuda 的 FFI（运行时不依赖 llama.cpp）
```

亮点：

- **小卡长上下文**：q4_0 KV + FlashAttention 让每 token 成本基本不随上下文涨；
  `-c` 只吃显存、不降速（172K → 200K 只慢 0.4%）。
- **MTP 投机解码**：用模型自带的 NextN 头起草，配合 GDN 状态快照，部分接受时能正确回退。
- **成本感知的起草深度**：一个草稿 token 的边际成本 ≈ 一次完整 decode 的 60%（校验要多读它
  激活的专家权重），而接受率随深度衰减，所以最优点就是**每步 1 个草稿** —— 更深在本模型上是
  净亏。引擎会在线实测两种模式并锁定胜者（`--spec-adaptive`），而不是靠拍参数。
- **Prompt cache**：前缀复用 + GDN 状态检查点，多轮对话首字延迟约好 11 倍。
- **Tool calling**：模板注入 + 文本解析（与 llama.cpp 同路线），支持流式与多轮 `role: "tool"` 结果。
- **内置诊断**：分阶段计时、图节点直方图、按模块消融。

## 性能

同机、同会话、**两边 `ggml-cuda.dll` 完全相同**、ABBA 交叉、都走 HTTP server、greedy、生成 1200 token：

| 配置 | sqwen | llama.cpp | 比值 |
|---|---:|---:|---:|
| Decode，无投机 | 128 t/s | 143 t/s | 对方快 1.12× |
| Decode，MTP（每步 1 个草稿） | **137 t/s** | 151 t/s | 对方快 1.10× |
| Prefill `pp512` | ~3 000 t/s | ~3 200 t/s | 对方快 ~5% |
| 启动（纯加载） | 12.3 s | 19.2 s | — |
| 172K 上下文显存 | 23 676 MiB | 24 083 MiB | — |

一句话总结：**prefill 打平、显存更省、decode 落后 ~11%**，而投机解码对两边收益相当
（各 +5~7%，我们略多），所以这个差距没有被投机放大。投机解码以前在这里是**净亏**（`n_max=2` 时 −13%），
现在变成净赚 —— 因为最优点其实是每步只起草 1 个 token（见下文「近期变更」），
而不是 llama.cpp 默认参数暗示的 3 个。

> 测量纪律很重要：跨会话的绝对值会漂 ±5%，只有**同会话、同 DLL** 的 A/B 才可信。

## 近期变更

| 日期 | 改动 | 效果 |
|---|---|---|
| 2026-10-26 | **投机解码改为成本感知自适应**：只在「0 / 1 个草稿」之间选；每个候选取 5 轮样本、**在 32 轮暖机之后**才探测（生成初期草稿质量偏低），按**吞吐中位数**比较；`--spec-draft-n-max` 默认 3 → 1，新增 `--spec-adaptive`（默认开） | decode **+6.8%**（HTTP 1200 token：128.3 → 137.0 t/s）；bench tg600：148 vs 131 t/s（+13%）；与 llama.cpp 差距 1.36× → 1.10× |
| 2026-10-23 | **升级 ggml DLL 到 llama.cpp 10-23**（含 flash_attn_ext_vec 的 shared-memory 竞态修复），同步 `ggml_op` 枚举并加 ABI 守卫测试 | decode **+3~4%** |

## 环境要求

- Windows 10/11（构建脚本与工具链是 PowerShell + MSVC，其它平台未验证）
- Visual Studio 2022（MSVC）、CUDA 12.5（nvcc）、CMake
- 24 GB 显存的 NVIDIA 显卡（开发机为 RTX 3090 / sm_86）
- **一份已构建好 CUDA `ggml` 的 llama.cpp checkout** —— ggml 的库不随本仓库分发
- 契合 `qwen35moe` 的 GGUF 模型（Qwen3.6-35B-A3B，可选带 `nextn.*` MTP 头）

## 构建

```powershell
# 1. 指定你的 llama.cpp 路径
$env:SQWEN_LLAMA_CPP = "D:\path\to\llama.cpp"
.\build_ggml.ps1            # 把 ggml 的 .lib/.dll 复制到 .\build-ggml\

# 2. 编译
cargo build --release       # 产物：target\release\sqwen.exe
```

注意事项：

- `build.rs` **每次构建都会**把 ggml 的 DLL 复制到可执行文件旁边。手动替换 DLL 时要放进
  `build-ggml\bin\Release\`，否则下一次 `cargo build` 会被覆盖回去。
- ggml 的 ABI 必须与 `src/backend/ffi.rs` 一致。FFI 是对着 llama.cpp `0.7.0-dev`（build 1218）
  核对过的。

## 运行

起一个 OpenAI 兼容端点（这是实际在用的配置）：

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

- `--repeat-penalty 1.08 --repeat-last-n 128` **不要省**：默认值 1.0 等于关闭，配上低温度时，
  模型一旦开始复读就自己停不下来（agent 场景表现为"卡住"）。
- `--reasoning-format deepseek` 把思考流进 `reasoning_content`，`content` 只留答案；
  `--reasoning-budget` 限制思考长度，且每次生成会自动留 256 token 给正文。
- `-c 172032` 是日常档位；`-c 200000` 也能装下，但只剩 ~380 MiB，再开任何 GPU 程序就会
  触发 WDDM 分页（速度断崖式下跌）。

其它模式：

```powershell
.\target\release\sqwen.exe chat  <model> -p "你好" -n 256 --temp 0.1
.\target\release\sqwen.exe bench <model> --prompt-tokens 512 --gen-tokens 128 -r 2
```

## HTTP API

| 端点 | 说明 |
|---|---|
| `POST /v1/chat/completions` | OpenAI 兼容；支持流式与非流式 |
| `GET /v1/models`、`GET /health` | 列模型 / 就绪检查（端口在模型预热完成后才绑定） |
| `GET /` | 内置 WebUI |
| `POST /v1/bench` | 快速跑分端点 |

**Tool calling** 的做法是把工具定义渲染进 chat template，生成后再从文本里抽
`<tool_call>{...}</tool_call>` —— 与 llama.cpp 同一条路线。流式下工具调用在**最后一个分片**整块发出。

已知限制（都是有意的取舍，见文档）：

- 请求**完全串行**（单引擎）；超过 `--queue-capacity` 返回 `429`。
- `tool_choice: "required"` 和指定函数名不强制，只尊重 `"none"`。
- 不支持 `stop`、`n>1`、`logprobs`、`response_format`、JSON Schema 约束解码。
- 工具调用的 JSON 偶尔会带未转义的引号；服务端会修常见形态，修不好就回退成正文，而不是发出坏调用。

## 诊断开关

| 环境变量 | 作用 |
|---|---|
| `SQWEN_PROFILE=1` | 逐步打印 `build / alloc / fill / compute / read` 计时 |
| `SQWEN_DUMP_GRAPH=1` | 图节点直方图 + 融合模式命中数 |
| `SQWEN_ABLATE=<模块>` | 把模块替换成零来量它的耗时（`attn,gdn,moe,lm_head,norm,gdn_proj,gdn_conv,gdn_core,gdn_out`） |
| `SQWEN_TOOL_DEBUG=1` | 每请求的工具调用解析摘要 |
| `SQWEN_NO_LOOP_GUARD=1` | 关闭重复死循环兜底 |
| `SQWEN_EXPERT_USED=n` | 覆盖 top-k 专家数（仅供测量） |

## 目录结构

```
src/
  main.rs        CLI：chat / bench / server（clap）
  lib.rs
  backend/       ggml + ggml-cuda 的 FFI，张量 / 上下文封装
  model/         架构、权重、建图、attn、gdn、moe、mtp
  engine/        runner（prefill/decode/投机）、kv、sampler、prompt_cache、reasoning、timing
  server/        api、queue、bench、toolcall、webui.html
  template/      minijinja 模板渲染、tokenizer
assets/          回退用的 chat template 与 tokenizer
scripts/         ABBA 对照脚本、SSE 探针、OpenAI SDK 校验
tests/           graph / attn / gdn / moe / sampler / server 测试
```

## 状态

早期项目：只支持一个架构族、只能 Windows 构建、同一时刻只处理一个序列。
它已经快到可以日常使用（本项目的开发过程就是用它配合 agent 框架跑出来的），
但不是 llama.cpp 的平替。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。
