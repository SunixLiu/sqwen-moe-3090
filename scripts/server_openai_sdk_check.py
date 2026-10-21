"""「外部应用」视角的验收脚本：用官方 OpenAI Python SDK 打 sqwen 的 server。

起因（retro 会话复盘（内部资料，未随仓库发布） 的 ⛳）：
用其他应用调用 sqwen 的推理服务「完全没有响应」。SDK 会做三件我们当时没满足的事：
  1. 先 GET /v1/models（列模型）；
  2. 流式：按 SSE 规范拆 `data: ` 前缀、要求 `[DONE]` 才结束、每片 `json.loads`；
  3. 非流式：读 `choices[0].message.content`，且 id/created/finish_reason/usage 必须存在。

用法：python scripts/server_openai_sdk_check.py [port] [max_tokens]
"""
import os
import sys
import time

# 这台机器的 NO_PROXY 里写了 `[::1]`，httpx 解析它就抛 InvalidURL（与 sqwen 无关）。
# 为了不把环境问题混进验收，这里先把代理环境变量清掉。
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
           "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
    os.environ.pop(_k, None)

from openai import OpenAI

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 512

client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="not-needed", timeout=600.0)

print("== 1. GET /v1/models ==")
models = [m.id for m in client.models.list()]
print("   models:", models)
assert "sqwen" in models, "SDK 列不出模型（很多应用会在这里直接罢手）"

prompt = "用一句话说明什么是缓存。"

print(f"== 2. 非流式（max_tokens={max_tokens}）==")
t0 = time.time()
r = client.chat.completions.create(
    model="sqwen",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1,
    max_tokens=max_tokens,
)
print(f"   id={r.id} created={r.created} object={r.object}")
print(f"   finish_reason={r.choices[0].finish_reason} usage={r.usage}")
print(f"   content={r.choices[0].message.content!r}")
print(f"   耗时 {time.time()-t0:.2f}s")
assert r.choices[0].message.content, "非流式回答为空（客户端会以为没响应）"
assert r.usage.total_tokens is not None, "usage 缺 total_tokens（SDK 会解析成 None）"

print(f"== 3. 流式（max_tokens={max_tokens}）==")
t0 = time.time()
first = None
chunks = 0
text = []
reasoning = 0
finish = None
for chunk in client.chat.completions.create(
    model="sqwen",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1,
    max_tokens=max_tokens,
    stream=True,
):
    chunks += 1
    if first is None:
        first = time.time() - t0
    d = chunk.choices[0].delta
    if getattr(d, "reasoning_content", None):
        reasoning += len(d.reasoning_content)
    if d.content:
        text.append(d.content)
    if chunk.choices[0].finish_reason:
        finish = chunk.choices[0].finish_reason
total = time.time() - t0
answer = "".join(text)
print(f"   分片数={chunks} 首帧={first:.3f}s 总耗时={total:.2f}s finish_reason={finish}")
print(f"   reasoning_content 字符数={reasoning}")
print(f"   content={answer!r}")
assert first is not None and first < total, "首帧不比总耗时早 —— 还是假流式"
print("   提示：`--reasoning-format answer`（默认）下推理不进流，首帧 = 思考结束的时刻；")
print("         想让首帧落在 prefill 结束（思考跟着流出来）就用 `--reasoning-format deepseek`。")

print("== 4. 小 max_tokens（思考吃掉全部预算）==")
tiny = client.chat.completions.create(
    model="sqwen",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1,
    max_tokens=64,
    stream=True,
)
n_tiny = 0
tiny_finish = None
for chunk in tiny:
    n_tiny += 1
    if chunk.choices[0].finish_reason:
        tiny_finish = chunk.choices[0].finish_reason
print(f"   分片数={n_tiny} finish_reason={tiny_finish}")
assert tiny_finish == "length", f"用满 max_tokens 应报 length：{tiny_finish}"
print("   流正常收尾（不是挂住）→ server stderr 上应有一条 `warning: 回答为空…`")
print("OK：OpenAI SDK 走通了列模型 / 非流式 / 流式 / 小预算收尾四条路")
