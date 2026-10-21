"""原始 SSE 探针：按时间分桶统计分片到达，看「真流式」还是「攒完再发」。

这是定位「外部应用完全没有响应」的第一步（retro ⛳ 清单第 1 条）。
修之前的输出是：HTTP 头 1.3s 就到，**第一个 body 字节 7.99s 才到，然后 126 个分片
全挤在同一时刻**；而且每个分片都是 `data: data: {...}`（双前缀）。

用法：python scripts/server_sse_probe.py [port] [max_tokens] [stream|nostream]
"""
import http.client
import json
import sys
import time

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
stream = (sys.argv[3] if len(sys.argv) > 3 else "stream") == "stream"

body = json.dumps({
    "model": "sqwen",
    "messages": [{"role": "user", "content": "写一段 200 字左右的短文，讲缓存为什么会快。"}],
    "temperature": 0.1,
    "max_tokens": max_tokens,
    "stream": stream,
})

conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
t0 = time.time()
conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
resp = conn.getresponse()
print(f"[{time.time()-t0:7.3f}s] HTTP {resp.status}")

chunks = []          # (到达时刻, 字节数, 前 90 字节)
total = 0
while True:
    piece = resp.read1(1 << 16)
    if not piece:
        break
    total += len(piece)
    chunks.append((time.time() - t0, len(piece), piece[:90]))

if not chunks:
    print("!! 一个字节都没收到")
    sys.exit(1)

# 按 250ms 分桶
print("\n分片到达分布（250ms 一桶）：")
buckets = {}
for t, n, _ in chunks:
    buckets.setdefault(int(t / 0.25), [0, 0])
    buckets[int(t / 0.25)][0] += 1
    buckets[int(t / 0.25)][1] += n
for k in sorted(buckets):
    c, b = buckets[k]
    print(f"  {k*0.25:6.2f}~{k*0.25+0.25:6.2f}s | {'#' * min(c, 60):60s} {c:4d} 片 / {b:6d} B")

print("\n前 3 片与最后 2 片：")
for i, (t, n, head) in enumerate(chunks[:3]):
    print(f"  #{i+1} [{t:7.3f}s] {head!r}")
for i, (t, n, head) in enumerate(chunks[-2:]):
    print(f"  #末-{1-i} [{t:7.3f}s] {head!r}")

first = chunks[0][0]
last = chunks[-1][0]
joined = b"".join(c[2] for c in chunks)
print(f"\n分片={len(chunks)} 字节={total} 首字节={first:.3f}s 末字节={last:.3f}s")
print("有 data: data: 双前缀？", b"data: data:" in joined)
print("有 [DONE]？", b"[DONE]" in joined)
print("是合法 SSE JSON 行？", joined.startswith(b"data: {") or joined.startswith(b"data:["))
print(
    "真流式？（末字节明显晚于首字节，或首字节远早于总耗时）:",
    (last - first) > 0.1,
)
