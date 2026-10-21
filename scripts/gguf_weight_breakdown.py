"""按类别统计 GGUF 权重字节数，并估算 decode 每 token 的权重流量。

用途：判断 decode 到底是"权重带宽受限"还是"kernel 固定开销受限"。
"""
import struct
import sys
import collections

path = sys.argv[1]
f = open(path, 'rb')

def rd(fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

magic = f.read(4)
assert magic == b'GGUF', magic
version, = rd('<I')
n_tensors, = rd('<Q')
n_kv, = rd('<Q')

GGML_TYPE = {
    0: ('F32', 1, 4), 1: ('F16', 1, 2), 2: ('Q4_0', 32, 18), 3: ('Q4_1', 32, 20),
    6: ('Q5_0', 32, 22), 7: ('Q5_1', 32, 24), 8: ('Q8_0', 32, 34), 9: ('Q8_1', 32, 36),
    10: ('Q2_K', 256, 84), 11: ('Q3_K', 256, 110), 12: ('Q4_K', 256, 144),
    13: ('Q5_K', 256, 176), 14: ('Q6_K', 256, 210), 15: ('Q8_K', 256, 292),
    20: ('IQ4_NL', 32, 18), 23: ('IQ4_XS', 256, 136), 24: ('I8', 1, 1),
    25: ('I16', 1, 2), 26: ('I32', 1, 4), 27: ('I64', 1, 8), 28: ('F64', 1, 8),
    30: ('BF16', 1, 2),
}

def read_string(f):
    n, = rd('<Q')
    return f.read(n).decode('utf-8', 'replace')

def read_value(f, t):
    if t == 0: return rd('<B')[0]
    if t == 1: return rd('<b')[0]
    if t == 2: return rd('<H')[0]
    if t == 3: return rd('<h')[0]
    if t == 4: return rd('<I')[0]
    if t == 5: return rd('<i')[0]
    if t == 6: return rd('<f')[0]
    if t == 7: return rd('<?')[0]
    if t == 8: return read_string(f)
    if t == 9:
        et, = rd('<I')
        n, = rd('<Q')
        return [read_value(f, et) for _ in range(n)]
    if t == 10: return rd('<Q')[0]
    if t == 11: return rd('<q')[0]
    if t == 12: return rd('<d')[0]
    raise ValueError(t)

meta = {}
for _ in range(n_kv):
    key = read_string(f)
    t, = rd('<I')
    meta[key] = read_value(f, t)

tensors = []
for _ in range(n_tensors):
    name = read_string(f)
    nd, = rd('<I')
    dims = [rd('<Q')[0] for _ in range(nd)]
    tt, = rd('<I')
    rd('<Q')  # offset
    tensors.append((name, dims, tt))
f.close()

arch = meta.get('general.architecture', '?')
print(f"arch = {arch}   GGUF v{version}   tensors = {n_tensors}")
print("---- 关键元数据 ----")
for k in sorted(meta):
    if any(s in k for s in ('block_count', 'expert', 'embedding_length',
                            'feed_forward_length', 'head_count', 'context_length')):
        v = meta[k]
        if isinstance(v, list) and len(v) > 8:
            v = f"[{len(v)} 项]"
        print(f"  {k} = {v}")

n_layer = meta.get(f'{arch}.block_count')
n_expert = meta.get(f'{arch}.expert_count')
n_used = meta.get(f'{arch}.expert_used_count')
print(f"\n==== 每层 {n_expert} 个专家，激活 {n_used} 个 ====\n")

def nbytes(dims, tt):
    ne = 1
    for d in dims:
        ne *= d
    name, bs, ts = GGML_TYPE[tt]
    assert ne % bs == 0, (dims, tt)
    return ne // bs * ts

cat = collections.defaultdict(int)
cat_n = collections.defaultdict(int)
expert_bytes_per_layer = collections.defaultdict(int)
per_layer = collections.defaultdict(int)

for name, dims, tt in tensors:
    nb = nbytes(dims, tt)
    if '_exps' in name:
        c = 'expert (每层 256 个)'
        expert_bytes_per_layer[name.split('.')[0] if False else name] += nb
    elif 'shexp' in name or 'shared' in name:
        c = 'shared expert'
    elif name == 'token_embd.weight':
        c = 'token_embd (只读 1 行)'
    elif name == 'output.weight':
        c = 'lm_head (每 token 全读)'
    elif '.attn_' in name or '.ffn_' in name or '.ssm_' in name or '.conv' in name:
        c = 'dense (attn / ssm / 共享)'
    else:
        c = 'norm / 其他'
    cat[c] += nb
    cat_n[c] += 1

def mb(x):
    return f"{x/1024/1024:,.0f} MiB"

total = sum(cat.values())
print(f"{'类别':<26}{'字节':>14}{'张量数':>8}")
for c, v in sorted(cat.items(), key=lambda kv: -kv[1]):
    print(f"{c:<26}{mb(v):>14}{cat_n[c]:>8}")
print(f"{'合计':<26}{mb(total):>14}{n_tensors:>8}")

# 每 token 权重流量
expert_total = cat['expert (每层 256 个)']
expert_active = expert_total * n_used / n_expert
dense = cat['dense (attn / ssm / 共享)'] + cat['norm / 其他']
lm = cat['lm_head (每 token 全读)']
embd_row = 0  # get_rows 只读一行

per_tok = dense + expert_active + lm + embd_row
print(f"\n==== decode 每 token 权重流量 ====")
print(f"  稠密 (attn/ssm/norm)      {mb(dense):>14}   每 token 全读")
print(f"  lm_head                   {mb(lm):>14}   每 token 全读")
print(f"  专家 top-{n_used}/{n_expert}          {mb(expert_active):>14}   = {mb(expert_total)} × {n_used}/{n_expert}")
print(f"  ------------------------------------------------")
print(f"  合计                      {mb(per_tok):>14}")

for bw in (936, 800, 700):
    print(f"    带宽 {bw} GB/s → 理论下限 {per_tok/1e9/bw*1000:.2f} ms/token")
