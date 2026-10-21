import re
import collections

import sys

# 用法：python scripts/opdump_layer_histogram.py <opdump.txt>
# 文件内容形如 SQWEN_DUMP_GRAPH=1 sqwen bench ... 2> opdump.txt 的输出。
path = sys.argv[1] if len(sys.argv) > 1 else "opdump.txt"
lines = open(path, encoding='utf-8', errors='replace').read().splitlines()

blocks, cur = [], None
for ln in lines:
    if '[graph] n_tokens=' in ln:
        cur = {'hdr': ln, 'seq': []}
        blocks.append(cur)
    elif ln.startswith('[seq]') and cur is not None:
        m = re.match(r'\[seq\]\s+(\d+)\s+(\S+)', ln)
        if m:
            cur['seq'].append(m.group(2))

seq = [b for b in blocks if 'n_tokens=1' in b['hdr']][0]['seq']
FREE = {'VIEW', 'RESHAPE', 'PERMUTE', 'TRANSPOSE', 'NONE'}

# 用 MUL_MAT_ID 每层 3 个来切层：第 3 个 MUL_MAT_ID 之后为层尾
mmid = [i for i, o in enumerate(seq) if o == 'MUL_MAT_ID']
print(f"MUL_MAT_ID 共 {len(mmid)} 个 = {len(mmid)/3:.0f} 层 × 3")

# 更稳的切法：用 GATED_DELTA_NET / FLASH_ATTN_EXT 作为层内的锚，
# 层的边界取「下一个 MUL_MAT（MoE 的 down 投影之后）」
# 直接按 n_embd 投影：MoE 的 down 是 MUL_MAT_ID，取每层最后一个 MUL_MAT_ID 作为层结束
layer_ends = [mmid[2], mmid[5], mmid[8], mmid[len(mmid)-1]]
bounds = [0] + [mmid[i] for i in range(2, len(mmid), 3)] + [len(seq)]
# 去重、排序
bounds = sorted(set(bounds))

print(f"\n==== 每层算子数（共 {len(bounds)-1} 段）====")
sizes = []
for a, b in zip(bounds, bounds[1:]):
    seg = seq[a:b]
    real = [o for o in seg if o not in FREE]
    sizes.append((len(seg), len(real), seg))
    print(f"  [{a:5},{b:5})  节点 {len(seg):3}  真 kernel {len(real):3}")
    if a > 400:
        break

print("\n==== 一个全注意力层的算子序列（第 4 段，含 MoE）====")
# 找包含 FLASH_ATTN_EXT 的那段
for a, b in zip(bounds, bounds[1:]):
    seg = seq[a:b]
    if 'FLASH_ATTN_EXT' in seg:
        c = collections.Counter(seg)
        print(f"  段 [{a},{b}) 节点 {len(seg)}")
        print(f"  直方图: {dict(sorted(c.items(), key=lambda kv: -kv[1]))}")
        print("  序列:")
        for i, o in enumerate(seg):
            print(f"    {a+i:5} {o}")
        break

print("\n==== 全图：真 kernel 的构成 ====")
tot = collections.Counter(seq)
real = {k: v for k, v in tot.items() if k not in FREE}
s = sum(real.values())
for k, v in sorted(real.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<18} {v:5}   {v/s*100:5.1f}%")
print(f"  真 kernel 合计 {s} / 节点 {len(seq)}")
