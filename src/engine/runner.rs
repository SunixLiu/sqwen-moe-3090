// src/engine/runner.rs — prefill / decode 生成循环
//
// 一次 `generate` 的流程：
//
//   1. **重置**：清空 KV cache + GDN 常驻状态（同一 engine 连续跑两段序列必须互不影响）
//   2. **prefill**：prompt 按 `ubatch_size` 分块，每块一次前向（第一块里 30 个 GDN 层
//      是整块并行、10 个注意力层写自己的 KV 位置）
//   3. 取最后一块的 logits 采出第一个 token（这就是 ttft 的时刻）
//   4. **decode**：每步喂 1 个 token、位置 +1，采样下一个；遇到 EOS / 达到
//      `max_tokens` / 达到 `max_context` 就停
//
// 每步用一个 scratch ctx 建图；**decode 的图跨步复用**（P2c，见 `CachedGraph`）：
// 图节点（每步 ~3400 个）连同它们的后端 buffer 一次建好，之后每步只重写输入张量的
// 内容。权重与常驻状态住在 `Engine` 自己的 `ctx` 里，跨步存活（`StateTensor` 让这两者
// 能被同一个结构持有 —— 见 backend/ctx.rs 的说明）。

use std::time::Instant;

use anyhow::{anyhow, Result};

use crate::backend::ctx::{
    BackendBuffer, BackendKind, GgmlCtx, GraphAllocator, GraphBuilder, RawGraph, StateTensor,
};
use crate::backend::ffi;
use crate::engine::sampler::Sampler;
use crate::engine::timing::Timing;
use crate::model::arch::Qwen35MoeArch;
use crate::model::graph::{forward, EngineState, GRAPH_NODE_BUDGET};
use crate::model::mtp::MtpMode;
use crate::model::weights::{GgufFile, ModelWeights};

/// 生成配置。
/// FlashAttention 模式（P4）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlashAttnMode {
    /// 后端支持就用（默认）
    Auto,
    /// 必须用；不支持则加载失败
    On,
    /// 禁用（走手工注意力，V cache 退回 f32）
    Off,
}

/// 投机解码类型（对齐 llama.cpp 的 `--spec-type`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecType {
    /// 关闭（默认，同 llama.cpp）
    None,
    /// `draft-mtp`：用模型自带的 MTP / NextN 头（`blk.{n_layer}` + `nextn.*`）起草
    DraftMtp,
}

// ---------------------------------------------------------------------------
// 成本感知的起草预算（adaptive speculation）
// ---------------------------------------------------------------------------
//
// 实测结论（RTX 3090 / Qwen3.6-35B-A3B Q4_K_M / bench pp64 tg600 / `-r 3` ABBA）：
//
//   不投机         125~130 t/s
//   每轮起草 1 个  145~147 t/s   ← 最优
//   每轮起草 2 个  125 t/s
//   每轮起草 3 个  106 t/s
//
// 原因：草稿 token 的**边际成本**很高 —— 校验批里每多一个 token，就要多读它激活的
// 专家权重、多跑一遍 GDN/注意力（实测约等于一次完整 decode 的 60%），而接受率随
// 位置衰减（本模型 pos0≈0.9、pos1≈0.7）。所以固定深度越大越亏，最优通常就是 1。
//
// 因此自适应只在「0 个草稿」与「1 个草稿」之间选：
//   - 未决定 / 需要重估时：按 `PROBE` 序列各测 2 轮，只取每个深度的第 2 轮
//     （第 1 轮要吃 CUDA graph 切换/重捕获的开销），比较实测吞吐后**锁定**；
//   - 已决定则沿用，本请求内不再切换（切换一次 15~25 ms ≈ 2~3 个 token，而单次
//     生成的内容基本一致，来回切不划算）；
//   - 每 `REEVAL_REQUESTS` 个请求重新评估一次，跟随会话内容漂移。
#[derive(Clone, Copy)]
struct SpecObs {
    /// 该深度最近 5 轮的墙钟（ms，已剔除仅为保持 MTP 同步的重建时间）
    ms: [f64; SpecObs::MAX],
    /// 与该轮对应的被接受草稿数
    acc: [f64; SpecObs::MAX],
    /// 已收集的样本数
    n: usize,
}

impl SpecObs {
    /// 每个深度收集多少个样本再下结论。
    const MAX: usize = 5;
    const EMPTY: Self = Self {
        ms: [0.0; Self::MAX],
        acc: [0.0; Self::MAX],
        n: 0,
    };

    fn push(&mut self, ms: f64, acc: f64) {
        if self.n < Self::MAX {
            self.ms[self.n] = ms;
            self.acc[self.n] = acc;
            self.n += 1;
        }
    }

    /// **逐样本**算吞吐再取中位数。
    ///
    /// 不能「取最省时那一轮、再用它的接受数」：接受与否和那一轮的耗时并不相关，
    /// 实测出现过「最快的一轮恰好草稿被拒（acc=0）」→ 吞吐被低估成 86 t/s，
    /// 于是策略误判成不投机（真实稳态是 145 t/s）。中位数同时抗冷启动与坏样本。
    fn throughput(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        let mut t = [0.0f64; Self::MAX];
        for i in 0..self.n {
            t[i] = if self.ms[i] > 0.0 {
                (1.0 + self.acc[i]) / self.ms[i]
            } else {
                0.0
            };
        }
        let s = &mut t[..self.n];
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        s[self.n / 2]
    }

    /// 中位数耗时（诊断用）
    fn ms_median(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        let mut m = [0.0f64; Self::MAX];
        m[..self.n].copy_from_slice(&self.ms[..self.n]);
        let s = &mut m[..self.n];
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        s[self.n / 2]
    }

    /// 平均接受数（诊断用）
    fn acc_mean(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.acc[..self.n].iter().sum::<f64>() / self.n as f64
        }
    }

    fn valid(&self) -> bool {
        self.n > 0
    }
}

struct SpecPolicy {
    /// `--spec-draft-n-max` 上限（自适应模式下实际只用到 1）
    n_max: usize,
    adaptive: bool,
    /// 候选深度 0（不投机）与 1（起草一个）的最近观测
    obs: [SpecObs; 2],
    /// 已确定的决定：`None` = 还没测过
    decision: Option<bool>,
    /// 本请求锁定为不投机（可直接省掉 MTP KV 重建）
    plain_latched: bool,
    /// 探测序列进行到第几步（不在探测期为 `None`）
    probe_step: Option<u32>,
    /// 本请求已经过的轮数（探测前的暖机判断）
    rounds: u32,
    /// 进程内请求计数（决定何时重新评估）
    requests: u64,
}

impl SpecPolicy {
    /// 探测序列：先测投机、再测不投机（锁定不投机后 MTP 状态就再没人读了）。
    /// 每个深度测 `SpecObs::MAX` 轮，**逐样本算吞吐取中位数**：首轮仍可能吃 CUDA graph
    /// 重捕获，中位数天然把它滤掉；也避免「最快那轮恰好草稿被拒」把吞吐算低。
    const PROBE: [usize; 10] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
    /// 探测前先按投机跑这么多轮，把该形状的 CUDA graph 跑热（前几步是冷启动区）。
    const PROBE_DELAY: u32 = 32;
    /// 每隔多少个请求重新评估一次（内容漂移）。
    const REEVAL_REQUESTS: u64 = 8;
    /// 投机要胜出所需的领先幅度：切换深度有固定成本，优势不显著就不切。
    const SPEC_MARGIN: f64 = 1.05;

    fn new(n_max: usize, adaptive: bool) -> Self {
        Self {
            n_max,
            adaptive,
            obs: [SpecObs::EMPTY; 2],
            decision: None,
            plain_latched: false,
            probe_step: None,
            rounds: 0,
            requests: 0,
        }
    }

    /// 新请求：沿用既有决定，必要时安排一次重新探测。
    fn begin_request(&mut self) {
        self.requests = self.requests.wrapping_add(1);
        self.plain_latched = self.decision == Some(false);
        let need_probe =
            self.decision.is_none() || self.requests % Self::REEVAL_REQUESTS == 0;
        self.probe_step = if need_probe && self.adaptive && self.n_max > 0 {
            Some(0)
        } else {
            None
        };
        self.rounds = 0;
        self.obs = [SpecObs::EMPTY; 2];
    }

    /// 本请求是否已锁定为不投机。
    fn is_plain(&self) -> bool {
        self.plain_latched
    }

    /// 选本轮起草几个 token（0 = 不投机）。
    fn pick(&mut self) -> usize {
        if !self.adaptive {
            return self.n_max;
        }
        if let Some(step) = self.probe_step {
            if self.rounds < Self::PROBE_DELAY {
                // 探测前的暖机轮：统一按投机跑，把该形状的 CUDA graph 跑热
                return 1.min(self.n_max);
            }
            let k = Self::PROBE[(step as usize).min(Self::PROBE.len() - 1)];
            return k.min(self.n_max);
        }
        match self.decision {
            Some(true) => 1.min(self.n_max),
            Some(false) => 0,
            None => self.n_max,
        }
    }

    /// 一轮结束后反馈：探测期记录观测、序列走完时做决定；其余时候只推进轮数。
    fn on_round(&mut self, k: usize, accepted: usize, round_ms: f64) {
        self.rounds = self.rounds.saturating_add(1);
        let Some(step) = self.probe_step else { return };
        if self.rounds <= Self::PROBE_DELAY {
            return; // 暖机轮不采样
        }
        if k < 2 && round_ms > 0.0 {
            // 逐样本记录；吞吐在 `SpecObs::throughput` 里按中位数算（抗冷启动/坏样本）
            self.obs[k].push(round_ms, accepted as f64);
        }
        let next = step + 1;
        if next as usize >= Self::PROBE.len() {
            self.finish_probe();
        } else {
            self.probe_step = Some(next);
        }
    }

    fn finish_probe(&mut self) {
        self.probe_step = None;
        let (plain, spec) = (self.obs[0], self.obs[1]);
        if !plain.valid() || !spec.valid() {
            return; // 样本不完整：保持未决定，下次请求再测
        }
        let spec_wins = spec.throughput() >= plain.throughput() * Self::SPEC_MARGIN;
        self.decision = Some(spec_wins);
        self.plain_latched = !spec_wins;
    }

    /// 诊断：决定 + 观测。
    fn report(&self) -> String {
        let d = match self.decision {
            None => "未定".to_string(),
            Some(true) => "投机 k=1".to_string(),
            Some(false) => "不投机".to_string(),
        };
        format!(
            "决定={d} | k0 {:.1}ms/{:.0}t/s(n{}) | k1 {:.1}ms/acc{:.2}/{:.0}t/s(n{})",
            self.obs[0].ms_median(),
            self.obs[0].throughput() * 1000.0,
            self.obs[0].n,
            self.obs[1].ms_median(),
            self.obs[1].acc_mean(),
            self.obs[1].throughput() * 1000.0,
            self.obs[1].n,
        )
    }
}

/// `SQWEN_PHASE=1`：累积 decode 每步的分阶段耗时，只在每轮生成结束时打一行。
/// （逐步 `eprintln!` 自己就要 1~2 ms，会把被测的每步耗时抬高。）
fn phase_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("SQWEN_PHASE").is_some())
}

#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// 上下文上限，同时也是 KV cache 的容量（超过就报错，不做抢占/滚动）
    pub max_context: i64,
    /// 逻辑 batch 上限（对齐 llama.cpp 的 `-b/--batch-size`，默认 2048）。
    /// 目前只用来约束 ubatch（`ubatch = min(ubatch, batch)`）；server 连续批处理会用到。
    pub batch_size: i64,
    /// **物理** batch：一次前向最多处理多少 token（对齐 llama.cpp 的 `-ub/--ubatch-size`，
    /// 默认 512）。prefill 按它分块，每块建一次图。
    pub ubatch_size: i64,
    /// FlashAttention 模式（决定 V cache 用 q4_0 还是 f32）
    pub flash_attn: FlashAttnMode,
    /// 主机侧并行度（`-t/--threads`），0 = 自动（物理核数）。
    /// 目前用于并行填充因果 mask（大上下文下每块上百 MB 的纯 CPU 访存）。
    /// 注意：全 GPU 模式下它**不影响** GPU 计算，只影响主机侧的那部分工作。
    pub threads: i64,
    /// prompt 前缀复用（`--cache-prompt`，P6.3）。默认 **true**（同 llama.cpp 的
    /// `cache_prompt`）：多轮请求里新 prompt 的前缀通常与上一次完全相同，复用能把
    /// TTFT 从 O(历史长度) 降到 O(新增 token)。
    pub cache_prompt: bool,
    /// 投机解码类型（`--spec-type`）
    pub spec_type: SpecType,
    /// `--spec-draft-n-max`：每轮最多起草几个 token（默认 3，同 llama.cpp）
    pub spec_draft_n_max: usize,
    /// 深度自适应（`--spec-adaptive`，默认开）：按实测吞吐在线决定每轮起草几个
    /// token；关掉则固定用 `spec_draft_n_max`（旧行为）。
    pub spec_adaptive: bool,
    pub sampler: Sampler,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_context: 4096,
            batch_size: 2048,
            ubatch_size: 512,
            flash_attn: FlashAttnMode::Auto,
            threads: 0,
            cache_prompt: true,
            spec_type: SpecType::None,
            spec_draft_n_max: 3,
            spec_adaptive: true,
            sampler: Sampler::new(0.0, 0, 1.0, 42),
        }
    }
}

/// 校验 prompt + max_tokens 是否放得下；放不下在**建图之前**报错。
pub fn check_budget(prompt_len: usize, max_tokens: usize, max_context: usize) -> Result<()> {
    if prompt_len == 0 {
        return Err(anyhow!("prompt 为空"));
    }
    if prompt_len > max_context {
        return Err(anyhow!(
            "prompt 长度 {prompt_len} 超过上下文上限 {max_context}"
        ));
    }
    if prompt_len + max_tokens > max_context {
        return Err(anyhow!(
            "prompt {prompt_len} + 生成 {max_tokens} 超过上下文上限 {max_context}"
        ));
    }
    Ok(())
}

/// 决定"本步的图与上一步是否同构"的量（P2c）。
///
/// 图输入张量的**内容**（tokens / pos / mask / kv_idxs 的值）不在签名里 —— 那些每步
/// 都要重写，但改内容不影响节点属性，CUDA graph 照样重放。会改变图结构的只有这三项：
///   - `n_tokens`：prefill 分块长度（decode 恒为 1）；
///   - `out_ids`：最后一层要取哪几行 logits（prefill 取 `[n-1]`，decode 取 `[0]`）；
///   - `n_kv_padded`：mask 与 K/V 视图的形状，`KV_PAD`（256）对齐 → 256 步一个窗口。
#[derive(PartialEq, Eq, Debug)]
struct StepSig {
    n_tokens: usize,
    out_ids: Option<Vec<i32>>,
    n_kv_padded: usize,
}

/// 跨步复用的图（P2c）。
///
/// 为什么值得做：decode 每步的图在 256 步窗口内**完全同构**，而重建它要付
/// build 1.0 ms + `gallocr.alloc` 0.45 ms 的主机侧开销（`SQWEN_PROFILE=1` 实测，
/// 约占 decode 每 token 11 ms 的 13%）。省掉后每步只剩"重写输入 → compute → 回读"。
///
/// 为什么安全：整张图是「常驻状态 + 4 个输入张量」的纯数据流 —— K/V 用 `set_rows`
/// 写进固定指针的整块 cache，GDN 的递推状态用 `cpy` 写回常驻张量，RoPE 的位置来自
/// `pos` 张量的内容。所以重跑同一张图 = 正确的下一步（llama.cpp 的 CUDA graph 重放
/// 依赖的也是同一个性质）。
///
/// **失效条件**（必须成对处理）：任何一次 `scratch.reset()` 都会把图节点所在的 arena
/// 内存交还给后续分配，因此重建路径必须先把它置 `None`；同理，缓存存活期间不能再对
/// 别的图调 `gallocr.alloc`（会重排 buffer，把 `data` 指针改写掉）。
struct CachedGraph {
    graph: RawGraph,
    /// 图的输出（logits）张量；复用时不重建，直接读它。
    logits: StateTensor,
    sig: StepSig,
}

/// `--cache-prompt` 的检查点间隔：每多少个 token 存一份 GDN 状态快照。
///
/// 它决定"分歧点落在 prompt 中间"时要多算多少 token（最多一个间隔）。
/// 4096 在 1700 t/s 的 prefill 下约 2.4 s；同时把快照数量控制在
/// `prompt_len / 4096`（真正保留多少由 `PromptCache` 的字节预算决定）。
const CKPT_INTERVAL: usize = 4096;

/// 前缀复用门槛：省下的 token 占比低于它就不复用（同 llama.cpp 的 `f_keep ≥ 0.25`）。
/// 低于这个比例时，恢复状态 + 重新攒图的固定开销反而不划算。
const MIN_KEEP_FRAC: f32 = 0.25;

/// MTP 图的节点上限。
///
/// MTP 头只有 1 个解码块（attn + MoE）加几条投影，实测约 70 个节点；主干是 ~3400。
/// 因此 MTP 图**不做跨步复用**：每步重建 + 分配的开销（约 0.03 ms）远小于主干图的
/// 1.4 ms，而换来的是"每步可以有不同的批大小"（起草是 1 行、KV 重建是 n 行）。
const MTP_NODE_BUDGET: usize = 2048;

/// 稳态速率窗口的起点：生成开始后多少毫秒才计入 `gen_tokens_steady`。
///
/// 取 3 秒是为了对齐 llama.cpp server 的 `tg_3s`（实测它也是约 5~6 秒后才到稳态）。
/// GPU 从空闲 P8/270MHz 爬到负载 P2/1900MHz 要几十秒，前几秒基本在爬升段 ——
/// 把它算进平均会让累计速率系统性偏低（同一段生成实测 133.97 vs 窗口 143）。
const STEADY_SKIP_MS: f64 = 3000.0;

/// 稳态窗口内至少要这么多 token，速率才有意义；否则报 0 让调用方不显示。
const MIN_STEADY_TOKENS: usize = 32;

/// 每次生成给**正文**预留的 token 数（从思考预算里扣）。
///
/// 没有它的话，`max_tokens` 小一点的请求会被思考吃干：`finish_reason=length`、
/// `content` 为空，看起来像"模型没回答"。256 是"最短的有意义答案"的量级；
/// 真要写长文，客户端本来也该给更大的 `max_tokens`。
const ANSWER_TOKEN_RESERVE: usize = 256;

/// 重复死循环检测：窗口大小（token）与要求连续重复的次数。
///
/// 为什么要它：`--repeat-penalty` 默认是 1.0（关闭），temp 又常常很低，
/// 模型一旦开始整段复读就会自我强化，几十秒都停不下来（agent 侧表现就是
/// "卡住不动"）。这是兜底，不是采样策略 —— 96 个 token **逐字**重复 3 次，
/// 正常的代码/文本几乎不可能出现。
const LOOP_WINDOW: usize = 96;
const LOOP_REPEATS: usize = 3;
/// 每多少个 token 检查一次（太密浪费，太疏会让用户多等）。
const LOOP_CHECK_EVERY: usize = 16;

/// `out` 的末尾是否已经「连续重复同一段 `LOOP_WINDOW` 个 token、共 `LOOP_REPEATS` 次」。
fn is_degenerate_loop(out: &[u32]) -> bool {
    let w = LOOP_WINDOW;
    if out.len() < w * LOOP_REPEATS {
        return false;
    }
    let n = out.len();
    let last = &out[n - w..];
    for k in 1..LOOP_REPEATS {
        let start = n - w * (k + 1);
        if &out[start..start + w] != last {
            return false;
        }
    }
    true
}

/// 把 `[start, end)` 切成 prefill 分块：每块 ≤ `ubatch`，并在 `cut` 里的位置**正好切一刀**。
///
/// 为什么要在检查点位置切开：状态快照只能在"一步做完"的位置取，否则拿到的是
/// 半个 chunk 的中间状态。调用方给的历史边界（例如聊天历史的末尾）必须落在块边界上，
/// 下一个请求才能从它直接接上。
fn plan_chunks(
    start: usize,
    end: usize,
    ubatch: usize,
    cut: &[usize],
) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut from = start;
    while from < end {
        let mut to = (from + ubatch).min(end);
        if let Some(&c) = cut.iter().find(|&&c| c > from && c < to) {
            to = c;
        }
        out.push((from, to));
        from = to;
    }
    out
}

/// 取 logits 的 argmax（草稿 token；草稿本身不需要随机性 —— 同 llama.cpp 的
/// "top-k=10 后取候选 0"）。并列时取下标最小的那个，保证确定性。
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// 解析 `-t/--threads`：0 = 自动。
///
/// 自动值 = `min(逻辑核数, 8)`。理由：
///   - `available_parallelism()` 是逻辑核（含超线程），物理核通常是一半；
///   - 目前用线程的地方是**因果 mask 填充**（纯 memcpy/memset、访存带宽敏感），
///     实测 8 线程左右就饱和，再多会因带宽争用略慢；
///   - 与用户在 llama.cpp 上的经验一致（20 核机器上 8 线程最优）。
/// 显式传值时按用户给的来（只做 >= 1 的下限保护），便于自己扫最优。
fn resolve_threads(requested: i64) -> usize {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    if requested <= 0 {
        logical.min(8).max(1)
    } else {
        requested.max(1) as usize
    }
}

/// 探测后端是否支持 FlashAttention（P4）。
///
/// 做法：用与真实推理**完全相同的类型与形状**建一个 FA 节点，然后问后端
/// `ggml_backend_supports_op`。这是必须的 —— 不支持时 ggml-cuda 的 FA 派发会
/// `GGML_ABORT` 直接杀进程，而不是返回错误。
///
/// 形状：q `[head_dim, 1, n_head, 1]`、k/v `[head_dim, n_kv, n_head_kv, 1]`、
/// mask F16 `[n_kv, 1, 1, 1]`。`n_kv` 用 256（对齐粒度）即可，支持性是类型/架构驱动的。
fn probe_flash_attn(
    ctx: &GgmlCtx,
    kind: BackendKind,
    arch: &Qwen35MoeArch,
    v_type: ffi::GgmlType,
    n_kv: usize,
) -> bool {
    // 临时后端的 buffer 只用来问一句 supports_op，张量不需要真实内存
    let backend = unsafe {
        match kind {
            BackendKind::Cpu => ffi::ggml_backend_cpu_init(),
            BackendKind::Cuda(d) => ffi::ggml_backend_cuda_init(d),
        }
    };
    if backend.is_null() {
        return false;
    }
    let head_dim = arch.head_dim as i64;
    let n_head = arch.n_head as i64;
    let n_head_kv = arch.n_head_kv as i64;
    let n_kv = n_kv.max(256) as i64;
    let q = ctx.new_tensor_f32(&[head_dim, 1, n_head, 1]);
    let k = ctx.new_tensor(crate::engine::kv::KV_TYPE_K, &[head_dim, n_kv, n_head_kv, 1]);
    let v = ctx.new_tensor(v_type, &[head_dim, n_kv, n_head_kv, 1]);
    let mask = ctx.new_tensor(ffi::GgmlType::F16, &[n_kv, 1, 1, 1]);
    let c = ctx.as_ptr();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let fa = ctx.wrap(unsafe {
        ffi::ggml_flash_attn_ext(
            c,
            q.ptr(),
            k.ptr(),
            v.ptr(),
            mask.ptr(),
            scale,
            0.0,
            0.0,
        )
    });
    let ok = unsafe { ffi::ggml_backend_supports_op(backend, fa.ptr()) };
    unsafe { ffi::ggml_backend_free(backend) };
    ok
}

/// `--ctx-size` 的解析结果（用于启动提示）。
#[derive(Clone, Copy, PartialEq, Debug)]
enum CtxNote {
    Ok,
    /// 用户要的比模型训练上下文大（允许，但要提示外推风险）
    ExceedsTrain,
    /// 模型没声明 context_length，用了兜底值
    NoTrainInfo,
}

impl CtxNote {
    fn text(self) -> &'static str {
        match self {
            CtxNote::Ok => "",
            CtxNote::ExceedsTrain => " ⚠ 超过训练上下文，长序列质量可能下降",
            CtxNote::NoTrainInfo => "（模型未声明 context_length）",
        }
    }
}

/// 把 `--ctx-size` 解析成实际使用的上下文长度。
///
/// - `0` → 模型训练上下文（llama.cpp `n_ctx = 0` 的语义）；模型没声明时用 4096
/// - 大于训练上下文 → 允许（与 llama.cpp 一致），但调用方会打印提示
fn resolve_ctx_size(requested: i64, n_ctx_train: usize) -> (usize, CtxNote) {
    let train = if n_ctx_train > 0 { n_ctx_train } else { 4096 };
    let note = if n_ctx_train == 0 {
        CtxNote::NoTrainInfo
    } else if requested as usize > train {
        CtxNote::ExceedsTrain
    } else {
        CtxNote::Ok
    };
    let ctx = if requested <= 0 {
        train
    } else {
        requested as usize
    };
    // 提示按实际情况给：0 + 模型没声明 → 告诉用户用了兜底值；显式超训练上下文 → 提示外推
    (ctx.max(1), note)
}

/// 推理引擎：拥有 ctx、权重、常驻状态与后端 buffer。
pub struct Engine {
    /// 权重 + 常驻状态所在的后端 buffer（必须在 ctx 之前释放，所以声明在最前）
    persistent: BackendBuffer,
    /// 每步图的分配器：复用同一块 buffer（地址稳定 → 可复现，且不每步新建后端）
    galloc: GraphAllocator,
    weights: ModelWeights,
    state: EngineState,
    arch: Qwen35MoeArch,
    sampler: Sampler,
    /// 思考预算状态机（`--reasoning-budget`）；默认「不干预」。
    reasoning: crate::engine::reasoning::ReasoningBudget,
    /// prompt 前缀复用缓存（`--cache-prompt`，P6.3）
    prompt_cache: crate::engine::prompt_cache::PromptCache,
    /// 调用方希望存检查点的 token 位置（稳定历史边界），见 `set_prompt_boundaries`
    prompt_boundaries: Vec<usize>,
    max_context: usize,
    batch_size: usize,
    ubatch_size: usize,
    /// 结束 token（来自 GGUF，可追加）
    eos: Vec<u32>,
    n_vocab: usize,
    /// 上一张建好的图（P2c）：签名一致时整步跳过 reset + 建图 + `gallocr.alloc`。
    ///
    /// 必须在 `scratch` **之前**声明（同一 arena：先失效缓存、再 reset）。
    decode_graph: Option<CachedGraph>,
    /// MTP（投机解码）图专用的分配器与 scratch ctx。
    ///
    /// **必须与主干分开**：`ggml_gallocr_alloc_graph` 对"上一步不是同一张图"的调用会
    /// 重新 reserve，进而可能重排/重建整块 buffer —— 那会把缓存的 decode 图的
    /// `data` 指针改成悬垂的。分开之后两者互不干扰。
    /// 声明在 `ctx` 之前（buffer 要先于 tensor arena 释放）。
    mtp_galloc: Option<GraphAllocator>,
    mtp_scratch: Option<GgmlCtx>,
    /// 投机解码类型
    spec: SpecType,
    /// `--spec-draft-n-max`
    n_spec_max: usize,
    /// 深度自适应策略（成本感知的起草预算）
    spec_policy: SpecPolicy,
    /// 最近一次 `generate` 的投机统计：`(轮数, 起草数, 被接受数)`
    spec_stats: (u64, u64, u64),
    /// MTP 图输入里的 mask 字节缓冲（与主干的那块分开，免得借用打架）
    mtp_mask_buf: Vec<u8>,
    /// MTP 前向的累计墙钟：`(起草步合计 ms, KV 重建合计 ms)`（诊断）
    mtp_ms: (f64, f64),
    /// MTP 前向的分阶段累计：build / alloc / fill / compute / read（诊断）
    mtp_phase: [f64; 5],
    /// decode 每步的分阶段累计 ms：build / alloc / fill / compute / read。
    /// 只在 `SQWEN_PHASE=1` 时累积，**不逐步打印**（逐步 eprintln 会自己把耗时抬高 1~2 ms）。
    step_phase: [f64; 5],
    /// `step_phase` 的样本数
    step_phase_n: u64,
    /// 主机侧线程数（mask 填充用；MTP 图也要）
    threads: usize,
    /// **每步图专用的 scratch ctx**（只存张量元数据）。
    ///
    /// 每步 `reset()` 后重建图：地址完全确定 → ggml-cuda 的 CUDA graph key
    /// （`cgraph->nodes[0]`）与所有节点属性逐字段一致 → warmup 一次后持续重放。
    /// 每步新建 ctx 会让地址漂移，导致永远退回 eager 执行（实测 TG 62 → 82 t/s 的差别）。
    ///
    /// 声明在 `ctx`（持久 ctx）之前无所谓，但必须在 `galloc`/`state` 之后使用。
    scratch: GgmlCtx,
    /// ggml context：张量 arena，声明在最后 → 最后释放
    ctx: GgmlCtx,
}

// 引擎内部使用 Mutex 保护，裸指针不会跨线程误用，所以可以安全地 Send/Sync
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    /// 打开模型并完成分配 + 加载（约 20 GiB 权重）。
    pub fn load(model_path: &str, kind: BackendKind, cfg: EngineConfig) -> Result<Self> {
        let ctx = GgmlCtx::new_no_alloc().map_err(|e| anyhow!("{e}"))?;
        let gguf = GgufFile::open(model_path)?;
        let arch = Qwen35MoeArch::from_gguf(&gguf)?;

        let weights = ModelWeights::create_with_mtp(
            &ctx,
            &gguf,
            &arch,
            cfg.spec_type == SpecType::DraftMtp,
        )?;
        // 状态张量必须在第一次 from_ctx 之前创建，否则拿不到 buffer
        //
        // `max_context == 0` 表示"用模型训练上下文"（与 llama.cpp `n_ctx = 0` 同义）。
        // 注意 KV 显存：本模型 K(q4_0)+V(f32) 约 22.8 KiB/token，
        // 262144 上下文就是 ~5.7 GiB —— 20.2 GiB 权重 + 24 GiB 卡会 OOM。
        let (max_context, ctx_note) = resolve_ctx_size(cfg.max_context, arch.n_ctx_train as usize);

        // ---- FlashAttention 支持探测（P4）----
        // 必须用 ggml_backend_supports_op 门控：不支持时 ggml-cuda 会直接 ABORT。
        // 要在建 state 之前探测，因为它决定 V cache 的类型。
        let probe_ctx = GgmlCtx::new_no_alloc().map_err(|e| anyhow!("{e}"))?;
        let supported = probe_flash_attn(
            &probe_ctx,
            kind,
            &arch,
            crate::engine::kv::KV_TYPE_V_FA,
            max_context.min(4096),
        );
        let flash_attn = match cfg.flash_attn {
            FlashAttnMode::Auto => supported,
            FlashAttnMode::On => {
                if !supported {
                    return Err(anyhow!(
                        "后端不支持 FlashAttention（--flash-attn on 要求支持）；\
                         用 --flash-attn off 走手工注意力"
                    ));
                }
                true
            }
            FlashAttnMode::Off => false,
        };
        let v_type = if flash_attn {
            crate::engine::kv::KV_TYPE_V_FA
        } else {
            crate::engine::kv::KV_TYPE_V_NO_FA
        };

        // 主机侧线程数：0 = 自动。物理核通常比逻辑核少一半，而 mask 填充是访存带宽
        // 敏感型的活，**8 个线程左右就饱和**（实测更多线程反而略慢），所以上限夹到 8。
        let threads = resolve_threads(cfg.threads);
        // 投机解码：需要 `arch.n_rs` 个递推状态快照（= 每轮最多起草的 token 数）
        let spec = if cfg.spec_type == SpecType::DraftMtp && arch.n_layer_nextn > 0 {
            SpecType::DraftMtp
        } else {
            if cfg.spec_type == SpecType::DraftMtp {
                eprintln!(
                    "[engine] 模型没有 MTP / NextN 层（nextn_predict_layers=0），\
                     --spec-type draft-mtp 不生效"
                );
            }
            SpecType::None
        };
        let n_rs = if spec == SpecType::DraftMtp {
            cfg.spec_draft_n_max.max(1)
        } else {
            0
        };
        let n_spec_max = cfg.spec_draft_n_max.max(1);
        // 主干一次前向最多 n_tokens：prefill 分块（ubatch）或校验批（n_max + 1）
        let ubatch_capped = cfg
            .ubatch_size
            .max(1)
            .min(cfg.batch_size.max(1))
            .min(max_context as i64) as usize;
        let mtp_cap = ubatch_capped.max(n_spec_max + 1);

        let state = EngineState::new(
            &ctx,
            &arch,
            max_context,
            v_type,
            flash_attn,
            threads,
            n_rs,
            mtp_cap,
        )
        .map_err(|e| anyhow!("{e}"))?;
        let kv_bytes = state.kv.bytes_total();
        let kv_per_tok = state.kv.bytes_per_token();
        if ctx_note != CtxNote::Ok || std::env::var_os("SQWEN_QUIET").is_none() {
            eprintln!(
                "[engine] ctx={} (模型训练上下文 {}){} | batch={} ubatch={} | flash-attn={} (V={}) | KV cache ≈ {:.1} MiB ({:.1} KiB/token × {} 全注意力层)",
                max_context,
                arch.n_ctx_train,
                ctx_note.text(),
                cfg.batch_size.max(1),
                ubatch_capped,
                if flash_attn { "on" } else { "off" },
                if flash_attn { "q4_0" } else { "f32" },
                kv_bytes as f64 / (1024.0 * 1024.0),
                kv_per_tok as f64 / 1024.0,
                state.kv.n_layers(),
            );
            if spec == SpecType::DraftMtp {
                eprintln!(
                    "[engine] 投机解码: draft-mtp（n_max={}）| MTP 权重 +{:.1} MiB | \
                     MTP KV ≈ {:.1} MiB | 递推回退快照 ×{}",
                    n_spec_max,
                    weights
                        .mtp_weight_bytes(&ctx)
                        .map(|b| b as f64 / (1024.0 * 1024.0))
                        .unwrap_or(0.0),
                    (state.kv.bytes_total() / state.kv.n_layers().max(1)) as f64
                        / (1024.0 * 1024.0),
                    n_rs + 1,
                );
            }
        }
        let persistent =
            BackendBuffer::from_ctx(&ctx, kind).map_err(|e| anyhow!("分配后端 buffer 失败: {e}"))?;
        weights.load_data(&ctx, &gguf)?;
        // backend buffer 不清零，递推状态必须显式清零
        state.zero_states();
        // KV cache 也要清零（一次性）：n_kv 对齐到 256 后，[n_used, n_kv_padded)
        // 这几行会被注意力读到，未初始化显存里的 NaN/Inf 会污染整行 softmax。
        // 之后不用再清：写入只落在已初始化的前缀里。
        state.kv.zero_caches();
        // MTP 的 KV 同理（它现在只在投机解码时存在）
        if let Some(mtp) = state.mtp.as_ref() {
            mtp.kv.zero_caches();
        }

        // EOS：优先 `eos_token_id`，兼容数组形式 `eos_token_ids`
        let mut eos = Vec::new();
        if let Some(id) = gguf.kv_get_u32("tokenizer.ggml.eos_token_id") {
            eos.push(id);
        }
        let mut arr = [0i32; 16];
        if let Some(n) = gguf.kv_get_i32_array("tokenizer.ggml.eos_token_ids", &mut arr) {
            for &v in arr.iter().take(n) {
                if v >= 0 && !eos.contains(&(v as u32)) {
                    eos.push(v as u32);
                }
            }
        }

        let n_vocab = weights.output.get(&ctx).ne()[1] as usize;
        let galloc = GraphAllocator::new(kind).map_err(|e| anyhow!("{e}"))?;
        let scratch = GgmlCtx::new_no_alloc().map_err(|e| anyhow!("{e}"))?;
        let (mtp_galloc, mtp_scratch) = if spec == SpecType::DraftMtp {
            (
                Some(GraphAllocator::new(kind).map_err(|e| anyhow!("{e}"))?),
                Some(GgmlCtx::new_no_alloc().map_err(|e| anyhow!("{e}"))?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            persistent,
            galloc,
            weights,
            state,
            arch,
            sampler: cfg.sampler,
            reasoning: crate::engine::reasoning::ReasoningBudget::new(Default::default()),
            prompt_cache: crate::engine::prompt_cache::PromptCache::new(
                cfg.cache_prompt,
                crate::engine::prompt_cache::DEFAULT_CACHE_BYTES,
            ),
            prompt_boundaries: Vec::new(),
            max_context,
            batch_size: cfg.batch_size.max(1) as usize,
            // llama.cpp 的 clamp：n_ubatch = min(n_ubatch, n_batch)；再夹一层 ubatch <= ctx
            // （一次前向不可能超过 KV 容量）
            ubatch_size: ubatch_capped,
            eos,
            n_vocab,
            decode_graph: None,
            mtp_galloc,
            mtp_scratch,
            spec,
            n_spec_max,
            spec_policy: SpecPolicy::new(n_spec_max, cfg.spec_adaptive),
            spec_stats: (0, 0, 0),
            mtp_mask_buf: Vec::new(),
            mtp_ms: (0.0, 0.0),
            mtp_phase: [0.0; 5],
            step_phase: [0.0; 5],
            step_phase_n: 0,
            threads,
            scratch,
            ctx,
        })
    }

    /// 是否启用了 MTP 投机解码。
    pub fn spec_enabled(&self) -> bool {
        self.spec == SpecType::DraftMtp && self.state.mtp.is_some()
    }

    /// 投机解码统计（诊断）：`(轮数, 起草 token 数, 被接受数)`。
    pub fn spec_stats(&self) -> (u64, u64, u64) {
        self.spec_stats
    }

    /// ggml context（张量 arena）。Engine 持有它，权重与常驻状态都活在其中。
    pub fn ctx(&self) -> &GgmlCtx {
        &self.ctx
    }

    /// 常驻引擎状态（诊断用）。
    pub fn state(&self) -> &EngineState {
        &self.state
    }

    /// 每步图的分配器（复用 buffer；诊断用）。
    pub fn allocator(&self) -> &GraphAllocator {
        &self.galloc
    }

    /// 实际生效的逻辑 batch 上限（`-b`，已按 ctx 约束）。
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// 实际生效的物理 batch / prefill 分块大小（`-ub`，已按 batch 与 ctx 约束）。
    pub fn ubatch_size(&self) -> usize {
        self.ubatch_size
    }

    /// 上下文上限（KV 容量）。
    pub fn max_context(&self) -> usize {
        self.max_context
    }

    /// 权重 + 常驻状态所在的后端 buffer。
    pub fn buffer(&self) -> &BackendBuffer {
        &self.persistent
    }

    pub fn arch(&self) -> &Qwen35MoeArch {
        &self.arch
    }

    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    pub fn eos_tokens(&self) -> &[u32] {
        &self.eos
    }

    /// 追加一个结束 token（chat 模型常有多套 end token，由上层从模板/词表补充）。
    pub fn add_eos(&mut self, id: u32) {
        if !self.eos.contains(&id) {
            self.eos.push(id);
        }
    }

    /// 直接替换结束 token 集合。bench 用它临时清空 EOS（与 llama-bench 一致：
    /// 忽略 EOS、固定生成 N 个 token，否则吞吐数字会被提前结束污染）。
    pub fn set_eos(&mut self, eos: Vec<u32>) {
        self.eos = eos;
    }

    /// 采样器（改温度/top-k/top-p）。
    pub fn sampler_mut(&mut self) -> &mut Sampler {
        &mut self.sampler
    }

    /// 设置思考预算（`--reasoning-budget`，P6.2）。
    ///
    /// 引擎侧只做"超预算就强制输出思考结束标签"这一件事 —— 是否开启思考、标签的 token
    /// id 都由调用方（CLI / server）从模板与 tokenizer 推出来。
    pub fn set_reasoning(&mut self, cfg: crate::engine::reasoning::ReasoningConfig) {
        self.reasoning = crate::engine::reasoning::ReasoningBudget::new(cfg);
    }

    /// 开关 prompt 前缀复用（`--cache-prompt`，P6.3）。默认开（同 llama.cpp）。
    pub fn set_cache_prompt(&mut self, enabled: bool) {
        self.prompt_cache.set_enabled(enabled);
    }

    /// 启动预热：把**一次性成本**从第一个真实请求挪到加载阶段。
    ///
    /// 对齐 llama.cpp 的 `warmup`（`common/common.cpp:1464`，默认值见
    /// `common/common.h:580` 的 `bool warmup = true;  // warmup run`）—— 我们一直没有。
    ///
    /// # 为什么需要（实测）
    ///
    /// 第一次前向的 `compute` 是 **351 ms**（34 token 的 prefill），而同一形状在稳态
    /// 只要 ~11 ms。这多出来的 ~340 ms 是**固定的**：n=512 时是 581 vs 158 ms
    /// （多出 ~420 ms），基本不随 n 变 —— 来自 CUDA 模块的惰性加载、gallocr 首次
    /// reserve、以及后端上下文的首次初始化。它**每个进程都要付一次**，表现就是
    /// 首字延迟（TTFT）比稳态高约 200 ms（实测冷 252 ms vs 热 43 ms）。
    ///
    /// # 覆盖范围
    ///
    /// 1. **一次 prefill**（`n = min(ubatch, 64)`）—— 覆盖 prefill 专用内核
    ///    （`mul_mat_q`/mmq 这一路，只有在 token 数够多时才会被选中，所以不能只喂 1 个）；
    /// 2. **两次 decode**（`n = 1`）—— 覆盖 decode 专用内核（`mul_mat_vec_q`/mmvq），
    ///    并让 ggml-cuda 完成 decode 图的 CUDA graph capture（它要求连续两次同 key
    ///    且节点属性不变才会 capture，见 `ggml_backend_cuda_graph_compute` 的 warmup 状态机）；
    /// 3. 开了投机时再补两次 MTP 起草 —— MTP 图的 CUDA graph key 与主干不同，要单独预热。
    ///
    /// # 复位
    ///
    /// 跑完必须回到"刚加载完"的状态：GDN 递推状态、卷积历史、KV 游标、MTP 的 `h_buf`
    /// 全部清掉，否则第一个真实请求会接着 dummy 序列往下写。采样器也重启
    /// （同 llama.cpp：`res->reset_samplers()`），保证 RNG 与 seed 对齐。
    pub fn warmup(&mut self) -> Result<()> {
        /// 预热 prefill 的 token 数。够大才会走到 prefill 专用内核；不必等于 ubatch ——
        /// 一次性成本基本与 n 无关，喂大了只是白等。
        const WARMUP_PREFILL: usize = 64;

        let n = self.ubatch_size.clamp(1, WARMUP_PREFILL);
        let ids: Vec<i32> = vec![0; n];
        let last = (n - 1) as i32;

        let reset = |s: &mut crate::model::graph::EngineState| {
            s.reset_caches();
            s.zero_states();
        };

        reset(&mut self.state);

        // 1) prefill：只要最后一行的 logits（够触发全部 kernel 就行）
        self.step(&ids, 0, Some(&[last]))
            .map_err(|e| anyhow!("warmup prefill: {e}"))?;

        // 2) decode ×2：第 2 次让 ggml-cuda 完成 capture
        reset(&mut self.state);
        for _ in 0..2 {
            self.state.rewind_to(0).map_err(|e| anyhow!("{e}"))?;
            self.step(&[0], 0, Some(&[0]))
                .map_err(|e| anyhow!("warmup decode: {e}"))?;
        }

        // 3) 投机：MTP 的图是另一套 key，单独预热（顺带把计时清零，别把预热算进统计）
        if self.spec_enabled() {
            for _ in 0..2 {
                self.mtp_forward(MtpMode::Draft, &[0], 0, 0)
                    .map_err(|e| anyhow!("warmup MTP: {e}"))?;
            }
            self.mtp_ms = (0.0, 0.0);
            self.mtp_phase = [0.0; 5];
        }

        // 4) 复位到"刚加载完"
        reset(&mut self.state);
        self.sampler.restart();
        Ok(())
    }

    /// 指定"希望在这里存一个状态检查点"的 token 位置（升序，可空）。
    ///
    /// 用途：调用方知道自己的 prompt 里哪一段是**稳定的历史**（例如 server 渲染聊天
    /// 历史的部分），把历史末尾的位置报上来，引擎就会正好在那儿存一份 GDN 状态快照。
    /// 下一个请求的历史只要没变，就能从这一点直接接上 —— 多轮对话的典型情况省掉
    /// 整段历史的重算。除这些位置外，引擎还会按 `CKPT_INTERVAL` 定期存（见 `generate`）。
    pub fn set_prompt_boundaries(&mut self, positions: Vec<usize>) {
        self.prompt_boundaries = positions;
    }

    /// 缓存统计（诊断用）：`(命中次数, 未命中次数, 复用掉的 token 数)`。
    pub fn cache_stats(&self) -> (u64, u64, u64) {
        (
            self.prompt_cache.hits,
            self.prompt_cache.misses,
            self.prompt_cache.reused_tokens,
        )
    }

    /// 生成 `max_tokens` 个 token（可能因 EOS 或上下文上限提前停止）。
    ///
    /// 每个 token 采出来就立刻通过 `on_token` 回调（流式）。
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        max_tokens: usize,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<(Vec<u32>, Timing)> {
        check_budget(prompt_ids.len(), max_tokens, self.max_context)?;
        if let Some(&bad) = prompt_ids.iter().find(|&&t| t as usize >= self.n_vocab) {
            return Err(anyhow!("prompt 里有越界的 token id {bad}"));
        }

        // ---- 前缀复用（--cache-prompt，P6.3）----
        // 命中就恢复检查点的 GDN 状态 + 把 KV 游标回退到检查点位置，只 prefill 后缀；
        // 未命中才做整段 reset（注意：**命中时绝对不能 zero_states**，那会把恢复的
        // 递推状态清掉）。
        let plan = self.prompt_cache.plan(prompt_ids, MIN_KEEP_FRAC);
        let start_pos = match &plan {
            Some(p) => {
                self.state
                    .restore_state(self.prompt_cache.checkpoint_data(p.checkpoint_idx))
                    .map_err(|e| anyhow!("恢复检查点失败: {e}"))?;
                self.state
                    .rewind_to(p.start_pos)
                    .map_err(|e| anyhow!("KV 回退失败: {e}"))?;
                // 前缀一旦分叉，位置 > lcp 的检查点就失效了（KV 行会被覆盖）
                let lcp = self.prompt_cache.lcp(prompt_ids);
                self.prompt_cache.truncate_to(lcp);
                self.prompt_cache.note_hit(p.start_pos);
                p.start_pos
            }
            None => {
                self.prompt_cache.note_miss();
                self.state.reset_caches();
                self.state.zero_states();
                0
            }
        };
        // 采样器始终重置（新序列；不重置的话随机流会接上一次）
        self.sampler.restart();
        // 递推状态的平面选择在序列开始时必须是 0（它由上一轮的"部分接受"设定）
        self.state.set_recurrent_sel(0);

        let t0 = Instant::now();

        // ---- prefill（按 ubatch 分块，且在检查点位置切一刀）----
        // 每个 chunk 只要**最后一个位置**的 logits（生成第一个 token 用），
        // 由 `out_ids` 在最后一层裁行 —— 否则要把 [n_vocab, chunk] 整块拷回主机
        // （512 token 就是 508 MB，实测占 prefill 的 317 ms）。
        let ckpt_at = self.checkpoint_positions(start_pos, prompt_ids.len());
        let mut pos = start_pos;
        let mut vals = Vec::new();
        for (from, to) in plan_chunks(start_pos, prompt_ids.len(), self.ubatch_size, &ckpt_at) {
            let chunk = &prompt_ids[from..to];
            let ids: Vec<i32> = chunk.iter().map(|&t| t as i32).collect();
            let n = ids.len();
            vals = self.step(&ids, pos, Some(&[(n - 1) as i32]))?;
            // MTP：本块的 KV 重建（teacher forcing），并把 `h_buf[0]` 推到本块最后一个
            // 已提交位置的主干 h。**每个 prefill 块都要跑**，否则 draft 阶段读到的
            // MTP KV 会有空洞。
            if self.spec_enabled() {
                self.mtp_forward(MtpMode::Catchup, &ids, pos, n - 1)?;
            }
            pos += n;
            // 正好落在检查点位置上 → 存一份状态快照（下一个请求可能从这里接）
            if ckpt_at.contains(&pos) && pos < prompt_ids.len() {
                self.prompt_cache.push_checkpoint(pos, self.state.snapshot_state());
            }
        }
        let row = vals;
        assert_eq!(
            row.len(),
            self.n_vocab,
            "out_ids 裁行后应当只剩需要的那一行"
        );
        let ttft0 = Instant::now();
        let prompt_ms = (ttft0 - t0).as_secs_f64() * 1000.0;
        let prompt_cached = start_pos as u32;

        // max_tokens == 0：只做 prefill（bench 的 prompt-only 模式），不产出 token
        if max_tokens == 0 {
            return Ok((
                Vec::new(),
                Timing {
                    ttft_ms: prompt_ms,
                    prompt_tokens: prompt_ids.len() as u32,
                    prompt_cached,
                    prompt_ms,
                    gen_tokens: 0,
                    gen_ms: 0.0,
                    gen_tokens_steady: 0,
                    gen_ms_steady: 0.0,
                },
            ));
        }

        // ---- decode ----
        let profiling = std::env::var_os("SQWEN_PROFILE").is_some();
        let mut t_step = 0f64;
        let mut t_sample = 0f64;
        // MTP 的计时器是**引擎级累加字段**，必须每个请求归零。
        //
        // 不归零的后果（`-r N` 的 bench 上非常明显，且很难看出来）：
        //   - `MTP 起草 ms/步 = self.mtp_ms.0 / n_drafted`，分子是跨请求累积的、
        //     分母是本请求的 → 第 k 次生成会放大 k 倍；
        //   - `主干校验 = (t_step - mtp_ms) / 轮数` 里 `t_step` 是本请求的，
        //     减掉累积值之后会直接变成**负数**。
        self.mtp_ms = (0.0, 0.0);
        self.mtp_phase = [0.0; 5];
        let mut out: Vec<u32> = Vec::new();
        // 稳态窗口的起点：第一次跨过 `STEADY_SKIP_MS` 时记下 `(已耗时 ms, 已产出 token 数)`。
        // 末尾用它算出**跳过冷启动段**的速率 —— 累计速率会被开头几十秒的 GPU 时钟爬升
        // （空闲 P8/270MHz → 负载 P2/1900MHz）平均进去，系统性偏低约 6%。
        // llama.cpp 的 server 报 `tg`（累计）与 `tg_3s`（窗口）两个数，我们也要有可比的稳态口径。
        let mut steady_mark: Option<(f64, usize)> = None;
        self.reasoning.restart();
        // 给正文留余地：预算是启动时定的、`max_tokens` 是每请求的，撞上就会
        // 「思考吃光预算 → content 为空」。固定留 256 token（一个短答案的长度）。
        self.reasoning.ensure_answer_room(max_tokens, ANSWER_TOKEN_RESERVE);
        let ts = Instant::now();
        // 思考预算（`--reasoning-budget`）可能在起飞时就要强制结束思考
        let mut next = match self.reasoning.next_forced() {
            Some(t) => t,
            None => self.sampler.sample(&row),
        };
        t_sample += ts.elapsed().as_secs_f64() * 1000.0;
        // 把采出的 token 喂回采样器（重复/presence/frequency 惩罚需要"已生成历史"）。
        // 与 llama.cpp 一致：只喂生成出来的 token，不喂 prompt。
        self.sampler.accept(next);
        self.reasoning.accept(next);

        // ---- 投机解码的轮次统计（诊断）----
        let mut n_rounds = 0u64;
        let mut n_drafted = 0u64;
        let mut n_accepted = 0u64;
        // 每个草稿位置各自"被到达时的命中数 / 被到达次数"（对齐 llama.cpp 的 acc rate/pos）
        let mut pos_hit = [0u64; 16];
        let mut pos_tot = [0u64; 16];
        // 新请求：重置「锁定不投机」状态（统计跨请求保留）
        self.spec_policy.begin_request();

        // `SQWEN_NO_EOS=1`：**只给消融/压测量测用** —— 不许被 EOS 提前终止。
        // 消融后 logits 是垃圾，第一个 token 常常就是 EOS，于是 decode 一步都跑不到。
        let ignore_eos = std::env::var_os("SQWEN_NO_EOS").is_some();
        // `SQWEN_NO_LOOP_GUARD=1`：关掉重复死循环兜底（压测/复现问题时用）
        let ignore_loop_guard = std::env::var_os("SQWEN_NO_LOOP_GUARD").is_some();

        // 不变式（循环顶部）：`next` 已经输出，但它**还没有**被提交 —— 它将被放到
        // 位置 `pos`。因此"已提交的 token 数"总是 `out.len() - 1`（最后输出的那个
        // 还没喂进去），这正是 prompt cache 记录 `processed` 的依据。
        loop {
            on_token(next);
            out.push(next);
            // 稳态窗口起点：第一次跨过 STEADY_SKIP_MS。粒度是"一轮"（无投机时=1 token，
            // 投机时=平均 2.6 token），对窗口速率的影响可以忽略。
            if steady_mark.is_none() {
                let el = (Instant::now() - ttft0).as_secs_f64() * 1000.0;
                if el >= STEADY_SKIP_MS {
                    steady_mark = Some((el, out.len()));
                }
            }
            // 重复死循环兜底：模型（尤其 temp 很低时）会整段重复同一段推理，
            // 自己停不下来。每 16 个 token 检查一次「最近 96 个 token 连续重复 3 次」。
            if !ignore_loop_guard
                && out.len() % LOOP_CHECK_EVERY == 0
                && is_degenerate_loop(&out)
            {
                eprintln!(
                    "[engine] 检测到重复死循环（最后 {LOOP_WINDOW} 个 token 连续重复 {LOOP_REPEATS} 次），\
                     已提前停止；共 {} token。加大 --repeat-penalty 或提高 --temp 可缓解",
                    out.len()
                );
                break;
            }
            if (!ignore_eos && self.eos.contains(&next)) || out.len() >= max_tokens || pos >= self.max_context {
                break;
            }

            // ---- 本轮的起草预算 --------------------------------------------
            // 每轮产出 `m + 1` 个 token（m 个被接受的草稿 + 1 个纠正），所以 k 要留
            // 出至少 1 个位置；上下文同理（校验批占 `[pos, pos+k]`）。
            //
            // 深度由 `SpecPolicy` 在线决定：目标函数是「每毫秒产出多少 token」，
            // 不是「接受长度」—— 固定深度在低接受率内容上会亏（见 `SpecPolicy` 注释）。
            let room = max_tokens.saturating_sub(out.len());
            let ctx_room = self.max_context.saturating_sub(pos);
            let spec_ok = self.spec_enabled() && !self.reasoning.is_forcing();
            let k = if spec_ok {
                self.spec_policy
                    .pick()
                    .min(self.n_spec_max)
                    .min(room.saturating_sub(1))
                    .min(ctx_room.saturating_sub(1))
            } else {
                0
            };
            let t_round = Instant::now();

            // ---- 起草：链式，每一步只喂 1 行（草稿 token 依赖上一步的 argmax，
            //      不可能合成一个批）-----------------------------------------
            let mut drafts: Vec<u32> = Vec::with_capacity(k);
            for i in 0..k {
                let tok = if i == 0 { next } else { drafts[i - 1] };
                let ts = Instant::now();
                let lg = self
                    .mtp_forward(MtpMode::Draft, &[tok as i32], pos + i, 0)?
                    .ok_or_else(|| anyhow!("MTP Draft 没有返回 logits"))?;
                t_step += ts.elapsed().as_secs_f64() * 1000.0;
                drafts.push(argmax(&lg));
            }

            // ---- 校验批：`[next, d0..d_{k-1}]`，所有行都要 logits -------------
            let mut ids: Vec<i32> = Vec::with_capacity(k + 1);
            ids.push(next as i32);
            ids.extend(drafts.iter().map(|&d| d as i32));
            let n_batch = ids.len();
            let out_rows: Vec<i32> = (0..n_batch as i32).collect();
            let ts = Instant::now();
            let rows = self.step(&ids, pos, Some(&out_rows))?;
            t_step += ts.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(rows.len(), self.n_vocab * n_batch);

            // ---- 逐行用**用户采样器**采一次，按最长匹配前缀接受 -----------------
            // 每个产出 token 恰好一次目标采样（顺序、历史都与不投机时完全一致），
            // 所以随机流与非投机路径对齐。
            let ts = Instant::now();
            let mut m = 0usize; // 被接受的草稿数
            let mut produced: Vec<u32> = Vec::with_capacity(k + 1);
            for i in 0..=k {
                let row_i = &rows[i * self.n_vocab..(i + 1) * self.n_vocab];
                let s = self.sampler.sample(row_i);
                self.sampler.accept(s);
                self.reasoning.accept(s);
                produced.push(s);
                pos_tot[i] += 1;
                if i < k && s == drafts[i] {
                    m += 1;
                    pos_hit[i] += 1;
                    continue;
                }
                break;
            }
            t_sample += ts.elapsed().as_secs_f64() * 1000.0;
            n_rounds += 1;
            n_drafted += k as u64;
            n_accepted += m as u64;

            // ---- 输出被接受的草稿（中途遇到 EOS / 达到上限就停）-----------------
            let mut stopped = false;
            for &t in produced.iter().take(m) {
                on_token(t);
                out.push(t);
                if (!ignore_eos && self.eos.contains(&t)) || out.len() >= max_tokens {
                    stopped = true;
                    break;
                }
            }

            // ---- 提交：状态退回"接受前缀之后" ---------------------------------
            // `committed` = 状态推进到的位置数。
            //   - 正常轮次：`next` + m 个草稿都被吃掉 → `pos + m + 1`；纠正 token
            //     （`produced[m]`）留到下一轮顶部输出，所以"最后输出的那个没提交"
            //     的不变式仍然成立。
            //   - 提前停止（EOS / 达到上限）：本轮最后输出的那个 token 不提交，
            //     于是提交量正好等于本轮输出的个数，即 `prompt_len + out.len() - 1`。
            let committed = if stopped {
                prompt_ids.len() + out.len() - 1
            } else {
                pos + m + 1
            };
            let consumed = committed - pos;
            self.state
                .rewind_to(committed)
                .map_err(|e| anyhow!("状态回退失败: {e}"))?;
            // 递推状态的读取平面：校验批有 `n_batch` 个 token，只提交了 `consumed` 个，
            // 所以要往回 `n_batch - consumed` 个快照。**这个选择留到下一轮 `step()`
            // 时才被消费**（本轮的 step 已经用掉了上一次设的值），所以这里不能清零。
            self.state.set_recurrent_sel(n_batch - consumed);

            // ---- MTP KV 的 teacher forcing 重建 + `h_buf[0]` 刷新 --------------
            // 只在「本轮起草过」或「还在探测不投机基线」时跑：一旦锁定不投机，
            // 之后再也不用 MTP，省掉这整段开销。
            let mut catchup_ms = 0.0;
            if self.spec_enabled() && (k > 0 || !self.spec_policy.is_plain()) {
                let tc = Instant::now();
                self.mtp_forward(MtpMode::Catchup, &ids, pos, consumed - 1)?;
                catchup_ms = tc.elapsed().as_secs_f64() * 1000.0;
            }
            // 本轮成本（剔除探测期仅为保持 MTP 同步的重建时间，否则会高估不投机的成本）
            if spec_ok {
                let round_ms = t_round.elapsed().as_secs_f64() * 1000.0 - catchup_ms;
                self.spec_policy.on_round(k, m, round_ms);
            }
            pos = committed;

            if stopped {
                break;
            }
            // 纠正 token 成为下一轮的 `next`（在循环顶部输出）
            next = produced[m];
        }
        if profiling && !out.is_empty() {
            let n = out.len() as f64;
            eprintln!(
                "[profile] decode: {} tok, step={:.2} ms/tok, sample={:.2} ms/tok",
                out.len(),
                t_step / n,
                t_sample / n,
            );
            if self.spec_enabled() && n_rounds > 0 {
                eprintln!(
                    "[profile] spec: {} 轮, 起草 {} 个, 接受 {} 个（{:.2} 个/轮, 平均接受长度 {:.2}）",
                    n_rounds,
                    n_drafted,
                    n_accepted,
                    n_accepted as f64 / n_rounds as f64,
                    1.0 + n_accepted as f64 / n_rounds as f64,
                );
                let rates: Vec<String> = (0..self.n_spec_max.min(16))
                    .map(|i| {
                        format!(
                            "{:.3}",
                            pos_hit[i] as f64 / (pos_tot[i].max(1) as f64)
                        )
                    })
                    .collect();
                eprintln!("[profile] spec 各位置命中率: {}", rates.join(", "));
                eprintln!("[profile] spec 成本模型: {}", self.spec_policy.report());
                eprintln!(
                    "[profile] spec 耗时: MTP 起草 {:.2} ms/步 × {}, MTP 重建 {:.2} ms/次 × {}, 主干校验 {:.2} ms/轮",
                    self.mtp_ms.0 / n_drafted.max(1) as f64,
                    n_drafted,
                    self.mtp_ms.1 / n_rounds.max(1) as f64,
                    n_rounds,
                    (t_step - self.mtp_ms.0 - self.mtp_ms.1) / n_rounds.max(1) as f64,
                );
                let calls = (n_drafted + n_rounds).max(1) as f64;
                eprintln!(
                    "[profile] MTP 分阶段 ms/次: build={:.2} alloc={:.2} fill={:.2} compute={:.2} read={:.2}",
                    self.mtp_phase[0] / calls,
                    self.mtp_phase[1] / calls,
                    self.mtp_phase[2] / calls,
                    self.mtp_phase[3] / calls,
                    self.mtp_phase[4] / calls,
                );
            }
        }
        self.spec_stats = (n_rounds, n_drafted, n_accepted);

        // `SQWEN_PHASE=1`：本轮 decode 的平均分阶段耗时（不逐步打印，避免被测对象被 IO 抬高）
        if phase_on() && self.step_phase_n > 0 {
            let n = self.step_phase_n as f64;
            eprintln!(
                "[phase] decode 每步 {:.2} ms = build {:.2} + alloc {:.2} + fill {:.2} + compute {:.2} + read {:.2}（{} 步）",
                self.step_phase.iter().sum::<f64>() / n,
                self.step_phase[0] / n,
                self.step_phase[1] / n,
                self.step_phase[2] / n,
                self.step_phase[3] / n,
                self.step_phase[4] / n,
                self.step_phase_n,
            );
            if self.spec_enabled() {
                eprintln!("[phase] spec 探测: {}", self.spec_policy.report());
            }
        }
        self.step_phase = [0.0; 5];
        self.step_phase_n = 0;

        let gen_ms = (Instant::now() - ttft0).as_secs_f64() * 1000.0;
        // 稳态窗口：跳过冷启动段之后的 token 数与耗时（见 `STEADY_SKIP_MS` 的注释）
        let (gen_tokens_steady, gen_ms_steady) = match steady_mark {
            Some((t_mark, n_mark)) if out.len() > n_mark + MIN_STEADY_TOKENS => {
                ((out.len() - n_mark) as u32, gen_ms - t_mark)
            }
            _ => (0, 0.0),
        };
        // 记下本序列**已处理**的 token 列表供下一次请求求 lcp：prompt + 生成的 token，
        // 但**不含最后一个**（它被采出来了却没喂进模型 —— `out` 的末尾是 EOS 或截断点，
        // 状态只推进到 `out.len()-1`）。含进去会让 lcp 指到一个我们没有状态的边界。
        //
        // 上面的提交逻辑保证了这一点对投机解码同样成立（见循环里 `committed` 的推导）。
        let mut processed: Vec<u32> = Vec::with_capacity(prompt_ids.len() + out.len());
        processed.extend_from_slice(prompt_ids);
        if !out.is_empty() {
            processed.extend_from_slice(&out[..out.len() - 1]);
        }
        debug_assert_eq!(processed.len(), pos, "状态推进位置必须与 processed 一致");
        // 序列末尾也存一个检查点（"延续式"请求会直接命中它）
        self.prompt_cache
            .push_checkpoint(processed.len(), self.state.snapshot_state());
        self.prompt_cache.finish(processed);

        let timing = Timing {
            ttft_ms: prompt_ms,
            prompt_tokens: prompt_ids.len() as u32,
            prompt_cached,
            prompt_ms,
            gen_tokens: out.len() as u32,
            gen_ms,
            gen_tokens_steady,
            gen_ms_steady,
        };
        Ok((out, timing))
    }

    /// 跑一次 MTP 头的图（每步重建，不做跨步复用）。
    ///
    /// - `MtpMode::Catchup`：把本批（`ids.len()` 个 token、位置起点 `start`）的
    ///   MTP KV 按 teacher forcing 重建（每条 KV 是 `(x_p, h_{p-1})`），然后
    ///   `h_buf[0] ← h_buf[h_sel + 1]`（本批里最后一个**已提交**位置的主干 h）。
    ///   返回 `None`（不算 logits —— 省掉一次 ~300 MB 的 lm_head 读）。
    /// - `MtpMode::Draft`：单行草稿（`ids.len() == 1`，位置 `start`），返回
    ///   `[n_vocab]` 的 logits；同时把 `h_nextn` 写回 `h_buf[0]` 供链式下一步用。
    ///
    /// 用的是**独立的** `mtp_galloc` / `mtp_scratch`：MTP 图每步重建，若不分开，
    /// `ggml_gallocr_alloc_graph` 会在"上一步不是同一张图"时重新 reserve，从而可能
    /// 重排主干的缓冲区、把缓存的 decode 图的 `data` 指针变成悬垂。
    fn mtp_forward(
        &mut self,
        mode: MtpMode,
        ids: &[i32],
        start: usize,
        h_sel: usize,
    ) -> Result<Option<Vec<f32>>> {
        let t_mtp0 = Instant::now();
        let n = ids.len();
        assert!(n > 0, "MTP 批不能为空");
        let n_i64 = n as i64;

        // KV 游标退到本批起点：起草期写下的 MTP KV 只在本轮内有效，
        // 每批都从起点重写（teacher forcing 会覆盖成正确值）。
        {
            let mtp = self.state.mtp.as_mut().ok_or_else(|| anyhow!("MTP 未启用"))?;
            mtp.kv.rewind_to(start).map_err(|e| anyhow!("MTP {e}"))?;
            mtp.kv.begin_step(n).map_err(|e| anyhow!("MTP {e}"))?;
        }
        let n_kv = self.state.mtp.as_ref().expect("刚判过 Some").kv.n_kv_padded();
        let flash_attn = self.state.flash_attn();
        let rope = *self.state.rope();

        let scratch = self.mtp_scratch.as_ref().ok_or_else(|| anyhow!("MTP scratch 缺失"))?;
        scratch.reset();
        let mut tokens = scratch.new_tensor_1d_i32(n_i64);
        tokens.set_name("mtp_tokens");
        let mut pos_t = scratch.new_tensor_1d_i32(4 * n_i64);
        pos_t.set_name("mtp_pos");
        let mask_type = if flash_attn {
            ffi::GgmlType::F16
        } else {
            ffi::GgmlType::F32
        };
        let mut mask = scratch.new_tensor_2d(mask_type, n_kv as i64, n_i64);
        mask.set_name("mtp_mask");
        let mut kv_idxs = scratch.new_tensor(ffi::GgmlType::I64, &[n_i64]);
        kv_idxs.set_name("mtp_kv_idxs");

        let mut g = GraphBuilder::new_with_size(scratch, MTP_NODE_BUDGET);
        let out = {
            let mtp_w = self.weights.mtp.as_ref().ok_or_else(|| anyhow!("MTP 权重未加载"))?;
            let st = self.state.mtp.as_mut().expect("刚判过 Some");
            crate::model::mtp::build_mtp(
                scratch,
                &mut g,
                mtp_w,
                &self.weights,
                &self.arch,
                st,
                mode,
                &tokens,
                &pos_t,
                &mask,
                &kv_idxs,
                n_i64,
                &rope,
                flash_attn,
                h_sel,
            )
            .map_err(|e| anyhow!("{e}"))?
        };

        let galloc = self.mtp_galloc.as_ref().ok_or_else(|| anyhow!("MTP gallocr 缺失"))?;
        let tp0 = Instant::now();
        self.mtp_phase[0] += tp0.duration_since(t_mtp0).as_secs_f64() * 1000.0; // build
        galloc.alloc(&g).map_err(|e| anyhow!("MTP gallocr 分配失败: {e}"))?;
        let tp1 = Instant::now();
        self.mtp_phase[1] += tp1.duration_since(tp0).as_secs_f64() * 1000.0; // alloc

        crate::model::graph::write_step_inputs(
            &mut self.mtp_mask_buf,
            &StateTensor::new(tokens),
            &StateTensor::new(pos_t),
            &StateTensor::new(mask),
            &StateTensor::new(kv_idxs),
            ids,
            start,
            n_kv,
            flash_attn,
            self.threads,
        );
        let tp2 = Instant::now();
        self.mtp_phase[2] += tp2.duration_since(tp1).as_secs_f64() * 1000.0; // fill
        galloc.compute(&g).map_err(|e| anyhow!("MTP compute: {e}"))?;
        let tp3 = Instant::now();
        self.mtp_phase[3] += tp3.duration_since(tp2).as_secs_f64() * 1000.0; // compute

        let ms = tp3.duration_since(t_mtp0).as_secs_f64() * 1000.0;
        match mode {
            MtpMode::Draft => self.mtp_ms.0 += ms,
            MtpMode::Catchup => self.mtp_ms.1 += ms,
        }

        Ok(out.logits.map(|lg| {
            let vals = lg.read_f32_at(0, lg.nbytes() / 4);
            self.mtp_phase[4] += Instant::now().duration_since(tp3).as_secs_f64() * 1000.0; // read
            vals
        }))
    }

    /// 本步 prefill 要在哪些位置存检查点（升序、去重、都 < `prompt_len`）。
    ///
    /// 两类：调用方报的稳定边界（例如聊天历史的末尾）+ 每 `CKPT_INTERVAL` 一个。
    /// 后者保证"分歧点落在 prompt 中间"时也能就近恢复（最多多算 `CKPT_INTERVAL`
    /// 个 token），代价是被 `PromptCache` 的字节预算淘汰最旧的那些。
    fn checkpoint_positions(&self, start_pos: usize, prompt_len: usize) -> Vec<usize> {
        if !self.prompt_cache.enabled() {
            return Vec::new();
        }
        let mut v: Vec<usize> = self
            .prompt_boundaries
            .iter()
            .copied()
            .filter(|&p| p > start_pos && p < prompt_len)
            .collect();
        let mut p = (start_pos / CKPT_INTERVAL + 1) * CKPT_INTERVAL;
        while p < prompt_len {
            v.push(p);
            p += CKPT_INTERVAL;
        }
        v.sort_unstable();
        v.dedup();
        v
    }

    /// 跑一步前向，返回 `[n_vocab, n_tokens]` 的 logits（flatten）。
    ///
    /// **图跨步复用**（P2c）：与上一步同构（签名一致）时跳过 `reset` + 建图 +
    /// `gallocr.alloc`，只重写图输入的内容再 compute。图结构在 256 步窗口内不变，
    /// 所以 decode 只有每 256 步会重建一次。`SQWEN_NO_GRAPH_REUSE=1` 可关掉复用做 A/B。
    ///
    /// `SQWEN_PROFILE=1` 时打印每步的分阶段耗时（build/alloc/fill/compute/read），
    /// 这是定位 TG 瓶颈的主要工具（消融数据见 .superpowers/docs-archive/PERFORMANCE_COMPARISON.md，内部资料）。
    fn step(&mut self, ids: &[i32], pos: usize, out_ids: Option<&[i32]>) -> Result<Vec<f32>> {
        let profiling = std::env::var_os("SQWEN_PROFILE").is_some();
        let t0 = Instant::now();
        let n = ids.len();

        // 本步的签名：形状/取行方式一致就能复用上一步的图。
        // `n_kv_padded_after` 是纯计算（不动 KV 状态），真正的推进在 forward / reuse_step 里。
        let sig = StepSig {
            n_tokens: n,
            out_ids: out_ids.map(|v| v.to_vec()),
            n_kv_padded: self.state.kv.n_kv_padded_after(n),
        };
        let reuse = !matches!(std::env::var_os("SQWEN_NO_GRAPH_REUSE"), Some(_))
            && self.decode_graph.as_ref().is_some_and(|c| c.sig == sig);

        let mut t_build = t0.elapsed();
        let mut t_alloc = t0.elapsed();
        let (graph, logits): (RawGraph, StateTensor) = if reuse {
            // 缓存命中：只把「位置」推到本步（KV 写入区间 / step_start / n_past / out_rows），
            // 图本身、以及它在 gallocr 里的分配都不动。
            self.state
                .reuse_step(pos, n, out_ids.map(|v| v.to_vec()))
                .map_err(|e| anyhow!("{e}"))?;
            let c = self.decode_graph.as_ref().expect("上面刚判过 Some");
            (c.graph, c.logits)
        } else {
            // 复用同一个 scratch ctx：reset 只清对象表，arena 基址不变 →
            // 每步重建出的张量地址完全一致 → CUDA graph 的 key 与节点属性稳定。
            // （每步新建 ctx 会让地址漂移，ggml-cuda 每步都判成"新图"，永远不重放。）
            //
            // 顺序要紧：先让缓存失效（图节点就在这块 arena 里），再 reset。
            self.decode_graph = None;
            self.scratch.reset();
            let step_ctx = &self.scratch;
            let mut g = GraphBuilder::new_with_size(step_ctx, GRAPH_NODE_BUDGET);
            let out = forward(
                step_ctx,
                &mut g,
                &self.weights,
                &self.arch,
                &mut self.state,
                ids,
                pos,
                out_ids,
            )
            .map_err(|e| anyhow!("{e}"))?;
            t_build = t0.elapsed();

            // 先给本步张量分配（复用同一块 buffer，地址稳定），再写图输入
            self.galloc
                .alloc(&g)
                .map_err(|e| anyhow!("gallocr 分配失败: {e}"))?;
            t_alloc = t0.elapsed();

            let entry = CachedGraph {
                graph: g.as_raw(),
                logits: StateTensor::new(out),
                sig,
            };
            let (graph, logits) = (entry.graph, entry.logits);
            self.decode_graph = Some(entry);
            (graph, logits)
        };

        self.state
            .fill_inputs(ids)
            .map_err(|e| anyhow!("fill_inputs: {e}"))?;
        let t_fill = t0.elapsed();
        self.galloc
            .compute_graph(graph)
            .map_err(|e| anyhow!("compute: {e}"))?;
        let t_compute = t0.elapsed();

        // 直接把 D2H 的字节读进 `Vec<f32>`：投机校验批每轮要回读 `n_max+1` 行 logits
        // （4 行 = 4 MB + 103 万个元素），中间多一次 `Vec<u8>` 和一次逐元素重组是白付的。
        let vals: Vec<f32> = logits.read_f32_at(0, logits.nbytes() / 4);

        // `SQWEN_PHASE=1`：累积每步分阶段耗时，**只在 generate 结束时打一行**
        // （逐步 eprintln 自己就要 1~2 ms，会把被测对象抬高）。
        if phase_on() {
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
            self.step_phase[0] += ms(t_build);
            self.step_phase[1] += ms(t_alloc - t_build);
            self.step_phase[2] += ms(t_fill - t_alloc);
            self.step_phase[3] += ms(t_compute - t_fill);
            self.step_phase[4] += ms(t0.elapsed() - t_compute);
            self.step_phase_n += 1;
        }

        if profiling {
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
            eprintln!(
                "[profile] n={:4} build={:6.2} alloc={:6.2} fill={:5.2} compute={:7.2} read={:6.2} total={:7.2} ms reuse={}",
                ids.len(),
                ms(t_build),
                ms(t_alloc - t_build),
                ms(t_fill - t_alloc),
                ms(t_compute - t_fill),
                ms(t0.elapsed() - t_compute),
                ms(t0.elapsed()),
                u8::from(reuse),
            );
            // P8：图节点直方图（只打前几笔，避免刷屏）。
            // 每个非 view 节点至少一次 kernel launch，所以节点构成 ≈ kernel 构成。
            // P2c 之后 decode 的图是复用的，所以这里给 prefill 与 decode 各留计数器
            // —— 否则 dump 全被最前面的 3 个 prefill 块吃掉，看不到 decode 的图。
            if std::env::var_os("SQWEN_DUMP_GRAPH").is_some() {
                use std::sync::atomic::{AtomicUsize, Ordering};
                static DUMPED_PREFILL: AtomicUsize = AtomicUsize::new(0);
                static DUMPED_DECODE: AtomicUsize = AtomicUsize::new(0);
                let budget = if n == 1 { 2 } else { 3 };
                let ctr = if n == 1 { &DUMPED_DECODE } else { &DUMPED_PREFILL };
                if ctr.fetch_add(1, Ordering::Relaxed) < budget {
                    let (nodes, hist) = graph.op_histogram();
                    let views: usize = hist
                        .iter()
                        .filter(|(name, _)| {
                            matches!(
                                name.as_str(),
                                "VIEW" | "RESHAPE" | "PERMUTE" | "TRANSPOSE" | "CONT"
                            )
                        })
                        .map(|(_, c)| *c)
                        .sum();
                    eprintln!(
                        "[graph] n_tokens={} 节点={} （view/reshape/permute/cont {} 个）→ 约 {} 次 kernel launch",
                        ids.len(),
                        nodes,
                        views,
                        nodes - views
                    );
                    for (name, c) in hist.iter().take(20) {
                        eprintln!("[graph]   {name:<20} {c}");
                    }
                    // P8：融合潜力 —— ggml-cuda 的融合要求模式在**节点序列里相邻**。
                    // 数一下每个已知模式实际出现了多少次（每个融合点省 1~2 个 kernel）。
                    let seq = graph.op_sequence();
                    let pat = |names: &[&str]| -> usize {
                        if names.is_empty() || seq.len() < names.len() {
                            return 0;
                        }
                        (0..=seq.len() - names.len())
                            .filter(|&i| {
                                names
                                    .iter()
                                    .enumerate()
                                    .all(|(k, n)| seq[i + k] == *n)
                            })
                            .count()
                    };
                    // MoE：40 层 × (routed + shared) = 80 处期望
                    // TOPK_MOE 是 ggml-cuda 把整条专家路由折成一个 kernel 的模式
                    // （`ggml_cuda_topk_moe_fusion`），要求下面这 5 个节点**严格相邻**；
                    // 期望 40 处。见 `moe.rs` 里 `builder.expand(&w_norm)` 的注释。
                    eprintln!(
                        "[graph] 融合模式命中: MAT_ID+MAT_ID+GLU={} MAT+MAT+GLU={} NORM+MUL+ROPE={} ROPE+VIEW+SET_ROWS={} TOPK_MOE={} GDN_CACHE={}",
                        pat(&["MUL_MAT_ID", "MUL_MAT_ID", "GLU"]),
                        pat(&["MUL_MAT", "MUL_MAT", "GLU"]),
                        pat(&["RMS_NORM", "MUL", "ROPE"]),
                        pat(&["ROPE", "VIEW", "SET_ROWS"]),
                        pat(&["SOFT_MAX", "RESHAPE", "ARGSORT", "VIEW", "GET_ROWS"]),
                        // GDN 快照回写：`ggml_cuda_try_gdn_cache_fusion`（ggml-cuda.cu:2897）
                        // 要求 GDN 之后第一个非 view 节点就是 CPY（view 会被跳过），命中后
                        // 省掉那次拷贝。中间夹 1~2 个 view 都正常（新状态视图 + 目标视图）。
                        // 期望 30 处（30 个 GDN 层）。见 `graph.rs::gdn_step` 的注释。
                        pat(&["GATED_DELTA_NET", "VIEW", "CPY"])
                            + pat(&["GATED_DELTA_NET", "VIEW", "VIEW", "CPY"]),
                    );
                }
            }
        }
        Ok(vals)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// 关掉自适应时必须完全退回旧的「固定 n_max」行为。
    /// 关掉自适应时必须完全退回旧的「固定 n_max」行为。
    #[test]
    fn spec_policy_fixed_when_adaptive_off() {
        let mut p = SpecPolicy::new(3, false);
        p.begin_request();
        for _ in 0..50 {
            assert_eq!(p.pick(), 3);
            p.on_round(3, 0, 20.0);
        }
    }

    /// 跑完 8 轮暖机后按探测序列取样：先 5 轮投机、再 5 轮不投机，然后锁定。
    fn run_probe(spec: &[(f64, usize)], plain: &[(f64, usize)]) -> SpecPolicy {
        let mut p = SpecPolicy::new(3, true);
        p.begin_request();
        for _ in 0..SpecPolicy::PROBE_DELAY {
            let k = p.pick();
            assert_eq!(k, 1, "暖机轮应统一按投机跑");
            p.on_round(k, 1, 30.0);
        }
        for &(ms, acc) in spec {
            let k = p.pick();
            assert_eq!(k, 1, "探测前半段应是投机");
            p.on_round(k, acc, ms);
        }
        for &(ms, acc) in plain {
            let k = p.pick();
            assert_eq!(k, 0, "探测后半段应是不投机");
            p.on_round(k, acc, ms);
        }
        p
    }

    /// 投机明显更快 → 锁定投机（此后一直 1）。
    #[test]
    fn spec_policy_probe_sequence_then_decide() {
        // 前两轮含 CUDA graph 重捕获开销，中位数把它滤掉
        let spec = [(40.0, 1), (35.0, 1), (13.0, 1), (13.0, 1), (13.0, 1)];
        let plain = [(40.0, 0), (33.0, 0), (7.6, 0), (7.6, 0), (7.6, 0)];
        let mut p = run_probe(&spec, &plain);
        // k1 中位吞吐 ≈ 2/13.0 = 0.154 t/ms；k0 ≈ 1/7.6 = 0.132 → 投机胜
        assert_eq!(p.decision, Some(true));
        assert!(!p.is_plain());
        for _ in 0..20 {
            assert_eq!(p.pick(), 1);
            p.on_round(1, 1, 13.0);
        }
    }

    /// **回归测试**：最快的那轮恰好草稿被拒（acc=0）不能把吞吐算低。
    /// 旧实现取「最省时那轮 + 它的 acc」→ 0.091 t/ms < 不投机 0.132 → 误判成不投机。
    #[test]
    fn spec_policy_ignores_unlucky_fast_sample() {
        let spec = [(11.0, 0), (13.0, 1), (13.0, 1), (13.0, 1), (13.0, 1)];
        let plain = [(7.6, 0), (7.6, 0), (7.6, 0), (7.6, 0), (7.6, 0)];
        let p = run_probe(&spec, &plain);
        assert_eq!(p.decision, Some(true), "中位数应把最快但被拒的那轮排掉");
    }

    /// 不投机明显更快 → 锁定不投机（此后一直 0）。
    #[test]
    fn spec_policy_latches_plain_when_speculation_loses() {
        let spec = [(40.0, 0), (40.0, 0), (40.0, 0), (40.0, 0), (40.0, 0)];
        let plain = [(8.0, 0), (8.0, 0), (8.0, 0), (8.0, 0), (8.0, 0)];
        let mut p = run_probe(&spec, &plain);
        assert_eq!(p.decision, Some(false));
        assert!(p.is_plain());
        for _ in 0..20 {
            assert_eq!(p.pick(), 0);
        }
    }

    /// 优势不足 `SPEC_MARGIN` 时维持不投机（避免为了 2% 反复付切换成本）。
    #[test]
    fn spec_policy_needs_margin_to_switch_to_spec() {
        let plain = [(8.0, 0), (8.0, 0), (8.0, 0), (8.0, 0), (8.0, 0)];
        // 15ms 出 2 个 → 0.133 t/ms，比 0.125 高 6.7% > 5% → 投机
        let p = run_probe(&[(15.0, 1); 5], &plain);
        assert_eq!(p.decision, Some(true));
        // 16.3ms 出 2 个 → 0.123 t/ms，只高 2% → 维持不投机
        let q = run_probe(&[(16.3, 1); 5], &plain);
        assert_eq!(q.decision, Some(false));
        assert!(q.is_plain());
    }

    /// 检查点位置必须**正好**落在分块边界上，否则快照会是半个 chunk 的中间状态。
    #[test]
    fn plan_chunks_cuts_at_checkpoints() {
        // 无切割点：按 ubatch 均分
        assert_eq!(
            plan_chunks(0, 100, 30, &[]),
            vec![(0, 30), (30, 60), (60, 90), (90, 100)]
        );
        // 切割点落在块中间 → 块被缩短到切割点
        assert_eq!(
            plan_chunks(0, 100, 30, &[10, 75]),
            vec![(0, 10), (10, 40), (40, 70), (70, 75), (75, 100)]
        );
        // 切割点在块边界上（或范围外）→ 不产生影响
        assert_eq!(
            plan_chunks(0, 100, 50, &[50, 0, 100, 200]),
            vec![(0, 50), (50, 100)]
        );
        // 复用时从 start 开始切
        assert_eq!(plan_chunks(40, 100, 30, &[60]), vec![(40, 60), (60, 90), (90, 100)]);
        assert!(plan_chunks(5, 5, 10, &[]).is_empty());
    }

    #[test]
    fn budget_accepts_exact_fit() {        assert!(check_budget(1, 1, 2).is_ok());
        assert!(check_budget(1000, 3096, 4096).is_ok());
    }

    #[test]
    fn budget_rejects_empty_and_overflow() {
        assert!(check_budget(0, 1, 16).is_err());
        assert!(check_budget(4097, 1, 4096).is_err());
        assert!(check_budget(1000, 3097, 4096).is_err());
        // prompt 本身放得下、但加上生成就超了
        assert!(check_budget(4096, 1, 4096).is_err());
    }

    /// `--ctx-size` 语义（对齐 llama.cpp 的 `n_ctx = 0`）：
    /// 0 = 模型训练上下文；正数 = 用户指定；超过训练上下文要给出提示。
    #[test]
    fn ctx_size_resolution() {
        // 0 → 训练上下文
        assert_eq!(resolve_ctx_size(0, 262144).0, 262144);
        assert_eq!(resolve_ctx_size(0, 262144).1, CtxNote::Ok);
        // 显式指定
        assert_eq!(resolve_ctx_size(4096, 262144).0, 4096);
        assert_eq!(resolve_ctx_size(4096, 262144).1, CtxNote::Ok);
        assert_eq!(resolve_ctx_size(262144, 262144).0, 262144);
        assert_eq!(resolve_ctx_size(262144, 262144).1, CtxNote::Ok);
        // 超过训练上下文：允许但要提示
        assert_eq!(resolve_ctx_size(300000, 262144).1, CtxNote::ExceedsTrain);
        assert_eq!(resolve_ctx_size(300000, 262144).0, 300000);
        // 模型没声明 context_length → 兜底 4096
        assert_eq!(resolve_ctx_size(0, 0).0, 4096);
        assert_eq!(resolve_ctx_size(0, 0).1, CtxNote::NoTrainInfo);
        // 非法值（<=0 且非 0 的负数）按"用训练上下文"处理，且结果至少为 1
        assert_eq!(resolve_ctx_size(-5, 0).0, 4096);
        assert_eq!(resolve_ctx_size(1, 0).0, 1);
    }

    /// 重复死循环检测：整段逐字重复 3 次才算，差一个 token 都不算。
    #[test]
    fn loop_guard_detects_exact_repetition() {
        let w = LOOP_WINDOW;
        let unit: Vec<u32> = (0..w as u32).map(|i| 100 + i).collect();
        // 2 次 → 不触发
        let mut two = Vec::new();
        two.extend_from_slice(&unit);
        two.extend_from_slice(&unit);
        assert!(!is_degenerate_loop(&two));
        // 3 次 → 触发
        let mut three = Vec::new();
        for _ in 0..3 {
            three.extend_from_slice(&unit);
        }
        assert!(is_degenerate_loop(&three));
        // 第三次只差一个 token → 不触发
        let mut almost = Vec::new();
        for _ in 0..3 {
            almost.extend_from_slice(&unit);
        }
        almost.pop();
        assert!(!is_degenerate_loop(&almost));
        // 普通不重复的长文本不触发
        let plain: Vec<u32> = (0..w as u32 * 4).collect();
        assert!(!is_degenerate_loop(&plain));
    }
}
