// src/engine/sampler.rs — 确定性采样器（greedy / top-k / top-p / min-p / 温度 / 重复惩罚）
//
// 语义对齐 llama.cpp 的 sampler chain，**顺序也一致**：
//
//     penalties(repeat/presence/frequency, repeat_last_n)
//       → top_k → top_p → min_p → temp → dist
//
// 性能是硬要求：TG（逐 token 生成）的最大单点就在这里。这块演进过三轮：
//   1. 旧实现只要 temp>0 就把全部 248320 个候选 `sort_by` 全排序，实测 **18 ms/token**；
//   2. 按 llama.cpp 的做法改成"部分选择"（`select_nth_unstable` quickselect）之后，
//      仍然要 **1.55 ms/token**（占 TG ~15%）—— 瓶颈不在复杂度而在常数：每次比较都要
//      间接寻址 `data[*a]`/`data[*b]`，在 993 KB 的 logits 上是纯随机访问；而且为了拿到
//      下标还要先写出一个 248320 项的 `Vec<u32>`（993 KB）；
//   3. 现在换成 `select_top_k`：顺序扫描 + k 大小的降序缓冲，读 `data` 全程线性。
// 其余各步：
//   - top_p：先取前 256 个，累积概率不够再成倍扩大候选数（`llama_sampler_top_p_apply`）；
//   - min_p：本质是 logit 阈值 `max_logit + ln(min_p)`，一次 O(n) 扫描（`llama_sampler_min_p_apply`）；
//   - 完全不截断（top_k=0 且 top_p>=1 且 min_p<=0）：只需要一次 O(n) 的 softmax 累积，**不需要排序**。
//
// **确定性**是硬要求（golden test、可复现 bench、复现 bug）：
//   - 随机源是 splitmix64（无外部依赖；seed=0 的首个输出有公开标准值，单元测试钉住）；
//   - 排序按 `(logit 降序, 下标升序)`，并列时不依赖 sort 的稳定性；
//   - greedy 路径不消耗 RNG，且相同最大值取最小下标；
//   - 惩罚历史只在真的启用惩罚时维护；默认参数（repeat=1.0 / presence=0 / frequency=0）
//     下与旧实现逐 bit 等价（golden / tokenizer parity 不受影响）。
//
// 边界语义（都有测试）：
//   - `temp <= 0` 或 NaN → greedy；
//   - NaN logit 当作 `-inf`（引擎真出过 NaN logits：GGUF 偏移 bug 让权重全错，
//     采样器至少不能把 NaN 当成最大 logit 把垃圾 token 选出来）；
//   - `top_k == 0` → 不截断；`top_k >= 候选数` → 不截断；
//   - `top_p >= 1.0` → 不截断；`top_p <= 0` → 只留概率最大的那个（退化为 greedy）；
//   - `min_p <= 0` → 不截断；`min_keep` 保证至少留 1 个（llama.cpp 的 "first token always matches"）。

use std::collections::{HashMap, VecDeque};

/// top_p 首次部分排序的候选数（llama.cpp `llama_sampler_top_p_apply` 用 256）。
///
/// 累积概率不够时按 ×4 扩大，最坏退化到全量（分布极平时才会发生）。
const TOP_P_INITIAL_K: usize = 256;

/// splitmix64：Steele 等人的混合函数，状态就是一个 u64。
#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// [0, 1) 的均匀分布（取高 53 位，正好用满 f64 的尾数）。
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// NaN 归一成 `-inf`，其余原样返回。
#[inline]
fn sanitize(x: f32) -> f32 {
    if x.is_nan() {
        f32::NEG_INFINITY
    } else {
        x
    }
}

/// 取最大值的下标；并列取最小下标；NaN 视作 `-inf`。
fn greedy(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = sanitize(logits[0]);
    for (i, &x) in logits.iter().enumerate().skip(1) {
        let v = sanitize(x);
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

// ---------------------------------------------------------------------------
// 采样参数
// ---------------------------------------------------------------------------

/// 采样参数。字段名与 llama.cpp CLI / server 一一对应。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParams {
    /// 温度；`<= 0`（或 NaN）走 greedy
    pub temp: f32,
    /// top-k；0 = 关闭
    pub top_k: u32,
    /// top-p（核采样）；>= 1.0 = 关闭
    pub top_p: f32,
    /// min-p；<= 0 = 关闭。语义：保留 `logit >= max_logit + ln(min_p)` 的候选
    pub min_p: f32,
    /// 重复惩罚；1.0 = 关闭。正 logit 除以它、非正 logit 乘以它（llama.cpp 同款修正）
    pub repeat_penalty: f32,
    /// presence 惩罚；0 = 关闭。出现过的 token 减一个常数
    pub presence_penalty: f32,
    /// frequency 惩罚；0 = 关闭。按出现次数线性扣分
    pub frequency_penalty: f32,
    /// 惩罚回溯窗口；0 = 关闭；<0 = 全部已生成 token
    pub repeat_last_n: i32,
    /// 至少保留的候选数（top_p / min_p 不会把集合压到比它更小）
    pub min_keep: usize,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temp: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_last_n: 0,
            min_keep: 1,
            seed: 42,
        }
    }
}

/// 与 llama.cpp `common_params_sampling` 对齐的默认值（`common/common.h:225-243`）。
///
/// CLI / server 用这几个常量；`SamplingParams::default()` 保持"中性"
/// （temp=1.0、不截断、惩罚关闭），这样库调用方与老测试的语义不被隐式改变。
pub const DEFAULT_TEMP: f32 = 0.80;
pub const DEFAULT_TOP_K: u32 = 40;
pub const DEFAULT_TOP_P: f32 = 0.95;
pub const DEFAULT_MIN_P: f32 = 0.05;
pub const DEFAULT_REPEAT_LAST_N: i32 = 64;

impl SamplingParams {
    /// 惩罚是否真的需要生效（对应 llama.cpp `llama_sampler_penalties::is_disabled`）。
    pub fn penalties_active(&self) -> bool {
        self.repeat_last_n != 0
            && (self.repeat_penalty != 1.0
                || self.presence_penalty != 0.0
                || self.frequency_penalty != 0.0)
    }

    /// llama.cpp 的默认采样配置（temp=0.8、top_k=40、top_p=0.95、min_p=0.05、惩罚窗口 64）。
    pub fn llama_cpp_default() -> Self {
        Self {
            temp: DEFAULT_TEMP,
            top_k: DEFAULT_TOP_K,
            top_p: DEFAULT_TOP_P,
            min_p: DEFAULT_MIN_P,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_last_n: DEFAULT_REPEAT_LAST_N,
            min_keep: 1,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------

/// 采样器：参数 + RNG 状态 + 惩罚历史 + 复用缓冲。
#[derive(Clone, Debug)]
pub struct Sampler {
    params: SamplingParams,
    /// 随机流状态（每次随机采样推进一次）
    rng: SplitMix64,
    /// 已生成 token 的惩罚回溯窗口（仅惩罚开启时维护）
    hist: VecDeque<u32>,
    /// 窗口内每个 token 的出现次数
    counts: HashMap<u32, u32>,
    /// 惩罚用的可写 logits 副本（复用，避免每 token 分配 1 MB）
    logits_buf: Vec<f32>,
    /// 不截断路径的 softmax 概率（复用，避免每 token 重新分配/重复算 exp）
    probs_buf: Vec<f64>,
    /// 候选下标（复用）
    cand: Vec<u32>,
}

impl Sampler {
    /// 兼容旧签名：只有 temp / top_k / top_p / seed，惩罚关闭。
    pub fn new(temp: f32, top_k: u32, top_p: f32, seed: u64) -> Self {
        Self::from_params(SamplingParams {
            temp,
            top_k,
            top_p,
            seed,
            ..Default::default()
        })
    }

    pub fn from_params(params: SamplingParams) -> Self {
        let rng = SplitMix64::new(params.seed);
        Self {
            params,
            rng,
            hist: VecDeque::new(),
            counts: HashMap::new(),
            logits_buf: Vec::new(),
            probs_buf: Vec::new(),
            cand: Vec::new(),
        }
    }

    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// 改参数（每请求覆盖用）。不改 RNG 状态，也不清历史。
    pub fn set_params(&mut self, params: SamplingParams) {
        self.params = params;
    }

    /// 构造时用的 seed。
    pub fn seed(&self) -> u64 {
        self.params.seed
    }

    /// 把随机流与惩罚历史重置回起点。generate 每次开始都会调用它，
    /// 所以「同一 prompt + 同一配置」永远给出同一串 token（golden test / bench 需要）。
    pub fn restart(&mut self) {
        self.rng = SplitMix64::new(self.params.seed);
        self.hist.clear();
        self.counts.clear();
    }

    /// 换一个 seed 并把随机流重置到它（重跑同一段生成以复现问题时用）。
    pub fn reseed(&mut self, seed: u64) {
        self.params.seed = seed;
        self.restart();
    }

    fn penalties_active(&self) -> bool {
        self.params.penalties_active()
    }

    /// 记录一个**已生成**的 token（llama.cpp 的 `llama_sampler_accept`）。
    ///
    /// 注意与 llama.cpp 一致：只喂生成出来的 token，不喂 prompt。
    pub fn accept(&mut self, token: u32) {
        if !self.penalties_active() {
            return;
        }
        let last_n = if self.params.repeat_last_n < 0 {
            usize::MAX
        } else {
            self.params.repeat_last_n as usize
        };
        if last_n == 0 {
            return;
        }
        *self.counts.entry(token).or_insert(0) += 1;
        self.hist.push_back(token);
        if self.hist.len() > last_n {
            if let Some(old) = self.hist.pop_front() {
                if let Some(c) = self.counts.get_mut(&old) {
                    *c -= 1;
                    if *c == 0 {
                        self.counts.remove(&old);
                    }
                }
            }
        }
    }

    /// 从 `logits` 采一个 token id。
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        assert!(!logits.is_empty(), "sample: logits 不能为空");
        let params = self.params;
        let n = logits.len();

        // ---- 1. penalties：只在开启时复制一份可写 logits ----
        // 顺序与 llama.cpp 一致：先 repeat（正 logit 除、非正 logit 乘），
        // 再减去 frequency*count + presence。
        // 复杂度只看"窗口内出现过的 token 数"（`counts` 的元素数，通常 ≤ repeat_last_n），
        // 不要对整张词表做 248k 次哈希查找。
        let penal = self.penalties_active();
        if penal {
            self.logits_buf.clear();
            self.logits_buf.extend_from_slice(logits);
            let Self { logits_buf, counts, .. } = self;
            for (&tok, &c) in counts.iter() {
                let idx = tok as usize;
                if idx >= logits_buf.len() {
                    continue;
                }
                let v = &mut logits_buf[idx];
                if *v <= 0.0 {
                    *v *= params.repeat_penalty;
                } else {
                    *v /= params.repeat_penalty;
                }
                *v -= c as f32 * params.frequency_penalty + params.presence_penalty;
            }
        }
        let data: &[f32] = if penal { &self.logits_buf } else { logits };

        // ---- 2. greedy：不碰 RNG ----
        if !(params.temp > 0.0) {
            return greedy(data);
        }

        // ---- 3. 求 max（NaN 视作 -inf）；全 -inf 时退化为下标 0 ----
        let mut max_l = f32::NEG_INFINITY;
        for &v in data {
            let s = sanitize(v);
            if s > max_l {
                max_l = s;
            }
        }
        if max_l == f32::NEG_INFINITY {
            return 0;
        }
        let temp = params.temp as f64;
        let weight = |i: u32| -> f64 { (((sanitize(data[i as usize]) as f64) - max_l as f64) / temp).exp() };

        // ---- 4. 完全不截断：一次 O(n) exp + 一次廉价累积，**不排序** ----
        let need_trunc = params.top_k > 0 || params.top_p < 1.0 || params.min_p > 0.0;
        if !need_trunc {
            // 概率只算一次存进复用缓冲（旧实现是"求和 + 累积"各算一遍 exp，
            // debug 构建下 libm 的 exp 是这块的主要开销）。
            let mut probs = std::mem::take(&mut self.probs_buf);
            probs.clear();
            probs.reserve(n);
            let mut total = 0.0f64;
            for &v in data {
                let w = (((sanitize(v) as f64) - max_l as f64) / temp).exp();
                total += w;
                probs.push(w);
            }
            let target = total * self.rng.next_f64();
            let mut cum = 0.0f64;
            let mut chosen = greedy(data); // 浮点尾差没选中时的安全回退
            for (i, &w) in probs.iter().enumerate() {
                cum += w;
                if cum >= target {
                    chosen = i as u32;
                    break;
                }
            }
            self.probs_buf = probs;
            return chosen;
        }

        // ---- 5. 截断路径：top_k → top_p → min_p（顺序同 llama.cpp）----
        //
        // 只对"候选集"排序/选择，绝不对 248k 全排序：
        //   - top_k>0：quickselect 取前 k（O(n)），只排序这 k 个；
        //   - top_k==0 且 top_p<1：llama.cpp 的自适应策略 —— 先部分排序前 256 个，
        //     累积概率不够就成倍扩大（实测本模型第一轮就够，top_p 核只有几十个 token）；
        //   - 两者都没开（只有 min_p）：候选集保持下标序，min_p 是 O(cand) 的阈值过滤。
        let min_keep = params.min_keep.max(1);
        let mut cand = std::mem::take(&mut self.cand);

        // 5.1 选出候选集，统一走 `select_top_k`（顺序扫描 + k 大小的降序缓冲）：
        //   - top_k>0：直接取前 k，返回时已按 `(logit 降序, 下标升序)` 排好；
        //   - top_k==0 且 top_p<1：llama.cpp 的自适应策略 —— 先取前 256，累积概率
        //     不够就成倍扩大（实测本模型第一轮就够，top_p 核只有几十个 token）；
        //   - 两者都没开（只有 min_p）：候选集保持下标序，min_p 是 O(cand) 的阈值过滤。
        if params.top_k > 0 {
            select_top_k(data, params.top_k as usize, &mut cand);
        } else if params.top_p < 1.0 {
            let p = params.top_p.max(0.0) as f64;
            let mut m = TOP_P_INITIAL_K.min(n);
            loop {
                select_top_k(data, m, &mut cand);
                let total: f64 = cand.iter().map(|&i| weight(i)).sum();
                let keep = prefix_len(&cand, &weight, total * p, min_keep);
                if keep < cand.len() || m >= n {
                    cand.truncate(keep);
                    break;
                }
                m = (m * 4).min(n);
            }
        } else {
            cand.clear();
            cand.extend(0..n as u32);
        }

        // 5.2 top_p（top_k 已排序的分支）：直接截断降序前缀
        if params.top_k > 0 && params.top_p < 1.0 {
            let p = params.top_p.max(0.0) as f64;
            let total: f64 = cand.iter().map(|&i| weight(i)).sum();
            let keep = prefix_len(&cand, &weight, total * p, min_keep);
            cand.truncate(keep);
        }

        // 5.3 min_p：本质是 logit 阈值 `max_logit + ln(min_p)`
        if params.min_p > 0.0 {
            let max_cand = cand
                .iter()
                .map(|&i| sanitize(data[i as usize]))
                .fold(f32::NEG_INFINITY, f32::max);
            if max_cand > f32::NEG_INFINITY {
                let min_logit = max_cand + params.min_p.ln();
                let mut visited = 0usize;
                cand.retain(|&i| {
                    let keep = visited < min_keep || sanitize(data[i as usize]) >= min_logit;
                    visited += 1;
                    keep
                });
            }
        }

        // 5.4 temp + dist：单次累积（llama.cpp `llama_sampler_dist_apply`）
        let total: f64 = cand.iter().map(|&i| weight(i)).sum();
        let target = total * self.rng.next_f64();
        let mut cum = 0.0f64;
        let mut chosen = *cand.first().unwrap_or(&0);
        for &i in cand.iter() {
            cum += weight(i);
            if cum >= target {
                chosen = i;
                break;
            }
        }

        self.cand = cand;
        chosen
    }
}

/// 降序候选里，累积权重达到 `target` 所需的前缀长度（至少 `min_keep`）。
///
/// 累积始终到不了 `target` 时返回 `cand.len()`——调用方据此判断"候选集不够大，需要扩大"。
fn prefix_len(cand: &[u32], weight: &impl Fn(u32) -> f64, target: f64, min_keep: usize) -> usize {
    let mut cum = 0.0f64;
    for (pos, &i) in cand.iter().enumerate() {
        cum += weight(i);
        if cum >= target && pos + 1 >= min_keep {
            return pos + 1;
        }
    }
    cand.len()
}

/// 候选下标比较：logit 降序，并列取下标升序（确定性）。
fn cmp_logit(data: &[f32], a: &u32, b: &u32) -> std::cmp::Ordering {
    let (la, lb) = (sanitize(data[*a as usize]), sanitize(data[*b as usize]));
    lb.total_cmp(&la).then_with(|| a.cmp(b))
}

/// 顺序挑选 `data` 里按全序 `(logit 降序, 下标升序)` 排前 `k` 个的下标，写进 `out`。
///
/// 返回时 `out` 已按该全序排好（`out[0]` 最好、`out[k-1]` 最差），下游可以直接按
/// 降序前缀做 top_p 截断。
///
/// **为什么不用 `select_nth_unstable_by`**：那是 O(n) 的 quickselect，渐进复杂度没问题，
/// 常数却极差 —— 每次比较都要间接寻址 `data[*a]`/`data[*b]`，在 248320 个 logit（993 KB）
/// 上是纯随机访问，缓存全打穿；而且为了拿到下标还得先写出一个 248320 项的 `Vec<u32>`。
/// 实测这条路径 **1.55 ms/token**，占 TG 约 15%。
///
/// 换成"顺序扫描 + k 大小的降序缓冲"后：缓冲尾元素就是阈值，绝大多数元素只需要一次
/// **顺序**加载 + 一次比较就被淘汰。`data` 全程线性读，没有间接寻址、没有下标数组。
///
/// **分块预筛**：外层先按 256 个一组，用一次可向量化的计数判断整块有没有元素能过阈值；
/// 一个都过不了就整块跳过。阈值只会单调上升（每轮要么不变、要么变大），所以"用块入口
/// 的阈值判空"永远安全 —— 过不了入口阈值的元素，更过不了之后更高的阈值。
/// 这个计数循环没有分支、没有提前退出，是 LLVM 最擅长向量化的归约形态
/// （也正因如此，`-C target-cpu=native` 在这里才终于有东西可加速）。
///
/// **与旧路径逐位等价**：两者求的都是同一个全序下的前 k 个，而全序下的前 k 个是唯一
/// 确定的。`tests` 里有对随机（含并列、NaN、±inf）数据的逐项比对。
fn select_top_k(data: &[f32], k: usize, out: &mut Vec<u32>) {
    /// 分块预筛的粒度：256 个 f32 = 1 KB，一次计数循环的开销可以忽略。
    const CHUNK: usize = 256;

    let n = data.len();
    let k = k.min(n);
    out.clear();
    if k == 0 {
        return;
    }

    // k 相对 n 太大时，插入缓冲会退化成 O(n·k)：直接快选 + 全排序（旧实现）。
    if k * 4 > n {
        out.extend(0..n as u32);
        if k < n {
            out.select_nth_unstable_by(k - 1, |a, b| cmp_logit(data, a, b));
            out.truncate(k);
        }
        out.sort_unstable_by(|a, b| cmp_logit(data, a, b));
        return;
    }

    // 前 k 个先直接进缓冲，做一次插入排序。降序：`out[0]` 最好、`out[k-1]` 最差。
    // 并列时不下移，于是并列者保持下标升序（正好是想要的 tie-break）。
    out.extend(0..k as u32);
    for i in 1..k {
        let v = sanitize(data[i]);
        let mut j = i;
        while j > 0 && sanitize(data[out[j - 1] as usize]) < v {
            out.swap(j, j - 1);
            j -= 1;
        }
    }
    let mut worst = sanitize(data[out[k - 1] as usize]);

    let mut base = k;
    while base < n {
        let end = (base + CHUNK).min(n);
        let chunk = &data[base..end];

        // 预筛刻意用**原始值**比较：NaN 与任何数的有序比较都是 false，而
        // `sanitize(NaN) = -inf` 也永远过不了阈值（`worst >= -inf`）—— 两者完全等价，
        // 却省掉了逐元素的 NaN 判断，让这个计数循环保持无分支、可向量化。
        let mut hits = 0usize;
        for &x in chunk {
            if x > worst {
                hits += 1;
            }
        }
        if hits == 0 {
            base = end;
            continue;
        }

        for (off, &x) in chunk.iter().enumerate() {
            if x > worst {
                // 走到这里 `x` 一定不是 NaN（NaN 的比较为 false），所以插入位置用原始值
                // 即可；只有缓冲里可能残留的 NaN（来自最开始那 k 个）需要 sanitize。
                let mut j = k - 1;
                while j > 0 && sanitize(data[out[j - 1] as usize]) < x {
                    out[j] = out[j - 1];
                    j -= 1;
                }
                out[j] = (base + off) as u32;
                worst = sanitize(data[out[k - 1] as usize]);
            }
        }
        base = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 旧实现（写全量下标 + quickselect + 全排序）—— 作为 `select_top_k` 的等价性基准。
    fn reference_top_k(data: &[f32], k: usize) -> Vec<u32> {
        let n = data.len();
        let k = k.min(n);
        let mut v: Vec<u32> = (0..n as u32).collect();
        if k == 0 {
            v.clear();
            return v;
        }
        if k < n {
            v.select_nth_unstable_by(k - 1, |a, b| cmp_logit(data, a, b));
            v.truncate(k);
        }
        v.sort_unstable_by(|a, b| cmp_logit(data, a, b));
        v
    }

    fn check(data: &[f32], k: usize) {
        let mut got = Vec::new();
        select_top_k(data, k, &mut got);
        assert_eq!(got, reference_top_k(data, k), "n={} k={k}", data.len());
    }

    /// 随机数据（刻意混入大量并列、NaN、±inf）下，新实现必须与旧路径**逐项**一致。
    ///
    /// 这条是这次改写的主要安全网：采样器是确定性的硬要求，任何 tie-break 偏移都会
    /// 让同一 seed 的输出不可复现。
    #[test]
    fn select_top_k_matches_quickselect() {
        let mut rng = SplitMix64::new(0xC0FFEE);
        for _ in 0..200 {
            let n = 1 + (rng.next_u64() % 700) as usize;
            let mut data: Vec<f32> = Vec::with_capacity(n);
            for i in 0..n {
                let r = rng.next_u64();
                let v = match r % 8 {
                    // 大量并列：只有 5 个取值，逼出 tie-break 差异
                    0..=2 => ((r % 5) as f32) * 0.5,
                    3 => f32::NAN,
                    4 => {
                        if i % 2 == 0 {
                            f32::INFINITY
                        } else {
                            f32::NEG_INFINITY
                        }
                    }
                    _ => ((r % 100_000) as f32) / 1000.0 - 50.0,
                };
                data.push(v);
            }
            for &k in &[0usize, 1, 2, 3, 7, 40, n / 4, n / 2, n, n + 5] {
                check(&data, k);
            }
        }
    }

    /// 边界：空、单元素、全相等、全 NaN。
    #[test]
    fn select_top_k_edges() {
        let mut out = Vec::new();
        select_top_k(&[], 3, &mut out);
        assert!(out.is_empty());

        select_top_k(&[1.5], 0, &mut out);
        assert!(out.is_empty());
        select_top_k(&[1.5], 1, &mut out);
        assert_eq!(out, vec![0]);

        // 全相等：按「下标升序」取前 k 个
        select_top_k(&[2.0f32; 8], 3, &mut out);
        assert_eq!(out, vec![0, 1, 2]);

        // 全 NaN 视作 -inf：同样是下标升序前 k 个
        select_top_k(&[f32::NAN; 8], 3, &mut out);
        assert_eq!(out, vec![0, 1, 2]);
    }

    /// 跨多个预筛块的大数组，以及触发 `k*4 > n` 退化分支的 k。
    #[test]
    fn select_top_k_spans_many_chunks() {
        let n = 5000;
        let data: Vec<f32> = (0..n).map(|i| ((i * 7919) % 1000) as f32).collect();
        check(&data, 40);
        check(&data, 256);
        check(&data, 257);
        check(&data, 3000); // k*4 > n → 退化到 quickselect + 全排序
    }
}
