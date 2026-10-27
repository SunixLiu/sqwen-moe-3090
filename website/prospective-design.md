# sqwen landing page — prospective design

Status: **draft, awaiting review** (revision 1)

## Product truth

sqwen is a local inference engine for one model family (Qwen3.6-35B-A3B, `qwen35moe`: Gated DeltaNet linear attention + full attention + MoE), running the whole 20.2 GiB Q4_K_M model on a single 24 GB GPU (RTX 3090) and serving an OpenAI-compatible HTTP API. It is written in Rust over hand-written FFI to ggml/ggml-cuda; the compute graph, KV cache, sampler, prompt cache, speculative decoding and HTTP server are all implemented in-repo.

Audience: developers who run local LLMs or build agent frameworks on one consumer GPU. Primary action: download v0.1.1 (73.5 MB zip) and point an existing OpenAI client at it. What the visitor should remember: **one 24 GB card, ~200K context, every byte of VRAM accounted for** — and that the project measures itself honestly instead of claiming parity.

Verified facts used on the page (all from README.zh-CN.md / DEVELOPMENT_LOG.md):
- Weights 20.2 GiB resident; `-c 172032` → 23,676 MiB used, ~530 MiB free; `-c 200000` → ~380 MiB free
- KV cache 5.6 KiB/token (q4_0 K/V, 10 full-attention layers)
- 172K → 200K context costs only 0.4% speed
- Speculative decode measured: d=0 125–130 t/s, d=1 145–147 t/s, d=2 125 t/s, d=3 106 t/s; draft-token marginal cost ≈ 60% of a full decode; acceptance decays pos0≈0.9, pos1≈0.7; engine probes online and locks the winner (`--spec-adaptive`)
- ABBA vs llama.cpp (same session, same ggml-cuda.dll): decode 137 vs 151 t/s (1.10×), prefill ~3,000 vs ~3,200 t/s (~5%), startup 12.3 s vs 19.2 s, VRAM 23,676 vs 24,083 MiB
- Prompt cache: prefix reuse + GDN state checkpoints, multi-turn first-token latency ~11× better
- Phase timing: host side ~0.4 ms/step (6%), GPU compute ~95%
- v0.1.1 release: 73.5 MB zip = exe + 4 ggml DLLs + cudart + QUICKSTART.txt; MIT license
- Honest limits: serial requests (429 beyond `--queue-capacity`), no `stop`/`n>1`/`logprobs`/`response_format`, Windows-only build, one architecture family, "not a llama.cpp replacement"

## Governing idea

**The first frame is the VRAM ledger, not a headline.** The product's identity is squeezing one 24 GB card dry ("把 24 GB 显存榨到位"), and a conventional hero-plus-cards page would hide exactly that. The opening is an interactive memory map of the 24 GiB bar: weights (fixed), KV cache (grows with context), runtime/driver overhead, free space. Dragging the context slider (32K→200K) rebalances the ledger live — KV grows, free shrinks, and the speed readout stays flat (the product's core claim: context costs VRAM, not speed).

Why this is more than an editorial restyle: meaning changes through agency. The visitor manipulates the constraint that defines the product and watches the consequence stay boring (0.4% slower at 200K) — which is the point. The ledger grammar then carries through the whole page: benchmark numbers as ledger rows, spec-decode tradeoff as a cost table, phase timing as a stacked bar.

Competitor-swap test: swap in "llama.cpp" and the premise breaks — its default KV is f16 (far more than 5.6 KiB/token), its spec defaults differ, and none of the anchor numbers hold. The interactive math is architecture-specific to qwen35moe + q4_0 KV + GDN state.

## First-frame composition

- Top-left: wordmark `sqwen` (display face) + one-line honest positioning ("只针对 qwen35moe，把 24 GB 显存榨到位。不是 llama.cpp 的平替。")
- Dominant object: the 24 GiB memory bar, segmented with exact MiB readouts in mono; a context slider under it (32K–200K, default 172,032); three live readouts: KV used / free / speed delta vs 172K
- No eyebrow, no giant centered headline, no floating widget. The bar is the hero.

## Visual grammar

- **Palette** (derived from subject: a hardware budget, not a SaaS brand):
  - Canvas `#15171b` (warm near-black, instrument panel)
  - Text `#e9e4d8` (warm off-white), secondary `#9a948a`
  - Weights segment `#c99a4b` (amber — the fixed, dominant allocation)
  - KV segment `#6f9a8b` (muted teal — the variable spend; data-viz role, not an accent glow)
  - Runtime/driver segments `#3a3e45` / `#2c2f35` (neutral darks)
  - Free space `#1d2025` with a hairline top edge
  - Interactive state (slider thumb, active row, links) `#e8b23e` (bright amber)
  - No gradients anywhere (deliberate absence: flat instrument look)
- **Type**: Bricolage Grotesque (display + Latin body; distinctive cuts, OFL) · JetBrains Mono (every measured number, code, table data — semantic, not decorative) · Noto Sans SC (CJK body). Fallbacks: system sans / mono while loading.
- **Shapes**: right angles, 2px radius max; hairline rules as ledger separators; no cards, no pills, no shadows-as-decoration. Buttons are rectangular blocks with a 2px radius.
- **Icons**: none except one bespoke mark — a segmented-bar glyph echoing the memory map, used beside the wordmark. No icon tiles, no stock icons.
- **Motion**: slider-driven rebalance (the signature); count-up of readouts on load; restrained section reveals (8px translate + opacity, ≤300 ms). `prefers-reduced-motion` → all static, values shown at final state.

## Signature behavior

Context slider: drag → KV segment width animates, free segment shrinks, readouts tick (KV GiB, free MiB, "speed vs 172K: −0.4% at 200K"), and a warning state appears below ~500 MiB free ("再开任何 GPU 程序就会触发 WDDM 分页" — the README's real caution). Keyboard: arrow keys step the slider by ~1.7K tokens (1% of range); ARIA slider semantics; static fallback shows the 172K ledger with the same numbers as text. Verified: Playwright functional test — initial 172,032 → KV 0.92 GiB / free 530 MiB; max drag → exactly 200,000 → KV 1.07 GiB / free 377 MiB + warning visible; min → 32,768 → free 1,292 MiB, warning hidden; all three fonts confirmed loaded via document.fonts.check.

## Journey

1. Arrival: the ledger + honest one-liner
2. The math: why context is cheap here (5.6 KiB/token, KV table by context size)
3. The measurement: spec-decode cost table (d=0/1/2/3) + "deeper is a net loss" insight + ABBA benchmark ledger vs llama.cpp with the discipline note
4. The interface: endpoints table, tool calling, prompt cache, the real server command block
5. The honesty: known limits, stated plainly
6. Action: download v0.1.1 zip (real release link), MIT, repo link

## Asset route

No raster assets. All visuals are DOM/SVG drawn from real numbers. Fonts self-hosted as local woff2: Bricolage Grotesque + JetBrains Mono served as single variable-font files (weight ranges declared in @font-face); Noto Sans SC instanced at 400/500/700 and subset via pyftsubset to exactly the characters used on the page (~55 KB each, OFL). Total font payload ~270 KB. Provisional: runtime/driver segment sizes (~2.0 GiB / ~370 MiB) are derived by interpolation between the README's two anchor points; labeled as derived in the page footnote. Unverified: none of the claims on the page go beyond README/dev-log evidence.

## Stack and pipeline

Static HTML + CSS + vanilla JS, one file (`design-specimen.html` → production `index.html`), no build step, no framework. Rationale: the product is a zero-runtime-dependency engine that already ships a single-file webui.html; a static page matches that character and deploys anywhere (GitHub Pages). Alternative considered: Astro for content-led pages — rejected, overkill for one page with one interactive island. 2D only; motion is interactive (slider) plus restrained scroll reveals; no 3D.

## Acceptance evidence

- Specimen rendered via Playwright capture at desktop / laptop / tablet / phone / 320px + reduced-motion pass
- Slider math checked against both README anchors (172K → ~530 MiB free; 200K → ~380 MiB free)
- Every number on the page traceable to README.zh-CN.md or DEVELOPMENT_LOG.md
- Contrast measured on canvas/text/segment colors; keyboard path through slider and links
- Independent subagent review with screenshots before production

## Unresolved assumptions

- Release asset URL verified via GitHub API: `https://github.com/SunixLiu/sqwen-moe-3090/releases/download/v0.1.1/sqwen-v0.1.1-windows-x64.zip` (77,105,729 B = 73.5 MB) ✓
- Deployment: **local only for now** — user explicitly said not to publish to GitHub (no Pages deploy, no commit of website/ unless asked later)

## Revision ledger

| Revision | User feedback and exact decision | Changes made | Evidence/preview | State |
| --- | --- | --- | --- | --- |
| 1 | (none yet) | Initial direction + specimen built | design-specimen.html + captures | awaiting review |
| 2 | internal review round 1: phone/narrow horizontal overflow (absolutely-positioned segment tags escaping the bar); em-dash flag | removed kv/runtime/driver segment tags (kept weights tag only), added max-width/ellipsis guard on .seg-tag; fixed KV table values to exact model math (179/358/717 MiB rows); slider step 8192 → "any" so 200,000 is exactly reachable | round-2 captures: scrollWidth = viewport at 390/320; vision review of all 7 shots (CJK render, overflow, contrast, layout) all OK; contrast pairs all PASS (3.34–14.15:1); Playwright functional test (slider min/max/keyboard, fonts loaded, no page errors); em-dashes: 4 total, justified (title, 2 table placeholders, 1 prose) — not spam | approved by user |
| 3 | user: "我觉得没问题" — explicit green light on direction; then "这个web暂时不发布到github" — production stays local, no GitHub Pages deploy | production `index.html` created from specimen + bespoke SVG favicon (data URI); release asset URL verified via GitHub API (sqwen-v0.1.1-windows-x64.zip, 77,105,729 B = 73.5 MB) | round-5 capture of index.html pending; independent subagent review in flight | production (local only) |
| 4 | independent subagent review: PASS 95/100, no hard failures; 5 findings (1 moderate, 4 low) | (1) ≤380px media query tightens lede/bar/margins so the slider input is fully inside the first frame at 320×640 (measured: input bottom = 640 = fold); (2) "decode 落后 ~11%" → "~10%" (matches table's 1.10×); (3) prose verified against DEVELOPMENT_LOG.md — "1200 token" and "5 轮样本" confirmed kept; "32 轮暖机" corrected to "跳过前 32 步（约 45 token）" (log: PROBE_DELAY=32, warmup is 8 rounds); "+5~7%" corrected to "各 ~+7%" (log: sqwen +7.6%, llama +7.4%); (4) `.nowrap` span keeps "显存分配" from wrapping mid-word at 320px (verified via cropped re-inspection; earlier vision flag was a false positive on the two-line layout); (5) readouts get `role="status" aria-live="polite"` for screen-reader feedback during drag; Noto Sans SC re-subset for the new characters (跳/过/前) | round-8 captures (narrow/desktop/phone) all clean per vision review; fold measured by Playwright bounding boxes | production (local only), review findings closed |
