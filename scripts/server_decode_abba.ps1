# scripts/server_decode_abba.ps1 — 两边的 **server** 走同一条 HTTP 路径、ABBA 交叉的 decode 对比
#
# 为什么需要它（retro §2.9）：同一模型、同一套参数，两个环境给出**相反**的相对结论
# （本机 ABBA：llama +24%；用户环境分两次跑：sqwen +5%）。已知的差异是
#   ① 本机当时用 llama-cli 单轮，用户用 llama-server；
#   ② 用户那次 llama.cpp 冷启动很长，而两边都只看了**累计**口径。
# 要定论只能：同机、同会话、ABBA 顺序、**两边都走 server**、**都取稳态口径**。
#
# 用法：
#   .\scripts\server_decode_abba.ps1                      # 默认跑 无投机 + 投机 两个 block
#   .\scripts\server_decode_abba.ps1 -Which spec          # 只跑投机
#   .\scripts\server_decode_abba.ps1 -Cycles 1            # 每个 block 只跑一轮 ABBA
#
# 输出：每轮一行的表格 + 每个引擎的平均值 + 自检（两边 prompt_tokens / completion_tokens /
#       greedy 正文是否一致）。全部原始日志落在 -LogDir 下。

param(
    [string]$Model = $(if ($env:SQWEN_TEST_MODEL) { $env:SQWEN_TEST_MODEL } else { "E:\LLMModels\bigbang\bigbang-v1-i1-gguf\BigBang-v1.i1-Q4_K_M.gguf" }),
    [string]$Template = "E:\LLMModels\unsloth\qwen3.6-35B-A3B-Q4_K_M\chat_template_v22.5.jinja",
    [string]$Sqwen = "target\release\sqwen.exe",
    [string]$LlamaServer = $(if ($env:SQWEN_LLAMA_CPP) { Join-Path $env:SQWEN_LLAMA_CPP "build\bin\Release\llama-server.exe" } else { "D:\01Projects\llama.cpp\build\bin\Release\llama-server.exe" }),
    [int]$Ctx = 4096,
    [int]$MaxTokens = 1200,
    [int]$WarmupTokens = 400,
    [int]$Threads = 8,
    [int]$SqwenPort = 8131,
    [int]$LlamaPort = 8132,
    [ValidateSet("both", "nospec", "spec")][string]$Which = "both",
    [int]$SpecNMax = 3,
    [string]$LogDir = "target\abba",
    [int]$LoadTimeoutSec = 420,
    [switch]$SkipPreflight
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# 与 sqwen 的 bench/对比脚本一致：固定 greedy，避免采样随机性把两边拉开。
# 但 server 路径下**不能**清 EOS，所以 prompt 必须保证 1200 token 内不会自然结束。
$ContextPad = @"
缓存是计算机系统中用于加速数据访问的一种常见技术。它的基本原理是利用局部性：程序在一段时间内往往只访问一小部分数据，把这部分数据放在更快的存储介质上，就能减少平均访问延迟。设计缓存需要在容量、命中率、替换策略和一致性之间做权衡。现代处理器通常有多级缓存，每一级的容量和速度都不同；操作系统用页缓存来掩盖磁盘延迟；数据库用缓冲池来减少随机读。缓存失效被认为是计算机科学中最难的两件事之一。
"@
$MeasurePrompt = $ContextPad + "`n`n请从 1 开始逐个列出整数，每个数字后面跟一个逗号，中间不要换行，一直列下去，不要提前结束。"
$WarmPrompt = "请用一句话解释什么是哈希表。"

function New-ChatBody([string]$prompt, [int]$maxTokens) {
    return (@{
        model       = "sqwen"
        messages    = @(@{ role = "user"; content = $prompt })
        temperature = 0.0
        max_tokens  = $maxTokens
        stream      = $false
    } | ConvertTo-Json -Depth 6 -Compress)
}

function Get-VramFreeMiB {
    $out = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
    if (-not $out) { return 0 }
    return [int]($out | Select-Object -First 1)
}

function Wait-PortFree([int]$port, [int]$timeoutSec = 60) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        $c = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
        if (-not $c) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Wait-Healthy([int]$port, [int]$timeoutSec) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3 -UseBasicParsing
            if ($r.StatusCode -eq 200) { return $true }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

# 一次完整的测量：起 server → 等健康 → 预热 → 测量 → 关 server
function Invoke-OneRun {
    param(
        [string]$Engine,      # sqwen | llama
        [string]$Config,      # nospec | spec
        [string]$Tag
    )

    $port = if ($Engine -eq "sqwen") { $SqwenPort } else { $LlamaPort }
    $outLog = Join-Path $LogDir "$Tag.out.log"
    $errLog = Join-Path $LogDir "$Tag.err.log"
    if (-not (Wait-PortFree $port 30)) { throw "端口 $port 仍被占用" }

    if ($Engine -eq "sqwen") {
        $argv = @(
            "server", $Model,
            "--port", "$port",
            "-c", "$Ctx",
            "-b", "2048", "-u", "512",
            "-t", "$Threads",
            "--template", $Template,
            # 与 llama.cpp 的默认输出形状对齐：推理进 reasoning_content、content 只留回答。
            # 这一项**不影响生成**（只影响把哪部分放进响应），所以对计时是中性的；
            # 对齐它是为了让两边的正文可比（否则我们默认 answer 会把推理直接丢掉）。
            "--reasoning", "on",
            "--reasoning-effort", "medium",
            "--reasoning-preserve",
            "--reasoning-format", "deepseek"
        )
        if ($Config -eq "spec") { $argv += @("--spec-type", "draft-mtp", "--spec-draft-n-max", "$SpecNMax") }
        $exePath = (Resolve-Path -LiteralPath $Sqwen).Path
    } else {
        $argv = @(
            "-m", $Model,
            "--host", "127.0.0.1", "--port", "$port",
            "-c", "$Ctx",
            "-b", "2048", "-ub", "512",
            "-t", "$Threads",
            "-ngl", "99",
            "-fa", "on",
            "-ctk", "q4_0", "-ctv", "q4_0",
            "--chat-template-file", $Template,
            "--jinja",
            # 显式写死，不依赖这个 build 的 auto 默认（实测 auto 会选 deepseek）
            "--reasoning", "on",
            "--reasoning-effort", "medium",
            "--reasoning-preserve",
            "--reasoning-format", "deepseek"
        )
        if ($Config -eq "spec") { $argv += @("--spec-type", "draft-mtp", "--spec-draft-n-max", "$SpecNMax") }
        $exePath = (Resolve-Path -LiteralPath $LlamaServer).Path
    }

    $tLoad = Get-Date
    $proc = Start-Process -FilePath $exePath -ArgumentList $argv -PassThru -NoNewWindow `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog
    try {
        if (-not (Wait-Healthy $port $LoadTimeoutSec)) {
            throw "$Engine 在 $LoadTimeoutSec 秒内没起来；看 $errLog"
        }
        $loadSec = ((Get-Date) - $tLoad).TotalSeconds

        # 预热：① 把 GPU 时钟从空闲 P8 拉起来（retro 方法学第 1 条）
        #       ② 让 CUDA graph capture / 惰性模块加载都发生在测量之外
        $body = New-ChatBody $WarmPrompt $WarmupTokens
        Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $body -TimeoutSec 600 | Out-Null

        # 测量
        $body = New-ChatBody $MeasurePrompt $MaxTokens
        $t0 = Get-Date
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $body -TimeoutSec 900
        $wall = ((Get-Date) - $t0).TotalSeconds
    } finally {
        if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Seconds 2
        # 等显存真的还回来（下一个引擎要 20.2 GiB）
        $deadline = (Get-Date).AddSeconds(90)
        while ((Get-Date) -lt $deadline) {
            if ((Get-VramFreeMiB) -ge 23000) { break }
            Start-Sleep -Seconds 2
        }
    }

    # ---- 解析两个引擎各自的字段 ----
    # `usage.prompt_tokens` 两边都是**整段 prompt**；llama 的 `timings.prompt_n` 是
    # 「本次新算的 token」（命中缓存后只有几个），拿它做自检会误判 —— 实测踩过一次。
    $row = [ordered]@{
        Tag          = $Tag
        Engine       = $Engine
        Config       = $Config
        LoadSec      = [math]::Round($loadSec, 1)
        WallSec      = [math]::Round($wall, 2)
        PromptTokens = $null
        Cached       = $null
        GenTokens    = $null
        Cumulative   = $null   # sqwen eval_rate      / llama timings.predicted_per_second
        Steady       = $null   # sqwen eval_rate_steady / llama log 里的最后一个 tg_3s
        Drafted      = $null   # 只有 llama 的 timings 直接给（sqwen 要开 SQWEN_PROFILE，会拖慢吞吐）
        Accepted     = $null
        Finish       = $null
        Reasoning    = ""
        Content      = ""
    }
    if ($Engine -eq "sqwen") {
        $row.PromptTokens = [int]$resp.usage.prompt_tokens
        $row.Cached = [int]$resp.cached_tokens
        $row.GenTokens = [int]$resp.usage.completion_tokens
        $row.Cumulative = [double]$resp.eval_rate
        $row.Steady = [double]$resp.eval_rate_steady
        $row.Finish = [string]$resp.choices[0].finish_reason
        $row.Reasoning = [string]$resp.choices[0].message.reasoning_content
        $row.Content = [string]$resp.choices[0].message.content
    } else {
        $row.PromptTokens = [int]$resp.usage.prompt_tokens
        $row.Cached = [int]$resp.usage.prompt_tokens_details.cached_tokens
        $row.GenTokens = [int]$resp.timings.predicted_n
        $row.Cumulative = [double]$resp.timings.predicted_per_second
        $row.Finish = [string]$resp.choices[0].finish_reason
        $row.Reasoning = [string]$resp.choices[0].message.reasoning_content
        $row.Content = [string]$resp.choices[0].message.content
        $row.NewPromptTokens = [int]$resp.timings.prompt_n
        # llama.cpp 直接把投机统计放进 timings —— 解释差距时要用（是内核慢还是草稿质量差）
        if ($resp.timings.PSObject.Properties.Name -contains "draft_n") {
            $row.Drafted = [int]$resp.timings.draft_n
            $row.Accepted = [int]$resp.timings.draft_n_accepted
        }
        # tg_3s 只在 server 的 stderr 日志里（`n_gen = …, tg = …, tg_3s = …`，每 ≥3 秒一行）
        $tg = Select-String -Path $errLog -Pattern "tg_3s = *([0-9.]+) t/s" -AllMatches |
            ForEach-Object { $_.Matches } | ForEach-Object { [double]$_.Groups[1].Value }
        $row.Steady = if ($tg) { [double]($tg | Select-Object -Last 1) } else { 0.0 }
        $row.TgLines = if ($tg) { $tg.Count } else { 0 }
    }
    return [pscustomobject]$row
}

# ---------------------------------------------------------------------------
# 预检：把「比较前提」再核实一遍（retro 的教训：脚手架/前提的 bug 是沉默的）
# ---------------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Write-Host "=== 预检 ===" -ForegroundColor Cyan
Write-Host ("PowerShell {0}" -f $PSVersionTable.PSVersion)
if (-not (Test-Path -LiteralPath $Model)) { throw "模型不存在：$Model" }
Write-Host ("模型 : {0}  ({1:N2} GiB)" -f $Model, ((Get-Item $Model).Length / 1GB))
$sqwenDll = "target\release\ggml-cuda.dll"
$llamaDll = Join-Path (Split-Path $LlamaServer) "ggml-cuda.dll"
$h1 = (Get-FileHash $sqwenDll -Algorithm SHA256).Hash
$h2 = (Get-FileHash $llamaDll -Algorithm SHA256).Hash
Write-Host ("sqwen ggml-cuda.dll : {0}  {1}" -f $h1.Substring(0, 16), (Get-Item $sqwenDll).LastWriteTime)
Write-Host ("llama ggml-cuda.dll : {0}  {1}" -f $h2.Substring(0, 16), (Get-Item $llamaDll).LastWriteTime)
if ($h1 -ne $h2) {
    Write-Host "!!! 两个 ggml-cuda.dll 不同 —— 内核不是同一份，比较前提不成立 !!!" -ForegroundColor Red
}
$llamaVer = (& $LlamaServer --version 2>&1 | Select-String "version:") -join " "
Write-Host ("llama.cpp : {0}" -f $llamaVer.Trim())
Write-Host ("sqwen     : {0}" -f (Get-Item (Resolve-Path $Sqwen).Path).LastWriteTime)
if (-not $SkipPreflight) {
    $free = Get-VramFreeMiB
    Write-Host ("空闲显存 : {0} MiB" -f $free)
    if ($free -lt 23000) { throw "显存不足（{0} MiB < 23 GiB），先关掉别的推理进程" -f $free }
}
Get-Process sqwen, llama-server -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host ("杀掉残留进程 {0} ({1})" -f $_.ProcessName, $_.Id) -ForegroundColor Yellow
    Stop-Process -Id $_.Id -Force
}
Start-Sleep -Seconds 3

# ---------------------------------------------------------------------------
# 跑 ABBA：每个 cycle 的顺序是 A B B A（A=sqwen，B=llama.cpp）
# ---------------------------------------------------------------------------
$blocks = @()
if ($Which -in @("both", "nospec")) { $blocks += @{ Config = "nospec"; Cycles = 1 } }
if ($Which -in @("both", "spec")) { $blocks += @{ Config = "spec"; Cycles = 2 } }

$all = @()
foreach ($b in $blocks) {
    $cfg = $b.Config
    Write-Host ""
    Write-Host ("=== block: {0}（{1} 个 ABBA 轮）===" -f $cfg, $b.Cycles) -ForegroundColor Cyan
    for ($c = 1; $c -le $b.Cycles; $c++) {
        foreach ($eng in @("sqwen", "llama", "llama", "sqwen")) {
            $tag = "{0}-c{1}-{2}-{3}" -f $cfg, $c, $eng, (Get-Date -Format "HHmmss")
            Write-Host ("  -> {0} ({1})" -f $tag, $eng) -NoNewline
            $row = Invoke-OneRun -Engine $eng -Config $cfg -Tag $tag
            $all += $row
            Write-Host ("  载入 {0}s | 累计 {1:N1} t/s | 稳态 {2:N1} t/s | gen {3}" -f `
                $row.LoadSec, $row.Cumulative, $row.Steady, $row.GenTokens) -ForegroundColor DarkGray
        }
    }
}

# ---------------------------------------------------------------------------
# 汇总 + 自检
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== 逐轮明细 ===" -ForegroundColor Green
$all | Format-Table Tag, Engine, Config, PromptTokens, Cached, GenTokens, WallSec, Cumulative, Steady, Drafted, Accepted, Finish -AutoSize

Write-Host "=== 每个引擎的平均（ABBA 配对）===" -ForegroundColor Green
$summary = foreach ($cfg in ($all.Config | Select-Object -Unique)) {
    foreach ($eng in @("sqwen", "llama")) {
        $rows = $all | Where-Object { $_.Config -eq $cfg -and $_.Engine -eq $eng }
        if (-not $rows) { continue }
        $dr = ($rows | Measure-Object Drafted -Sum).Sum
        $ac = ($rows | Measure-Object Accepted -Sum).Sum
        [pscustomobject]@{
            Config        = $cfg
            Engine        = $eng
            N             = $rows.Count
            CumulativeAvg = [math]::Round(($rows | Measure-Object Cumulative -Average).Average, 2)
            SteadyAvg     = [math]::Round(($rows | Measure-Object Steady -Average).Average, 2)
            CumulativeMin = [math]::Round(($rows | Measure-Object Cumulative -Minimum).Minimum, 2)
            CumulativeMax = [math]::Round(($rows | Measure-Object Cumulative -Maximum).Maximum, 2)
            DraftAccept   = if ($dr -gt 0) { [math]::Round($ac / $dr, 3) } else { $null }
        }
    }
}
$summary | Format-Table -AutoSize

Write-Host "=== 相对差距（llama / sqwen，>1 表示 llama 更快）===" -ForegroundColor Green
foreach ($cfg in ($summary.Config | Select-Object -Unique)) {
    $s = $summary | Where-Object { $_.Config -eq $cfg -and $_.Engine -eq "sqwen" }
    $l = $summary | Where-Object { $_.Config -eq $cfg -and $_.Engine -eq "llama" }
    Write-Host ("{0,-8} 累计 {1:N3}x   稳态 {2:N3}x" -f $cfg,
        ($l.CumulativeAvg / $s.CumulativeAvg), ($l.SteadyAvg / $s.SteadyAvg))
}

Write-Host "=== 自检（不通过就说明这次对比不可信）===" -ForegroundColor Green
$problems = 0
foreach ($cfg in ($all.Config | Select-Object -Unique)) {
    $rows = $all | Where-Object { $_.Config -eq $cfg }
    $pt = ($rows.PromptTokens | Select-Object -Unique)
    if ($pt.Count -ne 1) {
        Write-Host ("  [X] {0}: 两边 prompt_tokens 不一致 {1} —— prompt 渲染没对齐，decode 对比也不干净" -f $cfg, ($pt -join "/")) -ForegroundColor Red
        $problems++
    } else {
        Write-Host ("  [OK] {0}: prompt_tokens 一致 = {1}" -f $cfg, $pt)
    }
    if ($rows | Where-Object { $_.Cached -gt 0 }) {
        Write-Host ("  [!] {0}: 有轮次命中了 prompt 缓存（{1}）—— 预热与测量应当用不同 prompt" -f $cfg,
            (($rows | Where-Object { $_.Cached -gt 0 } | ForEach-Object { "$($_.Tag):$($_.Cached)" }) -join ",")) -ForegroundColor Yellow
    }
    $gt = ($rows.GenTokens | Select-Object -Unique)
    if ($gt.Count -ne 1) {
        Write-Host ("  [!] {0}: completion_tokens 不一致 {1}（贪心下本该逐 token 一致；不影响速率对比，但值得查）" -f $cfg, ($gt -join "/")) -ForegroundColor Yellow
    } else {
        Write-Host ("  [OK] {0}: completion_tokens 一致 = {1}" -f $cfg, $gt)
    }
    if ($rows | Where-Object { $_.Finish -ne "length" }) {
        Write-Host ("  [!] {0}: 有请求不是 length 结束（可能撞上 EOS，稳态窗口变短）：{1}" -f $cfg, (($rows | Where-Object { $_.Finish -ne "length" }).Tag -join ",")) -ForegroundColor Yellow
    }
    if ($rows | Where-Object { $_.Steady -le 0 }) {
        Write-Host ("  [X] {0}: 有轮次稳态速率为 0（窗口没成型）—— 该轮作废" -f $cfg) -ForegroundColor Red
        $problems++
    }
    $minGen = ($rows | Measure-Object GenTokens -Minimum).Minimum
    if ($minGen -lt 600) {
        Write-Host ("  [!] {0}: 最少只生成了 {1} 个 token，稳态窗口偏短" -f $cfg, $minGen) -ForegroundColor Yellow
    }
    # 贪心正文（推理 + 回答拼起来，因为两边都是 deepseek 形状）：理论上应产出同一串 token。
    # 数值差异（不同 kernel 选择）可能让它在某处提前分叉，所以这一条只报警不判死。
    $fps = $rows | ForEach-Object { ($_.Reasoning + $_.Content) }
    $texts = $fps | ForEach-Object { $_.Substring(0, [Math]::Min(300, $_.Length)) } | Select-Object -Unique
    if ($texts.Count -ne 1) {
        Write-Host ("  [!] {0}: 两边贪心正文前 300 字不同（{1} 种）—— 数值/模板可能有差异" -f $cfg, $texts.Count) -ForegroundColor Yellow
    } else {
        Write-Host ("  [OK] {0}: 两边贪心正文前 300 字完全一致" -f $cfg)
    }
}
if ($problems -gt 0) { Write-Host "结论：这次对比有硬伤，先修脚手架再下结论" -ForegroundColor Red }
else { Write-Host "结论：没有硬伤" -ForegroundColor Green }

$csv = Join-Path $LogDir ("abba-{0}.csv" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
$all | Export-Csv -Path $csv -NoTypeInformation -Encoding UTF8
Write-Host ("明细已存 {0}" -f $csv)
