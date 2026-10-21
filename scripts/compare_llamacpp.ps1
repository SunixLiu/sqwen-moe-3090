# scripts/compare_llamacpp.ps1 — sqwen vs llama.cpp 同条件对比
#
# 用法：
#   .\scripts\compare_llamacpp.ps1 -Model <gguf> [-PromptTokens 512] [-GenTokens 64] [-Reps 3] [-Sqwen <exe>]
#
# 说明：
#   - 两边都用 **greedy（temp=0）** 且固定生成 N 个 token（sqwen bench 内部会临时清空 EOS，
#     与 llama-bench 的行为一致），否则吞吐数字不可比；
#   - llama-bench 自报 mean（-r 次重复），sqwen bench 也是 warmup + -r 次均值；
#   - 额外跑一组 sqwen 的"真实采样"配置（temp=1 + top_k/top_p/min_p + 重复惩罚），
#     用来看采样开销占比——llama-bench 不测这个。

param(
    [Parameter(Mandatory = $true)][string]$Model,
    [int]$PromptTokens = 512,
    [int]$GenTokens = 64,
    [int]$Reps = 3,
    [string]$Sqwen = "",
    [string]$LlamaBench = $(if ($env:SQWEN_LLAMA_CPP) { Join-Path $env:SQWEN_LLAMA_CPP "build\bin\Release\llama-bench.exe" } else { "D:\01Projects\llama.cpp\build\bin\Release\llama-bench.exe" }),
    [int]$NGpuLayers = 99
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Model)) { throw "model not found: $Model" }
if ([string]::IsNullOrWhiteSpace($Sqwen)) {
    $candidates = @("target\release\sqwen.exe", "target\debug\sqwen.exe")
    $Sqwen = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $Sqwen) { throw "找不到 sqwen.exe，请用 -Sqwen 指定" }
}
if (-not (Test-Path -LiteralPath $Sqwen)) { throw "sqwen not found: $Sqwen" }

function Get-SqwenNumbers([string]$line) {
    # 形如：pp512: 899.0 t/s | tg48: 60.8 t/s | TTFT ... | tg_ms/tok 16.45 (std 0.59) | runs 3
    if ($line -match 'pp(\d+):\s*([\d.]+)\s*t/s.*tg(\d+):\s*([\d.]+)\s*t/s.*tg_ms/tok\s*([\d.]+)') {
        return [pscustomobject]@{
            PP = [double]$Matches[2]
            TG = [double]$Matches[4]
            TGms = [double]$Matches[5]
        }
    }
    throw "无法解析 sqwen bench 输出：$line"
}

Write-Host "=== sqwen: greedy (temp=0) ===" -ForegroundColor Cyan
$sqwenGreedy = & $Sqwen bench $Model --prompt-tokens $PromptTokens --gen-tokens $GenTokens --repetitions $Reps 2>&1 |
    Select-String -Pattern '^pp\d+' | Select-Object -Last 1
if (-not $sqwenGreedy) { throw "sqwen bench 没有输出结果行" }
$sq = Get-SqwenNumbers $sqwenGreedy.Line
Write-Host $sqwenGreedy.Line

Write-Host "=== sqwen: 真实采样 (temp=1, top_k=40, top_p=0.95, min_p=0.05, repeat=1.1/64) ===" -ForegroundColor Cyan
$sampling = "--temp 1.0 --top-k 40 --top-p 0.95 --min-p 0.05 --repeat-penalty 1.1 --repeat-last-n 64"
$sqwenSampled = & $Sqwen bench $Model --prompt-tokens $PromptTokens --gen-tokens $GenTokens --repetitions $Reps $sampling.Split(" ") 2>&1 |
    Select-String -Pattern '^pp\d+' | Select-Object -Last 1
$sqs = Get-SqwenNumbers $sqwenSampled.Line
Write-Host $sqwenSampled.Line

$llamaPP = $null; $llamaTG = $null
if (Test-Path -LiteralPath $LlamaBench) {
    Write-Host "=== llama.cpp: llama-bench (greedy) ===" -ForegroundColor Cyan
    $out = & $LlamaBench -m $Model -p $PromptTokens -n $GenTokens -ngl $NGpuLayers -r $Reps -o md 2>&1
    $out | Where-Object { $_ -match '^\|' } | ForEach-Object { Write-Host $_ }
    foreach ($line in $out) {
        if ($line -match 'pp' + $PromptTokens + '\s*\|\s*([\d.]+)') { $llamaPP = [double]$Matches[1] }
        if ($line -match 'tg' + $GenTokens + '\s*\|\s*([\d.]+)') { $llamaTG = [double]$Matches[1] }
    }
} else {
    Write-Host "WARNING: 找不到 llama-bench ($LlamaBench)，跳过对比" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== 汇总 ===" -ForegroundColor Green
"{0,-34} {1,10} {2,10}" -f "实现/配置", "PP t/s", "TG t/s"
"{0,-34} {1,10:N1} {2,10:N1}" -f "sqwen greedy", $sq.PP, $sq.TG
"{0,-34} {1,10:N1} {2,10:N1}" -f "sqwen temp=1 + 截断/惩罚", $sqs.PP, $sqs.TG
if ($llamaPP) {
    "{0,-34} {1,10:N1} {2,10:N1}" -f "llama.cpp (llama-bench)", $llamaPP, $llamaTG
    "{0,-34} {1,10:N2} {2,10:N2}" -f "llama/squeen 倍数（越大越好）", ($llamaPP / $sq.PP), ($llamaTG / $sq.TG)
}
