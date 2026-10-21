# cache-prompt 的正确性验证：命中路径与未命中路径的答案都必须正常
$ErrorActionPreference = "Stop"
$m = "E:\LLMModels\bigbang\bigbang-v1-i1-gguf\BigBang-v1.i1-Q4_K_M.gguf"
$t = "E:\LLMModels\unsloth\qwen3.6-35B-A3B-Q4_K_m\chat_template_v22.5.jinja"
$port = 8397

$long = (1..120 | ForEach-Object { "这是一段用于撑长上下文的填充文本，第 $_ 段，主题是若干无关紧要的叙述。" }) -join ""
$turn1 = "$long`n`n请只回答：这段话一共分成多少段？"

$p = Start-Process -FilePath ".\target\release\sqwen.exe" -ArgumentList @(
    "server", $m, "--port", $port, "-c", "8192", "--template", $t,
    "--reasoning", "off", "--reasoning-format", "answer", "--cache-prompt") `
    -RedirectStandardError "target\cp_verify.err" -RedirectStandardOutput "target\cp_verify.out" -PassThru
Start-Sleep -Seconds 45

function Ask($msgs, $label) {
    $b = @{ messages = $msgs; max_tokens = 48 } | ConvertTo-Json -Depth 6
    $r = (Invoke-WebRequest -Uri "http://127.0.0.1:$port/v1/chat/completions" -Method Post `
          -ContentType "application/json" -Body $b -TimeoutSec 300).Content | ConvertFrom-Json
    # Write-Host 不进管道，避免被赋值捕获（否则调用处什么都打印不出来）
    Write-Host "[$label] cached=$($r.cached_tokens)/$($r.usage.prompt_tokens) TTFT=$([math]::Round($r.ttft_ms,1))ms"
    Write-Host "        答: $($r.choices[0].content)"
    return $r
}

$r1 = Ask @(@{ role = "user"; content = $turn1 }) "第1轮"
$r2 = Ask @(@{ role = "user"; content = $turn1 },
            @{ role = "assistant"; content = $r1.choices[0].content },
            @{ role = "user"; content = "再用一句话回答：这段文本在讲什么？" }) "第2轮(延续→应命中)"
$r3 = Ask @(@{ role = "user"; content = "1+1 等于几？只回答数字。" }) "第3轮(完全不同→应未命中)"
$r4 = Ask @(@{ role = "user"; content = $turn1 },
            @{ role = "assistant"; content = $r1.choices[0].content },
            @{ role = "user"; content = "再数一次段数。" }) "第4轮(回到旧历史→应命中)"

Get-Process sqwen -ErrorAction SilentlyContinue | Stop-Process -Force
