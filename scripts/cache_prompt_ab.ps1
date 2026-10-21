# 多轮对话 TTFT 对比：--cache-prompt 开 vs 关
# 用法：pwsh -File scripts/cache_prompt_ab.ps1
$ErrorActionPreference = "Stop"
$m = "E:\LLMModels\bigbang\bigbang-v1-i1-gguf\BigBang-v1.i1-Q4_K_M.gguf"
$t = "E:\LLMModels\unsloth\qwen3.6-35B-A3B-Q4_K_m\chat_template_v22.5.jinja"
$port = 8396

# 第一轮塞一段长文本（约 2~3k token），第二轮把它作为历史重发 —— 模拟真实多轮
$long = (1..120 | ForEach-Object { "这是一段用于撑长上下文的填充文本，第 $_ 段，主题是若干无关紧要的叙述。" }) -join ""
$turn1 = "$long`n`n用一句话总结上面这段话。"

function Run-Case($label, $extraArgs) {
    $args = @("server", $m, "--port", $port, "-c", "8192", "--template", $t,
              "--reasoning", "off", "--reasoning-format", "answer") + $extraArgs
    $p = Start-Process -FilePath ".\target\release\sqwen.exe" -ArgumentList $args `
        -RedirectStandardError "target\cp_$label.err" -RedirectStandardOutput "target\cp_$label.out" -PassThru
    Start-Sleep -Seconds 45
    $r1 = $null; $r2 = $null
    try {
        $b1 = @{ messages = @(@{ role = "user"; content = $turn1 }); max_tokens = 48 } | ConvertTo-Json -Depth 6
        $r1 = (Invoke-WebRequest -Uri "http://127.0.0.1:$port/v1/chat/completions" -Method Post `
              -ContentType "application/json" -Body $b1 -TimeoutSec 300).Content | ConvertFrom-Json
        $b2 = @{ messages = @(
                    @{ role = "user"; content = $turn1 },
                    @{ role = "assistant"; content = $r1.choices[0].content },
                    @{ role = "user"; content = "再换个说法总结一次。" }
                ); max_tokens = 48 } | ConvertTo-Json -Depth 6
        $r2 = (Invoke-WebRequest -Uri "http://127.0.0.1:$port/v1/chat/completions" -Method Post `
              -ContentType "application/json" -Body $b2 -TimeoutSec 300).Content | ConvertFrom-Json
    } catch { "  请求失败: $_" }
    Get-Process sqwen -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 3
    if ($r1 -and $r2) {
        "=== $label ==="
        "  第 1 轮: prompt_tokens=$($r1.usage.prompt_tokens) cached=$($r1.cached_tokens) TTFT=$([math]::Round($r1.ttft_ms,1)) ms"
        "  第 2 轮: prompt_tokens=$($r2.usage.prompt_tokens) cached=$($r2.cached_tokens) TTFT=$([math]::Round($r2.ttft_ms,1)) ms"
    }
}

Run-Case "cache_off" @("--no-cache-prompt")
Run-Case "cache_on"  @("--cache-prompt")
