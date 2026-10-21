# build_ggml.ps1
# Copy a prebuilt CUDA ggml (import libs + DLLs) into .\build-ggml\ so that
# build.rs can link against it.
#
# Strategy: reuse the artifacts already produced by a llama.cpp checkout
#           (CMake + MSVC + nvcc), instead of rebuilding ggml from scratch.
#
# Usage:
#   $env:SQWEN_LLAMA_CPP = "D:\path\to\llama.cpp"; .\build_ggml.ps1
#   .\build_ggml.ps1 -LlamaCpp "D:\path\to\llama.cpp"
#   .\build_ggml.ps1 -LlamaCpp ... -CudaRoot "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
#
# Output:
#   .\build-ggml\lib\Release\*.lib   (link time)
#   .\build-ggml\bin\Release\*.dll   (run time)

param(
    [string]$LlamaCpp = $env:SQWEN_LLAMA_CPP,
    [string]$CudaRoot = $env:SQWEN_CUDA_ROOT
)

$ErrorActionPreference = "Stop"

if (-not $LlamaCpp) {
    throw "llama.cpp path not set. Pass -LlamaCpp <path> or set `$env:SQWEN_LLAMA_CPP."
}
if (-not (Test-Path $LlamaCpp)) {
    throw "llama.cpp path does not exist: $LlamaCpp"
}

# Default CUDA location; only used for cudart, and only if present.
if (-not $CudaRoot) {
    $CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
}

$build_root = Join-Path $PSScriptRoot "build-ggml"
$lib_dir = Join-Path $build_root "lib\Release"
$bin_dir = Join-Path $build_root "bin\Release"

New-Item -ItemType Directory -Force -Path $lib_dir | Out-Null
New-Item -ItemType Directory -Force -Path $bin_dir | Out-Null

Write-Host "llama.cpp : $LlamaCpp"
Write-Host "output    : $build_root"

Write-Host ""
Write-Host "=== copying ggml import libraries -> $lib_dir ==="

$ggml_libs = @(
    "$LlamaCpp\build\ggml\src\Release\ggml.lib",
    "$LlamaCpp\build\ggml\src\Release\ggml-base.lib",
    "$LlamaCpp\build\ggml\src\Release\ggml-cpu.lib",
    "$LlamaCpp\build\ggml\src\ggml-cuda\Release\ggml-cuda.lib"
)
$missing = 0
foreach ($src in $ggml_libs) {
    $name = Split-Path $src -Leaf
    if (Test-Path $src) {
        Write-Host "  [copy] $name"
        Copy-Item -Path $src -Destination (Join-Path $lib_dir $name) -Force
    } else {
        Write-Warning "  [missing] $src"
        $missing++
    }
}

# cudart.lib (from the CUDA SDK)
$cudart_src = Join-Path $CudaRoot "lib\x64\cudart.lib"
if (Test-Path $cudart_src) {
    Write-Host "  [copy] cudart.lib"
    Copy-Item -Path $cudart_src -Destination (Join-Path $lib_dir "cudart.lib") -Force
} else {
    Write-Warning "  [missing] cudart.lib at $cudart_src (pass -CudaRoot if your CUDA lives elsewhere)"
}

Write-Host ""
Write-Host "=== copying ggml DLLs -> $bin_dir ==="

$ggml_dlls = @(
    "$LlamaCpp\build\bin\Release\ggml.dll",
    "$LlamaCpp\build\bin\Release\ggml-base.dll",
    "$LlamaCpp\build\bin\Release\ggml-cpu.dll",
    "$LlamaCpp\build\bin\Release\ggml-cuda.dll"
)
foreach ($src in $ggml_dlls) {
    $name = Split-Path $src -Leaf
    if (Test-Path $src) {
        Write-Host "  [copy] $name"
        Copy-Item -Path $src -Destination (Join-Path $bin_dir $name) -Force
    } else {
        Write-Warning "  [missing] $src"
        $missing++
    }
}

# cudart64_*.dll (from the CUDA SDK)
$cudart_dll = Get-ChildItem -Path (Join-Path $CudaRoot "bin") -Filter "cudart64_*.dll" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($cudart_dll) {
    Write-Host "  [copy] $($cudart_dll.Name)"
    Copy-Item -Path $cudart_dll.FullName -Destination (Join-Path $bin_dir $cudart_dll.Name) -Force
} else {
    Write-Warning "  [missing] cudart64_*.dll under $CudaRoot\bin"
}

Write-Host ""
Write-Host "=== result ==="
$lib_files = Get-ChildItem -Path $lib_dir -Filter "*.lib" | Select-Object -ExpandProperty Name
$dll_files = Get-ChildItem -Path $bin_dir -Filter "*.dll" | Select-Object -ExpandProperty Name
Write-Host "libraries: $($lib_files -join ', ')"
Write-Host "DLLs     : $($dll_files -join ', ')"
Write-Host ""

if ($missing -gt 0) {
    Write-Warning "$missing file(s) missing - did you build llama.cpp with GGML_CUDA=ON in Release mode?"
    exit 1
}

Write-Host "Done. Next: cargo build --release"
