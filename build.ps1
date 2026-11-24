<#
  Builds LiveTranscriber (PyInstaller, one-folder) and zips it for a GitHub release.

    .\build.ps1               # app only        -> dist\LiveTranscriber-win64.zip
    .\build.ps1 -GpuPack      # + optional GPU  -> dist\LiveTranscriber-gpu-pack.zip
                              #   (CUDA 12 cuBLAS/cuDNN/cudart/cuFFT DLLs taken from NVIDIA's official PyPI wheels)

  Needs Windows 10/11 x64 and Python 3.10-3.12 on PATH. Creates .venv on first run.
#>
param([switch]$GpuPack, [switch]$SkipInstall)
$ErrorActionPreference = "Continue"   # pip/pyinstaller write notices to stderr; we check exit codes instead
Set-Location $PSScriptRoot
function Run([string]$what, [scriptblock]$cmd) { & $cmd; if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" } }

if (-not (Test-Path .venv)) { Run "venv" { python -m venv .venv } }
$py = ".\.venv\Scripts\python.exe"
if (-not $SkipInstall) {
    Run "pip upgrade" { & $py -m pip install --upgrade pip }
    Run "pip install" { & $py -m pip install -r requirements.txt }
    # faster-whisper depends on the CPU-only "onnxruntime", which shadows "onnxruntime-gpu".
    # Keep only the GPU build (it also runs fine on CPU).
    Run "pip uninstall" { & $py -m pip uninstall -y onnxruntime onnxruntime-gpu }
    Run "pip install onnxruntime-gpu" { & $py -m pip install "onnxruntime-gpu>=1.22,<1.24" }
}

# NOTE: CUDA DLLs are deliberately NOT bundled in the exe (see GPU pack below). If PyInstaller finds any on
# this machine's PATH it would pull them in, so build with a PATH that has no CUDA/cuDNN folders.
$cleanPath = ($env:PATH -split ";" | Where-Object { $_ -and $_ -notmatch "CUDA|cudnn|NVIDIA GPU Computing|torch" }) -join ";"
$oldPath = $env:PATH; $env:PATH = $cleanPath
try {
    Run "PyInstaller" { & $py -m PyInstaller --noconfirm --clean --windowed --name LiveTranscriber `
        --collect-all faster_whisper --collect-all ctranslate2 --collect-all onnxruntime `
        --collect-all sherpa_onnx --collect-all onnx_asr --collect-binaries av --hidden-import pyaudiowpatch `
        --copy-metadata onnx-asr --copy-metadata huggingface-hub --copy-metadata faster-whisper `
        --copy-metadata ctranslate2 --copy-metadata onnxruntime-gpu --copy-metadata sherpa-onnx `
        --exclude-module torch --exclude-module matplotlib --exclude-module tensorboard `
        --distpath dist --workpath build --specpath build app.py }
} finally { $env:PATH = $oldPath }

# TensorRT provider is huge and unused (we only use CUDA/CPU)
Get-ChildItem dist\LiveTranscriber -Recurse -Filter "onnxruntime_providers_tensorrt.dll" | Remove-Item -Force
# Safety net: no NVIDIA runtime libs inside the app zip
Get-ChildItem dist\LiveTranscriber -Recurse -Include "cublas*.dll","cudnn*.dll","cudart*.dll","cufft*.dll","nvrtc*.dll" | Remove-Item -Force

Compress-Archive -Path dist\LiveTranscriber -DestinationPath dist\LiveTranscriber-win64.zip -Force
Write-Host "Built dist\LiveTranscriber-win64.zip"

if ($GpuPack) {
    $w = "dist\gpu-wheels"; $pack = "dist\gpu-pack\cuda"
    Remove-Item $w, dist\gpu-pack -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $w, $pack | Out-Null
    foreach ($pkg in "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12", "nvidia-cufft-cu12", "nvidia-cudnn-cu12", "nvidia-cuda-nvrtc-cu12") {
        Run "download $pkg" { & $py -m pip download $pkg --no-deps --only-binary=:all: --platform win_amd64 --python-version 3.12 -d $w }
    }
    foreach ($whl in Get-ChildItem $w -Filter *.whl) {
        $x = Join-Path $w ($whl.BaseName); Copy-Item $whl.FullName "$x.zip"; Expand-Archive "$x.zip" -DestinationPath $x -Force
        Get-ChildItem $x -Recurse -Filter *.dll | Copy-Item -Destination $pack -Force
    }
    Compress-Archive -Path dist\gpu-pack\cuda -DestinationPath dist\LiveTranscriber-gpu-pack.zip -Force
    Write-Host "Built dist\LiveTranscriber-gpu-pack.zip"
}
