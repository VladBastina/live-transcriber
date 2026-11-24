#!/usr/bin/env bash
# Builds the Linux app (PyInstaller, one-folder) and tars it for a GitHub release.
#
#   ./build.sh               # app only        -> dist/LiveTranscriber-linux-x86_64.tar.gz
#   ./build.sh --gpu-pack    # + optional GPU  -> dist/LiveTranscriber-linux-gpu-pack.tar.gz
#                            #   (CUDA 12 cuBLAS/cuDNN/cudart/cuFFT taken from NVIDIA's official PyPI wheels)
#
# Needs: python3 (3.10-3.12) with venv + tkinter, binutils (objdump), x86_64 Linux.
#   Debian/Ubuntu: sudo apt install python3-venv python3-tk binutils
# Build on the OLDEST distro you want to support (the binary needs at least the build machine's glibc).
set -euo pipefail
cd "$(dirname "$0")"

GPU_PACK=0; SKIP_INSTALL=0
for a in "$@"; do
  case "$a" in
    --gpu-pack) GPU_PACK=1 ;;
    --skip-install) SKIP_INSTALL=1 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done

[ -d .venv ] || python3 -m venv .venv
PY=.venv/bin/python
if [ "$SKIP_INSTALL" = 0 ]; then
  $PY -m pip install --upgrade pip
  $PY -m pip install -r requirements.txt
  # faster-whisper pulls the CPU-only "onnxruntime", which shadows "onnxruntime-gpu"; keep only the GPU build.
  $PY -m pip uninstall -y onnxruntime onnxruntime-gpu
  $PY -m pip install "onnxruntime-gpu>=1.22,<1.24"
fi

# CUDA libs are deliberately NOT bundled (they are the optional GPU pack): build without LD_LIBRARY_PATH
env -u LD_LIBRARY_PATH $PY -m PyInstaller --noconfirm --clean --windowed --name LiveTranscriber \
  --collect-all faster_whisper --collect-all ctranslate2 --collect-all onnxruntime \
  --collect-all sherpa_onnx --collect-all onnx_asr --collect-binaries av \
  --copy-metadata onnx-asr --copy-metadata huggingface-hub --copy-metadata faster-whisper \
  --copy-metadata ctranslate2 --copy-metadata onnxruntime-gpu --copy-metadata sherpa-onnx \
  --exclude-module torch --exclude-module matplotlib --exclude-module tensorboard \
  --distpath dist --workpath build --specpath build app.py

find dist/LiveTranscriber -name 'libonnxruntime_providers_tensorrt*' -delete
# safety net: no NVIDIA runtime libs inside the app archive
find dist/LiveTranscriber \( -name 'libcublas*' -o -name 'libcudnn*' -o -name 'libcudart*' \
  -o -name 'libcufft*' -o -name 'libnvrtc*' \) -delete

tar czf dist/LiveTranscriber-linux-x86_64.tar.gz -C dist LiveTranscriber
echo "Built dist/LiveTranscriber-linux-x86_64.tar.gz"

if [ "$GPU_PACK" = 1 ]; then
  rm -rf dist/gpu-wheels dist/gpu-pack; mkdir -p dist/gpu-wheels dist/gpu-pack/cuda
  for pkg in nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12; do
    $PY -m pip download "$pkg" --no-deps --only-binary=:all: --python-version 3.10 \
      --platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64 \
      --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 -d dist/gpu-wheels
  done
  for whl in dist/gpu-wheels/*.whl; do
    $PY - "$whl" dist/gpu-pack/cuda <<'EOF'
import sys, zipfile, os
whl, out = sys.argv[1:3]
with zipfile.ZipFile(whl) as z:
    for n in z.namelist():
        b = os.path.basename(n)
        if ".so" in b and "/lib/" in n:
            open(os.path.join(out, b), "wb").write(z.read(n))
EOF
  done
  tar czf dist/LiveTranscriber-linux-gpu-pack.tar.gz -C dist/gpu-pack cuda
  echo "Built dist/LiveTranscriber-linux-gpu-pack.tar.gz"
fi
