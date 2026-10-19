#!/usr/bin/env bash
# Source this file in EVERY new terminal. Set START_ROOT first if using another directory.
export START_ROOT="${START_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# Reuse the platform runtime paths from the existing offline model environment.
export SMOLVLA_BASE_ROOT="${SMOLVLA_BASE_ROOT:-$START_ROOT/../SothisAI/smolvla}"
if [[ -f "$SMOLVLA_BASE_ROOT/env.sh" ]]; then
  source "$SMOLVLA_BASE_ROOT/env.sh"
fi
# Private OSMesa extraction: no system Mesa packages are replaced.
if [[ -d "$START_ROOT/.runtime/osmesa/usr/lib/x86_64-linux-gnu" ]]; then
  export LD_LIBRARY_PATH="$START_ROOT/.runtime/osmesa/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# Configure network only when explicitly supplied for this instance.
if [[ -n "${START_PROXY:-}" ]]; then
  export HTTP_PROXY="$START_PROXY"
  export HTTPS_PROXY="$START_PROXY"
  export http_proxy="$START_PROXY"
  export https_proxy="$START_PROXY"
fi
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_PROGRESS_BAR=off
export HF_HOME="$START_ROOT/cache/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_LEROBOT_HOME="$START_ROOT/cache/lerobot"
export LIBERO_CONFIG_PATH="$START_ROOT/cache/libero-config"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=120
export TOKENIZERS_PARALLELISM=false
# Keep visibility settings supplied by the platform. If none are set, use device 0.
# A vendor DCU torch build still uses the PyTorch 'cuda' API.
if [[ -z "${HIP_VISIBLE_DEVICES:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES=0
fi
export OMP_NUM_THREADS=4
export ACCELERATE_MIXED_PRECISION=no
mkdir -p "$START_ROOT/cache" "$START_ROOT/data" "$START_ROOT/logs"
source "$START_ROOT/.venv/bin/activate"
