#!/usr/bin/env bash
# Build the lingbot-map docker image used by the pipeline's `lingbot` step (camera-only 3-D map of the front camera).
# lingbot-map: https://github.com/robbyant/lingbot-map (Apache-2.0). The image targets Blackwell (sm_120, RTX PRO 6000);
# set TORCH_CUDA_ARCH_LIST for other GPUs (e.g. "9.0" for H100). Needs docker + NVIDIA container toolkit.
#   bash deploy/nebius/lingbot/build.sh          # ~10 min; image: lingbot:blackwell (LINGBOT_IMAGE)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
IMAGE="${LINGBOT_IMAGE:-lingbot:blackwell}"
[ -d lingbot-map ] || git clone --depth 1 https://github.com/robbyant/lingbot-map.git
DOCKER=docker; docker info >/dev/null 2>&1 || DOCKER="sudo docker"
$DOCKER build -t "$IMAGE" --build-arg TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" .
mkdir -p hf_cache
echo "built $IMAGE; weights (robbyant/lingbot-map, ~1 GB) download into $HERE/hf_cache on the first run"
