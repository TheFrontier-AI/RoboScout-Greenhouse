#!/usr/bin/env bash
# Pipeline hook (cloud/pipeline/steps.py step_lingbot): run lingbot-map on a folder of front-camera JPEGs and write
#   <prefix>_poses.json  <prefix>_small.ply  <prefix>_timing.json   (+ <prefix>.npz raw predictions)
# usage: run.sh <image_folder> <out_prefix> [gpu_index]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SEQ="$(cd "$1" && pwd)"; PREFIX="$2"; GPU="${3:-0}"
OUT_DIR="$(cd "$(dirname "$PREFIX")" && pwd)"; NAME="$(basename "$PREFIX")"
IMAGE="${LINGBOT_IMAGE:-lingbot:blackwell}"; MODE="${LINGBOT_MODE:-windowed}"
REPO_ID=robbyant/lingbot-map; CACHE="$HERE/hf_cache"; mkdir -p "$CACHE"
DOCKER=docker; docker info >/dev/null 2>&1 || DOCKER="sudo docker"
# the images folder may be symlinks into the run dir: mount the run dir too so they resolve inside the container
RUN_ROOT="$(cd "$SEQ/../.." && pwd)"
# weights: downloaded once into hf_cache by huggingface_hub inside the container
MODEL="$(ls "$CACHE"/hub/models--robbyant--lingbot-map/snapshots/*/lingbot-map.pt 2>/dev/null | head -1 || true)"
if [ -z "$MODEL" ]; then
  $DOCKER run --rm -v "$CACHE:/hf_cache" -e HF_HOME=/hf_cache "$IMAGE" \
    python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('$REPO_ID', 'lingbot-map.pt'))"
  MODEL="$(ls "$CACHE"/hub/models--robbyant--lingbot-map/snapshots/*/lingbot-map.pt | head -1)"
fi
MODEL_IN="/hf_cache${MODEL#$CACHE}"
T0=$(date +%s.%N)
$DOCKER run --rm --gpus "device=$GPU" -v "$CACHE:/hf_cache" -v "$HERE:/work:ro" -v "$RUN_ROOT:$RUN_ROOT" -v "$OUT_DIR:/out" \
  -e HF_HOME=/hf_cache "$IMAGE" python /work/run_lingbot.py --image_folder "$SEQ" --out "/out/$NAME.npz" \
  --model_path "$MODEL_IN" --mode "$MODE" --window_size "${LINGBOT_WINDOW:-128}" --overlap_size 16 --overlap_keyframes 16
WALL=$(echo "$(date +%s.%N) - $T0" | bc)
PY="${LINGBOT_PY:-python3}"
"$PY" "$HERE/lingbot_to_route.py" "$OUT_DIR/$NAME.npz" "$PREFIX" --wall-s "$WALL" --mode "$MODE" --gpu "$(nvidia-smi --id="$GPU" --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
