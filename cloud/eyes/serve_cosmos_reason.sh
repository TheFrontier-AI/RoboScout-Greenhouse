#!/usr/bin/env bash
# vLLM OpenAI-compatible server for NVIDIA Cosmos Reason 2 on one RTX PRO 6000 Blackwell (96 GB).
# Cosmos-Reason2-32B in BF16 is ~66 GB of weights, which leaves ~20 GB for KV cache at 0.92 utilisation.
# For more frames/s use COSMOS_MODEL=nvidia/Cosmos-Reason2-8B (gated on HF: accept the terms and set HF_TOKEN).
# Env overrides: COSMOS_MODEL, MAX_MODEL_LEN, GPU_MEM_UTIL, PORT, VLLM_API_KEY (optional bearer key).
set -euo pipefail

MODEL="${COSMOS_MODEL:-nvidia/Cosmos-Reason2-32B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"   # one full-res photo can be >10k vision tokens; 4k+ reserved for reasoning
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
PORT="${PORT:-8000}"

args=(
  serve "$MODEL"
  --served-model-name "$MODEL"
  --host 0.0.0.0 --port "$PORT"
  --dtype bfloat16
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-num-seqs 16
  --limit-mm-per-prompt '{"image": 1, "video": 0}'
  --media-io-kwargs '{"video": {"num_frames": -1}}'
  --reasoning-parser qwen3
)
[[ -n "${VLLM_API_KEY:-}" ]] && args+=(--api-key "$VLLM_API_KEY")

exec vllm "${args[@]}" "$@"
