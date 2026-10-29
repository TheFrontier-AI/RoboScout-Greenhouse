#!/usr/bin/env bash
# One system on the Nebius server: the robot hub (agent/robot_api.py: frames, status, POST /run_complete) and the
# analysis pipeline watcher (python3 -m cloud.pipeline watch) that runs every analysis on each finished run and writes
# <run>/analysis/report.json + report_th.md + report_en.md. Started by pipeline.service; can also be run by hand:
#   deploy/nebius/start_pipeline.sh          env: RUNS_DIR (~/runs)  PY (~/venv2/bin/python)  HUB_PORT (8090)
#                                                 PIPELINE_AUTO (0: the watcher does the work; 1: the hub also spawns a run)
# Keys come from deploy/nebius/.env (NEBIUS_API_KEY, optional EYES_BASE_URL); never put them on the command line.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
RUNS_DIR="${RUNS_DIR:-$HOME/runs}"; HUB_PORT="${HUB_PORT:-8090}"
PY="${PY:-$([ -x "$REPO/.venv/bin/python" ] && echo "$REPO/.venv/bin/python" || echo "$HOME/venv2/bin/python")}"
[ -f deploy/nebius/.env ] && { set -a; . deploy/nebius/.env; set +a; }
export PYTHONUNBUFFERED=1 PYTHONPATH="$REPO"
export DETECTOR_MODEL="${DETECTOR_MODEL:-vision_runs/rtdetr_v2_mixed/best}"
export EYES2_BASE_URL="${EYES2_BASE_URL:-https://api.tokenfactory.nebius.com/v1/}" EYES2_MODEL="${EYES2_MODEL:-nvidia/Nemotron-3.5-Super-VL}"
export CUVSLAM_PYTHON="${CUVSLAM_PYTHON:-$HOME/cuvslam_venv/bin/python}" LINGBOT_RUN="${LINGBOT_RUN:-$REPO/deploy/nebius/lingbot/run.sh}"
export PIPELINE_AUTO="${PIPELINE_AUTO:-0}" PIPELINE_RUNS_DIR="$RUNS_DIR" PIPELINE_VLM_WORKERS="${PIPELINE_VLM_WORKERS:-4}"
mkdir -p "$RUNS_DIR/hub" "$RUNS_DIR/logs"

"$PY" agent/robot_api.py --host 0.0.0.0 --port "$HUB_PORT" --data "$RUNS_DIR/hub" --map data/tunnel09_aisle_map_v18.json \
  >> "$RUNS_DIR/logs/hub.log" 2>&1 &
HUB=$!; trap 'kill $HUB 2>/dev/null || true' EXIT
for _ in $(seq 20); do curl -sf "localhost:$HUB_PORT/health" >/dev/null && break; sleep 0.5; done
echo "hub on :$HUB_PORT (data $RUNS_DIR/hub)  pipeline watching $RUNS_DIR  logs in $RUNS_DIR/logs"
"$PY" -m cloud.pipeline watch --runs-dir "$RUNS_DIR" --poll "${PIPELINE_POLL_S:-30}" 2>&1 | tee -a "$RUNS_DIR/logs/pipeline.log"
