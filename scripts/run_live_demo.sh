#!/usr/bin/env bash
# Full live loop on one workstation: robot hub + dashboard here, R1 in local Isaac Sim, Cosmos Reason on the Nebius VM
# (SSH tunnel), HERMES (Nemotron on Nebius Token Factory). Open http://127.0.0.1:8080 to watch / screen-record.
#   scripts/run_live_demo.sh <run_dir> [survey_until_m=18] [disease_metre=15]
# Needs: deploy/nebius/.env (NEBIUS_API_KEY), ssh access to the VM (VM_HOST), ISAAC_PY pointing at Isaac's python.sh.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
RUN="${1:?run dir}"; UNTIL="${2:-18}"; DISEASE="${3:-15}"
VM_HOST="${VM_HOST:-ubuntu@204.12.163.30}"
ISAAC_PY="${ISAAC_PY:-$HOME/Documents/isaac-sim/IsaacSim/_build/linux-aarch64/release/python.sh}"
mkdir -p "$RUN/data"
set -a; . deploy/nebius/.env; set +a
export EYES_BASE_URL=http://127.0.0.1:18000/v1 EYES_MODEL="${EYES_MODEL:-nvidia/Cosmos-Reason2-8B}" EYES_API_KEY=x
export ROBOT_API_URL=http://127.0.0.1:8090 PYTHONUNBUFFERED=1

curl -sf --max-time 5 localhost:18000/v1/models >/dev/null || \
  ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 18000:127.0.0.1:8000 "$VM_HOST"
curl -sf --max-time 20 localhost:18000/v1/models >/dev/null || { echo "Cosmos endpoint not reachable via tunnel"; exit 1; }

python3 agent/robot_api.py --data "$RUN/data" --map data/tunnel09_aisle_map_v18.json --port 8090 > "$RUN/api.log" 2>&1 &
for _ in $(seq 20); do curl -sf localhost:8090/health >/dev/null && break; sleep 0.5; done
python3 dashboard/serve.py --data "$RUN/data" --port 8080 > "$RUN/dash.log" 2>&1 &
"$ISAAC_PY" -u scripts/run_demo_sim.py --backend isaac --api-url "$ROBOT_API_URL" --disease-metre "$DISEASE" \
  --record "$RUN/rec" --record-every 2 --outbox "$RUN/outbox" --idle-exit-s 600 > "$RUN/sim.log" 2>&1 &
for _ in $(seq 120); do grep -q '^\[demo.*backend=isaac' "$RUN/sim.log" 2>/dev/null && break; sleep 2; done
python3 agent/brain_loop.py --house tunnel-09 --poll 5 --survey-until "$UNTIL" > "$RUN/brain.log" 2>&1 &
echo "running: dashboard http://127.0.0.1:8080  logs in $RUN  (report at the end of $RUN/brain.log)"
