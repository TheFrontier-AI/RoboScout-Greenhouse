#!/usr/bin/env bash
# Cloud-side install on a fresh Nebius GPU VM (Ubuntu 24.04 + CUDA image, e.g. 1x RTX PRO 6000, user `ubuntu`).
# Installs the hub + analysis pipeline into this checkout, verifies it with the offline tests, and (optionally)
# installs the systemd service. Re-runnable.
#
#   git clone https://github.com/TheFrontier-AI/RoboScout-Greenhouse.git && cd RoboScout-Greenhouse
#   bash deploy/nebius/install_cloud.sh               # venv + python deps + tests
#   cp deploy/nebius/.env.example deploy/nebius/.env  # then put NEBIUS_API_KEY in it (never commit .env)
#   # fine-tuned detector weights -> vision_runs/rtdetr_v2_mixed/best/{config.json,model.safetensors} (ask the author)
#   bash deploy/nebius/install_cloud.sh --service      # systemd unit `pipeline` (hub :8090 + watcher on ~/runs)
#
# Optional GPU steps (the pipeline skips them when absent): --cuvslam  --lingbot   (see docs/INSTALL.md)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
VENV="${VENV:-$REPO/.venv}"; RUNS_DIR="${RUNS_DIR:-$HOME/runs}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"   # match the VM's driver: nvidia-smi -> CUDA 13.x
WANT_SERVICE=0; WANT_CUVSLAM=0; WANT_LINGBOT=0
for a in "$@"; do case "$a" in --service) WANT_SERVICE=1;; --cuvslam) WANT_CUVSLAM=1;; --lingbot) WANT_LINGBOT=1;; esac; done

echo "== apt packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip ffmpeg rsync curl git >/dev/null
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || echo "no GPU visible (detect/cuvslam/lingbot will run on CPU or be skipped)"

echo "== python venv $VENV"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q torch --index-url "$TORCH_INDEX"
"$VENV/bin/pip" install -q -r requirements/cloud.txt -r requirements/dev.txt
"$VENV/bin/python" - <<'EOF'
import torch, transformers
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  transformers {transformers.__version__}")
EOF

echo "== offline tests"
"$VENV/bin/python" -m pytest -q tests/test_pipeline.py tests/test_robot_api.py tests/test_vision_finding.py tests/test_eyes_parse.py

echo "== runs dir $RUNS_DIR"
mkdir -p "$RUNS_DIR/hub" "$RUNS_DIR/logs"
[ -f deploy/nebius/.env ] || echo "note: deploy/nebius/.env not found - copy .env.example and set NEBIUS_API_KEY (Token Factory) for Nemotron steps"
[ -f vision_runs/rtdetr_v2_mixed/best/model.safetensors ] || echo "note: detector weights missing (vision_runs/rtdetr_v2_mixed/best) - the detect step will be skipped; request them from the author"

if [ "$WANT_CUVSLAM" = 1 ]; then
  echo "== cuVSLAM (NVIDIA wheel, CUDA 12 runtime libs)"
  CV="${CUVSLAM_VENV:-$HOME/cuvslam_venv}"
  [ -x "$CV/bin/python" ] || python3 -m venv "$CV"
  "$CV/bin/pip" install -q --upgrade pip
  "$CV/bin/pip" install -q numpy pillow scipy
  "$CV/bin/pip" install -q cuvslam || echo "pip could not find the cuvslam wheel: download it from https://github.com/nvidia-isaac/cuVSLAM/releases and pip install the .whl into $CV"
  "$CV/bin/python" -c "import cuvslam; print('cuvslam', cuvslam.__version__)" || true
fi

if [ "$WANT_LINGBOT" = 1 ]; then
  echo "== lingbot-map (docker image lingbot:blackwell)"
  bash deploy/nebius/lingbot/build.sh
fi

if [ "$WANT_SERVICE" = 1 ]; then
  echo "== systemd service"
  sed -e "s#__USER__#$USER#g" -e "s#__HOME__#$HOME#g" -e "s#__REPO__#$REPO#g" -e "s#__PY__#$VENV/bin/python#g" \
      -e "s#__RUNS__#$RUNS_DIR#g" deploy/nebius/pipeline.service | sudo tee /etc/systemd/system/pipeline.service >/dev/null
  sudo systemctl daemon-reload && sudo systemctl enable --now pipeline
  sleep 3; systemctl --no-pager --lines=5 status pipeline || true
  curl -s localhost:8090/health && echo
fi
echo "cloud side OK. Next: docs/INSTALL.md section 'Cloud'."
