#!/usr/bin/env bash
# One-time setup of a fresh Nebius AI Cloud VM (Ubuntu, RTX PRO 6000) for the R1 demo. Idempotent; safe to re-run.
#   scp -r r1_greenhouse_nav ubuntu@<vm>:~ && ssh ubuntu@<vm> 'bash r1_greenhouse_nav/deploy/nebius/bootstrap_vm.sh'
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

echo "== GPU"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || { echo "no NVIDIA driver: pick a Nebius GPU image with drivers"; exit 1; }

if ! command -v docker >/dev/null; then
  echo "== installing docker"; curl -fsSL https://get.docker.com | sudo sh; sudo usermod -aG docker "$USER"
fi
if ! sudo docker info 2>/dev/null | grep -qi nvidia; then   # sudo: a fresh user may not be in the docker group yet
  echo "== installing NVIDIA container toolkit"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
fi
sudo docker run --rm --gpus all ubuntu nvidia-smi -L

[ -f "$REPO/deploy/nebius/.env" ] || { cp "$REPO/deploy/nebius/.env.example" "$REPO/deploy/nebius/.env"; echo "!! fill $REPO/deploy/nebius/.env then re-run"; exit 2; }

echo "== NGC login for the Isaac Sim image (needs NGC_API_KEY in the environment)"
[ -n "${NGC_API_KEY:-}" ] && echo "$NGC_API_KEY" | sudo docker login nvcr.io -u '$oauthtoken' --password-stdin || echo "skip: NGC_API_KEY not set"

echo "== building + starting eyes / robot-api / dashboard"
cd "$REPO/deploy/nebius" && sudo docker compose up -d --build
echo "== waiting for Cosmos Reason"; for i in $(seq 1 120); do curl -sf localhost:8000/v1/models >/dev/null && break; sleep 10; done
curl -s localhost:8000/v1/models | head -c 300; echo
curl -s localhost:8090/health; echo
echo "OK. From your laptop: ssh -L 8080:localhost:8080 -L 8090:localhost:8090 ubuntu@<vm>  → http://localhost:8080"
