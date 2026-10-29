#!/usr/bin/env bash
# Robot-side install (Jetson Orin / any ARM64 or x86 Linux with Python >= 3.10). No GPU needed for the navigation
# stack, the recorder and the cloud hand-off; cuVSLAM (optional, NVIDIA wheel) is the only GPU piece on the robot.
#
#   git clone https://github.com/TheFrontier-AI/RoboScout-Greenhouse.git && cd RoboScout-Greenhouse
#   bash deploy/robot/install_robot.sh            # creates .venv-robot, installs requirements/robot.txt, runs the tests
#
# Afterwards (see docs/INSTALL.md):
#   source .venv-robot/bin/activate
#   export ROBOT_API_URL=http://<server>:8090
#   python3 scripts/robot_upload_run.py --rec <run>/rec --dest ubuntu@<server>:/home/ubuntu/runs/<run>/rec --hub $ROBOT_API_URL --wait
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
VENV="${VENV:-$REPO/.venv-robot}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1 (sudo apt-get install -y $2)"; exit 1; }; }
need python3 python3; need rsync rsync; need ssh openssh-client
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 required"'

if [ ! -x "$VENV/bin/pip" ]; then
  rm -rf "$VENV"
  if ! python3 -m venv "$VENV" 2>/dev/null; then
    # JetPack images ship python3 without ensurepip (no python3-venv package, no passwordless sudo):
    # make the venv without pip and bootstrap pip from PyPA instead
    rm -rf "$VENV"; python3 -m venv --without-pip "$VENV"
    python3 -c 'import urllib.request; urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", "/tmp/get-pip.py")'
    "$VENV/bin/python" /tmp/get-pip.py -q
  fi
fi
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r requirements/robot.txt -r requirements/dev.txt
echo "venv: $VENV  ($("$VENV/bin/python" --version), $(uname -m))"

# offline self-test of the navigation stack, hub client and recorder (no simulator, no network)
"$VENV/bin/python" -m pytest -q tests/test_commands.py tests/test_controller.py tests/test_mission.py \
  tests/test_sim_cloud_link.py tests/test_sim_executor.py tests/test_sim_geometry.py \
  tests/test_lidar_odometry.py tests/test_scan_localizer.py tests/test_robot_api.py
echo "robot side OK. Next: docs/INSTALL.md section 'Robot'."
