"""Adapter to the R180 Isaac environment (robot_design/r1_v3/diy_r50/physical_ai_r180). Import only after SimulationApp."""
from __future__ import annotations
import os, sys
from pathlib import Path


def make_r180_env(simulation_app, r180_dir: str | None = None):
    d = Path(r180_dir or os.environ["R180_DIR"]).resolve()
    sys.path.insert(0, str(d))
    from r180_env import make_env            # noqa: E402  (Isaac modules; needs a running SimulationApp)
    return make_env(simulation_app)


def observation_to_scan(obs):
    """Pull (ranges_m, angles_rad) out of the R180 observation dict."""
    return obs["lidar_m"], obs["lidar_angles_rad"]
