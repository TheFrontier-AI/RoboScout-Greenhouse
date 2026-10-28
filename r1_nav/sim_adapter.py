"""Adapter to the R180 Isaac environment (robot_design/r1_v3/diy_r50/physical_ai_r180). Import only after SimulationApp."""
from __future__ import annotations
import math, os, sys
from pathlib import Path


def make_r180_env(simulation_app, r180_dir: str | None = None):
    d = Path(r180_dir or os.environ["R180_DIR"]).resolve()
    sys.path.insert(0, str(d))
    from r180_env import make_env            # noqa: E402  (Isaac modules; needs a running SimulationApp)
    return make_env(simulation_app)


R180_LIDAR_FORWARD_RAD = -math.pi / 2   # R180 scan puts robot forward at -pi/2 (physical_ai_runtime/stop_controller.py)


def observation_to_scan(obs, max_range_m: float = 12.0):
    """(ranges_m, angles_rad) in the controller convention (0 = forward, +pi/2 = left); inf/NaN → max range."""
    ranges = [float(r) if math.isfinite(r) else max_range_m for r in obs["lidar_m"]]
    angles = [(float(a) - R180_LIDAR_FORWARD_RAD + math.pi) % (2 * math.pi) - math.pi for a in obs["lidar_angles_rad"]]
    return ranges, angles
