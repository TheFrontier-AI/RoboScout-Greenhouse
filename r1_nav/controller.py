"""Sensor-only aisle controller: keep the robot centred between the two bed walls using the planar LiDAR,
stop for anything closer than `stop_range_m` ahead. Pure functions; no simulator state."""
from __future__ import annotations
import math
from dataclasses import dataclass

WHEEL_RADIUS_M = 0.127
MAX_WHEEL_RAD_S = 2.0          # env action bound


@dataclass
class ControllerConfig:
    cruise_m_s: float = 0.25
    reverse_m_s: float = 0.15
    stop_range_m: float = 0.6
    front_half_angle_rad: float = math.radians(20)
    side_centre_rad: float = math.radians(90)
    side_half_angle_rad: float = math.radians(25)
    k_lateral: float = 1.2     # rad/s of wheel-speed difference per metre of lateral error
    max_diff_rad_s: float = 0.6


def sector_min(ranges, angles, centre, half, no_return_value=12.0):
    best = no_return_value
    for r, a in zip(ranges, angles):
        d = (a - centre + math.pi) % (2 * math.pi) - math.pi
        if abs(d) <= half and r < best:
            best = r
    return best


def wheel_command(ranges, angles, cfg: ControllerConfig = ControllerConfig(), reverse: bool = False):
    """Return ([left, right] wheel rad/s, info). Angles: 0 = forward, +pi/2 = left (robot frame)."""
    front = sector_min(ranges, angles, 0.0, cfg.front_half_angle_rad)
    left = sector_min(ranges, angles, cfg.side_centre_rad, cfg.side_half_angle_rad)
    right = sector_min(ranges, angles, -cfg.side_centre_rad, cfg.side_half_angle_rad)
    info = {"front_m": front, "left_m": left, "right_m": right, "stopped": False}
    if not reverse and front < cfg.stop_range_m:
        info["stopped"] = True
        return [0.0, 0.0], info
    speed = (cfg.reverse_m_s if reverse else cfg.cruise_m_s) / WHEEL_RADIUS_M
    lateral_error = left - right                      # >0: more room on the left -> steer left
    diff = max(-cfg.max_diff_rad_s, min(cfg.max_diff_rad_s, cfg.k_lateral * lateral_error))
    if reverse:
        speed, diff = -speed, -diff
    l, r = speed - diff / 2, speed + diff / 2
    scale = max(1.0, abs(l) / MAX_WHEEL_RAD_S, abs(r) / MAX_WHEEL_RAD_S)
    return [l / scale, r / scale], info
