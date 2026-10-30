"""LiDAR odometry for the aisle: scan-to-keyframe ICP on the planar C1 scan, fused with wheel + gyro odometry.

Why: wheel odometry over-reads on a wet floor (slip) and nothing on the wheels can tell. Plants and posts along the
beds give the scan structure along the aisle, so matching consecutive scans measures how far the robot really moved.
A bare corridor (two smooth walls) is degenerate along its axis; then ICP is not trusted and the wheels carry on.
Pure numpy/scipy, runs every control tick on the robot. Same PoseSource interface as executor.WheelImuOdometry."""
from __future__ import annotations
import math
import numpy as np
from scipy.spatial import cKDTree
from sim.executor import INTO_AISLE_HEADING_DEG, R180_LIDAR_FORWARD_RAD, WheelImuOdometry, scan_from_obs
from sim.scan_localizer import LIDAR_AHEAD_M

BODY_SELF_M = 0.6      # LiDAR returns closer than this to the body origin are the robot itself (mast, electronics box)


def scan_points(obs, max_range_m: float = 8.0, self_m: float = 0.25) -> np.ndarray:
    """Planar scan → (N, 2) points in the LiDAR frame (x forward, y left)."""
    r, a = scan_from_obs(obs, R180_LIDAR_FORWARD_RAD)
    r, a = np.asarray(r), np.asarray(a)
    keep = (r > self_m) & (r < max_range_m)
    pts = np.stack([r[keep] * np.cos(a[keep]), r[keep] * np.sin(a[keep])], -1)
    body = pts + (LIDAR_AHEAD_M, 0.0)                     # returns from the robot's own mast / box sit 0.3-0.6 m behind the LiDAR
    return pts[np.hypot(body[:, 0], body[:, 1]) >= BODY_SELF_M]


def icp(src: np.ndarray, dst: np.ndarray, init=(0.0, 0.0, 0.0), iters: int = 20, max_pair_m: float = 0.25):
    """Point-to-point ICP, src → dst. Returns ((dx, dy, dth), inlier fraction, along-x constraint strength).
    The constraint is the share of matched pairs whose dst neighbourhood is not a line parallel to x (walls)."""
    tree = cKDTree(dst)
    dx, dy, th = init
    frac, strength = 0.0, 0.0
    for _ in range(iters):
        c, s = math.cos(th), math.sin(th)
        p = src @ np.array([[c, s], [-s, c]]) + (dx, dy)
        d, idx = tree.query(p, distance_upper_bound=max_pair_m)
        ok = np.isfinite(d)
        frac = ok.mean() if len(ok) else 0.0
        if ok.sum() < 20:
            return (dx, dy, th), frac, 0.0
        a, b = src[ok], dst[idx[ok]]
        ma, mb = a.mean(0), b.mean(0)
        H = (a - ma).T @ (b - mb)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[1] *= -1; R = Vt.T @ U.T
        th = math.atan2(R[1, 0], R[0, 0])
        dx, dy = mb - R @ ma
    # along-x observability: local direction of dst around each match; wall points run along x and say nothing
    _, nn = tree.query(dst[idx[ok]], k=min(6, len(dst)))
    dirs = dst[nn] - dst[nn].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", dirs, dirs)
    w, v = np.linalg.eigh(cov)
    major = v[:, :, 1]                                     # local line direction
    strength = float(np.mean(np.abs(major[:, 0]) < math.cos(math.radians(30))))  # not parallel to the aisle
    return (float(dx), float(dy), float(th)), float(frac), strength


class LidarWheelOdometry:
    """Along-aisle metre from LiDAR scan matching when the scan constrains it, else from the wheels.
    Emits slip warnings (wheel vs LiDAR disagreement) in `events`; heading still from the gyro."""

    def __init__(self, metre: float = 0.0, heading_deg: float = INTO_AISLE_HEADING_DEG, keyframe_m: float = 1.0,
                 min_inliers: float = 0.5, min_strength: float = 0.08, slip_ratio: float = 0.07, **wheel_kw):
        self.wheel = WheelImuOdometry(metre, heading_deg, **wheel_kw)
        self.keyframe_m, self.min_inliers, self.min_strength, self.slip_ratio = keyframe_m, min_inliers, min_strength, slip_ratio
        self.reset(metre, heading_deg)

    def reset(self, metre: float, heading_deg: float | None = None) -> None:
        self.wheel.reset(metre, heading_deg)
        self.metre = float(metre)
        self.kf_pts = None
        self.kf_metre = self.kf_wheel = self.metre
        self.lidar_used = self.lidar_total = 0
        self.events: list[dict] = []

    def correct(self, metre: float) -> None:
        self.metre = float(metre); self.wheel.correct(metre)
        self.kf_pts = None

    @property
    def heading(self) -> float:
        return self.wheel.heading

    def set_heading(self, heading_deg: float) -> None:
        self.wheel.set_heading(heading_deg)

    def update(self, obs):
        prev_wheel = self.wheel.metre
        w_metre, heading, w_conf = self.wheel.update(obs)
        wheel_step = w_metre - prev_wheel
        pts = scan_points(obs)
        if self.kf_pts is None or len(pts) < 30:
            self.metre += wheel_step
            if len(pts) >= 30:
                self.kf_pts, self.kf_metre, self.kf_wheel = pts, self.metre, w_metre
            return self.metre, heading, w_conf
        # robot moves along its own x; metre sign follows the heading relative to the aisle axis
        sign = math.cos(math.radians(heading - INTO_AISLE_HEADING_DEG))
        guess = (w_metre - self.kf_wheel) * sign
        (dx, _, _), frac, strength = icp(pts, self.kf_pts, init=(guess, 0.0, 0.0))
        self.lidar_total += 1
        if frac >= self.min_inliers and strength >= self.min_strength:
            self.lidar_used += 1
            self.metre, conf = self.kf_metre + dx * sign, max(w_conf, 0.9)
        else:
            self.metre += wheel_step
            conf = w_conf
        if abs(self.metre - self.kf_metre) >= self.keyframe_m:
            # slip check over the whole keyframe window (one metre), not per tick: per-tick ICP noise is not slip
            wheel_moved, lidar_moved = w_metre - self.kf_wheel, self.metre - self.kf_metre
            if abs(wheel_moved - lidar_moved) > max(0.04, self.slip_ratio * abs(wheel_moved)):
                self.events.append({"type": "slip", "wheel_m": round(wheel_moved, 3), "lidar_m": round(lidar_moved, 3),
                                    "metre": round(self.metre, 2), "t": float(obs["timestamp_s"])})
            self.kf_pts, self.kf_metre, self.kf_wheel = pts, self.metre, w_metre
        return self.metre, heading, conf
