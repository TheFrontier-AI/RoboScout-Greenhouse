"""Warehouse-style localization: match each planar LiDAR scan against a prior 2-D map of the farm (scan-to-map ICP).

The map is a point set at LiDAR height (walls, posts, plants, door frames) in the scene frame. On the real robot it
comes from a mapping run (SLAM) or the farm twin; in the sim it is built once from the twin's collision geometry
(scripts/run_farm.py: build_prior_map). Odometry (wheels + gyro) gives the guess; the scan fixes x, y and heading,
so skid-steer scrub and gyro drift do not accumulate. Pure numpy/scipy; no simulator truth."""
from __future__ import annotations
import math
import numpy as np
from scipy.spatial import cKDTree

LIDAR_AHEAD_M = 0.34


def _rot(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


class ScanToMapLocalizer:
    def __init__(self, map_xy: np.ndarray, max_pair_m: float = 0.3, iters: int = 20, min_inliers: float = 0.55,
                 max_rmse_m: float = 0.12, max_jump_m: float = 1.2, max_jump_deg: float = 25.0):
        self.map = np.asarray(map_xy, float)
        self.tree = cKDTree(self.map)
        self.max_pair, self.iters, self.min_inliers, self.max_rmse = max_pair_m, iters, min_inliers, max_rmse_m
        self.max_jump, self.max_jump_deg = max_jump_m, max_jump_deg

    def localize(self, scan_xy: np.ndarray, x: float, y: float, heading_deg: float):
        """scan_xy: (N,2) in the LiDAR frame (x forward, y left). Pose guess in the scene frame (heading 0 = +X, CCW).
        Returns ((x, y, heading_deg), quality dict) — the pose is the guess when the match is not trusted."""
        if len(scan_xy) < 40:
            return (x, y, heading_deg), {"ok": False, "why": "few points"}
        h = math.radians(heading_deg)
        body = scan_xy + (LIDAR_AHEAD_M, 0.0)
        R_acc, t_acc = np.eye(2), np.zeros(2)
        pts = body @ _rot(h).T + (x, y)
        frac = rmse = 0.0
        for i in range(self.iters):
            pair = max(self.max_pair, 1.0 * 0.75 ** i)          # coarse to fine: tolerate a large odometry error
            d, idx = self.tree.query(pts, distance_upper_bound=pair)
            ok = np.isfinite(d)
            frac = float(ok.mean())
            if ok.sum() < 30:
                return (x, y, heading_deg), {"ok": False, "why": "no overlap", "inliers": frac}
            a, b = pts[ok], self.map[idx[ok]]
            ma, mb = a.mean(0), b.mean(0)
            U, _, Vt = np.linalg.svd((a - ma).T @ (b - mb))
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[1] *= -1; R = Vt.T @ U.T
            t = mb - R @ ma
            pts = pts @ R.T + t
            R_acc, t_acc = R @ R_acc, R @ t_acc + t
            rmse = float(np.sqrt(np.mean(d[ok] ** 2)))
            if pair <= self.max_pair and np.hypot(*t) < 1e-4 and abs(math.atan2(R[1, 0], R[0, 0])) < 1e-5:
                break
        nx, ny = R_acc @ np.array([x, y]) + t_acc
        dth = math.degrees(math.atan2(R_acc[1, 0], R_acc[0, 0]))
        q = {"inliers": round(frac, 3), "rmse_m": round(rmse, 4), "jump_m": round(math.hypot(nx - x, ny - y), 3),
             "jump_deg": round(dth, 2)}
        q["ok"] = (frac >= self.min_inliers and rmse <= self.max_rmse and q["jump_m"] <= self.max_jump
                   and abs(dth) <= self.max_jump_deg)
        if not q["ok"]:
            return (x, y, heading_deg), q
        return (float(nx), float(ny), heading_deg + dth), q
