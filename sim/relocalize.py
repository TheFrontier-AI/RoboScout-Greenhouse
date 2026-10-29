"""Global relocalisation on the taught map: "where am I?" with no starting guess.

The robot may be put down anywhere on a path it was led along (scripts/teach_map.py). Hypotheses are every
`spacing_m` along the taught path, in the taught heading and the reverse (and a few heading offsets); each is scored by
the share of the scan's points that land within `hit_m` of a map point, the best few are refined with the same ICP the
robot localises with (ScanToMapLocalizer), and the answer carries an ambiguity flag: two identical greenhouses can
match a scan equally well, and then the honest output is "ambiguous between A and B", not a guess.

    reloc = Relocalizer(map_xy, path_xyh)           # map and taught path from teach_map.py
    r = reloc.locate(scan_xy)                        # scan in the LiDAR frame (x forward, y left)
    r.pose, r.score, r.ambiguous, r.alternatives
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import cKDTree

from sim.scan_localizer import LIDAR_AHEAD_M, ScanToMapLocalizer


def _rot(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


@dataclass
class Relocation:
    pose: tuple[float, float, float] | None          # (x, y, heading_deg) in the farm frame, None = not found
    score: float                                      # share of scan points on the map after refinement (0..1)
    rmse_m: float
    ambiguous: bool                                   # another far-away place matches almost as well
    alternatives: list[dict] = field(default_factory=list)   # [{"pose", "score", "dist_m"}] the runner-up places
    hypotheses: int = 0


class Relocalizer:
    def __init__(self, map_xy: np.ndarray, path_xyh: np.ndarray, spacing_m: float = 0.5,
                 heading_offsets_deg=(0.0, 180.0, 15.0, -15.0, 195.0, 165.0), hit_m: float = 0.25,
                 refine_top: int = 12, ambiguity_margin: float = 0.02, same_place_m: float = 3.0):
        self.map = np.asarray(map_xy, float); self.tree = cKDTree(self.map)
        self.hit_m, self.refine_top, self.margin, self.same_place_m = hit_m, refine_top, ambiguity_margin, same_place_m
        self.loc = ScanToMapLocalizer(self.map, max_pair_m=0.3, iters=25, max_jump_m=3.0, max_jump_deg=60.0)
        # hypotheses: the taught path resampled every spacing_m, each with the taught heading + offsets
        p = np.asarray(path_xyh, float)
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))]
        marks = np.arange(0.0, d[-1], spacing_m)
        xs = np.interp(marks, d, p[:, 0]); ys = np.interp(marks, d, p[:, 1])
        hs = np.degrees(np.unwrap(np.radians(p[:, 2]))); hs = np.interp(marks, d, hs)
        self.hyp = np.array([(x, y, (h + off + 180) % 360 - 180) for x, y, h in zip(xs, ys, hs) for off in heading_offsets_deg])

    def _score(self, scan_body: np.ndarray, x: float, y: float, h_deg: float) -> float:
        pts = scan_body @ _rot(math.radians(h_deg)).T + (x, y)
        d, _ = self.tree.query(pts, distance_upper_bound=self.hit_m)
        return float(np.isfinite(d).mean())

    def locate(self, scan_xy: np.ndarray, max_range_m: float = 11.0, subsample: int = 4) -> Relocation:
        scan = np.asarray(scan_xy, float)
        scan = scan[np.linalg.norm(scan, axis=1) < max_range_m]
        if len(scan) < 40:
            return Relocation(None, 0.0, 0.0, False, hypotheses=len(self.hyp))
        body = scan + (LIDAR_AHEAD_M, 0.0)
        coarse = body[::subsample]
        scores = np.array([self._score(coarse, *h) for h in self.hyp])
        order = np.argsort(-scores)[: self.refine_top]
        refined = []
        for i in order:
            x, y, h = self.hyp[i]
            (fx, fy, fh), q = self.loc.localize(scan, x, y, h)
            s = self._score(body, fx, fy, fh)
            refined.append((s, float(q.get("rmse_m", 1.0) or 1.0), fx, fy, fh))
        refined.sort(key=lambda r: -r[0])
        best = refined[0]
        alts = []
        for s, rm, fx, fy, fh in refined[1:]:
            dist = math.hypot(fx - best[2], fy - best[3])
            if dist > self.same_place_m and s >= best[0] - self.margin:
                alts.append({"pose": (round(fx, 2), round(fy, 2), round(fh, 1)), "score": round(s, 3), "dist_m": round(dist, 1)})
        return Relocation((round(best[2], 3), round(best[3], 3), round(best[4], 2)), round(best[0], 3), round(best[1], 4),
                          bool(alts), alts, hypotheses=len(self.hyp))


@dataclass
class Track:
    pose: tuple[float, float, float] | None
    decided: bool                      # one hypothesis dominates
    metres: float                      # distance driven while deciding
    steps: int
    survivors: int
    margin: float                      # score gap between the best and the runner-up far-away hypothesis
    history: list[dict] = field(default_factory=list)


class HypothesisTracker:
    """Keep the ambiguous candidates alive and let motion decide. Start from one scan's candidates (Relocalizer),
    then for every new scan: move each hypothesis by the robot's own odometry delta, re-refine it on the map, score it,
    accumulate; drop hypotheses that stop matching; stop when the best beats every far-away rival by `decide_margin`.
    This is how the robot, put down on a known path, finds out which greenhouse it is in by driving a little."""

    def __init__(self, reloc: Relocalizer, keep: int = 8, decide_margin: float = 0.12, drop_below: float = 0.35,
                 same_place_m: float = 3.0):
        self.reloc, self.keep, self.decide_margin, self.drop_below, self.same_place_m = reloc, keep, decide_margin, drop_below, same_place_m
        self.hyps: list[list] = []          # [x, y, h, cumulative score, n]
        self.last_odom = None
        self.metres = 0.0; self.steps = 0

    def start(self, scan_xy: np.ndarray, odom_xyh) -> None:
        r = self.reloc.locate(scan_xy)
        cands = [r.pose] + [a["pose"] for a in r.alternatives] if r.pose else []
        body = np.asarray(scan_xy, float) + (LIDAR_AHEAD_M, 0.0)
        # also keep the best coarse hypotheses so a near-miss is not lost before motion can judge it
        scores = np.array([self.reloc._score(body[::4], *h) for h in self.reloc.hyp])
        for i in np.argsort(-scores)[: self.keep * 2]:
            x, y, h = self.reloc.hyp[i]
            if all(math.hypot(x - c[0], y - c[1]) > self.same_place_m for c in cands):
                cands.append((float(x), float(y), float(h)))
        self.hyps = [[c[0], c[1], c[2], self.reloc._score(body, *c), 1] for c in cands[: self.keep]]
        self.last_odom = tuple(odom_xyh); self.metres = 0.0; self.steps = 0

    def update(self, scan_xy: np.ndarray, odom_xyh) -> Track:
        ox, oy, oh = odom_xyh; lx, ly, lh = self.last_odom
        dh = oh - lh; dx, dy = ox - lx, oy - ly
        # odometry delta expressed in the old body frame, then applied in each hypothesis' frame
        c, s = math.cos(math.radians(-lh)), math.sin(math.radians(-lh))
        bx, by = c * dx - s * dy, s * dx + c * dy
        self.metres += math.hypot(dx, dy); self.steps += 1; self.last_odom = (ox, oy, oh)
        scan = np.asarray(scan_xy, float); body = scan + (LIDAR_AHEAD_M, 0.0)
        new = []
        for x, y, h, acc, n in self.hyps:
            hr = math.radians(h); nx = x + math.cos(hr) * bx - math.sin(hr) * by; ny = y + math.sin(hr) * bx + math.cos(hr) * by; nh = h + dh
            (fx, fy, fh), q = self.reloc.loc.localize(scan, nx, ny, nh)
            sc = self.reloc._score(body, fx, fy, fh)
            if sc < self.drop_below and n > 3:
                continue
            new.append([fx, fy, fh, acc + sc, n + 1])
        # merge hypotheses that converged to the same place
        new.sort(key=lambda r: -r[3] / r[4]); merged = []
        for r in new:
            if all(math.hypot(r[0] - m[0], r[1] - m[1]) > self.same_place_m for m in merged):
                merged.append(r)
        self.hyps = merged[: self.keep] or new[:1]
        best = self.hyps[0]; best_mean = best[3] / best[4]
        rivals = [r[3] / r[4] for r in self.hyps[1:]]
        margin = best_mean - max(rivals) if rivals else 1.0
        decided = len(self.hyps) == 1 or margin >= self.decide_margin
        return Track((round(best[0], 3), round(best[1], 3), round(best[2], 2)), decided, round(self.metres, 2), self.steps,
                     len(self.hyps), round(margin, 3))
