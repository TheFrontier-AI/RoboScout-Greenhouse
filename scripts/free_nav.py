#!/usr/bin/env python3
"""Navigation from the robot's own map, with no stored line to follow.

The robot has a LiDAR point map it built itself and the scans it took while building it. From these alone:
  1. FREE SPACE  - a 0.25 m grid; a cell is drivable when a LiDAR ray has passed through it (so the robot has SEEN it
     empty) and it is farther than the robot's half-width from every map point. Nothing it has not seen is drivable:
     ponds, orchard and road stay out without anyone drawing a boundary.
  2. STRUCTURE   - the map's dominant wall direction is found from the points themselves; long narrow free corridors
     between parallel walls are the aisles (a plant row on each side); the robot's start pose is the dock.
  3. PLANNING    - A* over the free grid with a clearance cost, from any drivable cell to any other: the path is made
     when it is needed, from where the robot believes it is, and remade when the belief changes.
  4. MISSION     - survey = traverse every aisle end to end (nearest first), then return to the dock.

  python3 scripts/free_nav.py --scans scans_all.jsonl --rec <run>/rec --out <dir>     # builds, saves, draws"""
from __future__ import annotations
import argparse, heapq, json, math, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.teach_map import beam_angles, scan_xy, to_scene       # noqa: E402
from sim.scan_localizer import LIDAR_AHEAD_M                        # noqa: E402

CELL, ROBOT_HALF_M, AISLE_W = 0.25, 0.35, (0.8, 2.6)


class FreeSpace:
    """Grid of cells the robot has seen empty and can fit through."""

    def __init__(self, map_xy: np.ndarray, scan_poses, scans, cell=CELL, clearance=ROBOT_HALF_M, max_range=10.0):
        self.cell = cell; self.map = map_xy; self.tree = cKDTree(map_xy)
        lo = map_xy.min(0) - 2.0; hi = map_xy.max(0) + 2.0
        self.x0, self.y0 = float(lo[0]), float(lo[1]); self.nx = int((hi[0] - lo[0]) / cell) + 1; self.ny = int((hi[1] - lo[1]) / cell) + 1
        seen = np.zeros((self.ny, self.nx), bool); ang = beam_angles(500)
        for (x, y, h), ranges in zip(scan_poses, scans):               # ray-cast every beam: the cells it crossed were empty
            pts = scan_xy(ranges, ang, max_range); hits = to_scene(pts, x, y, h)
            hr = math.radians(h); ox, oy = x + math.cos(hr) * LIDAR_AHEAD_M, y + math.sin(hr) * LIDAR_AHEAD_M
            for hx, hy in hits:
                n = int(math.hypot(hx - ox, hy - oy) / (cell * 0.5)) + 1
                xs = np.linspace(ox, hx, n)[:-1]; ys = np.linspace(oy, hy, n)[:-1]
                i = ((ys - self.y0) / cell).astype(int); j = ((xs - self.x0) / cell).astype(int)
                ok = (i >= 0) & (i < self.ny) & (j >= 0) & (j < self.nx); seen[i[ok], j[ok]] = True
        ys, xs = np.mgrid[0:self.ny, 0:self.nx]; cx = self.x0 + (xs + 0.5) * cell; cy = self.y0 + (ys + 0.5) * cell
        d, _ = self.tree.query(np.stack([cx.ravel(), cy.ravel()], -1)); self.dist = d.reshape(self.ny, self.nx)
        self.free = seen & (self.dist >= clearance)
        lab, n = ndimage.label(self.free); sizes = ndimage.sum(self.free, lab, range(1, n + 1))
        self.free &= np.isin(lab, 1 + np.nonzero(sizes >= 30)[0])             # drop specks the robot could not use

    @classmethod
    def from_saved(cls, d: Path) -> "FreeSpace":
        """Re-open a saved free grid (free_grid.npy + map_xy.npy + structure.json) without the scans."""
        self = cls.__new__(cls); st = json.load(open(Path(d) / "structure.json"))
        self.cell = st["cell"]; self.x0, self.y0 = st["x0"], st["y0"]; self.map = np.load(Path(d) / "map_xy.npy"); self.tree = cKDTree(self.map)
        self.free = np.load(Path(d) / "free_grid.npy"); self.ny, self.nx = self.free.shape
        ys, xs = np.mgrid[0:self.ny, 0:self.nx]; cx = self.x0 + (xs + 0.5) * self.cell; cy = self.y0 + (ys + 0.5) * self.cell
        self.dist = self.tree.query(np.stack([cx.ravel(), cy.ravel()], -1))[0].reshape(self.ny, self.nx)
        return self

    def cell_of(self, xy):
        return int((xy[1] - self.y0) / self.cell), int((xy[0] - self.x0) / self.cell)

    def xy_of(self, ij):
        return (self.x0 + (ij[1] + 0.5) * self.cell, self.y0 + (ij[0] + 0.5) * self.cell)

    def is_free(self, xy):
        i, j = self.cell_of(xy); return 0 <= i < self.ny and 0 <= j < self.nx and bool(self.free[i, j])

    def nearest_free(self, xy, max_m=3.0):
        i0, j0 = self.cell_of(xy); r = int(max_m / self.cell); best = None
        for i in range(max(0, i0 - r), min(self.ny, i0 + r + 1)):
            for j in range(max(0, j0 - r), min(self.nx, j0 + r + 1)):
                if self.free[i, j]:
                    d = math.hypot(i - i0, j - j0)
                    if best is None or d < best[0]:
                        best = (d, (i, j))
        return None if best is None or best[0] * self.cell > max_m else best[1]

    def astar(self, start_xy, goal_xy, max_m=3.0):
        """Shortest drivable path (list of xy) with a clearance cost; None when start or goal is not drivable."""
        s = self.nearest_free(start_xy, max_m); g = self.nearest_free(goal_xy, max_m)
        if s is None or g is None:
            return None
        cost = 1.0 + 1.5 * np.clip(1.0 - (self.dist - ROBOT_HALF_M) / 0.8, 0, 1)    # prefer the middle of a corridor
        openq = [(0.0, s)]; came = {s: None}; gsc = {s: 0.0}
        nbrs = [(di, dj, math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == g:
                break
            for di, dj, w in nbrs:
                n = (cur[0] + di, cur[1] + dj)
                if not (0 <= n[0] < self.ny and 0 <= n[1] < self.nx) or not self.free[n]:
                    continue
                if di and dj and not (self.free[cur[0] + di, cur[1]] and self.free[cur[0], cur[1] + dj]):
                    continue                                                       # no corner cutting through walls
                ng = gsc[cur] + w * cost[n]
                if ng < gsc.get(n, 1e18):
                    gsc[n] = ng; came[n] = cur; heapq.heappush(openq, (ng + math.hypot(n[0] - g[0], n[1] - g[1]), n))
        if g not in came:
            return None
        path = []; c = g
        while c is not None:
            path.append(self.xy_of(c)); c = came[c]
        return [tuple(start_xy[:2])] + path[::-1][1:-1] + [tuple(goal_xy[:2])]


def wall_direction(map_xy: np.ndarray) -> float:
    """Dominant direction of the long structures in the map (degrees): the angle whose projection histogram is
    sharpest - long straight walls pile up in a few bins."""
    best = (0.0, 0.0)
    for a in np.arange(0.0, 180.0, 1.0):
        r = math.radians(a); p = map_xy[:, 0] * -math.sin(r) + map_xy[:, 1] * math.cos(r)   # distance across lines at angle a
        h, _ = np.histogram(p, bins=np.arange(p.min(), p.max() + 0.1, 0.1))
        score = float(np.sort(h)[-12:].sum()) / len(map_xy)
        if score > best[1]:
            best = (float(a), score)
    return best[0]


def find_aisles(fs: FreeSpace, theta_deg: float, min_len_m=8.0):
    """Long narrow free corridors between parallel walls, in the map's own wall direction. Returns a list of
    {"id", "a": (x, y) one end, "b": (x, y) other end, "width_m", "length_m"} in scene coordinates."""
    r = math.radians(theta_deg); u = np.array([math.cos(r), math.sin(r)]); v = np.array([-math.sin(r), math.cos(r)])
    ys, xs = np.nonzero(fs.free); pts = np.stack([fs.x0 + (xs + 0.5) * fs.cell, fs.y0 + (ys + 0.5) * fs.cell], -1)
    along = pts @ u; across = pts @ v; dist = fs.dist[ys, xs]
    step = fs.cell; bins = np.arange(across.min(), across.max() + step, step); col = np.digitize(across, bins)
    # per cross-position column: the longest contiguous run of free cells along the wall direction
    runs = {}
    for c in np.unique(col):
        sel = np.sort(along[col == c]); gaps = np.diff(sel) > 1.5 * step
        starts = np.r_[0, np.nonzero(gaps)[0] + 1]; ends = np.r_[np.nonzero(gaps)[0], len(sel) - 1]
        k = int(np.argmax(sel[ends] - sel[starts])); runs[c] = (sel[starts[k]], sel[ends[k]], float(np.median(dist[col == c])))
    # corridors: contiguous columns whose longest run is >= min_len and whose width is aisle-like (0.8-2.6 m)
    cols = sorted(runs); aisles = []; cur = []
    def flush(cur):
        if not cur: return
        w = len(cur) * step + 2 * ROBOT_HALF_M                     # wall to wall: the free band plus the clearance on each side
        if AISLE_W[0] <= w <= AISLE_W[1]:
            a0 = float(np.percentile([runs[c][0] for c in cur], 25)); a1 = float(np.percentile([runs[c][1] for c in cur], 75))
            if a1 - a0 >= min_len_m:
                mid = bins[int(np.mean(cur)) - 1] + step / 2
                A = u * a0 + v * mid; B = u * a1 + v * mid
                # what is beside this corridor? count point layers within 2.5 m on each side (0.1 m bins across)
                mp_along = fs.map @ u; mp_across = fs.map @ v; inside = (mp_along > a0) & (mp_along < a1)
                layers = []
                for sign in (+1, -1):
                    dcross = sign * (mp_across[inside] - mid); sel = (dcross > 0.2) & (dcross < 2.5)
                    h, _ = np.histogram(dcross[sel], bins=np.arange(0.2, 2.6, 0.1)); peaks = int(((h[1:-1] > h[:-2]) & (h[1:-1] >= h[2:]) & (h[1:-1] > 0.03 * max(1, h.sum()))).sum())
                    layers.append(peaks)
                kind = "plant aisle" if min(layers) >= 2 else "corridor"
                aisles.append({"id": f"{'aisle' if kind == 'plant aisle' else 'corridor'}-{len(aisles) + 1:02d}", "kind": kind, "side_layers": layers, "a": (float(A[0]), float(A[1])), "b": (float(B[0]), float(B[1])), "width_m": round(w, 2), "length_m": round(float(a1 - a0), 1)})
    for c in cols:
        a0, a1, _ = runs[c]
        if a1 - a0 >= min_len_m and (not cur or c == cur[-1] + 1):
            cur.append(c)
        else:
            flush(cur); cur = [c] if a1 - a0 >= min_len_m else []
    flush(cur)
    return aisles


def build(scans_jsonl: Path, rec: Path, drive_every=10):
    rows = [json.loads(l) for l in open(scans_jsonl)]; mp = [r for r in rows if r["tag"] == "map"]
    S = [s for s in (json.loads(l) for l in open(rec / "steps.jsonl")) if s.get("lidar") and s.get("xy")][::drive_every]
    poses = [(r["x"], r["y"], r["heading_deg"]) for r in mp] + [(s["xy"][0], s["xy"][1], s["heading_deg"]) for s in S]
    scans = [r["lidar"] for r in mp] + [s["lidar"] for s in S]
    from scripts.offpath_eval import build_from_scans
    m = np.load(rec / "teach" / "prior_map_xy.npy"); look = build_from_scans(mp)
    merged = np.unique(np.floor(np.vstack([m, look]) / 0.05).astype(int), axis=0) * 0.05 + 0.025   # taught + look-around, own-body returns removed
    fs = FreeSpace(merged, poses, scans); theta = wall_direction(merged); aisles = find_aisles(fs, theta)
    first = next(s for s in (json.loads(l) for l in open(rec / "steps.jsonl")) if s.get("xy"))
    dock = (first["xy"][0], first["xy"][1], first["heading_deg"])          # where the robot was switched on
    return fs, merged, theta, aisles, dock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", type=Path, required=True); ap.add_argument("--rec", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    fs, m, theta, aisles, dock = build(a.scans, a.rec)
    print(f"free cells {int(fs.free.sum())} of {fs.free.size} ({fs.free.sum() * fs.cell ** 2:.0f} m2 seen empty and drivable); wall direction {theta:.0f} deg")
    for x in aisles: print("  ", x)
    # a demo plan: dock -> each aisle end to end -> dock
    plan = []; cur = dock[:2]; order = []
    todo = [x for x in aisles if x["kind"] == "plant aisle"]
    while todo:
        k = min(range(len(todo)), key=lambda i: min(math.dist(cur, todo[i]["a"]), math.dist(cur, todo[i]["b"])))
        ai = todo.pop(k); ends = (ai["a"], ai["b"]) if math.dist(cur, ai["a"]) <= math.dist(cur, ai["b"]) else (ai["b"], ai["a"])
        p1 = fs.astar(cur, ends[0]); p2 = fs.astar(ends[0], ends[1])
        if p1 is None or p2 is None:
            print("   cannot reach", ai["id"]); continue
        plan += p1 + p2; cur = ends[1]; order.append(ai["id"])
    back = fs.astar(cur, dock[:2]); plan += back or []
    L = sum(math.dist(p, q) for p, q in zip(plan, plan[1:]))
    print(f"mission from the dock: {' -> '.join(order)} -> dock, {L:.0f} m")
    np.save(a.out / "free_grid.npy", fs.free); np.save(a.out / "map_xy.npy", m); json.dump({"x0": fs.x0, "y0": fs.y0, "cell": fs.cell, "wall_direction_deg": theta, "aisles": aisles, "dock": dock,
                                                         "mission_order": order, "mission_length_m": round(L, 1)}, open(a.out / "structure.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 15), facecolor="#0d1117"); ax.set_facecolor("#0d1117")
    ax.imshow(fs.free, origin="lower", extent=(fs.x0, fs.x0 + fs.nx * fs.cell, fs.y0, fs.y0 + fs.ny * fs.cell), cmap="Greens", alpha=0.35, vmin=0, vmax=1.3)
    ax.scatter(m[:, 0], m[:, 1], s=0.6, c="#8b949e", lw=0, rasterized=True)
    for x in aisles:
        ax.plot([x["a"][0], x["b"][0]], [x["a"][1], x["b"][1]], c="#ffd33d" if x["kind"] == "plant aisle" else "#79c0ff", lw=2.5); ax.annotate(f"{x['id']}  {x['length_m']:.0f} m", ((x["a"][0] + x["b"][0]) / 2, (x["a"][1] + x["b"][1]) / 2), color="#ffd33d", fontsize=9, rotation=90 if abs(theta - 90) < 45 else 0, xytext=(6, 0), textcoords="offset points")
    P = np.array(plan); ax.plot(P[:, 0], P[:, 1], c="#3fb950", lw=1.6, alpha=0.9, label=f"path the robot planned itself ({L:.0f} m)")
    ax.plot(dock[0], dock[1], marker="*", ms=18, c="#ffd33d", mec="#000"); ax.annotate("dock = where it was switched on", dock[:2], color="#ffd33d", fontsize=9, xytext=(8, 8), textcoords="offset points")
    ax.set_aspect("equal"); ax.tick_params(colors="#8b949e"); ax.legend(loc="lower left", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.set_title("From the robot's own map: green = seen empty and drivable, yellow = aisles it found, line = the survey path it planned (no stored route)", color="#fff", fontsize=10)
    fig.tight_layout(); fig.savefig(a.out / "free_nav_structure.png", dpi=140, facecolor=fig.get_facecolor()); print("figure", a.out / "free_nav_structure.png")


if __name__ == "__main__":
    main()
