"""Mission planning on the robot's own map, offered to HERMES through the hub.

The robot reads its map (scripts/free_nav.py): drivable space, plant aisles, the dock. HERMES is the decision-maker:
it asks for the structure, asks the robot to plan a survey in an order, sees every leg's length and a picture of the
plan on the map, approves or changes the order, and sends the approved mission. Nothing here follows a stored route.

Hub routes (agent/robot_api.py):  GET /structure   POST /plan {"order": [...], "from": [x, y]}   POST /mission_plan {...}
"""
from __future__ import annotations
import base64, io, itertools, json, math, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.free_nav import FreeSpace            # noqa: E402


class MissionPlanner:
    def __init__(self, structure_dir: Path):
        d = Path(structure_dir); self.dir = d; self.fs = FreeSpace.from_saved(d); st = json.load(open(d / "structure.json"))
        self.dock = tuple(st["dock"]); self.theta = st["wall_direction_deg"]
        self.aisles = [a for a in st["aisles"] if a.get("kind") == "plant aisle" and self.fs.astar(self.dock[:2], a["a"]) is not None and self.fs.astar(self.dock[:2], a["b"]) is not None]
        self.unreachable = [a["id"] for a in st["aisles"] if a.get("kind") == "plant aisle" and a not in self.aisles]

    def structure(self) -> dict:
        """What the robot found in its own map, in words a planner can use."""
        return {"source": "the robot's own LiDAR map; nothing drawn by a person", "wall_direction_deg": self.theta,
                "drivable_area_m2": round(float(self.fs.free.sum()) * self.fs.cell ** 2, 1),
                "dock": {"xy": [round(self.dock[0], 2), round(self.dock[1], 2)], "meaning": "where the robot was switched on"},
                "plant_aisles": [{"id": a["id"], "from_xy": [round(a["a"][0], 2), round(a["a"][1], 2)], "to_xy": [round(a["b"][0], 2), round(a["b"][1], 2)],
                                  "length_m": a["length_m"], "width_m": a["width_m"], "reachable": True} for a in self.aisles],
                "aisles_seen_but_not_reachable": self.unreachable}

    def goal_points(self, order, start_xy):
        pts = []; cur = tuple(start_xy)
        for aid in order:
            a = next(x for x in self.aisles if x["id"] == aid)
            if math.dist(cur, a["a"]) <= math.dist(cur, a["b"]):
                pts += [(f"{aid}/start", tuple(a["a"])), (f"{aid}/end", tuple(a["b"]))]; cur = tuple(a["b"])
            else:
                pts += [(f"{aid}/start", tuple(a["b"])), (f"{aid}/end", tuple(a["a"]))]; cur = tuple(a["a"])
        pts.append(("dock", self.dock[:2])); return pts

    def plan(self, order, start_xy) -> dict:
        """Every leg planned on the free grid from `start_xy`; None metres = that leg cannot be planned."""
        ids = {a["id"] for a in self.aisles}; bad = [o for o in order if o not in ids]
        if bad:
            return {"ok": False, "error": f"unknown or unreachable aisle ids: {bad}", "known": sorted(ids)}
        legs = []; cur = tuple(start_xy); poly = [cur]; total = 0.0; ok = True; prev = None
        for name, g in self.goal_points(order, start_xy):
            leg = self.astar_in_aisle(cur, g, name[:-4]) if name.endswith("/end") and prev == name[:-4] + "/start" else self.fs.astar(cur, g)
            prev = name; L = None if leg is None else sum(math.dist(p, q) for p, q in zip(leg, leg[1:]))
            legs.append({"to": name, "metres": None if L is None else round(L, 1)}); ok &= L is not None; total += L or 0.0
            poly += list(leg[1:]) if leg else [g]; cur = tuple(g)
        missing = sorted(ids - set(order))
        return {"ok": ok and not missing, "order": list(order), "legs": legs, "total_m": round(total, 1), "aisles_missing_from_order": missing,
                "polyline_xy": [[round(x, 2), round(y, 2)] for x, y in poly[:: max(1, len(poly) // 400)]]}

    def astar_in_aisle(self, start, goal, aid):
        """The survey leg of an aisle is planned inside that aisle (free grid masked to 1.2 m either side of it)."""
        a = next(x for x in self.aisles if x["id"] == aid); A, B = np.array(a["a"]), np.array(a["b"]); u = (B - A) / np.linalg.norm(B - A)
        fs = self.fs; ys, xs = np.mgrid[0:fs.ny, 0:fs.nx]; cx = fs.x0 + (xs + 0.5) * fs.cell; cy = fs.y0 + (ys + 0.5) * fs.cell
        d = np.stack([cx - A[0], cy - A[1]], -1); t = d @ u; off = np.abs(d[..., 0] * u[1] - d[..., 1] * u[0])
        saved = fs.free; fs.free = saved & (off <= 1.2) & (t >= -1.5) & (t <= np.linalg.norm(B - A) + 1.5)
        try:
            return fs.astar(start, goal)
        finally:
            fs.free = saved

    def rpp_order(self, start_xy, aisle_ids=None) -> dict:
        """Rural Postman Problem on the small graph of aisle ends + dock: every chosen aisle is a REQUIRED edge
        (driven end to end), the connections between ends are A* distances on the free grid, the robot starts where
        it is and ends at the dock. Exact (every order and direction) up to 7 aisles, nearest-first + 2-opt beyond.
        Runs in well under a second on the robot - no model call needed to choose the order."""
        ids = aisle_ids or [a["id"] for a in self.aisles]; A = {a["id"]: a for a in self.aisles if a["id"] in ids}
        ends = {"start": tuple(start_xy), "dock": self.dock[:2]}
        for i, a in A.items():
            ends[f"{i}/a"] = tuple(a["a"]); ends[f"{i}/b"] = tuple(a["b"])
        keys = list(ends); D = {}
        for u in keys:
            for v in keys:
                if u == v or (u, v) in D:
                    continue
                leg = self.fs.astar(ends[u], ends[v]); L = None if leg is None else sum(math.dist(p, q) for p, q in zip(leg, leg[1:]))
                D[(u, v)] = D[(v, u)] = L
        inside = {i: self.astar_in_aisle(ends[f"{i}/a"], ends[f"{i}/b"], i) for i in A}
        inside = {i: (None if leg is None else sum(math.dist(p, q) for p, q in zip(leg, leg[1:]))) for i, leg in inside.items()}
        def cost(seq):
            """seq = [(aisle id, direction 'ab' or 'ba'), ...]"""
            cur = "start"; total = 0.0
            for i, d in seq:
                first, last = (f"{i}/a", f"{i}/b") if d == "ab" else (f"{i}/b", f"{i}/a")
                c1 = D.get((cur, first)) if cur != first else 0.0
                if c1 is None or inside[i] is None:
                    return None
                total += c1 + inside[i]; cur = last
            c = D.get((cur, "dock")); return None if c is None else total + c
        import itertools
        best = None
        if len(A) <= 7:
            for perm in itertools.permutations(A):
                for dirs in itertools.product(("ab", "ba"), repeat=len(perm)):
                    c = cost(list(zip(perm, dirs)))
                    if c is not None and (best is None or c < best[0]):
                        best = (c, list(zip(perm, dirs)))
        else:
            todo = set(A); cur = "start"; seq = []
            while todo:
                cand = [(D.get((cur, f"{i}/a")), i, "ab") for i in todo] + [(D.get((cur, f"{i}/b")), i, "ba") for i in todo]
                cand = [c for c in cand if c[0] is not None]
                if not cand:
                    break
                _, i, d = min(cand); seq.append((i, d)); todo.discard(i); cur = f"{i}/b" if d == "ab" else f"{i}/a"
            improved = True
            while improved and len(seq) > 2:
                improved = False
                for x in range(len(seq) - 1):
                    for y in range(x + 1, len(seq)):
                        alt = seq[:x] + [(i, "ba" if d == "ab" else "ab") for i, d in reversed(seq[x:y + 1])] + seq[y + 1:]
                        if (cost(alt) or 1e18) < (cost(seq) or 1e18):
                            seq = alt; improved = True
            best = (cost(seq), seq)
        if best is None or best[0] is None:
            return {"ok": False, "error": "no closed route covers every chosen aisle"}
        return {"ok": True, "order": [i for i, _ in best[1]], "directions": dict(best[1]), "total_m": round(best[0], 1), "method": "exact RPP" if len(A) <= 7 else "RPP heuristic (nearest + 2-opt)"}

    def alternatives(self, start_xy) -> list:
        out = []
        for o in itertools.permutations([a["id"] for a in self.aisles]):
            p = self.plan(list(o), start_xy)
            if p["ok"]:
                out.append({"order": list(o), "total_m": p["total_m"]})
        return sorted(out, key=lambda x: x["total_m"])

    def picture(self, order=None, start_xy=None) -> bytes:
        """JPEG of the map, the drivable space, the aisles, the dock and (if given) the plan - what HERMES looks at."""
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fs = self.fs; fig, ax = plt.subplots(figsize=(5, 8), dpi=100); ax.set_facecolor("#111")
        ax.imshow(fs.free, origin="lower", extent=(fs.x0, fs.x0 + fs.nx * fs.cell, fs.y0, fs.y0 + fs.ny * fs.cell), cmap="Greens", alpha=0.5, vmin=0, vmax=1.3)
        ax.scatter(fs.map[:, 0], fs.map[:, 1], s=0.3, c="#aaa", lw=0)
        for a in self.aisles:
            ax.plot([a["a"][0], a["b"][0]], [a["a"][1], a["b"][1]], c="#ffd33d", lw=1.5, alpha=0.7)
            ax.annotate(a["id"], ((a["a"][0] + a["b"][0]) / 2, (a["a"][1] + a["b"][1]) / 2), color="#ffd33d", fontsize=8, rotation=90)
        if order and start_xy is not None:
            P = np.array(self.plan(order, start_xy)["polyline_xy"]); ax.plot(P[:, 0], P[:, 1], c="#2ecc71", lw=2)
            ax.plot(start_xy[0], start_xy[1], "o", c="#ff7b72", ms=8); ax.annotate("robot", start_xy, color="#ff7b72", fontsize=8, xytext=(5, 5), textcoords="offset points")
        ax.plot(self.dock[0], self.dock[1], "*", c="#ffd33d", ms=14); ax.annotate("dock", self.dock[:2], color="#ffd33d", fontsize=8, xytext=(5, 5), textcoords="offset points")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="jpeg", facecolor="#111"); plt.close(fig); return buf.getvalue()

    def picture_b64(self, order=None, start_xy=None) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(self.picture(order, start_xy)).decode()


def load(structure_dir: str | None = None) -> MissionPlanner | None:
    d = structure_dir or os.environ.get("R1_STRUCTURE")
    if not d or not (Path(d) / "structure.json").exists():
        return None
    return MissionPlanner(Path(d))
