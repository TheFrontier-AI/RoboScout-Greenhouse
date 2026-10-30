#!/usr/bin/env python3
"""The farm the way a person understands it, built by the robot from its own map: houses (blocks of parallel walls),
their doors (where the drivable space crosses an end wall), the plant rows inside each house, the open yards that
connect the doors, and the dock. This is what the agent plans on - named places, not metre coordinates.

  python3 scripts/scene_graph.py --structure <free_nav dir> --out scene_graph.json"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.free_nav import FreeSpace            # noqa: E402


def build_scene_graph(fs: FreeSpace, aisles: list, dock, theta_deg: float) -> dict:
    """Axis-aligned after rotating by the wall direction; walls run along +v, houses are blocks of dense wall columns."""
    r = math.radians(theta_deg); u = np.array([math.cos(r), math.sin(r)]); v = np.array([-math.sin(r), math.cos(r)])
    to_uv = lambda p: (float(np.dot(p, u)), float(np.dot(p, v)))          # along the walls, across the walls
    to_xy = lambda a, c: (float(a * u[0] + c * v[0]), float(a * u[1] + c * v[1]))
    # dense wall "columns" across (v) and dense wall "rows" along (u) from the map points
    P = fs.map; A = P @ u; C = P @ v
    cb = np.arange(C.min(), C.max() + 0.25, 0.25); ab = np.arange(A.min(), A.max() + 0.25, 0.25)
    H, _, _ = np.histogram2d(C, A, bins=[cb, ab]); occ = H > 0
    col_occ = occ.mean(axis=1); dense_cols = [cb[i] + 0.125 for i in np.nonzero(col_occ > 0.45)[0]]
    # houses: runs of dense columns with gaps < 1.5 m
    houses = []; cur = []
    for c in dense_cols:
        if cur and c - cur[-1] > 1.5:
            houses.append((cur[0] - 0.3, cur[-1] + 0.3)); cur = []
        cur.append(c)
    if cur:
        houses.append((cur[0] - 0.3, cur[-1] + 0.3))
    out_houses = []
    for k, (c0, c1) in enumerate(houses):
        hid = f"house-{chr(65 + k)}"
        rows = []
        for a in aisles:
            if a.get("kind") != "plant aisle":
                continue
            pa, pb = to_uv(np.array(a["a"])), to_uv(np.array(a["b"]))
            if c0 <= (pa[1] + pb[1]) / 2 <= c1:
                ends = sorted([(pa, a["a"]), (pb, a["b"])], key=lambda e: e[0][0])
                rows.append({"id": f"{hid}/row-{len(rows) + 1}", "from_map_id": a["id"], "end_low": [round(x, 2) for x in ends[0][1]], "end_high": [round(x, 2) for x in ends[1][1]],
                             "length_m": a["length_m"], "width_m": a["width_m"], "_lo": ends[0][0][0], "_hi": ends[1][0][0]})
        # the house ends = the end walls nearest to where the rows end (a wall = an along-position where the points fill
        # most of the house's width; the road or fence farther out is not the house)
        sel = (C >= c0) & (C <= c1); Hh, _, _ = np.histogram2d(A[sel], C[sel], bins=[ab, np.arange(c0, c1 + 0.25, 0.25)])
        fill = (Hh > 0).mean(axis=1); walls = [ab[i] + 0.125 for i in np.nonzero(fill > 0.35)[0]]
        row_lo = min(r["_lo"] for r in rows) if rows else float(np.percentile(A[sel], 1)); row_hi = max(r["_hi"] for r in rows) if rows else float(np.percentile(A[sel], 99))
        lo_c = [w for w in walls if row_lo - 3.0 <= w <= row_lo + 1.5]; hi_c = [w for w in walls if row_hi - 1.5 <= w <= row_hi + 3.0]
        a_lo = float(min(lo_c)) if lo_c else row_lo - 1.5; a_hi = float(max(hi_c)) if hi_c else row_hi + 1.5
        for r_ in rows:
            r_.pop("_lo"); r_.pop("_hi")
        doors = []
        for side, a_wall in (("low", a_lo), ("high", a_hi)):
            # a door = drivable cells on the wall line within the house span (one per gap)
            cs = np.arange(c0 + 0.25, c1, 0.25); free_on_wall = [c for c in cs if fs.is_free(to_xy(a_wall, c))]
            groups = []
            for c in free_on_wall:
                if groups and c - groups[-1][-1] <= 0.3:
                    groups[-1].append(c)
                else:
                    groups.append([c])
            for g in groups:
                cm = float(np.mean(g)); x, y = to_xy(a_wall, cm)
                doors.append({"id": f"{hid}/door-{side}" + (f"-{len(doors) + 1}" if len(groups) > 1 else ""), "xy": [round(x, 2), round(y, 2)], "side": side, "width_m": round(len(g) * 0.25 + 0.5, 2)})
        out_houses.append({"id": hid, "span_across_m": [round(c0, 2), round(c1, 2)], "span_along_m": [round(a_lo, 2), round(a_hi, 2)], "rows": rows, "doors": doors,
                           "corners_xy": [[round(x, 2) for x in to_xy(a, c)] for a, c in ((a_lo, c0), (a_lo, c1), (a_hi, c1), (a_hi, c0))]})
    # yards: open space beyond the house ends, and the gaps between houses
    yards = [{"id": "yard-low", "where": "beyond the low end of the houses", "connects": [d["id"] for h in out_houses for d in h["doors"] if d["side"] == "low"]},
             {"id": "yard-high", "where": "beyond the high end of the houses", "connects": [d["id"] for h in out_houses for d in h["doors"] if d["side"] == "high"]}]
    for (c0, c1), (d0, d1) in zip(houses, houses[1:]):
        yards.append({"id": f"gap-{len(yards) - 1}", "where": f"between {out_houses[len(yards) - 2]['id']} and {out_houses[len(yards) - 1]['id']}", "connects": ["yard-low", "yard-high"]})
    dk = to_uv(np.array(dock[:2])); dock_yard = "yard-high" if dk[0] > np.mean([h["span_along_m"][1] for h in out_houses]) else "yard-low"
    return {"frame": "map metres, x right, y up", "wall_direction_deg": theta_deg, "houses": out_houses, "yards": yards,
            "dock": {"xy": [round(dock[0], 2), round(dock[1], 2)], "in": dock_yard, "meaning": "where the robot was switched on"}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--structure", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); st = json.load(open(a.structure / "structure.json")); fs = FreeSpace.from_saved(a.structure)
    g = build_scene_graph(fs, st["aisles"], st["dock"], st["wall_direction_deg"]); json.dump(g, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in g.items() if k != "houses"}, indent=1)); [print(json.dumps(h)) for h in g["houses"]]


if __name__ == "__main__":
    main()
