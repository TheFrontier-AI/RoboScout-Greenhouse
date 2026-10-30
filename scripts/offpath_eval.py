#!/usr/bin/env python3
"""Off-path relocalisation with REAL simulated scans (scripts/offpath_scans.py output):
  1. build the "look-around" map from the map-tagged scans (poses along the taught path, 8 headings each) and merge it
     with the taught map, so the map also holds what is visible beside the path;
  2. relocalise every off-path scan (robot put down 1-3 m beside the path) on that map, path + grid hypotheses;
  3. for every found pose, plan the way back: rejoin the nearest site-graph node, then the normal route to the dock.

  python3 scripts/offpath_eval.py --scans scans.jsonl --teach <run>/rec/teach [--out <dir>]"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.relocalize import Relocalizer                       # noqa: E402
from scripts.teach_map import beam_angles, scan_xy, to_scene  # noqa: E402
from r1_nav.site_map import NoRoute, SiteMap                 # noqa: E402


def build_from_scans(rows, cell=0.05, min_hits=2, max_range=11.0):
    ang = beam_angles(len(rows[0]["lidar"])); hits = {}
    for r in rows:
        pts = to_scene(scan_xy(r["lidar"], ang, max_range), r["x"], r["y"], r["heading_deg"])
        for cx, cy in np.floor(pts / cell).astype(int):
            hits[(cx, cy)] = hits.get((cx, cy), 0) + 1
    cells = np.array([k for k, v in hits.items() if v >= min_hits], float)
    return (cells + 0.5) * cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", type=Path, required=True); ap.add_argument("--teach", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None); ap.add_argument("--no-lookaround", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.scans)]
    maps_rows = [r for r in rows if r["tag"] == "map"]; off = [r for r in rows if r["tag"] == "offpath"]
    taught = np.load(a.teach / "prior_map_xy.npy"); path = np.load(a.teach / "path_xyh.npy")
    site = SiteMap.load(a.teach / "site_map.json")
    if a.no_lookaround:
        m = taught
    else:
        look = build_from_scans(maps_rows)
        m = np.unique(np.floor(np.vstack([taught, look]) / 0.05).astype(int), axis=0) * 0.05 + 0.025
    print(f"map points: taught {len(taught)}  look-around {0 if a.no_lookaround else len(look)}  merged {len(m)}", flush=True)
    ang = beam_angles(len(rows[0]["lidar"]))
    reloc = Relocalizer(m, path, grid_m=0.5, grid_headings=12)
    res = {"n": 0, "found_1m": 0, "ambiguous": 0, "ambiguous_best_correct": 0, "wrong": 0, "errs": [], "rejoin_ok": 0, "rejoin_refused": 0, "cases": []}
    t0 = time.time()
    for r in off:
        rl = reloc.locate(scan_xy(r["lidar"], ang)); res["n"] += 1
        e = math.hypot(rl.pose[0] - r["x"], rl.pose[1] - r["y"]) if rl.pose else 99.0
        kind = "ambiguous" if rl.ambiguous else ("found" if e < 1.0 else "wrong")
        if kind == "ambiguous": res["ambiguous"] += 1; res["ambiguous_best_correct"] += int(e < 1.0)
        elif kind == "found": res["found_1m"] += 1; res["errs"].append(e)
        else: res["wrong"] += 1
        back = None
        if kind == "found":
            try:
                rt = site.route_from_xy(rl.pose[:2], "dock"); back = {"nodes": rt.nodes, "length_m": rt.length_m}; res["rejoin_ok"] += 1
            except NoRoute as ex:
                back = {"refused": str(ex)}; res["rejoin_refused"] += 1
        res["cases"].append({"x": r["x"], "y": r["y"], "off_m": r.get("off_m"), "heading_deg": r["heading_deg"], "result": kind,
                             "err_m": round(e, 2), "score": rl.score, "route_back": back})
    res["median_err_m"] = round(float(np.median(res["errs"])), 3) if res["errs"] else None; del res["errs"]
    res["s_per_scan"] = round((time.time() - t0) / max(len(off), 1), 2); res["hypotheses"] = int(len(reloc.hyp))
    summ = {k: v for k, v in res.items() if k != "cases"}
    print(json.dumps(summ, indent=1))
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "offpath_eval.json").write_text(json.dumps(res, indent=1)); np.save(a.out / "map_lookaround_xy.npy", m)


if __name__ == "__main__":
    main()
