#!/usr/bin/env python3
"""HERMES plans the survey on the farm as a person understands it - the scene graph the robot built from its map
(houses, rows, doors, yards, dock) - as a sequence of NAMED places, never as metre coordinates. The robot turns each
named leg into geometry (A* on its free grid, row centre lines) and checks the result: every leg drivable, every row
covered end to end, finishing at the dock. The same start poses as the drawing experiment.

  python3 scripts/hermes_plans_on_graph.py --scans scans_all.jsonl --rec <run>/rec --cases vlm_offpath.jsonl --out <dir> [--n 12]"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.free_nav import build as build_structure            # noqa: E402
from scripts.freenav_mission_eval import FreeFarm, _last_json     # noqa: E402
from agent.mission_planner import MissionPlanner                 # noqa: E402


def where_am_i(g: dict, xy) -> str:
    x, y = xy
    for h in g["houses"]:
        xs = [c[0] for c in h["corners_xy"]]; ys = [c[1] for c in h["corners_xy"]]
        if min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys):
            return h["id"]
    ys_hi = max(h["span_along_m"][1] for h in g["houses"]); ys_lo = min(h["span_along_m"][0] for h in g["houses"])
    return "yard-high" if y > ys_hi else ("yard-low" if y < ys_lo else "between the houses")


def ask(g: dict, start_xy, place: str):
    from openai import OpenAI
    client = OpenAI(base_url=os.environ.get("HERMES_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"), api_key=os.environ.get("NEBIUS_API_KEY"), timeout=180)
    names = [d["id"] for h in g["houses"] for d in h["doors"]] + [r["id"] for h in g["houses"] for r in h["rows"] if r["length_m"] >= 20] + [y["id"] for y in g["yards"]] + ["dock"]
    msg = ("You are HERMES, mission planner of a greenhouse survey robot. The robot built this description of the farm from its own map. "
           "Plan the survey as a sequence of NAMED places (no coordinates): the robot must drive every long plant row (length >= 20 m) from one end to the other once, "
           "pass through doors to leave or enter a house, cross the yard that connects the doors, and finish at the dock. Rows are traversed by naming them with a "
           "direction 'low->high' or 'high->low' (low/high = the two ends of the house). Keep the driving short.\n"
           "Farm: " + json.dumps(g) + f"\nThe robot is now at {list(map(lambda v: round(v, 1), start_xy))}, which is in {place}.\n"
           "Usable place names: " + json.dumps(names) +
           "\nReply with JSON only: {\"route\": [\"place or row-id low->high\", ...], \"reason\": \"one sentence\"}.")
    t0 = time.time()
    r = client.chat.completions.create(model=os.environ.get("HERMES_MODEL", "nvidia/nemotron-3-super-120b-a12b"), temperature=0.1, max_tokens=8000, messages=[{"role": "user", "content": msg}])
    m = r.choices[0].message; txt = (m.content or getattr(m, "reasoning_content", None) or "").split("</think>")[-1].replace("```json", "").replace("```", "")
    return _last_json(txt), round(time.time() - t0, 1)


def realise(g: dict, farm: FreeFarm, start_xy, route):
    """Named legs -> points; rows -> their centre line; everything else by A* on the free grid."""
    doors = {d["id"]: tuple(d["xy"]) for h in g["houses"] for d in h["doors"]}; rows = {r["id"]: r for h in g["houses"] for r in h["rows"]}
    pts = [tuple(start_xy)]; legs = []; cur = tuple(start_xy); rows_done = []; unknown = []
    for step in route:
        s = str(step).strip(); name = s.split()[0]
        if name in doors:
            tgt = doors[name]; leg = farm.fs.astar(cur, tgt); legs.append((name, leg is not None)); pts += list(leg[1:]) if leg else [tgt]; cur = tgt
        elif name in rows:
            r = rows[name]; lo, hi = tuple(r["end_low"]), tuple(r["end_high"])
            a, b = (lo, hi) if "high->low" not in s else (hi, lo)
            leg = farm.fs.astar(cur, a); legs.append((name + " (to its end)", leg is not None)); pts += list(leg[1:]) if leg else [a]
            inside = farm.astar_in_aisle(a, b, r["from_map_id"]); legs.append((name, inside is not None)); pts += list(inside[1:]) if inside else [b]; cur = b; rows_done.append(name)
        elif name == "dock":
            leg = farm.fs.astar(cur, farm.dock[:2]); legs.append(("dock", leg is not None)); pts += list(leg[1:]) if leg else [farm.dock[:2]]; cur = farm.dock[:2]
        elif any(name == y["id"] for y in g["yards"]):
            legs.append((name, True))                                         # a yard is passed through, no point of its own
        else:
            unknown.append(name)
    P = np.array(pts); t = cKDTree(P); cov = {}
    for aid, st in farm.stops.items():
        cov[aid] = round(float((t.query(st)[0] < 0.6).mean()), 3)
    L = float(sum(math.dist(p, q) for p, q in zip(pts, pts[1:])))
    ok = all(okk for _, okk in legs) and not unknown and all(v >= 0.98 for v in cov.values()) and math.dist(P[-1], farm.dock[:2]) < 2.0
    return {"ok": ok, "legs": legs, "unknown_names": unknown, "coverage": cov, "ends_at_dock": math.dist(P[-1], farm.dock[:2]) < 2.0, "length_m": round(L, 1), "polyline": [[round(x, 2), round(y, 2)] for x, y in pts[:: max(1, len(pts) // 300)]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", type=Path, required=True); ap.add_argument("--rec", type=Path, required=True); ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    fs, m, theta, aisles, dock = build_structure(a.scans, a.rec); farm = FreeFarm(fs, aisles, dock); g = json.load(open(a.rec / "teach" / "free_nav" / "scene_graph.json"))
    mp = MissionPlanner.__new__(MissionPlanner); mp.fs = fs; mp.dock = farm.dock; mp.aisles = farm.aisles; mp.astar_in_aisle = farm.astar_in_aisle
    cases = [json.loads(l) for l in open(a.cases)][: a.n]; out = []
    for i, c in enumerate(cases):
        start = tuple(c["truth"][:2]); place = where_am_i(g, start)
        try:
            j, secs = ask(g, start, place); res = realise(g, farm, start, j.get("route", [])); res["route"] = j.get("route"); res["reason"] = str(j.get("reason", ""))[:160]
        except Exception as e:
            res = {"ok": False, "why": f"model: {str(e)[:80]}"}; secs = None
        rpp = mp.rpp_order(start); ref = mp.plan(rpp["order"], start) if rpp["ok"] else None
        row = {"case": i + 1, "start_xy": [round(start[0], 2), round(start[1], 2)], "robot_in": place, "hermes": res, "seconds": secs, "rpp_astar_length_m": ref["total_m"] if ref else None}
        out.append(row); print(json.dumps({k: (v if k != "hermes" else {kk: vv for kk, vv in v.items() if kk != "polyline"}) for k, v in row.items()}), flush=True)
    good = [r for r in out if r["hermes"].get("ok")]
    summ = {"cases": len(out), "routes_returned": sum(1 for r in out if r["hermes"].get("route")), "complete_and_drivable": len(good),
            "length_vs_rpp_astar_median_ratio": round(float(np.median([r["hermes"]["length_m"] / r["rpp_astar_length_m"] for r in good if r["rpp_astar_length_m"]])), 2) if good else None,
            "seconds_median": round(float(np.median([r["seconds"] for r in out if r["seconds"]])), 1) if any(r["seconds"] for r in out) else None, "model": os.environ.get("HERMES_MODEL", "nvidia/nemotron-3-super-120b-a12b")}
    print(json.dumps(summ, indent=1)); (a.out / "hermes_on_graph.json").write_text(json.dumps({"summary": summ, "scene_graph": g, "cases": out}, indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = min(10, len(out)); fig, axs = plt.subplots(1, n, figsize=(2.3 * n, 5.8), facecolor="#fff")
    for ax, r in zip(np.atleast_1d(axs), out[:n]):
        ax.scatter(m[:, 0], m[:, 1], s=0.3, c="#6e7681", lw=0); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.set_xlim(-8, 17); ax.set_ylim(-44, 13)
        for h in g["houses"]:
            cs = np.array(h["corners_xy"] + [h["corners_xy"][0]]); ax.plot(cs[:, 0], cs[:, 1], c="#e6a800", lw=1, alpha=0.7); ax.annotate(h["id"], cs[0], fontsize=6, color="#a06a00")
            for d in h["doors"]: ax.plot(d["xy"][0], d["xy"][1], "D", c="#fff", mec="#000", ms=5)
        hh = r["hermes"]
        if hh.get("polyline"):
            P = np.array(hh["polyline"]); ax.plot(P[:, 0], P[:, 1], c="#4cc9f0", lw=1.6)
        ax.plot(r["start_xy"][0], r["start_xy"][1], "o", c="#000", ms=6); ax.plot(dock[0], dock[1], "*", c="#ffd33d", ms=12, mec="#000")
        ax.set_title(f"#{r['case']} {'✓' if hh.get('ok') else '✗'} {hh.get('length_m', 0):.0f} m vs {r['rpp_astar_length_m'] or 0:.0f} m\n" + " → ".join(str(x).replace("house-", "").replace("/row-", " row").replace("/door-", " door ") for x in (hh.get("route") or []))[:70], fontsize=6.5)
    fig.suptitle(f"HERMES plans on the farm as a person understands it (houses, rows, doors, yards, dock - named places, no coordinates); the robot makes the geometry: complete and drivable in {len(good)} of {len(out)}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(a.out / "hermes_on_graph.png", dpi=120); print("figure", a.out / "hermes_on_graph.png")


if __name__ == "__main__":
    main()
