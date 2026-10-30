#!/usr/bin/env python3
"""Can HERMES draw the survey route itself? The agent's vision model gets the robot's map as a picture with metre
axes (drivable space, the plant aisles, the dock, the robot) and the structure as JSON, and must return the whole
survey route as waypoints in metres. The robot then checks every segment on its own free grid: a segment that crosses
something the map says is not drivable is repaired with the grid planner between the two waypoints. We measure how
much of what HERMES drew was drivable as drawn, how much had to be repaired, whether the route covers every aisle end
to end and ends at the dock, and its length against the Rural Postman + A* plan.

  python3 scripts/hermes_draws_route.py --scans scans_all.jsonl --rec <run>/rec --cases vlm_offpath.jsonl --out <dir> [--n 20]"""
from __future__ import annotations
import argparse, base64, io, json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.free_nav import build as build_structure            # noqa: E402
from scripts.freenav_mission_eval import FreeFarm, _last_json     # noqa: E402
from agent.mission_planner import MissionPlanner                 # noqa: E402


def picture_with_axes(farm: FreeFarm, start_xy) -> str:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fs = farm.fs; fig, ax = plt.subplots(figsize=(6, 10), dpi=100); ax.set_facecolor("#f4f4f4")
    ax.imshow(fs.free, origin="lower", extent=(fs.x0, fs.x0 + fs.nx * fs.cell, fs.y0, fs.y0 + fs.ny * fs.cell), cmap="Greens", alpha=0.6, vmin=0, vmax=1.3)
    ax.scatter(fs.map[:, 0], fs.map[:, 1], s=0.4, c="#444", lw=0)
    for a in farm.aisles:
        ax.plot([a["a"][0], a["b"][0]], [a["a"][1], a["b"][1]], c="#e6a800", lw=2, alpha=0.8)
        ax.annotate(f'{a["id"]}  A=({a["a"][0]:.1f},{a["a"][1]:.1f})  B=({a["b"][0]:.1f},{a["b"][1]:.1f})', ((a["a"][0] + a["b"][0]) / 2, (a["a"][1] + a["b"][1]) / 2), color="#a06a00", fontsize=7, rotation=90)
    ax.plot(start_xy[0], start_xy[1], "o", c="#d62728", ms=9); ax.annotate(f"robot ({start_xy[0]:.1f},{start_xy[1]:.1f})", start_xy, color="#d62728", fontsize=8, xytext=(6, 6), textcoords="offset points")
    ax.plot(farm.dock[0], farm.dock[1], "*", c="#e6a800", ms=16, mec="#000"); ax.annotate(f"dock ({farm.dock[0]:.1f},{farm.dock[1]:.1f})", farm.dock[:2], color="#000", fontsize=8, xytext=(6, 6), textcoords="offset points")
    ax.set_xticks(np.arange(math.floor(fs.x0), fs.x0 + fs.nx * fs.cell, 2)); ax.set_yticks(np.arange(math.floor(fs.y0), fs.y0 + fs.ny * fs.cell, 2))
    ax.grid(True, color="#bbb", lw=0.4); ax.tick_params(labelsize=6); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal"); fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="jpeg", pil_kwargs={"quality": 90}); plt.close(fig)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ask_hermes(farm: FreeFarm, start_xy, img: str):
    from openai import OpenAI
    client = OpenAI(base_url=os.environ.get("EYES2_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"), api_key=os.environ.get("EYES2_API_KEY") or os.environ.get("NEBIUS_API_KEY"), timeout=240)
    structure = {"robot_xy": [round(start_xy[0], 2), round(start_xy[1], 2)], "dock_xy": [round(farm.dock[0], 2), round(farm.dock[1], 2)],
                 "plant_aisles": [{"id": a["id"], "end_A": [round(a["a"][0], 2), round(a["a"][1], 2)], "end_B": [round(a["b"][0], 2), round(a["b"][1], 2)], "width_m": a["width_m"]} for a in farm.aisles],
                 "rules": ["drive every plant aisle from one end to the other, once, along its centre line", "stay inside the green (drivable) area; dark dots are walls and plant rows", "end at the dock", "waypoints in metres, as on the axes; put a waypoint every 1-3 m in the open and at both ends of every aisle"]}
    msg = ("You are HERMES, planning the survey route for a greenhouse robot on the map it built itself. The picture shows the map with metre axes: green = drivable space "
           "the robot has seen empty, dark dots = walls and plant rows, orange lines = the plant aisles with their end coordinates, red dot = the robot, star = the dock.\n"
           "Draw the complete survey route as a list of waypoints the robot will follow in order.\n" + json.dumps(structure) +
           "\nReply with JSON only: {\"waypoints\": [[x, y], ...], \"aisle_order\": [ids], \"reason\": \"one sentence\"}.")
    t0 = time.time()
    r = client.chat.completions.create(model=os.environ.get("EYES2_MODEL", "nvidia/Nemotron-3.5-Super-VL"), temperature=0.1, max_tokens=8000,
                                       messages=[{"role": "user", "content": [{"type": "text", "text": msg}, {"type": "image_url", "image_url": {"url": img}}]}])
    m = r.choices[0].message; txt = (m.content or getattr(m, "reasoning_content", None) or "").split("</think>")[-1].replace("```json", "").replace("```", "")
    j = _last_json(txt); return j, round(time.time() - t0, 1)


def check_route(farm: FreeFarm, wps):
    """Validate HERMES's drawing on the robot's grid: drivable segments as drawn, repairs with A*, coverage of every aisle, end at the dock."""
    fs = farm.fs; wps = [tuple(map(float, w)) for w in wps if isinstance(w, (list, tuple)) and len(w) == 2]
    if len(wps) < 2:
        return {"ok": False, "why": "no route"}
    ok_segs = 0; repaired = 0; poly = [wps[0]]; failed = 0
    for a, b in zip(wps, wps[1:]):
        if fs.free_segment(np.array(a), np.array(b), clearance=0.30):
            ok_segs += 1; poly.append(b)
        else:
            leg = fs.astar(a, b)
            if leg is None:
                failed += 1; poly.append(b)
            else:
                repaired += 1; poly.extend(leg[1:])
    P = np.array(poly); L = float(sum(math.dist(p, q) for p, q in zip(poly, poly[1:])))
    from scipy.spatial import cKDTree
    t = cKDTree(P); cov = {}
    for aid, stops in farm.stops.items():
        cov[aid] = round(float((t.query(stops)[0] < 0.6).mean()), 3)
    drawn_m = float(sum(math.dist(p, q) for p, q in zip(wps, wps[1:])))
    return {"ok": failed == 0 and all(v >= 0.98 for v in cov.values()) and math.dist(P[-1], farm.dock[:2]) < 2.0, "segments": len(wps) - 1, "drivable_as_drawn": ok_segs, "repaired_by_planner": repaired, "unrepairable": failed,
            "aisle_coverage": cov, "ends_at_dock": math.dist(P[-1], farm.dock[:2]) < 2.0, "drawn_length_m": round(drawn_m, 1), "driven_length_m": round(L, 1), "polyline": [[round(x, 2), round(y, 2)] for x, y in poly]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", type=Path, required=True); ap.add_argument("--rec", type=Path, required=True); ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    fs, m, theta, aisles, dock = build_structure(a.scans, a.rec); farm = FreeFarm(fs, aisles, dock)
    mp = MissionPlanner.__new__(MissionPlanner); mp.fs = fs; mp.dock = farm.dock; mp.aisles = farm.aisles; mp.astar_in_aisle = farm.astar_in_aisle
    cases = [json.loads(l) for l in open(a.cases)][: a.n]; out = []
    for i, c in enumerate(cases):
        start = tuple(c["truth"][:2]); img = picture_with_axes(farm, start)
        try:
            j, secs = ask_hermes(farm, start, img); chk = check_route(farm, j.get("waypoints", [])); chk["reason"] = str(j.get("reason", ""))[:160]; chk["aisle_order"] = j.get("aisle_order")
        except Exception as e:
            chk = {"ok": False, "why": f"model: {str(e)[:80]}"}; secs = None
        rpp = mp.rpp_order(start); ref = mp.plan(rpp["order"], start) if rpp["ok"] else None
        row = {"case": i + 1, "start_xy": [round(start[0], 2), round(start[1], 2)], "hermes": chk, "hermes_seconds": secs, "rpp_astar_length_m": ref["total_m"] if ref else None}
        out.append(row); print(json.dumps({k: (v if k != "hermes" else {kk: vv for kk, vv in v.items() if kk != "polyline"}) for k, v in row.items()}), flush=True)
    good = [r for r in out if r["hermes"].get("ok")]
    segs = sum(r["hermes"].get("segments", 0) for r in out); drawn_ok = sum(r["hermes"].get("drivable_as_drawn", 0) for r in out); rep = sum(r["hermes"].get("repaired_by_planner", 0) for r in out)
    summ = {"cases": len(out), "route_complete_and_drivable_after_check": len(good), "segments_drawn": segs, "drivable_as_drawn": drawn_ok, "repaired_by_planner": rep, "unrepairable": sum(r["hermes"].get("unrepairable", 0) for r in out),
            "length_vs_rpp_astar_median_ratio": round(float(np.median([r["hermes"]["driven_length_m"] / r["rpp_astar_length_m"] for r in good if r["rpp_astar_length_m"]])), 2) if good else None,
            "hermes_seconds_median": round(float(np.median([r["hermes_seconds"] for r in out if r["hermes_seconds"]])), 1) if any(r["hermes_seconds"] for r in out) else None,
            "model": os.environ.get("EYES2_MODEL", "nvidia/Nemotron-3.5-Super-VL")}
    print(json.dumps(summ, indent=1)); (a.out / "hermes_draws.json").write_text(json.dumps({"summary": summ, "cases": out}, indent=1))
    # figure: what HERMES drew (dashed) and what the robot would drive after the check (solid), first 10 cases
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = min(10, len(out)); fig, axs = plt.subplots(1, n, figsize=(2.3 * n, 5.8), facecolor="#fff")
    for ax, r in zip(np.atleast_1d(axs), out[:n]):
        ax.scatter(m[:, 0], m[:, 1], s=0.3, c="#6e7681", lw=0); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.set_xlim(-8, 17); ax.set_ylim(-44, 13)
        h = r["hermes"]
        if h.get("polyline"):
            P = np.array(h["polyline"]); ax.plot(P[:, 0], P[:, 1], c="#4cc9f0", lw=1.6)
        ax.plot(r["start_xy"][0], r["start_xy"][1], "o", c="#000", ms=6); ax.plot(dock[0], dock[1], "*", c="#ffd33d", ms=12, mec="#000")
        ax.set_title(f"#{r['case']} {'✓' if h.get('ok') else '✗'} drawn ok {h.get('drivable_as_drawn', 0)}/{h.get('segments', 0)} seg\nrepaired {h.get('repaired_by_planner', 0)} · {h.get('driven_length_m', 0):.0f} m vs {r['rpp_astar_length_m'] or 0:.0f} m", fontsize=7.5)
    fig.suptitle(f"HERMES draws the survey route from the map picture; the robot checks it on its grid: complete and drivable in {len(good)} of {len(out)}; "
                 f"{drawn_ok} of {segs} drawn segments drivable as drawn, {rep} repaired by the planner", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(a.out / "hermes_draws.png", dpi=120); print("figure", a.out / "hermes_draws.png")


if __name__ == "__main__":
    main()
