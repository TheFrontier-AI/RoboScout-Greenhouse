#!/usr/bin/env python3
"""Small animated GIFs of the plans and the drives, so a reader sees the order of travel instead of guessing it from a
tangle of lines: (1) HERMES's plan on the scene graph, the robot moving place to place with each place named as it is
reached; (2) a real put-down survey from a wrong first guess, the robot moving along what it really drove, with the
moment it worked out its true position. Palette GIFs at ~600 px wide, well under 2 MB each.

  python3 scripts/make_plan_gifs.py --rec <run>/rec --graph <dir>/hermes_on_graph.json --survey <dir>/mission_cases.json --out <gallery dir>"""
from __future__ import annotations
import argparse, io, json, math
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def frame_to_image(fig) -> Image.Image:
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, facecolor="#fff"); buf.seek(0); return Image.open(buf).convert("RGB")


def densify(P, step=0.5):
    out = [P[0]]
    for a, b in zip(P, P[1:]):
        n = max(1, int(math.dist(a, b) / step))
        for k in range(1, n + 1):
            out.append((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n))
    return np.array(out)


def save_gif(frames, path, ms=90):
    pal = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=ms, loop=0, optimize=True)
    return path.stat().st_size


def plan_gif(graph_json, map_xy, dock, out: Path, cases=(4, 2)):
    j = json.load(open(graph_json)); g = j["scene_graph"]; rows = {c["case"]: c for c in j["cases"]}
    panels = [rows[c] for c in cases if c in rows]; frames = []
    paths = [densify(np.array(c["hermes"]["polyline"]), 0.6) for c in panels]; n = max(len(p) for p in paths) + 8
    for t in range(0, n, 2):
        fig, axs = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 5.2), facecolor="#fff")
        for ax, c, P in zip(np.atleast_1d(axs), panels, paths):
            ax.scatter(map_xy[:, 0], map_xy[:, 1], s=0.25, c="#9aa0a6", lw=0, rasterized=True); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.set_xlim(-8, 17); ax.set_ylim(-44, 13)
            for h in g["houses"]:
                cs = np.array(h["corners_xy"] + [h["corners_xy"][0]]); ax.plot(cs[:, 0], cs[:, 1], c="#e6a800", lw=1.2, alpha=0.8); ax.annotate(h["id"], cs[1] + [0.3, 1.0], fontsize=7, color="#a06a00")
                for d in h["doors"]: ax.plot(d["xy"][0], d["xy"][1], "D", c="#fff", mec="#000", ms=5)
            ax.plot(dock[0], dock[1], "*", c="#ffd33d", ms=13, mec="#000"); ax.plot(c["start_xy"][0], c["start_xy"][1], "o", c="#000", ms=6)
            k = min(t, len(P) - 1); ax.plot(P[:k + 1, 0], P[:k + 1, 1], c="#4cc9f0", lw=2); ax.plot(P[k, 0], P[k, 1], "o", c="#d62728", ms=8, mec="#000")
            # which named place is the robot at right now?
            here = ""
            for h in g["houses"]:
                for d in h["doors"]:
                    if math.dist(P[k], d["xy"]) < 2.0: here = d["id"]
                for r in h["rows"]:
                    A, B = np.array(r["end_low"]), np.array(r["end_high"]); u = (B - A) / np.linalg.norm(B - A); tt = np.dot(P[k] - A, u); off = abs(np.cross(u, P[k] - A))
                    if 0 <= tt <= np.linalg.norm(B - A) and off < 1.0 and not here: here = f"{r['id']}  {tt:.0f} m"
            if not here and math.dist(P[k], dock[:2]) < 2.0: here = "dock"
            if not here: here = "yard-high" if P[k][1] > 4 else ("yard-low" if P[k][1] < -36 else "gap between the houses")
            ax.set_title(f"put down #{c['case']}\nnow at: {here}", fontsize=8, loc="left", color="#1a7f37")
        routes = [" → ".join(str(x).replace("house-", "").replace("/row-", " row").replace("/door-", " door ").replace("yard-", "yard ").replace(" low->high", "↑").replace(" high->low", "↓") for x in c["hermes"]["route"]) for c in panels]
        import textwrap
        lines = ["HERMES plans in named places on the robot's scene graph; the robot drives the geometry"]
        for c, r in zip(panels, routes):
            lines += textwrap.wrap(f"#{c['case']}: {r}", 86)
        fig.suptitle("\n".join(lines), fontsize=6.6, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        frames.append(frame_to_image(fig)); plt.close(fig)
    frames += [frames[-1]] * 12
    return save_gif(frames, out)


def survey_gif(survey_json, map_xy, dock, out: Path, picks=None):
    """Three good surveys side by side: one that knew where it was from the start, two that started with a wrong
    guess and worked out their true position early."""
    d = json.load(open(survey_json))
    good = lambda r: r["outcome"] == "mission_done" and all(v >= 0.98 for v in r["coverage"].values()) and (r.get("aisle_steps_logged_right") or 0) >= 0.95
    if picks:
        chosen = [next(x for x in d if x["case"] == c) for c in picks]
    else:
        known = sorted([r for r in d if good(r) and r["start_belief_err_m"] < 1.0], key=lambda r: r["metres"])[:1]
        fixed = sorted([r for r in d if good(r) and r["start_belief_err_m"] >= 1.0 and r.get("corrected_at_m")], key=lambda r: r["corrected_at_m"])[:2]
        chosen = known + fixed
    tps = [np.array(r["true_path"])[:, :2] for r in chosen]; bes = [np.array(r["belief_err"]) for r in chosen]
    cums = [np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(t[:, 0]), np.diff(t[:, 1])))]) for t in tps]
    n = max(len(t) for t in tps); step = max(1, n // 80); frames = []
    for k in range(0, n, step):
        fig, axs = plt.subplots(1, len(chosen), figsize=(2.9 * len(chosen), 5.4), facecolor="#fff")
        for ax, r, tp, be, cum in zip(np.atleast_1d(axs), chosen, tps, bes, cums):
            i = min(k, len(tp) - 1); m = min(len(tp), len(be))
            ax.scatter(map_xy[:, 0], map_xy[:, 1], s=0.25, c="#9aa0a6", lw=0, rasterized=True); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.set_xlim(-8, 17); ax.set_ylim(-44, 13)
            ax.plot(dock[0], dock[1], "*", c="#ffd33d", ms=13, mec="#000"); ax.plot(tp[0, 0], tp[0, 1], "o", c="#000", ms=6); ax.annotate("start", tp[0], fontsize=7, xytext=(5, 4), textcoords="offset points")
            ax.plot(tp[:i + 1, 0], tp[:i + 1, 1], c="#4cc9f0", lw=2); ax.plot(tp[i, 0], tp[i, 1], "o", c="#d62728", ms=8, mec="#000")
            knows = be[min(i, m - 1)] < 1.0; fixed_i = int(np.searchsorted(cum, r["corrected_at_m"])) if r.get("corrected_at_m") and r["start_belief_err_m"] >= 1.0 else None
            if fixed_i is not None and i >= fixed_i:
                ax.plot(tp[fixed_i, 0], tp[fixed_i, 1], "o", ms=10, mfc="none", mec="#ff4d4d", mew=2)
            head = "knew where it was from the start" if r["start_belief_err_m"] < 1.0 else f"first guess {r['start_belief_err_m']:.0f} m off"
            state = ("done: both rows surveyed, records right" if i >= len(tp) - 1 else ("knows where it is" if knows else "still believes a wrong place"))
            if fixed_i is not None and i >= fixed_i and i < len(tp) - 1: state += f" (fixed after {r['corrected_at_m']:.0f} m)"
            ax.set_title(f"put down #{r['case']} · {head}\n{min(cum[i], cum[-1]):.0f} m driven · {state}", fontsize=7.2, loc="left", color="#1a7f37" if knows else "#b35900")
        fig.suptitle("Put down anywhere, no stored route: three surveys as they were really driven", fontsize=8.5); fig.tight_layout(rect=(0, 0, 1, 0.95))
        frames.append(frame_to_image(fig)); plt.close(fig)
    frames += [frames[-1]] * 15
    return save_gif(frames, out), [r["case"] for r in chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", type=Path, required=True); ap.add_argument("--graph", type=Path, required=True); ap.add_argument("--survey", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); st = json.load(open(a.rec / "teach" / "free_nav" / "structure.json")); dock = st["dock"]
    map_xy = np.load(a.rec / "teach" / "offpath2" / "map_lookaround_xy.npy")[::2]
    s1 = plan_gif(a.graph, map_xy, dock, a.out / "35_plan_on_scene_graph.gif"); print("35_plan_on_scene_graph.gif", round(s1 / 1e6, 2), "MB")
    s2, cases = survey_gif(a.survey, map_xy, dock, a.out / "36_survey_from_wrong_guess.gif"); print("36_survey_from_wrong_guess.gif", round(s2 / 1e6, 2), "MB", "cases", cases)


if __name__ == "__main__":
    main()
