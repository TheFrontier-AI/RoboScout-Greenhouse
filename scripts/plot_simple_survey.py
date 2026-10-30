#!/usr/bin/env python3
"""The plain figure of a put-down survey run (scripts/freenav_mission_eval.py output): per case one blue line = the
path the robot really drove, the start, the dock, and - for cases that started with a wrong guess - one red circle
where it worked out its real position. The panel title is the verdict.

  python3 scripts/plot_simple_survey.py --cases <dir>/mission_cases.json --rec <run>/rec --out <png>"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PATH, FIX, DOCK, MAP = "#4cc9f0", "#ff4d4d", "#ffd33d", "#6e7681"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True); ap.add_argument("--rec", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); teach = a.rec / "teach"
    cases = json.load(open(a.cases)); st = json.load(open(teach / "free_nav" / "structure.json")); dock = st["dock"]
    look = np.load(teach / "offpath2" / "map_lookaround_xy.npy")
    n = len(cases); cols = 10; rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(2.3 * cols, 5.6 * rows), facecolor="#ffffff")
    for ax in axs.flat: ax.axis("off")
    ok_n = 0
    for ax, r in zip(axs.flat, cases):
        ax.axis("on"); ax.set_facecolor("#ffffff"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#cccccc")
        ax.scatter(look[:, 0], look[:, 1], s=0.3, c=MAP, lw=0, rasterized=True)
        tp = np.array(r["true_path"])[:, :2]; ax.plot(tp[:, 0], tp[:, 1], c=PATH, lw=1.6, zorder=3)
        t = r["truth"]; ax.plot(t[0], t[1], "o", c="#000", ms=7, zorder=6); ax.annotate("start", (t[0], t[1]), fontsize=7, xytext=(5, 4), textcoords="offset points", color="#000")
        ax.plot(dock[0], dock[1], marker="*", ms=14, c=DOCK, mec="#000", mew=0.6, zorder=6); ax.annotate("dock", dock[:2], fontsize=7, xytext=(6, 5), textcoords="offset points", color="#000")
        be = np.array(r["belief_err"]); m = min(len(tp), len(be))
        if be[0] >= 1.0 and r.get("corrected_at_m") is not None:
            cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(tp[:m, 0]), np.diff(tp[:m, 1])))]); i = min(int(np.searchsorted(cum, r["corrected_at_m"])), m - 1)
            ax.plot(tp[i, 0], tp[i, 1], "o", ms=9, mfc="none", mec=FIX, mew=2, zorder=7)
            ax.annotate(f"knew where it was\nafter {r['corrected_at_m']:.0f} m", (tp[i, 0], tp[i, 1]), fontsize=6.5, color=FIX, xytext=(6, -4), textcoords="offset points")
        cov = r["coverage"]; lab = r.get("aisle_steps_logged_right") or 0
        ok = r["outcome"] == "mission_done" and all(v >= 0.98 for v in cov.values()) and lab >= 0.95; ok_n += ok
        if ok: verdict, col = "✓ both rows surveyed, records right", "#1a7f37"
        elif r["outcome"] == "mission_done" and all(v >= 0.98 for v in cov.values()): verdict, col = "△ both rows surveyed, wrong house name", "#b35900"
        elif r["outcome"] == "mission_done": verdict, col = f"△ finished, rows covered {' / '.join(f'{v * 100:.0f}%' for v in cov.values())}", "#b35900"
        else: verdict, col = "✗ did not finish", "#c0392b"
        start = "started knowing where it was" if be[0] < 1.0 else f"started believing a place {be[0]:.0f} m away"
        ax.set_title(f"#{r['case']} {r['tag']} · {r['metres']:.0f} m\n{verdict}\n{start}", fontsize=7.4, color=col, pad=3)
        ax.set_xlim(-8, 17); ax.set_ylim(-44, 13); ax.set_aspect("equal")
    leg = [Line2D([], [], marker="o", ls="", c="#000", ms=8, label="●  start = where the robot was really put down"),
           Line2D([], [], c=PATH, lw=3, label="━  the path the robot really drove"),
           Line2D([], [], marker="o", ls="", mfc="none", mec=FIX, ms=9, mew=2, label="○  where it worked out its real position (cases that started with a wrong guess)"),
           Line2D([], [], marker="*", ls="", c=DOCK, ms=14, mec="#000", label="★  dock (end of the survey)"),
           Line2D([], [], marker=".", ls="", c=MAP, ms=9, label="grey = the LiDAR map the robot built (walls, plant rows)")]
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9.5, frameon=True)
    near = [r for r in cases if r["tag"] == "near"]; far = [r for r in cases if r["tag"] == "far"]
    fig.suptitle(f"Put down anywhere, no stored route: the robot finds the rows in its own map, plans and drives the survey itself.  Result: {ok_n} of {n} surveys complete with correct records "
                 f"({len(near)} 'near' starts beside the old path, {len(far)} 'far' starts in the open; nobody tells the robot where it is)", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.035, 1, 0.985)); a.out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out, dpi=110, facecolor="#fff"); print("wrote", a.out)


if __name__ == "__main__":
    main()
