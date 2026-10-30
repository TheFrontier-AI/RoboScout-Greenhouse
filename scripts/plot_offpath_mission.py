#!/usr/bin/env python3
"""Figures for scripts/offpath_mission_eval.py: the TRUE path each put-down robot drove while doing the survey,
coloured by whether it knew where it was at that moment (green = belief within 1 m, orange = wrong belief), START,
END (dock), the point where the belief was corrected, and the plant stops it really covered.

  python3 scripts/plot_offpath_mission.py --cases <dir>/mission_cases.json --rec <run>/rec --out <dir>"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

OK, BAD, DOCK, MAP, PATH = "#3fb950", "#f0883e", "#ffd33d", "#8b949e", "#58a6ff"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True); ap.add_argument("--rec", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--map", type=Path, default=None, help="map_lookaround_xy.npy (default: <rec>/teach/offpath2/map_lookaround_xy.npy)")
    ap.add_argument("--freenav", type=Path, default=None, help="free_nav output dir (free_grid.npy + structure.json): draw the robot's own drivable space and aisles, no taught drive")
    a = ap.parse_args(); teach = a.rec / "teach"; a.out.mkdir(parents=True, exist_ok=True)
    cases = json.load(open(a.cases)); path = np.load(teach / "path_xyh.npy")
    look = np.load(a.map or (teach / "offpath2" / "map_lookaround_xy.npy"))
    nodes = {n["id"]: np.array(n["xy"]) for n in json.load(open(teach / "site_map.json"))["nodes"]}
    fn = None
    if a.freenav:
        st = json.load(open(a.freenav / "structure.json")); fn = {"grid": np.load(a.freenav / "free_grid.npy"), "st": st}
        nodes = {"dock": np.array(st["dock"][:2])}

    def base(ax, s, dock_ms=12, fs=7):
        ax.scatter(look[:, 0], look[:, 1], s=s, c=MAP, lw=0, rasterized=True)
        if fn is None:
            ax.plot(path[:, 0], path[:, 1], c=PATH, lw=0.6, alpha=0.6)
        else:
            g = fn["grid"]; st = fn["st"]
            ax.imshow(g, origin="lower", extent=(st["x0"], st["x0"] + g.shape[1] * st["cell"], st["y0"], st["y0"] + g.shape[0] * st["cell"]), cmap="Greens", alpha=0.28, vmin=0, vmax=1.3, zorder=0)
            for x in st["aisles"]:
                if x["kind"] == "plant aisle": ax.plot([x["a"][0], x["b"][0]], [x["a"][1], x["b"][1]], c=DOCK, lw=1.2, alpha=0.6, zorder=1)
        for k, xy in nodes.items():
            if k == "dock":
                ax.plot(*xy, marker="*", ms=dock_ms * 1.6, c=DOCK, mec="#000", mew=0.6, zorder=10)
                ax.annotate("END = dock", xy, color=DOCK, fontsize=fs, fontweight="bold", xytext=(7, 7), textcoords="offset points", zorder=11)
            else:
                ax.plot(*xy, marker="D", ms=dock_ms * 0.5, c="#fff", mec="#000", mew=0.5, zorder=6)
        ax.set_aspect("equal"); ax.set_facecolor("#0d1117"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#30363d")

    def draw(ax, r, lw, ms, fs):
        tp = np.array(r["true_path"])[:, :2]; be = np.array(r["belief_err"])
        n = min(len(tp), len(be))
        if n >= 2:
            segs = np.stack([tp[:n - 1], tp[1:n]], axis=1); cols = [OK if e < 1.0 else BAD for e in be[1:n]]
            ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, alpha=0.95, zorder=4))
        t = r["truth"]; c0 = OK if be[0] < 1.0 else BAD
        ax.plot(t[0], t[1], "o", c=c0, ms=ms, mec="#000", mew=0.6, zorder=8)
        ax.annotate("", xy=(t[0] + 2.0 * math.cos(math.radians(t[2])), t[1] + 2.0 * math.sin(math.radians(t[2]))), xytext=(t[0], t[1]), arrowprops=dict(arrowstyle="->", color=c0, lw=1.4), zorder=9)
        ax.annotate("START", (t[0], t[1]), color=c0, fontsize=fs, fontweight="bold", xytext=(-16, 8), textcoords="offset points", zorder=9)
        if r.get("surprise_xy"):
            sx, sy = r["surprise_xy"]; ax.plot(sx, sy, marker="s", ms=ms * 0.9, c="#f0463c", mec="#000", mew=0.5, zorder=8)
        if r.get("corrected_at_m") is not None and be[0] >= 1.0:
            cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(tp[:n, 0]), np.diff(tp[:n, 1])))])
            i = int(np.searchsorted(cum, r["corrected_at_m"])); i = min(i, n - 1)
            ax.plot(tp[i, 0], tp[i, 1], marker="o", ms=ms * 1.3, mfc="none", mec="#fff", mew=1.6, zorder=9)
            ax.annotate(f"fixed @ {r['corrected_at_m']:.0f} m", (tp[i, 0], tp[i, 1]), color="#fff", fontsize=fs, xytext=(8, -4), textcoords="offset points", zorder=9)

    def title(r):
        cov = r["coverage"]; lab = r.get("aisle_steps_logged_right")
        head = {"mission_done": "complete", "stays_put": "stays put", "ran_out": "unfinished", "gave_up": "gave up"}[r["outcome"]]
        s = f"#{r['case']} {r['tag']} · {head} · {r['metres']:.0f} m"
        if r["outcome"] != "stays_put":
            s += f"\ncovered {cov['tunnel-09'] * 100:.0f}% / {cov['tunnel-10'] * 100:.0f}% · right {'-' if lab is None else f'{lab * 100:.0f}%'}"
        s += "\n" + ("knew from the start" if r["start_belief_err_m"] < 1.0 else f"wrong by {r['start_belief_err_m']:.0f} m" + (f", fixed @ {r['corrected_at_m']:.0f} m" if r["corrected_at_m"] is not None else ", never fixed"))
        return s

    # grid: one panel per case
    n = len(cases); cols = 10; rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(2.4 * cols, 6.4 * rows), facecolor="#0d1117")
    for ax in axs.flat: ax.axis("off")
    for ax, r in zip(axs.flat, cases):
        ax.axis("on"); base(ax, 0.3, 6, 6); ax.set_xlim(-8, 17); ax.set_ylim(-44, 13); draw(ax, r, 1.6, 7, 6)
        ok = r["outcome"] == "mission_done" and all(v >= 0.98 for v in r["coverage"].values()) and (r.get("aisle_steps_logged_right") or 0) >= 0.95
        ax.set_title(title(r), color=OK if ok else (BAD if r["outcome"] != "stays_put" else "#e3b341"), fontsize=6.6, pad=3)
    leg = [Line2D([], [], marker="o", ls="", c="#fff", ms=9, label="●  START = where the robot was put down; arrow = the way it faces"),
           Line2D([], [], c=OK, lw=3, label="━  the path it really drove while it knew where it was (belief within 1 m)"),
           Line2D([], [], c=BAD, lw=3, label="━  the path it really drove while believing a wrong place"),
           Line2D([], [], marker="o", ls="", mfc="none", mec="#fff", ms=10, mew=1.6, label="○  the point where it worked out where it really was"),
           Line2D([], [], marker="*", ls="", c=DOCK, ms=14, label="★  END = the dock"), (Line2D([], [], marker="s", ls="", c="#f0463c", ms=8, label="■  an obstacle that was not on the map, dropped on the planned path") if fn else Line2D([], [], marker="D", ls="", c="#fff", ms=7, label="◆  greenhouse door")),
           (Line2D([], [], c=DOCK, lw=2, alpha=0.6, label="━  plant aisles the robot found in its own map; green area = space it has seen empty") if fn else Line2D([], [], c=PATH, lw=2, label="━  the taught drive (what the mission follows)")), Line2D([], [], marker=".", ls="", c=MAP, ms=9, label="grey dots = LiDAR map")]
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9.5, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", title="how to read each panel", title_fontsize=10)
    fig.legends[0].get_title().set_color("#fff")
    near = [r for r in cases if r["tag"] == "near"]; far = [r for r in cases if r["tag"] == "far"]
    def ok_n(rs): return sum(r["outcome"] == "mission_done" and all(v >= 0.98 for v in r["coverage"].values()) and (r.get("aisle_steps_logged_right") or 0) >= 0.95 for r in rs)
    fig.suptitle((f"Put down, then survey with no stored route: the robot found the aisles in its own map, a decision-maker chose the order, every path was planned on the spot" if fn else "Put down, then do the whole survey (both houses, back to the dock)") + f": full coverage with correct records in {ok_n(near)} / {len(near)} near cases and {ok_n(far)} / {len(far)} far cases   (synthetic LiDAR on the self-built map)",
                 color="#fff", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.985)); fig.savefig(a.out / "02_mission_each_case.png", dpi=110, facecolor=fig.get_facecolor()); plt.close(fig)
    # overview: all true paths on one map
    fig, ax = plt.subplots(figsize=(11, 15), facecolor="#0d1117"); base(ax, 0.8, 14, 10)
    for r in cases: draw(ax, r, 1.0, 6, 0.1)
    for txt in [t for t in ax.texts if t.get_text() == "START" or "knows" in t.get_text()]: txt.remove()
    ax.set_title("All put-down cases: the paths really driven during the survey (green = knew where it was, orange = wrong belief)", color="#fff", fontsize=11)
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8.5, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(a.out / "01_mission_overview.png", dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
