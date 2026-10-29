#!/usr/bin/env python3
"""Mapping by demonstration ("teach mode"): a person walks, pushes or drives the robot once through the farm while
the recorder runs; this script turns that recording into the maps the robot navigates with - nobody types a distance.

  python3 scripts/teach_map.py --rec <run>/rec [--out <run>/rec/teach] [--validate]

From steps.jsonl (the robot's own pose estimate + the 2-D LiDAR scan at every step; ground truth is never used):
  prior_map_xy.npy    the farm as the LiDAR saw it: scans placed with the robot's own poses, voxel-filtered - the
                      map ScanToMapLocalizer matches against (same format as the plan-derived map run_farm.py builds)
  site_map.json       skeleton of the site graph in r1_nav/site_map.py schema: dock, a door node wherever the taught
                      drive entered or left a house, edges along the taught path with their measured lengths.
                      Houses are named from the recording; the farmer renames / confirms them (farm-site-map skill)
  teach_report.json   what was built, and with --validate how well the robot localises on its own map versus the
                      plan-derived map on the same drive (fix error against the recorded truth, inlier fraction)
  teach_map.png       the map with the taught path, dock and doors"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.scan_localizer import LIDAR_AHEAD_M, ScanToMapLocalizer      # noqa: E402

N_BEAMS = 500
FORWARD_RAD = -math.pi / 2                                           # R180: robot forward sits at -pi/2 in the raw scan


def beam_angles(n: int = N_BEAMS) -> np.ndarray:
    """r1_nav convention (0 = forward, +left) for the raw R180 azimuths -pi .. pi."""
    raw = -math.pi + 2 * math.pi * np.arange(n) / n
    return (raw - FORWARD_RAD + math.pi) % (2 * math.pi) - math.pi


def scan_xy(ranges, angles: np.ndarray, max_range: float = 11.0, self_m: float = 0.25) -> np.ndarray:
    r = np.array([v if v is not None else np.inf for v in ranges], float)
    keep = (r > self_m) & (r < max_range)
    return np.stack([r[keep] * np.cos(angles[keep]), r[keep] * np.sin(angles[keep])], -1)


def to_scene(pts_lidar: np.ndarray, x: float, y: float, heading_deg: float) -> np.ndarray:
    h = math.radians(heading_deg); c, s = math.cos(h), math.sin(h)
    body = pts_lidar + (LIDAR_AHEAD_M, 0.0)
    return body @ np.array([[c, -s], [s, c]]).T + (x, y)


def load_steps(rec: Path) -> list[dict]:
    rows = [json.loads(l) for l in open(rec / "steps.jsonl")]
    return [r for r in rows if r.get("lidar") and r.get("xy") and r.get("heading_deg") is not None]


def build_map(steps: list[dict], every: int = 2, cell: float = 0.05, min_hits: int = 2, max_range: float = 11.0) -> np.ndarray:
    """Accumulate every `every`-th scan in the farm frame using the robot's OWN pose, keep cells hit >= min_hits times."""
    ang = beam_angles(len(steps[0]["lidar"]))
    hits: dict[tuple[int, int], int] = {}
    for s in steps[::every]:
        pts = to_scene(scan_xy(s["lidar"], ang, max_range), s["xy"][0], s["xy"][1], s["heading_deg"])
        for cx, cy in np.floor(pts / cell).astype(int):
            hits[(cx, cy)] = hits.get((cx, cy), 0) + 1
    cells = np.array([k for k, v in hits.items() if v >= min_hits], float)
    return (cells + 0.5) * cell


def site_skeleton(steps: list[dict], source: str) -> dict:
    """Dock + a door node at every change of leg kind between outdoors and a house, edges along the taught path."""
    from r1_nav.site_map import default_meta
    nodes, edges = [], []
    xy = np.array([s["xy"] for s in steps], float)
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]     # distance along the taught path
    nodes.append({"id": "dock", "kind": "dock", "xy": [round(float(xy[0, 0]), 3), round(float(xy[0, 1]), 3)],
                  "heading_deg": round(float(steps[0]["heading_deg"]), 1), "label": "where the taught drive started"})
    inside = ("enter", "drive_in", "drive_out", "turn_around", "hold")
    last_node, last_i, prev_in, doors = "dock", 0, False, {}
    for i, s in enumerate(steps):
        now_in = s.get("leg") in inside
        if now_in != prev_in and i > 0:
            # the house this door belongs to: the aisle the robot is in / is about to be in / has just left
            ahead = next((t.get("aisle") for t in steps[i:i + 300] if t.get("leg") in ("drive_in", "drive_out") and t.get("aisle")), None)
            behind = next((t.get("aisle") for t in steps[i - 1::-1][:300] if t.get("leg") in ("drive_in", "drive_out") and t.get("aisle")), None)
            house = ((ahead if now_in else behind) or ahead or behind or f"house-{len(doors) + 1}").split("/")[0]
            near = [n for n in nodes if n["kind"] == "door" and n.get("house") == house
                    and math.hypot(n["xy"][0] - xy[i, 0], n["xy"][1] - xy[i, 1]) < 2.0]
            if near:                                  # the same door used again (revisit, second pass): reuse the node
                nid = near[0]["id"]
            else:
                doors[house] = doors.get(house, 0) + 1
                nid = f"{house}/door-{doors[house]}"
                nodes.append({"id": nid, "kind": "door", "house": house, "xy": [round(float(xy[i, 0]), 3), round(float(xy[i, 1]), 3)],
                              "heading_deg": round(float(s["heading_deg"]), 1),
                              "label": ("entered" if now_in else "left") + f" {house} here during the taught drive"})
            if nid != last_node and not any({e["a"], e["b"]} == {last_node, nid} for e in edges):
                edges.append({"a": last_node, "b": nid, "length_m": round(float(seg[i] - seg[last_i]), 2), "width_m": None,
                              "surface": "unknown", "notes": ["taught: the robot was driven along this path"], "blocked": False,
                              "provenance": "teach_map.py: path length measured by the robot's own odometry"})
            last_node, last_i = nid, i
        prev_in = now_in
    if last_node != "dock" and not any({e["a"], e["b"]} == {last_node, "dock"} for e in edges):
        edges.append({"a": last_node, "b": "dock", "length_m": round(float(seg[-1] - seg[last_i]), 2), "width_m": None,
                      "surface": "unknown", "notes": ["taught: return to the dock"], "blocked": False,
                      "provenance": "teach_map.py: path length measured by the robot's own odometry"})
    # inside a house the taught path is the aisle: record its measured length for the aisle map
    houses = {}
    for s in steps:
        if s.get("leg") in ("drive_in", "drive_out") and s.get("aisle") and s.get("metre") is not None:
            h = s["aisle"].split("/")[0]; houses.setdefault(h, {"aisles": {}})
            a = houses[h]["aisles"].setdefault(s["aisle"], {"metre_min": 1e9, "metre_max": -1e9})
            a["metre_min"] = min(a["metre_min"], s["metre"]); a["metre_max"] = max(a["metre_max"], s["metre"])
    for h in houses.values():
        for a in h["aisles"].values():
            a["length_m"] = round(a["metre_max"] - a["metre_min"], 2); a["metre_min"] = round(a["metre_min"], 2); a["metre_max"] = round(a["metre_max"], 2)
    meta = default_meta(source, scale_known=True, taught=True, taught_from="steps.jsonl",
                        note="built by scripts/teach_map.py from a taught drive; rename houses and confirm before cross-house commands")
    return {"schema_version": 1, "frame": "site_world_z_up_metres", "metadata": meta, "nodes": nodes, "edges": edges,
            "layers": [], "landmarks": [], "houses_taught": houses}


def validate(steps: list[dict], maps: dict[str, np.ndarray], every: int = 5) -> dict:
    """Localise every `every`-th scan on each map from the robot's recorded guess; score the fix against the truth."""
    ang = beam_angles(len(steps[0]["lidar"]))
    out = {}
    for name, m in maps.items():
        loc = ScanToMapLocalizer(m); errs, raw, ok_n, n = [], [], 0, 0
        for s in steps[::every]:
            if not s.get("gt_scene_pose"):
                continue
            n += 1
            pts = scan_xy(s["lidar"], ang)
            (fx, fy, fh), q = loc.localize(pts, s["xy"][0], s["xy"][1], s["heading_deg"])
            gt = s["gt_scene_pose"][:2]
            raw.append(math.hypot(s["xy"][0] - gt[0], s["xy"][1] - gt[1]))
            if q.get("ok"):
                ok_n += 1; errs.append(math.hypot(fx - gt[0], fy - gt[1]))
        e = np.array(errs); r = np.array(raw)
        out[name] = {"points": int(len(m)), "scans": n, "accepted": ok_n, "accepted_frac": round(ok_n / max(n, 1), 3),
                     "fix_rmse_m": round(float(np.sqrt((e ** 2).mean())), 3) if len(e) else None,
                     "fix_median_m": round(float(np.median(e)), 3) if len(e) else None,
                     "guess_rmse_m": round(float(np.sqrt((r ** 2).mean())), 3)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", type=Path, required=True); ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--every", type=int, default=2); ap.add_argument("--cell", type=float, default=0.05)
    ap.add_argument("--min-hits", type=int, default=2); ap.add_argument("--validate", action="store_true")
    a = ap.parse_args(); out = a.out or (a.rec / "teach"); out.mkdir(parents=True, exist_ok=True)
    steps = load_steps(a.rec)
    m = build_map(steps, a.every, a.cell, a.min_hits)
    np.save(out / "prior_map_xy.npy", m)
    np.save(out / "path_xyh.npy", np.array([[s["xy"][0], s["xy"][1], s["heading_deg"]] for s in steps], float))  # for sim.relocalize
    site = site_skeleton(steps, source="robot")          # SOURCES in r1_nav.site_map: the robot measured it
    (out / "site_map.json").write_text(json.dumps(site, indent=1, ensure_ascii=False))
    xy = np.array([s["xy"] for s in steps]); path_m = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    rep = {"rec": str(a.rec), "steps": len(steps), "taught_path_m": round(path_m, 1), "map_points": int(len(m)),
           "cell_m": a.cell, "nodes": [n["id"] for n in site["nodes"]], "edges": len(site["edges"]),
           "houses": site["houses_taught"]}
    if a.validate:
        maps = {"self_built": m}
        if (a.rec / "prior_map_xy.npy").exists():
            maps["plan_derived"] = np.load(a.rec / "prior_map_xy.npy")
        rep["localisation"] = validate(steps, maps)
    (out / "teach_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 12), dpi=120); fig.patch.set_facecolor("#0e1116"); ax.set_facecolor("#0e1116")
        ax.scatter(m[:, 0], m[:, 1], s=0.5, c="#8fa3b7", label=f"map from the robot's own LiDAR ({len(m)} pts)")
        ax.plot(xy[:, 0], xy[:, 1], c="#76b900", lw=1.0, label=f"taught path ({path_m:.0f} m)")
        for n in site["nodes"]:
            col = "#ff2fd6" if n["kind"] == "dock" else "#ffd23f"
            ax.scatter([n["xy"][0]], [n["xy"][1]], s=70, c=col, zorder=5, marker="s" if n["kind"] == "dock" else "o")
            ax.annotate(n["id"], n["xy"], color="white", fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_aspect("equal"); ax.tick_params(colors="#cfd6dd"); [sp.set_color("#3a424c") for sp in ax.spines.values()]
        ax.set_title("Mapping by demonstration: the farm as the robot recorded it on one taught drive - no distances typed",
                     color="white", fontsize=11)
        ax.legend(loc="lower right", fontsize=8, facecolor="#161b22", labelcolor="white", edgecolor="#3a424c")
        plt.tight_layout(); plt.savefig(out / "teach_map.png", facecolor=fig.get_facecolor())
    except Exception as e:
        print("no figure:", e)


if __name__ == "__main__":
    main()
