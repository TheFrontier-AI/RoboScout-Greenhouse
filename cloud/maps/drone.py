"""Put a drone cloud-SLAM result (cloud/mapping/run_job.py: topdown.png + summary.json with ground basis e1/e2) into
the site frame from ≥ 2 anchor correspondences (a pixel on topdown.png ↔ known site x,y, e.g. two greenhouse doors),
and attach the top-down image to a site map as a background layer (and the drone track as a landmark polyline).
The layer does not change the route graph, so it does not clear a confirmation; nodes/edges traced on it later do.
  python -m cloud.maps.drone --slam runs/drone_slam --pair 410,388=195.07,156.67 --pair 702,620=255.22,79.59 \
      --site data/site_map_v18_draft.json --out data/site_map_v18_draft.json"""
from __future__ import annotations
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cloud.mapping.align_to_house import umeyama_2d  # noqa: E402
from r1_nav.site_map import SiteMap, now_iso  # noqa: E402


def topdown_px_to_uv(px, meta: dict) -> np.ndarray:
    """topdown.png pixel (col, row) → ground-plane uv (run_job.topdown: row 0 = max e2)."""
    px = np.atleast_2d(np.asarray(px, float))
    H = meta["size_px"][1]
    return np.asarray(meta["origin_uv"], float) + meta["resolution_per_px"] * np.stack([px[:, 0], H - 1 - px[:, 1]], 1)


def align_topdown(summary: dict, pairs: list[tuple[tuple[float, float], tuple[float, float]]]) -> dict:
    """Fit uv → site similarity from (pixel, site_xy) pairs. Returns scale, rotation, translation, rmse and the
    2x3 affine pixel→site for the image layer."""
    if len(pairs) < 2:
        raise ValueError("need ≥ 2 anchor correspondences")
    meta = summary["topdown"]
    src = topdown_px_to_uv([p for p, _ in pairs], meta)
    dst = np.array([xy for _, xy in pairs], float)
    s, R, t = umeyama_2d(src, dst)
    res = (s * (R @ src.T)).T + t - dst
    # pixel → uv is affine: uv = o + r*[c, H-1-row]; compose into one 2x3 matrix pixel → site
    r, H = meta["resolution_per_px"], meta["size_px"][1]
    A_uv = np.array([[r, 0.0], [0.0, -r]])
    b_uv = np.asarray(meta["origin_uv"], float) + np.array([0.0, r * (H - 1)])
    A = s * R @ A_uv
    b = s * R @ b_uv + t
    return {"scale": float(s), "rotation_deg": math.degrees(math.atan2(R[1, 0], R[0, 0])), "R": R.tolist(),
            "t": t.tolist(), "rmse_m": round(float(np.sqrt((res ** 2).sum(1).mean())), 3), "pairs": len(pairs),
            "pixel_to_site": np.hstack([A, b[:, None]]).round(6).tolist()}


def track_to_site(trajectory: dict, summary: dict, fit: dict) -> list[list[float]]:
    meta = summary["topdown"]
    cams = np.array([f["position"] for f in trajectory["frames"]], float)
    uv = np.stack([cams @ np.asarray(meta["e1"]), cams @ np.asarray(meta["e2"])], 1)
    xy = fit["scale"] * (np.asarray(fit["R"]) @ uv.T).T + np.asarray(fit["t"])
    return xy.round(3).tolist()


def attach_topdown(site: SiteMap, slam_dir: str | Path, pairs, anchor_names: list[str] | None = None) -> tuple[SiteMap, dict]:
    slam_dir = Path(slam_dir)
    summary = json.loads((slam_dir / "summary.json").read_text())
    fit = align_topdown(summary, pairs)
    m = copy.deepcopy(site)
    W, H = summary["topdown"]["size_px"]
    A = np.asarray(fit["pixel_to_site"])
    corners = [(A @ np.array([c, r, 1.0])).round(3).tolist() for c, r in ((0, 0), (W - 1, 0), (W - 1, H - 1), (0, H - 1))]
    m.layers = [l for l in m.layers if l.get("path") != str(slam_dir / "topdown.png")]
    m.layers.append({"kind": "drone_topdown", "path": str(slam_dir / "topdown.png"), "size_px": [W, H],
                     "pixel_to_site": fit["pixel_to_site"], "corners_site": corners, "rmse_m": fit["rmse_m"],
                     "model": summary.get("model"), "anchors": anchor_names or []})
    if (slam_dir / "trajectory.json").exists():
        track = track_to_site(json.loads((slam_dir / "trajectory.json").read_text()), summary, fit)
        m.landmarks = [l for l in m.landmarks if l.get("id") != "drone-track"]
        m.landmarks.append({"id": "drone-track", "kind": "polyline", "label": "drone flight path", "xy": track})
    m.meta["anchors_used"] = sorted(set(m.meta.get("anchors_used", [])) | set(anchor_names or []))
    m.meta.setdefault("history", []).append({"op": "attach_layer", "kind": "drone_topdown", "at": now_iso(),
                                             "rmse_m": fit["rmse_m"], "revision": m.revision})
    return m, fit


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slam", required=True, type=Path, help="run_job.py output dir (summary.json, topdown.png)")
    ap.add_argument("--pair", action="append", default=[], help="COL,ROW=X,Y (topdown pixel = site metres); ≥ 2")
    ap.add_argument("--name", action="append", default=[], help="anchor names, same order as --pair")
    ap.add_argument("--site", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    pairs = []
    for p in a.pair:
        l, r = p.split("=")
        pairs.append((tuple(map(float, l.split(","))), tuple(map(float, r.split(",")))))
    m, fit = attach_topdown(SiteMap.load(a.site), a.slam, pairs, a.name)
    m.save(a.out)
    print(json.dumps({k: fit[k] for k in ("scale", "rotation_deg", "rmse_m", "pairs")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
