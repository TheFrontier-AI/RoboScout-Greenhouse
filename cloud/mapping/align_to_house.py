"""Put a cloud SLAM result (run_job.py output: SLAM-world frame, arbitrary scale) into house-local metres for the dashboard.
Fits a 2-D similarity (scale, rotation, translation; Umeyama) between SLAM camera centres on the ground plane and reference
positions of the robot at the same video times, then writes <out>/trajectory.json in the dashboard schema and warps
topdown.png into house bounds.

Reference = robot frames.jsonl from the hub/recording (aisle + metre per frame, time from `ts`), or any JSONL of
{"t_sec", "x", "y"} in house-local metres.
  python3 cloud/mapping/align_to_house.py --slam runs/slam --frames runs/demo/frames.jsonl --map data/tunnel09_aisle_map_v18.json --out runs/demo/slam"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def umeyama_2d(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R (2x2), t with dst ≈ s R src + t (least squares, no reflection)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (xs ** 2).sum(1).mean())
    return s, R, mu_d - s * R @ mu_s


def refs_from_frames(frames_jsonl: Path, map_path: Path, camera: str = "front") -> list[dict]:
    from r1_nav.aisle_map import AisleMap
    amap = AisleMap.load(map_path)
    rows = [json.loads(l) for l in frames_jsonl.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("camera") == camera and r.get("aisle") and r.get("metre") is not None]
    if not rows:
        return []
    t0 = min(r["ts"] for r in rows)
    out = []
    for r in rows:
        a = amap.get(r["aisle"])
        x, y, _ = a.point_at(min(max(r["metre"], 0.0), a.length_m))
        out.append({"t_sec": r["ts"] - t0, "x": x, "y": y})
    return out


def align(slam_dir: Path, refs: list[dict], out_dir: Path, max_dt: float = 0.6) -> dict:
    traj = json.loads((slam_dir / "trajectory.json").read_text())["frames"]
    meta = json.loads((slam_dir / "summary.json").read_text())["topdown"]
    e1, e2 = np.array(meta["e1"]), np.array(meta["e2"])
    cams = np.array([f["position"] for f in traj], float)
    uv = np.stack([cams @ e1, cams @ e2], 1)
    t = np.array([f["t_sec"] for f in traj], float)
    rt = np.array([r["t_sec"] for r in refs], float)
    pairs = [(i, int(np.argmin(abs(rt - ti)))) for i, ti in enumerate(t) if len(rt) and abs(rt - ti).min() <= max_dt]
    if len(pairs) < 3:
        raise ValueError(f"only {len(pairs)} time-matched frames; need ≥ 3 (check clocks / --max-dt)")
    src = uv[[i for i, _ in pairs]]
    dst = np.array([[refs[j]["x"], refs[j]["y"]] for _, j in pairs])
    s, R, tr = umeyama_2d(src, dst)
    xy = (s * (R @ uv.T)).T + tr
    rmse = float(np.sqrt(((xy[[i for i, _ in pairs]] - dst) ** 2).sum(1).mean()))
    poses = []
    for k in range(len(xy)):
        d = xy[min(k + 1, len(xy) - 1)] - xy[max(k - 1, 0)]
        poses.append({"ts": float(t[k]), "x": float(xy[k, 0]), "y": float(xy[k, 1]),
                      "heading_deg": round(math.degrees(math.atan2(d[1], d[0])), 1)})
    # topdown.png: pixel (c, r) ↔ uv = origin + res*(c, H-1-r); map its corners to house xy
    W, H = meta["size_px"]; res = meta["resolution_per_px"]; o = np.array(meta["origin_uv"], float)
    px = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], float)        # topdown.png pixels (col, row)
    px_uv = o + res * np.stack([px[:, 0], H - 1 - px[:, 1]], 1)                  # row 0 = max e2
    px_xy = (s * (R @ px_uv.T)).T + tr
    lo, hi = np.minimum(px_xy.min(0), xy.min(0)), np.maximum(px_xy.max(0), xy.max(0))
    out_dir.mkdir(parents=True, exist_ok=True)
    src_png = slam_dir / "topdown.png"
    if src_png.exists():
        import cv2
        img = cv2.imread(str(src_png), cv2.IMREAD_UNCHANGED)
        out_res = max(res * s, 0.02)                                             # metres per output pixel
        OW, OH = int(math.ceil((hi[0] - lo[0]) / out_res)) + 1, int(math.ceil((hi[1] - lo[1]) / out_res)) + 1
        dst = np.stack([(px_xy[:, 0] - lo[0]) / out_res, OH - 1 - (px_xy[:, 1] - lo[1]) / out_res], 1)  # top row = max y
        A = cv2.getAffineTransform(np.float32(px[:3]), np.float32(dst[:3]))
        cv2.imwrite(str(out_dir / "topdown.png"), cv2.warpAffine(img, A, (OW, OH)))
    report = {"bounds_local": {"min": lo.round(3).tolist(), "max": hi.round(3).tolist()}, "poses": poses,
              "alignment": {"scale": s, "rotation_deg": math.degrees(math.atan2(R[1, 0], R[0, 0])), "rmse_m": round(rmse, 3),
                            "matched": len(pairs)}}
    (out_dir / "trajectory.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slam", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--frames", help="robot frames.jsonl (aisle+metre+ts)"); ap.add_argument("--map", default="data/tunnel09_aisle_map_v18.json")
    ap.add_argument("--refs", help="JSONL of {t_sec,x,y} house-local"); ap.add_argument("--max-dt", type=float, default=0.6)
    a = ap.parse_args()
    refs = [json.loads(l) for l in Path(a.refs).read_text().splitlines() if l.strip()] if a.refs else \
        refs_from_frames(Path(a.frames), Path(a.map))
    r = align(Path(a.slam), refs, Path(a.out), a.max_dt)
    print(json.dumps(r["alignment"]))
