#!/usr/bin/env python3
"""Convert run_lingbot.py's raw .npz (lingbot-map predictions for one image sequence) into the route files the pipeline
and the viewers read:  <prefix>_poses.json  (camera-to-world 4x4, centres, intrinsics)   <prefix>_small.ply
(confidence-filtered, subsampled coloured points)   <prefix>_timing.json.  Only numpy is needed.

  python3 lingbot_to_route.py route.npz <out_prefix> [--wall-s 151] [--max-points 1500000]"""
from __future__ import annotations
import argparse, json, struct
import numpy as np


def as_4x4(E: np.ndarray) -> np.ndarray:
    if E.shape[-2:] == (4, 4):
        return E.astype(np.float64)
    out = np.tile(np.eye(4), (len(E), 1, 1)); out[:, :3, :4] = E
    return out


def write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    hdr = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\n"
           "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(hdr.encode()); f.write(rec.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("prefix")
    ap.add_argument("--wall-s", type=float, default=None); ap.add_argument("--mode", default=None); ap.add_argument("--gpu", default=None)
    ap.add_argument("--max-points", type=int, default=1_500_000); ap.add_argument("--conf-quantile", type=float, default=0.5)
    a = ap.parse_args()
    z = np.load(a.npz, allow_pickle=True)
    keys = set(z.files)
    # poses: lingbot-map's postprocess stores camera-to-world (N,3,4) under "extrinsic"
    E = as_4x4(np.asarray(z["extrinsic"] if "extrinsic" in keys else z["c2w"]))
    K = np.asarray(z["intrinsic"]) if "intrinsic" in keys else np.asarray(z.get("intrinsics", np.zeros((len(E), 3, 3))))
    names = [str(s) for s in z["paths"]] if "paths" in keys else [f"{i:05d}" for i in range(len(E))]
    hw = [int(v) for v in z["image_hw"]] if "image_hw" in keys else list(np.asarray(z["images"]).shape[1:3])
    poses = {"convention": "camera-to-world, OpenCV camera axes (x right, y down, z forward)", "image_hw": hw,
             "frames": names, "centers": E[:, :3, 3].round(5).tolist(), "c2w": E.round(6).tolist(), "intrinsics": K.round(3).tolist()}
    with open(a.prefix + "_poses.json", "w") as f:
        json.dump(poses, f)
    # points: world_points (N,H,W,3) + colours from images (N,H,W,3); keep the more confident half, then subsample
    P = np.asarray(z["world_points"], dtype=np.float32).reshape(-1, 3)
    C = np.asarray(z["images"]).reshape(-1, 3) if "images" in keys else np.full((len(P), 3), 160, np.uint8)
    n_raw = len(P)
    conf_key = next((k for k in ("world_points_conf", "depth_conf", "conf") if k in keys), None)
    thr = None
    if conf_key:
        conf = np.asarray(z[conf_key], dtype=np.float32).reshape(-1)
        thr = float(np.quantile(conf, a.conf_quantile)); keep = conf >= thr
        P, C = P[keep], C[keep]
    finite = np.isfinite(P).all(1); P, C = P[finite], C[finite]
    if len(P) > a.max_points:
        idx = np.random.default_rng(0).choice(len(P), a.max_points, replace=False); P, C = P[idx], C[idx]
    write_ply(a.prefix + "_small.ply", P, C.astype(np.uint8))
    timing = {"frames": len(E), "wall_s": a.wall_s, "inference_s": float(z["t_inference"]) if "t_inference" in keys else None,
              "fps": (len(E) / float(z["t_inference"])) if "t_inference" in keys and float(z["t_inference"]) > 0 else None,
              "input_hw": hw, "mode": a.mode, "keyframe_interval": int(z["keyframe_interval"]) if "keyframe_interval" in keys else None,
              "points_raw": int(n_raw), "points": int(len(P)), "conf_key": conf_key, "conf_threshold": thr, "gpu": a.gpu}
    with open(a.prefix + "_timing.json", "w") as f:
        json.dump(timing, f, indent=1)
    print(json.dumps({k: timing[k] for k in ("frames", "wall_s", "fps", "points")}))


if __name__ == "__main__":
    main()
