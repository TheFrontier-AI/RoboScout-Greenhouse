"""Cloud mapping batch job: video(s) -> frames (ffmpeg) -> feed-forward 3D reconstruction on one GPU
-> trajectory.json, points.ply, topdown.png, occupancy.png, summary.json.

primary  = MapAnything (facebook/map-anything, one memory-efficient pass over all frames, ~metric scale)
fallback = VGGT-1B     (facebook/VGGT-1B, single pass, frames capped by --fallback-max-frames)

Importable without a GPU / torch: everything heavy is imported inside the reconstruction functions."""
from __future__ import annotations
import argparse
import json
import math
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

PRIMARY_MODEL = "facebook/map-anything"      # CC-BY-NC 4.0; "facebook/map-anything-apache" for Apache 2.0
FALLBACK_MODEL = "facebook/VGGT-1B"          # non-commercial; "facebook/VGGT-1B-Commercial" is gated


# ---------------------------------------------------------------- arguments / sampling (pure, unit-tested)

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video", nargs="+", required=True, help="local path, http(s):// or s3:// (one or more)")
    p.add_argument("--out", required=True, help="output directory (e.g. an s3 volume mount)")
    p.add_argument("--fps", type=float, default=2.0, help="sampling rate per video before the frame cap")
    p.add_argument("--max-frames", type=int, default=600, help="cap over all videos together")
    p.add_argument("--method", choices=("primary", "fallback"), default="primary")
    p.add_argument("--fisheye", action="store_true", help="input is equidistant fisheye: reproject a pinhole centre crop")
    p.add_argument("--fisheye-fov", type=float, default=175.0, help="fisheye input diagonal-ish FOV, degrees")
    p.add_argument("--crop-fov", type=float, default=100.0, help="horizontal FOV of the pinhole crop, degrees")
    p.add_argument("--width", type=int, default=0, help="resize extracted frames to this width (0 = keep)")
    p.add_argument("--fallback-max-frames", type=int, default=300, help="VGGT single-pass frame cap")
    p.add_argument("--up", choices=("cam-y", "cam-z"), default="cam-y",
                   help="gravity hint: cam-y = forward-looking cameras (robot), cam-z = nadir drone camera")
    p.add_argument("--max-points", type=int, default=4_000_000)
    p.add_argument("--conf-percentile", type=float, default=30.0, help="drop the lowest-confidence N%% of points")
    p.add_argument("--grid-px", type=int, default=1024, help="long side of the top-down image")
    p.add_argument("--frames-only", action="store_true", help="stop after frame extraction (smoke test)")
    a = p.parse_args(argv)
    if a.fps <= 0 or a.max_frames < 1:
        p.error("--fps must be > 0 and --max-frames >= 1")
    return a


def plan_fps(durations_s, fps, max_frames):
    """One sampling rate for all videos so that sum(duration * rate) <= max_frames (never above `fps`)."""
    total = sum(max(d, 0.0) for d in durations_s)
    if total <= 0:
        return fps
    return min(fps, max_frames / total)


def uniform_subsample(n, k):
    """k indices spread evenly over range(n), first and last included (all of them when k >= n)."""
    if k >= n:
        return list(range(n))
    if k <= 1:
        return [0][:k]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def fisheye_filter(fisheye_fov, crop_fov, width, height):
    """ffmpeg v360 filter: equidistant fisheye -> pinhole crop of `crop_fov` horizontal FOV (same size)."""
    vfov = math.degrees(2 * math.atan(math.tan(math.radians(crop_fov) / 2) * height / width))
    return (f"v360=input=fisheye:output=flat:ih_fov={fisheye_fov:g}:iv_fov={fisheye_fov:g}"
            f":h_fov={crop_fov:g}:v_fov={vfov:.3f}:w={width}:h={height}:interp=cubic")


def pinhole_intrinsics(crop_fov, width, height):
    f = (width / 2) / math.tan(math.radians(crop_fov) / 2)
    return [[f, 0.0, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]]


def frame_filter(rate, args, width, height):
    chain = [f"fps={rate:.6f}"]
    if args.fisheye:
        chain.append(fisheye_filter(args.fisheye_fov, args.crop_fov, width, height))
    if args.width:
        chain.append(f"scale={args.width}:-2")
    return ",".join(chain)


# ---------------------------------------------------------------- inputs / frames

def fetch(src, dst_dir):
    """Return a local path for `src`; downloads http(s) and s3 (boto3, AWS_* env + AWS_ENDPOINT_URL) inputs."""
    if src.startswith(("http://", "https://")):
        dst = Path(dst_dir) / Path(src.split("?")[0]).name
        urllib.request.urlretrieve(src, dst)
        return dst
    if src.startswith("s3://"):
        import boto3
        bucket, _, key = src[5:].partition("/")
        dst = Path(dst_dir) / Path(key).name
        boto3.client("s3").download_file(bucket, key, str(dst))
        return dst
    return Path(src)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:format=duration", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    return float(j["format"]["duration"]), int(s["width"]), int(s["height"])


def extract_frames(videos, args, frame_dir):
    """Extract frames of every video at one shared rate, then cap the total. Returns list of frame records."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")
    frame_dir = Path(frame_dir)
    shutil.rmtree(frame_dir, ignore_errors=True)       # stale frames would be globbed in
    frame_dir.mkdir(parents=True)
    infos = [probe(v) for v in videos]
    rate = plan_fps([d for d, _, _ in infos], args.fps, args.max_frames)
    frames = []
    for vi, (video, (_, w, h)) in enumerate(zip(videos, infos)):
        pattern = frame_dir / f"v{vi:02d}_%05d.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", frame_filter(rate, args, w, h),
                        "-q:v", "2", str(pattern)], check=True)
        for k, img in enumerate(sorted(frame_dir.glob(f"v{vi:02d}_*.jpg"))):
            frames.append({"video": vi, "source": str(video), "t_sec": round(k / rate, 3), "image": img.name})
    keep = set(uniform_subsample(len(frames), args.max_frames))
    for i, f in enumerate(frames):
        if i not in keep:
            (frame_dir / f["image"]).unlink()
    frames = [f for i, f in enumerate(frames) if i in keep]
    return frames, {"sample_fps": rate, "videos": [{"path": str(v), "duration_s": d, "width": w, "height": h}
                                                     for v, (d, w, h) in zip(videos, infos)]}


# ---------------------------------------------------------------- reconstruction (GPU)

def _torch_device():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for reconstruction")
    return torch, "cuda", torch.cuda.get_device_name(0)


def run_primary(paths):
    """MapAnything, images only. Returns (cam2world Nx4x4, points Mx3, colors Mx3 uint8, conf M)."""
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch, dev, _ = _torch_device()
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images
    model = MapAnything.from_pretrained(PRIMARY_MODEL).to(dev)
    views = load_images([str(p) for p in paths])
    with torch.no_grad():
        preds = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                            apply_mask=True, mask_edges=True, apply_confidence_mask=False)
    poses, pts, cols, conf = [], [], [], []
    for p in preds:
        poses.append(p["camera_poses"][0].float().cpu().numpy())
        m = p["mask"][0].squeeze(-1).bool().cpu().numpy() if "mask" in p else None
        xyz = p["pts3d"][0].float().cpu().numpy()
        rgb = p["img_no_norm"][0].float().cpu().numpy()
        c = p["conf"][0].float().cpu().numpy()
        m = np.ones(c.shape, bool) if m is None else m
        pts.append(xyz[m]); cols.append(rgb[m]); conf.append(c[m])
    return _pack(poses, pts, cols, conf)


def run_fallback(paths):
    """VGGT-1B single pass (depth branch unprojected, as recommended by the authors)."""
    torch, dev, _ = _torch_device()
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT.from_pretrained(FALLBACK_MODEL).to(dev).eval()
    images = load_and_preprocess_images([str(p) for p in paths]).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        out = model(images)
    extr, intr = pose_encoding_to_extri_intri(out["pose_enc"], images.shape[-2:])
    extr, intr = extr[0].float().cpu().numpy(), intr[0].float().cpu().numpy()
    depth = out["depth"][0].float().cpu().numpy()
    conf = out["depth_conf"][0].float().cpu().numpy()
    xyz = unproject_depth_map_to_point_map(depth, extr, intr)
    rgb = images.permute(0, 2, 3, 1).float().cpu().numpy()
    poses = []
    for e in extr:                                  # camera-from-world 3x4 -> world-from-camera 4x4
        T = np.eye(4); T[:3] = e
        poses.append(np.linalg.inv(T))
    return _pack(poses, list(xyz), list(rgb), list(conf))


def _pack(poses, pts, cols, conf):
    pts = np.concatenate([np.asarray(p).reshape(-1, 3) for p in pts]).astype(np.float32)
    cols = np.concatenate([np.asarray(c).reshape(-1, 3) for c in cols])
    if cols.max(initial=0) <= 1.0:
        cols = cols * 255.0
    conf = np.concatenate([np.asarray(c).reshape(-1) for c in conf]).astype(np.float32)
    ok = np.isfinite(pts).all(1)
    return np.stack(poses).astype(np.float64), pts[ok], np.clip(cols[ok], 0, 255).astype(np.uint8), conf[ok]


# ---------------------------------------------------------------- outputs (pure numpy, unit-tested)

def filter_points(pts, cols, conf, conf_percentile, max_points, seed=0):
    if len(pts) and conf_percentile > 0:
        keep = conf >= np.percentile(conf, conf_percentile)
        pts, cols = pts[keep], cols[keep]
    if len(pts) > max_points:
        idx = np.random.default_rng(seed).choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    return pts, cols


def write_ply(path, pts, cols):
    """Binary little-endian PLY with float xyz + uchar rgb (no plyfile dependency)."""
    rec = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["red"], rec["green"], rec["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(pts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())


def up_vector(poses, mode="cam-y"):
    """World 'up' from camera orientations (OpenCV cams: +y down, +z forward)."""
    axis = poses[:, :3, 1] if mode == "cam-y" else poses[:, :3, 2]
    v = -axis.mean(0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def ground_basis(up, forward=None):
    """Orthonormal (e1, e2, up) with e1 = `forward` projected on the ground plane (first camera heading),
    so the top-down image shows the start heading pointing right."""
    f = np.zeros(3) if forward is None else forward - (forward @ up) * up
    if np.linalg.norm(f) < 1e-6:
        ref = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
        f = np.cross(ref, up)
    e1 = f / np.linalg.norm(f)
    return e1, np.cross(up, e1), up


def topdown(pts, poses, up, grid_px=1024):
    """Max-height grid in the ground plane. Returns (height HxW float, nan = empty), occupancy uint8
    (255 obstacle, 128 free, 0 unknown), traj pixel coords, metadata."""
    e1, e2, n = ground_basis(up, poses[0, :3, 2] if len(poses) else None)
    uv, h = np.stack([pts @ e1, pts @ e2], 1), pts @ n
    cams = poses[:, :3, 3]
    cuv = np.stack([cams @ e1, cams @ e2], 1)
    allxy = np.concatenate([uv, cuv]) if len(uv) else cuv
    lo = np.minimum(np.percentile(allxy, 0.5, 0), cuv.min(0))      # robust to outliers, keeps the path
    hi = np.maximum(np.percentile(allxy, 99.5, 0), cuv.max(0))
    span = max(float((hi - lo).max()), 1e-6)
    res = span / (grid_px - 1)
    W, H = (np.ceil((hi - lo) / res).astype(int) + 1)
    hmap = np.full((H, W), -np.inf, np.float32)
    ij = np.floor((uv - lo) / res).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
    np.maximum.at(hmap, (H - 1 - ij[ok, 1], ij[ok, 0]), h[ok])
    hmap[~np.isfinite(hmap)] = np.nan
    floor = float(np.nanpercentile(h, 5)) if len(h) else 0.0
    top = float(np.nanpercentile(h, 95)) if len(h) else 1.0
    thresh = floor + 0.15 * max(top - floor, 1e-6)  # relative: VGGT scale is arbitrary
    occ = np.zeros((H, W), np.uint8)
    seen = np.isfinite(hmap)
    occ[seen] = 128
    occ[seen & (np.nan_to_num(hmap, nan=-np.inf) > thresh)] = 255
    cpx = np.stack([(cuv[:, 0] - lo[0]) / res, H - 1 - (cuv[:, 1] - lo[1]) / res], 1)
    meta = {"resolution_per_px": res, "origin_uv": lo.tolist(), "floor_h": floor, "obstacle_h": thresh,
            "up": n.tolist(), "e1": e1.tolist(), "e2": e2.tolist(), "size_px": [int(W), int(H)]}
    return hmap, occ, cpx, meta


def save_topdown_png(path, hmap, cpx):
    import cv2
    valid = np.isfinite(hmap)
    norm = np.zeros(hmap.shape, np.uint8)
    if valid.any():
        lo, hi = np.percentile(hmap[valid], [2, 98])
        norm[valid] = np.clip((hmap[valid] - lo) / max(hi - lo, 1e-9) * 254 + 1, 1, 255).astype(np.uint8)
    img = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    img[~valid] = 0
    pts = np.round(cpx).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, (255, 255, 255), 2)
    if len(pts):
        cv2.circle(img, tuple(int(v) for v in pts[0, 0]), 6, (0, 255, 0), -1)
    cv2.imwrite(str(path), img)


def trajectory_records(frames, poses):
    out = []
    for f, T in zip(frames, poses):
        out.append({**f, "T_world_cam": np.round(T, 6).tolist(), "position": np.round(T[:3, 3], 6).tolist()})
    return out


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


# ---------------------------------------------------------------- main

def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    out = Path(args.out)
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    videos = [fetch(v, out / "inputs") for v in args.video]
    frames, finfo = extract_frames(videos, args, out / "frames")
    t_frames = time.time() - t0
    summary = {"method": args.method, "frames": len(frames), **finfo, "fisheye": args.fisheye,
               "args": vars(args), "runtime_s": {"frames": round(t_frames, 2)}}
    if args.fisheye:
        w, h = finfo["videos"][0]["width"], finfo["videos"][0]["height"]
        if args.width:
            w, h = args.width, int(round(h * args.width / w / 2) * 2)
        summary["pinhole_K"] = pinhole_intrinsics(args.crop_fov, w, h)
    if args.frames_only:
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: summary[k] for k in ("frames", "sample_fps")}))
        return 0
    if args.method == "fallback" and len(frames) > args.fallback_max_frames:
        frames = [frames[i] for i in uniform_subsample(len(frames), args.fallback_max_frames)]
    paths = [out / "frames" / f["image"] for f in frames]
    t1 = time.time()
    poses, pts, cols, conf = (run_primary if args.method == "primary" else run_fallback)(paths)
    t_recon = time.time() - t1
    pts, cols = filter_points(pts, cols, conf, args.conf_percentile, args.max_points)
    write_ply(out / "points.ply", pts, cols)
    (out / "trajectory.json").write_text(json.dumps(
        {"convention": "T_world_cam, OpenCV camera (x right, y down, z forward), world = first camera",
         "frames": trajectory_records(frames, poses)}, indent=1))
    up = up_vector(poses, args.up)
    hmap, occ, cpx, tmeta = topdown(pts, poses, up, args.grid_px)
    save_topdown_png(out / "topdown.png", hmap, cpx)
    import cv2
    cv2.imwrite(str(out / "occupancy.png"), occ)
    summary.update({"frames_reconstructed": len(frames), "points": int(len(pts)), "gpu": gpu_name(),
                    "model": PRIMARY_MODEL if args.method == "primary" else FALLBACK_MODEL, "topdown": tmeta})
    summary["runtime_s"].update({"reconstruction": round(t_recon, 2), "total": round(time.time() - t0, 2)})
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("method", "frames_reconstructed", "points", "gpu", "runtime_s")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
