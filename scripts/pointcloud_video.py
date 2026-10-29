"""Presentation video of a 3-D point-cloud map growing as the robot drives: chase camera behind the current position,
points revealed in route order, the estimated path drawn in. Pure numpy renderer (painter's algorithm), no GPU.

  python3 scripts/pointcloud_video.py --ply map.ply --reveal map.reveal.npy --path map.path.npy --title "cuVSLAM" --out cuvslam_3d.mp4"""
from __future__ import annotations
import argparse, math, subprocess, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])


def read_ply(p: Path):
    raw = open(p, "rb").read(); i = raw.index(b"end_header\n") + 11; a = np.frombuffer(raw[i:], dtype=dt)
    return np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float32), np.stack([a["r"], a["g"], a["b"]], 1)


def look_at(eye, target, up=(0, 0, 1.0)):
    f = target - eye; f /= np.linalg.norm(f); r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
    return np.stack([r, -u, f])                      # rows: camera x (right), y (down), z (forward)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", type=Path, required=True); ap.add_argument("--reveal", type=Path, required=True); ap.add_argument("--path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--title", default=""); ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--fps", type=int, default=12); ap.add_argument("--size", default="1280x720"); ap.add_argument("--back", type=float, default=4.0)
    ap.add_argument("--up", type=float, default=2.6); ap.add_argument("--fov", type=float, default=70.0); ap.add_argument("--splat", type=int, default=2)
    ap.add_argument("--end-orbit", type=int, default=120, help="extra frames orbiting the finished map")
    a = ap.parse_args()
    W, H = (int(v) for v in a.size.split("x")); f = (W / 2) / math.tan(math.radians(a.fov / 2))
    P, C = read_ply(a.ply); rv = np.load(a.reveal); path = np.load(a.path)
    n_steps = int(rv.max()) + 1; path = path[:n_steps] if len(path) >= n_steps else np.vstack([path, np.repeat(path[-1:], n_steps - len(path), 0)])
    ctr = path.mean(0); ext = max(np.ptp(path[:, 0]), np.ptp(path[:, 1]))
    tmp = Path(tempfile.mkdtemp(prefix="pcv_")); font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26) if Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf").exists() else ImageFont.load_default()
    ks = np.linspace(0, n_steps - 1, a.frames).astype(int); total = len(ks) + a.end_orbit
    head_s = None
    for fi in range(total):
        if fi < len(ks):
            k = ks[fi]; j = min(n_steps - 1, k + 8); pos = path[k]; ahead = path[j]
            d = ahead[:2] - pos[:2]; hd = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 0.05 else (head_s if head_s is not None else 0.0)
            head_s = hd if head_s is None else head_s + 0.15 * ((hd - head_s + math.pi) % (2 * math.pi) - math.pi)
            eye = np.array([pos[0] - a.back * math.cos(head_s), pos[1] - a.back * math.sin(head_s), pos[2] + a.up])
            tgt = np.array([pos[0] + 3 * math.cos(head_s), pos[1] + 3 * math.sin(head_s), pos[2] + 0.3]); shown = rv <= k
        else:
            th = 2 * math.pi * (fi - len(ks)) / a.end_orbit; k = n_steps - 1
            eye = np.array([ctr[0] + 0.9 * ext * math.cos(th), ctr[1] + 0.9 * ext * math.sin(th), ctr[2] + 0.55 * ext]); tgt = ctr; shown = np.ones(len(P), bool)
        Rc = look_at(eye, tgt); Q = (P[shown] - eye) @ Rc.T; z = Q[:, 2]; ok = z > 0.15
        u = (f * Q[ok, 0] / z[ok] + W / 2).astype(np.int32); v = (f * Q[ok, 1] / z[ok] + H / 2).astype(np.int32); zz = z[ok]; cc = C[shown][ok]
        inb = (u >= 0) & (u < W - a.splat) & (v >= 0) & (v < H - a.splat); u, v, zz, cc = u[inb], v[inb], zz[inb], cc[inb]
        o = np.argsort(-zz); u, v, cc = u[o], v[o], cc[o]
        img = np.full((H, W, 3), (10, 12, 14), np.uint8)
        for dy in range(a.splat):
            for dx in range(a.splat):
                img[v + dy, u + dx] = cc
        im = Image.fromarray(img); dr = ImageDraw.Draw(im)
        pp = (path[:k + 1] - eye) @ Rc.T; pz = pp[:, 2]; m = pz > 0.15
        pts = [(float(f * x / zq + W / 2), float(f * y / zq + H / 2)) for (x, y, zq), mm in zip(pp, m) if mm]
        if len(pts) > 1:
            dr.line(pts, fill=(118, 185, 0), width=3)
        if pts:
            x, y = pts[-1]; dr.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(240, 80, 60))
        dr.rectangle([0, 0, W, 44], fill=(0, 0, 0)); dr.text((14, 8), f"{a.title}   {int(100 * min(1, k / max(1, n_steps - 1)))}% of route   {shown.sum():,} points", font=font, fill=(255, 255, 255))
        im.save(tmp / f"{fi:05d}.jpg", quality=90)
        if fi % 100 == 0:
            print(f"frame {fi}/{total}", flush=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(tmp / "%05d.jpg"), "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", str(a.out)], check=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
