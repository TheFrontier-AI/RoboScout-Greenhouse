"""Cinematic LiDAR view for render_cinematic.py runs: a tilted 3-D chase view of the planar scan.
Recent scans persist in the world frame and fade out; the current scan glows and shows its beam fan.
Colour = range from the sensor (near = warm, far = cool). Same frame clock as the other raw videos.

  python3 scripts/lidar_viz.py --run <render dir> --out <dir> [--fps 20] [--size 1920x1080]"""
from __future__ import annotations
import argparse, json, math, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops

RMAX, HIST = 12.0, 50                      # metres; scans kept on screen (2.5 s at 20 fps)
SENSOR_Z = 0.512                           # C1 mount height (m)
CAM = (-6.5, 0.0, 4.2)                     # robot frame: forward x, left y, up z
LOOK = (3.5, 0.0, 0.0)
BG_TOP, BG_BOT = np.array([4, 7, 14]), np.array([10, 16, 26])


def turbo(u: np.ndarray) -> np.ndarray:
    """Polynomial approximation of Google's Turbo colormap, u in [0,1] -> uint8 RGB."""
    u = np.clip(u, 0, 1)[..., None]
    c = np.concatenate([
        0.13572138 + u * (4.61539260 + u * (-42.66032258 + u * (132.13108234 + u * (-152.94239396 + u * 59.28637943)))),
        0.09140261 + u * (2.19418839 + u * (4.84296658 + u * (-14.18503333 + u * (4.27729857 + u * 2.82956604)))),
        0.10667330 + u * (12.64194608 + u * (-60.58204836 + u * (110.36276771 + u * (-89.90310912 + u * 27.34824973)))),
    ], -1)
    return (np.clip(c, 0, 1) * 255).astype(np.uint8)


class Camera:
    def __init__(self, w, h, fov_deg=55.0):
        self.w, self.h, self.f = w, h, (w / 2) / math.tan(math.radians(fov_deg) / 2)
        p, t = np.array(CAM, float), np.array(LOOK, float)
        fwd = (t - p) / np.linalg.norm(t - p)
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        self.p, self.R = p, np.stack([right, up, fwd])

    def project(self, pts: np.ndarray):
        """pts (N,3) in robot frame -> (N,2) pixels, depth, visible mask."""
        c = (pts - self.p) @ self.R.T
        z = c[:, 2]; ok = z > 0.3
        zs = np.where(ok, z, 1.0)
        uv = np.stack([self.w / 2 + self.f * c[:, 0] / zs, self.h / 2 - self.f * c[:, 1] / zs], -1)
        return uv, z, ok


def hits_world(rec):
    l = rec["lidar"]; rr = l["ranges"]; n = len(rr); ox, oy = l["origin"]
    k = np.array([i for i, r in enumerate(rr) if r is not None]); r = np.array([rr[i] for i in k], float)
    if not len(k):
        return np.zeros((0, 2)), np.zeros(0)
    a = math.radians(l["yaw_deg"]) - math.pi / 2 + (-math.pi + 2 * math.pi * k / n)
    return np.stack([ox + r * np.cos(a), oy + r * np.sin(a)], -1), r


def to_robot(xy, rec):
    ox, oy = rec["lidar"]["origin"]; h = math.radians(rec["lidar"]["yaw_deg"]) - math.pi / 2   # robot forward heading
    d = xy - [ox, oy]; c, s = math.cos(h), math.sin(h)
    return np.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c], -1)


def background(w, h):
    g = np.linspace(0, 1, h)[:, None, None]
    return Image.fromarray(np.broadcast_to(BG_TOP * (1 - g) + BG_BOT * g, (h, w, 3)).astype(np.uint8))


def draw_grid(d, cam, rec):
    """1 m ground grid fixed to the world, so it scrolls past as the robot drives."""
    ox, oy = rec["lidar"]["origin"]; R = 16
    for axis in (0, 1):
        for v in np.arange(math.floor((ox, oy)[axis]) - R, math.floor((ox, oy)[axis]) + R + 1):
            t = np.linspace(-R, R, 64)
            xy = np.stack([np.full_like(t, v), (oy, ox)[axis] + t], -1) if axis == 0 else np.stack([ox + t, np.full_like(t, v)], -1)
            p = to_robot(xy, rec); P = np.c_[p, np.zeros(len(p))]
            uv, z, ok = cam.project(P)
            dist = np.hypot(p[:, 0], p[:, 1])
            for i in range(len(t) - 1):
                if ok[i] and ok[i + 1] and dist[i] < 15:
                    a = int(46 * (1 - dist[i] / 15) ** 1.4) + 6
                    d.line([tuple(uv[i]), tuple(uv[i + 1])], fill=(a // 2, a, a + 10), width=1)


def draw_rings(d, cam):
    for r in (2, 4, 6, 8, 10, 12):
        t = np.linspace(0, 2 * math.pi, 180)
        uv, z, ok = cam.project(np.c_[r * np.cos(t), r * np.sin(t), np.zeros_like(t)])
        col = tuple(int(x) for x in turbo(np.array([r / RMAX]))[0] // 4)
        for i in range(len(t) - 1):
            if ok[i] and ok[i + 1]:
                d.line([tuple(uv[i]), tuple(uv[i + 1])], fill=col, width=1)


def draw_robot(d, cam):
    L, W, H = 0.62, 0.42, 0.55                       # R1 footprint (approx.), drawn as a wireframe box
    xs, ys = (-L / 2, L / 2), (-W / 2, W / 2)
    corners = np.array([(x, y, z) for z in (0, H) for x in xs for y in ys])
    uv, _, ok = cam.project(corners)
    edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    base = [tuple(uv[i]) for i in (0, 1, 3, 2)]
    d.polygon(base, fill=(30, 52, 8))
    for a, b in edges:
        if ok[a] and ok[b]:
            d.line([tuple(uv[a]), tuple(uv[b])], fill=(118, 185, 0), width=3)
    s, _, _ = cam.project(np.array([[0, 0, SENSOR_Z + 0.08]]))                # the sensor
    x, y = s[0]; d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(230, 255, 200))
    tip, _, _ = cam.project(np.array([[L / 2 + 0.5, 0, 0.02], [L / 2 + 0.15, 0.15, 0.02], [L / 2 + 0.15, -0.15, 0.02]]))
    d.polygon([tuple(p) for p in tip], fill=(118, 185, 0))                    # heading arrow


def render(recs, i, cam, w, h) -> Image.Image:
    cur = recs[i]
    base = background(w, h); d = ImageDraw.Draw(base)
    draw_grid(d, cam, cur); draw_rings(d, cam)

    glow = Image.new("RGB", (w, h)); g = ImageDraw.Draw(glow)
    sharp = Image.new("RGB", (w, h)); s = ImageDraw.Draw(sharp)
    for j in range(max(0, i - HIST), i + 1):                                   # oldest first, current on top
        xy, rng = hits_world(recs[j])
        if not len(rng):
            continue
        age = (i - j) / HIST; fade = (1 - age) ** 1.8
        p = to_robot(xy, cur)
        P = np.c_[p, np.full(len(p), SENSOR_Z)]
        uv, z, ok = cam.project(P)
        col = turbo(np.clip(rng / RMAX, 0, 1) * 0.9 + 0.05).astype(float) * fade
        rad = np.clip(26 / np.maximum(z, 1), 1.2, 5) * (1.0 if j == i else 0.7)
        for (u, v), c, r_, k in zip(uv, col, rad, ok):
            if not k or not (-20 < u < w + 20 and -20 < v < h + 20):
                continue
            c = tuple(int(x) for x in c)
            s.ellipse([u - r_, v - r_, u + r_, v + r_], fill=c)
            if j == i:
                g.ellipse([u - r_ * 3, v - r_ * 3, u + r_ * 3, v + r_ * 3], fill=c)
        if j == i:                                                             # current scan: beam fan + drop lines
            src, _, _ = cam.project(np.array([[0, 0, SENSOR_Z]]))
            G, _, okg = cam.project(np.c_[p, np.zeros(len(p))])
            for n_, ((u, v), c, k) in enumerate(zip(uv, col, ok)):
                if not k:
                    continue
                c = tuple(int(x) for x in c)
                if n_ % 3 == 0:
                    d.line([tuple(src[0]), (u, v)], fill=tuple(x // 7 for x in c), width=1)
                if okg[n_]:
                    s.line([(u, v), tuple(G[n_])], fill=tuple(x // 3 for x in c), width=1)

    glow = glow.filter(ImageFilter.GaussianBlur(9))
    out = ImageChops.add(base, glow)
    out = ImageChops.add(out, ImageChops.multiply(glow, glow))                 # hot core
    out = ImageChops.add(out, sharp)
    draw_robot(ImageDraw.Draw(out), cam)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=20); ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--frames", default="", help="a:b subset for previews")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    w, h = map(int, a.size.split("x")); cam = Camera(w, h)
    recs = [json.loads(l) for l in (a.run / "frames.jsonl").read_text().splitlines() if l.strip()]
    recs = sorted([r for r in recs if r.get("lidar")], key=lambda r: r["frame"])
    lo, hi = (map(int, a.frames.split(":")) if a.frames else (0, len(recs)))
    tmp = a.out / "_lidar3d_frames"; tmp.mkdir(exist_ok=True)
    for i in range(lo, min(hi, len(recs))):
        render(recs, i, cam, w, h).save(tmp / f"{i - lo:05d}.png")
    if not a.frames:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(tmp / "%05d.png"),
                        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(a.out / "lidar.mp4")], check=True)
        print("ok lidar.mp4", len(recs), "scans")


if __name__ == "__main__":
    main()
