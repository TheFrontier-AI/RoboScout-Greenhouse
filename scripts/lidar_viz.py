"""Paper-style LiDAR view for render_cinematic.py runs: an oblique 3-D view of the real planar (2-D) scan.
Every scan so far is accumulated in the world frame into one point cloud; the current scan is highlighted.
The camera is world-fixed in orientation and pans smoothly with the robot, so the map builds up on screen.
Colour = range from the sensor at capture (turbo). Real sensor data only: nothing is extruded or invented.
Same frame clock as the other raw videos.

  python3 scripts/lidar_viz.py --run <render dir> --out <dir> [--fps 20] [--size 1920x1080]"""
from __future__ import annotations
import argparse, json, math, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 22); FONT_S = ImageFont.truetype("DejaVuSans.ttf", 16)
except OSError:
    FONT = FONT_S = ImageFont.load_default()
RMAX = 12.0                                # metres
SENSOR_Z = 0.512                           # C1 mount height (m)
CAM_OFFSET = np.array([-19.0, -21.0, 25.0])  # world-frame offset of the camera from its look-at point
FOLLOW = 0.45                              # look-at = FOLLOW * robot + (1 - FOLLOW) * route centre
SMOOTH_S = 1.5                             # look-at smoothing window (s)


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
    def __init__(self, w, h, pos, look, fov_deg=50.0):
        self.w, self.h, self.f = w, h, (w / 2) / math.tan(math.radians(fov_deg) / 2)
        p, t = np.asarray(pos, float), np.asarray(look, float)
        fwd = (t - p) / np.linalg.norm(t - p)
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        self.p, self.R = p, np.stack([right, up, fwd])

    def project(self, pts: np.ndarray):
        """pts (N,3) world -> (N,2) pixels, depth, visible mask."""
        c = (pts - self.p) @ self.R.T
        z = c[:, 2]; ok = z > 1.0
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


def heading(rec) -> float:
    return math.radians(rec["lidar"]["yaw_deg"]) - math.pi / 2       # robot forward direction in the world


def look_points(recs, fps):
    """Smoothed look-at point per frame: part robot, part route centre, so the built map stays in view."""
    xy = np.array([r["robot"][:2] for r in recs], float); centre = (xy.min(0) + xy.max(0)) / 2
    tgt = FOLLOW * xy + (1 - FOLLOW) * centre
    k = max(1, int(SMOOTH_S * fps)); pad = np.pad(tgt, ((k, k), (0, 0)), mode="edge")
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    sm = np.stack([np.convolve(pad[:, j], ker, mode="valid") for j in range(2)], -1)
    return np.c_[sm, np.zeros(len(sm))]


def draw_grid(d, cam, look, R=30, step=1.0):
    """World-fixed 1 m ground grid (5 m lines brighter), faded with distance from the look-at point."""
    cx, cy = math.floor(look[0]), math.floor(look[1])
    t = np.linspace(-R, R, 120)
    for axis in (0, 1):
        for v in np.arange(-R, R + 1, step):
            line_at = (cx if axis == 0 else cy) + v
            xy = np.stack([np.full_like(t, line_at), cy + t], -1) if axis == 0 else np.stack([cx + t, np.full_like(t, line_at)], -1)
            uv, _, ok = cam.project(np.c_[xy, np.zeros(len(xy))])
            dist = np.hypot(xy[:, 0] - look[0], xy[:, 1] - look[1])
            major = line_at % 5 == 0
            for i in range(len(t) - 1):
                if ok[i] and ok[i + 1] and dist[i] < R:
                    a = int((44 if major else 26) * (1 - dist[i] / R) ** 1.1) + 8
                    d.line([tuple(uv[i]), tuple(uv[i + 1])], fill=(a, a, a), width=1)


def draw_robot(d, cam, rec):
    L, W, H = 0.62, 0.42, 0.55                       # R1 footprint (approx.), drawn as a wireframe box
    x0, y0 = rec["robot"][:2]; h = heading(rec); c, s = math.cos(h), math.sin(h)

    def world(p):
        p = np.atleast_2d(np.asarray(p, float))
        return np.c_[x0 + p[:, 0] * c - p[:, 1] * s, y0 + p[:, 0] * s + p[:, 1] * c, p[:, 2]]
    corners = world([(x, y, z) for z in (0, H) for x in (-L / 2, L / 2) for y in (-W / 2, W / 2)])
    uv, _, ok = cam.project(corners)
    edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        if ok[a] and ok[b]:
            d.line([tuple(uv[a]), tuple(uv[b])], fill=(255, 220, 0), width=2)
    tip, _, okt = cam.project(world([(L / 2 + 0.7, 0, 0.02), (L / 2 + 0.2, 0.22, 0.02), (L / 2 + 0.2, -0.22, 0.02)]))
    if okt.all():
        d.polygon([tuple(p) for p in tip], fill=(255, 220, 0))                 # heading arrow
    sen, _, oks = cam.project(np.array([[*rec["lidar"]["origin"], SENSOR_Z]]))
    if oks[0]:
        x, y = sen[0]; d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 255, 255))


def precompute(recs):
    """World-frame hits of every scan, coloured once by range at capture (stable colours, no flicker)."""
    out = []
    for r in recs:
        xy, rng = hits_world(r)
        out.append((xy, turbo(np.clip(rng / RMAX, 0, 1) * 0.85 + 0.08)))
    return out


def splat(img: np.ndarray, uv, col, ok, size: int):
    w, h = img.shape[1], img.shape[0]
    u = np.round(uv[:, 0]).astype(int) - size // 2; v = np.round(uv[:, 1]).astype(int) - size // 2
    for du in range(size):
        for dv in range(size):
            m = ok & (u + du >= 0) & (u + du < w) & (v + dv >= 0) & (v + dv < h)
            img[v[m] + dv, u[m] + du] = np.maximum(img[v[m] + dv, u[m] + du], col[m])


def colorbar(d, w, h):
    x0, y0, bw, bh = w - 70, h // 2 - 160, 16, 320
    for k in range(bh):
        c = tuple(int(x) for x in turbo(np.array([(1 - k / bh) * 0.85 + 0.08]))[0])
        d.line([(x0, y0 + k), (x0 + bw, y0 + k)], fill=c)
    d.rectangle([x0, y0, x0 + bw, y0 + bh], outline=(120, 120, 120))
    for m in (0, 4, 8, 12):
        y = y0 + bh - bh * m / RMAX
        d.line([(x0 - 5, y), (x0, y)], fill=(160, 160, 160)); d.text((x0 - 44, y - 9), f"{m} m", fill=(170, 170, 170), font=FONT_S)
    d.text((x0 - 30, y0 - 28), "range", fill=(170, 170, 170), font=FONT_S)


def render(recs, i, w, h, cloud, looks) -> Image.Image:
    cur = recs[i]; look = looks[i]
    cam = Camera(w, h, look + CAM_OFFSET, look)
    base = Image.new("RGB", (w, h), (0, 0, 0)); d = ImageDraw.Draw(base)
    draw_grid(d, cam, look)
    path = np.array([r["robot"][:2] for r in recs[: i + 1]], float)              # driven path on the ground
    uv, _, ok = cam.project(np.c_[path, np.zeros(len(path))])
    pts = [tuple(p) for p, k in zip(uv, ok) if k]
    if len(pts) > 1:
        d.line(pts, fill=(235, 235, 235), width=2)

    img = np.asarray(base).copy()
    old = [c for c in cloud[: i] if len(c[0])]
    if old:                                                                    # accumulated map (all earlier scans)
        xy = np.concatenate([c[0] for c in old]); col = (np.concatenate([c[1] for c in old]) * 0.8).astype(np.uint8)
        uv, z, ok = cam.project(np.c_[xy, np.full(len(xy), SENSOR_Z)])
        near = z < 26; splat(img, uv, col, ok & ~near, 2); splat(img, uv, col, ok & near, 3)
    xy, col = cloud[i]
    if len(xy):                                                                # current scan, highlighted
        uv, z, ok = cam.project(np.c_[xy, np.full(len(xy), SENSOR_Z)])
        o = cur["lidar"]["origin"]; src, _, oks = cam.project(np.array([[*o, SENSOR_Z]]))
        rays = Image.new("RGB", (w, h)); rd = ImageDraw.Draw(rays)                # faint beam fan of the live scan
        if oks[0]:
            for n_, ((u, v), c, k) in enumerate(zip(uv, col, ok)):
                if k and n_ % 2 == 0:
                    rd.line([tuple(src[0]), (u, v)], fill=tuple(int(x) // 6 for x in c), width=1)
        img = np.maximum(img, np.asarray(rays))
        splat(img, uv, np.clip(col.astype(int) + 90, 0, 255).astype(np.uint8), ok, 4)
    out = Image.fromarray(img); d = ImageDraw.Draw(out)
    draw_robot(d, cam, cur)
    colorbar(d, w, h)
    n_pts = sum(len(c[0]) for c in cloud[: i + 1])
    d.text((32, h - 50), f"R1 planar LiDAR (C1, 500 beams, 12 m)  |  scans: {i + 1}  |  points: {n_pts:,}  |  t = {cur['t']:.2f} s",
           fill=(180, 180, 180), font=FONT)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=20); ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--frames", default="", help="a:b subset for previews")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    w, h = map(int, a.size.split("x"))
    recs = [json.loads(l) for l in (a.run / "frames.jsonl").read_text().splitlines() if l.strip()]
    recs = sorted([r for r in recs if r.get("lidar")], key=lambda r: r["frame"])
    lo, hi = (map(int, a.frames.split(":")) if a.frames else (0, len(recs)))
    tmp = a.out / "_lidar3d_frames"; tmp.mkdir(exist_ok=True)
    cloud, looks = precompute(recs), look_points(recs, a.fps)
    for i in range(lo, min(hi, len(recs))):
        render(recs, i, w, h, cloud, looks).save(tmp / f"{i - lo:05d}.png")
    if not a.frames:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(tmp / "%05d.png"),
                        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(a.out / "lidar.mp4")], check=True)
        print("ok lidar.mp4", len(recs), "scans")


if __name__ == "__main__":
    main()
