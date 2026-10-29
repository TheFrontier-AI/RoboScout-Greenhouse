"""LiDAR view for render_cinematic.py runs, styled after driving-dataset point-cloud figures.
Camera sits behind and above the robot and turns with it (game-style chase), robot in the centre.
Points are the real planar (2-D) C1 scans: the last few seconds are kept in the world frame for density,
drawn large with a soft glow and coloured by distance from the robot (turbo: near = blue, far = red).
Nothing is extruded or invented. Same frame clock as the other raw videos.

  python3 scripts/lidar_viz.py --run <render dir> --out <dir> [--fps 20] [--size 1920x1080]"""
from __future__ import annotations
import argparse, json, math, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 22); FONT_S = ImageFont.truetype("DejaVuSans.ttf", 16)
except OSError:
    FONT = FONT_S = ImageFont.load_default()
RMAX = 12.0                                # metres
SENSOR_Z = 0.512                           # C1 mount height (m)
CAM_BACK, CAM_UP, LOOK_AHEAD = 9.0, 7.0, 4.0   # chase camera, robot frame (m)
KEEP_S = 6.0                               # seconds of scans kept on screen
HEADING_SMOOTH_S = 0.6                     # camera yaw smoothing so turns in place don't snap


def turbo(u: np.ndarray) -> np.ndarray:
    """Polynomial approximation of Google's Turbo colormap, u in [0,1] -> uint8 RGB."""
    u = np.clip(u, 0, 1)[..., None]
    c = np.concatenate([
        0.13572138 + u * (4.61539260 + u * (-42.66032258 + u * (132.13108234 + u * (-152.94239396 + u * 59.28637943)))),
        0.09140261 + u * (2.19418839 + u * (4.84296658 + u * (-14.18503333 + u * (4.27729857 + u * 2.82956604)))),
        0.10667330 + u * (12.64194608 + u * (-60.58204836 + u * (110.36276771 + u * (-89.90310912 + u * 27.34824973)))),
    ], -1)
    return (np.clip(c, 0, 1) * 255).astype(np.uint8)


def cmap(d: np.ndarray) -> np.ndarray:
    return turbo(0.1 + 0.85 * np.clip(d / RMAX, 0, 1))


class Camera:
    def __init__(self, w, h, pos, look, fov_deg=60.0):
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
        return np.zeros((0, 2))
    a = math.radians(l["yaw_deg"]) - math.pi / 2 + (-math.pi + 2 * math.pi * k / n)
    return np.stack([ox + r * np.cos(a), oy + r * np.sin(a)], -1)


def heading(rec) -> float:
    return math.radians(rec["lidar"]["yaw_deg"]) - math.pi / 2       # robot forward direction in the world


def smoothed_headings(recs, fps):
    h = np.unwrap([heading(r) for r in recs]); k = max(1, int(HEADING_SMOOTH_S * fps))
    pad = np.pad(h, k, mode="edge"); return np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="valid")


def chase_camera(w, h, rec, yaw):
    x, y = rec["robot"][:2]; c, s = math.cos(yaw), math.sin(yaw)
    pos = (x - CAM_BACK * c, y - CAM_BACK * s, CAM_UP)
    look = (x + LOOK_AHEAD * c, y + LOOK_AHEAD * s, 0.0)
    return Camera(w, h, pos, look)


def draw_grid(d, cam, rec, R=24, step=2.0):
    """World-fixed ground grid, faded with distance from the robot."""
    x0, y0 = rec["robot"][:2]; cx, cy = step * math.floor(x0 / step), step * math.floor(y0 / step)
    t = np.linspace(-R, R, 96)
    for axis in (0, 1):
        for v in np.arange(-R, R + step, step):
            xy = np.stack([np.full_like(t, cx + v), cy + t], -1) if axis == 0 else np.stack([cx + t, np.full_like(t, cy + v)], -1)
            uv, _, ok = cam.project(np.c_[xy, np.zeros(len(xy))])
            dist = np.hypot(xy[:, 0] - x0, xy[:, 1] - y0)
            for i in range(len(t) - 1):
                if ok[i] and ok[i + 1] and dist[i] < R:
                    a = int(40 * (1 - dist[i] / R)) + 10
                    d.line([tuple(uv[i]), tuple(uv[i + 1])], fill=(a, a, a), width=1)


def draw_robot(d, cam, rec):
    L, W, H = 0.62, 0.42, 0.55                       # R1 footprint (approx.), drawn as a wireframe box
    x0, y0 = rec["robot"][:2]; h = heading(rec); c, s = math.cos(h), math.sin(h)

    def world(p):
        p = np.atleast_2d(np.asarray(p, float))
        return np.c_[x0 + p[:, 0] * c - p[:, 1] * s, y0 + p[:, 0] * s + p[:, 1] * c, p[:, 2]]
    uv, _, ok = cam.project(world([(x, y, z) for z in (0, H) for x in (-L / 2, L / 2) for y in (-W / 2, W / 2)]))
    for a, b in [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        if ok[a] and ok[b]:
            d.line([tuple(uv[a]), tuple(uv[b])], fill=(255, 220, 0), width=2)
    tip, _, okt = cam.project(world([(L / 2 + 0.6, 0, 0.02), (L / 2 + 0.2, 0.2, 0.02), (L / 2 + 0.2, -0.2, 0.02)]))
    if okt.all():
        d.polygon([tuple(p) for p in tip], fill=(255, 220, 0))                 # heading arrow


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
        d.line([(x0, y0 + k), (x0 + bw, y0 + k)], fill=tuple(int(x) for x in cmap(np.array([(1 - k / bh) * RMAX]))[0]))
    d.rectangle([x0, y0, x0 + bw, y0 + bh], outline=(120, 120, 120))
    for m in (0, 4, 8, 12):
        y = y0 + bh - bh * m / RMAX
        d.line([(x0 - 5, y), (x0, y)], fill=(160, 160, 160)); d.text((x0 - 44, y - 9), f"{m} m", fill=(170, 170, 170), font=FONT_S)
    d.text((x0 - 50, y0 - 28), "distance", fill=(170, 170, 170), font=FONT_S)


def render(recs, i, w, h, cloud, yaws, keep) -> Image.Image:
    cur = recs[i]; cam = chase_camera(w, h, cur, yaws[i])
    base = Image.new("RGB", (w, h)); draw_grid(ImageDraw.Draw(base), cam, cur)
    pts = Image.new("RGB", (w, h)); img = np.asarray(pts).copy()
    rx, ry = cur["robot"][:2]
    scans = [c for c in cloud[max(0, i - keep): i + 1] if len(c)]
    if scans:
        xy = np.concatenate(scans)
        uv, z, ok = cam.project(np.c_[xy, np.full(len(xy), SENSOR_Z)])
        col = cmap(np.hypot(xy[:, 0] - rx, xy[:, 1] - ry))                      # colour by distance from the robot now
        size = np.clip(np.round(60 / np.maximum(z, 1)), 3, 7).astype(int)       # bigger when closer to the camera
        for s in np.unique(size):
            splat(img, uv, col, ok & (size == s), int(s))
    pts = Image.fromarray(img)
    glow = pts.filter(ImageFilter.GaussianBlur(6))
    out = np.clip(np.asarray(base, int) + np.asarray(glow, int) * 1.6 + np.asarray(pts, int), 0, 255).astype(np.uint8)
    out = Image.fromarray(out); d = ImageDraw.Draw(out)
    draw_robot(d, cam, cur); colorbar(d, w, h)
    d.text((32, h - 50), f"R1 planar LiDAR (C1, 500 beams, 12 m)  |  last {KEEP_S:.0f} s of scans  |  t = {cur['t']:.2f} s",
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
    cloud, yaws, keep = [hits_world(r) for r in recs], smoothed_headings(recs, a.fps), int(KEEP_S * a.fps)
    for i in range(lo, min(hi, len(recs))):
        render(recs, i, w, h, cloud, yaws, keep).save(tmp / f"{i - lo:05d}.png")
    if not a.frames:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(tmp / "%05d.png"),
                        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(a.out / "lidar.mp4")], check=True)
        print("ok lidar.mp4", len(recs), "scans")


if __name__ == "__main__":
    main()
