"""Raw, unedited videos from one scripts/run_farm.py recording (no captions, no cuts; real sim time):
  chase.mp4        third-person camera locked behind the robot (rendered every --chase-every steps)
  front.mp4        robot front camera
  side_<cam>.mp4   the six plant cameras, one frame per plant stop (full 1080x1920 PNGs stay in rec/side/)
  lidar.mp4        planar C1 scan every 0.1 s, robot-centred top view (forward = up), colour = range
  slam_map.mp4     2-D map growing from the scans, placed with the robot's OWN pose estimate (scan-to-map + odometry);
                   dark = unknown, grey = seen free, bright = LiDAR returns; green line = driven path
  slam_map_final.png

  python3 scripts/make_farm_videos.py --rec <run>/rec --out <dir>"""
from __future__ import annotations
import argparse, json, math, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

SIDE = ("mast_left_low", "mast_left_mid", "mast_left_high", "mast_right_low", "mast_right_mid", "mast_right_high")


def encode(pattern: str, out: Path, fps: float, crf: int = 18, glob: bool = False):
    inp = ["-pattern_type", "glob", "-i", pattern] if glob else ["-i", pattern]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), *inp, "-c:v", "libx264", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(out)], check=True)


def range_colour(r: float, rmax: float = 12.0):
    u = max(0.0, min(1.0, r / rmax))
    return (int(255 * (1 - u) ** 0.6), int(200 * (1 - abs(u - 0.5) * 2) + 40), int(255 * u ** 0.8))


def scan_xy(ranges):
    n = len(ranges); out = []
    for k, r in enumerate(ranges):
        if r is None or r >= 11.9:
            continue
        a = -math.pi + 2 * math.pi * k / n + math.pi / 2          # R180 raw azimuth -> robot frame (0 = forward)
        out.append((r * math.cos(a), r * math.sin(a), r))
    return out


def lidar_frame(ranges, size=1080, view=8.0):
    im = Image.new("RGB", (size, size), (6, 8, 12)); d = ImageDraw.Draw(im); c = size / 2; s = c / view
    for r in range(2, int(view) + 1, 2):
        d.ellipse([c - r * s, c - r * s, c + r * s, c + r * s], outline=(30, 38, 46))
    for x, y, r in scan_xy(ranges):
        px, py = c - y * s, c - x * s
        d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=range_colour(r))
    d.polygon([(c, c - 14), (c - 9, c + 10), (c + 9, c + 10)], fill=(118, 185, 0))
    return im


class Map:
    def __init__(self, xmin, xmax, ymin, ymax, res=0.05):
        self.res, self.x0, self.y0 = res, xmin, ymin
        self.w, self.h = int((xmax - xmin) / res) + 1, int((ymax - ymin) / res) + 1
        self.hit = np.zeros((self.h, self.w), np.float32); self.free = np.zeros((self.h, self.w), np.float32)

    def add(self, x, y, heading_deg, ranges, lidar_ahead=0.34):
        h = math.radians(heading_deg); lx, ly = x + lidar_ahead * math.cos(h), y + lidar_ahead * math.sin(h)
        for px, py, r in scan_xy(ranges):                          # only beams that returned: no guessed free space
            wx = lx + px * math.cos(h) - py * math.sin(h); wy = ly + px * math.sin(h) + py * math.cos(h)
            n = int(r / self.res)
            xs = np.linspace(lx, wx, n, endpoint=False); ys = np.linspace(ly, wy, n, endpoint=False)
            ci = ((xs - self.x0) / self.res).astype(int); cj = ((ys - self.y0) / self.res).astype(int)
            ok = (ci >= 0) & (ci < self.w) & (cj >= 0) & (cj < self.h)
            self.free[cj[ok], ci[ok]] += 1
            i, j = int((wx - self.x0) / self.res), int((wy - self.y0) / self.res)
            if 0 <= i < self.w and 0 <= j < self.h:
                self.hit[j, i] += 4

    def image(self, trail, size=1080):
        img = np.zeros((self.h, self.w, 3), np.uint8); img[:] = (10, 12, 16)                  # unknown: dark
        seen = self.free > 0; img[seen] = (48, 54, 62)                                       # seen free: grey
        occ = self.hit > np.maximum(2, self.free * 0.5); img[occ] = (255, 196, 64)            # returns: amber
        im = Image.fromarray(np.flipud(img))
        sc = size / max(im.width, im.height); im = im.resize((int(im.width * sc), int(im.height * sc)), Image.NEAREST)
        d = ImageDraw.Draw(im)
        pts = [((x - self.x0) / self.res * sc, (self.h - (y - self.y0) / self.res) * sc) for x, y in trail]
        if len(pts) > 1:
            d.line(pts, fill=(118, 185, 0), width=3)
        if pts:
            x, y = pts[-1]; d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(240, 80, 60))
        return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chase-every", type=int, default=2); ap.add_argument("--map-every", type=int, default=5)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    step_s = 0.1
    if any((a.rec / "chase").glob("*.jpg")):
        encode(str(a.rec / "chase" / "*.jpg"), a.out / "chase.mp4", 1 / (step_s * a.chase_every), glob=True); print("ok chase")
    if any((a.rec / "front").glob("*.jpg")):
        encode(str(a.rec / "front" / "*.jpg"), a.out / "front.mp4", 1.0, glob=True); print("ok front")
    for cam in SIDE:
        if any((a.rec / "side").glob(f"*_{cam}.png")):
            encode(str(a.rec / "side" / f"*_{cam}.png"), a.out / f"side_{cam}.mp4", 2.0, glob=True); print("ok", cam)
    steps = [json.loads(l) for l in (a.rec / "steps.jsonl").read_text().splitlines() if l.strip()]
    steps = [s for s in steps if s.get("lidar") and s.get("xy")]
    tl, tm = a.out / "_lidar", a.out / "_map"; tl.mkdir(exist_ok=True); tm.mkdir(exist_ok=True)
    xs = [s["xy"][0] for s in steps]; ys = [s["xy"][1] for s in steps]
    m = Map(min(xs) - 6, max(xs) + 6, min(ys) - 6, max(ys) + 6)
    trail, k = [], 0
    for i, s in enumerate(steps):
        lidar_frame(s["lidar"]).save(tl / f"{i:06d}.png")
        m.add(s["xy"][0], s["xy"][1], s["heading_deg"], s["lidar"]); trail.append(tuple(s["xy"]))
        if i % a.map_every == 0:
            m.image(trail).save(tm / f"{k:06d}.png"); k += 1
    encode(str(tl / "%06d.png"), a.out / "lidar.mp4", 1 / step_s)
    encode(str(tm / "%06d.png"), a.out / "slam_map.mp4", 1 / (step_s * a.map_every))
    m.image(trail).save(a.out / "slam_map_final.png")
    print("ok lidar, slam_map", len(steps), "scans")


if __name__ == "__main__":
    main()
