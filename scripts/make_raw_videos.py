"""Raw, unedited videos from one render_cinematic.py run (no captions, no cuts; all share the same frame clock):
  chase.mp4        third-person follow camera
  side_left.mp4    robot-left plant camera (1080x1920), PNG frames kept for detection datasets
  side_right.mp4   robot-right plant camera (1080x1920)
  front.mp4        front camera
  lidar.mp4        planar LiDAR scan, 3-D chase view with fading trail and glow (scripts/lidar_viz.py)
  lidar_top.mp4    the same scans, robot-centred top view (forward = up), colour = range
  slam_map.mp4     occupancy map built from the LiDAR scans along the route (known poses: mapping, not pose estimation)

  python3 scripts/make_raw_videos.py --run <render_cinematic out dir> --out <dir> [--fps 20]"""
from __future__ import annotations
import argparse, json, math, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def encode(pattern: str, out: Path, fps: int, crf: int = 16):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern, "-c:v", "libx264",
                    "-crf", str(crf), "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(out)], check=True)


def range_colour(r: float, rmax: float = 12.0):
    u = max(0.0, min(1.0, r / rmax))                  # near = warm, far = cool
    return (int(255 * (1 - u) ** 0.6), int(200 * (1 - abs(u - 0.5) * 2) + 40), int(255 * u ** 0.8))


def lidar_frame(rec: dict, size: int = 1080, view_m: float = 12.0) -> Image.Image:
    im = Image.new("RGB", (size, size), (6, 8, 12)); d = ImageDraw.Draw(im); c = size / 2; s = c / view_m
    for r in range(2, int(view_m) + 1, 2):                                   # range rings every 2 m
        d.ellipse([c - r * s, c - r * s, c + r * s, c + r * s], outline=(28, 36, 44))
    rr = rec["lidar"]["ranges"]; n = len(rr)
    for k, r in enumerate(rr):
        if r is None:
            continue
        a = -math.pi + 2 * math.pi * k / n                                   # robot frame: 0 = forward, +left
        x, y = c - r * math.sin(a) * s, c - r * math.cos(a) * s             # forward drawn up, left drawn left
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=range_colour(r))
    d.polygon([(c, c - 14), (c - 9, c + 10), (c + 9, c + 10)], fill=(118, 185, 0))   # the robot
    return im


class Grid:
    """Simple occupancy evidence grid (hits vs pass-throughs) in the scene frame."""

    def __init__(self, xmin, xmax, ymin, ymax, res=0.05):
        self.res, self.x0, self.y0 = res, xmin, ymin
        self.w, self.h = int((xmax - xmin) / res) + 1, int((ymax - ymin) / res) + 1
        self.hit = np.zeros((self.h, self.w), np.float32); self.free = np.zeros((self.h, self.w), np.float32)

    def cell(self, x, y):
        return int((x - self.x0) / self.res), int((y - self.y0) / self.res)

    def add(self, rec, rmax=12.0):
        ox, oy = rec["lidar"]["origin"]; yaw = rec["lidar"]["yaw_deg"]; rr = rec["lidar"]["ranges"]; n = len(rr)
        for k, r in enumerate(rr):
            a = math.radians(yaw) - math.pi / 2 + (-math.pi + 2 * math.pi * k / n)
            dist = r if r is not None else rmax
            steps = int(dist / self.res)
            xs = ox + np.cos(a) * np.arange(steps) * self.res; ys = oy + np.sin(a) * np.arange(steps) * self.res
            ci = ((xs - self.x0) / self.res).astype(int); cj = ((ys - self.y0) / self.res).astype(int)
            ok = (ci >= 0) & (ci < self.w) & (cj >= 0) & (cj < self.h)
            self.free[cj[ok], ci[ok]] += 1
            if r is not None:
                i, j = self.cell(ox + math.cos(a) * r, oy + math.sin(a) * r)
                if 0 <= i < self.w and 0 <= j < self.h:
                    self.hit[j, i] += 3

    def image(self, trail, size=1080) -> Image.Image:
        known = (self.hit + self.free) > 0
        occ = np.where(known, self.hit / np.maximum(self.hit + self.free, 1e-6), 0.5)
        g = np.full(occ.shape, 70, np.uint8)                                   # unknown = grey
        g[known] = (235 - 235 * np.clip(occ[known] * 4, 0, 1)).astype(np.uint8)  # free = light, occupied = dark
        im = Image.fromarray(np.flipud(g)).convert("RGB")                       # +Y up
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
    ap.add_argument("--run", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=20)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    for name in ("chase", "side_left", "side_right", "front"):
        if (a.run / name).is_dir() and any((a.run / name).glob("*.png")):
            encode(str(a.run / name / "%05d.png"), a.out / f"{name}.mp4", a.fps); print("ok", name)
    recs = [json.loads(l) for l in (a.run / "frames.jsonl").read_text().splitlines() if l.strip()]
    recs = sorted([r for r in recs if r.get("lidar")], key=lambda r: r["frame"])
    if not recs:
        print("no lidar records"); return
    tmp_l, tmp_m = a.out / "_lidar_frames", a.out / "_map_frames"; tmp_l.mkdir(exist_ok=True); tmp_m.mkdir(exist_ok=True)
    xs = [r["robot"][0] for r in recs]; ys = [r["robot"][1] for r in recs]
    grid = Grid(min(xs) - 8, max(xs) + 8, min(ys) - 8, max(ys) + 8)
    trail = []
    for i, r in enumerate(recs):
        lidar_frame(r).save(tmp_l / f"{i:05d}.png")
        grid.add(r); trail.append(tuple(r["lidar"]["origin"]))
        grid.image(trail).save(tmp_m / f"{i:05d}.png")
    encode(str(tmp_l / "%05d.png"), a.out / "lidar_top.mp4", a.fps); encode(str(tmp_m / "%05d.png"), a.out / "slam_map.mp4", a.fps)
    grid.image(trail).save(a.out / "slam_map_final.png")
    subprocess.run(["python3", str(Path(__file__).with_name("lidar_viz.py")), "--run", str(a.run), "--out", str(a.out),
                    "--fps", str(a.fps)], check=True)
    print("ok lidar, slam_map", len(recs), "scans")


if __name__ == "__main__":
    main()
