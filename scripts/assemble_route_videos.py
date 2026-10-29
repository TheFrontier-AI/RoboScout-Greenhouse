"""Join scripts/render_route.py frames (from all GPUs / VMs) into raw, real-time videos, one per camera. No cuts, no
captions, no speed change: frame rate = the rate the frames were rendered at along the recorded run.

  chase.mp4 (5 fps), front.mp4 (2 fps), zed_left.mp4 / zed_right.mp4 / zed_depth.mp4 (10 fps),
  shots/<camera>_<n>.mp4 (10 fps; a new clip each time the robot re-enters that camera's zone),
  side_<camera>.mp4 (one frame per plant stop, 2 fps), lidar.mp4 (10 fps, robot-centred 2-D scan)

  python3 scripts/assemble_route_videos.py --frames <render_out> --out <videos_dir>"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def encode(files, out: Path, fps: float, crf: int = 16):
    """ffmpeg concat of an explicit, time-ordered file list."""
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{f}'\nduration {1 / fps:.6f}\n" for f in files) + f"file '{files[-1]}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-r", f"{fps:g}",
                    "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    str(out)], check=True)
    lst.unlink()
    print(f"{out.name}: {len(files)} frames, {len(files) / fps:.1f} s")


def ns_sorted(d: Path, pattern="*"):
    return sorted(d.glob(pattern), key=lambda p: int(p.stem.split("_")[0]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frames", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); F = a.frames
    for name, fps in (("chase", 5), ("front", 2)):
        fs = ns_sorted(F / name, "*.jpg")
        if fs:
            encode(fs, a.out / f"{name}.mp4", fps)
    for cam, sub in (("zed_left", "cam0"), ("zed_right", "cam1")):
        fs = ns_sorted(F / "zed" / sub, "*.png")
        if fs:
            encode(fs, a.out / f"{cam}.mp4", 10)
    dep = ns_sorted(F / "zed" / "depth0", "*.png")
    if dep:                                                     # depth as a viewable greyscale video (0-12 m)
        import numpy as np
        from PIL import Image
        vis = F / "zed" / "depth_vis"; vis.mkdir(exist_ok=True)
        for f in dep:
            o = vis / (f.stem + ".jpg")
            if not o.exists():
                d = np.asarray(Image.open(f), dtype=np.float32) / 1000.0
                Image.fromarray((255 * (1 - np.clip(d / 12.0, 0, 1)) * (d > 0)).astype(np.uint8)).save(o, quality=92)
        encode(ns_sorted(vis, "*.jpg"), a.out / "zed_depth.mp4", 10)
    (a.out / "shots").mkdir(exist_ok=True)
    for cam in sorted((F / "shots").glob("*")) if (F / "shots").exists() else []:
        fs = ns_sorted(cam, "*.jpg"); clips, cur = [], []
        for f in fs:
            if cur and int(f.stem) - int(cur[-1].stem) > 0.15e9:
                clips.append(cur); cur = []
            cur.append(f)
        if cur:
            clips.append(cur)
        for i, c in enumerate(clips):
            if len(c) >= 5:
                encode(c, a.out / "shots" / f"{cam.name}_{i + 1}.mp4", 10)
    for cam in ("left_low", "left_mid", "left_high", "right_low", "right_mid", "right_high"):
        fs = ns_sorted(F / "side", f"*_plant_{cam}.png")
        if fs:
            encode(fs, a.out / f"side_{cam}.mp4", 2)
    rows = []
    for p in sorted((F / "lidar").glob("scans_part*.jsonl")):
        rows += [json.loads(l) for l in open(p)]
    if rows:
        from make_farm_videos import lidar_frame
        rows.sort(key=lambda r: r["ns"]); d = F / "lidar" / "frames"; d.mkdir(exist_ok=True)
        for r in rows:
            o = d / f"{r['ns']}.png"
            if not o.exists():
                lidar_frame(r["lidar"]).save(o)
        encode(ns_sorted(d, "*.png"), a.out / "lidar.mp4", 10)


if __name__ == "__main__":
    main()
