"""One camera at a time: draw RT-DETR boxes (per-image detections) on side_<cam>.mp4 -> det_<cam>.mp4, plus a per-camera
count summary. python3 scripts/annotate_side_cam.py --videos <dir> --detection <dir> --cam left_mid --out det_left_mid.mp4"""
from __future__ import annotations
import argparse, collections, json, subprocess, tempfile
from pathlib import Path
from PIL import Image, ImageDraw
from annotate_side import COL, NAME, frames, font

ap = argparse.ArgumentParser(); ap.add_argument("--videos", type=Path, required=True); ap.add_argument("--detection", type=Path, required=True)
ap.add_argument("--cam", required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--fps", type=float, default=2.0)
a = ap.parse_args()
rows = sorted((json.loads(l) for l in open(a.detection / "findings.jsonl") if f"mast_{a.cam}" in l), key=lambda r: int(Path(r["image"]).stem.split("_")[0]))
rows = [r for r in rows if r["camera"] == f"mast_{a.cam}"]
f_big, f_small = font(40), font(26); tmp = Path(tempfile.mkdtemp(prefix=f"det_{a.cam}_")); tot = collections.Counter()
for k, (r, im) in enumerate(zip(rows, frames(a.videos / f"side_{a.cam}.mp4"))):
    d = ImageDraw.Draw(im)
    for b in r.get("detections", []):
        x1, y1, x2, y2 = (v * s for v, s in zip(b["bbox_norm"], (1080, 1920, 1080, 1920))); c = COL.get(b["label"], (255, 255, 255))
        d.rectangle([x1, y1, x2, y2], outline=c, width=4); d.text((x1 + 4, max(0, y1 - 30)), f"{NAME.get(b['label'], b['label'])} {b['score']:.2f}", font=f_small, fill=c)
        tot[b["label"]] += 1
    d.rectangle([0, 0, 1080, 60], fill=(0, 0, 0)); d.text((14, 10), f"{a.cam.replace('_', ' ')}  {r['aisle']}  metre {r['metre']:.2f}  stop {k + 1}/{len(rows)}", font=f_big, fill=(255, 255, 255))
    im.save(tmp / f"{k:05d}.jpg", quality=90)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{a.fps:g}", "-i", str(tmp / "%05d.jpg"), "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(a.out)], check=True)
(a.out.with_suffix(".json")).write_text(json.dumps({"camera": a.cam, "stops": len(rows), "detections_by_class": dict(tot)}, indent=1))
print(a.out.name, len(rows), "stops", dict(tot))
