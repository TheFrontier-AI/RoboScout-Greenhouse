"""Draw the AI results onto the side-camera plant stops: RT-DETR boxes coloured by category and the Nemotron VL verdict
for the left/right plant, one 3x2 grid frame per stop (left high/mid/low | right high/mid/low) -> side_ai.mp4.

  python3 scripts/annotate_side.py --videos <videos_dir> --detection <detection_dir> --vlm <vlm_stops_dir> --out side_ai.mp4"""
from __future__ import annotations
import argparse, collections, json, subprocess, tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ORDER_L = ("left_high", "left_mid", "left_low"); ORDER_R = ("right_high", "right_mid", "right_low")
COL = {"fruit_ripe": (40, 200, 70), "fruit_turning": (255, 200, 0), "fruit_green": (255, 200, 0), "fruit_lesion": (230, 30, 30),
       "plant_base": (60, 160, 255), "leaf_chlorosis": (220, 60, 255), "leaf_spots": (220, 60, 255)}
NAME = {"fruit_ripe": "ripe", "fruit_turning": "turning", "fruit_green": "green", "fruit_lesion": "lesion", "plant_base": "plant",
        "leaf_chlorosis": "chlorosis", "leaf_spots": "leaf spots"}
VCOL = {"healthy": (40, 160, 70), "suspect": (230, 150, 0), "disease": (220, 30, 30), "not_plant": (90, 90, 90), "error": (90, 90, 90)}


def frames(video: Path, w=1080, h=1920):
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    while (b := p.stdout.read(w * h * 3)) and len(b) == w * h * 3:
        yield Image.frombytes("RGB", (w, h), b)


def font(sz):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, required=True); ap.add_argument("--detection", type=Path, required=True)
    ap.add_argument("--vlm", type=Path); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--only", choices=("all", "harvest", "disease"), default="all",
                    help="harvest = fruit grading + plant bases; disease = chlorosis, leaf spots, fruit lesions")
    a = ap.parse_args()
    det = collections.defaultdict(list)
    for l in open(a.detection / "findings.jsonl"):
        r = json.loads(l); det[r["camera"]].append((int(Path(r["image"]).stem.split("_")[0]), r))
    for k in det:
        det[k].sort(key=lambda x: x[0])
    vlm = [json.loads(l) for l in open(a.vlm / "vlm_stops.jsonl")] if a.vlm else []
    vlm = {v["stop"]: v for v in vlm}
    gens = {c: frames(a.videos / f"side_{c}.mp4") for c in ORDER_L + ORDER_R}
    n = min(len(det[f"mast_{c}"]) for c in ORDER_L + ORDER_R)
    f_big, f_small = font(34), font(22)
    tmp = Path(tempfile.mkdtemp(prefix="side_ai_"))
    W, H = 540, 960
    for k in range(n):
        PANEL = 300 if any("description" in vlm.get(k, {}).get(sd, {}) for sd in ("left", "right")) else 0
        canvas = Image.new("RGB", (W * 6 + 20, H + 120 + PANEL), (14, 16, 18)); d = ImageDraw.Draw(canvas)
        r0 = det["mast_left_mid"][k][1]
        d.text((20, 14), f"{r0['aisle']}  metre {r0['metre']:.2f}   stop {k + 1}/{n}", font=f_big, fill=(235, 235, 235))
        for side_i, (side, cams) in enumerate((("left", ORDER_L), ("right", ORDER_R))):
            v = vlm.get(k, {}).get(side, {})
            st = v.get("status", "-"); labels = ", ".join(i.get("label", "") for i in v.get("issues", []))[:60]
            x0 = 10 + side_i * (W * 3 + 10)
            d.rectangle([x0, 62, x0 + W * 3 - 10, 108], fill=VCOL.get(st, (60, 60, 60)))
            d.text((x0 + 12, 70), f"Nemotron VL ({side} plant): {st}" + (f" - {labels}" if labels else ""), font=f_small, fill=(255, 255, 255))
            for j, cam in enumerate(cams):
                im = next(gens[cam]).resize((W, H)); dd = ImageDraw.Draw(im)
                keep = {"all": None, "harvest": {"fruit_ripe", "fruit_turning", "fruit_green", "fruit_lesion", "plant_base"},
                        "disease": {"leaf_chlorosis", "leaf_spots", "fruit_lesion"}}[a.only]
                for b in det[f"mast_{cam}"][k][1].get("detections", []):
                    if keep is not None and b["label"] not in keep:
                        continue
                    x1, y1, x2, y2 = (v_ * s for v_, s in zip(b["bbox_norm"], (W, H, W, H)))
                    c = COL.get(b["label"], (255, 255, 255))
                    dd.rectangle([x1, y1, x2, y2], outline=c, width=3)
                    dd.text((x1 + 3, max(0, y1 - 20)), f"{NAME.get(b['label'], b['label'])} {b['score']:.2f}", font=f_small, fill=c)
                canvas.paste(im, (x0 + j * W, 116))
                d.text((x0 + j * W + 8, 120 + H - 34), cam.replace("_", " "), font=f_small, fill=(255, 255, 255))
            if PANEL:                                     # Nemotron VL explanation, agronomist style
                import textwrap
                y = 116 + H + 12
                for line in textwrap.wrap(v.get("description", ""), 118)[:8]:
                    d.text((x0 + 12, y), line, font=f_small, fill=(225, 225, 225)); y += 28
                if v.get("action"):
                    for line in textwrap.wrap("Next step: " + v["action"], 118)[:2]:
                        d.text((x0 + 12, y + 6), line, font=f_small, fill=(255, 210, 90)); y += 28
        canvas.save(tmp / f"{k:05d}.jpg", quality=90)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{a.fps:g}", "-i", str(tmp / "%05d.jpg"), "-c:v", "libx264",
                    "-crf", "20", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(a.out)], check=True)
    print(f"{a.out}: {n} stops")


if __name__ == "__main__":
    main()
