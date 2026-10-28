"""Draw synthetic labels over their images for visual QA.  python3 vision/tools/draw_labels.py DATASET_DIR OUT.jpg [--n 3]"""
from __future__ import annotations
import argparse, glob, json, os
from PIL import Image, ImageDraw

FRUIT_COLOURS = {"ripe": (255, 40, 40), "turning": (255, 170, 0), "green": (60, 255, 60), "lesion": (255, 0, 255)}


def draw(label_path: str, root: str) -> Image.Image:
    lab = json.load(open(label_path))
    im = Image.open(os.path.join(root, lab["image"])).convert("RGB")
    d = ImageDraw.Draw(im)
    for o in lab["objects"]:
        x, y, w, h = o["bbox"]
        if o["type"] == "fruit":
            col = (128, 128, 128) if o.get("ignore") else FRUIT_COLOURS[o["cls"]]
            d.rectangle([x - 1, y - 1, x + w + 1, y + h + 1], outline=col, width=1 if o.get("ignore") else 2)
        elif o["type"] == "plant_base":
            d.rectangle([x, y, x + w, y + h], outline=(255, 255, 255), width=2)
            d.text((x + 2, y - 10), f"P{o['id']}", fill=(255, 255, 255))
        else:
            d.rectangle([x, y, x + w, y + h], outline=(0, 200, 255), width=2)
            d.text((x + 3, y + 3), f"{o['cls']}/{o['severity']} P{o['id']}", fill=(0, 200, 255))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    ims = [draw(p, a.root) for p in sorted(glob.glob(os.path.join(a.root, "labels", "*.json")))[:a.n]]
    w, h = ims[0].size
    sheet = Image.new("RGB", (w * len(ims), h))
    for i, im in enumerate(ims):
        sheet.paste(im, (i * w, 0))
    sheet.save(a.out, quality=90)


if __name__ == "__main__":
    main()
