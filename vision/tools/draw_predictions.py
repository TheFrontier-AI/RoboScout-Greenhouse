"""Draw detector predictions (thick) over images, optionally next to ground truth (thin, from labels/).
  .venv-vision/bin/python vision/tools/draw_predictions.py --model vision_runs/rtdetr_v1_synth/best OUT.jpg IMG [IMG...]"""
from __future__ import annotations
import argparse, json, os, sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vision.detector import INPUT_WH, Detector              # noqa: E402

COL = {"fruit_ripe": (255, 40, 40), "fruit_turning": (255, 170, 0), "fruit_green": (60, 255, 60),
       "fruit_lesion": (255, 0, 255), "plant_base": (255, 255, 255), "leaf_chlorosis": (0, 200, 255),
       "leaf_spots": (0, 120, 255)}
SHORT = {"fruit_ripe": "ripe", "fruit_turning": "turn", "fruit_green": "grn", "fruit_lesion": "LES",
         "plant_base": "base", "leaf_chlorosis": "chl", "leaf_spots": "spot"}


def gt_path(img: str) -> str | None:
    d, f = os.path.split(img)
    p = os.path.join(os.path.dirname(d), "labels", os.path.splitext(f)[0] + ".json")
    return p if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("out")
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    det = Detector(a.model, threshold=a.threshold)
    tiles = []
    for p in a.images:
        im = Image.open(p).convert("RGB")
        dets = det.detect(im)
        im = im.resize(INPUT_WH, Image.BILINEAR)
        W, H = im.size
        gt = im.copy()
        g = ImageDraw.Draw(gt)
        lp = gt_path(p)
        if lp:
            for o in json.load(open(lp))["objects"]:
                from train_rtdetr import class_of                      # noqa: E402
                c = class_of(o)
                if c:
                    x, y, w, h = o["bbox"]
                    g.rectangle([x, y, x + w, y + h], outline=COL[c], width=2)
        d = ImageDraw.Draw(im)
        for r in dets:
            x0, y0, x1, y1 = (v * s for v, s in zip(r["bbox_norm"], (W, H, W, H)))
            d.rectangle([x0, y0, x1, y1], outline=COL[r["label"]], width=3)
            d.text((x0 + 2, y0 + 1), f"{SHORT[r['label']]} {r['score']:.2f}", fill=COL[r["label"]])
        tiles += [gt, im] if lp else [im]
    per_row = 6 if len(tiles) >= 6 else len(tiles)
    rows = (len(tiles) + per_row - 1) // per_row
    sheet = Image.new("RGB", (W * per_row, H * rows), (20, 24, 28))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % per_row) * W, (i // per_row) * H))
    sheet.thumbnail((3000, 3000))
    sheet.save(a.out, quality=88)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
