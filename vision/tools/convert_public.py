"""Convert public CC BY 4.0 datasets into the synthetic label format used by vision/train_rtdetr.py.

All output frames are 576x1024 portrait, like the mast cameras and the synthetic set.
  agrob    AgRobTomato (Zenodo 5596799, CC BY 4.0): robot-height views between greenhouse rows, 4 ripeness stages.
           Each 1280x720 frame is cut into three overlapping portrait crops.
  plantdoc PlantDoc object detection (github pratikkayal/PlantDoc-Object-Detection-Dataset, CC BY 4.0): tomato leaf
           images only. Spot/blight classes map to leaf_spots, yellow-leaf virus maps to leaf_chlorosis. Images containing
           unmapped tomato diseases (mosaic, mold, mites) are skipped, so no visible disease goes unlabelled.

`scene.index` is a block id used for the train/val split, so neighbouring video frames stay on the same side.

  python3 -I vision/tools/convert_public.py agrob RAW_DIR OUT_DIR
  python3 -I vision/tools/convert_public.py plantdoc REPO_DIR OUT_DIR"""
from __future__ import annotations
import argparse, glob, json, os
import xml.etree.ElementTree as ET

from PIL import Image

OUT_WH = (576, 1024)
AGROB_CLASS = {"unriped": "green", "breaking": "turning", "reddish": "ripe", "riped": "ripe"}
PLANTDOC_MAP = {"Tomato Early blight leaf": "leaf_spots", "Tomato Septoria leaf spot": "leaf_spots",
                "Tomato leaf bacterial spot": "leaf_spots", "Tomato leaf late blight": "leaf_spots",
                "Tomato leaf yellow virus": "chlorosis", "Tomato leaf": None}            # None = healthy, no box
MIN_INSIDE = 0.4                                     # keep a cut box if at least 40 % of it is inside the crop


def voc_objects(xml_path: str):
    root = ET.parse(xml_path).getroot()
    for o in root.findall("object"):
        b = o.find("bndbox")
        yield o.find("name").text.strip(), [float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]


def write(out: str, stem: str, im: Image.Image, objs: list[dict], source: str, block: int):
    im.save(os.path.join(out, "images", stem + ".jpg"), quality=92)
    lab = {"image": f"images/{stem}.jpg", "width": OUT_WH[0], "height": OUT_WH[1], "source": source,
           "scene": {"index": block}, "objects": objs}
    json.dump(lab, open(os.path.join(out, "labels", stem + ".json"), "w"))


def crop_boxes(boxes, x0, y0, x1, y1, sx, sy):
    out = []
    for cls, (a, b, c, d) in boxes:
        ix0, iy0, ix1, iy1 = max(a, x0), max(b, y0), min(c, x1), min(d, y1)
        if ix1 <= ix0 or iy1 <= iy0 or (ix1 - ix0) * (iy1 - iy0) < MIN_INSIDE * (c - a) * (d - b):
            continue
        out.append((cls, [round((ix0 - x0) * sx, 1), round((iy0 - y0) * sy, 1),
                          round((ix1 - ix0) * sx, 1), round((iy1 - iy0) * sy, 1)]))
    return out


def agrob(raw: str, out: str):
    root = glob.glob(os.path.join(raw, "**", "JPEGImages"), recursive=True)[0].rsplit("/", 1)[0]
    frames = sorted(glob.glob(os.path.join(root, "JPEGImages", "*.jpg")))
    n = 0
    for i, f in enumerate(frames):
        stem = os.path.basename(f)[:-4]
        boxes = [(AGROB_CLASS[c.lower()], b) for c, b in voc_objects(os.path.join(root, "Annotations", stem + ".xml"))]
        im = Image.open(f).convert("RGB")
        W, H = im.size
        cw = round(H * OUT_WH[0] / OUT_WH[1])                     # 405 px wide portrait crop at full height
        for k, x0 in enumerate((0, (W - cw) // 2, W - cw)):
            cb = crop_boxes(boxes, x0, 0, x0 + cw, H, OUT_WH[0] / cw, OUT_WH[1] / H)
            objs = [{"type": "fruit", "cls": c, "bbox": b, "ignore": False} for c, b in cb]
            write(out, f"agrob_{stem}_c{k}", im.crop((x0, 0, x0 + cw, H)).resize(OUT_WH, Image.BILINEAR), objs,
                  "agrobtomato_cc_by_4.0", i // 25)
            n += 1
    print(f"agrob: {len(frames)} frames -> {n} crops")


def plantdoc(repo: str, out: str):
    n = skipped = 0
    for xml in sorted(glob.glob(os.path.join(repo, "**", "*.xml"), recursive=True)):
        objs = list(voc_objects(xml))
        names = {c for c, _ in objs}
        if not any(c.startswith("Tomato") for c in names):
            continue
        if any(c not in PLANTDOC_MAP for c in names):
            skipped += 1
            continue
        img = next((xml[:-4] + e for e in (".jpg", ".jpeg", ".png", ".JPG") if os.path.exists(xml[:-4] + e)), None)
        if img is None:
            continue
        try:
            im = Image.open(img).convert("RGB")
        except OSError:
            continue
        W, H = im.size
        s = min(OUT_WH[0] / W, OUT_WH[1] / H)                       # letterbox: whole image, grey padding
        nw, nh = round(W * s), round(H * s)
        ox, oy = (OUT_WH[0] - nw) // 2, (OUT_WH[1] - nh) // 2
        canvas = Image.new("RGB", OUT_WH, (114, 114, 114))
        canvas.paste(im.resize((nw, nh), Image.BILINEAR), (ox, oy))
        boxes = []
        for c, (a, b, c2, d) in objs:
            kind = PLANTDOC_MAP[c]
            if kind is None:
                continue
            boxes.append({"type": "anomaly", "cls": kind, "severity": "real",
                          "bbox": [round(ox + a * s, 1), round(oy + b * s, 1), round((c2 - a) * s, 1), round((d - b) * s, 1)]})
        stem = "plantdoc_" + os.path.basename(xml)[:-4].replace(" ", "_")[:80]
        write(out, f"{stem}_{n}", canvas, boxes, "plantdoc_cc_by_4.0", n)
        n += 1
    print(f"plantdoc: {n} tomato images, {skipped} skipped (unmapped tomato disease)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=("agrob", "plantdoc"))
    ap.add_argument("src")
    ap.add_argument("out")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "labels"), exist_ok=True)
    (agrob if a.dataset == "agrob" else plantdoc)(a.src, a.out)


if __name__ == "__main__":
    main()
