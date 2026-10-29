"""Side-camera detector: fruit grading, plant bases (for counting) and leaf anomalies -> one finding per frame.

The finding keeps the Cosmos schema (status / issues with bbox_norm / summary) so the hub and dashboard need no
change, and adds `harvest` (fruit counts and pass/fail) and `plants` (plant-base positions along the aisle).

Grading rule (explainable): ripe red = PASS, lesion = REJECT, green/turning = NOT READY. The detector's colour
class is cross-checked with the mean hue inside the box; disagreements are reported, not silently fixed.

Each side frame "owns" the 0.5 m strip of row centred on its capture metre (r1_nav.cameras.SIDE_STEP_M) and the
height band of its mast camera (split halfway between camera heights), so fruit and plants are counted once even
though neighbouring frames and the low/mid/high cameras overlap.

  .venv-vision/bin/python vision/detector.py --model vision_runs/rtdetr_v1/best IMAGE [--metre 12.5 --camera mast_left_low]"""
from __future__ import annotations
import argparse, colorsys, json, math, os, sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from r1_nav.cameras import SIDE_STEP_M                       # noqa: E402

INPUT_WH = (576, 1024)                                    # training resolution (9:16 portrait, like the C260 mast cams)
ROW_DEPTH_M = 0.95                                        # camera to near plant row: centre aisle 1.27 m, mast offset
SIDE_HFOV_DEG = 40.0                                      # 9:16 portrait at ~65 deg vertical FOV
SIDE_VFOV_DEG = 65.0
MAST_Z_M = {"low": 0.7945, "mid": 1.2145, "high": 1.6445}  # robot_sensors.usda plant cameras, above the floor
HEIGHT_BANDS_M = {"low": (-1.0, 1.0045), "mid": (1.0045, 1.4295), "high": (1.4295, 9.0)}
FRUIT_GRADE = {"fruit_ripe": "pass", "fruit_lesion": "reject", "fruit_turning": "not_ready", "fruit_green": "not_ready"}
DISEASE_LABELS = {"leaf_chlorosis": "chlorosis (yellowing leaves)", "leaf_spots": "leaf spots",
                  "fruit_lesion": "fruit lesion"}


def owned_strip(row_depth_m: float = ROW_DEPTH_M, hfov_deg: float = SIDE_HFOV_DEG) -> tuple[float, float]:
    """Image x-range (0..1) covering SIDE_STEP_M of row at row_depth_m."""
    width_m = 2 * row_depth_m * math.tan(math.radians(hfov_deg) / 2)
    half = min(0.5, SIDE_STEP_M / width_m / 2)
    return 0.5 - half, 0.5 + half


def mast_slot(camera: str) -> str | None:
    return next((k for k in MAST_Z_M if str(camera).endswith(k)), None)


def box_hue_class(rgb: np.ndarray, box) -> str:
    """Hue of the central half of a fruit box, mapped like the training labels (red / turning / green)."""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = box
    cx0, cx1 = int((3 * x0 + x1) / 4 * W), int((x0 + 3 * x1) / 4 * W) + 1
    cy0, cy1 = int((3 * y0 + y1) / 4 * H), int((y0 + 3 * y1) / 4 * H) + 1
    patch = rgb[max(0, cy0):cy1, max(0, cx0):cx1].reshape(-1, 3).astype(np.float32) / 255
    if not len(patch):
        return "unknown"
    h, s, v = colorsys.rgb_to_hsv(*patch.mean(axis=0))
    if v < 0.12:
        return "dark"
    h *= 360
    return "fruit_ripe" if (h < 25 or h > 340) else "fruit_turning" if h < 70 else "fruit_green"


class Detector:
    def __init__(self, model_dir: str, device: str | None = None, threshold: float = 0.5):
        import torch
        from transformers import RTDetrV2ForObjectDetection
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RTDetrV2ForObjectDetection.from_pretrained(model_dir).to(self.device).eval()
        self.labels = self.model.config.id2label
        self.threshold = threshold

    def detect(self, image: Image.Image) -> list[dict]:
        im = image.convert("RGB").resize(INPUT_WH, Image.BILINEAR)
        x = self.torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(self.device)
        with self.torch.no_grad():
            out = self.model(pixel_values=x)
        prob = out.logits.sigmoid()[0].float().cpu().numpy()
        boxes = out.pred_boxes[0].float().cpu().numpy()
        dets = []
        for q in range(prob.shape[0]):
            c = int(prob[q].argmax())
            sc = float(prob[q, c])
            if sc < self.threshold:
                continue
            cx, cy, w, h = boxes[q]
            dets.append({"label": self.labels[c], "score": round(sc, 3),
                         "bbox_norm": [round(float(v), 4) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)]})
        return nms(dets)


def nms(dets: list[dict], thr: float = 0.6) -> list[dict]:
    keep = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(d["label"] != k["label"] or iou(d["bbox_norm"], k["bbox_norm"]) < thr for k in keep):
            keep.append(d)
    return keep


def iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def to_finding(dets: list[dict], image: Image.Image, meta: dict | None = None,
               row_depth_m: float = ROW_DEPTH_M, hfov_deg: float = SIDE_HFOV_DEG) -> dict:
    meta = meta or {}
    rgb = np.asarray(image.convert("RGB"))
    s0, s1 = owned_strip(row_depth_m, hfov_deg)
    width_m = 2 * row_depth_m * math.tan(math.radians(hfov_deg) / 2)
    # left cameras look out of the robot's left side: image x increases toward the aisle start
    sign = -1.0 if str(meta.get("camera", "")).startswith("mast_left") else 1.0
    metre = meta.get("metre")
    slot = mast_slot(meta.get("camera", ""))
    height_m = 2 * row_depth_m * math.tan(math.radians(SIDE_VFOV_DEG) / 2)

    harvest = {"pass": 0, "reject": 0, "not_ready": 0, "by_class": {}, "colour_disagreements": 0}
    plants, issues = [], []
    for d in dets:
        cx = (d["bbox_norm"][0] + d["bbox_norm"][2]) / 2
        cy = (d["bbox_norm"][1] + d["bbox_norm"][3]) / 2
        owned = s0 <= cx < s1
        if slot is not None:
            z = MAST_Z_M[slot] + (0.5 - cy) * height_m
            d["height_m"] = round(z, 2)
            lo, hi = HEIGHT_BANDS_M[slot]
            owned = owned and (d["label"] == "plant_base" or lo <= z < hi)
        d["counted"] = owned
        d["metre"] = None if metre is None else round(float(metre) + sign * (cx - 0.5) * width_m, 2)
        lab = d["label"]
        if lab in FRUIT_GRADE:
            d["grade"] = FRUIT_GRADE[lab]
            if lab != "fruit_lesion":
                hue = box_hue_class(rgb, d["bbox_norm"])
                d["hue_check"] = hue
                if hue not in (lab, "dark", "unknown"):
                    harvest["colour_disagreements"] += owned
            if owned:
                harvest[d["grade"]] += 1
                harvest["by_class"][lab] = harvest["by_class"].get(lab, 0) + 1
        elif lab == "plant_base" and owned:
            plants.append({"metre": d["metre"], "score": d["score"], "bbox_norm": d["bbox_norm"]})
        if lab in DISEASE_LABELS:
            issues.append({"label": DISEASE_LABELS[lab], "confidence": d["score"], "bbox_norm": d["bbox_norm"],
                           "evidence": f"RT-DETR {lab} detection" + (f" at ~{d['metre']} m" if d["metre"] is not None else "")})
    n_sick = len(issues)
    status = "disease" if any(i["confidence"] >= 0.7 for i in issues) else "suspect" if n_sick else "healthy"
    summary = (f"{harvest['pass']} ripe (pass), {harvest['not_ready']} not ready, {harvest['reject']} rejected fruit; "
               f"{len(plants)} plant(s) in strip; " + (f"{n_sick} anomaly region(s)" if n_sick else "no leaf anomaly"))
    return {"status": status, "issues": issues, "summary": summary, "source": "rtdetr",
            "harvest": harvest, "plants": plants, "detections": dets}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--metre", type=float)
    ap.add_argument("--camera", default="mast_right_mid")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    det = Detector(a.model, threshold=a.threshold)
    for p in a.images:
        im = Image.open(p)
        f = to_finding(det.detect(im), im, {"metre": a.metre, "camera": a.camera})
        print(json.dumps({"image": p} | {k: f[k] for k in ("status", "summary", "harvest", "plants")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
