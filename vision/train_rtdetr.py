"""Fine-tune RT-DETRv2 (Apache-2.0, PekingU/rtdetr_v2_r18vd) on the synthetic side-camera dataset.

Classes: fruit ripe / turning / green / lesion, plant_base (one per plant, for counting), leaf chlorosis and
strong leaf spots. Split is by rendered scene, so validation rows were never seen in training.

  .venv-vision/bin/python vision/train_rtdetr.py --data vision_data/synth_v1 --out vision_runs/rtdetr_v1 --epochs 40"""
from __future__ import annotations
import argparse, glob, json, math, os, random, time

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from transformers import RTDetrV2ForObjectDetection

CLASSES = ("fruit_ripe", "fruit_turning", "fruit_green", "fruit_lesion", "plant_base", "leaf_chlorosis", "leaf_spots")
BASE_MODEL = "PekingU/rtdetr_v2_r18vd"


def class_of(o: dict) -> str | None:
    if o["type"] == "fruit":
        return None if o.get("ignore") else "fruit_" + o["cls"]
    if o["type"] == "plant_base":
        return "plant_base"
    if o["type"] == "anomaly":
        return {"chlorosis": "leaf_chlorosis", "leaf_spots": "leaf_spots"}.get(o["cls"])
    return None


def load_split(root: str, val_every: int = 8):
    """Scene-level split: every val_every-th scene of each seed is validation."""
    train, val = [], []
    for p in sorted(glob.glob(os.path.join(root, "labels", "*.json"))):
        lab = json.load(open(p))
        boxes, cls = [], []
        for o in lab["objects"]:
            c = class_of(o)
            if c is not None:
                boxes.append(o["bbox"])
                cls.append(CLASSES.index(c))
        item = {"image": os.path.join(root, lab["image"]), "boxes": boxes, "cls": cls}
        (val if lab["scene"]["index"] % val_every == 0 else train).append(item)
    return train, val


class SynthDataset(Dataset):
    def __init__(self, items, train: bool):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        im = Image.open(it["image"]).convert("RGB")
        W, H = im.size
        boxes = np.array(it["boxes"], dtype=np.float32).reshape(-1, 4)
        if self.train:
            im, boxes = augment(im, boxes)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        cxcywh = np.stack([(boxes[:, 0] + boxes[:, 2] / 2) / W, (boxes[:, 1] + boxes[:, 3] / 2) / H,
                           boxes[:, 2] / W, boxes[:, 3] / H], axis=1) if len(boxes) else np.zeros((0, 4), np.float32)
        return x, {"class_labels": torch.tensor(it["cls"], dtype=torch.long),
                   "boxes": torch.from_numpy(cxcywh.clip(0, 1)).float()}


def augment(im: Image.Image, boxes: np.ndarray):
    """Photometric + low-resolution augmentation. Hue is barely touched: colour is the ripeness label."""
    W, H = im.size
    if random.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = boxes.copy()
        boxes[:, 0] = W - boxes[:, 0] - boxes[:, 2]
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = a ** random.uniform(0.6, 1.6)                                    # exposure / backlight
    a = a * random.uniform(0.6, 1.3) + random.uniform(-0.05, 0.05)
    grey = a.mean(axis=2, keepdims=True)
    a = grey + (a - grey) * random.uniform(0.7, 1.2)                      # saturation
    a = a * np.array([random.uniform(0.95, 1.05) for _ in range(3)], np.float32)   # mild white balance
    if random.random() < 0.3:
        a = a + np.random.normal(0, random.uniform(0.005, 0.03), a.shape).astype(np.float32)
    im = Image.fromarray((a.clip(0, 1) * 255).astype(np.uint8))
    if random.random() < 0.3:                                             # e.g. Isaac 180x320 side cameras
        s = random.uniform(0.3, 0.7)
        im = im.resize((int(W * s), int(H * s)), Image.BILINEAR).resize((W, H), Image.BILINEAR)
    elif random.random() < 0.2:
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5)))
    return im, boxes


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


# ---------------------------------------------------------------- evaluation (per-class AP@0.5, counting error)
def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = (np.minimum(ax2, bx2) - np.maximum(ax1, bx1)).clip(0)
    ih = (np.minimum(ay2, by2) - np.maximum(ay1, by1)).clip(0)
    inter = iw * ih
    return inter / (a[2] * a[3] + b[:, 2] * b[:, 3] - inter + 1e-9)


@torch.no_grad()
def evaluate(model, loader, device, score_thr=0.5):
    model.eval()
    dets = {c: [] for c in range(len(CLASSES))}            # (score, is_tp)
    n_gt = {c: 0 for c in range(len(CLASSES))}
    count_err = {c: [] for c in range(len(CLASSES))}
    for x, targets in loader:
        out = model(pixel_values=x.to(device))
        prob = out.logits.sigmoid().float().cpu().numpy()
        pbox = out.pred_boxes.float().cpu().numpy()
        for b, t in enumerate(targets):
            gl, gb = t["class_labels"].numpy(), t["boxes"].numpy()
            for c in range(len(CLASSES)):
                g = gb[gl == c]
                n_gt[c] += len(g)
                sc = prob[b, :, c]
                order = np.argsort(-sc)
                used = np.zeros(len(g), bool)
                kept = 0
                for q in order[:100]:
                    if sc[q] < 0.05:
                        break
                    tp = False
                    if len(g):
                        ious = iou(pbox[b, q], g)
                        ious[used] = 0
                        j = int(ious.argmax())
                        if ious[j] >= 0.5:
                            used[j], tp = True, True
                    dets[c].append((float(sc[q]), tp))
                    kept += sc[q] >= score_thr
                count_err[c].append(kept - len(g))
    res = {}
    for c, name in enumerate(CLASSES):
        d = sorted(dets[c], key=lambda z: -z[0])
        tps = np.cumsum([z[1] for z in d]) if d else np.zeros(0)
        prec = tps / np.arange(1, len(d) + 1) if d else np.zeros(0)
        rec = tps / max(n_gt[c], 1)
        ap = 0.0
        for r in np.linspace(0, 1, 101):                     # COCO-style 101-point interpolation
            p = prec[rec >= r]
            ap += (p.max() if len(p) else 0.0) / 101
        res[name] = {"AP50": round(ap, 4), "n_gt": n_gt[c],
                     "count_mae_per_image": round(float(np.mean(np.abs(count_err[c]))), 3),
                     "count_bias_per_image": round(float(np.mean(count_err[c])), 3)}
    res["mAP50"] = round(float(np.mean([res[n]["AP50"] for n in CLASSES if res[n]["n_gt"]])), 4)
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    train, val = [], []
    for d in a.data:
        t, v = load_split(d)
        train += t; val += v
    print(f"train {len(train)} images, val {len(val)} images", flush=True)
    dev = torch.device("cuda")
    model = RTDetrV2ForObjectDetection.from_pretrained(
        BASE_MODEL, num_labels=len(CLASSES), ignore_mismatched_sizes=True,
        id2label=dict(enumerate(CLASSES)), label2id={c: i for i, c in enumerate(CLASSES)}).to(dev)
    tl = DataLoader(SynthDataset(train, True), a.batch, shuffle=True, num_workers=a.workers, collate_fn=collate,
                    drop_last=True, persistent_workers=True)
    vl = DataLoader(SynthDataset(val, False), a.batch, num_workers=a.workers, collate_fn=collate)
    backbone = [p for n, p in model.named_parameters() if "backbone" in n]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n]
    opt = torch.optim.AdamW([{"params": backbone, "lr": a.lr * 0.1}, {"params": rest, "lr": a.lr}], weight_decay=1e-4)
    total = a.epochs * len(tl)
    warm = min(300, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    best, history = -1.0, []
    step = 0
    for ep in range(a.epochs):
        t0, losses = time.time(), []
        for x, targets in tl:
            targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(pixel_values=x.to(dev), labels=targets)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step(); sched.step(); step += 1
            losses.append(out.loss.item())
        rec = {"epoch": ep + 1, "loss": round(float(np.mean(losses)), 4), "sec": round(time.time() - t0, 1)}
        if (ep + 1) % 5 == 0 or ep + 1 == a.epochs:
            rec["val"] = evaluate(model, vl, dev)
            if rec["val"]["mAP50"] > best:
                best = rec["val"]["mAP50"]
                model.save_pretrained(os.path.join(a.out, "best"))
        history.append(rec)
        print(json.dumps(rec), flush=True)
        json.dump({"base_model": BASE_MODEL, "classes": CLASSES, "data": a.data, "n_train": len(train),
                   "n_val": len(val), "args": vars(a), "history": history, "best_mAP50": best},
                  open(os.path.join(a.out, "train_report.json"), "w"), indent=1)
    model.save_pretrained(os.path.join(a.out, "last"))


if __name__ == "__main__":
    main()
