"""Evaluate a trained checkpoint on one dataset's validation split (same scene/block split as training).

  .venv-vision/bin/python vision/eval_rtdetr.py --model vision_runs/rtdetr_v1_synth/best --data vision_data/public_agrob
Use --all to score every image (e.g. a real test set that was never used for training)."""
from __future__ import annotations
import argparse, json, os, sys

import torch
from torch.utils.data import DataLoader
from transformers import RTDetrV2ForObjectDetection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rtdetr import SynthDataset, collate, evaluate, load_split      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    train, val = load_split(a.data)
    items = train + val if a.all else val
    model = RTDetrV2ForObjectDetection.from_pretrained(a.model).to("cuda")
    res = evaluate(model, DataLoader(SynthDataset(items, False), 8, num_workers=4, collate_fn=collate), "cuda")
    res = {"model": a.model, "data": a.data, "images": len(items), **res}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
