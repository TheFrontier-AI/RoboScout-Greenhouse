"""Run the RT-DETRv2 side-camera detector on every plant-stop image of a re-rendered route and report the results in
separate categories: harvest (fruit grading), plant count, leaf anomalies (disease screening). Leaf anomalies and fruit
lesions are checked against the scene's ground-truth anomaly plants (sim.crop_swap DEFAULT_ANOMALIES).

  python3 scripts/detect_side.py --frames <render_out> --rec <run>/rec --model vision_runs/rtdetr_v2_mixed/best --out det/

`summarize(rows, rec, ...)` is reused by cloud/pipeline (step `detect`) on the robot's own rec/side PNGs."""
from __future__ import annotations
import argparse, collections, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HEAD_Y = 3.26                                      # aisle metre 0 (first plant) in the scene frame
AISLE_X = {"tunnel-09": 0.0, "tunnel-10": 9.015}
DETECTABLE = {"chlorosis_strong": "leaf_chlorosis", "leaf_spots_strong": "leaf_spots", "fruit_lesion_strong": "fruit_lesion"}


def summarize(rows: list[dict], rec: Path, model: str, threshold: float, match_m: float = 0.7) -> dict:
    """Per-image findings (vision.detector.to_finding + aisle/metre/camera/image) -> the three categories.
    The ground-truth check is only filled in when the run has rec/ground_truth_plants.json (sim runs)."""
    # 1. harvest (fruit grading), per aisle; each frame counts only its own strip/height band (detector rule)
    harvest = collections.defaultdict(collections.Counter)
    for r in rows:
        for k, v in (r.get("harvest") or {}).items():
            if k in ("pass", "reject", "not_ready"):
                harvest[r["aisle"]][k] += v
    # 2. plant count (plant bases owned by each frame)
    plants = collections.Counter(r["aisle"] for r in rows for _ in (r.get("plants") or []))
    # 3. leaf anomalies, clustered by aisle + 0.5 m
    anomalies = []
    for r in rows:
        for i in r.get("issues") or []:
            anomalies.append({"aisle": r["aisle"], "metre": r["metre"], "camera": r["camera"], "label": i.get("label"),
                              "confidence": round(float(i.get("confidence", 0)), 3), "image": r["image"]})
    gt_path = rec / "ground_truth_plants.json"
    gt = [g for g in json.load(open(gt_path)) if g.get("synthetic_anomaly")] if gt_path.exists() else []
    checks = []
    for g in gt:
        aisle = f"{g['tunnel']}/aisle-02"; m = HEAD_Y - g["xy"][1]
        want = DETECTABLE.get(g["synthetic_anomaly"])
        near = [x for x in anomalies if x["aisle"] == aisle and x["metre"] is not None and abs(x["metre"] - m) <= match_m]
        fruit_rej = [r for r in rows if r["aisle"] == aisle and r["metre"] is not None and abs(r["metre"] - m) <= match_m
                     and (r.get("harvest") or {}).get("reject", 0) > 0]
        hit = (bool(fruit_rej) if want == "fruit_lesion" else any(x["label"] and want and want.split("_")[-1] in x["label"] for x in near)) if want else None
        checks.append({"plant": g["prim"], "aisle": aisle, "metre": round(m, 2), "truth": g["synthetic_anomaly"],
                       "detectable_by_this_model": want is not None, "found": hit,
                       "detections_nearby": [(x["label"], x["confidence"]) for x in near][:5]})
    far = [x for x in anomalies if not any(x["aisle"] == f"{g['tunnel']}/aisle-02" and x["metre"] is not None
                                           and abs(x["metre"] - (HEAD_Y - g["xy"][1])) <= match_m for g in gt)]
    return {"images": len(rows), "model": model, "threshold": threshold,
            "1_harvest_fruit_grading": {k: dict(v) for k, v in harvest.items()},
            "2_plant_count": dict(plants),
            "3_leaf_anomalies": {"detections": len(anomalies), "by_label": dict(collections.Counter(x["label"] for x in anomalies)),
                                 "spots": anomalies,
                                 "ground_truth_check": checks if gt else None,
                                 "found_of_detectable": f"{sum(1 for c in checks if c['found'])}/{sum(1 for c in checks if c['detectable_by_this_model'])}" if gt else None,
                                 "detections_not_near_any_anomaly_plant": len(far) if gt else None}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True); ap.add_argument("--rec", type=Path, required=True)
    ap.add_argument("--model", default="vision_runs/rtdetr_v2_mixed/best"); ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--match-m", type=float, default=0.7)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    from vision.detector import Detector, to_finding
    caps = {}                                            # frames.jsonl: what the robot knew at each capture
    for line in open(a.rec / "frames.jsonl"):
        e = json.loads(line)
        if str(e.get("camera", "")).startswith("mast_") and e.get("ts") is not None:
            e["t"] = e["ts"]; caps[(int(round(e["ts"] * 1e9)), e["camera"])] = e
    det = Detector(a.model, threshold=a.threshold)
    rows = []
    for f in sorted((a.frames / "side").glob("*_plant_*.png")):
        ns, cam = int(f.stem.split("_")[0]), "mast_" + f.stem.split("_plant_")[1]
        e = caps.get((ns, cam)) or min((v for k, v in caps.items() if k[1] == cam), key=lambda v: abs(v["t"] * 1e9 - ns))
        meta = {"camera": cam, "metre": e.get("metre"), "aisle": e.get("aisle")}
        im = Image.open(f).convert("RGB")
        fd = to_finding(det.detect(im), im, meta)
        rows.append({"image": str(f.relative_to(a.frames)), **meta, **fd})
    with open(a.out / "findings.jsonl", "w") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize(rows, a.rec, a.model, a.threshold, a.match_m)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "3_leaf_anomalies"}, indent=1, ensure_ascii=False))
    la = summary["3_leaf_anomalies"]
    print("leaf anomalies:", la["by_label"], "found", la["found_of_detectable"], "far", la["detections_not_near_any_anomaly_plant"])


if __name__ == "__main__":
    main()
