"""Run a VLM eye (cloud/eyes/inspect.py prompt) on every side-camera plant stop of a re-rendered route, reading the
frames straight out of the raw side_<camera>.mp4 videos (no extracted files on disk), and check the result against
the scene's ground-truth anomaly plants, especially the shape-level ones the RT-DETR detector cannot box (wilting,
defoliation).

  EYES_API_KEY=... python3 scripts/vlm_side.py --videos <videos_dir> --detection <detection_dir> --rec <run>/rec \
      --base-url https://api.tokenfactory.nebius.com/v1/ --model nvidia/Nemotron-3.5-Super-VL --out vlm/"""
from __future__ import annotations
import argparse, collections, io, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HEAD_Y = 3.26
CAMS = ("left_low", "left_mid", "left_high", "right_low", "right_mid", "right_high")


def frames(video: Path, w=1080, h=1920):
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, required=True); ap.add_argument("--detection", type=Path, required=True)
    ap.add_argument("--rec", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--match-m", type=float, default=0.7)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    from openai import OpenAI
    from cloud.eyes.inspect import inspect_image
    client = OpenAI(base_url=a.base_url, api_key=os.environ["EYES_API_KEY"], timeout=180.0)
    det = [json.loads(l) for l in open(a.detection / "findings.jsonl")]
    meta = collections.defaultdict(list)                  # per camera, in time order = video frame order
    for r in det:
        ns = int(Path(r["image"]).stem.split("_")[0]); meta[r["camera"]].append((ns, {k: r[k] for k in ("camera", "metre", "aisle", "image")}))
    jobs = []
    for cam in CAMS:
        ms = [m for _, m in sorted(meta[f"mast_{cam}"], key=lambda x: x[0])]
        for m, raw in zip(ms, frames(a.videos / f"side_{cam}.mp4")):
            buf = io.BytesIO(); Image.frombytes("RGB", (1080, 1920), raw).save(buf, "JPEG", quality=92)
            jobs.append((m, buf.getvalue()))
    print(f"{len(jobs)} frames from the side videos -> {a.model}", flush=True)

    def run(job):
        m, img = job
        try:
            f = inspect_image(img, dict(m, purpose="inspect"), client=client, model=a.model)
            f = f.model_dump() if hasattr(f, "model_dump") else dict(f)
        except Exception as e:
            f = {"status": "error", "issues": [], "summary": str(e)[:300]}
        return {**m, **f}
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run, jobs))
    with open(a.out / "vlm_findings.jsonl", "w") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    flagged = [r for r in rows if r.get("status") in ("suspect", "disease")]
    gt = [g for g in json.load(open(a.rec / "ground_truth_plants.json")) if g.get("synthetic_anomaly")]
    checks = []
    for g in gt:
        aisle = f"{g['tunnel']}/aisle-02"; m = HEAD_Y - g["xy"][1]
        near = [r for r in flagged if r.get("aisle") == aisle and r.get("metre") is not None and abs(r["metre"] - m) <= a.match_m]
        checks.append({"plant": g["prim"], "aisle": aisle, "metre": round(m, 2), "truth": g["synthetic_anomaly"], "flagged": bool(near),
                       "labels_nearby": sorted({i.get("label", "") for r in near for i in r.get("issues", [])})[:6]})
    far = [r for r in flagged if not any(r.get("aisle") == f"{g['tunnel']}/aisle-02" and r.get("metre") is not None
                                         and abs(r["metre"] - (HEAD_Y - g["xy"][1])) <= a.match_m for g in gt)]
    summary = {"model": a.model, "frames": len(rows), "status": dict(collections.Counter(r.get("status") for r in rows)),
               "found": f"{sum(c['flagged'] for c in checks)}/{len(checks)}", "flagged_not_near_any_anomaly_plant": len(far),
               "ground_truth_check": checks}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "ground_truth_check"}, ensure_ascii=False))
    for c in checks:
        print(c["aisle"], c["metre"], c["truth"], "FLAGGED" if c["flagged"] else "missed", c["labels_nearby"][:3])


if __name__ == "__main__":
    main()
