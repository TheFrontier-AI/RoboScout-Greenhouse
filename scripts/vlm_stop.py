"""One VLM request per plant stop: all six side cameras in one message (left high/mid/low = the plant in the left row,
right high/mid/low = the plant in the right row), so the model sees each plant top to bottom and the number of calls
is the number of stops, not stops x cameras. Frames come straight out of the raw side_<camera>.mp4 videos. Results are
checked against the scene's ground-truth anomaly plants.

  EYES_API_KEY=... python3 scripts/vlm_stop.py --videos <videos_dir> --detection <detection_dir> --rec <run>/rec \
      --base-url https://api.tokenfactory.nebius.com/v1/ --model nvidia/Nemotron-3.5-Super-VL --out vlm_stops/

The pieces (group_png_stops, encode_views, ask_stop, summarize) are reused by cloud/pipeline (step `vlm_stops`)."""
from __future__ import annotations
import argparse, base64, collections, io, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HEAD_Y = 3.26
ORDER = ("left_high", "left_mid", "left_low", "right_high", "right_mid", "right_low")

PROMPT = """These six images come from the side cameras of a greenhouse robot that has stopped beside two tomato plants.
Images 1-3 (high, mid, low) show the plant in the LEFT row, images 4-6 (high, mid, low) the plant in the RIGHT row.
Context: {ctx}.
For each of the two plants, look at the whole plant top to bottom for: wilting (drooping, limp leaves and stems),
defoliation (missing or dropped leaves, bare stem sections), yellowing/chlorosis, leaf spots or lesions (early blight
rings, late blight patches, leaf mold), fruit lesions or rot, pests. Be conservative: "disease" only for clear visible
evidence, "suspect" when unsure or occluded, "healthy" when nothing is wrong, "not_plant" if no plant is visible.
Do not invent issues. confidence is 0..1.
{detail}Reply with ONLY this JSON object, no markdown:
{{"left": {{"status": "healthy|suspect|disease|not_plant", "issues": [{{"label": str, "confidence": float, "evidence": str}}], "summary": str{extra}}},
  "right": {{"status": "...", "issues": [...], "summary": str{extra}}}}}"""
DETAIL = """Explain like an agronomist walking the row: in "description" write 3-5 sentences per plant covering the canopy
(leaf colour, turgor, density, any missing leaves), the stems, the fruit you can see (how many, which colours/ripeness,
any damage), where in the six views each observation is (high/mid/low), and in "action" one practical next step for the
farmer. Only describe what is visible.
"""


def frames(video: Path, w=1080, h=1920):
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    n = w * h * 3
    while (b := p.stdout.read(n)) and len(b) == n:
        yield b


def parse(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    d = json.loads(m.group(0))
    return {s: d.get(s, {"status": "error", "issues": [], "summary": "missing"}) for s in ("left", "right")}


def group_png_stops(rec: Path, side_dir: Path | None = None) -> list[dict]:
    """Group the six side captures of each plant stop from a run_farm recording (rec/frames.jsonl + rec/side PNGs,
    falling back to the JPEGs in rec/frames). Returns [{"stop", "ns", "aisle", "metre", "purpose", "paths": {cam: Path}}]."""
    side_dir = side_dir or rec / "side"
    groups: dict = collections.OrderedDict()
    for line in open(rec / "frames.jsonl"):
        f = json.loads(line)
        if str(f.get("camera", "")).startswith("mast_") and f.get("purpose") in ("inspect", "revisit"):
            key = (f.get("aisle"), round(float(f["metre"]), 2) if f.get("metre") is not None else None, f["purpose"])
            groups.setdefault(key, {})[f["camera"][5:]] = f
    stops = []
    for (aisle, metre, purpose), g in groups.items():
        if not all(c in g for c in ORDER):
            continue
        paths = {}
        for cam in ORDER:
            fid = g[cam]["frame_id"]
            p = side_dir / f"{fid:06d}_mast_{cam}.png"
            if not p.exists():
                p = next(side_dir.glob(f"*{fid}_mast_{cam}.png"), None) or rec / g[cam].get("image_path", f"frames/{fid:06d}_mast_{cam}.jpg")
            paths[cam] = p
        if all(p.exists() for p in paths.values()):
            stops.append({"stop": len(stops), "ns": int(float(g["left_mid"].get("ts", 0)) * 1e9), "aisle": aisle, "metre": metre,
                          "purpose": purpose, "frame_ids": {cam: g[cam]["frame_id"] for cam in ORDER}, "paths": paths})
    return stops


def encode_views(images, long_side: int = 1280) -> list[str]:
    """PIL images (or paths) in ORDER -> base64 JPEGs with the long side resized to long_side."""
    from PIL import Image
    out = []
    for im in images:
        im = (Image.open(im) if not hasattr(im, "save") else im).convert("RGB"); im.thumbnail((long_side, long_side))
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=88); out.append(base64.b64encode(buf.getvalue()).decode())
    return out


def ask_stop(client, model: str, s: dict, detailed: bool = False, retries: int = 3) -> dict:
    """One chat request for one stop (s["images"] = six base64 JPEGs in ORDER). Never raises: errors become status=error."""
    ctx = f"{s['aisle']}, metre {s['metre']:.2f} along the aisle" if s.get("metre") is not None else "greenhouse aisle"
    content = []
    for i, (cam, b64) in enumerate(zip(ORDER, s["images"]), 1):
        content += [{"type": "text", "text": f"Image {i}: {cam.replace('_', ' ')}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]
    content.append({"type": "text", "text": PROMPT.format(ctx=ctx, detail=DETAIL if detailed else "",
                                                          extra=', "description": str, "action": str' if detailed else "")})
    keep = {k: s[k] for k in ("stop", "ns", "aisle", "metre", "purpose", "frame_ids") if k in s}
    t0 = time.time(); err = "no attempt"
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}],
                                               temperature=0.2, max_tokens=4096)
            msg = r.choices[0].message
            out = parse(msg.content or getattr(msg, "reasoning_content", None) or "")
            return keep | out | {"latency_s": round(time.time() - t0, 1), "tokens": getattr(r.usage, "total_tokens", None)}
        except Exception as e:
            err = str(e)[:300]; time.sleep(3 if attempt < retries - 1 else 0)
    return keep | {"left": {"status": "error", "issues": [], "summary": err}, "right": {"status": "error", "issues": [], "summary": err}}


def summarize(rows: list[dict], rec: Path, model: str, match_m: float = 0.7) -> dict:
    """Status counts + check of flagged stops against the scene's ground-truth anomaly plants (if the run has them)."""
    flagged = [(r, side) for r in rows for side in ("left", "right") if r[side].get("status") in ("suspect", "disease")]
    gt_path = rec / "ground_truth_plants.json"
    gt = [g for g in json.load(open(gt_path)) if g.get("synthetic_anomaly")] if gt_path.exists() else []
    checks = []
    for g in gt:
        aisle = f"{g['tunnel']}/aisle-02"; m = HEAD_Y - g["xy"][1]
        near = [(r, sd) for r, sd in flagged if r["aisle"] == aisle and r["metre"] is not None and abs(r["metre"] - m) <= match_m]
        checks.append({"plant": g["prim"], "aisle": aisle, "metre": round(m, 2), "truth": g["synthetic_anomaly"], "flagged": bool(near),
                       "labels_nearby": sorted({i.get("label", "") for r, sd in near for i in r[sd].get("issues", [])})[:6]})
    far = [1 for r, sd in flagged if not any(r["aisle"] == f"{g['tunnel']}/aisle-02" and r["metre"] is not None
                                             and abs(r["metre"] - (HEAD_Y - g["xy"][1])) <= match_m for g in gt)]
    tok = sum(r.get("tokens") or 0 for r in rows)
    return {"model": model, "requests": len(rows), "plants_judged": 2 * len(rows), "total_tokens": tok,
            "status": dict(collections.Counter(r[s].get("status") for r in rows for s in ("left", "right"))),
            "flagged_plants": len(flagged),
            "anomaly_plants_flagged": f"{sum(c['flagged'] for c in checks)}/{len(checks)}" if gt else None,
            "flags_not_near_any_anomaly_plant": len(far) if gt else None, "ground_truth_check": checks if gt else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path); ap.add_argument("--detection", type=Path)
    ap.add_argument("--side-pngs", type=Path, help="instead of videos: a run_farm rec/side dir (<frame_id>_<mast_cam>.png) + rec/frames.jsonl")
    ap.add_argument("--rec", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--match-m", type=float, default=0.7)
    ap.add_argument("--detailed", action="store_true", help="ask for an agronomist-style description and a next step per plant")
    ap.add_argument("--long-side", type=int, default=1280, help="resize each view so its long side is this many pixels")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key=os.environ["EYES_API_KEY"], timeout=240.0)
    if a.side_pngs:                                       # original run: group the six captures of each plant stop
        stops = [s | {"images": encode_views([s["paths"][c] for c in ORDER], a.long_side)} for s in group_png_stops(a.rec, a.side_pngs)]
    else:
        meta = collections.defaultdict(list)
        for line in open(a.detection / "findings.jsonl"):
            r = json.loads(line); ns = int(Path(r["image"]).stem.split("_")[0])
            meta[r["camera"]].append((ns, {"metre": r["metre"], "aisle": r["aisle"]}))
        per_cam = {}
        for cam in ORDER:
            ms = sorted(meta[f"mast_{cam}"], key=lambda x: x[0])
            per_cam[cam] = list(zip(ms, frames(a.videos / f"side_{cam}.mp4")))
        n_stops = min(len(v) for v in per_cam.values())
        stops = []
        for k in range(n_stops):
            (ns, m), _ = per_cam["left_mid"][k]
            imgs = encode_views([Image.frombytes("RGB", (1080, 1920), per_cam[cam][k][1]) for cam in ORDER], a.long_side)
            stops.append({"stop": k, "ns": ns, **m, "images": imgs})
    n_stops = len(stops)
    print(f"{n_stops} stops x 6 views -> {n_stops} requests to {a.model}", flush=True)

    def run(s):
        return ask_stop(client, a.model, s, a.detailed)
    done, rows = 0, []
    with ThreadPoolExecutor(a.workers) as ex, open(a.out / "vlm_stops.jsonl", "w") as fo:
        for r in ex.map(run, stops):
            rows.append(r); fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush(); done += 1
            if done % 10 == 0:
                print(f"{done}/{n_stops} stops", flush=True)
    summary = summarize(rows, a.rec, a.model, a.match_m)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "ground_truth_check"}, ensure_ascii=False))
    for c in summary.get("ground_truth_check") or []:
        print(c["aisle"], c["metre"], c["truth"], "FLAGGED" if c["flagged"] else "missed", c["labels_nearby"][:3])


if __name__ == "__main__":
    main()
