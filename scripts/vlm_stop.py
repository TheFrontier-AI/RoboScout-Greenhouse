"""One VLM request per plant stop: all six side cameras in one message (left high/mid/low = the plant in the left row,
right high/mid/low = the plant in the right row), so the model sees each plant top to bottom and the number of calls
is the number of stops, not stops x cameras. Frames come straight out of the raw side_<camera>.mp4 videos. Results are
checked against the scene's ground-truth anomaly plants.

  EYES_API_KEY=... python3 scripts/vlm_stop.py --videos <videos_dir> --detection <detection_dir> --rec <run>/rec \
      --base-url https://api.tokenfactory.nebius.com/v1/ --model nvidia/Nemotron-3.5-Super-VL --out vlm_stops/"""
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
        fr = [json.loads(l) for l in open(a.rec / "frames.jsonl")]
        groups = collections.OrderedDict()
        for f in fr:
            if str(f.get("camera", "")).startswith("mast_") and f.get("purpose") == "inspect":
                groups.setdefault((f["aisle"], round(f["metre"], 2)), {})[f["camera"][5:]] = f
        per_cam = {cam: [] for cam in ORDER}
        for (aisle, metre), g in groups.items():
            if all(c in g for c in ORDER):
                for cam in ORDER:
                    p_ = a.side_pngs / f"{g[cam]['frame_id']:06d}_mast_{cam}.png"
                    if not p_.exists():
                        p_ = next(a.side_pngs.glob(f"*{g[cam]['frame_id']}_mast_{cam}.png"), p_)
                    per_cam[cam].append(((int(g[cam]["ts"] * 1e9), {"metre": metre, "aisle": aisle}), Image.open(p_).convert("RGB").tobytes()))
        meta = None
    else:
        meta = collections.defaultdict(list)
    for line in (open(a.detection / "findings.jsonl") if meta is not None else []):
        r = json.loads(line); ns = int(Path(r["image"]).stem.split("_")[0])
        meta[r["camera"]].append((ns, {"metre": r["metre"], "aisle": r["aisle"]}))
    if meta is not None:
        per_cam = {}
        for cam in ORDER:
            ms = sorted(meta[f"mast_{cam}"], key=lambda x: x[0])
            per_cam[cam] = list(zip(ms, frames(a.videos / f"side_{cam}.mp4")))
    n_stops = min(len(v) for v in per_cam.values())
    stops = []
    for k in range(n_stops):
        (ns, m), _ = per_cam["left_mid"][k]
        imgs = []
        for cam in ORDER:
            im = Image.frombytes("RGB", (1080, 1920), per_cam[cam][k][1]); im.thumbnail((a.long_side, a.long_side))
            buf = io.BytesIO(); im.save(buf, "JPEG", quality=88); imgs.append(base64.b64encode(buf.getvalue()).decode())
        stops.append({"stop": k, "ns": ns, **m, "images": imgs})
    print(f"{n_stops} stops x 6 views -> {n_stops} requests to {a.model}", flush=True)

    def run(s):
        ctx = f"{s['aisle']}, metre {s['metre']:.2f} along the aisle" if s.get("metre") is not None else "greenhouse aisle"
        content = []
        for i, (cam, b64) in enumerate(zip(ORDER, s["images"]), 1):
            content += [{"type": "text", "text": f"Image {i}: {cam.replace('_', ' ')}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]
        content.append({"type": "text", "text": PROMPT.format(ctx=ctx, detail=DETAIL if a.detailed else "",
                                                              extra=', "description": str, "action": str' if a.detailed else "")})
        t0 = time.time()
        for attempt in range(3):
            try:
                r = client.chat.completions.create(model=a.model, messages=[{"role": "user", "content": content}],
                                                   temperature=0.2, max_tokens=4096)
                msg = r.choices[0].message
                out = parse(msg.content or getattr(msg, "reasoning_content", None) or "")
                return {k: s[k] for k in ("stop", "ns", "aisle", "metre")} | out | {"latency_s": round(time.time() - t0, 1),
                        "tokens": getattr(r.usage, "total_tokens", None)}
            except Exception as e:
                err = str(e)[:300]; time.sleep(3)
        return {k: s[k] for k in ("stop", "ns", "aisle", "metre")} | {"left": {"status": "error", "issues": [], "summary": err},
                                                                     "right": {"status": "error", "issues": [], "summary": err}}
    done, rows = 0, []
    with ThreadPoolExecutor(a.workers) as ex, open(a.out / "vlm_stops.jsonl", "w") as fo:
        for r in ex.map(run, stops):
            rows.append(r); fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush(); done += 1
            if done % 10 == 0:
                print(f"{done}/{n_stops} stops", flush=True)
    flagged = [(r, side) for r in rows for side in ("left", "right") if r[side].get("status") in ("suspect", "disease")]
    gt = [g for g in json.load(open(a.rec / "ground_truth_plants.json")) if g.get("synthetic_anomaly")]
    checks = []
    for g in gt:
        aisle = f"{g['tunnel']}/aisle-02"; m = HEAD_Y - g["xy"][1]
        near = [(r, sd) for r, sd in flagged if r["aisle"] == aisle and r["metre"] is not None and abs(r["metre"] - m) <= a.match_m]
        checks.append({"plant": g["prim"], "aisle": aisle, "metre": round(m, 2), "truth": g["synthetic_anomaly"], "flagged": bool(near),
                       "labels_nearby": sorted({i.get("label", "") for r, sd in near for i in r[sd].get("issues", [])})[:6]})
    far = [1 for r, sd in flagged if not any(r["aisle"] == f"{g['tunnel']}/aisle-02" and r["metre"] is not None
                                             and abs(r["metre"] - (HEAD_Y - g["xy"][1])) <= a.match_m for g in gt)]
    tok = sum(r.get("tokens") or 0 for r in rows)
    summary = {"model": a.model, "requests": len(rows), "plants_judged": 2 * len(rows), "total_tokens": tok,
               "status": dict(collections.Counter(r[s].get("status") for r in rows for s in ("left", "right"))),
               "anomaly_plants_flagged": f"{sum(c['flagged'] for c in checks)}/{len(checks)}",
               "flags_not_near_any_anomaly_plant": len(far), "ground_truth_check": checks}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "ground_truth_check"}, ensure_ascii=False))
    for c in checks:
        print(c["aisle"], c["metre"], c["truth"], "FLAGGED" if c["flagged"] else "missed", c["labels_nearby"][:3])


if __name__ == "__main__":
    main()
