"""Cosmos Reason on the robot's front camera at the moments it should "open its eyes" (layer 3 of the navigation
design): every --every-s seconds of driving plus each change of leg (leaving the dock, entering/leaving a greenhouse,
crossing the yard). Measures the round-trip latency of every call.

  python3 scripts/cosmos_front.py --front <render_out>/front --rec <run>/rec --base-url http://localhost:8000/v1 \
      --model nvidia/Cosmos-Reason2-8B --out cosmos_front/"""
from __future__ import annotations
import argparse, base64, json, re, time
from pathlib import Path

PROMPT = """You are the forward-looking eyes of R1, a small 4-wheel robot (0.53 m wide) that inspects tomato greenhouses.
This is its front camera, {where}. The robot is about to {doing}.
Describe in 2-4 sentences what is ahead (the path, doorways, plants, structures, people, objects, ground condition) and
say whether it is safe to continue. Reply with ONLY this JSON: {{"scene": str, "path_clear": true|false,
"hazards": [str], "recommendation": "continue|slow_down|stop|ask_human", "confidence": 0.0-1.0}}"""
DOING = {"path": "drive across open ground between greenhouses", "enter": "enter a greenhouse through its doorway",
         "drive_in": "drive down a planted aisle, stopping at each plant", "drive_out": "drive down a planted aisle, stopping at each plant",
         "dock": "return to its charging dock"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", type=Path, required=True); ap.add_argument("--rec", type=Path, required=True)
    ap.add_argument("--base-url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--every-s", type=float, default=10.0); ap.add_argument("--api-key", default="EMPTY")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key=a.api_key, timeout=120)
    steps = [json.loads(l) for l in open(a.rec / "steps.jsonl")]
    frames = sorted(a.front.glob("*.jpg"), key=lambda p: int(p.stem)); ts = [int(p.stem) / 1e9 for p in frames]
    leg_at = lambda t: next((s for s in reversed(steps) if s["t"] <= t), steps[0])
    picks, last, prev_leg = [], -1e9, None
    for p, t in zip(frames, ts):
        s = leg_at(t); leg = s.get("leg")
        if leg is None:
            continue
        if leg != prev_leg or t - last >= a.every_s:
            picks.append((p, t, leg, s.get("aisle"), s.get("metre"), "leg change" if leg != prev_leg else "periodic")); last = t
        prev_leg = leg
    print(f"{len(picks)} checks out of {len(frames)} front frames", flush=True)
    rows = []
    for p, t, leg, aisle, metre, why in picks:
        where = f"in {aisle} at metre {metre:.1f}" if aisle and leg in ("drive_in", "drive_out") else "outdoors between greenhouses" if leg in ("path", "dock") else "at a greenhouse doorway"
        b64 = base64.b64encode(p.read_bytes()).decode()
        t0 = time.time()
        r = client.chat.completions.create(model=a.model, temperature=0.2, max_tokens=1024, messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": PROMPT.format(where=where, doing=DOING.get(leg, "keep driving"))}]}])
        lat = time.time() - t0
        txt = re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S)
        try:
            out = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
        except Exception:
            out = {"scene": txt[:400], "parse_error": True}
        row = {"t": round(t, 2), "frame": p.name, "leg": leg, "aisle": aisle, "metre": metre, "trigger": why, "latency_s": round(lat, 2), **out}
        rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    lats = sorted(r["latency_s"] for r in rows)
    summ = {"model": a.model, "checks": len(rows), "front_frames": len(frames), "latency_s": {"median": lats[len(lats) // 2],
            "p90": lats[int(len(lats) * 0.9)], "max": lats[-1]},
            "recommendations": {k: sum(1 for r in rows if r.get("recommendation") == k) for k in ("continue", "slow_down", "stop", "ask_human")}}
    (a.out / "cosmos_front.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ))


if __name__ == "__main__":
    main()
