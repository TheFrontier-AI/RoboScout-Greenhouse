#!/usr/bin/env python3
"""Visual tie-break for relocalisation: when the LiDAR says "I could be at A or at B" (two identical greenhouses),
ask the vision-language model which reference view from the taught drive matches what the front camera sees now.

  python3 scripts/vlm_place_check.py --rec <run>/rec --teach <run>/rec/teach --n 16 [--out <dir>]

For N random steps that have a front frame: run sim.relocalize on the LiDAR scan; if it is ambiguous, take the two
far-apart candidates A and B, fetch the taught drive's front frame from ~1.5 m before each candidate (a different
viewpoint of the same place, never the current frame itself), and ask Nemotron VL which of them is where the robot is.
Scores against the recorded truth. Needs NEBIUS_API_KEY (Token Factory)."""
from __future__ import annotations
import argparse, base64, json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.relocalize import Relocalizer                        # noqa: E402
from scripts.teach_map import beam_angles, scan_xy            # noqa: E402

PROMPT = """You are helping a greenhouse robot find out where it is. Image 1 is what its front camera sees right now.
Images 2 (place A) and 3 (place B) are reference photos taken earlier on a led drive at two candidate places on the
same farm; the two places look alike because the greenhouses are identical, so look for what differs: what is visible
through the door or at the end of the row, the growth stage and arrangement of the plants nearest the camera, any
damaged or yellowing plants, objects, structures, light. Decide which reference shows the SAME place as image 1.
Keep your reasoning short. Reply with ONLY this JSON: {"match": "A" | "B" | "unsure", "confidence": 0.0-1.0, "evidence": "one or two sentences"}"""


def b64(p: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()


def ref_frame(front: Path, step: int, back_steps: int = 60) -> Path | None:
    s = max(1, step - back_steps); s -= (s - 1) % 5                      # front frames exist every 5th step (1, 6, 11, ...)
    p = front / f"{s:06d}.jpg"
    return p if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", type=Path, required=True); ap.add_argument("--teach", type=Path, default=None)
    ap.add_argument("--n", type=int, default=16); ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--model", default=os.environ.get("EYES2_MODEL", "nvidia/Nemotron-3.5-Super-VL"))
    ap.add_argument("--base-url", default=os.environ.get("EYES2_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(); teach = a.teach or a.rec / "teach"
    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key=os.environ.get("EYES2_API_KEY") or os.environ["NEBIUS_API_KEY"], timeout=120)
    m = np.load(teach / "prior_map_xy.npy"); path = np.load(teach / "path_xyh.npy")
    S = [json.loads(l) for l in open(a.rec / "steps.jsonl")]
    S = [s for s in S if s.get("lidar") and s.get("gt_scene_pose") and s.get("xy")]
    ang = beam_angles(len(S[0]["lidar"])); reloc = Relocalizer(m, path)
    rng = np.random.default_rng(a.seed); order = rng.permutation(len(S))
    rows = []
    for k in order:
        if len(rows) >= a.n:
            break
        s = S[k]
        if (s["step"] - 1) % 5 or not (a.rec / "front" / f"{s['step']:06d}.jpg").exists():
            continue
        r = reloc.locate(scan_xy(s["lidar"], ang))
        if not r.pose or not r.ambiguous:
            continue
        cands = [r.pose, r.alternatives[0]["pose"]]
        # the taught-path step nearest each candidate -> its reference frame ~1.5 m earlier on the path
        refs, steps_at = [], []
        for c in cands:
            i = int(np.argmin(np.hypot(path[:, 0] - c[0], path[:, 1] - c[1]))); st = S[min(i, len(S) - 1)]["step"]
            p = ref_frame(a.rec / "front", st); refs.append(p); steps_at.append(st)
        if any(p is None for p in refs) or abs(steps_at[0] - steps_at[1]) < 300:
            continue
        gt = s["gt_scene_pose"]
        truth = "A" if math.hypot(cands[0][0] - gt[0], cands[0][1] - gt[1]) < math.hypot(cands[1][0] - gt[0], cands[1][1] - gt[1]) else "B"
        t0 = time.time()
        try:
            content = [{"type": "text", "text": PROMPT},
                       {"type": "image_url", "image_url": {"url": b64(a.rec / "front" / f"{s['step']:06d}.jpg")}},
                       {"type": "image_url", "image_url": {"url": b64(refs[0])}},
                       {"type": "image_url", "image_url": {"url": b64(refs[1])}}]
            resp = client.chat.completions.create(model=a.model, temperature=0.1, max_tokens=2000,
                                                  messages=[{"role": "user", "content": content}])
            msg = resp.choices[0].message                        # Token Factory may put the text in reasoning_content
            txt = (msg.content or getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or "")
            txt = txt.split("</think>")[-1].replace("```json", "").replace("```", "")
            if "{" not in txt:
                raise ValueError("no JSON in reply: " + txt[:160].replace("\n", " "))
            j = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
        except Exception as e:
            j = {"match": "error", "confidence": 0, "evidence": str(e)[:160]}
        row = {"step": s["step"], "leg": s.get("leg"), "truth": truth, "vlm": j.get("match"), "confidence": j.get("confidence"),
               "correct": j.get("match") == truth, "lidar_scores": [r.score, r.alternatives[0]["score"]],
               "cand_dist_m": r.alternatives[0]["dist_m"], "latency_s": round(time.time() - t0, 1), "evidence": j.get("evidence")}
        rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    n = len(rows); correct = sum(1 for r in rows if r["correct"]); unsure = sum(1 for r in rows if r["vlm"] == "unsure")
    wrong = sum(1 for r in rows if r["vlm"] in ("A", "B") and not r["correct"])
    summ = {"model": a.model, "ambiguous_cases": n, "vlm_correct": correct, "vlm_wrong": wrong, "vlm_unsure": unsure,
            "errors": n - correct - wrong - unsure, "accuracy_when_decided": round(correct / max(correct + wrong, 1), 3),
            "median_latency_s": float(np.median([r["latency_s"] for r in rows])) if rows else None}
    print(json.dumps(summ, indent=1))
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "vlm_place_check.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        (a.out / "summary.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
