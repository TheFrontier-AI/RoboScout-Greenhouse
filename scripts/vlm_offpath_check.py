#!/usr/bin/env python3
"""Camera tie-break for OFF-PATH relocalisation (real simulated scans + front images from scripts/offpath_scans.py and
scripts/offpath_images.py): LiDAR proposes the candidate places; when they are ambiguous, Nemotron VL compares the
robot's front image with a reference view from the led drive near each candidate and picks one; the chosen pose then
gets its way back to the dock (SiteMap.route_from_xy).

  python3 scripts/vlm_offpath_check.py --scans scans_all.jsonl --images <dir> --rec <run>/rec --out <dir>"""
from __future__ import annotations
import argparse, base64, json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.relocalize import Relocalizer                       # noqa: E402
from scripts.teach_map import beam_angles, scan_xy           # noqa: E402
from scripts.offpath_eval import build_from_scans            # noqa: E402
from scripts.vlm_place_check import PROMPT, b64, ref_frame   # noqa: E402
from r1_nav.site_map import NoRoute, SiteMap                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", type=Path, required=True); ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--rec", type=Path, required=True); ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model", default=os.environ.get("EYES2_MODEL", "nvidia/Nemotron-3.5-Super-VL"))
    ap.add_argument("--base-url", default=os.environ.get("EYES2_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"))
    a = ap.parse_args(); teach = a.rec / "teach"
    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key=os.environ.get("EYES2_API_KEY") or os.environ["NEBIUS_API_KEY"], timeout=180)
    rows = [json.loads(l) for l in open(a.scans)]
    off = [(i, r) for i, r in enumerate(rows) if r["tag"] == "offpath"]; mp = [r for r in rows if r["tag"] == "map"]
    taught = np.load(teach / "prior_map_xy.npy"); path = np.load(teach / "path_xyh.npy")
    look = build_from_scans(mp); m = np.unique(np.floor(np.vstack([taught, look]) / 0.05).astype(int), axis=0) * 0.05 + 0.025
    reloc = Relocalizer(m, path, grid_m=0.5, grid_headings=24, refine_top=24)
    site = SiteMap.load(teach / "site_map.json")
    S = [json.loads(l) for l in open(a.rec / "steps.jsonl")]; S = [s for s in S if s.get("xy")]
    ang = beam_angles(len(rows[0]["lidar"]))
    out = []; n_amb = n_dec = n_ok = n_lidar_ok = n_back = 0
    for i, r in off:
        img = a.images / f"{i:05d}.jpg"
        rl = reloc.locate(scan_xy(r["lidar"], ang)); truth = (r["x"], r["y"])
        err = lambda p: math.hypot(p[0] - truth[0], p[1] - truth[1])
        row = {"i": i, "truth": [r["x"], r["y"], r["heading_deg"]], "lidar": {"pose": rl.pose, "score": rl.score, "ambiguous": rl.ambiguous,
               "err_m": round(err(rl.pose), 2) if rl.pose else None}}
        final = None
        if rl.pose and not rl.ambiguous:
            final = rl.pose; n_lidar_ok += int(err(rl.pose) < 1.0)
        elif rl.pose and rl.ambiguous and img.exists():
            n_amb += 1
            cands = [rl.pose, rl.alternatives[0]["pose"]]
            refs = []
            for c in cands:
                j = int(np.argmin(np.hypot(path[:, 0] - c[0], path[:, 1] - c[1]))); st = S[min(j, len(S) - 1)]["step"]
                refs.append(ref_frame(a.rec / "front", st))
            if all(refs):
                content = [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": b64(img)}},
                           {"type": "image_url", "image_url": {"url": b64(refs[0])}}, {"type": "image_url", "image_url": {"url": b64(refs[1])}}]
                t0 = time.time()
                try:
                    resp = client.chat.completions.create(model=a.model, temperature=0.1, max_tokens=4000, messages=[{"role": "user", "content": content}])
                    msg = resp.choices[0].message
                    txt = (msg.content or getattr(msg, "reasoning_content", None) or "").split("</think>")[-1].replace("```json", "").replace("```", "")
                    j = json.loads(txt[txt.index("{"): txt.rindex("}") + 1]) if "{" in txt else {"match": "error", "evidence": txt[:120]}
                except Exception as e:
                    j = {"match": "error", "evidence": str(e)[:120]}
                row["vlm"] = {"match": j.get("match"), "confidence": j.get("confidence"), "evidence": (j.get("evidence") or "")[:200], "latency_s": round(time.time() - t0, 1),
                              "A": cands[0], "B": cands[1]}
                if j.get("match") in ("A", "B"):
                    n_dec += 1; final = cands[0] if j["match"] == "A" else cands[1]
                    n_ok += int(err(final) < 1.0)
        if final is not None:
            row["final_pose"] = final; row["final_err_m"] = round(err(final), 2)
            try:
                rt = site.route_from_xy(final[:2], "dock"); row["route_back"] = {"nodes": rt.nodes, "length_m": rt.length_m}
                n_back += int(err(final) < 1.0)
            except NoRoute as ex:
                row["route_back"] = {"refused": str(ex)}
        out.append(row); print(json.dumps({k: row[k] for k in row if k != "lidar"} | {"lidar_err": row["lidar"]["err_m"], "amb": rl.ambiguous}, ensure_ascii=False)[:260], flush=True)
    summ = {"off_path_cases": len(off), "lidar_unambiguous_correct": n_lidar_ok, "ambiguous_sent_to_camera": n_amb, "camera_decided": n_dec,
            "camera_correct": n_ok, "correct_final_pose_with_route_home": n_back, "model": a.model}
    print(json.dumps(summ, indent=1))
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "vlm_offpath.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
        (a.out / "summary.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
