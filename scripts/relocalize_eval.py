#!/usr/bin/env python3
"""Put the robot down anywhere on the taught map: can it work out where it is?  Offline evaluation on a recording.

  python3 scripts/relocalize_eval.py --rec <run>/rec [--teach <run>/rec/teach] [--n 40] [--out <dir>]

For N random steps of the recording: (1) one-scan relocalisation from scratch (sim.relocalize.Relocalizer) - correct
within 1 m, flagged ambiguous, or wrong-and-confident; (2) multi-hypothesis tracking while the robot keeps driving the
recorded route (HypothesisTracker) - metres driven until one hypothesis dominates, and whether it is the right one.
Ground truth is used only to score."""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.relocalize import HypothesisTracker, Relocalizer    # noqa: E402
from scripts.teach_map import beam_angles, scan_xy           # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", type=Path, required=True); ap.add_argument("--teach", type=Path, default=None)
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-drive-s", type=float, default=120.0); ap.add_argument("--decide-margin", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(); teach = a.teach or a.rec / "teach"
    m = np.load(teach / "prior_map_xy.npy"); path = np.load(teach / "path_xyh.npy")
    S = [json.loads(l) for l in open(a.rec / "steps.jsonl")]
    S = [s for s in S if s.get("lidar") and s.get("gt_scene_pose") and s.get("xy") and s.get("leg")]
    ang = beam_angles(len(S[0]["lidar"])); reloc = Relocalizer(m, path)
    rng = np.random.default_rng(a.seed)
    # (1) single scan
    one = {"n": 0, "correct_1m": 0, "ambiguous": 0, "wrong_confident": 0, "errs": []}
    t0 = time.time()
    for k in rng.choice(len(S), a.n, replace=False):
        s = S[k]; r = reloc.locate(scan_xy(s["lidar"], ang)); gt = s["gt_scene_pose"]; one["n"] += 1
        e = math.hypot(r.pose[0] - gt[0], r.pose[1] - gt[1]) if r.pose else 99.0
        if r.ambiguous: one["ambiguous"] += 1
        elif e < 1.0: one["correct_1m"] += 1; one["errs"].append(e)
        else: one["wrong_confident"] += 1
    one["s_per_scan"] = round((time.time() - t0) / a.n, 3)
    one["median_err_m_when_correct"] = round(float(np.median(one["errs"])), 3) if one["errs"] else None; del one["errs"]
    # (2) keep driving the recorded route, let motion decide
    steps_budget = int(a.max_drive_s * 10)
    track = {"n": 0, "decided_correct_1m": 0, "decided_wrong": 0, "undecided": 0, "undecided_but_best_correct": 0, "metres": []}
    for k in rng.choice(len(S) - steps_budget, a.n, replace=False):
        tr = HypothesisTracker(reloc, decide_margin=a.decide_margin); s0 = S[k]
        tr.start(scan_xy(s0["lidar"], ang), (*s0["xy"], s0["heading_deg"])); track["n"] += 1; done = False
        for j in range(k + 1, k + steps_budget, 2):
            s = S[j]; t = tr.update(scan_xy(s["lidar"], ang), (*s["xy"], s["heading_deg"]))
            if t.decided and t.metres > 1.0:
                gt = s["gt_scene_pose"]; e = math.hypot(t.pose[0] - gt[0], t.pose[1] - gt[1]); done = True
                if e < 1.0: track["decided_correct_1m"] += 1; track["metres"].append(t.metres)
                else: track["decided_wrong"] += 1
                break
        if not done:
            gt = s["gt_scene_pose"]; e = math.hypot(t.pose[0] - gt[0], t.pose[1] - gt[1]); track["undecided"] += 1
            if e < 1.0: track["undecided_but_best_correct"] += 1
    track["metres_to_decide_median"] = round(float(np.median(track["metres"])), 1) if track["metres"] else None
    track["metres_to_decide_max"] = round(float(max(track["metres"])), 1) if track["metres"] else None; del track["metres"]
    rep = {"rec": str(a.rec), "map_points": int(len(m)), "hypotheses": int(len(reloc.hyp)), "single_scan": one,
           "drive_and_track": track, "note": "two identical greenhouses: LiDAR geometry alone cannot tell tunnel-09 from tunnel-10 "
           "inside the aisles; outdoors (dock, road, yard) it can. The ambiguous cases are flagged, not guessed."}
    print(json.dumps(rep, indent=1))
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True); (a.out / "relocalize_eval.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
