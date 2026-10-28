"""Batch evaluation of BOOK (sensor-only aisle driving) in the Isaac R180 env, for Nebius Serverless Jobs / a GPU VM.
N episodes x randomisation (seed, wheel-friction scale for wet floor, obstacle on/off, start offset). Metrics follow the
README success criteria; ground truth is evaluation-only (GroundTruthEvaluator → info["gt"]), the robot never sees it.

  /isaac-sim/python.sh scripts/run_eval_job.py --episodes 50 --out /results/run1
  python3 scripts/run_eval_job.py --backend fake --episodes 20 --out /tmp/eval_fake      # plumbing check, no physics
  python3 scripts/run_eval_job.py --dry-run --episodes 5                                 # print sampled episodes only

Writes <out>/results.jsonl (one row per episode) and <out>/summary.json (success rate overall and per condition)."""
import argparse, json, math, os, platform, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r1_nav.aisle_map import AisleMap                     # noqa: E402
from r1_nav.mission import Leg                            # noqa: E402
from sim.backends import make_sim                         # noqa: E402
from sim.executor import ExecutorConfig, LegExecutor, WheelImuOdometry   # noqa: E402
from sim.isaac_env import SceneFrame, house_translation   # noqa: E402

STOP_TOL_M, MIN_CLEARANCE_M, OBSTACLE_STOP_M, SURVEY_MAX_S = 0.15, 0.05, 0.5, 8 * 60   # README criteria 1, 2, 4


def log(msg: str) -> None:
    print(f"[eval {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sample_episode(i: int, args, start_metre: float, aisle_len: float) -> dict:
    rng = random.Random(args.seed0 * 100003 + i)
    task = args.task
    target = aisle_len if task == "survey" else round(rng.uniform(*args.target_range), 2)
    target = min(target, aisle_len - 0.5) if task == "goto" else target
    obstacle = rng.random() < args.obstacle_prob
    ep = {"episode": i, "seed": args.seed0 + i, "task": task, "target_metre": target,
          "friction_scale": round(rng.uniform(*args.friction_range), 3),
          "start_offset": [round(rng.uniform(-args.start_along_m, args.start_along_m), 3),
                           round(rng.uniform(-args.start_lateral_m, args.start_lateral_m), 3),
                           round(rng.uniform(-args.start_yaw_deg, args.start_yaw_deg), 2)],
          "obstacle_metre": None}
    if obstacle:                                          # box somewhere between start and target (centre of the box)
        lo = start_metre + ep["start_offset"][0] + 2.5
        ep["obstacle_metre"] = round(rng.uniform(lo, max(lo + 0.5, min(target, aisle_len) - 1.0)), 2)
    return ep


def run_episode(sim, ep: dict, aisle, args) -> dict:
    t_wall = time.monotonic()
    obs, info = sim.reset(seed=ep["seed"], friction_scale=ep["friction_scale"], obstacle_metre=ep["obstacle_metre"],
                          start_offset=tuple(ep["start_offset"]))
    gt0, unsupported = info.get("gt"), info.get("unsupported", [])
    # The robot only knows its nominal start (as if from the head tag); offsets are unknown disturbances.
    ex = LegExecutor(WheelImuOdometry(metre=sim.start_metre), aisle.id.split("/")[0],
                     ExecutorConfig(blocked_timeout_s=args.obstacle_hold_s))
    legs = [Leg("drive_in", aisle.id, ep["target_metre"])]
    if ep["task"] == "survey":
        legs.append(Leg("reverse_out", aisle.id, args.survey_exit_metre))
    ex.load(legs, ep["task"])
    t0 = float(obs["timestamp_s"])
    min_strip = min_bed = math.inf
    gt, term_reason, steps, events_seen = gt0, None, 0, []
    det = None                                            # (gt front metre, t) when SAFETY first stopped for the box
    limit_s = args.max_episode_s or (abs(ep["target_metre"] - sim.start_metre) / 0.2 + (300 if ep["task"] == "survey" else 30))
    while ex.busy:
        wheel, events = ex.step(obs)
        for e in events:
            if e["type"] in ("obstacle_stop", "stuck", "aborted", "leg_done"):
                events_seen.append({k: v for k, v in e.items() if k in ("type", "reason", "why", "metre", "error_m")})
            if e["type"] == "obstacle_stop" and det is None and gt:
                det = gt["front_metre"]
        obs, _, term, trunc, info = sim.step(wheel)
        steps += 1
        gt = info.get("gt") or gt
        if gt:
            min_strip, min_bed = min(min_strip, gt["clearance_strip_m"]), min(min_bed, gt["clearance_bed_m"])
        if term or trunc:
            term_reason = info.get("termination"); ex.abort(f"sim: {term_reason}"); break
        if float(obs["timestamp_s"]) - t0 > limit_s:
            term_reason = "time_limit"; ex.abort("episode time limit"); break
    for _ in range(args.settle_steps):                    # let the robot coast to rest before measuring the stop
        obs, _, term, _, info = sim.step([0.0, 0.0])
        gt = info.get("gt") or gt
        if term:
            break
    sim_s = float(obs["timestamp_s"]) - t0
    row = {**ep, "status": ex.status, "reason": ex.reason, "termination": term_reason, "steps": steps,
           "sim_s": round(sim_s, 2), "wall_s": round(time.monotonic() - t_wall, 2), "events": events_seen[-6:],
           "unsupported": unsupported, "gt_available": gt is not None,
           "odometry_metre": round(ex.metre, 3)}
    if gt is None:
        row.update(success=None, why="no ground truth from env")
        return row
    row.update(final_metre=round(gt["metre"], 3), odometry_error_m=round(ex.metre - gt["metre"], 3),
               final_lateral_m=round(gt["lateral_m"], 3), final_yaw_deg=round(gt["yaw_deg"], 2),
               min_clearance_strip_m=round(min_strip, 3), min_clearance_bed_m=round(min_bed, 3),
               start_gt_metre=round(gt0["metre"], 3) if gt0 else None)
    clear = (min_strip if args.clearance_ref == "strip" else min_bed) >= MIN_CLEARANCE_M
    checks = {"no_tip": term_reason not in ("tip", "invalid_physics", "side_collision"), "side_clearance": clear}
    if ep["obstacle_metre"] is not None:
        face = gt.get("obstacle_face_metre") or (ep["obstacle_metre"] - 0.09)
        gap = face - gt["front_metre"]
        row.update(obstacle_gap_m=round(gap, 3),
                   stop_distance_after_detect_m=None if det is None else round(gt["front_metre"] - det, 3))
        checks.update(stopped_for_obstacle=det is not None and ex.reason.startswith("obstacle"),
                      no_contact=gap >= 0.02,
                      stop_within_0_5m=det is not None and gt["front_metre"] - det <= OBSTACLE_STOP_M)
    else:
        target = ep["target_metre"] if ep["task"] == "goto" else args.survey_exit_metre
        err = gt["metre"] - target
        row["stop_error_m"] = round(err, 3)
        checks.update(legs_done=ex.status == "done", stop_accuracy=abs(err) <= STOP_TOL_M)
        if ep["task"] == "survey":
            checks["time_le_8min"] = sim_s <= SURVEY_MAX_S
    row.update(checks=checks, success=all(checks.values()))
    return row


def summarise(rows: list[dict], meta: dict) -> dict:
    def rate(rs):
        judged = [r for r in rs if r.get("success") is not None]
        return {"n": len(rs), "judged": len(judged),
                "success_rate": round(sum(r["success"] for r in judged) / len(judged), 3) if judged else None}
    by = {"obstacle": rate([r for r in rows if r["obstacle_metre"] is not None]),
          "no_obstacle": rate([r for r in rows if r["obstacle_metre"] is None]),
          "wet_floor_friction_lt_0.6": rate([r for r in rows if r["friction_scale"] < 0.6]),
          "dry_floor_friction_ge_0.6": rate([r for r in rows if r["friction_scale"] >= 0.6])}
    errs = [abs(r["stop_error_m"]) for r in rows if r.get("stop_error_m") is not None]
    clr = [r["min_clearance_strip_m"] for r in rows if r.get("min_clearance_strip_m") is not None]
    odo = [abs(r["odometry_error_m"]) for r in rows if r.get("odometry_error_m") is not None]
    fails = {}
    for r in rows:
        for k, v in (r.get("checks") or {}).items():
            fails[k] = fails.get(k, 0) + (not v)
    return {**meta, "overall": rate(rows), "by_condition": by, "failed_checks": fails,
            "stop_error_abs_m": {"mean": round(sum(errs) / len(errs), 3), "max": round(max(errs), 3)} if errs else None,
            "min_clearance_strip_m": round(min(clr), 3) if clr else None,
            "odometry_error_abs_m": {"mean": round(sum(odo) / len(odo), 3), "max": round(max(odo), 3)} if odo else None,
            "sim_s_total": round(sum(r["sim_s"] for r in rows), 1), "wall_s_total": round(sum(r["wall_s"] for r in rows), 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("isaac", "fake"), default="isaac")
    ap.add_argument("--episodes", type=int, default=int(os.environ.get("EVAL_EPISODES", 20)))
    ap.add_argument("--seed0", type=int, default=int(os.environ.get("EVAL_SEED0", 0)), help="shard jobs by seed0")
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("EVAL_OUT", "runs/eval")))
    ap.add_argument("--task", choices=("goto", "survey"), default="goto")
    ap.add_argument("--target-range", type=float, nargs=2, default=(8.0, 30.0), metavar=("MIN", "MAX"))
    ap.add_argument("--friction-range", type=float, nargs=2, default=(0.3, 1.0), metavar=("MIN", "MAX"))
    ap.add_argument("--obstacle-prob", type=float, default=0.3)
    ap.add_argument("--start-along-m", type=float, default=0.0,
                    help="unknown along-aisle start error; >0 tests localisation (odometry has no tag fix), not BOOK")
    ap.add_argument("--start-lateral-m", type=float, default=0.05)
    ap.add_argument("--start-yaw-deg", type=float, default=3.0)
    ap.add_argument("--obstacle-hold-s", type=float, default=3.0, help="executor gives up (aborted) after this")
    ap.add_argument("--survey-exit-metre", type=float, default=0.0, help="reverse_out target for --task survey")
    ap.add_argument("--clearance-ref", choices=("strip", "bed"), default="strip",
                    help="strip = 0.38 m concrete half-width (Astra's conservative check); bed = bed-envelope half-gap")
    ap.add_argument("--settle-steps", type=int, default=10)
    ap.add_argument("--max-episode-s", type=float, default=None)
    ap.add_argument("--map", type=Path, default=ROOT / "data/tunnel09_aisle_map_v18.json")
    ap.add_argument("--aisle", default="tunnel-09/aisle-02")
    ap.add_argument("--r180-dir", default=os.environ.get("R180_DIR"))
    ap.add_argument("--stage", default=os.environ.get("EVAL_STAGE", "default"))
    ap.add_argument("--dry-run", action="store_true", help="print sampled episodes and exit (no sim)")
    args = ap.parse_args()

    amap = AisleMap.load(args.map)
    aisle, house_xy = amap.get(args.aisle), house_translation(args.map)
    start = SceneFrame(aisle, house_xy).start_metre
    episodes = [sample_episode(i, args, start, aisle.length_m) for i in range(args.episodes)]
    if args.dry_run:
        print(json.dumps(episodes, indent=1)); return 0
    args.out.mkdir(parents=True, exist_ok=True)
    sim, app = make_sim(args.backend, aisle, house_xy, args.r180_dir, args.stage, side_cameras=False, log=log)
    unsupported = sorted(k for k, v in sim.knobs.items() if not v)
    if unsupported:
        log(f"WARNING unsupported randomisation knobs (held at nominal): {unsupported}")
    meta = {"backend": sim.backend, "stage": args.stage, "task": args.task, "episodes": args.episodes, "seed0": args.seed0,
            "knobs": sim.knobs, "start_metre": round(start, 3), "clearance_ref": args.clearance_ref,
            "criteria": {"stop_tol_m": STOP_TOL_M, "min_clearance_m": MIN_CLEARANCE_M,
                         "obstacle_stop_after_detect_m": OBSTACLE_STOP_M, "survey_max_s": SURVEY_MAX_S},
            "host": platform.node(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "note": "fake backend is kinematic, NOT physics: plumbing check only" if sim.backend == "fake" else ""}
    rows = []
    try:
        with open(args.out / "results.jsonl", "w") as f:
            for ep in episodes:
                try:
                    row = run_episode(sim, ep, aisle, args)
                except Exception as e:                          # one bad episode must not kill a batch job
                    row = {**ep, "status": "error", "reason": repr(e), "success": False, "sim_s": 0.0, "wall_s": 0.0}
                rows.append(row)
                f.write(json.dumps(row, default=float) + "\n"); f.flush()
                log(f"ep {ep['episode']:3d} success={row.get('success')} status={row['status']} "
                    f"err={row.get('stop_error_m')} gap={row.get('obstacle_gap_m')} clr={row.get('min_clearance_strip_m')} "
                    f"mu={ep['friction_scale']} obst={ep['obstacle_metre']} wall={row['wall_s']}s")
    finally:
        summary = summarise(rows, meta)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        log(f"summary {json.dumps(summary['overall'])} → {args.out}")
        sim.close()
        if app is not None:
            app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
