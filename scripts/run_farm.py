"""Full-farm run: R1 drives itself through tunnel-09 and tunnel-10 (front to back, across the yard, back to front)
and home to the dock, stopping at every plant to shoot all six side cameras. Goals come from HERMES through the robot
hub (agent/robot_api.py): `survey_farm`, `revisit`, `return_to_dock`, `stop`; or from --offline (scripted farm survey).

Robot side (no simulator truth): BOOK aisle driving + SAFETY, LiDAR+wheel+gyro odometry (sim.lidar_odometry),
yard dead reckoning, landmark checks. Scene (session layer only): Tomato-Assets-V1 crops, open tunnel ends, collision,
yard ground (sim.scene_prep). Rendering on demand (sim.fast_env): side cameras only at plant stops, a game-style
third-person chase camera every --chase-every steps, front camera every --front-every steps. Records everything
for videos, detection datasets and 3-D SLAM. No training.

  /isaac-sim/python.sh scripts/run_farm.py --record /runs/farm --api-url http://127.0.0.1:8090
  /isaac-sim/python.sh scripts/run_farm.py --record /runs/farm --offline"""
from __future__ import annotations
import argparse, collections, json, math, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r1_nav.cameras import FRONT_CAMERA                                  # noqa: E402
from r1_nav.mission import Leg, farm_loop, revisit_from_yard             # noqa: E402
from sim.cloud_link import CloudLink, encode_image                       # noqa: E402
from sim.executor import ExecutorConfig, LegExecutor                     # noqa: E402
from sim.lidar_odometry import LidarWheelOdometry, scan_points          # noqa: E402
from sim.scan_localizer import ScanToMapLocalizer                        # noqa: E402

R180 = Path(os.environ.get("R180_DIR", "/home/passin/Documents/Blender/TheFarm/robot_design/r1_v3/diy_r50/physical_ai_r180"))
STAGE = R180 / "tomato_realism_2026-10-28/C2_R180_Interior_PT_Sensor_Fixture.usda"
# Scene frame (the twin moves /Farm by -(195.065, 153.0)): aisle-02 centre lines, metre 0 = first plant (bed start),
# front doors at y 3.66, back openings at y -35.19 (open for this run), dock outside tunnel-09 (site map).
HOUSES = [("tunnel-09/aisle-02", 0.0), ("tunnel-10/aisle-02", 9.015)]
HEAD_Y, BACK_Y, ROWS_M = 3.26, -35.19, (0.0, 38.04)
DOCK = (2.0, 4.9)
ROAD_Y = 8.99


def build_prior_map(region=(-5.0, 14.0, -40.0, 10.0), step=1.0, n_rays=180, z=0.546, rmax=12.0, voxel=0.03):
    """2-D map at LiDAR height from the twin's collision geometry (PhysX raycasts from a grid of positions).
    Stands in for the mapping run / site survey a real deployment would provide. Robot hits are excluded."""
    import carb, numpy as np, omni.physx
    sq = omni.physx.get_physx_scene_query_interface()
    pts = []
    dirs = [(math.cos(2 * math.pi * k / n_rays), math.sin(2 * math.pi * k / n_rays)) for k in range(n_rays)]
    x0, x1, y0, y1 = region
    for x in np.arange(x0, x1 + 1e-6, step):
        for y in np.arange(y0, y1 + 1e-6, step):
            for dx, dy in dirs:
                h = sq.raycast_closest(carb.Float3(float(x), float(y), z), carb.Float3(dx, dy, 0.0), rmax)
                if h["hit"] and not str(h["collision"]).startswith("/Robot") and h["distance"] > 0.02:
                    pts.append((h["position"][0], h["position"][1]))
    a = np.asarray(pts)
    return np.unique(np.round(a / voxel).astype(np.int64), axis=0) * voxel


def log(m: str) -> None:
    print(f"[farm {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", type=Path, required=True)
    ap.add_argument("--offline", action="store_true", help="scripted: survey_farm then stop (no hub, no HERMES)")
    ap.add_argument("--api-url", default=os.environ.get("ROBOT_API_URL", "http://127.0.0.1:8090"))
    ap.add_argument("--outbox", type=Path, default=Path("/runs/outbox"))
    ap.add_argument("--plant-res", default=os.environ.get("R1_PLANT_RES", "1080x1920"))
    ap.add_argument("--crop-res", default="2k", choices=("2k", "8k"))
    ap.add_argument("--chase-every", type=int, default=2); ap.add_argument("--front-every", type=int, default=5)
    ap.add_argument("--front-res", default="1280x720", help="front camera (navigation + visual SLAM / MapAnything input)")
    ap.add_argument("--max-sim-s", type=float, default=7200.0); ap.add_argument("--idle-exit-s", type=float, default=600.0)
    ap.add_argument("--only", choices=("09", "10", "both"), default="both", help="survey just one house (testing)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N control steps (testing)")
    a = ap.parse_args()
    rec = a.record; [(rec / d).mkdir(parents=True, exist_ok=True) for d in ("frames", "chase", "front", "side")]
    files = {n: open(rec / f"{n}.jsonl", "a") for n in ("steps", "events", "frames")}
    write = lambda n, row: files[n].write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
    link = CloudLink(None if a.offline else a.api_url, outbox=a.outbox, log=log)
    pending = collections.deque([{"cmd": "survey_farm"}] if a.offline else [])

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    code, step, obs, ex = 0, 0, None, None
    try:
        for p in (R180, R180.parent / "physical_ai_runtime", R180 / "tomato_realism_2026-10-28"):
            sys.path.insert(0, str(p))
        from gated_render_sensor_env import GatedRenderSensorEnv
        from sim.fast_env import YardEvaluator, make_fast_env
        from sim.scene_prep import prepare_farm_run
        from scripts.farm_smoke import ChaseCam
        from PIL import Image
        import numpy as np
        Env = make_fast_env(GatedRenderSensorEnv)
        w, h = (int(v) for v in a.plant_res.split("x"))
        env = Env(app, stage_path=STAGE, aisle_half_width=0.38, body_half_length=0.375, plant_cameras=True, plant_resolution=(w, h),
                  front_resolution=tuple(int(v) for v in a.front_res.split("x")))
        houses = HOUSES if a.only == "both" else [x for x in HOUSES if x[0].startswith(f"tunnel-{a.only}")]
        report = prepare_farm_run(env.stage, tunnels=tuple(x[0][7:9] for x in HOUSES), res=a.crop_res, log=log)
        (rec / "scene_prep.json").write_text(json.dumps(report, indent=1, default=str))
        (rec / "ground_truth_plants.json").write_text(json.dumps(report.get("crops", []), indent=1))
        chase = ChaseCam(env.stage); env.extra_products.append(chase.rp.hydra_texture)
        obs, info = env.reset(options={"obstacle_enabled": False, "task_mode": "navigation", "max_steps": None,
                                       "start_pose": {"x": DOCK[0], "y": DOCK[1], "yaw_deg": 0.0}})
        env.set_evaluator(YardEvaluator())
        t0 = time.time(); prior = build_prior_map()
        np.save(rec / "prior_map_xy.npy", prior); loc = ScanToMapLocalizer(prior)
        log(f"prior 2-D map: {len(prior)} points in {time.time() - t0:.1f}s")
        pose = LidarWheelOdometry(metre=0.0, heading_deg=-90.0)
        cfg = ExecutorConfig(aisle_heads={aid: (x, HEAD_Y) for aid, x in HOUSES}, rows_m=ROWS_M, blocked_timeout_s=120.0)
        ex = LegExecutor(pose, "farm", cfg); ex.set_xy(*DOCK)
        log(f"ready: dock {DOCK}, houses {houses}, plants swapped {len(report.get('crops', []))}")
        last_front = None

        def plan(cmd: dict) -> list[Leg] | str:
            name = cmd.get("cmd")
            if name in ("survey_farm", "survey_house"):
                return farm_loop(houses, HEAD_Y, BACK_Y, DOCK, ROAD_Y, ROWS_M[1])
            if name == "revisit":
                aid = cmd.get("aisle") or f"{cmd.get('house')}/aisle-02"
                x = dict(HOUSES).get(aid)
                if x is None or cmd.get("metre") is None:
                    return f"unknown aisle {aid}"
                return revisit_from_yard(aid, x, HEAD_Y, float(cmd["metre"]), DOCK)
            if name == "return_to_dock":
                return []
            return f"unsupported command {name}"

        def publish(extra=None):
            st = {**ex.status_dict(), "ts": time.time(), "sim_time_s": round(float(obs["timestamp_s"]), 2), "backend": "isaac",
                  "frames_queued": link.backlog, "pending_commands": len(pending), "xy": ex.xy, **(extra or {})}
            st["legs_remaining"] += len(pending)
            if link.enabled:
                link.post_status(st)
            write("events", {"type": "status", **st})

        def start(cmd):
            legs = plan(cmd)
            ok = isinstance(legs, list)
            write("events", {"type": "command", "t": float(obs["timestamp_s"]), "cmd": cmd, "accepted": ok,
                             "legs": [l.kind for l in legs] if ok else None, "reason": None if ok else legs})
            log(f"cmd {cmd.get('cmd')} → {'accepted ' + str([l.kind for l in legs]) if ok else 'rejected: ' + legs}")
            if ok and legs:
                ex.load(legs, cmd.get("cmd"))
            publish({"reply": "accepted" if ok else "rejected"})

        sit_n = [0]

        def front_range(o, half_deg=15.0):
            r = o.get("lidar_m") if o else None
            if r is None or not len(r):
                return None
            n = len(r)                                         # R180 raw azimuth: beam k at -pi + 2pi k/n + pi/2
            vals = [float(r[k]) for k in range(n)
                    if abs(math.degrees(((-math.pi + 2 * math.pi * k / n + math.pi / 2) + math.pi) % (2 * math.pi) - math.pi)) <= half_deg
                    and math.isfinite(float(r[k]))]
            return round(min(vals), 3) if vals else None

        def report_situation(ev):
            """SAFETY stopped the robot: send the latest front frame for Cosmos Reason to explain to HERMES."""
            img = obs.get("front_rgb", last_front) if obs else last_front
            if img is None:
                return
            sit_n[0] += 1
            meta = {"frame_id": 900000 + sit_n[0], "camera": FRONT_CAMERA, "purpose": "situation",
                    "trigger": ev["type"], "why": ev.get("why") or ev.get("reason") or "", "aisle": ex.aisle,
                    "metre": round(ex.metre, 2) if ex.aisle else None, "xy": ex.xy, "lidar_front_m": front_range(obs),
                    "t": float(obs["timestamp_s"]), "pose_confidence": round(ex.confidence, 3)}
            data, mime = encode_image(img)
            if link.enabled:
                link.send_frame(meta, data, mime)
            rel = f"frames/situation_{sit_n[0]:03d}.jpg"; (rec / rel).write_bytes(data)
            write("frames", {**meta, "image_path": rel})
            log(f"situation report {sit_n[0]} ({ev['type']}) sent to HERMES via Cosmos Reason")

        def dispatch(events):
            for ev in events:
                if ev["type"] in ("obstacle_stop", "stuck") or (ev["type"] == "aborted" and "obstacle" in str(ev.get("reason", ""))):
                    report_situation(ev)
                write("events", {"t": float(obs["timestamp_s"]), **{k: v for k, v in ev.items() if k != "meta"},
                                 **({"frame_id": ev["meta"]["frame_id"]} if "meta" in ev else {})})
                if ev["type"] != "capture":
                    if ev["type"] not in ("leg_start", "waypoint"):
                        log(f"{ev['type']}: {ev.get('reason') or ev.get('note') or (ev.get('leg') or {}).get('kind', '')} "
                            f"metre={ev.get('metre')} {ev.get('error_m', '')}")
                    continue
                meta, cam = dict(ev["meta"]), ev["camera"]
                if cam == FRONT_CAMERA:
                    img = obs.get("front_rgb", last_front); meta["image_source"] = "front_rgb" if "front_rgb" in obs else "front_rgb_last"
                else:
                    img = obs.get("plant_rgb", {}).get(cam.replace("mast_", "plant_")); meta["image_source"] = cam
                if img is None:
                    write("events", {"type": "capture_missing", "camera": cam, "frame_id": meta["frame_id"]}); continue
                if cam != FRONT_CAMERA:                                # full-size PNG kept for datasets / SLAM
                    Image.fromarray(img).save(rec / "side" / f"{meta['frame_id']:06d}_{cam}.png")
                data, mime = encode_image(img)
                if link.enabled:
                    link.send_frame(meta, data, mime)
                rel = f"frames/{meta['frame_id']:06d}_{cam}.jpg"; (rec / rel).write_bytes(data)
                write("frames", {**meta, "image_path": rel, "xy": ex.xy})

        t_poll = t_status = 0.0; idle_since = time.monotonic(); t_wall0 = time.time()
        while True:
            now = time.monotonic()
            if link.enabled and now - t_poll >= 1.0:
                t_poll = now; cmd = link.next_command()
                if cmd:
                    if cmd.get("cmd") == "stop":
                        pending.clear(); ex.abort("stop command"); publish(); log("STOP")
                    elif ex.busy:
                        pending.append(cmd); log(f"queued {cmd.get('cmd')}")
                    else:
                        start(cmd)
            if not ex.busy and pending:
                start(pending.popleft()); continue
            if not ex.busy:
                if a.offline or now - idle_since >= a.idle_exit_s:
                    break
                if now - t_status >= 2.0:
                    publish(); t_status = now
                link.flush(); time.sleep(1.0); continue
            idle_since = time.monotonic()
            wheel, events = ex.step(obs)
            if events:
                dispatch(events)
            t_next = float(obs["timestamp_s"]) + 0.1
            pos, quat = env.robot.get_world_poses(); chase.update(pos[0], quat[0])
            env.want = {"side": ex.wants_side_images(t_next), "extra": step % a.chase_every == 0,
                        "front": step % a.front_every == 0}
            want = dict(env.want)
            obs, _, term, _, info = env.step(wheel)
            step += 1
            leg = ex.current_leg
            fix = None
            if ex.xy is not None and leg is not None:                    # warehouse-style: scan-to-map every step
                (fx, fy, fh), q = loc.localize(scan_points(obs, max_range_m=11.0), ex.xy[0], ex.xy[1], ex.heading)
                outside = leg.kind in ("path", "enter")
                if q["ok"]:
                    ex.apply_fix(fx, fy, fh, xy=outside, t=float(obs["timestamp_s"]))
                fix = {**q, "pose": [round(fx, 3), round(fy, 3), round(fh, 2)], "applied_xy": q["ok"] and outside}
            ex.note_command(float(obs["timestamp_s"]), wheel)
            if "front_rgb" in obs:
                last_front = obs["front_rgb"]
                Image.fromarray(last_front).save(rec / "front" / f"{step:06d}.jpg", quality=92)
            if want["extra"]:
                Image.fromarray(chase.image()).save(rec / "chase" / f"{step:06d}.jpg", quality=93)
            write("steps", {"step": step, "t": round(float(obs["timestamp_s"]), 3), "wheel": [round(v, 4) for v in wheel],
                            "leg": ex.current_leg.kind if ex.current_leg else None, "aisle": ex.aisle,
                            "metre": round(ex.metre, 3), "heading_deg": round(ex.heading, 2), "xy": ex.xy,
                            "confidence": round(ex.confidence, 3), "rendered": "".join(k[0] for k, v in want.items() if v),
                            "lidar": [round(r, 3) if math.isfinite(r) else None for r in obs["lidar_m"]],
                            "gt_scene_pose": info.get("scene_pose"), "scan_to_map": fix})
            if step % 50 == 0:
                el = time.time() - t_wall0
                log(f"step {step} sim {float(obs['timestamp_s']):.1f}s wall {el / 60:.1f} min ({el / step:.2f} s/step, render "
                    f"{env.render_seconds / 60:.1f} min) leg={ex.current_leg.kind if ex.current_leg else None} "
                    f"aisle={ex.aisle} metre={ex.metre:.2f} xy={tuple(round(v, 2) for v in ex.xy)} "
                    f"gt={[round(v, 2) for v in (info.get('scene_pose') or [])[:4]]}")
            if time.monotonic() - t_status >= 2.0:
                publish(); t_status = time.monotonic()
            if term:
                ex.abort(f"sim ended: {info.get('termination')}"); log(f"sim ended {info}"); code = 2; break
            if float(obs["timestamp_s"]) > a.max_sim_s or (a.max_steps and step >= a.max_steps):
                ex.abort("step/time limit"); log("limit reached"); break
    except Exception:
        import traceback
        log("ERROR " + traceback.format_exc()); code = 3
    finally:
        try:
            if ex is not None and obs is not None:
                publish()
            link.flush()
            summ = {"steps": step, "sim_time_s": float(obs["timestamp_s"]) if obs else None,
                    "final_status": ex.status if ex else None, "reason": ex.reason if ex else None,
                    "frames": ex.frame_id if ex else 0, "frames_sent": link.sent}
            log(f"summary {json.dumps(summ, default=float)}")
            (rec / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
            for f in files.values():
                f.close()
        finally:
            os._exit(code)


if __name__ == "__main__":
    sys.exit(main())
