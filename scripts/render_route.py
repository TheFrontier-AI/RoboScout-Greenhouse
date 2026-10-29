"""Re-render a recorded R1 run on a new scene, every camera, split across GPUs (render farm). The robot is placed at its
Isaac ground-truth pose of each instant (no new drive, no AI decisions), so every frame is independent and the parts
join into one continuous, real-time video per camera.

Streams (timestamps = sim time of the recorded run, ns):
  chase   game-style camera locked behind the robot      1920x1080, 5 Hz   (yaw smoothed over the whole path)
  front   robot front camera                             1280x720,  2 Hz
  zed     R181-S ZED 2i left/right/depth                 --zed-res, 15 Hz  (+ IMU, ground truth, calib: part 0)
  shots   sim.shots CCTV + bystander views, in zone       1920x1080, 10 Hz
  side    the six plant cameras at every recorded plant stop, 1080x1920    (run with --profile interior: ISO 400)
  lidar   planar C1 scan by PhysX raycasts at the LiDAR pose, 10 Hz (no render)

  /isaac-sim/python.sh scripts/render_route.py --rec /rec --out /out --streams chase,front,zed,shots,lidar --part 0 --parts 7
  /isaac-sim/python.sh scripts/render_route.py --rec /rec --out /out --streams side --profile interior"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from run_farm import R180, log                     # noqa: E402
from replay_route import camera_imu, quat_wxyz      # noqa: E402

STAGE = ROOT / "sim/stages/R181S_permanent_review2.usda"
PROFILES = R180 / "portable_scene_review2/render_profiles.json"
BASE_HZ = 30
EVERY = {"zed": 2, "shots": 3, "lidar": 3, "chase": 6, "front": 15}       # ticks of the 30 Hz grid
LIDAR_POS = (0.0, -0.34, 0.512)            # physical_ai_r167_lidar POSITION in the Physics_Base frame
LIDAR_BEAMS, LIDAR_MAX = 500, 12.0


def survey_window(rec: Path):
    """From the first recorded step to the dock arrival (revisits after that are not part of the route)."""
    steps = [json.loads(l) for l in open(rec / "steps.jsonl")]
    t0 = next(s["t"] for s in steps if s.get("gt_scene_pose"))
    dock = [s["t"] for s in steps if s.get("leg") == "dock"]
    return t0, (min(dock) if dock else steps[-1]["t"])


def smoothed_heading(t, x, y, yaw, tau_s: float = 1.2):
    """Chase-camera heading: robot forward heading low-pass filtered forward AND backward in time (zero lag, stateless
    per frame, so any GPU renders the same view at the same instant)."""
    import numpy as np
    fwd = np.unwrap(yaw - math.pi / 2)                      # robot forward = body -Y
    a = 1 - math.exp(-(t[1] - t[0]) / tau_s)
    f = fwd.copy()
    for i in range(1, len(f)):
        f[i] = f[i - 1] + a * (f[i] - f[i - 1])
    for i in range(len(f) - 2, -1, -1):
        f[i] = f[i + 1] + a * (f[i] - f[i + 1])
    return f


def build_jobs(rec, t0, t1, streams, shots_mod):
    """[(t, {stream: [cameras]})] on the 30 Hz grid, plus side-camera stops at the recorded capture instants."""
    import numpy as np
    from replay_route import load_track
    t, x, y, z, yaw = load_track(rec, 0, 10 ** 9)
    jobs = {}
    n = int((t1 - t0) * BASE_HZ)
    for k in range(n + 1):
        tt = round(t0 + k / BASE_HZ, 6)
        want = {}
        for s in ("chase", "front", "zed", "lidar"):
            if s in streams and k % EVERY[s] == 0:
                want[s] = [s]
        if "shots" in streams and k % EVERY["shots"] == 0:
            px, py = float(np.interp(tt, t, x)), float(np.interp(tt, t, y))
            cams = [sh.name for sh in shots_mod.SHOTS if shots_mod.in_zone(sh, px, py)]
            if cams:
                want["shots"] = cams
        if want:
            jobs[tt] = want
    if "side" in streams:
        stops = sorted({round(e["t"], 6) for e in map(json.loads, open(rec / "events.jsonl"))
                        if e.get("type") == "capture" and str(e.get("camera", "")).startswith("mast_") and t0 <= e["t"] <= t1})
        for tt in stops:
            jobs.setdefault(tt, {})["side"] = ["side"]
    return sorted(jobs.items())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage", type=Path, default=STAGE)
    ap.add_argument("--streams", default="chase,front,zed,shots,lidar")
    ap.add_argument("--part", type=int, default=0); ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--profile", default="exterior_reference", choices=("exterior_reference", "interior_inspection"))
    ap.add_argument("--spp", type=int, help="override path-tracing samples per pixel (total); sensor-only parts can use fewer")
    ap.add_argument("--zed-every", type=int, default=EVERY["zed"], help="30 Hz ticks per ZED frame (2 = 15 Hz, 3 = 10 Hz)")
    ap.add_argument("--zed-res", default="1280x720"); ap.add_argument("--t-from", type=float); ap.add_argument("--t-to", type=float)
    a = ap.parse_args()
    streams = set(a.streams.split(","))
    EVERY["zed"] = a.zed_every
    import numpy as np
    from PIL import Image
    from replay_route import load_track
    t0, t1 = survey_window(a.rec)
    t0, t1 = (a.t_from if a.t_from is not None else t0), (a.t_to if a.t_to is not None else t1)
    t, x, y, z, yaw = load_track(a.rec, 0, 10 ** 9)
    head = smoothed_heading(t, x, y, yaw)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        for p in (R180, R180.parent / "physical_ai_runtime", R180 / "tomato_realism_2026-10-28"):
            sys.path.insert(0, str(p))
        import carb, omni.physx, omni.replicator.core as rep
        from pxr import Usd, UsdGeom
        from gated_render_sensor_env import GatedRenderSensorEnv
        from sim import shots as shots_mod
        from sim.fast_env import YardEvaluator, make_fast_env
        from sim.zed2i_rig import add_zed2i
        from farm_smoke import ChaseCam
        jobs = build_jobs(a.rec, t0, t1, streams, shots_mod)
        mine = jobs[a.part::a.parts]
        log(f"part {a.part}/{a.parts}: {len(mine)} of {len(jobs)} instants, streams {sorted(streams)}, {t1 - t0:.0f} s of route")
        Env = make_fast_env(GatedRenderSensorEnv)
        env = Env(app, stage_path=a.stage, aisle_half_width=0.38, body_half_length=0.375, plant_cameras="side" in streams,
                  plant_resolution=(1080, 1920), front_resolution=(1280, 720))
        env.reset(options={"obstacle_enabled": False, "task_mode": "navigation", "max_steps": None,
                           "start_pose": {"x": float(x[0]), "y": float(y[0]), "yaw_deg": 0.0}})
        env.set_evaluator(YardEvaluator())
        prof = json.loads(PROFILES.read_text())["profiles"][a.profile]
        st = carb.settings.get_settings()
        for k, v in prof.items():
            st.set(k, v)
        env._spp_full = (prof["/rtx/pathtracing/spp"], prof["/rtx/pathtracing/totalSpp"])
        if a.spp:
            env._spp_full = (max(1, a.spp // 2), a.spp)
        zw, zh = (int(v) for v in a.zed_res.split("x"))
        rig = add_zed2i(env.stage, (zw, zh)) if "zed" in streams else None
        chase = ChaseCam(env.stage) if "chase" in streams else None
        shots = shots_mod.ShotCameras(env.stage) if "shots" in streams else None
        tex = {"chase": [chase.rp.hydra_texture] if chase else [], "front": [env.camera._render_product.hydra_texture],
               "zed": rig.hydra_textures if rig else [],
               "side": [c._render_product.hydra_texture for c in env.plant_cameras.values()]}
        alltex = [tx for v in tex.values() for tx in v] + (shots.textures([s.name for s in shots_mod.SHOTS]) if shots else [])
        for tx in alltex + env._all_products():
            tx.set_updates_enabled(False)
        _, q0 = env.robot.get_world_poses(); q0 = np.asarray(q0[0], float)
        yaw0 = math.atan2(2 * (q0[0] * q0[3] + q0[1] * q0[2]), 1 - 2 * (q0[2] ** 2 + q0[3] ** 2))

        def place(tt):
            xx, yy, zz, hh = (float(np.interp(tt, t, v)) for v in (x, y, z, yaw))
            dq = hh - yaw0; c, s = math.cos(dq / 2), math.sin(dq / 2); w0, a0, b0, c0 = q0
            env.robot.set_world_poses(positions=np.array([[xx, yy, zz]]),
                                      orientations=np.array([[c * w0 - s * c0, c * a0 - s * b0, c * b0 + s * a0, c * c0 + s * w0]]))
            env.robot.set_velocities(np.zeros((1, 6), dtype=np.float32))
            return xx, yy, zz, hh

        def chase_set(tt, xx, yy):
            from pxr import Gf
            h = float(np.interp(tt, t, head)); c, s = math.cos(h), math.sin(h)
            eye = Gf.Vec3d(xx - chase.back * c, yy - chase.back * s, chase.up)
            tgt = Gf.Vec3d(xx + chase.ahead * c, yy + chase.ahead * s, chase.look_z)
            chase.op.Set(Gf.Matrix4d().SetLookAt(eye, tgt, Gf.Vec3d(0, 0, 1)).GetInverse())

        sq = omni.physx.get_physx_scene_query_interface()

        def lidar_scan(xx, yy, zz, hh):
            """R180 raw azimuth order (beam k at -pi + 2 pi k / n in the sensor frame, sensor +X = body +X)."""
            c, s = math.cos(hh), math.sin(hh)
            lx, ly, lz = LIDAR_POS
            ox, oy = xx + c * lx - s * ly, yy + s * lx + c * ly
            out = []
            for k in range(LIDAR_BEAMS):
                az = -math.pi + 2 * math.pi * k / LIDAR_BEAMS + hh
                h_ = sq.raycast_closest(carb.Float3(ox, oy, lz + zz), carb.Float3(math.cos(az), math.sin(az), 0.0), LIDAR_MAX)
                if h_["hit"] and str(h_["collision"]).startswith("/Robot"):
                    out.append(None); continue        # own body: the real driver masks it too
                out.append(round(h_["distance"], 3) if h_["hit"] and h_["distance"] >= 0.05 else None)
            return out

        root_cam = None
        if rig:
            M = lambda p: np.array(UsdGeom.Xformable(env.stage.GetPrimAtPath(p)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            root_cam = np.linalg.inv(M("/Robot/Physics_Base")) @ M(rig.cams["left"]); root_cam[:3, :3] = root_cam[:3, :3] @ np.diag([1.0, -1.0, -1.0])

        def rotm(q):
            w_, x_, y_, z_ = (float(v) for v in q)
            return np.array([[1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - w_ * z_), 2 * (x_ * z_ + w_ * y_)],
                             [2 * (x_ * y_ + w_ * z_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - w_ * x_)],
                             [2 * (x_ * z_ - w_ * y_), 2 * (y_ * z_ + w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)]])

        out = a.out
        for d in ("chase", "front", "zed/cam0", "zed/cam1", "zed/depth0", "lidar", "side", "shots"):
            (out / d).mkdir(parents=True, exist_ok=True)
        gt_rows, lidar_rows, t_start = [], [], time.time()
        for i, (tt, want) in enumerate(mine):
            ns = int(round(tt * 1e9))
            xx, yy, zz, hh = place(tt); env.world.step(render=False); xx, yy, zz, hh = place(tt)
            if "lidar" in want:
                lidar_rows.append({"t": tt, "ns": ns, "pose": [xx, yy, math.degrees(hh)], "lidar": lidar_scan(xx, yy, zz, hh)})
            on = []
            if "chase" in want:
                chase_set(tt, xx, yy); on += tex["chase"]
            if "shots" in want:
                shots.update(tt, (xx, yy, zz)); on += shots.textures(want["shots"])
            for s in ("front", "zed", "side"):
                if s in want:
                    on += tex[s]
            if on:
                env._quality(True)
                for tx in on:
                    tx.set_updates_enabled(True)
                for _ in range(2):
                    rep.orchestrator.step(delta_time=0.0, pause_timeline=False, rt_subframes=1)
                for tx in on:
                    tx.set_updates_enabled(False)
            if "chase" in want:
                Image.fromarray(chase.image()).save(out / "chase" / f"{ns}.jpg", quality=95)
            if "front" in want:
                Image.fromarray(np.asarray(env.camera.get_rgba())[..., :3]).save(out / "front" / f"{ns}.jpg", quality=95)
            if "zed" in want:
                Image.fromarray(rig.rgb("left")).save(out / "zed/cam0" / f"{ns}.png")
                Image.fromarray(rig.rgb("right")).save(out / "zed/cam1" / f"{ns}.png")
                Image.fromarray(np.round(rig.depth() * 1000).astype(np.uint16)).save(out / "zed/depth0" / f"{ns}.png")
                pos, quat = env.robot.get_world_poses()
                Mb = np.eye(4); Mb[:3, :3] = rotm(quat[0]); Mb[:3, 3] = pos[0]; C = Mb @ root_cam
                gt_rows.append((ns, *C[:3, 3], *quat_wxyz(C[:3, :3])))
            if "shots" in want:
                for n in want["shots"]:
                    (out / "shots" / n).mkdir(exist_ok=True)
                    Image.fromarray(shots.image(n)).save(out / "shots" / n / f"{ns}.jpg", quality=95)
            if "side" in want:
                for n, cam in env.plant_cameras.items():
                    Image.fromarray(np.asarray(cam.get_rgba())[..., :3]).save(out / "side" / f"{ns}_{n}.png")
            if i % 25 == 0:
                el = time.time() - t_start
                log(f"part {a.part}: {i}/{len(mine)} t={tt:.1f}s {el / max(1, i):.2f} s/instant, eta {(len(mine) - i) * el / max(1, i) / 60:.0f} min")
        if gt_rows:
            (out / "zed").mkdir(exist_ok=True)
            with open(out / "zed" / f"groundtruth_part{a.part}.csv", "w") as f:
                for r in gt_rows:
                    f.write(",".join([str(r[0])] + [f"{v:.6f}" for v in r[1:]]) + "\n")
        if lidar_rows:
            with open(out / "lidar" / f"scans_part{a.part}.jsonl", "w") as f:
                for r in lidar_rows:
                    f.write(json.dumps(r) + "\n")
        if rig and a.part == 0:                                # IMU + calibration for the whole window (cheap)
            Rq0 = rotm(q0); Rz0 = np.array([[math.cos(yaw0), -math.sin(yaw0), 0], [math.sin(yaw0), math.cos(yaw0), 0], [0, 0, 1.0]])
            T = np.eye(4); T[:3, :3] = Rz0.T @ Rq0 @ root_cam[:3, :3]; T[:3, 3] = Rz0.T @ Rq0 @ root_cam[:3, 3]
            t_imu = np.arange(t0, t1, 1.0 / 400)
            with open(out / "zed" / "imu0.csv", "w") as f:
                f.write("#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n")
                for ti, (wv, fv) in zip(t_imu, camera_imu(t_imu, t, x, y, z, yaw, T)):
                    f.write(f"{int(round(ti * 1e9))}," + ",".join(f"{v:.6f}" for v in (*wv, *fv)) + "\n")
            (out / "zed" / "calib.json").write_text(json.dumps({**rig.calibration(), "T_base_leftcam": T.tolist(), "robot": "R181-S",
                                                                "imu_frame": "left camera, OpenCV axes", "stage": str(a.stage),
                                                                "window_s": [t0, t1]}, indent=1))
        (out / f"part{a.part}_{a.profile}.done").write_text(json.dumps({"instants": len(mine), "minutes": (time.time() - t_start) / 60}))
        log(f"part {a.part} done: {len(mine)} instants in {(time.time() - t_start) / 60:.1f} min")
    except Exception:
        import traceback
        log("ERROR " + traceback.format_exc())
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
