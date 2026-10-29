"""Replay a recorded R1 run (the robot's real, AI-driven path) and re-render it with new sensors or a new scene.

The robot is placed at the Isaac ground-truth pose of every recorded step (interpolated to --hz) and only cameras
are rendered: no physics decisions, no AI, so this is a re-render of the route, never a new drive. Uses:
  - R181-S sensor data: ZED 2i stereo pair (+ depth, + IMU computed from the real motion) for cuVSLAM and other SLAM;
  - animation: the game-style chase camera along the same route (e.g. after a scene update).

Output (EuRoC-style, timestamps = sim time in ns):
  cam0/data/<ns>.png  cam1/data/<ns>.png   left / right, rectified
  depth0/data/<ns>.png                     left depth, uint16 millimetres, 0 = no depth (outside 0.3-20 m)
  chase/<ns>.jpg                           third-person camera (--chase)
  imu0/data.csv                            #timestamp [ns], w_x, w_y, w_z [rad s^-1], a_x, a_y, a_z [m s^-2]
                                           in the LEFT CAMERA optical frame (x right, y down, z forward), noise-free
  groundtruth.csv                          #timestamp [ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z: left camera (OpenCV) in the scene
  calib.json                               intrinsics, baseline, IMU noise, which run and steps were replayed

  /isaac-sim/python.sh scripts/replay_route.py --rec /runs/rec --out /runs/replay --from-step 334 --to-step 634 --hz 30
  /isaac-sim/python.sh scripts/replay_route.py --rec /runs/rec --out /runs/anim --hz 10 --no-stereo --chase"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from run_farm import R180, STAGE, HOUSES, log          # noqa: E402

G = 9.81


def load_track(rec: Path, a, b):
    import numpy as np
    rows = []
    for line in open(rec / "steps.jsonl"):
        d = json.loads(line)
        if a <= d["step"] <= b and d.get("gt_scene_pose"):
            x, y, z, yaw = d["gt_scene_pose"][:4]
            rows.append((d["t"], x, y, z, math.radians(yaw)))
    t, x, y, z, yaw = (np.array(c, float) for c in zip(*rows))
    return t, x, y, z, np.unwrap(yaw)


def camera_imu(t_out, t, x, y, z, yaw, T_mount):
    """Specific force and angular rate of the left camera (OpenCV frame) from the smoothed real robot path."""
    import numpy as np
    from scipy.interpolate import CubicSpline
    sx, sy, sz, sw = (CubicSpline(t, v) for v in (x, y, z, yaw))
    out = []
    Rm, pm = T_mount[:3, :3], T_mount[:3, 3]
    for tt in t_out:
        h = sw(tt); c, s = math.cos(h), math.sin(h)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        w_dot = float(sw(tt, 1)); w_dd = float(sw(tt, 2))
        acc_base = np.array([sx(tt, 2), sy(tt, 2), sz(tt, 2)])
        r = Rz @ pm                                                   # mount lever arm in the world
        omega = np.array([0, 0, w_dot]); alpha = np.array([0, 0, w_dd])
        acc_cam = acc_base + np.cross(alpha, r) + np.cross(omega, np.cross(omega, r))
        Rc = Rz @ Rm                                                  # world <- camera (OpenCV)
        f = Rc.T @ (acc_cam - np.array([0, 0, -G]))
        out.append((Rc.T @ omega, f))
    return out


def quat_wxyz(R):
    import numpy as np
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", type=Path, required=True, help="run_farm.py record dir (steps.jsonl with gt_scene_pose)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--from-step", type=int, default=0); ap.add_argument("--to-step", type=int, default=10**9)
    ap.add_argument("--hz", type=float, default=30.0); ap.add_argument("--imu-hz", type=float, default=400.0)
    ap.add_argument("--res", default="1920x1080", help="stereo resolution (ZED 2i HD1080; 960x540 for quick tests)")
    ap.add_argument("--no-stereo", action="store_true"); ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--chase", action="store_true", help="also render the third-person camera (animation)")
    ap.add_argument("--crop-res", default="2k", choices=("2k", "8k"))
    a = ap.parse_args()
    import numpy as np
    from PIL import Image
    t, x, y, z, yaw = load_track(a.rec, a.from_step, a.to_step)
    t_frames = np.arange(t[0], t[-1], 1.0 / a.hz)
    log(f"replay {len(t)} recorded poses, {t[-1] - t[0]:.1f} s -> {len(t_frames)} frames at {a.hz:g} Hz")
    out = a.out; [(out / d).mkdir(parents=True, exist_ok=True) for d in ("cam0/data", "cam1/data", "depth0/data", "imu0", "chase")]

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        for p in (R180, R180.parent / "physical_ai_runtime", R180 / "tomato_realism_2026-10-28"):
            sys.path.insert(0, str(p))
        import omni.replicator.core as rep
        from pxr import Usd, UsdGeom
        from gated_render_sensor_env import GatedRenderSensorEnv
        from sim.fast_env import YardEvaluator, make_fast_env
        from sim.scene_prep import prepare_farm_run
        from sim.zed2i_rig import add_zed2i
        from farm_smoke import ChaseCam
        Env = make_fast_env(GatedRenderSensorEnv)
        env = Env(app, stage_path=STAGE, aisle_half_width=0.38, body_half_length=0.375, plant_cameras=False)
        report = prepare_farm_run(env.stage, tunnels=tuple(h[0][7:9] for h in HOUSES), res=a.crop_res, log=log)
        env.reset(options={"obstacle_enabled": False, "task_mode": "navigation", "max_steps": None,
                           "start_pose": {"x": float(x[0]), "y": float(y[0]), "yaw_deg": 0.0}})
        env.set_evaluator(YardEvaluator())
        w, h = (int(v) for v in a.res.split("x"))
        rig = None if a.no_stereo else add_zed2i(env.stage, (w, h), depth=not a.no_depth)
        chase = ChaseCam(env.stage) if a.chase else None
        for tex in env._all_products():
            tex.set_updates_enabled(False)
        mine = (rig.hydra_textures if rig else []) + ([chase.rp.hydra_texture] if chase else [])
        _, q0 = env.robot.get_world_poses(); q0 = np.asarray(q0[0], float)          # authored base orientation
        yaw0 = math.atan2(2 * (q0[0] * q0[3] + q0[1] * q0[2]), 1 - 2 * (q0[2] ** 2 + q0[3] ** 2))

        def place(i_t):
            xx, yy, zz, hh = (float(np.interp(i_t, t, v)) for v in (x, y, z, yaw))
            dq = hh - yaw0; c, s = math.cos(dq / 2), math.sin(dq / 2)
            w0, x0, y0, z0 = q0
            q = [c * w0 - s * z0, c * x0 - s * y0, c * y0 + s * x0, c * z0 + s * w0]   # yaw about world Z, as reset does
            env.robot.set_world_poses(positions=np.array([[xx, yy, zz]]), orientations=np.array([q]))
            env.robot.set_velocities(np.zeros((1, 6), dtype=np.float32))
            return xx, yy, zz, hh

        def usd_world(path):
            return np.array(UsdGeom.Xformable(env.stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T

        def rot(qw):
            w_, x_, y_, z_ = (float(v) for v in qw)
            return np.array([[1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - w_ * z_), 2 * (x_ * z_ + w_ * y_)],
                             [2 * (x_ * y_ + w_ * z_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - w_ * x_)],
                             [2 * (x_ * z_ - w_ * y_), 2 * (y_ * z_ + w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)]])

        # fixed mount: physics base (the pose get_world_poses reports; /Robot itself stays at the origin) -> left camera
        T_root_cam = None
        if rig:
            T_root_cam = np.linalg.inv(usd_world("/Robot/Physics_Base")) @ usd_world(rig.cams["left"])
            T_root_cam[:3, :3] = T_root_cam[:3, :3] @ np.diag([1.0, -1.0, -1.0])

        def cam_world(pos, quat):
            M = np.eye(4); M[:3, :3] = rot(quat); M[:3, 3] = np.asarray(pos, float)
            C = M @ T_root_cam
            return C[:3, :3], C[:3, 3]

        gt_rows, t0w = [], time.time()
        for k, tt in enumerate(t_frames):
            px, py, pz, ph = place(tt)
            for tex in env._all_products() + mine:
                tex.set_updates_enabled(False)
            env.world.step(render=False)          # one physics tick so the new pose reaches the renderer
            place(tt)                             # undo the tick's gravity/contact drift
            pos, quat = env.robot.get_world_poses()
            if chase:
                chase.update(pos[0], quat[0])
            env._quality(True)
            for tex in mine:
                tex.set_updates_enabled(True)
            for _ in range(2):
                rep.orchestrator.step(delta_time=0.0, pause_timeline=False, rt_subframes=1)
            ns = int(round(tt * 1e9))
            if rig:
                Image.fromarray(rig.rgb("left")).save(out / "cam0/data" / f"{ns}.png")
                Image.fromarray(rig.rgb("right")).save(out / "cam1/data" / f"{ns}.png")
                if not a.no_depth:
                    Image.fromarray(np.round(rig.depth() * 1000).astype(np.uint16)).save(out / "depth0/data" / f"{ns}.png")
                R, p = cam_world(pos[0], quat[0]); gt_rows.append((ns, *p, *quat_wxyz(R)))
            if chase:
                Image.fromarray(chase.image()).save(out / "chase" / f"{ns}.jpg", quality=92)
            if k % 50 == 0:
                el = time.time() - t0w
                log(f"frame {k}/{len(t_frames)} t={tt:.2f}s xy=({px:.2f},{py:.2f}) {el / max(1, k):.2f} s/frame")
        if rig:
            with open(out / "groundtruth.csv", "w") as f:
                f.write("#timestamp [ns],p_x,p_y,p_z,q_w,q_x,q_y,q_z\n")
                for r in gt_rows:
                    f.write(",".join([str(r[0])] + [f"{v:.6f}" for v in r[1:]]) + "\n")
            # IMU mount in the yaw-only base frame used by camera_imu (the authored base is upright)
            Rq0 = rot(q0); Rz0 = np.array([[math.cos(yaw0), -math.sin(yaw0), 0], [math.sin(yaw0), math.cos(yaw0), 0], [0, 0, 1.0]])
            T = np.eye(4); T[:3, :3] = Rz0.T @ Rq0 @ T_root_cam[:3, :3]; T[:3, 3] = Rz0.T @ Rq0 @ T_root_cam[:3, 3]
            t_imu = np.arange(t[0], t[-1], 1.0 / a.imu_hz)
            with open(out / "imu0/data.csv", "w") as f:
                f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                        "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
                for ti, (wv, fv) in zip(t_imu, camera_imu(t_imu, t, x, y, z, yaw, T)):
                    f.write(f"{int(round(ti * 1e9))}," + ",".join(f"{v:.6f}" for v in (*wv, *fv)) + "\n")
            (out / "calib.json").write_text(json.dumps({**rig.calibration(), "T_base_leftcam": T.tolist(),
                                                        "imu_frame": "left camera, OpenCV axes", "rec": str(a.rec),
                                                        "steps": [a.from_step, a.to_step], "hz": a.hz,
                                                        "scene_changes": list(report)}, indent=1))
        log(f"done: {len(t_frames)} frames in {(time.time() - t0w) / 60:.1f} min -> {out}")
    except Exception:
        import traceback
        log("ERROR " + traceback.format_exc())
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
