"""Storyboard: one still from every camera we will present, on Astra's permanent scene (R181-S robot), with the robot
placed at its recorded pose from a real run. Robot cameras: chase, front, six side cameras at a plant stop, ZED 2i
left/right/depth, the 2-D LiDAR scan. Extra cameras: sim.shots (CCTV + bystander views).

  /isaac-sim/python.sh scripts/shot_preview.py --rec /rec --out /out"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from run_farm import R180, log                     # noqa: E402

STAGE = ROOT / "sim/stages/R181S_permanent_review2.usda"
# moments of run 3 (control step -> what to show)
MOMENTS = [(150, "leaving the dock"), (1000, "tunnel-09 aisle, metre ~7"), (2450, "tunnel-09 near the back end"),
           (3000, "back yard crossing"), (3800, "tunnel-10 near the back end"), (5600, "tunnel-10 near the front end"),
           (6300, "returning to the dock")]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--rec", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plant-metre", type=float, default=12.1, help="side-camera stop to show (tunnel-09 chlorosis plant at 12.1 m)")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    from PIL import Image
    steps = {}
    for line in open(a.rec / "steps.jsonl"):
        d = json.loads(line); steps[d["step"]] = d
    caps = [json.loads(l) for l in open(a.rec / "events.jsonl")]
    caps = [e for e in caps if e.get("type") == "capture" and str(e.get("camera", "")).startswith("mast_")
            and str(e.get("aisle", "")).startswith("tunnel-09") and e.get("metre") is not None]
    stop = min(caps, key=lambda e: abs(e["metre"] - a.plant_metre)) if caps else None
    stop_step = min(steps, key=lambda s: abs(steps[s]["t"] - stop["t"])) if stop else 1000

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        for p in (R180, R180.parent / "physical_ai_runtime", R180 / "tomato_realism_2026-10-28"):
            sys.path.insert(0, str(p))
        import omni.replicator.core as rep
        from gated_render_sensor_env import GatedRenderSensorEnv
        from sim.fast_env import YardEvaluator, make_fast_env
        from sim.shots import SHOTS, ShotCameras, in_zone
        from sim.zed2i_rig import add_zed2i
        from farm_smoke import ChaseCam
        from make_farm_videos import lidar_frame
        Env = make_fast_env(GatedRenderSensorEnv)
        env = Env(app, stage_path=STAGE, aisle_half_width=0.38, body_half_length=0.375, plant_cameras=True,
                  plant_resolution=(1080, 1920), front_resolution=(1280, 720))
        x0, y0 = steps[min(steps)]["gt_scene_pose"][:2]
        env.reset(options={"obstacle_enabled": False, "task_mode": "navigation", "max_steps": None,
                           "start_pose": {"x": x0, "y": y0, "yaw_deg": 0.0}})
        env.set_evaluator(YardEvaluator())
        rig = add_zed2i(env.stage, (1920, 1080)); chase = ChaseCam(env.stage); shots = ShotCameras(env.stage)
        _, q0 = env.robot.get_world_poses(); q0 = np.asarray(q0[0], float)
        yaw0 = math.atan2(2 * (q0[0] * q0[3] + q0[1] * q0[2]), 1 - 2 * (q0[2] ** 2 + q0[3] ** 2))

        def place(st):
            x, y, z, yaw = steps[st]["gt_scene_pose"][:4]; dq = math.radians(yaw) - yaw0
            c, s = math.cos(dq / 2), math.sin(dq / 2); w0, a0, b0, c0 = q0
            env.robot.set_world_poses(positions=np.array([[x, y, z]]),
                                      orientations=np.array([[c * w0 - s * c0, c * a0 - s * b0, c * b0 + s * a0, c * c0 + s * w0]]))
            env.robot.set_velocities(np.zeros((1, 6), dtype=np.float32))
            return x, y, z

        own = [tex for tex in env._all_products()] + rig.hydra_textures + [chase.rp.hydra_texture] + shots.textures([s.name for s in SHOTS])

        def render(texs):
            for tex in own:
                tex.set_updates_enabled(False)
            env._quality(True)
            for tex in texs:
                tex.set_updates_enabled(True)
            for _ in range(2):
                rep.orchestrator.step(delta_time=0.0, pause_timeline=False, rt_subframes=1)

        index = []
        for st, label in MOMENTS + [(stop_step, f"plant stop at metre {stop['metre']:.1f}" if stop else "plant stop")]:
            st = min(steps, key=lambda s: abs(s - st))
            place(st); obs, *_ = env.step([0.0, 0.0]); xyz = place(st)
            for tex in own:
                tex.set_updates_enabled(False)
            env.world.step(render=False); xyz = place(st)
            pos, quat = env.robot.get_world_poses(); chase.update(pos[0], quat[0]); shots.update(steps[st]["t"], xyz)
            tag = f"s{st:05d}"
            lidar_frame([float(r) if np.isfinite(r) else None for r in obs["lidar_m"]]).save(a.out / f"{tag}_lidar.png")
            index.append({"step": st, "label": label, "camera": "lidar", "file": f"{tag}_lidar.png"})
            want = [("chase", [chase.rp.hydra_texture], lambda: chase.image()),
                    ("front", [env.camera._render_product.hydra_texture], lambda: np.asarray(env.camera.get_rgba())[..., :3])]
            if (st, label) == (stop_step, label) and "plant stop" in label:
                for n, cam in env.plant_cameras.items():
                    want.append((n, [cam._render_product.hydra_texture], (lambda c=cam: np.asarray(c.get_rgba())[..., :3])))
            if st in (1000, 3000):
                want.append(("zed_left", rig.hydra_textures, lambda: rig.rgb("left")))
            want += [(s.name, shots.textures([s.name]), (lambda n=s.name: shots.image(n))) for s in SHOTS
                     if in_zone(s, xyz[0], xyz[1])]
            for name, texs, grab in want:
                t0 = time.time(); render(texs); img = grab()
                Image.fromarray(np.ascontiguousarray(img)).save(a.out / f"{tag}_{name}.jpg", quality=92)
                if name == "zed_left":
                    Image.fromarray(rig.rgb("right")).save(a.out / f"{tag}_zed_right.jpg", quality=92)
                    d = rig.depth(); Image.fromarray((np.clip(d / 12.0, 0, 1) * 255).astype(np.uint8)).save(a.out / f"{tag}_zed_depth.png")
                index.append({"step": st, "label": label, "camera": name, "file": f"{tag}_{name}.jpg", "render_s": round(time.time() - t0, 2)})
            log(f"{label}: {len(want)} cameras")
        (a.out / "index.json").write_text(json.dumps(index, indent=1))
        log(f"done {len(index)} images -> {a.out}")
    except Exception:
        import traceback
        log("ERROR " + traceback.format_exc())
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
