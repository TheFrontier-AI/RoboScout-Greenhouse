"""Smoke test for the full-farm run: realism stage + scene_prep (crop swap, open ends, collision, yard ground),
render-on-demand env, game-style chase camera. Drives straight for N steps, times physics-only vs rendered steps,
saves side/front/chase images and LiDAR hit counts. No training.

  /isaac-sim/python.sh scripts/farm_smoke.py --out /runs/smoke --steps 40"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R180 = Path(os.environ.get("R180_DIR", "/home/passin/Documents/Blender/TheFarm/robot_design/r1_v3/diy_r50/physical_ai_r180"))
STAGE = R180 / "tomato_realism_2026-10-28/C2_R180_Interior_PT_Sensor_Fixture.usda"


class ChaseCam:
    """Third-person camera locked behind the robot (fixed offset in the robot frame, yaw smoothed), like a game."""

    def __init__(self, stage, res=(1920, 1080), back=2.4, up=1.35, ahead=2.0, look_z=0.55, smooth=0.25):
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom
        self.back, self.up, self.ahead, self.look_z, self.smooth = back, up, ahead, look_z, smooth
        cam = UsdGeom.Camera.Define(stage, "/ChaseCam")
        cam.CreateFocalLengthAttr(18.0); cam.CreateHorizontalApertureAttr(36.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500.0))
        self.op = UsdGeom.Xformable(cam.GetPrim()).AddTransformOp()
        self.rp = rep.create.render_product("/ChaseCam", res)
        self.ann = rep.AnnotatorRegistry.get_annotator("rgb"); self.ann.attach([self.rp])
        self.yaw = None

    def update(self, pos, quat_wxyz):
        from pxr import Gf
        w, x, y, z = (float(v) for v in quat_wxyz)
        fwd_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) - math.pi / 2   # robot forward = body -Y
        if self.yaw is None:
            self.yaw = fwd_yaw
        d = (fwd_yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
        self.yaw += self.smooth * d
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        px, py = float(pos[0]), float(pos[1])
        eye = Gf.Vec3d(px - self.back * c, py - self.back * s, self.up)
        tgt = Gf.Vec3d(px + self.ahead * c, py + self.ahead * s, self.look_z)
        self.op.Set(Gf.Matrix4d().SetLookAt(eye, tgt, Gf.Vec3d(0, 0, 1)).GetInverse())

    def image(self):
        import numpy as np
        return np.asarray(self.ann.get_data())[..., :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--res", default="8k"); ap.add_argument("--plant-res", default="1080x1920")
    ap.add_argument("--physics-render", action="store_true")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[smoke {time.strftime('%H:%M:%S')}] {m}", flush=True)
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True})
        for p in (R180, R180.parent / "physical_ai_runtime", R180 / "tomato_realism_2026-10-28"):
            sys.path.insert(0, str(p))
        from gated_render_sensor_env import GatedRenderSensorEnv
        from sim.fast_env import YardEvaluator, make_fast_env
        from sim.scene_prep import prepare_farm_run
        from PIL import Image
        import numpy as np
        Env = make_fast_env(GatedRenderSensorEnv); Env.physics_render = a.physics_render
        w, h = (int(v) for v in a.plant_res.split("x"))
        t0 = time.time()
        env = Env(app, stage_path=STAGE, aisle_half_width=0.38, body_half_length=0.375, plant_cameras=True, plant_resolution=(w, h))
        log(f"env built {time.time() - t0:.1f}s")
        t0 = time.time(); report = prepare_farm_run(env.stage, res=a.res, log=log); log(f"scene prep {time.time() - t0:.1f}s")
        (a.out / "scene_prep.json").write_text(json.dumps(report, indent=1, default=str))
        chase = ChaseCam(env.stage)
        env.extra_products.append(chase.rp.hydra_texture)
        t0 = time.time()
        obs, info = env.reset(options={"obstacle_enabled": False, "task_mode": "navigation", "max_steps": None})
        env.set_evaluator(YardEvaluator()); log(f"reset {time.time() - t0:.1f}s")
        times = []
        for i in range(a.steps):
            pos, quat = env.robot.get_world_poses(); chase.update(pos[0], quat[0])
            env.want = {"extra": i % 2 == 0, "front": i % 5 == 0, "side": i in (5, a.steps - 1)}
            t0 = time.time(); obs, _, term, trunc, info = env.step([1.2, 1.2]); dt = time.time() - t0
            hits = sum(1 for r in obs["lidar_m"] if math.isfinite(r) and r < 11.9)
            times.append({"i": i, "s": round(dt, 3), "want": env_want_str(i, a.steps), "hits": hits, "pose": info.get("scene_pose")})
            log(f"step {i} {dt:.2f}s hits {hits} pose {[round(v, 2) for v in (info.get('scene_pose') or [])[:4]]}")
            if i % 2 == 0:
                Image.fromarray(chase.image()).save(a.out / f"chase_{i:03d}.png")
            if "front_rgb" in obs:
                Image.fromarray(obs["front_rgb"]).save(a.out / f"front_{i:03d}.png")
            for n, img in obs.get("plant_rgb", {}).items():
                Image.fromarray(img).save(a.out / f"{n}_{i:03d}.png")
            if term:
                log(f"terminated {info}"); break
        (a.out / "timing.json").write_text(json.dumps(times, indent=1))
        log("done")
    except Exception:
        import traceback
        log("ERROR " + traceback.format_exc())
    finally:
        os._exit(0)


def env_want_str(i, n):
    return "".join(k for k, v in (("C", i % 2 == 0), ("F", i % 5 == 0), ("S", i in (5, n - 1))) if v)


if __name__ == "__main__":
    main()
