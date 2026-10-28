"""Still PathTracing views of the opened tunnel ends (front + back of tunnel-09 and tunnel-10) after scene_prep.
  /isaac-sim/python.sh scripts/door_views.py --out /runs/doors"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_farm import STAGE, R180            # noqa: E402

VIEWS = {"t09_front": ((1.8, 9.5, 1.7), (0.0, 1.0, 1.0)), "t09_back": ((-1.8, -41.5, 1.7), (0.0, -33.0, 1.0)),
         "t10_front": ((10.8, 9.5, 1.7), (9.015, 1.0, 1.0)), "t10_back": ((7.2, -41.5, 1.7), (9.015, -33.0, 1.0)),
         "t09_inside_to_back": ((0.0, -27.0, 1.3), (0.0, -40.0, 0.9))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        for p in (R180, R180.parent / "physical_ai_runtime", R180 / "tomato_realism_2026-10-28"):
            sys.path.insert(0, str(p))
        from gated_render_sensor_env import GatedRenderSensorEnv
        from sim.scene_prep import prepare_farm_run
        import omni.replicator.core as rep, numpy as np
        from pxr import Gf, UsdGeom
        from PIL import Image
        env = GatedRenderSensorEnv(app, stage_path=STAGE, aisle_half_width=0.38, body_half_length=0.375, plant_cameras=False)
        prepare_farm_run(env.stage, res="2k")
        env.reset(options={"obstacle_enabled": False, "task_mode": "navigation", "max_steps": None,
                           "start_pose": {"x": 2.0, "y": 4.9, "yaw_deg": 0.0}})
        for name, (eye, tgt) in VIEWS.items():
            cam = UsdGeom.Camera.Define(env.stage, f"/DoorCam_{name}"); cam.CreateFocalLengthAttr(20.0)
            cam.CreateHorizontalApertureAttr(36.0); cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500))
            UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(
                Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*tgt), Gf.Vec3d(0, 0, 1)).GetInverse())
            rp = rep.create.render_product(f"/DoorCam_{name}", (1600, 900))
            ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])
            for _ in range(3):
                rep.orchestrator.step(delta_time=0.0, pause_timeline=False, rt_subframes=1)
            Image.fromarray(np.asarray(ann.get_data())[..., :3]).save(a.out / f"{name}.jpg", quality=92)
            ann.detach(); rp.destroy()
            print("saved", name, flush=True)
    except Exception:
        import traceback; print("ERROR", traceback.format_exc(), flush=True)
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
