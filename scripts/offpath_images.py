"""Render the robot's front camera at given poses in the farm twin (Isaac Sim, PathTracing): the image the robot would
see when put down there. Pairs with scripts/offpath_scans.py (same pose list) for the camera tie-break test.

  <isaac python.sh> scripts/offpath_images.py --poses poses.json --out <dir> [--tag offpath] [--spp 16]"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "sim/stages/R181S_permanent_review2.usda"
FRONT = "/Robot/Physics_Base/Sensors/front_rgb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="offpath"); ap.add_argument("--spp", type=int, default=16)
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--stage", type=Path, default=STAGE)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    poses = [(i, p) for i, p in enumerate(json.load(open(a.poses))) if p.get("tag") == a.tag]
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": a.width, "height": a.height})
    import carb, omni.usd, omni.replicator.core as rep
    from pxr import Gf, Sdf, UsdGeom
    from PIL import Image
    import numpy as np
    st = carb.settings.get_settings()
    omni.usd.get_context().open_stage(str(a.stage))
    for _ in range(60):
        app.update()
    stage = omni.usd.get_context().get_stage()
    scene_layer = next((l for l in stage.GetLayerStack() if l.identifier.endswith("scene.usdc")), None)
    if scene_layer is not None:
        for k, v in dict(scene_layer.customLayerData.get("renderSettings", {})).items():
            st.set("/" + k.replace(":", "/"), v)
    st.set("/rtx/rendermode", "PathTracing"); st.set("/rtx/pathtracing/spp", a.spp); st.set("/rtx/pathtracing/totalSpp", a.spp)
    st.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    stage.SetEditTarget(stage.GetSessionLayer())
    for o in [o for o in stage.GetPrimAtPath("/TestFixtures").GetChildren()] if stage.GetPrimAtPath("/TestFixtures") else []:
        UsdGeom.Imageable(o).MakeInvisible()
    robot = UsdGeom.Xformable(stage.GetPrimAtPath("/Robot"))
    t_op = robot.GetOrderedXformOps()[0] if robot.GetOrderedXformOps() else robot.AddTranslateOp()
    r_op = robot.AddRotateZOp(opSuffix="img")
    rp = rep.create.render_product(FRONT, (a.width, a.height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach([rp])
    t0 = time.time()
    for i, p in poses:
        t_op.Set(Gf.Vec3d(p["x"], p["y"], 0.034)); r_op.Set(float(p["heading_deg"] + 90.0))
        for _ in range(4):
            rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
        Image.fromarray(np.asarray(rgb.get_data())[:, :, :3]).save(a.out / f"{i:05d}.jpg", quality=92)
        print(f"[img] {i} {time.time() - t0:.0f}s", flush=True)
    print(f"[img] done {len(poses)} images in {time.time() - t0:.0f}s", flush=True)
    app.close()


if __name__ == "__main__":
    main()
