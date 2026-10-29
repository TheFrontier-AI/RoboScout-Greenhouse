"""Three aerial stills of the current farm twin (Isaac Sim, PathTracing) for the gallery: whole farm, the greenhouse
block with the yard, and a low view along the tunnels from the dock. Session-layer overrides only; the scene files are
not modified. Camera poses come from the stage's bounding box, so they survive a scene update.

  <isaac python.sh> scripts/render_aerials.py --out <dir> [--spp 64] [--width 1800 --height 1200] [--test]"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "sim/stages/R181S_permanent_review2.usda"      # sublayers portable_scene_review2/scene.usdc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--width", type=int, default=1800); ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--stage", type=Path, default=STAGE); ap.add_argument("--test", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of shot names")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if a.test:
        a.spp, a.width, a.height = 8, 900, 600
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": a.width, "height": a.height})
    import carb, omni.usd, omni.replicator.core as rep
    from pxr import Gf, Sdf, Usd, UsdGeom
    from PIL import Image
    import numpy as np
    st = carb.settings.get_settings()
    omni.usd.get_context().open_stage(str(a.stage))
    for _ in range(60):
        app.update()
    stage = omni.usd.get_context().get_stage()
    scene_layer = next((l for l in stage.GetLayerStack() if l.identifier.endswith("scene.usdc")), None)
    if scene_layer is not None:
        for k, v in dict(scene_layer.customLayerData.get("renderSettings", {})).items():   # Astra's PT profile
            st.set("/" + k.replace(":", "/"), v)
    st.set("/rtx/rendermode", "PathTracing"); st.set("/rtx/pathtracing/spp", a.spp); st.set("/rtx/pathtracing/totalSpp", a.spp)
    st.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    stage.SetEditTarget(stage.GetSessionLayer())
    for p in ("/TestFixtures",):
        prim = stage.GetPrimAtPath(p)
        if prim:
            for o in prim.GetChildren():
                UsdGeom.Imageable(o).MakeInvisible()
    # world extent of the farm (ignore the sky dome / ground plane if they are huge: use the /Farm prim when present)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    root = stage.GetPrimAtPath("/Farm") if stage.GetPrimAtPath("/Farm") else stage.GetPseudoRoot()
    bb = cache.ComputeWorldBound(root).ComputeAlignedRange()
    lo, hi = bb.GetMin(), bb.GetMax(); c = (lo + hi) / 2.0; size = hi - lo
    print(f"[aerials] farm bbox min={tuple(round(v, 1) for v in lo)} max={tuple(round(v, 1) for v in hi)}", flush=True)
    span = max(size[0], size[1])
    dock = Gf.Vec3d(2.0, 4.9, 0.0)                                 # robot dock in scene metres (run_farm.py DOCK)
    tunnels = Gf.Vec3d(4.5, -16.0, 0.0)                            # between tunnel-09 (x=0) and tunnel-10 (x=9), mid-aisle
    shots = {
        # name: (camera position, look-at, focal length mm)
        "whole_farm": (Gf.Vec3d(c[0] - 0.55 * span, c[1] - 0.75 * span, 0.55 * span), Gf.Vec3d(c[0], c[1], 0.0), 28.0),
        "greenhouses_yard": (Gf.Vec3d(tunnels[0] + 55.0, tunnels[1] - 35.0, 38.0), Gf.Vec3d(tunnels[0], tunnels[1] + 4.0, 0.0), 32.0),
        "dock_low": (Gf.Vec3d(dock[0] + 14.0, dock[1] + 18.0, 5.0), Gf.Vec3d(tunnels[0] - 1.0, tunnels[1] + 6.0, 1.5), 40.0),
    }
    if a.only:
        shots = {k: v for k, v in shots.items() if k in a.only.split(",")}
    cam = UsdGeom.Camera.Define(stage, "/AerialCam"); cam.CreateFocalLengthAttr(28.0)
    cam.CreateHorizontalApertureAttr(36.0); cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 5000.0))
    cop = UsdGeom.Xformable(cam.GetPrim()).AddTransformOp()
    rp = rep.create.render_product("/AerialCam", (a.width, a.height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach([rp])
    meta = {}
    for name, (pos, look, fl) in shots.items():
        cam.GetFocalLengthAttr().Set(float(fl))
        cop.Set(Gf.Matrix4d().SetLookAt(pos, look, Gf.Vec3d(0, 0, 1)).GetInverse())
        for _ in range(4):                                         # let the path tracer accumulate totalSpp
            rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
        im = np.asarray(rgb.get_data())[:, :, :3]
        Image.fromarray(im).save(a.out / f"{name}.png")
        meta[name] = {"camera": [round(v, 2) for v in pos], "look_at": [round(v, 2) for v in look], "focal_mm": fl,
                      "spp": a.spp, "size": [a.width, a.height]}
        print(f"[aerials] wrote {name}.png", flush=True)
    (a.out / "aerials.json").write_text(json.dumps({"stage": str(a.stage), "bbox": [list(lo), list(hi)], "shots": meta}, indent=1))
    app.close()


if __name__ == "__main__":
    main()
