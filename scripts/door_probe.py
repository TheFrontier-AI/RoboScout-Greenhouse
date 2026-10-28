"""Doorway check for the full-farm run: what does PhysX see at each tunnel opening, and can the robot drive through?
  - horizontal rays across each doorway at low heights (sills, ramps, thresholds a 2-D LiDAR would miss),
  - a vertical height profile along each aisle centre line through the door,
  - the robot placed 1 m outside tunnel-10's back door, driven straight in for 6 s (ground-truth motion reported).

  /isaac-sim/python.sh scripts/door_probe.py --out /runs/rec/door_probe.json"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_farm import R180, STAGE, HOUSES, BACK_Y, log        # noqa: E402

FRONT_Y = 3.67


def probe(cx: float, y_door: float, outward: float) -> dict:
    import carb, omni.physx
    sq = omni.physx.get_physx_scene_query_interface()
    out = {"horizontal": [], "profile": []}
    for z in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.55):
        for dx in (-0.3, 0.0, 0.3):
            o = carb.Float3(cx + dx, y_door + outward * 1.0, z)
            h = sq.raycast_closest(o, carb.Float3(0.0, -outward, 0.0), 2.0)
            if h["hit"]:
                out["horizontal"].append({"z": z, "dx": dx, "dist_from_1m_out": round(h["distance"], 3),
                                          "y": round(h["position"][1], 3), "prim": str(h["collision"])})
    for k in range(61):
        y = y_door + outward * (1.5 - 0.05 * k)
        h = sq.raycast_closest(carb.Float3(cx, y, 1.0), carb.Float3(0.0, 0.0, -1.0), 2.0)
        out["profile"].append([round(y, 3), round(h["position"][2], 3) if h["hit"] else None,
                               str(h["collision"]).split("/")[-1] if h["hit"] else None])
    return out


def colliders_in(stage, xr, yr, zr=(0.0, 1.2)) -> list:
    """Every enabled collider prim whose world bounding box overlaps the given box (the doorway the robot drives through)."""
    from pxr import Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy,
                                                       UsdGeom.Tokens.guide])
    out = []
    for p in stage.Traverse():
        if not p.HasAPI(UsdPhysics.CollisionAPI) or str(p.GetPath()).startswith("/Robot"):
            continue
        en = UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
        if en is False:
            continue
        r = cache.ComputeWorldBound(p).ComputeAlignedRange()
        if r.IsEmpty():
            continue
        lo, hi = r.GetMin(), r.GetMax()
        if lo[0] < xr[1] and hi[0] > xr[0] and lo[1] < yr[1] and hi[1] > yr[0] and lo[2] < zr[1] and hi[2] > zr[0]:
            out.append([str(p.GetPath()), [round(v, 3) for v in lo], [round(v, 3) for v in hi]])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--drive-s", type=float, default=6.0); ap.add_argument("--wheel", type=float, default=1.5)
    a = ap.parse_args()
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    res = {}
    try:
        for p in (R180, R180.parent / "physical_ai_runtime", R180 / "tomato_realism_2026-10-28"):
            sys.path.insert(0, str(p))
        from gated_render_sensor_env import GatedRenderSensorEnv
        from sim.fast_env import YardEvaluator, make_fast_env
        from sim.scene_prep import prepare_farm_run
        Env = make_fast_env(GatedRenderSensorEnv)
        env = Env(app, stage_path=STAGE, aisle_half_width=0.38, body_half_length=0.375, plant_cameras=False)
        prepare_farm_run(env.stage, tunnels=tuple(x[0][7:9] for x in HOUSES), res="2k", log=log)
        cx10 = HOUSES[1][1]
        res["colliders_in_t10_back_door"] = colliders_in(env.stage, (cx10 - 0.4, cx10 + 0.4), (BACK_Y - 0.3, BACK_Y + 0.5))
        res["colliders_in_t09_back_door"] = colliders_in(env.stage, (-0.4, 0.4), (BACK_Y - 0.3, BACK_Y + 0.5))
        log(f"t10 back door colliders {res['colliders_in_t10_back_door']}")
        obs, info = env.reset(options={"obstacle_enabled": False, "task_mode": "navigation", "max_steps": None,
                                       "start_pose": {"x": cx10, "y": BACK_Y - 1.0, "yaw_deg": 180.0}})
        env.set_evaluator(YardEvaluator())
        env.want = {"side": False, "front": False, "extra": False}
        for name, x in HOUSES:
            res[name] = {"back": probe(x, BACK_Y, -1.0), "front": probe(x, FRONT_Y, 1.0)}
        track = []
        for i in range(int(a.drive_s / 0.1)):
            obs, _, term, _, info = env.step([a.wheel, a.wheel])
            if i % 5 == 0:
                track.append([round(v, 3) for v in (info.get("scene_pose") or [])[:4]])
            if term:
                track.append(["terminated", str(info.get("termination"))]); break
        res["drive_t10_back"] = track
        log(f"drive track {track}")
    except Exception:
        import traceback
        res["error"] = traceback.format_exc(); log("ERROR " + res["error"])
    finally:
        a.out.write_text(json.dumps(res, indent=1)); log(f"wrote {a.out}")
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
