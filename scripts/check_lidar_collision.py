"""Before/after check for sim/scene_collision.py: planar C1 scans (PhysX scene queries) at given robot poses, with and
without the runtime crop/tunnel-10 colliders. Writes <out>/scans_<mode>.json. Session-layer only. No training.

  python.sh scripts/check_lidar_collision.py --out <dir> --mode before|after --poses x,y,yaw [x,y,yaw ...]"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.render_cinematic import SCENE                       # same scene as the demo renders

LIDAR_LOCAL, N_BEAMS, RMAX = (0.0, -0.34, 0.512), 500, 12.0     # C1 pose on the robot (lidar_runtime.POSITION)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--mode", choices=("before", "after"), required=True)
    ap.add_argument("--poses", nargs="+", required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    import carb, omni.usd, omni.physx
    omni.usd.get_context().open_stage(str(SCENE))
    for _ in range(30):
        app.update()
    stage = omni.usd.get_context().get_stage(); stage.SetEditTarget(stage.GetSessionLayer())
    info = {}
    if a.mode == "after":
        from sim.scene_collision import add_crop_collision
        t0 = time.time(); info = add_crop_collision(stage); info["author_s"] = round(time.time() - t0, 1)
    t0 = time.time(); omni.physx.get_physx_interface().force_load_physics_from_usd(); info["physics_load_s"] = round(time.time() - t0, 1)
    sq = omni.physx.get_physx_scene_query_interface()
    scans = []
    for pose in a.poses:
        x, y, yaw = (float(v) for v in pose.split(","))
        c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        ox, oy = x + c * LIDAR_LOCAL[0] - s_ * LIDAR_LOCAL[1], y + s_ * LIDAR_LOCAL[0] + c * LIDAR_LOCAL[1]
        oz = 0.034 + LIDAR_LOCAL[2]; rr = []; hit_paths = {}
        for k in range(N_BEAMS):
            world = math.radians(yaw) - math.pi / 2 + (-math.pi + 2 * math.pi * k / N_BEAMS)
            best = [None, None]
            def hit(h):
                if not str(h.collision).startswith("/Robot") and (best[0] is None or h.distance < best[0]):
                    best[0], best[1] = h.distance, str(h.collision)
                return True
            sq.raycast_all(carb.Float3(ox, oy, oz), carb.Float3(math.cos(world), math.sin(world), 0.0), RMAX, hit)
            ok = best[0] is not None and best[0] >= 0.05
            rr.append(round(best[0], 3) if ok else None)
            if ok:
                g = best[1].split("/")[4] if best[1].count("/") >= 4 else best[1]
                hit_paths[g] = hit_paths.get(g, 0) + 1
        scans.append({"pose": [x, y, yaw], "origin": [ox, oy], "yaw_deg": yaw, "ranges": rr,
                      "hits": sum(r is not None for r in rr), "hit_groups": dict(sorted(hit_paths.items(), key=lambda kv: -kv[1])[:12])})
        print(pose, "hits", scans[-1]["hits"], flush=True)
    (a.out / f"scans_{a.mode}.json").write_text(json.dumps({"info": info, "scans": scans}, indent=1))
    print("info", info, flush=True)
    app.close()


if __name__ == "__main__":
    main()
