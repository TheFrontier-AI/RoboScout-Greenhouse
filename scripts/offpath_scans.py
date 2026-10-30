"""Collect 2-D LiDAR scans at given robot poses in the farm twin (Isaac Sim, headless, no rendering): the robot is placed
kinematically at each pose and the C1 LiDAR is simulated with PhysX scene queries, exactly as run_farm records it.
Used for (a) a "look-around" map of the farm (poses along the taught path with several headings) and (b) off-path
test poses (robot put down beside the path). Output rows use the recorder convention of steps.jsonl: `lidar` is 500
ranges indexed by raw azimuth -pi..pi with the robot's forward at -pi/2; `heading_deg` is the scene heading (0 = +X).

  <isaac python.sh> scripts/offpath_scans.py --poses poses.json --out scans.jsonl [--stage <usda>]"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "sim/stages/R181S_permanent_review2.usda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage", type=Path, default=STAGE)
    a = ap.parse_args()
    poses = json.load(open(a.poses))
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": 64, "height": 64})
    import carb, omni.usd, omni.physx
    from pxr import Gf, UsdGeom
    omni.usd.get_context().open_stage(str(a.stage))
    for _ in range(60):
        app.update()
    stage = omni.usd.get_context().get_stage(); stage.SetEditTarget(stage.GetSessionLayer())
    for o in [o for o in stage.GetPrimAtPath("/TestFixtures").GetChildren()] if stage.GetPrimAtPath("/TestFixtures") else []:
        UsdGeom.Imageable(o).MakeInvisible()
    robot = UsdGeom.Xformable(stage.GetPrimAtPath("/Robot"))
    t_op = robot.GetOrderedXformOps()[0] if robot.GetOrderedXformOps() else robot.AddTranslateOp()
    r_op = robot.AddRotateZOp(opSuffix="scan")
    omni.physx.get_physx_interface().force_load_physics_from_usd()
    for _ in range(10):
        app.update()
    q = omni.physx.get_physx_scene_query_interface()
    LIDAR_LOCAL, N, RMAX = (0.0, -0.34, 0.512), 500, 12.0           # C1 pose on the robot (lidar_runtime.POSITION)

    def scan(x, y, heading_deg):
        yaw = heading_deg + 90.0                                     # robot forward is -Y at yaw 0
        c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        ox, oy = x + c * LIDAR_LOCAL[0] - s_ * LIDAR_LOCAL[1], y + s_ * LIDAR_LOCAL[0] + c * LIDAR_LOCAL[1]
        oz = 0.034 + LIDAR_LOCAL[2]; out = []
        for k in range(N):
            a_k = -math.pi + 2 * math.pi * k / N                        # raw azimuth, forward at -pi/2
            w = math.radians(heading_deg) + a_k + math.pi / 2           # world direction of this beam
            d = (math.cos(w), math.sin(w), 0.0); best = [None]
            def hit(h):
                if not str(h.collision).startswith("/Robot") and (best[0] is None or h.distance < best[0]):
                    best[0] = h.distance
                return True
            q.raycast_all(carb.Float3(ox, oy, oz), carb.Float3(*d), RMAX, hit)
            out.append(round(best[0], 3) if best[0] is not None and best[0] >= 0.05 else None)
        return out

    t0 = time.time()
    with open(a.out, "w") as f:
        for i, p in enumerate(poses):
            t_op.Set(Gf.Vec3d(p["x"], p["y"], 0.034)); r_op.Set(float(p["heading_deg"] + 90.0))
            app.update()
            row = {"i": i, "tag": p.get("tag"), "x": p["x"], "y": p["y"], "heading_deg": p["heading_deg"], "lidar": scan(p["x"], p["y"], p["heading_deg"])}
            f.write(json.dumps(row) + "\n")
            if i % 100 == 0:
                print(f"[scans] {i}/{len(poses)} {time.time() - t0:.0f}s", flush=True)
    print(f"[scans] done {len(poses)} scans in {time.time() - t0:.0f}s -> {a.out}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
