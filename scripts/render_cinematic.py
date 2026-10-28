"""Cinematic one-shot of R1 in the farm twin (Isaac Sim, PathTracing): survey in tunnel-09, reverse out, follow the
site-map route along the road, enter tunnel-10. The robot is posed KINEMATICALLY along the planned route (outdoor
transit is not physics-simulated); the camera flies a smooth continuous path. Session-layer overrides only — the
scene files are not modified. Frames → <out>/frame_00000.png. No training.

  /isaac-sim/python.sh scripts/render_cinematic.py --out /runs/cine --seconds 40 --fps 24 [--spp 16] [--test 3]"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENE = Path("/home/passin/Documents/Blender/TheFarm/robot_design/r1_v3/diy_r50/physical_ai_r180/"
             "tomato_realism_2026-10-28/C2_R180_Interior_PT_Sensor_Fixture.usda")
ORIGIN = (195.065185546875, 153.0)                     # /Farm is moved by -ORIGIN in the twin


def site_route() -> list[tuple[float, float]]:
    """Scene-frame waypoints: tunnel-09 aisle-02 (metre 4 → 11 → reverse to apron), road, tunnel-10 aisle (metre 6)."""
    sys.path.insert(0, str(ROOT))
    from r1_nav.site_map import SiteMap
    s = SiteMap.load(ROOT / "data/site_map_v18_draft.json")
    r = s.route(s.door_of("tunnel-09"), s.door_of("tunnel-10"))
    node = {n.id: n for n in (s.nodes.values() if isinstance(s.nodes, dict) else s.nodes)}
    xy = [(node[n].xy[0] - ORIGIN[0], node[n].xy[1] - ORIGIN[1]) for n in r.nodes]
    door9, door10 = xy[0], xy[-1]
    return xy, door9, door10


def smooth(t):                                         # ease in/out 0..1
    return t * t * (3 - 2 * t)


def build_timeline(seconds: float):
    """Robot keyframes (t, x, y, yaw_deg) and camera keyframes (t, pos, look_at) in the scene frame (Z up)."""
    xy, d9, d10 = site_route()
    x9, x10 = d9[0], d10[0]
    in9, deep9 = (x9, d9[1] - 6.0), (x9, d9[1] - 13.0)             # inside tunnel-09 aisle-02
    in10 = (x10, d10[1] - 8.0)
    T = seconds / 40.0
    robot = [  # forward = -Y at yaw 0 (twin convention); reverse keeps yaw
        (0 * T, *in9, 0), (9 * T, *deep9, 0),                       # survey drive-in
        (17 * T, *xy[1], 0),                                        # reverse out to the apron
        (19 * T, *xy[1], 180), (21 * T, *xy[2], 180),               # turn to face +Y, out to the road
        (22.5 * T, *xy[2], 90), (28 * T, *xy[3], 90),               # turn, drive along the road (+X)
        (29.5 * T, *xy[3], 0), (32 * T, *xy[4], 0),                 # turn, back down to tunnel-10's apron
        (34 * T, *d10, 0), (40 * T, *in10, 0)]                      # enter tunnel-10
    mid = ((x9 + x10) / 2, d9[1])
    cam = [  # (t, (x,y,z), look target or 'robot')
        (0 * T, (x9 + 0.15, in9[1] + 2.4, 1.35), "robot"),          # low chase inside the aisle
        (8 * T, (x9 + 0.25, deep9[1] + 3.0, 1.6), "robot"),
        (15 * T, (x9 - 2.0, d9[1] + 6.5, 4.0), "robot"),            # robot reverses out under us
        (22 * T, (mid[0], d9[1] + 16.0, 11.0), (mid[0], d9[1] - 4.0, 0.5)),   # wide aerial: both greenhouses
        (28 * T, (x10 + 4.0, d10[1] + 9.0, 5.0), "robot"),          # chase along the road
        (35 * T, (x10 + 0.2, d10[1] + 3.0, 1.7), "robot"),          # follow into tunnel-10
        (40 * T, (mid[0], d10[1] + 20.0, 16.0), (mid[0], d10[1] - 6.0, 0.0))]  # pull out
    return robot, cam


def interp(keys, t):
    for (t0, *a), (t1, *b) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            u = smooth((t - t0) / max(t1 - t0, 1e-9))
            return [p + (q - p) * u if not isinstance(p, (tuple, str)) else (p, q, u) for p, q in zip(a, b)]
    return list(keys[-1][1:])


def robot_pose(keys, t):
    for (t0, x0, y0, a0), (t1, x1, y1, a1) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            u = smooth((t - t0) / max(t1 - t0, 1e-9))
            da = (a1 - a0 + 180) % 360 - 180
            return x0 + (x1 - x0) * u, y0 + (y1 - y0) * u, a0 + da * u
    return keys[-1][1:]


def cam_pose(keys, t, robot_xy):
    def tgt(v):
        return (robot_xy[0], robot_xy[1], 0.75) if v == "robot" else v
    # Catmull-Rom over positions for a continuous, smooth flight; targets blended linearly
    times = [k[0] for k in keys]
    i = max(0, min(len(keys) - 2, next((j for j in range(len(keys) - 1) if times[j] <= t <= times[j + 1]), len(keys) - 2)))
    p0, p1, p2, p3 = (keys[max(i - 1, 0)][1], keys[i][1], keys[i + 1][1], keys[min(i + 2, len(keys) - 1)][1])
    u = smooth((t - times[i]) / max(times[i + 1] - times[i], 1e-9))
    cr = lambda a, b, c, d: 0.5 * (2 * b + (-a + c) * u + (2 * a - 5 * b + 4 * c - d) * u * u + (-a + 3 * b - 3 * c + d) * u ** 3)
    pos = tuple(cr(p0[k], p1[k], p2[k], p3[k]) for k in range(3))
    a, b = tgt(keys[i][2]), tgt(keys[i + 1][2])
    look = tuple(a[k] + (b[k] - a[k]) * u for k in range(3))
    return pos, look


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--fps", type=int, default=24); ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080); ap.add_argument("--spp", type=int, default=16)
    ap.add_argument("--test", type=int, default=0, help="render only N evenly spaced frames")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--no-sensors", action="store_true", help="chase camera only (no side/front cameras, no LiDAR)")
    ap.add_argument("--side-res", default="1080x1920", help="side camera WxH (real Rapoo C260 portrait)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    robot_keys, cam_keys = build_timeline(a.seconds)
    n = int(a.seconds * a.fps)
    frames = [round(i * (n - 1) / max(a.test - 1, 1)) for i in range(a.test)] if a.test else list(range(a.start, n))
    (a.out / "timeline.json").write_text(json.dumps({"robot": robot_keys, "camera": [[k[0], k[1], k[2]] for k in cam_keys],
                                                     "fps": a.fps, "seconds": a.seconds}, indent=1))

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": a.width, "height": a.height})
    import carb, omni.usd, omni.replicator.core as rep
    from pxr import Gf, Sdf, UsdGeom
    from PIL import Image
    import numpy as np
    layer = Sdf.Layer.FindOrOpen(str(SCENE)); st = carb.settings.get_settings()
    omni.usd.get_context().open_stage(str(SCENE))
    for _ in range(60):
        app.update()
    for k, v in dict(layer.customLayerData.get("renderSettings", {})).items():   # Astra's profile (PT ISO400)
        st.set("/" + k.replace(":", "/"), v)
    st.set("/rtx/rendermode", "PathTracing"); st.set("/rtx/pathtracing/spp", a.spp); st.set("/rtx/pathtracing/totalSpp", a.spp)
    st.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())                                 # runtime-only overrides
    robot = UsdGeom.Xformable(stage.GetPrimAtPath("/Robot"))
    t_op = robot.GetOrderedXformOps()[0] if robot.GetOrderedXformOps() else robot.AddTranslateOp()
    r_op = robot.AddRotateZOp(opSuffix="cine")
    for o in [o for o in stage.GetPrimAtPath("/TestFixtures").GetChildren()] if stage.GetPrimAtPath("/TestFixtures") else []:
        UsdGeom.Imageable(o).MakeInvisible()                                     # hide the obstacle test box
    cam = UsdGeom.Camera.Define(stage, "/CineCam"); cam.CreateFocalLengthAttr(24.0)
    cam.CreateHorizontalApertureAttr(36.0); cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 2000.0))
    cx = UsdGeom.Xformable(cam.GetPrim()); cop = cx.AddTransformOp()
    rp = rep.create.render_product("/CineCam", (a.width, a.height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach([rp])
    sensors = {}                                      # name -> annotator; robot-left uses authored plant_right_* (Astra)
    if not a.no_sensors:
        sw, sh = (int(v) for v in a.side_res.lower().split("x"))
        for name, path, res in (("side_left", "/Robot/Physics_Base/Sensors/plant_right_mid", (sw, sh)),
                                ("side_right", "/Robot/Physics_Base/Sensors/plant_left_mid", (sw, sh)),
                                ("front", "/Robot/Physics_Base/Sensors/front_rgb", (640, 360))):
            if stage.GetPrimAtPath(path):
                ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rep.create.render_product(path, res)])
                sensors[name] = ann; (a.out / name).mkdir(exist_ok=True)
    (a.out / "chase").mkdir(exist_ok=True)
    lidar = None
    if not a.no_sensors:
        try:
            import omni.physx
            omni.physx.get_physx_interface().force_load_physics_from_usd()
            lidar = omni.physx.get_physx_scene_query_interface()
        except Exception as e:                                         # keep rendering even without scans
            print("LIDAR unavailable:", e, flush=True)
    LIDAR_LOCAL, N_BEAMS, RMAX = (0.0, -0.34, 0.512), 500, 12.0      # C1 pose on the robot (lidar_runtime.POSITION)

    def scan(x, y, yaw):
        """Planar 360° scan from the C1 pose via PhysX scene queries; hits on the robot itself are skipped."""
        c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        ox, oy = x + c * LIDAR_LOCAL[0] - s_ * LIDAR_LOCAL[1], y + s_ * LIDAR_LOCAL[0] + c * LIDAR_LOCAL[1]
        oz = 0.034 + LIDAR_LOCAL[2]; out = []
        for k in range(N_BEAMS):
            ang = -math.pi + 2 * math.pi * k / N_BEAMS                 # robot frame, 0 = robot forward (-Y at yaw 0)
            world = math.radians(yaw) - math.pi / 2 + ang
            d = (math.cos(world), math.sin(world), 0.0); best = [None]
            def hit(h):
                if not str(h.collision).startswith("/Robot") and (best[0] is None or h.distance < best[0]):
                    best[0] = h.distance
                return True
            lidar.raycast_all(carb.Float3(ox, oy, oz), carb.Float3(*d), RMAX, hit)
            out.append(round(best[0], 3) if best[0] is not None and best[0] >= 0.05 else None)
        return {"origin": [round(ox, 3), round(oy, 3)], "yaw_deg": round(yaw, 2), "ranges": out}
    for f in frames:
        t = f / a.fps
        x, y, yaw = robot_pose(robot_keys, t)
        t_op.Set(Gf.Vec3d(x, y, 0.034)); r_op.Set(float(yaw))
        pos, look = cam_pose(cam_keys, t, (x, y))
        m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*pos), Gf.Vec3d(*look), Gf.Vec3d(0, 0, 1)).GetInverse()
        cop.Set(m)
        for _ in range(2):
            rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
        Image.fromarray(np.asarray(rgb.get_data())[:, :, :3]).save(a.out / "chase" / f"{f:05d}.png")
        for name, ann in sensors.items():
            Image.fromarray(np.asarray(ann.get_data())[:, :, :3]).save(a.out / name / f"{f:05d}.png")
        rec = {"frame": f, "t": round(t, 4), "robot": [round(x, 4), round(y, 4), round(yaw, 3)], "camera": [round(v, 3) for v in pos]}
        if lidar is not None:
            rec["lidar"] = scan(x, y, yaw)
        with open(a.out / "frames.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"frame {f}/{n} t={t:.2f} robot=({x:.2f},{y:.2f},{yaw:.0f}) cam={tuple(round(v, 1) for v in pos)}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
