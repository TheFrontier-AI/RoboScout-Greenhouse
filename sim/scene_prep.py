"""Runtime scene preparation for the full-farm run (session layer only, scene files untouched):
  - open both ends of each tunnel: hide the door curtain (Hero_Gate_Drape), delete the faces that close the doorway
    in the end walls and any door threshold below 0.3 m, so the robot can drive through a tunnel and out the back;
  - an invisible ground collider for the yard/road between tunnels (the farm terrain has no collision in this stage);
  - collision for tunnel structures that lack it (tunnel-10) and the Tomato-Assets-V1 crop swap (sim.crop_swap).
Every change is reported so the run log says exactly what was altered for simulation."""
from __future__ import annotations
import numpy as np

ROOT = "/Farm/SiteV18_Root"
DOOR_HALF_W = 0.65                 # doorway half-width around the aisle centre line (door frame posts at ±0.59)
DOOR_TOP_Z = 2.05


def _faces(mesh):
    cnt = np.array(mesh.GetFaceVertexCountsAttr().Get()); idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    return cnt, idx, np.r_[0, np.cumsum(cnt)[:-1]]


def _drop_faces(prim, keep_fn) -> int:
    """Remove faces whose world-space centroid fails keep_fn(centroids) -> bool mask. Returns faces removed."""
    from pxr import Usd, UsdGeom, Vt
    mesh = UsdGeom.Mesh(prim)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=float)
    cnt, idx, starts = _faces(mesh)
    X = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    w = (np.c_[pts, np.ones(len(pts))] @ X.T)[:, :3]
    cen = np.array([w[idx[s:s + c]].mean(0) for s, c in zip(starts, cnt)])
    keep = keep_fn(cen)
    if keep.all():
        return 0
    new_idx = np.concatenate([idx[s:s + c] for s, c, k in zip(starts, cnt, keep) if k])
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([int(c) for c in cnt[keep]]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([int(i) for i in new_idx]))
    for sub in UsdGeom.Subset.GetAllGeomSubsets(mesh):          # face subsets would point past the new face list
        sub.GetPrim().SetActive(False)
    return int((~keep).sum())


def open_tunnel_ends(stage, tunnels=("09", "10")) -> dict:
    from pxr import Usd, UsdGeom
    out = {}
    for t in tunnels:
        anchor = stage.GetPrimAtPath(f"{ROOT}/SiteV18_Anchor_tunnel_{t}")
        if not anchor:
            continue
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        box = cache.ComputeWorldBound(anchor).ComputeAlignedRange()
        cx = (box.GetMin()[0] + box.GetMax()[0]) / 2
        y_front, y_back = box.GetMax()[1], box.GetMin()[1]
        rep = {"centre_x": round(cx, 3), "front_y": round(y_front, 3), "back_y": round(y_back, 3), "hidden": [], "faces_removed": {}}
        for grp in anchor.GetChildren():
            name = grp.GetName()
            if "Gate_Drape" in name:
                UsdGeom.Imageable(grp).MakeInvisible(); rep["hidden"].append(name)
                continue
            if not any(k in name for k in ("_Walls", "Entrance")):
                continue
            for p in Usd.PrimRange(grp):
                if not p.IsA(UsdGeom.Mesh):
                    continue
                walls = "_Walls" in name

                def keep(c, walls=walls):
                    in_door = (np.abs(c[:, 0] - cx) < DOOR_HALF_W) & (c[:, 2] < DOOR_TOP_Z) & \
                              ((np.abs(c[:, 1] - y_front) < 0.4) | (np.abs(c[:, 1] - y_back) < 0.4))
                    if not walls:                                   # door frame: keep posts/lintel, drop the threshold
                        in_door &= c[:, 2] < 0.3
                    return ~in_door
                n = _drop_faces(p, keep)
                if n:
                    rep["faces_removed"][p.GetName()] = n
        rep["colliders"] = _wall_colliders(stage, anchor, t, cx, y_front, y_back)
        out[f"tunnel-{t}"] = rep
    return out


def _wall_colliders(stage, anchor, t, cx, y_front, y_back, half_w=3.26, door_half=0.62, h=2.0, th=0.04) -> int:
    """PhysX cooked the wall/door-frame meshes before the doorway faces were removed, so their collision would stay
    closed. Disable those mesh colliders (a live attribute) and stand in simple boxes: side walls + end walls with the
    doorway gap. The LiDAR sees the boxes exactly where the walls are."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    for grp in anchor.GetChildren():
        if any(k in grp.GetName() for k in ("_Walls", "Entrance")):
            for p in Usd.PrimRange(grp):
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr().Set(False)
    boxes = [(cx - half_w, (y_front + y_back) / 2, th, y_front - y_back), (cx + half_w, (y_front + y_back) / 2, th, y_front - y_back)]
    seg = half_w - door_half
    for y in (y_front, y_back):
        boxes += [(cx - door_half - seg / 2, y, seg, th), (cx + door_half + seg / 2, y, seg, th)]
    for i, (x, y, sx, sy) in enumerate(boxes):
        cube = UsdGeom.Cube.Define(stage, f"/SimHelpers/Walls_T{t}_{i}")
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, h / 2)); xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, h))
        UsdGeom.Imageable(cube).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return len(boxes)


def add_yard_ground(stage, centre=(4.5, -16.0), size=(80.0, 90.0), top_z: float = 0.005) -> str:
    """Invisible static box whose top is the yard surface (outside the tunnels the terrain has no collider)."""
    from pxr import Gf, UsdGeom, UsdPhysics
    path = "/SimHelpers/YardGround"
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(cube)
    x.AddTranslateOp().Set(Gf.Vec3d(centre[0], centre[1], top_z - 0.05))
    x.AddScaleOp().Set(Gf.Vec3f(size[0], size[1], 0.1))
    UsdGeom.Imageable(cube).MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return path


def prepare_farm_run(stage, tunnels=("09", "10"), crops: bool = True, res: str = "8k", seed: int = 7, log=print) -> dict:
    from sim.scene_collision import add_crop_collision
    # collision first: open_tunnel_ends then switches off every wall/door-frame mesh collider (tunnel-10's are only
    # created here) and stands in the box walls with the doorway gap (farm_full, 29 Oct: tunnel-10's back door frame
    # was still solid because its collider was added after the switch-off)
    collision = add_crop_collision(stage, tunnels, plants=not crops)
    report = {"open_ends": open_tunnel_ends(stage, tunnels), "yard_ground": add_yard_ground(stage), "collision": collision}
    if crops:
        from sim.crop_swap import swap_crops
        report["crops"] = swap_crops(stage, tunnels, seed=seed, res=res)
    log(f"[scene_prep] open ends {report['open_ends']}")
    log(f"[scene_prep] collision {report['collision']['added']}; crops swapped: {len(report.get('crops', []))}")
    return report
