"""Runtime collision for the farm twin, so PhysX range sensors (the C1 LiDAR) see what the cameras see.

The authored scene gives collision only to tunnel-09's structure (beds, walls, frame, ground...). Tomato plants have
none anywhere, and tunnel-10 has none at all, so LiDAR rays pass straight through the crop and tunnel-10 is invisible.
This adds static triangle-mesh colliders on the *current edit target* (call it with the session layer active):
  - every tunnel listed gets the same structure colliders tunnel-09 has (mirrored by group name),
  - every Hero_Tomato_Row_* plant mesh gets a collider (leaves are alpha-cut cards, so hits land on the whole card:
    slightly denser than real foliage, far closer than "no plants").
The scene files themselves are never modified."""
from __future__ import annotations
import re

ROOT = "/Farm/SiteV18_Root"
PLANT = re.compile(r"_Hero_Tomato_Row_\d+_Plant_\d+$")


def _collide(prim, UsdPhysics) -> int:
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("none")   # static triangle mesh
    return 1


def add_crop_collision(stage, tunnels=("09", "10"), plants=True) -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics
    ref = stage.GetPrimAtPath(f"{ROOT}/SiteV18_Anchor_tunnel_09")
    struct = set()                                            # tunnel-09 group names that already collide
    if ref:
        for p in Usd.PrimRange(ref):
            if p.IsA(UsdGeom.Gprim) and p.HasAPI(UsdPhysics.CollisionAPI):
                g = str(p.GetPath()).split("/")[4]
                struct.add(re.sub(r"^SiteV18_Building_tunnel_09_", "", re.sub(r"_\d{3}$", "", g)))
    counts = {}
    for t in tunnels:
        anchor = stage.GetPrimAtPath(f"{ROOT}/SiteV18_Anchor_tunnel_{t}")
        if not anchor:
            continue
        n_struct = n_plant = 0
        for grp in anchor.GetChildren():
            name = grp.GetName()
            kind = re.sub(rf"^SiteV18_Building_tunnel_{t}_", "", re.sub(r"_\d{3}$", "", name))
            is_plant = plants and PLANT.search(name)
            if not (is_plant or kind in struct):
                continue
            for p in Usd.PrimRange(grp):
                if p.IsA(UsdGeom.Mesh) and not p.HasAPI(UsdPhysics.CollisionAPI):
                    if is_plant:
                        n_plant += _collide(p, UsdPhysics)
                    else:
                        n_struct += _collide(p, UsdPhysics)
        counts[t] = {"structure": n_struct, "plants": n_plant}
    return {"reference_groups": sorted(struct), "added": counts}
