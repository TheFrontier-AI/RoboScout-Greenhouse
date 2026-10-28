"""Replace the farm twin's V18 tomato plants with Tomato-Assets-V1 plants (Astra's Isaac library) at runtime.

The V18 web-model plants render with black fruit in the realism stage and carry no disease. The library has 8 healthy
plants with green/orange/red fruit and 12 synthetic visual anomalies, already baked to USD for Isaac. Every V18 plant
in the chosen tunnels is hidden and a library plant is referenced at its base (random healthy variant + random yaw);
listed anomaly plants are placed at chosen row/plant slots. The swap is authored on the current edit target (use the
session layer): scene files are never modified. Returns a manifest = the ground truth for the run (no boxes drawn).
Synthetic anomalies are generated visual variations, not diagnosed diseases."""
from __future__ import annotations
import random
import re
from pathlib import Path

LIBRARY = Path("/home/passin/Documents/Blender/TheFarm/robot_design/r1_v3/diy_r50/physical_ai_r180/"
               "tomato_realism_2026-10-28")
HEALTHY = [f"Tomato_Healthy_S2026{d}" for d in range(1028, 1036)]
ROOT = "/Farm/SiteV18_Root"
PLANT = re.compile(r"_Hero_Tomato_Row_(\d+)_Plant_(\d+)(?:_\d{3})?$")   # some tunnel-10 plants carry a _001 suffix

# Clearly visible anomalies only (mild ones are near invisible at mast distance), spread over both aisle-02 rows
# (row 2 = robot's right driving in, row 3 = robot's left) and both tunnels. (tunnel, row, plant index, kind)
DEFAULT_ANOMALIES = (
    ("09", 3, 14, "leaf_spots_strong"), ("09", 2, 27, "chlorosis_strong"), ("09", 3, 41, "fruit_lesion_strong"),
    ("09", 2, 58, "wilting_strong"), ("09", 3, 72, "leaf_spots_strong"),
    ("10", 2, 11, "chlorosis_strong"), ("10", 3, 33, "fruit_lesion_strong"), ("10", 2, 49, "leaf_spots_strong"),
    ("10", 3, 66, "defoliation_strong"),
)


def asset_path(name: str, res: str = "8k") -> Path:
    d = LIBRARY / ("library_baked_8k" if res == "8k" else "library_baked")
    return d / name / "asset.usda"


def anomaly_asset(kind: str) -> str:
    return f"Tomato_SYNTHETIC_{kind}_S20261028"


def swap_crops(stage, tunnels=("09", "10"), anomalies=DEFAULT_ANOMALIES, seed: int = 7, res: str = "8k",
               collision: bool = True) -> list[dict]:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    rng = random.Random(seed)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    special = {(t, r, p): k for t, r, p, k in anomalies}
    UsdGeom.Scope.Define(stage, "/CropSwap")
    manifest = []
    for t in tunnels:
        anchor = stage.GetPrimAtPath(f"{ROOT}/SiteV18_Anchor_tunnel_{t}")
        if not anchor:
            continue
        for grp in anchor.GetChildren():
            m = PLANT.search(grp.GetName())
            if not m:
                continue
            row, idx = int(m.group(1)), int(m.group(2))
            box = cache.ComputeWorldBound(grp).ComputeAlignedRange()
            if box.IsEmpty():
                continue
            lo, hi = box.GetMin(), box.GetMax()
            base = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2])
            UsdGeom.Imageable(grp).MakeInvisible()
            for p in Usd.PrimRange(grp):                       # the hidden original must not block the LiDAR either
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr().Set(False)
            kind = special.get((t, row, idx))
            name = anomaly_asset(kind) if kind else rng.choice(HEALTHY)
            path = f"/CropSwap/T{t}_R{row}_P{idx:02d}"
            if stage.GetPrimAtPath(path):                          # a "_001" twin of the same row/plant number
                path += "_b"
            x = UsdGeom.Xform.Define(stage, path)
            x.AddTranslateOp().Set(Gf.Vec3d(*base))
            x.AddRotateZOp().Set(rng.uniform(0.0, 360.0))
            x.GetPrim().GetReferences().AddReference(str(asset_path(name, res)))
            if collision:                                     # LiDAR must see the plants (static triangle meshes)
                for p in Usd.PrimRange(x.GetPrim()):
                    if p.IsA(UsdGeom.Mesh) and p.IsActive():
                        UsdPhysics.CollisionAPI.Apply(p)
                        UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr().Set("none")
            manifest.append({"tunnel": f"tunnel-{t}", "row": row, "plant": idx, "asset": name,
                             "synthetic_anomaly": kind, "xy": [round(base[0], 3), round(base[1], 3)],
                             "v18_height_m": round(hi[2] - lo[2], 2), "prim": path})
    return manifest
