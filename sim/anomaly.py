"""Labelled test fixture for the digital twin: a textured 'diseased foliage' panel on the bed edge at an aisle metre,
so the inspect → revisit loop has something real to find (like the obstacle box for SAFETY). Texture from
sim/make_disease_texture.py. Authored at runtime on the live stage; never saved into the scene files."""
from __future__ import annotations
from pathlib import Path

PANEL_PATH = "/TestFixtures/disease_panel"


def add_disease_panel(stage, frame, metre: float, side: str, texture: str | Path,
                      bed_lateral_m: float = 0.60, z0: float = 0.75, size=(0.40, 0.70)) -> str:
    """side: 'left'|'right' as seen by the robot driving into the aisle (facing -Y): right = scene -X."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    lateral = -bed_lateral_m if side == "right" else bed_lateral_m
    cx, cy = frame.to_scene(metre, lateral)
    w, h = size
    face = 1.0 if lateral < 0 else -1.0                          # normal points back into the aisle
    UsdGeom.Xform.Define(stage, "/TestFixtures")
    mesh = UsdGeom.Mesh.Define(stage, PANEL_PATH)
    pts = [Gf.Vec3f(cx, cy - face * w / 2, z0), Gf.Vec3f(cx, cy + face * w / 2, z0),
           Gf.Vec3f(cx, cy + face * w / 2, z0 + h), Gf.Vec3f(cx, cy - face * w / 2, z0 + h)]
    mesh.CreatePointsAttr(pts); mesh.CreateFaceVertexCountsAttr([4]); mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(face, 0, 0)] * 4); mesh.SetNormalsInterpolation("vertex")
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    mat = UsdShade.Material.Define(stage, PANEL_PATH + "_mat")
    sh = UsdShade.Shader.Define(stage, PANEL_PATH + "_mat/surface"); sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    tex = UsdShade.Shader.Define(stage, PANEL_PATH + "_mat/tex"); tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(Path(texture).resolve()))
    reader = UsdShade.Shader.Define(stage, PANEL_PATH + "_mat/st"); reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    return PANEL_PATH
