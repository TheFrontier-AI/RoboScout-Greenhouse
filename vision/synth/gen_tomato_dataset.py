"""Synthetic, auto-labelled side-camera dataset for fruit grading and plant anomaly detection.

Builds randomised tomato rows from Tomato-Assets-V1 (8 healthy plants + 12 synthetic anomaly plants), renders
portrait frames from R1's three mast heights with Cycles, and writes exact labels from the object-index pass:
every fruit and every plant is its own object, so a box is the extent of its visible pixels. No hand annotation.

  blender -b --factory-startup ASSETS/tomato-assets.blend --python vision/synth/gen_tomato_dataset.py -- \
      --out vision_data/synth_v1 --scenes 40 --frames-per-scene 24 --seed 1

Labels describe synthetic visual alterations from the asset pack, not diagnosed diseases."""
from __future__ import annotations
import argparse, colorsys, json, math, os, random, sys, time

import bpy, bmesh
import numpy as np
import OpenImageIO as oiio
from mathutils import Euler, Matrix, Vector

MAST_HEIGHTS_M = (0.7945, 1.2145, 1.6445)          # robot_sensors.usda plant_* cameras, base frame z
PLANT_CLASSES = ("healthy", "chlorosis", "leaf_spots", "wilting", "stunted", "defoliation", "fruit_lesion")
FRUIT_CLASSES = ("ripe", "turning", "green", "lesion")
PLANT_ID0, STEM_ID0, FRUIT_ID0 = 1, 5000, 10000
BASE_WINDOW_M = (0.04, 0.35)                        # plant_base label: stem within +-4 cm and 0-35 cm above the bed
MIN_FRUIT_PX, MIN_FRUIT_FILL = 20, 0.3              # a whole visible fruit fills ~0.78 of its box
TISSUE_AOV = "ANOMALY_TISSUE"                       # asset pack's exact anomaly-pixel mask (VALUE AOV)
REGION_KINDS = ("chlorosis", "leaf_spots")          # locally visible (leaf_spots only when strong; mild is invisible
                                                    # at mast distance); wilting/stunted/defoliation are whole-plant shape
ANOMALY_CELL_PX, ANOMALY_CELL_COVER, ANOMALY_MIN_PX = 16, 0.05, 150


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--scenes", type=int, default=2)
    p.add_argument("--frames-per-scene", type=int, default=6)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--width", type=int, default=576)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--anomaly-rate", type=float, default=0.3)
    return p.parse_args(argv)


def fruit_class_from_rgb(rgb) -> str:
    """Ripeness from the asset's col_botanical colour: hue 0-25 deg red, 25-70 turning (orange/yellow), else green."""
    h = colorsys.rgb_to_hsv(*rgb)[0] * 360.0
    if h < 25.0 or h > 340.0:
        return "ripe"
    return "turning" if h < 70.0 else "green"


def variant_of(name: str) -> tuple[str, str]:
    if name.startswith("Tomato_Healthy_"):
        return "healthy", "none"
    parts = name.split("_")              # Tomato_SYNTHETIC_<type...>_<mild|strong>_S...
    sev = parts[-2]
    return "_".join(parts[2:-2]), sev


# ---------------------------------------------------------------- kits: plant body + separated fruit objects
def separate_slot(body, keyword: str):
    """Split the faces using the material whose name contains keyword into a new object; return it."""
    slot = next(i for i, s in enumerate(body.material_slots) if keyword in s.material.name)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    body.active_material_index = slot
    bpy.ops.object.material_slot_select()
    bpy.ops.mesh.separate(type="SELECTED")
    bpy.ops.object.mode_set(mode="OBJECT")
    return next(o for o in bpy.context.selected_objects if o is not body)


def build_kits(kit_col) -> list[dict]:
    kits = []
    for col in list(bpy.data.collections):
        if not (col.name.startswith("Tomato_Healthy_") or col.name.startswith("Tomato_SYNTHETIC_")):
            continue
        src = col.objects[0]
        body = src.copy()
        body.data = src.data.copy()
        body.name = "KIT_" + src.name
        kit_col.objects.link(body)
        stem = separate_slot(body, "Stem")
        fruit_all = separate_slot(body, "Fruit")
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = fruit_all
        fruit_all.select_set(True)
        bpy.ops.mesh.separate(type="LOOSE")
        fruits = [o for o in bpy.context.selected_objects]

        kind, sev = variant_of(src.name)
        fdata = []
        for f in fruits:
            att = f.data.attributes["col_botanical"].data
            rgb = np.array([c.color[:3] for c in att]).mean(axis=0)
            fcls = "lesion" if kind == "fruit_lesion" else fruit_class_from_rgb(rgb)
            fdata.append({"obj": f, "cls": fcls, "rgb": [round(float(x), 4) for x in rgb]})
        kits.append({"name": src.name, "kind": kind, "severity": sev, "body": body, "stem": stem, "fruits": fdata})
    return kits


# ---------------------------------------------------------------- static set dressing
def mat(name, rgb, rough=0.8, emit=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1)
    bsdf.inputs["Roughness"].default_value = rough
    if emit:
        bsdf.inputs["Emission Color"].default_value = (*rgb, 1)
        bsdf.inputs["Emission Strength"].default_value = emit
    return m


def add_box(col, name, loc, size, material):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    o.location, o.scale = loc, size
    me.materials.append(material)
    col.objects.link(o)
    return o


def add_cyl(col, name, p0, p1, radius, material):
    p0, p1 = Vector(p0), Vector(p1)
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=6, radius1=radius, radius2=radius, depth=1.0)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    d = p1 - p0
    o.location = (p0 + p1) / 2
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = d.to_track_quat("Z", "Y")
    o.scale = (1, 1, d.length)
    me.materials.append(material)
    col.objects.link(o)
    return o


# ---------------------------------------------------------------- one randomised row
def build_scene(rng: random.Random, kits, scene_col, anomaly_rate):
    for o in list(scene_col.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    healthy = [k for k in kits if k["kind"] == "healthy"]
    sick = [k for k in kits if k["kind"] != "healthy"]

    row_x = rng.uniform(0.75, 1.25)              # camera (x=0) to near plant row
    bed_h = rng.uniform(0.15, 0.40)
    bed_w = rng.uniform(0.6, 1.0)
    length = 14.0
    grey = rng.uniform(0.25, 0.55)
    concrete = mat("concrete", (grey, grey * 0.98, grey * 0.93), 0.9)
    soil_v = rng.uniform(0.06, 0.16)
    soil = mat("soil", (soil_v * 1.3, soil_v, soil_v * 0.7), 1.0)
    string = mat("string", (0.8, 0.8, 0.78), 0.7)
    pipe = mat("pipe", (0.02, 0.02, 0.02), 0.4)
    film_v = rng.uniform(0.6, 1.0)
    film = mat("film", (film_v, film_v, film_v * 0.97), 0.5, emit=rng.uniform(0.3, 3.0))

    add_box(scene_col, "ground", (row_x, 0, -0.01), (12, length + 6, 0.02), soil)
    for bx in (row_x, row_x + bed_w + 1.3):      # near bed + the next bed for background clutter
        add_box(scene_col, "bed", (bx + bed_w / 2 - 0.25, 0, bed_h / 2), (bed_w, length, bed_h), concrete)
        add_cyl(scene_col, "drip", (bx - 0.15, -length / 2, bed_h + 0.01), (bx - 0.15, length / 2, bed_h + 0.01), 0.008, pipe)
    add_box(scene_col, "film_wall", (row_x + rng.uniform(3.5, 6.0), 0, 2.0), (0.02, length + 6, 4.5), film)
    add_cyl(scene_col, "wire", (row_x, -length / 2, 2.6), (row_x, length / 2, 2.6), 0.004, string)

    plants, fruits = [], []
    pid, fid = PLANT_ID0, FRUIT_ID0
    for row_off in (0.0, bed_w * 0.5, bed_w + 1.3, bed_w * 1.5 + 1.3):
        near = row_off < bed_w                    # both rows of the bed beside the aisle
        spacing = rng.uniform(0.35, 0.6)
        y = -length / 2 + rng.uniform(0, spacing)
        while y < length / 2:
            kit = rng.choice(sick) if rng.random() < anomaly_rate else rng.choice(healthy)
            px, py = row_x + row_off + rng.uniform(-0.04, 0.04), y + rng.uniform(-0.05, 0.05)
            m = (Matrix.Translation((px, py, bed_h)) @ Matrix.Rotation(rng.uniform(0, 2 * math.pi), 4, "Z")
                 @ Matrix.Scale(rng.uniform(0.85, 1.25), 4))
            body = bpy.data.objects.new(f"P{pid}", kit["body"].data)
            body.matrix_world, body.pass_index = m, pid
            body.color = (rng.uniform(0.85, 1.1), rng.uniform(0.9, 1.1), rng.uniform(0.85, 1.1), 1)
            scene_col.objects.link(body)
            st = bpy.data.objects.new(f"S{pid}", kit["stem"].data)
            st.matrix_world, st.pass_index = m, STEM_ID0 + pid
            scene_col.objects.link(st)
            plants.append({"id": pid, "kit": kit["name"], "kind": kit["kind"], "severity": kit["severity"],
                           "world_xyz": [round(px, 3), round(py, 3), round(bed_h, 3)], "near_row": near})
            for f in kit["fruits"]:
                fo = bpy.data.objects.new(f"F{fid}", f["obj"].data)
                fo.matrix_world, fo.pass_index = m, fid
                scene_col.objects.link(fo)
                c = fo.matrix_world @ (sum((Vector(b) for b in f["obj"].bound_box), Vector()) / 8)
                fruits.append({"id": fid, "plant_id": pid, "cls": f["cls"], "rgb": f["rgb"],
                               "world_xyz": [round(v, 3) for v in c]})
                fid += 1
            add_cyl(scene_col, "string", (px, py, bed_h), (px + rng.uniform(-0.05, 0.05), py, 2.6), 0.0025, string)
            pid += 1
            y += spacing * rng.uniform(0.85, 1.15)
    return {"row_x": row_x, "bed_h": bed_h, "plants": plants, "fruits": fruits}


def randomise_light(rng: random.Random, scene):
    world = scene.world
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    sky = rng.uniform(0.75, 1.0)
    bg.inputs["Color"].default_value = (sky * rng.uniform(0.85, 1.0), sky, sky * rng.uniform(1.0, 1.1), 1)
    bg.inputs["Strength"].default_value = rng.uniform(0.6, 3.0)
    sun = bpy.data.objects["SUN"]
    sun.rotation_euler = Euler((math.radians(rng.uniform(5, 70)), 0, math.radians(rng.uniform(0, 360))))
    sun.data.energy = rng.uniform(0.5, 6.0)
    sun.data.angle = math.radians(rng.uniform(0.5, 25))       # film diffuses the sun
    scene.view_settings.exposure = rng.uniform(-1.0, 1.0)


def setup_render(scene, a):
    try:
        scene.render.engine = "CYCLES"
    except TypeError as e:
        raise SystemExit(f"Cycles unavailable: {e}")
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "CUDA"
    scene.cycles.device = "GPU"
    scene.cycles.samples = a.samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.cycles.transparent_max_bounces = 16
    scene.render.resolution_x, scene.render.resolution_y = a.width, a.height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    vl = scene.view_layers[0]
    vl.use_pass_object_index = True
    vl.pass_alpha_threshold = 0.5           # index ignores leaf-cutout texels below alpha 0.5
    if TISSUE_AOV not in vl.aovs:
        aov = vl.aovs.add()
        aov.name, aov.type = TISSUE_AOV, "VALUE"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    sun_data = bpy.data.lights.new("SUN", "SUN")
    sun = bpy.data.objects.new("SUN", sun_data)
    scene.collection.objects.link(sun)
    cam = bpy.data.objects.new("CAM", bpy.data.cameras.new("CAM"))
    scene.collection.objects.link(cam)
    cam.data.sensor_fit = "VERTICAL"
    scene.camera = cam
    return cam


def set_format(scene, multilayer: bool):
    """PNG beauty or multilayer EXR (Blender 5 moved multilayer to media_type; 4.x used OPEN_EXR_MULTILAYER)."""
    st = scene.render.image_settings
    if hasattr(st, "media_type"):
        st.media_type = "MULTI_LAYER_IMAGE" if multilayer else "IMAGE"
        st.file_format = "OPEN_EXR_MULTILAYER" if multilayer else "PNG"
    else:
        st.file_format = "OPEN_EXR_MULTILAYER" if multilayer else "PNG"
    if multilayer:
        st.color_depth = "32"
    else:
        st.color_mode, st.color_depth = "RGB", "8"


def read_pass(path: str, keys: tuple[str, ...]) -> np.ndarray:
    """One single-channel pass from a multi-part EXR (Blender 5 writes each pass as its own subimage)."""
    n = oiio.ImageBuf(path).nsubimages
    for sub in range(max(n, 1)):
        buf = oiio.ImageBuf(path, sub, 0)
        for c, name in enumerate(buf.spec().channelnames):
            if any(k in name for k in keys):
                return buf.get_pixels(oiio.FLOAT)[:, :, c]
    raise RuntimeError(f"no {keys} pass in {path}")


def boxes_from_index(idx: np.ndarray, min_px: int):
    ids, counts = np.unique(idx, return_counts=True)
    out = {}
    for i, n in zip(ids.tolist(), counts.tolist()):
        if i <= 0 or n < min_px:
            continue
        ys, xs = np.nonzero(idx == i)
        out[i] = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1), int(n))
    return out


def label_frame(idx: np.ndarray, tissue: np.ndarray, scene, cam, plants: dict, fruits: dict) -> list[dict]:
    """fruit boxes (visible pixels; heavily occluded ones flagged ignore), one plant_base per visible near-bed
    stem, and anomaly regions = connected clusters of a plant's visible ANOMALY_TISSUE pixels."""
    from bpy_extras.object_utils import world_to_camera_view
    H, W = idx.shape
    boxes = boxes_from_index(idx, 1)
    objs = []
    for oid, (x, y, w, h, n) in boxes.items():
        if oid in fruits:
            f = fruits[oid]
            fill = n / float(w * h)
            objs.append({"type": "fruit", "cls": f["cls"], "bbox": [x, y, w, h], "px": n, "fill": round(fill, 3),
                         "id": oid, "plant_id": f["plant_id"],
                         "ignore": n < MIN_FRUIT_PX or fill < MIN_FRUIT_FILL})
    for pid, p in plants.items():
        if not p["near_row"] or STEM_ID0 + pid not in boxes:
            continue
        bx, by, bz = p["world_xyz"]
        pts = [world_to_camera_view(scene, cam, Vector((bx, by + dy, bz + dz)))
               for dy in (-BASE_WINDOW_M[0], BASE_WINDOW_M[0]) for dz in (0.0, BASE_WINDOW_M[1])]
        if any(q.z <= 0 for q in pts):
            continue
        xs = [q.x * W for q in pts]
        ys = [(1 - q.y) * H for q in pts]
        x0, x1 = max(0, int(min(xs))), min(W, int(max(xs)) + 1)
        y0, y1 = max(0, int(min(ys))), min(H, int(max(ys)) + 1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        sub = idx[y0:y1, x0:x1] == STEM_ID0 + pid
        n = int(sub.sum())
        if n < 20:
            continue
        ys_, xs_ = np.nonzero(sub)
        objs.append({"type": "plant_base", "cls": "plant_base", "id": pid, "px": n,
                     "bbox": [x0 + int(xs_.min()), y0 + int(ys_.min()), int(np.ptp(xs_)) + 1, int(np.ptp(ys_)) + 1],
                     "world_xyz": p["world_xyz"]})
    c = ANOMALY_CELL_PX
    gh, gw = H // c, W // c
    for pid, p in plants.items():
        if p["kind"] not in REGION_KINDS or (p["kind"] == "leaf_spots" and p["severity"] != "strong") or pid not in boxes:
            continue
        mask = tissue[:gh * c, :gw * c] & (idx[:gh * c, :gw * c] == pid)
        if mask.sum() < ANOMALY_MIN_PX:
            continue
        cover = mask.reshape(gh, c, gw, c).mean(axis=(1, 3)) >= ANOMALY_CELL_COVER
        seen = np.zeros_like(cover)
        for gy, gx in zip(*np.nonzero(cover)):
            if seen[gy, gx]:
                continue
            stack, comp = [(gy, gx)], []
            seen[gy, gx] = True
            while stack:
                cy, cx = stack.pop()
                comp.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < gh and 0 <= nx < gw and cover[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            cy_, cx_ = zip(*comp)
            x0, y0, x1, y1 = (int(v) for v in (min(cx_) * c, min(cy_) * c, (max(cx_) + 1) * c, (max(cy_) + 1) * c))
            n = int(mask[y0:y1, x0:x1].sum())
            if n < ANOMALY_MIN_PX:
                continue
            objs.append({"type": "anomaly", "cls": p["kind"], "severity": p["severity"], "id": pid,
                         "bbox": [x0, y0, x1 - x0, y1 - y0], "px": n, "world_xyz": p["world_xyz"]})
    return objs


def main():
    a = parse_args()
    rng = random.Random(a.seed)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "labels"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "index"), exist_ok=True)
    scene = bpy.context.scene

    for c in list(bpy.data.collections):          # drop the display layout; keep asset collections out of view
        if c.name.startswith("DISPLAY"):
            for o in list(c.objects):
                bpy.data.objects.remove(o, do_unlink=True)
    kit_col = bpy.data.collections.new("KITS")
    scene.collection.children.link(kit_col)
    kits = build_kits(kit_col)
    for lc in bpy.context.view_layer.layer_collection.children:
        lc.exclude = lc.name != "SCENE"
    scene_col = bpy.data.collections.new("SCENE")
    scene.collection.children.link(scene_col)
    for lc in bpy.context.view_layer.layer_collection.children:
        lc.exclude = lc.name != "SCENE"
    cam = setup_render(scene, a)
    print(f"[gen] kits: {[(k['name'], len(k['fruits'])) for k in kits]}", flush=True)

    tmp_exr = os.path.join(a.out, f"_tmp_index_s{a.seed}_{os.getpid()}.exr")   # one per process: runs share --out
    n_done = 0
    for s in range(a.scenes):
        layout = build_scene(rng, kits, scene_col, a.anomaly_rate)
        plants = {p["id"]: p for p in layout["plants"]}
        fruits = {f["id"]: f for f in layout["fruits"]}
        for k in range(a.frames_per_scene):
            t0 = time.time()
            randomise_light(rng, scene)
            slot = k % 3
            cam.location = (rng.uniform(-0.05, 0.1), rng.uniform(-5.5, 5.5),
                            MAST_HEIGHTS_M[slot] + rng.uniform(-0.08, 0.08))
            cam.rotation_euler = Euler((math.radians(90 + rng.uniform(-10, 10) + (8 if slot == 0 else -6 if slot == 2 else 0)),
                                        math.radians(rng.uniform(-3, 3)), math.radians(-90 + rng.uniform(-10, 10))))
            cam.data.angle = math.radians(rng.uniform(55, 75))
            stem = f"s{a.seed:03d}_{s:04d}_{k:03d}"

            set_format(scene, multilayer=True)        # write_still keeps every pass; save_render keeps only Combined
            scene.render.filepath = tmp_exr
            bpy.ops.render.render(write_still=True)
            set_format(scene, multilayer=False)
            bpy.data.images["Render Result"].save_render(os.path.join(a.out, "images", stem + ".png"), scene=scene)

            idx = np.rint(read_pass(tmp_exr, ("Object Index", "IndexOB"))).astype(np.int32)
            tissue = read_pass(tmp_exr, (TISSUE_AOV,)) > 0.5
            oiio.ImageBuf(idx.astype(np.uint16)).write(os.path.join(a.out, "index", stem + ".png"), oiio.UINT16)

            objs = label_frame(idx, tissue, scene, cam, plants, fruits)
            label = {"image": f"images/{stem}.png", "width": a.width, "height": a.height,
                     "camera": {"slot": ("low", "mid", "high")[slot], "location": [round(v, 3) for v in cam.location],
                                "rotation_euler": [round(v, 4) for v in cam.rotation_euler],
                                "vfov_deg": round(math.degrees(cam.data.angle), 2)},
                     "scene": {"seed": a.seed, "index": s, "row_x": round(layout["row_x"], 3), "bed_h": round(layout["bed_h"], 3)},
                     "objects": objs,
                     "plants_in_scene": layout["plants"], "fruits_in_scene": layout["fruits"]}
            with open(os.path.join(a.out, "labels", stem + ".json"), "w") as fh:
                json.dump(label, fh)
            n_done += 1
            cnt = {t: sum(o["type"] == t and not o.get("ignore") for o in objs) for t in ("fruit", "plant_base", "anomaly")}
            print(f"[gen] {stem} {cnt} {time.time() - t0:.1f}s ({n_done})", flush=True)
    if os.path.exists(tmp_exr):
        os.remove(tmp_exr)


main()
