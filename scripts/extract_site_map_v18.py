"""Draft farm site map from the V18 farm GLB (stdlib only: reads just the GLB JSON chunk, never the 150 MB buffer).

Greenhouses = `SiteV18-Anchor-<feature>` nodes with feature_kind tunnel_greenhouse | greenhouse_block. Coordinates are
converted glTF Y-up → Blender Z-up exactly like data/tunnel09_aisle_map_v18.json: (x, y, z)_site = (x, -z, y)_gltf.
Doors: the aisle map's +Y entry for the house it covers; else the side the authored `entrance` view preset looks in from;
else the long-axis end facing the other greenhouses (door_estimated). Each door gets an apron junction 2 m outside it.
Corridors: the blueprint road centrelines of site-layout-v5.json (same frame; checked against the anchors), short
straight links from aprons to roads, nearest-neighbour links, then the shortest links that join leftover components.
Every edge has width_m null and the map is unconfirmed: a human must review it before the robot uses it.
  python3 scripts/extract_site_map_v18.py --out data/site_map_v18_draft.json"""
from __future__ import annotations
import argparse, hashlib, json, math, struct, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r1_nav.mission import HEADLAND_OFFSET_M          # noqa: E402
from r1_nav.site_map import Edge, Node, SiteMap, default_meta  # noqa: E402

FARM = ROOT.parent / "robot_design/r1_v3/farm_integration/latest_delivery/prototype"
GLB = FARM / "assets/farm-realistic-v18.glb"
LAYOUT = FARM / "site-layout-v5.json"
HOUSE_KINDS = ("tunnel_greenhouse", "greenhouse_block")
OBSTACLE_KINDS = HOUSE_KINDS + ("service_building", "pond", "pavilion", "utility_pad", "nursery_plot")
ATTACH_MAX_M = 35.0          # apron → road straight link
KNN, KNN_MAX_M = 2, 40.0


# ---------------------------------------------------------------- GLB scene graph
def read_glb_json(path: Path) -> dict:
    with open(path, "rb") as f:
        magic, _, _ = struct.unpack("<III", f.read(12))
        length, kind = struct.unpack("<II", f.read(8))
        if magic != 0x46546C67 or kind != 0x4E4F534A:
            raise ValueError(f"{path} is not a GLB with a leading JSON chunk")
        return json.loads(f.read(length))


def _mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def local_matrix(n: dict):
    if "matrix" in n:
        m = n["matrix"]
        return [[m[c * 4 + r] for c in range(4)] for r in range(4)]          # glTF is column-major
    tx, ty, tz = n.get("translation", (0, 0, 0))
    x, y, z, w = n.get("rotation", (0, 0, 0, 1))
    sx, sy, sz = n.get("scale", (1, 1, 1))
    r = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    return [[r[0][0] * sx, r[0][1] * sy, r[0][2] * sz, tx], [r[1][0] * sx, r[1][1] * sy, r[1][2] * sz, ty],
            [r[2][0] * sx, r[2][1] * sy, r[2][2] * sz, tz], [0, 0, 0, 1]]


def world_matrices(j: dict) -> dict[int, list]:
    eye = [[float(i == k) for k in range(4)] for i in range(4)]
    out, stack = {}, [(r, eye) for s in j.get("scenes", []) for r in s.get("nodes", [])]
    while stack:
        i, parent = stack.pop()
        out[i] = _mul(parent, local_matrix(j["nodes"][i]))
        stack += [(c, out[i]) for c in j["nodes"][i].get("children", [])]
    return out


def to_site(p):                                       # glTF Y-up → Blender/site Z-up
    return (p[0], -p[2], p[1])


def anchors(j: dict) -> list[dict]:
    W, feats = world_matrices(j), []
    for i, n in enumerate(j["nodes"]):
        if not n.get("name", "").startswith("SiteV18-Anchor-") or i not in W:
            continue
        e, m = n.get("extras", {}), W[i]
        x, y, _ = to_site((m[0][3], m[1][3], m[2][3]))
        ax = to_site((m[0][0], m[1][0], m[2][0]))     # local +X axis in the site frame → yaw
        yaw = math.atan2(ax[1], ax[0])
        f = {"id": e.get("feature_id", n["name"][15:]), "kind": e.get("feature_kind"), "label": e.get("label"),
             "node": n["name"], "xy": (x, y), "yaw": yaw, "size": (e.get("width_m", 0.0), e.get("depth_m", 0.0)),
             "interior_system": e.get("interior_system"), "confidence": e.get("confidence")}
        ent = (e.get("view_presets") or {}).get("entrance")
        if ent:
            f["entrance"] = {"position": ent["position"][:2], "target": ent["target"][:2]}
        feats.append(f)
    return feats


# ---------------------------------------------------------------- geometry
def to_local(f, p):
    c, s = math.cos(f["yaw"]), math.sin(f["yaw"])
    dx, dy = p[0] - f["xy"][0], p[1] - f["xy"][1]
    return (c * dx + s * dy, -s * dx + c * dy)


def to_world(f, q):
    c, s = math.cos(f["yaw"]), math.sin(f["yaw"])
    return (f["xy"][0] + c * q[0] - s * q[1], f["xy"][1] + s * q[0] + c * q[1])


def seg_hits_rect(f, p, q, shrink=0.3) -> bool:
    """Does segment p–q pass through feature f's footprint (shrunk by `shrink`)? Liang–Barsky in local coords."""
    hx, hy = f["size"][0] / 2 - shrink, f["size"][1] / 2 - shrink
    if hx <= 0 or hy <= 0:
        return False
    (x0, y0), (x1, y1) = to_local(f, p), to_local(f, q)
    t0, t1, dx, dy = 0.0, 1.0, x1 - x0, y1 - y0
    for pk, qk in ((-dx, x0 + hx), (dx, hx - x0), (-dy, y0 + hy), (dy, hy - y0)):
        if pk == 0:
            if qk < 0:
                return False
        else:
            t = qk / pk
            if pk < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def project(p, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    L2 = vx * vx + vy * vy or 1e-12
    t = max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / L2))
    q = (a[0] + t * vx, a[1] + t * vy)
    return t, q, math.dist(p, q)


def seg_intersect(a, b, c, d):
    r, s = (b[0] - a[0], b[1] - a[1]), (d[0] - c[0], d[1] - c[1])
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-9:
        return None
    t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / den
    u = ((c[0] - a[0]) * r[1] - (c[1] - a[1]) * r[0]) / den
    if -1e-6 <= t <= 1 + 1e-6 and -1e-6 <= u <= 1 + 1e-6:
        return t, (a[0] + t * r[0], a[1] + t * r[1])
    return None


# ---------------------------------------------------------------- doors
def entry_face(f: dict):
    """Outward normal (local) of the footprint face the entrance view ray enters through, or None."""
    p, q = to_local(f, f["entrance"]["position"]), to_local(f, f["entrance"]["target"])
    d, h = (q[0] - p[0], q[1] - p[1]), (f["size"][0] / 2, f["size"][1] / 2)
    best = None
    for ax in (0, 1):
        for sign in (-1.0, 1.0):
            if d[ax] == 0 or (p[ax] - sign * h[ax]) * sign <= 0:     # camera must be outside this face
                continue
            t = (sign * h[ax] - p[ax]) / d[ax]
            other = p[1 - ax] + t * d[1 - ax]
            if t > 0 and abs(other) <= h[1 - ax] and (best is None or t < best[0]):
                best = (t, (sign, 0.0) if ax == 0 else (0.0, sign))
    return best[1] if best else None


def place_door(f: dict, centroid, aisle_maps: dict) -> dict:
    hx, hy = f["size"][0] / 2, f["size"][1] / 2
    if f["id"] in aisle_maps:                         # aisle-map entry convention: house-local +Y end
        T = aisle_maps[f["id"]]["house_local_to_world"]
        local, est, how = (0.0, hy), False, "aisle map entry convention (house-local +Y)"
        world = (T[0][0] * local[0] + T[0][1] * local[1] + T[0][3], T[1][0] * local[0] + T[1][1] * local[1] + T[1][3])
        out = (world[0] - T[0][3], world[1] - T[1][3])
    else:
        face = entry_face(f) if "entrance" in f else None
        if face:                                      # side through which the entrance camera looks into the house
            ref, est, how = face, False, "authored entrance view preset (side the camera looks in through)"
            lat = to_local(f, f["entrance"]["target"])
        else:                                         # no preset: long-axis end facing the other greenhouses
            ref, est, how = to_local(f, centroid), True, "estimated: long-axis end facing the other greenhouses"
            ref = (ref[0], 0.0) if hx >= hy else (0.0, ref[1])
            lat = (0.0, 0.0)
        if abs(ref[0]) / max(hx, 1e-6) >= abs(ref[1]) / max(hy, 1e-6):
            local = (math.copysign(hx, ref[0]), max(-hy + 0.5, min(hy - 0.5, lat[1])))
            out = (math.copysign(1.0, ref[0]), 0.0)
        else:
            local = (max(-hx + 0.5, min(hx - 0.5, lat[0])), math.copysign(hy, ref[1]))
            out = (0.0, math.copysign(1.0, ref[1]))
        world = to_world(f, local)
        c, s = math.cos(f["yaw"]), math.sin(f["yaw"])
        out = (c * out[0] - s * out[1], s * out[0] + c * out[1])
    n = math.hypot(*out) or 1.0
    out = (out[0] / n, out[1] / n)
    heading = math.degrees(math.atan2(-out[1], -out[0])) % 360.0           # driving in = against the outward normal
    return {"xy": world, "out": out, "estimated": est, "how": how, "heading_deg": round(heading, 1)}


# ---------------------------------------------------------------- roads
def load_roads(layout_path: Path, feats: list[dict]) -> tuple[list[dict], str]:
    if not layout_path or not layout_path.exists():
        return [], "no site-layout file: roads skipped"
    L = json.loads(layout_path.read_text())
    sx, sy = L["scale"]["x_metres_per_pixel"], L["scale"]["y_metres_per_pixel"]
    ox, oy = L["coordinate_system"]["model_origin_px"]
    conv = lambda p: ((p[0] - ox) * sx, (oy - p[1]) * sy)                  # X=(px-27)*sx; Y=(1039-py)*sy
    byid = {f["id"]: f for f in feats}
    errs = [math.dist(conv(((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)), byid[x["id"]]["xy"])
            for x in L.get("features", []) if x["id"] in byid and (r := x.get("rect_px"))]
    if not errs or max(errs) > 0.5:
        return [], f"layout frame does not match the GLB anchors (max {max(errs or [math.inf]):.2f} m): roads skipped"
    roads = [{"id": r["id"], "label": r.get("note", ""), "pts": [conv(p) for p in r["path_px"]],
              "width_m": round(r.get("width_px", 0) * (sx + sy) / 2, 2), "confidence": r.get("confidence")}
             for r in L.get("roads", [])]
    return roads, f"{len(roads)} blueprint road centrelines (frame check max {max(errs):.3f} m over {len(errs)} features)"


# ---------------------------------------------------------------- build
def build(glb: Path, layout: Path | None, aisle_map_paths: list[Path], dock_house: str, dock_offset) -> tuple[SiteMap, dict]:
    j = read_glb_json(glb)
    feats = anchors(j)
    houses = [f for f in feats if f["kind"] in HOUSE_KINDS]
    obstacles = [f for f in feats if f["kind"] in OBSTACLE_KINDS]
    aisle_maps = {}
    for p in aisle_map_paths:
        d = json.loads(p.read_text())
        aisle_maps[d["source"]["feature_id"]] = d
    centroid = (sum(f["xy"][0] for f in houses) / len(houses), sum(f["xy"][1] for f in houses) / len(houses))
    nodes: dict[str, Node] = {}
    edges: list[Edge] = []
    report = {"houses": [], "estimated_doors": [], "frame_checks": {}}

    def add_edge(a, b, provenance, surface="unknown", notes=None, **extra):
        """`provenance` says where the edge came from; `notes` are for things a human must know (shown to HERMES)."""
        if a != b and not any(e.key() == frozenset((a, b)) for e in edges):
            edges.append(Edge(a, b, round(math.dist(nodes[a].xy, nodes[b].xy), 3), None, surface, notes or [], False,
                              {"provenance": provenance, **extra}))

    def crosses(a, b):
        p, q = nodes[a].xy, nodes[b].xy                # aprons sit 2 m outside, so even the own house counts
        return [f["id"] for f in obstacles if seg_hits_rect(f, p, q)]

    for f in sorted(houses, key=lambda f: f["id"]):
        if f["id"] in aisle_maps:
            T = aisle_maps[f["id"]]["house_local_to_world"]
            report["frame_checks"][f["id"]] = round(math.dist((T[0][3], T[1][3]), f["xy"]), 4)
        d = place_door(f, centroid, aisle_maps)
        did, aid = f"{f['id']}/door", f"{f['id']}/apron"
        extra = {"door_estimated": d["estimated"], "door_from": d["how"], "feature_kind": f["kind"]}
        nodes[did] = Node(did, "door", d["xy"], f["id"], d["heading_deg"], f["label"], extra)
        apron = (d["xy"][0] + HEADLAND_OFFSET_M * d["out"][0], d["xy"][1] + HEADLAND_OFFSET_M * d["out"][1])
        nodes[aid] = Node(aid, "junction", apron, None, None, f"{f['label']} apron", {"door": did})
        add_edge(did, aid, "door apron (headland in front of the door), not measured")
        report["houses"].append(f["id"])
        if d["estimated"]:
            report["estimated_doors"].append(f["id"])

    roads, road_msg = load_roads(layout, feats) if layout else ([], "roads disabled")
    report["roads"] = road_msg
    # cut points per road: vertices, road–road intersections, apron attachment projections
    cuts: dict[str, list[tuple[float, tuple]]] = {r["id"]: [] for r in roads}
    rlen = lambda r, k: sum(math.dist(r["pts"][i], r["pts"][i + 1]) for i in range(k))
    for r in roads:
        cuts[r["id"]] += [(rlen(r, k), p) for k, p in enumerate(r["pts"])]
    for i, r in enumerate(roads):
        for r2 in roads[i + 1:]:
            for k in range(len(r["pts"]) - 1):
                for k2 in range(len(r2["pts"]) - 1):
                    hit = seg_intersect(r["pts"][k], r["pts"][k + 1], r2["pts"][k2], r2["pts"][k2 + 1])
                    if hit:
                        p = hit[1]
                        cuts[r["id"]].append((rlen(r, k) + math.dist(r["pts"][k], p), p))
                        cuts[r2["id"]].append((rlen(r2, k2) + math.dist(r2["pts"][k2], p), p))
    attach = {}
    for aid in [n.id for n in nodes.values() if n.kind == "junction"]:
        best = None
        for r in roads:
            for k in range(len(r["pts"]) - 1):
                t, q, dist = project(nodes[aid].xy, r["pts"][k], r["pts"][k + 1])
                if dist <= ATTACH_MAX_M and (best is None or dist < best[0]):
                    best = (dist, r["id"], rlen(r, k) + math.dist(r["pts"][k], q), q)
        if best:
            attach[aid] = best
            cuts[best[1]].append((best[2], best[3]))
    # road nodes: merge cut points closer than 1 m (also across roads, so crossings become one junction)
    road_nodes: list[tuple[str, tuple]] = []

    def road_node(p):
        for nid, q in road_nodes:
            if math.dist(p, q) < 1.0:
                return nid
        nid = f"road/{len(road_nodes) + 1:02d}"
        road_nodes.append((nid, p))
        nodes[nid] = Node(nid, "waypoint", p, label="road")
        return nid

    for r in roads:
        seq = [road_node(p) for _, p in sorted(cuts[r["id"]], key=lambda c: c[0])]
        for a, b in zip(seq, seq[1:]):
            add_edge(a, b, f"blueprint road {r['id']} (drawn width ~{r['width_m']} m, provisional scale; material unknown)",
                     "road", road=r["id"])
    for aid, (dist, rid, _, q) in attach.items():
        rn = road_node(q)
        if not crosses(aid, rn):
            add_edge(aid, rn, f"straight link to road {rid} ({dist:.1f} m), not surveyed")
    # nearest-neighbour links for aprons that are not on a road yet
    aprons = [n.id for n in nodes.values() if n.kind == "junction"]
    for aid in aprons:
        if any(aid in (e.a, e.b) and "door" not in (nodes[e.a].kind, nodes[e.b].kind) for e in edges):
            continue
        near = sorted((math.dist(nodes[aid].xy, nodes[b].xy), b) for b in aprons + [r for r, _ in road_nodes] if b != aid)
        for dist, b in [x for x in near if x[0] <= KNN_MAX_M and not crosses(aid, x[1])][:KNN]:
            add_edge(aid, b, f"nearest-neighbour straight link ({dist:.1f} m), not surveyed")

    # dock beside the chosen house door (house-local offset, rotated with the house)
    hd = nodes[f"{dock_house}/door"]
    f = next(f for f in houses if f["id"] == dock_house)
    c, s = math.cos(f["yaw"]), math.sin(f["yaw"])
    dock_xy = (hd.xy[0] + c * dock_offset[0] - s * dock_offset[1], hd.xy[1] + s * dock_offset[0] + c * dock_offset[1])
    nodes["dock"] = Node("dock", "dock", dock_xy, None, None, f"dock near {dock_house}",
                         {"dock_offset_house_local": list(dock_offset), "dock_estimated": True})
    add_edge("dock", f"{dock_house}/apron", "dock beside the door apron; position is a placeholder")

    # join leftover components with the shortest links (prefer ones that do not cross a footprint)
    m = SiteMap(nodes, edges, {})
    while len(comps := m._components()) > 1:
        main = next(c for c in comps if "dock" in c)
        rest = set().union(*(c for c in comps if c is not main))
        cand = sorted((math.dist(nodes[a].xy, nodes[b].xy), a, b) for a in main for b in rest
                      if not a.endswith("/door") and not b.endswith("/door"))
        dist, a, b = next(((d, a, b) for d, a, b in cand if not crosses(a, b)), cand[0])
        hit = crosses(a, b)
        add_edge(a, b, f"gap-closing straight link ({dist:.1f} m), not surveyed",
                 notes=[f"WARNING straight line crosses {', '.join(hit)}; real path unknown"] if hit else None)
        m = SiteMap(nodes, edges, {})

    meta = default_meta("v18", True, source_detail={
        "glb": str(glb), "glb_sha256": hashlib.sha256(glb.read_bytes()).hexdigest(), "layout": str(layout) if layout else None,
        "aisle_maps": [str(p) for p in aisle_map_paths], "conversion": "glTF Y-up → Z-up: (x, y, z) = (x, -z, y)",
        "status": "Draft from illustrative model; blueprint scale provisional; not surveyed. Edge widths unknown.",
        "estimated_doors": report["estimated_doors"]})
    landmarks = [{"id": f["id"], "kind": f["kind"], "label": f["label"], "xy": [round(v, 3) for v in f["xy"]],
                  "size_m": [round(v, 3) for v in f["size"]], "yaw_deg": round(math.degrees(f["yaw"]), 3),
                  "confidence": f["confidence"]} for f in feats]
    site = SiteMap(nodes, edges, meta, [], landmarks)
    report.update(features=len(feats), nodes=len(nodes), edges=len(edges), warnings=site.validate(),
                  houses_by_kind={k: sum(f["kind"] == k for f in houses) for k in HOUSE_KINDS})
    return site, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--glb", type=Path, default=GLB)
    ap.add_argument("--layout", type=Path, default=LAYOUT, help="site-layout-v5.json with road centrelines ('' = none)")
    ap.add_argument("--aisle-map", type=Path, nargs="*", default=[ROOT / "data/tunnel09_aisle_map_v18.json"])
    ap.add_argument("--dock-house", default="tunnel-09")
    ap.add_argument("--dock-offset", default="2.0,1.0", help="dock position relative to that door, house-local metres")
    ap.add_argument("--out", type=Path, default=ROOT / "data/site_map_v18_draft.json")
    a = ap.parse_args(argv)
    site, report = build(a.glb, a.layout if str(a.layout) else None, a.aisle_map, a.dock_house,
                         tuple(float(v) for v in a.dock_offset.split(",")))
    site.save(a.out)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
