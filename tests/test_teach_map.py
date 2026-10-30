"""Mapping by demonstration (scripts/teach_map.py) on a tiny synthetic taught drive: no files, no GPU."""
import math, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.teach_map import beam_angles, build_map, scan_xy, site_skeleton, to_scene  # noqa: E402

N = 500


def corridor_scan(x, y, heading_deg, half_w=1.0):
    """Ranges of a 2 m wide corridor along +Y seen from (x, y): walls at x = -half_w and x = +half_w."""
    ang = beam_angles(N); h = math.radians(heading_deg); out = []
    for a in ang:
        d = np.array([math.cos(a + h), math.sin(a + h)])           # beam direction in the world
        r = 12.0
        for wall in (-half_w, half_w):
            if abs(d[0]) > 1e-6:
                t = (wall - x) / d[0]
                if t > 0.3:
                    r = min(r, t)
        out.append(round(r, 3) if r < 12.0 else None)
    return out


def taught_drive():
    steps = []
    t = 0.0
    for i in range(60):                                             # outdoors: drive +Y from the dock
        y = 0.1 * i; steps.append({"t": t, "leg": "path", "aisle": None, "xy": [0.0, y], "heading_deg": 90.0, "lidar": corridor_scan(0.0, y, 90.0)}); t += 0.1
    for i in range(100):                                            # inside house A, aisle metre counting up
        y = 6.0 + 0.1 * i; steps.append({"t": t, "leg": "drive_in", "aisle": "house-A/aisle-01", "metre": 0.1 * i, "xy": [0.0, y], "heading_deg": 90.0, "lidar": corridor_scan(0.0, y, 90.0)}); t += 0.1
    for i in range(30):                                             # back outdoors, return
        y = 16.0 - 0.1 * i; steps.append({"t": t, "leg": "path", "aisle": None, "xy": [0.0, y], "heading_deg": -90.0, "lidar": corridor_scan(0.0, y, -90.0)}); t += 0.1
    return steps


def test_scan_projection_puts_walls_where_they_are():
    pts = to_scene(scan_xy(corridor_scan(0.0, 3.0, 90.0), beam_angles(N)), 0.0, 3.0, 90.0)
    assert np.all(np.abs(np.abs(pts[:, 0]) - 1.0) < 0.05)            # every point sits on a wall at x = +-1


def test_build_map_and_site_skeleton():
    steps = taught_drive()
    m = build_map(steps, every=1, cell=0.1, min_hits=2)
    assert len(m) > 50 and np.all(np.abs(np.abs(m[:, 0]) - 1.0) < 0.1)
    site = site_skeleton(steps, "robot")
    ids = [n["id"] for n in site["nodes"]]
    assert ids[0] == "dock" and "house-A/door-1" in ids and "house-A/door-2" in ids
    assert site["houses_taught"]["house-A"]["aisles"]["house-A/aisle-01"]["length_m"] == 9.9
    from r1_nav.site_map import SiteMap
    sm = SiteMap.from_dict(site)
    assert sm.houses() == ["house-A"] and not sm.confirmed and not sm.validate()


def test_route_from_off_path_position():
    """A robot relocalised 2 m beside the taught path gets a 'rejoin' leg to the nearest node, then the normal route."""
    import math
    from r1_nav.site_map import NoRoute, SiteMap
    site = site_skeleton(taught_drive(), "robot"); sm = SiteMap.from_dict(site)
    door1 = next(n["id"] for n in site["nodes"] if n["id"].endswith("door-1")); door2 = next(n["id"] for n in site["nodes"] if n["id"].endswith("door-2"))
    xy = (2.0, sm.nodes[door1].xy[1])                                 # 2 m to the side of door-1
    r = sm.route_from_xy(xy, door2)
    assert r.nodes[0] == "__here__" and r.nodes[1] == door1 and r.nodes[-1] == door2
    assert abs(r.edges[0].length_m - 2.0) < 0.05 and r.length_m > 2.0
    try:
        sm.route_from_xy((40.0, 0.0), door2); assert False, "should refuse a far-away start"
    except NoRoute:
        pass
