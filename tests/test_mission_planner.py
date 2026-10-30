"""The mission planner on a tiny synthetic map: two aisles between walls, a dock in the yard, no stored route."""
import json, math, sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.free_nav import FreeSpace, find_aisles, wall_direction   # noqa: E402
from agent.mission_planner import MissionPlanner                      # noqa: E402


def _toy(tmp_path: Path):
    # walls along y: two aisles 1.6 m wide between three rows of "stems" (x = 0, 2, 4) and outer walls at x = -1, 5
    pts = []
    for x in (-1.0, 5.0):
        pts += [(x, y) for y in np.arange(-10, 10, 0.05)]
    for x in (0.0, 2.0, 4.0):                       # plant rows: stems every 0.45 m plus a bed edge 0.3 m each side
        for y in np.arange(-10, 10, 0.45):
            pts += [(x, y), (x - 0.3, y + 0.1), (x + 0.3, y + 0.1)]
    m = np.array(pts)
    # scans: the robot drove the two aisles (x = 1, x = 3) and the yard (y = 12), looking around
    poses, scans = [], []
    from scripts.teach_map import beam_angles, to_scene   # noqa
    ang = beam_angles(500)
    def fake_scan(x, y, h):
        # ranges from ray-casting against the toy walls (brute force on a KD-tree bin)
        from scipy.spatial import cKDTree
        t = cKDTree(m); out = []
        for k, a in enumerate(ang):
            if k % 5:
                out.append(None); continue                   # every fifth beam is enough for the toy
            w = math.radians(h) + a; best = None
            for r in np.arange(0.3, 11.0, 0.1):
                px, py = x + 0.34 * math.cos(math.radians(h)) + r * math.cos(w), y + 0.34 * math.sin(math.radians(h)) + r * math.sin(w)
                if t.query((px, py))[0] < 0.08:
                    best = r; break
            out.append(best)
        return out
    for x in (1.0, 3.0):
        for y in np.arange(-9, 10, 2.0):
            for h in (90.0, -90.0):
                poses.append((x, y, h)); scans.append(fake_scan(x, y, h))
    for x in np.arange(-0.5, 5.0, 1.0):
        for h in (0.0, 90.0, 180.0, -90.0):
            poses.append((x, 12.0, h)); scans.append(fake_scan(x, 12.0, h))
    fs = FreeSpace(m, poses, scans); theta = wall_direction(m); aisles = find_aisles(fs, theta, min_len_m=8.0)
    d = tmp_path / "structure"; d.mkdir()
    np.save(d / "free_grid.npy", fs.free); np.save(d / "map_xy.npy", m)
    json.dump({"x0": fs.x0, "y0": fs.y0, "cell": fs.cell, "wall_direction_deg": theta, "aisles": aisles, "dock": [2.0, 12.0, -90.0]}, open(d / "structure.json", "w"))
    return d, aisles


def test_planner_finds_aisles_and_plans_without_a_route(tmp_path):
    d, aisles = _toy(tmp_path)
    plant = [a for a in aisles if a["kind"] == "plant aisle"]
    assert len(plant) == 2, aisles
    mp = MissionPlanner(d)
    st = mp.structure(); assert len(st["plant_aisles"]) == 2 and st["dock"]["xy"] == [2.0, 12.0]
    p = mp.plan([a["id"] for a in mp.aisles], (2.0, 12.0))
    assert p["ok"] and all(l["metres"] for l in p["legs"]) and p["legs"][-1]["to"] == "dock" and p["total_m"] > 40
    alt = mp.alternatives((2.0, 12.0)); assert len(alt) == 2 and alt[0]["total_m"] <= alt[1]["total_m"]
    bad = mp.plan(["aisle-99"], (2.0, 12.0)); assert not bad["ok"]
    assert mp.picture(p["order"], (2.0, 12.0))[:3] == b"\xff\xd8\xff"       # a JPEG for HERMES to look at
