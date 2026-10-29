"""Global relocalisation (sim/relocalize.py) on a synthetic taught drive: a corridor with one side opening.
No files, no GPU."""
import math, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.teach_map import beam_angles, build_map, scan_xy  # noqa: E402
from sim.relocalize import HypothesisTracker, Relocalizer      # noqa: E402

N = 500


def world_scan(x, y, heading_deg):
    """Corridor along +Y between x = -1 and x = +1 for y in [0, 20], with an opening (no wall) on the right
    for y in [8, 10] - the one feature that makes position along the corridor observable."""
    ang = beam_angles(N); h = math.radians(heading_deg); out = []
    for a in ang:
        d = np.array([math.cos(a + h), math.sin(a + h)]); r = 12.0
        for wall in (-1.0, 1.0):
            if abs(d[0]) > 1e-6:
                t = (wall - x) / d[0]
                if t > 0.3:
                    yy = y + t * d[1]
                    if wall == 1.0 and 8.0 <= yy <= 10.0:
                        continue                      # the opening
                    if 0.0 <= yy <= 20.0:
                        r = min(r, t)
        out.append(round(r, 3) if r < 12.0 else None)
    return out


def taught():
    steps = []
    for i in range(190):
        y = 0.5 + 0.1 * i
        steps.append({"t": 0.1 * i, "leg": "path", "xy": [0.0, y], "heading_deg": 90.0, "lidar": world_scan(0.0, y, 90.0)})
    return steps


def test_relocalise_from_scratch_near_the_feature():
    steps = taught(); m = build_map(steps, every=1, cell=0.1, min_hits=2)
    path = np.array([[s["xy"][0], s["xy"][1], s["heading_deg"]] for s in steps])
    reloc = Relocalizer(m, path, spacing_m=0.5)
    r = reloc.locate(scan_xy(world_scan(0.0, 8.5, 90.0), beam_angles(N)))     # the opening is in view: unambiguous
    assert r.pose is not None and not r.ambiguous
    assert math.hypot(r.pose[0] - 0.0, r.pose[1] - 8.5) < 0.6


def test_motion_resolves_a_featureless_start():
    steps = taught(); m = build_map(steps, every=1, cell=0.1, min_hits=2)
    path = np.array([[s["xy"][0], s["xy"][1], s["heading_deg"]] for s in steps])
    reloc = Relocalizer(m, path, spacing_m=0.5); tr = HypothesisTracker(reloc)
    y0 = 3.0                                                        # far from the opening: every y looks the same
    tr.start(scan_xy(world_scan(0.0, y0, 90.0), beam_angles(N)), (0.0, y0, 90.0))
    decided = None
    for k in range(1, 80):                                          # drive +Y 0.1 m per step past the opening
        y = y0 + 0.1 * k
        t = tr.update(scan_xy(world_scan(0.0, y, 90.0), beam_angles(N)), (0.0, y, 90.0))
        if t.decided and t.metres > 1.0:
            decided = (t, y); break
    assert decided is not None, "the tracker never decided"
    t, y = decided
    assert math.hypot(t.pose[0] - 0.0, t.pose[1] - y) < 0.8
