"""Scan-to-map ICP recovers x, y and heading from a wrong odometry guess (skid-steer scrub, gyro drift)."""
import math
import numpy as np
from sim.scan_localizer import LIDAR_AHEAD_M, ScanToMapLocalizer


def farm_map():
    rng = np.random.default_rng(0)
    pts = []
    for cx in (0.0, 9.0):                                    # two tunnels: side walls, end walls with a doorway, posts
        for y in np.arange(-35, 3.6, 0.05):
            pts += [(cx - 3.25, y), (cx + 3.25, y)]
        for x in np.arange(cx - 3.25, cx + 3.25, 0.05):
            if abs(x - cx) > 0.6:
                pts += [(x, 3.66), (x, -35.19)]
        for y in np.arange(-34, 3, 1.5):
            pts += [(cx - 1.5 + 0.03 * math.cos(a), y + 0.03 * math.sin(a)) for a in np.linspace(0, 6.28, 8)]
    pts += [(x, 9.0) for x in np.arange(-6, 15, 0.05)]       # road edge
    return np.array(pts) + rng.normal(0, 0.005, (len(pts), 2))


def scan_from(map_xy, x, y, h_deg, rmax=8.0):
    h = math.radians(h_deg)
    lx, ly = x + LIDAR_AHEAD_M * math.cos(h), y + LIDAR_AHEAD_M * math.sin(h)
    d = map_xy - (lx, ly)
    near = np.hypot(d[:, 0], d[:, 1]) < rmax
    c, s = math.cos(-h), math.sin(-h)
    return d[near] @ np.array([[c, -s], [s, c]]).T


def test_recovers_pose_outside_the_houses():
    m = farm_map(); loc = ScanToMapLocalizer(m)
    truth = (4.5, 5.2, -90.0)
    (x, y, h), q = loc.localize(scan_from(m, *truth), 4.5 + 0.35, 5.2 - 0.3, -90.0 + 9.0)
    assert q["ok"] and math.hypot(x - truth[0], y - truth[1]) < 0.05 and abs(h - truth[2]) < 0.7


def test_rejects_a_scan_that_does_not_fit():
    m = farm_map(); loc = ScanToMapLocalizer(m)
    junk = np.random.default_rng(1).uniform(-6, 6, (300, 2))
    (x, y, h), q = loc.localize(junk, 4.5, 5.2, -90.0)
    assert not q["ok"] and (x, y, h) == (4.5, 5.2, -90.0)
