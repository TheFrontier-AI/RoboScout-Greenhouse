"""Leg executor on a kinematic fake aisle (no Isaac): corridor driving, stop accuracy, obstacle, stuck, hold, stop."""
import math
import pytest
from r1_nav.cameras import SIDE_CAMERAS, FRONT_CAMERA
from r1_nav.mission import Leg
from sim.executor import (ExecutorConfig, LegExecutor, WheelImuOdometry, R180_LIDAR_FORWARD_RAD, scan_from_obs,
                          wall_yaw)
from sim.fake_env import FakeR180Sim

A = "tunnel-09/aisle-02"


def run(legs, max_steps=4000, cfg=None, **reset_kw):
    sim = FakeR180Sim(aisle_length_m=20.0, start_metre=1.0)
    obs, info = sim.reset(seed=0, **reset_kw)
    ex = LegExecutor(WheelImuOdometry(metre=sim.start_metre), "tunnel-09", cfg or ExecutorConfig(blocked_timeout_s=3.0))
    ex.load(legs, "test")
    events, cmds, gts = [], [], []
    for _ in range(max_steps):
        if not ex.busy:
            break
        cmd, ev = ex.step(obs)
        events += ev; cmds.append(cmd)
        obs, _, term, _, info = sim.step(cmd)
        gts.append(info["gt"])
        assert not term, "robot left the aisle"
    return ex, events, cmds, gts


def kinds(events, t):
    return [e for e in events if e["type"] == t]


def test_scan_rotation_puts_r180_forward_at_zero():
    obs = {"lidar_m": [1.0, float("inf")], "lidar_angles_rad": [R180_LIDAR_FORWARD_RAD, 0.0]}
    r, a = scan_from_obs(obs)
    assert r == [1.0, 12.0] and abs(a[0]) < 1e-9 and abs(a[1] - math.pi / 2) < 1e-9     # raw 0 rad = robot left


def test_wall_yaw_sees_robot_turned_left():
    sim = FakeR180Sim()
    obs, _ = sim.reset(start_offset=(0.0, 0.0, 6.0))
    assert abs(wall_yaw(*scan_from_obs(obs)) - math.radians(6.0)) < math.radians(1.0)


def test_drive_in_stops_within_tolerance_and_stays_centred():
    ex, ev, cmds, gts = run([Leg("drive_in", A, 9.0)], start_offset=(0.0, 0.06, 3.0))
    assert ex.status == "done"
    assert abs(gts[-1]["metre"] - 9.0) <= 0.15 and kinds(ev, "leg_done")[0]["within_tolerance"]
    assert min(g["clearance_bed_m"] for g in gts) > 0.05 and abs(gts[-1]["lateral_m"]) < 0.03
    assert all(abs(c) <= 2.0 for cmd in cmds for c in cmd)


def test_side_cameras_every_half_metre_front_every_two():
    ex, ev, _, _ = run([Leg("drive_in", A, 6.0)])
    caps = kinds(ev, "capture")
    side = [c["meta"]["metre"] for c in caps if c["camera"] == SIDE_CAMERAS[0]]
    front = [c for c in caps if c["camera"] == FRONT_CAMERA]
    assert len(side) == 10 and all(0.45 <= b - a <= 0.6 for a, b in zip(side, side[1:]))
    assert len(front) == 3 and all(c["meta"]["purpose"] == "navigate" for c in front)
    assert len({c["meta"]["frame_id"] for c in caps}) == len(caps)


def test_reverse_out_goes_backwards_to_target():
    ex, ev, cmds, gts = run([Leg("drive_in", A, 5.0), Leg("reverse_out", A, 1.5)])
    assert ex.status == "done" and abs(gts[-1]["metre"] - 1.5) <= 0.15
    assert any(c[0] < 0 and c[1] < 0 for c in cmds)
    rev_caps = [e for e in kinds(ev, "capture") if e["meta"]["metre"] < 4.9]
    assert rev_caps and all(e["camera"] == FRONT_CAMERA for e in rev_caps[-1:])


def test_front_obstacle_stops_before_contact_then_aborts_after_timeout():
    ex, ev, _, gts = run([Leg("drive_in", A, 15.0)], obstacle_metre=6.0)
    assert ex.status == "aborted" and ex.reason.startswith("obstacle")
    assert kinds(ev, "obstacle_stop")
    gap = gts[-1]["obstacle_face_metre"] - gts[-1]["front_metre"]
    assert 0.3 < gap < 0.7


def test_stuck_when_wheels_commanded_but_no_progress():
    ex, ev, cmds, _ = run([Leg("drive_in", A, 15.0)], jam_metre=4.0)
    assert ex.status == "stuck" and kinds(ev, "stuck") and cmds[-1] == [0.0, 0.0]
    assert ex.legs_remaining == 1


def test_hold_keeps_still_and_captures_six_side_cameras_as_revisit():
    ex, ev, cmds, gts = run([Leg("hold", A, 1.0)])
    assert ex.status == "done" and all(c == [0.0, 0.0] for c in cmds)
    caps = kinds(ev, "capture")
    assert sorted(c["camera"] for c in caps) == sorted(SIDE_CAMERAS)
    assert all(c["meta"]["purpose"] == "revisit" for c in caps)
    assert 2.9 <= kinds(ev, "leg_done")[0]["held_s"] <= 3.2


def test_stop_leg_aborts_and_transit_is_marked_not_simulated():
    ex, ev, _, _ = run([Leg("transit", A, 0.0), Leg("stop", None, None), Leg("drive_in", A, 9.0)])
    assert ex.status == "aborted" and ex.reason == "stop command" and ex.legs_remaining == 0
    assert kinds(ev, "leg_done")[0]["note"] == "headland transit not simulated"


def test_external_abort_and_status_dict():
    sim = FakeR180Sim()
    obs, _ = sim.reset()
    ex = LegExecutor(WheelImuOdometry(metre=3.0), "tunnel-09")
    assert ex.step(obs) == ([0.0, 0.0], []) and ex.status == "idle"
    ex.load([Leg("drive_in", A, 9.0)], "go_to")
    cmd, _ = ex.step(obs)
    assert cmd[0] > 0 and ex.status_dict()["state"] == "drive_in" and ex.status_dict()["legs_remaining"] == 1
    ex.abort("stop command")
    assert ex.step(obs)[0] == [0.0, 0.0] and ex.status_dict()["state"] == "aborted"


def test_odometry_tracks_metre_and_confidence_decays():
    odo = WheelImuOdometry(metre=2.0)
    base = {"imu_accel_gyro": [0, 0, 9.81, 0, 0, 0.0]}
    odo.update({**base, "timestamp_s": 0.0, "wheel_position_velocity": [0.0] * 8})
    for k in range(1, 41):                                    # 40 × 0.25 rad = 10 rad; Isaac wraps shaft angles at ±pi
        a = (0.25 * k + math.pi) % (2 * math.pi) - math.pi
        m, h, c = odo.update({**base, "timestamp_s": 0.1 * k, "wheel_position_velocity": [a] * 4 + [0.0] * 4})
    assert m == pytest.approx(2.0 + 1.27) and h == pytest.approx(-90.0) and c < 1.0


def test_drive_in_to_a_metre_behind_the_robot_reverses_to_it():
    ex, ev, cmds, gts = run([Leg("drive_in", A, 6.0), Leg("drive_in", A, 3.0)])
    assert ex.status == "done" and abs(gts[-1]["metre"] - 3.0) <= 0.15
    assert kinds(ev, "note") and cmds[-2][0] < 0
