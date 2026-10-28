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


def test_side_cameras_at_every_plant_while_stopped_front_every_two():
    ex, ev, cmds, gts = run([Leg("drive_in", A, 6.0)])
    caps = kinds(ev, "capture")
    side = [c["meta"]["metre"] for c in caps if c["camera"] == SIDE_CAMERAS[0]]
    front = [c for c in caps if c["camera"] == FRONT_CAMERA]
    assert len(side) == 11 and all(0.4 <= b - a <= 0.55 for a, b in zip(side, side[1:]))   # 0.45 m plant pitch
    assert sum(1 for c in cmds if c == [0.0, 0.0]) >= 11 * 4                                 # stood still >= 0.5 s per shot
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
    ex.load([Leg("drive_in", A, 9.0)], "go_to", )
    for _ in range(8):                                   # first stop-to-shoot at the start metre, then it drives
        cmd, _ = ex.step(obs); obs, *_ = sim.step(cmd)
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


def test_turn_around_spins_180_verifies_with_lidar_then_drives_out_forward():
    ex, ev, cmds, gts = run([Leg("drive_in", A, 8.0), Leg("turn_around", A, 8.0), Leg("drive_out", A, 2.0)], max_steps=8000)
    assert ex.status == "done", ex.reason
    chk = kinds(ev, "turn_check")[0]
    assert chk["ok"] and abs(chk["gyro_turned_deg"] - 180) <= 2.5 and abs(chk["lidar_yaw_deg"]) <= 4
    assert abs(gts[-1]["metre"] - 2.0) <= 0.15 and abs(abs(gts[-1]["yaw_deg"]) - 180) < 6
    assert min(g["clearance_bed_m"] for g in gts) > 0.0
    out = [c for c in kinds(ev, "capture") if c["meta"]["metre"] < 7.5 and c["camera"] == SIDE_CAMERAS[0]]
    assert len(out) >= 10                                 # second camera pass on the way out
    assert not any(c[0] < 0 and c[1] < 0 for c in cmds)   # never reversed


def test_turn_refused_when_spin_circle_is_blocked():
    sim = FakeR180Sim(aisle_length_m=20.0, start_metre=1.0, bed_half_m=0.45)   # 0.9 m aisle: a spin needs ~1.02 m
    obs, _ = sim.reset()
    ex = LegExecutor(WheelImuOdometry(metre=1.0), "tunnel-09")
    ex.load([Leg("turn_around", A, 1.0)], "survey")
    cmd, ev = ex.step(obs)
    assert cmd == [0.0, 0.0] and ex.status == "aborted" and "no room to turn" in ex.reason


def test_end_wall_pose_check_corrects_wheel_slip():
    sim = FakeR180Sim(aisle_length_m=20.0, start_metre=1.0, end_wall_metre=20.4)
    obs, _ = sim.reset(friction_scale=0.45)                # wet floor: wheels over-read ~7 %
    cfg = ExecutorConfig(end_wall_metre=20.4)
    ex = LegExecutor(WheelImuOdometry(metre=1.0), "tunnel-09", cfg)
    ex.load([Leg("drive_in", A, 19.0)], "survey")
    ev, gt = [], None
    for _ in range(9000):
        if not ex.busy:
            break
        cmd, e = ex.step(obs); ev += e
        obs, _, _, _, info = sim.step(cmd); gt = info["gt"]
    checks = kinds(ev, "pose_check")
    assert checks and checks[0]["corrected"] and checks[0]["error_m"] > 0.3   # odometry ran ahead of the truth
    assert ex.status == "done" and abs(gt["metre"] - 19.0) <= 0.15            # stopped at the real metre anyway


def test_large_pose_disagreement_stops_the_robot():
    sim = FakeR180Sim(aisle_length_m=20.0, start_metre=1.0, end_wall_metre=20.4)
    obs, _ = sim.reset(friction_scale=0.1)                 # 25 % slip: odometry far off
    ex = LegExecutor(WheelImuOdometry(metre=1.0), "tunnel-09", ExecutorConfig(end_wall_metre=20.4))
    ex.load([Leg("drive_in", A, 19.0)], "survey")
    for _ in range(9000):
        if not ex.busy:
            break
        cmd, _ = ex.step(obs); obs, *_ = sim.step(cmd)
    assert ex.status == "aborted" and "position" in ex.reason
