import math
from r1_nav.controller import wheel_command, ControllerConfig, MAX_WHEEL_RAD_S

N = 500
ANGLES = [(-math.pi + 2 * math.pi * i / N) for i in range(N)]


def scan(left, right, front=12.0):
    out = []
    for a in ANGLES:
        if abs(a) < math.radians(20):
            out.append(front)
        elif abs(a - math.pi / 2) < math.radians(25):
            out.append(left)
        elif abs(a + math.pi / 2) < math.radians(25):
            out.append(right)
        else:
            out.append(12.0)
    return out


def test_centred_drives_straight_at_cruise():
    cmd, info = wheel_command(scan(0.38, 0.38), ANGLES)
    assert abs(cmd[0] - cmd[1]) < 1e-9 and 1.9 < cmd[0] <= MAX_WHEEL_RAD_S and not info["stopped"]


def test_more_room_on_left_steers_left():
    cmd, _ = wheel_command(scan(0.5, 0.26), ANGLES)
    assert cmd[1] > cmd[0]            # right wheel faster -> turns left


def test_obstacle_ahead_stops():
    cmd, info = wheel_command(scan(0.38, 0.38, front=0.5), ANGLES)
    assert cmd == [0.0, 0.0] and info["stopped"]


def test_reverse_is_slower_and_negative():
    cmd, _ = wheel_command(scan(0.38, 0.38), ANGLES, reverse=True)
    assert cmd[0] < 0 and abs(cmd[0]) < 1.3


def test_commands_stay_within_env_bounds():
    for l, r in [(0.9, 0.1), (0.1, 0.9), (0.05, 0.05)]:
        cmd, _ = wheel_command(scan(l, r), ANGLES, ControllerConfig(k_lateral=10.0, max_diff_rad_s=5.0))
        assert max(abs(c) for c in cmd) <= MAX_WHEEL_RAD_S + 1e-9
