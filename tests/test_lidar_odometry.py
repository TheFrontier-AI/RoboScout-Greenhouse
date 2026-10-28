"""LiDAR odometry (scan matching on the stems) keeps the true metre when the wheels slip."""
from r1_nav.mission import Leg
from sim.executor import ExecutorConfig, LegExecutor, WheelImuOdometry
from sim.fake_env import FakeR180Sim
from sim.lidar_odometry import LidarWheelOdometry

A = "tunnel-09/aisle-02"


def drive(pose, friction, target=12.0):
    sim = FakeR180Sim(aisle_length_m=20.0, start_metre=1.0, plant_pitch_m=0.45)
    obs, _ = sim.reset(seed=1, friction_scale=friction)
    ex = LegExecutor(pose, "tunnel-09", ExecutorConfig(stop_to_shoot=False))
    ex.load([Leg("drive_in", A, target)], "survey")
    gt = None
    for _ in range(6000):
        if not ex.busy:
            break
        cmd, _ = ex.step(obs)
        obs, _, _, _, info = sim.step(cmd); gt = info["gt"]
    return ex, gt


def test_wheels_alone_overshoot_belief_on_wet_floor():
    ex, gt = drive(WheelImuOdometry(metre=1.0), friction=0.4)
    assert gt["metre"] < 12.0 - 0.5                       # robot believes 12 m but is really short of it


def test_lidar_odometry_holds_true_metre_and_reports_slip():
    pose = LidarWheelOdometry(metre=1.0)
    ex, gt = drive(pose, friction=0.4)
    assert ex.status == "done" and abs(gt["metre"] - 12.0) <= 0.15        # wheels alone: ~1.1 m short
    assert pose.lidar_used > 0.8 * pose.lidar_total and any(e["type"] == "slip" for e in pose.events)


def test_dry_floor_lidar_and_wheels_agree():
    pose = LidarWheelOdometry(metre=1.0)
    ex, gt = drive(pose, friction=1.0)
    assert abs(gt["metre"] - 12.0) <= 0.15 and not any(e["type"] == "slip" for e in pose.events)
