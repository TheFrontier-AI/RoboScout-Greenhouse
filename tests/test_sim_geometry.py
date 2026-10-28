"""Scene ↔ aisle frame used by the Isaac wrapper and evaluator (pure math, no Isaac)."""
import math
from pathlib import Path
import pytest
from r1_nav.aisle_map import AisleMap
from sim.isaac_env import SceneFrame, house_translation, ground_truth, GroundTruthEvaluator

MAP = Path(__file__).resolve().parents[1] / "data/tunnel09_aisle_map_v18.json"


def frame():
    return SceneFrame(AisleMap.load(MAP).get("tunnel-09/aisle-02"), house_translation(MAP))


def test_robot_start_is_about_3_3_m_into_centre_aisle():
    assert frame().start_metre == pytest.approx(3.26, abs=0.02)


def test_round_trip_and_forward_is_minus_y():
    f = frame()
    sx, sy = f.to_scene(15.0, 0.1)
    assert f.to_aisle(sx, sy) == pytest.approx((15.0, 0.1))
    assert f.to_scene(16.0)[1] < f.to_scene(15.0)[1]


def test_ground_truth_clearance_and_tip():
    f = frame()
    sx, sy = f.to_scene(10.0, 0.05)
    gt = ground_truth((sx, sy, 0.03), (1, 0, 0, 0), f)
    assert gt["metre"] == pytest.approx(10.0) and gt["clearance_strip_m"] == pytest.approx(0.38 - 0.05 - 0.26395)
    s = math.sin(math.radians(30) / 2)
    _, term, _, info = GroundTruthEvaluator(f).evaluate((sx, sy, 0.03), (math.cos(math.radians(15)), s, 0, 0), [0] * 4)
    assert term and info["termination"] == "tip"
