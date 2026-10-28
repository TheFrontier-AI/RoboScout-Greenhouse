from pathlib import Path
from r1_nav.aisle_map import AisleMap
from r1_nav.commands import plan_command
from r1_nav.cameras import due_cameras, frame_meta, SIDE_CAMERAS, FRONT_CAMERA

MAP = AisleMap.load(Path(__file__).resolve().parents[1] / "data/tunnel09_aisle_map_v18.json")


def test_revisit_goes_then_holds():
    r = plan_command({"cmd": "revisit", "house": "tunnel-09", "aisle": "tunnel-09/aisle-02", "metre": 15}, MAP)
    assert r.status == "accepted" and [l.kind for l in r.legs] == ["transit", "drive_in", "hold"]
    assert r.legs[-1].target_metre == 15.0


def test_rejects_narrow_aisle_bad_metre_unknown_cmd_and_wrong_house():
    assert plan_command({"cmd": "go_to", "aisle": "tunnel-09/aisle-01", "metre": 3}, MAP).status == "rejected"
    assert plan_command({"cmd": "go_to", "aisle": "tunnel-09/aisle-02", "metre": 99}, MAP).status == "rejected"
    assert plan_command({"cmd": "go_to", "aisle": "nope", "metre": 1}, MAP).status == "rejected"
    assert plan_command({"cmd": "set_wheels", "left": 2}, MAP).status == "rejected"
    assert plan_command({"cmd": "survey_house", "house": "tunnel-07"}, MAP).status == "rejected"


def test_without_map_only_discovery_dock_stop():
    assert plan_command({"cmd": "survey_house"}, None).status == "rejected"
    for c in ("discover_house", "return_to_dock", "stop"):
        assert plan_command({"cmd": c}, None).status == "accepted"


def test_capture_schedule_side_every_half_metre_front_every_two():
    assert due_cameras(0.0, None, None) == list(SIDE_CAMERAS) + [FRONT_CAMERA]
    assert due_cameras(0.3, 0.0, 0.0) == []
    assert due_cameras(0.5, 0.0, 0.0) == list(SIDE_CAMERAS)
    assert due_cameras(2.0, 1.5, 0.0) == list(SIDE_CAMERAS) + [FRONT_CAMERA]


def test_front_frames_are_navigation_and_confidence_clamped():
    m = frame_meta(1.0, "tunnel-09", "tunnel-09/aisle-02", 15.04, 270.0, "front", 7, 1.7)
    assert m.purpose == "navigate" and m.pose_confidence == 1.0 and m.metre == 15.04
