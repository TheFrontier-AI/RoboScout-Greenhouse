from pathlib import Path
import pytest
from r1_nav.aisle_map import AisleMap
from r1_nav.mission import go_to, survey_aisle, survey_house, coverage_distance_m

MAP = Path(__file__).resolve().parents[1] / "data/tunnel09_aisle_map_v18.json"


def test_map_loads_three_aisles_left_to_right():
    m = AisleMap.load(MAP)
    assert m.house_id == "tunnel-09" and len(m.aisles) == 3
    assert [a.id for a in m.aisles] == ["tunnel-09/aisle-01", "tunnel-09/aisle-02", "tunnel-09/aisle-03"]
    assert all(abs(a.length_m - 38.04) < 0.01 for a in m.aisles)


def test_only_centre_aisle_fits_r180():
    m = AisleMap.load(MAP)
    assert [a.id for a in m.traversable_aisles()] == ["tunnel-09/aisle-02"]
    with pytest.raises(ValueError):
        go_to(m, "tunnel-09/aisle-01", 5.0)


def test_point_at_runs_from_door_into_house():
    a = AisleMap.load(MAP).get("tunnel-09/aisle-02")
    x, y0, _ = a.point_at(0.0); _, y10, _ = a.point_at(10.0)
    assert x == 0.0 and y0 > y10 and abs((y0 - y10) - 10.0) < 1e-6
    with pytest.raises(ValueError):
        a.point_at(40.0)


def test_survey_house_turns_around_in_the_wide_aisle_and_reports_skipped():
    m = AisleMap.load(MAP)
    legs, skipped = survey_house(m)
    kinds = [l.kind for l in legs]
    assert kinds == ["transit", "drive_in", "turn_around", "drive_out", "dock"]     # 1.27 m aisle fits a 1.04 m spin
    assert {a.id for a in skipped} == {"tunnel-09/aisle-01", "tunnel-09/aisle-03"}
    assert abs(coverage_distance_m(legs, m) - 2 * (38.04 - 0.8) - 0.5) < 0.05


def test_narrow_or_no_turn_survey_reverses_out():
    m = AisleMap.load(MAP)
    assert [l.kind for l in survey_aisle(m, "tunnel-09/aisle-02", turnaround=False)] == ["transit", "drive_in", "reverse_out"]
    assert m.get("tunnel-09/aisle-02").turnable() and not m.get("tunnel-09/aisle-01").traversable()


def test_farm_loop_any_number_of_houses():
    """Three (or more) houses: the drive-through alternates direction and crosses behind / in front, then docks."""
    from r1_nav.mission import farm_loop
    houses = [("tunnel-09/aisle-02", 0.0), ("tunnel-10/aisle-02", 9.0), ("tunnel-11/aisle-02", 18.0)]
    legs = farm_loop(houses, head_y=3.26, back_y=-35.19, dock_xy=(2.0, 4.9), road_y=8.99, rows_end_m=38.04)
    kinds = [(l.kind, l.aisle_id) for l in legs]
    drives = [k for k in kinds if k[0] in ("drive_in", "drive_out")]
    assert [a for _, a in drives] == [h for h, _ in houses]              # every house surveyed once, in order
    assert [k for k, _ in drives] == ["drive_in", "drive_out", "drive_in"]  # front->back, back->front, front->back
    assert kinds[-1][0] == "dock"
    assert sum(1 for k, _ in kinds if k == "path") >= len(houses)        # a path leg between houses and home
