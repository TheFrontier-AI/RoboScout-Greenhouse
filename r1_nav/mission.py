"""Turn a high-level mission into an ordered list of legs. Aisles wide enough to spin in place: drive in to the end, turn
around 180 deg, drive forward out (second camera pass, no long blind reverse). Narrower aisles: drive in, reverse out."""
from __future__ import annotations
from dataclasses import dataclass
from .aisle_map import AisleMap, Aisle

HEADLAND_OFFSET_M = 2.0        # how far beyond the aisle head the robot waits/turns (door-side yard)
END_STOP_M = 0.8               # stop this far before the far end of the beds (end wall + spin clearance)
EXIT_METRE = -0.5              # drive_out target: just past the bed ends, back at the head


@dataclass(frozen=True)
class Leg:
    kind: str                  # drive_in | reverse_out | turn_around | drive_out | transit | dock | hold | stop | discover
    aisle_id: str | None
    target_metre: float | None
    note: str = ""


@dataclass(frozen=True)
class SiteLeg(Leg):
    """Farm-level leg between greenhouses (kinds: exit_house | site_transit | enter_house), site-map node ids."""
    house: str | None = None
    from_node: str | None = None
    to_node: str | None = None
    length_m: float | None = None


def go_to(amap: AisleMap, aisle_id: str, metre: float) -> list[Leg]:
    a = amap.get(aisle_id)
    if not a.traversable():
        raise ValueError(f"{aisle_id} clearance {a.clearance_m:.2f} m is too narrow for the robot")
    a.point_at(metre)
    return [Leg("transit", aisle_id, 0.0, "reach aisle head on the headland and align"),
            Leg("drive_in", aisle_id, metre, "sensor-only centring, stop at metre")]


def survey_aisle(amap: AisleMap, aisle_id: str, turnaround: bool = True) -> list[Leg]:
    a = amap.get(aisle_id)
    if not a.traversable():
        raise ValueError(f"{aisle_id} clearance {a.clearance_m:.2f} m is too narrow for the robot")
    if turnaround and a.turnable():
        end = round(a.length_m - END_STOP_M, 2)
        return [Leg("transit", aisle_id, 0.0, "align at head"),
                Leg("drive_in", aisle_id, end, "stop at every plant, capture all side cameras"),
                Leg("turn_around", aisle_id, end, "spin 180 deg in place, verify with gyro + LiDAR"),
                Leg("drive_out", aisle_id, EXIT_METRE, "drive forward out, second camera pass")]
    return [Leg("transit", aisle_id, 0.0, "align at head"),
            Leg("drive_in", aisle_id, a.length_m, "stop at every plant, capture all side cameras"),
            Leg("reverse_out", aisle_id, -HEADLAND_OFFSET_M, "too narrow to turn: reverse until clear of the head")]


def survey_house(amap: AisleMap) -> tuple[list[Leg], list[Aisle]]:
    """Cover every traversable aisle left→right; returns legs and the aisles that were skipped (too narrow)."""
    legs: list[Leg] = []
    skipped = [a for a in amap.aisles if not a.traversable()]
    for a in amap.traversable_aisles():
        legs += survey_aisle(amap, a.id)
    legs.append(Leg("dock", None, None, "return to dock"))
    return legs, skipped


def coverage_distance_m(legs: list[Leg], amap: AisleMap) -> float:
    total = 0.0
    for leg in legs:
        if leg.kind == "drive_in":
            total += leg.target_metre
        elif leg.kind == "reverse_out":
            total += amap.get(leg.aisle_id).length_m + HEADLAND_OFFSET_M
        elif leg.kind == "drive_out":
            total += amap.get(leg.aisle_id).length_m - END_STOP_M - leg.target_metre
    return total
