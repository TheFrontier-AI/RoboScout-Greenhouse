"""Turn a high-level mission into an ordered list of legs. No U-turns inside aisles: drive in forward, reverse out."""
from __future__ import annotations
from dataclasses import dataclass
from .aisle_map import AisleMap, Aisle

HEADLAND_OFFSET_M = 2.0        # how far beyond the aisle head the robot waits/turns (door-side yard)


@dataclass(frozen=True)
class Leg:
    kind: str                  # drive_in | reverse_out | transit | dock | hold | stop | discover (+ SiteLeg kinds)
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


def survey_aisle(amap: AisleMap, aisle_id: str) -> list[Leg]:
    a = amap.get(aisle_id)
    if not a.traversable():
        raise ValueError(f"{aisle_id} clearance {a.clearance_m:.2f} m is too narrow for the robot")
    return [Leg("transit", aisle_id, 0.0, "align at head"),
            Leg("drive_in", aisle_id, a.length_m, "capture every 0.5 m"),
            Leg("reverse_out", aisle_id, -HEADLAND_OFFSET_M, "reverse until clear of the head")]


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
    return total
