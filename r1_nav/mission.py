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


@dataclass(frozen=True)
class PathLeg(Leg):
    """Yard/road leg outside the houses: follow waypoints (scene metres) then face final_heading_deg (0 = +X, CCW)."""
    waypoints: tuple = ()
    final_heading_deg: float | None = None


def farm_loop(houses: list[tuple[str, float]], head_y: float, back_y: float, dock_xy: tuple[float, float],
              road_y: float, rows_end_m: float, yard_offset_m: float = 1.6) -> list[Leg]:
    """Drive-through survey of several tunnels whose ends are open: enter house 1 at the front, survey its aisle to
    the back, cross behind the houses, enter house 2 at the back and survey it toward the front, ... then home.
    houses: [(aisle_id, aisle centre x)], all aisles parallel to Y with the door (head, metre 0) at head_y and the back
    opening at back_y. Metres are measured from the head in every house, so a back-to-front pass counts down."""
    legs: list[Leg] = []
    beyond_back = (head_y - back_y) + yard_offset_m           # metre just outside the back opening
    beyond_head = -yard_offset_m
    x0, y0 = dock_xy
    for i, (aisle_id, x) in enumerate(houses):
        front_to_back = i % 2 == 0
        if i == 0:
            legs.append(PathLeg("path", None, None, "leave the dock, line up with the door",
                                waypoints=((x0, max(y0, head_y + yard_offset_m)), (x, head_y + yard_offset_m)),
                                final_heading_deg=-90.0))
            legs.append(Leg("enter", aisle_id, beyond_head, "enter through the front door"))
        if front_to_back:
            legs.append(Leg("drive_in", aisle_id, beyond_back, "stop at every plant, drive out of the back"))
        else:
            legs.append(Leg("drive_out", aisle_id, beyond_head, "stop at every plant, drive out of the front"))
        if i + 1 < len(houses):
            nxt = houses[i + 1][1]
            yy = back_y - yard_offset_m if front_to_back else head_y + yard_offset_m
            legs.append(PathLeg("path", None, None, "cross to the next house",
                                waypoints=((x, yy), (nxt, yy)), final_heading_deg=90.0 if front_to_back else -90.0))
            legs.append(Leg("enter", houses[i + 1][0], beyond_back if front_to_back else beyond_head,
                            "enter through the " + ("back" if front_to_back else "front")))
        else:
            yy = back_y - yard_offset_m if front_to_back else head_y + yard_offset_m
            pts = ((x, yy),) + (((x + 4.0, yy), (x + 4.0, road_y), (x0, road_y)) if front_to_back else ((x0, yy),)) + ((x0, y0),)
            legs.append(PathLeg("path", None, None, "return to the dock", waypoints=pts, final_heading_deg=-90.0))
    legs.append(Leg("dock", None, None, "docked"))
    return legs


def revisit_from_yard(aisle_id: str, x: float, head_y: float, metre: float, here_xy: tuple[float, float],
                      yard_offset_m: float = 1.6, hold_note: str = "revisit") -> list[Leg]:
    """From the front yard / dock: to the door, in to the metre, hold for the revisit shots, turn, out, back."""
    return [PathLeg("path", None, None, "to the door", waypoints=((x, head_y + yard_offset_m),), final_heading_deg=-90.0),
            Leg("enter", aisle_id, -yard_offset_m, "enter through the front door"),
            Leg("drive_in", aisle_id, metre, "drive to the spot"),
            Leg("hold", aisle_id, metre, hold_note),
            Leg("turn_around", aisle_id, metre, "turn to leave"),
            Leg("drive_out", aisle_id, -yard_offset_m, "drive out of the front"),
            PathLeg("path", None, None, "back to the dock", waypoints=(here_xy,), final_heading_deg=-90.0)]


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
