"""Validate HERMES (cloud) goal commands and turn them into mission legs. The cloud only sends goals, never wheel
commands; anything that does not fit the confirmed aisle map is rejected with a reason (docs/INTERFACES_th.md §1)."""
from __future__ import annotations
from dataclasses import dataclass, field
from .aisle_map import AisleMap
from .mission import Leg, go_to, survey_aisle, survey_house

COMMANDS = ("go_to", "survey_aisle", "survey_house", "revisit", "discover_house", "return_to_dock", "stop")
REVISIT_HOLD_S = 3.0           # dwell at the revisit point so all 6 side cameras get a steady shot


@dataclass
class Reply:
    status: str                # accepted | rejected
    reason: str = ""
    legs: list[Leg] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"status": self.status, "reason": self.reason, "legs": [l.__dict__ for l in self.legs]}


def _reject(reason: str) -> Reply:
    return Reply("rejected", reason)


def plan_command(cmd: dict, amap: AisleMap | None) -> Reply:
    """Map one command dict to legs. `amap` is None before the house has been discovered and confirmed."""
    name = cmd.get("cmd")
    if name not in COMMANDS:
        return _reject(f"unknown cmd {name!r}; allowed: {', '.join(COMMANDS)}")
    if name == "stop":
        return Reply("accepted", "stop now", [Leg("stop", None, None, "halt and hold")])
    if name == "return_to_dock":
        return Reply("accepted", "", [Leg("dock", None, None, "return to dock")])
    if name == "discover_house":
        return Reply("accepted", "map unknown house with SLAM", [Leg("discover", None, None, "follow headland, probe every opening")])
    if amap is None:
        return _reject("no confirmed aisle map yet; send discover_house first")
    if cmd.get("house") not in (None, amap.house_id):
        return _reject(f"robot is in {amap.house_id}, not {cmd.get('house')}")
    try:
        if name == "survey_house":
            legs, skipped = survey_house(amap)
            note = f"skipping too-narrow aisles: {', '.join(a.id for a in skipped)}" if skipped else ""
            return Reply("accepted", note, legs)
        aisle = cmd.get("aisle")
        if name == "survey_aisle":
            return Reply("accepted", "", survey_aisle(amap, aisle))
        metre = float(cmd["metre"])
        legs = go_to(amap, aisle, metre)
        if name == "revisit":
            legs.append(Leg("hold", aisle, metre, f"capture all side cameras for {REVISIT_HOLD_S:.0f} s (purpose=revisit)"))
        return Reply("accepted", "", legs)
    except KeyError as e:
        return _reject(f"missing or unknown field/aisle: {e}")
    except (TypeError, ValueError) as e:
        return _reject(str(e))
