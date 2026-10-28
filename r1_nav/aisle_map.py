"""Aisle map of one greenhouse, loaded from the team's r1-aisle-map.json (house-local metres, Z up, door at +Y)."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

ROBOT_WIDTH_M = 0.528          # R180 visual bounds (robot_design/.../visual_bounds.json)
SIDE_MARGIN_M = 0.06           # required clearance each side


@dataclass(frozen=True)
class Aisle:
    id: str
    label: str
    x_local: float             # lateral centre line
    y_start: float             # head (door side)
    y_end: float               # far end
    clearance_m: float         # bed-envelope to bed-envelope
    floor_z: float

    @property
    def length_m(self) -> float:
        return abs(self.y_start - self.y_end)

    def traversable(self, robot_width_m: float = ROBOT_WIDTH_M, margin_m: float = SIDE_MARGIN_M) -> bool:
        return self.clearance_m >= robot_width_m + 2 * margin_m

    def point_at(self, metre: float) -> tuple[float, float, float]:
        """House-local point `metre` metres into the aisle from its head (0 = head)."""
        if not 0 <= metre <= self.length_m:
            raise ValueError(f"metre {metre} outside aisle length {self.length_m:.2f}")
        direction = -1.0 if self.y_end < self.y_start else 1.0
        return (self.x_local, self.y_start + direction * metre, self.floor_z)


@dataclass(frozen=True)
class AisleMap:
    house_id: str
    aisles: tuple[Aisle, ...]

    @classmethod
    def load(cls, path: str | Path) -> "AisleMap":
        d = json.loads(Path(path).read_text())
        aisles = tuple(sorted((Aisle(a["id"], a["label"], a["start_local"][0], a["start_local"][1], a["end_local"][1],
                                     a["clearance_m"], a["floor_height_local_m"]) for a in d["aisles"]),
                              key=lambda a: a.x_local))
        return cls(d["source"]["feature_id"], aisles)

    def get(self, aisle_id: str) -> Aisle:
        for a in self.aisles:
            if a.id == aisle_id:
                return a
        raise KeyError(aisle_id)

    def traversable_aisles(self, **kw) -> list[Aisle]:
        return [a for a in self.aisles if a.traversable(**kw)]
