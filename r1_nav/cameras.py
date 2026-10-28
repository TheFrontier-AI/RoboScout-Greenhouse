"""R1 camera rig: 6 side (mast) cameras look at the plants, 1 front camera drives/verifies navigation.
Decides which cameras fire at a given aisle metre and builds the frame metadata sent to the cloud (docs/INTERFACES_th.md)."""
from __future__ import annotations
from dataclasses import dataclass, asdict

SIDE_CAMERAS = ("mast_left_low", "mast_left_mid", "mast_left_high",
                "mast_right_low", "mast_right_mid", "mast_right_high")
FRONT_CAMERA = "front"
ALL_CAMERAS = SIDE_CAMERAS + (FRONT_CAMERA,)
SIDE_STEP_M = 0.5              # plant-inspection spacing (README success criterion 3)
PLANT_PITCH_M = 0.45           # tomato pitch in the rows (84 plants per 38 m): stop-to-shoot spacing
FRONT_STEP_M = 2.0             # navigation check spacing (cloud verifies the robot moved/turned as commanded)


@dataclass(frozen=True)
class FrameMeta:
    ts: float
    house: str
    aisle: str
    metre: float
    heading_deg: float
    camera: str
    frame_id: int
    pose_confidence: float     # 0..1 from SLAM; < 0.5 → cloud must not pin findings to this metre
    purpose: str               # inspect | navigate | revisit

    def to_dict(self) -> dict:
        return asdict(self)


def due_cameras(metre: float, last_side_m: float | None, last_front_m: float | None,
                side_step_m: float = SIDE_STEP_M) -> list[str]:
    """Cameras that should capture now, given the metres of the previous side/front captures."""
    due: list[str] = []
    if last_side_m is None or abs(metre - last_side_m) >= side_step_m - 1e-9:
        due += SIDE_CAMERAS
    if last_front_m is None or abs(metre - last_front_m) >= FRONT_STEP_M - 1e-9:
        due.append(FRONT_CAMERA)
    return due


def frame_meta(ts: float, house: str, aisle: str, metre: float, heading_deg: float, camera: str, frame_id: int,
               pose_confidence: float, purpose: str = "inspect") -> FrameMeta:
    if camera not in ALL_CAMERAS:
        raise ValueError(f"unknown camera {camera}")
    if purpose not in ("inspect", "navigate", "revisit"):
        raise ValueError(f"unknown purpose {purpose}")
    return FrameMeta(ts, house, aisle, round(metre, 2), round(heading_deg, 1), camera, frame_id,
                     round(max(0.0, min(1.0, pose_confidence)), 3),
                     "navigate" if camera == FRONT_CAMERA and purpose == "inspect" else purpose)
