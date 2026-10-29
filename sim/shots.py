"""Extra presentation cameras along the farm route (added to the robot's own cameras, never replacing them):
fixed CCTV (high, wide, like a farm security camera) and handheld "bystander" views (eye height, following the robot,
slight hand shake, no person modelled). Scene frame = farm plan: x east, y north, z up; tunnel-09 aisle at x 0,
tunnel-10 aisle at x 9.015, front doors y 3.67, back doors y -35.19, yard connector y ~ -37.5, dock (2.0, 4.9).

Each CCTV records only while the robot is inside its watch zone (like a motion-triggered camera); bystanders film
while the robot is within their range. Every clip is raw, real-time footage of the recorded route."""
from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Shot:
    name: str
    kind: str                       # "cctv" (fixed) | "bystander" (eye height, follows the robot)
    eye: tuple[float, float, float]
    look: tuple[float, float, float] | None   # fixed target for CCTV; None = aim at the robot
    hfov_deg: float
    zone: tuple[float, float, float, float]   # robot x0, x1, y0, y1 that triggers recording
    note: str


SHOTS = [
    Shot("cctv1_t09_front", "cctv", (3.6, 8.2, 3.0), (0.4, 3.4, 0.4), 95, (-2, 5, 0, 9), "pole in front of tunnel-09: leaving the dock, entering the front door"),
    Shot("cctv2_t09_head", "cctv", (0.0, 3.35, 2.4), (0.0, -8.0, 0.3), 80, (-1, 1, -16, 3.4), "inside tunnel-09 at the front end, looking down aisle-02"),
    Shot("cctv3_t09_tail", "cctv", (0.0, -34.9, 2.4), (0.0, -24.0, 0.3), 80, (-1, 1, -35, -18), "inside tunnel-09 at the back end, looking up the aisle"),
    Shot("cctv4_yard", "cctv", (4.5, -41.5, 4.2), (4.5, -36.3, 0.2), 100, (-2, 11, -42, -33), "pole in the back yard: out of 09, across, into 10"),
    Shot("cctv5_t10_tail", "cctv", (9.015, -34.9, 2.4), (9.015, -24.0, 0.3), 80, (8, 10, -35, -18), "inside tunnel-10 at the back end, looking up the aisle"),
    Shot("cctv6_t10_head", "cctv", (9.015, 3.35, 2.4), (9.015, -8.0, 0.3), 80, (8, 10, -16, 3.4), "inside tunnel-10 at the front end, looking down the aisle"),
    Shot("cctv7_front_overview", "cctv", (4.5, 13.0, 5.5), (4.5, 3.5, 0.3), 100, (-3, 12, 0, 10), "high pole over the front yard: both front doors and the dock"),
    Shot("person1_t09_door", "bystander", (2.3, 6.6, 1.6), None, 55, (-2, 4, -2, 8), "someone filming by tunnel-09's front door"),
    Shot("person2_yard", "bystander", (6.8, -39.8, 1.6), None, 55, (-1, 10, -40, -33), "someone filming in the back yard"),
    Shot("person3_t10_door", "bystander", (11.3, 6.4, 1.6), None, 55, (2, 11, -2, 8), "someone filming by tunnel-10's front door and the dock"),
]


def in_zone(shot: Shot, x: float, y: float) -> bool:
    x0, x1, y0, y1 = shot.zone
    return x0 <= x <= x1 and y0 <= y <= y1


def handheld(t: float, amp_m: float = 0.012, amp_deg: float = 0.35):
    """Deterministic small hand shake (two incommensurate sines per axis): position offset (m), aim offset (deg)."""
    s = lambda f, p: math.sin(2 * math.pi * f * t + p)
    pos = (amp_m * s(0.37, 0.1) + 0.4 * amp_m * s(1.13, 2.0), amp_m * s(0.29, 1.3), amp_m * s(0.43, 0.7) + 0.3 * amp_m * s(1.7, 0.2))
    ang = (amp_deg * s(0.31, 0.5), amp_deg * s(0.47, 1.9))
    return pos, ang


class ShotCameras:
    """USD cameras + render products for SHOTS (session layer). update(t, robot_xyz) aims bystanders at the robot."""

    def __init__(self, stage, shots=SHOTS, res=(1920, 1080), aperture_mm: float = 36.0):
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom
        self.shots, self.res, self.ops, self.rp, self.ann = list(shots), res, {}, {}, {}
        for s in self.shots:
            cam = UsdGeom.Camera.Define(stage, f"/ShotCams/{s.name}")
            cam.CreateHorizontalApertureAttr(aperture_mm); cam.CreateVerticalApertureAttr(aperture_mm * res[1] / res[0])
            cam.CreateFocalLengthAttr(aperture_mm / 2 / math.tan(math.radians(s.hfov_deg / 2)))
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500.0))
            self.ops[s.name] = UsdGeom.Xformable(cam.GetPrim()).AddTransformOp()
            self.rp[s.name] = rep.create.render_product(f"/ShotCams/{s.name}", res)
            a = rep.AnnotatorRegistry.get_annotator("rgb"); a.attach([self.rp[s.name]]); self.ann[s.name] = a
            if s.look is not None:
                self._aim(s.name, s.eye, s.look)

    def _aim(self, name, eye, look):
        from pxr import Gf
        self.ops[name].Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*look), Gf.Vec3d(0, 0, 1)).GetInverse())

    def update(self, t: float, robot_xyz) -> None:
        for s in self.shots:
            if s.kind != "bystander":
                continue
            (dx, dy, dz), (dyaw, dpitch) = handheld(t)
            eye = (s.eye[0] + dx, s.eye[1] + dy, s.eye[2] + dz)
            tx, ty, tz = float(robot_xyz[0]), float(robot_xyz[1]), 0.55
            d = math.hypot(tx - eye[0], ty - eye[1]) or 1.0
            yaw = math.atan2(ty - eye[1], tx - eye[0]) + math.radians(dyaw)
            pitch = math.atan2(tz - eye[2], d) + math.radians(dpitch)
            look = (eye[0] + d * math.cos(yaw) * math.cos(pitch), eye[1] + d * math.sin(yaw) * math.cos(pitch), eye[2] + d * math.sin(pitch))
            self._aim(s.name, eye, look)

    def textures(self, names):
        return [self.rp[n].hydra_texture for n in names]

    def image(self, name):
        import numpy as np
        d = np.asarray(self.ann[name].get_data())
        return d[..., :3] if d.ndim == 3 else d
