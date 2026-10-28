"""Kinematic stand-in for the Isaac R180 env (NOT physics): straight aisle between two bed walls, optional front
obstacle, wheel slip and a jam point. Emits R180-format observations so the executor, runners and cloud link can be
tested without Isaac. Ground truth goes into info["gt"] only (same keys as sim.isaac_env.ground_truth)."""
from __future__ import annotations
import math
import random
from sim.executor import R180_LIDAR_FORWARD_RAD

WHEEL_R, TRACK_M = 0.127, 0.45
LIDAR_AHEAD_M, BODY_HALF_LEN, BODY_HALF_W = 0.34, 0.375, 0.26395
STRIP_HALF_M = 0.38            # concrete strip half-width used by Astra's route/evaluator checks
N_BEAMS = 500


class FakeR180Sim:
    backend = "fake"

    def __init__(self, aisle_length_m: float = 38.04, bed_half_m: float = 0.634, start_metre: float = 3.26,
                 dt: float = 0.1, side_cameras: bool = True, end_wall_metre: float | None = None):
        self.L, self.W, self.start_metre, self.dt = aisle_length_m, bed_half_m, start_metre, dt
        self.end_wall = end_wall_metre             # far end wall across the aisle (None = open)
        self.has_side_cameras = side_cameras
        self.knobs = {"seed": True, "friction_scale": True, "obstacle": True, "start_offset": True, "ground_truth": True}
        self.angles = [-math.pi + 2 * math.pi * i / N_BEAMS for i in range(N_BEAMS)]   # raw R180 azimuths
        self.reset()

    def reset(self, seed=None, friction_scale: float | None = None, obstacle_metre: float | None = None,
              start_offset: tuple[float, float, float] = (0.0, 0.0, 0.0), jam_metre: float | None = None):
        rng = random.Random(seed)
        d_m, lat, yaw_deg = start_offset
        self.s, self.y, self.psi = self.start_metre + d_m, lat, math.radians(yaw_deg)
        self.obstacle, self.jam = obstacle_metre, jam_metre
        mu = 1.0 if friction_scale is None else friction_scale
        self.slip = max(0.0, min(0.25, (0.6 - mu) * 0.5))          # illustrative only: wet floor → wheels over-read
        self.t, self.wl, self.wr, self.ol, self.or_ = 1.0, 0.0, 0.0, 0.0, 0.0
        self.noise = rng.uniform(0.0, 0.004)
        self._rng = rng
        return self._obs(), {"backend": self.backend, "gt": self.ground_truth(), "unsupported": []}

    def step(self, action):
        l, r = (max(-2.0, min(2.0, float(a))) for a in action)
        jammed = self.jam is not None and self.s >= self.jam and (l + r) > 0
        if jammed:
            l = r = 0.0                                          # wheels blocked: encoders stop too
        v = WHEEL_R * (l + r) / 2 * (1 - self.slip)
        w = WHEEL_R * (r - l) / TRACK_M * (1 - self.slip)
        self.psi += w * self.dt
        self.s += v * math.cos(self.psi) * self.dt
        self.y += v * math.sin(self.psi) * self.dt
        if self.obstacle is not None:                            # cannot drive through the box
            self.s = min(self.s, self.obstacle - 0.09 - BODY_HALF_LEN)
        self.wl, self.wr, self.ol, self.or_ = self.wl + l * self.dt, self.wr + r * self.dt, l, r
        self.t += self.dt
        gt = self.ground_truth()
        term = gt["clearance_bed_m"] < 0
        return self._obs(), 0.0, term, False, {"termination": "side_collision" if term else None, "gt": gt}

    def ground_truth(self) -> dict:
        halfx = BODY_HALF_W * abs(math.cos(self.psi)) + BODY_HALF_LEN * abs(math.sin(self.psi))
        return {"metre": self.s, "lateral_m": self.y, "yaw_deg": math.degrees(self.psi), "tilt_deg": 0.0,
                "clearance_strip_m": STRIP_HALF_M - abs(self.y) - halfx, "clearance_bed_m": self.W - abs(self.y) - halfx,
                "front_metre": self.s + BODY_HALF_LEN, "obstacle_face_metre": None if self.obstacle is None else self.obstacle - 0.09}

    def side_images(self) -> dict:
        return {} if not self.has_side_cameras else {c: self._image(60) for c in (
            "mast_left_low", "mast_left_mid", "mast_left_high", "mast_right_low", "mast_right_mid", "mast_right_high")}

    def close(self):
        pass

    # ---- sensors -------------------------------------------------------------------------------------------------
    def _ranges(self):
        lx, ly = self.s + LIDAR_AHEAD_M * math.cos(self.psi), self.y + LIDAR_AHEAD_M * math.sin(self.psi)
        out = []
        for a in self.angles:
            phi = self.psi + (a - R180_LIDAR_FORWARD_RAD)          # beam direction in aisle frame (0 = into aisle)
            sn, cs, best = math.sin(phi), math.cos(phi), 12.0
            if sn > 1e-6:
                best = min(best, (self.W - ly) / sn)
            elif sn < -1e-6:
                best = min(best, (self.W + ly) / -sn)
            if self.end_wall is not None and cs > 1e-6:
                best = min(best, (self.end_wall - lx) / cs)
            if self.obstacle is not None and cs > 1e-6:
                d = (self.obstacle - 0.09 - lx) / cs
                if d > 0 and abs(ly + d * sn) < 0.325:
                    best = min(best, d)
            out.append(max(0.05, best + self._rng.gauss(0.0, self.noise)))
        return out

    def _image(self, shade: int):
        try:
            import numpy as np
            img = np.zeros((180, 320, 3), dtype=np.uint8)
            img[:, :, 1] = shade + int(self.s * 4) % 100
            img[90:, :, 0] = 120
            return img
        except ImportError:
            return [[[shade, 120, 60]] * 32] * 18

    def _obs(self) -> dict:
        return {"timestamp_s": self.t, "lidar_m": self._ranges(), "lidar_angles_rad": list(self.angles),
                "imu_accel_gyro": [0.0, 0.0, 9.81, 0.0, 0.0, WHEEL_R * (self.or_ - self.ol) / TRACK_M * (1 - self.slip)],
                "wheel_position_velocity": [self.wl, self.wr, self.wl, self.wr, self.ol, self.or_, self.ol, self.or_],
                "front_rgb": self._image(100)}
