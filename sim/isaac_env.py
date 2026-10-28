"""Wrapper around Astra's R180 SensorStopEnv (diy_r50/physical_ai_runtime/sensor_env.py) for long aisle runs and
batch evaluation. Isaac modules are imported lazily, so the pure geometry here is importable (and tested) without Isaac.

What the env natively offers (read 2026-10-28): `SensorStopEnv(app, stage_path=, aisle_half_width=, body_half_length=)`,
`reset(*, seed=None, options={'obstacle_y': -2.0|-2.8|-3.2})`, `step([l, r])` → (obs, reward, terminated, truncated, info).
Its evaluator truncates at 300 steps and terminates on any approach to the fixed obstacle, so this wrapper swaps in a
GroundTruthEvaluator after every reset (evaluation-only world pose → info["gt"], never into obs). Friction, obstacle
placement, start offset and side cameras are applied from outside the env by editing the stage: all UNVERIFIED until
run on Isaac (see knobs / unsupported lists in reset())."""
from __future__ import annotations
import json
import math
import os
import sys
from pathlib import Path

# greenhouse/build_scene.py moves /Farm by -SCENE_ORIGIN so the robot starts at scene (0, 0, 0.034), facing -Y.
SCENE_ORIGIN_XY = (195.065185546875, 153.0)
ROBOT_START_SCENE_XY = (0.0, 0.0)
STAGES = {"default": "greenhouse/sensor_stop_fixture.usda",                      # what r180_env.make_env loads
          "photo": "scene_fix_2026-10-28/sensor_stop_fixture_pt.usda"}          # photo-matched (README criteria)
OBSTACLE_PRIM = "/TestFixtures/removable_front_obstacle"                         # cube 0.65 x 0.18 x 0.70 m
OBSTACLE_HALF_DEPTH_M = 0.09
FRICTION_PRIM = "/Robot/Materials/ContactEstimate"                               # robot contact material 0.7 / 0.6
BASE_FRICTION = (0.7, 0.6)                                                       # static, dynamic
SIDE_CAMERA_PRIMS = {f"mast_{side}_{h}": f"/Robot/Physics_Base/Sensors/plant_{side}_{h}"
                     for side in ("left", "right") for h in ("low", "mid", "high")}
STRIP_HALF_M, BODY_HALF_W, BODY_HALF_LEN = 0.38, 0.26395, 0.375                  # as Astra's route/evaluator checks
DEFAULT_R180_DIR = "/home/passin/Documents/Blender/TheFarm/robot_design/r1_v3/diy_r50/physical_ai_r180"


def house_translation(map_path: str | Path) -> tuple[float, float]:
    """World XY of the house-local origin, from the aisle map's house_local_to_world matrix."""
    m = json.loads(Path(map_path).read_text())["house_local_to_world"]
    return float(m[0][3]), float(m[1][3])


class SceneFrame:
    """Scene (stage) XY ↔ (aisle metre, lateral) for one aisle. lateral > 0 = scene +X = robot's left facing -Y."""

    def __init__(self, aisle, house_xy: tuple[float, float]):
        self.aisle = aisle
        self.dx, self.dy = SCENE_ORIGIN_XY[0] - house_xy[0], SCENE_ORIGIN_XY[1] - house_xy[1]
        self.sign = -1.0 if aisle.y_end < aisle.y_start else 1.0

    def to_aisle(self, sx: float, sy: float) -> tuple[float, float]:
        hx, hy = sx + self.dx, sy + self.dy
        return (hy - self.aisle.y_start) * self.sign, hx - self.aisle.x_local

    def to_scene(self, metre: float, lateral: float = 0.0) -> tuple[float, float]:
        return self.aisle.x_local + lateral - self.dx, self.aisle.y_start + self.sign * metre - self.dy

    @property
    def start_metre(self) -> float:
        return self.to_aisle(*ROBOT_START_SCENE_XY)[0]


def ground_truth(position, quat_wxyz, frame: SceneFrame) -> dict:
    """Evaluation-only robot state in aisle terms (robot forward = scene -Y at yaw 0)."""
    x, y, _ = (float(v) for v in position)
    w, qx, qy, qz = (float(v) for v in quat_wxyz)
    yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (qx * qx + qy * qy)))))
    metre, lateral = frame.to_aisle(x, y)
    halfx = BODY_HALF_W * abs(math.cos(yaw)) + BODY_HALF_LEN * abs(math.sin(yaw))
    return {"metre": metre, "lateral_m": lateral, "yaw_deg": math.degrees(yaw), "tilt_deg": tilt,
            "clearance_strip_m": STRIP_HALF_M - abs(lateral) - halfx,
            "clearance_bed_m": frame.aisle.clearance_m / 2 - abs(lateral) - halfx,
            "front_metre": metre + BODY_HALF_LEN, "scene_xy": [x, y]}


class GroundTruthEvaluator:
    """Drop-in for SensorStopEnv.evaluator: never ends an episode except on tip / non-finite physics."""

    def __init__(self, frame: SceneFrame, obstacle_face_metre: float | None = None, max_tilt_deg: float = 20.0):
        self.frame, self.obstacle_face, self.max_tilt = frame, obstacle_face_metre, max_tilt_deg
        self.last: dict | None = None

    def evaluate(self, position, quaternion_wxyz, wheel_velocity):
        vals = [float(v) for v in (*position, *quaternion_wxyz, *wheel_velocity)]
        if not all(math.isfinite(v) for v in vals):
            return 0.0, True, False, {"termination": "invalid_physics", "gt": self.last}
        gt = ground_truth(position, quaternion_wxyz, self.frame)
        gt["obstacle_face_metre"] = self.obstacle_face
        self.last = gt
        if gt["tilt_deg"] > self.max_tilt:
            return 0.0, True, False, {"termination": "tip", "gt": gt}
        return 0.0, False, False, {"termination": None, "gt": gt}


class R180Sim:
    """Backend-agnostic sim object (same interface as sim.fake_env.FakeR180Sim). Create the SimulationApp first."""
    backend = "isaac"

    def __init__(self, app, aisle, house_xy, r180_dir: str | None = None, stage: str = "default",
                 side_cameras: bool = True, settle_steps: int = 3, log=print):
        self.app, self.log, self.settle_steps = app, log, settle_steps
        d = Path(r180_dir or os.environ.get("R180_DIR", DEFAULT_R180_DIR)).resolve()
        for p in (d, d.parent / "physical_ai_runtime"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from sensor_env import SensorStopEnv                     # noqa: E402  (Isaac; needs a running SimulationApp)
        stage_path = Path(STAGES.get(stage, stage))
        stage_path = stage_path if stage_path.is_absolute() else d / stage_path
        self.env = SensorStopEnv(app, stage_path=stage_path, aisle_half_width=STRIP_HALF_M, body_half_length=BODY_HALF_LEN)
        self.stage, self.frame = self.env.stage, SceneFrame(aisle, house_xy)
        self.start_metre = self.frame.start_metre
        self.knobs = self._detect_knobs()
        self.side_cams: dict = {}
        self.has_side_cameras = False
        if side_cameras:
            self._attach_side_cameras()
        self._default_pose = None
        self.log(f"[R180Sim] stage={stage_path.name} start_metre={self.start_metre:.2f} knobs={self.knobs} "
                 f"side_cameras={sorted(self.side_cams) or 'NONE'}")

    # ---- capability detection ------------------------------------------------------------------------------------
    def _detect_knobs(self) -> dict:
        prim = self.stage.GetPrimAtPath
        fr = prim(FRICTION_PRIM)
        return {"seed": True,                                      # native: obstacle_y choice only
                "friction_scale": bool(fr and fr.GetAttribute("physics:staticFriction")),
                "obstacle": bool(prim(OBSTACLE_PRIM)),
                "start_offset": hasattr(self.env.robot, "set_default_state") or hasattr(self.env.robot, "set_world_poses"),
                "ground_truth": True}

    def _attach_side_cameras(self) -> None:
        # The six plant cameras are authored prims in robot_sensors.usda but SensorStopEnv only wires front_rgb.
        # UNVERIFIED: adding Camera objects before world.reset mirrors how the env adds front_rgb.
        try:
            from isaacsim.sensors.camera import Camera
            for name, path in SIDE_CAMERA_PRIMS.items():
                if self.stage.GetPrimAtPath(path):
                    self.side_cams[name] = self.env.world.scene.add(
                        Camera(prim_path=path, name=name, resolution=(180, 320), frequency=30))  # portrait 9:16
            self.has_side_cameras = len(self.side_cams) == len(SIDE_CAMERA_PRIMS)
        except Exception as e:                                     # keep running on front_rgb placeholders
            self.log(f"WARNING side cameras not attached ({e}); side captures will use front_rgb placeholders")
            self.side_cams, self.has_side_cameras = {}, False

    # ---- episode -------------------------------------------------------------------------------------------------
    def reset(self, seed=None, friction_scale: float | None = None, obstacle_metre: float | None = None,
              start_offset: tuple[float, float, float] = (0.0, 0.0, 0.0), jam_metre: float | None = None):
        """obstacle_metre: centre of the test box along the aisle (None = removed). start_offset: (along m, lateral m, yaw deg)."""
        from pxr import Gf
        unsupported = []
        if jam_metre is not None:
            unsupported.append("jam_metre")
        if friction_scale is not None:
            if self.knobs["friction_scale"]:                       # edit target is the session layer (set by env)
                p = self.stage.GetPrimAtPath(FRICTION_PRIM)
                p.GetAttribute("physics:staticFriction").Set(BASE_FRICTION[0] * friction_scale)
                p.GetAttribute("physics:dynamicFriction").Set(BASE_FRICTION[1] * friction_scale)
            else:
                unsupported.append("friction_scale")
        offset_applied = self._set_start_default(start_offset) if any(start_offset) else True
        obs, info = self.env.reset(seed=seed, options={"obstacle_y": -2.8})
        face = None
        if self.knobs["obstacle"]:
            if obstacle_metre is None:
                pos = (0.0, 0.0, -50.0)                            # park the box under the floor
            else:
                sx, sy = self.frame.to_scene(obstacle_metre)
                pos, face = (sx, sy, 0.35), obstacle_metre - OBSTACLE_HALF_DEPTH_M
            self.stage.GetPrimAtPath(OBSTACLE_PRIM).GetAttribute("xformOp:translate").Set(Gf.Vec3d(*pos))
        elif obstacle_metre is not None:
            unsupported.append("obstacle")
        if not offset_applied:
            offset_applied = self._teleport(start_offset)
            if not offset_applied:
                unsupported.append("start_offset")
        self.env.evaluator = GroundTruthEvaluator(self.frame, face)
        info = {"termination": None}
        for _ in range(self.settle_steps):                        # fresh sweep/images after stage edits
            obs, _, _, _, info = self.env.step([0.0, 0.0])
        return obs, {"backend": self.backend, "gt": info.get("gt"), "unsupported": unsupported,
                     "start_metre": self.start_metre}

    def _set_start_default(self, start_offset) -> bool:
        robot = self.env.robot
        if not hasattr(robot, "set_default_state"):
            return False
        try:
            import numpy as np
            if self._default_pose is None:
                st = robot.get_default_state()
                self._default_pose = (np.array(st.positions, dtype=float), np.array(st.orientations, dtype=float))
            pos, quat = self._pose_with_offset(*self._default_pose, start_offset)
            robot.set_default_state(positions=pos, orientations=quat)
            return True
        except Exception as e:
            self.log(f"WARNING start offset via default state failed: {e}")
            return False

    def _teleport(self, start_offset) -> bool:
        try:
            p, q = self.env.robot.get_world_poses()
            pos, quat = self._pose_with_offset(p, q, start_offset)
            self.env.robot.set_world_poses(positions=pos, orientations=quat)
            return True
        except Exception as e:
            self.log(f"WARNING start offset teleport failed: {e}")
            return False

    @staticmethod
    def _pose_with_offset(pos, quat, start_offset):
        import numpy as np
        d_m, lat, yaw_deg = start_offset
        pos = np.array(pos, dtype=float).reshape(1, 3).copy()
        pos[0, 0] += lat                                           # +lateral = scene +X
        pos[0, 1] -= d_m                                           # +metre = scene -Y (into the aisle)
        h = math.radians(yaw_deg) / 2
        w0, x0, y0, z0 = np.array(quat, dtype=float).reshape(4)
        w1, z1 = math.cos(h), math.sin(h)                          # yaw about Z, left-multiplied
        q = np.array([[w1 * w0 - z1 * z0, w1 * x0 - z1 * y0, w1 * y0 + z1 * x0, w1 * z0 + z1 * w0]])
        return pos, q

    def step(self, action):
        return self.env.step([float(action[0]), float(action[1])])

    def side_images(self) -> dict:
        out = {}
        for name, cam in self.side_cams.items():
            try:
                rgba = cam.get_rgba()
                if rgba is not None and getattr(rgba, "ndim", 0) == 3 and rgba.size:
                    out[name] = rgba[:, :, :3]
            except Exception:
                pass
        return out

    def close(self):
        self.env.close()
