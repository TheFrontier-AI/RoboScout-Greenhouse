"""Steppable leg executor: drives r1_nav mission legs on any robot that yields R180-style observations.
Pure Python (no Isaac). One call per control tick: `cmd, events = ex.step(obs)`; cmd = [left, right] wheel rad/s.
Position along the aisle comes from a pluggable PoseSource (wheel+IMU odometry now, SLAM later); never sim ground truth."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Protocol
from r1_nav.cameras import FRONT_CAMERA, SIDE_CAMERAS, due_cameras, frame_meta
from r1_nav.commands import REVISIT_HOLD_S
from r1_nav.controller import MAX_WHEEL_RAD_S, WHEEL_RADIUS_M, ControllerConfig, sector_min, wheel_command
from r1_nav.mission import Leg

# R180 planar LiDAR reports azimuth with robot forward (body -Y) at -pi/2 (physical_ai_runtime/stop_controller.py).
# r1_nav.controller expects 0 = forward, +pi/2 = left, so scans are rotated by -R180_LIDAR_FORWARD_RAD.
R180_LIDAR_FORWARD_RAD = -math.pi / 2
INTO_AISLE_HEADING_DEG = -90.0  # house frame (0 = +X, CCW): aisles run from the door toward -Y
NOT_SIMULATED = ("transit", "dock", "discover")
STATUSES = ("idle", "running", "done", "stuck", "aborted")


def _window_median(ranges, angles, centre: float, half: float = math.radians(4)):
    vals = sorted(r for r, a in zip(ranges, angles) if abs((a - centre + math.pi) % (2 * math.pi) - math.pi) <= half)
    return vals[len(vals) // 2] if vals else None


def wall_yaw(ranges, angles, max_range_m: float = 3.0) -> float | None:
    """Robot yaw (rad, + = turned left) relative to the bed walls, from two LiDAR samples per side (r1_nav angles).
    Returns None when no side wall is within max_range_m (open headland)."""
    yaws = []
    for side in (1.0, -1.0):
        fa, ra = side * math.radians(60), side * math.radians(120)
        rf, rr = _window_median(ranges, angles, fa), _window_median(ranges, angles, ra)
        if rf is None or rr is None or rf > max_range_m or rr > max_range_m:
            continue
        dx, dy = rf * math.cos(fa) - rr * math.cos(ra), rf * math.sin(fa) - rr * math.sin(ra)
        yaws.append(-math.atan2(dy, dx))
    return sum(yaws) / len(yaws) if yaws else None


def scan_from_obs(obs, forward_rad: float = R180_LIDAR_FORWARD_RAD, max_range_m: float = 12.0):
    """(ranges, angles) in r1_nav convention from an R180 observation; inf/NaN become no-return (max range)."""
    ranges = [float(r) if math.isfinite(r) else max_range_m for r in obs["lidar_m"]]
    angles = [(float(a) - forward_rad + math.pi) % (2 * math.pi) - math.pi for a in obs["lidar_angles_rad"]]
    return ranges, angles


class PoseSource(Protocol):
    def reset(self, metre: float, heading_deg: float | None = None) -> None: ...
    def update(self, obs) -> tuple[float, float, float]:
        """Return (metre along current aisle, heading_deg in house frame, confidence 0..1)."""
        ...


class WheelImuOdometry:
    """Fallback pose: mean wheel angle x radius along the aisle, heading from integrated gyro z.
    Wheel slip (wet floor) is invisible to it, so confidence decays with distance travelled since the last reset."""

    def __init__(self, metre: float = 0.0, heading_deg: float = INTO_AISLE_HEADING_DEG, wheel_radius_m: float = WHEEL_RADIUS_M,
                 gyro_z_index: int = 5, gyro_sign: float = 1.0, drift_per_m: float = 0.005, min_confidence: float = 0.3):
        self.r, self.gz, self.gsign = wheel_radius_m, gyro_z_index, gyro_sign
        self.drift_per_m, self.min_conf = drift_per_m, min_confidence
        self.reset(metre, heading_deg)

    def reset(self, metre: float, heading_deg: float | None = None) -> None:
        self.metre, self.travelled = float(metre), 0.0
        if heading_deg is not None:
            self.heading = float(heading_deg)
        self._wheel = self._t = None

    def update(self, obs) -> tuple[float, float, float]:
        wpv, t = obs["wheel_position_velocity"], float(obs["timestamp_s"])
        wheel = [float(x) for x in wpv[:4]]                # LF, RF, LR, RR shaft angles (rad), + = forward; wrap at ±pi
        if self._wheel is not None:
            dt = t - self._t
            if 0.0 < dt < 1.0:
                self.heading += math.degrees(self.gsign * float(obs["imu_accel_gyro"][self.gz]) * dt)
            # per-wheel delta unwrapped to (-pi, pi]: one control step turns a wheel far less than half a revolution
            d = sum((w - p + math.pi) % (2 * math.pi) - math.pi for w, p in zip(wheel, self._wheel)) / 4.0 * self.r
            self.metre += d * math.cos(math.radians(self.heading - INTO_AISLE_HEADING_DEG))
            self.travelled += abs(d)
        self._wheel, self._t = wheel, t
        return self.metre, self.heading, max(self.min_conf, 1.0 - self.drift_per_m * self.travelled)


@dataclass
class ExecutorConfig:
    tolerance_m: float = 0.15           # README success criterion 1
    stop_lead_m: float = 0.03           # declare arrival this much early; wheels coast a few mm after zero command
    slow_zone_m: float = 0.6            # ramp speed down over the last metres before the target
    min_speed_frac: float = 0.3
    stuck_window_s: float = 5.0         # wheels commanded but pose advanced < stuck_min_progress_m in this window
    stuck_min_progress_m: float = 0.05
    blocked_timeout_s: float | None = 60.0   # obstacle still there after this → aborted (None = wait forever)
    hold_s: float = REVISIT_HOLD_S
    hold_capture_after_s: float = 1.0   # let the robot settle before the revisit shots
    rear_check: bool = True             # SAFETY while reversing (controller only guards the front)
    rear_self_m: float = 0.72           # LiDAR sits 0.34 m ahead of centre, rear bumper 0.375 m behind: closer returns = own body
    rear_stop_m: float = 0.4            # clearance behind the rear bumper that stops a reverse
    rear_half_angle_rad: float = math.radians(15)
    lidar_forward_rad: float = R180_LIDAR_FORWARD_RAD
    # Heading damping: r1_nav.controller is pure lateral-P (an undamped oscillator in the aisle). The executor adds
    # -k_heading * yaw, yaw measured from the bed walls with the LiDAR (gyro/odometry heading as fallback).
    k_heading: float = 2.0              # wheel-speed difference (rad/s) per rad of yaw relative to the aisle axis
    wall_max_m: float = 3.0             # wall samples farther than this are ignored (open headland, gaps)
    controller: ControllerConfig = field(default_factory=ControllerConfig)


class LegExecutor:
    def __init__(self, pose: PoseSource, house: str, cfg: ExecutorConfig | None = None):
        self.pose, self.house, self.cfg = pose, house, cfg or ExecutorConfig()
        self.status, self.reason, self.cmd_name = "idle", "", None
        self.legs: list[Leg] = []
        self.i, self.frame_id = 0, 0
        self.aisle: str | None = None
        self.metre, self.heading, self.confidence = 0.0, INTO_AISLE_HEADING_DEG, 0.0
        self._reset_leg()

    # ---- public -------------------------------------------------------------------------------------------------
    def load(self, legs: list[Leg], cmd_name: str | None = None) -> None:
        self.legs, self.i, self.cmd_name, self.reason = list(legs), 0, cmd_name, ""
        self.status = "running" if self.legs else "done"
        self._reset_leg()

    def abort(self, reason: str) -> list[dict]:
        self.status, self.reason, self.legs, self.i = "aborted", reason, [], 0
        self._reset_leg()
        return [{"type": "aborted", "reason": reason}]

    @property
    def busy(self) -> bool:
        return self.status == "running"

    @property
    def current_leg(self) -> Leg | None:
        return self.legs[self.i] if self.status == "running" and self.i < len(self.legs) else None

    @property
    def legs_remaining(self) -> int:
        return max(0, len(self.legs) - self.i)

    def status_dict(self) -> dict:
        leg = self.current_leg
        return {"state": leg.kind if leg else self.status, "status": self.status, "reason": self.reason,
                "cmd": self.cmd_name, "house": self.house, "aisle": self.aisle, "metre": round(self.metre, 2),
                "heading_deg": round(self.heading, 1), "pose_confidence": round(self.confidence, 3),
                "leg": leg.__dict__ if leg else None, "legs_remaining": self.legs_remaining if self.busy else 0}

    def step(self, obs) -> tuple[list[float], list[dict]]:
        self.metre, self.heading, self.confidence = self.pose.update(obs)
        events: list[dict] = []
        leg = self.current_leg
        if leg is None:
            return [0.0, 0.0], events
        t = float(obs["timestamp_s"])
        if self._t0 is None:
            self._t0 = t
            events.append({"type": "leg_start", "index": self.i, "leg": leg.__dict__, "metre": round(self.metre, 3)})
        if leg.kind in NOT_SIMULATED:
            # TODO(sim): headland transit / dock / SLAM discovery are not simulated: the R180 env has a single start
            # pose inside aisle-02 and no headland navigation. Marked done so mission sequencing can be exercised.
            if leg.aisle_id:
                self.aisle = leg.aisle_id
            self._finish(events, note="headland transit not simulated")
            return [0.0, 0.0], events
        if leg.kind == "stop":
            events += self.abort("stop command")
            return [0.0, 0.0], events
        if leg.kind == "hold":
            return self._hold(leg, t, events), events
        if leg.kind in ("drive_in", "reverse_out"):
            if self._reverse is None:
                self._reverse = leg.kind == "reverse_out"
                if not self._reverse and leg.target_metre < self.metre - self.cfg.tolerance_m:
                    # Robot is already past the goal (headland transit back to the head is not simulated): back up to it.
                    self._reverse = True
                    events.append({"type": "note", "note": "target behind robot: reversing to it"})
            return self._drive(leg, obs, t, events, reverse=self._reverse), events
        events += self.abort(f"unknown leg kind {leg.kind!r}")
        return [0.0, 0.0], events

    # ---- legs ---------------------------------------------------------------------------------------------------
    def _reset_leg(self) -> None:
        self._t0 = self._blocked_since = self._anchor = None
        self._last_side = self._last_front = None
        self._held_capture = False
        self._reverse = None

    def _finish(self, events: list[dict], **info) -> None:
        events.append({"type": "leg_done", "index": self.i, "leg": self.legs[self.i].__dict__,
                       "metre": round(self.metre, 3), **info})
        self.i += 1
        self._reset_leg()
        if self.i >= len(self.legs):
            self.status = "done"

    def _terminal(self, status: str, reason: str, events: list[dict]) -> list[float]:
        self.status, self.reason = status, reason
        events.append({"type": status, "reason": reason, "metre": round(self.metre, 3)})
        return [0.0, 0.0]

    def _hold(self, leg: Leg, t: float, events: list[dict]) -> list[float]:
        self.aisle = leg.aisle_id or self.aisle
        if not self._held_capture and t - self._t0 >= min(self.cfg.hold_capture_after_s, self.cfg.hold_s):
            self._capture(SIDE_CAMERAS, t, "revisit", events)
            self._held_capture = True
        if t - self._t0 >= self.cfg.hold_s:
            self._finish(events, held_s=round(t - self._t0, 2))
        return [0.0, 0.0]

    def _drive(self, leg: Leg, obs, t: float, events: list[dict], reverse: bool) -> list[float]:
        c = self.cfg
        self.aisle = leg.aisle_id or self.aisle
        remaining = (self.metre - leg.target_metre) if reverse else (leg.target_metre - self.metre)
        if remaining <= c.stop_lead_m:
            err = self.metre - leg.target_metre
            self._finish(events, error_m=round(err, 3), within_tolerance=abs(err) <= c.tolerance_m)
            return [0.0, 0.0]
        ranges, angles = scan_from_obs(obs, c.lidar_forward_rad)
        cmd, info = wheel_command(ranges, angles, c.controller, reverse=reverse)
        blocked, why = info["stopped"], f"front {info['front_m']:.2f} m"
        if reverse and c.rear_check:
            rear = sector_min([r if r >= c.rear_self_m else 99.0 for r in ranges], angles, math.pi, c.rear_half_angle_rad)
            if rear < c.rear_self_m + c.rear_stop_m:
                blocked, why = True, f"rear {rear - c.rear_self_m:.2f} m behind bumper"
        if blocked:
            self._anchor = (t, self.metre)                     # not commanding wheels: no stuck timer
            if self._blocked_since is None:
                self._blocked_since = t
                events.append({"type": "obstacle_stop", "why": why, "metre": round(self.metre, 3), "t": t})
            elif c.blocked_timeout_s is not None and t - self._blocked_since >= c.blocked_timeout_s:
                return self._terminal("aborted", f"obstacle did not clear in {c.blocked_timeout_s:.0f} s ({why})", events)
            return [0.0, 0.0]
        if self._blocked_since is not None:
            events.append({"type": "obstacle_clear", "waited_s": round(t - self._blocked_since, 2), "metre": round(self.metre, 3)})
            self._blocked_since = None
        if self._anchor is None or abs(self.metre - self._anchor[1]) >= c.stuck_min_progress_m:
            self._anchor = (t, self.metre)
        elif t - self._anchor[0] >= c.stuck_window_s:
            return self._terminal("stuck", f"no progress for {c.stuck_window_s:.0f} s with wheels commanded", events)
        yaw = wall_yaw(ranges, angles, c.wall_max_m)
        if yaw is None:
            yaw = math.radians((self.heading - INTO_AISLE_HEADING_DEG + 180) % 360 - 180)
        damp = max(-c.controller.max_diff_rad_s, min(c.controller.max_diff_rad_s, -c.k_heading * yaw))
        cmd = [cmd[0] - damp / 2, cmd[1] + damp / 2]
        scale = max(1.0, abs(cmd[0]) / MAX_WHEEL_RAD_S, abs(cmd[1]) / MAX_WHEEL_RAD_S)
        frac = max(c.min_speed_frac, min(1.0, remaining / c.slow_zone_m)) / scale
        if reverse:
            if self._last_front is None or abs(self.metre - self._last_front) >= 2.0 - 1e-9:
                self._capture((FRONT_CAMERA,), t, "navigate", events)
        else:
            self._capture(due_cameras(self.metre, self._last_side, self._last_front), t, "inspect", events)
        return [cmd[0] * frac, cmd[1] * frac]

    def _capture(self, cameras, t: float, purpose: str, events: list[dict]) -> None:
        for cam in cameras:
            self.frame_id += 1
            meta = frame_meta(t, self.house, self.aisle or "", self.metre, self.heading, cam, self.frame_id,
                              self.confidence, purpose)
            events.append({"type": "capture", "camera": cam, "meta": meta.to_dict()})
            if cam == FRONT_CAMERA:
                self._last_front = self.metre
            else:
                self._last_side = self.metre
