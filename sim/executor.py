"""Steppable leg executor: drives r1_nav mission legs on any robot that yields R180-style observations.
Pure Python (no Isaac). One call per control tick: `cmd, events = ex.step(obs)`; cmd = [left, right] wheel rad/s.
Position along the aisle comes from a pluggable PoseSource (wheel+IMU odometry now, SLAM later); never sim ground truth."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Protocol
from r1_nav.cameras import FRONT_CAMERA, PLANT_PITCH_M, SIDE_CAMERAS, due_cameras, frame_meta
from r1_nav.commands import REVISIT_HOLD_S
from r1_nav.controller import MAX_WHEEL_RAD_S, WHEEL_RADIUS_M, ControllerConfig, sector_min, wheel_command
from r1_nav.mission import Leg, PathLeg

# R180 planar LiDAR reports azimuth with robot forward (body -Y) at -pi/2 (physical_ai_runtime/stop_controller.py).
# r1_nav.controller expects 0 = forward, +pi/2 = left, so scans are rotated by -R180_LIDAR_FORWARD_RAD.
R180_LIDAR_FORWARD_RAD = -math.pi / 2
INTO_AISLE_HEADING_DEG = -90.0  # house frame (0 = +X, CCW): aisles run from the door toward -Y
NOT_SIMULATED = ("transit", "dock", "discover", "exit_house", "site_transit", "enter_house")
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


def front_wall_range(ranges, angles, half: float = math.radians(3)) -> float | None:
    """Median LiDAR range straight ahead (r1_nav angles), None if nothing returns."""
    return _window_median(ranges, angles, 0.0, half)


def spin_clearance_ok(ranges, angles, lidar_ahead_m: float, radius_m: float, self_m: float) -> tuple[bool, float]:
    """Is the circle swept by a spin in place (about the body centre) free? LiDAR sits lidar_ahead_m ahead of centre;
    returns closer than self_m are the robot's own body. Returns (ok, closest obstacle distance from the centre)."""
    closest = 99.0
    for r, a in zip(ranges, angles):
        if r < self_m or r >= 11.9:
            continue
        x, y = lidar_ahead_m + r * math.cos(a), r * math.sin(a)
        closest = min(closest, math.hypot(x, y))
    return closest > radius_m, closest


class WheelImuOdometry:
    """Fallback pose: mean wheel angle x radius along the aisle, heading from integrated gyro z.
    Wheel slip (wet floor) is invisible to it, so confidence decays with distance travelled since the last reset."""

    def __init__(self, metre: float = 0.0, heading_deg: float = INTO_AISLE_HEADING_DEG, wheel_radius_m: float = WHEEL_RADIUS_M,
                 gyro_z_index: int = 5, gyro_sign: float = 1.0, drift_per_m: float = 0.005, min_confidence: float = 0.3):
        self.r, self.gz, self.gsign = wheel_radius_m, gyro_z_index, gyro_sign
        self.drift_per_m, self.min_conf = drift_per_m, min_confidence
        self.reset(metre, heading_deg)

    def set_heading(self, heading_deg: float) -> None:
        self.heading = float(heading_deg)

    def correct(self, metre: float) -> None:
        """Re-anchor the along-aisle position to an external fix (LiDAR landmark); drift restarts from here."""
        self.metre, self.travelled = float(metre), 0.0

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
    # Inspection: stop at every plant so all six side cameras get a still, sharp frame (no motion blur).
    stop_to_shoot: bool = True
    shoot_settle_s: float = 0.5
    side_step_m: float = PLANT_PITCH_M
    # Turn around in place (wide aisles): wheel speed, finish tolerance, LiDAR alignment check after the spin.
    spin_wheel_rad_s: float = 1.6
    spin_min_frac: float = 0.4          # skid-steer: below ~0.6 rad/s the wheels scrub without turning
    spin_tolerance_deg: float = 2.0
    spin_radius_m: float = 0.46         # half-diagonal of the 0.528 x 0.75 m body
    spin_margin_m: float = 0.05
    spin_lidar_max_yaw_deg: float = 4.0  # after the spin the bed walls must look parallel again (LiDAR)
    lidar_ahead_m: float = 0.34
    # Position checks against map landmarks (warehouse-style: odometry is checked, not trusted).
    end_wall_metre: float | None = None  # aisle metre of the far end wall (from the map); None = no check
    pose_check_every_m: float = 0.5
    pose_check_max_range_m: float = 8.0  # C1 reaches 12 m; a short window lets a slipping robot stop short unchecked
    pose_correct_max_m: float = 1.0     # larger disagreement: do not trust either, report and stop
    pose_check_max_yaw_deg: float = 3.0
    # Yard / road legs (outside the houses): dead reckoning (wheels + gyro) between known poses, LiDAR safety on.
    aisle_heads: dict = field(default_factory=dict)   # aisle_id -> (centre x, scene y of metre 0); aisles run toward -Y
    rows_m: tuple = (0.0, 38.04)        # plant rows span these metres: side cameras shoot only here
    path_speed_m_s: float = 0.25
    path_k_heading: float = 1.5         # wheel-speed difference (rad/s) per rad of bearing error
    path_spin_above_deg: float = 20.0   # larger bearing error: turn in place first
    path_reach_m: float = 0.12
    controller: ControllerConfig = field(default_factory=ControllerConfig)


class LegExecutor:
    def __init__(self, pose: PoseSource, house: str, cfg: ExecutorConfig | None = None):
        self.pose, self.house, self.cfg = pose, house, cfg or ExecutorConfig()
        self.status, self.reason, self.cmd_name = "idle", "", None
        self.legs: list[Leg] = []
        self.i, self.frame_id = 0, 0
        self._pose_events_seen = 0
        self.aisle: str | None = None
        self.metre, self.heading, self.confidence = 0.0, INTO_AISLE_HEADING_DEG, 0.0
        self.xy: tuple[float, float] | None = None       # scene-frame dead-reckoned position (set by set_xy)
        self._fixes: list[tuple[float, float, float]] = []
        self._cmd_hist: list[tuple[float, float]] = []
        self._wheels = None
        self._reset_leg()

    def set_xy(self, x: float, y: float) -> None:
        self.xy = (float(x), float(y))

    def apply_fix(self, x: float, y: float, heading_deg: float, xy: bool = True, t: float | None = None) -> None:
        """External absolute fix (LiDAR scan-to-map): heading always, position only outside the aisles.
        Fix positions are also kept for SAFETY: wheels turning while the scan says the robot is not moving = stuck."""
        if t is not None:
            self._fixes.append((t, float(x), float(y)))
            self._fixes = [f for f in self._fixes if t - f[0] <= self.cfg.stuck_window_s + 0.5]
        self.heading = float(heading_deg)
        if hasattr(self.pose, "set_heading"):
            self.pose.set_heading(heading_deg)
        if xy:
            self.xy = (float(x), float(y))

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
        self._dead_reckon(obs)
        events: list[dict] = []
        for e in getattr(self.pose, "events", [])[self._pose_events_seen:]:
            events.append(e)
        self._pose_events_seen = len(getattr(self.pose, "events", []))
        leg = self.current_leg
        if leg is None:
            return [0.0, 0.0], events
        t = float(obs["timestamp_s"])
        if self._t0 is None:
            self._t0 = t
            events.append({"type": "leg_start", "index": self.i, "leg": leg.__dict__, "metre": round(self.metre, 3)})
        if self._scan_says_stuck(t):
            return self._terminal("stuck", "wheels turning but the LiDAR scan shows no motion for "
                                  f"{self.cfg.stuck_window_s:.0f} s (wheel slip / blocked)", events), events
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
        if isinstance(leg, PathLeg) or leg.kind == "path":
            return self._path(leg, obs, t, events), events
        if leg.kind == "enter":
            self._enter(leg, events)
            return [0.0, 0.0], events
        if leg.kind == "turn_around":
            return self._turn(leg, obs, t, events), events
        if leg.kind == "drive_out":
            self._reverse = False
            return self._drive(leg, obs, t, events, reverse=False, outward=True), events
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
        self._shoot_since = None
        self._spin_start = self._spin_done_t = None
        self._last_check = None
        self._wp = 0
        self._last_front_xy = (1e9, 1e9)

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

    def _drive(self, leg: Leg, obs, t: float, events: list[dict], reverse: bool, outward: bool = False) -> list[float]:
        c = self.cfg
        self.aisle = leg.aisle_id or self.aisle
        decreasing = reverse or outward                       # metre falls when backing up or driving out after a turn
        remaining = (self.metre - leg.target_metre) if decreasing else (leg.target_metre - self.metre)
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
            ref = INTO_AISLE_HEADING_DEG + (180.0 if outward else 0.0)
            yaw = math.radians((self.heading - ref + 180) % 360 - 180)
        if not decreasing and not self._pose_check(ranges, angles, yaw, t, events):
            return self._terminal("aborted", "position disagreement: odometry vs LiDAR end wall beyond "
                                  f"{c.pose_correct_max_m:.1f} m, waiting for help", events)
        in_rows = c.rows_m[0] - 0.05 <= self.metre <= c.rows_m[1] + 0.05 and self.cmd_name != "revisit"
        if c.stop_to_shoot and not reverse and in_rows:
            due = due_cameras(self.metre, self._last_side, self._last_front, c.side_step_m)
            if any(cam != FRONT_CAMERA for cam in due):
                if self._shoot_since is None:
                    self._shoot_since = t
                if t - self._shoot_since < c.shoot_settle_s:
                    self._anchor = (t, self.metre)            # standing still on purpose: not stuck
                    return [0.0, 0.0]
                self._capture(due, t, "inspect", events)
                self._shoot_since = None
        damp = max(-c.controller.max_diff_rad_s, min(c.controller.max_diff_rad_s, -c.k_heading * yaw))
        cmd = [cmd[0] - damp / 2, cmd[1] + damp / 2]
        scale = max(1.0, abs(cmd[0]) / MAX_WHEEL_RAD_S, abs(cmd[1]) / MAX_WHEEL_RAD_S)
        frac = max(c.min_speed_frac, min(1.0, remaining / c.slow_zone_m)) / scale
        if reverse:
            if self._last_front is None or abs(self.metre - self._last_front) >= 2.0 - 1e-9:
                self._capture((FRONT_CAMERA,), t, "navigate", events)
        elif in_rows:
            self._capture(due_cameras(self.metre, self._last_side, self._last_front, c.side_step_m), t, "inspect", events)
        elif self._last_front is None or abs(self.metre - self._last_front) >= 2.0 - 1e-9:
            self._capture((FRONT_CAMERA,), t, "navigate", events)
        return [cmd[0] * frac, cmd[1] * frac]

    def note_command(self, t: float, cmd) -> None:
        """Remember how hard the wheels were driven forward/backward (for the scan-based stuck check)."""
        self._cmd_hist.append((t, abs(cmd[0] + cmd[1]) / 2))
        self._cmd_hist = [c for c in self._cmd_hist if t - c[0] <= self.cfg.stuck_window_s + 0.5]

    def _scan_says_stuck(self, t: float) -> bool:
        w = self.cfg.stuck_window_s
        if len(self._fixes) < 10 or not self._cmd_hist or t - self._cmd_hist[0][0] < w - 0.2:
            return False
        if min(c[1] for c in self._cmd_hist if t - c[0] <= w) < 0.8:   # stopped / shooting / spinning: not stuck
            return False
        old = [f for f in self._fixes if t - f[0] >= w - 0.3]
        if not old:
            return False
        return math.hypot(self._fixes[-1][1] - old[0][1], self._fixes[-1][2] - old[0][2]) < self.cfg.stuck_min_progress_m

    def wants_side_images(self, t_next: float) -> bool:
        """Will the next step() capture side cameras? (lets a render-on-demand sim render them just in time)"""
        leg = self.current_leg
        if leg is None:
            return False
        if leg.kind == "hold":
            return self._t0 is not None and not self._held_capture and t_next - self._t0 >= min(self.cfg.hold_capture_after_s, self.cfg.hold_s)
        return self._shoot_since is not None and t_next - self._shoot_since >= self.cfg.shoot_settle_s - 1e-6

    def _dead_reckon(self, obs) -> None:
        """Scene XY from wheel distance along the gyro heading; snapped to the aisle centre line inside an aisle."""
        wheel = [float(x) for x in obs["wheel_position_velocity"][:4]]
        if self._wheels is not None and self.xy is not None:
            d = sum((w - p + math.pi) % (2 * math.pi) - math.pi for w, p in zip(wheel, self._wheels)) / 4.0 * WHEEL_RADIUS_M
            h = math.radians(self.heading)
            self.xy = (self.xy[0] + d * math.cos(h), self.xy[1] + d * math.sin(h))
        self._wheels = wheel
        leg = self.current_leg
        if self.xy is not None and leg is not None and leg.kind in ("drive_in", "drive_out", "reverse_out", "turn_around") \
                and self.aisle in self.cfg.aisle_heads:
            ax, hy = self.cfg.aisle_heads[self.aisle]
            self.xy = (ax, hy - self.metre)                      # LiDAR keeps the robot on the centre line

    def _enter(self, leg: Leg, events: list[dict]) -> None:
        """Hand over from yard dead reckoning to the aisle frame: aisle metre from the scene position."""
        self.aisle = leg.aisle_id
        if self.xy is not None and leg.aisle_id in self.cfg.aisle_heads:
            ax, hy = self.cfg.aisle_heads[leg.aisle_id]
            metre = hy - self.xy[1]
            self.pose.reset(metre, self.heading)
            self.metre = metre
            events.append({"type": "enter", "aisle": leg.aisle_id, "metre": round(metre, 3),
                           "lateral_offset_m": round(self.xy[0] - ax, 3)})
        self._finish(events)

    def _path(self, leg, obs, t: float, events: list[dict]) -> list[float]:
        c = self.cfg
        if self.xy is None:
            return self._terminal("aborted", "yard leg without a known start position", events)
        wps = getattr(leg, "waypoints", ())
        ranges, angles = scan_from_obs(obs, c.lidar_forward_rad)
        while self._wp < len(wps) and math.dist(self.xy, wps[self._wp]) <= c.path_reach_m:
            events.append({"type": "waypoint", "index": self._wp, "xy": [round(v, 3) for v in self.xy], "t": t})
            self._wp += 1
        if self._wp >= len(wps):                                   # face the final heading, then done
            fh = getattr(leg, "final_heading_deg", None)
            err = 0.0 if fh is None else (fh - self.heading + 180) % 360 - 180
            if abs(err) <= c.spin_tolerance_deg:
                self._finish(events, xy=[round(v, 3) for v in self.xy], heading_deg=round(self.heading, 1))
                return [0.0, 0.0]
            w = c.spin_wheel_rad_s * max(c.spin_min_frac, min(1.0, abs(err) / 30.0)) * (1 if err > 0 else -1)
            self._anchor = (t, self.metre)
            return [-w, w]
        tx, ty = wps[self._wp]
        bearing = math.degrees(math.atan2(ty - self.xy[1], tx - self.xy[0]))
        err = (bearing - self.heading + 180) % 360 - 180
        if abs(err) > c.path_spin_above_deg:
            w = c.spin_wheel_rad_s * max(c.spin_min_frac, min(1.0, abs(err) / 30.0)) * (1 if err > 0 else -1)
            return [-w, w]
        front = sector_min(ranges, angles, 0.0, c.controller.front_half_angle_rad)
        if front < c.controller.stop_range_m:
            if self._blocked_since is None:
                self._blocked_since = t
                events.append({"type": "obstacle_stop", "why": f"front {front:.2f} m", "xy": [round(v, 2) for v in self.xy], "t": t})
            elif c.blocked_timeout_s is not None and t - self._blocked_since >= c.blocked_timeout_s:
                return self._terminal("aborted", f"yard path blocked for {c.blocked_timeout_s:.0f} s", events)
            return [0.0, 0.0]
        self._blocked_since = None
        dist = math.dist(self.xy, (tx, ty))
        v = c.path_speed_m_s * max(0.3, min(1.0, dist / 0.6)) / WHEEL_RADIUS_M
        diff = max(-c.controller.max_diff_rad_s, min(c.controller.max_diff_rad_s, c.path_k_heading * math.radians(err)))
        if self._last_front is None or math.dist(self.xy, self._last_front_xy) >= 2.0:
            self._capture((FRONT_CAMERA,), t, "navigate", events); self._last_front_xy = self.xy
        l, r = v - diff / 2, v + diff / 2
        scale = max(1.0, abs(l) / MAX_WHEEL_RAD_S, abs(r) / MAX_WHEEL_RAD_S)
        return [l / scale, r / scale]

    def _pose_check(self, ranges, angles, yaw: float, t: float, events: list[dict]) -> bool:
        """Compare odometry with the far end wall seen by the LiDAR (map landmark); correct small drift, report all.
        Returns False when the two disagree by more than pose_correct_max_m."""
        c = self.cfg
        if c.end_wall_metre is None or abs(math.degrees(yaw)) > c.pose_check_max_yaw_deg:
            return True
        if self._last_check is not None and abs(self.metre - self._last_check) < c.pose_check_every_m:
            return True
        r = front_wall_range(ranges, angles)
        if r is None or r > c.pose_check_max_range_m:
            return True
        lidar_metre = c.end_wall_metre - c.lidar_ahead_m - r * math.cos(yaw)
        err = self.metre - lidar_metre
        self._last_check = self.metre
        ok = abs(err) <= c.pose_correct_max_m
        events.append({"type": "pose_check", "landmark": "end_wall", "odometry_metre": round(self.metre, 3),
                       "lidar_metre": round(lidar_metre, 3), "error_m": round(err, 3), "corrected": ok, "t": t})
        if ok and hasattr(self.pose, "correct"):
            self.pose.correct(lidar_metre)
            self.metre = lidar_metre
        return ok

    def _turn(self, leg: Leg, obs, t: float, events: list[dict]) -> list[float]:
        """Spin 180 deg in place on the gyro, refuse if the swept circle is not clear, then verify with the LiDAR."""
        c = self.cfg
        self.aisle = leg.aisle_id or self.aisle
        ranges, angles = scan_from_obs(obs, c.lidar_forward_rad)
        if self._spin_start is None:
            ok, closest = spin_clearance_ok(ranges, angles, c.lidar_ahead_m, c.spin_radius_m + c.spin_margin_m, 0.25)
            if not ok:
                return self._terminal("aborted", f"no room to turn: obstacle {closest:.2f} m from centre "
                                      f"(needs {c.spin_radius_m + c.spin_margin_m:.2f} m)", events)
            self._spin_start = self.heading
            events.append({"type": "turn_start", "heading_deg": round(self.heading, 1), "closest_m": round(closest, 3), "t": t})
        turned = self.heading - self._spin_start
        remaining = 180.0 - turned
        if self._spin_done_t is None:
            if abs(remaining) > c.spin_tolerance_deg:
                w = c.spin_wheel_rad_s * max(c.spin_min_frac, min(1.0, abs(remaining) / 30.0)) * (1 if remaining > 0 else -1)
                self._anchor = (t, self.metre)
                return [-w, w]                                    # left turn: right wheel forward
            self._spin_done_t = t
        if t - self._spin_done_t < 0.5:                           # settle, then check alignment with the bed walls
            return [0.0, 0.0]
        lidar_yaw = wall_yaw(ranges, angles, c.wall_max_m)
        lidar_ok = lidar_yaw is not None and abs(math.degrees(lidar_yaw)) <= c.spin_lidar_max_yaw_deg
        events.append({"type": "turn_check", "gyro_turned_deg": round(turned, 1),
                       "lidar_yaw_deg": None if lidar_yaw is None else round(math.degrees(lidar_yaw), 2),
                       "ok": lidar_ok, "t": t})
        if not lidar_ok:
            return self._terminal("aborted", "turn not verified by LiDAR (bed walls not parallel)", events)
        self._finish(events, turned_deg=round(turned, 1))
        return [0.0, 0.0]

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
