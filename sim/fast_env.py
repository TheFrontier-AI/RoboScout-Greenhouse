"""Render-on-demand wrapper around Astra's GatedRenderSensorEnv (subclass; her runtime files are not modified).

Physics + LiDAR + IMU + wheels advance every control step (one C1 sweep, 0.1 s). Cameras render only when asked:
  env.want = {"side": bool, "front": bool, "extra": bool}   (set before step; cleared after)
side = the six 1080x1920 plant cameras (the robot stops at each plant to shoot), front = navigation camera,
extra = render products added by the runner (the third-person chase camera). Observations without a fresh render
carry no image keys, so nothing stale is ever passed off as a new frame. The evaluator only ends an episode on tip-over
or invalid physics (the robot is meant to leave the aisle and drive in the yard)."""
from __future__ import annotations
import math
import sys
import numpy as np


def make_fast_env(base_cls):
    mod = sys.modules[base_cls.__module__]
    review = sys.modules.get("sensor_env_review", mod)
    g = {**vars(review), **vars(mod)}                           # update, observation, read_packet, compose, ...

    class FastFarmEnv(base_cls):
        physics_render = True                                    # the C1 LiDAR bins need a render tick per physics step;
                                                                 # all camera products are disabled, so this is cheap

        def __init__(self, *a, front_resolution=None, **kw):
            super().__init__(*a, **kw)
            self.front_resolution = front_resolution             # the base env's 320x180 is too small for visual SLAM;
            self.want = {"side": False, "front": False, "extra": False}
            self.extra_products = []                             # hydra textures of runner-owned render products
            self.render_seconds = 0.0

        def reset(self, *a, **kw):
            out = super().reset(*a, **kw)
            if self.front_resolution and tuple(self.camera.get_resolution()) != tuple(self.front_resolution):
                self.camera.set_resolution(tuple(self.front_resolution))   # render product exists only after reset
            return out

        def set_evaluator(self, evaluator):
            self.evaluator = evaluator

        def _products(self, which):
            out = []
            if which.get("front"):
                out.append(self.camera._render_product.hydra_texture)
            if which.get("side"):
                out += [c._render_product.hydra_texture for c in self.plant_cameras.values()]
            if which.get("extra"):
                out += list(self.extra_products)
            return out

        def _all_products(self):
            return self._products({"front": True, "side": True, "extra": True})

        def step(self, action):
            import time
            import omni.replicator.core as rep
            import omni.timeline
            if self.done:
                raise RuntimeError("Call reset before stepping a finished episode")
            a = np.asarray(action, dtype=float)
            if a.shape != (2,) or not np.isfinite(a).all() or np.any(abs(a) > 2.0):
                self._stop(); self.done = True
                raise ValueError("Action must be left/right wheel rad/s within [-2,2]")
            cmds = [a[0] if "LEFT" in n else a[1] for n in self.robot.dof_names]
            self.robot.set_joint_velocity_targets(np.asarray([cmds], dtype=np.float32))
            want, self.want = dict(self.want), {"side": False, "front": False, "extra": False}
            try:
                for tex in self._all_products():
                    tex.set_updates_enabled(False)
                self._quality(False)
                complete = False
                for _ in range(25):
                    self.world.step(render=self.physics_render)
                    if g["update"](self.lidar, self.world.current_time):
                        complete = True; break
                if not complete:
                    raise RuntimeError("No complete scan within watchdog interval")
                rendered = self._products(want)
                if rendered:
                    t0 = time.time()
                    self._quality(True)
                    for tex in rendered:
                        tex.set_updates_enabled(True)
                    for _ in range(2):
                        rep.orchestrator.step(delta_time=0.0, pause_timeline=False, rt_subframes=1)
                    self.render_seconds += time.time() - t0
                obs = self._obs(want)
                pos, quat = self.robot.get_world_poses()
                reward, term, trunc, info = self.evaluator.evaluate(pos[0], quat[0], self.robot.get_joint_velocities()[0])
                self.done = term or trunc
                if self.done:
                    self._stop()
                return obs, float(reward), bool(term), bool(trunc), info
            except Exception:
                self._stop(); self.done = True
                raise
            finally:
                tl = omni.timeline.get_timeline_interface()
                tl.set_auto_update(True); tl.commit_silently()

        def _quality(self, full: bool):
            """PathTracing samples: full profile for captures, 1 spp for physics/LiDAR ticks (nothing is captured then)."""
            import carb
            st = carb.settings.get_settings()
            if not hasattr(self, "_spp_full"):
                self._spp_full = (st.get("/rtx/pathtracing/spp") or 16, st.get("/rtx/pathtracing/totalSpp") or 16)
            spp, total = self._spp_full if full else (1, 1)
            st.set("/rtx/pathtracing/spp", spp); st.set("/rtx/pathtracing/totalSpp", total)

        def _obs(self, want):
            now = float(self.world.current_time)
            scan = g["observation"](self.lidar)
            if not scan["scan_complete"] or now - min(scan["beam_timestamps_s"]) > 0.2:
                raise RuntimeError("Invalid sweep age")
            ip = g["read_packet"](self.iface, self.imu.prim_path, previous_timestamp_s=self.previous_imu)
            self.previous_imu = ip["timestamp_s"]
            pkt = g["compose"](now_s=now, lidar=g["SensorPacket"](scan["timestamp_s"], tuple(scan["ranges_m"]), "lidar_c1",
                                                                   "simulated_sensor"),
                               imu=g["SensorPacket"](**ip), encoder=g["read_wheels"](self.robot, now),
                               range_min_m=0.05, range_max_m=12.0, lidar_count=500)
            pkt["lidar_angles_rad"] = scan["angles_rad"].tolist()
            pkt["plant_rgb"] = {}
            if want.get("front"):
                pkt["front_rgb"] = np.asarray(self.camera.get_rgba())[..., :3]
            if want.get("side"):
                for name, cam in self.plant_cameras.items():
                    pkt["plant_rgb"][name] = np.asarray(cam.get_rgba())[..., :3]
            return pkt

    return FastFarmEnv


class YardEvaluator:
    """Ends the episode only on tip-over or non-finite physics; reports scene pose for the log (evaluation only)."""

    def __init__(self, max_tilt_deg: float = 20.0):
        self.max_tilt = max_tilt_deg

    def evaluate(self, position, quat_wxyz, wheel_velocity):
        vals = [float(v) for v in (*position, *quat_wxyz, *wheel_velocity)]
        if not all(math.isfinite(v) for v in vals):
            return 0.0, True, False, {"termination": "invalid_physics"}
        w, x, y, z = (float(v) for v in quat_wxyz)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
        yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        info = {"termination": None, "scene_pose": [float(position[0]), float(position[1]), float(position[2]), yaw, tilt]}
        if tilt > self.max_tilt:
            info["termination"] = "tip"
            return 0.0, True, False, info
        return 0.0, False, False, info
