# Side (plant) cameras in the R180 env: what is needed (for Astra)

## Update verified 2026-10-28

The original request below is retained as history. `SensorStopEnv` now exposes six real
`obs['plant_rgb']` images with separate timestamps and freshness checks. Public left/right
names follow robot forward **-Y**: public left uses authored USD `plant_right_*`; public
right uses authored USD `plant_left_*`. Protected USD prims were not renamed.

Native reset accepts `start_pose`, `start_offset`, `friction_scale`, `obstacle_enabled`,
`obstacle_position`, `max_steps`, and `task_mode`. Default `max_steps=None` removes the
300-step limit; `task_mode='navigation'` disables the approach-and-stop objective but
retains bounds, tip, and obstacle collision checks.

Evidence: `../../robot_design/r1_v3/diy_r50/physical_ai_r180/side_camera_runtime_2026-10-28/`
contains `native_api_result.json`, `native_api_realism_result.json`, `camera_axis_audit.json`,
and reproducible `check_native_api.py` / `check_realism_api.py`.
The step-301 check sets the evaluator counter to 300 then executes one real physics step;
it is **not a continuous 301-step navigation rollout**. Native sensor tests passed; the
navigation wrapper still needs an end-to-end run. Eleven evaluator/geometry unit tests pass.

For the updated material scene use `C2_R180_Realism_Sensor_Fixture.usda` from that folder.
The runtime now applies the scene renderSettings (including fractionalCutoutOpacity).
Actual six-camera frames show leaves again. Film/lighting realism remains incomplete.
Rendering every physics substep is still expensive; no performance improvement is claimed.

## Original request

**At request time:** `physical_ai_runtime/sensor_env.py` (`SensorStopEnv`) only wires `front_rgb` (320x180) into `obs`.
The six plant cameras are already authored in `physical_ai_r180/robot_sensors.usda` under
`/Robot/Physics_Base/Sensors/`:

| r1_nav name (`r1_nav/cameras.py`) | USD prim | position in base frame (m) | looks along |
|---|---|---|---|
| mast_left_low  | plant_left_low   | (-0.1745, -0.1967, 0.7945) | base -X |
| mast_left_mid  | plant_left_mid   | (-0.1745, -0.1967, 1.2145) | base -X |
| mast_left_high | plant_left_high  | (-0.1745, -0.1967, 1.6445) | base -X |
| mast_right_low | plant_right_low  | (+0.1745, -0.3333, 0.7945) | base +X |
| mast_right_mid | plant_right_mid  | (+0.1745, -0.3333, 1.2145) | base +X |
| mast_right_high| plant_right_high | (+0.1745, -0.3333, 1.6445) | base +X |

Optics: focal 20 mm, aperture 21.40 x 38.05 mm (portrait), so about 56° H x 87° V.
`scripts/run_demo_sim.py` tries to attach `isaacsim.sensors.camera.Camera` objects to these prims at 180x320 (see
`sim/isaac_env.py: R180Sim._attach_side_cameras`). This has NOT been run yet. If it fails, the run logs a WARNING and
side captures fall back to `front_rgb` with `meta.image_source = "front_rgb_placeholder"`.

## Please add or check in the env
1. **Expose the 6 cameras in `obs`** as `plant_rgb[name]` (uint8 HxWx3) with their own `*_timestamp_s`, using the same
   freshness checks as `front_rgb` (`rgb_adapter.read_rgb_packet`). Resolution 180x320 (portrait) is enough for the
   demo, and 360x640 is better for Cosmos Reason.
2. **Check left/right naming.** The robot drives toward base **-Y**, so its left is **+X**. `plant_left_*` sits at
   x = -0.17 and looks along **-X**, which is the robot's *right*. Either swap the names or confirm which convention
   is intended. The executor tags frames with these names, and the cloud pins findings to the left or right bed
   based on them.
3. **FOV and placement.** Target: 3 per side (low/mid/high, about 0.8 / 1.2 / 1.65 m), looking sideways (yaw ±90°,
   level or tilted down 5–10° for `high`), about 90° FOV. Vertical is 87° now. Horizontal 56° covers about 0.5 m of
   bed edge at 0.46 m distance, so the 0.5 m capture spacing is just contiguous. Consider a landscape 90° H instead.
   Note that left and right are offset in Y (-0.197 vs -0.333).
4. **Performance.** Six more render products cost a lot because the env renders every physics substep (12 per step).
   Options: render the plant cameras only on capture steps (an `env.capture(names)` call that does one rendered
   update), or render all cameras every Nth substep. The executor needs plant images only every 0.5 m (about every
   2 s at 0.25 m/s).
5. **Optional:** a depth or semantic annotator on the plant cameras, which would give ground truth for "leaf anomaly"
   placement in the evaluation.

## Other env gaps found while wiring this (also for Astra)
- `reset()` only accepts `options={'obstacle_y': -2.0|-2.8|-3.2}`. Native options for **start pose** (aisle, metre,
  lateral, yaw), **friction scale**, and **obstacle on/off plus position** would replace the stage edits in
  `sim/isaac_env.py`, which are unverified.
- The built-in `ApproachStopEvaluator` truncates at 300 steps (about 30 s) and terminates near y = obstacle_y even
  when the box is moved. `R180Sim` swaps it for `GroundTruthEvaluator` after each reset.
- **Reverse:** the controller commands negative wheel speeds. `check_reverse_drive.py` shows the open-loop reverse is
  physically fine. Behind the aisle head (metre < 0, the door side) there is no measured headland collision geometry.
- LiDAR angle convention: robot forward is at azimuth -π/2 (taken from `stop_controller.py`). `r1_nav.controller`
  assumes 0 = forward, so `sim.executor.scan_from_obs` rotates the scan. Please confirm the azimuth is CCW (left = 0 rad).
- IMU gyro z sign and frame (`imu_accel_gyro[5]`): we assume + = CCW (left turn).
