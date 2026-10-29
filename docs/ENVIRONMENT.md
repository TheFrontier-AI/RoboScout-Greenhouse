# Simulated environment (Isaac Sim 5.1, DGX Spark machine)

- Scene: `robot_design/r1_v3/diy_r50/physical_ai_r180/scene_fix_2026-10-28/C2_R180_tunnel09_photo_match.usda`
  (+ `sensor_stop_fixture_pt.usda` with a test box) — already compared against photos of the real farm; see the README in that folder
- env API (from the R180 handoff): create the `SimulationApp` first, then `from r180_env import make_env`;
  `obs, info = env.reset(seed)`; `obs, r, term, trunc, info = env.step([left_rad_s, right_rad_s])` in the range ±2 rad/s
- obs: `front_rgb` 320×180 uint8, `lidar_m` 500 values + `lidar_angles_rad`, `imu_accel_gyro` 6 values, `wheel_position_velocity` 8 values, `timestamp_s`
  **No world pose** — the controller must use sensors only; the ground-truth pose may be used only for evaluation
- Rendering: ACES settings are baked into the scene; real-time mode must settle ~200 frames before images are used; PathTracing for high-quality images
- Known limitations: leaves have no collision and the LiDAR cannot see leaves; mass/friction are estimates; the env renders on every physics substep (slow; asked Astra to fix)
- Limitations of the current env that this repo needs added: random start point in an aisle, starting in any of the 3 aisles, several obstacle types, reverse command, camera images every 0.5 m
