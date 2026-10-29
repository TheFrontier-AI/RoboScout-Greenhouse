# R1 LiDAR in simulation

## Sensor
Simulated **Slamtec RPLIDAR C1**: a 2-D, 360° laser scanner, modelled from datasheet rev 1.2. Nominal, not calibrated.

| | |
|---|---|
| Isaac Sim 5.1 sensor | **PhysX range sensor** (`isaacsim.sensors.physx`, `RangeSensorCreateLidar`, `_range_sensor.acquire_lidar_sensor_interface()`). Not the RTX LiDAR |
| Field of view | 360° horizontal, 1° vertical (single plane) |
| Resolution | 0.72° → about 500 beams per scan |
| Rate | 10 Hz (`rotation_rate=10`) |
| Range | 0.05 m to 12.0 m |
| Prim | `/Robot/Physics_Base/Sensors/lidar_c1` |
| Mount | 0.34 m ahead of the base centre, 0.512 m above the base (scan plane about 0.55 m above the floor) |
| Physics | 120 Hz |

The sensor is created by the R180 runtime (`physical_ai_r167_lidar/lidar_runtime.py`, scene owner's package, outside this repo). This repo reads it through `sim/fast_env.py` as `obs["lidar_m"]`. Beam *k* of *n* has raw azimuth −π + 2πk/n; adding π/2 gives the robot frame (0 = forward). See `scripts/make_farm_videos.py: scan_xy`.

## Issues seen and how they are handled
- **"Hit outside configured range"**: never seen, in any of our runs (container logs and all recorded runs checked, 29 Oct 2026).
- **Raw returns outside 0.05–12 m**: PhysX can report a far end wall (>12 m) from the middle of a 38 m aisle. The runtime turns any return outside 0.05–12.0001 m into "no return" (intensity 0) and counts it, as the real C1 does. No error is raised.
- **"Missing/duplicate LiDAR bin"**: this is the runtime's own check, not an Isaac error. It fires when physics steps run without a render/update tick. Fix: `sim/fast_env.py` renders on every physics step (`physics_render = True`), with all camera products disabled and path tracing at 1 spp, so the tick is cheap. The LiDAR update is called after each step.
- **Rays passing through tomato plants and tunnel-10**: the scene had no plant colliders, and tunnel-10 had no collision at all. Fixes, all in the session layer only:
  - `sim/crop_swap.py`: an invisible cylinder per plant stem (r = 0.10 m, h = 2 m). Alpha leaf-card triangle colliders caused false SAFETY stops.
  - `sim/scene_collision.py`: mirrors tunnel-09's structure colliders onto tunnel-10.
  - `sim/scene_prep.py`: box colliders for the walls, with a 1.24 m doorway gap at each end.
  - Order matters: collision must be added before the wall and door-frame meshes are switched off. On 29 Oct, tunnel-10's back door frame stayed solid and blocked R1.
- **Own body in the scan**: returns closer than `rear_self_m` (0.72 m) behind the LiDAR are the robot itself and are ignored (`sim/executor.py`).

## What the LiDAR is used for (robot side, no simulator truth)
- `sim/lidar_odometry.py`: scan matching (ICP) on plant stems, fused with wheels and gyro; slip check.
- `sim/scan_localizer.py`: warehouse-style scan-to-map ICP against a prior 2-D map. In the sim, `scripts/run_farm.py: build_prior_map` builds this map from the twin's collision geometry with PhysX raycasts at the same height.
- `sim/executor.py`:
  - SAFETY front stop;
  - end-wall and row-end landmark checks;
  - stuck check: 90th-percentile raw scan change over 5 s below 0.03 m while the wheels are driven. On the recorded run: moving ≥ 0.70 m, blocked ≤ 0.002 m.
- `scripts/make_farm_videos.py`: the 2-D LiDAR video and the SLAM occupancy map (returns only; no guessed free space).
