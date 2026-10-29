# R181-S: the stereo-camera (ZED 2i) variant of the R1 robot

Status: **spec and sim ready to use** (29 Oct 2026). Body, shape, wheels, mast and side cameras are identical to the R180; only the front camera changes.
Astra's robot files are not modified; the stereo camera is added through a session layer when the sim runs.

## Spec changes from the R180

| | R180 | R181-S |
|---|---|---|
| Front camera | single RGB (sim 1280×720) | **Stereolabs ZED 2i Polarizer** (2.1 mm lens) |
| Images | 1 | left and right cameras, **12 cm** apart, HD1080 at 30 fps (2K at 15 fps) |
| Field of view | ~69° horizontal | 110° × 70° before rectification (after rectification at 1080p fx ≈ 1066 px, ~84° horizontal) |
| Depth | none | 0.3–20 m (computed from the stereo pair) |
| IMU | the robot's IMU | an extra IMU in the camera (gyro + accelerometer, ~400 Hz) |
| Other | – | IP66 water/dust proof, polarising filter to cut reflections from plastic film/leaves, USB 3 |
| Mount position | front of the robot | **the same spot as the old front camera** (`/Robot/Physics_Base/Sensors/front_rgb`) |
| Camera price | – | ~US$549 from Stereolabs (excluding import duty) |

LiDAR C1, wheels, the robot's gyro, the 6 side cameras: same as the R180

## New capabilities
1. **On-robot localisation with cuVSLAM** (NVIDIA, runs on the Jetson): stereo / stereo+IMU / RGB-D, used together with the LiDAR
2. **3-D obstacles**: depth sees low/hanging objects (hoses, branches) that the single-plane LiDAR cannot
3. **Event reports with real distances**: Cosmos Reason → HERMES gets the distance to a person/obstacle
4. **Metric-scale 3-D map** for lingbot-map / MapAnything on the cloud

Items 2–4 are still next steps; item 1 has been tested (below).

## Sim test results (29 Oct, the real run-3 route, first few metres of house 09, 30 seconds, 450 stereo pairs at 15 Hz, 960×540)

| cuVSLAM | tracked | RMSE error | scale | speed (RTX PRO 6000) |
|---|---|---|---|---|
| stereo | 450/450 | 3.8 mm | 1.007 | 1,308 fps |
| stereo + IMU | 450/450 | 4.2 mm | 1.008 | 646 fps |
| RGB-D | 450/450 | 6.4 mm | 0.975 | 507 fps |

For comparison: the R180's single front camera (cuVSLAM Mono, whole route, 2 Hz) failed to initialise, error 15.8 m.

**Caution:** sim images are unrealistically perfect (no noise, motion blur, rolling shutter, calibration error; the IMU has no noise) and the test segment is short, 5.7 m.
Real values in the greenhouse will be much worse. The whole route and the real camera must be tested before quoting numbers.

## How to use (on a VM with Isaac Sim 5.1)

1. Render stereo data along the recorded real route (no new driving, only re-rendering):
   ```
   /isaac-sim/python.sh scripts/replay_route.py --rec <run>/rec --out <out> --from-step 334 --to-step 6476 --hz 30
   ```
   Produces `cam0/ cam1/ depth0/ imu0/data.csv groundtruth.csv calib.json` (EuRoC format). Add `--chase` for a chase camera for animation.
   Speed ~1.1 s/frame at 960×540; the whole route at 30 Hz and 1080p takes several hours, so select a segment with `--from-step/--to-step`.
2. Run cuVSLAM:
   ```
   python3 scripts/cuvslam_r181.py --data <out> --mode stereo|inertial|rgbd --out result.json
   ```
   Requires the `cuvslam` wheel v17.0.1 (cu12, cp312) and the CUDA 12 libraries (`nvidia-cusolver-cu12` etc.) on LD_LIBRARY_PATH.

Related files: `sim/zed2i_rig.py` (spec + stereo camera in the sim), `scripts/replay_route.py`, `scripts/cuvslam_r181.py`
