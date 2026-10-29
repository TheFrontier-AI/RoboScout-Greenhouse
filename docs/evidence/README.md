# Evidence: it ran

Raw logs and outputs from the real runs, copied as they were (only the server IP and a scratch path are replaced by
`SERVER` / `JETSON` / `SCRATCH`). Nothing here is mocked; the offline test suite is separate.

| File | What it proves |
|---|---|
| [01_cloud_install_fresh_vm.log](01_cloud_install_fresh_vm.log) | Fresh checkout of `main` on a Nebius `gpu-rtx6000` VM: `deploy/nebius/install_cloud.sh` → torch 2.14.1+cu130 with CUDA, 28 offline tests pass, 2 min 15 s |
| [02_cloud_handoff_run3_pipeline.log](02_cloud_handoff_run3_pipeline.log) | Hub `/health`, the robot hand-off of run 3 (`robot_upload_run.py`), `/run_complete` accepted, the watcher running `detect → vlm_stops → lidar_map → report`, `/run_status` = `done` after 9 min 59 s, `report_en.md` / `report_th.md` written |
| [03_robot_install_dgx_spark_arm64.log](03_robot_install_dgx_spark_arm64.log) | Robot side on an ARM64 machine (DGX Spark): `deploy/robot/install_robot.sh`, 50 tests pass |
| [04_robot_install_jetson_orin_nano.log](04_robot_install_jetson_orin_nano.log) | **Real Jetson Orin Nano Super (JetPack 6, Python 3.10)**: install 1 min 33 s, 50 tests pass in 49 s |
| [05_jetson_handoff_to_cloud.log](05_jetson_handoff_to_cloud.log) | **Jetson → cloud**: a recorded run uploaded from the Jetson over the farm's internet, hub accepted it, pipeline ran (83 s), `/run_status` = `done` with both reports |
| [06_run3_report_en.md](06_run3_report_en.md) / [06_run3_report_th.md](06_run3_report_th.md) | The farmer's report the pipeline wrote for run 3 (English / Thai), as produced — not edited |
| [06_run3_report_summary.json](06_run3_report_summary.json) | The machine-readable report for run 3: harvest counts per aisle, plant count, disease screening summary, navigation events, localisation, step status |
| [08_two_observer_fusion_run3.json](08_two_observer_fusion_run3.json) | `scripts/fuse_localisation.py` on run 3: the robot's own estimate (wheels + gyro + LiDAR scan-to-map) vs cuVSLAM stereo as an independent second observer, and the two fused — scored against the recorded truth (figure: [../media/gallery/23_two_observers_fusion.png](../media/gallery/23_two_observers_fusion.png)) |
| [09_cuvslam_on_jetson.json](09_cuvslam_on_jetson.json) | **cuVSLAM on the real Jetson Orin Nano** (25 W): stereo 188 fps, stereo + IMU 82 fps, 600/600 frames tracked over 60 s / 11 m of ZED 2i data, metric scale (Sim3 scale 1.003), ATE 3–4 mm on the segment |
| [07_run3_recording_summary.json](07_run3_recording_summary.json) | What the robot recorded while driving run 3 on its own: 7 500 control steps over 750 s, legs (`path`, `enter`, `drive_in`, `drive_out`, `hold`, `turn_around`, `dock`), 4 726 events incl. 1 035 camera captures |

The drive itself:

- Route the robot drove in run 3 (from its own recording): [../media/gallery/22_route_run3.png](../media/gallery/22_route_run3.png) — 117 m, 12.5 min, 161 plant stops.
- The film (chase camera, nothing scripted): `docs/demo/robofarm_demo.mp4` (gitignored, 260 MB — supplied with the submission).
- Per-camera raw renders, detection videos, 3-D maps and the full run-3 recording (5.8 GB) are kept off-git; see
  [../SYSTEM_OVERVIEW.md](../SYSTEM_OVERVIEW.md) §"Figures and videos" for the file list, available on request.

How to reproduce any of it: [../INSTALL.md](../INSTALL.md).
