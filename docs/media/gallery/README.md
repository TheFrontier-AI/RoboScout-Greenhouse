# RoboFarm gallery

All 3:2, ≤ 2 MB. Renders are from the NVIDIA Isaac Sim digital twin of our farm (Astra scene) unless marked *real photo*.

| # | Image | Caption |
|---|---|---|
| 01 | ![](01_farm_overview_isaac_sim.jpg) | Digital twin of our tomato farm in NVIDIA Isaac Sim — CCTV view as R1 leaves its dock |
| 02 | ![](02_robot_in_tunnel09.jpg) | R1 driving itself down tunnel-09, stopping at every plant — chase camera |
| 03 | ![](03_cctv_tunnel09.jpg) | Greenhouse CCTV: R1 working the aisle, six mast cameras capturing each plant from both sides |
| 04 | ![](04_robot_crossing_yard.jpg) | Bystander view: R1 crossing the yard between greenhouses, Cosmos Reason watching the front camera |
| 05 | ![](05_robot_between_greenhouses.jpg) | Out the back door and across the yard to tunnel-10 — the full route is driven, not scripted |
| 06 | ![](06_front_camera.jpg) | R1 front camera: the aisle as the robot sees it; this feed goes to NVIDIA Cosmos Reason on events |
| 07 | ![](07_six_cameras_nemotron_explains.jpg) | Plant stop 21/161: six views, RT-DETR boxes, NVIDIA Nemotron 3.5 Super VL verdict and next step |
| 08 | ![](08_fruit_grading_rtdetr.jpg) | RT-DETRv2 fine-tuned on Isaac Sim renders + real greenhouse photos: every fruit graded ripe / not ready / reject |
| 09 | ![](09_zed2i_stereo_depth.jpg) | R181-S: ZED 2i stereo depth from the robot's own camera, used for cuVSLAM and the 3-D map |
| 10 | ![](10_lidar_scan.jpg) | 2-D LiDAR scan in the robot frame — matched against the farm map for localisation, no markers or tags |
| 11 | ![](11_cuvslam_3d_map.jpg) | 3-D map of the greenhouse: ZED stereo depth placed with NVIDIA cuVSLAM poses, built on Nebius GPUs |
| 12 | ![](12_lingbot_map_3d.jpg) | lingbot-map: a second 3-D reconstruction from the front camera alone, run as a Nebius GPU job |
| 13 | ![](13_plant_health_card.jpg) | Plant-health card at a sick plant: RT-DETR boxes, Nemotron's description, recommended action — separate from harvest |
| 14 | ![](14_survey_report.jpg) | The farmer's report: ripe / not ready / reject per greenhouse, flagged plants with photos, location and next step |
| 15 | ![](15_r180_assembly_cad.jpg) | R180 assembly overview from our CAD model — built around locally available components |
| 16 | ![](16_real_farm_aerial_tunnel_greenhouses.jpg) | *Real photo* — the row of tunnel greenhouses the digital twin was built from |
| 17 | ![](17_real_tomato_aisle.jpg) | *Real photo* — inside a tomato tunnel on the farm, the environment R1 is designed to survey |
| 18 | ![](18_real_tunnel_greenhouses_exterior.jpg) | *Real photo* — the tunnel greenhouses from outside, matched in the Isaac Sim twin |
| 19 | ![](19_sim_whole_farm_aerial.jpg) | The whole farm in the twin: greenhouses, ponds, orchard plots, sheds (Blender scene V16) |
| 20 | ![](20_sim_farm_aerial_greenhouses_ponds.jpg) | Aerial of the greenhouse block and the four ponds in the twin |
| 21 | ![](21_sim_ponds_and_greenhouses.jpg) | Ponds and the tunnel greenhouses in the twin |
| 22 | ![](22_route_run3.png) | Run 3, the route R1 drove on its own: dock → tunnel-09 → yard → tunnel-10 → dock — 117 m, 12.5 min, 161 plant stops |
