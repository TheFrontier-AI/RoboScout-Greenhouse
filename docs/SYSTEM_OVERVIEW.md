# RoboFarm — System Overview (29 Oct 2026)

*For the hackathon write-up and the demo narration. Every number below comes from runs made on 28–29 Oct 2026; the files that back them are listed in §10. Thai summary in §12 (สรุปภาษาไทย).*

---

## 1. What it is

RoboFarm is a small four-wheel greenhouse robot (R1, now R181-S) that drives itself through tomato greenhouses, stops at every plant, photographs it from six cameras, and sends the pictures to a server. On the server, AI models grade the fruit, screen the plants for disease, build 3-D maps of the farm, and write a report for the farmer. A planner (HERMES, running NVIDIA Nemotron on Nebius Token Factory) decides where the robot should go next, including going back to look again at a plant that looked sick.

The whole system was built and tested in an NVIDIA Isaac Sim digital twin of our real farm. Nothing in the loop is scripted: the robot plans its own wheel commands from its sensors, and the route shown in every video is the route it actually drove.

## 2. What runs where

| Part | Where it runs | What it does | Status |
|---|---|---|---|
| **BOOK driver** | Robot (edge) | Follows the aisle map: enter the door, drive the aisle, stop every 0.45 m at a plant, turn, exit, cross the yard, dock | Working; drove the full farm route |
| **SAFETY** | Robot | Stops the wheels when the LiDAR sees an obstacle, when the scan shows the robot is stuck, or when the pose disagrees with the map | Working; one real stuck case caught and fixed |
| **Localisation** | Robot | Wheels + gyro, LiDAR scan matching on plant stems, scan-to-map ICP against a prior 2-D map. No fiducial tags | 0.23 m RMSE over 123 m, max 0.50 m |
| **R181-S stereo (new)** | Robot | ZED 2i stereo camera: cuVSLAM gives metric position at 500+ fps on one GPU | 6.4 cm RMSE over 111 m (sim) |
| **Robot API hub** | Nebius server | Receives frames and status, queues them for the AI eyes, hands goals back to the robot | Working |
| **HERMES planner** | Nebius server | Nemotron 3 (Token Factory) picks goals: `survey_farm`, `revisit`, `return_to_dock`, `notify`. Never wheel speeds | Working; it ordered two revisits on run 3 |
| **Plant eyes** | Nebius server | RT-DETRv2 (per image) + Nemotron 3.5 Super VL (one request per plant stop, six images) | Working |
| **Front eyes** | Nebius server | Cosmos Reason 2 explains SAFETY stops (person? cart? hose?) so HERMES can wait or call the farmer | Built and tested with stubs; live run pending |
| **3-D mapping** | Nebius server | lingbot-map, cuVSLAM + ZED depth fusion, MapAnything; LiDAR 2-D map | Working |
| **Post-run pipeline** | Nebius server | One command runs every step above on a finished run and writes `report.json` + Thai/English report | On `main`, 114 tests |

Everything on the server is stateless per run: the robot uploads, the server computes, the report comes back. A hundred robots are a hundred upload queues into the same pipeline.

## 3. Models

| Model | Vendor / licence | Runs on | Role in RoboFarm |
|---|---|---|---|
| **NVIDIA Nemotron 3 Super** (120B-A12B) | NVIDIA, Token Factory API | Nebius Token Factory | HERMES planner reasoning and the farmer report |
| **NVIDIA Nemotron 3.5 Super VL** | NVIDIA, Token Factory API | Nebius Token Factory | Plant-health eye: six side views per stop, agronomist-style description and next step |
| **NVIDIA Cosmos Reason 2 (8B)** | NVIDIA, HF-gated | vLLM on RTX PRO 6000 | Front-camera situation reports for navigation |
| **RT-DETRv2-r18** fine-tuned | PekingU, Apache-2.0 | RTX PRO 6000 / Jetson | Fruit grading (ripe / turning / green / lesion), plant base, leaf chlorosis, leaf spots. Trained only on synthetic renders |
| **NVIDIA cuVSLAM** v17 | NVIDIA Community Licence | RTX PRO 6000 (Jetson-ready) | Stereo / stereo-inertial / RGB-D visual odometry for R181-S |
| **lingbot-map** | Robbyant, Apache-2.0 | RTX PRO 6000 | Streaming 3-D reconstruction from the front camera |
| **MapAnything**, **VGGT**, **MASt3R-SLAM**, **SLAM3R** | Meta / NAVER / PKU, research licences | RTX PRO 6000 | Benchmarked against the above; see §7.3 |
| **NVIDIA Isaac Sim 5.1** | NVIDIA | RTX PRO 6000 VMs | Digital twin, RTX path tracing, PhysX LiDAR |

## 4. The robot: R180 → R181-S

| | R180 (run 3) | R181-S (re-render) |
|---|---|---|
| Drive | 4-wheel skid steer, 0.53 m wide, 0.75 m long, 0.23 m/s in the aisle | same |
| LiDAR | RPLIDAR C1 class, 2-D, 360°, 10 Hz, 12 m | same |
| Front camera | one RGB, 1280×720 | **Stereolabs ZED 2i** (polariser): stereo 12 cm baseline, 1080p30, depth 0.3–20 m, IMU |
| Side cameras | six fixed cameras on a mast (left/right × low/mid/high), 1080×1920 portrait | same |
| Compute target | Jetson | Jetson (cuVSLAM runs there) |

The body, mast and side cameras did not change; the stereo camera is mounted where the old front camera was. Nothing was trained for R181-S.

## 5. Driving and deciding

1. **Goal**: HERMES sends `survey_farm`. The robot expands it into legs: path → enter → drive_in (tunnel-09) → path across the yard → enter → drive_out (tunnel-10) → path → dock.
2. **In the aisle**: BOOK keeps the robot centred between the beds (LiDAR), stops every 0.45 m (plant pitch), waits 0.5 s for the image to settle, fires all six side cameras, drives on. 161 stops on the route.
3. **Localisation**: every 0.1 s the scan is matched to the prior map; the match is applied to heading in the aisles and to full position outside. The front and back walls and the row ends are used as landmarks.
4. **Safety**: LiDAR front stop; a scan-based stuck check (90th-percentile beam change < 3 cm over 5 s while the wheels are driven) that survives plant stops; spin-clearance check before turning around.
5. **Server loop**: frames go to the hub; findings come back; HERMES clusters disease flags within 1 m and, after the survey, sends the robot back to the two strongest spots. On run 3 both revisits came back *healthy*, so the two flags were cleared as false alarms — the loop did its job.
6. **Situations**: when SAFETY stops the robot, the front frame goes to Cosmos Reason; HERMES waits for people and animals, and notifies the farmer for a hose, a cart or a fallen plant.

## 6. Simulation and scenes

- **Isaac Sim 5.1**, 120 Hz PhysX, LiDAR as PhysX range sensor, cameras rendered on demand (side cameras only at stops).
- **Original scene** (run 3): our farm twin with tunnel-09/10, 672 tomato plants swapped to the Tomato-Assets-V1 library, 9 synthetic anomaly plants (2 chlorosis, 3 leaf spots, 2 fruit lesion, 1 wilting, 1 defoliation), doors opened and colliders added at runtime.
- **Astra permanent scene** (29 Oct): proper doors with frames and rolled curtains, yard connector, same 672 plants and anomalies, physics colliders authored. RTX path tracing, 32 spp + OptiX denoiser, HDRI daylight, ISO 100 (ISO 400 for the side cameras inside the canopy).
- **Re-render**: the recorded route of run 3 was replayed on the Astra scene with every camera: chase, front, six side cameras, ZED 2i left/right/depth + IMU, seven fixed CCTV cameras and three handheld "bystander" views that record only while the robot passes. ~25,000 frames on 16 RTX PRO 6000 (two VMs) in ~2 h.

## 7. Results

### 7.1 The route (run 3, original scene, robot driving itself)
- Dock → tunnel-09 front door → 38 m aisle → back door → yard → tunnel-10 back door → aisle → front door → dock, plus two revisits. 123 m, 12.5 min of sim time, 7,500 control steps, no collision, no stuck event.
- Robot's own position vs. truth: **RMSE 0.23 m, median 0.14 m, max 0.50 m** (wheels + gyro + LiDAR scan-to-map; 7,497 of 7,498 scan matches accepted).

### 7.2 Localisation and 3-D mapping, Astra scene, full route (111 m, 6,476 stereo frames)
| Method | Input | Error vs. true path (RMSE / median / max) | Speed | GPU memory |
|---|---|---|---|---|
| **cuVSLAM stereo** (R181-S) | ZED 2i left+right, 15 Hz | **0.064 / 0.041 / 0.16 m**, metric scale 1.003 | 526 fps | ~1.7 GB |
| cuVSLAM stereo + IMU | + simulated IMU 400 Hz | 0.22 / 0.21 / 0.34 m | 422 fps | ~1.7 GB |
| cuVSLAM RGB-D | left + depth | failed on the full route (13.8 m; 6 mm on a 30 s segment) | 243 fps | — |
| **lingbot-map** | front camera only, 2 Hz | 1.06 / 0.87 / 1.95 m (scale fitted) | 11.6 fps | 17 GB |
| R1 LiDAR + wheels (reference) | 2-D LiDAR + odometry | 0.25 / 0.18 / 0.49 m | real time on the robot | — |

cuVSLAM's dense map: 2.75 M points from the ZED depth placed with cuVSLAM's own poses (no ground truth), both greenhouses with correct arched roofs and plant rows (fig. `fig_cuvslam_map_top_and_section.png`, video `cuvslam_3d.mp4`).

### 7.3 SLAM benchmark, original scene, full route (1,296 front frames at 2 Hz)
| Method | Error (RMSE) | Speed | GPU | Note |
|---|---|---|---|---|
| lingbot-map | 1.22 m | 12.1 fps | 15.5 GB | only camera method that kept the two greenhouses apart |
| MASt3R-SLAM | 7.45 m | 8.4 fps | 62.7 GB | good inside one aisle (1.07 m), lost the layout across the yard |
| MapAnything | 12.6 m | 5.0 fps | 45.5 GB | batch; identical aisles collapsed |
| VGGT | 13.1 m | 3.9 fps | 27.2 GB | batch; same failure |
| SLAM3R | 14.5 m | 9.2 fps | 30 GB | only the first ~380 frames registered |
| cuVSLAM mono | 15.8 m | 302 fps | 1.7 GB | mono at 2 Hz cannot initialise; stereo fixes it (§7.2) |

Lesson: a single camera at 2 Hz is a hard case for every method; stereo (R181-S) or LiDAR is what makes localisation reliable, and lingbot-map is the right tool for the 3-D map.

### 7.4 Plant inspection, Astra scene, 161 stops × 6 cameras = 966 images
**Harvest (RT-DETR, fruit counted once per 0.5 m strip and height band):**

| Aisle | Ripe (pass) | Not ready | Lesion (reject) |
|---|---|---|---|
| tunnel-09 | 1,780 | 2,242 | 53 |
| tunnel-10 | 1,769 | 2,242 | 32 |

**Plant count (RT-DETR plant base):** 16 and 29 plants found of ~168 per aisle — not usable on this scene (the base is hidden by the beds); counting should use the LiDAR stem detector instead.

**Disease screening against the 9 hidden anomaly plants:**

| Anomaly (count) | RT-DETR | Nemotron VL (6 views/stop) | Either |
|---|---|---|---|
| Chlorosis (2) | 2/2 | 2/2 | 2/2 |
| Fruit lesion (2) | 2/2 | 0/2 | 2/2 |
| Leaf spots (3) | 0/3 | 2/3 | 2/3 |
| Wilting (1) | n/a | 0/1 | 0/1 |
| Defoliation (1) | n/a | 0/1 | 0/1 |
| **Total** | 4/7 detectable | 4/9 | **6/9** |
| False alarms | 568 boxes away from any anomaly plant | 12 plants | — |

Nemotron VL per stop: 161 requests, 322 plants judged, 1.25 M tokens, ~8 s per request, each answer a 3–5 sentence agronomist description plus a next step (e.g. "upper-canopy yellowing, lower leaves green; check soil nitrogen and irrigation uniformity"). Wilting and defoliation are whole-plant shapes that the 1 m side cameras cannot see; a farther view (CCTV or the front camera) is the fix.

### 7.5 Live run 3 (original scene): the two-eye screen + HERMES
- 978 side images → Cosmos Reason 2 (8B) + Nemotron 3.5 Super VL per image: 52 suspect shots, 4/9 anomaly plants found, 37 shots flagged away from any anomaly.
- RT-DETR in the same run: fruit grading on every image; the dashboard keeps *Plant health* and *Harvest* as separate panels.
- HERMES ordered revisits at tunnel-09 metre 0 and 1.4 (chlorosis flags); both revisits returned healthy → cleared.

### 7.6 Compute and cost (29 Oct)
- Render farm: 16 × RTX PRO 6000 for ~2 h, ~46 GPU-hours for the day in total (roughly US$45–85 at list prices).
- Token Factory: ~2.5 M tokens for all Nemotron VL runs (cents to a few dollars).
- cuVSLAM: 12 s for 6,476 stereo frames on one GPU.

## 8. The one-system pipeline (what runs automatically next time)

```
robot (sim or Jetson)                       Nebius server
 survey ─ frames ──────────────▶ hub /frame  (RT-DETR + VLM eyes, HERMES goals back)
 survey done ─ /run_complete ──▶ run_complete.json ──▶ pipeline watcher
                                               detect ▸ vlm_stops ▸ lidar_map ▸ cuvslam ▸ lingbot ▸ cosmos_front ▸ report
                                               <run>/analysis/report.json · report_th.md · report_en.md
```
`python3 -m cloud.pipeline watch --runs-dir <dir>` (systemd unit provided). Steps are idempotent and isolated; a missing GPU tool is a *skip with reason*, not a failure. Details: `docs/PIPELINE_th.md`.

## 9. Honest limitations
- Everything is simulation; the sim images are noise-free and perfectly calibrated, so cuVSLAM's 6 cm and the detector's numbers will be worse on a real camera.
- RT-DETR was trained on synthetic renders only and over-flags chlorosis on this scene; Nemotron VL is the filter that keeps the farmer's report short.
- Wilting and defoliation are not detected yet.
- Plant counting by RT-DETR failed on the new scene.
- Cosmos Reason's navigation role is implemented and unit-tested but has not been run live on the new scene (model download needs the owner's Hugging Face token on the server).
- The scene, doors and plant colliders are modelled, not surveyed.

## 10. Figures and video sources (for the document)

All figures are in `docs/media/figures/`. Videos are raw renders (no cuts) in `TheFarm/r1_runs/farm_2026-10-29/render_v2_astra_scene/videos/`; the original run's videos in `…/vm_runs/farm_full3/videos/`.

| File | Shows | Source |
|---|---|---|
| `docs/media/r1_cctv_tunnel09.gif` | cover: R1 driving tunnel-09, CCTV view, 8× | `shots/cctv2_t09_head_2.mp4` 0–48 s |
| `fig_cctv7_overview.jpg` | both greenhouses, robot leaving the dock | `shots/cctv7_front_overview_1.mp4` @12 s |
| `fig_chase_door_t09.jpg` | chase camera at the tunnel-09 door | `chase.mp4` @38 s |
| `fig_chase_yard.jpg` | crossing the yard behind the greenhouses | `chase.mp4` @5:00 |
| `fig_storyboard_cctv.jpg`, `fig_storyboard_bystander.jpg` | all CCTV / bystander angles | storyboard renders |
| `fig_r181s_zed_stereo_depth.jpg` | ZED 2i left / right / depth in the aisle and yard | storyboard |
| `fig_run3_route_vs_estimate.png` | the real route vs. the robot's own estimate, plant stops | run 3 `steps.jsonl` |
| `fig_lidar_2d_scan.png`, `fig_lidar_slam_map_astra_scene.png` | 2-D LiDAR scan; LiDAR SLAM map of the farm | `lidar.mp4` @5:00; `slam_map_final.png` |
| `fig_slam_paths_astra_scene.png` | cuVSLAM / lingbot / LiDAR paths vs. truth | §7.2 |
| `fig_cuvslam_map_top_and_section.png` | cuVSLAM dense map, top view and cross-section | `cuvslam_map.ply` |
| `fig_cuvslam_3d_chase.jpg`, `fig_cuvslam_3d_orbit.jpg` | 3-D map growing; final orbit | `cuvslam_3d.mp4` @30 s, @46 s |
| `fig_lingbot_map_topdown.png`, `fig_lingbot_3d_orbit.jpg` | lingbot-map 3-D map | `lingbot_3d.mp4` @44 s |
| `fig_mast3r_original_scene.jpg` | MASt3R-SLAM path and map, original scene | benchmark |
| `fig_cuvslam_mono_failure.png` | why mono at 2 Hz fails | benchmark |
| `fig_side_ai_chlorosis_stop26.jpg` | RT-DETR boxes + Nemotron VL explanation on the chlorosis plant | `side_ai_disease_explained.mp4` @49 s |
| `fig_side_ai_harvest.jpg` | fruit grading boxes | `side_ai_harvest.mp4` @20 s |
| `fig_astra_doors.jpg` | the four greenhouse doors of the permanent scene | Astra |
| Interactive 3-D maps | https://claude.ai/artifact/CExpeKcE3DEmoCgeUKftwk | — |
| SLAM benchmark page | https://claude.ai/artifact/N2YRTHArKStwgH7mWNpigM | — |

## 11. Demo video plan (one continuous 5-minute film, English narration)

Picture-in-picture: main = chase camera (continuous, real time or 2×), inset = front camera (ZED left) and one side camera (right mid). The robot never cuts away; overlays pause the action.

| Time | On screen | Narration (gist) |
|---|---|---|
| 0:00 | CCTV-7 overview, robot leaves the dock | What RoboFarm is; one sentence on Isaac Sim twin; "nothing you see is scripted" |
| 0:30 | Chase cam through the door; LiDAR inset | Edge stack: BOOK, SAFETY, LiDAR localisation; 0.23 m over 123 m |
| 1:00 | First plant stops; side inset flashes | Six cameras per plant; frames stream to the Nebius hub |
| 1:30 | Stop at metre 11.85 — **freeze**, box slides in with RT-DETR boxes, then Nemotron's text types out | Harvest vs. disease kept separate; "Nemotron 3.5 Super VL reads all six views in one request" |
| 2:15 | Resume; HERMES panel: `revisit tunnel-09 m 12.1` | The planner (Nemotron 3) decides; it only ever picks goals |
| 2:40 | Back door → yard; CCTV-4 cut-in; SAFETY stop demo (bystander) | Cosmos Reason explains stops; HERMES waits or notifies |
| 3:10 | Tunnel-10; ZED stereo + depth inset; cuVSLAM path overlay | R181-S: 6 cm over 111 m at 500 fps, Jetson-ready |
| 3:45 | 3-D map growing (cuVSLAM), then lingbot-map | Server-side mapping on Nebius; the benchmark in one line |
| 4:20 | Return to dock; report card: harvest numbers, 6/9 anomalies, 12 false alarms | Results and limits, honestly |
| 4:50 | Pipeline diagram | "Next run, all of this happens by itself" |

Narration voice: a modern neural TTS (edge-tts `en-US-AndrewMultilingualNeural` or ElevenLabs if a key is available); script written for speech, not for reading.

## 12. สรุปภาษาไทย

RoboFarm คือหุ่นตรวจแปลงมะเขือเทศ R1 (รุ่นใหม่ R181-S ใส่กล้องคู่ ZED 2i) ขับเองด้วยแผนที่แปลง + LiDAR ไม่ใช้แท็ก หยุดถ่าย 6 กล้องทุกต้น แล้วส่งภาพขึ้นเซิร์ฟเวอร์ Nebius ฝั่งเซิร์ฟเวอร์: RT-DETR คัดเกรดผล/หาใบผิดปกติทีละภาพ, Nemotron 3.5 Super VL ดู 6 ภาพต่อต้นในคำขอเดียวแล้วอธิบายแบบนักวิชาการเกษตร, HERMES (Nemotron 3) ตัดสินใจย้อนกลับไปดูหรือแจ้งเกษตรกร, Cosmos Reason อธิบายเหตุการณ์หน้ารถให้ HERMES, และสร้างแผนที่ 3 มิติด้วย cuVSLAM (จากกล้องคู่) และ lingbot-map (จากกล้องหน้า) ทุกอย่างทดสอบใน Isaac Sim ฝาแฝดฟาร์มจริง เส้นทางในวิดีโอคือเส้นทางที่หุ่นขับเองจริง

ผลหลัก: ตำแหน่งจาก LiDAR คลาด 0.23 ม. ตลอด 123 ม.; cuVSLAM กล้องคู่คลาด 6.4 ซม. ตลอด 111 ม. ที่ 526 ภาพ/วินาที; lingbot-map 1.06 ม.; RT-DETR + Nemotron VL ร่วมกันเจอต้นผิดปกติ 6 จาก 9 ต้น (ยังไม่เจอต้นเหี่ยวกับใบร่วง) โดย Nemotron แจ้งเตือนเกินแค่ 12 ต้น; นับผลสุก/ไม่สุก/เสียแยกรายโรงเรือน; ระบบอัตโนมัติครบวงจรอยู่บน git แล้ว (114 เทสต์) ข้อจำกัดอยู่ใน §9 รูปและคลิปต้นทางสำหรับทำเอกสารอยู่ใน §10 และแผนวิดีโอเดโม 5 นาทีใน §11
