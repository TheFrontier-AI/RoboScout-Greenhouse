# RoboFarm — Devpost submission draft (Physical AI track)

**Tagline:** An agentic greenhouse robot: it drives its own aisles, spots sick tomato plants with NVIDIA Cosmos, and lets a Hermes agent on Nemotron decide where to look again. It is built sim-to-real on Nebius.

## Inspiration
Our farm grows tomatoes in plastic tunnels. Scouting every aisle by eye is slow, and early leaf disease is easy to miss.
We are building R1, a narrow differential-drive robot with six side cameras. It should walk the aisles every day, flag what looks wrong and go back for a second look before a person is called.

## What it does
- **Drives the aisle on its own:** LiDAR centring, an onboard SAFETY layer (obstacle stop, STUCK detection, reverse-out because the robot can't turn around inside an aisle) and odometry.
  - It stays within 2–16 mm of the true aisle metre and within a few millimetres of the centreline in the digital twin.
- **Sees:** six plant cameras (1080×1920, the real Rapoo C260 resolution) plus a front camera.
  - Every 0.5 m the frames are screened by **two NVIDIA models**: Cosmos-Reason2-8B (self-hosted on an RTX PRO 6000) and Nemotron-3.5-Super-VL (Nebius Token Factory).
  - If either one says "suspect", the spot is revisited.
- **Decides:** **HERMES** is Hermes Agent plus a fallback loop, running **NVIDIA Nemotron 3 Super** on **Nebius Token Factory**.
  - It sends goals only, never wheel commands: survey, revisit a metre, go to another greenhouse, dock.
  - It explains every decision in plain language and writes a Thai and English report for the farmer.
- **Maps:** cloud SLAM (MapAnything) turns real drone video into a 3-D farm map as a **Nebius Serverless Job** (H200, 55 s for 200 frames).
  - Farm-level maps can come from the V18 farm model, the drone, the robot, or **a farmer's hand-drawn sketch read by Nemotron-VL**.
  - A human always confirms a map before the robot uses it.

## How we built it (Nebius + NVIDIA)
| Piece | Where it runs | NVIDIA model / tech |
|---|---|---|
| HERMES planner | Nebius Token Factory | Nemotron 3 Super 120B |
| Plant screening, eye 1 | Nebius AI Cloud VM, RTX PRO 6000 Blackwell (vLLM) | Cosmos-Reason2-8B |
| Plant screening, eye 2 + sketch reading | Nebius Token Factory | Nemotron-3.5-Super-VL |
| Digital twin + robot sim | RTX PRO 6000 (Isaac Sim 5.1 container) | Isaac Sim, PathTracing ISO 400 |
| Cloud SLAM from drone video | **Nebius Serverless Job** (H200) + Nebius Container Registry | MapAnything (VGGT fallback) |
| Robot code | Python, ROS 2-ready contracts | — |

**Sim-to-real.** The twin was built from our real farm photos and drone survey: tomato plants, film, light and exposure were matched by our realism workstream. Real greenhouse photos go through the same eyes as the sim frames. `r1_nav/` uses sensors only (no simulator pose), so the same code moves to the Jetson Orin Nano on the real robot.

## Challenges
- **Subtle disease is hard.**
  - At 180×320 a test lesion was found 2/5 times.
  - At the real 1080×1920 resolution, Cosmos-8B missed a "leaf spots" plant that Nemotron-VL flagged. This led to the two-eye design.
  - Screening is tuned for recall; revisits are tuned for precision.
- **Bugs that offline tests couldn't catch, found in live runs:**
  - Wrapped wheel angles made odometry jump 1.6 m.
  - Far-wall LiDAR returns over 12 m crashed the sensor.
  - Revisit shots from the other bed cleared a real finding. HERMES now waits for all six revisit shots before clearing.
- **Speed and bandwidth:**
  - Real-resolution rendering took 29 s per step. The gated runtime (render once per LiDAR sweep) brought it to about 6 s.
  - A 17 KB/s uplink to the cloud was solved with zstd plus 48 parallel streams.

## Accomplishments
- The full loop runs end to end with real NVIDIA models on Nebius: survey → detect → HERMES revisits → confirm or clear → report.
- **Final run** (RTX PRO 6000, Astra's realism scene, gated runtime about 6 s/step, cameras at 1080×1920):
  - **84 screened frames, 0 false alarms.**
  - **Nemotron-VL flagged the subtle test plant at 15.13 m.** Cosmos-8B missed it.
  - HERMES sent a revisit with its reasoning.
  - Both eyes saw the six still shots as healthy, so the spot was **cleared**. The report says so honestly and asks a person to check.
- An earlier run, with a more visible test lesion, was **confirmed** on revisit (early blight 0.89–0.99).
- About 80 offline tests.
- Every live failure became a regression test.

## What's next
- The physical robot is being assembled from parts sold in Thailand; it will run the same stack on a Jetson Orin Nano with ROS 2.
- Watering: the pump, tank and nozzles are designed; the API is drafted. Then measure the side aisles on site and fine-tune the eyes on our own frames.

## Honest status
- The robot is simulated: all runs are in the Isaac twin.
- The diseased plant is a labelled synthetic test asset, not a real diagnosis.
- Two of tunnel-09's three aisles are narrower than the robot plus margin and are reported as skipped.
- Model outputs are screening aids, not lab diagnoses.

## Links
- Repo: https://github.com/TheFrontier-AI/RoboScout-Greenhouse (Apache-2.0)
- Video: <YouTube link> (file: TheFarm/r1_runs/RoboFarm_demo_v3.mp4, 2:19)
