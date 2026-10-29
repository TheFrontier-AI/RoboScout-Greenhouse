# Competition plan: Nebius x NVIDIA Global AI Hackathon — Physical AI track

**Entry:** RoboFarm (R1 robot): the robot enters a greenhouse it has never seen, builds its own map, finds disease by itself and decides by itself where to go and look.
**Submission:** 30 Oct 2026 10:00 PT = **31 Oct 00:00 Thailand time** (target to actually submit: 30 Oct 20:00)
**Principle:** use off-the-shelf models that can be deployed immediately, everywhere. Train nothing in this round, except TRAIN, which is a side task.

## Against the Physical AI track requirements
| Requirement | Our plan |
|---|---|
| An embodied/edge agent that **perceives and acts in the real world** | The R1 robot in a **digital twin built from the real farm** (the tunnel-09 scene tuned against real photos) + **real data through the system** (real greenhouse photos, real drone video) |
| Powered by Nemotron / GR00T / Cosmos / Sonic | **Nemotron** (brain) + **Cosmos Reason** (eyes) + **Cosmos Transfer** (synthetic data); GR00T/Sonic are for humanoid robots, we do not force them in |
| Coordinated through an agent runtime | **Hermes Agent** is the runtime that commands SLAM / the robot / Cosmos through skills |
| **Nebius Serverless Jobs**: simulation, synthetic data generation, policy evaluation, sensor processing | (1) Isaac Sim headless running BOOK evaluation with randomised wet floor/obstacles, hundreds of rounds (2) Cosmos Transfer generating greenhouse images under many conditions (3) 3-D reconstruction from video the robot filmed |
| **Nebius Serverless Endpoints**: real-time inference | Cosmos Reason endpoint inspecting images while driving |
| The video must include a clip of ≥ 1 minute of **real hardware working** (or the core module if there is no hardware) | No hardware ready in this round → **a clip of the core module running continuously for ≥ 1 minute** in the digital twin (allowed by the rules) |

## Sim-to-Real approach (told in the video as Real → Sim → Real)
1. **Real → Sim (done):** the twin is built from the real farm (evidence below)
2. **Sim (this competition round):** develop and evaluate every module in the twin, on Nebius Serverless Jobs, hundreds of rounds including wet floor/obstacles
3. **Sim → Real (next):** the same stack goes onto the Jetson on the real R1 robot; the SAFETY/MISSION/BOOK code does not depend on the sim
   (`r1_nav/` does not use the ground-truth pose from the sim, only sensors). The only part that changes is the adapter (`sim_adapter.py` → ROS 2 on the Jetson),
   and Cosmos Transfer makes the sim images varied and realistic to close the sim-to-real gap of the image inspector.

## Real → Sim: evidence that the sim comes from the real world (already available)
- Real greenhouse photos: `refs/02_tomato_greenhouses/` (aisles A/B/C, raised beds, irrigation, doors/fans)
- A 2:33 min drone flight over the real farm: `refs/06_video/drone_flyover_site_2m33s.mp4`
- Scene tuned against the real photos: `C2_R180_tunnel09_photo_match.usda` + the 4-way comparison image `compare_4way_RT100_PT100.jpg`
  (aisle brightness: real 126 / Isaac 128)
- The R1 robot is designed from parts that can actually be bought in Thailand (`MASTER_PROCUREMENT_REGISTER.csv`)
- **Real data through the real system:** the drone video goes through Serverless Jobs to build the 3-D farm (MASt3R/VGGT), then is compared with the V18 model;
  real greenhouse photos go through the Cosmos Reason endpoint to inspect plant condition

## What the judges will see in 3 minutes
1. **0:00–0:25 Real → twin:** real photos next to the Isaac scene, and the real drone video becoming a 3-D farm (Serverless Jobs)
2. **0:25–1:45 Core module running continuously (≥ 1 minute):** R1 enters tunnel-09 with no map → SLAM builds the map → HERMES (Nemotron) plans
   → the Cosmos Reason endpoint finds an abnormal leaf → HERMES orders a revisit → the robot arrives and confirms
3. **1:45–2:15 Real photos into Cosmos:** the same system inspects photos from the real greenhouse
4. **2:15–2:40 Serverless Jobs:** BOOK evaluated over hundreds of rounds on wet floor/obstacles, with success rates
5. **2:40–3:00** the map with disease pins, the report from HERMES to the farmer, the plan ahead (real robot being assembled, Jetson)

## Architecture
```
🤖 Edge: Jetson (real) / R1 (Isaac Sim)              ☁️ Nebius Serverless Endpoints
 ├ 🛑 SAFETY   stop/STUCK/e-stop                       └ 🟢 Cosmos Reason   disease inspection + confirms the robot drove correctly
 ├ 🗺️ MISSION  discover/survey/revisit/dock           ☁️ Nebius Serverless Jobs
 ├ 📘 BOOK     drives the wheels down the aisle centre   ├ 🧪 Isaac Sim headless  BOOK evaluation, hundreds of rounds
 └ 📡 SLAM     RTAB-Map / slam_toolbox (LiDAR+camera)  ├ 🟢 Cosmos Transfer     synthetic images (wet/sunny/diseased)
        │ images + pose + status ▲ goals               └ 🧊 MASt3R-SLAM/VGGT + 3DGS  3-D greenhouse
        └──────────────────────┴─────────────────────  ☁️ Nebius Token Factory
                                                        └ 🧠 HERMES (agent runtime) + 🟢 Nemotron 3
```

## 4 work streams (in parallel)
| Stream | Work | Fallback if blocked (still within the plan) |
|---|---|---|
| **A Twin + robot** | R1 in the sim: BOOK, SLAM, `discover_house`, abnormal leaves in the scene; real ↔ twin comparison clip | Camera SLAM blocked → slam_toolbox (LiDAR) |
| **B Nebius Serverless** | Endpoint: Cosmos Reason (sim + real photos); Jobs: BOOK evaluation in Isaac, real drone video → 3-D, Cosmos Transfer | Isaac in Jobs does not support an RTX GPU → run the evaluation on a Nebius AI Cloud 6000 VM instead |
| **C HERMES** | skill `r1-robot-control`, Nemotron via Token Factory, the loop: cover every aisle → find disease → revisit → return to dock | tool calls unstable → JSON schema + the robot re-validates commands |
| **D Submission** | Dashboard with map + disease pins, README, LICENSE Apache 2.0, video ≤ 3 min, Devpost | — |

## Schedule
| When | A Edge + sim | B Serverless | C HERMES | D Submission |
|---|---|---|---|---|
| **28 Oct** | sim env working, BOOK drives the centre aisle | Cosmos Reason endpoint, drone video → 3-D Job | Token Factory key, Hermes ↔ Nemotron | register on Devpost |
| **29 Oct** | SLAM + discover in the sim, abnormal leaves in the scene | BOOK evaluation Jobs, Cosmos Transfer, Cosmos on real photos | robot-command skill, revisit loop | dashboard, real ↔ twin clip |
| **30 Oct** | run the full demo 3 times | 3DGS + disease pins, evaluation summary | report to the farmer | record the video, public repo, **submit 20:00** |

## Side tasks (the submission is complete without them)
- **COMPASS / X-Mobility** (NVIDIA, deploy only), evaluated against BOOK in the same Serverless Jobs set
- **TRAIN (TD-MPC2)**: the only thing that needs real training; kept for after the competition

## Main risks
1. No real hardware in the video: compensated by the Real → Sim evidence and real data through the system (real photos + real drone video)
2. Serverless Jobs: must check whether a GPU with RT cores is available for Isaac Sim, and use an x86 container
3. The Isaac env (Astra): slow, and reversing is not yet supported
4. Cosmos Reason not yet found on Token Factory → build our own Serverless Endpoint
5. Licence: the repo must be Apache 2.0; farm data, 3-D scenes and our own fine-tuned models need not be in the repo
