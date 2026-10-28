# RoboFarm

**R1 is our greenhouse robot: it enters a house it has never seen, maps it, finds sick plants, and decides by itself where to look again.**
Built for the *Nebius x NVIDIA Global AI Hackathon 2026 — Physical AI track*. Thai docs: [README_th.md](README_th.md).

- **Brain — HERMES:** [Hermes Agent](https://github.com/NousResearch/hermes-agent) is the agent runtime, running **NVIDIA Nemotron 3** on **Nebius Token Factory**. It picks goals (survey, revisit, dock), never wheel speeds.
- **Eyes:** **NVIDIA Cosmos Reason 2** on a **Nebius Serverless Endpoint** / RTX PRO 6000 VM. It inspects the 6 side-camera views of tomato plants and checks from the front camera that the robot moved as commanded.
- **Cloud SLAM AI:** **Nebius GPU Jobs** run feed-forward 3-D reconstruction (MapAnything, VGGT fallback) on robot or drone video. The result is aligned to the greenhouse map.
- **Robot (edge):** the R1 differential-drive robot, simulated in **NVIDIA Isaac Sim** in a digital twin built from our real farm. It runs a rule-based driver (BOOK), a SAFETY layer and MISSION legs that are identical for the real Jetson.
- **Evaluation at scale:** Isaac Sim batch jobs randomise wet-floor friction, obstacles and start offsets, and report success rates against the acceptance criteria below.

```
 Edge: R1 (Isaac twin → Jetson)            Nebius
  SAFETY · MISSION · BOOK · pose ──frames──▶ robot API hub ──▶ Cosmos Reason (Serverless Endpoint)
            ▲ goals only                      │ findings
            └──── HERMES (Hermes Agent + Nemotron @ Token Factory) ◀┘
                     Cloud SLAM AI + Isaac eval  (Nebius GPU Jobs)  ──▶ live dashboard
```

## Sim-to-Real
1. **Real → Sim (done):** the tunnel-09 twin is calibrated against real photos (aisle brightness: real 126, Isaac 128). We also have a real drone survey of the site, and the robot's bill of materials uses parts sold in Thailand.
2. **Sim:** all modules are developed and evaluated in the twin. Real photos and the real drone video go through the same cloud pipeline.
3. **Sim → Real (next):** `r1_nav/` uses only sensors (no simulator pose). Moving to the Jetson means swapping `sim_adapter` for ROS 2.

## Repo map
| Path | What |
|---|---|
| `r1_nav/` | Aisle map, mission legs, BOOK LiDAR aisle-centring controller, goal validation (`commands.py`), 7-camera schedule (`cameras.py`), farm site map between greenhouses (`site_map.py`) |
| `cloud/maps/`, `scripts/extract_site_map_v18.py` | Site-map drafts from the V18 model, a farmer's sketch (Nemotron VL) or a drone top-down; human confirms before use ([docs/SITE_MAP_th.md](docs/SITE_MAP_th.md)) |
| `sim/`, `scripts/run_demo_sim.py` | Leg executor (STUCK, rear stop, heading damping), cloud link with offline outbox, Isaac wrapper, kinematic fake backend |
| `scripts/run_eval_job.py`, `jobs/isaac_eval/` | Batch evaluation (wet floor / obstacle / offsets) as a GPU job |
| `agent/robot_api.py` | Goal hub between HERMES, the robot and Cosmos; writes the dashboard data |
| `agent/hermes_skills/r1-robot-control/` | Hermes Agent skill that commands the robot |
| `agent/brain_loop.py` | Deterministic fallback brain (same policy; Nemotron writes the reasoning and the farmer report) |
| `cloud/eyes/` | Cosmos Reason serving (vLLM) and an inspection client with strict JSON output |
| `cloud/brain/` | Nemotron client for Token Factory and the Hermes provider config |
| `cloud/mapping/` | Cloud SLAM AI job (MapAnything / VGGT) and alignment into house metres |
| `dashboard/` | Live map, finding pins, 7-camera views, HERMES decision feed |
| `deploy/nebius/` | One-VM stack (RTX PRO 6000): bootstrap and docker compose |
| `docs/` | Mission spec, interfaces, environment, roadmap, hackathon plan (Thai) |

## Quick start
```bash
python3 -m pytest -q                                   # 72 offline tests

# Full loop rehearsal without GPU (fake robot + stub eyes):
python3 scripts/stub_eyes.py --port 8001 --spot 15 &
EYES_BASE_URL=http://127.0.0.1:8001/v1 EYES_MODEL=stub EYES_API_KEY=x \
  python3 agent/robot_api.py --data runs/demo --map data/tunnel09_aisle_map_v18.json --port 8090 &
python3 dashboard/serve.py --data runs/demo --port 8080 &          # open http://127.0.0.1:8080
python3 scripts/run_demo_sim.py --backend fake --idle-exit-s 60 &
python3 agent/brain_loop.py --house tunnel-09                     # or: hermes, with the r1-robot-control skill
```
Real stack on Nebius: [deploy/nebius/README_th.md](deploy/nebius/README_th.md). Cloud details and verified model IDs: [cloud/README_cloud.md](cloud/README_cloud.md).

## Phase-1 acceptance criteria (Isaac, tunnel-09 photo-matched scene)
1. Stop at a commanded point in the centre aisle within ±0.15 m, side clearance ≥ 0.05 m, 10/10 runs.
2. Full aisle in and reverse out, no collision, stop within 0.5 m of an obstacle, 10/10 runs.
3. Side images with (aisle, metre, time) every 0.5 m (`docs/INTERFACES_th.md`).
4. One aisle (76 m round trip) in ≤ 8 min at 0.25 m/s.

## Honest status
- The robot itself is still being assembled. Everything robot-side runs in the Isaac twin.
- The side aisles of tunnel-09 (0.55 m) are narrower than the robot plus margin. They are reported as skipped, not attempted.
- Headland transit and docking are not simulated yet (the env has a single start). No policy training was done for this entry.

## License and credits
Apache-2.0 (see `LICENSE`). Third-party models keep their own licences (see `cloud/README_cloud.md` and `cloud/mapping/README_mapping.md`).
The R1 concept is inspired by Sumith Shridhar's R1; no code or models from it are used.
