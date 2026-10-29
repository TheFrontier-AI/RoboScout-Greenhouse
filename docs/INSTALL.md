# RoboFarm — install and run (cloud + robot)

RoboFarm is two programs that talk over HTTP + rsync:

```
 ROBOT (Jetson Orin / any Linux, ARM64 or x86)          CLOUD (Nebius GPU VM, Ubuntu 24.04 + CUDA)
 ┌──────────────────────────────────────────┐           ┌──────────────────────────────────────────────┐
 │ r1_nav/      navigation, LiDAR localiser │           │ agent/robot_api.py   hub :8090                │
 │ sim/         recorder + hub client       │  frames,  │   POST /frame /status /run_complete            │
 │ scripts/run_farm.py (Isaac Sim drive) or │  status   │   GET  /next_command /run_status /health       │
 │ scripts/robot_upload_run.py (hand-off)   │ ────────▶ │ cloud/pipeline/   detect → vlm_stops → lidar_map│
 │ cuVSLAM (optional, NVIDIA wheel)         │  rsync    │   → cuvslam → lingbot → cosmos_front → report  │
 └──────────────────────────────────────────┘           │ Token Factory: Nemotron 3.5 Super VL, Nemotron 3│
                                                        └──────────────────────────────────────────────┘
```

A survey produces one **run directory** on the robot (`<run>/rec/`: `steps.jsonl`, `frames.jsonl`, `events.jsonl`,
`side/*.png` six mast cameras per plant stop, `front/*.jpg`, optional `zed/`). The robot rsyncs it to the server and
posts `/run_complete`; the pipeline watcher writes `<run>/rec/analysis/report.json`, `report_th.md`, `report_en.md`.

Everything below was verified from a fresh `git clone` on a Nebius `gpu-rtx6000` VM (Ubuntu 24.04, CUDA 13) and on an
ARM64 machine standing in for the Jetson. Steps marked *optional* can be skipped; the pipeline skips those steps.

---

## 0. Supported platforms

Verified (logs in [evidence/](evidence/README.md)): **robot** on a Jetson Orin Nano Super (JetPack 6, Python 3.10) and on an ARM64 DGX Spark; **cuVSLAM** on the Jetson (cuvslam 17.0.1 cu12 aarch64, 188 FPS stereo); **cloud** on a Nebius `gpu-rtx6000` VM (Ubuntu 24.04, CUDA 13, Python 3.12). Expected to work: any Linux with Python ≥ 3.10 for the robot side; any x86-64 Linux with an NVIDIA GPU (CUDA 12.x/13.x) for the cloud side; CPU-only cloud runs with GPU steps skipped. Simulation needs NVIDIA Isaac Sim 5.1 on x86-64.

## 1. Cloud (Nebius VM)

**VM:** Ubuntu 24.04 + CUDA image (`ubuntu24.04-cuda13.0`), one RTX PRO 6000 (`gpu-rtx6000`, preset `1gpu-24vcpu-218gb`)
is enough; 200 GB disk; a public IP; port 8090 reachable from the robot (or an SSH tunnel). Default user `ubuntu`.

```bash
git clone https://github.com/TheFrontier-AI/RoboScout-Greenhouse.git
cd RoboScout-Greenhouse
bash deploy/nebius/install_cloud.sh          # apt (ffmpeg, rsync, curl, python3-venv), .venv with torch cu130 + deps, offline tests
```

**Keys** — copy the template and put your [Nebius Token Factory](https://tokenfactory.nebius.com) key in it
(`NEBIUS_API_KEY`). Nothing else is required; the file is also the place for the optional settings listed in §4.

```bash
cp deploy/nebius/.env.example deploy/nebius/.env
nano deploy/nebius/.env                      # NEBIUS_API_KEY=...   (never commit .env)
```

**Detector weights** — the fine-tuned RT-DETRv2 (`vision_runs/rtdetr_v2_mixed/best/`, 77 MB: `config.json` +
`model.safetensors`) is **not in the repository**. It was trained by the author on Isaac Sim renders of our greenhouse
mixed with public greenhouse photos (AgRob, PlantDoc). **Contact the author for the weights** and place them at
`vision_runs/rtdetr_v2_mixed/best/`. Without them the `detect` step is skipped and the report has no fruit counts.
(To train your own: `vision/README.md`.)

**Start the system** (hub + pipeline watcher), either by hand or as a service:

```bash
bash deploy/nebius/start_pipeline.sh                   # foreground: hub on :8090, watcher on ~/runs, logs in ~/runs/logs
# or
bash deploy/nebius/install_cloud.sh --service          # systemd unit `pipeline`: enable --now, restarts on reboot
journalctl -u pipeline -f
curl -s localhost:8090/health                          # {"ok": true, ...}
```

The watcher polls `~/runs/*/rec` (`RUNS_DIR`) for a `run_complete.json` without `analysis/pipeline.done` and runs the
pipeline on it. You can also run it on any run dir directly:

```bash
.venv/bin/python -m cloud.pipeline run ~/runs/farm_001/rec            # all steps; --steps detect,report --force
```

*Optional GPU steps* (installed with flags; skipped when absent):

```bash
bash deploy/nebius/install_cloud.sh --cuvslam     # NVIDIA cuVSLAM wheel into ~/cuvslam_venv (stereo / stereo+IMU / RGB-D on zed/)
bash deploy/nebius/install_cloud.sh --lingbot     # docker image lingbot:blackwell (camera-only 3-D map of the front camera)
```
cuVSLAM: if `pip install cuvslam` finds no wheel for your platform, download it from
<https://github.com/nvidia-isaac/cuVSLAM/releases> and `~/cuvslam_venv/bin/pip install <wheel>`. lingbot-map needs
docker + the NVIDIA container toolkit (`deploy/nebius/bootstrap_vm.sh` installs both); its weights (robbyant/lingbot-map,
~1 GB) download on the first run. Cosmos Reason 2 for the front camera is a vLLM container (`deploy/nebius/docker-compose.yml`,
service `eyes`, gated model: set `HF_TOKEN`); set `EYES_BASE_URL=http://127.0.0.1:8000/v1` to enable the `cosmos_front` step.

---

## 2. Robot (Jetson Orin or any Linux box)

Python ≥ 3.10, `rsync`, `ssh`. The navigation stack, recorder and hub client are pure Python (numpy + pillow); nothing
here needs a GPU. On a real robot the sensors arrive through the ROS 2 adapter (see `r1_nav/README` roadmap); in
simulation `scripts/run_farm.py` drives the digital twin in Isaac Sim 5.1.

```bash
git clone https://github.com/TheFrontier-AI/RoboScout-Greenhouse.git
cd RoboScout-Greenhouse
bash deploy/robot/install_robot.sh                     # .venv-robot + requirements/robot.txt + the robot-side offline tests
source .venv-robot/bin/activate
export ROBOT_API_URL=http://<SERVER_IP>:8090           # the hub
```

*cuVSLAM on the Jetson (R181-S stereo camera, optional):* install NVIDIA's wheel into the same venv and point it at
the JetPack CUDA libraries — verified on a Jetson Orin Nano Super (JetPack 6): stereo 188 fps, stereo + IMU 82 fps.

```bash
.venv-robot/bin/pip install https://github.com/nvidia-isaac/cuVSLAM/releases/download/v17.0.1/cuvslam-17.0.1+cu12-cp310-cp310-manylinux_2_35_aarch64.whl
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
.venv-robot/bin/python scripts/cuvslam_r181.py --data <zed EuRoC dir> --mode stereo      # or inertial
```

The robot must be able to `ssh ubuntu@<SERVER_IP>` without a password (put its public key in the VM's
`~/.ssh/authorized_keys`); that is the rsync channel for the run directory.

**Drive a survey in Isaac Sim** (needs the scene assets; not required for the hand-off test):

```bash
/isaac-sim/python.sh scripts/run_farm.py --record runs/farm_001 --api-url $ROBOT_API_URL \
    --upload-rec ubuntu@<SERVER_IP>:/home/ubuntu/runs/farm_001/rec
```
At the end of the drive `run_farm.py` writes `summary.json`, rsyncs the run and posts `/run_complete` itself.

**Hand off an already recorded run** (what a real robot does when it docks; also the end-to-end test):

```bash
python3 scripts/robot_upload_run.py --rec runs/farm_001/rec \
    --dest ubuntu@<SERVER_IP>:/home/ubuntu/runs/farm_001/rec --hub $ROBOT_API_URL --wait
```
It uploads only what the pipeline reads (jsonl, `side/`, `front/`, `zed/`), posts `/run_complete`, then polls
`GET /run_status?run=farm_001` until `analysis/report.json` exists. Exit code 0 = the report is on the server.

---

## 2b. Surveying more greenhouses (three, four, …)

The route is generated, not hard-coded: `r1_nav.mission.farm_loop(houses, …)` takes any list of houses and builds the
drive-through — front-to-back in the first, cross behind, back-to-front in the next, and so on, then home. Three ways
to command it (1 is for the simulator; on a real farm you use 2 or 3):

1. **Simulation driver** — give the houses in order (`aisle id : aisle-centre x` in site metres):
   ```bash
   /isaac-sim/python.sh scripts/run_farm.py --record runs/farm_003 --api-url $ROBOT_API_URL \
       --houses "tunnel-09/aisle-02:0.0,tunnel-10/aisle-02:9.015,tunnel-11/aisle-02:18.03"
   ```
   The house must exist in the twin (scene prep swaps its plants in from the same list).
2. **From HERMES / the hub, one house at a time** — `survey_house` and `go_to_house` are validated commands
   (`r1_nav/commands.py`): the target house must be a `door` node on the **site map** (`data/site_map_v18_draft.json`,
   `r1_nav/site_map.py`) and the site map must be **confirmed by a human** (the `farm-site-map` skill walks the farmer
   through it); an unknown or unconfirmed house is rejected with the list of houses the map knows.
3. **Whole-farm order** — `agent/brain_loop.py --farm` sends `survey_farm`; the houses it covers are the ones in the
   confirmed site map, in map order.

Each house needs an aisle map (`data/<house>_aisle_map_*.json`, like `tunnel09_aisle_map_v18.json`) for its row
geometry; identical tunnels can share one map with an x offset, which is how tunnel-10 is handled today. The report
groups harvest and health per house automatically (`by_aisle`), so nothing changes on the cloud side.

## 2d. Mapping by demonstration — teach mode (no distances typed)

The robot makes its own map. Nobody measures the farm or draws a route:

1. **Teach drive.** Put the robot at its dock and walk it once through the farm — push it, drive it with a controller, or
   (in simulation) let it drive — with the recorder running. It records its 2-D LiDAR scans and its own pose estimate
   at 10 Hz (`steps.jsonl`). That is all the input there is.
2. **Build the maps.**
   ```bash
   python3 scripts/teach_map.py --rec <run>/rec --validate
   ```
   writes `<run>/rec/teach/`:
   - `prior_map_xy.npy` — the farm as the LiDAR saw it, scans placed with the robot's own poses (the map the
     localiser matches against; same format the plan-derived map had);
   - `site_map.json` — the site graph: the dock, a door node wherever the drive entered or left a house, and edges
     along the taught path with their measured lengths, plus each house's aisle length — in the `r1_nav.site_map`
     schema, unconfirmed;
   - `teach_map.png`, `teach_report.json`.
3. **Name and confirm.** The farmer renames the houses (`house-1` → "tunnel-11") and confirms the map — by hand in
   the JSON, or by talking to HERMES through the `farm-site-map` skill. Confirmation is what unlocks cross-house
   commands.
4. **Use it.** Start the hub with `R1_SITE_MAP=<run>/rec/teach/site_map.json` and copy `prior_map_xy.npy` where
   the robot loads its prior; then command as in 2b: *survey the farm*, *go to tunnel-11*, *return to dock*. The robot
   plans the route on the graph it built and localises on the map it recorded.

**Verified on run 3 as the taught drive** (`docs/evidence/10_teach_map_run3.json`, figure
`docs/media/gallery/24_teach_map_self_built.png`): from 7 500 steps / 123 m of driving, `teach_map.py` built a
32 967-point map of both greenhouses and the yard, a site graph of the dock and four doors (dock → tunnel-09 front
→ tunnel-09 back → 12.8 m behind the houses → tunnel-10 back → tunnel-10 front → 14.0 m → home) with both aisles
measured at 41.5–41.6 m, and the site map loads, validates and routes in `r1_nav.site_map`. Localising the same
1 500 scans on the self-built map accepted every scan (1 500 / 1 500) with a fix error of 0.23 m RMSE — the same as
on the plan-derived map (0.23 m).

The self-built map carries the pose error of the drive that taught it (median fix 0.14 m against 0.03 m for the
drawn plan); building it from cuVSLAM poses, and letting the robot teach itself with `discover_house`, are the next
steps. Teach mode on the physical robot uses the same recorder with the motors idle.

**Who plans what.** HERMES plans the *mission*: which houses to survey today and in what order, which plant to go
back to, when to come home, whom to tell. The *route* — the actual path from where the robot is to the door of the
house it was sent to — is planned by the robot itself on the site graph (`r1_nav/site_map.py` finds the shortest
confirmed path; `r1_nav/commands.py` turns it into legs), and the *driving* along that route is the robot's
navigation stack alone. HERMES never sees a coordinate or a wheel; it names a destination, and the robot gets there.

## 2c. Which farm? Deploying on another farm (anywhere)

(Put the robot down somewhere other than the dock? See [NEW_SITE_GUIDE.md §6](NEW_SITE_GUIDE.md): `scripts/relocalize_eval.py`, `scripts/vlm_place_check.py`.)

Nothing in the code names our farm. A farm is **three data files plus one confirmation**; point the system at them and
every command above works on that farm. House names are whatever those files call them.

| What | File | Who makes it | How the system finds it |
|---|---|---|---|
| **Site map** — the farm as a graph: `dock`, one `door` per house, `junction`/`waypoint` nodes, edges with length / width / surface, in site metres | `data/<farm>_site_map.json` (schema: `r1_nav/site_map.py`; ours: `data/site_map_v18_draft.json`) | from the farm plan or a walk with a tape measure; the `farm-site-map` skill builds and edits it from the farmer's words | `R1_SITE_MAP=/path/<farm>_site_map.json` in the environment of the hub / brain (comma-separate several) |
| **Aisle map** — one per house: beds, aisle centre lines, lengths, clearance, house-to-site transform | `data/<house>_aisle_map.json` (schema: `r1_nav/aisle_map.py`; ours: `data/tunnel09_aisle_map_v18.json`) | measured once per house type; identical houses share a file with an x-offset; `discover_house` lets the robot survey an unmapped house with LiDAR first | hub `--map <file>` (default house) and the `aisle_maps` passed to `plan_command` |
| **Confirmation** — a human has checked the site map | `metadata.confirmed_by / confirmed_at` inside the site map | the farmer, through the `farm-site-map` skill (`SiteMap.confirm(by=...)`) | commands that move between houses (`go_to_house`, `survey_house` of another house, `survey_farm`) are **rejected until confirmed** |
| **Scene** (simulation only) — the farm's digital twin | USD stage for Isaac Sim | built from photos/plans (Blender → Isaac) | `scripts/run_farm.py` (`STAGE`, `--houses`) — not used on a real robot |

**So, to run the robot on a different farm, in Thailand or abroad:**

1. Write the site map for that farm (`data/<farm>_site_map.json`) with its houses and paths; have its farmer confirm it.
2. Write one aisle map per house type (`data/<house>_aisle_map.json`).
3. Start the hub with `R1_SITE_MAP=data/<farm>_site_map.json` and `--map data/<house>_aisle_map.json`; start the robot with
   `ROBOT_API_URL` pointing at that hub.
4. Command it exactly as in 2b: `survey_house` / `go_to_house` / `survey_farm`. The report is grouped by the house ids of
   that farm automatically.
5. Crops: the detector and the Nemotron prompts are tuned for tomato; another crop needs its own classes (retrain
   `vision/train_rtdetr.py`) and prompt text — the navigation, mapping, agent and report need no change.

The simulation flag `--houses` only tells the digital twin which houses to prepare; it is not how a real robot is
commanded. On a real robot the list of houses *is* the confirmed site map.

## 3. End-to-end test (what we ran)

Raw logs of every step below are in [evidence/](evidence/README.md).

Verified 2026-10-30 from a fresh checkout of `main` (commit `24d2f61`):

1. **Cloud** — Nebius `gpu-rtx6000` VM (Ubuntu 24.04, driver 580, CUDA 13): `install_cloud.sh` took 2 min 15 s
   (torch 2.14.1+cu130, transformers 5.22, 28 offline tests pass); `.env` with the Token Factory key; weights copied
   in; `start_pipeline.sh` → `/health` ok.
2. **Robot** — fresh clone on an ARM64 machine (DGX Spark): `install_robot.sh` → 50 offline tests pass
   (this test caught a missing `scipy` dependency, fixed in the same commit series). Then
   `robot_upload_run.py --wait` handed over the recorded run `farm_full3` (161 plant stops, two greenhouses,
   978 side images, 1 501 front frames, 2.9 GB).
   **Real Jetson** — the same on a **Jetson Orin Nano Super 8 GB (JetPack 6, L4T R36.4.7, Python 3.10)**:
   `install_robot.sh` 1 min 33 s (the JetPack image has no `python3-venv`; the script bootstraps pip itself —
   found and fixed here), **50 tests pass in 49 s**, and `robot_upload_run.py --wait` uploaded a 2-stop run over the
   farm's internet to the hub, which ran the pipeline in 83 s → `"state": "done"`, `report_th.md`/`report_en.md`
   (first contact needed `StrictHostKeyChecking=accept-new` in the rsync ssh options — also fixed here).
3. **Result** — the hub accepted `/run_complete`; the watcher ran `detect → vlm_stops → lidar_map → report`
   (cuvslam skipped: no `zed/` on R1; lingbot skipped: docker not usable by the service user; cosmos_front skipped:
   `EYES_BASE_URL` unset) and finished in **9 min 59 s** end to end; `GET /run_status?run=farm_full3` → `"state": "done"`,
   `analysis/report.json`, `report_en.md`, `report_th.md` present. The report counted 3 276 ripe / 4 534 not ready /
   47 reject fruit over the two houses and listed the suspect plants with Nemotron's descriptions and actions.

Offline checks on either side (no GPU, no network): `python3 -m pytest -q` (114 tests).

---

## 4. Configuration reference (`deploy/nebius/.env`, read by `start_pipeline.sh`)

| Variable | Default | Meaning |
|---|---|---|
| `NEBIUS_API_KEY` | — | Token Factory key: Nemotron 3.5 Super VL per-stop inspection, Nemotron 3 report text |
| `DETECTOR_MODEL` | `vision_runs/rtdetr_v2_mixed/best` | RT-DETRv2 dir (ask the author); `DETECTOR_THRESHOLD` 0.5 |
| `EYES2_BASE_URL` / `EYES2_MODEL` | Token Factory / `nvidia/Nemotron-3.5-Super-VL` | per-plant-stop VLM, six views in one request |
| `BRAIN_BASE_URL` / `BRAIN_MODEL` | Token Factory / `nvidia/nemotron-3-super-120b-a12b` | HERMES planner and report writer |
| `EYES_BASE_URL` / `EYES_MODEL` | unset / `nvidia/Cosmos-Reason2-8B` | Cosmos Reason endpoint for the front camera (optional) |
| `CUVSLAM_PYTHON` | `~/cuvslam_venv/bin/python` | interpreter with the cuvslam wheel (optional) |
| `LINGBOT_RUN` / `LINGBOT_IMAGE` | `deploy/nebius/lingbot/run.sh` / `lingbot:blackwell` | lingbot-map hook (optional) |
| `PIPELINE_STEPS` | all | subset, e.g. `detect,vlm_stops,report` |
| `PIPELINE_AUTO` | 0 | 1: the hub also launches the pipeline on `/run_complete` (watcher still works) |
| `RUNS_DIR`, `HUB_PORT`, `PY` | `~/runs`, 8090, `.venv/bin/python` | start_pipeline.sh |
| `ROBOT_API_URL` | — | robot side: the hub URL |

Models: Nemotron 3.5 Super VL and Nemotron 3 run hosted on Nebius Token Factory (no download). RT-DETRv2 base
(`PekingU/rtdetr_v2_r18vd`) and lingbot-map are public; our fine-tuned RT-DETR weights are available from the author.

## 5. Troubleshooting

- `run_complete` returns 400 `run dir not found` — the `--dest` path and the hub's view of the filesystem differ; the hub
  needs the **server-side** path (`robot_upload_run.py` sends the part after `host:`).
- `detect` skipped — weights missing at `DETECTOR_MODEL`; `vlm_stops`/`report` template only — `NEBIUS_API_KEY` unset.
- torch without CUDA — the wheel index must match the driver (`nvidia-smi` shows CUDA 13.x → `cu130`; 12.x → `cu128`):
  `TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash deploy/nebius/install_cloud.sh`.
- `docker: permission denied` — `sudo usermod -aG docker $USER` and log in again (the lingbot hook falls back to `sudo docker`).
- Watcher sees nothing — the run dir must contain `run_complete.json` and `frames.jsonl`/`steps.jsonl`, one or two levels under `RUNS_DIR`.

---

## สรุปภาษาไทย

ระบบมี 2 ฝั่ง: **หุ่น** (Jetson Orin หรือ Linux อะไรก็ได้ — `deploy/robot/install_robot.sh`) และ **คลาวด์** (Nebius GPU VM —
`deploy/nebius/install_cloud.sh` แล้ว `start_pipeline.sh` หรือ `--service`) หุ่นบันทึก run แล้วส่งขึ้นเซิร์ฟเวอร์ด้วย
`scripts/robot_upload_run.py --wait` เซิร์ฟเวอร์รัน pipeline (ตรวจจับ → Nemotron 6 มุม → แผนที่ LiDAR → cuVSLAM/lingbot (ถ้ามี) →
รายงาน TH/EN) ใส่ key Token Factory ใน `deploy/nebius/.env` อย่างเดียว **น้ำหนักโมเดลตรวจจับที่ฝึกเอง ไม่อยู่ใน git —
ติดต่อผู้เขียน** แล้ววางที่ `vision_runs/rtdetr_v2_mixed/best/`
