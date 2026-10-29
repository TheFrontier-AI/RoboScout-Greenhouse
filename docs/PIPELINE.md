# Post-survey analysis system (cloud/pipeline) — robot finishes → server analyses by itself → one report

Previously, after the robot finished a survey we had to run the scripts one by one (`detect_side.py`, `vlm_stop.py`, `make_farm_videos.py`,
`cuvslam_r181.py`, lingbot, `cosmos_front.py` — see `deploy/nebius/post_render.sh`). Now all of it is **one system that runs automatically**:

```
 Robot (Isaac sim / Jetson)                        Nebius server (RTX PRO 6000)
 ─────────────────────────                         ──────────────────────────────────────────────────────────
 scripts/run_farm.py                               agent/robot_api.py (hub :8090)
   drives the survey, records rec/ (steps/events/frames.jsonl,   ◀── POST /status, /frame (while driving: per-image inspection as before)
   side/*.png, front/*.jpg, zed/ if present)
   on finish (finally):                             ◀── POST /run_complete {"run","path","summary"}
     [--upload-rec user@vm:/runs/<name>/rec  rsync]        │ writes <run>/run_complete.json  (+ runs.jsonl in the data dir)
     POST /run_complete                                    │ PIPELINE_AUTO=1 → spawn `python3 -m cloud.pipeline run <run>`
                                                           ▼
                                                   python3 -m cloud.pipeline watch --runs-dir ~/runs   (pipeline.service)
                                                     sees a run_complete.json without analysis/pipeline.done → runs one run at a time:
                                                     detect → vlm_stops → lidar_map → cuvslam → lingbot → cosmos_front → report
                                                           ▼
                                                   <run>/analysis/report.json + report_th.md + report_en.md
```

- In the **sim** the robot's `rec/` folder and the hub are on the same disk (shared mount), so only the path is sent; no upload is needed.
- On the **real robot** pass `--upload-rec user@server:/home/ubuntu/runs/<name>/rec` (or env `R1_UPLOAD_REC`); the robot rsyncs `rec/` up first,
  then POSTs the server-side path. If rsync fails it does not POST (it can be rerun later with `--upload-rec`).
- Hub down when the robot finishes → write `run_complete.json` into rec yourself (the watcher only looks for this file), or `curl -X POST :8090/run_complete -d '{"path": "..."}'`
- Steps that need a GPU or an external service (cuvslam, lingbot, cosmos_front) are **skipped with a reason** when the tool is missing; they do not make the other steps fail.

## Steps (cloud/pipeline/steps.py) and their outputs
Every step writes to `<run>/analysis/<step>/` and leaves a marker `analysis/<step>.done` (JSON: status `ok|skipped`, reason, result, timing).
A rerun skips steps that already have `.done` (`--force` = run again). A failed step gets `.error` instead and is retried on the next round.

| Step | What it does | Existing code used | Output files | Skipped when |
|---|---|---|---|---|
| `detect` | RT-DETR on every side-camera image (`side/*.png` or `frames/*_mast_*.jpg`); summary per camera + 3 categories: ① fruit grading ② plant count ③ leaf anomalies (compared with ground truth if available) | `vision.detector`, `scripts.detect_side.summarize` | `findings.jsonl`, `by_camera.json`, `summary.json` | no images / no `DETECTOR_MODEL` |
| `vlm_stops` | **One request per plant stop**: groups the 6 images from `frames.jsonl` (left high/mid/low = left plant, right = right plant), sends them to Nemotron VL via Token Factory with N concurrent workers, asks for an agronomist-style description + `action` per plant | `scripts.vlm_stop` (`group_png_stops`, `ask_stop`, `summarize`) | `vlm_stops.jsonl` (per stop), `plants.json` (per plant: status, issues, description, action, images), `summary.json` | no `EYES2_API_KEY`/`NEBIUS_API_KEY` |
| `lidar_map` | 2-D map from LiDAR + the robot's own pose (`steps.jsonl`); video + final PNG; measures odometry error against `gt_scene_pose`, scan-to-map statistics | `scripts.make_farm_videos.Map` | `slam_map.mp4` (if ffmpeg is present), `slam_map_final.png`, `localisation.json`, `path.json` | `steps.jsonl` has no `xy`+`lidar` |
| `cuvslam` | cuVSLAM on the ZED 2i data (`zed/` in EuRoC layout, or the raw layout from `render_route.py` → an EuRoC view is built with symlinks in `analysis/cuvslam/euroc`) for every mode in `CUVSLAM_MODES` | `scripts/cuvslam_r181.py` (subprocess with `CUVSLAM_PYTHON`) | `cuvslam_<mode>.json/.log`, `summary.json` | no `zed/` / `cuvslam` cannot be imported |
| `lingbot` | lingbot-map on the front camera (`front/*.jpg` → sequence `front_seq/00000.jpg`) via the docker image `lingbot:blackwell` | `~/lingbot/run.sh <seq> <prefix> <gpu>` | `route_poses.json`, `route_small.ply`, `route_timing.json`, `lingbot.log` | no `run.sh` / docker / image |
| `cosmos_front` | Cosmos Reason looks at the front camera when the robot "opens its eyes" (leg changes + every N seconds); measures latency | `scripts.cosmos_front` | `cosmos_front.jsonl`, `summary.json` | no `EYES_BASE_URL` |
| `report` | Merges everything into **separate sections**: harvest / plant_count / disease / navigation_events / localisation (+ the steps that ran), then has Nemotron write Markdown in Thai + English (no key → template) | `cloud.brain.token_factory.chat` | `analysis/report.json`, `report_th.md`, `report_en.md` | never skipped |

Run-level files: `analysis/pipeline.log` (log of the whole round), `analysis/pipeline.done` (status of every step + exit code),
`analysis/pipeline.lock` (prevents the hub and the watcher running concurrently), `analysis/pipeline.out` (stdout when the hub spawns it itself)

## Commands
```bash
python3 -m cloud.pipeline run <run_dir>                 # run every step enabled in PIPELINE_STEPS
python3 -m cloud.pipeline run <run_dir> --force         # rerun everything even if .done exists
python3 -m cloud.pipeline run <run_dir> --steps vlm_stops,report
python3 -m cloud.pipeline watch --runs-dir ~/runs [--poll 30] [--once]
```
`<run_dir>` = the `rec/` folder of `run_farm.py` (contains `frames.jsonl`/`steps.jsonl`). The watcher looks 2 levels deep (`~/runs/<name>/rec/run_complete.json`).
Exit code: 0 every step ok/skipped, 1 a step failed (see `.error`), 3 not a run dir, 4 currently being run by another process.

## Environment variables (cloud/pipeline/config.py)
| Variable | Default | Used by |
|---|---|---|
| `PIPELINE_STEPS` | all steps in order | step selection (comma-separated) |
| `DETECTOR_MODEL`, `DETECTOR_THRESHOLD` | `vision_runs/rtdetr_v2_mixed/best`, 0.5 | detect |
| `EYES2_BASE_URL`, `EYES2_MODEL`, `EYES2_API_KEY` (or `NEBIUS_API_KEY`) | Token Factory, `nvidia/Nemotron-3.5-Super-VL` | vlm_stops |
| `PIPELINE_VLM_WORKERS`, `PIPELINE_VLM_LONG_SIDE`, `PIPELINE_VLM_DETAILED` | 4, 1280, 1 | vlm_stops (concurrent requests, image size, ask for description + action) |
| `EYES_BASE_URL`, `EYES_MODEL`, `EYES_API_KEY`, `PIPELINE_COSMOS_EVERY_S` | –, Cosmos-Reason2-8B, EMPTY, 10 | cosmos_front |
| `CUVSLAM_PYTHON`, `CUVSLAM_MODES` | the running python, `stereo,inertial,rgbd` | cuvslam |
| `LINGBOT_RUN`, `LINGBOT_IMAGE`, `LINGBOT_GPU` | `~/lingbot/run.sh`, `lingbot:blackwell`, 0 | lingbot |
| `NEBIUS_API_KEY`, `BRAIN_BASE_URL`, `BRAIN_MODEL` | –, Token Factory, `nvidia/nemotron-3-super-120b-a12b` | report (written by Nemotron) |
| `PIPELINE_MATCH_M`, `PIPELINE_MAP_EVERY` | 0.7 m, 5 | ground-truth match radius, steps per map-video frame |
| `PIPELINE_AUTO`, `PIPELINE_RUNS_DIR` | 0, – | hub: spawn the pipeline itself on `/run_complete`; run folder when the robot sends only a name |
All keys live in `deploy/nebius/.env` only (`start_pipeline.sh` sources it). Never put them on the command line or in git.

## Installing on the Nebius server
```bash
# code at ~/TheFarm/r1_greenhouse_nav, python with torch/transformers/openai at ~/venv2, cuvslam at ~/cuvslam_venv, lingbot at ~/lingbot
cd ~/TheFarm/r1_greenhouse_nav
sudo cp deploy/nebius/pipeline.service /etc/systemd/system/ && sudo systemctl daemon-reload
sudo systemctl enable --now pipeline            # hub :8090 + watcher on ~/runs
journalctl -u pipeline -f                       # or tail -f ~/runs/logs/pipeline.log ~/runs/logs/hub.log
```
Adjust the paths/user in `pipeline.service` (`User`, `WorkingDirectory`, `RUNS_DIR`, `PY`) if they differ; it can also be run by hand with `deploy/nebius/start_pipeline.sh`.
Test without a robot: put an old rec at `~/runs/<name>/rec`, then `curl -X POST localhost:8090/run_complete -H 'Content-Type: application/json' -d '{"run":"<name>","path":"/home/ubuntu/runs/<name>/rec"}'`

## How the robot calls the system
1. `run_farm.py` finishes (the `finally` block after writing `summary.json`) and the CloudLink is open (`--api-url`, not `--offline`)
2. If `--upload-rec` is set → `sim.cloud_link.rsync_dir(rec, target)`, then the server-side path (the part after `:`) is used
3. `CloudLink.post_run_complete(rec, summary, server_path)` → `POST /run_complete {"run": <name of rec's parent folder>, "path", "summary", "source": "robot"}`
4. The hub checks that the path exists on the server (missing → 400), writes `run_complete.json` → the watcher (or the hub itself if `PIPELINE_AUTO=1`) carries on
Everything is wrapped in try/except: the robot always shuts down normally even if the send fails.

## report.json layout (separate sections, never mixed)
```
harvest            {source: rtdetr, by_aisle: {aisle: {pass, reject, not_ready}}, total, rule}
plant_count        {by_aisle, total, plant_stops, plants_judged_by_vlm}
disease            {rtdetr: {detections, by_label, spots[], ground_truth_check}, vlm: {status, flagged[] (plant+description+action), ground_truth_check}, healthy_plants}
navigation_events  {counts, commands[], legs_done[], incidents[] (obstacle_stop/stuck/aborted/...), final_state, front_camera_checks}
localisation       {lidar_odometry: {odometry_vs_truth, scan_to_map, path_m}, cuvslam: {mode: ate...}, lingbot}
steps              {step: {status, reason, seconds}}      files: location of every step's files
```

## Tests (no GPU/network)
`python3 -m pytest -q tests/test_pipeline.py` — a small fake run (PNG, steps/frames/events.jsonl), RT-DETR and the OpenAI client are stubbed:
checks the `analysis/` layout, markers, that a rerun does no repeated work, `--force`, optional steps skipped correctly, a failed step not stopping the others,
`/run_complete` writing the marker and `PIPELINE_AUTO` spawning, CLI + watcher

---

## Summary
One automatic system replaces the hand-run scripts: the robot (`scripts/run_farm.py`) finishes its survey and POSTs
`/run_complete {"run","path","summary"}` to the hub (`agent/robot_api.py`; in the sim `rec/` is a shared mount, on a
real robot `--upload-rec user@server:/runs/<name>/rec` rsyncs it first). The hub writes `<run>/run_complete.json`
(and spawns `python3 -m cloud.pipeline run <run>` when `PIPELINE_AUTO=1`). The watcher
(`python3 -m cloud.pipeline watch --runs-dir ~/runs`, started by `deploy/nebius/pipeline.service` via
`start_pipeline.sh` together with the hub) processes every such run once, one at a time, in order:
`detect` (RT-DETR per side image: harvest grading / plant count / leaf anomalies), `vlm_stops` (one Nemotron VL request
per plant stop with all six views, N workers, agronomist description + action per plant), `lidar_map` (2-D LiDAR map
video + PNG + odometry vs truth), `cuvslam` and `lingbot` (optional GPU steps, skipped with a reason when the wheel /
docker image is missing), `cosmos_front` (needs `EYES_BASE_URL`) and `report` (`analysis/report.json` with separate
harvest / plant count / disease / navigation events / localisation sections, plus `report_th.md` and `report_en.md`
written by Nemotron on Token Factory when `NEBIUS_API_KEY` is set, else a template). Each step writes
`analysis/<step>/` and an `analysis/<step>.done` marker (idempotent; `--force` reruns; a failure leaves `.error` and
does not stop the other steps). Config is env-only (`cloud/pipeline/config.py`); tests in `tests/test_pipeline.py`
run with stubs and no GPU or network.
