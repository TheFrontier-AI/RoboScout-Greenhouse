# ระบบวิเคราะห์หลังสำรวจ (cloud/pipeline) — หุ่นจบงาน → เซิร์ฟเวอร์วิเคราะห์เอง → รายงานที่เดียว

เมื่อก่อนหลังหุ่นเดินสำรวจเสร็จ เราต้องรันสคริปต์ทีละตัวเอง (`detect_side.py`, `vlm_stop.py`, `make_farm_videos.py`,
`cuvslam_r181.py`, lingbot, `cosmos_front.py` — ดู `deploy/nebius/post_render.sh`) ตอนนี้ทั้งหมดเป็น **ระบบเดียวที่ทำงานอัตโนมัติ**:

```
 หุ่น (Isaac sim / Jetson)                         เซิร์ฟเวอร์ Nebius (RTX PRO 6000)
 ─────────────────────────                         ──────────────────────────────────────────────────────────
 scripts/run_farm.py                               agent/robot_api.py (hub :8090)
   เดินสำรวจ, อัด rec/ (steps/events/frames.jsonl,   ◀── POST /status, /frame (ระหว่างเดิน: ตรวจทีละภาพเหมือนเดิม)
   side/*.png, front/*.jpg, zed/ ถ้ามี)
   จบงาน (finally):                                 ◀── POST /run_complete {"run","path","summary"}
     [--upload-rec user@vm:/runs/<ชื่อ>/rec  rsync]        │ เขียน <run>/run_complete.json  (+ runs.jsonl ใน data dir)
     POST /run_complete                                    │ PIPELINE_AUTO=1 → spawn `python3 -m cloud.pipeline run <run>`
                                                           ▼
                                                   python3 -m cloud.pipeline watch --runs-dir ~/runs   (pipeline.service)
                                                     เห็น run_complete.json ที่ยังไม่มี analysis/pipeline.done → รันทีละ run:
                                                     detect → vlm_stops → lidar_map → cuvslam → lingbot → cosmos_front → report
                                                           ▼
                                                   <run>/analysis/report.json + report_th.md + report_en.md
```

- ใน **sim** โฟลเดอร์ `rec/` ของหุ่นกับ hub อยู่บนดิสก์เดียวกัน (shared mount) จึงส่งแค่ path ไม่ต้องอัปโหลด
- บน **หุ่นจริง** ใส่ `--upload-rec user@server:/home/ubuntu/runs/<ชื่อ>/rec` (หรือ env `R1_UPLOAD_REC`) หุ่นจะ rsync `rec/` ขึ้นไปก่อน
  แล้วค่อย POST path ฝั่งเซิร์ฟเวอร์; ถ้า rsync ล้ม จะไม่ POST (รันใหม่ด้วย `--upload-rec` ทีหลังได้)
- hub ล่มตอนหุ่นจบงาน → เขียน `run_complete.json` ลงใน rec เองก็ได้ (watcher ดูแค่ไฟล์นี้) หรือ `curl -X POST :8090/run_complete -d '{"path": "..."}'`
- ขั้นที่ต้อง GPU/บริการภายนอก (cuvslam, lingbot, cosmos_front) **ข้ามพร้อมเหตุผล** เมื่อไม่มีเครื่องมือ ไม่ทำให้ขั้นอื่นล้ม

## ขั้นตอน (cloud/pipeline/steps.py) และไฟล์ที่ได้
ทุกขั้นเขียนลง `<run>/analysis/<step>/` และทิ้ง marker `analysis/<step>.done` (JSON: status `ok|skipped`, reason, result, เวลา)
รันซ้ำจะข้ามขั้นที่มี `.done` แล้ว (`--force` = รันใหม่) ขั้นที่ล้มได้ `.error` แทนและจะถูกลองใหม่รอบหน้า

| ขั้น | ทำอะไร | ใช้โค้ดเดิม | ไฟล์ออก | ข้ามเมื่อ |
|---|---|---|---|---|
| `detect` | RT-DETR ทุกภาพกล้องข้าง (`side/*.png` หรือ `frames/*_mast_*.jpg`) สรุปต่อกล้อง + 3 หมวด: ①คัดเกรดผล ②นับต้น ③ความผิดปกติใบ (เทียบ ground truth ถ้ามี) | `vision.detector`, `scripts.detect_side.summarize` | `findings.jsonl`, `by_camera.json`, `summary.json` | ไม่มีภาพ / ไม่มี `DETECTOR_MODEL` |
| `vlm_stops` | **หนึ่งคำขอต่อหนึ่งจุดหยุด**: จับกลุ่ม 6 ภาพจาก `frames.jsonl` (ซ้ายสูง/กลาง/ต่ำ = ต้นซ้าย, ขวา = ต้นขวา) ส่ง Nemotron VL ผ่าน Token Factory พร้อมกัน N worker ขอคำอธิบายแบบนักเกษตร + `action` ต่อต้น | `scripts.vlm_stop` (`group_png_stops`, `ask_stop`, `summarize`) | `vlm_stops.jsonl` (ต่อจุด), `plants.json` (ต่อต้น: status, issues, description, action, ภาพ), `summary.json` | ไม่มี `EYES2_API_KEY`/`NEBIUS_API_KEY` |
| `lidar_map` | แผนที่ 2 มิติจาก LiDAR + pose ของหุ่นเอง (`steps.jsonl`) วิดีโอ + PNG สุดท้าย; วัดค่าคลาดเคลื่อน odometry เทียบ `gt_scene_pose`, สถิติ scan-to-map | `scripts.make_farm_videos.Map` | `slam_map.mp4` (ถ้ามี ffmpeg), `slam_map_final.png`, `localisation.json`, `path.json` | `steps.jsonl` ไม่มี `xy`+`lidar` |
| `cuvslam` | cuVSLAM บนข้อมูล ZED 2i (`zed/` แบบ EuRoC หรือแบบดิบจาก `render_route.py` → สร้าง view EuRoC ด้วย symlink ใน `analysis/cuvslam/euroc`) ทุกโหมดใน `CUVSLAM_MODES` | `scripts/cuvslam_r181.py` (subprocess ด้วย `CUVSLAM_PYTHON`) | `cuvslam_<mode>.json/.log`, `summary.json` | ไม่มี `zed/` / import `cuvslam` ไม่ได้ |
| `lingbot` | lingbot-map บนกล้องหน้า (`front/*.jpg` → ลำดับ `front_seq/00000.jpg`) ผ่าน docker image `lingbot:blackwell` | `~/lingbot/run.sh <seq> <prefix> <gpu>` | `route_poses.json`, `route_small.ply`, `route_timing.json`, `lingbot.log` | ไม่มี `run.sh` / docker / image |
| `cosmos_front` | Cosmos Reason ดูกล้องหน้าตอน "ลืมตา" (เปลี่ยนขา + ทุก N วินาที) วัด latency | `scripts.cosmos_front` | `cosmos_front.jsonl`, `summary.json` | ไม่มี `EYES_BASE_URL` |
| `report` | รวมทุกอย่าง **แยกหมวด**: harvest / plant_count / disease / navigation_events / localisation (+ steps ที่รัน) แล้วให้ Nemotron เขียน Markdown ไทย+อังกฤษ (ไม่มี key → แม่แบบ) | `cloud.brain.token_factory.chat` | `analysis/report.json`, `report_th.md`, `report_en.md` | ไม่ข้าม |

ไฟล์ระดับ run: `analysis/pipeline.log` (log ทั้งรอบ), `analysis/pipeline.done` (สถานะทุกขั้น + exit code),
`analysis/pipeline.lock` (กันรันซ้อนระหว่าง hub กับ watcher), `analysis/pipeline.out` (stdout เมื่อ hub spawn เอง)

## คำสั่ง
```bash
python3 -m cloud.pipeline run <run_dir>                 # รันทุกขั้นที่เปิดใน PIPELINE_STEPS
python3 -m cloud.pipeline run <run_dir> --force         # รันใหม่หมดแม้มี .done
python3 -m cloud.pipeline run <run_dir> --steps vlm_stops,report
python3 -m cloud.pipeline watch --runs-dir ~/runs [--poll 30] [--once]
```
`<run_dir>` = โฟลเดอร์ `rec/` ของ `run_farm.py` (มี `frames.jsonl`/`steps.jsonl`) watcher มองลึก 2 ชั้น (`~/runs/<ชื่อ>/rec/run_complete.json`)
Exit code: 0 ทุกขั้น ok/skipped, 1 มีขั้นล้ม (ดู `.error`), 3 ไม่ใช่ run dir, 4 กำลังถูกรันโดย process อื่น

## ตัวแปรแวดล้อม (cloud/pipeline/config.py)
| ตัวแปร | ค่าเริ่มต้น | ใช้กับ |
|---|---|---|
| `PIPELINE_STEPS` | ทุกขั้นตามลำดับ | เลือกขั้น (คั่นด้วย `,`) |
| `DETECTOR_MODEL`, `DETECTOR_THRESHOLD` | `vision_runs/rtdetr_v2_mixed/best`, 0.5 | detect |
| `EYES2_BASE_URL`, `EYES2_MODEL`, `EYES2_API_KEY` (หรือ `NEBIUS_API_KEY`) | Token Factory, `nvidia/Nemotron-3.5-Super-VL` | vlm_stops |
| `PIPELINE_VLM_WORKERS`, `PIPELINE_VLM_LONG_SIDE`, `PIPELINE_VLM_DETAILED` | 4, 1280, 1 | vlm_stops (คำขอพร้อมกัน, ขนาดภาพ, ขอคำอธิบาย+action) |
| `EYES_BASE_URL`, `EYES_MODEL`, `EYES_API_KEY`, `PIPELINE_COSMOS_EVERY_S` | –, Cosmos-Reason2-8B, EMPTY, 10 | cosmos_front |
| `CUVSLAM_PYTHON`, `CUVSLAM_MODES` | python ตัวที่รัน, `stereo,inertial,rgbd` | cuvslam |
| `LINGBOT_RUN`, `LINGBOT_IMAGE`, `LINGBOT_GPU` | `~/lingbot/run.sh`, `lingbot:blackwell`, 0 | lingbot |
| `NEBIUS_API_KEY`, `BRAIN_BASE_URL`, `BRAIN_MODEL` | –, Token Factory, `nvidia/nemotron-3-super-120b-a12b` | report (Nemotron เขียน) |
| `PIPELINE_MATCH_M`, `PIPELINE_MAP_EVERY` | 0.7 ม., 5 | รัศมีเทียบ ground truth, step ต่อเฟรมวิดีโอแผนที่ |
| `PIPELINE_AUTO`, `PIPELINE_RUNS_DIR` | 0, – | hub: spawn pipeline เองเมื่อได้ `/run_complete`; โฟลเดอร์ run เมื่อหุ่นส่งแค่ชื่อ |
key ทั้งหมดอยู่ใน `deploy/nebius/.env` เท่านั้น (`start_pipeline.sh` source ให้) ห้ามใส่ในคำสั่งหรือ git

## ติดตั้งบนเซิร์ฟเวอร์ Nebius
```bash
# โค้ดอยู่ที่ ~/TheFarm/r1_greenhouse_nav, python ที่มี torch/transformers/openai ที่ ~/venv2, cuvslam ที่ ~/cuvslam_venv, lingbot ที่ ~/lingbot
cd ~/TheFarm/r1_greenhouse_nav
sudo cp deploy/nebius/pipeline.service /etc/systemd/system/ && sudo systemctl daemon-reload
sudo systemctl enable --now pipeline            # hub :8090 + watcher บน ~/runs
journalctl -u pipeline -f                       # หรือ tail -f ~/runs/logs/pipeline.log ~/runs/logs/hub.log
```
ปรับ path/ผู้ใช้ใน `pipeline.service` (`User`, `WorkingDirectory`, `RUNS_DIR`, `PY`) ถ้าไม่ตรง; รันมือได้ด้วย `deploy/nebius/start_pipeline.sh`
ทดสอบโดยไม่ต้องมีหุ่น: วาง rec เก่าไว้ที่ `~/runs/<ชื่อ>/rec` แล้ว `curl -X POST localhost:8090/run_complete -H 'Content-Type: application/json' -d '{"run":"<ชื่อ>","path":"/home/ubuntu/runs/<ชื่อ>/rec"}'`

## หุ่นเรียกระบบอย่างไร
1. `run_farm.py` จบ (บล็อก `finally` หลังเขียน `summary.json`) และ CloudLink เปิดอยู่ (`--api-url`, ไม่ใช่ `--offline`)
2. ถ้ามี `--upload-rec` → `sim.cloud_link.rsync_dir(rec, target)` แล้วใช้ path ฝั่งเซิร์ฟเวอร์ (ส่วนหลัง `:`)
3. `CloudLink.post_run_complete(rec, summary, server_path)` → `POST /run_complete {"run": <ชื่อโฟลเดอร์แม่ของ rec>, "path", "summary", "source": "robot"}`
4. hub ตรวจว่า path มีจริงบนเซิร์ฟเวอร์ (ไม่มี → 400) เขียน `run_complete.json` → watcher (หรือ hub เองถ้า `PIPELINE_AUTO=1`) รันต่อ
ทุกอย่างมี try/except: หุ่นปิดตัวตามปกติเสมอแม้ส่งไม่ได้

## รูปแบบ report.json (หมวดแยกกัน ไม่ปนกัน)
```
harvest            {source: rtdetr, by_aisle: {aisle: {pass, reject, not_ready}}, total, rule}
plant_count        {by_aisle, total, plant_stops, plants_judged_by_vlm}
disease            {rtdetr: {detections, by_label, spots[], ground_truth_check}, vlm: {status, flagged[] (ต้น+description+action), ground_truth_check}, healthy_plants}
navigation_events  {counts, commands[], legs_done[], incidents[] (obstacle_stop/stuck/aborted/...), final_state, front_camera_checks}
localisation       {lidar_odometry: {odometry_vs_truth, scan_to_map, path_m}, cuvslam: {mode: ate...}, lingbot}
steps              {step: {status, reason, seconds}}      files: ที่อยู่ไฟล์ของทุกขั้น
```

## ทดสอบ (ไม่ใช้ GPU/เน็ต)
`python3 -m pytest -q tests/test_pipeline.py` — run ปลอมเล็ก ๆ (PNG, steps/frames/events.jsonl), RT-DETR และ OpenAI client เป็น stub:
ตรวจโครงสร้าง `analysis/`, marker, การรันซ้ำไม่ทำงานซ้ำ, `--force`, ขั้น optional ข้ามอย่างถูกต้อง, ขั้นล้มไม่หยุดขั้นอื่น,
`/run_complete` เขียน marker และ `PIPELINE_AUTO` spawn, CLI + watcher

---

## English summary
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
