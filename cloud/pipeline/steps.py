"""The analysis steps of the post-survey pipeline. Each step is `fn(run_dir, cfg) -> dict`, reads the robot's recording
(run_dir = a scripts/run_farm.py rec dir: steps.jsonl, events.jsonl, frames.jsonl, side/, front/, optional zed/) and
writes into <run_dir>/analysis/<step>/. The runner (runner.py) writes the <step>.done marker and isolates failures;
a step that cannot run here raises Skip(reason) instead of failing.

  detect        RT-DETR on every side image (vision.detector), per camera + the 3 categories (scripts.detect_side.summarize)
  vlm_stops     one Nemotron VL request per plant stop, six views (scripts.vlm_stop), N workers; description + action per plant
  lidar_map     2-D LiDAR map video + final PNG from steps.jsonl (scripts.make_farm_videos.Map) + odometry vs truth
  cuvslam       cuVSLAM on zed/ (EuRoC layout) via scripts/cuvslam_r181.py   [optional GPU]
  lingbot       lingbot-map docker on the front camera (~/lingbot/run.sh)    [optional GPU]
  cosmos_front  Cosmos Reason on the front camera at "open your eyes" moments (scripts.cosmos_front) [needs EYES_BASE_URL]
  report        merges everything into analysis/report.json + report_th.md / report_en.md (Nemotron text or template)"""
from __future__ import annotations
import collections
import glob as _glob
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cloud.pipeline.config import ALL_STEPS, PipelineConfig

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:                             # scripts/ is a namespace package: `import scripts.vlm_stop`
    sys.path.insert(0, str(ROOT))


class Skip(Exception):
    """Raised by a step that cannot run here (no data, no model, no GPU tool); recorded as status=skipped."""


def out_dir(run_dir: Path, step: str) -> Path:
    d = run_dir / "analysis" / step
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _openai_client(base_url: str, api_key: str, timeout: float):
    """Separate so tests can swap in a fake client."""
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout)


# ---- detect ---------------------------------------------------------------------------------------------------------
def side_images(run_dir: Path) -> list[tuple[Path, int, str]]:
    """(path, frame_id, camera) for every side image: full PNGs in side/ (run_farm) or the JPEGs sent to the hub."""
    out = []
    for d, ext in ((run_dir / "side", "png"), (run_dir / "frames", "jpg")):
        for p in sorted(d.glob(f"*_mast_*.{ext}")):
            fid, cam = p.stem.split("_", 1)
            if fid.isdigit():
                out.append((p, int(fid), cam))
        if out:
            return out
    return out


def step_detect(run_dir: Path, cfg: PipelineConfig) -> dict:
    from PIL import Image
    import vision.detector as vd
    import scripts.detect_side as detect_side
    images = side_images(run_dir)
    if not images:
        raise Skip("no side images (side/*_mast_*.png or frames/*_mast_*.jpg)")
    model = cfg.detector_model if Path(cfg.detector_model).is_absolute() else str(cfg.repo / cfg.detector_model)
    if not Path(model).exists():
        raise Skip(f"DETECTOR_MODEL not found: {model}")
    meta_by_id = {f["frame_id"]: f for f in read_jsonl(run_dir / "frames.jsonl") if "frame_id" in f}
    det = vd.Detector(model, threshold=cfg.detector_threshold)
    rows = []
    for p, fid, cam in images:
        m = meta_by_id.get(fid, {})
        meta = {"camera": cam, "metre": m.get("metre"), "aisle": m.get("aisle"), "frame_id": fid, "purpose": m.get("purpose")}
        im = Image.open(p).convert("RGB")
        rows.append({"image": str(p.relative_to(run_dir)), **meta, **vd.to_finding(det.detect(im), im, meta)})
    o = out_dir(run_dir, "detect")
    write_jsonl(o / "findings.jsonl", rows)
    by_cam: dict[str, dict] = {}
    for r in rows:
        c = by_cam.setdefault(r["camera"], {"images": 0, "pass": 0, "reject": 0, "not_ready": 0, "plants": 0, "anomalies": 0, "status": {}})
        c["images"] += 1; c["plants"] += len(r.get("plants") or []); c["anomalies"] += len(r.get("issues") or [])
        c["status"][r["status"]] = c["status"].get(r["status"], 0) + 1
        for k in ("pass", "reject", "not_ready"):
            c[k] += (r.get("harvest") or {}).get(k, 0)
    (o / "by_camera.json").write_text(json.dumps(by_cam, indent=1))
    summary = detect_side.summarize(rows, run_dir, model, cfg.detector_threshold, cfg.match_m)
    (o / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    la = summary["3_leaf_anomalies"]
    return {"images": len(rows), "harvest": summary["1_harvest_fruit_grading"], "plant_count": summary["2_plant_count"],
            "leaf_anomalies": la["detections"], "found_of_detectable": la["found_of_detectable"]}


# ---- vlm_stops ------------------------------------------------------------------------------------------------------
def step_vlm_stops(run_dir: Path, cfg: PipelineConfig) -> dict:
    import scripts.vlm_stop as vlm_stop
    if not cfg.eyes2_api_key:
        raise Skip("no EYES2_API_KEY / NEBIUS_API_KEY for the per-stop VLM")
    if not (run_dir / "frames.jsonl").exists():
        raise Skip("no frames.jsonl")
    stops = vlm_stop.group_png_stops(run_dir)
    if not stops:
        raise Skip("no complete 6-camera plant stops in frames.jsonl")
    client = _openai_client(cfg.eyes2_base_url, cfg.eyes2_api_key, 240.0)
    o = out_dir(run_dir, "vlm_stops")

    def one(s: dict) -> dict:
        s = dict(s); paths = s.pop("paths")
        s["images"] = vlm_stop.encode_views([paths[c] for c in vlm_stop.ORDER], cfg.vlm_long_side)
        r = vlm_stop.ask_stop(client, cfg.eyes2_model, s, cfg.vlm_detailed)
        return r | {"images": {c: str(paths[c].relative_to(run_dir)) for c in vlm_stop.ORDER}}
    rows = []
    with ThreadPoolExecutor(max(1, cfg.vlm_workers)) as ex, open(o / "vlm_stops.jsonl", "w") as fo:
        for r in ex.map(one, stops):                        # written as they finish: a crash keeps partial results
            rows.append(r); fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush()
    plants = [{"aisle": r["aisle"], "metre": r["metre"], "row": side, "purpose": r.get("purpose"),
               "status": r[side].get("status"), "issues": r[side].get("issues", []), "summary": r[side].get("summary", ""),
               "description": r[side].get("description"), "action": r[side].get("action"),
               "images": [r["images"][c] for c in vlm_stop.ORDER if c.startswith(side)]}
              for r in rows for side in ("left", "right")]
    (o / "plants.json").write_text(json.dumps(plants, indent=1, ensure_ascii=False))
    summary = vlm_stop.summarize(rows, run_dir, cfg.eyes2_model, cfg.match_m) | {"workers": cfg.vlm_workers, "detailed": cfg.vlm_detailed}
    (o / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return {k: summary[k] for k in ("requests", "plants_judged", "status", "flagged_plants", "anomaly_plants_flagged", "total_tokens")}


# ---- lidar_map ------------------------------------------------------------------------------------------------------
def _dist_stats(errs: list[float]) -> dict | None:
    if not errs:
        return None
    s = sorted(errs)
    return {"n": len(s), "rmse_m": round(math.sqrt(sum(e * e for e in s) / len(s)), 3), "median_m": round(s[len(s) // 2], 3),
            "p90_m": round(s[int(len(s) * 0.9)], 3), "max_m": round(s[-1], 3)}


def step_lidar_map(run_dir: Path, cfg: PipelineConfig) -> dict:
    from scripts.make_farm_videos import Map, encode
    steps = [s for s in read_jsonl(run_dir / "steps.jsonl") if s.get("lidar") and s.get("xy")]
    if not steps:
        raise Skip("steps.jsonl has no rows with xy + lidar (run_farm recording needed)")
    o = out_dir(run_dir, "lidar_map"); tm = o / "_map"; tm.mkdir(exist_ok=True)
    xs = [s["xy"][0] for s in steps]; ys = [s["xy"][1] for s in steps]
    m = Map(min(xs) - 6, max(xs) + 6, min(ys) - 6, max(ys) + 6)
    trail, k = [], 0
    for i, s in enumerate(steps):
        m.add(s["xy"][0], s["xy"][1], s["heading_deg"], s["lidar"]); trail.append(tuple(s["xy"]))
        if i % cfg.map_every == 0:
            m.image(trail).save(tm / f"{k:06d}.png"); k += 1
    m.image(trail).save(o / "slam_map_final.png")
    video = None
    if shutil.which("ffmpeg"):
        encode(str(tm / "%06d.png"), o / "slam_map.mp4", 1 / (0.1 * cfg.map_every)); video = "slam_map.mp4"
    errs = [math.hypot(s["xy"][0] - s["gt_scene_pose"][0], s["xy"][1] - s["gt_scene_pose"][1])
            for s in steps if s.get("gt_scene_pose") and len(s["gt_scene_pose"]) >= 2]
    fixes = [s["scan_to_map"] for s in steps if isinstance(s.get("scan_to_map"), dict)]
    loc = {"scans": len(steps), "path_m": round(sum(math.dist(a, b) for a, b in zip(trail, trail[1:])), 2),
           "odometry_vs_truth": _dist_stats(errs),
           "scan_to_map": {"attempts": len(fixes), "ok": sum(1 for f in fixes if f.get("ok")),
                           "applied_xy": sum(1 for f in fixes if f.get("applied_xy"))} if fixes else None,
           "mean_confidence": round(sum(float(s.get("confidence", 0)) for s in steps) / len(steps), 3)}
    (o / "localisation.json").write_text(json.dumps(loc, indent=1))
    (o / "path.json").write_text(json.dumps([[round(x, 3), round(y, 3)] for x, y in trail[::cfg.map_every]]))
    return loc | {"map_frames": k, "video": video, "final_png": "slam_map_final.png"}


# ---- cuvslam --------------------------------------------------------------------------------------------------------
def euroc_dir(run_dir: Path, out: Path) -> Path | None:
    """zed/ already in EuRoC layout -> it; raw render_route layout (zed/cam0, imu0.csv, groundtruth_part*.csv) ->
    EuRoC view built with symlinks under out/euroc (like deploy/nebius/post_render.sh); nothing -> None."""
    z = run_dir / "zed"
    if (z / "cam0" / "data").is_dir() and (z / "calib.json").exists():
        return z
    if not ((z / "cam0").is_dir() and (z / "calib.json").exists()):
        return None
    e = out / "euroc"
    for cam in ("cam0", "cam1", "depth0"):
        (e / cam).mkdir(parents=True, exist_ok=True)
        link = e / cam / "data"
        if not link.exists() and (z / cam).is_dir():
            link.symlink_to((z / cam).resolve(), target_is_directory=True)
    (e / "imu0").mkdir(exist_ok=True)
    if (z / "imu0.csv").exists():
        shutil.copyfile(z / "imu0.csv", e / "imu0" / "data.csv")
    shutil.copyfile(z / "calib.json", e / "calib.json")
    if (z / "groundtruth.csv").exists():
        shutil.copyfile(z / "groundtruth.csv", e / "groundtruth.csv")
    else:
        rows = sorted((l for p in z.glob("groundtruth_part*.csv") for l in p.read_text().splitlines() if l and not l.startswith("#")),
                      key=lambda l: int(l.split(",")[0]))
        (e / "groundtruth.csv").write_text("#timestamp [ns],p_x,p_y,p_z,q_w,q_x,q_y,q_z\n" + "\n".join(rows) + "\n")
    return e


def cuvslam_available(python: str) -> str | None:
    """None when `import cuvslam` works in that interpreter, else the reason."""
    try:
        r = subprocess.run([python, "-c", "import cuvslam"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"{python}: {e}"
    return None if r.returncode == 0 else f"cuvslam not importable in {python}: {(r.stderr or r.stdout).strip().splitlines()[-1:] or ['?']}"


def step_cuvslam(run_dir: Path, cfg: PipelineConfig) -> dict:
    o = out_dir(run_dir, "cuvslam")
    data = euroc_dir(run_dir, o)
    if data is None:
        raise Skip("no zed/ stereo recording in this run")
    why = cuvslam_available(cfg.cuvslam_python)
    if why:
        raise Skip(why)
    env = dict(os.environ)
    try:                                                   # CUDA libs shipped as nvidia-* wheels next to cuvslam
        sp = subprocess.run([cfg.cuvslam_python, "-c", "import site;print(site.getsitepackages()[0])"],
                            capture_output=True, text=True, timeout=60).stdout.strip()
        libs = _glob.glob(f"{sp}/nvidia/*/lib")
        if libs:
            env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    except (OSError, subprocess.TimeoutExpired):
        pass
    res = {}
    for mode in cfg.cuvslam_modes:
        out = o / f"cuvslam_{mode}.json"
        r = subprocess.run([cfg.cuvslam_python, str(cfg.repo / "scripts/cuvslam_r181.py"), "--data", str(data), "--mode", mode,
                            "--out", str(out)], capture_output=True, text=True, env=env, cwd=str(cfg.repo))
        (o / f"cuvslam_{mode}.log").write_text((r.stdout or "") + (r.stderr or ""))
        if r.returncode == 0 and out.exists():
            d = json.loads(out.read_text())
            res[mode] = {k: d[k] for k in ("method", "frames", "tracked", "fps", "path_m", "ate_rmse_m", "ate_median_m", "ate_max_m", "sim3_scale") if k in d}
        else:
            res[mode] = {"error": f"exit {r.returncode}: {(r.stderr or r.stdout or '').strip()[-300:]}"}
    (o / "summary.json").write_text(json.dumps(res, indent=1))
    if all("error" in v for v in res.values()):
        raise RuntimeError(f"cuVSLAM failed in every mode: {res}")
    return res


# ---- lingbot --------------------------------------------------------------------------------------------------------
def step_lingbot(run_dir: Path, cfg: PipelineConfig) -> dict:
    front = sorted((run_dir / "front").glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if not front:
        raise Skip("no front/*.jpg frames")
    if not cfg.lingbot_run.exists():
        raise Skip(f"lingbot runner not found: {cfg.lingbot_run}")
    if not shutil.which("docker"):
        raise Skip("docker not installed")
    r = subprocess.run(["docker", "image", "inspect", cfg.lingbot_image], capture_output=True, text=True)
    if r.returncode != 0:
        raise Skip(f"docker image {cfg.lingbot_image} not available")
    o = out_dir(run_dir, "lingbot"); seq = o / "front_seq"; seq.mkdir(exist_ok=True)
    with open(o / "front_seq_index.txt", "w") as f:
        for i, p in enumerate(front):
            dst = seq / f"{i:05d}.jpg"
            if not dst.exists():
                dst.symlink_to(p.resolve())
            f.write(f"{i} {p.stem}\n")
    prefix = o / "route"
    t0 = time.time()
    r = subprocess.run(["bash", str(cfg.lingbot_run), str(seq), str(prefix), str(cfg.lingbot_gpu)], capture_output=True, text=True)
    (o / "lingbot.log").write_text((r.stdout or "") + (r.stderr or ""))
    if r.returncode != 0:
        raise RuntimeError(f"lingbot run.sh exit {r.returncode}: {(r.stderr or r.stdout or '').strip()[-300:]}")
    poses = json.loads((prefix.parent / "route_poses.json").read_text()) if (prefix.parent / "route_poses.json").exists() else None
    timing = json.loads((prefix.parent / "route_timing.json").read_text()) if (prefix.parent / "route_timing.json").exists() else None
    return {"frames": len(front), "poses": len(poses) if isinstance(poses, (list, dict)) else None, "timing": timing,
            "ply": "route_small.ply" if (prefix.parent / "route_small.ply").exists() else None, "wall_s": round(time.time() - t0, 1)}


# ---- cosmos_front ---------------------------------------------------------------------------------------------------
def step_cosmos_front(run_dir: Path, cfg: PipelineConfig) -> dict:
    import scripts.cosmos_front as cf
    if not cfg.eyes_base_url:
        raise Skip("EYES_BASE_URL not set (Cosmos Reason endpoint)")
    steps = read_jsonl(run_dir / "steps.jsonl")
    if not steps or not (run_dir / "front").is_dir():
        raise Skip("needs steps.jsonl and front/ frames")
    frames = cf.front_frames(run_dir / "front", steps)
    picks = cf.pick_checks(steps, frames, cfg.cosmos_every_s)
    if not picks:
        raise Skip("no front frames during a leg")
    client = _openai_client(cfg.eyes_base_url, cfg.eyes_api_key, 120.0)
    o = out_dir(run_dir, "cosmos_front")
    rows = []
    with open(o / "cosmos_front.jsonl", "w") as fo:
        for pick in picks:
            try:
                row = cf.ask_front(client, cfg.eyes_model, pick)
            except Exception as e:                           # one bad call must not lose the rest
                row = {"t": round(pick[1], 2), "frame": pick[0].name, "leg": pick[2], "trigger": pick[5], "latency_s": 0.0, "error": str(e)[:300]}
            rows.append(row); fo.write(json.dumps(row, ensure_ascii=False) + "\n"); fo.flush()
    summ = cf.summarize(rows, cfg.eyes_model, len(frames)) | {"errors": sum(1 for r in rows if "error" in r)}
    (o / "summary.json").write_text(json.dumps(summ, indent=1))
    return summ


# ---- report ---------------------------------------------------------------------------------------------------------
def _load(p: Path):
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except ValueError:
        return None


def step_markers(run_dir: Path) -> dict[str, dict]:
    a = run_dir / "analysis"
    out = {}
    for s in ALL_STEPS:
        for suffix in (".done", ".error"):
            d = _load(a / f"{s}{suffix}")
            if d:
                out[s] = d
    return out


def build_report(run_dir: Path, cfg: PipelineConfig) -> dict:
    a = run_dir / "analysis"
    det = _load(a / "detect/summary.json") or {}
    vlm = _load(a / "vlm_stops/summary.json") or {}
    plants = _load(a / "vlm_stops/plants.json") or []
    loc = _load(a / "lidar_map/localisation.json")
    cuv = _load(a / "cuvslam/summary.json")
    cosmos = _load(a / "cosmos_front/summary.json")
    markers = step_markers(run_dir)
    events = read_jsonl(run_dir / "events.jsonl")
    counts = collections.Counter(e.get("type") for e in events)
    incidents = [{k: e.get(k) for k in ("t", "type", "reason", "why", "note", "leg", "aisle", "metre", "frame_id") if e.get(k) is not None}
                 for e in events if e.get("type") in ("obstacle_stop", "stuck", "aborted", "capture_missing", "blocked", "replan")]
    commands = [{"t": e.get("t"), "cmd": (e.get("cmd") or {}).get("cmd"), "accepted": e.get("accepted"), "legs": e.get("legs"), "reason": e.get("reason")}
                for e in events if e.get("type") == "command"]
    legs = [{k: e.get(k) for k in ("t", "leg", "aisle", "metre", "error_m", "note") if e.get(k) is not None} for e in events if e.get("type") == "leg_done"]
    status_rows = [e for e in events if e.get("type") == "status"]
    harvest_by = det.get("1_harvest_fruit_grading") or {}
    total = collections.Counter()
    for v in harvest_by.values():
        total.update(v)
    la = det.get("3_leaf_anomalies") or {}
    flagged = [p for p in plants if p.get("status") in ("suspect", "disease")]
    return {
        "run": run_dir.name if run_dir.name != "rec" else run_dir.parent.name, "run_dir": str(run_dir), "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "robot_summary": _load(run_dir / "summary.json"), "run_complete": _load(run_dir / "run_complete.json"),
        "harvest": {"source": "rtdetr", "model": det.get("model"), "images": det.get("images"), "by_aisle": harvest_by,
                    "total": {k: total.get(k, 0) for k in ("pass", "reject", "not_ready")},
                    "rule": "ripe red = pass, lesion = reject, green/turning = not ready; each frame counts only its own 0.5 m strip"},
        "plant_count": {"source": "rtdetr plant_base", "by_aisle": det.get("2_plant_count") or {},
                        "total": sum((det.get("2_plant_count") or {}).values()), "plant_stops": vlm.get("requests"),
                        "plants_judged_by_vlm": vlm.get("plants_judged")},
        "disease": {"rtdetr": {"detections": la.get("detections"), "by_label": la.get("by_label"), "spots": (la.get("spots") or [])[:200],
                               "ground_truth_check": la.get("ground_truth_check"), "found_of_detectable": la.get("found_of_detectable"),
                               "detections_not_near_any_anomaly_plant": la.get("detections_not_near_any_anomaly_plant")},
                    "vlm": {"model": vlm.get("model"), "requests": vlm.get("requests"), "status": vlm.get("status"),
                            "flagged": flagged, "anomaly_plants_flagged": vlm.get("anomaly_plants_flagged"),
                            "flags_not_near_any_anomaly_plant": vlm.get("flags_not_near_any_anomaly_plant"),
                            "ground_truth_check": vlm.get("ground_truth_check")},
                    "healthy_plants": sum(1 for p in plants if p.get("status") == "healthy")},
        "navigation_events": {"counts": dict(counts), "commands": commands, "legs_done": legs, "incidents": incidents,
                              "final_state": (status_rows[-1].get("state") if status_rows else None),
                              "front_camera_checks": cosmos},
        "localisation": {"lidar_odometry": loc, "cuvslam": cuv, "lingbot": (markers.get("lingbot") or {}).get("result")},
        "steps": {s: {k: m.get(k) for k in ("status", "reason", "seconds")} for s, m in markers.items()},
        "files": {"detect": "analysis/detect/{findings.jsonl,by_camera.json,summary.json}",
                  "vlm_stops": "analysis/vlm_stops/{vlm_stops.jsonl,plants.json,summary.json}",
                  "lidar_map": "analysis/lidar_map/{slam_map.mp4,slam_map_final.png,localisation.json,path.json}",
                  "cuvslam": "analysis/cuvslam/cuvslam_<mode>.json", "lingbot": "analysis/lingbot/route_{poses.json,small.ply,timing.json}",
                  "cosmos_front": "analysis/cosmos_front/{cosmos_front.jsonl,summary.json}",
                  "report": "analysis/{report.json,report_th.md,report_en.md}"}}


# Thai farmer-report text (feature: report_th.md). Section titles / labels of the template report in Thai and English.
_T = {"th": {"title": "รายงานการสำรวจโรงเรือน", "harvest": "1. ผลผลิต (คัดเกรดผล)", "count": "2. จำนวนต้น", "disease": "3. โรค/ความผิดปกติ",
             "nav": "4. เหตุการณ์ระหว่างเดิน", "loc": "5. การระบุตำแหน่ง", "steps": "ขั้นตอนที่รัน", "pass": "ผ่าน (สุกแดง)", "reject": "คัดทิ้ง (มีแผล)",
             "not_ready": "ยังไม่พร้อม", "aisle": "ทางเดิน", "plants": "ต้น", "none": "ไม่มี", "flagged": "ต้นที่ควรดู", "action": "สิ่งที่ควรทำ",
             "incidents": "เหตุการณ์ผิดปกติ", "commands": "คำสั่งที่ได้รับ", "odo": "ระยะคลาดเคลื่อนตำแหน่งหุ่น (odometry เทียบค่าจริง)",
             "template": "(รายงานจากแม่แบบ: ไม่มี NEBIUS_API_KEY จึงไม่ได้ให้ Nemotron เขียน)", "not_run": "ไม่ได้รัน"},
      "en": {"title": "Greenhouse survey report", "harvest": "1. Harvest (fruit grading)", "count": "2. Plant count", "disease": "3. Disease / anomalies",
             "nav": "4. Navigation events", "loc": "5. Localisation", "steps": "Steps run", "pass": "pass (ripe red)", "reject": "reject (lesion)",
             "not_ready": "not ready", "aisle": "aisle", "plants": "plants", "none": "none", "flagged": "plants to look at", "action": "action",
             "incidents": "incidents", "commands": "commands received", "odo": "robot position error (odometry vs truth)",
             "template": "(template report: no NEBIUS_API_KEY, so Nemotron did not write it)", "not_run": "not run"}}


def template_report(rep: dict, lang: str) -> str:
    t = _T[lang]; L = [f"# {t['title']}: {rep['run']}", "", f"{rep['generated']}  {t['template']}", ""]
    h = rep["harvest"]; det_ok = h["images"] is not None; vlm_ok = rep["disease"]["vlm"]["requests"] is not None
    L += [f"## {t['harvest']}", f"- {t['pass']}: {h['total']['pass']}, {t['reject']}: {h['total']['reject']}, {t['not_ready']}: {h['total']['not_ready']}"
          if det_ok else f"- {t['not_run']} (detect)"]
    L += [f"  - {t['aisle']} {a}: {v}" for a, v in h["by_aisle"].items()] + [""]
    c = rep["plant_count"]; L += [f"## {t['count']}", (f"- {c['total']} {t['plants']} (RT-DETR)" if det_ok else f"- {t['not_run']} (detect)")
                                  + (f"; {c['plant_stops']} stops x 2 = {c['plants_judged_by_vlm']} {t['plants']} (VLM)" if vlm_ok else "")]
    L += [f"  - {t['aisle']} {a}: {v}" for a, v in c["by_aisle"].items()] + [""]
    d = rep["disease"]; L += [f"## {t['disease']}", (f"- RT-DETR: {d['rtdetr']['detections']} {d['rtdetr']['by_label'] or ''}"
                              + (f", ground truth {d['rtdetr']['found_of_detectable']}" if d['rtdetr']['found_of_detectable'] else "")) if det_ok else f"- RT-DETR: {t['not_run']}",
                              (f"- VLM ({d['vlm']['model']}): {d['vlm']['status'] or t['none']}"
                              + (f", ground truth {d['vlm']['anomaly_plants_flagged']}" if d['vlm']['anomaly_plants_flagged'] else "")) if vlm_ok else f"- VLM: {t['not_run']}",
                              f"- {t['flagged']}:"]
    L += [f"  - {p['aisle']} @ {p['metre']} m ({p['row']}): **{p['status']}** {p.get('summary', '')}"
          + (f"\n    - {p['description']}" if p.get("description") else "") + (f"\n    - {t['action']}: {p['action']}" if p.get("action") else "")
          for p in d["vlm"]["flagged"]] or [f"  - {t['none']}"]
    L += [""]
    n = rep["navigation_events"]; L += [f"## {t['nav']}", f"- {dict(n['counts'])}", f"- {t['commands']}: " + ", ".join(f"{x['cmd']} ({'ok' if x['accepted'] else x['reason']})" for x in n["commands"]) if n["commands"] else f"- {t['commands']}: {t['none']}",
                                        f"- {t['incidents']}: " + ("; ".join(f"{x.get('type')} @ {x.get('aisle')} {x.get('metre')} m {x.get('reason') or x.get('why') or ''}" for x in n["incidents"]) if n["incidents"] else t["none"])]
    if n.get("front_camera_checks"):
        L.append(f"- Cosmos front checks: {n['front_camera_checks']}")
    L += [""]
    lo = rep["localisation"]; L += [f"## {t['loc']}"]
    if lo.get("lidar_odometry"):
        L.append(f"- {t['odo']}: {lo['lidar_odometry'].get('odometry_vs_truth')}; scan_to_map {lo['lidar_odometry'].get('scan_to_map')}; path {lo['lidar_odometry'].get('path_m')} m")
    L.append(f"- cuVSLAM: {lo.get('cuvslam') or t['none']}")
    L.append(f"- lingbot-map: {lo.get('lingbot') or t['none']}")
    L += ["", f"## {t['steps']}"] + [f"- {s}: {m['status']}" + (f" ({m['reason']})" if m.get("reason") else "") for s, m in rep["steps"].items()]
    return "\n".join(L) + "\n"


def _brain_chat(cfg: PipelineConfig, messages: list[dict]) -> str:
    """Nemotron text via cloud.brain.token_factory (NEBIUS_API_KEY / BRAIN_* env). Separate so tests can stub it."""
    from cloud.brain.token_factory import chat
    return chat(messages, model=cfg.brain_model, temperature=0.2, max_tokens=6000).content


def nemotron_report(rep: dict, lang: str, cfg: PipelineConfig) -> str:
    slim = json.loads(json.dumps(rep))
    slim["disease"]["rtdetr"]["spots"] = slim["disease"]["rtdetr"]["spots"][:40]
    for p in slim["disease"]["vlm"]["flagged"]:
        p.pop("images", None)
    # Thai farmer-report text (feature: report_th.md): the Thai report is written by Nemotron in plain farmer Thai.
    language = "Thai, plain words a farmer understands; keep model/file names in English" if lang == "th" else "English"
    sys_msg = ("You write the post-survey report of R1, a greenhouse inspection robot. Write Markdown in " + language + ". "
               "Use EXACTLY these five sections, in this order, kept separate: 1 Harvest (fruit grading), 2 Plant count, "
               "3 Disease / anomalies (RT-DETR screening and the VLM's per-plant verdicts with their descriptions and actions), "
               "4 Navigation events (commands, incidents, legs), 5 Localisation (LiDAR odometry vs truth, cuVSLAM, lingbot-map). "
               "Then a short 'Steps run' list. Only use numbers that are in the JSON; say 'not run' for missing parts; never invent findings.")
    return _brain_chat(cfg, [{"role": "system", "content": sys_msg},
                             {"role": "user", "content": f"Report data (JSON):\n{json.dumps(slim, ensure_ascii=False)}"}])


def step_report(run_dir: Path, cfg: PipelineConfig) -> dict:
    rep = build_report(run_dir, cfg)
    a = run_dir / "analysis"
    (a / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    writer = {}
    for lang in ("th", "en"):
        text = None
        if cfg.nebius_api_key:
            try:
                text = nemotron_report(rep, lang, cfg); writer[lang] = cfg.brain_model
            except Exception as e:                         # endpoint down: still deliver the template
                writer[lang] = f"template (nemotron failed: {str(e)[:120]})"
        if not text:
            text = template_report(rep, lang); writer.setdefault(lang, "template")
        (a / f"report_{lang}.md").write_text(text)
    return {"report": "analysis/report.json", "markdown": ["analysis/report_th.md", "analysis/report_en.md"], "writer": writer,
            "harvest_total": rep["harvest"]["total"], "plants": rep["plant_count"]["total"], "flagged_plants": len(rep["disease"]["vlm"]["flagged"]),
            "incidents": len(rep["navigation_events"]["incidents"])}


STEPS = {"detect": step_detect, "vlm_stops": step_vlm_stops, "lidar_map": step_lidar_map, "cuvslam": step_cuvslam,
         "lingbot": step_lingbot, "cosmos_front": step_cosmos_front, "report": step_report}
assert tuple(STEPS) == ALL_STEPS
