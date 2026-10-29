"""cloud/pipeline on a tiny fake run_farm recording, no GPU / network: RT-DETR and the OpenAI clients are stubbed.
Checks the analysis/ layout, .done markers, idempotency / --force, graceful skips of optional steps, error isolation,
the hub's POST /run_complete (+ PIPELINE_AUTO) and the CLI / watcher."""
import json, os, subprocess, sys, threading, time, urllib.error, urllib.request
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloud.pipeline import runner, steps                     # noqa: E402
from cloud.pipeline.config import ALL_STEPS, PipelineConfig  # noqa: E402

CAMS = ("mast_left_low", "mast_left_mid", "mast_left_high", "mast_right_low", "mast_right_mid", "mast_right_high")
AISLE = "tunnel-09/aisle-02"
HEAD_Y = 3.26


def make_run(root: Path, stops=(1.0, 1.5), with_gt=True) -> Path:
    """rec/ like scripts/run_farm.py: side PNGs + frames/steps/events jsonl + front jpgs + ground truth."""
    rec = root / "farm_test" / "rec"
    for d in ("side", "frames", "front", "chase"):
        (rec / d).mkdir(parents=True)
    fid, frames, steps_rows, events = 0, [], [], []
    events.append({"type": "command", "t": 0.0, "cmd": {"cmd": "survey_farm"}, "accepted": True, "legs": ["drive_in", "dock"]})
    events.append({"type": "leg_start", "t": 0.1, "leg": {"kind": "drive_in"}})
    for k, m in enumerate(stops):
        t = 1.0 + k
        for cam in CAMS:
            fid += 1
            Image.new("RGB", (90, 160), (40 + 50 * k, 120, 60)).save(rec / "side" / f"{fid:06d}_{cam}.png")
            frames.append({"ts": t, "house": "tunnel-09", "aisle": AISLE, "metre": m, "heading_deg": -90.0, "camera": cam,
                           "frame_id": fid, "pose_confidence": 0.95, "purpose": "inspect", "image_path": f"frames/{fid:06d}_{cam}.jpg"})
            events.append({"type": "capture", "t": t, "camera": cam, "frame_id": fid})
    for i in range(12):
        t = 0.1 * (i + 1); y = HEAD_Y - 0.15 * i
        steps_rows.append({"step": i + 1, "t": round(t, 2), "wheel": [1.0, 1.0], "leg": "drive_in" if i < 10 else "dock", "aisle": AISLE,
                           "metre": round(HEAD_Y - y, 3), "heading_deg": -90.0, "xy": [0.01 * i, y], "confidence": 0.9,
                           "rendered": "s", "lidar": [3.0 + 0.05 * (j % 7) for j in range(36)],
                           "gt_scene_pose": [0.0, y + 0.02, 0.0, -90.0, 0.0], "scan_to_map": {"ok": i % 2 == 0, "applied_xy": False}})
    for s in (1, 6, 11):
        Image.new("RGB", (64, 36), (90, 90, 200)).save(rec / "front" / f"{s:06d}.jpg", quality=80)
    events.append({"type": "obstacle_stop", "t": 0.5, "reason": "lidar 0.4 m", "aisle": AISLE, "metre": 0.4})
    events.append({"type": "leg_done", "t": 1.2, "leg": "drive_in", "aisle": AISLE, "metre": 1.5, "error_m": 0.03})
    events.append({"type": "status", "state": "done", "ts": time.time()})
    for name, rows in (("frames", frames), ("steps", steps_rows), ("events", events)):
        (rec / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (rec / "summary.json").write_text(json.dumps({"steps": 12, "sim_time_s": 1.2, "final_status": "done", "frames": fid}))
    if with_gt:
        gt = [{"tunnel": "tunnel-09", "row": 1, "plant": 3, "asset": "x", "synthetic_anomaly": "leaf_spots_strong",
               "xy": [0.5, HEAD_Y - stops[0]], "prim": "/Farm/p3"},
              {"tunnel": "tunnel-09", "row": 1, "plant": 4, "asset": "y", "synthetic_anomaly": None, "xy": [0.5, 1.0], "prim": "/Farm/p4"}]
        (rec / "ground_truth_plants.json").write_text(json.dumps(gt))
    return rec


class FakeDetector:
    made = 0

    def __init__(self, model_dir, device=None, threshold=0.5):
        FakeDetector.made += 1
        self.model_dir = model_dir

    def detect(self, image):
        return [{"label": "fruit_ripe", "score": 0.9, "bbox_norm": [0.45, 0.45, 0.55, 0.55]},
                {"label": "plant_base", "score": 0.8, "bbox_norm": [0.4, 0.8, 0.6, 0.95]},
                {"label": "leaf_spots", "score": 0.85, "bbox_norm": [0.5, 0.2, 0.6, 0.3]}]


class _Resp:
    def __init__(self, text):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text, "reasoning_content": None})()})()]
        self.usage = type("U", (), {"total_tokens": 1234})()


class FakeOpenAI:
    """Answers the six-view stop prompt (scripts/vlm_stop) and the front-camera prompt (scripts/cosmos_front)."""
    calls = []

    def __init__(self, fail=False):
        self.fail = fail
        self.chat = type("Chat", (), {})(); self.chat.completions = self

    def create(self, model, messages, **kw):
        FakeOpenAI.calls.append(model)
        if self.fail:
            raise RuntimeError("endpoint down")
        text = "".join(p.get("text", "") for p in messages[0]["content"] if isinstance(p, dict))
        n_img = sum(1 for p in messages[0]["content"] if isinstance(p, dict) and p.get("type") == "image_url")
        if "six images" in text:
            assert n_img == 6
            sick = "metre 1.00" in text
            return _Resp(json.dumps({"left": {"status": "disease" if sick else "healthy", "issues": [{"label": "leaf spots", "confidence": 0.8, "evidence": "rings"}] if sick else [],
                                              "summary": "spots on mid leaves" if sick else "fine", "description": "Canopy dense, mid view shows rings.", "action": "scout and remove leaves"},
                                     "right": {"status": "healthy", "issues": [], "summary": "fine", "description": "Green, turgid.", "action": "none"}}))
        return _Resp('<think>x</think>{"scene": "aisle ahead", "path_clear": true, "hazards": [], "recommendation": "continue", "confidence": 0.9}')


@pytest.fixture
def env(tmp_path, monkeypatch):
    model = tmp_path / "model"; model.mkdir()
    for k in ("EYES_BASE_URL", "EYES2_API_KEY", "PIPELINE_STEPS", "PIPELINE_AUTO", "PIPELINE_RUNS_DIR", "LD_LIBRARY_PATH"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DETECTOR_MODEL", str(model)); monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("EYES_BASE_URL", "http://127.0.0.1:1/v1"); monkeypatch.setenv("LINGBOT_RUN", str(tmp_path / "no_lingbot.sh"))
    monkeypatch.setenv("CUVSLAM_PYTHON", sys.executable); monkeypatch.setenv("PIPELINE_VLM_WORKERS", "2")
    import vision.detector
    FakeDetector.made = 0; FakeOpenAI.calls = []
    monkeypatch.setattr(vision.detector, "Detector", FakeDetector)
    monkeypatch.setattr(steps, "_openai_client", lambda base, key, timeout: FakeOpenAI())
    monkeypatch.setattr(steps, "_brain_chat", lambda cfg, messages: f"# NEMOTRON {messages[0]['content'][:30]}\n" + messages[1]["content"][:60])
    return tmp_path


def test_config_from_env(env):
    cfg = PipelineConfig.from_env()
    assert cfg.steps == ALL_STEPS and cfg.eyes2_api_key == "test-key" and cfg.vlm_workers == 2 and cfg.eyes_base_url
    assert "test-key" not in json.dumps(cfg.public())
    assert PipelineConfig.from_env({"PIPELINE_STEPS": "detect, report"}).steps == ("detect", "report")
    with pytest.raises(ValueError):
        PipelineConfig.from_env({"PIPELINE_STEPS": "nope"})


def test_full_pipeline_layout_markers_and_idempotency(env):
    rec = make_run(env)
    res = runner.run(rec)
    a = rec / "analysis"
    assert res["exit_code"] == 0 and res["ok"], res
    assert res["steps"] == {"detect": "ok", "vlm_stops": "ok", "lidar_map": "ok", "cuvslam": "skipped", "lingbot": "skipped",
                            "cosmos_front": "ok", "report": "ok"}
    for s in ALL_STEPS:
        m = json.loads((a / f"{s}.done").read_text()); assert m["step"] == s and m["status"] in ("ok", "skipped")
    assert "zed" in json.loads((a / "cuvslam.done").read_text())["reason"]
    assert "lingbot" in json.loads((a / "lingbot.done").read_text())["reason"]
    assert (a / "pipeline.log").exists() and (a / "pipeline.done").exists() and not (a / "pipeline.lock").exists()
    # detect: per image, per camera, 3 categories
    rows = [json.loads(l) for l in (a / "detect/findings.jsonl").read_text().splitlines()]
    assert len(rows) == 12 and {r["camera"] for r in rows} == set(CAMS) and all(r["metre"] in (1.0, 1.5) for r in rows)
    det = json.loads((a / "detect/summary.json").read_text())
    assert set(det) >= {"1_harvest_fruit_grading", "2_plant_count", "3_leaf_anomalies"}
    assert det["1_harvest_fruit_grading"][AISLE]["pass"] > 0 and det["2_plant_count"][AISLE] > 0
    assert det["3_leaf_anomalies"]["found_of_detectable"] == "1/1"
    assert set(json.loads((a / "detect/by_camera.json").read_text())) == set(CAMS)
    # vlm_stops: one request per stop with six views, description + action per plant, ground-truth check
    vrows = [json.loads(l) for l in (a / "vlm_stops/vlm_stops.jsonl").read_text().splitlines()]
    assert len(vrows) == 2 and FakeOpenAI.calls.count("nvidia/Nemotron-3.5-Super-VL") == 2
    assert {r["metre"] for r in vrows} == {1.0, 1.5} and all(len(r["images"]) == 6 for r in vrows)
    plants = json.loads((a / "vlm_stops/plants.json").read_text())
    assert len(plants) == 4 and all(p["description"] and p["action"] for p in plants)
    vs = json.loads((a / "vlm_stops/summary.json").read_text())
    assert vs["requests"] == 2 and vs["anomaly_plants_flagged"] == "1/1" and vs["status"]["disease"] == 1
    # lidar_map
    assert (a / "lidar_map/slam_map_final.png").exists() and (a / "lidar_map/localisation.json").exists()
    loc = json.loads((a / "lidar_map/localisation.json").read_text())
    assert loc["scans"] == 12 and 0.01 < loc["odometry_vs_truth"]["rmse_m"] < 0.2 and loc["scan_to_map"]["ok"] == 6
    # cosmos_front: leg change at the first frame and at the dock leg
    cf = json.loads((a / "cosmos_front/summary.json").read_text())
    assert cf["checks"] >= 2 and cf["recommendations"]["continue"] == cf["checks"] and cf["errors"] == 0
    # report: separate sections + both languages via the (stubbed) Nemotron
    rep = json.loads((a / "report.json").read_text())
    assert set(rep) >= {"harvest", "plant_count", "disease", "navigation_events", "localisation", "steps"}
    assert rep["harvest"]["total"]["pass"] == det["1_harvest_fruit_grading"][AISLE]["pass"]
    assert rep["disease"]["vlm"]["flagged"][0]["action"] == "scout and remove leaves"
    assert rep["navigation_events"]["incidents"][0]["type"] == "obstacle_stop" and rep["navigation_events"]["commands"][0]["cmd"] == "survey_farm"
    assert rep["localisation"]["lidar_odometry"]["odometry_vs_truth"]["n"] == 12 and rep["localisation"]["cuvslam"] is None
    assert rep["steps"]["cuvslam"]["status"] == "skipped"
    assert (a / "report_th.md").read_text().startswith("# NEMOTRON") and (a / "report_en.md").read_text().startswith("# NEMOTRON")
    # idempotent: second run touches nothing; --force reruns
    made, calls = FakeDetector.made, len(FakeOpenAI.calls)
    res2 = runner.run(rec)
    assert res2["exit_code"] == 0 and FakeDetector.made == made and len(FakeOpenAI.calls) == calls
    res3 = runner.run(rec, force=True, steps=("detect",))
    assert res3["steps"] == {"detect": "ok"} and FakeDetector.made == made + 1


def test_optional_step_failure_does_not_stop_the_rest(env, monkeypatch):
    rec = make_run(env, with_gt=False)

    def boom(run_dir, cfg):
        raise RuntimeError("ffmpeg exploded")
    monkeypatch.setitem(steps.STEPS, "lidar_map", boom)
    monkeypatch.setattr(steps, "_openai_client", lambda base, key, timeout: FakeOpenAI(fail=True))
    monkeypatch.setattr(steps, "_brain_chat", lambda cfg, messages: (_ for _ in ()).throw(RuntimeError("token factory down")))
    monkeypatch.setattr("scripts.vlm_stop.time.sleep", lambda s: None)
    res = runner.run(rec)
    a = rec / "analysis"
    assert res["exit_code"] == 1 and not res["ok"]
    assert res["steps"]["lidar_map"] == "error" and res["steps"]["report"] == "ok" and res["steps"]["detect"] == "ok"
    assert "ffmpeg exploded" in json.loads((a / "lidar_map.error").read_text())["reason"] and not (a / "lidar_map.done").exists()
    assert res["steps"]["vlm_stops"] == "ok"                       # per-request errors become status=error rows, not a crash
    vs = json.loads((a / "vlm_stops/summary.json").read_text()); assert vs["status"] == {"error": 4} and vs["ground_truth_check"] is None
    assert res["steps"]["cosmos_front"] == "ok" and json.loads((a / "cosmos_front/summary.json").read_text())["errors"] >= 1
    rep = json.loads((a / "report.json").read_text())
    assert rep["steps"]["lidar_map"]["status"] == "error" and rep["disease"]["rtdetr"]["ground_truth_check"] is None
    th = (a / "report_th.md").read_text(); assert "รายงานการสำรวจ" in th and "แม่แบบ" in th        # template fallback, Thai
    assert json.loads((a / "report.done").read_text())["result"]["writer"]["en"].startswith("template")
    # a failed step is retried on the next run (no .done); after it succeeds again everything is cached
    monkeypatch.setitem(steps.STEPS, "lidar_map", steps.step_lidar_map)
    res2 = runner.run(rec)
    assert res2["steps"]["lidar_map"] == "ok" and res2["exit_code"] == 0 and not (a / "lidar_map.error").exists()


def test_skips_without_optional_inputs_and_keys(env, monkeypatch):
    rec = make_run(env)
    monkeypatch.delenv("NEBIUS_API_KEY"); monkeypatch.delenv("EYES_BASE_URL")
    monkeypatch.setenv("PIPELINE_STEPS", "vlm_stops,cosmos_front,report")
    res = runner.run(rec)
    assert res["steps"] == {"vlm_stops": "skipped", "cosmos_front": "skipped", "report": "ok"} and res["exit_code"] == 0
    assert "EYES2_API_KEY" in json.loads((rec / "analysis/vlm_stops.done").read_text())["reason"]
    assert "EYES_BASE_URL" in json.loads((rec / "analysis/cosmos_front.done").read_text())["reason"]
    en = (rec / "analysis/report_en.md").read_text()
    assert "template report" in en and "## 1. Harvest" in en and "## 5. Localisation" in en
    # detect skips cleanly when the model dir is missing (no torch import attempted)
    monkeypatch.setenv("DETECTOR_MODEL", str(env / "missing_model"))
    m = runner.run_step("detect", rec, PipelineConfig.from_env())
    assert m["status"] == "skipped" and "DETECTOR_MODEL" in m["reason"]


def test_cuvslam_needs_zed_and_the_wheel(env, monkeypatch):
    rec = make_run(env)
    cfg = PipelineConfig.from_env()
    assert runner.run_step("cuvslam", rec, cfg)["status"] == "skipped"
    z = rec / "zed"                                              # raw render_route layout -> EuRoC view with symlinks
    for d in ("cam0", "cam1", "depth0"):
        (z / d).mkdir(parents=True); Image.new("L", (8, 8)).save(z / d / "1000.png")
    (z / "imu0.csv").write_text("#t,gx,gy,gz,ax,ay,az\n1000,0,0,0,0,0,9.8\n")
    (z / "calib.json").write_text(json.dumps({"K": [[1, 0, 4], [0, 1, 4], [0, 0, 1]], "size": [8, 8], "baseline_m": 0.12, "imu_hz": 400, "imu_noise": {}}))
    (z / "groundtruth_part1.csv").write_text("2000,0,1,0,1,0,0,0\n"); (z / "groundtruth_part0.csv").write_text("1000,0,0,0,1,0,0,0\n")
    m = runner.run_step("cuvslam", rec, cfg, force=True)
    assert m["status"] == "skipped" and "cuvslam" in m["reason"]            # the wheel is not installed here
    e = rec / "analysis/cuvslam/euroc"
    assert (e / "cam0/data/1000.png").exists() and (e / "imu0/data.csv").exists()
    assert (e / "groundtruth.csv").read_text().splitlines()[1].startswith("1000,")

    def fake_run(cmd, **kw):                                      # wheel present, script writes its result
        if cmd[1:3] == ["-c", "import cuvslam"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[1].endswith("cuvslam_r181.py"):
            Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps({"method": f"cuVSLAM {cmd[cmd.index('--mode') + 1]}", "frames": 1, "tracked": 1,
                                                                      "ate_rmse_m": 0.01, "poses": []}))
            return subprocess.CompletedProcess(cmd, 0, "ok", "")
        return subprocess.CompletedProcess(cmd, 0, "/tmp/sp", "")
    monkeypatch.setattr(steps.subprocess, "run", fake_run)
    monkeypatch.setenv("CUVSLAM_MODES", "stereo,rgbd")
    m = runner.run_step("cuvslam", rec, PipelineConfig.from_env(), force=True)
    assert m["status"] == "ok" and set(m["result"]) == {"stereo", "rgbd"} and m["result"]["stereo"]["ate_rmse_m"] == 0.01


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_hub_run_complete_marker_and_auto_pipeline(env, monkeypatch):
    import agent.robot_api as api
    from sim.cloud_link import CloudLink
    rec = make_run(env)
    launched = []
    monkeypatch.setattr(api, "launch_pipeline", lambda run_dir: launched.append(Path(run_dir)))
    srv = api.serve(env / "hub", None, port=0); port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        code, r = _post(base + "/run_complete", {"run": "farm_test", "path": str(rec), "summary": {"steps": 12}})
        assert code == 200 and r["pipeline"] == "marker written"
        m = json.loads((rec / "run_complete.json").read_text())
        assert m["run"] == "farm_test" and m["path"] == str(rec) and m["summary"] == {"steps": 12}
        assert json.loads((env / "hub/runs.jsonl").read_text().splitlines()[0])["run"] == "farm_test"
        assert _post(base + "/run_complete", {"path": str(env / "nowhere")})[0] == 400
        assert launched == []
        monkeypatch.setenv("PIPELINE_AUTO", "1")
        link = CloudLink(base, outbox=env / "outbox", log=lambda m: None)      # robot side (run_farm.py finally block)
        assert link.post_run_complete(rec, {"steps": 12})
        time.sleep(0.2)
        assert launched == [rec] and json.loads((rec / "run_complete.json").read_text())["run"] == "farm_test"
        with urllib.request.urlopen(base + "/health") as h:
            assert json.loads(h.read())["pipeline_auto"] is True
        assert not CloudLink("http://127.0.0.1:9", outbox=env / "o2", timeout_s=0.3, log=lambda m: None).post_run_complete(rec)
    finally:
        srv.shutdown()


def test_cli_run_and_watcher(env, monkeypatch):
    rec = make_run(env)
    e = {**os.environ, "PIPELINE_STEPS": "lidar_map,report", "NEBIUS_API_KEY": "", "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-m", "cloud.pipeline", "run", str(rec)], cwd=ROOT, env=e, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["steps"] == {"lidar_map": "ok", "report": "ok"} and (rec / "analysis/report_en.md").exists()
    # watcher: only runs with run_complete.json and no pipeline.done
    rec2 = make_run(env / "second"); (rec2 / "run_complete.json").write_text("{}")
    assert runner.pending_runs(env) == [] and runner.pending_runs(env / "second") == [rec2]
    (rec / "run_complete.json").write_text("{}")
    assert runner.pending_runs(env) == []                          # already processed: the CLI wrote pipeline.done
    (rec / "analysis/pipeline.done").unlink(); (rec / "analysis/pipeline.lock").write_text(json.dumps({"pid": os.getpid()}))
    assert runner.pending_runs(env) == []                          # locked by a live process -> left alone
    (rec / "analysis/pipeline.lock").unlink()
    assert runner.pending_runs(env) == [rec]
    monkeypatch.setenv("PIPELINE_STEPS", "report")
    assert runner.watch(env, poll_s=0, once=True) == 1 and (rec / "analysis/pipeline.done").exists()
    assert runner.run(rec, PipelineConfig.from_env())["exit_code"] == 0
    assert runner.run(env / "not_a_run")["exit_code"] == 3
