"""Robot command hub: HERMES (cloud) posts goals, the robot (sim or Jetson) pulls them, pushes status and camera frames.
Side-camera frames are sent to Cosmos Reason (cloud/eyes) when EYES_BASE_URL is set. Everything is also written to a data dir
in the dashboard format (status.json, frames.jsonl, findings.jsonl, decisions.jsonl, aisle_map.json). Stdlib only.

  python3 agent/robot_api.py --data runs/demo --map data/tunnel09_aisle_map_v18.json --port 8090

HERMES side:  POST /command {cmd...,"reasoning": "..."}  GET /status  GET /findings?since=N  GET /map  GET /health
Robot side:   GET /next_command (204 if none)  POST /status {..}  POST /frame (multipart: meta JSON + image jpeg)
              POST /run_complete {"run": name, "path": <run dir on this server>, "summary": {..}} when the survey ends:
              writes <run dir>/run_complete.json (the cloud/pipeline watcher picks it up); with PIPELINE_AUTO=1 the hub
              also spawns `python3 -m cloud.pipeline run <run dir>` itself (docs/PIPELINE.md)."""
from __future__ import annotations
import argparse, json, os, queue, shutil, subprocess, sys, threading, time
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r1_nav.aisle_map import AisleMap            # noqa: E402
from r1_nav.commands import plan_command          # noqa: E402
from r1_nav.cameras import SIDE_CAMERAS           # noqa: E402


class Hub:
    mission_plan: dict | None = None

    def __init__(self, data: Path, map_path: Path | None):
        self.data, self.lock = data, threading.Lock()
        (data / "frames").mkdir(parents=True, exist_ok=True)
        self.amap = AisleMap.load(map_path) if map_path else None
        if map_path:
            shutil.copyfile(map_path, data / "aisle_map.json")
        self.commands: queue.Queue[dict] = queue.Queue()
        self.status: dict = {"state": "idle", "ts": time.time()}
        self.findings: list[dict] = []
        self.inspect_q: queue.Queue[tuple[bytes, dict]] = queue.Queue()
        threading.Thread(target=self._inspect_worker, daemon=True).start()
        self.detect_q: queue.Queue[tuple[bytes, dict]] = queue.Queue()
        self.situations: list[dict] = []             # SAFETY stops explained by Cosmos Reason (cloud/eyes/situation)
        self.situation_q: queue.Queue[tuple[bytes, dict]] = queue.Queue()
        threading.Thread(target=self._situation_worker, daemon=True).start()
        if os.environ.get("DETECTOR_MODEL"):       # local RT-DETR: fruit grading, plant count, leaf anomalies
            threading.Thread(target=self._detect_worker, daemon=True).start()
        env = os.environ.get                       # dashboard service chips (configured, not health-checked)
        (data / "services.json").write_text(json.dumps({
            "nemotron": {"live": bool(env("NEBIUS_API_KEY")), "model": env("BRAIN_MODEL", ""), "detail": "Nebius Token Factory"},
            "cosmos": {"live": bool(env("EYES_BASE_URL")), "model": env("EYES_MODEL", ""), "detail": env("EYES_BASE_URL", "")}}))

    def _append(self, name: str, row: dict) -> None:
        with self.lock, open(self.data / name, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def command(self, cmd: dict) -> dict:
        if cmd.get("cmd") == "notify":             # HERMES -> farmer alert; nothing is sent to the robot
            reply = {"status": "accepted", "reason": "farmer notified", "legs": []}
            self._append("alerts.jsonl", {"ts": time.time(), "message": cmd.get("message", ""),
                                          "situation": cmd.get("situation"), "reasoning": cmd.get("reasoning", "")})
        elif cmd.get("cmd") == "set_map":            # after a human confirms the discovered map
            p = Path(cmd["path"]); self.amap = AisleMap.load(p); shutil.copyfile(p, self.data / "aisle_map.json")
            reply = {"status": "accepted", "reason": f"map {self.amap.house_id} loaded", "legs": []}
        elif cmd.get("cmd") == "survey_farm" or (cmd.get("cmd") == "revisit" and self.amap is not None
                                                  and not str(cmd.get("aisle", "")).startswith(self.amap.house_id + "/")):
            # farm-level goals span several houses: the robot plans them against its own farm map (run_farm.py)
            reply = {"status": "accepted", "reason": "farm-level goal: planned on the robot", "legs": []}
            self.commands.put(cmd)
        else:
            reply = plan_command(cmd, self.amap).to_dict()
            if reply["status"] == "accepted":
                if cmd["cmd"] == "stop":           # stop jumps the queue
                    with self.commands.mutex:
                        self.commands.queue.clear()
                self.commands.put(cmd)
        self._append("decisions.jsonl", {"ts": time.time(), "reasoning": cmd.get("reasoning", ""),
                                         "command": {k: v for k, v in cmd.items() if k != "reasoning"},
                                         "reply": reply["status"], "reason": reply["reason"]})
        return reply

    def planner(self):
        """The mission planner on the robot's own map (env R1_STRUCTURE = a scripts/free_nav.py output dir), lazily."""
        if not hasattr(self, "_planner"):
            try:
                from agent.mission_planner import load
                self._planner = load()
            except Exception:
                self._planner = None
        return self._planner

    def set_mission_plan(self, plan: dict) -> None:
        """An approved map-only mission: stored, logged, and queued for the robot as `survey_aisles`."""
        plan = dict(plan, t=time.time()); self.mission_plan = plan
        (self.data / "mission_plan.json").write_text(json.dumps(plan, indent=1)); self._append("commands.jsonl", {"cmd": "survey_aisles", **plan})
        self.commands.put({"cmd": "survey_aisles", "order": plan["order"], "approved_by": plan.get("approved_by"), "t": plan["t"]})

    def set_status(self, st: dict) -> None:
        st = {**st, "ts": st.get("ts", time.time())}
        with self.lock:
            self.status = st
            (self.data / "status.json").write_text(json.dumps(st, ensure_ascii=False))

    def frame(self, meta: dict, image: bytes) -> dict:
        rel = f"frames/{meta.get('frame_id', int(time.time() * 1000))}_{meta.get('camera', 'cam')}.jpg"
        (self.data / rel).write_bytes(image)
        self._append("frames.jsonl", {**meta, "image_path": rel})
        if meta.get("purpose") == "situation":
            self.situation_q.put((image, {**meta, "image_path": rel}))
            return {"stored": rel, "situation_queued": self.situation_q.qsize()}
        if os.environ.get("EYES_BASE_URL") and meta.get("camera") in SIDE_CAMERAS and meta.get("pose_confidence", 1) >= 0.5:
            self.inspect_q.put((image, {**meta, "image_path": rel}))
        if os.environ.get("DETECTOR_MODEL") and meta.get("camera") in SIDE_CAMERAS and meta.get("pose_confidence", 1) >= 0.5:
            self.detect_q.put((image, {**meta, "image_path": rel}))
        return {"stored": rel, "inspect_queued": self.inspect_q.qsize(), "detect_queued": self.detect_q.qsize()}

    def run_complete(self, body: dict) -> dict:
        """Robot says its survey is over: write <run dir>/run_complete.json (+ runs.jsonl here) and, with PIPELINE_AUTO=1,
        start the analysis pipeline on it. The run dir is body["path"] (a dir on this server: shared mount or rsync
        target), else PIPELINE_RUNS_DIR/<run>, else this hub's data dir."""
        run = str(body.get("run") or time.strftime("run_%Y%m%d_%H%M%S"))
        if body.get("path"):
            run_dir = Path(body["path"])
        elif os.environ.get("PIPELINE_RUNS_DIR"):
            run_dir = Path(os.environ["PIPELINE_RUNS_DIR"]) / run
        else:
            run_dir = self.data
        if not run_dir.is_dir():
            raise ValueError(f"run dir not found on this server: {run_dir}")
        marker = {"run": run, "path": str(run_dir), "ts": time.time(), "summary": body.get("summary"),
                  "source": body.get("source", "robot")}
        (run_dir / "run_complete.json").write_text(json.dumps(marker, ensure_ascii=False, indent=1))
        self._append("runs.jsonl", marker)
        auto = os.environ.get("PIPELINE_AUTO") == "1"
        if auto:
            threading.Thread(target=launch_pipeline, args=(run_dir,), daemon=True).start()
        return {"ok": True, "run": run, "path": str(run_dir), "pipeline": "spawned" if auto else "marker written"}

    def run_status(self, run: str) -> dict:
        """Where a handed-off run stands: the newest runs.jsonl row for `run`, plus what the pipeline has written in
        <run dir>/analysis (step markers, pipeline.done, report.json). Lets the robot or a farmer poll for the report."""
        rows = [json.loads(l) for l in (self.data / "runs.jsonl").read_text().splitlines()] if (self.data / "runs.jsonl").exists() else []
        rows = [r for r in rows if r.get("run") == run]
        if not rows:
            return {"run": run, "state": "unknown"}
        run_dir = Path(rows[-1]["path"]); an = run_dir / "analysis"
        done = sorted(p.stem for p in an.glob("*.done")) if an.is_dir() else []
        failed = sorted(p.stem for p in an.glob("*.failed")) if an.is_dir() else []
        report = (an / "report.json").exists()
        state = "done" if (an / "pipeline.done").exists() else "failed" if failed and not report else "running" if done else "queued"
        return {"run": run, "state": state, "path": str(run_dir), "received_ts": rows[-1].get("ts"), "steps_done": done,
                "steps_failed": failed, "report": "analysis/report.json" if report else None,
                "report_md": [p.name for p in an.glob("report_*.md")] if an.is_dir() else []}

    def _inspect_worker(self) -> None:
        while True:
            image, meta = self.inspect_q.get()
            try:
                from cloud.eyes.inspect import inspect_image          # lazy: needs openai + endpoint
                f = inspect_image(image, meta)
                f = f.model_dump() if hasattr(f, "model_dump") else dict(f)
                row = {k: meta.get(k) for k in ("frame_id", "aisle", "metre", "camera", "image_path", "purpose")} | f
            except Exception as e:                                     # keep the hub alive if the endpoint is down
                row = {k: meta.get(k) for k in ("frame_id", "aisle", "metre", "camera", "image_path")} | \
                      {"status": "error", "issues": [], "summary": f"inspect failed: {e}"}
            with self.lock:
                self.findings.append(row)
            self._append("findings.jsonl", row)


    def _situation_worker(self) -> None:
        keep = ("frame_id", "trigger", "why", "aisle", "metre", "xy", "lidar_front_m", "image_path", "t")
        while True:
            image, meta = self.situation_q.get()
            row = {k: meta.get(k) for k in keep}
            try:
                if not os.environ.get("EYES_BASE_URL"):
                    raise RuntimeError("no EYES_BASE_URL")
                from cloud.eyes.situation import describe_situation      # lazy: needs openai + endpoint
                row |= describe_situation(image, meta)
            except Exception as e:
                row |= {"what": "unknown", "description": f"situation report failed: {e}", "blocking_path": True,
                        "likely_to_move": False, "crop_concern": "", "confidence": 0.0}
            row["ts"] = time.time()
            with self.lock:
                self.situations.append(row)
            self._append("situations.jsonl", row)

    def _detect_worker(self) -> None:
        import io
        from PIL import Image
        from vision.detector import Detector, to_finding                # lazy: needs torch + transformers
        det = Detector(os.environ["DETECTOR_MODEL"], threshold=float(os.environ.get("DETECTOR_THRESHOLD", "0.5")))
        while True:
            image, meta = self.detect_q.get()
            try:
                im = Image.open(io.BytesIO(image))
                f = to_finding(det.detect(im), im, meta)
                row = {k: meta.get(k) for k in ("frame_id", "aisle", "metre", "camera", "image_path", "purpose")} | f
            except Exception as e:
                row = {k: meta.get(k) for k in ("frame_id", "aisle", "metre", "camera", "image_path")} | \
                      {"status": "error", "issues": [], "source": "rtdetr", "summary": f"detector failed: {e}"}
            with self.lock:
                self.findings.append(row)
            self._append("findings.jsonl", row)


def launch_pipeline(run_dir: Path) -> int:
    """`python3 -m cloud.pipeline run <run_dir>` as a child process (own torch/CUDA, own log in <run_dir>/analysis/)."""
    (run_dir / "analysis").mkdir(exist_ok=True)
    with open(run_dir / "analysis" / "pipeline.out", "a") as out:
        r = subprocess.run([sys.executable, "-m", "cloud.pipeline", "run", str(run_dir)], cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT)
    print(f"[hub] pipeline for {run_dir} exited {r.returncode}", flush=True)
    return r.returncode


def make_handler(hub: Hub):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):                 # quiet
            pass

        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def _body(self) -> bytes:
            return self.rfile.read(int(self.headers.get("Content-Length", 0)))

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/health":
                return self._json(200, {"ok": True, "map": hub.amap.house_id if hub.amap else None,
                                        "eyes": bool(os.environ.get("EYES_BASE_URL")),
                                        "pipeline_auto": os.environ.get("PIPELINE_AUTO") == "1"})
            if u.path == "/status":
                return self._json(200, hub.status)
            if u.path == "/map":
                if not hub.amap:
                    return self._json(200, {"house": None, "aisles": []})
                return self._json(200, {"house": hub.amap.house_id, "aisles": [
                    {"id": a.id, "label": a.label, "length_m": round(a.length_m, 2), "clearance_m": round(a.clearance_m, 3),
                     "traversable": a.traversable()} for a in hub.amap.aisles]})
            if u.path == "/findings":
                since = int(parse_qs(u.query).get("since", ["0"])[0])
                with hub.lock:
                    rows = hub.findings[since:]
                return self._json(200, {"next": since + len(rows), "findings": rows})
            if u.path == "/situations":
                since = int(parse_qs(u.query).get("since", ["0"])[0])
                with hub.lock:
                    rows = hub.situations[since:]
                return self._json(200, {"next": since + len(rows), "situations": rows})
            if u.path == "/run_status":
                return self._json(200, hub.run_status(parse_qs(u.query).get("run", [""])[0]))
            if u.path == "/structure":
                mp = hub.planner()
                if mp is None:
                    return self._json(200, {"available": False, "why": "no structure built from the robot's map yet (scripts/free_nav.py; env R1_STRUCTURE)"})
                return self._json(200, {"available": True} | mp.structure())
            if u.path == "/scene_graph":
                mp = hub.planner()
                if mp is None or not (mp.dir / "scene_graph.json").exists():
                    return self._json(200, {"available": False, "why": "no scene graph yet (scripts/free_nav.py writes scene_graph.json next to the structure)"})
                g = json.load(open(mp.dir / "scene_graph.json"))
                g["legend"] = {"house-X": "a greenhouse: a block of parallel walls the robot's LiDAR saw; corners_xy outline it", "house-X/row-N": "a plant row inside that house: a drivable corridor with plants on both sides; end_low/end_high are its two ends",
                               "house-X/door-low|high": "where the drivable space crosses the house's end wall - the way in and out", "yard-low|yard-high": "open ground beyond the houses' ends; it connects the doors listed",
                               "gap-N": "the open strip between two houses, connecting the two yards", "dock": "where the robot was switched on; the survey ends there",
                               "not listed": "anything the robot has never seen empty - ponds, orchard, road - is not drivable and never planned through"}
                return self._json(200, {"available": True} | g)
            if u.path == "/structure/picture":
                mp = hub.planner()
                if mp is None:
                    return self._json(404, {"error": "no structure"})
                q = parse_qs(u.query); order = q.get("order", [""])[0].split(",") if q.get("order", [""])[0] else None
                start = [float(v) for v in q["from"][0].split(",")] if q.get("from") else None
                body = mp.picture(order, start); self.send_response(200); self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            if u.path == "/mission_plan":
                return self._json(200, hub.mission_plan or {"approved": False})
            if u.path == "/next_command":
                try:
                    return self._json(200, hub.commands.get_nowait())
                except queue.Empty:
                    self.send_response(204); self.end_headers(); return
            self._json(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                if path == "/command":
                    return self._json(200, hub.command(json.loads(self._body())))
                if path == "/status":
                    hub.set_status(json.loads(self._body())); return self._json(200, {"ok": True})
                if path == "/frame":
                    msg = BytesParser(policy=HTTP).parsebytes(
                        b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + self._body())
                    parts = {p.get_param("name", header="content-disposition"): p.get_payload(decode=True)
                             for p in msg.iter_parts()}
                    return self._json(200, hub.frame(json.loads(parts["meta"]), parts["image"]))
                if path == "/run_complete":
                    return self._json(200, hub.run_complete(json.loads(self._body() or b"{}")))
                if path == "/plan":
                    mp = hub.planner(); b = json.loads(self._body() or b"{}")
                    if mp is None:
                        return self._json(400, {"error": "no structure"})
                    start = b.get("from") or hub.status.get("xy") or list(mp.dock[:2])
                    rpp = mp.rpp_order(start, b.get("aisles")); order = b.get("order") or (rpp["order"] if rpp["ok"] else [a["id"] for a in mp.aisles])
                    out = mp.plan(order, start); out["alternatives"] = mp.alternatives(start)[:6]; out["rural_postman"] = rpp
                    if b.get("picture", True):
                        out["picture"] = mp.picture_b64(order, start)
                    return self._json(200, out)
                if path == "/mission_plan":
                    mp = hub.planner(); b = json.loads(self._body() or b"{}")
                    if mp is None:
                        return self._json(400, {"error": "no structure"})
                    start = b.get("from") or hub.status.get("xy") or list(mp.dock[:2]); p = mp.plan(b.get("order", []), start)
                    if not p["ok"]:
                        return self._json(400, {"error": "plan not acceptable", "plan": {k: v for k, v in p.items() if k != "polyline_xy"}})
                    hub.set_mission_plan({"approved": True, "approved_by": b.get("approved_by", "HERMES"), "reason": b.get("reason", ""), "order": p["order"],
                                          "legs": p["legs"], "total_m": p["total_m"], "mode": "map_only"})
                    return self._json(200, hub.mission_plan)
            except (KeyError, ValueError, TypeError) as e:
                return self._json(400, {"error": str(e)})
            self._json(404, {"error": "not found"})
    return H


def serve(data: Path, map_path: Path | None, host: str = "127.0.0.1", port: int = 8090) -> ThreadingHTTPServer:
    hub = Hub(data, map_path)
    srv = ThreadingHTTPServer((host, port), make_handler(hub)); srv.hub = hub
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/demo"); ap.add_argument("--map", default=None)
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    srv = serve(Path(a.data), Path(a.map) if a.map else None, a.host, a.port)
    print(f"robot API on http://{a.host}:{a.port}  data={a.data}  map={a.map}")
    srv.serve_forever()
