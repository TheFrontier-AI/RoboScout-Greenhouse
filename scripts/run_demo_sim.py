"""Headless demo loop: R1 in the Isaac R180 env takes goals from the robot hub (agent/robot_api.py), plans them with
r1_nav.commands, drives the legs with sim.executor and pushes status + camera frames back. No training.

  # online (hub at ROBOT_API_URL, default http://127.0.0.1:8090)
  R180_DIR=.../physical_ai_r180 /path/to/isaac-sim/python.sh scripts/run_demo_sim.py --record runs/demo_sim
  # scripted, no hub: discover_house → survey_aisle aisle-02 → revisit metre 15 → return_to_dock
  /path/to/isaac-sim/python.sh scripts/run_demo_sim.py --offline --record runs/demo_offline
  # same loop without Isaac (kinematic fake) to test the plumbing
  python3 scripts/run_demo_sim.py --backend fake --offline --record /tmp/demo_fake"""
import argparse, collections, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r1_nav.aisle_map import AisleMap                     # noqa: E402
from r1_nav.cameras import FRONT_CAMERA                   # noqa: E402
from r1_nav.commands import plan_command                  # noqa: E402
from sim.backends import make_sim                         # noqa: E402
from sim.cloud_link import CloudLink, encode_image        # noqa: E402
from sim.executor import ExecutorConfig, LegExecutor, WheelImuOdometry   # noqa: E402
from sim.isaac_env import house_translation               # noqa: E402

HOUSE = "tunnel-09"
OFFLINE_SCRIPT = [{"cmd": "discover_house"},
                  {"cmd": "survey_aisle", "house": HOUSE, "aisle": f"{HOUSE}/aisle-02"},
                  {"cmd": "revisit", "house": HOUSE, "aisle": f"{HOUSE}/aisle-02", "metre": 15},
                  {"cmd": "return_to_dock"}]


def log(msg: str) -> None:
    print(f"[demo {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Recorder:
    """Everything needed to cut the video: per-step log, events, every captured frame, front_rgb every N steps."""

    def __init__(self, root: Path, every: int):
        self.root, self.every = root, every
        for d in ("frames", "video"):
            (root / d).mkdir(parents=True, exist_ok=True)
        self.files = {n: open(root / f"{n}.jsonl", "a") for n in ("steps", "events", "frames")}

    def write(self, name: str, row: dict) -> None:
        self.files[name].write(json.dumps(row, ensure_ascii=False, default=float) + "\n")

    def frame(self, meta: dict, data: bytes, mime: str) -> None:
        rel = f"frames/{meta['frame_id']:06d}_{meta['camera']}.{'jpg' if mime == 'image/jpeg' else 'png'}"
        (self.root / rel).write_bytes(data)
        self.write("frames", {**meta, "image_path": rel})

    def video(self, step: int, rgb) -> None:
        if self.every and step % self.every == 0:
            data, mime = encode_image(rgb)
            (self.root / "video" / f"{step:06d}.{'jpg' if mime == 'image/jpeg' else 'png'}").write_bytes(data)

    def close(self) -> None:
        for f in self.files.values():
            f.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("isaac", "fake"), default="isaac")
    ap.add_argument("--offline", action="store_true", help="run a scripted command list, no hub")
    ap.add_argument("--script", type=Path, help="JSON list of commands for --offline (default: built-in demo)")
    ap.add_argument("--record", type=Path, help="save frames, per-step log, events and video frames here")
    ap.add_argument("--record-every", type=int, default=1, help="save front_rgb to video/ every N control steps (0 = off)")
    ap.add_argument("--map", type=Path, default=ROOT / "data/tunnel09_aisle_map_v18.json")
    ap.add_argument("--aisle", default=f"{HOUSE}/aisle-02", help="the aisle the R180 scene starts in (only one simulated)")
    ap.add_argument("--r180-dir", default=os.environ.get("R180_DIR"))
    ap.add_argument("--stage", default="default", help="default | photo | path relative to R180 dir")
    ap.add_argument("--no-side-cams", action="store_true")
    ap.add_argument("--api-url", default=os.environ.get("ROBOT_API_URL", "http://127.0.0.1:8090"))
    ap.add_argument("--outbox", type=Path, default=ROOT / "runs/sim_outbox")
    ap.add_argument("--poll-s", type=float, default=1.0)
    ap.add_argument("--status-s", type=float, default=1.0)
    ap.add_argument("--max-sim-s", type=float, default=3600.0)
    ap.add_argument("--idle-exit-s", type=float, default=0.0, help="online: exit after this long idle (0 = never)")
    ap.add_argument("--obstacle-metre", type=float, help="place the test box at this aisle metre (SAFETY clip); default none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disease-metre", type=float, help="labelled test fixture: diseased-foliage panel at this metre")
    ap.add_argument("--disease-side", choices=("left", "right"), default="right")
    args = ap.parse_args()

    amap = AisleMap.load(args.map)
    aisle = amap.get(args.aisle)
    house_xy = house_translation(args.map)
    link = CloudLink(None if args.offline else args.api_url, outbox=args.outbox, log=log)
    pending = collections.deque(json.loads(args.script.read_text()) if args.script else OFFLINE_SCRIPT) \
        if args.offline else collections.deque()
    rec = Recorder(args.record, args.record_every) if args.record else None
    if link.enabled:
        log(f"hub {link.base}: {link.health() or 'NOT REACHABLE (will retry; frames go to ' + str(args.outbox) + ')'}")

    sim, app = make_sim(args.backend, aisle, house_xy, args.r180_dir, args.stage, not args.no_side_cams, log=log)
    if not sim.has_side_cameras:
        log("WARNING: no side (plant) cameras in this env; side-camera captures use front_rgb as a PLACEHOLDER "
            "(meta.image_source = front_rgb_placeholder). See sim/SIDE_CAMERAS_TODO.md")
    obs, info = sim.reset(seed=args.seed, obstacle_metre=args.obstacle_metre)
    if info.get("unsupported"):
        log(f"WARNING env could not apply: {info['unsupported']}")
    if args.disease_metre is not None and sim.backend == "isaac":
        from sim.anomaly import add_disease_panel
        from sim.make_disease_texture import make
        tex = Path(args.outbox).parent / "disease_panel.png"; tex.parent.mkdir(parents=True, exist_ok=True)
        make().save(tex)
        log(f"TEST FIXTURE: diseased-foliage panel at metre {args.disease_metre} ({args.disease_side}) "
            f"→ {add_disease_panel(sim.stage, sim.frame, args.disease_metre, args.disease_side, tex)}")
    ex = LegExecutor(WheelImuOdometry(metre=sim.start_metre), amap.house_id, ExecutorConfig())
    log(f"backend={sim.backend} aisle={aisle.id} start_metre={sim.start_metre:.2f} knobs={sim.knobs}")

    def publish(extra: dict | None = None) -> None:
        st = {**ex.status_dict(), "ts": time.time(), "sim_time_s": round(float(obs["timestamp_s"]), 2),
              "backend": sim.backend, "frames_queued": link.backlog, "pending_commands": len(pending), **(extra or {})}
        st["legs_remaining"] += len(pending)                  # queued commands keep the brain from thinking we're idle
        if link.enabled:
            link.post_status(st)
        if rec:
            rec.write("events", {"type": "status", **st})

    def start(cmd: dict) -> None:
        reply = plan_command(cmd, amap)
        foreign = [l.aisle_id for l in reply.legs if l.aisle_id not in (None, aisle.id)]
        if reply.status == "accepted" and foreign:
            reply.status, reply.reason = "rejected", f"only {aisle.id} is simulated, not {foreign[0]}"
        log(f"cmd {cmd.get('cmd')} → {reply.status} {reply.reason} legs={[l.kind for l in reply.legs]}")
        if rec:
            rec.write("events", {"type": "command", "t": float(obs["timestamp_s"]), "cmd": cmd, "reply": reply.to_dict()})
        if reply.status == "accepted":
            ex.load(reply.legs, cmd.get("cmd"))
        publish({"reply": reply.status, "reason": reply.reason})

    def handle(cmd: dict) -> None:
        if cmd.get("cmd") == "stop":
            pending.clear()
            ex.abort("stop command")
            log("STOP")
            publish()
        elif ex.busy:
            pending.append(cmd)
            log(f"queued {cmd.get('cmd')} until current legs finish")
        else:
            start(cmd)

    def dispatch(events: list[dict]) -> None:
        side = None
        for ev in events:
            if rec:
                rec.write("events", {"t": float(obs["timestamp_s"]), **{k: v for k, v in ev.items() if k != "meta"},
                                     **({"frame_id": ev["meta"]["frame_id"]} if "meta" in ev else {})})
            if ev["type"] != "capture":
                if ev["type"] in ("leg_done", "stuck", "aborted", "obstacle_stop", "obstacle_clear"):
                    log(f"{ev['type']}: {ev.get('reason') or ev.get('leg', {}).get('kind', '')} metre={ev.get('metre')}")
                continue
            meta, cam = dict(ev["meta"]), ev["camera"]
            if cam == FRONT_CAMERA:
                img, meta["image_source"] = obs["front_rgb"], "front_rgb"
            else:
                side = sim.side_images() if side is None else side
                img = side.get(cam)
                meta["image_source"] = cam if img is not None else "front_rgb_placeholder"
                img = obs["front_rgb"] if img is None else img
            data, mime = encode_image(img)
            if link.enabled:
                link.send_frame(meta, data, mime)
            if rec:
                rec.frame(meta, data, mime)

    step, last_poll, last_status, code, idle_since = 0, 0.0, 0.0, 0, time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if link.enabled and now - last_poll >= args.poll_s:
                last_poll = now
                cmd = link.next_command()
                if cmd:
                    handle(cmd)
            if not ex.busy and pending:
                start(pending.popleft())
                continue
            if not ex.busy:                                       # idle: do not burn GPU stepping physics
                if args.offline or (args.idle_exit_s and now - idle_since >= args.idle_exit_s):
                    break
                if now - last_status >= args.status_s:
                    publish(); last_status = now
                link.flush()
                time.sleep(args.poll_s)
                continue
            idle_since = time.monotonic()
            wheel, events = ex.step(obs)
            if events:
                dispatch(events)
            obs, _, term, trunc, info = sim.step(wheel)
            step += 1
            if rec:
                rec.write("steps", {"step": step, "t": float(obs["timestamp_s"]), "wheel": [round(w, 4) for w in wheel],
                                    "status": ex.status, "leg": ex.current_leg.kind if ex.current_leg else None,
                                    "metre": round(ex.metre, 3), "heading_deg": round(ex.heading, 2),
                                    "confidence": round(ex.confidence, 3), "gt": info.get("gt")})
                rec.video(step, obs["front_rgb"])
            if time.monotonic() - last_status >= args.status_s:
                publish(); last_status = time.monotonic()
            if term or trunc:
                ex.abort(f"sim episode ended: {info.get('termination')}")
                log(f"sim ended: {info.get('termination')}"); code = 2
                break
            if float(obs["timestamp_s"]) > args.max_sim_s:
                ex.abort("max sim time"); log("max sim time reached"); break
    except KeyboardInterrupt:
        ex.abort("interrupted"); log("interrupted")
    except Exception as e:                                    # SimulationApp.close() exits the process: log before it
        import traceback
        ex.abort(f"error: {e!r}"); log("ERROR " + traceback.format_exc()); code = 3
    finally:
        publish()
        if link.enabled:
            link.flush()
        summary = {"steps": step, "sim_time_s": float(obs["timestamp_s"]), "final_status": ex.status, "reason": ex.reason,
                   "frames": ex.frame_id, "frames_sent": link.sent, "frames_queued": link.backlog, "backend": sim.backend,
                   "side_cameras": sim.has_side_cameras, "final_gt": info.get("gt")}
        log(f"summary {json.dumps(summary, default=float)}")
        if rec:
            (rec.root / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
            rec.close()
        sim.close()
        if app is not None:
            app.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
