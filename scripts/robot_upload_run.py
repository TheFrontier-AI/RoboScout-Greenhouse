#!/usr/bin/env python3
"""Robot-side hand-off of a finished survey to the cloud, without Isaac Sim: upload the recorded run directory to the
Nebius server and tell the hub it is complete, exactly as scripts/run_farm.py does at the end of a real drive.
Runs on the robot computer (Jetson Orin or any Linux box) with only the stdlib + rsync + ssh.

  python3 scripts/robot_upload_run.py --rec /data/runs/farm_001/rec \
      --dest ubuntu@SERVER:/home/ubuntu/runs/farm_001/rec --hub http://SERVER:8090 [--wait]

Uploads only what the analysis pipeline reads (steps/frames/events jsonl, side/, front/, zed/, summary and scene
files); the 30 Hz chase/third-person renders stay on the robot. --wait polls GET /run_status until the pipeline has
written analysis/report.json (or fails). Exit code 0 only when the hub accepted the run (and, with --wait, the
report exists)."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.cloud_link import CloudLink                                 # noqa: E402

INCLUDE = ["*.jsonl", "*.json", "*.npy", "*.csv", "side/***", "front/***", "zed/***", "frames/***"]


def rsync_run(rec: Path, dest: str, log=print) -> str | None:
    """rsync the pipeline inputs of `rec` to `dest` (user@host:/path or a local path). Returns the server path."""
    src = str(rec.resolve()).rstrip("/") + "/"
    # accept-new: a robot's first contact with a new server must not hang on a host-key prompt (known hosts still pinned)
    cmd = ["rsync", "-a", "--partial", "--mkpath", "-e", "ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=20"] + \
          [f"--include={p}" for p in INCLUDE] + ["--include=*/", "--exclude=*", src, dest.rstrip("/") + "/"]
    log("[upload] " + " ".join(cmd[:4]) + f" ... {src} -> {dest}")
    r = subprocess.run(cmd, text=True)
    if r.returncode != 0:
        log(f"[upload] rsync exit {r.returncode}")
        return None
    return dest.split(":", 1)[1] if ":" in dest else dest


def run_status(hub: str, run: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{hub.rstrip('/')}/run_status?run={run}", timeout=10) as r:
            return json.loads(r.read())
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rec", type=Path, required=True, help="recorded run dir (contains frames.jsonl, side/, front/ ...)")
    ap.add_argument("--dest", required=True, help="rsync target on the server, e.g. ubuntu@1.2.3.4:/home/ubuntu/runs/farm_001/rec")
    ap.add_argument("--hub", default=os.environ.get("ROBOT_API_URL", "http://127.0.0.1:8090"), help="hub base URL (ROBOT_API_URL)")
    ap.add_argument("--run", default=None, help="run name (default: parent dir name of rec)")
    ap.add_argument("--wait", action="store_true", help="poll the hub until analysis/report.json exists")
    ap.add_argument("--timeout-s", type=float, default=3 * 3600)
    a = ap.parse_args()
    rec = a.rec
    if not (rec / "frames.jsonl").exists() and not (rec / "steps.jsonl").exists():
        print(f"not a run dir (no frames.jsonl / steps.jsonl): {rec}"); return 2
    run = a.run or (rec.parent.name if rec.name == "rec" else rec.name)
    summary = json.loads((rec / "summary.json").read_text()) if (rec / "summary.json").exists() else None

    server_path = rsync_run(rec, a.dest)
    if server_path is None:
        return 3
    link = CloudLink(a.hub, outbox=rec.parent / "outbox", timeout_s=15.0)
    if link.health() is None:
        print(f"hub not reachable at {a.hub}"); return 4
    if not link.post_run_complete(rec, summary, server_path, run=run):
        return 5
    print(f"run '{run}' handed to {a.hub} (server path {server_path})")
    if not a.wait:
        return 0
    t0 = time.time(); last = None
    while time.time() - t0 < a.timeout_s:
        st = run_status(a.hub, run)
        if st and st.get("state") != last:
            last = st.get("state"); print(f"[{int(time.time() - t0):5d} s] {last}  done={st.get('steps_done')}")
        if st and st.get("state") in ("done", "failed"):
            print(json.dumps(st, ensure_ascii=False, indent=1))
            return 0 if st.get("state") == "done" and st.get("report") else 6
        time.sleep(20)
    print("timed out waiting for the report"); return 7


if __name__ == "__main__":
    sys.exit(main())
