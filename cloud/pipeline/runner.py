"""Runs the pipeline steps on one run dir, or watches a runs dir for finished robot runs and processes them one at a
time. Every step is isolated: a failing (optional) step is recorded in <run_dir>/analysis/<step>.error and the rest
still run; a finished step leaves <step>.done (status ok|skipped) and is not repeated unless --force. The whole run
ends with analysis/pipeline.done; analysis/pipeline.log keeps the log; analysis/pipeline.lock stops two runners
(hub PIPELINE_AUTO + watcher) from working on the same run.

  python3 -m cloud.pipeline run <run_dir> [--force] [--steps detect,report]
  python3 -m cloud.pipeline watch --runs-dir <dir> [--poll 30] [--once]

Exit codes: 0 all steps ok/skipped, 1 some step failed, 3 run dir unusable, 4 already being processed."""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

from cloud.pipeline.config import ALL_STEPS, PipelineConfig
from cloud.pipeline.steps import STEPS, Skip

LOG = logging.getLogger("pipeline")


def _setup_log(run_dir: Path) -> logging.Handler:
    (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(run_dir / "analysis" / "pipeline.log")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(h); LOG.setLevel(logging.INFO)
    if not any(isinstance(x, logging.StreamHandler) and not isinstance(x, logging.FileHandler) for x in LOG.handlers):
        s = logging.StreamHandler(sys.stderr); s.setFormatter(logging.Formatter("[pipeline %(asctime)s] %(message)s", "%H:%M:%S"))
        LOG.addHandler(s)
    return h


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_step(name: str, run_dir: Path, cfg: PipelineConfig, force: bool = False) -> dict:
    """Run one step with its .done/.error marker; returns the marker contents."""
    a = run_dir / "analysis"; done, err = a / f"{name}.done", a / f"{name}.error"
    if done.exists() and not force:
        m = json.loads(done.read_text()); LOG.info("%s: already %s (%s)", name, m.get("status"), done.name)
        return m | {"cached": True}
    t0 = time.time(); started = time.strftime("%Y-%m-%d %H:%M:%S")
    LOG.info("%s: start", name)
    try:
        result = STEPS[name](run_dir, cfg); status, reason = "ok", None
    except Skip as e:
        result, status, reason = None, "skipped", str(e)
    except Exception as e:
        result, status, reason = None, "error", f"{type(e).__name__}: {e}"
        LOG.error("%s: FAILED\n%s", name, traceback.format_exc())
    m = {"step": name, "status": status, "reason": reason, "result": result, "started": started,
         "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1)}
    if status == "error":
        err.write_text(json.dumps(m, indent=1, ensure_ascii=False)); done.unlink(missing_ok=True)
    else:
        done.write_text(json.dumps(m, indent=1, ensure_ascii=False)); err.unlink(missing_ok=True)
    LOG.info("%s: %s%s (%.1fs)", name, status, f" - {reason}" if reason else "", m["seconds"])
    return m


def run(run_dir: Path, cfg: PipelineConfig | None = None, force: bool = False, steps: tuple[str, ...] | None = None) -> dict:
    """All enabled steps in order. Returns the pipeline.done contents (with 'exit_code')."""
    cfg = cfg or PipelineConfig.from_env()
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir() or not any((run_dir / f).exists() for f in ("frames.jsonl", "steps.jsonl")):
        return {"run": str(run_dir), "ok": False, "exit_code": 3, "error": "not a run dir (needs frames.jsonl or steps.jsonl)"}
    a = run_dir / "analysis"; a.mkdir(exist_ok=True)
    lock = a / "pipeline.lock"
    if lock.exists():
        try:
            other = json.loads(lock.read_text())
        except ValueError:
            other = {}
        if other.get("pid") and _pid_alive(int(other["pid"])) and other["pid"] != os.getpid():
            return {"run": str(run_dir), "ok": False, "exit_code": 4, "error": f"already being processed by pid {other['pid']}"}
    lock.write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))
    h = _setup_log(run_dir)
    order = [s for s in (steps or cfg.steps) if s in ALL_STEPS]
    LOG.info("run %s steps=%s force=%s cfg=%s", run_dir, order, force, json.dumps(cfg.public()))
    t0 = time.time(); results = {}
    try:
        for name in ALL_STEPS:                              # canonical order whatever the config order
            if name in order:
                results[name] = run_step(name, run_dir, cfg, force)
    finally:
        lock.unlink(missing_ok=True)
    statuses = {k: v["status"] for k, v in results.items()}
    ok = all(s != "error" for s in statuses.values())
    done = {"run": run_dir.parent.name if run_dir.name == "rec" else run_dir.name, "run_dir": str(run_dir), "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1), "steps": statuses, "ok": ok,
            "exit_code": 0 if ok else 1, "report": "analysis/report.json" if (a / "report.json").exists() else None}
    (a / "pipeline.done").write_text(json.dumps(done, indent=1))
    LOG.info("done %s ok=%s %s", run_dir.name, ok, statuses)
    LOG.removeHandler(h); h.close()
    return done


def pending_runs(runs_dir: Path) -> list[Path]:
    """Run dirs under runs_dir (depth 1-2, e.g. <run>/ or <run>/rec/) with run_complete.json and no pipeline.done."""
    runs_dir = Path(runs_dir)
    markers = list(runs_dir.glob("*/run_complete.json")) + list(runs_dir.glob("*/*/run_complete.json"))
    out = []
    for m in sorted(markers, key=lambda p: p.stat().st_mtime):
        d = m.parent
        if (d / "analysis" / "pipeline.done").exists():
            continue
        lock = d / "analysis" / "pipeline.lock"
        if lock.exists():
            try:
                if _pid_alive(int(json.loads(lock.read_text()).get("pid", 0))):
                    continue
            except (ValueError, TypeError):
                pass
        out.append(d)
    return out


def watch(runs_dir: Path, cfg: PipelineConfig | None = None, poll_s: float = 30.0, once: bool = False) -> int:
    cfg = cfg or PipelineConfig.from_env()
    n = 0
    print(f"[pipeline] watching {runs_dir} every {poll_s:.0f}s (run_complete.json without analysis/pipeline.done)", flush=True)
    while True:
        for d in pending_runs(runs_dir):
            print(f"[pipeline] processing {d}", flush=True)
            r = run(d, cfg); n += 1
            print(f"[pipeline] {d.name}: exit {r.get('exit_code')} {r.get('steps') or r.get('error')}", flush=True)
        if once:
            return n
        time.sleep(poll_s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("run_dir", type=Path)
    r.add_argument("--force", action="store_true", help="rerun steps that already have a .done marker")
    r.add_argument("--steps", default=None, help="comma list; default PIPELINE_STEPS or all")
    w = sub.add_parser("watch"); w.add_argument("--runs-dir", type=Path, required=True)
    w.add_argument("--poll", type=float, default=30.0); w.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    cfg = PipelineConfig.from_env()
    if a.cmd == "run":
        res = run(a.run_dir, cfg, a.force, tuple(a.steps.split(",")) if a.steps else None)
        print(json.dumps({k: v for k, v in res.items() if k != "run_dir"}, ensure_ascii=False))
        return int(res.get("exit_code", 1))
    watch(a.runs_dir, cfg, a.poll, a.once)
    return 0
