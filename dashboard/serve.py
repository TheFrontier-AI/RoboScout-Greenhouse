#!/usr/bin/env python3
"""Tiny stdlib server for the R1 greenhouse live dashboard.

    python3 dashboard/serve.py --data DIR --port 8080

Routes
  GET /                -> dashboard/index.html (+ /app.js, /style.css)
  GET /api/state       -> one JSON snapshot assembled from the files the system writes into DIR
  GET /img/<rel path>  -> an image file under DIR (path traversal / symlink escape refused)

Files read from DIR (all optional; missing or half-written files never crash the server):
  aisle_map.json            same schema as data/tunnel09_aisle_map_v18.json
  status.json               {state, cmd, aisle, metre, heading_deg, pose_confidence, battery, ts, legs_remaining}
  frames.jsonl              one FrameMeta per line (+ image_path, relative to DIR or absolute inside DIR)
  findings.jsonl            one inspection result per line
  decisions.jsonl           HERMES decisions {ts, reasoning, command, reply}
  services.json             optional {"nemotron": {"live": true, "detail": "..."}, "cosmos": {...}, "slam": {...}}
  slam/topdown.png          optional SLAM top-down render
  slam/trajectory.json      optional {"bounds_local": {"min": [x, y], "max": [x, y]}, "poses": [{ts, x, y, heading_deg}]}
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import unquote, urlsplit

STATIC_DIR = Path(__file__).resolve().parent
STATIC_FILES = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".ppm"}
MAX_JSONL_ROWS = 6000          # keep /api/state bounded even after a long run
LIVE_WINDOW_S = 120.0          # a service counts as "live" if it produced output this recently


# ---------------------------------------------------------------- file readers (tolerant of partial writes)

def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_jsonl(path: Path, limit: int = MAX_JSONL_ROWS) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue           # the writer may be mid-line; skip it, we'll get it next poll
        if isinstance(row, dict):
            rows.append(row)
    return rows


def safe_path(data_dir: Path, rel: str) -> Path | None:
    """Resolve `rel` under data_dir; None if it escapes (.., absolute, symlink out) or is not a file."""
    rel = unquote(rel).replace("\\", "/")
    if not rel or "\x00" in rel:
        return None
    root = data_dir.resolve()
    try:
        candidate = (root / rel.lstrip("/")).resolve()
        candidate.relative_to(root)
    except (ValueError, OSError):
        return None
    return candidate if candidate.is_file() else None


def image_url(data_dir: Path, image_path) -> str | None:
    """Turn a frame's image_path (relative to DIR, or absolute inside DIR) into a /img/ URL."""
    if not image_path or not isinstance(image_path, str):
        return None
    p = Path(image_path)
    root = data_dir.resolve()
    if p.is_absolute():
        try:
            rel = p.resolve().relative_to(root).as_posix()
        except (ValueError, OSError):
            return None
    else:
        rel = p.as_posix()
    return "/img/" + rel if safe_path(data_dir, rel) else None


# ---------------------------------------------------------------- state assembly

def _services(data_dir: Path, decisions: list[dict], findings: list[dict], slam: dict, now: float) -> dict:
    """Which Nebius / NVIDIA services are live. services.json wins; otherwise inferred from recent output."""
    declared = _read_json(data_dir / "services.json") or {}

    def recent(path: Path) -> bool:
        try:
            return now - path.stat().st_mtime < LIVE_WINDOW_S
        except OSError:
            return False

    inferred = {
        "nemotron": {"name": "Nemotron", "where": "Nebius Token Factory",
                     "live": bool(decisions) and recent(data_dir / "decisions.jsonl"),
                     "detail": f"{len(decisions)} decisions"},
        "cosmos": {"name": "Cosmos Reason", "where": "Nebius Serverless Endpoint",
                   "live": bool(findings) and recent(data_dir / "findings.jsonl"),
                   "detail": f"{len(findings)} frames inspected"},
        "slam": {"name": "SLAM job", "where": "Nebius GPU job",
                 "live": slam.get("topdown_url") is not None or bool(slam.get("poses")),
                 "detail": f"{len(slam.get('poses') or [])} poses"},
    }
    out = {}
    for key, base in inferred.items():
        d = declared.get(key) if isinstance(declared, dict) else None
        merged = dict(base, source="inferred")
        if isinstance(d, dict):
            merged.update({k: v for k, v in d.items() if k in ("name", "where", "live", "detail", "model")})
            merged["source"] = "declared"
        merged["live"] = bool(merged.get("live"))
        out[key] = merged
    return out


def build_state(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    now = time.time()
    frames = _read_jsonl(data_dir / "frames.jsonl")
    for f in frames:
        f["image_url"] = image_url(data_dir, f.get("image_path"))
    findings = _read_jsonl(data_dir / "findings.jsonl")
    decisions = _read_jsonl(data_dir / "decisions.jsonl", limit=500)

    traj = _read_json(data_dir / "slam" / "trajectory.json")
    traj = traj if isinstance(traj, dict) else ({"poses": traj} if isinstance(traj, list) else {})
    slam = {
        "topdown_url": "/img/slam/topdown.png" if safe_path(data_dir, "slam/topdown.png") else None,
        "bounds_local": traj.get("bounds_local") or traj.get("bounds"),
        "poses": traj.get("poses") or traj.get("points") or [],
    }
    try:
        slam["topdown_mtime"] = (data_dir / "slam" / "topdown.png").stat().st_mtime
    except OSError:
        slam["topdown_mtime"] = None

    return {
        "server_ts": now,
        "data_dir": str(data_dir),
        "aisle_map": _read_json(data_dir / "aisle_map.json"),
        "status": _read_json(data_dir / "status.json"),
        "frames": frames,
        "findings": findings,
        "decisions": decisions,
        "slam": slam,
        "services": _services(data_dir, decisions, findings, slam, now),
    }


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    data_dir: Path = Path(".")
    server_version = "R1Dashboard/1.0"

    def log_message(self, fmt, *args):     # quiet: the page polls every second
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _error(self, code: int, msg: str) -> None:
        self._send(code, json.dumps({"error": msg}).encode(), "application/json")

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/state":
            body = json.dumps(build_state(self.data_dir), separators=(",", ":")).encode()
            return self._send(200, body, "application/json; charset=utf-8")
        if path.startswith("/img/"):
            target = safe_path(self.data_dir, path[len("/img/"):])
            if target is None or target.suffix.lower() not in IMAGE_EXTS:
                return self._error(HTTPStatus.NOT_FOUND, "not found")
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            try:
                return self._send(200, target.read_bytes(), ctype)
            except OSError:
                return self._error(HTTPStatus.NOT_FOUND, "not found")
        if path in STATIC_FILES:
            f = STATIC_DIR / STATIC_FILES[path]
            ctype = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}[f.suffix] + "; charset=utf-8"
            return self._send(200, f.read_bytes(), ctype)
        return self._error(HTTPStatus.NOT_FOUND, "not found")


def make_server(data_dir: str | Path, port: int = 8080, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"data_dir": Path(data_dir)})
    srv = ThreadingHTTPServer((host, port), handler)
    srv.daemon_threads = True
    return srv


def serve_in_thread(data_dir: str | Path, port: int = 0, host: str = "127.0.0.1") -> tuple[ThreadingHTTPServer, Thread]:
    """Start a server on a background thread (port 0 = pick a free one). Used by tests."""
    srv = make_server(data_dir, port, host)
    t = Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, t


def main() -> None:
    ap = argparse.ArgumentParser(description="R1 greenhouse live dashboard server")
    ap.add_argument("--data", required=True, help="directory the robot / cloud pipeline writes into")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to view from a phone on the same network")
    args = ap.parse_args()
    data_dir = Path(args.data)
    if not data_dir.is_dir():
        raise SystemExit(f"--data {data_dir} is not a directory")
    srv = make_server(data_dir, args.port, args.host)
    print(f"R1 dashboard: http://{args.host}:{srv.server_address[1]}/  (data: {data_dir.resolve()})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
