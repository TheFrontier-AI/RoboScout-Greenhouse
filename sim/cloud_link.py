"""Robot-side client for the robot command hub (agent/robot_api.py): pull goals, push status and camera frames.
Stdlib HTTP only. Tolerates the hub being down: frames are queued to an on-disk outbox and retried, the latest status
is re-sent, and while the hub is marked down no network call blocks the control loop.

  GET  /next_command → 200 cmd JSON | 204      POST /status  JSON
  POST /frame        multipart: meta (JSON string) + image (jpeg)      GET /health
  POST /run_complete {"run", "path", "summary"}  survey finished → server runs the analysis pipeline on that run dir"""
from __future__ import annotations
import io
import json
import os
import struct
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import zlib
from pathlib import Path


def encode_image(rgb, quality: int = 95) -> tuple[bytes, str]:
    """HxWx3 uint8 (numpy array or nested lists) → (bytes, mime). JPEG via PIL; stdlib PNG fallback without PIL."""
    try:
        from PIL import Image
        import numpy as np
        buf = io.BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)[:, :, :3]).save(buf, "JPEG", quality=quality)
        return buf.getvalue(), "image/jpeg"
    except ImportError:
        rows = [bytes(int(c) for px in row for c in list(px)[:3]) for row in rgb]
        h, w = len(rows), len(rows[0]) // 3
        raw = b"".join(b"\x00" + r for r in rows)
        chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
        return png, "image/png"


def multipart(meta: dict, image: bytes, mime: str = "image/jpeg") -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    ext = "jpg" if mime == "image/jpeg" else "png"
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"meta\"\r\nContent-Type: application/json\r\n\r\n"
            f"{json.dumps(meta, ensure_ascii=False)}\r\n--{b}\r\nContent-Disposition: form-data; name=\"image\"; "
            f"filename=\"frame.{ext}\"\r\nContent-Type: {mime}\r\n\r\n").encode() + image + f"\r\n--{b}--\r\n".encode()
    return body, f"multipart/form-data; boundary={b}"


def rsync_dir(src: str | Path, target: str, log=print, timeout_s: float = 3600.0) -> str | None:
    """rsync a run dir to `target` ("user@host:/runs/farm/rec" or a local path). Returns the path on the server (the
    part after ':'), or None when rsync failed. Never raises: the robot must finish its shutdown either way."""
    src = str(Path(src).resolve()).rstrip("/") + "/"
    try:
        r = subprocess.run(["rsync", "-a", "--partial", src, target.rstrip("/") + "/"], capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as e:
        log(f"[cloud_link] rsync failed: {e}")
        return None
    if r.returncode != 0:
        log(f"[cloud_link] rsync exit {r.returncode}: {(r.stderr or '').strip()[-200:]}")
        return None
    return target.split(":", 1)[1] if ":" in target else target


class CloudLink:
    def __init__(self, base_url: str | None = None, outbox: str | Path = "sim_outbox", timeout_s: float = 2.0,
                 retry_s: float = 5.0, flush_batch: int = 20, log=print):
        self.base = (base_url if base_url is not None else os.environ.get("ROBOT_API_URL", "")).rstrip("/")
        self.outbox = Path(outbox)
        self.timeout_s, self.retry_s, self.flush_batch, self.log = timeout_s, retry_s, flush_batch, log
        self._down_until = 0.0
        self._pending_status: dict | None = None
        self.sent = self.queued = 0

    @property
    def enabled(self) -> bool:
        return bool(self.base)

    @property
    def up(self) -> bool:
        return self.enabled and time.monotonic() >= self._down_until

    # ---- HTTP ---------------------------------------------------------------------------------------------------
    def _request(self, method: str, path: str, body: bytes | None = None, ctype: str | None = None):
        """(status, parsed JSON | None), or None when the hub is unreachable (marks it down for retry_s)."""
        if not self.up:
            return None
        req = urllib.request.Request(self.base + path, data=body, method=method,
                                     headers={"Content-Type": ctype} if ctype else {})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as e:                       # hub answered: not "down"
            return e.code, None
        except (urllib.error.URLError, OSError, ValueError) as e:
            if time.monotonic() >= self._down_until:
                self.log(f"[cloud_link] hub unreachable ({e}); retry in {self.retry_s:.0f} s")
            self._down_until = time.monotonic() + self.retry_s
            return None

    def health(self) -> dict | None:
        r = self._request("GET", "/health")
        return r[1] if r and r[0] == 200 else None

    def next_command(self) -> dict | None:
        r = self._request("GET", "/next_command")
        return r[1] if r and r[0] == 200 and isinstance(r[1], dict) else None

    def post_status(self, status: dict) -> bool:
        self._pending_status = status                              # only the latest status matters
        r = self._request("POST", "/status", json.dumps(status, ensure_ascii=False).encode(), "application/json")
        if r and r[0] == 200:
            self._pending_status = None
            return True
        return False

    def post_run_complete(self, run_dir: str | Path, summary: dict | None = None, server_path: str | None = None,
                          run: str | None = None) -> bool:
        """Tell the hub the survey is over. `server_path` is where the run dir lives on the server (rsync target or
        shared mount); default: the same absolute path (sim: rec dir is a shared mount). `run` names the run on the
        hub (what GET /run_status?run= is polled with); default: the run dir's name (its parent's when it is `rec`)."""
        run_dir = Path(run_dir).resolve()
        name = run or (run_dir.parent.name if run_dir.name == "rec" else run_dir.name)
        body = {"run": name, "path": server_path or str(run_dir), "summary": summary, "source": "robot"}
        r = self._request("POST", "/run_complete", json.dumps(body, ensure_ascii=False, default=float).encode(), "application/json")
        if r and r[0] == 200:
            self.log(f"[cloud_link] run_complete accepted: {r[1]}")
            return True
        self.log(f"[cloud_link] run_complete not delivered ({'hub down' if r is None else r[0]})")
        return False

    def send_frame(self, meta: dict, image: bytes, mime: str = "image/jpeg") -> bool:
        """Send now if possible (after the backlog, to keep order); otherwise queue to the outbox."""
        self.flush()
        if self.up and not self._outbox_files():
            if self._post_frame(meta, image, mime):
                self.sent += 1
                return True
        self._queue(meta, image, mime)
        return False

    def _post_frame(self, meta: dict, image: bytes, mime: str) -> bool:
        body, ctype = multipart(meta, image, mime)
        r = self._request("POST", "/frame", body, ctype)
        if r is not None and r[0] == 400:                          # hub rejects this frame: drop, do not retry forever
            self.log(f"[cloud_link] hub rejected frame {meta.get('frame_id')}")
            return True
        return bool(r and r[0] == 200)

    # ---- outbox -------------------------------------------------------------------------------------------------
    def _queue(self, meta: dict, image: bytes, mime: str) -> None:
        self.outbox.mkdir(parents=True, exist_ok=True)
        stem = f"{time.time_ns():020d}_{meta.get('frame_id', 0):07d}_{meta.get('camera', 'cam')}"
        (self.outbox / f"{stem}.img").write_bytes(image)
        (self.outbox / f"{stem}.json").write_text(json.dumps({"meta": meta, "mime": mime}, ensure_ascii=False))
        self.queued += 1

    def _outbox_files(self) -> list[Path]:
        return sorted(self.outbox.glob("*.json")) if self.outbox.exists() else []

    def flush(self) -> int:
        """Retry queued frames (oldest first) and the last unsent status. Returns frames delivered."""
        if not self.up:
            return 0
        if self._pending_status is not None:
            self.post_status(self._pending_status)
        n = 0
        for js in self._outbox_files()[: self.flush_batch]:
            img = js.with_suffix(".img")
            try:
                d = json.loads(js.read_text())
                data = img.read_bytes()
            except (OSError, ValueError):
                js.unlink(missing_ok=True); img.unlink(missing_ok=True)
                continue
            if not self._post_frame(d["meta"], data, d.get("mime", "image/jpeg")):
                break
            js.unlink(missing_ok=True); img.unlink(missing_ok=True)
            n += 1
        self.sent += n
        return n

    @property
    def backlog(self) -> int:
        return len(self._outbox_files())
