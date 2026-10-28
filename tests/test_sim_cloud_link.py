"""CloudLink against a stub hub: commands, status, multipart frames, and the on-disk outbox when the hub is down."""
import json
import threading
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from sim.cloud_link import CloudLink, encode_image, multipart


class Stub:
    def __init__(self):
        self.cmds, self.status, self.frames = [{"cmd": "go_to", "aisle": "tunnel-09/aisle-02", "metre": 5}], [], []
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _ok(self, obj, code=200):
                b = json.dumps(obj).encode()
                self.send_response(code); self.send_header("Content-Length", str(len(b))); self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path == "/health":
                    return self._ok({"ok": True})
                if stub.cmds:
                    return self._ok(stub.cmds.pop(0))
                self.send_response(204); self.end_headers()

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if self.path == "/status":
                    stub.status.append(json.loads(body))
                else:
                    msg = BytesParser(policy=HTTP).parsebytes(
                        b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
                    parts = {p.get_param("name", header="content-disposition"): p.get_payload(decode=True)
                             for p in msg.iter_parts()}
                    stub.frames.append((json.loads(parts["meta"]), parts["image"]))
                self._ok({"ok": True})

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"


def test_encode_image_and_multipart():
    data, mime = encode_image([[[255, 0, 0]] * 8] * 4)
    assert mime in ("image/jpeg", "image/png") and len(data) > 20
    body, ctype = multipart({"frame_id": 1}, data, mime)
    assert ctype.startswith("multipart/form-data; boundary=") and b'name="meta"' in body and data in body


def test_commands_status_frames_roundtrip(tmp_path):
    s = Stub()
    link = CloudLink(s.url, outbox=tmp_path / "out", log=lambda m: None)
    assert link.health() == {"ok": True}
    assert link.next_command()["cmd"] == "go_to" and link.next_command() is None
    assert link.post_status({"state": "drive_in", "metre": 1.0}) and s.status[-1]["state"] == "drive_in"
    assert link.send_frame({"frame_id": 7, "camera": "front"}, b"\xff\xd8jpeg")
    assert s.frames[-1] == ({"frame_id": 7, "camera": "front"}, b"\xff\xd8jpeg")
    s.srv.shutdown()


def test_hub_down_queues_to_outbox_then_flushes_in_order(tmp_path):
    down = CloudLink("http://127.0.0.1:9", outbox=tmp_path / "out", timeout_s=0.5, retry_s=60, log=lambda m: None)
    assert down.next_command() is None and not down.up              # marked down, no further blocking calls
    for i in (1, 2, 3):
        assert not down.send_frame({"frame_id": i, "camera": "front"}, bytes([i]))
    assert down.backlog == 3 and not down.post_status({"state": "x"})
    s = Stub()
    link = CloudLink(s.url, outbox=tmp_path / "out", log=lambda m: None)
    assert link.flush() == 3 and link.backlog == 0
    assert [m["frame_id"] for m, _ in s.frames] == [1, 2, 3]
    s.srv.shutdown()


def test_disabled_link_is_noop(tmp_path):
    link = CloudLink("", outbox=tmp_path / "out")
    assert not link.enabled and link.next_command() is None and link.flush() == 0
