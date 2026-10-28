import json, threading, urllib.request, uuid
from pathlib import Path
from agent.robot_api import serve

MAP = Path(__file__).resolve().parents[1] / "data/tunnel09_aisle_map_v18.json"


def _req(url, data=None, ctype="application/json"):
    r = urllib.request.Request(url, data=data, headers={"Content-Type": ctype} if data is not None else {})
    with urllib.request.urlopen(r, timeout=5) as resp:
        return resp.status, (json.loads(resp.read() or b"null") if resp.status != 204 else None)


def test_command_flow_status_and_frame(tmp_path, monkeypatch):
    monkeypatch.delenv("EYES_BASE_URL", raising=False)
    srv = serve(tmp_path, MAP, port=0); port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        assert _req(base + "/health")[1]["map"] == "tunnel-09"
        assert _req(base + "/next_command")[0] == 204
        ok = _req(base + "/command", json.dumps({"cmd": "revisit", "aisle": "tunnel-09/aisle-02", "metre": 15,
                                                 "reasoning": "disease at 15 m"}).encode())[1]
        bad = _req(base + "/command", json.dumps({"cmd": "go_to", "aisle": "tunnel-09/aisle-01", "metre": 2}).encode())[1]
        assert ok["status"] == "accepted" and bad["status"] == "rejected"
        assert _req(base + "/next_command")[1]["cmd"] == "revisit"
        _req(base + "/status", json.dumps({"state": "running", "aisle": "tunnel-09/aisle-02", "metre": 3.0}).encode())
        assert _req(base + "/status")[1]["metre"] == 3.0
        b = uuid.uuid4().hex
        meta = {"frame_id": 1, "camera": "mast_left_mid", "aisle": "tunnel-09/aisle-02", "metre": 3.0, "pose_confidence": 0.9}
        body = (f"--{b}\r\nContent-Disposition: form-data; name=\"meta\"\r\n\r\n{json.dumps(meta)}\r\n"
                f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"f.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n"
                ).encode() + b"\xff\xd8JPEGDATA\xff\xd9" + f"\r\n--{b}--\r\n".encode()
        res = _req(base + "/frame", body, f"multipart/form-data; boundary={b}")[1]
        assert (tmp_path / res["stored"]).read_bytes() == b"\xff\xd8JPEGDATA\xff\xd9"
        decisions = [json.loads(l) for l in (tmp_path / "decisions.jsonl").read_text().splitlines()]
        assert [d["reply"] for d in decisions] == ["accepted", "rejected"] and decisions[0]["reasoning"] == "disease at 15 m"
        assert json.loads((tmp_path / "frames.jsonl").read_text())["image_path"] == res["stored"]
    finally:
        srv.shutdown()
