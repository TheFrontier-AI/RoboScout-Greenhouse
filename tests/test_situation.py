"""Event-triggered situation reports: Cosmos Reason explains a SAFETY stop, HERMES decides (no motion commands)."""
import json, threading, uuid
from pathlib import Path
from agent.brain_loop import decide_situation
from agent.robot_api import serve
from cloud.eyes.situation import parse_situation
from tests.test_robot_api import _req

MAP = Path(__file__).resolve().parents[1] / "data/tunnel09_aisle_map_v18.json"


def test_parse_situation_reply_with_thinking_and_alias():
    s = parse_situation('<think>a man</think> {"what": "person", "description": "worker centre near", '
                        '"blocking_path": true, "likely_to_move": true, "confidence": 0.9}')
    assert s["what"] == "person" and s["likely_to_move"] and s["crop_concern"] == ""
    assert parse_situation('{"what": "spaceship"}')["what"] == "other"


def test_hermes_waits_for_people_and_calls_the_farmer_for_static_blocks():
    cmd, why = decide_situation({"what": "person", "aisle": "tunnel-10/aisle-02", "metre": 12.0, "likely_to_move": True})
    assert cmd is None and "waits" in why
    cmd, _ = decide_situation({"what": "tool_or_hose", "aisle": "tunnel-09/aisle-02", "metre": 3.0, "description": "hose"})
    assert cmd["cmd"] == "notify" and "metre 3.0" in cmd["message"]
    cmd, _ = decide_situation({"what": "fallen_plant_or_branch", "crop_concern": "plant 22 fell into the aisle"})
    assert cmd["cmd"] == "notify" and "Crop issue" in cmd["message"]
    assert decide_situation({"what": "nothing_visible"})[0]["cmd"] == "notify"


def test_hub_queues_situation_frames_and_logs_farmer_alerts(tmp_path, monkeypatch):
    monkeypatch.delenv("EYES_BASE_URL", raising=False)
    srv = serve(tmp_path, MAP, port=0); base = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        b = uuid.uuid4().hex
        meta = {"frame_id": 900001, "camera": "front", "purpose": "situation", "trigger": "obstacle_stop", "metre": 4.2}
        body = (f"--{b}\r\nContent-Disposition: form-data; name=\"meta\"\r\n\r\n{json.dumps(meta)}\r\n"
                f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"f.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n"
                ).encode() + b"\xff\xd8X\xff\xd9" + f"\r\n--{b}--\r\n".encode()
        assert "situation_queued" in _req(base + "/frame", body, f"multipart/form-data; boundary={b}")[1]
        import time
        for _ in range(50):
            rows = _req(base + "/situations?since=0")[1]["situations"]
            if rows:
                break
            time.sleep(0.05)
        assert rows[0]["trigger"] == "obstacle_stop" and rows[0]["what"] == "unknown"     # no eyes configured
        assert _req(base + "/command", json.dumps({"cmd": "notify", "message": "hose in aisle"}).encode())[1]["status"] == "accepted"
        assert json.loads((tmp_path / "alerts.jsonl").read_text())["message"] == "hose in aisle"
        assert _req(base + "/next_command")[0] == 204                                      # nothing sent to the robot
    finally:
        srv.shutdown()
