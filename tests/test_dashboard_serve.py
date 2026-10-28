"""Offline tests for dashboard/serve.py: state assembly, tolerant JSONL reading and /img path safety."""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "dashboard"))
import serve  # noqa: E402
import make_demo_data  # noqa: E402

PNG_1PX = make_demo_data._png_bytes(1, 1, [b"\x10\x80\x20"])


@pytest.fixture()
def data_dir(tmp_path):
    d = tmp_path / "data"
    (d / "frames").mkdir(parents=True)
    (d / "aisle_map.json").write_text((REPO / "data/tunnel09_aisle_map_v18.json").read_text())
    (d / "status.json").write_text(json.dumps({"state": "surveying", "aisle": "tunnel-09/aisle-02", "metre": 3.5,
                                               "heading_deg": -90, "pose_confidence": 0.9, "ts": 1.0}))
    (d / "frames" / "a.png").write_bytes(PNG_1PX)
    frames = [{"ts": 1, "aisle": "tunnel-09/aisle-02", "metre": 3.5, "camera": "mast_left_mid", "frame_id": 1,
               "image_path": "frames/a.png"},
              {"ts": 2, "aisle": "tunnel-09/aisle-02", "metre": 3.5, "camera": "front", "frame_id": 2,
               "image_path": "../outside.png"},
              {"ts": 3, "aisle": "tunnel-09/aisle-02", "metre": 4.0, "camera": "front", "frame_id": 3,
               "image_path": str(d / "frames" / "a.png")}]
    (d / "frames.jsonl").write_text("\n".join(json.dumps(f) for f in frames) + "\n{\"ts\": 4, \"aisle\"")  # torn last line
    (d / "findings.jsonl").write_text(json.dumps({"frame_id": 1, "aisle": "tunnel-09/aisle-02", "metre": 3.5,
                                                  "camera": "mast_left_mid", "status": "disease", "issues": [],
                                                  "summary": "x"}) + "\n\nnot json\n")
    (d / "decisions.jsonl").write_text(json.dumps({"ts": 1, "reasoning": "go", "command": {"cmd": "revisit"},
                                                   "reply": {"status": "accepted"}}) + "\n")
    (tmp_path / "outside.png").write_bytes(PNG_1PX)
    (tmp_path / "secret.txt").write_text("secret")
    return d


@pytest.fixture()
def server(data_dir):
    srv, _ = serve.serve_in_thread(data_dir, port=0)
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def status_of(url):
    try:
        return get(url)[0]
    except urllib.error.HTTPError as e:
        return e.code


def test_build_state_assembles_all_files_and_skips_bad_lines(data_dir):
    s = serve.build_state(data_dir)
    assert s["aisle_map"]["map_id"] == "v18-tunnel-09-bed-aisles-r1"
    assert s["status"]["metre"] == 3.5
    assert [f["frame_id"] for f in s["frames"]] == [1, 2, 3]            # torn 4th line skipped
    assert s["frames"][0]["image_url"] == "/img/frames/a.png"
    assert s["frames"][1]["image_url"] is None                         # ../ escapes DIR
    assert s["frames"][2]["image_url"] == "/img/frames/a.png"          # absolute path inside DIR is relativised
    assert len(s["findings"]) == 1 and s["findings"][0]["status"] == "disease"
    assert s["decisions"][0]["command"]["cmd"] == "revisit"
    assert s["slam"]["topdown_url"] is None and s["slam"]["poses"] == []
    assert set(s["services"]) == {"nemotron", "cosmos", "slam"}
    assert s["services"]["nemotron"]["live"] and s["services"]["cosmos"]["live"] and not s["services"]["slam"]["live"]


def test_empty_dir_does_not_crash(tmp_path):
    s = serve.build_state(tmp_path)
    assert s["aisle_map"] is None and s["status"] is None and s["frames"] == [] and s["decisions"] == []


def test_services_json_overrides_inference(data_dir):
    (data_dir / "services.json").write_text(json.dumps({"slam": {"live": True, "detail": "job 42"}}))
    s = serve.build_state(data_dir)
    assert s["services"]["slam"]["live"] and s["services"]["slam"]["source"] == "declared"


def test_safe_path_rejects_traversal(data_dir):
    assert serve.safe_path(data_dir, "frames/a.png") is not None
    for bad in ("../outside.png", "frames/../../outside.png", "%2e%2e/outside.png", "..%2Foutside.png",
                "/etc/passwd", "frames/../../secret.txt", "", "frames", "a\x00b"):
        assert serve.safe_path(data_dir, bad) is None, bad


def test_safe_path_rejects_symlink_escape(data_dir):
    (data_dir / "frames" / "link.png").symlink_to(data_dir.parent / "outside.png")
    assert serve.safe_path(data_dir, "frames/link.png") is None


def test_http_api_state_and_images(server):
    code, ctype, body = get(server + "/api/state")
    assert code == 200 and ctype.startswith("application/json")
    s = json.loads(body)
    assert s["status"]["state"] == "surveying" and len(s["frames"]) == 3
    code, ctype, body = get(server + s["frames"][0]["image_url"])
    assert code == 200 and ctype == "image/png" and body == PNG_1PX
    for path in ("/img/../outside.png", "/img/%2e%2e/outside.png", "/img/frames/%2e%2e/%2e%2e/outside.png",
                 "/img/..%2f..%2fsecret.txt", "/img/aisle_map.json", "/img/frames/missing.png", "/nope",
                 "/../../etc/passwd"):
        assert status_of(server + path) == 404, path


def test_http_serves_static_page(server):
    code, ctype, body = get(server + "/")
    assert code == 200 and ctype.startswith("text/html") and b"app.js" in body
    assert get(server + "/app.js")[0] == 200 and get(server + "/style.css")[0] == 200


def test_demo_data_generator(tmp_path):
    out = tmp_path / "demo"
    make_demo_data.main(["--out", str(out)])
    s = serve.build_state(out)
    assert s["status"]["aisle"] == "tunnel-09/aisle-02"
    assert any(f["status"] == "disease" and f["metre"] == 15.0 for f in s["findings"])
    assert any(d["command"].get("cmd") == "revisit" for d in s["decisions"])
    assert all(f["image_url"] for f in s["frames"])
    side = {f["camera"] for f in s["frames"] if f["metre"] == 15.0}
    assert len(side) == 7                                              # 6 side cameras + front
    assert s["slam"]["topdown_url"] and len(s["slam"]["poses"]) > 10
