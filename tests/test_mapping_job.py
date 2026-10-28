"""Offline tests for cloud/mapping/run_job.py: argument parsing, frame sampling, PLY + top-down outputs (no GPU)."""
import importlib.util
import math
from pathlib import Path
import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "cloud/mapping/run_job.py"
_spec = importlib.util.spec_from_file_location("mapping_run_job", _SRC)
job = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(job)


def test_args_defaults_and_multiple_videos():
    a = job.parse_args(["--video", "front.mp4", "s3://b/side1.mp4", "--out", "o"])
    assert a.video == ["front.mp4", "s3://b/side1.mp4"]
    assert (a.fps, a.max_frames, a.method, a.fisheye) == (2.0, 600, "primary", False)
    assert job.parse_args(["--video", "v", "--out", "o", "--method", "fallback", "--fisheye"]).fisheye


@pytest.mark.parametrize("bad", [["--method", "colmap"], ["--fps", "0"], ["--max-frames", "0"]])
def test_args_rejects_bad_values(bad):
    with pytest.raises(SystemExit):
        job.parse_args(["--video", "v", "--out", "o", *bad])


def test_plan_fps_caps_total_frames():
    assert job.plan_fps([152.8], 2.0, 600) == 2.0                       # 306 frames fit
    r = job.plan_fps([300.0] * 7, 2.0, 600)                              # 7 cameras x 5 min
    assert r < 2.0 and math.isclose(r * 2100, 600)
    assert job.plan_fps([], 2.0, 600) == 2.0


def test_uniform_subsample():
    assert job.uniform_subsample(5, 10) == [0, 1, 2, 3, 4]
    idx = job.uniform_subsample(1000, 600)
    assert len(idx) == 600 and idx[0] == 0 and idx[-1] == 999 and idx == sorted(set(idx))
    assert job.uniform_subsample(10, 1) == [0] and job.uniform_subsample(10, 0) == []


def test_fisheye_filter_and_intrinsics():
    f = job.fisheye_filter(175, 100, 960, 540)
    assert f.startswith("v360=input=fisheye:output=flat") and "ih_fov=175" in f and "h_fov=100" in f
    K = job.pinhole_intrinsics(90, 1000, 500)
    assert math.isclose(K[0][0], 500.0) and K[0][2] == 500 and K[1][2] == 250
    a = job.parse_args(["--video", "v", "--out", "o", "--fps", "1", "--fisheye", "--width", "640"])
    chain = job.frame_filter(0.5, a, 960, 540).split(",")
    assert chain[0] == "fps=0.500000" and chain[1].startswith("v360") and chain[2] == "scale=640:-2"


def test_write_ply_roundtrip(tmp_path):
    pts = np.array([[0, 0, 0], [1, 2, 3]], np.float32)
    cols = np.array([[255, 0, 0], [0, 255, 0]], np.uint8)
    job.write_ply(tmp_path / "p.ply", pts, cols)
    raw = (tmp_path / "p.ply").read_bytes()
    head, body = raw.split(b"end_header\n")
    assert b"element vertex 2" in head and len(body) == 2 * 15
    assert np.frombuffer(body[15:27], "<f4").tolist() == [1, 2, 3]


def test_topdown_flat_floor_with_box():
    # OpenCV cameras in a z-up world looking along +x: columns = cam x (-y), cam y (-z), cam z (+x).
    R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    poses = np.stack([np.eye(4)] * 3)
    for i, T in enumerate(poses):
        T[:3, :3] = R
        T[:3, 3] = [i, 0, 0.3]
    up = job.up_vector(poses, "cam-y")
    assert np.allclose(up, [0, 0, 1])
    g = np.mgrid[0:4:0.05, -2:2:0.05].reshape(2, -1).T
    floor = np.c_[g, np.zeros(len(g))]
    box = np.c_[np.random.default_rng(0).uniform([2, 1], [2.5, 1.5], (500, 2)), np.full(500, 1.0)]
    hmap, occ, cpx, meta = job.topdown(np.r_[floor, box], poses, up, grid_px=200)
    assert occ.max() == 255 and (occ == 128).sum() > (occ == 255).sum()
    assert meta["floor_h"] == pytest.approx(0, abs=1e-6) and 0 < meta["obstacle_h"] < 1
    assert cpx.shape == (3, 2) and np.all(cpx >= 0)


def test_filter_points_confidence_and_cap():
    pts = np.arange(300, dtype=np.float32).reshape(100, 3)
    cols = np.zeros((100, 3), np.uint8)
    p, _ = job.filter_points(pts, cols, np.arange(100, dtype=np.float32), 50, 10_000)
    assert len(p) == 50 and p[:, 0].min() >= 150
    p, _ = job.filter_points(pts, cols, np.ones(100, np.float32), 0, 7)
    assert len(p) == 7
