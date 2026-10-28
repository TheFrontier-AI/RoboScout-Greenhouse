import json, math
import numpy as np
from cloud.mapping.align_to_house import umeyama_2d, align


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 2))
    th, s, t = math.radians(37), 2.5, np.array([4.0, -1.0])
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    s2, R2, t2 = umeyama_2d(src, (s * (R @ src.T)).T + t)
    assert abs(s2 - s) < 1e-9 and np.allclose(R2, R) and np.allclose(t2, t)


def test_align_maps_slam_track_onto_aisle(tmp_path):
    # SLAM world: camera drives along +z (OpenCV forward), y down; ground plane basis e1=x, e2=z
    n = 30
    frames = [{"t_sec": k * 0.5, "image": f"{k}.jpg", "position": [0.0, 0.0, 0.1 * k]} for k in range(n)]
    (tmp_path / "trajectory.json").write_text(json.dumps({"frames": frames}))
    (tmp_path / "summary.json").write_text(json.dumps({"topdown": {"e1": [1, 0, 0], "e2": [0, 0, 1], "origin_uv": [-1, -1],
                                                                   "resolution_per_px": 0.05, "size_px": [40, 80]}}))
    refs = [{"t_sec": k * 0.5, "x": 0.0, "y": 19.0 - 0.25 * k} for k in range(n)]   # aisle-02: door +Y, 0.5 m/s
    r = align(tmp_path, refs, tmp_path / "out")
    assert r["alignment"]["rmse_m"] < 1e-6 and abs(r["alignment"]["scale"] - 2.5) < 1e-6
    assert abs(r["poses"][0]["y"] - 19.0) < 1e-6 and abs(r["poses"][-1]["heading_deg"] + 90) < 1e-6


def test_topdown_is_warped_into_house_bounds(tmp_path):
    cv2 = __import__("pytest").importorskip("cv2")
    n = 30
    frames = [{"t_sec": k * 0.5, "position": [0.0, 0.0, 0.1 * k]} for k in range(n)]
    (tmp_path / "trajectory.json").write_text(json.dumps({"frames": frames}))
    (tmp_path / "summary.json").write_text(json.dumps({"topdown": {"e1": [1, 0, 0], "e2": [0, 0, 1], "origin_uv": [-1, -1],
                                                                   "resolution_per_px": 0.05, "size_px": [40, 80]}}))
    img = np.zeros((80, 40), np.uint8); img[0:5, :] = 255          # bright band at max e2 = far end of the drive
    cv2.imwrite(str(tmp_path / "topdown.png"), img)
    refs = [{"t_sec": k * 0.5, "x": 0.0, "y": 19.0 - 0.25 * k} for k in range(n)]
    r = align(tmp_path, refs, tmp_path / "out")
    out = cv2.imread(str(tmp_path / "out" / "topdown.png"), cv2.IMREAD_GRAYSCALE)
    lo, hi = r["bounds_local"]["min"], r["bounds_local"]["max"]
    rows = np.where(out.max(1) > 128)[0]                            # far end (low y) must land at the bottom rows
    y_of_row = hi[1] - rows.mean() * (hi[1] - lo[1]) / (out.shape[0] - 1)
    assert y_of_row < 19.0 - 0.25 * (n - 1) + 0.5
