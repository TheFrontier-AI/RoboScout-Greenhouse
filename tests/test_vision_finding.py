"""Offline checks for the detector's finding builder (no model, no GPU)."""
import numpy as np
from PIL import Image

from vision.detector import owned_strip, to_finding


def img(colour=(200, 30, 30)):
    return Image.fromarray(np.full((1024, 576, 3), colour, np.uint8))


def det(label, cx, cy, s=0.04, score=0.9):
    return {"label": label, "score": score, "bbox_norm": [cx - s, cy - s, cx + s, cy + s]}


def test_strip_is_half_metre_of_row():
    s0, s1 = owned_strip(0.95, 40.0)
    width = 2 * 0.95 * np.tan(np.radians(20))
    assert abs((s1 - s0) * width - 0.5) < 1e-6


def test_grades_and_counts_only_owned_strip():
    dets = [det("fruit_ripe", 0.5, 0.5), det("fruit_ripe", 0.05, 0.5), det("fruit_lesion", 0.5, 0.45),
            det("fruit_green", 0.52, 0.55)]
    f = to_finding(dets, img(), {"metre": 10.0, "camera": "mast_right_mid"})
    assert f["harvest"]["pass"] == 1 and f["harvest"]["reject"] == 1 and f["harvest"]["not_ready"] == 1
    assert f["status"] == "disease" and f["issues"][0]["label"] == "fruit lesion"
    assert f["source"] == "rtdetr"


def test_height_band_prevents_double_count():
    # a fruit at the top of the low camera (z ~ 1.3 m) belongs to the mid camera's band
    f = to_finding([det("fruit_ripe", 0.5, 0.1)], img(), {"metre": 3.0, "camera": "mast_left_low"})
    assert f["harvest"]["pass"] == 0
    f = to_finding([det("fruit_ripe", 0.5, 0.5)], img(), {"metre": 3.0, "camera": "mast_left_low"})
    assert f["harvest"]["pass"] == 1


def test_left_camera_metre_direction_and_plants():
    f = to_finding([det("plant_base", 0.6, 0.85)], img(), {"metre": 5.0, "camera": "mast_left_low"})
    assert len(f["plants"]) == 1 and f["plants"][0]["metre"] < 5.0
    f = to_finding([det("plant_base", 0.6, 0.85)], img(), {"metre": 5.0, "camera": "mast_right_low"})
    assert f["plants"][0]["metre"] > 5.0


def test_hue_disagreement_is_reported():
    f = to_finding([det("fruit_ripe", 0.5, 0.5)], img((40, 160, 40)), {"metre": 1.0, "camera": "mast_right_mid"})
    assert f["harvest"]["colour_disagreements"] == 1
