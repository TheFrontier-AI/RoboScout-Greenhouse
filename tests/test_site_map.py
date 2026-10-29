import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from r1_nav.aisle_map import AisleMap
from r1_nav.commands import plan_command
from r1_nav.site_map import NoRoute, SiteMap

ROOT = Path(__file__).resolve().parents[1]
MAP = AisleMap.load(ROOT / "data/tunnel09_aisle_map_v18.json")
GLB = ROOT.parent / "robot_design/r1_v3/farm_integration/latest_delivery/prototype/assets/farm-realistic-v18.glb"


def farm(confirmed=True) -> SiteMap:
    """dock — A door ... two ways to B: short (j1–j2) and long (j1–j3–j2); C behind a narrow gate."""
    d = {"metadata": {"source": "v18", "scale_known": True},
         "nodes": [{"id": "dock", "kind": "dock", "xy": [0, 0]},
                   {"id": "A/door", "kind": "door", "house": "tunnel-09", "xy": [0, 2]},
                   {"id": "B/door", "kind": "door", "house": "B", "xy": [20, 2]},
                   {"id": "C/door", "kind": "door", "house": "C", "xy": [0, 30]},
                   {"id": "j1", "kind": "junction", "xy": [0, 5]}, {"id": "j2", "kind": "junction", "xy": [20, 5]},
                   {"id": "j3", "kind": "junction", "xy": [10, 15]}],
         "edges": [{"a": "dock", "b": "j1"}, {"a": "A/door", "b": "j1"}, {"a": "j1", "b": "j2", "notes": ["wet after watering"]},
                   {"a": "j1", "b": "j3"}, {"a": "j3", "b": "j2"}, {"a": "j2", "b": "B/door"},
                   {"a": "j1", "b": "C/door", "width_m": 0.5}]}
    m = SiteMap.from_dict(d)
    return m.confirm("farmer") if confirmed else m


def test_route_closed_edge_reroutes_and_bumps_revision(tmp_path):
    m = farm()
    r = m.route("A/door", "B/door")
    assert r.nodes == ["A/door", "j1", "j2", "B/door"] and r.length_m == pytest.approx(26.0)
    m2 = m.patch({"op": "close_edge", "a": "j1", "b": "j2", "reason": "truck parked"})
    assert m2.revision == m.revision + 1 and not m2.confirmed and m.confirmed
    assert m2.route("A/door", "B/door").nodes == ["A/door", "j1", "j3", "j2", "B/door"]
    m3 = m2.patch({"op": "close_edge", "a": "j3", "b": "j2"})
    with pytest.raises(NoRoute, match="closed"):
        m3.route("A/door", "B/door")
    m3.save(tmp_path / "s.json")
    back = SiteMap.load(tmp_path / "s.json")
    assert back.revision == m3.revision and back.find_edge("j1", "j2").blocked and back.meta["history"]


def test_width_filter_validate_and_merge():
    m = farm()
    with pytest.raises(NoRoute, match="narrower"):
        m.route("dock", "C/door")                          # 0.5 m < 0.528 + 2*0.06
    assert m.route("dock", "C/door", robot_width_m=0.3).nodes == ["dock", "j1", "C/door"]
    assert m.door_of("B") == "B/door" and m.houses() == ["B", "C", "tunnel-09"]
    bad = SiteMap.from_dict({"metadata": {"source": "robot"}, "nodes": [
        {"id": "a", "kind": "dock", "xy": [0, 0]}, {"id": "a", "kind": "door", "xy": [1, 1]},
        {"id": "lonely", "kind": "waypoint", "xy": [9, 9]}], "edges": [{"a": "a", "b": "ghost"}]})
    w = " | ".join(bad.validate())
    assert "duplicate node id" in w and "unknown node 'ghost'" in w and "disconnected" in w
    other = SiteMap.from_dict({"metadata": {"source": "robot", "scale_known": True},
                               "nodes": [{"id": "j9", "kind": "waypoint", "xy": [5, 5]}],
                               "edges": [{"a": "j1", "b": "j2", "notes": ["ruts"], "width_m": 2.0}, {"a": "j1", "b": "j9"}]})
    mm = m.merge(other)
    e = mm.find_edge("j1", "j2")
    assert e.notes == ["wet after watering", "ruts"] and e.width_m == 2.0 and "j9" in mm.nodes and not mm.confirmed
    noted = m.patch({"op": "add_note", "a": "j1", "b": "j3", "note": "do not pass 7-9am"})
    assert noted.find_edge("j1", "j3").notes == ["do not pass 7-9am"]


def test_go_to_house_and_cross_house_survey():
    m = farm()
    r = plan_command({"cmd": "go_to_house", "house": "B"}, MAP, site=m)
    assert r.status == "accepted", r.reason
    assert [l.kind for l in r.legs] == ["exit_house", "site_transit", "site_transit", "site_transit", "enter_house"]
    assert r.legs[1].from_node == "A/door" and r.legs[2].length_m == 20.0 and "wet after watering" in r.legs[2].note
    assert "wet after watering" in r.reason and r.to_dict()["legs"][2]["to_node"] == "j2"
    # B has no aisle map → discover; tunnel-09 known → its own legs then the route back to the dock
    r = plan_command({"cmd": "survey_house", "house": "B"}, MAP, site=m)
    assert r.status == "accepted" and r.legs[-1].kind == "discover"
    r = plan_command({"cmd": "survey_house", "house": "tunnel-09"}, None, site=m, aisle_maps=[MAP], current_house="B")
    kinds = [l.kind for l in r.legs]
    assert r.status == "accepted" and kinds[0] == "exit_house" and "drive_in" in kinds and kinds[-1] == "dock"
    assert kinds.index("enter_house") < kinds.index("drive_in") and "skipping too-narrow aisles" in r.reason
    # from the dock when no house is known
    r = plan_command({"cmd": "go_to_house", "house": "B"}, None, site=m)
    assert r.status == "accepted" and r.legs[0].kind == "site_transit" and r.legs[0].from_node == "dock"
    assert plan_command({"cmd": "go_to_house", "house": "tunnel-09"}, MAP, site=m).legs == []


def test_cross_house_rejections(monkeypatch):
    monkeypatch.delenv("R1_SITE_MAP", raising=False)
    m = farm()
    rej = lambda cmd, **kw: plan_command(cmd, MAP, **kw)
    assert "no site map" in rej({"cmd": "go_to_house", "house": "B"}).reason
    assert "not confirmed" in rej({"cmd": "survey_house", "house": "B"}, site=farm(confirmed=False)).reason
    assert "unknown house" in rej({"cmd": "go_to_house", "house": "Z"}, site=m).reason
    assert "narrower" in rej({"cmd": "go_to_house", "house": "C"}, site=m).reason
    closed = m.patch([{"op": "close_edge", "a": "j1", "b": "j2"}, {"op": "close_edge", "a": "j1", "b": "j3"}]).confirm("farmer")
    assert "closed" in rej({"cmd": "go_to_house", "house": "B"}, site=closed).reason
    assert "source 'drone'" in rej({"cmd": "go_to_house", "house": "B", "map_source": "drone"}, site=m).reason
    pixels = SiteMap.from_dict({**json.loads(json.dumps(m.to_dict())), "metadata": {**m.meta, "scale_known": False}})
    assert "no metric scale" in rej({"cmd": "go_to_house", "house": "B"}, site=pixels).reason
    # same-house commands keep the old behaviour; other-house go_to is still rejected
    assert rej({"cmd": "survey_house", "house": "tunnel-09"}, site=m).legs[0].kind == "transit"
    assert rej({"cmd": "go_to", "house": "B", "aisle": "tunnel-09/aisle-02", "metre": 3}, site=m).status == "rejected"


def test_site_map_from_env(monkeypatch, tmp_path):
    p = farm().save(tmp_path / "site.json")
    monkeypatch.setenv("R1_SITE_MAP", str(p))
    assert plan_command({"cmd": "go_to_house", "house": "B"}, MAP).status == "accepted"
    farm().patch({"op": "close_edge", "a": "j2", "b": "B/door"}).save(p)       # edited → unconfirmed again
    assert "not confirmed" in plan_command({"cmd": "go_to_house", "house": "B"}, MAP).reason


# ------------------------------------------------------------------ sketch (fake VLM client, no network)
class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        msg = SimpleNamespace(content=self.replies.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _png(w=400, h=300) -> bytes:
    import io
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(b, "PNG")
    return b.getvalue()


SKETCH_REPLY = """<think>three houses</think>```json
{"nodes": [{"id": "n1", "kind": "house", "label": "Tunnel 09", "xy": [250, 500], "door_marked": true},
           {"id": "n2", "kind": "house", "label": "House B", "xy": [750, 500], "door_marked": false},
           {"id": "n3", "kind": "dock", "label": "dock", "xy": [250, 900]},
           {"id": "n4", "kind": "junction", "label": "", "xy": "bad"}],
 "edges": [{"a": "n1", "b": "n2", "notes": "wet after watering", "blocked": false, "width_m": null},
           {"a": "n3", "b": "n1", "notes": [], "blocked": false, "width_m": 1.5},
           {"a": "n1", "b": "ghost"}],
 "areas": [{"label": "pond", "xy": [500, 100], "notes": ["do not pass 7-9am"]}], "notes": ["drawn by farmer"],}
```"""


def test_sketch_parse_and_anchor_alignment():
    from cloud.maps.sketch import norm_label, sketch_to_site_map
    assert norm_label("Tunnel 09") == norm_label("tunnel-09") and norm_label("House B") == norm_label("HouseB")
    client = FakeClient(["not json at all", SKETCH_REPLY])
    # image 400x300: n1 pixel (100,150) y-up (100,150); n3 (100,30). Anchors: 1 px = 0.1 m, shifted.
    anchors = {"tunnel-09": (195.0, 156.0), "dock": (195.0, 144.0)}
    m, rep = sketch_to_site_map(_png(), anchors, client=client, model="fake")
    assert len(client.calls) == 2                                 # one correction retry
    assert m.meta["scale_known"] and not m.confirmed and m.meta["source"] == "hand_sketch"
    assert rep["scale_m_per_px"] == pytest.approx(0.1) and rep["rmse_m"] < 1e-6
    assert set(rep["anchors_matched"]) == {"tunnel-09", "dock"}
    assert m.nodes["n1"].house == "tunnel-09" and m.nodes["n2"].house == "House B"
    assert m.nodes["n2"].extra["door_estimated"] and np.allclose(m.nodes["n2"].xy, (215.0, 156.0))
    assert "n4" not in m.nodes and len(m.edges) == 2
    e = m.find_edge("n1", "n2")
    assert e.notes == ["wet after watering"] and e.length_m == pytest.approx(20.0)
    assert m.landmarks[0]["notes"] == ["do not pass 7-9am"] and m.meta["sketch_notes"] == ["drawn by farmer"]
    r = plan_command({"cmd": "go_to_house", "house": "House B"}, MAP, site=m)
    assert r.status == "rejected" and "not confirmed" in r.reason
    assert plan_command({"cmd": "go_to_house", "house": "House B"}, MAP, site=m.confirm("farmer")).status == "accepted"


def test_sketch_without_anchors_stays_in_pixels():
    from cloud.maps.sketch import sketch_to_site_map
    m, rep = sketch_to_site_map(_png(), {"tunnel-09": (1.0, 2.0)}, client=FakeClient([SKETCH_REPLY]), model="fake")
    assert not m.meta["scale_known"] and rep["rmse_m"] is None and m.nodes["n1"].xy == (100.0, 150.0)
    assert m.nodes["n1"].extra["pixel_xy"] == [100.0, 150.0] and m.to_dict()["frame"] == "sketch_pixels"


def test_drone_topdown_alignment(tmp_path):
    from cloud.maps.drone import attach_topdown
    meta = {"resolution_per_px": 0.05, "origin_uv": [-5.0, -5.0], "size_px": [200, 200], "e1": [1, 0, 0], "e2": [0, 1, 0]}
    (tmp_path / "summary.json").write_text(json.dumps({"topdown": meta, "model": "fake"}))
    (tmp_path / "trajectory.json").write_text(json.dumps({"frames": [{"position": [0, 0, 1]}, {"position": [1, 0, 1]}]}))
    # uv (0,0) is pixel (100, 99); uv (1,0) is pixel (120, 99). Site: rotate 90° and shift to (195, 150).
    pairs = [((100, 99), (195.0, 150.0)), ((120, 99), (195.0, 151.0))]
    m, fit = attach_topdown(farm(), tmp_path, pairs, ["tunnel-09", "B"])
    assert fit["rmse_m"] < 1e-6 and fit["scale"] == pytest.approx(1.0) and fit["rotation_deg"] == pytest.approx(90.0)
    assert m.layers[0]["kind"] == "drone_topdown" and m.confirmed                     # a layer is not a graph edit
    assert np.allclose(m.landmarks[-1]["xy"], [[195.0, 150.0], [195.0, 151.0]])


@pytest.mark.skipif(not GLB.exists(), reason="V18 farm GLB not present")
def test_extract_v18_smoke(tmp_path):
    import subprocess, sys
    out = tmp_path / "v18.json"
    rep = json.loads(subprocess.run([sys.executable, str(ROOT / "scripts/extract_site_map_v18.py"), "--out", str(out)],
                                    check=True, capture_output=True, text=True).stdout)
    m = SiteMap.load(out)
    assert len(rep["houses"]) >= 20 and not m.confirmed and m.meta["source"] == "v18" and rep["warnings"] == []
    assert rep["frame_checks"]["tunnel-09"] < 0.01
    assert np.allclose(m.nodes[m.door_of("tunnel-09")].xy, (195.065, 156.667), atol=0.01)
    assert all(e.width_m is None for e in m.edges)
    assert m.route("dock", m.door_of("south-greenhouse-block-03")).length_m > 50
