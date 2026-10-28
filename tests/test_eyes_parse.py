import json
from types import SimpleNamespace
import pytest
from cloud.eyes.inspect import ParseError, Finding, build_prompt, inspect_image, parse_finding

MAST = {"house": "tunnel-09", "aisle": "tunnel-09/aisle-02", "metre": 15.0, "camera": "mast_left_mid"}
FRONT = {"house": "tunnel-09", "aisle": "tunnel-09/aisle-02", "metre": 16.0, "camera": "front"}
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16


def test_plain_json_parses_and_keeps_meta():
    f = parse_finding('{"status": "healthy", "issues": [], "summary": "dense green foliage"}', MAST)
    assert f.status == "healthy" and f.issues == [] and f.nav is None and f.meta["metre"] == 15.0


def test_think_answer_fences_and_trailing_commas():
    text = ('<think>lower leaves show rings {not json}</think>\n<answer>```json\n{"status": "Disease", "issues": '
            '[{"label": "early blight", "confidence": 0.8, "bbox_norm": [0.1, 0.2, 0.4, 0.6], '
            '"evidence": "concentric \\"target\\" spots {x}",},], "summary": "blight on lower leaves"}\n```</answer>')
    f = parse_finding(text, MAST)
    assert f.status == "disease"
    assert f.issues[0].label == "early blight" and f.issues[0].bbox_norm == [0.1, 0.2, 0.4, 0.6]
    assert "{x}" in f.issues[0].evidence


def test_prose_around_json_and_unclosed_think():
    f = parse_finding('reasoning...</think> Here you go: {"status": "suspect", "issues": [], "summary": "blurry"} ok')
    assert f.status == "suspect"


def test_confidence_and_bbox_normalisation():
    f = parse_finding(json.dumps({"status": "suspect", "summary": "", "issues": [
        {"label": "yellowing", "confidence": 85, "bbox_norm": [100, 200, 500, 900]},
        {"label": "wilting", "confidence": -1, "bbox_norm": [0.5, 0.5, 0.2, 0.9]},
        {"label": "pests", "confidence": 0.3, "bbox_norm": "somewhere"}]}))
    a, b, c = f.issues
    assert a.confidence == pytest.approx(0.85) and a.bbox_norm == pytest.approx([0.1, 0.2, 0.5, 0.9])
    assert b.confidence == 0.0 and b.bbox_norm is None          # inverted box dropped
    assert c.bbox_norm is None


@pytest.mark.parametrize("text", ["no json here", '{"status": "dead", "issues": [], "summary": ""}',
                                  '{"status": "healthy", "issues": [{"confidence": 0.5}]}', '{"status": ',
                                  "[1, 2, 3]"])
def test_invalid_replies_raise(text):
    with pytest.raises(ParseError):
        parse_finding(text, MAST)


def test_front_camera_nav_fields():
    f = parse_finding('{"aisle_clear": false, "obstacle": "orange crate at 2 m", "appears_centered": true, '
                      '"issues": [{"label": "obstacle:crate", "confidence": 0.9, "bbox_norm": null, "evidence": ""}],'
                      ' "summary": "blocked"}', FRONT)
    assert f.status == "not_plant" and f.nav.aisle_clear is False and f.nav.appears_centered is True
    assert f.nav.obstacle == "orange crate at 2 m"
    with pytest.raises(ParseError):
        parse_finding('{"status": "not_plant", "issues": [], "summary": "?"}', FRONT)   # missing aisle_clear


def test_prompt_selection():
    assert "early blight" in build_prompt(MAST) and "aisle_clear" not in build_prompt(MAST)
    assert "aisle_clear" in build_prompt(FRONT) and "metre=16.0" in build_prompt(FRONT)


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        msg = SimpleNamespace(content=self.replies.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_inspect_retries_once_then_succeeds():
    fc = FakeClient(["Sorry, I think it is fine.", '{"status": "healthy", "issues": [], "summary": "ok"}'])
    f = inspect_image(JPEG, MAST, client=fc, model="m")
    assert isinstance(f, Finding) and f.status == "healthy" and len(fc.calls) == 2
    first = fc.calls[0]["messages"][1]["content"][0]["image_url"]["url"]
    assert first.startswith("data:image/jpeg;base64,")
    assert fc.calls[1]["messages"][-1]["role"] == "user" and "ONLY the JSON" in fc.calls[1]["messages"][-1]["content"]


def test_inspect_gives_up_after_one_retry():
    fc = FakeClient(["nope", "still nope", "never asked"])
    with pytest.raises(ParseError):
        inspect_image(JPEG, MAST, client=fc, model="m")
    assert len(fc.calls) == 2


def test_rejects_unknown_image_format():
    with pytest.raises(ValueError):
        inspect_image(b"GIF89a....", MAST, client=FakeClient([]), model="m")
