"""Ask Cosmos Reason (vLLM, OpenAI-compatible) about one R1 camera frame and return a validated Finding.
Mast cameras get a conservative tomato plant-health prompt; the front camera gets a navigation sanity prompt.
Env: EYES_BASE_URL (e.g. http://localhost:8000/v1), EYES_MODEL, EYES_API_KEY, EYES_REASONING=1, EYES_STRUCTURED=1."""
from __future__ import annotations
import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

FRONT_CAMERA = "front"
STATUSES = ("healthy", "suspect", "disease", "not_plant")
_STATUS_ALIASES = {"diseased": "disease", "sick": "disease", "ok": "healthy", "suspicious": "suspect",
                   "uncertain": "suspect", "no_plant": "not_plant", "not plant": "not_plant", "none": "not_plant"}
DEFAULT_MODEL = "nvidia/Cosmos-Reason2-32B"


class ParseError(ValueError):
    pass


class Issue(BaseModel):
    label: str
    confidence: float = Field(0.0)
    bbox_norm: list[float] | None = None
    evidence: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, v):
        v = float(v if v is not None else 0.0)
        return max(0.0, min(1.0, v / 100.0 if v > 1.0 else v))

    @field_validator("bbox_norm", mode="before")
    @classmethod
    def _bbox(cls, v):
        """Accept [x0,y0,x1,y1] in 0..1 or Qwen-style 0..1000; anything else becomes None."""
        if v is None or not isinstance(v, (list, tuple)) or len(v) != 4:
            return None
        try:
            b = [float(x) for x in v]
        except (TypeError, ValueError):
            return None
        if max(b) > 1.0:
            if max(b) > 1000.0:
                return None
            b = [x / 1000.0 for x in b]
        b = [max(0.0, min(1.0, x)) for x in b]
        return b if b[0] < b[2] and b[1] < b[3] else None


class NavCheck(BaseModel):
    aisle_clear: bool
    obstacle: str | None = None
    appears_centered: bool | None = None


class Finding(BaseModel):
    status: Literal["healthy", "suspect", "disease", "not_plant"]
    issues: list[Issue] = []
    summary: str = ""
    nav: NavCheck | None = None          # front camera only
    meta: dict = {}

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, v):
        s = str(v).strip().lower().replace("-", "_")
        return _STATUS_ALIASES.get(s, s)


PLANT_PROMPT = """You inspect one image from a side-looking camera on a greenhouse robot driving along a tomato aisle.
Context: {ctx}.
Look for: early blight (concentric brown rings, yellow halo, older leaves), late blight (dark greasy water-soaked
patches, white mould on undersides), leaf mold (yellow upper-leaf spots, olive-grey velvet below), yellowing/chlorosis,
wilting, pests (whitefly, aphids, mites, leaf-miner trails, caterpillar damage), fruit rot (blossom-end rot, soft or
mouldy fruit). Be conservative: use "disease" only for clear visible evidence, "suspect" when unsure or the view is
blurred/occluded/far, "healthy" when foliage is visible and nothing is wrong, "not_plant" when no tomato plant is visible.
Do not invent issues; an empty list is fine. confidence is 0..1. bbox_norm is [x0,y0,x1,y1] normalised to 0..1 of the
image (origin top-left) or null.
Reply with ONLY this JSON object, no markdown:
{{"status": "healthy|suspect|disease|not_plant", "issues": [{{"label": str, "confidence": float, "bbox_norm": [x0,y0,x1,y1] or null, "evidence": str}}], "summary": str}}"""

NAV_PROMPT = """You check one image from the forward-facing wide-angle (175 deg fisheye) camera of a greenhouse robot.
Context: {ctx}.
The robot should be driving straight down the middle of a narrow aisle between tomato beds. Report whether the aisle
ahead is clear, describe any obstacle (person, crate, hose, fallen plant, cart, wall/end of aisle) and whether the robot
appears centred between the two beds and aligned with the aisle. Edges are distorted by the fisheye lens.
Use status "not_plant" (this is a navigation check). Put each obstacle in issues with label "obstacle:<what>".
Reply with ONLY this JSON object, no markdown:
{{"aisle_clear": bool, "obstacle": str or null, "appears_centered": bool, "status": "not_plant", "issues": [{{"label": str, "confidence": float, "bbox_norm": [x0,y0,x1,y1] or null, "evidence": str}}], "summary": str}}"""

REASONING_SUFFIX = ("\nAnswer the question using the following format:\n<think>\nyour reasoning\n</think>\n\n"
                    "Write your final answer immediately after the </think> tag.")
RETRY_PROMPT = ("Your previous reply was not valid JSON for the requested schema ({err}). "
                "Reply again with ONLY the JSON object, no prose, no markdown fences.")


SCREEN_SUFFIX = ("\nThis is the fast SCREENING pass while driving; every flagged spot gets a closer, stricter re-inspection. "
                 "Prefer recall: if ANY leaf shows spots, lesions, rings, yellow halos or discoloured patches, even small or "
                 "partly visible, answer at least \"suspect\" and box them. Say \"healthy\" only if all visible leaves look clean.")


def build_prompt(meta: dict) -> str:
    keys = ("house", "aisle", "metre", "heading_deg", "camera")
    ctx = ", ".join(f"{k}={meta[k]}" for k in keys if meta.get(k) is not None) or "unknown position"
    prompt = (NAV_PROMPT if meta.get("camera") == FRONT_CAMERA else PLANT_PROMPT).format(ctx=ctx)
    if meta.get("camera") != FRONT_CAMERA and meta.get("purpose", "inspect") == "inspect":
        prompt += SCREEN_SUFFIX                  # screening = sensitive; revisit keeps the conservative wording
    return prompt + (REASONING_SUFFIX if os.environ.get("EYES_REASONING") == "1" else "")


def _extract_json_text(text: str) -> str:
    """Strip <think> blocks, <answer> tags and ``` fences, then return the first balanced {...} object."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    if "</think>" in t:                                  # reasoning parser off and an unclosed opening tag
        t = t.split("</think>", 1)[1]
    m = re.search(r"<answer>(.*?)</answer>", t, flags=re.S)
    if m:
        t = m.group(1)
    t = re.sub(r"```(?:json)?", "", t)
    start = t.find("{")
    if start < 0:
        raise ParseError("no JSON object in reply")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(t[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]
    raise ParseError("unbalanced JSON object in reply")


def parse_finding(text: str, meta: dict | None = None) -> Finding:
    """Parse and validate a model reply. Raises ParseError on anything that is not a usable Finding."""
    meta = dict(meta or {})
    raw = _extract_json_text(text)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        try:                                             # common slip: trailing commas
            obj = json.loads(re.sub(r",\s*([}\]])", r"\1", raw))
        except json.JSONDecodeError as e:
            raise ParseError(f"invalid JSON: {e}") from None
    if not isinstance(obj, dict):
        raise ParseError("reply JSON is not an object")
    nav = None
    if meta.get("camera") == FRONT_CAMERA or "aisle_clear" in obj:
        obj.setdefault("status", "not_plant")
        try:
            nav = NavCheck(aisle_clear=obj["aisle_clear"], obstacle=obj.get("obstacle") or None,
                           appears_centered=obj.get("appears_centered"))
        except (KeyError, ValidationError) as e:
            raise ParseError(f"bad navigation fields: {e}") from None
    try:
        return Finding(status=obj.get("status"), issues=obj.get("issues") or [], summary=str(obj.get("summary") or ""),
                       nav=nav, meta=meta)
    except ValidationError as e:
        raise ParseError(f"schema: {e.errors()[0].get('msg')} at {e.errors()[0].get('loc')}") from None


def _mime(image_bytes: bytes) -> str:
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("unsupported image format (need JPEG, PNG or WebP)")


def _response_format() -> dict | None:
    if os.environ.get("EYES_STRUCTURED") != "1":
        return None
    schema = {"type": "object", "required": ["status", "issues", "summary"], "properties": {
        "status": {"type": "string", "enum": list(STATUSES)}, "summary": {"type": "string"},
        "aisle_clear": {"type": "boolean"}, "obstacle": {"type": ["string", "null"]},
        "appears_centered": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "object", "required": ["label", "confidence"], "properties": {
            "label": {"type": "string"}, "confidence": {"type": "number"}, "evidence": {"type": "string"},
            "bbox_norm": {"anyOf": [{"type": "null"}, {"type": "array", "items": {"type": "number"},
                                                       "minItems": 4, "maxItems": 4}]}}}}}}
    return {"type": "json_schema", "json_schema": {"name": "finding", "schema": schema}}


def make_client():
    from openai import OpenAI
    return OpenAI(base_url=os.environ.get("EYES_BASE_URL", "http://localhost:8000/v1"),
                  api_key=os.environ.get("EYES_API_KEY", "EMPTY"), timeout=180.0)


def inspect_image(image_bytes: bytes, meta: dict, client=None, model: str | None = None) -> Finding:
    """One frame -> Finding. Retries once with a correction message if the reply does not parse."""
    client = client or make_client()
    model = model or os.environ.get("EYES_MODEL", DEFAULT_MODEL)
    url = f"data:{_mime(image_bytes)};base64,{base64.b64encode(image_bytes).decode()}"
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                             {"type": "text", "text": build_prompt(meta)}]}]
    extra = {"response_format": rf} if (rf := _response_format()) else {}
    err = None
    for _ in range(2):
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2, top_p=0.95,
                                              max_tokens=4096, **extra)
        msg = resp.choices[0].message
        # vLLM's qwen3 reasoning parser puts the whole reply in `reasoning` when the model emits no </think>
        text = msg.content or getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
        try:
            return parse_finding(text, meta)
        except ParseError as e:
            err = e
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": RETRY_PROMPT.format(err=e)}]
    raise ParseError(f"model reply unparseable after retry: {err}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image", type=Path)
    ap.add_argument("--camera", default="mast_left_mid")
    ap.add_argument("--house", default=None)
    ap.add_argument("--aisle", default=None)
    ap.add_argument("--metre", type=float, default=None)
    ap.add_argument("--heading-deg", type=float, default=None)
    a = ap.parse_args(argv)
    house = a.house or (a.aisle.split("/")[0] if a.aisle and "/" in a.aisle else None)
    meta = {"house": house, "aisle": a.aisle, "metre": a.metre, "heading_deg": a.heading_deg, "camera": a.camera}
    finding = inspect_image(a.image.read_bytes(), meta)
    print(finding.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
