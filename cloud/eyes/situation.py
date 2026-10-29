"""Situation report: when SAFETY stops the robot (LiDAR obstacle, blocked path, stuck), the front-camera frame goes to
Cosmos Reason, which says *what* is there in farm terms (a worker, a cart, a hose across the aisle, a fallen plant...).
HERMES gets the report and decides what to do (cloud/brain policy in agent/brain_loop.decide_situation).

Event-triggered only, never a continuous video feed: the robot's own SAFETY has already stopped the wheels; the model
explains the stop so the planner (and the farmer) can act on it. Env: EYES_BASE_URL, EYES_MODEL, EYES_API_KEY."""
from __future__ import annotations
import base64
import json
import os
import re

WHAT = ("person", "animal", "vehicle_or_cart", "crate_or_bin", "tool_or_hose", "fallen_plant_or_branch",
        "door_or_gate_closed", "water_or_mud", "debris", "nothing_visible", "other")
MOVABLE = {"person", "animal", "vehicle_or_cart"}           # usually clears by itself


def build_prompt(meta: dict) -> str:
    where = meta.get("aisle") or "the yard between greenhouses"
    trig = meta.get("trigger", "obstacle_stop")
    rng = meta.get("lidar_front_m")
    return (
        "You are the eyes of R1, a small 4-wheel inspection robot (0.53 m wide) in a tomato greenhouse farm. "
        f"The robot just stopped by itself ({trig}"
        + (f", LiDAR return {rng:.2f} m ahead" if isinstance(rng, (int, float)) else "")
        + f") while driving in {where}"
        + (f" at metre {meta['metre']:.1f}" if isinstance(meta.get("metre"), (int, float)) else "")
        + ". This is its front camera. Describe what is in front of the robot, in farm terms.\n"
        "Answer with JSON only:\n"
        '{"what": one of ' + json.dumps(list(WHAT)) + ",\n"
        ' "description": "one sentence: what you see and where (left/centre/right, near/far)",\n'
        ' "blocking_path": true if it blocks the robot\'s lane,\n'
        ' "likely_to_move": true if it will probably clear by itself (a person walking, an animal),\n'
        ' "crop_concern": "short note if the crop itself is affected (fallen plant, broken stake, flooding), else empty",\n'
        ' "confidence": 0.0-1.0}\n'
        "Say nothing_visible if the lane looks clear. Do not guess beyond what is visible.")


def _client():
    from openai import OpenAI
    return OpenAI(base_url=os.environ.get("EYES_BASE_URL", "http://localhost:8000/v1"),
                  api_key=os.environ.get("EYES_API_KEY", "EMPTY"), timeout=120.0)


def parse_situation(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {text[:200]!r}")
    d = json.loads(m.group(0))
    what = str(d.get("what", "other")).strip().lower().replace(" ", "_")
    out = {"what": what if what in WHAT else "other",
           "description": str(d.get("description", "")).strip(),
           "blocking_path": bool(d.get("blocking_path", what not in ("nothing_visible",))),
           "likely_to_move": bool(d.get("likely_to_move", what in MOVABLE)),
           "crop_concern": str(d.get("crop_concern", "") or "").strip(),
           "confidence": max(0.0, min(1.0, float(d.get("confidence", 0.5) or 0.0)))}
    return out


def describe_situation(image_bytes: bytes, meta: dict, client=None, model: str | None = None) -> dict:
    model = model or os.environ.get("EYES_MODEL", "nvidia/Cosmos-Reason2-8B")
    mime = "image/png" if image_bytes[:4] == b"\x89PNG" else "image/jpeg"
    url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                             {"type": "text", "text": build_prompt(meta)}]}]
    client = client or _client()
    err = None
    for _ in range(2):
        r = client.chat.completions.create(model=model, messages=messages, temperature=0.2, max_tokens=1024)
        msg = r.choices[0].message
        text = msg.content or getattr(msg, "reasoning_content", None) or ""
        try:
            return {**parse_situation(text), "model": model.split("/")[-1]}
        except (ValueError, TypeError) as e:
            err = e
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": "Reply again with the JSON object only."}]
    return {"what": "other", "description": f"unreadable reply: {err}", "blocking_path": True, "likely_to_move": False,
            "crop_concern": "", "confidence": 0.0, "model": model.split("/")[-1]}
