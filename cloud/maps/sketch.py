"""Hand-drawn farm map (photo of paper, or a drawing made with the V18 web draw tools) → draft SiteMap.
A vision-language model reads the sketch into STRICT JSON (nodes with normalised image coords and the labels as the
farmer wrote them, edges, free-text notes in Thai/English). With ≥ 2 labels matching known anchors the sketch is fitted
to site metres with a 2-D similarity (Umeyama), else it stays in pixel units (scale_known=false). Always unconfirmed.
Env: SKETCH_BASE_URL (default Nebius Token Factory), SKETCH_MODEL (default nvidia/Nemotron-3.5-Super-VL),
SKETCH_API_KEY (falls back to NEBIUS_API_KEY), SKETCH_THINKING=1 (model reasoning on; slower).
  python -m cloud.maps.sketch photo.jpg --anchor "tunnel-09=195.07,156.67" --anchor "dock=197.07,157.67" --out site_map_sketch.json"""
from __future__ import annotations
import argparse
import base64
import io
import json
import math
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cloud.eyes.inspect import ParseError, RETRY_PROMPT, _extract_json_text, _mime  # noqa: E402
from cloud.mapping.align_to_house import umeyama_2d  # noqa: E402
from r1_nav.site_map import Edge, Node, SiteMap, default_meta  # noqa: E402

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "nvidia/Nemotron-3.5-Super-VL"
KINDS = {"house": "door", "greenhouse": "door", "door": "door", "dock": "dock", "charger": "dock",
         "junction": "junction", "gate": "junction", "waypoint": "waypoint", "other": "waypoint"}

PROMPT = """This image is a hand-drawn (or annotated) map of a farm, made by a farmer for a greenhouse robot.
The image is {w} x {h} pixels. Read it and list:
- nodes: every greenhouse/building (kind "house"), the robot dock or charger ("dock"), path crossings or turns
  ("junction"), gates ("gate") and other labelled points ("other"). For a house give the point of its door/entrance if
  one is drawn (door_marked true), else the point where a path touches the house (door_marked false).
  "label" is copied exactly as written (any language, e.g. "House A", "dock"); use "" if unlabelled.
- edges: drawn paths/roads between two nodes (by node id). Put any text written along or next to a path in "notes"
  (copy it as written, e.g. "wet after watering", "do not pass 7-9am"). blocked=true only if the path is crossed out or
  marked closed. width_m only if a width is written, else null.
- areas: other written notes not tied to one path, with the point they refer to.
Coordinates "xy" are [x, y] normalised to 0..1000 of the image width/height, origin top-left.
Do not invent nodes, paths or notes that are not drawn. Reply with ONLY this JSON object, no markdown:
{{"nodes": [{{"id": "n1", "kind": "house|dock|junction|gate|other", "label": str, "xy": [x, y], "door_marked": bool}}],
 "edges": [{{"a": "n1", "b": "n2", "notes": [str], "blocked": bool, "width_m": number or null}}],
 "areas": [{{"label": str, "xy": [x, y], "notes": [str]}}], "notes": [str]}}"""


def norm_label(s: str) -> str:
    """Case/space/punctuation-insensitive key ("Tunnel 09" == "tunnel-09"); Thai letters are kept."""
    s = unicodedata.normalize("NFC", str(s or "")).casefold()
    return "".join(ch for ch in s if ch.isalnum() or unicodedata.category(ch).startswith("M"))


def parse_sketch_json(text: str) -> dict:
    """Model reply → {"nodes", "edges", "areas", "notes"} with sane types. Raises ParseError."""
    raw = _extract_json_text(text)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        try:
            obj = json.loads(re.sub(r",\s*([}\]])", r"\1", raw))
        except json.JSONDecodeError as e:
            raise ParseError(f"invalid JSON: {e}") from None
    if not isinstance(obj, dict) or not isinstance(obj.get("nodes"), list) or not obj["nodes"]:
        raise ParseError("reply has no 'nodes' list")
    as_list = lambda v: [str(x) for x in v] if isinstance(v, list) else ([str(v)] if v else [])
    nodes, ids = [], set()
    for i, n in enumerate(obj["nodes"]):
        try:
            x, y = (float(v) for v in n["xy"][:2])
        except (KeyError, TypeError, ValueError):
            continue                                     # a node without a usable position is dropped
        nid = str(n.get("id") or f"n{i + 1}")
        if nid in ids:
            nid = f"{nid}_{i + 1}"
        ids.add(nid)
        kind = str(n.get("kind", "other")).strip().lower()
        nodes.append({"id": nid, "kind": kind if kind in KINDS else "other", "label": str(n.get("label") or ""),
                      "xy": [x, y], "door_marked": bool(n.get("door_marked", False))})
    if not nodes:
        raise ParseError("no node has a usable xy")
    edges = []
    for e in obj.get("edges") or []:
        if not isinstance(e, dict) or e.get("a") not in ids or e.get("b") not in ids or e["a"] == e["b"]:
            continue
        w = e.get("width_m")
        edges.append({"a": e["a"], "b": e["b"], "notes": as_list(e.get("notes")), "blocked": bool(e.get("blocked", False)),
                      "width_m": float(w) if isinstance(w, (int, float)) and w > 0 else None})
    areas = [{"label": str(a.get("label") or ""), "xy": a.get("xy"), "notes": as_list(a.get("notes"))}
             for a in obj.get("areas") or [] if isinstance(a, dict)]
    return {"nodes": nodes, "edges": edges, "areas": areas, "notes": as_list(obj.get("notes"))}


def _image_size(image_bytes: bytes) -> tuple[int, int]:
    from PIL import Image
    with Image.open(io.BytesIO(image_bytes)) as im:
        return im.size


def _to_px(xy, w, h, coord_scale: float) -> tuple[float, float]:
    """Model coords (0..coord_scale) → pixels with y UP (u, h - v), so the pixel→site fit is a proper similarity."""
    return (float(xy[0]) / coord_scale * w, h - float(xy[1]) / coord_scale * h)


def build_site_map(parsed: dict, size: tuple[int, int], anchors: dict | None = None, coord_scale: float = 1000.0,
                   image_name: str | None = None) -> tuple[SiteMap, dict]:
    """Parsed sketch → unconfirmed SiteMap. Anchors {name: (x, y) site metres} are matched to node labels."""
    w, h = size
    if max(max(n["xy"]) for n in parsed["nodes"]) <= 1.0:
        coord_scale = 1.0                                # model answered in 0..1 fractions
    px = {n["id"]: _to_px(n["xy"], w, h, coord_scale) for n in parsed["nodes"]}
    keys = {norm_label(k): (k, tuple(map(float, v))) for k, v in (anchors or {}).items()}
    matches = []
    for n in parsed["nodes"]:
        k = keys.get(norm_label(n["label"])) or (keys.get(norm_label(n["kind"])) if n["kind"] == "dock" else None)
        if k and k[0] not in [m[1] for m in matches]:
            matches.append((n["id"], k[0], k[1]))
    report = {"anchors_matched": [m[1] for m in matches], "rmse_m": None, "scale_m_per_px": None}
    fit = None
    if len(matches) >= 2:
        src = np.array([px[i] for i, _, _ in matches], float)
        dst = np.array([xy for _, _, xy in matches], float)
        s, R, t = umeyama_2d(src, dst)
        fit = lambda p: tuple((s * R @ np.asarray(p, float) + t).tolist())
        res = np.array([fit(p) for p in src]) - dst
        report.update(rmse_m=round(float(np.sqrt((res ** 2).sum(1).mean())), 3), scale_m_per_px=round(s, 5),
                      rotation_deg=round(math.degrees(math.atan2(R[1, 0], R[0, 0])), 2))
    nodes = {}
    for n in parsed["nodes"]:
        kind = KINDS[n["kind"]]
        xy = fit(px[n["id"]]) if fit else px[n["id"]]
        house = (n["label"] or n["id"]) if kind == "door" else None
        anchor = next((a for i, a, _ in matches if i == n["id"]), None)
        if kind == "door" and anchor:
            house = anchor                               # matched house uses the known id (e.g. tunnel-09)
        extra = {"pixel_xy": [round(v, 1) for v in (px[n["id"]][0], h - px[n["id"]][1])]}
        if kind == "door":
            extra["door_estimated"] = not n["door_marked"]
        if anchor:
            extra["anchor"] = anchor
        nodes[n["id"]] = Node(n["id"], kind, xy, house, None, n["label"] or None, extra)
    edges = [Edge(e["a"], e["b"], None, e["width_m"], "unknown", e["notes"], e["blocked"]) for e in parsed["edges"]]
    for e in edges:
        if fit:
            e.length_m = round(math.dist(nodes[e.a].xy, nodes[e.b].xy), 3)
    landmarks = [{"id": f"note-{i + 1}", "kind": "note_area", "label": a["label"], "notes": a["notes"],
                  **({"xy": list(fit(_to_px(a["xy"], w, h, coord_scale))) if fit else
                      list(_to_px(a["xy"], w, h, coord_scale))} if isinstance(a["xy"], list) and len(a["xy"]) >= 2 else {})}
                 for i, a in enumerate(parsed["areas"])]
    meta = default_meta("hand_sketch", fit is not None, anchors_used=report["anchors_matched"],
                        sketch_notes=parsed["notes"], sketch_image={"name": image_name, "size_px": [w, h]},
                        alignment=report if fit else None)
    site = SiteMap(nodes, edges, meta, [{"kind": "sketch", "path": image_name, "size_px": [w, h]}] if image_name else [],
                   landmarks)
    report["warnings"] = site.validate()
    return site, report


def make_client():
    from openai import OpenAI
    key = os.environ.get("SKETCH_API_KEY") or os.environ.get("NEBIUS_API_KEY") or "EMPTY"
    return OpenAI(base_url=os.environ.get("SKETCH_BASE_URL", DEFAULT_BASE_URL), api_key=key, timeout=180.0)


def read_sketch(image_bytes: bytes, client=None, model: str | None = None) -> dict:
    """Ask the VLM once (plus one correction retry) and return the parsed sketch JSON."""
    client, model = client or make_client(), model or os.environ.get("SKETCH_MODEL", DEFAULT_MODEL)
    think = os.environ.get("SKETCH_THINKING") == "1"     # Nemotron VL reasoning: ~40 s, >4k tokens; off = ~3 s, same topology
    w, h = _image_size(image_bytes)
    url = f"data:{_mime(image_bytes)};base64,{base64.b64encode(image_bytes).decode()}"
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                             {"type": "text", "text": PROMPT.format(w=w, h=h)}]}]
    err = None
    for _ in range(2):
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0.1,
                                              max_tokens=16384 if think else 4096,
                                              extra_body={"chat_template_kwargs": {"enable_thinking": think}})
        msg = resp.choices[0].message
        text = msg.content or getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
        try:
            return parse_sketch_json(text)
        except ParseError as e:
            err = e
            messages += [{"role": "assistant", "content": text}, {"role": "user", "content": RETRY_PROMPT.format(err=e)}]
    raise ParseError(f"sketch reply unparseable after retry: {err}")


def sketch_to_site_map(image_bytes: bytes, anchors: dict | None = None, client=None, model: str | None = None,
                       image_name: str | None = None) -> tuple[SiteMap, dict]:
    parsed = read_sketch(image_bytes, client, model)
    site, report = build_site_map(parsed, _image_size(image_bytes), anchors, image_name=image_name)
    site.meta["sketch_model"] = model or os.environ.get("SKETCH_MODEL", DEFAULT_MODEL)
    return site, report


def parse_anchor(s: str) -> tuple[str, tuple[float, float]]:
    name, _, xy = s.rpartition("=")
    x, y = (float(v) for v in xy.split(","))
    if not name:
        raise argparse.ArgumentTypeError(f"anchor must be NAME=X,Y, got {s!r}")
    return name, (x, y)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image", type=Path)
    ap.add_argument("--anchor", action="append", type=parse_anchor, default=[],
                    help="label as written on the sketch (or house id) = site x,y metres; repeat (≥ 2 to get metres)")
    ap.add_argument("--anchors-from", type=Path, help="site map JSON: use its door/dock nodes as anchors (house ids)")
    ap.add_argument("--out", type=Path, default=Path("site_map_sketch.json"))
    a = ap.parse_args(argv)
    anchors = dict(a.anchor)
    if a.anchors_from:
        ref = SiteMap.load(a.anchors_from)
        for n in ref.nodes.values():
            if n.kind in ("door", "dock"):
                anchors.setdefault(n.house or n.id, n.xy)
    site, report = sketch_to_site_map(a.image.read_bytes(), anchors, image_name=str(a.image))
    site.save(a.out)
    print(json.dumps({"out": str(a.out), "nodes": len(site.nodes), "edges": len(site.edges),
                      "scale_known": site.meta["scale_known"], **report}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
