#!/usr/bin/env python3
"""Generate a realistic fake dashboard data dir so the UI can be built / recorded before the real system runs.

    python3 dashboard/make_demo_data.py --out /tmp/dash_demo            # write a finished scenario at once
    python3 dashboard/make_demo_data.py --out /tmp/dash_demo --live     # replay it in real time (for recording)

Scenario: R1 enters tunnel-09 aisle-02 from the door (+Y), surveys toward the far end capturing the 6 side
cameras every 0.5 m and the front camera every 2 m. Cosmos Reason flags a 'suspect' at 9.5 m and a 'disease'
(leaf spot) at 15.0 m. HERMES (Nemotron) decides to revisit 15.0 m after reaching 22 m; the robot drives back,
holds, re-captures, and the revisit confirms the disease.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import struct
import sys
import time
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MAP = REPO / "data" / "tunnel09_aisle_map_v18.json"
SIDE_CAMERAS = ("mast_left_low", "mast_left_mid", "mast_left_high",
                "mast_right_low", "mast_right_mid", "mast_right_high")
FRONT = "front"
AISLE = "tunnel-09/aisle-02"
HOUSE = "tunnel-09"
SURVEY_TO_M = 22.0
REVISIT_M = 15.0
SPEED_MPS = 0.25

try:
    from PIL import Image, ImageDraw, ImageFont   # optional
except Exception:                                  # pragma: no cover - depends on the machine
    Image = None


# ---------------------------------------------------------------- images

def _png_bytes(w: int, h: int, rgb_rows: list[bytes]) -> bytes:
    """Minimal PNG encoder (stdlib zlib) for when PIL is missing."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + r for r in rgb_rows)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def plant_image(path: Path, seed: int, label: str, spots: int = 0, kind: str = "plant", size=(320, 240)) -> None:
    """A stylised plant-row photo: soil, leaves, optional brown disease spots, caption."""
    rnd = random.Random(seed)
    w, h = size
    path.parent.mkdir(parents=True, exist_ok=True)
    if Image is None:
        rows = []
        for y in range(h):
            row = bytearray()
            for x in range(w):
                leaf = (math.sin(x * 0.07 + seed) + math.cos(y * 0.09 + seed * 0.3)) > 0.4
                g = 120 + int(60 * math.sin((x + y) * 0.03 + seed))
                px = (40, max(60, g), 45) if leaf and kind == "plant" else (92, 70, 52)
                if spots and leaf and (x * 7 + y * 13 + seed) % 97 < spots:
                    px = (120, 80, 30)
                row += bytes(px)
            rows.append(bytes(row))
        path.write_bytes(_png_bytes(w, h, rows))
        return
    img = Image.new("RGB", (w, h), (24, 32, 26))
    d = ImageDraw.Draw(img)
    for y in range(h):                                          # sky / film glow to soil gradient
        t = y / h
        d.line([(0, y), (w, y)], fill=(int(60 + 40 * (1 - t)), int(72 + 30 * (1 - t)), int(64 + 20 * (1 - t))))
    d.rectangle([0, int(h * 0.78), w, h], fill=(86, 64, 46))   # bed soil
    if kind == "front":
        d.polygon([(w * 0.42, h * 0.45), (w * 0.58, h * 0.45), (w * 0.9, h), (w * 0.1, h)], fill=(70, 66, 60))
        for sx in (-1, 1):
            for i in range(14):
                x = w / 2 + sx * (w * 0.12 + i * 9)
                y = h * 0.45 + i * 10
                d.ellipse([x - 6 - i, y - 4 - i, x + 6 + i, y + 4 + i], fill=(40 + i * 3, 110 + i * 4, 50))
    elif kind == "plant":
        for _ in range(26):
            cx, cy = rnd.uniform(0, w), rnd.uniform(h * 0.05, h * 0.8)
            rx, ry = rnd.uniform(14, 34), rnd.uniform(8, 18)
            g = rnd.randint(100, 170)
            d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(rnd.randint(30, 60), g, rnd.randint(35, 60)))
            d.line([(cx, cy), (cx, h * 0.8)], fill=(50, 90, 40), width=2)
        for _ in range(spots):
            cx, cy = rnd.uniform(w * 0.45, w * 0.8), rnd.uniform(h * 0.2, h * 0.6)
            r = rnd.uniform(3, 8)
            d.ellipse([cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2], fill=(190, 170, 60))
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(105, 62, 28))
    else:                                                       # not_plant: frame post / irrigation
        d.rectangle([w * 0.4, 0, w * 0.52, h], fill=(150, 152, 156))
        d.line([(0, h * 0.3), (w, h * 0.32)], fill=(30, 30, 30), width=4)
    d.rectangle([0, 0, w, 18], fill=(0, 0, 0))
    d.text((6, 3), label, fill=(220, 230, 220), font=ImageFont.load_default())
    img.save(path, optimize=True)


def slam_topdown(path: Path, amap: dict, bounds: dict, px_per_m: int = 24) -> None:
    """Fake SLAM occupancy-style top-down: image +x = house-local +X, image top row = max Y."""
    (x0, y0), (x1, y1) = bounds["min"], bounds["max"]
    w, h = int((x1 - x0) * px_per_m), int((y1 - y0) * px_per_m)
    rnd = random.Random(7)

    def to_px(x, y):
        return (x - x0) * px_per_m, (y1 - y) * px_per_m
    path.parent.mkdir(parents=True, exist_ok=True)
    if Image is None:
        path.write_bytes(_png_bytes(w, h, [bytes([18, 22, 30]) * w for _ in range(h)]))
        return
    img = Image.new("RGB", (w, h), (14, 18, 24))
    d = ImageDraw.Draw(img)
    for _ in range(4000):                                       # speckle = point cloud floor hits
        x, y = rnd.uniform(0, w), rnd.uniform(0, h)
        d.point((x, y), fill=(40, 60, 70))
    for b in amap["bed_bounds_local_m"]:
        ax, ay = to_px(b["min"][0], b["max"][1])
        bx, by = to_px(b["max"][0], b["min"][1])
        d.rectangle([ax, ay, bx, by], fill=(30, 70, 60))
        for _ in range(500):
            d.point((rnd.uniform(ax, bx), rnd.uniform(ay, by)), fill=(70, 160, 120))
    img.save(path, optimize=True)


# ---------------------------------------------------------------- scenario

def build_events(amap: dict) -> list[dict]:
    """Ordered list of {t, kind, payload} where t is seconds since the mission started."""
    events: list[dict] = []
    frame_id = 0
    t = 0.0
    rnd = random.Random(42)

    def status(state, cmd, metre, heading, conf, legs, battery):
        return {"t": t, "kind": "status", "payload": {
            "state": state, "cmd": cmd, "house": HOUSE, "aisle": AISLE, "metre": round(metre, 2),
            "heading_deg": heading, "pose_confidence": conf, "battery": battery, "legs_remaining": legs}}

    def decision(reasoning, command, reply):
        events.append({"t": t, "kind": "decision", "payload": {
            "reasoning": reasoning, "command": command, "reply": reply, "model": "nvidia/nemotron (Token Factory)"}})

    def capture(metre, heading, purpose, conf):
        nonlocal frame_id
        cams = list(SIDE_CAMERAS) + ([FRONT] if abs(metre / 2 - round(metre / 2)) < 1e-6 or purpose == "revisit" else [])
        for cam in cams:
            frame_id += 1
            fname = f"frames/{AISLE.split('/')[-1]}_{metre:05.1f}_{cam}_{frame_id}.png"
            events.append({"t": t, "kind": "frame", "payload": {
                "house": HOUSE, "aisle": AISLE, "metre": round(metre, 2), "heading_deg": heading, "camera": cam,
                "frame_id": frame_id, "pose_confidence": conf, "purpose": purpose if cam != FRONT else "navigate",
                "image_path": fname}})
            if cam == FRONT:
                continue
            events.append({"t": t + 1.2, "kind": "finding", "payload": finding_for(frame_id, metre, cam, purpose, rnd)})

    # --- mission start
    decision("Map loaded: aisle-02 is the only aisle wide enough (1.27 m) for R1 (needs 0.65 m). Survey it head to end.",
             {"cmd": "survey_aisle", "house": HOUSE, "aisle": AISLE}, {"status": "accepted", "reason": "3 legs"})
    events.append(status("transit", "survey_aisle", 0.0, -90.0, 0.93, 3, 0.97))
    t += 4.0
    # --- survey forward
    metre = 0.0
    while metre <= SURVEY_TO_M + 1e-9:
        conf = round(0.9 + 0.06 * math.sin(metre * 0.7) - (0.25 if 11.0 <= metre <= 12.0 else 0.0), 3)
        events.append(status("surveying", "survey_aisle", metre, round(-90.0 + rnd.uniform(-2, 2), 1), conf, 2,
                             round(0.97 - metre * 0.004, 3)))
        capture(metre, -90.0, "inspect", conf)
        if abs(metre - 15.5) < 1e-6:
            t += 0.5
            decision("Cosmos flagged DISEASE at 15.0 m (right bed, leaf spot, conf 0.86). Finish the survey leg to 22 m "
                     "first, then revisit 15.0 m for a close confirmation shot.",
                     {"cmd": "revisit", "house": HOUSE, "aisle": AISLE, "metre": REVISIT_M, "after": "current_leg"},
                     {"status": "queued", "reason": "revisit after leg"})
        if abs(metre - 11.5) < 1e-6:
            decision("Pose confidence dipped to 0.66 near 11.5 m (repeated bed texture). Below 0.5 I would hold; "
                     "continue at reduced speed and keep pinning findings.",
                     {"cmd": "set_speed", "mps": 0.15}, {"status": "rejected", "reason": "unknown cmd 'set_speed'"})
        metre = round(metre + 0.5, 2)
        t += 0.5 / SPEED_MPS
    # --- revisit
    decision("Survey leg reached 22.0 m. Executing queued revisit of 15.0 m (disease suspected, right-mid camera).",
             {"cmd": "revisit", "house": HOUSE, "aisle": AISLE, "metre": REVISIT_M},
             {"status": "accepted", "reason": "legs: drive_in, hold"})
    metre = SURVEY_TO_M
    while metre > REVISIT_M + 1e-9:
        metre = round(metre - 1.0, 2)
        t += 1.0 / (SPEED_MPS * 2)
        events.append(status("revisit", "revisit", metre, 90.0, 0.94, 1, round(0.88 - (22 - metre) * 0.003, 3)))
    t += 1.0
    events.append(status("hold", "revisit", REVISIT_M, -90.0, 0.95, 0, 0.86))
    capture(REVISIT_M, -90.0, "revisit", 0.95)
    t += 3.0
    decision("Revisit confirms leaf spot at 15.0 m on 2 of 3 right cameras. Logged for the grower; resume survey "
             "from 22.0 m to the aisle end.",
             {"cmd": "survey_aisle", "house": HOUSE, "aisle": AISLE, "from_metre": 22.0},
             {"status": "accepted", "reason": "2 legs"})
    events.append(status("hold", "survey_aisle", REVISIT_M, -90.0, 0.95, 2, 0.86))
    return sorted(events, key=lambda e: e["t"])


def finding_for(frame_id: int, metre: float, cam: str, purpose: str, rnd: random.Random) -> dict:
    base = {"frame_id": frame_id, "house": HOUSE, "aisle": AISLE, "metre": round(metre, 2), "camera": cam,
            "purpose": purpose, "model": "nvidia/Cosmos-Reason (Serverless Endpoint)"}
    if abs(metre - REVISIT_M) < 1e-6 and cam in ("mast_right_mid", "mast_right_low"):
        conf = 0.86 if purpose == "inspect" else 0.91
        return dict(base, status="disease", issues=[{
            "label": "leaf spot (fungal, likely Cercospora)", "confidence": conf if cam.endswith("mid") else conf - 0.12,
            "bbox_norm": [0.48, 0.22, 0.80, 0.60],
            "evidence": "Circular brown lesions 3-8 mm with yellow halos on 4+ leaves, clustered on lower canopy."}],
            summary="Brown circular lesions with chlorotic halos — consistent with fungal leaf spot.")
    if abs(metre - 9.5) < 1e-6 and cam == "mast_left_mid":
        return dict(base, status="suspect", issues=[{
            "label": "chlorosis (possible N deficiency)", "confidence": 0.58, "bbox_norm": [0.1, 0.3, 0.5, 0.65],
            "evidence": "Uniform yellowing of older leaves; no lesions visible. Could be lighting."}],
            summary="Mild yellowing on older leaves — re-check under better light.")
    if metre == 0.0 and cam == "mast_left_high":
        return dict(base, status="not_plant", issues=[],
                    summary="Frame shows a greenhouse post and irrigation line, no plants.")
    return dict(base, status="healthy", issues=[],
                summary=rnd.choice(["Dense green canopy, no visible lesions or pests.",
                                    "Healthy leaves, uniform colour.", "No anomalies detected."]))


def _image_for(out: Path, ev: dict, findings_by_frame: dict) -> None:
    p = ev["payload"]
    f = findings_by_frame.get(p["frame_id"])
    status = f["status"] if f else "healthy"
    kind = "front" if p["camera"] == FRONT else ("other" if status == "not_plant" else "plant")
    spots = 9 if status == "disease" else 0
    label = f"{p['camera']}  {p['metre']:.1f} m  #{p['frame_id']}"
    plant_image(out / p["image_path"], p["frame_id"] * 31 + int(p["metre"] * 10), label, spots, kind)


def write_all(out: Path, map_path: Path, live: bool, speed: float, with_slam: bool) -> None:
    amap = json.loads(map_path.read_text())
    if out.exists():
        for name in ("frames.jsonl", "findings.jsonl", "decisions.jsonl", "status.json", "services.json"):
            (out / name).unlink(missing_ok=True)
        shutil.rmtree(out / "frames", ignore_errors=True)
        shutil.rmtree(out / "slam", ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "aisle_map.json").write_text(json.dumps(amap, ensure_ascii=False, indent=1))

    (out / "services.json").write_text(json.dumps({
        "nemotron": {"live": True, "model": "nemotron-super", "detail": "HERMES planner"},
        "cosmos": {"live": True, "model": "Cosmos-Reason", "detail": "plant inspection"},
        "slam": {"live": with_slam, "detail": "cuVSLAM batch job" if with_slam else "not started"}}))
    events = build_events(amap)
    findings_by_frame = {e["payload"]["frame_id"]: e["payload"] for e in events if e["kind"] == "finding"}
    total = events[-1]["t"]
    start = time.time() if live else time.time() - total
    def ts_of(t: float) -> float:
        return round(start + t / (speed if live else 1.0), 3)
    files = {k: open(out / f"{k}s.jsonl", "a", encoding="utf-8") for k in ("frame", "finding", "decision")}
    print(f"demo: {len(events)} events over {total:.0f} s simulated -> {out}", flush=True)
    try:
        for i, ev in enumerate(events):
            if live:
                wait = start + ev["t"] / speed - time.time()
                if wait > 0:
                    time.sleep(wait)
            ts = ts_of(ev["t"])
            row = dict(ev["payload"], ts=ts)
            if ev["kind"] == "status":
                tmp = out / "status.json.tmp"
                tmp.write_text(json.dumps(row))
                tmp.replace(out / "status.json")        # atomic so the server never reads half a file
                continue
            if ev["kind"] == "frame":
                _image_for(out, ev, findings_by_frame)
            files[ev["kind"]].write(json.dumps(row, ensure_ascii=False) + "\n")
            files[ev["kind"]].flush()
            if with_slam and (i == len(events) - 1 or (live and ev["kind"] == "decision")):
                write_slam(out, amap, events, ts_of, upto=ev["t"])
    finally:
        for fh in files.values():
            fh.close()
    if with_slam:
        write_slam(out, amap, events, ts_of, upto=total)



def write_slam(out: Path, amap: dict, events: list[dict], ts_of, upto: float) -> None:
    xs = [v for b in amap["bed_bounds_local_m"] for v in (b["min"][0], b["max"][0])]
    ys = [v for b in amap["bed_bounds_local_m"] for v in (b["min"][1], b["max"][1])]
    bounds = {"min": [min(xs) - 0.4, min(ys) - 0.6], "max": [max(xs) + 0.4, max(ys) + 0.6]}
    if not (out / "slam" / "topdown.png").exists():
        slam_topdown(out / "slam" / "topdown.png", amap, bounds)
    aisle = next(a for a in amap["aisles"] if a["id"] == AISLE)
    rnd = random.Random(3)
    poses = []
    for ev in events:
        if ev["kind"] != "status" or ev["t"] > upto:
            continue
        p = ev["payload"]
        poses.append({"ts": ts_of(ev["t"]), "x": round(aisle["start_local"][0] + rnd.gauss(0, 0.03), 3),
                      "y": round(aisle["start_local"][1] - p["metre"] + rnd.gauss(0, 0.05), 3),
                      "heading_deg": p["heading_deg"]})
    tmp = out / "slam" / "trajectory.json.tmp"
    tmp.write_text(json.dumps({"bounds_local": bounds, "image_axes": "+x right = local +X, top row = max local Y",
                               "poses": poses}))
    tmp.replace(out / "slam" / "trajectory.json")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="data dir to (re)create")
    ap.add_argument("--map", default=str(DEFAULT_MAP))
    ap.add_argument("--live", action="store_true", help="replay events in real time instead of writing at once")
    ap.add_argument("--speed", type=float, default=2.0, help="--live replay speed multiplier")
    ap.add_argument("--no-slam", action="store_true", help="skip slam/topdown.png + slam/trajectory.json")
    args = ap.parse_args(argv)
    write_all(Path(args.out), Path(args.map), args.live, args.speed, not args.no_slam)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
