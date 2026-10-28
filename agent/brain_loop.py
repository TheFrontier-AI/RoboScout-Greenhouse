"""Deterministic HERMES fallback: the same decision procedure as hermes_skills/r1-robot-control, with Nemotron
(Token Factory, via cloud/brain) only writing the `reasoning` lines and the final farmer report. Use it if the Hermes
runtime misbehaves during the demo; the robot API cannot tell the difference.

  ROBOT_API_URL=http://127.0.0.1:8090 python3 agent/brain_loop.py --house tunnel-09"""
from __future__ import annotations
import argparse, json, os, sys, time, urllib.request
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # repo root, for cloud.brain when run as a script

API = os.environ.get("ROBOT_API_URL", "http://127.0.0.1:8090")
MERGE_M, DISEASE_CONF, BATTERY_MIN = 1.0, 0.6, 25
SUSPECT_MIN = 1               # two-eye screening is tuned for recall: one suspect shot is enough to go back and look
REVISIT_SHOTS = 6             # all side cameras fire on a revisit hold (r1_nav.cameras.SIDE_CAMERAS)


def _get(path: str):
    with urllib.request.urlopen(API + path, timeout=10) as r:
        return json.loads(r.read() or b"null")


def _post(path: str, obj: dict):
    req = urllib.request.Request(API + path, json.dumps(obj).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _nemotron(prompt: str, fallback: str) -> str:
    try:
        from cloud.brain.token_factory import chat
        out = chat([{"role": "system", "content": "You are HERMES, a concise greenhouse robot planner."},
                    {"role": "user", "content": prompt}])
        text = out if isinstance(out, str) else getattr(out, "content", None)   # cloud.brain.token_factory.Reply
        return (text or "").strip() or fallback
    except Exception:
        return fallback


@dataclass
class Spot:
    aisle: str
    metre: float
    labels: list[str] = field(default_factory=list)
    best: float = 0.0
    n_suspect: int = 0
    state: str = "new"         # new | revisiting | confirmed | cleared


def cluster(findings: list[dict], spots: list[Spot]) -> list[Spot]:
    """Merge disease/suspect findings within MERGE_M of each other into spots; return spots that now need a revisit."""
    for f in findings:
        if f.get("status") not in ("disease", "suspect") or f.get("purpose") == "revisit" or f.get("metre") is None:
            continue
        s = next((s for s in spots if s.aisle == f["aisle"] and abs(s.metre - f["metre"]) <= MERGE_M), None)
        if s is None:
            s = Spot(f["aisle"], float(f["metre"])); spots.append(s)
        conf = max((i.get("confidence", 0.0) for i in f.get("issues", [])), default=0.0)
        s.labels += [i.get("label", "?") for i in f.get("issues", [])]
        s.best = max(s.best, conf if f["status"] == "disease" else 0.0)
        s.n_suspect += f["status"] == "suspect"
    return [s for s in spots if s.state == "new" and (s.best >= DISEASE_CONF or s.n_suspect >= SUSPECT_MIN)]


def judge_revisit(findings: list[dict], spot: Spot, need: int = REVISIT_SHOTS) -> None:
    """Decide only once every side camera's revisit shot is back (they arrive one by one; an early healthy
    shot from the other bed must not clear the spot). Any disease/suspect revisit shot confirms it."""
    near = [f for f in findings if f.get("purpose") == "revisit" and f.get("aisle") == spot.aisle
            and abs(f.get("metre", 1e9) - spot.metre) <= MERGE_M]
    if any(f.get("status") in ("disease", "suspect") for f in near):
        spot.state = "confirmed"
        spot.labels += [i.get("label", "?") for f in near for i in f.get("issues", [])]
    elif len(near) >= need:
        spot.state = "cleared"


def wait_idle(poll_s: float) -> dict:
    while True:
        st = _get("/status")
        if st.get("state") in ("done", "idle", "stuck", "aborted") and not st.get("legs_remaining"):
            return st
        time.sleep(poll_s)


def run(house: str, poll_s: float = 3.0, survey_until: float | None = None) -> str:
    since, findings, spots, stuck, quiet = 0, [], [], 0, 0

    def send(cmd: dict, why: str) -> dict:
        cmd["reasoning"] = _nemotron(f"In one sentence, explain this robot command to a farmer: {json.dumps(cmd)}. Context: {why}", why)
        return _post("/command", cmd)

    if not _get("/health").get("map"):
        send({"cmd": "discover_house"}, "no map of this greenhouse yet, mapping it first"); wait_idle(poll_s)
        print("discovered; confirm the map then POST {'cmd':'set_map','path':...}"); return ""
    if survey_until is None:
        send({"cmd": "survey_house", "house": house}, "daily inspection of every reachable aisle")
    else:                                     # demo / partial survey: inspect the first part of the first reachable aisle
        aisle = next(a["id"] for a in _get("/map")["aisles"] if a["traversable"])
        send({"cmd": "go_to", "house": house, "aisle": aisle, "metre": survey_until},
             f"inspect {aisle} from the door to metre {survey_until:g}")
    while True:
        time.sleep(poll_s)
        res = _get(f"/findings?since={since}"); since = res["next"]; findings += res["findings"]
        st = _get("/status")
        for s in [s for s in spots if s.state == "revisiting"]:
            judge_revisit(findings, s)
        for s in cluster(res["findings"], spots):
            s.state = "revisiting"
            send({"cmd": "revisit", "house": house, "aisle": s.aisle, "metre": round(s.metre, 1)},
                 f"Cosmos flagged {', '.join(sorted(set(s.labels)))} near {s.aisle} metre {s.metre:.1f}")
        if st.get("state") == "stuck":
            stuck += 1; send({"cmd": "stop"}, "robot reports it is stuck; holding for a human")
            if stuck >= 2:
                break
        batt = st.get("battery")
        if batt is not None and (batt * 100 if batt <= 1 else batt) < BATTERY_MIN:   # accept 0–1 or %
            break
        robot_idle = st.get("state") in ("done", "idle") and not st.get("legs_remaining")
        quiet = quiet + 1 if robot_idle and not res["findings"] else 0
        if quiet >= 3:                        # robot parked, no shots arriving: judge revisits on what we have
            for s in [s for s in spots if s.state == "revisiting"]:
                judge_revisit(findings, s, need=1)
                if s.state == "revisiting":
                    s.state = "uncertain"
        if robot_idle and not any(s.state == "revisiting" for s in spots):
            break
    send({"cmd": "return_to_dock"}, "inspection finished")
    summary = [{"aisle": s.aisle, "metre": s.metre, "labels": sorted(set(s.labels)), "state": s.state} for s in spots]
    report = _nemotron(
        "Write a short farmer report (Thai then English) from these greenhouse inspection spots. Spot states mean: "
        "confirmed = the robot re-inspected and still sees the symptoms (Thai: ยืนยันว่าพบอาการ); "
        "cleared = the re-inspection did not see the symptoms (Thai: ตรวจซ้ำแล้วไม่พบอาการ — NOT treated/removed); "
        "revisiting/new = not yet re-inspected. Never claim the disease was treated or removed. List the labels the "
        "model reported, recommend a person checks the plant, and note this is not a lab diagnosis. Spots: "
        + json.dumps(summary, ensure_ascii=False), json.dumps(summary, ensure_ascii=False, indent=1))
    print(report)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--house", default="tunnel-09"); ap.add_argument("--poll", type=float, default=3.0)
    ap.add_argument("--survey-until", type=float, help="partial survey: go_to this metre of the first reachable aisle")
    a = ap.parse_args(); run(a.house, a.poll, a.survey_until)
