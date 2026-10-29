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
    by: list[str] = field(default_factory=list)   # which eyes flagged it (finding meta.votes)


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
        s.by += [m for m, v in (f.get("meta") or {}).get("votes", {}).items() if v in ("disease", "suspect") and m not in s.by]
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


def decide_situation(sit: dict) -> tuple[dict | None, str]:
    """HERMES policy for a SAFETY stop explained by Cosmos Reason. Returns (command or None, why).
    The robot's SAFETY already holds the wheels and resumes by itself when the lane clears (it gives up after
    blocked_timeout_s), so HERMES never commands motion here: it waits, or tells the farmer what needs a person."""
    what, where = sit.get("what", "unknown"), sit.get("aisle") or "the yard"
    at = f"{where}" + (f" metre {sit['metre']:.1f}" if isinstance(sit.get("metre"), (int, float)) else "")
    desc = sit.get("description", "")
    if sit.get("crop_concern"):
        return ({"cmd": "notify", "message": f"Crop issue at {at}: {sit['crop_concern']} ({desc})"},
                f"Cosmos Reason saw a crop problem in the lane at {at}; a person should fix it")
    if what in ("person", "animal") or (sit.get("likely_to_move") and what != "unknown"):
        return None, f"Cosmos Reason: {what} at {at} ({desc}); it should move on, the robot waits safely"
    if what == "nothing_visible":
        return ({"cmd": "notify", "message": f"LiDAR stop at {at} but the camera sees a clear lane; check the sensor"},
                "LiDAR and camera disagree; the robot keeps waiting under SAFETY")
    if what in ("vehicle_or_cart", "crate_or_bin", "tool_or_hose", "fallen_plant_or_branch", "debris",
                "door_or_gate_closed", "water_or_mud", "other"):
        return ({"cmd": "notify", "message": f"Path blocked at {at}: {desc or what}. Please clear it; the robot waits."},
                f"Cosmos Reason: {what} blocks the lane at {at}; it will not move by itself")
    return ({"cmd": "notify", "message": f"Robot stopped at {at}; the camera report failed, please check"},
            "no usable camera report")


def run(house: str, poll_s: float = 3.0, survey_until: float | None = None, farm: bool = False,
        max_revisits: int | None = None) -> str:
    """farm=True: one drive-through survey of every house (survey_farm); revisits are queued by the robot until the
    survey is done, and at most max_revisits spots are revisited (most votes first; the rest are reported as new)."""
    since, findings, spots, stuck, quiet = 0, [], [], 0, 0
    revisits, sit_since, situations = 0, 0, []

    def send(cmd: dict, why: str) -> dict:
        cmd["reasoning"] = _nemotron(f"In one sentence, explain this robot command to a farmer: {json.dumps(cmd)}. Context: {why}", why)
        return _post("/command", cmd)

    if farm:
        send({"cmd": "survey_farm"}, "daily inspection: drive through tunnel-09 and tunnel-10, photograph every plant")
    elif not _get("/health").get("map"):
        send({"cmd": "discover_house"}, "no map of this greenhouse yet, mapping it first"); wait_idle(poll_s)
        print("discovered; confirm the map then POST {'cmd':'set_map','path':...}"); return ""
    elif survey_until is None:
        send({"cmd": "survey_house", "house": house}, "daily inspection of every reachable aisle")
    else:                                     # demo / partial survey: inspect the first part of the first reachable aisle
        aisle = next(a["id"] for a in _get("/map")["aisles"] if a["traversable"])
        send({"cmd": "go_to", "house": house, "aisle": aisle, "metre": survey_until},
             f"inspect {aisle} from the door to metre {survey_until:g}")
    while True:
        time.sleep(poll_s)
        res = _get(f"/findings?since={since}"); since = res["next"]; findings += res["findings"]
        st = _get("/status")
        try:
            sr = _get(f"/situations?since={sit_since}"); sit_since = sr["next"]
        except Exception:
            sr = {"situations": []}
        for sit in sr["situations"]:                  # Cosmos Reason explains each SAFETY stop; HERMES decides
            situations.append(sit)
            cmd, why = decide_situation(sit)
            if cmd is not None:
                send(cmd, why)
            else:
                send({"cmd": "notify", "message": why, "situation": sit.get("what"), "wait": True}, why)
        for s in [s for s in spots if s.state == "revisiting"]:
            judge_revisit(findings, s)
        for s in cluster(res["findings"], spots):
            if max_revisits is not None and revisits >= max_revisits:
                continue                          # stays "new": reported, not re-inspected this round
            s.state = "revisiting"; revisits += 1
            send({"cmd": "revisit", "house": (s.aisle or house).split("/")[0], "aisle": s.aisle, "metre": round(s.metre, 1)},
                 f"{' and '.join(s.by) or 'the plant-inspection model'} flagged {', '.join(sorted(set(s.labels)))} "
                 f"near {s.aisle} metre {s.metre:.1f}")
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
    events = [{"what": x.get("what"), "where": x.get("aisle") or "yard", "metre": x.get("metre"),
               "description": x.get("description"), "crop_concern": x.get("crop_concern")} for x in situations]
    report = _nemotron(
        "Write a short farmer report (Thai then English) from these greenhouse inspection spots. Spot states mean: "
        "confirmed = the robot re-inspected and still sees the symptoms (Thai: ยืนยันว่าพบอาการ); "
        "cleared = the re-inspection did not see the symptoms (Thai: ตรวจซ้ำแล้วไม่พบอาการ — NOT treated/removed); "
        "revisiting/new = not yet re-inspected. Never claim the disease was treated or removed. List the labels the "
        "model reported, recommend a person checks the plant, and note this is not a lab diagnosis. Also list, under a "
        "separate heading, what stopped the robot on the way (from its camera reports) and what a person should do. "
        "Spots: " + json.dumps(summary, ensure_ascii=False) + " Stops: " + json.dumps(events, ensure_ascii=False),
        json.dumps({"spots": summary, "stops": events}, ensure_ascii=False, indent=1))
    print(report)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--house", default="tunnel-09"); ap.add_argument("--poll", type=float, default=3.0)
    ap.add_argument("--survey-until", type=float, help="partial survey: go_to this metre of the first reachable aisle")
    ap.add_argument("--farm", action="store_true", help="drive-through survey of the whole farm (survey_farm)")
    ap.add_argument("--max-revisits", type=int, help="cap revisits (farm runs: each one is a trip back into a house)")
    a = ap.parse_args(); run(a.house, a.poll, a.survey_until, a.farm, a.max_revisits)
