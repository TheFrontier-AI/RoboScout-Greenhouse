#!/usr/bin/env python3
"""Put down anywhere, then survey the farm with NO stored route: the robot reads its own map (scripts/free_nav.py) to
find the plant aisles and the drivable space, a decision-maker chooses the mission order from that structure, the
robot plans every path itself (A*) from where it believes it is, and it changes the plan when something unexpected
happens: an obstacle that is not on its map (drives around it), a belief that the scans contradict (searches the whole
map again, re-places its history, relabels its records, re-derives the mission), a dock that the camera says is not
the dock (relabels and drives to the real one).

  python3 scripts/freenav_mission_eval.py --scans scans_all.jsonl --rec <run>/rec --cases vlm_offpath.jsonl --out <dir>
         [--far 10] [--surprise] [--decider nearest|nemotron]"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.relocalize import HypothesisTracker, Relocalizer                       # noqa: E402
from sim.scan_localizer import LIDAR_AHEAD_M                                    # noqa: E402
from scripts.free_nav import FreeSpace, build as build_structure                # noqa: E402
from scripts.offpath_mission_eval import World, compose, inv, rel_motion, resample, nearest_ref, camera_pick  # noqa: E402

STEP_M, NODE_M, STOP_PITCH, AWAY_M, TWIN_M = 0.25, 1.5, 0.45, 3.0, 15.0   # TWIN_M: a camera rival must be a different place, not a spot nearby
G: dict = {}


class FreeFarm:
    """What the robot knows from its own map: drivable space, plant aisles, the dock. No route."""

    def __init__(self, fs: FreeSpace, aisles, dock):
        self.fs = fs; self.dock = tuple(dock); self.stops = {}
        # only aisles the robot can actually drive: both ends reachable from the dock on the free grid
        self.aisles = [a for a in aisles if a["kind"] == "plant aisle" and fs.astar(self.dock[:2], a["a"]) is not None and fs.astar(self.dock[:2], a["b"]) is not None]
        for a in self.aisles:
            A, B = np.array(a["a"]), np.array(a["b"]); n = int(np.linalg.norm(B - A) / STOP_PITCH)
            self.stops[a["id"]] = np.array([A + (B - A) * k * STOP_PITCH / np.linalg.norm(B - A) for k in range(n)])

    def goal_points(self, order):
        """Mission as goal points: each aisle end to end (the end nearer the previous goal first), then the dock."""
        pts = []; cur = None
        for aid in order:
            a = next(x for x in self.aisles if x["id"] == aid)
            if cur is None or math.dist(cur, a["a"]) <= math.dist(cur, a["b"]):
                pts += [(f"{aid}/start", a["a"]), (f"{aid}/end", a["b"])]; cur = a["b"]
            else:
                pts += [(f"{aid}/start", a["b"]), (f"{aid}/end", a["a"])]; cur = a["a"]
        pts.append(("dock", self.dock[:2])); return pts

    def plan(self, pose, goals):
        """Polyline through the remaining goals, every leg planned on the free grid from where the robot believes it
        is. A leg from an aisle's start to its end is the survey itself, so it is planned inside that aisle."""
        cur = tuple(pose[:2]); pts = [np.array(cur, float)]; prev = None
        for name, g in goals:
            if name.endswith("/end") and prev == name[:-4] + "/start":
                leg = self.astar_in_aisle(cur, g, name[:-4])
            else:
                leg = self.fs.astar(cur, g)
            if leg is None:
                return None
            pts.extend(np.array(leg[1:], float)); cur = tuple(g); prev = name
        return np.array(pts, float)

    def astar_in_aisle(self, start, goal, aid):
        """A* with the free grid masked to the aisle corridor (1.2 m either side of its centre line)."""
        a = next(x for x in self.aisles if x["id"] == aid); A, B = np.array(a["a"]), np.array(a["b"]); u = (B - A) / np.linalg.norm(B - A)
        ys, xs = np.mgrid[0:self.fs.ny, 0:self.fs.nx]; cx = self.fs.x0 + (xs + 0.5) * self.fs.cell; cy = self.fs.y0 + (ys + 0.5) * self.fs.cell
        d = np.stack([cx - A[0], cy - A[1]], -1); t = d @ u; off = np.abs(d[..., 0] * u[1] - d[..., 1] * u[0])
        mask = (off <= 1.2) & (t >= -1.5) & (t <= np.linalg.norm(B - A) + 1.5)
        saved = self.fs.free; self.fs.free = saved & mask
        try:
            return self.fs.astar(start, goal)
        finally:
            self.fs.free = saved

    def where(self, pose):
        for a in self.aisles:
            A, B = np.array(a["a"]), np.array(a["b"]); d = B - A; L = np.linalg.norm(d); u = d / L
            t = float(np.dot(np.array(pose[:2]) - A, u)); off = abs(float(np.cross(u, np.array(pose[:2]) - A)))
            if -1.0 <= t <= L + 1.0 and off < 1.0:
                return a["id"], round(t, 2)
        return None, None


def _last_json(txt: str) -> dict:
    """The last complete JSON object in a model reply (replies may carry reasoning text and quoted examples)."""
    end = txt.rfind("}")
    while end > 0:
        depth = 0
        for i in range(end, -1, -1):
            if txt[i] == "}": depth += 1
            elif txt[i] == "{":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(txt[i:end + 1])
                    except Exception:
                        break
        end = txt.rfind("}", 0, end)
    raise ValueError("no JSON object in the reply")


def plan_review(farm: FreeFarm, start_xy, order, decider="nemotron"):
    """HERMES checks the concrete plan before the robot moves: it sees every leg the robot planned (from, to, metres),
    the alternatives (other orders with their total length) and a picture of the plan on the map, and approves or
    changes the order. Returns (order, verdict text)."""
    import itertools, os
    legs_of = lambda o: [(a, b, farm.fs.astar(farm.dock[:2] if False else prev, g)) for (a, b), prev, g in []]
    def legs(o):
        pts = farm.goal_points(o); cur = tuple(start_xy); out = []
        for name, g in pts:
            leg = farm.fs.astar(cur, g); L = None if leg is None else sum(math.dist(p, q) for p, q in zip(leg, leg[1:]))
            out.append({"to": name, "metres": None if L is None else round(L, 1)}); cur = tuple(g)
        return out
    cands = {}
    for o in itertools.permutations([a["id"] for a in farm.aisles]):
        lg = legs(list(o))
        if all(x["metres"] is not None for x in lg):
            cands[" -> ".join(o)] = {"order": list(o), "legs": lg, "total_m": round(sum(x["metres"] for x in lg), 1)}
    if decider != "nemotron" or not cands:
        return order, "no review"
    try:
        import base64, io, os
        from openai import OpenAI
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        # the picture HERMES sees: the robot's map, its drivable space, the aisles, the dock and the proposed path
        fs = farm.fs; fig, ax = plt.subplots(figsize=(5, 8), dpi=100); ax.set_facecolor("#111")
        ax.imshow(fs.free, origin="lower", extent=(fs.x0, fs.x0 + fs.nx * fs.cell, fs.y0, fs.y0 + fs.ny * fs.cell), cmap="Greens", alpha=0.5, vmin=0, vmax=1.3)
        ax.scatter(fs.map[:, 0], fs.map[:, 1], s=0.3, c="#aaa", lw=0)
        pts = farm.goal_points(order); cur = tuple(start_xy); P = [cur]
        for name, g in pts:
            leg = fs.astar(cur, g); P += list(leg[1:]) if leg else [g]; cur = tuple(g)
        P = np.array(P); ax.plot(P[:, 0], P[:, 1], c="#2ecc71", lw=2)
        for a in farm.aisles:
            ax.annotate(a["id"], ((a["a"][0] + a["b"][0]) / 2, (a["a"][1] + a["b"][1]) / 2), color="#ffd33d", fontsize=8, rotation=90)
        ax.plot(start_xy[0], start_xy[1], "o", c="#ff7b72", ms=8); ax.annotate("robot", start_xy, color="#ff7b72", fontsize=8, xytext=(5, 5), textcoords="offset points")
        ax.plot(farm.dock[0], farm.dock[1], "*", c="#ffd33d", ms=14); ax.annotate("dock", farm.dock[:2], color="#ffd33d", fontsize=8, xytext=(5, 5), textcoords="offset points")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="jpeg", facecolor="#111"); plt.close(fig)
        img = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        client = OpenAI(base_url=os.environ.get("EYES2_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"), api_key=os.environ.get("EYES2_API_KEY") or os.environ.get("NEBIUS_API_KEY"), timeout=180)
        msg = ("You are HERMES, the mission supervisor of a greenhouse survey robot. The robot has planned its survey on the map it built itself; the image shows that map "
               "(grey = LiDAR points, green area = space the robot has seen empty, yellow labels = the plant aisles it found, red dot = the robot, star = the dock, green line = the proposed path). "
               "Check the plan before it moves: every plant aisle must be driven end to end exactly once, the robot must end at the dock, the path must stay in the green area, and the driving should be short. "
               "Here is the plan it proposes and the alternatives it computed, each leg with the metres the planner measured on the map.\n"
               + json.dumps({"proposed": " -> ".join(order), "alternatives": cands}) +
               "\nReply with JSON only: {\"approve\": true/false, \"order\": [aisle ids], \"reason\": \"one sentence\"}.")
        r = client.chat.completions.create(model=os.environ.get("EYES2_MODEL", "nvidia/Nemotron-3.5-Super-VL"), temperature=0.1, max_tokens=6000,
                                           messages=[{"role": "user", "content": [{"type": "text", "text": msg}, {"type": "image_url", "image_url": {"url": img}}]}])
        m = r.choices[0].message; txt = (m.content or getattr(m, "reasoning_content", None) or "").split("</think>")[-1].replace("```json", "").replace("```", "")
        j = _last_json(txt)
        ids = {a["id"] for a in farm.aisles}; new = [x for x in j.get("order", []) if x in ids]
        if not j.get("approve", True) and len(new) == len(ids):
            return new, f"HERMES changed the plan: {str(j.get('reason', ''))[:140]}"
        return order, f"HERMES approved the plan: {str(j.get('reason', ''))[:140]}"
    except Exception as e:
        return order, f"review unavailable ({str(e)[:60]})"


def decide_order(farm: FreeFarm, start_xy, decider="nearest"):
    """Mission-level decision: which aisles, in what order. 'nearest' = nearest-first; 'nemotron' = the agent's model
    (Nemotron 3 Super via Nebius Token Factory) chooses from the structure the robot found, with its reasoning kept."""
    todo = list(farm.aisles); order = []; cur = tuple(start_xy)
    while todo:
        k = min(range(len(todo)), key=lambda i: min(math.dist(cur, todo[i]["a"]), math.dist(cur, todo[i]["b"])))
        a = todo.pop(k); order.append(a["id"]); cur = a["b"] if math.dist(cur, a["a"]) <= math.dist(cur, a["b"]) else a["a"]
    if decider == "rpp":
        from agent.mission_planner import MissionPlanner
        mp = MissionPlanner.__new__(MissionPlanner); mp.fs = farm.fs; mp.dock = farm.dock; mp.aisles = farm.aisles; mp.astar_in_aisle = farm.astar_in_aisle
        r = mp.rpp_order(start_xy)
        return (r["order"], f"Rural Postman solver on the robot ({r['method']}, {r['total_m']} m)") if r["ok"] else (order, "nearest first (RPP found no closed route)")
    if decider != "nemotron":
        return order, "nearest aisle first, then the next nearest, then the dock"
    try:
        from openai import OpenAI
        client = OpenAI(base_url=os.environ.get("HERMES_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"), api_key=os.environ.get("NEBIUS_API_KEY"), timeout=120)
        structure = {"robot_position_xy_m": [round(start_xy[0], 1), round(start_xy[1], 1)], "dock_xy_m": [round(farm.dock[0], 1), round(farm.dock[1], 1)],
                     "plant_aisles_found": [{"id": a["id"], "from_xy": [round(a["a"][0], 1), round(a["a"][1], 1)], "to_xy": [round(a["b"][0], 1), round(a["b"][1], 1)], "length_m": a["length_m"], "width_m": a["width_m"]} for a in farm.aisles]}
        msg = ("You are the mission planner of a greenhouse survey robot. The robot built this map itself and found these plant aisles. "
               "The mission: inspect every plant in every aisle (drive each aisle end to end once), then return to the dock, with the least driving. "
               "Reply with JSON only: {\"order\": [aisle ids in the order to survey], \"reason\": \"one sentence\"}.\n" + json.dumps(structure))
        r = client.chat.completions.create(model=os.environ.get("HERMES_MODEL", "nvidia/nemotron-3-super-120b-a12b"), temperature=0.1, max_tokens=6000, messages=[{"role": "user", "content": msg}])
        m = r.choices[0].message; txt = (m.content or getattr(m, "reasoning_content", None) or "").split("</think>")[-1].replace("```json", "").replace("```", "")
        j = _last_json(txt)
        ids = {a["id"] for a in farm.aisles}
        chosen = [x for x in j.get("order", []) if x in ids] + [x for x in order if x not in j.get("order", [])]
        return chosen, "Nemotron: " + str(j.get("reason", ""))[:160]
    except Exception as e:
        return order, f"nearest first (model unavailable: {str(e)[:60]})"


def run_case(case):
    world, reloc, farm = G["world"], G["reloc"], G["farm"]; fs = farm.fs
    fs.free = G["free0"].copy()                                      # a fresh grid per case: local obstacle marks do not carry over
    truth = tuple(case["truth"]); err = lambda b, t: math.hypot(b[0] - t[0], b[1] - t[1])
    surprise_xy = None
    if case.get("final_pose"):
        belief = tuple(case["final_pose"]); how = "camera pick"
    elif case.get("lidar", {}).get("pose") and case.get("tag") != "far":
        belief = tuple(case["lidar"]["pose"]); how = "LiDAR best (camera undecided)"
    else:
        r = reloc.locate(world.scan(*truth)); belief = tuple(r.pose) if r.pose else truth; how = "LiDAR best" + (" (ambiguous)" if r.ambiguous else "")
    true = truth
    order, why = decide_order(farm, belief[:2], G.get("decider", "nearest"))
    order, verdict = plan_review(farm, belief[:2], order, "nemotron" if G.get("review") else "none")
    goals = farm.goal_points(order); all_goals = list(goals)
    log = {"true_path": [list(truth)], "belief_err": [round(err(belief, true), 2)], "events": [{"m": 0.0, "what": f"mission decided: {' -> '.join(order)} -> dock ({why})"}, {"m": 0.0, "what": f"plan checked before moving: {verdict}"}], "visits": [], "first_belief": how,
           "mission_order": order, "decider": why, "plan_review": verdict, "surprise_xy": surprise_xy}
    tr = HypothesisTracker(reloc); odom = (0.0, 0.0, 0.0); tr.start(world.scan(*truth), odom)
    tr.hyps.sort(key=lambda r: err(r, belief)); tr.hyps[0][0:3] = list(belief)
    metres = 0.0; replans = 0; corrected_at = None; outcome = "ran_out"; hist = []
    state = {"belief": belief, "wp": None, "k": 1}; local_blocks = []

    def record(e):
        log["belief_err"].append(round(e, 2)); log["true_path"].append([round(v, 3) for v in true])

    def replan():
        p = farm.plan(state["belief"], goals)
        if p is None:
            return False
        state["wp"] = resample(p, STEP_M); state["k"] = 1; return True

    def goals_left(corrected):
        """Which goals the corrected believed history has reached, in order; an aisle's end counts only when the
        corrected history really covered that aisle (>= 90 % of its plant stops)."""
        j = 0; ct = cKDTree(np.array([c[:2] for c in corrected])) if corrected else None
        for n, (name, g) in enumerate(all_goals):
            hit = next((i for i in range(j, len(corrected)) if math.dist(corrected[i][:2], g) < NODE_M), None)
            if hit is None:
                return list(all_goals[n:])
            if name.endswith("/end") and ct is not None and float((ct.query(farm.stops[name[:-4]])[0] < 0.6).mean()) < 0.9:
                return list(all_goals[n - 1:]) if n >= 1 and all_goals[n - 1][0] == name[:-4] + "/start" else list(all_goals[n:])
            j = hit
        return []

    def re_anchor(nb, reason):
        nonlocal replans, corrected_at
        old = state["belief"]; state["belief"] = nb; replans += 1
        T = compose(nb, inv(odom)); corrected = [compose(T, o) for o in hist]
        for v in log["visits"]:
            v[3], v[4] = farm.where(corrected[v[5]])
        before = len(goals); goals[:] = goals_left(corrected)
        log["events"].append({"m": round(metres, 1), "what": f"{reason}: now believes ({nb[0]:.1f}, {nb[1]:.1f}), error {err(nb, true):.1f} m; relabelled {len(log['visits'])} plant records" + (f"; mission goals left {before} -> {len(goals)}" if len(goals) != before else "")})
        tr.hyps = [h for h in tr.hyps if err(h, old) >= AWAY_M] or tr.hyps
        tr.hyps.sort(key=lambda r: err(r, nb)); tr.hyps[0][0:3] = list(nb)
        e = err(nb, true); corrected_at = round(metres, 1) if e < 1.0 else None
        return bool(goals) and replan()

    def research(reason):
        old = state["belief"]; scan = world.scan(*true); r = reloc.locate(scan)
        cands = ([r.pose] if r.pose else []) + [a["pose"] for a in r.alternatives]
        cands = [c for c in cands if err(c, old) >= AWAY_M] or cands
        tr._restart = True; tr.start(scan, odom); tr._restart = False
        for c in cands:
            if farm.plan(c, goals) is not None:
                return re_anchor(tuple(c), reason + ": searched the whole map again" + (" (ambiguous)" if r.ambiguous else ""))
        return False

    def probe():
        nonlocal true, odom, metres
        sc = world.scan(*true)
        if len(sc) == 0:
            return False
        d = np.hypot(sc[:, 0], sc[:, 1]); a = math.degrees(math.atan2(sc[d.argmax(), 1], sc[d.argmax(), 0])); moved = 0.0
        true = compose(true, (0.0, 0.0, a)); odom = compose(odom, (0.0, 0.0, a))
        for _ in range(8):
            nxt = compose(true, (STEP_M, 0.0, 0.0))
            if world.blocked(nxt[0], nxt[1]):
                break
            true = nxt; odom = compose(odom, (STEP_M, 0.0, 0.0)); metres += STEP_M; moved += STEP_M; record(err(state["belief"], true)); hist.append(odom)
        log["events"].append({"m": round(metres, 1), "what": f"probe: moved {moved:.1f} m into open space, searching again"})
        return moved > 0

    def recover(reason):
        ok = research(reason); probes = 0
        while not ok and probes < 5 and probe():
            probes += 1; ok = research("after the probe")
        return ok

    if G.get("surprise") and replan():
        # something that is not on the map: a 0.8 m block dropped on the robot's own planned path, in the open
        rng = np.random.default_rng(case.get("seed", 0) + 11); wp = state["wp"]
        cand = [i for i in range(len(wp)) if fs.tree.query(wp[i])[0] >= 1.2 and farm.where((wp[i][0], wp[i][1], 0))[0] is None and 8 < i < len(wp) - 8]
        if cand:
            i = cand[rng.integers(len(cand))]; sx, sy = float(wp[i][0]), float(wp[i][1])
            # placed in the TRUE world along the direction the robot will really travel from its true start
            T = compose(true, inv(state["belief"])); sx, sy, _ = compose(T, (sx, sy, 0.0))
            blk = np.array([(sx + dx, sy + dy) for dx in np.arange(-0.4, 0.41, 0.1) for dy in np.arange(-0.4, 0.41, 0.1)])
            world = World(np.vstack([world.m, blk])); surprise_xy = (round(sx, 2), round(sy, 2)); log["surprise_xy"] = surprise_xy
    log["first_belief_pose"] = [round(v, 3) for v in belief]
    _p0 = farm.plan(belief, goals); log["initial_plan_xy"] = [[round(x, 2), round(y, 2)] for x, y in (_p0[:: max(1, len(_p0) // 300)] if _p0 is not None else [])]
    if not replan():
        log["events"].append({"m": 0.0, "what": "cannot plan from the believed pose"})
        if not recover("cannot plan from the believed pose"):
            return log | {"outcome": "stays_put", "metres": 0.0, "corrected_at_m": None, "replans": replans, "coverage": {a["id"]: 0.0 for a in farm.aisles}, "aisle_steps_logged_right": None,
                          "final_belief_err_m": round(err(state["belief"], true), 2), "at_dock": False, "unexpected_obstacle": None}
    unexpected = None; iters = 0; last_m = -1.0; stalls = 0
    while metres < 360.0 and replans <= 14 and iters < 6000:
        iters += 1
        if metres == last_m:
            stalls += 1
            if stalls > 40:                                   # not moving for 40 rounds: something is wrong with this place
                log["events"].append({"m": round(metres, 1), "what": "stuck: no progress for 40 rounds, searching again"})
                stalls = 0
                if not recover("stuck in place"):
                    break
        else:
            stalls = 0; last_m = metres
        if not goals:
            outcome = "mission_done"; break
        belief = state["belief"]; wp = state["wp"]; k = state["k"]
        if wp is None or k >= len(wp):
            break
        tgt = wp[k]; hb = math.degrees(math.atan2(tgt[1] - belief[1], tgt[0] - belief[0]))
        nxt_b = (float(tgt[0]), float(tgt[1]), hb); rel = rel_motion(belief, nxt_b); nxt_t = compose(true, rel)
        if world.blocked(nxt_t[0], nxt_t[1]) and fs.tree.query(nxt_b[:2])[0] >= 0.5:
            scan = world.scan(*true); fit = reloc._score(np.asarray(scan) + (LIDAR_AHEAD_M, 0.0), *belief)
            if fit >= 0.7:
                # the place fits the map well: this is a new obstacle, not a wrong belief -> mark it and plan around it
                i, j = fs.cell_of(nxt_b[:2]); bi, bj = fs.cell_of(belief[:2]); keep = fs.free[bi, bj]
                fs.free[max(0, i - 1):i + 2, max(0, j - 1):j + 2] = False; fs.free[bi, bj] = keep; local_blocks.append(nxt_b[:2])
                log["events"].append({"m": round(metres, 1), "what": f"unexpected obstacle at ({nxt_b[0]:.1f}, {nxt_b[1]:.1f}), not on the map: marked it and planned around"})
                unexpected = unexpected or round(metres, 1)
                if not replan() and not recover("no way round the obstacle from the believed place"):
                    break
                continue
            if not recover("obstacle where the map said free and the scans do not fit the believed place"):
                break
            record(err(state["belief"], true)); continue
        true = nxt_t; odom = compose(odom, rel); metres += math.hypot(rel[0], rel[1]); state["k"] = k + 1
        scan_now = world.scan(*true); t = tr.update(scan_now, odom)
        if state["k"] % 40 == 0:
            r = reloc.locate(scan_now)
            for c in ([r.pose] if r.pose else []) + [a["pose"] for a in r.alternatives]:
                if all(err(c, h) > 3.0 for h in tr.hyps):
                    tr.hyps.append([c[0], c[1], c[2], reloc._score(np.asarray(scan_now) + (LIDAR_AHEAD_M, 0.0), *c), 1])
            tr.hyps = tr.hyps[: tr.keep + 4]
        alive = [h for h in tr.hyps if err(h, nxt_b) < 2.5]; body_now = np.asarray(scan_now) + (LIDAR_AHEAD_M, 0.0)
        if not alive:
            (fx, fy, fh), _q = reloc.loc.localize(np.asarray(scan_now), *nxt_b); sc = reloc._score(body_now, fx, fy, fh)
            if sc >= tr.drop_below:
                tr.hyps.insert(0, [fx, fy, fh, sc, 1]); alive = [tr.hyps[0]]
        rival = None
        if alive:
            b_mean = alive[0][3] / alive[0][4]; far = [h for h in tr.hyps if err(h, nxt_b) >= AWAY_M and h[4] >= 8]
            if far:
                best = max(far, key=lambda h: h[3] / h[4])
                if best[3] / best[4] - b_mean >= tr.decide_margin:
                    rival = best
        if alive and rival is not None:
            if not re_anchor((rival[0], rival[1], rival[2]), f"a rival place fits the scans clearly better (margin {rival[3] / rival[4] - b_mean:.2f})"):
                if goals and not recover("cannot plan after switching place"):
                    break
        elif alive:
            state["belief"] = (alive[0][0], alive[0][1], alive[0][2])
        elif not recover("believed place no longer fits the scans"):
            break
        belief = state["belief"]; e = err(belief, true); record(e); hist.append(odom)
        # EYES WHILE DRIVING: every G["eyes_every"] metres, if the LiDAR still keeps a far-away rival place alive, the
        # camera compares the live view with "here" and with that rival; two rival verdicts in a row move the belief
        if G.get("eyes_every") and G.get("refs") and metres - state.get("last_eyes_m", -1e9) >= G["eyes_every"]:
            state["last_eyes_m"] = metres
            farr = sorted([h for h in tr.hyps if err(h, belief) >= AWAY_M and h[4] >= 4], key=lambda h: -h[3] / h[4])
            if farr:
                rv = farr[0]; live = nearest_ref(true); ra = nearest_ref(belief, away_from_xy=live[2] if live else None); rb = nearest_ref(rv, away_from_xy=live[2] if live else None)
                if live and ra and rb:
                    pick = camera_pick(live[1], ra[1], rb[1]); votes = state.setdefault("eye_votes", [])
                    votes.append(pick == "B"); log["events"].append({"m": round(metres, 1), "what": f"eyes on the way: live view vs here vs rival ({rv[0]:.1f}, {rv[1]:.1f}) -> {pick or 'no answer'}"})
                    if len(votes) >= 2 and votes[-1] and votes[-2]:
                        state["eye_votes"] = []
                        if not re_anchor((rv[0], rv[1], rv[2]), "the camera says twice that this is the rival place"):
                            if goals and not recover("cannot plan after the camera check"):
                                break
                        continue
        inside_now = farm.where(belief)[0]
        if "inside_prev" not in state:
            state["inside_prev"] = inside_now
        inside_prev = state["inside_prev"]
        if inside_now != inside_prev:
            state.setdefault("transitions", []).append({"m": round(metres, 1), "what": (f"entered {inside_now}" if inside_now else f"left {inside_prev} into the open")})
            state["inside_prev"] = inside_now
            if G.get("refs") and G.get("context_decide", True):
                r = reloc.locate(scan_now)
                rivals = [c for c in ([r.pose] if r.pose else []) + [a["pose"] for a in r.alternatives] if err(c, belief) >= TWIN_M]
                for h in sorted([h for h in tr.hyps if err(h, belief) >= TWIN_M and h[4] >= 4], key=lambda h: -h[3] / max(h[4], 1))[:1]:
                    if all(err(h, c) >= AWAY_M for c in rivals): rivals.insert(0, (h[0], h[1], h[2]))
                if rivals:
                    rv = rivals[0]; live = nearest_ref(true); ra = nearest_ref(belief, away_from_xy=live[2] if live else None); rb = nearest_ref(rv, away_from_xy=live[2] if live else None)
                    if live and ra and rb:
                        ctx = {"metres_driven": round(metres, 1), "mission_order": order, "goals_left": [g[0] for g in goals][:3],
                               "believed_place_xy": [round(belief[0], 1), round(belief[1], 1)], "rival_place_xy": [round(rv[0], 1), round(rv[1], 1)],
                               "lidar_fit_here": round(float(reloc._score(np.asarray(scan_now) + (LIDAR_AHEAD_M, 0.0), *belief)), 2), "lidar_fit_rival": round(float(reloc._score(np.asarray(scan_now) + (LIDAR_AHEAD_M, 0.0), *rv)), 2),
                               "doors_passed": state["transitions"][-6:], "camera_verdicts_so_far": [e["what"] for e in log["events"] if "camera" in e["what"] or "agent decided" in e["what"]][-4:],
                               "first_belief": log["first_belief"]}
                        where, why = context_decide(live[1], ra[1], rb[1], ctx)
                        log["events"].append({"m": round(metres, 1), "what": f"agent decided at the doorway: {where} ({why[:90]})"})
                        # a move is accepted only when the agent says 'rival' at two doorways in a row AND the LiDAR
                        # fits the rival place at least as well as the believed one; 'here' clears the count
                        votes = state.setdefault("rival_votes", 0)
                        state["rival_votes"] = votes + 1 if (where == "rival" and ctx["lidar_fit_rival"] >= ctx["lidar_fit_here"] - 0.05) else (0 if where == "here" else votes)
                        if state["rival_votes"] >= 2:
                            state["rival_votes"] = 0
                            if not re_anchor((rv[0], rv[1], rv[2]), "the agent concluded from the whole situation that this is the rival place"):
                                if goals and not recover("cannot plan after the agent's decision"):
                                    break
                            continue
        if state["wp"] is not None and state["k"] < len(state["wp"]) and math.dist(belief[:2], state["wp"][state["k"]]) > 1.0:
            log["events"].append({"m": round(metres, 1), "what": "off the planned path by more than 1 m: planned again from here"}); replan()
        if e < 1.0 and corrected_at is None:
            corrected_at = round(metres, 1)
        if e >= 1.0:
            corrected_at = None
        bh, bm = farm.where(belief); th, tm = farm.where(true)
        if th is not None:
            log["visits"].append([round(metres, 1), th, tm, bh, bm, len(hist) - 1])
        if goals and math.dist(belief[:2], goals[0][1]) < NODE_M:
            name = goals[0][0]
            log["events"].append({"m": round(metres, 1), "what": f"believes it reached {name} (true error {err(belief, true):.1f} m)"})
            if name == "dock" and G.get("refs") and name not in state.setdefault("checked", set()):
                # before committing to a place that matters (an aisle to survey, the dock): is this really it?
                # the camera compares the live view with this place and with each far-away place a fresh whole-map
                # search still finds plausible (up to 3)
                state["checked"].add(name); r = reloc.locate(scan_now)
                rivals = [c for c in ([r.pose] if r.pose else []) + [a["pose"] for a in r.alternatives] if err(c, belief) >= TWIN_M][:3]
                for h in sorted([h for h in tr.hyps if err(h, belief) >= TWIN_M], key=lambda h: -h[3] / max(h[4], 1))[:2]:
                    if all(err(h, c) >= AWAY_M for c in rivals): rivals.append((h[0], h[1], h[2]))
                live = nearest_ref(true); switched = False
                for rv in rivals[:2]:
                    ra = nearest_ref(belief, away_from_xy=live[2] if live else None); rb = nearest_ref(rv, away_from_xy=live[2] if live else None)
                    if not (live and ra and rb):
                        continue
                    pick = camera_pick(live[1], ra[1], rb[1])
                    if pick == "B":                                         # a verdict that moves the robot is asked twice
                        pick = "B" if camera_pick(live[1], ra[1], rb[1]) == "B" else "unsure"
                    log["events"].append({"m": round(metres, 1), "what": f"camera check at {name}: live view vs here vs rival ({rv[0]:.1f}, {rv[1]:.1f}) -> {pick or 'no answer'}"})
                    if pick == "B":
                        if not re_anchor((rv[0], rv[1], rv[2]), f"this is not {name}"):
                            if goals and not recover("cannot plan after the camera check"):
                                outcome = "ran_out"; goals.clear()
                        switched = True; break
                if switched:
                    continue
            goals.pop(0)
            if not goals:
                outcome = "mission_done"; break
    tp = np.array(log["true_path"])[:, :2]; tpt = cKDTree(tp); cov = {}; ok_label = 0; n_vis = 0
    for aid, st in farm.stops.items():
        cov[aid] = round(float((tpt.query(st)[0] < 0.6).mean()), 3)
    for m, th, tm, bh, bm, _i in log["visits"]:
        n_vis += 1; ok_label += int(bh == th and bm is not None and abs(bm - tm) < 1.0)
    return log | {"outcome": outcome, "metres": round(metres, 1), "corrected_at_m": corrected_at, "replans": replans, "coverage": cov,
                  "aisle_steps_logged_right": round(ok_label / n_vis, 3) if n_vis else None, "final_belief_err_m": round(err(state["belief"], true), 2),
                  "at_dock": math.dist(true[:2], farm.dock[:2]) < NODE_M, "unexpected_obstacle": unexpected}


def far_cases(fs: FreeSpace, n, seed=7):
    rng = np.random.default_rng(seed); ys, xs = np.nonzero(fs.free); out = []; tries = 0
    while len(out) < n and tries < 5000:
        tries += 1; k = rng.integers(len(ys)); x, y = fs.xy_of((ys[k], xs[k]))
        if fs.dist[ys[k], xs[k]] < 0.8 or any(math.hypot(x - c["truth"][0], y - c["truth"][1]) < 4.0 for c in out):
            continue
        if any(a for a in G["farm"].aisles if G["farm"].where((x, y, 0))[0]):
            continue                                                  # not inside an aisle: in the open
        out.append({"truth": [round(x, 2), round(y, 2), round(float(rng.uniform(-180, 180)), 1)], "tag": "far", "seed": int(k)})
    return out


def context_decide(live, ref_here, ref_rival, context: dict):
    """The agent decides WHERE the robot is from everything it knows: the LiDAR's two candidate places with their
    scores, how far it has driven, the doors it passed and when, the camera verdicts so far, what the plan expected,
    plus the live view against the reference views of both places. Returns ('here'|'rival'|'undecided', reason)."""
    import os
    from openai import OpenAI
    from scripts.vlm_place_check import b64
    client = OpenAI(base_url=os.environ.get("EYES2_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"), api_key=os.environ.get("EYES2_API_KEY") or os.environ.get("NEBIUS_API_KEY"), timeout=180)
    msg = ("You are the navigator of a greenhouse robot. Two identical greenhouses stand side by side, so the LiDAR cannot tell them apart and single pictures "
           "inside the rows look alike. The robot has just passed a doorway (rows look the same, yards and doorways do not). Decide where it really is from ALL of "
           "the evidence below, not from one picture: the odometry history is reliable (if it entered a row 40 m ago it is still in that row), the plan says "
           "what it expected to see, doors and yards are the places that differ between the two houses.\n"
           "Image 1 = what the robot sees now. Image 2 = the reference view at the place it BELIEVES it is. Image 3 = the reference view at the RIVAL place.\n"
           + json.dumps(context, ensure_ascii=False) +
           "\nReply with JSON only: {\"where\": \"here\" | \"rival\" | \"undecided\", \"reason\": \"one sentence\"}. Say 'undecided' when the evidence does not settle it.")
    content = [{"type": "text", "text": msg}] + [{"type": "image_url", "image_url": {"url": b64(f)}} for f in (live, ref_here, ref_rival)]
    try:
        r = client.chat.completions.create(model=os.environ.get("EYES2_MODEL", "nvidia/Nemotron-3.5-Super-VL"), temperature=0.1, max_tokens=6000, messages=[{"role": "user", "content": content}])
        m = r.choices[0].message; txt = (m.content or getattr(m, "reasoning_content", None) or "").split("</think>")[-1].replace("```json", "").replace("```", "")
        j = _last_json(txt); w = j.get("where"); return (w if w in ("here", "rival", "undecided") else "undecided"), str(j.get("reason", ""))[:160]
    except Exception as e:
        return "undecided", f"model unavailable: {str(e)[:60]}"


def _run(i_case):
    i, c = i_case; t1 = time.time(); c = dict(c, seed=c.get("seed", i)); r = run_case(c)
    r["case"] = i + 1; r["tag"] = c.get("tag", "near"); r["start_belief_err_m"] = r["belief_err"][0]; r["seconds"] = round(time.time() - t1, 1); r["truth"] = c["truth"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", type=Path, required=True); ap.add_argument("--rec", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", type=str, default=None); ap.add_argument("--far", type=int, default=0); ap.add_argument("--surprise", action="store_true")
    ap.add_argument("--decider", default="rpp", choices=["rpp", "nearest", "nemotron"]); ap.add_argument("--review", action="store_true", help="HERMES reviews the plan on the map picture before the robot moves"); ap.add_argument("--eyes-every", type=float, default=0.0, help="metres between camera checks while driving (0 = off; the doorway decision and the dock check stay)"); ap.add_argument("--no-context", action="store_true", help="switch off the agent's doorway decision"); ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args(); teach = a.rec / "teach"
    fs, m, theta, aisles, dock = build_structure(a.scans, a.rec)
    G["world"] = World(m); G["reloc"] = Relocalizer(m, None, grid_m=0.5, grid_headings=24, refine_top=24); G["farm"] = FreeFarm(fs, aisles, dock)
    G["decider"] = a.decider; G["surprise"] = a.surprise; G["review"] = a.review; G["eyes_every"] = a.eyes_every; G["context_decide"] = not a.no_context; G["free0"] = fs.free.copy()
    ys, xs = np.nonzero(fs.free & (fs.dist >= 1.2)); G["yard_cells"] = [fs.xy_of((i, j)) for i, j in zip(ys, xs) if G["farm"].where(fs.xy_of((i, j)) + (0,))[0] is None]
    ap_refs = teach / "offpath_images_map"; poses_f = teach / "offpath_scans" / "scan_poses.json"; G["refs"] = []
    if ap_refs.is_dir() and poses_f.exists():
        for i, pz in enumerate(json.load(open(poses_f))):
            if pz.get("tag") == "map" and (ap_refs / f"{i:05d}.jpg").exists():
                G["refs"].append((pz["x"], pz["y"], pz["heading_deg"], ap_refs / f"{i:05d}.jpg"))
    import scripts.offpath_mission_eval as _ome; _ome.G["refs"] = G["refs"]          # nearest_ref lives there
    print(f"structure: {len(G['farm'].aisles)} plant aisles {[x['id'] for x in G['farm'].aisles]}, dock {tuple(round(v, 1) for v in dock)}, free {int(fs.free.sum())} cells, refs {len(G['refs'])}", flush=True)
    cases = [json.loads(l) for l in open(a.cases)]
    if a.far:
        cases += far_cases(fs, a.far)
    sel = [int(x) - 1 for x in a.only.split(",")] if a.only else list(range(len(cases)))
    a.out.mkdir(parents=True, exist_ok=True); out = []; t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(_run, [(i, cases[i]) for i in sel]):
            out.append(r); (a.out / "mission_cases.json").write_text(json.dumps(out))
            print(json.dumps({k: r.get(k) for k in ("case", "tag", "first_belief", "start_belief_err_m", "outcome", "metres", "corrected_at_m", "coverage", "aisle_steps_logged_right", "final_belief_err_m", "at_dock", "unexpected_obstacle", "seconds")}
                             | {"events": [e["what"][:90] for e in r["events"]]}), flush=True)
    def summ_of(rs):
        done = [r for r in rs if r["outcome"] != "stays_put"]
        return {"cases": len(rs), "survey_complete": sum(r["outcome"] == "mission_done" for r in rs),
                "every_aisle_fully_covered": sum(all(v >= 0.98 for v in r["coverage"].values()) for r in done),
                "records_labelled_right": sum(r["outcome"] == "mission_done" and all(v >= 0.98 for v in r["coverage"].values()) and (r["aisle_steps_logged_right"] or 0) >= 0.95 for r in rs),
                "started_wrong": sum(r["start_belief_err_m"] >= 1.0 for r in rs), "wrong_corrected_by_end": sum(r["start_belief_err_m"] >= 1.0 and r["final_belief_err_m"] < 1.0 for r in rs),
                "unexpected_obstacle_handled": sum(r.get("unexpected_obstacle") is not None and r["outcome"] == "mission_done" for r in rs),
                "at_dock_at_end": sum(r["at_dock"] for r in rs), "stays_put": sum(r["outcome"] == "stays_put" for r in rs)}
    summ = {"near": summ_of([r for r in out if r["tag"] == "near"]), "far": summ_of([r for r in out if r["tag"] == "far"]), "decider": a.decider, "hermes_review": a.review, "eyes_every_m": a.eyes_every, "agent_doorway_decision": not a.no_context, "surprise_obstacles": a.surprise,
            "structure": {"plant_aisles": [x["id"] for x in G["farm"].aisles], "aisle_lengths_m": [x["length_m"] for x in G["farm"].aisles], "dock": list(dock)}, "seconds": round(time.time() - t0, 1),
            "scans": "synthetic, ray-cast on the self-built map (taught + look-around), own-body returns removed; no stored route anywhere"}
    print(json.dumps(summ, indent=1))
    (a.out / "mission_cases.json").write_text(json.dumps(out)); (a.out / "summary.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
