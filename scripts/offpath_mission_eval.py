#!/usr/bin/env python3
"""Put down somewhere, decide where you are, then DO THE MISSION: survey every plant in every taught house and return
to the dock. The question is not "can it drive to a point" but "does a wrong first belief cost the survey?".

Cases: (a) the 40 off-path poses of scripts/vlm_offpath_check.py, starting from the pose the robot BELIEVED there
(the camera's pick, sometimes wrong); (b) `--far N` extra poses 4-8 m from the taught path inside the area the
look-around scans covered, starting from the LiDAR's best single-scan guess (no camera images exist for them).

The robot plans the survey from its belief (rejoin the taught path at the nearest reachable point, aisle of tunnel-09,
yard, aisle of tunnel-10, dock) and executes it; its TRUE pose moves by the same body-frame motion. Every 0.25 m a LiDAR
scan is synthesised at the true pose from the self-built map and the HypothesisTracker checks that the believed place
still fits. A belief that stops fitting the scans, or an obstacle where the map said the way was free, means the belief
was wrong: the robot searches the whole map again, keeps away from the place just contradicted, adopts the best place,
re-plans what is left of the mission and carries on. We record the true path, the belief error along the way, which
plant stops were really covered, and whether each covered stop was logged under the right house and metre.

Scans are synthetic (ray-cast on the self-built 2-D map): this measures the travelling logic, not sensor noise.

  python3 scripts/offpath_mission_eval.py --scans scans_all.jsonl --rec <run>/rec --cases vlm_offpath.jsonl --out <dir> [--far 10]"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.relocalize import HypothesisTracker, Relocalizer       # noqa: E402
from scripts.offpath_eval import build_from_scans               # noqa: E402
from sim.scan_localizer import LIDAR_AHEAD_M                    # noqa: E402

STEP_M, MAX_RANGE, N_BEAMS, OBSTACLE_M, NODE_M, STOP_PITCH, REJOIN_M, AWAY_M = 0.25, 11.0, 500, 0.25, 1.5, 0.45, 8.0, 3.0
MISSION = ["tunnel-09/door-1", "tunnel-09/door-2", "tunnel-10/door-1", "tunnel-10/door-2", "dock"]
AISLES = {"tunnel-09": ("tunnel-09/door-1", "tunnel-09/door-2"), "tunnel-10": ("tunnel-10/door-1", "tunnel-10/door-2")}
G: dict = {}                                                      # per-process world (filled before the pool forks)


class World:
    """The farm as a 2-D point map: synthetic LiDAR and obstacle checks for the TRUE pose."""

    def __init__(self, map_xy):
        self.m = map_xy; self.tree = cKDTree(map_xy); self.bin = 2 * math.pi / N_BEAMS

    def scan(self, x, y, h_deg):
        h = math.radians(h_deg); c, s = math.cos(h), math.sin(h)
        ox, oy = x + c * LIDAR_AHEAD_M, y + s * LIDAR_AHEAD_M
        idx = self.tree.query_ball_point((ox, oy), MAX_RANGE)
        if not idx:
            return np.zeros((0, 2))
        p = self.m[idx] - (ox, oy); q = p @ np.array([[c, s], [-s, c]]).T       # lidar frame, 0 = forward
        r = np.hypot(q[:, 0], q[:, 1]); a = np.arctan2(q[:, 1], q[:, 0])
        b = np.floor((a + math.pi) / self.bin).astype(int) % N_BEAMS
        best = np.full(N_BEAMS, np.inf); np.minimum.at(best, b, r)
        ok = np.isfinite(best) & (best > 0.25); ang = -math.pi + (np.arange(N_BEAMS) + 0.5) * self.bin
        return np.stack([best[ok] * np.cos(ang[ok]), best[ok] * np.sin(ang[ok])], -1)

    def blocked(self, x, y):
        return self.tree.query((x, y))[0] < OBSTACLE_M

    def free_segment(self, a, b, clearance=0.30):
        n = max(2, int(np.linalg.norm(np.subtract(b, a)) / 0.2)); pts = np.linspace(a, b, n)
        return bool((self.tree.query(pts)[0] >= clearance).all())


class Farm:
    """Taught path, graph nodes, aisle metre bookkeeping, mission plans."""

    def __init__(self, teach: Path, world: World):
        self.path = np.load(teach / "path_xyh.npy"); sm = json.load(open(teach / "site_map.json")); self.world = world
        self.nodes = {n["id"]: np.array(n["xy"], float) for n in sm["nodes"]}
        self.nidx = {k: self.pidx(v) for k, v in self.nodes.items()}; self.tree = cKDTree(self.path[:, :2])
        self.rj_idx = np.arange(0, len(self.path), 15); self.rj_tree = cKDTree(self.path[self.rj_idx, :2])
        self.cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(self.path[:, 0]), np.diff(self.path[:, 1])))])
        self.stops = {}
        for h, (d1, d2) in AISLES.items():
            i, j = self.nidx[d1], self.nidx[d2]; L = self.cum[j] - self.cum[i]
            self.stops[h] = np.array([self.path[i + np.searchsorted(self.cum[i:j + 1] - self.cum[i], k * STOP_PITCH), :2] for k in range(int(L / STOP_PITCH))])

    def pidx(self, xy):
        return int(np.argmin(np.hypot(self.path[:, 0] - xy[0], self.path[:, 1] - xy[1])))

    def where(self, pose):
        d, i = self.tree.query(pose[:2])
        if d > 2.0:
            return None, None
        for h, (d1, d2) in AISLES.items():
            a, b = self.nidx[d1], self.nidx[d2]
            if a <= i <= b:
                return h, round(float(self.cum[i] - self.cum[a]), 2)
        return None, None

    def along(self, i, j):
        """Path points from index i to index j, the shorter way (the taught drive starts and ends at the dock, so the
        two ends join)."""
        n = len(self.path)
        if i <= j:
            fwd = self.path[i:j + 1, :2]; wrap = np.vstack([self.path[i::-1, :2], self.path[n - 1:j - 1 if j > 0 else None:-1, :2]])
        else:
            fwd = self.path[i:j - 1 if j > 0 else None:-1, :2]; wrap = np.vstack([self.path[i:, :2], self.path[:j + 1, :2]])
        return fwd if len(fwd) <= len(wrap) else wrap

    def plan(self, pose, goals):
        """Polyline from the believed pose through the remaining goals: straight to the nearest taught-path point that
        is within REJOIN_M and reachable in a straight free line on the map, then along the taught path."""
        d, idx = self.rj_tree.query(pose[:2], k=min(300, len(self.rj_idx))); d = np.atleast_1d(d); idx = self.rj_idx[np.atleast_1d(idx)]
        rejoin = None
        for dd, ii in zip(d, idx):
            if dd > REJOIN_M:
                break
            if self.world.free_segment(np.array(pose[:2], float), self.path[ii, :2]):
                rejoin = int(ii); break
        if rejoin is None:
            return None
        pts = [np.array(pose[:2], float), self.path[rejoin, :2]]; cur = rejoin
        for g in goals:
            pts.extend(self.along(cur, self.nidx[g])); cur = self.nidx[g]
        return np.array(pts, float)


def resample(poly, step):
    out = [poly[0]]; acc = 0.0
    for a, b in zip(poly, poly[1:]):
        L = float(np.linalg.norm(b - a))
        if L < 1e-9:
            continue
        while acc + step <= L:
            acc += step; out.append(a + (b - a) * acc / L)
        acc -= L
    return np.array(out)


def compose(base, rel):
    h = math.radians(base[2]); c, s = math.cos(h), math.sin(h)
    return (base[0] + c * rel[0] - s * rel[1], base[1] + s * rel[0] + c * rel[1], base[2] + rel[2])


def inv(p):
    c, s = math.cos(math.radians(-p[2])), math.sin(math.radians(-p[2]))
    return (-(c * p[0] - s * p[1]), -(s * p[0] + c * p[1]), -p[2])


def rel_motion(a, b):
    h = math.radians(a[2]); c, s = math.cos(h), math.sin(h); dx, dy = b[0] - a[0], b[1] - a[1]
    return (c * dx + s * dy, -s * dx + c * dy, b[2] - a[2])


def search(reloc, scan, farm, goals, away_from=None):
    """Whole-map search. A place within AWAY_M of `away_from` (the belief just contradicted) is not accepted, and a
    place from which the map says the mission cannot be planned is rejected in favour of the next candidate."""
    r = reloc.locate(scan); cands = ([r.pose] if r.pose else []) + [a["pose"] for a in r.alternatives]
    if away_from is not None:
        cands = [c for c in cands if math.hypot(c[0] - away_from[0], c[1] - away_from[1]) >= AWAY_M] or cands
    for c in cands:
        p = farm.plan(c, goals)
        if p is not None:
            return tuple(c), r.ambiguous, p
    return (tuple(cands[0]) if cands else None), r.ambiguous, None


def remaining_goals(hist, farm):
    """Which mission goals the (corrected) believed history has already reached, in order."""
    j = 0
    for n, g in enumerate(MISSION):
        hit = next((i for i in range(j, len(hist)) if math.hypot(hist[i][0] - farm.nodes[g][0], hist[i][1] - farm.nodes[g][1]) < NODE_M), None)
        if hit is None:
            return list(MISSION[n:])
        j = hit
    return []


def run_case(case):
    world, reloc, farm = G["world"], G["reloc"], G["farm"]
    truth = tuple(case["truth"]); err = lambda b, t: math.hypot(b[0] - t[0], b[1] - t[1])
    if case.get("final_pose"):
        belief = tuple(case["final_pose"]); how = "camera pick"
    elif case.get("lidar", {}).get("pose") and case.get("tag") != "far":
        belief = tuple(case["lidar"]["pose"]); how = "LiDAR best (camera undecided)"
    else:
        belief, amb, _ = search(reloc, world.scan(*truth), farm, list(MISSION)); how = "LiDAR best" + (" (ambiguous)" if amb else "")
    true = truth
    log = {"true_path": [list(truth)], "belief_err": [round(err(belief, true), 2)], "metres": [0.0], "events": [], "visits": [], "first_belief": how}
    tr = HypothesisTracker(reloc); odom = (0.0, 0.0, 0.0); tr.start(world.scan(*truth), odom)
    tr.hyps.sort(key=lambda r: err(r, belief)); tr.hyps[0][0:3] = list(belief)
    goals = list(MISSION); plan = farm.plan(belief, goals)
    metres = 0.0; replans = 0; corrected_at = None; outcome = "ran_out"
    state = {"belief": belief, "plan": plan, "wp": None, "k": 1}; hist = []   # odometry pose per step

    def record(e):
        log["belief_err"].append(round(e, 2)); log["metres"].append(round(metres, 2)); log["true_path"].append([round(v, 3) for v in true])

    def research(reason):
        """The belief was contradicted: search the whole map again (away from the contradicted place), then treat the
        odometry since the last anchor as right and the anchor as wrong - shift the believed history by the same
        correction, relabel the plant records made in that stretch, and re-derive which mission goals are really done."""
        nonlocal replans
        old = state["belief"]; scan = world.scan(*true)
        nb, amb, p = search(reloc, scan, farm, goals, away_from=old)
        tr._restart = True; tr.start(scan, odom); tr._restart = False
        if nb is None:
            return False
        if err(nb, old) >= AWAY_M:                       # the contradicted place is given up for good
            tr.hyps = [h for h in tr.hyps if err(h, old) >= AWAY_M] or tr.hyps
        tr.hyps.sort(key=lambda r: err(r, nb)); tr.hyps[0][0:3] = list(nb)
        state["belief"] = nb; replans += 1
        # odometry is continuous for the whole run, so one new anchor re-places the WHOLE believed history
        T = compose(nb, inv(odom)); corrected = [compose(T, o) for o in hist]; relabelled = 0
        for v in log["visits"]:
            v[3], v[4] = farm.where(corrected[v[5]]); relabelled += 1
        before = list(goals); goals[:] = remaining_goals(corrected, farm)
        log["events"].append({"m": round(metres, 1), "what": f"{reason}: searched the whole map again, now believes ({nb[0]:.1f}, {nb[1]:.1f}), error {err(nb, true):.1f} m" + (" (ambiguous)" if amb else "")
                              + (f"; relabelled {relabelled} plant records" if relabelled else "") + (f"; mission goals left {len(before)} -> {len(goals)}" if len(goals) != len(before) else "")})
        if not goals:
            state["plan"] = None; return True
        state["plan"] = farm.plan(nb, goals)
        if state["plan"] is None:
            state["plan"] = p
        if state["plan"] is None:
            log["events"].append({"m": round(metres, 1), "what": "no reachable taught path within 8 m of the new belief: stops and asks for help"})
            return False
        state["wp"] = resample(state["plan"], STEP_M); state["k"] = 1
        return True

    def probe():
        """No candidate place lets the mission be planned: step 2 m into the most open direction the live LiDAR shows
        (never into an obstacle), then search again. Up to 5 probes before giving up and asking for help."""
        nonlocal true, odom, metres
        sc = world.scan(*true)
        if len(sc) == 0:
            return False
        d = np.hypot(sc[:, 0], sc[:, 1]); a = math.degrees(math.atan2(sc[d.argmax(), 1], sc[d.argmax(), 0]))
        moved = 0.0
        true = compose(true, (0.0, 0.0, a)); odom = compose(odom, (0.0, 0.0, a))      # turn towards the open direction first
        for _ in range(8):
            rel = (STEP_M, 0.0, 0.0); nxt = compose(true, rel)
            if world.blocked(nxt[0], nxt[1]):
                break
            true = nxt; odom = compose(odom, rel); metres += STEP_M; moved += STEP_M; record(err(state["belief"], true)); hist.append(odom)
        log["events"].append({"m": round(metres, 1), "what": f"probe: moved {moved:.1f} m into open space, searching again"})
        return moved > 0

    if plan is None:
        log["events"].append({"m": 0.0, "what": "no reachable taught path within 8 m of the believed pose: tries a search from here"})
        ok = research("cannot plan from the believed pose"); probes = 0
        while not ok and probes < 5 and probe():
            probes += 1; ok = research("after the probe")
        if not ok:
            return log | {"outcome": "stays_put", "metres": round(metres, 1), "corrected_at_m": None, "replans": replans, "coverage": {h: 0.0 for h in AISLES},
                          "aisle_steps_logged_right": None, "final_belief_err_m": round(err(state["belief"], true), 2), "at_dock": False}
    else:
        state["wp"] = resample(plan, STEP_M)
    while metres < 320.0 and replans <= 12:
        if not goals:
            outcome = "mission_done"; break
        belief = state["belief"]; wp = state["wp"]; k = state["k"]
        if wp is None or k >= len(wp):
            break
        tgt = wp[k]; hb = math.degrees(math.atan2(tgt[1] - belief[1], tgt[0] - belief[0]))
        nxt_b = (float(tgt[0]), float(tgt[1]), hb); rel = rel_motion(belief, nxt_b); nxt_t = compose(true, rel)
        if world.blocked(nxt_t[0], nxt_t[1]) and world.tree.query(nxt_b[:2])[0] >= 0.5:
            # the map says the believed next place has room (>= 0.5 m) but the robot is really blocked: the belief is wrong
            ok = research("obstacle where the map said the way was free: stop"); probes = 0
            while not ok and probes < 5 and probe():
                probes += 1; ok = research("after the probe")
            if not ok:
                break
            record(err(state["belief"], true)); continue
        true = nxt_t; odom = compose(odom, rel); metres += math.hypot(rel[0], rel[1]); state["k"] = k + 1
        scan_now = world.scan(*true); t = tr.update(scan_now, odom)
        if state["k"] % 40 == 0:
            # every 10 m: a fresh whole-map search; its candidates join the tracked rivals (they do not replace the
            # belief - a rival has to beat the believed place by the decide margin over 2 m of driving to win)
            r = reloc.locate(scan_now)
            for c in ([r.pose] if r.pose else []) + [a["pose"] for a in r.alternatives]:
                if all(err(c, h) > 3.0 for h in tr.hyps):
                    tr.hyps.append([c[0], c[1], c[2], reloc._score(np.asarray(scan_now) + (LIDAR_AHEAD_M, 0.0), *c), 1])
            tr.hyps = tr.hyps[: tr.keep + 4]
        alive = [h for h in tr.hyps if err(h, nxt_b) < 2.5]
        body_now = np.asarray(scan_now) + (LIDAR_AHEAD_M, 0.0)
        if not alive:
            # the believed place may have been pushed out of the tracker's list by fresh rivals: judge it on its own
            # fit to this scan, and keep it unless the fit is genuinely poor
            (fx, fy, fh), _q = reloc.loc.localize(np.asarray(scan_now), *nxt_b); sc = reloc._score(body_now, fx, fy, fh)
            if sc >= tr.drop_below:
                tr.hyps.insert(0, [fx, fy, fh, sc, 1]); alive = [tr.hyps[0]]
        rival = None
        if alive:
            b_mean = alive[0][3] / alive[0][4]
            far = [h for h in tr.hyps if err(h, nxt_b) >= AWAY_M and h[4] >= 8]
            if far:
                best = max(far, key=lambda h: h[3] / h[4])
                if best[3] / best[4] - b_mean >= tr.decide_margin:
                    rival = best
        if alive and rival is not None:
            # a far-away rival has fitted the scans clearly better than the believed place over >= 2 m of driving
            old = state["belief"]; nb = (rival[0], rival[1], rival[2]); state["belief"] = nb; replans += 1
            T = compose(nb, inv(odom)); corrected = [compose(T, o) for o in hist]
            for v in log["visits"]:
                v[3], v[4] = farm.where(corrected[v[5]])
            before = list(goals); goals[:] = remaining_goals(corrected, farm)
            log["events"].append({"m": round(metres, 1), "what": f"a rival place fits the scans clearly better (margin {rival[3] / rival[4] - b_mean:.2f}): now believes ({nb[0]:.1f}, {nb[1]:.1f}), error {err(nb, true):.1f} m; relabelled {len(log['visits'])} plant records" + (f"; mission goals left {len(before)} -> {len(goals)}" if len(goals) != len(before) else "")})
            tr.hyps = [h for h in tr.hyps if err(h, old) >= AWAY_M] or tr.hyps
            if goals:
                state["plan"] = farm.plan(nb, goals)
                if state["plan"] is None:
                    ok = research("cannot plan after switching place"); probes = 0
                    while not ok and probes < 5 and probe():
                        probes += 1; ok = research("after the probe")
                    if not ok:
                        break
                else:
                    state["wp"] = resample(state["plan"], STEP_M); state["k"] = 1
        elif alive:
            state["belief"] = (alive[0][0], alive[0][1], alive[0][2])
        else:
            ok = research("believed place no longer fits the scans"); probes = 0
            while not ok and probes < 5 and probe():
                probes += 1; ok = research("after the probe")
            if not ok:
                break
        belief = state["belief"]; e = err(belief, true); record(e); hist.append(odom)
        if e < 1.0 and corrected_at is None:
            corrected_at = round(metres, 1)
        if e >= 1.0:
            corrected_at = None
        bh, bm = farm.where(belief); th, tm = farm.where(true)
        if th is not None:
            log["visits"].append([round(metres, 1), th, tm, bh, bm, len(hist) - 1])
        if goals and err(belief, farm.nodes[goals[0]]) < NODE_M:
            log["events"].append({"m": round(metres, 1), "what": f"believes it reached {goals[0]} (true error {err(belief, true):.1f} m)"})
            if goals[0] == "dock" and G.get("refs") and not state.get("dock_checked"):
                # before docking: the camera compares the view here with the dock and with the strongest far rival
                state["dock_checked"] = True
                far = [h for h in tr.hyps if err(h, belief) >= AWAY_M]
                if not far:
                    r = reloc.locate(scan_now); far = [list(a["pose"]) + [0, 1] for a in r.alternatives if err(a["pose"], belief) >= AWAY_M]
                if far:
                    rv = max(far, key=lambda h: h[3] / max(h[4], 1)); live = nearest_ref(true)
                    ra = nearest_ref(belief, away_from_xy=live[2] if live else None); rb = nearest_ref(rv, away_from_xy=live[2] if live else None)
                    if live and ra and rb:
                        pick = camera_pick(live[1], ra[1], rb[1])
                        log["events"].append({"m": round(metres, 1), "what": f"camera check at the dock: live view vs dock vs rival ({rv[0]:.1f}, {rv[1]:.1f}) -> {pick or 'no answer'}"})
                        if pick == "B":
                            old = belief; nb = (rv[0], rv[1], rv[2]); state["belief"] = nb; replans += 1
                            T = compose(nb, inv(odom)); corrected = [compose(T, o) for o in hist]
                            for v in log["visits"]:
                                v[3], v[4] = farm.where(corrected[v[5]])
                            before = list(goals); goals[:] = remaining_goals(corrected, farm)
                            log["events"].append({"m": round(metres, 1), "what": f"this is not the dock: now believes ({nb[0]:.1f}, {nb[1]:.1f}), error {err(nb, true):.1f} m; relabelled {len(log['visits'])} plant records; mission goals left {len(before)} -> {len(goals)}"})
                            tr.hyps = [h for h in tr.hyps if err(h, old) >= AWAY_M] or tr.hyps
                            e = err(nb, true); corrected_at = round(metres, 1) if e < 1.0 else None
                            if goals:
                                state["plan"] = farm.plan(nb, goals)
                                if state["plan"] is None:
                                    break
                                state["wp"] = resample(state["plan"], STEP_M); state["k"] = 1
                                continue
            goals.pop(0)
            if not goals:
                outcome = "mission_done"; break
    tp = np.array(log["true_path"])[:, :2]; tpt = cKDTree(tp); cov = {}; ok_label = 0; n_vis = 0
    for h, st in farm.stops.items():
        cov[h] = round(float((tpt.query(st)[0] < 0.5).mean()), 3)
    for m, th, tm, bh, bm, _i in log["visits"]:
        n_vis += 1; ok_label += int(bh == th and bm is not None and abs(bm - tm) < 1.0)
    return log | {"outcome": outcome, "metres": round(metres, 1), "corrected_at_m": corrected_at, "replans": replans, "coverage": cov,
                  "aisle_steps_logged_right": round(ok_label / n_vis, 3) if n_vis else None,
                  "final_belief_err_m": round(err(state["belief"], true), 2), "at_dock": err(true, farm.nodes["dock"]) < NODE_M}


def far_cases(world, farm, n, seed=7):
    """Poses 4-8 m from the taught path, free of obstacles, inside the area the look-around scans covered."""
    rng = np.random.default_rng(seed); m = world.m
    lo, hi = m.min(0), m.max(0); out = []; tries = 0
    while len(out) < n and tries < 20000:
        tries += 1; x, y = rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])
        dp = farm.tree.query((x, y))[0]; dm = world.tree.query((x, y))[0]
        if not (4.0 <= dp <= 8.0 and dm >= 0.6 and len(world.tree.query_ball_point((x, y), 8.0)) >= 300):
            continue
        if any(math.hypot(x - c["truth"][0], y - c["truth"][1]) < 3.0 for c in out):
            continue
        if farm.plan((x, y, 0.0), MISSION) is None:
            continue
        out.append({"truth": [round(x, 2), round(y, 2), round(float(rng.uniform(-180, 180)), 1)], "tag": "far", "off_path_m": round(float(dp), 1)})
    return out


def nearest_ref(pose, away_from_xy=None, min_away=1.5):
    """Look-around image nearest to `pose` with a similar heading (<= 30 deg); optionally not the same spot as another."""
    best = None
    for x, y, h, img in G["refs"]:
        if abs((h - pose[2] + 180) % 360 - 180) > 30:
            continue
        if away_from_xy is not None and math.hypot(x - away_from_xy[0], y - away_from_xy[1]) < min_away:
            continue
        d = math.hypot(x - pose[0], y - pose[1])
        if best is None or d < best[0]:
            best = (d, img, (x, y))
    return best


def camera_pick(live, ref_a, ref_b):
    """Nemotron VL: which reference is where I am? -> 'A' / 'B' / None."""
    import os
    from openai import OpenAI
    from scripts.vlm_place_check import PROMPT, b64
    client = OpenAI(base_url=os.environ.get("EYES2_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"), api_key=os.environ.get("EYES2_API_KEY") or os.environ.get("NEBIUS_API_KEY"), timeout=180)
    content = [{"type": "text", "text": PROMPT}] + [{"type": "image_url", "image_url": {"url": b64(f)}} for f in (live, ref_a, ref_b)]
    try:
        resp = client.chat.completions.create(model=os.environ.get("EYES2_MODEL", "nvidia/Nemotron-3.5-Super-VL"), temperature=0.1, max_tokens=6000, messages=[{"role": "user", "content": content}])
        msg = resp.choices[0].message
        txt = (msg.content or getattr(msg, "reasoning_content", None) or "").split("</think>")[-1].replace("```json", "").replace("```", "")
        j = json.loads(txt[txt.index("{"): txt.rindex("}") + 1]) if "{" in txt else {}
        return j.get("match") if j.get("match") in ("A", "B") else None
    except Exception:
        return None


def _run(i_case):
    i, c = i_case; t1 = time.time(); r = run_case(c)
    r["case"] = i + 1; r["tag"] = c.get("tag", "near"); r["start_belief_err_m"] = r["belief_err"][0]; r["seconds"] = round(time.time() - t1, 1)
    r["truth"] = c["truth"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", type=Path, required=True); ap.add_argument("--rec", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", type=str, default=None, help="comma-separated 1-based case numbers"); ap.add_argument("--far", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args(); teach = a.rec / "teach"
    rows = [json.loads(l) for l in open(a.scans)]; mp = [r for r in rows if r["tag"] == "map"]
    taught = np.load(teach / "prior_map_xy.npy"); path = np.load(teach / "path_xyh.npy")
    look = build_from_scans(mp); m = np.unique(np.floor(np.vstack([taught, look]) / 0.05).astype(int), axis=0) * 0.05 + 0.025
    G["world"] = World(m); G["reloc"] = Relocalizer(m, path, grid_m=0.5, grid_headings=24, refine_top=24); G["farm"] = Farm(teach, G["world"])
    ap_refs = a.rec / "teach" / "offpath_images_map"; poses_f = a.rec / "teach" / "offpath_scans" / "scan_poses.json"
    G["refs"] = []
    if ap_refs.is_dir() and poses_f.exists():
        for i, pz in enumerate(json.load(open(poses_f))):
            if pz.get("tag") == "map" and (ap_refs / f"{i:05d}.jpg").exists():
                G["refs"].append((pz["x"], pz["y"], pz["heading_deg"], ap_refs / f"{i:05d}.jpg"))
    print("look-around references for the camera check:", len(G["refs"]), flush=True)
    cases = [json.loads(l) for l in open(a.cases)]
    if a.far:
        cases += far_cases(G["world"], G["farm"], a.far)
    sel = [int(x) - 1 for x in a.only.split(",")] if a.only else list(range(len(cases)))
    a.out.mkdir(parents=True, exist_ok=True); out = []; t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(_run, [(i, cases[i]) for i in sel]):
            out.append(r)
            print(json.dumps({k: r.get(k) for k in ("case", "tag", "first_belief", "start_belief_err_m", "outcome", "metres", "corrected_at_m", "coverage", "aisle_steps_logged_right", "final_belief_err_m", "at_dock", "seconds")}
                             | {"events": [e["what"][:80] for e in r["events"]]}), flush=True)
    def summ_of(rs):
        done = [r for r in rs if r["outcome"] != "stays_put"]
        return {"cases": len(rs), "mission_done": sum(r["outcome"] == "mission_done" for r in rs), "at_dock_at_end": sum(r.get("at_dock", False) for r in rs),
                "full_coverage_both_houses": sum(all(v >= 0.98 for v in r["coverage"].values()) for r in done),
                "started_wrong": sum(r["start_belief_err_m"] >= 1.0 for r in rs),
                "started_wrong_and_corrected_by_end": sum(r["start_belief_err_m"] >= 1.0 and r["final_belief_err_m"] < 1.0 for r in rs),
                "metres_driven_before_correction": sorted(r["corrected_at_m"] for r in rs if r["start_belief_err_m"] >= 1.0 and r["corrected_at_m"] is not None),
                "aisle_steps_logged_right_median": round(float(np.median([r["aisle_steps_logged_right"] for r in done if r["aisle_steps_logged_right"] is not None])), 3) if any(r["aisle_steps_logged_right"] is not None for r in done) else None,
                "stays_put": sum(r["outcome"] == "stays_put" for r in rs)}
    summ = {"near_40": summ_of([r for r in out if r["tag"] == "near"]), "far": summ_of([r for r in out if r["tag"] == "far"]), "seconds": round(time.time() - t0, 1),
            "scans": "synthetic, ray-cast on the self-built map (taught + look-around), own-body returns removed"}
    print(json.dumps(summ, indent=1))
    (a.out / "mission_cases.json").write_text(json.dumps(out)); (a.out / "summary.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
