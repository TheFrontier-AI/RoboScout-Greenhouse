"""Validate HERMES (cloud) goal commands and turn them into mission legs. The cloud only sends goals, never wheel
commands; anything that does not fit the confirmed aisle map is rejected with a reason (docs/INTERFACES.md §1).
Moving between greenhouses (go_to_house, survey_house of another house) also needs a CONFIRMED site map
(r1_nav/site_map.py): passed in, or loaded lazily from env R1_SITE_MAP (comma-separated JSON paths)."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from .aisle_map import AisleMap
from .mission import Leg, SiteLeg, go_to, survey_aisle, survey_house
from .site_map import NoRoute, Route, SiteMap

COMMANDS = ("go_to", "survey_aisle", "survey_house", "revisit", "discover_house", "return_to_dock", "stop", "go_to_house")
REVISIT_HOLD_S = 3.0           # dwell at the revisit point so all 6 side cameras get a steady shot


@dataclass
class Reply:
    status: str                # accepted | rejected
    reason: str = ""
    legs: list[Leg] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"status": self.status, "reason": self.reason, "legs": [l.__dict__ for l in self.legs]}


def _reject(reason: str) -> Reply:
    return Reply("rejected", reason)


_ENV_CACHE: dict[tuple, list[SiteMap]] = {}


def _env_sites() -> list[SiteMap]:
    """Site maps from R1_SITE_MAP, reloaded when a file changes (a patched map must be re-confirmed anyway)."""
    paths = [Path(p.strip()) for p in os.environ.get("R1_SITE_MAP", "").split(",") if p.strip()]
    key = tuple((str(p), (p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None) for p in paths)
    if key not in _ENV_CACHE:
        _ENV_CACHE.clear()
        _ENV_CACHE[key] = [SiteMap.load(p) for p in paths if p.exists()]
    return _ENV_CACHE[key]


def _pick_site(site, map_source: str | None) -> tuple[SiteMap | None, str]:
    """Newest confirmed metric site map (optionally from one source), or (None, reason)."""
    cands = list(site.values()) if isinstance(site, dict) else [site] if isinstance(site, SiteMap) else list(site or [])
    cands = cands or _env_sites()
    if not cands:
        return None, "no site map loaded (farm-level moves need a confirmed site map; see farm-site-map skill)"
    if map_source:
        cands = [s for s in cands if s.meta.get("source") == map_source]
        if not cands:
            return None, f"no site map from source {map_source!r} is loaded"
    ok = [s for s in cands if s.confirmed and s.meta.get("scale_known")]
    if not ok:
        s = max(cands, key=lambda m: m.revision)
        why = "has no metric scale (align it to ≥ 2 anchors)" if not s.meta.get("scale_known") else "is not confirmed"
        return None, (f"site map ({s.meta.get('source')}, rev {s.revision}) {why}; "
                      "ask a human to review and confirm it before the robot uses it")
    return max(ok, key=lambda m: (str(m.meta.get("confirmed_at")), m.revision)), ""


def _transit(site: SiteMap, route: Route) -> list[Leg]:
    legs = []
    for u, v, e in zip(route.nodes, route.nodes[1:], route.edges):
        note = f"follow site edge {u} → {v}" + (f" ({e.surface})" if e.surface != "unknown" else "")
        if e.notes:
            note += "; notes: " + " | ".join(e.notes)
        legs.append(SiteLeg("site_transit", None, None, note, from_node=u, to_node=v,
                            length_m=round(site.edge_length(e), 2)))
    return legs


def _route_notes(route: Route) -> str:
    notes = [f"{e.a}–{e.b}: {n}" for e in route.edges for n in e.notes]
    return ("; route notes: " + " | ".join(notes)) if notes else ""


def _cross_house(name: str, cmd: dict, amap: AisleMap | None, site, aisle_maps, current_house: str | None) -> Reply:
    target = cmd.get("house")
    if not target:
        return _reject(f"{name} needs a house")
    here = current_house or (amap.house_id if amap else None)
    s, why = _pick_site(site, cmd.get("map_source"))
    if s is None:
        return _reject((f"robot is in {here}, not {target}; " if here and why.startswith("no site map") else "") + why)
    try:
        goal = s.door_of(target)
    except KeyError:
        return _reject(f"unknown house {target!r}; houses on the site map: {', '.join(s.houses()) or 'none'}")
    if here == target and name == "go_to_house":
        return Reply("accepted", f"already in {target}", [])
    try:
        start = s.door_of(here) if here else (s.docks() or [None])[0]
    except KeyError:
        return _reject(f"current house {here!r} is not on the site map")
    if start is None:
        return _reject("robot position unknown: no current house and no dock on the site map")
    try:
        route = s.route(start, goal)
    except NoRoute as e:
        return _reject(str(e))
    legs: list[Leg] = []
    if here:
        legs.append(SiteLeg("exit_house", None, None, f"reverse out to the headland and leave {here} by its door",
                            house=here, to_node=start))
    legs += _transit(s, route)
    legs.append(SiteLeg("enter_house", None, None, f"align at door {goal} and enter {target}", house=target,
                        from_node=goal, to_node=goal))
    reason = (f"route {' → '.join(route.nodes)} = {route.length_m:.1f} m on {s.meta.get('source')} site map "
              f"rev {s.revision}{_route_notes(route)}")
    if name == "survey_house":
        maps = {m.house_id: m for m in (aisle_maps.values() if isinstance(aisle_maps, dict) else aisle_maps or [])}
        tm = maps.get(target)
        if tm is None:
            legs.append(Leg("discover", None, None, f"no aisle map for {target}: follow headland, probe every opening"))
            reason += f"; no confirmed aisle map for {target}: discovering it (confirm the map, then survey)"
        else:
            body, skipped = survey_house(tm)
            legs += body[:-1]                                    # dock leg replaced by the site route back
            if skipped:
                reason += f"; skipping too-narrow aisles: {', '.join(a.id for a in skipped)}"
            try:
                back = s.route(goal, s.docks()[0])
                legs += [SiteLeg("exit_house", None, None, f"leave {target} by its door", house=target, to_node=goal)]
                legs += _transit(s, back)
            except (IndexError, NoRoute):
                reason += "; no site route back to a dock"
            legs.append(Leg("dock", None, None, "return to dock"))
    return Reply("accepted", reason, legs)


def plan_command(cmd: dict, amap: AisleMap | None, site=None, aisle_maps=None, current_house: str | None = None) -> Reply:
    """Map one command dict to legs. `amap` is None before the house has been discovered and confirmed.
    `site`: SiteMap, list or {source: SiteMap} (None → env R1_SITE_MAP); `aisle_maps`: {house: AisleMap} or list of
    known houses; `current_house` defaults to amap.house_id."""
    name = cmd.get("cmd")
    if name not in COMMANDS:
        return _reject(f"unknown cmd {name!r}; allowed: {', '.join(COMMANDS)}")
    if name == "stop":
        return Reply("accepted", "stop now", [Leg("stop", None, None, "halt and hold")])
    if name == "return_to_dock":
        return Reply("accepted", "", [Leg("dock", None, None, "return to dock")])
    if name == "discover_house":
        return Reply("accepted", "map unknown house with SLAM", [Leg("discover", None, None, "follow headland, probe every opening")])
    here = current_house or (amap.house_id if amap else None)
    if name == "go_to_house" or (name == "survey_house" and cmd.get("house") not in (None, here)):
        return _cross_house(name, cmd, amap, site, aisle_maps, current_house)
    if amap is None:
        return _reject("no confirmed aisle map yet; send discover_house first")
    if cmd.get("house") not in (None, amap.house_id):
        return _reject(f"robot is in {amap.house_id}, not {cmd.get('house')}")
    try:
        if name == "survey_house":
            legs, skipped = survey_house(amap)
            note = f"skipping too-narrow aisles: {', '.join(a.id for a in skipped)}" if skipped else ""
            return Reply("accepted", note, legs)
        aisle = cmd.get("aisle")
        if name == "survey_aisle":
            return Reply("accepted", "", survey_aisle(amap, aisle))
        metre = float(cmd["metre"])
        legs = go_to(amap, aisle, metre)
        if name == "revisit":
            legs.append(Leg("hold", aisle, metre, f"capture all side cameras for {REVISIT_HOLD_S:.0f} s (purpose=revisit)"))
        return Reply("accepted", "", legs)
    except KeyError as e:
        return _reject(f"missing or unknown field/aisle: {e}")
    except (TypeError, ValueError) as e:
        return _reject(str(e))
