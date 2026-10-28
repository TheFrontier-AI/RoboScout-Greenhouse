"""Farm-level site map: a graph of docks, greenhouse doors, junctions and waypoints in site/world metres (the frame of
`house_local_to_world` in the aisle maps: Blender Z-up, X right, Y up). The robot only executes legs along routes of a
map a human has CONFIRMED; any edit (close a path, add a note) bumps the revision and clears the confirmation.

Sources: v18 (farm model), drone (cloud SLAM top-down), hand_sketch (farmer photo/drawing via a VLM), robot (recorded)."""
from __future__ import annotations
import copy
import heapq
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from .aisle_map import ROBOT_WIDTH_M, SIDE_MARGIN_M

SCHEMA_VERSION = 1
NODE_KINDS = ("dock", "door", "junction", "waypoint")
SOURCES = ("v18", "drone", "hand_sketch", "robot")
FRAME = "site_world_z_up_metres"           # same frame as house_local_to_world in data/*_aisle_map_*.json


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class NoRoute(ValueError):
    pass


@dataclass
class Node:
    id: str
    kind: str                              # dock | door | junction | waypoint
    xy: tuple[float, float]
    house: str | None = None               # doors only
    heading_deg: float | None = None       # doors: heading when driving INTO the house (CCW from +X)
    label: str | None = None               # as written on the sketch / model label
    extra: dict = field(default_factory=dict)   # e.g. door_estimated, pixel_xy

    def to_dict(self) -> dict:
        d = {"id": self.id, "kind": self.kind, "xy": [round(float(v), 3) for v in self.xy]}
        d.update({k: v for k, v in (("house", self.house), ("heading_deg", self.heading_deg), ("label", self.label))
                  if v is not None})
        return {**d, **self.extra}

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        known = ("id", "kind", "xy", "house", "heading_deg", "label")
        return cls(str(d["id"]), d.get("kind", "waypoint"), tuple(float(v) for v in d["xy"][:2]), d.get("house"),
                   d.get("heading_deg"), d.get("label"), {k: v for k, v in d.items() if k not in known})


@dataclass
class Edge:
    a: str
    b: str
    length_m: float | None = None          # None → straight-line distance between the nodes
    width_m: float | None = None           # None = unknown (not used to reject a route)
    surface: str = "unknown"
    notes: list[str] = field(default_factory=list)
    blocked: bool = False
    extra: dict = field(default_factory=dict)

    def key(self) -> frozenset:
        return frozenset((self.a, self.b))

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "length_m": None if self.length_m is None else round(self.length_m, 3),
                "width_m": self.width_m, "surface": self.surface, "notes": list(self.notes), "blocked": self.blocked,
                **self.extra}

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        known = ("a", "b", "length_m", "width_m", "surface", "notes", "blocked")
        notes = d.get("notes") or []
        return cls(str(d["a"]), str(d["b"]), d.get("length_m"), d.get("width_m"), d.get("surface") or "unknown",
                   [notes] if isinstance(notes, str) else list(notes), bool(d.get("blocked", False)),
                   {k: v for k, v in d.items() if k not in known})


@dataclass
class Route:
    nodes: list[str]
    length_m: float
    edges: list[Edge]


def default_meta(source: str, scale_known: bool = True, **kw) -> dict:
    return {"source": source, "created": now_iso(), "confirmed_by": None, "confirmed_at": None,
            "scale_known": scale_known, "anchors_used": [], "revision": 1, **kw}


@dataclass
class SiteMap:
    nodes: dict[str, Node]
    edges: list[Edge]
    meta: dict
    layers: list[dict] = field(default_factory=list)       # background images, e.g. drone top-down
    landmarks: list[dict] = field(default_factory=list)    # non-navigable features (ponds, buildings) for anchors
    problems: list[str] = field(default_factory=list)      # load-time issues (duplicate ids), reported by validate()

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_dict(cls, d: dict) -> "SiteMap":
        if d.get("schema_version", SCHEMA_VERSION) > SCHEMA_VERSION:
            raise ValueError(f"site map schema {d['schema_version']} is newer than {SCHEMA_VERSION}")
        nodes, problems = {}, []
        for nd in d.get("nodes", []):
            n = Node.from_dict(nd)
            if n.id in nodes:
                problems.append(f"duplicate node id {n.id!r} (later one ignored)")
                continue
            nodes[n.id] = n
        meta = {**default_meta(d.get("metadata", {}).get("source", "robot")), **d.get("metadata", {})}
        return cls(nodes, [Edge.from_dict(e) for e in d.get("edges", [])], meta, list(d.get("layers", [])),
                   list(d.get("landmarks", [])), problems)

    @classmethod
    def load(cls, path: str | Path) -> "SiteMap":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, "frame": FRAME if self.meta.get("scale_known") else "sketch_pixels",
                "metadata": self.meta, "nodes": [n.to_dict() for n in self.nodes.values()],
                "edges": [e.to_dict() for e in self.edges], "layers": self.layers, "landmarks": self.landmarks}

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False) + "\n")
        return p

    # ------------------------------------------------------------------ queries
    @property
    def revision(self) -> int:
        return int(self.meta.get("revision", 1))

    @property
    def confirmed(self) -> bool:
        return bool(self.meta.get("confirmed_by")) and bool(self.meta.get("confirmed_at"))

    def houses(self) -> list[str]:
        return sorted({n.house for n in self.nodes.values() if n.kind == "door" and n.house})

    def door_of(self, house: str) -> str:
        doors = [n.id for n in self.nodes.values() if n.kind == "door" and n.house == house]
        if not doors:
            raise KeyError(f"no door for house {house!r}; known houses: {', '.join(self.houses()) or 'none'}")
        return doors[0]

    def docks(self) -> list[str]:
        return [n.id for n in self.nodes.values() if n.kind == "dock"]

    def edge_length(self, e: Edge) -> float:
        if e.length_m is not None:
            return float(e.length_m)
        (x1, y1), (x2, y2) = self.nodes[e.a].xy, self.nodes[e.b].xy
        return math.hypot(x2 - x1, y2 - y1)

    def find_edge(self, a: str, b: str) -> Edge:
        for e in self.edges:
            if e.key() == frozenset((a, b)):
                return e
        raise KeyError(f"no edge {a}–{b}")

    def nearest_node(self, xy, kinds=NODE_KINDS) -> str:
        return min((n for n in self.nodes.values() if n.kind in kinds),
                   key=lambda n: math.hypot(n.xy[0] - xy[0], n.xy[1] - xy[1])).id

    def validate(self) -> list[str]:
        """Warnings only (a draft may be incomplete): duplicates, unknown nodes, bad kinds, disconnected nodes."""
        w = list(self.problems)
        for n in self.nodes.values():
            if n.kind not in NODE_KINDS:
                w.append(f"node {n.id}: unknown kind {n.kind!r}")
            if n.kind == "door" and not n.house:
                w.append(f"door {n.id} has no house")
        seen = set()
        for e in self.edges:
            for end in (e.a, e.b):
                if end not in self.nodes:
                    w.append(f"edge {e.a}–{e.b}: unknown node {end!r}")
            if e.a == e.b:
                w.append(f"edge {e.a}–{e.b} is a self-loop")
            if e.key() in seen:
                w.append(f"duplicate edge {e.a}–{e.b}")
            seen.add(e.key())
            if e.length_m is not None and e.length_m <= 0:
                w.append(f"edge {e.a}–{e.b}: non-positive length {e.length_m}")
        comps = self._components()
        if len(comps) > 1:
            main = max(comps, key=len)
            for c in comps:
                if c is not main:
                    w.append(f"disconnected from the main graph: {', '.join(sorted(c))}")
        if self.meta.get("source") not in SOURCES:
            w.append(f"unknown source {self.meta.get('source')!r}")
        return w

    def _components(self) -> list[set[str]]:
        adj = {n: set() for n in self.nodes}
        for e in self.edges:
            if e.a in adj and e.b in adj:
                adj[e.a].add(e.b); adj[e.b].add(e.a)
        left, comps = set(adj), []
        while left:
            stack, comp = [left.pop()], set()
            while stack:
                u = stack.pop(); comp.add(u)
                for v in adj[u] - comp:
                    left.discard(v); stack.append(v)
            comps.append(comp)
        return comps

    def route(self, start: str, goal: str, robot_width_m: float = ROBOT_WIDTH_M, margin_m: float = SIDE_MARGIN_M) -> Route:
        """Dijkstra over open edges wide enough for the robot (unknown width is allowed). Raises NoRoute with a reason."""
        for n in (start, goal):
            if n not in self.nodes:
                raise NoRoute(f"unknown node {n!r}")
        need = robot_width_m + 2 * margin_m
        adj: dict[str, list[tuple[str, Edge]]] = {}
        closed = narrow = 0
        for e in self.edges:
            if e.a not in self.nodes or e.b not in self.nodes:
                continue
            if e.blocked:
                closed += 1; continue
            if e.width_m is not None and e.width_m < need:
                narrow += 1; continue
            adj.setdefault(e.a, []).append((e.b, e)); adj.setdefault(e.b, []).append((e.a, e))
        dist, prev, pq = {start: 0.0}, {}, [(0.0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == goal:
                break
            if d > dist.get(u, math.inf):
                continue
            for v, e in adj.get(u, []):
                nd = d + self.edge_length(e)
                if nd < dist.get(v, math.inf):
                    dist[v], prev[v] = nd, (u, e)
                    heapq.heappush(pq, (nd, v))
        if goal not in dist:
            why = []
            if closed:
                why.append(f"{closed} closed edge(s)")
            if narrow:
                why.append(f"{narrow} edge(s) narrower than {need:.2f} m")
            raise NoRoute(f"no route {start} → {goal}" + (f" (excluded: {', '.join(why)})" if why else " (not connected)"))
        nodes, edges, u = [goal], [], goal
        while u != start:
            u, e = prev[u]
            nodes.append(u); edges.append(e)
        return Route(nodes[::-1], round(dist[goal], 3), edges[::-1])

    # ------------------------------------------------------------------ edits
    def confirm(self, by: str) -> "SiteMap":
        m = copy.deepcopy(self)
        m.meta.update(confirmed_by=by, confirmed_at=now_iso())
        return m

    def _bump(self, what: dict) -> None:
        self.meta["revision"] = self.revision + 1
        self.meta.update(confirmed_by=None, confirmed_at=None)
        self.meta.setdefault("history", []).append({**what, "revision": self.revision, "at": now_iso()})

    def patch(self, ops: dict | list[dict]) -> "SiteMap":
        """Apply edit operations to a copy. Each op is {"op": ..., ...}:
        close_edge/open_edge {a,b,reason?}, add_note {a,b,note} (or {node,note}), set_width {a,b,width_m},
        set_surface {a,b,surface}, add_node {node}, move_node {id,xy}, remove_node {id}, add_edge {edge},
        remove_edge {a,b}. Every applied op bumps the revision and clears the confirmation."""
        m = copy.deepcopy(self)
        for op in [ops] if isinstance(ops, dict) else ops:
            kind = op.get("op")
            if kind in ("close_edge", "open_edge", "set_width", "set_surface", "remove_edge") or \
                    (kind == "add_note" and "a" in op):
                e = m.find_edge(op["a"], op["b"])
                if kind == "close_edge":
                    e.blocked = True
                    if op.get("reason"):
                        e.notes.append(f"closed: {op['reason']}")
                elif kind == "open_edge":
                    e.blocked = False
                elif kind == "set_width":
                    e.width_m = None if op.get("width_m") is None else float(op["width_m"])
                elif kind == "set_surface":
                    e.surface = str(op["surface"])
                elif kind == "remove_edge":
                    m.edges.remove(e)
                else:
                    e.notes.append(str(op["note"]))
            elif kind == "add_note":
                n = m.nodes[op["node"]]
                n.extra.setdefault("notes", []).append(str(op["note"]))
            elif kind == "add_node":
                n = Node.from_dict(op["node"])
                if n.id in m.nodes:
                    raise ValueError(f"node {n.id!r} already exists")
                m.nodes[n.id] = n
            elif kind == "move_node":
                m.nodes[op["id"]].xy = tuple(float(v) for v in op["xy"][:2])
            elif kind == "remove_node":
                del m.nodes[op["id"]]
                m.edges = [e for e in m.edges if op["id"] not in (e.a, e.b)]
            elif kind == "add_edge":
                e = Edge.from_dict(op["edge"])
                if any(x.key() == e.key() for x in m.edges):
                    raise ValueError(f"edge {e.a}–{e.b} already exists")
                m.edges.append(e)
            else:
                raise ValueError(f"unknown patch op {kind!r}")
            m._bump(dict(op))
        return m

    def merge(self, other: "SiteMap") -> "SiteMap":
        """Union of two maps in the same frame. Same node id: keep ours. Same edge: notes joined, blocked if either is,
        narrowest known width. The result is a new unconfirmed revision."""
        if bool(self.meta.get("scale_known")) != bool(other.meta.get("scale_known")):
            raise ValueError("cannot merge a metric map with a pixel-unit sketch; align the sketch first")
        m = copy.deepcopy(self)
        for nid, n in other.nodes.items():
            m.nodes.setdefault(nid, copy.deepcopy(n))
        for e in other.edges:
            try:
                mine = m.find_edge(e.a, e.b)
            except KeyError:
                m.edges.append(copy.deepcopy(e)); continue
            mine.notes += [x for x in e.notes if x not in mine.notes]
            mine.blocked = mine.blocked or e.blocked
            widths = [w for w in (mine.width_m, e.width_m) if w is not None]
            mine.width_m = min(widths) if widths else None
        m.layers += [l for l in other.layers if l not in m.layers]
        m.meta["anchors_used"] = sorted(set(m.meta.get("anchors_used", [])) | set(other.meta.get("anchors_used", [])))
        m._bump({"op": "merge", "other_source": other.meta.get("source"), "other_revision": other.revision})
        return m


def summary(m: SiteMap) -> dict:
    return {"source": m.meta.get("source"), "revision": m.revision, "confirmed_by": m.meta.get("confirmed_by"),
            "confirmed_at": m.meta.get("confirmed_at"), "scale_known": m.meta.get("scale_known"),
            "houses": m.houses(), "docks": m.docks(), "nodes": len(m.nodes), "edges": len(m.edges),
            "closed_edges": [f"{e.a}–{e.b}" for e in m.edges if e.blocked],
            "notes": [f"{e.a}–{e.b}: {n}" for e in m.edges for n in e.notes][:50], "warnings": m.validate()}


def main(argv=None) -> int:
    """python -m r1_nav.site_map show|route|patch|confirm MAP.json ..."""
    import argparse
    ap = argparse.ArgumentParser(description="Inspect / edit / confirm a farm site map")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("show", "route", "patch", "confirm"):
        p = sub.add_parser(name)
        p.add_argument("map", type=Path)
        if name == "route":
            p.add_argument("start", help="node id or house id"); p.add_argument("goal", help="node id or house id")
            p.add_argument("--robot-width", type=float, default=ROBOT_WIDTH_M)
        if name == "patch":
            p.add_argument("ops", help='JSON op or list, e.g. \'{"op":"close_edge","a":"x","b":"y","reason":"mud"}\'')
            p.add_argument("--out", type=Path, help="default: overwrite MAP")
        if name == "confirm":
            p.add_argument("--by", required=True, help="name of the HUMAN who reviewed the map")
    a = ap.parse_args(argv)
    m = SiteMap.load(a.map)
    if a.cmd == "show":
        print(json.dumps(summary(m), ensure_ascii=False, indent=1))
    elif a.cmd == "route":
        ends = [x if x in m.nodes else m.door_of(x) for x in (a.start, a.goal)]
        try:
            r = m.route(*ends, robot_width_m=a.robot_width)
            print(json.dumps({"nodes": r.nodes, "length_m": r.length_m,
                              "notes": [f"{e.a}–{e.b}: {n}" for e in r.edges for n in e.notes]}, ensure_ascii=False, indent=1))
        except NoRoute as e:
            print(json.dumps({"error": str(e)}, ensure_ascii=False)); return 1
    elif a.cmd == "patch":
        m2 = m.patch(json.loads(a.ops))
        m2.save(a.out or a.map)
        print(json.dumps({"revision": m2.revision, "confirmed": m2.confirmed, "saved": str(a.out or a.map)}))
    else:
        m.confirm(a.by).save(a.map)
        print(json.dumps({"revision": m.revision, "confirmed_by": a.by}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
