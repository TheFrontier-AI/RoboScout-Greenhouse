---
name: farm-site-map
description: "Build, show, edit and get human confirmation of the farm site map (paths between greenhouses) that R1 uses for go_to_house / cross-house surveys. Sources: V18 model, drone SLAM, farmer sketch, robot recording."
version: 0.1.0
author: TheFarm R1 team
license: Apache-2.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Robotics, Greenhouse, Mapping, Physical AI, Nemotron, Navigation]
    related_skills: [r1-robot-control]
prerequisites:
  commands: [python3]
---

# Farm Site Map (between greenhouses)

The **site map** is a small graph in site metres (same frame as the aisle maps' `house_local_to_world`):
nodes = `dock`, `door` (one per greenhouse, `house` = house id), `junction`, `waypoint`;
edges = paths with `length_m`, `width_m` (null = unknown), `surface`, `notes[]` (farmer's words), `blocked`.
The robot only follows routes on a map a **human has confirmed**. Every edit makes a new revision and clears the
confirmation. Code: `r1_nav/site_map.py`; docs: `docs/SITE_MAP.md`.

## When to Use
- The farmer asks to go to / survey another greenhouse and the robot API rejects it with "no site map" or "not confirmed".
- The farmer sends a photo of a hand-drawn farm map, or drew paths on the V18 web model, or says a path is closed,
  wet, too narrow, or only usable at certain times.
- A drone survey finished and its top-down should be lined up with the farm.

## The 4 map sources (`metadata.source`)
| source | how it is made | scale | typical use |
|---|---|---|---|
| `v18` | `python3 scripts/extract_site_map_v18.py` from the V18 farm model: 22 greenhouse doors, blueprint roads, straight links | metres (blueprint scale provisional) | first draft, house ids |
| `drone` | cloud SLAM job → `python -m cloud.maps.drone --slam DIR --pair COL,ROW=X,Y ... --site MAP --out MAP` aligns the top-down with ≥ 2 known points and adds it as a background layer | metres after alignment | check paths/obstacles against reality |
| `hand_sketch` | farmer photo or V18-web drawing → `python -m cloud.maps.sketch photo.jpg --anchors-from data/site_map_v18_draft.json --out data/site_map_sketch.json` (Nemotron VL reads it as strict JSON) | metres if ≥ 2 labels match known anchors, else pixels | the farmer's own knowledge: real paths, notes, closures |
| `robot` | paths the robot actually drove and recorded | metres | most trusted once confirmed |

## Tools
```bash
python3 -m r1_nav.site_map show  MAP.json                 # source, revision, confirmed?, houses, closed edges, notes, warnings
python3 -m r1_nav.site_map route MAP.json tunnel-09 south-greenhouse-block-01   # node or house ids → route + notes
python3 -m r1_nav.site_map patch MAP.json '{"op":"close_edge","a":"road/13","b":"road/14","reason":"truck parked"}'
python3 -m r1_nav.site_map confirm MAP.json --by "<name of the human>"   # ONLY after the human says yes
```
The robot API reads confirmed maps from env `R1_SITE_MAP` (comma-separated paths), reloaded when the file changes.

## Sketch workflow
1. Farmer sends a photo of a paper sketch, or draws lines/arrows/notes on the V18 web model (draw tools in the
   "Comments on model" panel) and sends a screenshot.
2. Run `python -m cloud.maps.sketch <image> --anchors-from data/site_map_v18_draft.json --out data/site_map_sketch.json`
   (or `--anchor "House A=195.07,156.67"` for labels the V18 ids don't match). It prints matched anchors, rmse and scale.
3. **Show the draft** to the farmer: `show` output in plain words (houses found, paths, notes, anything unmatched,
   `scale_known`). If `scale_known` is false, ask for two known places (e.g. "which box is tunnel-09? where is the dock?").
4. Fix what the farmer corrects with `patch`; repeat 3.
5. Only when the farmer explicitly approves: `confirm --by "<their name>"`. Then commands may use `map_source: "hand_sketch"`.

## Natural language → patch ops
| farmer says | op |
|---|---|
| "the path from A to B is closed / blocked today" | `{"op":"close_edge","a":..,"b":..,"reason":"closed today"}` |
| "it's open again" | `{"op":"open_edge","a":..,"b":..}` |
| "wet after watering", "do not pass 7-9am", "muddy path" | `{"op":"add_note","a":..,"b":..,"note":"<their words>"}` |
| "that gate is only 60 cm" | `{"op":"set_width","a":..,"b":..,"width_m":0.6}` |
| "it's concrete" | `{"op":"set_surface","a":..,"b":..,"surface":"concrete"}` |
| "there is a path from X to Y" / "remove that path" | `add_edge {"edge":{"a":..,"b":..}}` / `remove_edge` |
| "the dock is next to tunnel-05 now" | `move_node {"id":"dock","xy":[x,y]}` |
Find `a`/`b` with `show`/`route`; never guess node ids. If unsure which path they mean, ask.

## Rules
- Never confirm a map yourself and never claim a draft is confirmed. Every patch clears the confirmation: ask again.
- Widths in V18/sketch drafts are unknown (null); a route is only rejected for width when a width is set.
- Read every route note aloud to the farmer before sending the command (wet, time windows, warnings).
- "Learning" the farm = keeping versioned maps (`revision`, `history`) and the farmer's notes, and preferring confirmed
  robot-recorded paths over time. It is **not** model training; nothing is fine-tuned.
- Keep the farmer's notes in their own words (Thai or English).
