# Farm map (site map) — driving between greenhouses

HERMES commands at the goal level, e.g. "survey house B" while the robot is in house A. The robot does not plan the route itself: `r1_nav/commands.py` finds the route on the
**human-confirmed farm map** and breaks it into legs (`exit_house` → `site_transit` one per path → `enter_house` → the in-house legs)

## File format (`r1_nav/site_map.py`, schema_version 1)
- `nodes`: `{id, kind: dock|door|junction|waypoint, xy: [x, y] in metres, house (door only), heading_deg (direction of driving into the house), label}`
- `edges`: `{a, b, length_m (null = computed from xy), width_m (null = unknown), surface, notes[], blocked}`
- `metadata`: `{source: v18|drone|hand_sketch|robot, created, confirmed_by, confirmed_at, scale_known, anchors_used[], revision, history[]}`
- `layers` (background images, e.g. a drone top-down), `landmarks` (ponds, buildings, area notes — not used for routing)
- Coordinates: the same frame as `house_local_to_world` in `data/tunnel09_aisle_map_v18.json` (Blender Z up, glTF `(x, y, z)` → `(x, −z, y)`)

## Rules
- Routing with Dijkstra over edges that are not `blocked` and whose `width_m` is null or ≥ robot width + 2×0.06 m (the same rule as for aisles inside the house)
- Only maps with a non-empty `confirmed_by` and `scale_known = true` may be used
- Every edit (`patch`: block/unblock a path, add a note, set a width, add/remove a node or edge) and every map merge (`merge`) → `revision + 1` and the confirmation is cleared
- The farm's "learning" = keeping several map versions with the farmer's notes, not training a model

## 4 sources
| source | how | notes |
|---|---|---|
| `v18` | `python3 scripts/extract_site_map_v18.py` → `data/site_map_v18_draft.json` | 22 houses (tunnel 12 + block 10), doors from the view preset `entrance`/aisle map, 5 roads from `site-layout-v5.json`; the doors of the 2 east greenhouses are estimates; the blueprint scale has not been measured on site |
| `drone` | the cloud SLAM job → `python -m cloud.maps.drone --slam DIR --pair COL,ROW=X,Y ...` | aligns the top-down image to the farm frame with ≥ 2 reference points (Umeyama), then attaches it as a background layer |
| `hand_sketch` | a photo of a hand-drawn map, or a drawing made on the V18 web view → `python -m cloud.maps.sketch img.jpg --anchors-from data/site_map_v18_draft.json` | Nemotron-3.5-Super-VL (Token Factory) reads it into JSON; ≥ 2 labels matching reference points → metres, otherwise pixels |
| `robot` | the route the robot actually drove and recorded | the most trustworthy once confirmed |

## Workflow
```bash
python3 -m r1_nav.site_map show data/site_map_v18_draft.json                      # summary for a human to check
python3 -m r1_nav.site_map route data/site_map_v18_draft.json tunnel-09 south-greenhouse-block-01
python3 -m r1_nav.site_map patch data/site_map_v18_draft.json '{"op":"add_note","a":"road/13","b":"road/14","note":"wet after watering"}'
python3 -m r1_nav.site_map confirm data/site_map_v18_draft.json --by "reviewer name"     # humans only
R1_SITE_MAP=data/site_map_v18_draft.json python3 agent/robot_api.py ...             # the robot API uses this map
```

## Limitations
- The V18 draft uses the blueprint roads + short straight lines to the doors; every edge width is unknown, so a human must check and edit before confirming
- Driving between houses (`exit_house`/`site_transit`/`enter_house`) is not yet simulated in Isaac (`sim/executor.py` skips these legs like `transit`)
