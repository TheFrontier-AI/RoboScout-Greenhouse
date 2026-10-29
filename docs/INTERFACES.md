# Hand-off contracts between the parts

Every part talks through the **robot API** (`agent/robot_api.py`, port 8090), which also stores everything in the data dir for the dashboard.
```
HERMES (Nemotron @ Token Factory) ──POST /command──▶ robot API ◀──GET /next_command── robot (Isaac sim / Jetson)
          ▲ GET /status /findings /map                  │  ▲ POST /status, POST /frame
          └─────────────────────────────────────────────┤  │
                                     Cosmos Reason ◀────┘ (side-camera images → findings)
```

## 1) HERMES → robot (goals only, never wheel commands)
`POST /command` JSON; always include a short `reasoning` (shown on the dashboard/video):
| cmd | fields | meaning |
|---|---|---|
| `discover_house` | – | a house with no map: SLAM builds the map + finds the aisle openings by itself |
| `survey_house` | `house` | cover every passable aisle (drive in, reverse out) |
| `survey_aisle` | `aisle` | one aisle, head to end |
| `go_to` | `aisle`, `metre` | go and stop at a point (metre 0 = the aisle head on the door side) |
| `revisit` | `aisle`, `metre` | go there and hold still for 3 seconds so all 6 side cameras capture (purpose=revisit) |
| `return_to_dock` / `stop` | – | return to the dock / stop immediately (clears the queue) |
| `set_map` | `path` | use the human-confirmed map after discover |
| `go_to_house` | `house`, `map_source`? | leave the current house → follow the confirmed farm map (site map) → enter the target house's door |
| `survey_house` (a different house) | `house`, `map_source`? | `go_to_house` first, then survey every aisle (if that house has no aisle map yet → `discover`), then return to the dock |

Response: `{"status": "accepted|rejected", "reason": "...", "legs": [...]}`; the robot validates every command against the map (`r1_nav/commands.py`)

### Farm map (site map) and cross-house commands
- A graph of `dock / door / junction / waypoint` + paths (`length_m`, `width_m` or null, `surface`, `notes[]`, `blocked`) in farm coordinates
  (metres, the same frame as `house_local_to_world` of the aisle map). Details: [SITE_MAP.md](SITE_MAP.md)
- The robot accepts only maps that are **human-confirmed** (`confirmed_by`) and have a metre scale; the robot API loads them from env `R1_SITE_MAP` (several files separated by `,`)
- `map_source` = `v18 | drone | hand_sketch | robot` selects the map source (omitted = the latest confirmed one)
- New legs in `legs`: `exit_house` (reverse out of the aisle to the door), `site_transit` (one leg per path, with `from_node`, `to_node`, `length_m`
  and the path's notes in `note`), `enter_house` (align at the door, then enter)
- Rejected with a reason when: no map, not confirmed, still in pixel units, unknown house, no route (blocked / narrower than robot width + 2×0.06 m)
- The `reason` on acceptance states the route, the distance and every note along the route (e.g. "wet after watering") so HERMES can tell the farmer

## 2) Robot → cloud (images + pose + status)
**7 cameras** (`r1_nav/cameras.py`): 6 side cameras `mast_{left,right}_{low,mid,high}` look at the plants, captured every 0.5 m;
1 front camera `front` for navigation, captured every 2 m so the cloud can check that the robot drove/turned correctly

`POST /frame` multipart: `meta` (JSON) + `image` (JPEG)
```json
{"ts": 0.0, "house": "tunnel-09", "aisle": "tunnel-09/aisle-02", "metre": 15.0, "heading_deg": 270.0,
 "camera": "mast_left_mid", "frame_id": 123, "pose_confidence": 0.92, "purpose": "inspect|navigate|revisit"}
```
`heading_deg` = the robot's heading in the house frame, counter-clockwise from the +X axis (driving from the door inward along −Y = −90)
`pose_confidence` comes from SLAM; below 0.5 → the image is not sent for disease inspection and no pin may be placed at that metre

`POST /status`: `{"state": "idle|running|done|stuck|aborted", "cmd": ..., "aisle": ..., "metre": ..., "heading_deg": ...,
"pose_confidence": ..., "battery": <percent 0–100>, "legs_remaining": n}`

## 3) Cosmos Reason → findings
Per side-camera image: `{"status": "healthy|suspect|disease|not_plant", "issues": [{"label", "confidence", "bbox_norm", "evidence"}], "summary"}`
`bbox_norm` = `[x0, y0, x1, y1]` as fractions 0–1 of the image (top-left to bottom-right) or `null`
Readable at `GET /findings?since=N`

## 4) Low-level layer on the robot (no network dependency)
`wheel_cmd = [left_rad_s, right_rad_s]` at 20 Hz; SAFETY (e-stop, stop when an obstacle is < 0.6 m, STUCK) always runs on the robot
Network loss: finish the current leg → reverse out and wait at the aisle head → keep the images in the outbox and send when reconnected

## 5) Robot → server: survey finished (`POST /run_complete`)
`{"run": "<run name>", "path": "<rec folder on the server>", "summary": {...}, "source": "robot"}`, sent once when `run_farm.py` finishes
(in the sim `rec/` is a shared mount; the real robot rsyncs first with `--upload-rec`). The hub writes `<path>/run_complete.json` and the analysis system
(`cloud/pipeline`: RT-DETR, VLM per stop, LiDAR map, cuVSLAM, lingbot, Cosmos front, report) runs by itself → `<path>/analysis/report.json`
+ `report_th.md` + `report_en.md`. Response `{"ok": true, "run", "path", "pipeline": "marker written|spawned"}`; path does not exist → 400
Details: [PIPELINE.md](PIPELINE.md)
