# Mission spec

## Coordinate frame
The greenhouse's local frame is used (metres, Z up): origin at the centre of the house, the door on the +Y side, aisles running along the Y axis.
A position in an aisle is given as (aisle_id, metre), where metre = 0 at the aisle head on the door side and increases into the house (matches the web team's `r1-aisle-map.json`).

## Missions
1. `go_to(aisle_id, metre)` — go and stop at a point in an aisle (always enter the aisle from the door side)
2. `survey_aisle(aisle_id)` — drive forward from the aisle head to the end (sending images every 0.5 m), then reverse out past the aisle head
3. `survey_house(house_id)` — `survey_aisle` on every passable aisle, left to right, moving between aisles across the area in front of the door (headland)
4. `return_to_dock()` — return to the charging dock in front of the house

## State machine
IDLE → ALIGN_AT_HEAD (align the robot with the aisle, using the row-head marker/LiDAR) → DRIVE_IN (drive down the aisle centre) → [STOP_OBSTACLE ↺ wait until clear]
→ AT_TARGET or AT_END → REVERSE_OUT → AT_HEAD → TRANSIT_HEADLAND → ... → DOCK
Every state has an e-stop and "stop when the LiDAR sees an obstacle ahead < 0.6 m", which overrides every other command

## Driving rules
- Forward speed ≤ 0.25 m/s, reverse ≤ 0.15 m/s (wheel radius 0.127 m → ≤ 2.0 / 1.2 rad/s)
- Hold the aisle centre: the left–right LiDAR distance difference → heading correction (a proportional controller with a limited turn rate)
- Never turn around inside an aisle; turning is only allowed on the headland in front of the door where there is ≥ 1.2 m of free space
- An aisle whose gap is < robot width + 2×0.06 m is "impassable" and is reported back instead of attempted

## Open questions needing a decision
- The side aisles of tunnel-09 in the model are 0.55 m, impassable for the R180 (0.53 m) → must be measured on site, or the robot adjusted
- The real area in front of the door (headland) has not been measured: moving between aisles needs room to turn
