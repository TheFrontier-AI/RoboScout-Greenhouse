---
name: r1-motion-policy
description: "Pick the task mode and travel speed for the R1 greenhouse robot (inspect, revisit close-up, transit, watering) from camera blur/overlap limits and the mission safety rules."
version: 0.1.0
author: TheFarm R1 team
license: Apache-2.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Robotics, Greenhouse, Physical AI, Navigation, Speed, Safety]
    related_skills: [r1-robot-control, plant-disease-diagnosis, greenhouse-watering]
prerequisites:
  commands: [curl]
---

# R1 Motion Policy (mode + speed)

Use with `r1-robot-control`. That skill says **where** R1 goes. This skill says **how it should move there**
(which mode, what speed, when to stop) and how to explain the choice to the farmer.

**Today's API has no speed field.** The robot always drives at its built-in limits (cruise 0.25 m/s forward,
0.15 m/s reverse, side images every 0.5 m, revisit hold 3 s). You influence motion only by choosing the
command: `survey_aisle`/`survey_house` (continuous), `revisit` (stop-and-hold), `go_to` (stop, no hold),
`stop`. Everything under "Proposed API extension" below is **not implemented**. Never send those fields
until `/health` or the robot team says they exist; the robot would reject or ignore them.

## When to Use
- Before sending any motion command, to choose between survey, revisit and go_to.
- When the farmer asks "how fast should it go to check for disease / to water?"
- When images come back blurred, frames have gaps, the floor is wet, or the robot reports `stuck`.

## Hard Safety Limits (MISSION_SPEC, enforced on the robot, never exceed in any proposal)
| Item | Limit |
|---|---|
| Forward speed | ≤ 0.25 m/s (wheel ≤ 2.0 rad/s, r = 0.127 m) |
| Reverse speed | ≤ 0.15 m/s (wheel ≤ 1.2 rad/s) |
| Obstacle stop | LiDAR obstacle ahead < 0.6 m → robot stops, waits; overrides every command |
| Turning | never inside an aisle; only on the headland with ≥ 1.2 m free |
| Narrow aisles | gap < 0.53 m + 2×0.06 m = 0.65 m → not traversable; report, don't attempt |
| E-stop / SAFETY / STUCK | always on the robot, not in HERMES |
HERMES may ask for **slower**, never faster. People in the aisle → `stop` and wait for a human.

## Modes
| Mode | How to command today | Target speed | Capture | Use for |
|---|---|---|---|---|
| `transit` | `go_to` / headland legs (automatic) | ≤ 0.25 fwd, ≤ 0.15 rev | front camera every 2 m only | moving between aisles, docking |
| `inspect` (continuous scan) | `survey_aisle` / `survey_house` | 0.20–0.25 m/s if exposure ≤ ~4 ms; 0.10–0.15 m/s in dim light | 6 side cams every 0.5 m | routine daily scouting |
| `inspect-stopgo` | not available (proposed `mode`) | stop at every capture point | settle ≥ 0.5 s, then capture | dim light / long exposure / high-res cameras |
| `revisit` (close-up) | `revisit` (aisle, metre) | approach ≤ 0.15 m/s for the last 1 m (proposed); full stop | hold 3 s, all 6 side cams (purpose=revisit) | confirming a suspect spot |
| `watering` | **not available** (see `greenhouse-watering`) | 0.10–0.15 m/s between plants, 0 while spraying | – | future spot watering |

## Researched numbers (details and sources in `references/speed_derivation.md`)
- **Motion blur** (pixels) = speed × exposure ÷ ground-sample distance. Keep ≤ 1 px for lesion detail
  (machine-vision rule of thumb; 0.5–2 px "not visible" but still hurts measurement).
- R1 side camera now: 56° H × 87° V portrait. At 0.4 m from the leaves one frame covers ~0.43 m of row;
  at 0.6 m ~0.64 m.
  - 360 px wide: ~1.2 mm/px at 0.4 m → max exposure for 1 px blur ≈ 4.7 ms at 0.25 m/s, 7.9 ms at 0.15, 11.8 ms at 0.10.
  - 1080 px wide: ~0.39 mm/px → ≈ 1.6 ms at 0.25 m/s, 2.6 ms at 0.15, 3.9 ms at 0.10.
  - So with a real high-res camera in a dim greenhouse (exposure often 10–30 ms), continuous scanning at
    0.25 m/s blurs 6 px or more → use 0.10 m/s or stop-and-go.
- **Overlap at 0.5 m spacing:** with 56° H at 0.4–0.46 m the frames **do not touch** (gap up to 7.5 cm).
  At 0.6 m there is about 22 % overlap. Aim for ≥ 30 % (every leaf seen at least once, often twice):
  capture step ≤ 0.30 m at 0.4 m distance, or a 90° H landscape camera (0.5 m step → 38 % overlap at 0.4 m).
- **Time budget** (38 m aisle, acceptance criterion: ≤ 8 min round trip):
  forward 0.25 + reverse 0.15 → ~6.8 min OK; forward 0.20 → ~7.4 min OK; forward 0.15 → ~8.4 min (misses);
  stop-and-go every 0.5 m → ~8.7 min (misses). Continuous at 0.20–0.25 m/s is the default; spend the slow
  modes only on spots that need it (revisits).
- Field references: commercial greenhouse rail sprayers run up to 0.67–1.17 m/s, but spraying, not
  imaging. A free-driving greenhouse sprayer study tested 0.15/0.25/0.35 m/s and found position error grows
  with speed. R1's 0.25 m/s cap is at the cautious end. That suits a robot that must also keep camera blur low.

## Decision Procedure
1. Read `/status` and `/health`. If `state` is `stuck`/`aborted`, see **Stuck** below; do not move it.
2. Pick the mode from the task: daily scouting → `inspect`; spot from findings → `revisit`;
   go somewhere without looking → `go_to` (transit); watering → refuse today (`greenhouse-watering`).
3. Check image quality from recent `/findings`: if ≥ 30 % of the last 20 side frames in an aisle say
   "blurred", "motion blur", "out of focus" or `suspect` only because of image quality:
   - today: queue `revisit` at the worst metres (stop-and-hold gives sharp frames) and tell the farmer the
     cruise speed is too fast for the light. Suggest more light or a shorter exposure.
   - with the extension: re-survey that aisle at `speed_mps: 0.10` or `mode: "inspect_stopgo"`.
4. If frames show gaps, meaning a suspect leaf is cut at a frame edge and the next frame is 0.5 m later, use `revisit`
   at that metre instead of trusting the single frame.
5. Always write the speed reason into `reasoning`, e.g.
   `"revisit 15.0 m: stop-and-hold for sharp close-up; continuous pass was blurred in low light"`.

## Wet floor, slip and stuck
- Wet floor (after irrigation, condensation, or any watering spill): the robot already has STUCK detection.
  HERMES should prefer fewer stops and starts (stop/start is where wheels slip). Do not queue many
  revisits in a wet aisle; merge them into one `survey_aisle` later or one revisit per 2 m.
  With the extension: ≤ 0.15 m/s forward, ≤ 0.10 m/s reverse on wet floor (engineering choice, not measured).
- `stuck` once: send `stop`, report aisle and metre, wait for a human. Never resend the same goal (rule from
  `r1-robot-control`). `stuck` twice in a mission → `return_to_dock` after the human clears it.
- `pose_confidence` < 0.5: don't pin findings to a metre. Survey the aisle again rather than revisiting a metre.
- Battery < 25 %: `return_to_dock`; slow modes cost battery time, so drop stop-and-go first.

## Proposed API extension (NOT implemented: for the robot team)
Optional fields on `POST /command`. The robot clamps every value to its limits and replies with the value it applied:
```jsonc
{"cmd": "survey_aisle", "aisle": "tunnel-09/aisle-02",
 "mode": "inspect|inspect_stopgo|transit|revisit|watering",
 "speed_mps": 0.15,            // forward; robot clamps to ≤ 0.25 (≤ 0.15 on wet floor)
 "reverse_speed_mps": 0.10,    // robot clamps to ≤ 0.15
 "capture_step_m": 0.30,       // side-camera spacing (default 0.5)
 "settle_s": 0.5,              // stop-and-go: wait after stopping before capture
 "hold_s": 3.0,                // revisit hold (today fixed REVISIT_HOLD_S = 3.0)
 "reasoning": "dim light: slower pass for sharp frames"}
```
Also useful in `/status`: `speed_mps`, `exposure_ms`, `floor_wet` (bool) and `slip_events`. With these,
HERMES could compute `speed = min(limit, blur_px_max × GSD / exposure)` itself.
Until they exist, describe them as "proposed" to the farmer and judges, never as working.
