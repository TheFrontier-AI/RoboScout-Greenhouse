---
name: farm-mission-planner
description: "Plan and supervise a greenhouse survey on the map the robot built itself: read the structure it found (plant aisles, dock, drivable space), ask it to plan, check every leg and the picture of the plan, approve or change the order, send the mission, and decide what to do when the robot reports something unexpected. No stored route."
version: 0.1.0
author: TheFarm R1 team
license: Apache-2.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Robotics, Greenhouse, Mission planning, Physical AI, Nemotron, Navigation]
    related_skills: [r1-robot-control, farm-site-map]
prerequisites:
  commands: [curl, python3]
---

# Farm Mission Planner (map only, no stored route)

You are the mission decision-maker. The robot is the geometry: it reads its own LiDAR map, finds the plant aisles and
the space it has seen empty, plans every path itself (A* with a clearance cost) and keeps checking where it is. You
decide **what** the mission is and **whether the plan is good**; you never send coordinates or steer.

Hub (`agent/robot_api.py`, `R1_HUB`, default `http://127.0.0.1:8090`). The structure comes from `scripts/free_nav.py`
run on the robot's map (`R1_STRUCTURE` points the hub at its output).

## When to use
- The farmer asks for a survey ("do the whole farm", "check house B first", "skip the wet aisle today").
- The robot was put down somewhere and has found itself (`/status` has `xy`), and no mission is running.
- The robot reports a situation during a mission (`/situations`): an aisle blocked, a place that does not look like
  the map, an obstacle that was not on the map, a camera verdict that it is not where it thought.

## Steps
1. **Read what the robot found.** `curl $R1_HUB/structure` → plant aisles (id, both ends, length, width,
   reachable), the dock (where it was switched on), drivable area, aisles seen but not reachable. Tell the farmer in
   plain words: "I found 2 aisles I can drive, 40 and 41 m long; 2 side aisles are too narrow or unseen."
2. **Ask for a plan.** `curl -X POST $R1_HUB/plan -d '{"aisles": ["aisle-03", "aisle-07"]}'` (omit `aisles` for every
   reachable aisle). The robot solves the order itself as a Rural Postman Problem (every chosen aisle is a required
   edge driven end to end, the connections are A* distances on its map, start = where it is, end = the dock; exact up
   to 7 aisles, well under a second) and returns it as `rural_postman`. You get every leg with its metres, `total_m`,
   `aisles_missing_from_order`, the shortest `alternatives`, and `picture` (a JPEG of the map with the drivable space,
   the aisles, the dock, the robot and the proposed path). Pass `order` only to override the solver (the farmer wants
   an aisle first); never invent a route of your own. **Look at the picture**: the path must stay inside the green area, pass every aisle end to
   end once, and finish at the dock.
3. **Decide.** Choose the order by what the farmer asked (an aisle first, an aisle skipped), then by `total_m`. If a
   leg has `metres: null` the robot cannot reach that place from where it is: drop that aisle and say so.
4. **Send the mission.** `curl -X POST $R1_HUB/mission_plan -d '{"order": [...], "approved_by": "HERMES", "reason": "..."}'`.
   The hub stores the approved plan (`mission_plan.json`), logs it, and queues `survey_aisles` for the robot.
   The robot plans each leg again from where it believes it is every time its belief changes.
5. **Supervise.** Poll `/status` and `/situations`. The robot handles the reflexes itself (stops for obstacles, plans
   around what is not on its map, searches the whole map again when the scans contradict its belief, asks its camera
   before committing to an aisle or the dock). It reports to you when a decision is yours:

| the robot reports | you decide |
|---|---|
| "aisle X blocked at metre M, no way round" | skip X and continue, or wait and retry, or return to the dock; tell the farmer |
| "this does not look like aisle X, camera says it is aisle Y" | accept the relabel (the robot already did it) and let it continue, or stop and ask the farmer |
| "obstacle not on the map at (x, y), planned around, +N m" | nothing, unless N is large: then re-plan the order with `/plan` |
| "cannot plan from where I believe I am" | ask the robot to probe (it steps into open space and searches again); if it still fails, tell the farmer where it last knew it was |
| "battery low" | return to the dock now; the records so far are kept |

## Rules
- Approve only a plan whose picture you have looked at and whose `aisles_missing_from_order` is empty (or the farmer
  said to skip them).
- Never invent an aisle id; use the ids from `/structure`.
- Say what you decided and why, in one sentence, every time (`reason`). The farmer reads it.
- Offline check of the same loop (synthetic scans): `python3 scripts/freenav_mission_eval.py ... --decider nemotron`;
  its evidence is `docs/evidence/17_freenav_mission_survey.json`.
