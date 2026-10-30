# Taking RoboFarm to a new farm

How the robot learns a site it has never seen, why a person leads it the first time, and how you then send it to
house 5, house 6, across houses, or through every house. Everything here runs today; the figures are from our own
farm's recorded drive.

---

## 1. Why a person leads the first drive

A farm is not a warehouse. Around the greenhouses there are ponds, an orchard, a road, soft ground, pipes and places
where a 58 kg robot must never go. A robot that "explores" a new site on its own has to be told all of that first, and
it would learn it by driving into it. So RoboFarm does not explore. **The first time it sees a site, a person leads it**
— walking beside it, pushing it, or driving it with a controller — along exactly the paths the farmer wants it to use:
dock → door of house 1 → down the aisle → out the back → behind the houses → into house 2 … → home. While it is led,
the robot records what its LiDAR sees and where it believes it is.

That one led drive answers the two questions a robot cannot answer safely by itself:

- **Where may I go?** Only along the paths it was led. Everything off that graph is "do not go". The robot refuses a
  route that is not on the map.
- **What is where?** The map it recorded, the doors it passed through, the length of every path and aisle — measured
  by the robot's own sensors, not typed by anyone.

The intelligence is not in the led drive. It is in what the robot does afterwards on its own: finding itself on the map
it recorded in rows where every plant looks the same, stopping at each plant, seeing what is wrong with it, deciding to
go back for a second look, and reporting — every day, without a person.

![The farm as the robot recorded it on one taught drive](media/gallery/24_teach_map_self_built.png)

*Figure 1 — Our farm as the robot recorded it on one led drive: the two tunnel greenhouses with their planting beds,
the road to the north, the yard behind the houses. 29 615 LiDAR points, 123 m of driving, no distances typed.*

---

## 2. Step by step: a site the robot has never seen

**You need:** the robot with LiDAR on, the recorder, and 15 minutes.

1. **Place the robot at its dock.** That is the origin of the map; the robot always starts and ends there.
2. **Start the recorder.** On the physical robot: the recorder runs with the motors idle while you push or drive it.
   In simulation: any drive serves (`scripts/run_farm.py`, or a previous survey).
3. **Lead it along every path it should ever use** — into each house through its door, down the aisle, out the other
   end, along the yard to the next house, and finally home. Walk at the speed you would want it to drive. Do not take
   it anywhere it should not go on its own.
4. **Build the maps** from the recording:
   ```bash
   python3 scripts/teach_map.py --rec <run>/rec --validate
   ```
   Output in `<run>/rec/teach/`: `prior_map_xy.npy` (the LiDAR map), `site_map.json` (the site graph),
   `teach_map.png`, `teach_report.json`.
5. **Name the houses.** The graph calls them `house-1`, `house-2` … in the order you entered them (or by the aisle id
   if the recorder knew it). Rename them to what the farm calls them — `tunnel-05`, `house-6` — in `site_map.json` or
   by telling HERMES ("the first house we entered is tunnel-05").
6. **Confirm the map.** A person approves it once (`farm-site-map` skill, or `SiteMap.confirm(by="…")`). Until then
   the robot accepts commands inside the house it is in and refuses to cross to another.
7. **Point the system at the map** and start:
   ```bash
   export R1_SITE_MAP=<run>/rec/teach/site_map.json      # the hub / HERMES read the graph from here
   # prior_map_xy.npy → where the robot loads its LiDAR prior
   bash deploy/nebius/start_pipeline.sh                   # hub + pipeline
   ```

![Site graph the robot built from the taught drive](media/gallery/25_teach_site_graph.png)

*Figure 2 — The site graph the robot built from that drive: the dock (magenta), one node at each door it passed
(yellow), and the paths between them with the lengths it measured — 3.6 m to the first door, 42 m through tunnel-09,
12.8 m behind the houses, 42 m through tunnel-10, 14 m home. Anything off the yellow graph is "do not go".*

---

## 3. Sending it somewhere: house 5, house 6, across houses, every house

**Who plans what.** HERMES plans the *mission*: which houses to survey today and in what order, which plant to go
back to, when to come home, whom to tell. The *route* — the actual path from where the robot is to the door of the
house it was sent to — is planned by the robot itself on the site graph (`r1_nav/site_map.py` finds the shortest
confirmed path; `r1_nav/commands.py` turns it into legs), and the *driving* along that route is the robot's
navigation stack alone. HERMES never sees a coordinate or a wheel; it names a destination, and the robot gets there.

Once the map is confirmed, you command by **name**, never by coordinates. The robot plans the route on the graph it
built and localises on the map it recorded. Commands go to the hub (`POST /command`) — from HERMES in plain
language, from the dashboard, or from a terminal:

| You want | Say to HERMES | Command the hub receives |
|---|---|---|
| Survey house 5 | "survey tunnel-05" | `{"cmd":"survey_house","house":"tunnel-05"}` |
| Go to house 6 and wait | "go to house 6" | `{"cmd":"go_to_house","house":"house-6"}` |
| From house 5, cross to house 6 and survey it | "now do house 6" | `{"cmd":"survey_house","house":"house-6"}` — the robot leaves house 5 by the door on the graph, follows the yard edge to house 6's door, enters |
| Survey every house | "survey the whole farm" | `{"cmd":"survey_farm"}` → one drive-through of every house on the confirmed map, in map order, then dock |
| Look again at a plant | "check metre 12 in tunnel-05 again" | `{"cmd":"revisit","aisle":"tunnel-05/aisle-01","metre":12.0}` |
| Come home | "return to dock" | `{"cmd":"return_to_dock"}` |
| Stop | "stop" | `{"cmd":"stop"}` |

```bash
curl -X POST http://<SERVER>:8090/command -H 'Content-Type: application/json' \
     -d '{"cmd":"survey_house","house":"tunnel-05"}'
```

What the robot does with a cross-house command: it looks up the route on the site graph (dock → door → … → door of the
target), drives it with the LiDAR map and its odometry, enters through the door node, and surveys the aisle front to
back, stopping at every plant. If the target house is not on the graph, or the graph is not confirmed, the command is
**rejected with the list of houses the robot knows** — it never guesses a path.

**Six houses in a row** look like this on the graph after a led drive that went through all of them:

```
 dock ─3.6 m─ h1/front ══ aisle 42 m ══ h1/back ─12 m─ h2/back ══ aisle ══ h2/front ─12 m─ h3/front ══ … ══ h6/back ─── home
```
`survey_farm` walks that graph once: h1 front→back, cross behind, h2 back→front, cross in front, h3 front→back … and
returns. `survey_house h5` takes the shortest graph path from wherever the robot is to h5's nearest door.

---

## 4. How you know the map is right

- **Look at it.** `teach_map.png` shows the LiDAR map with the taught path and every door node on it (Figure 1 and 2).
  A door in the wrong place is obvious.
- **The robot checks itself.** `--validate` re-localises every recorded scan on the new map and reports how many it
  accepted and how far its fixes land from the drive: on our farm 1 500 / 1 500 scans accepted, 0.23 m RMSE — the same
  as on the hand-drawn plan (`docs/evidence/10_teach_map_run3. Returns from the robot's own mast and electronics box (about 9 beams per scan, 0.3–0.6 m behind the LiDAR) are removed before mapping; left in, they draw a trace of the robot along the taught path and the map starts to behave like a painted line.json`).
- **Watch the first run.** The dashboard (`dashboard/serve.py`, port 8080) draws the robot on the map live; the
  robot's LiDAR estimate and its stereo cuVSLAM estimate are two independent observers, and a disagreement between
  them is flagged.
- **The robot refuses what is not on the map.** It cannot be sent off the graph by mistake.

---

## 6. Put the robot down anywhere: how it finds itself

Normally a survey starts at the dock. If someone puts the robot down somewhere else on a path it knows, it does not
assume it is at the dock — it works out where it is, in three steps, and says so when it cannot:

1. **LiDAR candidates (80 ms).** `sim/relocalize.py` tries every place along the taught path (every 0.5 m, both
   directions — 1 476 hypotheses on our farm), scores each by how well the current scan fits the map there, and refines
   the best with the same matcher the robot drives with. Outdoors — at the dock, on the road, in the yard — one place
   wins. Inside an aisle the answer is honest: "tunnel-09 or tunnel-10, same spot in either", because the two houses are
   identical to a LiDAR. On run 3, from a single scan at 40 random places: 8 found outright (5 cm median error),
   30 flagged ambiguous, 2 wrong.
2. **The camera decides (5 s).** For an ambiguous pair the robot's front image and a reference image from the led
   drive at each candidate place go to Nemotron 3.5 Super VL with one question: which reference is where I am? It
   answers with evidence a LiDAR cannot see — the lone tree visible through the far door, the grey block at the base of
   the white wall, the irrigation line, the building behind the house. On 16 ambiguous run-3 scans it decided 12 and
   was right in 10 (83 %); the other 4 ran out of reply length (`scripts/vlm_place_check.py`).
3. **Driving confirms.** The surviving candidates are carried along as the robot moves; one that stops fitting the map
   dies. Driving alone is not allowed to choose between two symmetric places — we measured that it chooses wrong (9 of
   10) in identical houses — so the choice stays with the camera or with reaching a place that is not symmetric
   (a door, the dock, the road), and until then the robot drives only what both candidates allow.

**Off the taught path.** A robot put down beside a path sees surfaces the led drive never recorded, so the led drive
records more: every 0.5 m along the path the robot also scans at eight headings, and those look-around scans are
merged with the taught map (984 look-around poses, 29 615 → 34 084 map points on our farm). The search then covers
every free cell of the map as well as the path (0.5 m grid × 24 headings, 101 148 hypotheses, 0.6 s per scan), and the
chosen pose gets its way home by rejoining the nearest graph node and following the normal route
(`SiteMap.route_from_xy`, refused beyond 6 m from the graph). Measured with LiDAR scans and front images rendered in
the current twin at 40 poses 1–3 m beside the path, in the front and back yards: the LiDAR settles 5 outright and
flags 33 as ambiguous (the right place leads in 14 of them); Nemotron 3.5 Super VL compares the front image with the
look-around reference image of matching heading at each candidate place, decided 28 and was right in 20; the final
pose is correct and a route to the dock is planned in **25 of 40 (62 %)**. The reference image must share the
viewpoint: with frames from the led drive as references the camera was right in 13 of 40. Evidence:
`docs/evidence/14_offpath_relocalisation_lidar.json` (LiDAR stage), `15_offpath_full_loop_summary.json` and
`15_offpath_full_loop.jsonl` (the full loop, per case).

**Then do the mission.** A put-down is only worth something if the survey that follows is complete and its records
carry the right house and metre, so the test continues from the believed pose (right or wrong) through the whole
mission: rejoin the taught path, the aisle of tunnel-09, the yard, the aisle of tunnel-10, the dock
(`scripts/offpath_mission_eval.py`, LiDAR scans synthesised on the self-built map). While driving, the believed place
is checked against every scan, a fresh whole-map search every 10 m keeps other candidate places alive, and a belief that stops
fitting, an obstacle where the map says the way is free, or another candidate place that fits clearly better over 2 m of driving
becomes a new anchor: the whole odometry history is re-placed, the plant records made so far are relabelled, and the
mission goals are re-derived from the corrected history. Before docking, Nemotron VL compares the view with the dock and
with the strongest other candidate place. On the 40 near put-downs the survey was completed with every plant stop covered in 40
of 40 and the records labelled right in 39 of 40; 13 had started with a wrong belief and 12 were corrected, 4 by the
scans within 2 m and 8 by the camera at the dock after a complete, then relabelled, survey. On 10 further put-downs
4–8 m from the path (LiDAR guess alone to start) the survey was complete in 10 of 10 and right in 9 of 10. Evidence:
`docs/evidence/16_offpath_mission_survey.json` (per case: every decision and the path driven).

**Without a stored route.** The led drive is a way to collect scans safely; the robot does not need its path afterwards.
From its own map alone (`scripts/free_nav.py`) it marks every 0.25 m cell that a LiDAR ray has crossed and that lies
farther than its half-width from any map point - space it has seen empty and can fit through; ponds, orchard and road
are never marked because it never saw them empty. It finds the dominant wall direction from the points, and the long
narrow free corridors between parallel rows with layered sides (stems, bed edge, next row) are the plant aisles; the
dock is where it was switched on. The mission is decided from that structure - which aisles, in what order - by the
agent's model (Nemotron 3 Super through Nebius Token Factory, with its reason kept) or by a nearest-first rule; every
path is then planned on the free grid when it is needed (A* with a clearance cost, which is why the robot keeps to the
middle of an aisle), from where the robot believes it is, and planned again whenever the belief changes or something
unexpected appears. Figure: `media/gallery/30_free_nav_structure.png`. Measured the same way as above, 40 near put-downs and 10 in the open, with an
obstacle that is not on the map dropped on the planned path in every case and HERMES reviewing the plan on the map
picture first: the survey was completed in 39 of 40 near cases (every row fully covered in 38, records right in 36;
12 of the 13 wrong starts corrected) and in 10 of 10 far cases (9 right); the unexpected obstacle was driven around in
41 of the 50; HERMES approved 31 plans and changed 10. Inside a row the robot asks for no pictures, odometry carries
its belief; at every doorway, when the LiDAR still keeps the twin place alive, the agent decides from the whole
situation (both places with their LiDAR fit, the metres driven, the doors passed, the verdicts so far, the plan, and the
live view against both reference views) and a move is accepted only after two doorways agree and the LiDAR fits the
other candidate place at least as well. Measured the other ways for the record: pictures every 10 m inside the rows gave 36 of 50 with
88 belief changes, single doorway verdicts without the two-doorway rule 39 of 50, the camera at the dock alone 43 of 50. The order came from the Rural Postman solver on the robot
(every chosen aisle a required edge, A* distances between aisle ends and the dock, about 0.1 s), so the model decides
which aisles and whether the plan is sound, never the geometry. Evidence: `docs/evidence/17_freenav_mission_survey.json`;
figures `media/gallery/31_freenav_survey_each_case.png`, `32_freenav_survey_overview.png`.

**Three ways to run a survey, from the same map.** (1) *Taught route*: the robot follows the corridors of the led
drive, node to node, with the site graph deciding what is allowed; the simplest and the one every number above was
measured on. (2) *Map only*: no route is kept; the robot finds the aisles and the drivable space in its own map, the
agent's model decides the order from that structure, the robot plans every path itself and plans again when something
unexpected appears. (3) *Map only with HERMES review*: as (2), and before the robot moves HERMES sees the plan drawn on
the map together with every leg's length and the alternatives, and approves or changes it; the same channel takes the
robot's reports during the mission (an aisle blocked, a place that does not look like the map) for the next decision.
A farm with ponds, an orchard and a road is run in (1) until its map has been seen widely enough for (2).

**Before the robot drives on its own in the field: a live ground check.** A 2-D LiDAR at 0.55 m returns nothing from
a drop-off and nothing from still water, so a map alone can send the robot into a pond or a ditch that the map never
held. A layer on the Jetson therefore gets the right to veto every forward command: the ZED 2i depth image flags a
ground plane that drops away or returns no depth within 2–3 m ahead, and a small real-time segmentation model
(SegFormer-B0 or PIDNet class, TensorRT) labels the ground ahead as drivable, water, ditch, vegetation or person. Two
frames per second are enough: at 0.3 m/s that is a check every 15 cm, about 5 % of the Orin Nano's GPU, trained on
the twin's render passes and real farm photographs. The vision-language models are too slow to be this brake; they
reason, the ground layer stops. On "not drivable ahead" the robot stops, marks the place on its map and plans again,
whatever the planner or HERMES asked for. The same pass labels the plants: canopy, stems, fruit and yellowing leaves become a plant layer of the map (where the rows really hold plants, where to stop for the close look), while the fine grading of fruit and lesions stays with RT-DETRv2 at the stops and Nemotron VL.

What the farmer sees: on the dashboard the robot reports "relocalised at tunnel-10, metre 14 (camera: the tree
through the door)" or "ambiguous: tunnel-09 or tunnel-10 — moving to the door to check". It never silently guesses.

Evidence: `docs/evidence/13_relocalize_eval_run3.json`, `12_vlm_place_check.jsonl`; beside the path
`14_offpath_relocalisation_lidar.json`, `15_offpath_full_loop_summary.json`.

## 7. What is and is not in this version

In this version: the led drive, the map and site graph built from it, naming and confirmation, route planning on the
graph, all commands above, and relocalisation from an unknown start (LiDAR candidates, camera tie-break, motion pruning, a route home from beside the path) — verified offline on our farm's recorded drive and on rendered put-down poses; wiring relocalisation into the live start-up of the robot is the next step. Next: building the map from cuVSLAM poses as well
as LiDAR (tighter than the odometry that taught it), and the aisle map generated from the LiDAR bed walls for houses the
recorder did not already know.

Related: [INSTALL.md §2b–2d](INSTALL.md) (commands, which files identify a farm, teach mode), `r1_nav/site_map.py`
(the graph schema and router), `agent/hermes_skills/farm-site-map` (naming and confirming by conversation).
