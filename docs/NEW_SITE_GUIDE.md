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
the road to the north, the yard behind the houses. 32 967 LiDAR points, 123 m of driving, no distances typed.*

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
  as on the hand-drawn plan (`docs/evidence/10_teach_map_run3.json`).
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

**Off the taught path - not yet.** The search can cover every free cell of the map (`Relocalizer(..., grid_m=0.5)`,
47 028 hypotheses, 0.7 s), and with the path hypotheses kept the accuracy on recorded scans is unchanged. A test with
scans synthesised from the robot's own map at 30 free cells 1-3 m off the path did not succeed (3 found, 19 ambiguous,
8 wrong): the map holds only what the LiDAR saw from the path, so a viewpoint beside the path sees surfaces that were
never recorded. Off-path starts therefore remain future work, with a clear plan: record a wider map on the led drive
(look around at the yard edges), use the cuVSLAM 3-D map which sees more than a 2-D scan, add a "go to the nearest
graph node" leg before the normal route, and test with real scans. Until then: put the robot down on a path it knows.

What the farmer sees: on the dashboard the robot reports "relocalised at tunnel-10, metre 14 (camera: the tree
through the door)" or "ambiguous: tunnel-09 or tunnel-10 — moving to the door to check". It never silently guesses.

Evidence: `docs/evidence/13_relocalize_eval_run3.json`, `12_vlm_place_check.jsonl`.

## 7. What is and is not in this version

In this version: the led drive, the map and site graph built from it, naming and confirmation, route planning on the
graph, all commands above, and relocalisation from an unknown start (LiDAR candidates, camera tie-break, motion pruning) — verified offline on our farm's recorded drive; wiring relocalisation into the live start-up of the robot is the next step. Next: building the map from cuVSLAM poses as well
as LiDAR (tighter than the odometry that taught it), and the aisle map generated from the LiDAR bed walls for houses the
recorder did not already know.

Related: [INSTALL.md §2b–2d](INSTALL.md) (commands, which files identify a farm, teach mode), `r1_nav/site_map.py`
(the graph schema and router), `agent/hermes_skills/farm-site-map` (naming and confirming by conversation).
