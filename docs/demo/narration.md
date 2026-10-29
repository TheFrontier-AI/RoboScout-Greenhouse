# RoboFarm demo — narration script (English)

Voice: en-US-AndrewMultilingualNeural

## s0_open  (~30 s, starts ≈ 0:00)
**On screen:** shots/cctv7_front_overview_1.mp4
**Overlay:** title: RoboFarm · R181-S · Nebius x NVIDIA Global AI Hackathon 2026

> This is RoboFarm. The robot you're watching is R1, built for tomato greenhouses, and right now it is leaving its dock in a digital twin of our own farm, rendered in NVIDIA Isaac Sim. Nothing in this film is scripted. Every turn, every stop, is the robot's own decision, and the route you'll see is the route it really drove.

## s1_edge  (~30 s, starts ≈ 0:30)
**On screen:** chase.mp4@30-62 · inset: lidar.mp4
**Overlay:** labels: BOOK driver · SAFETY · LiDAR localisation (no tags) — on the robot

> On board, three things run with no cloud at all. A rule-based driver we call BOOK follows the aisle map. A SAFETY layer stops the wheels the moment the LiDAR sees something in the way. And localisation fuses the wheels, the gyro and the LiDAR scan against a map of the farm, with no fiducial tags anywhere. Over this hundred-and-twenty-metre route, the robot's own idea of where it was stayed within twenty-three centimetres of the truth.

## s2_stops  (~30 s, starts ≈ 1:00)
**On screen:** chase.mp4@62-95 · inset: side_right_mid.mp4 (flash at each stop)
**Overlay:** counter: plant stops · six cameras per stop · frames → Nebius hub

> Inside the greenhouse it slows to a walk and stops at every single plant, forty-five centimetres apart. Six cameras on the mast, three on each side, low, mid and high, fire together. The pictures stream to a hub on Nebius, and the robot moves on without waiting for anyone.

## s3_disease_freeze  (~45 s, starts ≈ 1:30)
**On screen:** freeze chase.mp4 at the chlorosis stop (tunnel-09, metre 11.85) · inset: the six side images with RT-DETR boxes, then Nemotron VL text typing in
**Overlay:** card: Plant health (separate from Harvest) · RT-DETRv2 boxes · Nemotron 3.5 Super VL verdict + next step

> Here, at metre twelve of tunnel nine, something is wrong with the plant on the right. On the server two very different eyes look at it. RT-DETR, a small detector we trained only on synthetic renders, draws the boxes: the yellowing leaves, in purple, and, separately, every fruit graded ripe, not ready, or reject. Then NVIDIA Nemotron 3.5 Super VL reads all six views of the plant in a single request, and writes what an agronomist would say: yellowing in the upper canopy, lower leaves still green, stems and fruit fine. Its advice: check soil nitrogen and irrigation. Health and harvest are always reported apart, so nobody confuses a sick plant with a ripe one.

## s4_hermes  (~25 s, starts ≈ 2:15)
**On screen:** chase.mp4@95-120 · inset: HERMES panel: revisit tunnel-09/aisle-02 metre 12.1 — reasoning text from decisions.jsonl
**Overlay:** HERMES planner · NVIDIA Nemotron 3 · Nebius Token Factory

> Those findings go to the planner we call HERMES, running NVIDIA Nemotron 3 on Nebius Token Factory. HERMES never touches the wheels. It chooses goals: finish the survey, then go back and look again at the strongest suspects. On this run it ordered two revisits. Both came back healthy, so two false alarms were cleared before they ever reached the farmer.

## s5_yard_safety  (~30 s, starts ≈ 2:40)
**On screen:** chase.mp4@245-300 (back door → yard)
**Overlay:** SAFETY stop → front frame → Cosmos Reason 2 → HERMES: wait / notify farmer

> Out the back door and across the yard to the next greenhouse. If SAFETY ever stops the robot, the front camera frame goes to NVIDIA Cosmos Reason, which says in plain farm terms what is there: a person, a cart, a hose across the aisle, a fallen plant. HERMES then waits for people and animals, or sends the farmer a message for anything that will not move by itself. The robot stays safe; the AI explains why.

## s6_r181s  (~35 s, starts ≈ 3:10)
**On screen:** chase.mp4@330-365 (tunnel-10) · inset: zed_left_540p.mp4 + zed_depth.mp4; cuVSLAM path overlay on the farm plan
**Overlay:** R181-S · Stereolabs ZED 2i · NVIDIA cuVSLAM: 6.4 cm over 111 m · 526 fps · Jetson-ready

> This is the new sensor that makes R1 into R181-S: a ZED 2i stereo camera where the single front camera used to be. With stereo depth, NVIDIA cuVSLAM tracks the robot's position at over five hundred frames a second on a single GPU, and across the whole hundred-and-eleven-metre route it stays within six centimetres of the truth, at true metric scale. The same code runs on a Jetson inside the robot.

## s7_maps  (~40 s, starts ≈ 3:45)
**On screen:** cuvslam_3d.mp4@0-30 then lingbot_3d.mp4@28-48
**Overlay:** 3-D map: ZED depth placed by cuVSLAM poses · lingbot-map from the front camera · Nebius GPU

> Back on the server, those depth frames become a three-dimensional map of both greenhouses, placed with cuVSLAM's own poses and no ground truth at all. We also benchmarked seven camera-only mapping methods on the same route. Only lingbot-map kept the two identical greenhouses apart, at about a metre of error; the others folded them together. Stereo, or LiDAR, is what makes localisation reliable; lingbot-map is the right tool for the map.

## s8_results  (~35 s, starts ≈ 4:25)
**On screen:** chase.mp4@600-648 (return to dock)
**Overlay:** report card: tunnel-09 ripe 1,780 / not ready 2,242 / reject 53 · anomaly plants found 6 of 9 · Nemotron false alarms 12 · wilting & defoliation not yet detected

> Home to the dock, and the report is already written. Tunnel nine: one thousand seven hundred and eighty ripe fruit ready to pick, two thousand two hundred not ready, fifty-three to reject. Of nine diseased plants hidden in the scene, the two eyes together found six. Nemotron raised only twelve false alarms over three hundred and twenty plants. And we are honest about the misses: wilting and leaf drop are whole-plant shapes the close side cameras cannot see yet. That is the next thing to fix.

## s9_pipeline  (~25 s, starts ≈ 5:00)
**On screen:** pipeline diagram card over the final CCTV frame
**Overlay:** robot → /run_complete → pipeline: detect · vlm_stops · lidar_map · cuvslam · lingbot · cosmos · report

> Everything you have seen now runs as one system. When a survey ends, the robot posts one message, and the Nebius server does the rest by itself: detection, the six-view inspection, the maps, the report in Thai and English. Next time, nobody runs a single step by hand. RoboFarm, built for the Nebius and NVIDIA Global AI Hackathon.


Total target ≈ 5:25
