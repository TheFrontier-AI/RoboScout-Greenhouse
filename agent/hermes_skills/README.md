# HERMES skills (Hermes Agent)

Skills that HERMES (Hermes Agent + Nemotron on Nebius Token Factory) loads to run R1. Each folder is a
Hermes skill (`SKILL.md` + optional `references/`). Hermes reads skills from `~/.hermes/skills/`, so link
each folder there:
```bash
for s in r1-robot-control r1-motion-policy plant-disease-diagnosis greenhouse-watering; do
  ln -sfn "$PWD/agent/hermes_skills/$s" ~/.hermes/skills/$s
done
```

| Skill | What it does | Talks to |
|---|---|---|
| [`r1-robot-control`](r1-robot-control/SKILL.md) | Picks goals (discover, survey, revisit, dock) through the robot API; goals only, never wheels | `POST /command`, `GET /status /findings /map /health` |
| [`r1-motion-policy`](r1-motion-policy/SKILL.md) | Picks the task mode and speed (inspect, stop-and-go, revisit, transit, watering) from camera blur/overlap math and the safety limits; proposes `speed_mps`/`mode` API fields | same API (no new fields used today) |
| [`plant-disease-diagnosis`](plant-disease-diagnosis/SKILL.md) | Aggregates `/findings` per spot, decides revisits, confirms with a second model (Nemotron 3.5 Super VL or Cosmos Reason 2 32B), writes the Thai/English farmer report; tomato reference table | `/findings`, `cloud/eyes/inspect.py` |
| [`greenhouse-watering`](greenhouse-watering/SKILL.md) | Plans spot watering (dose, dwell, pump duty, speed) from R1's designed water hardware; **refuses to water** until a `water` command exists | none (planning only) |

## Third-party skills reviewed (NOT installed)
Found with `hermes skills search` (Hermes Skills Hub: clawhub, skills.sh, GitHub) and the web on 2026-10-28.
Install nothing here without a code review, and never anything that lets the LLM drive wheels or send farm
images to an unknown service.

| Skill | Source / licence | What it does | Verdict |
|---|---|---|---|
| [NVIDIA/skills](https://github.com/NVIDIA/skills), e.g. `tao-finetune-cosmos-reason`, `isaac-mission-control-showcase` | GitHub, Apache-2.0, ~3.8k stars, "trusted" in Hermes hub | NVIDIA's skills for Cosmos/TAO/Isaac | **Worth reviewing** for the dev team (fine-tuning Cosmos Reason on our labelled R1 frames). Not a HERMES runtime skill. |
| Isaac Sim `isaac-sim-ros2-bridge`, `occupancy-map` ([isaac-sim/IsaacSim](https://github.com/isaac-sim/IsaacSim)) | skills.sh (~39/32 installs), official repo, mixed licence | Isaac ROS 2 bridge, occupancy maps | Useful for the **sim → Jetson ROS 2** port (dev-time). Not for HERMES. |
| [dbwls99706/ros2-engineering-skills](https://github.com/dbwls99706/ros2-engineering-skills) | Apache-2.0, ~229 stars, updated 2026-10 (also on clawhub) | ROS 2 engineering guide incl. Nav2, MoveIt 2 | **Best ROS 2 coding skill found**; dev-time only. |
| [arpitg1304/robotics-agent-skills](https://github.com/arpitg1304/robotics-agent-skills) (`ros2`, `ros2-development` on skills.sh) | Apache-2.0, ~386 stars | ROS1/ROS2 production-code guidance | Good alternative to the above; dev-time only. |
| [adityakamath/ros2-skill](https://github.com/adityakamath/ros2-skill) (clawhub `ros2-skill`), [lpigeon/ros-skill](https://github.com/lpigeon/ros-skill) | Apache-2.0, 20–27 stars | Agent publishes/calls ROS topics directly (rclpy / rosbridge) | **Do not install in HERMES.** It would let the LLM publish `cmd_vel` and bypass the goal-only design and SAFETY. At most read-only introspection on the Jetson later. |
| clawhub `ros2-introspection` / `ros2-execution` | community | Sandboxed, allowlisted ROS 2 CLI | Introspection might help diagnostics after the ROS 2 port; review first. |
| Nav2 skills (smithery `nav2-navigation-skill`, MIUAV `wheeled-nav2`, robium `nav2`) | low installs; MIUAV has no licence | Nav2 configuration | Skip (R1 uses its own BOOK/MISSION legs, not Nav2). |
| clawhub `smyx-plant-leaf-disease-identification-analysis`, `smyx-plant-disease-recognition-analysis`, `smyx-crop-pest-identification-analysis`, `smyx-plant-nutrient-diagnosis-analysis`, `smyx-crop-specific-disease-analysis` | community, MIT-0 (skill text) | Send images/video to a vendor cloud analysis service (lifeemergence.com, "cloud report history") | **Do not install.** Farm images go to an unknown third party, the model is opaque, and there are no data terms. Our own Cosmos/Nemotron path does this already. |
| clawhub `agriculture` (@rithythul / koompi) | community, no licence in header | Generic farm-operations persona (crop plans, inventory, livestock, prices) | Not needed; generic prompt, no greenhouse/robot content. |
| clawhub `garden-irrigation` (@bowlderstudio) | MIT | Tuya soil sensors + weather → zone watering | Pattern is interesting if the farm adds Tuya soil sensors; needs Tuya cloud keys. Not now. |
| clawhub `fertigation`, `soil-water-management` | community | Reference text | Optional reading; not installed. |
| lobehub `xiao-zhi-greenie` | community | Horticulture persona | Skip. |
No dedicated tomato-greenhouse, plant-disease-from-robot or robot-speed skill exists in the hub. That is why
the three R1 skills above were written.
