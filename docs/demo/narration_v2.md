# RoboFarm demo — narration v2, 3-minute cut (final text from the user)

Judges watch three minutes, so the film is three minutes. ~390 words; the chase camera is the spine.

## 1. Open — CCTV overview, robot leaves the dock
> Meet RoboFarm — our custom-built autonomous robot designed to navigate tomato greenhouses, inspect every plant, and turn what it observes into actionable insights for farmers. What you're seeing is RoboFarm navigating autonomously through a digital twin of our greenhouse, built with NVIDIA Isaac Sim and powered by Nebius GPUs.

## 2. Where it is — chase cam into the aisle; LiDAR inset, then ZED insets
> For a robot to navigate autonomously, it first needs to understand where it is. RoboFarm combines LiDAR and a ZED stereo camera running NVIDIA cuVSLAM on the Jetson Orin inside the robot to track its position through greenhouse pathways — without requiring additional markers or modifications to the greenhouse.

## 3. Survey — plant stops; side-camera inset
> Once inside the greenhouse, RoboFarm begins its plant-by-plant inspection. Multiple cameras capture each tomato plant from different heights and angles, and the images are processed on Nebius, where HERMES Agent and Nebius Token Factory generate actionable insights for farmers.

## 4. The eyes — freeze at the sick plant; boxes draw on; Nemotron text types in
> This is where RoboFarm turns images into intelligence. Using NVIDIA Nemotron, the system brings together observations from multiple camera angles to assess fruit ripeness, identify potential diseases, and generate practical recommendations. When it detects yellowing leaves, it may suggest checking nutrient levels or irrigation. Plant health and harvest readiness are always kept separate.

## 5. HERMES + Cosmos — HERMES card over the chase; then the yard with the Cosmos flow
> HERMES, an AI agent powered by NVIDIA Nemotron through Nebius Token Factory, coordinates the robot's next actions — a closer inspection, or notifying the farmer. Between greenhouses, NVIDIA Cosmos Reason interprets people, equipment and unexpected obstacles, helping HERMES decide whether to proceed, wait, or request human assistance.

## 6. 3-D map — cuVSLAM 3-D map growing; lingbot-map orbit
> As RoboFarm moves through the greenhouse, it continuously collects spatial data to build and refine its 3D map. Using GPU infrastructure on Nebius, the system updates its representation of the environment, helping the robot navigate more accurately and adapt to changes in the farm.

## 7. Report — robot returns; report card fills in
> And this is what the farmer receives — a clear, actionable report from every survey: harvest readiness, the number of ripe and unripe tomatoes, and the plants that need attention, each with images, its location, and recommended actions. Available in Thai and English, it turns hours of manual inspection into decisions farmers can act on.

## 8. Close — assembly film (~15 s) under the first sentences, then the closing card
> We designed and assembled RoboFarm around locally available components, working directly with farmers — a step toward technological self-reliance. Built and tested in real agricultural environments, our vision is to make advanced robotics affordable, accessible, and locally sustainable. RoboFarm. Built for farmers. Designed for the real world.
