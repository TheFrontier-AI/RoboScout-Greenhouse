# Roadmap
1. (this week) Aisle map + mission planner + rule-based aisle-centring controller, with unit tests — **skeleton done**
2. Connect to the R180 env: drive to the end of the centre aisle and stop in front of the box, using sensors only (not the ground-truth pose); measure the edge clearance
3. Ask Astra to extend the env: reversing, random start point, several obstacle types, start in any aisle, images every 0.5 m
4. One full aisle (in and reverse out) 10/10 → the whole house (passable aisles only)
5. Connect central_brain (harness) through the INTERFACES.md contract and send images to the anomaly-detection team
6. Move the controller to the Jetson (ROS 2) and test on the real robot once assembled
