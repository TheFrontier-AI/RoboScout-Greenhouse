# Greenhouse survey report: farm_full3

2026-10-29 21:46:05  (template report: no NEBIUS_API_KEY, so Nemotron did not write it)

## 1. Harvest (fruit grading)
- pass (ripe red): 3275, reject (lesion): 47, not ready: 4532
  - aisle tunnel-09/aisle-02: {'pass': 1573, 'reject': 28, 'not_ready': 2231}
  - aisle tunnel-10/aisle-02: {'pass': 1673, 'reject': 19, 'not_ready': 2262}
  - aisle null: {'pass': 29, 'reject': 0, 'not_ready': 39}

## 2. Plant count
- 40 plants (RT-DETR)
  - aisle tunnel-09/aisle-02: 8
  - aisle tunnel-10/aisle-02: 32

## 3. Disease / anomalies
- RT-DETR: 1473 {'chlorosis (yellowing leaves)': 1358, 'fruit lesion': 112, 'leaf spots': 3}, ground truth 4/7
- VLM: not run
- plants to look at:
  - none

## 4. Navigation events
- {'status': 3637, 'command': 3, 'leg_start': 18, 'waypoint': 9, 'capture': 1035, 'leg_done': 16, 'enter': 4, 'slip': 2, 'aborted': 2}
- commands received: survey_farm (ok), revisit (ok), revisit (ok)
- incidents: aborted @ None -0.03 m no room to turn: obstacle 0.02 m from centre (needs 0.51 m); aborted @ None 1.384 m no room to turn: obstacle 0.02 m from centre (needs 0.51 m)

## 5. Localisation
- robot position error (odometry vs truth): {'n': 7500, 'rmse_m': 0.234, 'median_m': 0.144, 'p90_m': 0.412, 'max_m': 0.5}; scan_to_map {'attempts': 7498, 'ok': 7497, 'applied_xy': 2674}; path 122.97 m
- cuVSLAM: none
- lingbot-map: none

## Steps run
- detect: ok
- lidar_map: ok
