# รายงานการสำรวจโรงเรือน: farm_full3

2026-10-29 21:46:05  (รายงานจากแม่แบบ: ไม่มี NEBIUS_API_KEY จึงไม่ได้ให้ Nemotron เขียน)

## 1. ผลผลิต (คัดเกรดผล)
- ผ่าน (สุกแดง): 3275, คัดทิ้ง (มีแผล): 47, ยังไม่พร้อม: 4532
  - ทางเดิน tunnel-09/aisle-02: {'pass': 1573, 'reject': 28, 'not_ready': 2231}
  - ทางเดิน tunnel-10/aisle-02: {'pass': 1673, 'reject': 19, 'not_ready': 2262}
  - ทางเดิน null: {'pass': 29, 'reject': 0, 'not_ready': 39}

## 2. จำนวนต้น
- 40 ต้น (RT-DETR)
  - ทางเดิน tunnel-09/aisle-02: 8
  - ทางเดิน tunnel-10/aisle-02: 32

## 3. โรค/ความผิดปกติ
- RT-DETR: 1473 {'chlorosis (yellowing leaves)': 1358, 'fruit lesion': 112, 'leaf spots': 3}, ground truth 4/7
- VLM: ไม่ได้รัน
- ต้นที่ควรดู:
  - ไม่มี

## 4. เหตุการณ์ระหว่างเดิน
- {'status': 3637, 'command': 3, 'leg_start': 18, 'waypoint': 9, 'capture': 1035, 'leg_done': 16, 'enter': 4, 'slip': 2, 'aborted': 2}
- คำสั่งที่ได้รับ: survey_farm (ok), revisit (ok), revisit (ok)
- เหตุการณ์ผิดปกติ: aborted @ None -0.03 m no room to turn: obstacle 0.02 m from centre (needs 0.51 m); aborted @ None 1.384 m no room to turn: obstacle 0.02 m from centre (needs 0.51 m)

## 5. การระบุตำแหน่ง
- ระยะคลาดเคลื่อนตำแหน่งหุ่น (odometry เทียบค่าจริง): {'n': 7500, 'rmse_m': 0.234, 'median_m': 0.144, 'p90_m': 0.412, 'max_m': 0.5}; scan_to_map {'attempts': 7498, 'ok': 7497, 'applied_xy': 2674}; path 122.97 m
- cuVSLAM: ไม่มี
- lingbot-map: ไม่มี

## ขั้นตอนที่รัน
- detect: ok
- lidar_map: ok
