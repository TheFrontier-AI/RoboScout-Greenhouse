# สิ่งแวดล้อมจำลอง (Isaac Sim 5.1, เครื่อง DGX Spark)

- ฉาก: `robot_design/r1_v3/diy_r50/physical_ai_r180/scene_fix_2026-10-28/C2_R180_tunnel09_photo_match.usda`
  (+ `sensor_stop_fixture_pt.usda` มีกล่องทดสอบ) — เทียบกับภาพถ่ายฟาร์มจริงแล้ว ดู README_th.md ในโฟลเดอร์นั้น
- env API (จาก R180 handoff): สร้าง `SimulationApp` ก่อน แล้ว `from r180_env import make_env`;
  `obs, info = env.reset(seed)`; `obs, r, term, trunc, info = env.step([left_rad_s, right_rad_s])` ช่วง ±2 rad/s
- obs: `front_rgb` 320×180 uint8, `lidar_m` 500 ค่า + `lidar_angles_rad`, `imu_accel_gyro` 6 ค่า, `wheel_position_velocity` 8 ค่า, `timestamp_s`
  **ไม่มีตำแหน่งโลก** — ตัวควบคุมต้องใช้เซนเซอร์เท่านั้น; ตำแหน่งจริงใช้ได้เฉพาะตอนประเมินผล
- เรนเดอร์: ค่า ACES ฝังในฉากแล้ว; โหมดเรียลไทม์ต้อง settle ~200 เฟรมก่อนใช้ภาพ; PathTracing สำหรับภาพคุณภาพสูง
- ข้อจำกัดที่ทราบ: ใบไม้ไม่มี collision และ LiDAR มองไม่เห็นใบ; มวล/แรงเสียดทานเป็นค่าประมาณ; env เรนเดอร์ทุก physics substep (ช้า, ขอให้ Astra แก้)
- ข้อจำกัดของ env ปัจจุบันที่ repo นี้ต้องการเพิ่ม: จุดเริ่มสุ่มในทางเดิน, เริ่มได้ทั้ง 3 ทางเดิน, สิ่งกีดขวางหลายแบบ, สั่งถอยหลัง, กล้องส่งภาพทุก 0.5 ม.
