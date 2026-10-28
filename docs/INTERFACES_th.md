# สัญญาส่งต่อระหว่างส่วน

## 1) จาก central_brain → r1_greenhouse_nav (คำสั่ง)
JSON: `{"cmd": "go_to", "house": "tunnel-09", "aisle": "tunnel-09/aisle-02", "metre": 12.0}` |
`{"cmd": "survey_house", "house": "tunnel-09"}` | `{"cmd": "return_to_dock"}` | `{"cmd": "stop"}`
ตอบกลับ: `{"status": "accepted|rejected|done|aborted", "reason": "...", "pose": {"aisle": ..., "metre": ...}}`

## 2) จาก r1_greenhouse_nav → anomaly detection (ภาพ)
ทุก 0.5 ม. หรือเมื่อหยุด: `{"ts": float, "house": str, "aisle": str, "metre": float, "heading_deg": float,
"camera": "front_rgb|plant_left_mid|...", "image": <jpeg bytes หรือ path>, "frame_id": int}`
repo นี้ไม่ตีความภาพ; ผลตรวจจับส่งไป central_brain ไม่ย้อนกลับมาที่ตัวเดิน

## 3) จาก r1_greenhouse_nav → หุ่น (sim หรือจริง)
`wheel_cmd = [left_rad_s, right_rad_s]` ที่ 20 Hz; e-stop และหยุดเมื่อมีสิ่งกีดขวางทำในชั้นนี้ ไม่พึ่งเครือข่าย
