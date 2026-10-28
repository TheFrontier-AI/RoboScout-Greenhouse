# แผนที่ฟาร์ม (site map) — เดินระหว่างโรงเรือน

HERMES สั่งระดับเป้าหมาย เช่น "สำรวจโรงเรือน B" ขณะหุ่นอยู่โรง A หุ่นไม่คิดเส้นทางเอง: `r1_nav/commands.py` หาเส้นทางบน
**แผนที่ฟาร์มที่คนยืนยันแล้ว** แล้วแตกเป็นขา (`exit_house` → `site_transit` ทีละเส้นทาง → `enter_house` → ขาในโรงเรือน)

## รูปแบบไฟล์ (`r1_nav/site_map.py`, schema_version 1)
- `nodes`: `{id, kind: dock|door|junction|waypoint, xy: [x, y] เมตร, house (เฉพาะ door), heading_deg (ทิศขับเข้าโรงเรือน), label}`
- `edges`: `{a, b, length_m (null = คำนวณจาก xy), width_m (null = ไม่ทราบ), surface, notes[], blocked}`
- `metadata`: `{source: v18|drone|hand_sketch|robot, created, confirmed_by, confirmed_at, scale_known, anchors_used[], revision, history[]}`
- `layers` (ภาพพื้นหลัง เช่น top-down จากโดรน), `landmarks` (บ่อ อาคาร โน้ตพื้นที่ — ไม่ใช้หาเส้นทาง)
- พิกัด: กรอบเดียวกับ `house_local_to_world` ใน `data/tunnel09_aisle_map_v18.json` (Blender Z ขึ้น, glTF `(x, y, z)` → `(x, −z, y)`)

## กฎ
- หาเส้นทางด้วย Dijkstra เฉพาะเส้นที่ไม่ `blocked` และ `width_m` เป็น null หรือ ≥ กว้างหุ่น + 2×0.06 ม. (กฎเดียวกับทางเดินในโรงเรือน)
- ใช้ได้เฉพาะแผนที่ที่ `confirmed_by` ไม่ว่างและ `scale_known = true`
- ทุกการแก้ (`patch`: ปิด/เปิดเส้นทาง, เพิ่มโน้ต, ตั้งความกว้าง, เพิ่ม/ลบจุดหรือเส้น) และการรวมแผนที่ (`merge`) → `revision + 1` และล้างการยืนยัน
- "การเรียนรู้" ของฟาร์ม = เก็บแผนที่หลายเวอร์ชันพร้อมโน้ตของเกษตรกร ไม่ใช่การเทรนโมเดล

## 4 แหล่งที่มา
| source | ทำอย่างไร | หมายเหตุ |
|---|---|---|
| `v18` | `python3 scripts/extract_site_map_v18.py` → `data/site_map_v18_draft.json` | 22 โรงเรือน (tunnel 12 + block 10), ประตูจาก view preset `entrance`/ผังทางเดิน, ถนน 5 เส้นจาก `site-layout-v5.json`; ประตู east-greenhouse 2 หลังเป็นค่าประมาณ; สเกล blueprint ยังไม่ได้วัดจริง |
| `drone` | งาน SLAM บนคลาวด์ → `python -m cloud.maps.drone --slam DIR --pair COL,ROW=X,Y ...` | จัดภาพ top-down เข้ากรอบฟาร์มด้วยจุดอ้างอิง ≥ 2 จุด (Umeyama) แล้วแนบเป็นชั้นพื้นหลัง |
| `hand_sketch` | รูปถ่ายแผนที่วาดมือ หรือภาพที่วาดบนเว็บ V18 → `python -m cloud.maps.sketch img.jpg --anchors-from data/site_map_v18_draft.json` | Nemotron-3.5-Super-VL (Token Factory) อ่านเป็น JSON; ป้าย ≥ 2 อันตรงกับจุดอ้างอิง → เมตร ไม่งั้นเป็นพิกเซล |
| `robot` | เส้นทางที่หุ่นเดินจริงและบันทึกไว้ | น่าเชื่อที่สุดเมื่อยืนยันแล้ว |

## ขั้นตอนใช้งาน
```bash
python3 -m r1_nav.site_map show data/site_map_v18_draft.json                      # สรุปให้คนตรวจ
python3 -m r1_nav.site_map route data/site_map_v18_draft.json tunnel-09 south-greenhouse-block-01
python3 -m r1_nav.site_map patch data/site_map_v18_draft.json '{"op":"add_note","a":"road/13","b":"road/14","note":"เปียกหลังรดน้ำ"}'
python3 -m r1_nav.site_map confirm data/site_map_v18_draft.json --by "ชื่อคนตรวจ"     # คนเท่านั้น
R1_SITE_MAP=data/site_map_v18_draft.json python3 agent/robot_api.py ...             # robot API ใช้แผนที่นี้
```

## ข้อจำกัด
- ร่างจาก V18 ใช้ถนนตาม blueprint + เส้นตรงสั้น ๆ ไปประตู ความกว้างทุกเส้นไม่ทราบ ต้องให้คนตรวจและแก้ก่อนยืนยัน
- การเดินระหว่างโรงเรือน (`exit_house`/`site_transit`/`enter_house`) ยังไม่ได้จำลองใน Isaac (`sim/executor.py` ข้ามขาเหล่านี้เหมือน `transit`)
