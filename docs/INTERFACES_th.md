# สัญญาส่งต่อระหว่างส่วน

ทุกส่วนคุยกันผ่าน **robot API** (`agent/robot_api.py`, พอร์ต 8090) ซึ่งเก็บทุกอย่างลง data dir สำหรับแดชบอร์ดด้วย
```
HERMES (Nemotron @ Token Factory) ──POST /command──▶ robot API ◀──GET /next_command── หุ่น (Isaac sim / Jetson)
          ▲ GET /status /findings /map                  │  ▲ POST /status, POST /frame
          └─────────────────────────────────────────────┤  │
                                     Cosmos Reason ◀────┘ (ภาพกล้องข้าง → findings)
```

## 1) HERMES → หุ่น (เป้าหมายเท่านั้น ห้ามสั่งล้อ)
`POST /command` JSON, ใส่ `reasoning` สั้น ๆ ทุกครั้ง (แสดงบนแดชบอร์ด/วิดีโอ):
| cmd | ฟิลด์ | ความหมาย |
|---|---|---|
| `discover_house` | – | โรงเรือนที่ไม่มีผัง: SLAM สร้างแผนที่ + หาช่องทางเดินเอง |
| `survey_house` | `house` | เดินครบทุกทางเดินที่ผ่านได้ (เข้า–ถอยออก) |
| `survey_aisle` | `aisle` | ทางเดินเดียว หัวถึงท้าย |
| `go_to` | `aisle`, `metre` | ไปหยุดที่จุด (metre 0 = หัวทางเดินฝั่งประตู) |
| `revisit` | `aisle`, `metre` | ไปแล้วหยุดนิ่ง 3 วินาที ให้กล้องข้างทั้ง 6 ถ่าย (purpose=revisit) |
| `return_to_dock` / `stop` | – | กลับแท่น / หยุดทันที (ล้างคิว) |
| `set_map` | `path` | ใช้ผังที่คนยืนยันแล้วหลัง discover |
| `go_to_house` | `house`, `map_source`? | ออกจากโรงเรือนปัจจุบัน → เดินตามแผนที่ฟาร์ม (site map) ที่ยืนยันแล้ว → เข้าประตูโรงเรือนเป้าหมาย |
| `survey_house` (คนละโรงเรือน) | `house`, `map_source`? | `go_to_house` ก่อน แล้วสำรวจทุกทางเดิน (ถ้ายังไม่มีผังทางเดินของโรงนั้น → `discover`) แล้วกลับแท่น |

ตอบกลับ: `{"status": "accepted|rejected", "reason": "...", "legs": [...]}`; หุ่นตรวจทุกคำสั่งกับผัง (`r1_nav/commands.py`)

### แผนที่ฟาร์ม (site map) และคำสั่งข้ามโรงเรือน
- กราฟ `dock / door / junction / waypoint` + เส้นทาง (`length_m`, `width_m` หรือ null, `surface`, `notes[]`, `blocked`) ในพิกัดฟาร์ม
  (เมตร, กรอบเดียวกับ `house_local_to_world` ของผังทางเดิน) รายละเอียด: [SITE_MAP_th.md](SITE_MAP_th.md)
- หุ่นรับเฉพาะแผนที่ที่ **คนยืนยันแล้ว** (`confirmed_by`) และมีสเกลเมตร; robot API โหลดจาก env `R1_SITE_MAP` (หลายไฟล์คั่นด้วย `,`)
- `map_source` = `v18 | drone | hand_sketch | robot` เลือกแหล่งแผนที่ (ไม่ใส่ = ฉบับยืนยันล่าสุด)
- ขาใหม่ใน `legs`: `exit_house` (ถอยออกจากทางเดิน ไปที่ประตู), `site_transit` (หนึ่งขาต่อหนึ่งเส้นทาง มี `from_node`, `to_node`, `length_m`
  และโน้ตของเส้นทางใน `note`), `enter_house` (ตั้งหัวที่ประตูแล้วเข้า)
- ปฏิเสธพร้อมเหตุผลเมื่อ: ไม่มีแผนที่, ยังไม่ยืนยัน, ยังเป็นหน่วยพิกเซล, ไม่รู้จักโรงเรือน, ไม่มีเส้นทาง (ปิดอยู่/แคบกว่า กว้างหุ่น + 2×0.06 ม.)
- `reason` ตอนรับคำสั่งบอกเส้นทาง ระยะทาง และโน้ตทุกข้อบนเส้นทาง (เช่น "เปียกหลังรดน้ำ") ให้ HERMES แจ้งเกษตรกร

## 2) หุ่น → คลาวด์ (ภาพ + ตำแหน่ง + สถานะ)
**กล้อง 7 ตัว** (`r1_nav/cameras.py`): ข้าง 6 ตัว `mast_{left,right}_{low,mid,high}` ดูต้นไม้ ถ่ายทุก 0.5 ม.;
หน้า 1 ตัว `front` ไว้นำทาง ถ่ายทุก 2 ม. ให้คลาวด์ตรวจว่าหุ่นเดิน/หมุนถูก

`POST /frame` multipart: `meta` (JSON) + `image` (JPEG)
```json
{"ts": 0.0, "house": "tunnel-09", "aisle": "tunnel-09/aisle-02", "metre": 15.0, "heading_deg": 270.0,
 "camera": "mast_left_mid", "frame_id": 123, "pose_confidence": 0.92, "purpose": "inspect|navigate|revisit"}
```
`heading_deg` = มุมหัวหุ่นในกรอบโรงเรือน วัดทวนเข็มจากแกน +X (เดินจากประตูเข้าไปทาง −Y = −90)
`pose_confidence` มาจาก SLAM; ต่ำกว่า 0.5 → ไม่ส่งตรวจโรคและห้ามปักหมุดที่เมตรนั้น

`POST /status`: `{"state": "idle|running|done|stuck|aborted", "cmd": ..., "aisle": ..., "metre": ..., "heading_deg": ...,
"pose_confidence": ..., "battery": <เปอร์เซ็นต์ 0–100>, "legs_remaining": n}`

## 3) Cosmos Reason → findings
ต่อภาพกล้องข้าง: `{"status": "healthy|suspect|disease|not_plant", "issues": [{"label", "confidence", "bbox_norm", "evidence"}], "summary"}`
`bbox_norm` = `[x0, y0, x1, y1]` สัดส่วน 0–1 ของภาพ (มุมซ้ายบน–ขวาล่าง) หรือ `null`
อ่านได้ที่ `GET /findings?since=N`

## 4) ชั้นล่างบนตัวหุ่น (ไม่พึ่งเครือข่าย)
`wheel_cmd = [left_rad_s, right_rad_s]` ที่ 20 Hz; SAFETY (e-stop, หยุดเมื่อมีสิ่งกีดขวาง < 0.6 ม., STUCK) ทำบนหุ่นเสมอ
เน็ตหลุด: เดินช่วงปัจจุบันให้จบ → ถอยออกรอที่หัวทางเดิน → เก็บภาพไว้ใน outbox แล้วส่งเมื่อต่อได้

## 5) หุ่น → เซิร์ฟเวอร์: จบงานสำรวจ (`POST /run_complete`)
`{"run": "<ชื่อ run>", "path": "<โฟลเดอร์ rec บนเซิร์ฟเวอร์>", "summary": {...}, "source": "robot"}` ส่งครั้งเดียวตอน `run_farm.py` จบ
(ใน sim `rec/` เป็น shared mount; หุ่นจริงใช้ `--upload-rec` rsync ก่อน) hub เขียน `<path>/run_complete.json` แล้วระบบวิเคราะห์
(`cloud/pipeline`: RT-DETR, VLM ต่อจุดหยุด, แผนที่ LiDAR, cuVSLAM, lingbot, Cosmos หน้า, รายงาน) รันต่อเอง → `<path>/analysis/report.json`
+ `report_th.md` + `report_en.md` ตอบกลับ `{"ok": true, "run", "path", "pipeline": "marker written|spawned"}`; path ไม่มีจริง → 400
รายละเอียด: [PIPELINE_th.md](PIPELINE_th.md)
