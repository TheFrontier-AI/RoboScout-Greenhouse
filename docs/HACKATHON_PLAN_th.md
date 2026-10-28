# แผนแข่ง Nebius x NVIDIA Global AI Hackathon — สาย Physical AI

**ชื่องาน:** RoboFarm (หุ่น R1): หุ่นเข้าโรงเรือนที่ไม่เคยเห็น สร้างแผนที่เอง หาโรคเอง และตัดสินใจเองว่าจะไปดูตรงไหน
**ส่งงาน:** 30 ต.ค. 2026 10:00 PT = **31 ต.ค. 00:00 เวลาไทย** (ตั้งเป้าส่งจริง 30 ต.ค. 20:00)
**หลักการ:** ใช้โมเดลสำเร็จรูปที่ deploy ได้ทันทีทุกตัว ไม่เทรนอะไรเลยในรอบนี้ ยกเว้น TRAIN ซึ่งเป็นงานเสริม

## เทียบกับเงื่อนไขสาย Physical AI
| เงื่อนไข | แผนเรา |
|---|---|
| embodied/edge agent ที่ **รับรู้และลงมือในโลกจริง** | หุ่น R1 ใน **ดิจิทัลทวินที่สร้างจากฟาร์มจริง** (ฉาก tunnel-09 จูนเทียบภาพถ่ายจริง) + **ข้อมูลจริงเข้าระบบ** (ภาพโรงเรือนจริง, วิดีโอโดรนจริง) |
| ขับเคลื่อนด้วย Nemotron / GR00T / Cosmos / Sonic | **Nemotron** (สมอง) + **Cosmos Reason** (ตา) + **Cosmos Transfer** (ข้อมูลสังเคราะห์); GR00T/Sonic เป็นของหุ่นฮิวแมนนอยด์ ไม่ฝืนใช้ |
| ประสานงานผ่าน agent runtime | **Hermes Agent** เป็น runtime สั่ง SLAM / หุ่น / Cosmos ผ่าน skills |
| **Nebius Serverless Jobs**: จำลอง สร้างข้อมูลสังเคราะห์ ประเมินนโยบาย ประมวลผลเซนเซอร์ | (1) Isaac Sim headless รันประเมิน BOOK แบบสุ่มพื้นลื่น/สิ่งกีดขวาง หลายร้อยรอบ (2) Cosmos Transfer สร้างภาพโรงเรือนหลายสภาพ (3) สร้าง 3D จากวิดีโอที่หุ่นถ่ายมา |
| **Nebius Serverless Endpoints**: inference แบบเรียลไทม์ | Cosmos Reason endpoint ตรวจภาพระหว่างเดิน |
| วิดีโอต้องมีคลิป ≥ 1 นาทีของ **ฮาร์ดแวร์จริงทำงาน** (หรือโมดูลหลักถ้าไม่มีฮาร์ดแวร์) | ไม่มีฮาร์ดแวร์พร้อมในรอบนี้ → **คลิปโมดูลหลักทำงานต่อเนื่อง ≥ 1 นาที** ในดิจิทัลทวิน (กติกายอมรับ) |

## แนวทาง Sim-to-Real (เล่าในวิดีโอเป็น Real → Sim → Real)
1. **Real → Sim (ทำแล้ว):** สร้างทวินจากฟาร์มจริง (หลักฐานด้านล่าง)
2. **Sim (รอบแข่งนี้):** พัฒนาและประเมินทุกโมดูลในทวิน บน Nebius Serverless Jobs หลายร้อยรอบ รวมพื้นลื่น/สิ่งกีดขวาง
3. **Sim → Real (ต่อจากนี้):** stack เดียวกันลง Jetson บนหุ่น R1 จริง; โค้ด SAFETY/MISSION/BOOK ไม่ขึ้นกับ sim
   (`r1_nav/` ไม่ใช้ตำแหน่งจริงจาก sim ใช้แค่เซนเซอร์) ส่วนที่เปลี่ยนมีแค่ตัวต่อ (`sim_adapter.py` → ROS 2 บน Jetson)
   และ Cosmos Transfer ทำภาพ sim ให้หลากหลายเหมือนจริง เพื่อปิดช่องว่าง sim-to-real ของตัวตรวจภาพ

## Real → Sim: หลักฐานว่า sim มาจากโลกจริง (ของที่มีอยู่แล้ว)
- ภาพถ่ายโรงเรือนจริง: `refs/02_tomato_greenhouses/` (ทางเดิน A/B/C, แปลงยกสูง, ระบบน้ำ, ประตู/พัดลม)
- วิดีโอโดรนบินเหนือฟาร์มจริง 2:33 นาที: `refs/06_video/drone_flyover_site_2m33s.mp4`
- ฉากจูนเทียบภาพจริง: `C2_R180_tunnel09_photo_match.usda` + ภาพเทียบ 4 ทาง `compare_4way_RT100_PT100.jpg`
  (ความสว่างในทางเดิน: จริง 126 / Isaac 128)
- หุ่น R1 ออกแบบจากชิ้นส่วนที่ซื้อได้จริงในไทย (`MASTER_PROCUREMENT_REGISTER.csv`)
- **ข้อมูลจริงผ่านระบบจริง:** วิดีโอโดรนผ่าน Serverless Jobs สร้างฟาร์ม 3D (MASt3R/VGGT) แล้วเทียบกับโมเดล V18;
  ภาพโรงเรือนจริงผ่าน Cosmos Reason endpoint ตรวจสภาพต้น

## สิ่งที่กรรมการจะเห็นใน 3 นาที
1. **0:00–0:25 จริง → ทวิน:** ภาพจริงเทียบฉาก Isaac ข้างกัน และวิดีโอโดรนจริงกลายเป็นฟาร์ม 3D (Serverless Jobs)
2. **0:25–1:45 โมดูลหลักทำงานต่อเนื่อง (≥ 1 นาที):** R1 เข้า tunnel-09 โดยไม่มีผัง → SLAM สร้างแผนที่ → HERMES (Nemotron) วางแผน
   → Cosmos Reason endpoint เจอใบผิดปกติ → HERMES สั่งกลับไปดูซ้ำ → หุ่นไปถึงและยืนยัน
3. **1:45–2:15 ภาพจริงเข้า Cosmos:** ระบบเดียวกันตรวจภาพจากโรงเรือนจริง
4. **2:15–2:40 Serverless Jobs:** ประเมิน BOOK หลายร้อยรอบบนพื้นลื่น/สิ่งกีดขวาง พร้อมอัตราสำเร็จ
5. **2:40–3:00** แผนที่พร้อมหมุดโรค, รายงานจาก HERMES ถึงเกษตรกร, แผนต่อ (หุ่นจริงกำลังประกอบ, Jetson)

## สถาปัตยกรรม
```
🤖 Edge: Jetson (จริง) / R1 (Isaac Sim)              ☁️ Nebius Serverless Endpoints
 ├ 🛑 SAFETY   หยุด/STUCK/e-stop                       └ 🟢 Cosmos Reason   ตรวจโรค + ยืนยันว่าหุ่นเดินถูก
 ├ 🗺️ MISSION  discover/survey/revisit/dock           ☁️ Nebius Serverless Jobs
 ├ 📘 BOOK     ขับล้อกลางทางเดิน                         ├ 🧪 Isaac Sim headless  ประเมิน BOOK หลายร้อยรอบ
 └ 📡 SLAM     RTAB-Map / slam_toolbox (LiDAR+กล้อง)    ├ 🟢 Cosmos Transfer     ภาพสังเคราะห์ (เปียก/แดด/โรค)
        │ ภาพ + ตำแหน่ง + สถานะ ▲ เป้าหมาย               └ 🧊 MASt3R-SLAM/VGGT + 3DGS  โรงเรือน 3D
        └──────────────────────┴─────────────────────  ☁️ Nebius Token Factory
                                                        └ 🧠 HERMES (agent runtime) + 🟢 Nemotron 3
```

## งาน 4 สาย (ทำคู่ขนาน)
| สาย | งาน | ทางสำรองถ้าติด (ยังอยู่ในแผน) |
|---|---|---|
| **A ทวิน + หุ่น** | R1 ใน sim: BOOK, SLAM, `discover_house`, ใบผิดปกติในฉาก; คลิปเทียบจริง ↔ ทวิน | SLAM กล้องติด → slam_toolbox (LiDAR) |
| **B Nebius Serverless** | Endpoint: Cosmos Reason (ภาพ sim + ภาพจริง); Jobs: ประเมิน BOOK ใน Isaac, วิดีโอโดรนจริง → 3D, Cosmos Transfer | Isaac ใน Jobs ไม่รองรับ GPU RTX → รันประเมินบน VM 6000 ของ Nebius AI Cloud แทน |
| **C HERMES** | skill `r1-robot-control`, Nemotron ผ่าน Token Factory, วงจร ครบทุกทาง → พบโรค → revisit → กลับแท่น | tool-call ไม่นิ่ง → JSON schema + หุ่นตรวจคำสั่งซ้ำ |
| **D ส่งงาน** | แดชบอร์ดแผนที่ + หมุดโรค, README, LICENSE Apache 2.0, วิดีโอ ≤ 3 นาที, Devpost | — |

## ตารางเวลา
| เมื่อ | A Edge + sim | B Serverless | C HERMES | D ส่งงาน |
|---|---|---|---|---|
| **28 ต.ค.** | env sim ทำงาน, BOOK เดินทางเดินกลาง | Cosmos Reason endpoint, Job วิดีโอโดรน → 3D | Token Factory key, Hermes ↔ Nemotron | สมัคร Devpost |
| **29 ต.ค.** | SLAM + discover ใน sim, ใบผิดปกติในฉาก | Jobs ประเมิน BOOK, Cosmos Transfer, Cosmos ตรวจภาพจริง | skill สั่งหุ่น, วงจร revisit | แดชบอร์ด, คลิปจริง ↔ ทวิน |
| **30 ต.ค.** | รันเดโมเต็ม 3 ครั้ง | 3DGS + หมุดโรค, สรุปผลประเมิน | รายงานถึงเกษตรกร | อัดวิดีโอ, repo สาธารณะ, **ส่ง 20:00** |

## งานเสริม (ไม่ทำก็ส่งได้ครบ)
- **COMPASS / X-Mobility** (NVIDIA, deploy อย่างเดียว) ประเมินเทียบ BOOK ใน Serverless Jobs ชุดเดียวกัน
- **TRAIN (TD-MPC2)** ตัวเดียวที่ต้องเทรนจริง เก็บไว้หลังแข่ง

## ความเสี่ยงหลัก
1. ไม่มีฮาร์ดแวร์จริงในวิดีโอ: ชดเชยด้วยหลักฐาน Real → Sim และข้อมูลจริงที่ผ่านระบบ (ภาพจริง + วิดีโอโดรนจริง)
2. Serverless Jobs: ต้องเช็คว่ามี GPU ที่มี RT core สำหรับ Isaac Sim หรือไม่ และใช้ container x86
3. env Isaac (Astra): ช้า และยังไม่รองรับการถอยหลัง
4. Cosmos Reason ยังไม่พบใน Token Factory → ทำ Serverless Endpoint เอง
5. License: repo ต้องเป็น Apache 2.0; ข้อมูลฟาร์ม ฉาก 3D และโมเดลที่ปรับเองไม่ต้องใส่ใน repo
