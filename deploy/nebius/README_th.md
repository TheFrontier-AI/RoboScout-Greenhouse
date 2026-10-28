# รันระบบ R1 บน Nebius (RTX PRO 6000 Blackwell)

## สิ่งที่ต้องมีก่อน (เจ้าของบัญชีเตรียม)
| # | อะไร | ใช้กับ | ที่มา |
|---|---|---|---|
| 1 | บัญชี Nebius AI Cloud + เครดิต | ทุกอย่าง | console.nebius.com |
| 2 | **Token Factory API key** → `NEBIUS_API_KEY` | HERMES + Nemotron | tokenfactory.nebius.com |
| 3 | **VM RTX PRO 6000**: platform `gpu-rtx6000` (us-central1) หรือ `gpu-rtx6000-a` (uk-south2 / eu-south1), preset `1gpu-24vcpu-218gb`, Ubuntu + NVIDIA driver, ดิสก์ ≥ 300 GB, SSH key | Isaac Sim + Cosmos + SLAM + API + แดชบอร์ด | console → Compute |
| 4 | **Hugging Face token** และกดยอมรับ licence ของ `nvidia/Cosmos-Reason2-8B` | ตา (Cosmos) | huggingface.co |
| 5 | **NGC API key** | ดึง image Isaac Sim จาก nvcr.io | ngc.nvidia.com |
| 6 | Object Storage bucket + access key (ไม่บังคับ) | วิดีโอ/ผล SLAM, Serverless Jobs | console → Storage |
| 7 | GitHub repo สาธารณะ (ตอนส่ง) | ส่งงาน Devpost | github.com |

ห้ามส่ง key ในแชท: ใส่ในไฟล์ `deploy/nebius/.env` บน VM เท่านั้น (ไฟล์นี้ถูก ignore ไม่ขึ้น git)

## ขั้นตอน (ประมาณ 30–60 นาที ส่วนใหญ่คือดาวน์โหลดโมเดลและ image)
```bash
# บนเครื่องเรา
scp -r ~/Documents/Blender/TheFarm/r1_greenhouse_nav ubuntu@<VM>:~
scp -r ~/Documents/Blender/TheFarm/robot_design/r1_v3/diy_r50/physical_ai_r180 ubuntu@<VM>:~/r180   # env + ฉาก Isaac

# บน VM
cd ~/r1_greenhouse_nav/deploy/nebius
bash bootstrap_vm.sh                 # รอบแรกจะสร้าง .env แล้วหยุด → แก้ .env → รันซ้ำ
NGC_API_KEY=... bash bootstrap_vm.sh # Cosmos (eyes) + robot API + แดชบอร์ด ขึ้นทำงาน

# ดูจากเครื่องเรา
ssh -L 8080:localhost:8080 -L 8090:localhost:8090 ubuntu@<VM>   # แดชบอร์ด http://localhost:8080
```

## รันแต่ละส่วน
| งาน | คำสั่ง (บน VM) | ดูรายละเอียด |
|---|---|---|
| ตรวจภาพจริงด้วย Cosmos | `EYES_BASE_URL=http://127.0.0.1:8000/v1 python3 -m cloud.eyes.batch_real_photos` | `cloud/README_cloud.md` |
| SLAM AI จากวิดีโอโดรน | `docker build -f cloud/mapping/Dockerfile -t r1-mapping . && docker run --gpus all -v $PWD:/w r1-mapping --video /w/<video> --out /w/runs/demo/slam --up cam-z` | `cloud/mapping/nebius_job.md` |
| เดโมหุ่นใน Isaac | ดู `jobs/isaac_eval/README.md` และ `scripts/run_demo_sim.py --help` | |
| ประเมินหลายร้อยรอบ (Serverless Jobs) | `scripts/run_eval_job.py` / `jobs/isaac_eval/` | |
| สมอง HERMES | `hermes` บนเครื่องที่ต่อ API ได้ + config จาก `cloud/brain/hermes_provider.md` | skill: `agent/hermes_skills/r1-robot-control` |
| สมองสำรอง | `ROBOT_API_URL=http://127.0.0.1:8090 python3 agent/brain_loop.py` | |

## การใช้ GPU ใบเดียว (96 GB)
- Cosmos-Reason2-8B ~35 % (ค่าเริ่มต้นใน compose) + Isaac Sim ที่เหลือ
- งาน SLAM (MapAnything) ใช้หน่วยความจำมาก → รันตอนไม่ได้เปิด Isaac หรือหยุด `eyes` ชั่วคราว (`docker compose stop eyes`)
- ถ้าเปิด Serverless Endpoint แยกสำหรับ Cosmos ได้ → ใช้ 32B ที่นั่น แล้วตั้ง `EYES_BASE_URL` ชี้ไป endpoint

## สิ่งที่ยังไม่ได้ยืนยัน (เช็คเมื่อมีบัญชี)
- Serverless Endpoints/Jobs เลือก RTX PRO 6000 ได้ไหม (เอกสารมีแต่ H100/L40S); ถ้าไม่ได้ → รันทุกอย่างบน VM ข้างบน
- รหัสโมเดล Nemotron ใน Token Factory: เช็คด้วย `curl -H "Authorization: Bearer $NEBIUS_API_KEY" https://api.tokenfactory.nebius.com/v1/models`
- Docker image ของ SLAM และ Isaac ยังไม่ได้ build บนเครื่อง x86 จริง
