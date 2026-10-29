# Running the R1 system on Nebius (RTX PRO 6000 Blackwell)

## Prerequisites (prepared by the account owner)
| # | What | Used for | Where |
|---|---|---|---|
| 1 | Nebius AI Cloud account + credits | everything | console.nebius.com |
| 2 | **Token Factory API key** → `NEBIUS_API_KEY` | HERMES + Nemotron | tokenfactory.nebius.com |
| 3 | **RTX PRO 6000 VM**: platform `gpu-rtx6000` (us-central1) or `gpu-rtx6000-a` (uk-south2 / eu-south1), preset `1gpu-24vcpu-218gb`, Ubuntu + NVIDIA driver, disk ≥ 300 GB, SSH key | Isaac Sim + Cosmos + SLAM + API + dashboard | console → Compute |
| 4 | **Hugging Face token** and accept the licence of `nvidia/Cosmos-Reason2-8B` | eyes (Cosmos) | huggingface.co |
| 5 | **NGC API key** | pulling the Isaac Sim image from nvcr.io | ngc.nvidia.com |
| 6 | Object Storage bucket + access key (optional) | videos/SLAM results, Serverless Jobs | console → Storage |
| 7 | Public GitHub repo (at submission) | Devpost submission | github.com |

Never send keys in chat: put them only in the file `deploy/nebius/.env` on the VM (this file is git-ignored).

## Steps (about 30–60 minutes, mostly downloading models and images)
```bash
# on our machine
scp -r ~/Documents/Blender/TheFarm/r1_greenhouse_nav ubuntu@<VM>:~
scp -r ~/Documents/Blender/TheFarm/robot_design/r1_v3/diy_r50/physical_ai_r180 ubuntu@<VM>:~/r180   # env + Isaac scene

# on the VM
cd ~/r1_greenhouse_nav/deploy/nebius
bash bootstrap_vm.sh                 # first run creates .env and stops → edit .env → run again
NGC_API_KEY=... bash bootstrap_vm.sh # Cosmos (eyes) + robot API + dashboard come up

# view from our machine
ssh -L 8080:localhost:8080 -L 8090:localhost:8090 ubuntu@<VM>   # dashboard http://localhost:8080
```

## Running each part
| Task | Command (on the VM) | Details |
|---|---|---|
| Inspect real photos with Cosmos | `EYES_BASE_URL=http://127.0.0.1:8000/v1 python3 -m cloud.eyes.batch_real_photos` | `cloud/README_cloud.md` |
| SLAM AI from the drone video | `docker build -f cloud/mapping/Dockerfile -t r1-mapping . && docker run --gpus all -v $PWD:/w r1-mapping --video /w/<video> --out /w/runs/demo/slam --up cam-z` | `cloud/mapping/nebius_job.md` |
| Robot demo in Isaac | see `jobs/isaac_eval/README.md` and `scripts/run_demo_sim.py --help` | |
| Evaluation over hundreds of rounds (Serverless Jobs) | `scripts/run_eval_job.py` / `jobs/isaac_eval/` | |
| HERMES brain | `hermes` on a machine that can reach the API + config from `cloud/brain/hermes_provider.md` | skill: `agent/hermes_skills/r1-robot-control` |
| Fallback brain | `ROBOT_API_URL=http://127.0.0.1:8090 python3 agent/brain_loop.py` | |

## Sharing the single GPU (96 GB)
- Cosmos-Reason2-8B ~35 % (the default in compose) + Isaac Sim takes the rest
- The SLAM job (MapAnything) uses a lot of memory → run it when Isaac is not open, or stop `eyes` temporarily (`docker compose stop eyes`)
- If a separate Serverless Endpoint for Cosmos can be opened → use the 32B there and point `EYES_BASE_URL` at that endpoint

## Not yet verified (check once the account exists)
- Whether Serverless Endpoints/Jobs can select the RTX PRO 6000 (the docs only list H100/L40S); if not → run everything on the VM above
- The Nemotron model IDs on Token Factory: check with `curl -H "Authorization: Bearer $NEBIUS_API_KEY" https://api.tokenfactory.nebius.com/v1/models`
- The SLAM and Isaac Docker images have not yet been built on a real x86 machine
