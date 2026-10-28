# Which model for which job (R1, 2026-10-28)

Only models we can actually call are listed. Latency and cost figures are not measured here: time them on
our account before quoting numbers.

| Job | Model | Where | Why |
|---|---|---|---|
| **Per-frame screening** (6 side frames every 0.5 m ≈ 3 frames/s at 0.25 m/s; ~460 frames per 38 m aisle) | `nvidia/Cosmos-Reason2-8B`, reasoning **off**, `EYES_STRUCTURED=1` | self-hosted vLLM, RTX PRO 6000 | Highest throughput we control; fixed GPU cost, no per-token bill; physical-AI model trained for embodied scenes (camera on a moving robot). Already the pipeline's "eyes". Switch to `Cosmos-Reason2-32B` (repo default) when the queue keeps up. |
| **Confirmation on revisit** (1–2 sharp frames per spot, a handful per mission) | `nvidia/Nemotron-3.5-Super-VL` | Nebius Token Factory | Verified working on a real greenhouse photo. 120B-A12B hybrid MoE with C-RADIO vision encoder, released 2026-10-15. A **different model family** from the screener, so a matching answer is real evidence and not the same bias twice. Low volume keeps the token cost small. |
| Confirmation, alternative / tie-break | `nvidia/Cosmos-Reason2-32B` with `EYES_REASONING=1` | self-hosted | Strongest Cosmos checkpoint (Qwen3-VL-32B base); `<think>` trace gives an explanation the farmer can read; use when Token Factory is unreachable. |
| **Agent reasoning, decisions, farmer report** (HERMES) | `nvidia/nemotron-3-super-120b-a12b` | Token Factory | Default brain; agentic/tool-use model, 256K context holds a whole mission's findings. |
| Weekly/season summary, hard multi-house trade-offs | `nvidia/Nemotron-3-Ultra-550b-a55b` | Token Factory | Most capable, 1M context; slow/expensive, so use it rarely, not in the control loop. |
| Cheap routine text (status narration, Thai↔English rewrite, label normalisation) | `nvidia/Nemotron-3_5-Lightning` (or `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`) | Token Factory | 30B-A3B class: fast and cheap; text-only. Note the underscore in the Lightning id; copy ids from `GET /v1/models`. |

Sources: model ids and contexts `cloud/README_cloud.md`; Nemotron 3.5 Super VL release/specs
https://letsdatascience.com/news/nvidia-expands-nemotron-35-with-super-vl-and-bolt-ff81d36b ,
https://benchmarklist.com/models/nvidia-nemotron-3.5-super-vl-120b-a12b/ . NVIDIA reports, for example, V* 85.6 and
MathVision 78.2. Independent results are thin and the Terminal-Bench number conflicts between the model card and
the report. Treat it as strong, not proven, on plant pathology. Cosmos Reason 2: https://huggingface.co/nvidia/Cosmos-Reason2-32B ,
https://github.com/nvidia-cosmos/cosmos-reason2 .

No model above was trained specifically on tomato disease. All are general VLMs and can hallucinate a disease.
That is why the skill requires two views plus two model families before "confirmed", and why the report says
"suspected".

## Specialised plant-disease classifiers: add one as a first-pass filter?
Open checkpoints exist, e.g. `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` (PlantVillage,
38 classes, ~2.3k downloads, licence "other"), `wambugu71/crop_leaf_diseases_vit` (MIT),
`rodynaemad/plant-disease-dinov2-small`, `kimcomehome/plantvillage-vit-leaf-disease` (Hugging Face, checked
2026-10-28).

Known limits: PlantVillage images are single detached leaves on a plain background. Models score ~99 % on
held-out PlantVillage (Mohanty et al. 2016), then collapse on field images: ResNet-50 falls below ~30 %, and
a 67.7-point drop has been reported when transferring to PlantDoc. Grad-CAM shows the attention drifting to
the background (https://arxiv.org/pdf/2609.20290 ; https://www.researchsquare.com/article/rs-10437236/v1 ;
https://docs.lib.purdue.edu/fund/58). R1 frames show whole canopies at 0.4–0.6 m with many leaves, plastic,
pipes and soil. That is the worst case for these models, and they also cannot answer `not_plant`/"blurred".

**Recommendation: do not use one as a gate.** A false "healthy" would hide a sick plant before the VLM
ever sees it. Acceptable later uses:
1. as an extra **weak vote** on leaf crops cut out by the VLM's `bbox_norm` (closer to PlantVillage
   framing), logged but never decisive;
2. after fine-tuning on our own labelled R1 frames (the dashboard data dir already stores every frame with
   aisle/metre), as a cheap edge prefilter on the Jetson that only **raises** priority, never suppresses.
A cheaper and safer first-pass filter is image quality, not disease: drop blurred/dark/`not_plant` frames
before the VLM.
