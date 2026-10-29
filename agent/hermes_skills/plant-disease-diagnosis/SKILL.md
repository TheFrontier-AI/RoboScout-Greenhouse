---
name: plant-disease-diagnosis
description: "Ingest R1 plant findings from /findings, aggregate them per plant spot, decide confidence and revisits, get a second-model confirmation, and write the Thai/English farmer report for greenhouse tomatoes."
version: 0.1.0
author: TheFarm R1 team
license: Apache-2.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Agriculture, Plant Disease, Tomato, Greenhouse, Vision, Nemotron, Cosmos]
    related_skills: [r1-robot-control, r1-motion-policy, greenhouse-watering]
prerequisites:
  commands: [curl, python3]
---

# Plant Disease Diagnosis (R1 findings → confirmed spots → farmer report)

You are HERMES. The robot's eyes (Cosmos Reason 2 by default) look at every side-camera frame. You turn that
stream of per-frame opinions into **a few trustworthy spots** with a recommended action for the farmer.

> **Not a diagnosis.** R1 gives a visual screening only. It is not a substitute for an agronomist, an
> extension officer or a lab test (PCR/ELISA for viruses, culture for bacteria). Always include this disclaimer
> in the farmer's language (Thai on our farm): "This is a visual screening only, not a diagnosis by an agronomist
> or a lab result".

## When to Use
- New rows appear at `GET $ROBOT_API_URL/findings?since=N` during or after a survey.
- The farmer asks "which plants are sick?", "what is this disease?", or "what should I do?"
- A `revisit` finished and you must confirm or clear the spot.

## 1. Ingest
```bash
curl -s "$ROBOT_API_URL/findings?since=$NEXT"     # -> {"next": N, "findings": [row, ...]}; keep "next"
```
Row = `frame_id, aisle, metre, camera, image_path, purpose` + Finding
`{status: healthy|suspect|disease|not_plant|error, issues:[{label, confidence 0..1, bbox_norm, evidence}], summary, meta}`.
`meta` holds the frame metadata, including `pose_confidence` and `heading_deg`. `image_path` is relative to the robot API
data dir (`$R1_DATA_DIR`, e.g. `runs/demo/frames/123_mast_left_mid.jpg`). Images are **not** served over
HTTP, so you need file access on the same host to re-check one.
Drop: `not_plant`, `error` (count them; many errors → eyes are down, tell the operator),
`front` camera rows (navigation only), and rows with `meta.pose_confidence < 0.5` (keep them for the aisle,
don't pin them to a metre).

## 2. Aggregate into spots
- Spot key = `(aisle, side, metre_bin)`: side = `left`/`right` from the camera name, metre merged within
  **1.0 m** (same rule as `r1-robot-control`). Height (`low/mid/high`) is kept as evidence. Lower-leaf
  diseases (early blight, septoria, leaf mold, Mg deficiency) usually start on `low`. Virus and powdery mildew
  can show on `high`/young growth first.
- Normalise free-text labels to the canonical keys in `references/tomato_diseases.md`
  (e.g. "Alternaria", "target spots" → `early_blight`). Unknown label → `other:<text>`.
- Frame score: `disease` → max issue confidence; `suspect` → 0.5 × max confidence (0.3 if no issues);
  `healthy` → 0.
- Spot score = max frame score, raised by **+0.1 per additional independent view** (different camera or
  different pass) that names the same canonical label (cap 0.95).

## 3. Decide
| Condition | Spot state | Action |
|---|---|---|
| any `disease` frame ≥ 0.6, or ≥ 2 `suspect` within 1 m | `candidate` | one `revisit` (aisle, metre); merge spots < 1 m |
| `suspect` blamed on blur/occlusion/distance | `candidate-quality` | revisit (stop-and-hold gives a sharp frame; see `r1-motion-policy`) |
| single `suspect` < 0.6 | `watch` | no revisit; list in report; re-check next survey |
| revisit frames all `healthy` | `cleared` | report as cleared |
| ≥ 2 views agree on the same label ≥ 0.6, **and** the second-opinion model agrees (step 4) | `confirmed` | report + farmer action |
| models disagree, or label is a virus/bacteria look-alike | `uncertain` | ask farmer to check by hand / photo / lab |
Revisit each spot at most once per mission. Priority: high-risk spread labels first (`late_blight`,
`tylcv`, `bacterial_spot`), then by score.

## 4. Confirm with a second model (revisit frames only)
Pick the 1–2 sharpest `purpose=revisit` frames for the spot (highest confidence, bbox not at the image edge).
Re-run them through a **different** model with the repo's client (same prompt and JSON schema):
```bash
cd "$R1_REPO"   # r1_greenhouse_nav
# A) Nemotron 3.5 Super VL on Nebius Token Factory (hosted; verified working on a real greenhouse photo)
EYES_BASE_URL=https://api.tokenfactory.nebius.com/v1/ EYES_MODEL=nvidia/Nemotron-3.5-Super-VL \
EYES_API_KEY="$NEBIUS_API_KEY" python3 -m cloud.eyes.inspect "$R1_DATA_DIR/frames/123_mast_left_mid.jpg" \
  --camera mast_left_mid --aisle tunnel-09/aisle-02 --metre 15.0
# B) Cosmos Reason 2 32B with reasoning (self-hosted vLLM on the RTX PRO 6000)
EYES_BASE_URL="$COSMOS_URL/v1" EYES_MODEL=nvidia/Cosmos-Reason2-32B EYES_API_KEY="$COSMOS_KEY" \
EYES_REASONING=1 python3 -m cloud.eyes.inspect <image> --camera <cam> --aisle <aisle> --metre <m>
```
Rules: never put API keys in `reasoning`, reports or logs. Do not run `cloud/eyes/inspect.py` as a path; use
`python3 -m`. `EYES_REASONING=1` is the Cosmos `<think>` format; leave it off for Nemotron.
If the screening model was Cosmos, confirm with Nemotron VL (independent family), and the reverse.
Model choice and why: `references/models.md`.

## 5. Report to the farmer (farmer's language first, then English)
Write the report in the farmer's language (Thai on our farm) first, then the same in English. One block per
confirmed or uncertain spot, then a summary. Keep it short and concrete. The structure, shown here in English:
```
Inspection report tunnel-09 — 2026-10-28 09:40
Aisles inspected: aisle-01, aisle-02 | skipped: aisle-03 (too narrow, 0.55 m)
1) aisle-02 metre 15.0 left side, low — suspected late blight, confidence 0.8 (confirmed by 2 camera views + 2 models)
   Evidence: dark water-soaked lesions on the leaf, white mould under the leaf | image: frames/123_mast_left_low.jpg
   Action: inspect by eye today; pick the affected leaves/plant into a bag and dispose outside the greenhouse;
           lower humidity and improve airflow; ask the extension officer about registered products
Clear (re-inspected, looks normal): aisle-01 metre 8.5
Note: This is a visual screening only, not a diagnosis by an agronomist or a lab result
--- English ---
Covered aisle-01, aisle-02; skipped aisle-03 (too narrow). 1 confirmed spot: aisle-02 15.0 m left low,
suspected late blight (0.8; 2 views + 2 models)... This is visual screening, not an agronomist or lab diagnosis.
```
Include: aisle, metre, side, height, canonical label (local name + English), confidence, the evidence text, the
image path, the state (`confirmed`/`uncertain`/`cleared`/`watch`), and the action from the reference table.
Say "suspected", never "diagnosed". For `tylcv`, `bacterial_spot` and anything `uncertain`,
recommend a lab test or an extension officer.

## Rules
- Never invent findings, labels, metres or confidences. If nothing is wrong, say so plainly.
- Never recommend a specific pesticide product or dose. Say "registered product per label / ask the Thai
  Department of Agriculture or the local extension officer".
- Never command spraying or watering. See `greenhouse-watering`, which refuses today.
- Thresholds above are starting values, not calibrated. Log every decision so the farm team can tune them.
