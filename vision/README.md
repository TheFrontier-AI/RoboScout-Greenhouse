# R1 vision: fruit grading, plant count, leaf anomalies

Detector for R1's six mast (side) cameras. It runs next to Cosmos Reason and writes findings in the same format, so the hub and dashboard pick them up without changes.

| Class | What it means for the farmer |
|---|---|
| `fruit_ripe` | red, ready: **PASS** |
| `fruit_turning`, `fruit_green` | **NOT READY** |
| `fruit_lesion` | dark lesion: **REJECT** |
| `plant_base` | main stem 0–35 cm above the bed, one per plant, used for **plant counting** |
| `leaf_chlorosis`, `leaf_spots` | anomaly region; a pin is placed at the aisle metre where it was seen |

Wilting, stunted and defoliated plants are recognisable only from the shape of the whole plant, so this detector does not box them. Cosmos Reason covers them.

## Data: synthetic, labelled automatically (nobody draws boxes)
`synth/gen_tomato_dataset.py` (run inside Blender) builds random tomato rows from `Tomato-Assets-V1`, with eight healthy plants and twelve synthetic anomaly variants. It renders 576×1024 frames from R1's three mast heights. Labels come straight from the renderer:
- Every fruit and every plant stem is its own object, so its box is the extent of its visible pixels in the object-index pass.
- Fruit ripeness comes from the asset's per-fruit `col_botanical` colour.
- Anomaly boxes are clusters of the asset pack's exact `ANOMALY_TISSUE` AOV pixels.
- A fruit less than 30% visible behind leaves is marked `ignore`.

Mild leaf spots are not labelled because they are invisible at mast distance.

```bash
blender -b --factory-startup .../Tomato-Assets-V1/tomato-assets.blend --python vision/synth/gen_tomato_dataset.py -- \
    --out vision_data/synth_v1 --scenes 30 --frames-per-scene 20 --seed 11
python3 vision/tools/draw_labels.py vision_data/synth_v1 qa.jpg      # visual check of labels
```

## Train, run, integrate
```bash
.venv-vision/bin/python vision/train_rtdetr.py --data vision_data/synth_v1 --out vision_runs/rtdetr_v1 --epochs 40
.venv-vision/bin/python vision/detector.py --model vision_runs/rtdetr_v1/best frame.png --metre 12.5 --camera mast_left_low
DETECTOR_MODEL=vision_runs/rtdetr_v1/best .venv-vision/bin/python agent/robot_api.py ...   # hub adds RT-DETR findings
```
- **Model:** RT-DETRv2-r18 (`PekingU/rtdetr_v2_r18vd`, Apache-2.0), fine-tuned.
- **Validation split:** by rendered scene, so validation rows are never seen in training.
- **Counting:** each side frame owns the 0.5 m strip of row at its capture metre and the height band of its mast camera, so fruit and plants are counted once.

## Honest limits
- Training data is synthetic only. Real greenhouse photos (`refs/02_tomato_greenhouses/`) are a separate test set to be annotated by hand. They are never used for training.
- The anomaly classes are synthetic visual alterations from the asset pack, not diagnosed diseases.
