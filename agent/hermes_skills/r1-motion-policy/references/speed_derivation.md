# R1 speed derivation (inspection and watering)

Researched 2026-10-28. Numbers marked *assumption* are engineering estimates, not measurements on R1.

## 1. Inputs (from the repo)
| Input | Value | Where |
|---|---|---|
| Forward / reverse limit | 0.25 / 0.15 m/s | `docs/MISSION_SPEC_th.md` |
| Side capture spacing | 0.5 m (6 mast cams), front 2 m | `r1_nav/cameras.py` |
| Revisit hold | 3.0 s | `r1_nav/commands.py` `REVISIT_HOLD_S` |
| Side camera optics (sim) | focal 20 mm, aperture 21.40 × 38.05 mm portrait → ~56° H × 87° V; 180×320 now, 360×640 recommended | `sim/SIDE_CAMERAS_TODO.md` |
| Camera-to-leaf distance | ~0.4–0.6 m (centre aisle; bed edge ~0.46 m) | `sim/SIDE_CAMERAS_TODO.md` |
| Aisle length | ~38 m (76 m round trip in ≤ 8 min) | `README.md` acceptance criterion 4 |

## 2. Footprint and overlap
Along-row footprint `W = 2·d·tan(HFOV/2)`. Overlap at step `s`: `1 − s/W`.

| Camera | d (m) | W (m) | overlap @ 0.5 m | step for 30 % | step for 50 % |
|---|---|---|---|---|---|
| portrait 56° H | 0.40 | 0.43 | **−18 % (gap)** | 0.30 | 0.21 |
| portrait 56° H | 0.46 | 0.49 | −2 % (just touching) | 0.34 | 0.24 |
| portrait 56° H | 0.60 | 0.64 | 22 % | 0.45 | 0.32 |
| landscape 90° H | 0.40 | 0.80 | 38 % | 0.56 | 0.40 |
| landscape 90° H | 0.60 | 1.20 | 58 % | 0.84 | 0.60 |

Target ≥ 30 % so that a lesion near a frame edge is fully visible in at least one frame. Target ≥ 50 % when
you want two views of every leaf, which helps the confirmation vote.

## 3. Motion blur
`blur_px = v · t_exp / GSD`, with `GSD = W / N_px` (along-row pixels). Max exposure for `n` px blur:
`t_max = n · GSD / v` (Vision-Doctor, area-scan exposure time:
https://www.vision-doctor.com/en/camera/exposure-time-area-scan-camera.html, "a blur of 0.5 to 2 pixels is
usually no longer visible to the eye", still reduces measurement accuracy). Same relation for ground
vehicles: relative blur = t·v / (2·h·tan(FoV/2)) (https://arxiv.org/pdf/2505.11116).

Max exposure for 1 px blur, portrait 56° H:
| d | N_px | GSD | 0.10 m/s | 0.15 m/s | 0.20 m/s | 0.25 m/s |
|---|---|---|---|---|---|---|
| 0.4 m | 360 | 1.18 mm | 11.8 ms | 7.9 ms | 5.9 ms | 4.7 ms |
| 0.4 m | 1080 | 0.39 mm | 3.9 ms | 2.6 ms | 2.0 ms | 1.6 ms |
| 0.6 m | 360 | 1.77 mm | 17.7 ms | 11.8 ms | 8.9 ms | 7.1 ms |
| 0.6 m | 1080 | 0.59 mm | 5.9 ms | 3.9 ms | 3.0 ms | 2.4 ms |

Feature size check: early-blight spots are a few mm when young, and whitefly adults are ~1–1.5 mm. You need roughly
≥ 3–5 px across a feature, so detection of small lesions needs GSD ≤ ~0.5 mm/px (high-res camera). That makes
the short-exposure limit the binding one. Under plastic on a cloudy day, auto-exposure of 10–30 ms is common
(*assumption*; measure on R1). Then continuous 0.25 m/s gives 6–19 px blur at 1080 px, and 0.10 m/s or
stop-and-go is required. Other blur sources: rolling-shutter skew, mast vibration, and plants swaying (the
PhenoRob field robot notes wind-moved plants need fast shutter; another field phenotyping robot stops for
every capture: https://arxiv.org/pdf/2310.11516; Bonn ECMR 2021 talk:
https://www.ais.uni-bonn.de/behnke/talks/ECMR_2021_Agri_WS_Sven_Behnke_2021_08_30.pdf).

## 4. Time budget (38 m aisle, in forward then reverse out)
| Forward pattern | Forward | Reverse @ 0.15 | Total | ≤ 8 min? |
|---|---|---|---|---|
| continuous 0.25 m/s | 152 s | 253 s | 6.8 min | yes |
| continuous 0.20 m/s | 190 s | 253 s | 7.4 min | yes |
| continuous 0.15 m/s | 253 s | 253 s | 8.4 min | no |
| stop-and-go every 0.5 m (0.25 m/s, a = 0.25 m/s² *assumption*, 0.5 s settle) | ~266 s | 253 s | 8.7 min | no |
Conclusion: continuous 0.20–0.25 m/s for routine scouting. Use slow or stop-and-go only for revisits or for
a dim aisle, and accept the longer time. Option: also capture during the reverse pass (proposed) to get a
second view without extra time.

## 5. Field reference speeds
- Berg Meto greenhouse pipe-rail spray robot: max 70 m/min (~1.17 m/s), sprays forward and reverse at variable
  speed. https://hortamericas.com/catalog/greenhouse-equipment/berg-meto-spray-robot/
- Berg rail platform: 0–40 m/min (~0–0.67 m/s). https://www.mondomacchina.it/en/in-the-greenhouse-transport-and-spraying-c626
- Free-driving differential-drive greenhouse sprayer tested at 0.15/0.25/0.35 m/s; position RMSE rises with
  speed. https://www.magiran.com/paper/931232/?lang=en
- Scouting robots (PATHoBot pipe-rail trolley, WUR Plantalyzer six-camera cart, Viscon EVA Scoutr): no
  published travel speed found. https://www.hrl.uni-bonn.de/publications/2021/pathobot-a-robot-for-glasshouse-crop-phenotyping-and-intervention ,
  https://www.wur.nl/en/newsarticle/Tomato-grower-has-a-new-buddy.htm , https://viscongroup.eu/fresh-produce/eva-tomato-pest-control/

## 6. Watering speed (see `greenhouse-watering`)
Spot watering is stop-and-dwell: speed while dispensing = 0. Between plants 0.10–0.15 m/s with a full 10 L
tank (+10 kg high on the chassis, sloshing, wet floor afterwards). This is an *assumption*; validate in Isaac
with the tank mass. Spray-while-moving dose per metre = flow ÷ speed: one TXVK-8 head at 40 psi (~8.2 mL/s)
gives 82 mL/m at 0.10 m/s and 33 mL/m at 0.25 m/s. That is fine for a band, not for a per-plant water dose.
