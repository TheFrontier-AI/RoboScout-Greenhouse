# R1 water hardware (as designed, 2026-10-27/28)

Read from `robot_design/r1_v3/diy_r50/` (outside this repo). Every item is a **candidate, not released
for assembly** (`assembly_release: false` throughout). Nothing is purchased, bench-tested or wired.

| Part | Design value | Source file (under `robot_design/r1_v3/diy_r50/`) |
|---|---|---|
| Tank | Plasto Pack 203, **10 L** HDPE, 158 × 266 × 327 mm, 0.5 kg empty, low rear-removable tray | `tank_r91/fit_review.json`, `tank_suction_interface/decision.json` |
| Usable water | ~8 L (2 L low-water reserve) | `hydraulics/circuit_closure_r132/intermittent42/decision_and_mission.json` |
| Pump (preferred lightweight) | SEAFLO **SFDP2-030-055-42**, 24 V, 55 psi switch + internal bypass, 1.85 kg; **max ON 300 s, min OFF 600 s** → controller budget 290 s ON / 600 s OFF | same file; `duty_guard.py` (reference logic) |
| Pump (alternative) | Flojet **T202F-368S-1P** 24 V, 60 psi switch, internal bypass, 3.5 A, 2.7 kg | `hydraulics/circuit_closure_r132/integrated_pump_revision/decision.json` |
| Pump (earlier) | Shurflo 8000-953-238 24 V | `hydraulics/operating_point_screen.json`, `switching_summary.json` |
| Valves | 6 × SMC **VDW22** 24 V DC NC solenoid, 3 W / 0.125 A each (0.75 A all six) | `valve_driver_compact/decision.json`, `hydraulics/valve_r92/` |
| Valve driver | 2 × ULN2803 modules (3 valves each) candidate; hard power cutoff kept | `valve_driver_compact/decision.json` |
| Nozzles | 6 × TeeJet **TXVK-8** hollow-cone tips + anti-drip check, on **two 3-head risers** (left/right × low/mid/high) | `hydraulics/circuit_closure_r132/decision_and_placement.json` |
| Tip flow (TeeJet curve) | 30 psi 0.12 US gpm (0.45 L/min) · **40 psi 0.13 gpm (0.49 L/min ≈ 8.2 mL/s)** · 50 psi 0.15 · 60 psi 0.16 | `hydraulics/operating_point_screen.json` (TeeJet sheet https://www.teejet.com/-/media/dam/agricultural/usa/technical-documentation/product-data-sheet/36393_tj.pdf) |
| Six heads at 40 psi | ≈ **2.98 L/min**; pump must deliver ~47–51 psi | `decision_and_placement.json`, `intermittent42/decision_and_mission.json` |
| Tank time, all six | 10 L ≈ 3.4 min; 8 L usable ≈ 2.7 min | `intermittent42/decision_and_mission.json` |
| Pump power | SEAFLO 42: 40 psi 2.97 A = 71 W; 50 psi 3.28 A = 79 W. **Existing 24 V 60 W rail is undersized.** Pump power is not closed | `hydraulics/power_audit_r164/result.txt` |
| Open issues | pressure regulation at 40 psi not closed; actual operating point unknown (null); hose/fittings not selected; no pressure sensor selected | `decision_and_placement.json`, `switching_summary.json` |

## Watering numbers derived for HERMES
| Quantity | Value |
|---|---|
| Dwell for 250 mL, one head @ 40 psi | ~30 s (estimate; calibrate with a catch test) |
| Plants per 290 s window @ 250 mL | 1 head: ~9 · 2 heads (left+right low): ~19 |
| Plants per 8 L tank @ 250 mL | ~32 (needs 2 windows + 600 s rest) |
| Spray-while-moving, 1 head | 82 mL/m @ 0.10 m/s · 55 mL/m @ 0.15 · 33 mL/m @ 0.25 |

## Agronomy references
- Drip-irrigated greenhouse tomato: ~80 % ETc gave best yield and water-use efficiency, recommended
  0.37–0.46 L/plant/day for a tropical greenhouse (Nouri Emamzadei, Shahnazari, Karamvand, ISHS Acta Hort. 792:
  https://ishs.org/ishs-article/792_56/). Harmanto et al. 2005 (tropical greenhouse, ~75 % ETc) is cited
  elsewhere as 0.3–0.4 L/plant/day (secondary, not read).
- BC Trickle Irrigation Manual ch. 9: greenhouse tomato feed from 0.4 L up to 3 L/plant/day as plants grow.
  https://www2.gov.bc.ca/assets/gov/farming-natural-resources-and-industry/agricultural-land-and-environment/water/irrigation/trickle-irrigation-manual/chapter9.pdf
- Keep water off leaves; leaf-wetness duration drives fungal infection (UW-Madison Extension):
  https://hort.extension.wisc.edu/2021/04/01/water-water-everywhere-but-never-on-the-leaves/ ; drip rather
  than overhead for early blight (Connecticut AES fact sheet).
