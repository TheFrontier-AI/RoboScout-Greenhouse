---
name: greenhouse-watering
description: "Plan spot watering for R1 (dose, dwell, heads, pump duty, speed) from its designed 10 L tank / 24 V pump / 6-valve TXVK-8 spray hardware. Planning only: the robot API has no water command yet, so never send one."
version: 0.1.0
author: TheFarm R1 team
license: Apache-2.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Agriculture, Irrigation, Greenhouse, Tomato, Robotics, Planning]
    related_skills: [r1-robot-control, r1-motion-policy, plant-disease-diagnosis]
prerequisites:
  commands: [curl]
---

# Greenhouse Spot Watering (planning only)

**Status: hardware designed, API not wired.** R1's design includes a water system: a 10 L tank, a 24 V
diaphragm pump, 6 solenoid valves and 6 TeeJet TXVK-8 hollow-cone tips on two 3-head risers (left/right ×
low/mid/high). None of it is released for assembly, bench-tested, in the Isaac sim, or in the robot API.
Numbers: `references/r1_water_hardware.md`.

## Hard rule: refuse to water today
- The robot API accepts only `discover_house, survey_house, survey_aisle, go_to, revisit, return_to_dock,
  stop, set_map`. There is **no `water` command**. Never send one, never fake one with `go_to`/`revisit`, and
  never tell the farmer that R1 watered anything.
- If asked to water, reply in this spirit, in the farmer's language (Thai on our farm): "R1 cannot water yet.
  The water system is designed but not built, and the robot has no `water` command. I can make a watering plan
  (which plants, how many mL, how many seconds) for a person to follow now, or for R1 later."
- Never plan pesticide/fungicide/fertiliser spraying with R1. That needs a licensed applicator, product
  labels and re-entry intervals. Out of scope.
- Re-check `/health` each mission. Only when the robot team adds `water` and `/health` reports the water
  system as live may you use the proposed command below.

## When to Use
- The farmer asks R1 to water, or asks how much water or how long per plant.
- `plant-disease-diagnosis` reports `water_stress` (wilting, dry substrate) at specific spots.
- Planning the future watering mission for the hackathon demo or the robot team.

## Planning procedure
1. **Pick plants.** Use `water_stress` spots from findings or the farmer's list. One wilting plant in a
   drip-irrigated row usually means a **clogged or broken dripper**. Recommend fixing the line first. R1
   spot watering is a top-up, not the main irrigation.
2. **Dose per plant.** Mature greenhouse tomato uses about 0.3–0.8 L/plant/day (tropical optimum
   0.3–0.46 L; hot peak can reach 2–3 L). For a top-up use **150–300 mL per plant per visit**, split into 2
   pulses ≥ 5 min apart if the substrate is very dry (less runoff). The farmer's number always wins.
3. **Heads.** Use only the **`low`** head on the plant's side, aimed at the substrate/stem base. `mid`/`high`
   heads wet the foliage. Leaf wetness drives early blight, late blight, leaf mold and bacterial spot. Never use
   them for watering, and never spray over a spot labelled `late_blight`/`bacterial_spot`, because splash
   spreads it.
4. **Dwell.** `dwell_s = dose_mL / (n_heads × q_head)`, with q_head ≈ **8.2 mL/s** (TXVK-8 at 40 psi =
   0.13 US gpm ≈ 0.49 L/min; 7.6 mL/s at 30 psi, 10 mL/s at 60 psi). 250 mL with one low head ≈ **30 s**.
   Two plants facing each other (left low + right low together) take the same 30 s for both. Use **measured**
   flow once the bench catch-test exists; until then mark dwell as "estimated".
5. **Pump duty and tank.** The pump candidate (SEAFLO 42 series) is limited to 300 s ON then 600 s OFF. The
   design budget is **≤ 290 s cumulative pump-on, then 600 s rest**. Priming and bypass count, and short
   pauses do not reset it. Usable water is ~8 L (2 L low-level reserve).
   - one low head: 290 s × 8.2 mL/s ≈ 2.4 L per duty window ≈ 9 plants at 250 mL
   - two low heads: ≈ 4.75 L per window ≈ 19 plants at 250 mL
   - a full 8 L tank ≈ 32 plants at 250 mL → 2 duty windows (≥ 10 min rest between), then refill at the dock.
6. **Motion** (see `r1-motion-policy`): stop fully, settle ≥ 0.5 s, open valve(s), dwell, close, wait ~1 s
   (anti-drip check valve) before moving. Between plants drive **0.10–0.15 m/s**: the full tank adds ~10 kg
   high on the chassis, it sloshes, and the floor gets wet. That speed is an engineering choice to validate in
   Isaac. Never dispense while moving for a per-plant dose. Spray-while-moving gives only 33–82 mL per metre
   at 0.25–0.10 m/s per head.
7. **Timing.** Water in the **morning** so leaves and floor dry before night (extension guidance: evening
   wetting keeps plants wet overnight). Water before a survey only if the farmer accepts a wet aisle.
   Otherwise survey first.
8. **Output a plan, not commands**: table with aisle, metre, side, head, dose_mL, est. dwell_s, cumulative
   pump-on s, tank left L, rest breaks. Write the plan in the farmer's language (Thai on our farm) and English
   and mark it "plan for a person to execute (R1 cannot water yet)".

## Proposed API extension (NOT implemented: for the robot team)
```jsonc
// POST /command
{"cmd": "water", "aisle": "tunnel-09/aisle-02", "metre": 15.0,
 "side": "left|right|both", "heads": ["low"],          // mid/high refused for watering
 "dose_ml": 250, "max_dwell_s": 45,                   // robot computes dwell from measured flow, clamps to max
 "reasoning": "aisle-02 15 m left: morning wilting, substrate dry; dripper check requested"}
// reply: accepted|rejected + reason (no hardware, tank low, duty lockout, pressure out of range, person near)
// GET /status additions: water: {live, tank_l, low_level, pump_on_s_used, cooldown_s_left,
//                                pressure_psi, valves_open: [...], last_dose_ml}
```
On-robot rules (not HERMES): NC valves close and the pump relay opens on fault, E-stop or low level. A hardware timer
enforces the 290 s / 600 s duty limit, and a pressure check runs before opening any valve. HERMES only proposes
doses, and the robot has the final say.
