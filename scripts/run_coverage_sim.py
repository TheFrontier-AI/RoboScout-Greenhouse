"""Drive the centre aisle of tunnel-09 in Isaac with the sensor-only controller (phase-2 skeleton).
--dry-run prints the mission legs without starting Isaac. No training."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r1_nav.aisle_map import AisleMap
from r1_nav.mission import survey_house, coverage_distance_m

ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--steps", type=int, default=300)
args = ap.parse_args()
amap = AisleMap.load(Path(__file__).resolve().parents[1] / "data/tunnel09_aisle_map_v18.json")
legs, skipped = survey_house(amap)
print(json.dumps({"house": amap.house_id, "legs": [l.__dict__ for l in legs], "skipped_too_narrow": [a.id for a in skipped],
                  "distance_m": round(coverage_distance_m(legs, amap), 1)}, ensure_ascii=False, indent=1))
if args.dry_run:
    sys.exit(0)
from isaacsim import SimulationApp                  # noqa: E402
app = SimulationApp({"headless": True})
from r1_nav.sim_adapter import make_r180_env, observation_to_scan   # noqa: E402
from r1_nav.controller import wheel_command                         # noqa: E402
env = make_r180_env(app); obs, _ = env.reset(seed=0); log = []
for k in range(args.steps):
    cmd, info = wheel_command(*observation_to_scan(obs))
    obs, reward, term, trunc, einfo = env.step(cmd); log.append({"step": k, **info, "term": einfo.get("termination")})
    if term or trunc: break
Path("coverage_run_log.json").write_text(json.dumps(log, indent=1)); print("steps", len(log), "last", log[-1])
env.close(); app.close()
