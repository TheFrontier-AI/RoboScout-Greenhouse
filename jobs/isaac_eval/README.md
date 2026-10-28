# isaac_eval: batch evaluation of BOOK in Isaac Sim (Nebius)

`scripts/run_eval_job.py` runs N episodes of R1 driving the centre aisle (`tunnel-09/aisle-02`) of the R180 env and
randomises seed, wheel friction (0.3–1.0 x the 0.7/0.6 estimate, for a wet floor), the test box (on/off, position) and
the start pose (lateral/yaw). The robot uses only its sensors: BOOK centring, SAFETY stop and wheel+IMU odometry.
Ground truth is used for scoring only. Outputs:
- `results.jsonl`: one row per episode with checks and metrics
- `summary.json`: success rate overall, by obstacle/no obstacle, wet/dry, stop error, min clearance, odometry error

Criteria are from the repo README: stop ±0.15 m, side clearance ≥ 0.05 m (concrete-strip reference by default,
`--clearance-ref bed` for bed edge), stop within 0.5 m after the box is detected without contact, and survey ≤ 8 min.

Nothing here has run on Isaac yet. The status of each piece:
- **UNVERIFIED:** the friction, box and start-pose knobs. They edit the stage from outside `SensorStopEnv` (see
  `sim/isaac_env.py`). The job prints the detected knobs, and each episode lists the ones it could not apply.
- **Verified locally:** the `--backend fake` mode, which needs no GPU. Run it before spending GPU time:
  `python3 scripts/run_eval_job.py --backend fake --episodes 12 --out /tmp/eval_fake`

## 1. Stage the build context (on the workstation)
The scenes reference `../../../farm_integration` and `../../../presentation`, so those must be copied too (about
3.6 GB). Big archives and old copies are excluded.
```bash
FARM=/home/passin/Documents/Blender/TheFarm; R=$FARM/robot_design/r1_v3; CTX=/tmp/r1_eval_ctx
mkdir -p $CTX/r1_v3/diy_r50
rsync -a --delete --exclude '__pycache__' --exclude 'runs/' $FARM/r1_greenhouse_nav $CTX/
rsync -a --delete --exclude '*.zip' --exclude 'handoff/' --exclude 'survey_2026-10-28/' \
      --exclude 'tomato_realism_2026-10-28/' --exclude 'realism_gap_2026-10-28/' \
      $R/diy_r50/physical_ai_r180 $CTX/r1_v3/diy_r50/
for d in physical_ai_runtime physical_ai_r167_lidar physical_ai_r167_imu physical_ai_r168_camera; do
  rsync -a --delete --exclude '__pycache__' $R/diy_r50/$d $CTX/r1_v3/diy_r50/; done
rsync -a --delete $R/farm_integration $R/presentation $CTX/r1_v3/
cp $FARM/r1_greenhouse_nav/jobs/isaac_eval/Dockerfile $CTX/
```
The DGX Spark is ARM64. Build for amd64 with buildx and QEMU, or build on the Nebius VM itself, which is faster:
```bash
docker login nvcr.io                       # user: $oauthtoken, password: NGC API key
docker buildx build --platform linux/amd64 -t <registry>/r1-isaac-eval:0.1 --push $CTX
```

## 2. Run on a Nebius VM (RTX PRO 6000)
```bash
docker run --rm --gpus all --network host \
  -v $PWD/results:/results -v ~/isaac-cache:/isaac-sim/.cache \
  <registry>/r1-isaac-eval:0.1 --episodes 50 --seed0 0 --out /results/goto_s0
# survey (drive the whole aisle, reverse out), photo-matched scene:
docker run --rm --gpus all -v $PWD/results:/results <registry>/r1-isaac-eval:0.1 \
  --task survey --episodes 10 --stage photo --out /results/survey_photo
```
Sanity first: `--episodes 1 --obstacle-prob 0 --friction-range 1 1`. Check that `start_gt_metre` ≈ 3.26,
`final_metre` ≈ `target_metre` and `knobs` are all true in `summary.json`. The env renders every physics substep,
so expect roughly real-time or slower. A 30 m goto is about 120 s of sim time.

## 3. Nebius Serverless Jobs (UNVERIFIED)
- Isaac Sim needs an RTX GPU (RT cores) with a working Vulkan/driver stack in the container. Confirm that the
  Serverless Jobs GPU types include an RTX PRO 6000 or L40S class. If they offer only H100/H200 (no RT cores), use
  the VM path above. That is the fallback in `docs/HACKATHON_PLAN_th.md`.
- Shard by seed: submit K jobs with the same image and args `--episodes 25 --seed0 {0,25,50,...}
  --out /results/shard_{k}`. Mount object storage at `/results`, or copy out at the end. Merge the
  `results.jsonl` files and recompute the success rate.
- Environment overrides are supported: `EVAL_EPISODES`, `EVAL_SEED0`, `EVAL_OUT`, `EVAL_STAGE`.
- First start compiles shaders (several minutes). Persist `/isaac-sim/.cache` if the platform allows a volume.

## Known limits
- Only the start in aisle-02 at metre ≈ 3.26 exists in the env. Headland transit, other aisles and docking are not
  simulated.
- Odometry is wheel+IMU only. On a wet floor (friction < 0.6) wheel slip makes the robot stop short, so stop accuracy
  on wet floors measures the localisation gap that SLAM or tags must close.
- `--backend fake` is kinematic with an illustrative slip model. Use it to check the plumbing, never to report results.
