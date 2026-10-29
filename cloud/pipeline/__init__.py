"""Post-survey analysis pipeline: one robot run (rec dir) in, `analysis/` with every result and one report out.
  python3 -m cloud.pipeline run <run_dir> [--force] [--steps a,b]      python3 -m cloud.pipeline watch --runs-dir <dir>
Steps (cloud/pipeline/steps.py): detect, vlm_stops, lidar_map, cuvslam, lingbot, cosmos_front, report. Config: config.py."""
