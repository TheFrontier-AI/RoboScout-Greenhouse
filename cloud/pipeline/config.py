"""Pipeline configuration from environment variables (no YAML): paths, which steps run, model names, batch sizes,
Token Factory endpoints. `PipelineConfig.from_env()` reads os.environ (or a dict you pass) with sane defaults, so the
same code runs on the Nebius server (deploy/nebius/start_pipeline.sh) and in tests (everything stubbed).

  PIPELINE_STEPS        comma list, default all in order: detect,vlm_stops,lidar_map,cuvslam,lingbot,cosmos_front,report
  DETECTOR_MODEL        RT-DETR dir (vision_runs/rtdetr_v2_mixed/best)   DETECTOR_THRESHOLD 0.5
  EYES2_BASE_URL/MODEL  Nemotron VL on Token Factory (per-stop VLM)      EYES2_API_KEY | NEBIUS_API_KEY
  PIPELINE_VLM_WORKERS  parallel stop requests (4)   PIPELINE_VLM_LONG_SIDE 1280   PIPELINE_VLM_DETAILED 1
  EYES_BASE_URL/MODEL   Cosmos Reason vLLM endpoint for the front camera (optional)   PIPELINE_COSMOS_EVERY_S 10
  CUVSLAM_PYTHON        interpreter with the cuvslam wheel (default: this one)   CUVSLAM_MODES stereo,inertial,rgbd
  LINGBOT_RUN           ~/lingbot/run.sh   LINGBOT_IMAGE lingbot:blackwell   LINGBOT_GPU 0
  BRAIN_BASE_URL/MODEL  Nemotron text for report_th.md / report_en.md (needs NEBIUS_API_KEY)
  PIPELINE_MATCH_M      ground-truth match radius (0.7 m)   PIPELINE_MAP_EVERY 5 (steps per map video frame)"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALL_STEPS = ("detect", "vlm_stops", "lidar_map", "cuvslam", "lingbot", "cosmos_front", "report")
TOKEN_FACTORY = "https://api.tokenfactory.nebius.com/v1/"


@dataclass
class PipelineConfig:
    repo: Path = ROOT
    steps: tuple[str, ...] = ALL_STEPS
    # detect (RT-DETR)
    detector_model: str = "vision_runs/rtdetr_v2_mixed/best"
    detector_threshold: float = 0.5
    # vlm_stops (Nemotron VL on Token Factory, one request per plant stop)
    eyes2_base_url: str = TOKEN_FACTORY
    eyes2_model: str = "nvidia/Nemotron-3.5-Super-VL"
    eyes2_api_key: str = ""
    vlm_workers: int = 4
    vlm_long_side: int = 1280
    vlm_detailed: bool = True
    # cosmos_front (Cosmos Reason on a vLLM endpoint; optional)
    eyes_base_url: str = ""
    eyes_model: str = "nvidia/Cosmos-Reason2-8B"
    eyes_api_key: str = "EMPTY"
    cosmos_every_s: float = 10.0
    # cuvslam (optional GPU step)
    cuvslam_python: str = sys.executable
    cuvslam_modes: tuple[str, ...] = ("stereo", "inertial", "rgbd")
    # lingbot-map (optional GPU step, docker)
    lingbot_run: Path = Path.home() / "lingbot/run.sh"
    lingbot_image: str = "lingbot:blackwell"
    lingbot_gpu: int = 0
    # report (Nemotron text on Token Factory; template without a key)
    nebius_api_key: str = ""
    brain_base_url: str = TOKEN_FACTORY
    brain_model: str = "nvidia/nemotron-3-super-120b-a12b"
    # misc
    match_m: float = 0.7
    map_every: int = 5
    env: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_env(cls, env: dict | None = None) -> "PipelineConfig":
        e = dict(os.environ if env is None else env)
        g = lambda k, d="": e.get(k, d) or d
        steps = tuple(s.strip() for s in g("PIPELINE_STEPS", ",".join(ALL_STEPS)).split(",") if s.strip())
        bad = [s for s in steps if s not in ALL_STEPS]
        if bad:
            raise ValueError(f"unknown PIPELINE_STEPS {bad}; known: {ALL_STEPS}")
        return cls(
            steps=steps,
            detector_model=g("DETECTOR_MODEL", cls.detector_model), detector_threshold=float(g("DETECTOR_THRESHOLD", "0.5")),
            eyes2_base_url=g("EYES2_BASE_URL", TOKEN_FACTORY), eyes2_model=g("EYES2_MODEL", cls.eyes2_model),
            eyes2_api_key=g("EYES2_API_KEY") or g("NEBIUS_API_KEY"),
            vlm_workers=int(g("PIPELINE_VLM_WORKERS", "4")), vlm_long_side=int(g("PIPELINE_VLM_LONG_SIDE", "1280")),
            vlm_detailed=g("PIPELINE_VLM_DETAILED", "1") == "1",
            eyes_base_url=g("EYES_BASE_URL"), eyes_model=g("EYES_MODEL", cls.eyes_model), eyes_api_key=g("EYES_API_KEY", "EMPTY"),
            cosmos_every_s=float(g("PIPELINE_COSMOS_EVERY_S", "10")),
            cuvslam_python=g("CUVSLAM_PYTHON", sys.executable),
            cuvslam_modes=tuple(m.strip() for m in g("CUVSLAM_MODES", "stereo,inertial,rgbd").split(",") if m.strip()),
            lingbot_run=Path(g("LINGBOT_RUN", str(cls.lingbot_run))).expanduser(), lingbot_image=g("LINGBOT_IMAGE", cls.lingbot_image),
            lingbot_gpu=int(g("LINGBOT_GPU", "0")),
            nebius_api_key=g("NEBIUS_API_KEY"), brain_base_url=g("BRAIN_BASE_URL", TOKEN_FACTORY), brain_model=g("BRAIN_MODEL", cls.brain_model),
            match_m=float(g("PIPELINE_MATCH_M", "0.7")), map_every=int(g("PIPELINE_MAP_EVERY", "5")), env=e)

    def public(self) -> dict:
        """Loggable view: no keys."""
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()
                if k not in ("env", "eyes2_api_key", "eyes_api_key", "nebius_api_key")} | \
               {"eyes2_api_key": bool(self.eyes2_api_key), "nebius_api_key": bool(self.nebius_api_key)}
