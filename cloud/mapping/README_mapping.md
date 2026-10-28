# Cloud mapping job: video to trajectory, point cloud and top-down map

This is a batch job for one NVIDIA RTX PRO 6000 Blackwell (96 GB, sm_120, x86_64) on Nebius.
It takes greenhouse-robot video (a 175° front fisheye plus 6 side cameras) or drone video and writes:

| file | content |
|---|---|
| `trajectory.json` | per frame: source video, timestamp, image name, 4x4 `T_world_cam` (OpenCV camera axes; world = first camera) |
| `points.ply` | binary PLY, xyz + rgb, confidence-filtered, capped at `--max-points` |
| `topdown.png` | max-height map in the ground plane, with the camera path overlaid (green = start) |
| `occupancy.png` | 255 = obstacle (above floor + 15% of the height range), 128 = observed free, 0 = unknown |
| `summary.json` | frame count, sample fps, runtimes, GPU name, method/model, top-down grid metadata, pinhole K for `--fisheye` |

```bash
python run_job.py --video front.mp4 side1.mp4 ... --out /out --fps 2 --max-frames 600 --method primary [--fisheye]
python run_job.py --video drone.mp4 --out /out --up cam-z     # nadir drone: up = -camera z
python run_job.py --video x.mp4 --out /tmp/o --frames-only    # extraction only, no GPU
```

The frame budget is shared across videos. One sampling rate `min(fps, max_frames / total_duration)` is used for all of them, and a uniform cap is applied after extraction.
With `--fisheye`, ffmpeg `v360=input=fisheye:output=flat` reprojects an equidistant fisheye into a pinhole centre crop (`--crop-fov`, default 100°), and the matching K is written to the summary.
No model below handles 175° fisheye natively, so this crop is required for the front camera.

## Options surveyed (2026-10-28)

| method | type | install | Blackwell / torch | VRAM, speed | license | wide FOV / fisheye |
|---|---|---|---|---|---|---|
| **MapAnything** ([repo](https://github.com/facebookresearch/map-anything)) | feed-forward multi-view, metric scale; also wraps VGGT, π3, MASt3R, VGGT-Ω, AnyCalib | `pip install -e .` (python ≥3.10; deps include `uniception==0.1.7` and `opencv-python-headless==4.10.0.84`) | pure PyTorch, torch unpinned ("install PyTorch for your system"), so cu128 wheels work | `memory_efficient_inference=True` gives "up to 2000 views on 140 GB", so 600 views on 96 GB should fit (UNVERIFIED) | code Apache-2.0; weights `facebook/map-anything` CC-BY-NC-4.0, `facebook/map-anything-apache` Apache-2.0 (both ungated on HF) | pinhole intrinsics, or ray directions as an alternative input; fisheye not mentioned |
| **VGGT-1B** ([repo](https://github.com/facebookresearch/vggt)) | feed-forward, single pass, arbitrary scale | `pip install -e .`; the pyproject has no torch pin (docs suggest torch 2.3.1, which predates sm_120, so we use 2.8.0+cu128 instead) | pure PyTorch | "< 1 s" per scene; a 2026-05-15 fix allows "roughly 2-3x more input frames" in the same memory. Older README figure (H100, recalled, UNVERIFIED): ~21 GB for 100 frames, ~41 GB for 200 frames | code: VGGT License (commercial use allowed except military, since 2025-07-29); `facebook/VGGT-1B` CC-BY-NC-4.0 (ungated); `VGGT-1B-Commercial` gated | OpenCV pinhole; distortion not modelled |
| VGGT-Ω ([repo](https://github.com/facebookresearch/vggt-omega), [site](https://vggt-omega.github.io/)) | successor to VGGT (CVPR 2026), static and dynamic scenes | `pip install -r requirements.txt && pip install -e .` (`torch>=2.6`, `numpy<2`) | pure PyTorch | measured on A100 at 624x416: 13.4 GB for 100 frames, 28.3 GB for 300, 43.2 GB for 500 | FAIR Noncommercial Research License; **HF checkpoint gated with manual review** | not mentioned |
| π3 / π3X ([repo](https://github.com/yyfz/Pi3)) | feed-forward, permutation-equivariant (no reference view); π3X (2025-12) adds approximate metric scale and optional conditioning | `pip install -r requirements.txt`; `python example_mm.py --data_path video.mp4` writes a PLY | pure PyTorch | not stated | code BSD-3; weights CC BY-NC 4.0 | not mentioned |
| VGGT-Long ([repo](https://github.com/DengKaiCQ/VGGT-Long), [paper](https://arxiv.org/abs/2507.16443)) | chunked VGGT plus loop closure and alignment for kilometre-scale sequences | extra loop-closure dependencies | pure PyTorch plus extras | scales to long video by chunking | VGGT License (LICENSE.txt) | not mentioned |
| MASt3R-SLAM ([repo](https://github.com/rmurai0610/MASt3R-SLAM)) | real-time dense SLAM on MASt3R priors, works uncalibrated | conda, **torch 2.5.1** with cu11.8/12.1/12.4, `--recursive` submodules, compiled CUDA extensions (`pip install --no-build-isolation -e .`), checkpoints from naverlabs | the documented torch 2.5.1/cu12.4 has no sm_120. Needs a rebuild on torch ≥2.7 cu128 with `TORCH_CUDA_ARCH_LIST=12.0` (UNVERIFIED) | about 15 FPS on an RTX 4090 (paper) | CC BY-NC-SA 4.0 (plus MASt3R checkpoint terms) | generic central camera when uncalibrated; no fisheye model |
| DROID-SLAM ([repo](https://github.com/princeton-vl/DROID-SLAM)) | learned dense SLAM (BA + optical flow) | `--recursive`, compiles `lietorch` and `droid_backends` CUDA extensions; tested up to torch 2.7 | needs an sm_120 source build of the extensions (UNVERIFIED) | ≥11 GB inference | BSD-3 | **calibration required** (`fx fy cx cy [k1 k2 p1 p2 ...]`); radial-tangential model, not equidistant fisheye |

Blackwell notes: PyTorch cu128 wheels from 2.7 include sm_120. Users on RTX PRO 6000 report needing ≥2.7.1, and Hugging Face ZeroGPU on that card accepts ≥2.8.0 ([PyTorch forum](https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099), [HF forum](https://discuss.huggingface.co/t/nvidia-rtx-pro-6000-instead-of-h200-for-zerogpu/175960)). We pin torch 2.8.0+cu128 on the `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` base image.
Methods with custom CUDA extensions (MASt3R-SLAM, DROID-SLAM, and curope in MASt3R) must be compiled for `12.0`, which is the main risk on a one-day budget.

## Decision

**Primary: MapAnything** (`facebook/map-anything`).
- It is pure PyTorch: no CUDA extensions to compile for sm_120, and torch is unpinned, so it installs on the cu128 wheel.
- It does a single global, memory-efficient pass over hundreds of unordered views. That fits 7 cameras × N frames with no sequential tracking that could lose lock in repetitive greenhouse aisles.
- Its output is roughly metric, so the occupancy and height thresholds mean something physical.
- Weights are ungated. A license-clean variant (`map-anything-apache`) is available for after the competition by changing one constant.

**Fallback: VGGT-1B** (`facebook/VGGT-1B`).
- It has the smallest dependency surface (torch, einops, Pillow, huggingface_hub), it is ungated, and it is the most widely reproduced feed-forward model.
- Its failure modes are independent of MapAnything's (different code base and dependencies).
- It runs in a single pass, capped at `--fallback-max-frames 300` (uniform subsample). Its scale is arbitrary, so the top-down thresholds are relative.

Not chosen for this round:
- VGGT-Ω is the best-measured memory scaling, but its checkpoint is gated behind manual review, which is too risky with a 2-day deadline. Upgrade path: `pip install -e ".[vggt-omega]"` inside MapAnything once access is granted.
- MASt3R-SLAM and DROID-SLAM need sm_120 extension builds. DROID-SLAM also needs calibration.
- VGGT-Long is the natural next step if videos exceed about 600 frames per run.

## Status

- VERIFIED locally (DGX Spark, aarch64, no model run): argument parsing, fps planning, uniform cap, ffmpeg extraction (including the `v360` fisheye path), PLY writer, top-down and occupancy rasterisation on synthetic data. Run `python3 -m pytest -q`.
- UNVERIFIED (no x86 Blackwell here): Docker build, MapAnything and VGGT inference, VRAM for 600 views, the exact prediction-dict keys used in `run_primary` (`camera_poses`, `pts3d`, `conf`, `mask`, `img_no_norm`, taken from the README), and the fisheye `ih_fov`/`iv_fov` mapping for the real 175° lens (check one frame visually).

Sources: repo READMEs and Hugging Face model cards linked above; [VGGT-Ω memory table](https://github.com/facebookresearch/vggt-omega#runtime-and-gpu-memory); [MapAnything pyproject](https://github.com/facebookresearch/map-anything/blob/main/pyproject.toml); [MASt3R-SLAM LICENSE](https://github.com/rmurai0610/MASt3R-SLAM/blob/main/LICENSE.md); [MASt3R-SLAM paper](https://arxiv.org/abs/2412.12392).
