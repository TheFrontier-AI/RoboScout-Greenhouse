# BOOK batch evaluation in Isaac Sim 5.1 (the version the R180 env was tested with), headless, x86_64.
# Build context = a staged directory (see README.md) containing:
#   r1_greenhouse_nav/                      this repo
#   r1_v3/diy_r50/physical_ai_r180/         R180 env + greenhouse stages (zips/handoff copies excluded)
#   r1_v3/diy_r50/physical_ai_{runtime,r167_lidar,r167_imu,r168_camera}/
#   r1_v3/farm_integration/  r1_v3/presentation/   (stages reference ../../../farm_integration, ../../../presentation)
# UNVERIFIED: not built yet. Isaac Sim containers run as uid 1234 and need the EULA/consent env vars.
FROM --platform=linux/amd64 nvcr.io/nvidia/isaac-sim:5.1.0

ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    R180_DIR=/workspace/r1_v3/diy_r50/physical_ai_r180 \
    EVAL_OUT=/results \
    PYTHONUNBUFFERED=1

COPY --chown=1234:1234 r1_v3 /workspace/r1_v3
COPY --chown=1234:1234 r1_greenhouse_nav /workspace/r1_greenhouse_nav
USER root
RUN mkdir -p /results && chown 1234:1234 /results
USER 1234:1234
WORKDIR /workspace/r1_greenhouse_nav

# Args after the image name are passed to run_eval_job.py, e.g. --episodes 50 --seed0 100 --stage photo
ENTRYPOINT ["/isaac-sim/python.sh", "scripts/run_eval_job.py"]
CMD ["--episodes", "20"]
