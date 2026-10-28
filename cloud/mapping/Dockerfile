# Cloud mapping job image: x86_64, CUDA 12.8 (first CUDA with Blackwell sm_120), PyTorch 2.8.0+cu128.
# Build ON an x86_64 host (or `docker buildx build --platform linux/amd64`); NOT buildable natively on the DGX Spark (aarch64).
#   docker buildx build --platform linux/amd64 -t <registry>/r1-mapping:0.1 cloud/mapping
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG TORCH=2.8.0
ARG TORCHVISION=0.23.0
# Repo commits = main HEAD on 2026-10-28 (recorded in /opt/versions.txt at build time).
ARG MAPANYTHING_REF=3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9
ARG VGGT_REF=a288dd0f14786c93483e45524328726ab7b1b4ce
ARG PREFETCH_WEIGHTS=1

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/hf TORCH_HOME=/opt/torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev git ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv $VIRTUAL_ENV && pip install --no-cache-dir -U pip==25.2 wheel

# cu128 wheels include sm_120 kernels (check: torch.cuda.get_arch_list()).
RUN pip install --no-cache-dir torch==${TORCH} torchvision==${TORCHVISION} \
        --index-url https://download.pytorch.org/whl/cu128

# Primary: MapAnything (Apache-2.0 code; weights facebook/map-anything CC-BY-NC-4.0).
RUN git clone https://github.com/facebookresearch/map-anything /opt/map-anything \
    && git -C /opt/map-anything checkout ${MAPANYTHING_REF} \
    && pip install --no-cache-dir -e /opt/map-anything \
    && echo "map-anything $(git -C /opt/map-anything rev-parse HEAD)" >> /opt/versions.txt

# Fallback: VGGT (pure PyTorch; weights facebook/VGGT-1B non-commercial). --no-deps: its pyproject asks for
# numpy<2 and opencv-python, which would clash with MapAnything's opencv-python-headless; einops is the only extra.
RUN git clone https://github.com/facebookresearch/vggt /opt/vggt \
    && git -C /opt/vggt checkout ${VGGT_REF} \
    && pip install --no-cache-dir einops==0.8.1 && pip install --no-cache-dir --no-deps -e /opt/vggt \
    && echo "vggt $(git -C /opt/vggt rev-parse HEAD)" >> /opt/versions.txt

# boto3 for s3:// inputs when no bucket volume is mounted; fail the build if a dependency replaced the cu128 torch.
RUN pip install --no-cache-dir boto3==1.40.40 \
    && python -c "import torch; assert torch.__version__.startswith('${TORCH}+cu128'), torch.__version__" \
    && python -c "import torch; print('torch', torch.__version__, torch.version.cuda)" >> /opt/versions.txt

# Bake weights into the image (instantiate on CPU so anything fetched at init, e.g. torch.hub encoders, is cached too).
RUN if [ "$PREFETCH_WEIGHTS" = "1" ]; then \
      python -c "from mapanything.models import MapAnything; MapAnything.from_pretrained('facebook/map-anything')" \
      && python -c "from vggt.models.vggt import VGGT; VGGT.from_pretrained('facebook/VGGT-1B')"; fi

WORKDIR /app
COPY run_job.py /app/run_job.py
ENTRYPOINT ["python", "/app/run_job.py"]
