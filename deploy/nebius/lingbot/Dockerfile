# lingbot-map (https://github.com/robbyant/lingbot-map, Apache-2.0) on NVIDIA's PyTorch container, built for Blackwell by
# default (TORCH_CUDA_ARCH_LIST=12.0). Built by deploy/nebius/lingbot/build.sh; run by run.sh.
FROM nvcr.io/nvidia/pytorch:25.04-py3
ARG TORCH_CUDA_ARCH_LIST=12.0
ENV TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST PIP_NO_CACHE_DIR=1 HF_HOME=/hf_cache
RUN pip freeze | grep -iE "^(torch|torchvision|numpy|triton)==" > /tmp/constraints.txt; cat /tmp/constraints.txt
RUN pip install -c /tmp/constraints.txt einops safetensors huggingface_hub tqdm scipy Pillow opencv-python-headless trimesh
RUN pip install -c /tmp/constraints.txt flashinfer-python || echo "flashinfer failed"
COPY lingbot-map /opt/lingbot-map
RUN cd /opt/lingbot-map && pip install --no-deps -e .
WORKDIR /opt/lingbot-map
