# Cosmos Reason 2 "eyes" for a Nebius Serverless AI Endpoint (single GPU, x86_64, RTX PRO 6000 / H100 / L40S).
# Build:  docker build -t <registry>/r1-eyes:cr2 cloud/eyes
# Run:    docker run --gpus all -p 8000:8000 -e HF_TOKEN <registry>/r1-eyes:cr2
# Weights are pulled from Hugging Face at start-up (mount a volume at /root/.cache/huggingface to reuse them).
FROM vllm/vllm-openai:v0.31.0

ENV COSMOS_MODEL=nvidia/Cosmos-Reason2-32B \
    MAX_MODEL_LEN=32768 \
    GPU_MEM_UTIL=0.92 \
    PORT=8000

COPY serve_cosmos_reason.sh /app/serve_cosmos_reason.sh
RUN chmod +x /app/serve_cosmos_reason.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s CMD curl -fs http://localhost:${PORT}/health || exit 1
ENTRYPOINT ["/app/serve_cosmos_reason.sh"]
