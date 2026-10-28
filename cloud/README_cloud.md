# R1 cloud: eyes (Cosmos Reason) + brain (Nemotron) on Nebius

Researched 2026-10-28. "Verified" means read on the cited page that day; nothing here was tested against a live
Nebius account (no `NEBIUS_API_KEY` was available), so model ids must still be confirmed with `GET /v1/models`.
Items marked **UNVERIFIED** come from third-party pages or are inferred.

```
R1 (Isaac Sim) --frames+meta--> eyes: Cosmos-Reason2 on vLLM (Nebius Serverless Endpoint, RTX PRO 6000)
                                  |  Finding JSON
                                  v
                               brain: Nemotron 3 on Nebius Token Factory (Hermes Agent / cloud.brain.token_factory)
```

## 1. Nebius Token Factory (brain)

| Fact | Value | Source |
|---|---|---|
| OpenAI-compatible base URL | `https://api.tokenfactory.nebius.com/v1/` | https://docs.tokenfactory.nebius.com/quickstart.md |
| Auth | `Authorization: Bearer <key>`; docs use env var **`NEBIUS_API_KEY`** | same; https://docs.tokenfactory.nebius.com/api-reference/examples/list-of-models.md |
| List models | `GET /v1/models` (add `?verbose=true` for details) | list-of-models page above |
| Tool calling | standard OpenAI `tools` / `tool_choice` (`auto` default, or force a function); per-model support not listed, check Playground | https://docs.tokenfactory.nebius.com/ai-models-inference/function-calling.md |
| JSON output | `response_format` `{"type":"json_object"}` or `{"type":"json_schema","json_schema":{...}}`; per-model "JSON mode" tag on model cards | https://docs.tokenfactory.nebius.com/ai-models-inference/json.md |
| Vision input | `image_url` content parts (URL or base64) | https://docs.tokenfactory.nebius.com/api-reference/examples/vision-capabilities.md |

### Nemotron model ids on Token Factory

| Model | Id used in API calls | Context | Source / status |
|---|---|---|---|
| Nemotron 3 Super 120B-A12B | `nvidia/nemotron-3-super-120b-a12b` | 256K | Nebius page code sample https://nebius.com/services/token-factory/nemotron and Nebius cookbook https://github.com/nebius/token-factory-cookbook/blob/main/models/nemotron/README.md (**default brain model**) |
| Nemotron 3 Ultra 550B-A55B | `nvidia/Nemotron-3-Ultra-550b-a55b` | 1M | Nebius cookbook (above) |
| Nemotron 3.5 Lightning 30B-A3B | `nvidia/Nemotron-3_5-Lightning` (underscore!) | 1M | Nebius cookbook (above) |
| Nemotron 3 Nano 30B-A3B | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | 262K | **UNVERIFIED** (third-party: https://www.getmaxim.ai/bifrost/model-library/compare/nebius/nvidia-nemotron-3-nano-30b-a3b.md, https://www.edenai.co/models/nemotron-3-nano-30b-a3b); Nebius page lists "Nemotron 3 Nano 30b" without an id |
| Nemotron 3 Nano Omni | `nvidia/Nemotron-3-Nano-Omni` | ? | **UNVERIFIED**; listed by name on the Nebius Nemotron page; one public project reports 404 on its account |
| Nemotron 3.5 Super VL | `nvidia/Nemotron-3.5-Super-VL` | 1M | **UNVERIFIED** (https://mastra.ai/models/providers/nebius); would be a hosted vision fallback for the eyes |

Tool calling / JSON: Nebius docs do not state per-model support. Nemotron 3 Nano is listed with "Function Calling,
Reasoning" by a third party (Maxim, above). Nemotron 3 Super/Ultra are NVIDIA's agentic models and are expected to
support tools + JSON (**UNVERIFIED on Token Factory**; check the model card tags in the console/Playground).
Note: ids are case-sensitive and inconsistent across the family; always copy from `/v1/models`.

## 2. Cosmos Reason (eyes)

| Fact | Value | Source |
|---|---|---|
| Latest Reason checkpoint | **`nvidia/Cosmos-Reason2-32B`**, released 2026-04-29, post-trained from Qwen3-VL-32B-Instruct, ~33B params BF16 (~66 GB), up to 256K input context, not gated | https://huggingface.co/nvidia/Cosmos-Reason2-32B, https://github.com/nvidia-cosmos/cosmos-reason2 |
| Smaller | `nvidia/Cosmos-Reason2-8B` (8.77B, updated 2026-03-10, **gated**: accept terms, needs `HF_TOKEN`, min 32 GB GPU), `nvidia/Cosmos-Reason2-2B` | https://huggingface.co/nvidia/Cosmos-Reason2-8B |
| License | NVIDIA Open Model License (commercial use OK), additional Apache-2.0 terms | model cards above |
| Hardware | tested H100/A100, supported Blackwell + Hopper, BF16 only | model cards |
| vLLM | repo recommends `vllm>=0.11.0`; we pin image `vllm/vllm-openai:v0.31.0` (v0.32.0 is newest, 2026-10-18) | https://github.com/nvidia-cosmos/cosmos-reason2, https://github.com/vllm-project/vllm/releases |
| Official serve flags | `vllm serve nvidia/Cosmos-Reason2-2B --max-model-len 16384 --media-io-kwargs '{"video": {"num_frames": -1}}' --reasoning-parser qwen3 --port 8000` (8192-16384 suggested to avoid OOM on small GPUs) | cosmos-reason2 README |
| Prompting | system "You are a helpful assistant."; optional `<think>...</think>` then answer; give >= 4096 output tokens when reasoning; video at fps=4 | model cards |
| Image input | OpenAI chat `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}` (vLLM multimodal; not shown in Cosmos docs) | https://docs.vllm.ai/en/stable/features/multimodal_inputs/ |
| On Token Factory? | **Not found**: no Cosmos model on Nebius Nemotron page, cookbook or third-party catalogs → self-host on Serverless Endpoint. (UNVERIFIED for dedicated-endpoint templates / custom weights: https://docs.tokenfactory.nebius.com/ai-models-inference/dedicated-endpoints/custom-weights.md) | |
| Successor | Cosmos 3 (`nvidia/Cosmos3-Nano` 16B fits RTX PRO 6000, `Cosmos3-Super` 64B, `Cosmos3-Edge` 4B; OpenMDW-1.1 license) has a "Reasoner" tower; Cosmos-Reason2 repo is in maintenance mode. No standalone reasoner checkpoint/vLLM recipe confirmed → **UNVERIFIED** as a drop-in | https://github.com/NVIDIA/Cosmos |

Our choices (`eyes/serve_cosmos_reason.sh`): 32B BF16 on one RTX PRO 6000 (96 GB), `--gpu-memory-utilization 0.92`,
`--max-model-len 32768` (a full-resolution phone photo can cost >10k vision tokens on a Qwen3-VL backbone, plus the
reasoning budget; **UNVERIFIED** token count, lower to 16384 if sending ~1 MP sim frames), one image per prompt,
`--reasoning-parser qwen3` so `<think>` goes to `reasoning_content`. Switch to 8B with `COSMOS_MODEL=nvidia/Cosmos-Reason2-8B`
when throughput (7 cameras every 0.5 m) matters more than accuracy.

## 3. Nebius Serverless AI Endpoints (custom vLLM container)

Serverless AI (Endpoints + Jobs) is in public preview via console and CLI (https://nebius.com/blog/posts/ai-cloud-aether-3-5).
An Endpoint is a container on a Compute VM; billed as VM time; runs until stopped (no scale-to-zero in the vLLM recipe).

CLI (from https://docs.vllm.ai/en/latest/deployment/frameworks/nebius/ and https://docs.nebius.com/serverless/endpoints/manage.md):

```bash
# Endpoint bearer token (not the Token Factory key). Image: built from cloud/eyes/Dockerfile, or use
# vllm/vllm-openai:v0.31.0 with --container-command vllm --args "serve nvidia/Cosmos-Reason2-32B ...".
# HF_TOKEN only needed for the gated 8B/2B (prefer --env-secret; exact --env syntax UNVERIFIED).
export AUTH_TOKEN="$(openssl rand -hex 32)"
nebius ai endpoint create \
  --parent-id "$PROJECT_ID" --subnet-id "$SUBNET_ID" --name r1-eyes \
  --image <registry>/r1-eyes:cr2 \
  --env HF_TOKEN="$HF_TOKEN" \
  --platform gpu-rtx6000 --preset 1gpu-24vcpu-218gb \
  --container-port 8000/http \
  --auth token --token "$AUTH_TOKEN" \
  --disk-size 250Gi --shm-size 16Gi --public=false --preemptible=false
ENDPOINT_URL=$(nebius ai endpoint get <id> --format json | jq -r '.status.public_endpoints[] | select(startswith("https://"))' | head -1)
curl -fs "$ENDPOINT_URL/health"; curl -s "$ENDPOINT_URL/v1/models" -H "Authorization: Bearer $AUTH_TOKEN"
export EYES_BASE_URL="$ENDPOINT_URL/v1" EYES_API_KEY="$AUTH_TOKEN" EYES_MODEL=nvidia/Cosmos-Reason2-32B
```

Console: Serverless AI → Endpoints → Create endpoint → Custom configuration → image, registry creds, port 8000/http,
entrypoint/args, env vars/secrets, token auth → GPU, platform, preset → disk/volumes/network → Create (~5 min).
Other flags available: `--registry-username/--registry-password/--registry-secret`, `--token-secret`, `--volume`,
`--inject-file`, `--ssh-key`, `--preemptible`. Default disk 250Gi, shm 16Gi. The vLLM guide was tested with
Nebius CLI 0.12.265 on `--platform gpu-l40s-a --preset 1gpu-8vcpu-32gb`.

RTX PRO 6000: Compute platforms are `gpu-rtx6000` (us-central1) and `gpu-rtx6000-a` (uk-south2, eu-south1), presets
`1gpu-24vcpu-218gb` / `8gpu-192vcpu-1744gb` (https://docs.nebius.com/compute/virtual-machines/types). Whether these
platforms are selectable for **Serverless Endpoints** is **UNVERIFIED** (docs only show `gpu-h100-sxm` and `gpu-l40s-a`
examples; the Aether 3.5 post says to contact the account manager for availability). Fallback: `gpu-h100-sxm`
`1gpu-16vcpu-200gb` (80 GB fits 32B BF16 with `--max-model-len 16384`), or run the same container on a plain
RTX PRO 6000 Compute VM (`deploy/nebius/`).

## 4. Usage

```bash
# eyes (local GPU or endpoint)
bash cloud/eyes/serve_cosmos_reason.sh                   # port 8000
export EYES_BASE_URL=http://localhost:8000/v1 EYES_MODEL=nvidia/Cosmos-Reason2-32B EYES_API_KEY=EMPTY
python -m cloud.eyes.inspect frame.jpg --camera mast_left_mid --aisle tunnel-09/aisle-02 --metre 15
python -m cloud.eyes.batch_real_photos                   # refs/02_tomato_greenhouses -> cloud/eyes/out/real_photos.jsonl
# EYES_REASONING=1 asks for <think> first; EYES_STRUCTURED=1 sends a json_schema response_format (vLLM guided decoding)

# brain
export NEBIUS_API_KEY=...                                # never commit it
python -m cloud.brain.token_factory                      # prints "ready"
```

Finding schema (mast cameras): `{"status": "healthy|suspect|disease|not_plant", "issues": [{"label", "confidence",
"bbox_norm": [x0,y0,x1,y1] | null, "evidence"}], "summary"}`. The front camera adds `nav: {aisle_clear, obstacle,
appears_centered}` with `status: "not_plant"`. Parser strips `<think>`/`<answer>`/fences, takes the first balanced
JSON object, normalises 0..1000 boxes and percent confidences, and the client retries once with a correction message.
Do not run `cloud/eyes/inspect.py` as a script path (it would shadow stdlib `inspect`); use `python -m`.
