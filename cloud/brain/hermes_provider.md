# Hermes Agent → Nemotron on Nebius Token Factory

Key names checked against the comments in `~/.hermes/config.yaml` and the Hermes source in
`~/.hermes/hermes-agent` (commit bf0203a794, 2026-10-22). Do not paste a key into `config.yaml`.

## `~/.hermes/config.yaml` (replace the existing `model:` block's values)

```yaml
model:
  provider: "custom"                                   # any OpenAI-compatible endpoint
  base_url: "https://api.tokenfactory.nebius.com/v1/"
  default: "nvidia/nemotron-3-super-120b-a12b"         # "model:" works as the key name too
  api_key: "${NEBIUS_API_KEY}"                         # expanded from the environment at load time
  # context_length: 262144                             # only if auto-detection via /v1/models fails
```

Other Nemotron ids on Token Factory (see `cloud/README_cloud.md`): `nvidia/Nemotron-3-Ultra-550b-a55b`
(strongest, slower/pricier), `nvidia/Nemotron-3_5-Lightning` (fast 30B-A3B).

## Passing the API key (pick one)

1. `~/.hermes/.env` (Hermes loads it at start-up), one line: `NEBIUS_API_KEY=<your key>`.
2. Export it in the shell before running `hermes`: `export NEBIUS_API_KEY=...`.
3. One-off: `hermes --provider custom -m nvidia/nemotron-3-super-120b-a12b` with `NEBIUS_API_KEY` set.

`api_key: "${NEBIUS_API_KEY}"` is optional: for `provider: custom` Hermes also derives the env var from the
base_url host (`api.tokenfactory.nebius.com` → `NEBIUS_API_KEY`, see `_host_derived_api_key` in
`hermes_cli/runtime_provider.py`). It deliberately does NOT send `OPENAI_API_KEY`/`OPENROUTER_API_KEY` to
this host, so those will not work here.

## Optional: keep OpenRouter as default and add Nemotron as an alias

```yaml
model_aliases:
  nemotron:
    model: "nvidia/nemotron-3-super-120b-a12b"
    provider: custom
    base_url: "https://api.tokenfactory.nebius.com/v1/"
    key_env: NEBIUS_API_KEY
```

Then switch in a session with `/model nemotron`.

## Check

```bash
curl -s https://api.tokenfactory.nebius.com/v1/models -H "Authorization: Bearer $NEBIUS_API_KEY" | grep -i nemotron
hermes -z "Reply with the word ready."
```
