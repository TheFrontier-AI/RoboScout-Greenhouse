"""Thin client for NVIDIA Nemotron on Nebius Token Factory (OpenAI-compatible chat completions).
Env: NEBIUS_API_KEY (required), BRAIN_BASE_URL, BRAIN_MODEL."""
from __future__ import annotations
import os
from dataclasses import dataclass, field

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"


@dataclass
class Reply:
    content: str
    tool_calls: list[dict] = field(default_factory=list)   # [{"id", "name", "arguments"(str JSON)}]
    reasoning: str | None = None
    raw: object = None


_client = None


def client():
    global _client
    if _client is None:
        from openai import OpenAI
        key = os.environ.get("NEBIUS_API_KEY")
        if not key:
            raise RuntimeError("NEBIUS_API_KEY is not set")
        _client = OpenAI(base_url=os.environ.get("BRAIN_BASE_URL", DEFAULT_BASE_URL), api_key=key, timeout=120.0)
    return _client


def chat(messages: list[dict], tools: list[dict] | None = None, response_format: dict | None = None,
         model: str | None = None, temperature: float = 0.3, max_tokens: int = 4096, **kw) -> Reply:
    """One chat turn. `tools` are OpenAI function specs; `response_format` e.g. {"type": "json_object"}."""
    args = dict(model=model or os.environ.get("BRAIN_MODEL", DEFAULT_MODEL), messages=messages,
                temperature=temperature, max_tokens=max_tokens, **kw)
    if tools:
        args["tools"] = tools
        args.setdefault("tool_choice", "auto")
    if response_format:
        args["response_format"] = response_format
    resp = client().chat.completions.create(**args)
    msg = resp.choices[0].message
    calls = [{"id": c.id, "name": c.function.name, "arguments": c.function.arguments} for c in (msg.tool_calls or [])]
    return Reply(content=msg.content or "", tool_calls=calls,
                 reasoning=getattr(msg, "reasoning_content", None), raw=resp)


if __name__ == "__main__":
    print(chat([{"role": "user", "content": "Reply with the word ready."}]).content)
