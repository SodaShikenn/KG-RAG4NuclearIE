"""
Unified LLM client supporting OpenAI (cloud) and Llama (local) backends.
"""
from __future__ import annotations

import json
import os
from typing import Any


def _call_openai(
    prompt: str,
    system: str = "",
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    base_url: str | None = None,
) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "local"),
        base_url=base_url,
    )
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def call_llm(
    prompt: str,
    system: str = "",
    provider: str = "openai",
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """Call an LLM and return the text response."""
    if provider == "openai":
        return _call_openai(prompt, system, model, temperature, max_tokens)
    elif provider == "llama":
        # Llama served via OpenAI-compatible local endpoint (e.g. vLLM, Ollama)
        base_url = os.environ.get("LLAMA_API_BASE", "http://localhost:8000/v1")
        return _call_openai(prompt, system, model, temperature, max_tokens, base_url=base_url)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def call_llm_json(
    prompt: str,
    system: str = "",
    provider: str = "openai",
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> Any:
    """Call LLM and parse response as JSON."""
    raw = call_llm(prompt, system, provider, model, temperature, max_tokens)
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    return json.loads(cleaned)
