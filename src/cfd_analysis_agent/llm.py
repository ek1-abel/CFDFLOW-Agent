"""LLM construction helpers."""

from __future__ import annotations

from hello_agents import HelloAgentsLLM

from .config import RuntimeConfig


def build_llm(config: RuntimeConfig) -> HelloAgentsLLM:
    return HelloAgentsLLM(
        model=config.model_id,
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout,
    )
