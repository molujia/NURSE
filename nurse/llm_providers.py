"""Model routing and provider configuration for NURSE LLM calls."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMProviderConfig:
    name: str
    model: str
    env_key: str
    base_url: str = ""
    uses_openai_client: bool = True


OPENAI_MODELS = {"gpt-5-nano", "gpt-5-mini", "gpt-5"}
ZHIPU_MODELS = {"glm-4.5-air", "glm-4.6v"}
VOLCANO_PREFIXES = ("ep-", "ep-m-")

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_VOLCANO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


def _env(name: str) -> str:
    value = os.getenv(name)
    return str(value or "").strip()


def _normalize_model(model: str) -> str:
    return str(model or "").strip().lower()


def provider_for_model(model: str) -> LLMProviderConfig:
    normalized = _normalize_model(model)
    if normalized in OPENAI_MODELS or normalized.startswith("gpt-5"):
        return LLMProviderConfig(
            name="openai",
            model=str(model or "").strip(),
            env_key="OPENAI_API_KEY",
            base_url=_env("NURSE_OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL,
            uses_openai_client=True,
        )
    if normalized in ZHIPU_MODELS or normalized.startswith("glm-"):
        return LLMProviderConfig(
            name="zhipu",
            model=str(model or "").strip(),
            env_key="GLM_KEY",
            base_url=_env("NURSE_ZHIPU_BASE_URL") or DEFAULT_ZHIPU_BASE_URL,
            uses_openai_client=False,
        )
    if normalized.startswith(VOLCANO_PREFIXES):
        return LLMProviderConfig(
            name="volcano",
            model=str(model or "").strip(),
            env_key="VOLCANO_API_KEY",
            base_url=_env("NURSE_VOLCANO_BASE_URL") or DEFAULT_VOLCANO_BASE_URL,
            uses_openai_client=True,
        )
    return LLMProviderConfig(
        name="openai",
        model=str(model or "").strip(),
        env_key="OPENAI_API_KEY",
        base_url=_env("NURSE_OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL,
        uses_openai_client=True,
    )
