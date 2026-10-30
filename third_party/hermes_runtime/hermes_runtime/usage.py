# Derived from HERMES Agent 0.20.5 (agent/usage_pricing.py).
# Upstream commit: 8c8193dce5faf71d80bf98eb52bd896ee7be8a27
# SPDX-License-Identifier: MIT
# Local modifications: OpenAI-compatible usage only; removed pricing, routing and metadata.
"""Canonical usage accounting without provider pricing or remote metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CanonicalUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    request_count: int = 1

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    def __add__(self, other: CanonicalUsage) -> CanonicalUsage:
        if not isinstance(other, CanonicalUsage):
            return NotImplemented
        return CanonicalUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            request_count=self.request_count + other.request_count,
        )


def _get(value: Any, name: str, default: Any = 0) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def normalize_usage(response_usage: Any) -> CanonicalUsage:
    """Normalize the OpenAI chat-completions usage shape."""

    if not response_usage:
        return CanonicalUsage()
    prompt_total = _count(_get(response_usage, "prompt_tokens", 0))
    output_tokens = _count(_get(response_usage, "completion_tokens", 0))
    details = _get(response_usage, "prompt_tokens_details", None)
    cache_read_tokens = _count(_get(details, "cached_tokens", 0)) if details else 0
    cache_write_tokens = _count(_get(details, "cache_write_tokens", 0)) if details else 0
    reasoning_details = _get(response_usage, "completion_tokens_details", None)
    reasoning_tokens = (
        _count(_get(reasoning_details, "reasoning_tokens", 0))
        if reasoning_details
        else 0
    )
    return CanonicalUsage(
        input_tokens=max(0, prompt_total - cache_read_tokens - cache_write_tokens),
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        reasoning_tokens=reasoning_tokens,
    )
