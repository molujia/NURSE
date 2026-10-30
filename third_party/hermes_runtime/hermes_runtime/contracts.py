# Derived from HERMES Agent 0.20.5 (agent/transports/types.py).
# Upstream commit: 8c8193dce5faf71d80bf98eb52bd896ee7be8a27
# SPDX-License-Identifier: MIT
# Local modifications: fixed OpenAI-compatible fields, frozen records, no provider metadata.
"""Normalized response and tool-call contracts for the embedded runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A normalized function tool call."""

    id: str | None
    name: str
    arguments: str

    @property
    def type(self) -> str:
        return "function"

    @property
    def function(self) -> ToolCall:
        return self


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage attached to one normalized response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


@dataclass(frozen=True, slots=True)
class NormalizedResponse:
    """The provider-independent response surface used by the agent loop."""

    content: str | None
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str
    reasoning: str | None = None
    usage: Usage | None = None


def build_tool_call(
    call_id: str | None,
    name: str,
    arguments: Any,
) -> ToolCall:
    """Build a tool call and serialize mapping arguments deterministically."""

    serialized = (
        json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        if isinstance(arguments, dict)
        else str(arguments)
    )
    return ToolCall(id=call_id, name=name, arguments=serialized)


def map_finish_reason(reason: str | None, mapping: dict[str, str]) -> str:
    """Map provider finish reasons to the runtime's normalized vocabulary."""

    if reason is None:
        return "stop"
    return mapping.get(reason, "stop")
