# Derived from HERMES Agent 0.20.5 (agent/conversation_loop.py).
# Upstream commit: 8c8193dce5faf71d80bf98eb52bd896ee7be8a27
# SPDX-License-Identifier: MIT
# Local modifications: synchronous single-model loop; removed TUI, fallback, hooks and tools.
"""A narrow assistant/tool-result loop with injected provider and dispatcher."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import NormalizedResponse, ToolCall
from .tool_messages import make_tool_result_message
from .usage import CanonicalUsage

CompletionFunction = Callable[
    [tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]],
    NormalizedResponse,
]
DispatchFunction = Callable[[str, dict[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class ConversationResult:
    """Terminal text, full message trace, and accumulated raw usage."""

    content: str
    messages: tuple[dict[str, Any], ...]
    usage: CanonicalUsage
    turns: int


def _assistant_message(response: NormalizedResponse) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": response.content,
    }
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments,
                },
            }
            for call in response.tool_calls
        ]
    return message


def _response_usage(response: NormalizedResponse) -> CanonicalUsage:
    if response.usage is None:
        return CanonicalUsage()
    return CanonicalUsage(
        input_tokens=max(0, response.usage.prompt_tokens - response.usage.cached_tokens),
        output_tokens=response.usage.completion_tokens,
        cache_read_tokens=response.usage.cached_tokens,
    )


def _dispatch_tool(call: ToolCall, dispatch: DispatchFunction) -> dict[str, Any]:
    arguments = json.loads(call.arguments)
    if not isinstance(arguments, dict):
        arguments = {}
    result = dispatch(call.name, arguments)
    return make_tool_result_message(
        call.name,
        result,
        call.id or "",
    )


def run_conversation(
    *,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    complete: CompletionFunction,
    dispatch: DispatchFunction,
    max_turns: int = 16,
    return_at_turn_limit: bool = False,
) -> ConversationResult:
    """Run until the model emits no tool calls or the deterministic turn cap is hit."""

    if max_turns < 1:
        raise ValueError("max_turns must be at least one")
    trace = [dict(message) for message in messages]
    schemas = tuple(dict(tool) for tool in tools)
    usage = CanonicalUsage(request_count=0)
    for turn in range(1, max_turns + 1):
        response = complete(tuple(trace), schemas)
        usage += _response_usage(response)
        trace.append(_assistant_message(response))
        if not response.tool_calls:
            return ConversationResult(
                content=response.content or "",
                messages=tuple(trace),
                usage=usage,
                turns=turn,
            )
        trace.extend(_dispatch_tool(call, dispatch) for call in response.tool_calls)
    if return_at_turn_limit:
        # Bounded callers may stop without buying one more model request. The
        # last tool result remains in the trace; this is not a model stop.
        return ConversationResult(content="", messages=tuple(trace), usage=usage, turns=max_turns)
    raise RuntimeError("conversation exceeded max_turns")
