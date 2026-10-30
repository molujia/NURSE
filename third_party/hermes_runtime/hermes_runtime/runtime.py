# Project-owned assembly over the attributed HERMES-derived components.
# No upstream source file is copied into this module.
# SPDX-License-Identifier: MIT
"""Allowlisted mockable assembly for the embedded HERMES-derived runtime."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import NormalizedResponse, ToolCall
from .tool_messages import make_tool_result_message
from .transport import ChatCompletionsTransport
from .usage import CanonicalUsage

CompletionFunction = Callable[[dict[str, Any]], Any]
ToolHandler = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """One explicitly allowed function tool and its local handler."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: ToolHandler

    def definition(self) -> dict[str, Any]:
        """Return the OpenAI function-tool declaration sent to the model."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": copy.deepcopy(dict(self.parameters)),
            },
        }


@dataclass(frozen=True, slots=True)
class RuntimeAuditEvent:
    """Stable audit event for one model-requested tool call."""

    code: str
    tool_name: str
    tool_call_id: str


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Terminal response, transcript, usage, and tool-authorization audit."""

    content: str
    messages: tuple[dict[str, Any], ...]
    usage: CanonicalUsage
    turns: int
    audit_events: tuple[RuntimeAuditEvent, ...]


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


def _arguments(call: ToolCall) -> dict[str, Any]:
    value = json.loads(call.arguments)
    return value if isinstance(value, dict) else {}


def _rejection_content(tool_name: str) -> str:
    return json.dumps(
        {
            "error": {
                "code": "tool_not_allowed",
                "message": f"Tool '{tool_name}' is not registered.",
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class EmbeddedAgentRuntime:
    """Compose a fixed transport, injected provider call, and closed tool registry."""

    def __init__(
        self,
        *,
        model: str,
        complete: CompletionFunction,
        tools: Sequence[RegisteredTool],
        max_turns: int = 16,
    ) -> None:
        if not model:
            raise ValueError("model must be explicit")
        if max_turns < 1:
            raise ValueError("max_turns must be at least one")
        registry = {tool.name: tool for tool in tools}
        if len(registry) != len(tools):
            raise ValueError("tool names must be unique")
        self._model = model
        self._complete = complete
        self._tools = registry
        self._max_turns = max_turns
        self._transport = ChatCompletionsTransport()

    def run(self, *, messages: Sequence[Mapping[str, Any]]) -> AgentRunResult:
        """Run the injected model until it returns terminal text."""

        trace = [dict(message) for message in messages]
        definitions = [tool.definition() for tool in self._tools.values()]
        usage = CanonicalUsage(request_count=0)
        audit: list[RuntimeAuditEvent] = []

        for turn in range(1, self._max_turns + 1):
            request = self._transport.build_kwargs(
                self._model,
                trace,
                definitions,
            )
            response = self._transport.normalize_response(self._complete(request))
            usage += _response_usage(response)
            trace.append(_assistant_message(response))
            if not response.tool_calls:
                return AgentRunResult(
                    content=response.content or "",
                    messages=tuple(trace),
                    usage=usage,
                    turns=turn,
                    audit_events=tuple(audit),
                )

            for call in response.tool_calls:
                call_id = call.id or ""
                registered = self._tools.get(call.name)
                if registered is None:
                    content: Any = _rejection_content(call.name)
                    code = "tool_not_allowed"
                else:
                    content = registered.handler(_arguments(call))
                    code = "tool_executed"
                trace.append(
                    make_tool_result_message(
                        call.name,
                        content,
                        call_id,
                    )
                )
                audit.append(
                    RuntimeAuditEvent(
                        code=code,
                        tool_name=call.name,
                        tool_call_id=call_id,
                    )
                )

        raise RuntimeError("conversation exceeded max_turns")
