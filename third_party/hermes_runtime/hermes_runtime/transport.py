# Derived from HERMES Agent 0.20.5 (agent/transports/base.py and chat_completions.py).
# Upstream commit: 8c8193dce5faf71d80bf98eb52bd896ee7be8a27
# SPDX-License-Identifier: MIT
# Local modifications: one OpenAI-compatible shape; removed all provider profiles and fallbacks.
"""Transport conversion and response normalization without client discovery."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import NormalizedResponse, ToolCall, Usage, map_finish_reason
from .errors import EmptyStreamError
from .repair import deterministic_call_id, repair_tool_call_arguments, uniquify_tool_call_ids


def _member(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


class ProviderTransport(ABC):
    """Abstract provider data-path contract retained from HERMES."""

    @property
    @abstractmethod
    def api_mode(self) -> str:
        """Return the fixed API mode handled by this transport."""

    @abstractmethod
    def convert_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert normalized messages to the provider wire shape."""

    @abstractmethod
    def convert_tools(
        self,
        tools: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert normalized tool declarations to the provider wire shape."""

    @abstractmethod
    def build_kwargs(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """Build explicit chat-completions keyword arguments."""

    @abstractmethod
    def normalize_response(self, response: Any) -> NormalizedResponse:
        """Normalize one provider response."""

    def validate_response(self, response: Any) -> bool:
        """Return whether a raw response has a non-empty choices collection."""

        choices = _member(response, "choices", None)
        return (
            isinstance(choices, Sequence)
            and not isinstance(choices, str | bytes)
            and bool(choices)
        )


class ChatCompletionsTransport(ProviderTransport):
    """Narrow OpenAI-compatible transport used by the explicit ARK client."""

    @property
    def api_mode(self) -> str:
        return "chat_completions"

    def convert_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return [dict(message) for message in messages]

    def convert_tools(
        self,
        tools: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return [dict(tool) for tool in tools]

    def build_kwargs(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": self.convert_messages(messages),
            **params,
        }
        if tools:
            payload["tools"] = self.convert_tools(tools)
        return payload

    def normalize_response(self, response: Any) -> NormalizedResponse:
        if not self.validate_response(response):
            raise EmptyStreamError("provider returned no choices")
        choices = _member(response, "choices")
        choice = choices[0]
        message = _member(choice, "message")
        if message is None:
            raise EmptyStreamError("provider choice has no message")

        raw_calls = _member(message, "tool_calls", ()) or ()
        calls: list[ToolCall] = []
        for index, raw_call in enumerate(raw_calls):
            function = _member(raw_call, "function", {})
            name = str(_member(function, "name", ""))
            raw_arguments = str(_member(function, "arguments", "{}"))
            call_id = _member(raw_call, "id", None)
            calls.append(
                ToolCall(
                    id=str(call_id)
                    if call_id
                    else deterministic_call_id(name, raw_arguments, index),
                    name=name,
                    arguments=repair_tool_call_arguments(raw_arguments, name),
                )
            )

        raw_usage = _member(response, "usage", None)
        prompt_tokens = int(_member(raw_usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(_member(raw_usage, "completion_tokens", 0) or 0)
        cached_details = _member(raw_usage, "prompt_tokens_details", None)
        cached_tokens = int(_member(cached_details, "cached_tokens", 0) or 0)
        total_tokens = int(
            _member(raw_usage, "total_tokens", prompt_tokens + completion_tokens) or 0
        )
        finish_reason = map_finish_reason(
            _member(choice, "finish_reason", None),
            {
                "content_filter": "content_filter",
                "length": "length",
                "stop": "stop",
                "tool_calls": "tool_calls",
            },
        )
        content = _member(message, "content", None)
        reasoning = _member(message, "reasoning_content", None)
        return NormalizedResponse(
            content=str(content) if content is not None else None,
            tool_calls=uniquify_tool_call_ids(calls),
            finish_reason=finish_reason,
            reasoning=str(reasoning) if reasoning is not None else None,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
            ),
        )
