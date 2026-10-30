"""Strict wire checks before lossy HERMES normalization; no semantic acceptance."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from hermes_runtime.contracts import NormalizedResponse
from hermes_runtime.transport import ChatCompletionsTransport
from hermes_runtime.usage import CanonicalUsage, normalize_usage

from nurse_rcl.v2.exchange_json import (
    ExchangeBoundaryError,
    ExchangeLimits,
    decode_json_object,
    encode_json,
)


@dataclass(frozen=True, slots=True)
class ParsedExchangeResponse:
    """Private in-memory receipt, not a validated domain or public audit artifact."""

    output_json: bytes = field(repr=False)
    normalized: NormalizedResponse = field(repr=False)
    usage: CanonicalUsage | None
    usage_status: Literal["reported", "missing", "invalid"]
    raw_usage_json: bytes | None = field(repr=False)


def _valid_usage(value: Any) -> bool:
    if type(value) is not dict:
        return False
    for key in ("prompt_tokens", "completion_tokens"):
        if type(value.get(key)) is not int or value[key] < 0:
            return False
    total = value["prompt_tokens"] + value["completion_tokens"]
    if "total_tokens" in value and (
        type(value["total_tokens"]) is not int or value["total_tokens"] != total
    ):
        return False
    for key, buckets, ceiling in (
        ("prompt_tokens_details", ("cached_tokens", "cache_write_tokens"), value["prompt_tokens"]),
        ("completion_tokens_details", ("reasoning_tokens",), value["completion_tokens"]),
    ):
        details = value.get(key)
        if details is None:
            continue
        if type(details) is not dict:
            return False
        if any(
            type(details.get(bucket, 0)) is not int or details.get(bucket, 0) < 0
            for bucket in buckets
        ):
            return False
        if sum(details.get(bucket, 0) for bucket in buckets) > ceiling:
            return False
    return True


def parse_response(
    raw: bytes,
    *,
    output_tool_name: str | None,
    limits: ExchangeLimits,
) -> ParsedExchangeResponse:
    data = decode_json_object(raw, limits)
    choices = data.get("choices")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise ExchangeBoundaryError("response_shape")
    choice = choices[0]
    if type(choice.get("index")) is not int or choice["index"] != 0:
        raise ExchangeBoundaryError("response_shape")
    message = choice.get("message")
    if type(message) is not dict or message.get("role") != "assistant":
        raise ExchangeBoundaryError("response_shape")
    if message.get("refusal") not in (None, ""):
        raise ExchangeBoundaryError("provider_refusal")
    if message.get("function_call") is not None:
        raise ExchangeBoundaryError("unrequested_tool")
    expected_finish = "stop" if output_tool_name is None else "tool_calls"
    if choice.get("finish_reason") != expected_finish:
        raise ExchangeBoundaryError("incomplete_response")
    for key in ("content", "reasoning_content"):
        if message.get(key) is not None and type(message[key]) is not str:
            raise ExchangeBoundaryError("response_shape")
    if output_tool_name is None:
        if message.get("tool_calls") not in (None, []):
            raise ExchangeBoundaryError("unrequested_tool")
        text = message.get("content")
        if type(text) is not str:
            raise ExchangeBoundaryError("response_shape")
    else:
        calls = message.get("tool_calls")
        if type(calls) is not list or len(calls) != 1 or type(calls[0]) is not dict:
            raise ExchangeBoundaryError("response_shape")
        call = calls[0]
        function = call.get("function")
        if call.get("type") != "function" or type(function) is not dict:
            raise ExchangeBoundaryError("response_shape")
        if function.get("name") != output_tool_name:
            raise ExchangeBoundaryError("unrequested_tool")
        if call.get("id") is not None and type(call["id"]) is not str:
            raise ExchangeBoundaryError("response_shape")
        text = function.get("arguments")
        if type(text) is not str:
            raise ExchangeBoundaryError("response_shape")
    output = decode_json_object(text.encode("utf-8"), limits)
    raw_usage = data.get("usage")
    status: Literal["reported", "missing", "invalid"] = (
        "missing" if raw_usage is None else "reported" if _valid_usage(raw_usage) else "invalid"
    )
    normalized_input = dict(data)
    normalized_input["usage"] = raw_usage if status == "reported" else None
    normalized = ChatCompletionsTransport().normalize_response(normalized_input)
    if status != "reported":
        normalized = replace(normalized, usage=None)
    return ParsedExchangeResponse(
        output_json=encode_json(output, limits),
        normalized=normalized,
        usage=normalize_usage(raw_usage) if status == "reported" else None,
        usage_status=status,
        raw_usage_json=encode_json(raw_usage, limits) if raw_usage is not None else None,
    )
