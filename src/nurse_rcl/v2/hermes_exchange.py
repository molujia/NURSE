"""NURSE-owned one-exchange adapter; deliberately no HTTP, retries, or tool loop."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from hermes_runtime.transport import ChatCompletionsTransport

from nurse_rcl.v2.exchange_json import (
    ExchangeBoundaryError,
    ExchangeLimits,
    decode_json_object,
    encode_json,
    require_limits,
)
from nurse_rcl.v2.hermes_response import ParsedExchangeResponse, parse_response

ADAPTER_VERSION = "nurse.v2.hermes-single-exchange/1"


@dataclass(frozen=True, slots=True)
class PreparedExchange:
    payload_bytes: bytes = field(repr=False)
    limits: ExchangeLimits = field(default_factory=ExchangeLimits)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class ExchangeAttempt:
    """Transport receipt only. A ready response still needs domain schema/guards."""

    status: Literal["response_ready", "failed"]
    request_sha256: str
    exchange_attempted: Literal[True]
    failure_code: str | None
    response: ParsedExchangeResponse | None = field(repr=False)
    raw_response: bytes | None = field(repr=False)
    response_sha256: str | None
    received_bytes: int | None


def _validate_payload(payload: dict[str, Any], *, native_output: bool = False) -> str | None:
    base = {"model", "messages", "max_tokens", "temperature", "n", "stream"}
    if native_output:
        base.remove("max_tokens")
    tool_mode = "tools" in payload
    expected = base | (
        {"tools", "tool_choice", "parallel_tool_calls"} if tool_mode else {"response_format"}
    )
    if set(payload) != expected:
        raise ExchangeBoundaryError("invalid_request")
    if (
        type(payload["model"]) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", payload["model"]) is None
        or (not native_output and (type(payload["max_tokens"]) is not int
                                   or not 1 <= payload["max_tokens"] <= 65536))
        or type(payload["temperature"]) not in (int, float)
        or not 0 <= payload["temperature"] <= 2
        or type(payload["n"]) is not int
        or payload["n"] != 1
        or payload["stream"] is not False
    ):
        raise ExchangeBoundaryError("invalid_request")
    messages = payload["messages"]
    if type(messages) is not list or not messages:
        raise ExchangeBoundaryError("invalid_request")
    for message in messages:
        if (
            type(message) is not dict
            or set(message) != {"role", "content"}
            or message["role"] not in ("system", "user", "assistant")
            or type(message["content"]) is not str
            or not message["content"].strip()
        ):
            raise ExchangeBoundaryError("invalid_request")
    if not tool_mode:
        if payload["response_format"] != {"type": "json_object"}:
            raise ExchangeBoundaryError("invalid_request")
        return None
    tools = payload["tools"]
    if (
        type(tools) is not list
        or len(tools) != 1
        or type(tools[0]) is not dict
        or set(tools[0]) != {"type", "function"}
        or tools[0]["type"] != "function"
    ):
        raise ExchangeBoundaryError("invalid_request")
    function = tools[0]["function"]
    if (
        type(function) is not dict
        or set(function) != {"name", "description", "parameters"}
        or type(function["name"]) is not str
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", function["name"]) is None
        or type(function["description"]) is not str
        or not function["description"].strip()
        or type(function["parameters"]) is not dict
        or function["parameters"].get("type") != "object"
        or payload["parallel_tool_calls"] is not False
        or payload["tool_choice"] != {"type": "function", "function": {"name": function["name"]}}
    ):
        raise ExchangeBoundaryError("invalid_request")
    return str(function["name"])


def prepare_exchange(
    *,
    model: str,
    messages: object,
    max_output_tokens: int | None,
    temperature: float = 0.0,
    output_tool: object = None,
    limits: ExchangeLimits | None = None,
    native_output: bool = False,
) -> PreparedExchange:
    """Build a detached request from exact JSON values; never discover settings."""

    if limits is None:
        limits = ExchangeLimits()
    require_limits(limits)
    limits = ExchangeLimits(limits.max_bytes, limits.max_depth, limits.max_nodes)
    fields: dict[str, object] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_output_tokens,
        "temperature": temperature,
        "n": 1,
        "stream": False,
    }
    if native_output:
        if max_output_tokens is not None:
            raise ExchangeBoundaryError("native_output_requires_omission")
        del fields["max_tokens"]
    if output_tool is None:
        fields["response_format"] = {"type": "json_object"}
    else:
        # Snapshot first, before accessing even a purported dict supplied by a caller.
        tool = decode_json_object(encode_json(output_tool, limits), limits)
        fields["tools"] = [{"type": "function", "function": tool}]
        fields["tool_choice"] = {"type": "function", "function": {"name": tool.get("name")}}
        fields["parallel_tool_calls"] = False
    payload = decode_json_object(encode_json(fields, limits), limits)
    _validate_payload(payload, native_output=native_output)
    params = {
        key: value for key, value in payload.items() if key not in {"model", "messages", "tools"}
    }
    built = ChatCompletionsTransport().build_kwargs(
        payload["model"],
        payload["messages"],
        tools=payload.get("tools"),
        **params,
    )
    return PreparedExchange(encode_json(built, limits), limits)


def execute_once(request: PreparedExchange, complete: Callable[[bytes], bytes]) -> ExchangeAttempt:
    """Invoke once; physical retry policy and persistent accounting belong to NURSE."""

    if type(request) is not PreparedExchange:
        raise ExchangeBoundaryError("invalid_request")
    if not callable(complete):
        raise ExchangeBoundaryError("invalid_transport")
    require_limits(request.limits)
    limits = ExchangeLimits(
        request.limits.max_bytes, request.limits.max_depth, request.limits.max_nodes
    )
    payload_bytes = request.payload_bytes
    payload = decode_json_object(payload_bytes, limits)
    tool_name = _validate_payload(payload)
    if encode_json(payload, limits) != payload_bytes:
        raise ExchangeBoundaryError("noncanonical_request")
    request_hash = hashlib.sha256(payload_bytes).hexdigest()

    def failed(code: str, raw: bytes | None = None, received: int | None = None) -> ExchangeAttempt:
        return ExchangeAttempt(
            "failed",
            request_hash,
            True,
            code,
            None,
            raw,
            hashlib.sha256(raw).hexdigest() if raw is not None else None,
            received,
        )

    try:
        raw = complete(payload_bytes)
    except TimeoutError:
        return failed("timeout")
    except Exception:
        return failed("transport_error")
    if type(raw) is not bytes:
        return failed("response_type")
    if len(raw) > limits.max_bytes:
        return failed("response_limit", received=len(raw))
    try:
        parsed = parse_response(raw, output_tool_name=tool_name, limits=limits)
    except ExchangeBoundaryError as exc:
        return failed(exc.code, raw, len(raw))
    return ExchangeAttempt(
        "response_ready",
        request_hash,
        True,
        None,
        parsed,
        raw,
        hashlib.sha256(raw).hexdigest(),
        len(raw),
    )
