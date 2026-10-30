"""Explicit typed-profile durable receipt, not domain or semantic admission.

No key loading or default activation. The unchanged original client and retry
own sending/attempts; this injected adapter records body bytes, never headers.
Owner/source admission and any paid authorization remain separate boundaries.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from hermes_runtime.contracts import NormalizedResponse
from hermes_runtime.usage import normalize_usage

from nurse_rcl.agents.ark import (
    ARK_CHAT_COMPLETIONS_URL,
    ArkResponseResourceError,
    _bounded_response,
)
from nurse_rcl.agents.retry import AttemptRecord, AttemptUsage
from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object, encode_json
from nurse_rcl.v2.hermes_response import ParsedExchangeResponse, _valid_usage
from nurse_rcl.v2.live.ops_empirical_policy import RESPONSE_LIMITS
from nurse_rcl.v2.live.ops_usage import usage_parts

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nurse_rcl.v2.live.ark_bridge import LiveRequestBinding

PROTOCOL = "nurse.v2.sre-typed-report-exchange/1"


class TypedReportReceiptError(RuntimeError):
    """A retained outer receipt is unusable; never a retryable parsing error."""


class TypedReportAccountingError(RuntimeError):
    """Persisted incomplete/inconsistent accounting stops all further sends."""


def _parts(raw: bytes | None) -> dict[str, Any]:
    try:
        value = None if raw is None else decode_json_object(raw, RESPONSE_LIMITS).get("usage")
    except (ExchangeBoundaryError, UnicodeError):
        value = None
    parts = usage_parts(value)
    # Retain independently reported counts, but refuse invalid detail buckets too.
    parts["complete"] = not parts["usage_issues"] and _valid_usage(value)
    return parts


def typed_attempt_usage(raw: bytes | None) -> AttemptUsage:
    parts = _parts(raw)
    counts = tuple(
        parts[key] for key in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens")
    )
    if not parts["complete"] and all(value is not None for value in counts):
        # AttemptUsage has no 'inconsistent' status. Keep raw usage authoritative;
        # an invalid total is unknown, never silently reclassified as reported.
        counts = (counts[0], counts[1], None, counts[3])
    known = sum(value is not None for value in counts)
    return AttemptUsage(
        "reported" if known == 4 else "partial" if known else "unavailable", *counts
    )


def require_typed_accounting(raw: bytes | None) -> None:
    if not _parts(raw)["complete"]:
        raise TypedReportAccountingError("typed_accounting_unknown_or_inconsistent")


def parse_typed_receipt(raw: bytes) -> ParsedExchangeResponse:
    """Validate only complete outer receipt; preserve inner text, even bad JSON."""
    try:
        data = decode_json_object(raw, RESPONSE_LIMITS)
        choices = data.get("choices")
        if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
            raise TypedReportReceiptError("typed_outer_shape")
        choice = choices[0]
        message = choice.get("message")
        if (
            type(choice.get("index")) is not int
            or choice["index"] != 0
            or choice.get("finish_reason") != "stop"
            or type(message) is not dict
            or message.get("role") != "assistant"
            or message.get("refusal") not in (None, "")
            or message.get("function_call") is not None
            or message.get("tool_calls") not in (None, [])
            or type(message.get("content")) is not str
            or (
                message.get("reasoning_content") is not None
                and type(message["reasoning_content"]) is not str
            )
        ):
            raise TypedReportReceiptError("typed_outer_incomplete_or_unrequested")
        text = message["content"]
        native = text.encode("utf-8")
        if len(native) > RESPONSE_LIMITS.max_bytes:
            raise TypedReportReceiptError("typed_native_resource_limit")
        usage = data.get("usage")
        complete = _parts(raw)["complete"]
        return ParsedExchangeResponse(
            output_json=native,
            normalized=NormalizedResponse(content=text, tool_calls=(), finish_reason="stop"),
            usage=normalize_usage(usage) if complete else None,
            usage_status="reported" if complete else "missing" if usage is None else "invalid",
            raw_usage_json=None if usage is None else encode_json(usage, RESPONSE_LIMITS),
        )
    except (ExchangeBoundaryError, UnicodeError):
        raise TypedReportReceiptError("typed_outer_encoding_or_resource") from None


def interrupted_typed_record(number: int, raw: bytes | None, status: int | None) -> AttemptRecord:
    valid = False
    if raw is not None and status is None:
        try:
            parse_typed_receipt(raw)
            require_typed_accounting(raw)
            valid = True
        except (TypedReportReceiptError, TypedReportAccountingError):
            pass
    return AttemptRecord(
        attempt_number=number,
        category="success" if valid else "transport",
        outcome="success" if valid else "fatal",
        latency_seconds=0.0,
        usage=typed_attempt_usage(raw),
        status_code=None if valid else status,
        error_type=None if valid else "TypedInterruptedReceipt",
    )


def _durable(path: Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    # POSIX directory durability; Windows pure tests do not emulate a file lock.
    if os.name == "posix":
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class _CaptureStream(httpx.SyncByteStream):
    def __init__(self, response: httpx.Response, path: Path):
        self.response, self.path = response, path

    def __iter__(self) -> Iterator[bytes]:
        count = 0
        with self.path.open("xb") as handle:
            try:
                chunks = (
                    (self.response.content,)
                    if self.response.is_stream_consumed
                    else self.response.iter_raw()
                )
                for chunk in chunks:
                    remaining = RESPONSE_LIMITS.max_bytes - count
                    handle.write(chunk[:remaining])
                    count += len(chunk)
                    if count > RESPONSE_LIMITS.max_bytes:
                        raise ArkResponseResourceError("typed_http_resource_limit")
                    yield chunk
            finally:
                handle.flush()
                os.fsync(handle.fileno())

    def close(self) -> None:
        self.response.close()


class TypedReportTransport(httpx.BaseTransport):
    """One new capture directory, immutable wire, at most five original attempts."""

    def __init__(
        self,
        *,
        root: Path,
        expected_wire: bytes,
        code_sha256: str,
        execution_sha256: str,
        transport: httpx.BaseTransport | None = None,
        protocol: str = PROTOCOL,
    ):
        if protocol not in (PROTOCOL, "nurse.v2.sre-gplus-report-exchange/1"):
            raise ValueError("unregistered typed receiving protocol")
        self.protocol = protocol
        for pin in (code_sha256, execution_sha256):
            require_pin(pin)
        decode_json_object(expected_wire, RESPONSE_LIMITS)
        self.root, self.expected_wire = Path(root), expected_wire
        self.code, self.execution = code_sha256, execution_sha256
        self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        _durable(
            self.root / "identity.json",
            canonical_json_bytes(
                dict(
                    protocol=self.protocol,
                    code_sha256=self.code,
                    execution_sha256=self.execution,
                    request_sha256=sha256_hex(expected_wire),
                )
            ),
        )
        self.transport = transport if transport is not None else httpx.HTTPTransport()
        self.attempts = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if (
            request.method != "POST"
            or str(request.url) != ARK_CHAT_COMPLETIONS_URL
            or request.read() != self.expected_wire
        ):
            raise ValueError("typed actual wire or destination differs")
        self.attempts += 1
        if self.attempts > 5:
            raise ValueError("typed original five-attempt capacity exceeded")
        folder = self.root / f"attempt-{self.attempts}"
        folder.mkdir(mode=0o700, exist_ok=False)
        _durable(folder / "request.bin", request.content)
        status = None
        try:
            response = self.transport.handle_request(request)
            status = response.status_code
            original = response
            captured = _CaptureStream(original, folder / "http-body.bin")
            stream = httpx.Response(
                status, headers=response.headers, stream=captured, request=request
            )
            try:
                decoded = _bounded_response(stream, RESPONSE_LIMITS.max_bytes)
            finally:
                stream.close()
            _durable(folder / "decoded-body.bin", decoded.content)
            native = None
            try:
                value = decode_json_object(decoded.content, RESPONSE_LIMITS)
                choices = value.get("choices")
                if type(choices) is list and len(choices) == 1 and type(choices[0]) is dict:
                    message = choices[0].get("message")
                    if type(message) is dict and type(message.get("content")) is str:
                        native = message["content"].encode("utf-8")
                        _durable(folder / "native.bin", native)
            except (ExchangeBoundaryError, UnicodeError):
                pass  # Exact HTTP remains even when outer JSON/native is unusable.
            _durable(
                folder / "receive.json",
                canonical_json_bytes(
                    dict(
                        status_code=status,
                        request_sha256=sha256_hex(request.content),
                        http_body_sha256=sha256_hex((folder / "http-body.bin").read_bytes()),
                        decoded_body_sha256=sha256_hex(decoded.content),
                        native_sha256=None if native is None else sha256_hex(native),
                        semantic_certified=False,
                    )
                ),
            )
            return decoded
        except Exception as error:
            # A static error name only. Never exception text/headers/credentials.
            _durable(
                folder / "interruption.json",
                canonical_json_bytes(
                    dict(
                        status_code=status,
                        error_type=type(error).__name__,
                        accounting="unknown_until_original_ledger_record",
                    )
                ),
            )
            raise

    def close(self) -> None:
        self.transport.close()


def typed_delivery_origin(
    *,
    capture_root: Path,
    receipt: dict[str, Any],
    raw: bytes,
    binding: LiveRequestBinding,
    code_sha256: str,
    execution_sha256: str,
    protocol: str = PROTOCOL,
) -> dict[str, Any]:
    """Recompute actual-byte links; receipt is from the sole original ledger."""
    identity = decode_json_object((capture_root / "identity.json").read_bytes(), RESPONSE_LIMITS)
    if identity != dict(
        protocol=protocol,
        code_sha256=code_sha256,
        execution_sha256=execution_sha256,
        request_sha256=binding.request_sha256,
    ):
        raise ValueError("typed origin execution differs")
    ledger = receipt["ledger"]
    if ledger["binding"] != binding.model_dump(mode="json"):
        raise ValueError("typed origin ledger binding differs")
    attempts = ledger["attempts"]
    if not attempts or attempts[-1]["record"]["outcome"] != "success":
        raise ValueError("typed origin requires complete received ledger slot")
    number = attempts[-1]["number"]
    folder = capture_root / f"attempt-{number}"
    proof = decode_json_object((folder / "receive.json").read_bytes(), RESPONSE_LIMITS)
    request, http_body, decoded, native = (
        (folder / name).read_bytes()
        for name in ("request.bin", "http-body.bin", "decoded-body.bin", "native.bin")
    )
    if (
        sha256_hex(request) != binding.request_sha256
        or proof["request_sha256"] != binding.request_sha256
        or proof["http_body_sha256"] != sha256_hex(http_body)
        or proof["decoded_body_sha256"] != sha256_hex(decoded)
        or proof["native_sha256"] != sha256_hex(native)
        or not 200 <= proof["status_code"] < 300
        or encode_json(decode_json_object(decoded, RESPONSE_LIMITS), RESPONSE_LIMITS) != raw
        or attempts[-1]["response_json"].encode() != raw
        or attempts[-1]["response_sha256"] != sha256_hex(raw)
        or parse_typed_receipt(decoded).output_json != native
    ):
        raise ValueError("typed actual HTTP/native/ledger origin differs")
    require_typed_accounting(raw)
    return dict(
        protocol=protocol,
        request_sha256=binding.request_sha256,
        http_request_sha256=sha256_hex(request),
        http_body_sha256=sha256_hex(http_body),
        native_sha256=sha256_hex(native),
        ledger_response_sha256=sha256_hex(raw),
        run_identity_sha256=ledger["run_identity_sha256"],
        attempt_number=number,
        semantic_certified=False,
    )
