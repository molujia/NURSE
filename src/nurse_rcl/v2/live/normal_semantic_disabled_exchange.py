"""Explicit bounded non-thinking semantics; legacy wire and guards stay intact."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from nurse_rcl.artifacts import sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object, encode_json
from nurse_rcl.v2.hermes_exchange import ExchangeAttempt, PreparedExchange
from nurse_rcl.v2.hermes_response import parse_response
from nurse_rcl.v2.live.normal_semantic import parse_normal_semantic_response
from nurse_rcl.v2.live.normal_semantic_compact_exchange import COMPACT_LIMITS, COMPACT_POLICY
from nurse_rcl.v2.semantic_run_audit import SemanticRankingPolicy

DISABLED_SEMANTIC_EXCHANGE_VERSION = "nurse.v2.normal-semantic-disabled-exchange/1"
DISABLED_POLICY = SemanticRankingPolicy(
    configuration_version="normal-semantic-disabled-400k-8192/1",
    input_token_limit=400_000, output_token_limit=8192,
)
DISABLED_GUIDANCE = """
Compact output contract: use short local IDs h1,h2 and q1,q2, never concatenate
service or mechanism names into IDs. The FIRST queries array has at most 8 entries
TOTAL across all hypotheses, not 8 per hypothesis. Reuse a query ID only when its
candidate and mechanism needs are compatible. FOLLOW-UP queries and query_ids
must all be empty. Prefer a few useful competing testable hypotheses; do not
enumerate every candidate/mechanism pair. Keep each rationale to one short
sentence and unknown reasons brief. Preserve contrary/unknown evidence and exact
visible references. Finish the entire schema-valid JSON within the output budget.
"""


def semantic_policy(protocol: str) -> SemanticRankingPolicy:
    from nurse_rcl.v2.live.normal_semantic_followup import HYBRID_SEMANTIC_EXCHANGE_VERSION
    from nurse_rcl.v2.live.normal_semantic_indexed_exchange import (
        INDEXED_POLICY,
        INDEXED_SEMANTIC_EXCHANGE_VERSION,
    )

    if protocol == INDEXED_SEMANTIC_EXCHANGE_VERSION:
        return INDEXED_POLICY
    if protocol == DISABLED_SEMANTIC_EXCHANGE_VERSION:
        return DISABLED_POLICY
    if protocol == HYBRID_SEMANTIC_EXCHANGE_VERSION:
        return COMPACT_POLICY
    raise ExchangeBoundaryError("unregistered_semantic_protocol")


def validate_disabled_semantic_payload(
    payload: dict[str, Any],
) -> Literal["semantic_first", "semantic_followup"]:
    from nurse_rcl.v2.live.normal_semantic_followup import validate_hybrid_semantic_payload

    raw = encode_json(payload, COMPACT_LIMITS)
    if len(raw) + 256 > DISABLED_POLICY.input_token_limit:
        raise ExchangeBoundaryError("disabled_semantic_input_budget")
    if payload.get("thinking") != {"type": "disabled"} or (
        type(payload.get("max_tokens")) is not int
        or payload["max_tokens"] != DISABLED_POLICY.output_token_limit
    ):
        raise ExchangeBoundaryError("invalid_disabled_semantic_request")
    # A private validation copy removes ONLY the explicitly versioned additions.
    # The complete scientific body and all original shape/reference checks remain.
    legacy = decode_json_object(raw, COMPACT_LIMITS)
    del legacy["thinking"]
    legacy["max_tokens"] = COMPACT_POLICY.output_token_limit
    try:
        instruction = legacy["messages"][0]["content"]
        if type(instruction) is not str or not instruction.endswith(DISABLED_GUIDANCE):
            raise ValueError
        legacy["messages"][0]["content"] = instruction[:-len(DISABLED_GUIDANCE)]
    except (KeyError, IndexError, TypeError, ValueError):
        raise ExchangeBoundaryError("invalid_disabled_semantic_request") from None
    return validate_hybrid_semantic_payload(legacy)


def prepare_disabled_semantic_exchange(legacy: PreparedExchange) -> PreparedExchange:
    from nurse_rcl.v2.live.normal_semantic_followup import validate_hybrid_semantic_payload

    if type(legacy) is not PreparedExchange or legacy.limits != COMPACT_LIMITS:
        raise ExchangeBoundaryError("invalid_disabled_semantic_request")
    payload = decode_json_object(legacy.payload_bytes, COMPACT_LIMITS)
    validate_hybrid_semantic_payload(payload)
    if encode_json(payload, COMPACT_LIMITS) != legacy.payload_bytes:
        raise ExchangeBoundaryError("noncanonical_request")
    payload["thinking"] = {"type": "disabled"}
    payload["max_tokens"] = DISABLED_POLICY.output_token_limit
    payload["messages"][0]["content"] += DISABLED_GUIDANCE
    validate_disabled_semantic_payload(payload)
    return PreparedExchange(encode_json(payload, COMPACT_LIMITS), COMPACT_LIMITS)


def execute_disabled_semantic_once(
    request: PreparedExchange, complete: Callable[[bytes], bytes],
) -> ExchangeAttempt:
    """One exact exchange. The existing persistent bridge alone retries HTTP."""
    if type(request) is not PreparedExchange or request.limits != COMPACT_LIMITS:
        raise ExchangeBoundaryError("invalid_disabled_semantic_request")
    if not callable(complete):
        raise ExchangeBoundaryError("invalid_transport")
    payload = decode_json_object(request.payload_bytes, COMPACT_LIMITS)
    validate_disabled_semantic_payload(payload)
    if encode_json(payload, COMPACT_LIMITS) != request.payload_bytes:
        raise ExchangeBoundaryError("noncanonical_request")
    pin = request.sha256

    def failed(code: str, raw: bytes | None = None, received: int | None = None) -> ExchangeAttempt:
        return ExchangeAttempt("failed", pin, True, code, None, raw,
                               sha256_hex(raw) if raw is not None else None, received)

    try:
        raw = complete(request.payload_bytes)
    except TimeoutError:
        return failed("timeout")
    except Exception:
        return failed("transport_error")
    if type(raw) is not bytes:
        return failed("response_type")
    if len(raw) > COMPACT_LIMITS.max_bytes:
        return failed("response_limit", received=len(raw))
    try:
        parsed = parse_response(raw, output_tool_name=None, limits=COMPACT_LIMITS)
        parse_normal_semantic_response(parsed.output_json)
    except ExchangeBoundaryError as exc:
        return failed(exc.code, raw, len(raw))
    return ExchangeAttempt("response_ready", pin, True, None, parsed, raw,
                           sha256_hex(raw), len(raw))
