"""Explicit lossless indexed semantics; no retry, scoring or source authority."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from nurse_rcl.agents.ark import ARK_MODEL
from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object, encode_json
from nurse_rcl.v2.hermes_exchange import (
    ExchangeAttempt,
    PreparedExchange,
    _validate_payload,
    prepare_exchange,
)
from nurse_rcl.v2.hermes_response import parse_response
from nurse_rcl.v2.live.normal_semantic import parse_normal_semantic_response
from nurse_rcl.v2.live.normal_semantic_action_index import (
    build_action_index,
    decode_indexed_wire,
    encode_indexed_wire,
    require_action_index,
    require_indexed_wire,
)
from nurse_rcl.v2.live.normal_semantic_compact_exchange import COMPACT_GUIDANCE, COMPACT_LIMITS
from nurse_rcl.v2.live.normal_semantic_context import NormalSemanticContext
from nurse_rcl.v2.live.normal_semantic_disabled_exchange import DISABLED_GUIDANCE
from nurse_rcl.v2.live.normal_semantic_exchange import _system
from nurse_rcl.v2.live.normal_semantic_feedback import (
    NormalSemanticFeedback,
    SemanticFeedbackPackage,
    restore_semantic_feedback,
)
from nurse_rcl.v2.live.normal_semantic_wire import (
    WIRE_LIMITS,
    build_semantic_wire,
    restore_semantic_context,
)
from nurse_rcl.v2.semantic_run_audit import SemanticRankingPolicy

INDEXED_SEMANTIC_EXCHANGE_VERSION = "nurse.v2.normal-semantic-indexed-exchange/1"
INDEXED_BODY_VERSION = "nurse.v2.normal-semantic-indexed-request/1"
INDEXED_POLICY = SemanticRankingPolicy(
    configuration_version="normal-semantic-indexed-400k-8192/1",
    input_token_limit=400_000, output_token_limit=8192,
)
DESCRIPTOR_GUIDANCE = """
This wire additionally uses signal_descriptors, whose signal_descriptor_columns
name each field. Every signal remains a separate row: descriptor_index is a
zero-based INTEGER into signal_descriptors, not a score or independent evidence.
Reusing the exact descriptor removes only repeated text. All scientific values,
quality, units, baseline/fault summaries and facts remain attached to each signal.
"""
ACTION_GUIDANCE = """
action_index is the COMPLETE initial LEGAL query navigation, grouped by candidate
and mechanism in fixed order, NOT a ranking or an execution sequence. For each
verification query, select ONE option and a nonempty subset of that option's
needs. Copy only those need_key values, its request_key, and link the query_id
from a hypothesis with the group's candidate_service_key and mechanism_key.
Use its signal_reference as that hypothesis's source evidence if appropriate.
DO NOT copy all N references from the full requests table: that table deliberately
also retains NONQUERYABLE needs, refutations and unknowns. DO NOT mix mechanisms,
requests or candidates. Do not return action IDs; use the unchanged response schema.
The signal reference is an observation, NOT confirmation that its mechanism holds.
The index only guides initial choices: the executor reselects legal unresolved
needs after actual feedback. Different mechanism options for one R are the SAME
physical measurement, not additional independent support. All C remain eligible
for the original paid card query, even when absent from this index. Never invent
an executable option, infer health from absence, or silently discard contrary
evidence. A hypothesis needing no query and empty suggestions remain allowed.
"""


def _messages(body: dict[str, Any]) -> list[dict[str, str]]:
    from nurse_rcl.v2.live.normal_semantic_followup import FOLLOWUP_GUIDANCE

    guidance = ACTION_GUIDANCE if body["phase"] == "semantic_first" else FOLLOWUP_GUIDANCE
    return [dict(role="system", content=_system() + COMPACT_GUIDANCE + DESCRIPTOR_GUIDANCE
                 + guidance + DISABLED_GUIDANCE),
            dict(role="user", content=canonical_json_bytes(body).decode())]


def validate_indexed_semantic_payload(
    payload: dict[str, Any],
) -> Literal["semantic_first", "semantic_followup"]:
    from nurse_rcl.v2.live.normal_semantic_followup import _validate_feedback_display

    raw = encode_json(payload, COMPACT_LIMITS)
    if len(raw) + 256 > INDEXED_POLICY.input_token_limit:
        raise ExchangeBoundaryError("indexed_semantic_input_budget")
    if payload.get("thinking") != {"type": "disabled"}:
        raise ExchangeBoundaryError("invalid_indexed_semantic_request")
    plain = {key: value for key, value in payload.items() if key != "thinking"}
    if _validate_payload(plain) is not None or plain["model"] != ARK_MODEL or (
        plain["max_tokens"] != INDEXED_POLICY.output_token_limit or plain["temperature"] != 0
    ):
        raise ExchangeBoundaryError("invalid_indexed_semantic_request")
    try:
        messages = payload["messages"]
        if len(messages) != 2:
            raise ValueError
        body = decode_json_object(messages[1]["content"].encode(), COMPACT_LIMITS)
        if body["schema_version"] != INDEXED_BODY_VERSION:
            raise ValueError
        if body["phase"] == "semantic_first":
            if set(body) != {"schema_version", "phase", "opaque_case_id", "wire", "action_index"}:
                raise ValueError
            wire = decode_indexed_wire(body["wire"])
            require_action_index(body["action_index"], wire)
        elif body["phase"] == "semantic_followup":
            if set(body) != {"schema_version", "phase", "opaque_case_id", "feedback"}:
                raise ValueError
            feedback = dict(body["feedback"])
            wire = decode_indexed_wire(feedback["fixed_wire"])
            feedback["fixed_wire"] = wire
            _validate_feedback_display(feedback)
        else:
            raise ValueError
        if body["opaque_case_id"] != wire["opaque_case_id"] or messages != _messages(body):
            raise ValueError
    except (ValueError, KeyError, TypeError, AttributeError, IndexError):
        raise ExchangeBoundaryError("invalid_indexed_semantic_request") from None
    return "semantic_first" if body["phase"] == "semantic_first" else "semantic_followup"


def _prepare(body: dict[str, Any]) -> PreparedExchange:
    # Build the NEW body directly: never require the old expanded network body
    # to fit 400k before applying its lossless descriptor encoding.
    base = prepare_exchange(model=ARK_MODEL, messages=_messages(body),
                            max_output_tokens=INDEXED_POLICY.output_token_limit,
                            temperature=INDEXED_POLICY.temperature, limits=COMPACT_LIMITS)
    payload = decode_json_object(base.payload_bytes, COMPACT_LIMITS)
    payload["thinking"] = {"type": "disabled"}
    validate_indexed_semantic_payload(payload)
    return PreparedExchange(encode_json(payload, COMPACT_LIMITS), COMPACT_LIMITS)


def prepare_indexed_semantic_exchange(*, context: NormalSemanticContext) -> PreparedExchange:
    package = build_semantic_wire(context)
    restore_semantic_context(package, context)
    original = decode_json_object(package.wire_json.encode(), WIRE_LIMITS)
    wire = encode_indexed_wire(original)
    require_indexed_wire(wire, original)
    return _prepare(dict(schema_version=INDEXED_BODY_VERSION, phase="semantic_first",
                         opaque_case_id=context.opaque_case_id, wire=wire,
                         action_index=build_action_index(original)))


def prepare_indexed_followup_exchange(
    *, feedback: NormalSemanticFeedback, package: SemanticFeedbackPackage,
) -> PreparedExchange:
    restore_semantic_feedback(package, feedback)
    display = decode_json_object(package.public_json.encode(), WIRE_LIMITS)
    original = display["fixed_wire"]
    display["fixed_wire"] = encode_indexed_wire(original)
    require_indexed_wire(display["fixed_wire"], original)
    return _prepare(dict(schema_version=INDEXED_BODY_VERSION, phase="semantic_followup",
                         opaque_case_id=feedback.fixed.opaque_case_id, feedback=display))


def execute_indexed_semantic_once(
    request: PreparedExchange, complete: Callable[[bytes], bytes],
) -> ExchangeAttempt:
    if type(request) is not PreparedExchange or request.limits != COMPACT_LIMITS:
        raise ExchangeBoundaryError("invalid_indexed_semantic_request")
    if not callable(complete):
        raise ExchangeBoundaryError("invalid_transport")
    payload = decode_json_object(request.payload_bytes, COMPACT_LIMITS)
    validate_indexed_semantic_payload(payload)
    if encode_json(payload, COMPACT_LIMITS) != request.payload_bytes:
        raise ExchangeBoundaryError("noncanonical_request")

    def failed(code: str, raw: bytes | None = None, received: int | None = None) -> ExchangeAttempt:
        return ExchangeAttempt("failed", request.sha256, True, code, None, raw,
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
    return ExchangeAttempt("response_ready", request.sha256, True, None, parsed, raw,
                           sha256_hex(raw), len(raw))
