"""Isolated investigation protocol; no discovery, retries, or scoring authority."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nurse_rcl.agents.ark import ARK_MODEL
from nurse_rcl.artifacts import sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object, encode_json
from nurse_rcl.v2.hermes_exchange import _validate_payload, prepare_exchange
from nurse_rcl.v2.hermes_response import parse_response
from nurse_rcl.v2.live.ops_agent_decision import (
    CONTEXT_LIMITS,
    DECISION_LIMITS,
    SYSTEM,
    parse_decision,
)
from nurse_rcl.v2.live.ops_empirical_policy import (
    DECISION_PROTOCOL,
    MAX_OUTPUT_TOKENS,
    OUTPUT_POLICY,
)

if TYPE_CHECKING:
    from nurse_rcl.v2.live.ark_bridge import LiveArkBridge, LiveRequestBinding

OPS_DECISION_EXCHANGE_VERSION = "nurse.v2.ops-decision-exchange/1"
OPS_PROMPT_SHA256 = sha256_hex(encode_json(dict(
    protocol=OPS_DECISION_EXCHANGE_VERSION, system=SYSTEM, max_tokens=2048,
), CONTEXT_LIMITS))
OPS_EMPIRICAL_PROMPT_SHA256 = sha256_hex(encode_json(dict(
    protocol=DECISION_PROTOCOL, system=SYSTEM, output_policy=OUTPUT_POLICY,
    max_tokens=MAX_OUTPUT_TOKENS,
), CONTEXT_LIMITS))


def validate_ops_decision_payload(
    payload: dict[str, Any], *, empirical: bool = False,
) -> dict[str, Any]:
    """Wire/identity checks only. Trusted runner owns label-free source admission."""
    if payload.get("thinking") != {"type": "disabled"}:
        raise ExchangeBoundaryError("invalid_ops_request")
    base = {key: value for key, value in payload.items() if key != "thinking"}
    if _validate_payload(base) is not None or (
        base["model"] != ARK_MODEL
        or base["max_tokens"] != (MAX_OUTPUT_TOKENS if empirical else 2048)
        or base["temperature"] != 0
    ):
        raise ExchangeBoundaryError("invalid_ops_request")
    messages = base["messages"]
    if len(messages) != 2 or messages[0] != {"role": "system", "content": SYSTEM} or (
        messages[1]["role"] != "user"
    ):
        raise ExchangeBoundaryError("invalid_ops_messages")
    body = decode_json_object(messages[1]["content"].encode("utf-8"), CONTEXT_LIMITS)
    if set(body) != {"protocol", "opaque_case_id", "case_snapshot_id", "context"} or (
        body["protocol"] != (DECISION_PROTOCOL if empirical else OPS_DECISION_EXCHANGE_VERSION)
        or type(body["context"]) is not dict
    ):
        raise ExchangeBoundaryError("invalid_ops_context")
    context = body["context"]
    if empirical:
        expected = {"schema_version", "case_snapshot_id", "decision_index", "remaining_actions",
                    "remaining_decisions", "capabilities", "skills", "actions",
                    "previous_feedback", "critical_evidence", "omitted_detail"}
        if set(context) != expected or (
            context["schema_version"] != "nurse.v2.ops-investigation-input/2"
            or context["case_snapshot_id"] != body["case_snapshot_id"]
            or type(context["decision_index"]) is not int
            or not 1 <= context["decision_index"] <= 8
            or type(context["remaining_actions"]) is not int
            or not 0 <= context["remaining_actions"] <= 8
            or type(context["remaining_decisions"]) is not int
            or context["remaining_decisions"] != 9 - context["decision_index"]
            or type(context["capabilities"]) is not dict
            or context["capabilities"].get("case_snapshot_id") != body["case_snapshot_id"]
            or context["capabilities"].get("investigation_profile_id") != "hermes-readonly-ops/2"
            or any(type(context[k]) is not list for k in (
                "actions", "skills", "critical_evidence", "omitted_detail"))
        ):
            raise ExchangeBoundaryError("invalid_ops_context")
        return body
    if set(context) != {"messages", "capabilities", "decision_index", "remaining_actions",
                        "remaining_decisions"} or (
        type(context["decision_index"]) is not int or not 1 <= context["decision_index"] <= 8
        or type(context["remaining_actions"]) is not int
        or not 0 <= context["remaining_actions"] <= 8
        or type(context["remaining_decisions"]) is not int
        or context["remaining_decisions"] != 9 - context["decision_index"]
        or type(context["capabilities"]) is not dict
        or context["capabilities"].get("case_snapshot_id") != body["case_snapshot_id"]
        or type(context["messages"]) is not list or not context["messages"]
    ):
        raise ExchangeBoundaryError("invalid_ops_context")
    return body


def prepare_ops_decision(
    *, opaque_case_id: str, case_snapshot_id: str, decision_index: int, context: dict[str, Any],
    empirical: bool = False,
) -> tuple[bytes, LiveRequestBinding]:
    from nurse_rcl.v2.live.ark_bridge import LiveRequestBinding

    body = dict(protocol=DECISION_PROTOCOL if empirical else OPS_DECISION_EXCHANGE_VERSION,
                opaque_case_id=opaque_case_id,
                case_snapshot_id=case_snapshot_id, context=context)
    request = prepare_exchange(model=ARK_MODEL,
                               max_output_tokens=MAX_OUTPUT_TOKENS if empirical else 2048,
                               limits=CONTEXT_LIMITS,
                               messages=[{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": encode_json(
                                             body, CONTEXT_LIMITS).decode("utf-8")}])
    payload = decode_json_object(request.payload_bytes, CONTEXT_LIMITS)
    payload["thinking"] = {"type": "disabled"}
    validate_ops_decision_payload(payload, empirical=empirical)
    if context["decision_index"] != decision_index:
        raise ExchangeBoundaryError("ops_decision_index_drift")
    raw = encode_json(payload, CONTEXT_LIMITS)
    binding = LiveRequestBinding(
        opaque_case_id=opaque_case_id,
        variant="v2-ops-isolated/2" if empirical else "v2-ops-isolated/1", phase="investigation",
        decision_index=decision_index, input_sha256=case_snapshot_id,
        context_sha256=sha256_hex(encode_json(context, CONTEXT_LIMITS)),
        prompt_schema_sha256=OPS_EMPIRICAL_PROMPT_SHA256 if empirical else OPS_PROMPT_SHA256,
        request_sha256=sha256_hex(raw),
    )
    return raw, binding


def bind_empirical_decision(
    raw: bytes, *, opaque_case_id: str, case_snapshot_id: str, decision_index: int,
) -> LiveRequestBinding:
    """Validate a frozen wire without rebuilding it from a refreshed context."""
    from nurse_rcl.v2.live.ark_bridge import LiveRequestBinding

    payload = decode_json_object(raw, CONTEXT_LIMITS)
    body = validate_ops_decision_payload(payload, empirical=True)
    context = body["context"]
    if body["opaque_case_id"] != opaque_case_id or (
        body["case_snapshot_id"] != case_snapshot_id
        or context["decision_index"] != decision_index
        or encode_json(payload, CONTEXT_LIMITS) != raw
    ):
        raise ExchangeBoundaryError("ops_frozen_decision_binding")
    return LiveRequestBinding(
        opaque_case_id=opaque_case_id, variant="v2-ops-isolated/2", phase="investigation",
        decision_index=decision_index, input_sha256=case_snapshot_id,
        context_sha256=sha256_hex(encode_json(context, CONTEXT_LIMITS)),
        prompt_schema_sha256=OPS_EMPIRICAL_PROMPT_SHA256, request_sha256=sha256_hex(raw),
    )


class OpsArkCompletion:
    """Agent completion callable. The bridge owns lifetime retries/usage receipts.

    Construction is not authorization: the live runner must gate real transport
    on the user's approved cases/token estimate. No health probe is sent here.
    """

    def __init__(
        self, *, bridge: LiveArkBridge, opaque_case_id: str, case_snapshot_id: str,
    ) -> None:
        if bridge.request_protocol != OPS_DECISION_EXCHANGE_VERSION:
            raise ValueError("isolated investigation requires its own protocol")
        self._bridge = bridge
        self._case = opaque_case_id
        self._snapshot = case_snapshot_id

    def __call__(self, index: int, context: dict[str, Any]) -> bytes:
        payload, binding = prepare_ops_decision(opaque_case_id=self._case,
                                              case_snapshot_id=self._snapshot,
                                              decision_index=index, context=context)
        raw = self._bridge.complete(binding, payload)
        parsed = parse_response(raw, output_tool_name=None, limits=CONTEXT_LIMITS)
        return encode_json(parse_decision(parsed.output_json), DECISION_LIMITS)
