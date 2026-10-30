"""Candidate /25 request bytes only; never discover settings or activate a provider."""

from __future__ import annotations

from typing import Any

from nurse_rcl.agents.ark import ARK_MODEL
from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import decode_json_object, encode_json
from nurse_rcl.v2.hermes_exchange import _validate_payload, prepare_exchange
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_empirical_policy import MAX_OUTPUT_TOKENS, OUTPUT_POLICY
from nurse_rcl.v2.live.ops_provider import _require_outbound
from nurse_rcl.v2.live.ops_sre_atomic_selector_decision import parse_atomic_decision
from nurse_rcl.v2.live.ops_sre_atomic_selector_exchange import (
    ATOMIC_SYSTEM,
    atomic_response_contract,
)
from nurse_rcl.v2.live.ops_sre_encoding import expand
from nurse_rcl.v2.live.ops_sre_feedback_fact_model import (
    MODEL_SCHEMA,
    base_packet,
    pack_feedback_model_view,
    unpack_feedback_model_view,
)
from nurse_rcl.v2.live.ops_sre_model_view import resolve_model_response

FACT_PROTOCOL = "nurse.v2.sre-decision-exchange/25"
FACT_PRESENTATION_GUIDANCE = (
    "workbench.feedback_fact_check presents only currently visible registered records, "
    "qualified by their original sources. It is the same supplied material, not new "
    "evidence, an answer or semantic approval. Original complete receipts remain authoritative."
)


def feedback_decision_policy() -> dict[str, Any]:
    system = ATOMIC_SYSTEM + "\n" + FACT_PRESENTATION_GUIDANCE
    pinned = dict(
        protocol=FACT_PROTOCOL,
        system=system,
        output_policy=OUTPUT_POLICY,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return dict(
        pinned,
        model_schema=MODEL_SCHEMA,
        base_input_schema="nurse.v2.sre-investigation-input/24",
        response_contract=atomic_response_contract(),
        prompt_sha256=sha256_hex(canonical_json_bytes(pinned)),
    )


def prepare_feedback_decision(
    context: dict[str, Any], *, references: dict[str, Any], case_id: str, snapshot: str, index: int
) -> tuple[bytes, dict[str, Any]]:
    packet, updated = pack_feedback_model_view(
        context, references=references, case_id=case_id, snapshot=snapshot, index=index
    )
    policy = feedback_decision_policy()
    body = dict(
        protocol=FACT_PROTOCOL,
        base_input_schema=policy["base_input_schema"],
        opaque_case_id=case_id,
        case_snapshot_id=snapshot,
        context=packet,
    )
    _require_outbound(body, case=case_id, snapshot=snapshot)
    prepared = prepare_exchange(
        model=ARK_MODEL,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        limits=CONTEXT_LIMITS,
        messages=[
            dict(role="system", content=policy["system"]),
            dict(role="user", content=encode_json(body, CONTEXT_LIMITS).decode()),
        ],
    )
    payload = decode_json_object(prepared.payload_bytes, CONTEXT_LIMITS)
    payload["thinking"] = {"type": "disabled"}
    wire = encode_json(payload, CONTEXT_LIMITS)
    recovered, _ = feedback_context_from_wire(
        wire, references=updated, case_id=case_id, snapshot=snapshot, index=index
    )
    if canonical_json_bytes(recovered) != canonical_json_bytes(expand(context)):
        raise ValueError("feedback candidate request roundtrip differs")
    return wire, updated


def feedback_context_from_wire(
    raw: bytes, *, references: dict[str, Any] | None, case_id: str, snapshot: str, index: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = decode_json_object(raw, CONTEXT_LIMITS)
    base = {k: v for k, v in payload.items() if k != "thinking"}
    if (
        encode_json(payload, CONTEXT_LIMITS) != raw
        or payload.get("thinking") != {"type": "disabled"}
        or _validate_payload(base) is not None
        or base.get("model") != ARK_MODEL
        or base.get("max_tokens") != MAX_OUTPUT_TOKENS
        or type(base.get("temperature")) not in (int, float)
        or base["temperature"] != 0
    ):
        raise ValueError("feedback candidate wire settings differ")
    messages = base["messages"]
    if (
        len(messages) != 2
        or messages[0] != dict(role="system", content=feedback_decision_policy()["system"])
        or messages[1].get("role") != "user"
    ):
        raise ValueError("feedback candidate messages differ")
    body_bytes = messages[1]["content"].encode()
    body = decode_json_object(body_bytes, CONTEXT_LIMITS)
    if (
        encode_json(body, CONTEXT_LIMITS) != body_bytes
        or set(body)
        != {"protocol", "base_input_schema", "opaque_case_id", "case_snapshot_id", "context"}
        or body["protocol"] != FACT_PROTOCOL
        or body["base_input_schema"] != "nurse.v2.sre-investigation-input/24"
        or body["opaque_case_id"] != case_id
        or body["case_snapshot_id"] != snapshot
    ):
        raise ValueError("feedback candidate body identity differs")
    _require_outbound(body, case=case_id, snapshot=snapshot)
    context, display = unpack_feedback_model_view(
        body["context"], references=references, case_id=case_id, snapshot=snapshot, index=index
    )
    whole = dict(context=context, feedback_fact_check=display)
    encode_json(whole, CONTEXT_LIMITS)
    _require_outbound(whole, case=case_id, snapshot=snapshot)
    return context, display


def resolve_feedback_response(
    value: dict[str, Any],
    *,
    packet: dict[str, Any],
    references: dict[str, Any],
    case_id: str,
    snapshot: str,
    index: int,
) -> dict[str, Any]:
    """Preserve original shape/reference semantics; action/note admission stays separate."""
    context, _ = unpack_feedback_model_view(
        packet, references=references, case_id=case_id, snapshot=snapshot, index=index
    )
    base, _ = base_packet(packet)
    resolved = resolve_model_response(
        value,
        packet=base,
        references=references,
        case_id=case_id,
        snapshot=snapshot,
        index=index,
        report=False,
    )
    parse_atomic_decision(canonical_json_bytes(resolved), context=context)
    return resolved
