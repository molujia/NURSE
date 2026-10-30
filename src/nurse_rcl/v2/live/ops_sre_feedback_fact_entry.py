"""Explicit bounded /25 entry; private /24 shells validate, never send or persist."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import decode_json_object, encode_json
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_sre_feedback_fact_exchange import (
    FACT_PROTOCOL,
    feedback_context_from_wire,
    feedback_decision_policy,
    prepare_feedback_decision,
)
from nurse_rcl.v2.live.ops_sre_feedback_fact_model import base_packet

if TYPE_CHECKING:
    from nurse_rcl.v2.live.ark_bridge import LiveRequestBinding


def is_feedback_wire(raw: bytes) -> bool:
    payload = decode_json_object(raw, CONTEXT_LIMITS)
    body = decode_json_object(payload["messages"][1]["content"].encode(), CONTEXT_LIMITS)
    return body.get("protocol") == FACT_PROTOCOL


def _parts(raw: bytes) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    payload = decode_json_object(raw, CONTEXT_LIMITS)
    if encode_json(payload, CONTEXT_LIMITS) != raw:
        raise ValueError("entry request is not canonical")
    body = decode_json_object(payload["messages"][1]["content"].encode(), CONTEXT_LIMITS)
    if encode_json(body, CONTEXT_LIMITS).decode() != payload["messages"][1]["content"]:
        raise ValueError("entry body is not canonical")
    plain = deepcopy(payload)
    uncorrected = {k: v for k, v in body.items() if k != "correction"}
    plain["messages"][1]["content"] = encode_json(uncorrected, CONTEXT_LIMITS).decode()
    plain_wire = encode_json(plain, CONTEXT_LIMITS)
    if "correction" in body and (
        type(body["correction"]) is not dict
        or body["correction"].get("original_wire_sha256") != sha256_hex(plain_wire)
    ):
        raise ValueError("entry correction original request differs")
    return payload, body, plain_wire


def _validation_wire(body: dict[str, Any]) -> bytes:
    from nurse_rcl.v2.live.ops_sre_brief_exchange import _request

    packet, _ = base_packet(body["context"])
    legacy = dict(
        protocol="nurse.v2.sre-decision-exchange/24",
        context=packet,
        opaque_case_id=body["opaque_case_id"],
        case_snapshot_id=body["case_snapshot_id"],
    )
    original = _request(legacy, report=False)
    if "correction" in body:
        correction = deepcopy(body["correction"])
        correction["original_wire_sha256"] = sha256_hex(original)
        legacy["correction"] = correction
    return _request(legacy, report=False)


def entry_context_from_wire(
    raw: bytes, *, references: dict[str, Any] | None, case_id: str, snapshot: str, index: int
) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_context_from_wire

    _, body, plain = _parts(raw)
    context, _ = feedback_context_from_wire(
        plain, references=references, case_id=case_id, snapshot=snapshot, index=index
    )
    correction = body.get("correction")
    if correction is not None and (
        correction.get("decision_index") != index
        or type(correction.get("correction_index")) is not int
    ):
        raise ValueError("entry correction decision differs")
    validated = brief_context_from_wire(
        _validation_wire(body),
        references=references,
        opaque_case_id=case_id,
        case_snapshot_id=snapshot,
        decision_index=index,
    )
    if canonical_json_bytes(validated) != canonical_json_bytes(context):
        raise ValueError("entry complete context validation differs")
    return context


def prepare_entry_decision(
    context: dict[str, Any], *, references: dict[str, Any], case_id: str, snapshot: str, index: int
) -> bytes:
    raw, updated = prepare_feedback_decision(
        context, references=deepcopy(references), case_id=case_id, snapshot=snapshot, index=index
    )
    entry_context_from_wire(
        raw, references=updated, case_id=case_id, snapshot=snapshot, index=index
    )
    references.clear()
    references.update(updated)
    return raw


def bind_entry_decision(
    raw: bytes, *, case_id: str, snapshot: str, index: int
) -> LiveRequestBinding:
    from nurse_rcl.v2.live.ark_bridge import LiveRequestBinding

    context = entry_context_from_wire(
        raw, references=None, case_id=case_id, snapshot=snapshot, index=index
    )
    _, body, plain = _parts(raw)
    _, display = feedback_context_from_wire(
        plain, references=None, case_id=case_id, snapshot=snapshot, index=index
    )
    correction = body.get("correction")
    return LiveRequestBinding(
        opaque_case_id=case_id,
        variant="v2-ops-sre/14",
        phase="investigation_correction" if correction is not None else "investigation",
        decision_index=index,
        correction_index=1 if correction is not None else None,
        input_sha256=snapshot,
        context_sha256=sha256_hex(
            encode_json(dict(context=context, feedback_fact_check=display), CONTEXT_LIMITS)
        ),
        prompt_schema_sha256=feedback_decision_policy()["prompt_sha256"],
        request_sha256=sha256_hex(raw),
    )


def prepare_entry_correction(
    original_wire: bytes,
    *,
    native_proposal: bytes,
    proof: dict[str, Any],
    references: dict[str, Any],
    case_id: str,
    snapshot: str,
    index: int,
) -> bytes:
    from nurse_rcl.v2.live.ops_sre_brief_exchange import _body, prepare_brief_correction

    context = entry_context_from_wire(
        original_wire, references=references, case_id=case_id, snapshot=snapshot, index=index
    )
    payload, body, _ = _parts(original_wire)
    if "correction" in body:
        raise ValueError("entry correction cannot be corrected again")
    checked = prepare_brief_correction(
        _validation_wire(body),
        native_proposal=native_proposal,
        proof=proof,
        references=references,
        opaque_case_id=case_id,
        case_snapshot_id=snapshot,
        decision_index=index,
    )
    correction = _body(checked, report=False)["correction"]
    correction["original_wire_sha256"] = sha256_hex(original_wire)
    body["correction"] = correction
    payload["messages"][1]["content"] = encode_json(body, CONTEXT_LIMITS).decode()
    raw = encode_json(payload, CONTEXT_LIMITS)
    if (
        entry_context_from_wire(
            raw, references=references, case_id=case_id, snapshot=snapshot, index=index
        )
        != context
    ):
        raise ValueError("entry correction changed context")
    return raw


def require_feedback_history(wires: list[str], *, enabled: bool) -> None:
    expected = FACT_PROTOCOL if enabled else "nurse.v2.sre-decision-exchange/24"
    for wire in wires:
        try:
            payload = decode_json_object(wire.encode(), CONTEXT_LIMITS)
            body = decode_json_object(payload["messages"][1]["content"].encode(), CONTEXT_LIMITS)
            if body.get("protocol") != expected:
                raise ValueError("entry saved presentation differs")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ValueError("entry saved wire shape differs") from exc
