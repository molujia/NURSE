"""Independent /20 presentation wrapper. No change to /19's acceptance domain."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from nurse_rcl.v2.exchange_json import encode_json
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_sre_feedback_facts import (
    feedback_fact_check,
    validate_feedback_fact_check,
)
from nurse_rcl.v2.live.ops_sre_model_view import pack_model_view, unpack_model_view
from nurse_rcl.v2.live.ops_sre_feedback_representation import (
    BINDING_FIELD, representation_bindings,
)
from nurse_rcl.artifacts import canonical_json_bytes

MODEL_SCHEMA = "nurse.v2.sre-model-view/20"
OLD_MODEL_SCHEMA = "nurse.v2.sre-model-view/19"


def pack_feedback_model_view(
    context: dict[str, Any],
    *,
    references: dict[str, Any],
    case_id: str,
    snapshot: str,
    index: int,
    report: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        report
        or context.get("schema_version") != "nurse.v2.sre-investigation-input/24"
        or context.get("decision_index") != index
        or context.get("case_snapshot_id") != snapshot
    ):
        raise ValueError("feedback base input identity differs")
    feedback_fact_check(context)
    args: dict[str, Any] = dict(case_id=case_id, snapshot=snapshot, index=index, report=False)
    base, updated = pack_model_view(context, references=references, **args)
    if base["schema_version"] != OLD_MODEL_SCHEMA:
        raise ValueError("feedback old model identity differs")
    short = unpack_model_view(base, references=None, **args)
    packet = deepcopy(base)
    packet["schema_version"] = MODEL_SCHEMA
    relations = representation_bindings(context, references=updated,
        case_id=case_id, snapshot=snapshot, index=index)
    if relations:
        packet['workbench'][BINDING_FIELD] = relations
    packet["workbench"]["feedback_fact_check"] = feedback_fact_check(
        short, short_refs=True, representation_bindings=relations or None)
    encode_json(packet, CONTEXT_LIMITS)
    return packet, updated


def base_packet(packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(packet) is not dict or packet.get("schema_version") != MODEL_SCHEMA:
        raise ValueError("feedback model identity differs")
    encode_json(packet, CONTEXT_LIMITS)
    result = deepcopy(packet)
    if (
        type(result.get("workbench")) is not dict
        or "feedback_fact_check" not in result["workbench"]
    ):
        raise ValueError("feedback model check missing")
    aid = result["workbench"].pop("feedback_fact_check")
    result['workbench'].pop(BINDING_FIELD, None)
    result["schema_version"] = OLD_MODEL_SCHEMA
    return result, aid


def unpack_feedback_model_view(
    packet: dict[str, Any],
    *,
    references: dict[str, Any] | None,
    case_id: str,
    snapshot: str,
    index: int,
    report: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if report:
        raise ValueError("feedback report mode is not registered")
    base, aid = base_packet(packet)
    args: dict[str, Any] = dict(case_id=case_id, snapshot=snapshot, index=index, report=False)
    short = unpack_model_view(base, references=None, **args)
    if short.get("decision_index") != index or short.get("case_snapshot_id") != snapshot:
        raise ValueError("feedback current context/request identity differs")
    relations = packet['workbench'].get(BINDING_FIELD)
    validate_feedback_fact_check(aid, short, short_refs=True,
                                 representation_bindings=relations)
    decoded = unpack_model_view(base, references=references, **args)
    if references is not None:
        feedback_fact_check(decoded)
        if relations is not None and canonical_json_bytes(relations) != canonical_json_bytes(
            representation_bindings(decoded, references=references,
                case_id=case_id, snapshot=snapshot, index=index)
        ):
            raise ValueError('feedback representation binding differs from original owner')
    return decoded, deepcopy(aid)
