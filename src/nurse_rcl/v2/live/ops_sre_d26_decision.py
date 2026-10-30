"""D26 source-bound public plan admission against the exact frozen request."""

from __future__ import annotations

from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_sre_d23_decision import parse_balanced_decision
from nurse_rcl.v2.live.ops_sre_d26_questions import (
    apply_question_plan,
    validate_stop_assessment,
)

DECISION_FIELDS = {
    "actions",
    "stop_reason",
    "public_summary",
    "unresolved",
    "responds_to",
    "investigation_plan",
    "stop_basis",
    "latest_result_update",
    "stop_assessment",
}


def parse_question_decision(
    raw: bytes,
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        context is not None
        and context.get("schema_version") == "nurse.v2.sre-investigation-input/19"
    ):
        from nurse_rcl.v2.live.ops_sre_directed_decision import parse_directed_decision

        return parse_directed_decision(raw, context=context)
    if (
        context is not None
        and context.get("schema_version") == "nurse.v2.sre-investigation-input/18"
    ):
        from nurse_rcl.v2.live.ops_sre_action_decision import parse_action_decision

        return parse_action_decision(raw, context=context)
    value = decode_json_object(raw, CONTEXT_LIMITS)
    if set(value) != DECISION_FIELDS:
        raise ExchangeBoundaryError("sre_question_decision_shape")
    if context is None or type(context.get("question_state")) is not dict:
        raise ExchangeBoundaryError("sre_question_context_required")
    state = apply_question_plan(
        context["question_state"], value["investigation_plan"], context=context
    )
    plan = value["investigation_plan"]
    current = plan["current_question_id"]
    previous = {
        k: v
        for k, v in value.items()
        if k
        not in {
            "investigation_plan",
            "stop_assessment",
        }
    }
    previous["investigation_question"] = (
        None
        if current is None
        else dict(
            target_service_keys=next(
                q["target_service_keys"] for q in state["questions"] if q["question_id"] == current
            ),
            **plan["action_purpose"],
        )
    )
    update = value["latest_result_update"]
    if update is not None:
        if type(update) is not dict or set(update) != {
            "action_index",
            "answer_status",
            "summary",
            "remaining_question",
            "content_relation",
            "question_id",
        }:
            raise ExchangeBoundaryError("sre_question_latest_shape")
        if update["question_id"] != context["question_state"]["latest_action_question_id"]:
            raise ExchangeBoundaryError("sre_question_latest_differs")
        previous["latest_result_update"] = {k: v for k, v in update.items() if k != "question_id"}
    # Reuse, do not relax, the original action, responds_to, actual-feedback and budget gates.
    parse_balanced_decision(canonical_json_bytes(previous), context=context)
    if value["actions"]:
        if current is None or value["stop_assessment"] is not None:
            raise ExchangeBoundaryError("sre_question_continue_shape")
    else:
        if current is not None:
            raise ExchangeBoundaryError("sre_question_stop_shape")
        validate_stop_assessment(state, value["stop_assessment"], context=context)
    return value
