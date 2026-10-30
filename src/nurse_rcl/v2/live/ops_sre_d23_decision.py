"""D23 native decision admission against the exact frozen workbench request."""

from __future__ import annotations

from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_sre_feedback_context import parse_feedback_decision

CONTENT_RELATIONS = {"new_content", "no_new_content", "none_delivered"}


def validate_balanced_facts(value: dict[str, Any], context: dict[str, Any]) -> None:
    """Reject false process facts; do not judge diagnostic conclusions or select tools."""
    progress = context.get("investigation_progress", context.get("investigation_workbench"))
    if type(progress) is not dict:
        raise ExchangeBoundaryError("sre_progress_missing")
    latest = progress["latest_actual_result"]
    update = value.get("latest_result_update")
    if latest is None:
        if update is not None:
            raise ExchangeBoundaryError("sre_content_relation_differs")
    elif (
        type(update) is not dict
        or update.get("content_relation") != latest["content_relation"]
        or update.get("action_index") != latest["action_index"]
    ):
        raise ExchangeBoundaryError("sre_content_relation_differs")
    available = context["budget"]["actions_available_now"]
    if type(available) is not int or available < 0 or available > 8:
        raise ExchangeBoundaryError("sre_action_budget_differs")
    if not value["actions"] and value["stop_basis"] == "budget_exhausted" and available > 0:
        raise ExchangeBoundaryError("sre_stop_budget_available")


def parse_balanced_decision(
    raw: bytes, *, context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Use all previous shape/reference checks, then D23 frozen feedback checks."""
    value = decode_json_object(raw, CONTEXT_LIMITS)
    update = value.get("latest_result_update")
    if update is None:
        previous = value
    elif (
        type(update) is dict
        and set(update) == {
            "action_index", "answer_status", "summary", "remaining_question",
            "content_relation",
        }
        and update["content_relation"] in CONTENT_RELATIONS
    ):
        previous = dict(value, latest_result_update={
            key: item for key, item in update.items() if key != "content_relation"
        })
    else:
        raise ExchangeBoundaryError("sre_content_relation_shape")
    parse_feedback_decision(canonical_json_bytes(previous), context=context)
    if context is not None:
        validate_balanced_facts(value, context)
    return value
