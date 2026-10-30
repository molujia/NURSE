"""D14 acknowledgment of actual latest feedback, without semantic authority."""

from __future__ import annotations

from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_sre_brief_context import parse_brief_decision

FEEDBACK_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/8"
CONDITION_PROFILE = "hermes-sre-tools/7"
LATEST_RESULT_FIELDS = {"action_index", "answer_status", "summary", "remaining_question"}
ANSWER_STATUSES = frozenset({"answered", "partial", "unresolved", "no_observation", "deferred"})


def parse_feedback_decision(raw: bytes, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
    value = decode_json_object(raw, CONTEXT_LIMITS)
    if "latest_result_update" not in value:
        raise ExchangeBoundaryError("sre_latest_result_missing")
    parse_brief_decision(
        canonical_json_bytes({k: v for k, v in value.items() if k != "latest_result_update"}),
        context=context,
    )
    validate_latest_result_update(value, context=context)
    return value


def validate_latest_result_update(
    value: dict[str, Any], *, context: dict[str, Any] | None = None
) -> None:
    """The original real-action gate, reusable without requiring a public plan."""
    update = value["latest_result_update"]
    if update is not None:
        if type(update) is not dict or set(update) != LATEST_RESULT_FIELDS:
            raise ExchangeBoundaryError("sre_latest_result_shape")
        if (
            type(update["action_index"]) is not int
            or not 1 <= update["action_index"] <= 8
            or type(update["answer_status"]) is not str
            or update["answer_status"] not in ANSWER_STATUSES
            or type(update["summary"]) is not str
            or not update["summary"].strip()
            or (
                update["remaining_question"] is not None
                and (
                    type(update["remaining_question"]) is not str
                    or not update["remaining_question"].strip()
                )
            )
        ):
            raise ExchangeBoundaryError("sre_latest_result_shape")
    if context is not None:
        latest = context["actions"][-1]["action_index"] if context["actions"] else None
        if latest is None:
            if update is not None:
                raise ExchangeBoundaryError("sre_latest_result_without_action")
        elif (
            update is None
            or update["action_index"] != latest
            or not any(link["action_index"] == latest for link in value["responds_to"])
        ):
            raise ExchangeBoundaryError("sre_latest_result_not_addressed")


def latest_result_context(
    actions: list[dict[str, Any]],
    memory: list[dict[str, Any]],
    feedback: dict[str, Any],
) -> dict[str, Any] | None:
    """Point to the latest receipt, including failures; never fall back to success."""
    if not actions:
        return None
    latest = actions[-1]
    if feedback.get("action_index") != latest["action_index"]:
        raise ValueError("latest feedback identity differs")
    prior = next(
        (row for row in reversed(memory) if row.get("action_index") == latest["action_index"]), {}
    )
    result = dict(
        action_index=latest["action_index"],
        status=latest["status"],
        origin=latest.get("origin"),
        failure_type=latest.get("failure_type"),
        original_questions=prior.get("questions", []),
        result_location="previous_feedback",
        delivery_location="delivery_feedback",
        authority="process_metadata_not_diagnostic_evidence",
    )
    return decode_json_object(canonical_json_bytes(result), CONTEXT_LIMITS)
