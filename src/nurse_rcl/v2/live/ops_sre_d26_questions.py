"""Bounded public task questions projected from the original conversation.

Question state is an agent statement, never a new observation, fact authority,
diagnostic conclusion or program-selected next action.
"""

# ruff: noqa: RUF001 -- model-facing Chinese task goal.

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from nurse_rcl.v2.exchange_json import ExchangeBoundaryError

QUESTION_SCHEMA = "nurse.v2.sre-question-state/1"
AUTHORITY = "agent_statement_not_diagnostic_evidence"
GOAL = "解释唯一正式定位的实际依据，调查支持或削弱故障解释的证据及未决边界"
ROLES = {"boundary", "symptoms", "mechanisms", "dependency", "timing", "scoring"}
STATES = {"open", "answered", "blocked", "deferred", "superseded"}
UNRESOLVED = {"open", "blocked", "deferred"}
QUESTION_FIELDS = {
    "question_id",
    "parent_question_id",
    "role",
    "question",
    "target_service_keys",
    "state",
    "basis_refs",
    "remaining_gap",
    "superseded_by",
}
IDENTITY_FIELDS = {"question_id", "parent_question_id", "role", "question", "target_service_keys"}
BASIS_FIELDS = {"action_index", "card_id", "record_sha256", "delivery_kind"}


def _text(value: Any) -> bool:
    return type(value) is str and bool(value.strip())


def _id(value: Any) -> bool:
    # Deliberately distinct from typed request handles Q and report f/b aliases.
    return type(value) is str and re.fullmatch(r"iq_[a-z0-9_]{1,32}", value) is not None


def reachable_card_ids(context: dict[str, Any]) -> set[str]:
    """Only the current admitted request's navigation; no source loading."""
    from nurse_rcl.v2.live.ops_sre_delivered_navigation import navigation_targets

    caps = context.get("capabilities", {})
    result = {r["card_id"] for r in caps.get("published_node_summaries", [])}
    result.update(navigation_targets(context))
    result.update(
        k
        for r in context.get("investigation_routes", {}).get("entries", [])
        for k in r.get("card_ids", [])
    )
    result.update(
        r["card_id"]
        for c in caps.get("investigation_situation", {}).get("candidates", [])
        for r in c.get("scoring", {}).get("records", [])
    )
    result.update(k for r in context.get("actions", []) for k in r.get("visible_card_ids", []))
    return result


def _question(row: Any, context: dict[str, Any] | None) -> None:
    error = "sre_question_shape"
    if type(row) is not dict or set(row) != QUESTION_FIELDS:
        raise ExchangeBoundaryError(error)
    if (
        not _id(row["question_id"])
        or (row["parent_question_id"] is not None and not _id(row["parent_question_id"]))
        or type(row["role"]) is not str
        or row["role"] not in ROLES
        or not _text(row["question"])
        or type(row["state"]) is not str
        or row["state"] not in STATES
        or type(row["remaining_gap"]) is not str
        or (row["state"] in UNRESOLVED and not _text(row["remaining_gap"]))
        or (row["superseded_by"] is not None and not _id(row["superseded_by"]))
        or (row["state"] == "superseded") != (row["superseded_by"] is not None)
    ):
        raise ExchangeBoundaryError(error)
    targets = row["target_service_keys"]
    if (
        type(targets) is not list
        or len(targets) > 2
        or any(not _text(k) for k in targets)
        or len(set(targets)) != len(targets)
    ):
        raise ExchangeBoundaryError(error)
    if context is not None:
        candidates = {
            c["candidate_service_key"]
            for c in context.get("capabilities", {})
            .get("investigation_situation", {})
            .get("candidates", [])
        }
        if not set(targets) <= candidates:
            raise ExchangeBoundaryError("sre_question_target_not_visible")
    refs = row["basis_refs"]
    if type(refs) is not list or len(refs) > 16:
        raise ExchangeBoundaryError(error)
    seen: set[tuple[Any, ...]] = set()
    for ref in refs:
        if (
            type(ref) is not dict
            or set(ref) != BASIS_FIELDS
            or type(ref["action_index"]) is not int
            or not 1 <= ref["action_index"] <= 8
            or not _text(ref["card_id"])
            or not _text(ref["record_sha256"])
            or type(ref["delivery_kind"]) is not str
            or ref["delivery_kind"] not in {"complete", "brief", "partial"}
        ):
            raise ExchangeBoundaryError(error)
        identity = tuple(ref[k] for k in sorted(BASIS_FIELDS))
        if identity in seen:
            raise ExchangeBoundaryError(error)
        seen.add(identity)
        if context is None:
            continue
        actions = {r["action_index"]: r for r in context.get("actions", [])}
        action = actions.get(ref["action_index"], {})
        if (
            ref["action_index"] >= context["decision_index"]
            or action.get("status") != "completed"
            or ref["card_id"] not in action.get("visible_card_ids", [])
            or not any(
                all(r.get(k) == ref[k] for k in ("card_id", "record_sha256", "delivery_kind"))
                and ref["action_index"] in r.get("action_indices", [])
                for r in context.get("delivered_record_index", {}).get("records", [])
            )
        ):
            raise ExchangeBoundaryError("sre_question_basis_not_delivered")


def _relations(rows: dict[str, dict[str, Any]]) -> None:
    for field in ("parent_question_id", "superseded_by"):
        for key in rows:
            seen = {key}
            next_key = rows[key][field]
            while next_key is not None:
                if next_key not in rows or next_key in seen:
                    raise ExchangeBoundaryError("sre_question_relation")
                seen.add(next_key)
                next_key = rows[next_key][field]


def apply_question_plan(
    state: dict[str, Any],
    plan: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
    decision_index: int | None = None,
) -> dict[str, Any]:
    """Apply explicit updates without solving, hiding, selecting or rewriting questions."""
    error = "sre_question_plan_shape"
    if (
        type(plan) is not dict
        or set(plan) != {"updates", "current_question_id", "action_purpose"}
        or type(plan["updates"]) is not list
        or len(plan["updates"]) > 8
    ):
        raise ExchangeBoundaryError(error)
    index = context["decision_index"] if context is not None else decision_index
    if type(index) is not int or not 1 <= index <= 8:
        raise ExchangeBoundaryError(error)
    result = deepcopy(state)
    rows = {r["question_id"]: r for r in result["questions"]}
    seen = set()
    for update in plan["updates"]:
        _question(update, context)
        key = update["question_id"]
        if key in seen:
            raise ExchangeBoundaryError(error)
        seen.add(key)
        previous = rows.get(key)
        if previous is not None and any(previous[k] != update[k] for k in IDENTITY_FIELDS):
            raise ExchangeBoundaryError("sre_question_identity_changed")
        if (
            previous is not None
            and previous["state"] == "superseded"
            and (
                update["state"] != "superseded"
                or update["superseded_by"] != previous["superseded_by"]
            )
        ):
            raise ExchangeBoundaryError("sre_question_relation")
        rows[key] = dict(
            deepcopy(update),
            first_decision_index=(
                previous["first_decision_index"] if previous is not None else index
            ),
            last_update_decision_index=index,
        )
    if len(rows) > 8:
        raise ExchangeBoundaryError("sre_question_limit")
    if not rows:
        raise ExchangeBoundaryError("sre_question_missing")
    _relations(rows)
    current, purpose = plan["current_question_id"], plan["action_purpose"]
    if current is None:
        if purpose is not None:
            raise ExchangeBoundaryError(error)
    elif not _id(current) or current not in rows or rows[current]["state"] == "superseded":
        raise ExchangeBoundaryError("sre_current_question_missing")
    elif (
        type(purpose) is not dict
        or set(purpose) != {"card_ids", "discriminating_value", "why_now"}
        or not _text(purpose["discriminating_value"])
        or not _text(purpose["why_now"])
        or type(purpose["card_ids"]) is not list
        or len(purpose["card_ids"]) > 16
        or any(not _text(k) for k in purpose["card_ids"])
        or len(set(purpose["card_ids"])) != len(purpose["card_ids"])
    ):
        raise ExchangeBoundaryError(error)
    elif context is not None and not set(purpose["card_ids"]) <= reachable_card_ids(context):
        raise ExchangeBoundaryError("sre_question_entry_not_visible")
    result.update(questions=list(rows.values()), current_question_id=current)
    return result


def validate_stop_assessment(
    state: dict[str, Any],
    value: Any,
    *,
    context: dict[str, Any] | None = None,
    entry_field: str = "card_id",
) -> None:
    """Check coverage and visible entries, not whether the model's reason is persuasive."""
    error = "sre_stop_assessment"
    if (
        context is not None
        and context.get("schema_version") == "nurse.v2.sre-investigation-input/20"
    ):
        entry_field = "entry_ref"
    if entry_field not in {"card_id", "entry_ref"}:
        raise ExchangeBoundaryError(error)
    if (
        type(value) is not dict
        or set(value) != {"completed_scope", "remaining_questions", "unchecked_scope"}
        or not _text(value["completed_scope"])
        or type(value["unchecked_scope"]) is not list
        or len(value["unchecked_scope"]) > 8
        or any(not _text(s) for s in value["unchecked_scope"])
        or type(value["remaining_questions"]) is not list
        or len(value["remaining_questions"]) > 8
    ):
        raise ExchangeBoundaryError(error)
    expected = {r["question_id"] for r in state["questions"] if r["state"] in UNRESOLVED}
    seen = set()
    for row in value["remaining_questions"]:
        if (
            type(row) is not dict
            or set(row) != {"question_id", "why_stop", "entry_refs"}
            or not _id(row["question_id"])
            or row["question_id"] in seen
            or not _text(row["why_stop"])
            or type(row["entry_refs"]) is not list
            or len(row["entry_refs"]) > 16
        ):
            raise ExchangeBoundaryError(error)
        seen.add(row["question_id"])
        entries = set()
        for ref in row["entry_refs"]:
            if (
                type(ref) is not dict
                or set(ref) != {entry_field}
                or not _text(ref[entry_field])
                or ref[entry_field] in entries
            ):
                raise ExchangeBoundaryError(error)
            entries.add(ref[entry_field])
        if context is not None:
            available = (
                reachable_card_ids(context)
                if entry_field == "card_id"
                else {e["entry_ref"] for e in context.get("action_entries", {}).get("entries", [])}
            )
            if not entries <= available:
                raise ExchangeBoundaryError(error)
    if seen != expected:
        raise ExchangeBoundaryError(error)


def project_question_state(
    memory: list[dict[str, Any]], *, entry_field: str = "card_id"
) -> dict[str, Any]:
    """No migration: old public text without a plan remains old public text."""
    state: dict[str, Any] = dict(
        schema_version=QUESTION_SCHEMA,
        authority=AUTHORITY,
        goal=GOAL,
        questions=[],
        current_question_id=None,
        latest_action_question_id=None,
        stop_assessment=None,
    )
    if len(memory) > 8:
        raise ExchangeBoundaryError("sre_question_limit")
    for index, row in enumerate(memory, 1):
        if row.get("decision_index") != index or row.get("role") != "agent_statement":
            raise ExchangeBoundaryError("sre_question_history")
        if "investigation_plan" not in row:
            if "annotation_admission" in row:
                # A real action with an absent/rejected note has no accepted question binding.
                state["current_question_id"] = None
                if row.get("action_index") is not None:
                    state["latest_action_question_id"] = None
                state["stop_assessment"] = None
            continue
        state = apply_question_plan(state, row["investigation_plan"], decision_index=index)
        if row.get("action_index") is not None:
            state["latest_action_question_id"] = state["current_question_id"]
        assessment = row.get("stop_assessment")
        if assessment is not None:
            validate_stop_assessment(state, assessment, entry_field=entry_field)
        state["stop_assessment"] = deepcopy(assessment)
    return state


def validate_question_state(state: Any, *, context: dict[str, Any]) -> None:
    """Transport checks only; the owner independently projects the full conversation."""
    if (
        type(state) is not dict
        or set(state)
        != {
            "schema_version",
            "authority",
            "goal",
            "questions",
            "current_question_id",
            "latest_action_question_id",
            "stop_assessment",
        }
        or (state["schema_version"], state["authority"], state["goal"])
        != (
            QUESTION_SCHEMA,
            AUTHORITY,
            GOAL,
        )
        or type(state["questions"]) is not list
        or len(state["questions"]) > 8
    ):
        raise ExchangeBoundaryError("sre_question_state_shape")
    rows = {}
    for row in state["questions"]:
        if type(row) is not dict or set(row) != QUESTION_FIELDS | {
            "first_decision_index",
            "last_update_decision_index",
        }:
            raise ExchangeBoundaryError("sre_question_state_shape")
        _question({k: row[k] for k in QUESTION_FIELDS}, context)
        if row["question_id"] in rows or (
            type(row["first_decision_index"]) is not int
            or type(row["last_update_decision_index"]) is not int
            or not 1
            <= row["first_decision_index"]
            <= row["last_update_decision_index"]
            < context["decision_index"]
        ):
            raise ExchangeBoundaryError("sre_question_state_index")
        rows[row["question_id"]] = row
    _relations(rows)
    for key in ("current_question_id", "latest_action_question_id"):
        if state[key] is not None and state[key] not in rows:
            raise ExchangeBoundaryError("sre_question_state_identity")
    if state["stop_assessment"] is not None:
        validate_stop_assessment(state, state["stop_assessment"], context=context)
