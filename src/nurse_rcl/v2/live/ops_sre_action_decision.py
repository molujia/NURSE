"""Action-first B contract: public notes never grant action or fact authority."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_sre_d23_decision import CONTENT_RELATIONS, validate_balanced_facts
from nurse_rcl.v2.live.ops_sre_d26_questions import (
    QUESTION_FIELDS,
    ROLES,
    apply_question_plan,
    validate_stop_assessment,
)
from nurse_rcl.v2.live.ops_sre_feedback_context import validate_latest_result_update

ACTION_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/18"
ACTION_FIELDS = {
    "actions",
    "stop_reason",
    "public_summary",
    "unresolved",
    "responds_to",
    "stop_basis",
    "latest_result_update",
    "investigation_note",
}
NOTE_FIELDS = {
    "role",
    "target_service_keys",
    "why_now",
    "discriminating_value",
    "basis_refs",
    "question_updates",
    "stop_assessment",
}
QUESTION_UPDATE_FIELDS = {"question_id", "state", "remaining_gap", "basis_refs"}
QUESTION_UPDATE_STATES = frozenset({"open", "answered", "blocked", "deferred"})
STOP_BASES = frozenset({"no_useful_check", "budget_exhausted", "unavailable"})


def parse_action_decision(raw: bytes, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
    return _parse_action_decision(raw, context=context, input_schema=ACTION_INPUT_SCHEMA)


def _parse_action_decision(
    raw: bytes,
    *,
    context: dict[str, Any] | None,
    input_schema: str,
) -> dict[str, Any]:
    """Return the proposal untouched; ledger/tool admission is still independent."""
    from nurse_rcl.v2.live.ops_sre_context import parse_sre_decision

    value = decode_json_object(raw, CONTEXT_LIMITS)
    if set(value) != ACTION_FIELDS:
        raise ExchangeBoundaryError("sre_action_decision_shape")
    base = {
        k: value[k]
        for k in (
            "actions",
            "stop_reason",
            "public_summary",
            "unresolved",
            "responds_to",
        )
    }
    parse_sre_decision(canonical_json_bytes(base), context=context)
    basis = value["stop_basis"]
    if basis is not None and (type(basis) is not str or basis not in STOP_BASES):
        raise ExchangeBoundaryError("sre_stop_basis")
    if value["actions"] and (value["stop_reason"] is not None or basis is not None):
        raise ExchangeBoundaryError("sre_action_continue_shape")
    if not value["actions"] and value["stop_reason"] is None:
        raise ExchangeBoundaryError("sre_action_stop_shape")
    update = value["latest_result_update"]
    if update is not None:
        if (
            type(update) is not dict
            or set(update)
            != {
                "action_index",
                "answer_status",
                "summary",
                "remaining_question",
                "content_relation",
            }
            or type(update["content_relation"]) is not str
            or update["content_relation"] not in CONTENT_RELATIONS
        ):
            raise ExchangeBoundaryError("sre_content_relation_shape")
        base["latest_result_update"] = {k: v for k, v in update.items() if k != "content_relation"}
    else:
        base["latest_result_update"] = None
    validate_latest_result_update(base, context=context)
    if context is not None:
        if context.get("schema_version") != input_schema:
            raise ExchangeBoundaryError("sre_action_context_required")
        validate_balanced_facts(value, context)
    return value


def bind_note_sources(refs: Any, *, context: dict[str, Any]) -> list[dict[str, Any]]:
    """Bind only a unique, previously delivered record; no overview or later lookup."""
    if type(refs) is not list or len(refs) > 16:
        raise ExchangeBoundaryError("sre_note_source_shape")
    result, seen = [], set()
    actions = {r["action_index"]: r for r in context.get("actions", [])}
    for ref in refs:
        if (
            type(ref) is not dict
            or set(ref) != {"action_index", "card_id"}
            or type(ref["action_index"]) is not int
            or type(ref["card_id"]) is not str
        ):
            raise ExchangeBoundaryError("sre_note_source_shape")
        index, key = ref["action_index"], ref["card_id"]
        if (index, key) in seen:
            raise ExchangeBoundaryError("sre_note_source_shape")
        seen.add((index, key))
        action = actions.get(index, {})
        matches = [
            r
            for r in context.get("delivered_record_index", {}).get("records", [])
            if r["card_id"] == key and index in r["action_indices"]
        ]
        if (
            not 1 <= index < context["decision_index"]
            or action.get("status") != "completed"
            or key not in action.get("visible_card_ids", [])
            or len(matches) != 1
        ):
            raise ExchangeBoundaryError("sre_note_source_not_uniquely_delivered")
        source = matches[0]
        result.append(
            dict(
                action_index=index,
                card_id=key,
                record_sha256=source["record_sha256"],
                delivery_kind=source["delivery_kind"],
            )
        )
    return result


def compile_note_plan(
    note: dict[str, Any],
    actions: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Compile explicit question proposals, with no latest-answer ownership."""
    if type(note) is not dict or set(note) != NOTE_FIELDS:
        raise ExchangeBoundaryError("sre_note_shape")
    if type(note["role"]) is not str or note["role"] not in ROLES:
        raise ExchangeBoundaryError("sre_note_shape")
    for field in ("why_now", "discriminating_value"):
        if type(note[field]) is not str or not note[field].strip():
            raise ExchangeBoundaryError("sre_note_shape")
    targets = note["target_service_keys"]
    candidates = {
        r["candidate_service_key"]
        for r in context["capabilities"].get("investigation_situation", {}).get("candidates", [])
    }
    if (
        type(targets) is not list
        or len(targets) > 2
        or any(type(k) is not str for k in targets)
        or len(set(targets)) != len(targets)
        or not set(targets) <= candidates
    ):
        raise ExchangeBoundaryError("sre_question_target_not_visible")
    refs = bind_note_sources(note["basis_refs"], context=context)
    previous = {r["question_id"]: r for r in state["questions"]}
    updates = note["question_updates"]
    if type(updates) is not list or len(updates) > 8:
        raise ExchangeBoundaryError("sre_note_shape")
    rows = []
    for update in updates:
        if (
            type(update) is not dict
            or set(update) != QUESTION_UPDATE_FIELDS
            or type(update["question_id"]) is not str
            or update["question_id"] not in previous
            or type(update["state"]) is not str
            or update["state"] not in QUESTION_UPDATE_STATES
        ):
            raise ExchangeBoundaryError("sre_note_update_shape")
        row = {k: deepcopy(previous[update["question_id"]][k]) for k in QUESTION_FIELDS}
        row.update(
            state=update["state"],
            remaining_gap=update["remaining_gap"],
            basis_refs=bind_note_sources(update["basis_refs"], context=context),
        )
        rows.append(row)
    if actions:
        if (
            len(actions) != 1
            or type(actions[0]) is not dict
            or (type(actions[0].get("question")) is not str or not actions[0]["question"].strip())
        ):
            raise ExchangeBoundaryError("sre_note_action_question")
        key = f"iq_d{context['decision_index']}"
        if key in previous:
            raise ExchangeBoundaryError("sre_note_question_identity")
        rows.append(
            dict(
                question_id=key,
                parent_question_id=None,
                role=note["role"],
                question=actions[0]["question"],
                target_service_keys=targets,
                state="open",
                basis_refs=refs,
                remaining_gap=actions[0]["question"],
                superseded_by=None,
            )
        )
        if note["stop_assessment"] is not None:
            raise ExchangeBoundaryError("sre_note_continue_assessment")
        purpose = dict(
            card_ids=[], why_now=note["why_now"], discriminating_value=note["discriminating_value"]
        )
    else:
        key, purpose = None, None
    plan = dict(updates=rows, current_question_id=key, action_purpose=purpose)
    return plan


def _note_plan(value: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    note, actions, state = value["investigation_note"], value["actions"], context["question_state"]
    plan = compile_note_plan(note, actions, state, context=context)
    rows = plan["updates"]
    bound = apply_question_plan(state, plan, context=context)
    if context.get("schema_version") == "nurse.v2.sre-investigation-input/20":
        # Only an explicit update to this same bound question is comparable.
        # A narrow read can be answered while a different diagnostic gap remains;
        # neither delivery completeness nor prose decides that question for us.
        latest = state.get("latest_action_question_id")
        feedback = value["latest_result_update"]
        if latest is not None and feedback is not None:
            for row in rows:
                if row["question_id"] == latest and (
                    (row["state"] == "answered") != (feedback["answer_status"] == "answered")
                ):
                    raise ExchangeBoundaryError("sre_note_latest_question_status")
    if not actions:
        validate_stop_assessment(bound, note["stop_assessment"], context=context)
    return plan


def admit_note(value: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
    """A rejected annotation remains rejected, without buying a repair or changing an action."""
    if value["investigation_note"] is None:
        return dict(status="absent", code=None, plan=None)
    try:
        plan = _note_plan(value, context)
    except ExchangeBoundaryError as exc:
        return dict(status="rejected", code=exc.code, plan=None)
    return dict(status="accepted", code=None, plan=plan)


def note_memory(
    value: dict[str, Any], *, context: dict[str, Any], references: dict[str, Any] | None = None
) -> dict[str, Any]:
    if context.get("schema_version") in {
        "nurse.v2.sre-investigation-input/23",
        "nurse.v2.sre-investigation-input/24",
        "nurse.v2.sre-investigation-input/26",
    }:
        from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
        from nurse_rcl.v2.live.ops_sre_typed_purpose_questions import admit_typed_purpose_note

        if context.get("schema_version") == "nurse.v2.sre-investigation-input/24":
            from nurse_rcl.v2.live.ops_sre_atomic_selector_questions import (
                admit_atomic_note as admit_typed_purpose_note,
            )
        if context.get("schema_version") == "nurse.v2.sre-investigation-input/26":
            from nurse_rcl.v2.live.ops_sre_focused_questions import (
                admit_focused_note as admit_typed_purpose_note,
            )

        book = (
            None
            if references is None
            else SreActionReferenceBook.from_state(
                references, snapshot=references["snapshot"], run_id=references["run_id"]
            )
        )
        admission = admit_typed_purpose_note(value, context=context, book=book)
    elif context.get("schema_version") == "nurse.v2.sre-investigation-input/22":
        from nurse_rcl.v2.live.ops_sre_single_owner_decision import admit_single_owner_note

        admission = admit_single_owner_note(value, context=context)
    elif context.get("schema_version") == "nurse.v2.sre-investigation-input/21":
        from nurse_rcl.v2.live.ops_sre_observable_scope import admit_observable_note

        admission = admit_observable_note(value, context=context)
    else:
        admission = admit_note(value, context=context)
    result = dict(
        annotation_admission={k: admission[k] for k in ("status", "code")},
        stop_basis=value["stop_basis"],
    )
    if admission["status"] == "accepted":
        result.update(
            investigation_plan=admission["plan"],
            stop_assessment=deepcopy(value["investigation_note"]["stop_assessment"]),
        )
    return result
