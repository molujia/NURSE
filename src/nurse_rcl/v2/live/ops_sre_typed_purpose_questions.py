"""Explicit lifecycle with program scope titles. No IO, diagnosis or legacy adapters."""

from copy import deepcopy
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.live.ops_sre_action_decision import QUESTION_UPDATE_STATES, bind_note_sources
from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
from nurse_rcl.v2.live.ops_sre_d26_questions import (
    GOAL,
    IDENTITY_FIELDS,
    QUESTION_FIELDS,
    ROLES,
    _id,
    _question,
    _relations,
    _text,
    validate_stop_assessment,
)
from nurse_rcl.v2.live.ops_sre_observable_scope import _scoped_row, admit_observable_test
from nurse_rcl.v2.live.ops_sre_typed_purpose_decision import TYPED_INPUT_SCHEMA

QUESTION_SCHEMA = "nurse.v2.sre-question-state/3"
AUTHORITY = "explicit_agent_lifecycle_with_program_scope_not_diagnostic_evidence"
PROGRAM_ORIGIN = "program_from_admitted_observable_scope"
NAV_ORIGIN = "agent_navigation_question"
ROW_FIELDS = QUESTION_FIELDS | {"observable_scope", "remaining_facets", "question_origin"}
NOTE_FIELDS = {
    "role",
    "target_service_keys",
    "why_now",
    "basis_refs",
    "question_updates",
    "stop_assessment",
    "observable_test",
}
STATE_FIELDS = {
    "schema_version",
    "authority",
    "goal",
    "questions",
    "current_question_id",
    "latest_action_question_id",
    "stop_assessment",
}


def render_scope_title(
    scope: dict[str, Any],
    *,
    book: SreActionReferenceBook | None,
    decision_index: int,
    short_refs: bool = False,
) -> str:
    """Original frozen book only; public handles are not re-issued on later turns."""
    targets = scope["binding"]["targets"]
    if not short_refs:
        if type(book) is not SreActionReferenceBook:
            raise ExchangeBoundaryError("typed_title_references_required")
        if book.source_prefix(visible_at=decision_index) != scope["binding"]["source_prefix"]:
            raise ExchangeBoundaryError("typed_title_reference_binding")
        try:
            if (
                book.resolve(scope["entry_ref"], "action", visible_at=decision_index)
                != scope["entry_id"]
            ):
                raise ExchangeBoundaryError("typed_title_reference_binding")
            handles = {
                r[2]: r[1]
                for r in book.to_state()["entries"]
                if r[0] == "service" and r[3] <= decision_index
            }
            targets = [handles[t] for t in targets]
        except (ValueError, KeyError) as exc:
            raise ExchangeBoundaryError("typed_title_reference_binding") from exc

    def dump(value: Any) -> str:
        return canonical_json_bytes(value).decode()

    title = f"Inspect registered {scope['objective_kind']} for {dump(targets)}, "
    if scope["objective_kind"] == "metric_summary":
        title += f"metric {dump(scope['metric_name'])}, "
    return title + f"facets {dump(scope['requested_facets'])}, entry {dump(scope['entry_ref'])}."


def _row(row: Any, context: dict[str, Any] | None, *, short_refs: bool = False) -> None:
    if type(row) is not dict or set(row) != ROW_FIELDS:
        raise ExchangeBoundaryError("typed_question_shape")
    _question({k: row[k] for k in QUESTION_FIELDS}, context)
    _scoped_row(
        {k: row[k] for k in QUESTION_FIELDS | {"observable_scope", "remaining_facets"}},
        short_refs=short_refs,
    )
    if row["question_origin"] != (
        PROGRAM_ORIGIN if row["observable_scope"] is not None else NAV_ORIGIN
    ):
        raise ExchangeBoundaryError("typed_question_origin")
    if (
        row["parent_question_id"] is not None
        or row["state"] not in QUESTION_UPDATE_STATES
        or row["superseded_by"] is not None
    ):
        raise ExchangeBoundaryError("typed_question_shape")


def _state(
    state: Any,
    *,
    decision_index: int,
    context: dict[str, Any] | None = None,
    short_refs: bool = False,
) -> None:
    """Exact typed transport/state shape, without manufacturing a legacy state."""
    if (
        type(state) is not dict
        or set(state) != STATE_FIELDS
        or (state["schema_version"], state["authority"], state["goal"])
        != (QUESTION_SCHEMA, AUTHORITY, GOAL)
        or type(state["questions"]) is not list
        or len(state["questions"]) > 8
        or type(decision_index) is not int
        or not 1 <= decision_index <= 9
    ):
        raise ExchangeBoundaryError("typed_question_state_shape")
    rows = {}
    for row in state["questions"]:
        if type(row) is not dict or set(row) != ROW_FIELDS | {
            "first_decision_index",
            "last_update_decision_index",
        }:
            raise ExchangeBoundaryError("typed_question_state_shape")
        _row({k: row[k] for k in ROW_FIELDS}, context, short_refs=short_refs)
        first, last = row["first_decision_index"], row["last_update_decision_index"]
        if (
            type(first) is not int
            or type(last) is not int
            or not 1 <= first <= last < decision_index
            or row["question_id"] != f"iq_d{first}"
            or row["question_id"] in rows
        ):
            raise ExchangeBoundaryError("typed_question_state_index")
        rows[row["question_id"]] = row
    _relations(rows)
    for key in ("current_question_id", "latest_action_question_id"):
        if state[key] is not None and (not _id(state[key]) or state[key] not in rows):
            raise ExchangeBoundaryError("typed_question_state_identity")
    if state["stop_assessment"] is not None:
        validate_stop_assessment(
            state, state["stop_assessment"], context=context, entry_field="entry_ref"
        )


def validate_typed_questions(
    state: Any,
    *,
    context: dict[str, Any],
    book: SreActionReferenceBook | None = None,
    short_refs: bool = False,
) -> None:
    """Full titles use the original owner book; short titles only check display consistency."""
    _state(state, context=context, decision_index=context["decision_index"], short_refs=short_refs)
    for row in state["questions"]:
        if row["observable_scope"] is not None and row["question"] != render_scope_title(
            row["observable_scope"],
            book=book,
            decision_index=row["first_decision_index"],
            short_refs=short_refs,
        ):
            raise ExchangeBoundaryError("typed_question_title")


def apply_typed_plan(
    state: dict[str, Any],
    plan: dict[str, Any],
    *,
    decision_index: int,
    context: dict[str, Any] | None = None,
    short_refs: bool = False,
) -> dict[str, Any]:
    _state(state, decision_index=decision_index, context=context, short_refs=short_refs)
    if (
        type(plan) is not dict
        or set(plan) != {"updates", "current_question_id", "action_purpose"}
        or type(plan["updates"]) is not list
        or len(plan["updates"]) > 8
        or type(decision_index) is not int
        or not 1 <= decision_index <= 8
    ):
        raise ExchangeBoundaryError("sre_question_plan_shape")
    result = deepcopy(state)
    rows = {r["question_id"]: r for r in result["questions"]}
    seen = set()
    for update in plan["updates"]:
        _row(update, context, short_refs=short_refs)
        key = update["question_id"]
        if key in seen:
            raise ExchangeBoundaryError("sre_question_plan_shape")
        seen.add(key)
        previous = rows.get(key)
        if previous is not None and any(
            previous[k] != update[k]
            for k in IDENTITY_FIELDS | {"observable_scope", "question_origin"}
        ):
            raise ExchangeBoundaryError("sre_question_identity_changed")
        if previous is None and (
            key != f"iq_d{decision_index}"
            or update["parent_question_id"] is not None
            or update["state"] != "open"
        ):
            raise ExchangeBoundaryError("sre_note_question_identity")
        if previous is None and (
            update["remaining_gap"] != update["question"]
            or update["remaining_facets"]
            != (
                update["observable_scope"]["requested_facets"]
                if update["observable_scope"] is not None
                else None
            )
        ):
            raise ExchangeBoundaryError("sre_note_question_identity")
        if update["state"] not in QUESTION_UPDATE_STATES or update["superseded_by"] is not None:
            raise ExchangeBoundaryError("sre_note_update_shape")
        rows[key] = dict(
            deepcopy(update),
            first_decision_index=previous["first_decision_index"] if previous else decision_index,
            last_update_decision_index=decision_index,
        )
    if len(rows) > 8:
        raise ExchangeBoundaryError("sre_question_limit")
    _relations(rows)
    current, purpose = plan["current_question_id"], plan["action_purpose"]
    if current is None:
        if purpose is not None:
            raise ExchangeBoundaryError("sre_question_plan_shape")
    elif (
        not _id(current)
        or current not in rows
        or type(purpose) is not dict
        or set(purpose) != {"question_id", "why_now"}
        or purpose["question_id"] != current
        or not _text(purpose["why_now"])
    ):
        raise ExchangeBoundaryError("sre_question_plan_shape")
    result.update(questions=list(rows.values()), current_question_id=current)
    # An explicit update can legitimately invalidate the previous stop assessment.
    # The caller attaches only this decision's independently validated assessment.
    result["stop_assessment"] = None
    _state(result, decision_index=decision_index + 1, short_refs=short_refs)
    return result


def project_typed_questions(
    memory: list[dict[str, Any]],
    *,
    short_refs: bool = False,
) -> dict[str, Any]:
    state = dict(
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
                state["current_question_id"] = None
                if row.get("action_index") is not None:
                    state["latest_action_question_id"] = None
                state["stop_assessment"] = None
            continue
        state = apply_typed_plan(
            state, row["investigation_plan"], decision_index=index, short_refs=short_refs
        )
        if row.get("action_index") is not None:
            state["latest_action_question_id"] = state["current_question_id"]
        assessment = row.get("stop_assessment")
        if assessment is not None:
            validate_stop_assessment(state, assessment, entry_field="entry_ref")
        state["stop_assessment"] = deepcopy(assessment)
    return state


def admit_typed_purpose_note(
    value: dict[str, Any],
    *,
    context: dict[str, Any],
    book: SreActionReferenceBook | None = None,
    short_refs: bool = False,
) -> dict[str, Any]:
    if context.get("schema_version") != TYPED_INPUT_SCHEMA:
        raise ExchangeBoundaryError("typed_purpose_context_required")
    from nurse_rcl.v2.live.ops_sre_typed_purpose_decision import selection_shape

    return admit_purpose_note(
        value,
        context=context,
        book=book,
        short_refs=short_refs,
        selection_validator=selection_shape,
    )


def admit_purpose_note(
    value: dict[str, Any],
    *,
    context: dict[str, Any],
    selection_validator: Any,
    book: SreActionReferenceBook | None = None,
    short_refs: bool = False,
    atomic_selector: bool = False,
    focused: bool = False,
) -> dict[str, Any]:
    """Version-neutral scope/lifecycle core; wrappers own native and input identity."""
    note = value["investigation_note"]
    if note is None:
        return dict(status="absent", code=None, plan=None)
    try:
        if (
            type(note) is not dict
            or set(note) != NOTE_FIELDS
            or type(note["role"]) is not str
            or note["role"] not in ROLES
            or not _text(note["why_now"])
        ):
            raise ExchangeBoundaryError("typed_note_shape")
        actions = value["actions"]
        if type(actions) is not list or len(actions) > 1:
            raise ExchangeBoundaryError("sre_note_action_question")
        objective = admit_observable_test(
            note["observable_test"],
            selection=actions[0] if actions else None,
            table=context["action_entries"],
            _typed_purpose=not atomic_selector and not focused,
            _atomic_selector=atomic_selector and not focused,
            _focused=focused,
        )
        if objective["status"] != "accepted":
            raise ExchangeBoundaryError(objective["code"])
        scope = objective["scope"]
        targets = note["target_service_keys"]
        candidates = {
            r["candidate_service_key"]
            for r in context["capabilities"]
            .get("investigation_situation", {})
            .get("candidates", [])
        }
        if (
            type(targets) is not list
            or len(targets) > 2
            or any(not _text(t) for t in targets)
            or len(set(targets)) != len(targets)
            or not set(targets) <= candidates
        ):
            raise ExchangeBoundaryError("sre_question_target_not_visible")
        if scope is not None and targets != scope["binding"]["targets"]:
            raise ExchangeBoundaryError("objective_not_supplied")
        # The shared row validator applies the original visible-target/source checks.
        refs = bind_note_sources(note["basis_refs"], context=context)
        previous = {r["question_id"]: r for r in context["question_state"]["questions"]}
        updates = note["question_updates"]
        if type(updates) is not list or len(updates) > 8:
            raise ExchangeBoundaryError("sre_note_update_shape")
        rows = []
        for update in updates:
            if (
                type(update) is not dict
                or set(update)
                != {"question_id", "state", "remaining_gap", "basis_refs", "remaining_facets"}
                or type(update["question_id"]) is not str
                or update["question_id"] not in previous
                or type(update["state"]) is not str
                or update["state"] not in QUESTION_UPDATE_STATES
            ):
                raise ExchangeBoundaryError("sre_note_update_shape")
            row = {k: deepcopy(previous[update["question_id"]][k]) for k in ROW_FIELDS}
            row.update(
                state=update["state"],
                remaining_gap=deepcopy(update["remaining_gap"]),
                remaining_facets=deepcopy(update["remaining_facets"]),
                basis_refs=bind_note_sources(update["basis_refs"], context=context),
            )
            rows.append(row)
        key = None
        if actions:
            selection_validator(actions[0])
            key = f"iq_d{context['decision_index']}"
            if key in previous:
                raise ExchangeBoundaryError("sre_note_question_identity")
            title = (
                render_scope_title(
                    scope,
                    book=book,
                    decision_index=context["decision_index"],
                    short_refs=short_refs,
                )
                if scope is not None
                else actions[0]["question"]
            )
            rows.append(
                dict(
                    question_id=key,
                    parent_question_id=None,
                    role=note["role"],
                    question=title,
                    question_origin=PROGRAM_ORIGIN if scope is not None else NAV_ORIGIN,
                    target_service_keys=deepcopy(targets),
                    state="open",
                    basis_refs=refs,
                    remaining_gap=title,
                    superseded_by=None,
                    observable_scope=deepcopy(scope),
                    remaining_facets=deepcopy(scope["requested_facets"])
                    if scope is not None
                    else None,
                )
            )
            if note["stop_assessment"] is not None:
                raise ExchangeBoundaryError("sre_note_continue_assessment")
        plan = dict(
            updates=rows,
            current_question_id=key,
            action_purpose=dict(question_id=key, why_now=note["why_now"])
            if key is not None
            else None,
        )
        bound = apply_typed_plan(
            context["question_state"],
            plan,
            decision_index=context["decision_index"],
            context=context,
        )
        from nurse_rcl.v2.live.ops_sre_registered_field_delivery import already_supplied_facets

        try:
            contradiction = already_supplied_facets(context, updates)
        except ValueError as exc:
            raise ExchangeBoundaryError("registered_field_delivery_context") from exc
        if contradiction:
            raise ExchangeBoundaryError("remaining_facets_already_supplied")
        if not actions:
            validate_stop_assessment(
                bound, note["stop_assessment"], context=context, entry_field="entry_ref"
            )
        return dict(status="accepted", code=None, plan=plan)
    except ExchangeBoundaryError as exc:
        return dict(status="rejected", code=exc.code, plan=None)
