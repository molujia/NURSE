"""Opt-in registered supply declarations, not evidence or natural-language grading.

This module has no source reader, measurement, provider, retry or dispatch API.
Only current frozen entry metadata can license a purpose; unknown records stay generic.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.live.ops_sre_d26_questions import (
    QUESTION_FIELDS,
    apply_question_plan,
    project_question_state,
    validate_stop_assessment,
)

OBSERVABLE_ENTRY_SCHEMA = "nurse.v2.sre-action-entries/2"
OBSERVABLE_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/21"
OBSERVABLE_PROFILE = "v2-ops-sre/10"
OBSERVABLE_TOOLS = "hermes-sre-tools/20"
CONTRACT_SCHEMA = "nurse.v2.sre-observable-contract/1"
CONTRACT_AUTHORITY = "registered_supply_not_delivered_evidence"
SCOPED_QUESTION_SCHEMA = "nurse.v2.sre-question-state/2"
SCOPE_FIELDS = {
    "entry_ref",
    "entry_id",
    "objective_kind",
    "metric_name",
    "requested_facets",
    "binding",
}
FACETS = {
    "record_read": ["record_contents"],
    "metric_summary": ["baseline_value", "fault_value", "unit", "coverage", "episode_status"],
    "error_summary": ["error_count", "denominator", "fraction", "unit", "coverage"],
}
SUMMARY_TYPES = {
    ("ops-symptoms/1", "fixed_summary"),
    ("sre-prepared-summary/1", "prepared_summary"),
}


def observable_contract(
    context: dict[str, Any],
    binding: dict[str, Any],
    source_prefix: str,
) -> dict[str, Any]:
    """Metadata already admitted by the owner; never infer values or a diagnosis."""
    registered = (context.get("record_version"), context.get("record_kind"))
    metric = context.get("metric_name")
    objective = "record_read"
    if registered == ("ops-symptoms/1", "error_summary"):
        objective = "error_summary"
    elif registered in SUMMARY_TYPES and type(metric) is str and metric.strip():
        objective = "metric_summary"
    return dict(
        schema_version=CONTRACT_SCHEMA,
        authority=CONTRACT_AUTHORITY,
        objective_kind=objective,
        metric_name=metric if objective == "metric_summary" else None,
        allowed_facets=list(FACETS[objective]),
        binding=dict(
            targets=deepcopy(context.get("targets", context.get("scope", []))),
            source_prefix=deepcopy(source_prefix),
            **deepcopy(binding),
        ),
    )


def compile_observable_entries(capabilities: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_action_entries import compile_action_entries

    return compile_action_entries(capabilities, _observable=True, **kwargs)


def admit_observable_test(
    value: Any,
    *,
    selection: Any,
    table: dict[str, Any],
    _single_owner: bool = False,
    _typed_purpose: bool = False,
    _atomic_selector: bool = False,
    _focused: bool = False,
) -> dict[str, Any]:
    """Check explicit structure only; a rejected note cannot change native action/text."""

    def rejected(code: str) -> dict[str, Any]:
        return dict(status="rejected", code=code, scope=None)

    if sum((_single_owner, _typed_purpose, _atomic_selector, _focused)) > 1:
        return rejected("observable_context_required")
    schema = (
        "nurse.v2.sre-action-entries/6"
        if _focused
        else "nurse.v2.sre-action-entries/5"
        if _atomic_selector or _focused
        else "nurse.v2.sre-action-entries/4"
        if _typed_purpose
        else "nurse.v2.sre-action-entries/3"
        if _single_owner
        else OBSERVABLE_ENTRY_SCHEMA
    )
    if table.get("schema_version") != schema:
        return rejected("observable_context_required")
    concrete = (
        type(selection) is str
        if _atomic_selector or _focused
        else (selection is not None and "entry_ref" in selection)
    )
    if not concrete:
        return (
            dict(status="accepted", code=None, scope=None)
            if value is None
            else rejected("objective_not_supplied")
        )
    reference = selection if _atomic_selector or _focused else selection["entry_ref"]
    entries = [e for e in table["entries"] if e["entry_ref"] == reference]
    if len(entries) != 1:
        return rejected("objective_not_supplied")
    entry = entries[0]
    expected = observable_contract(
        entry["source_context"],
        entry["source_binding"],
        table["source_prefix"],
    )
    if entry.get("observable_contract") != expected:
        return rejected("observable_context_required")
    if (
        type(value) is not dict
        or set(value) != {"entry_ref", "objective_kind", "metric_name", "requested_facets"}
        or value["entry_ref"] != reference
        or value["objective_kind"] != expected["objective_kind"]
        or value["metric_name"] != expected["metric_name"]
        or type(value["requested_facets"]) is not list
        or not value["requested_facets"]
        or len(value["requested_facets"]) > len(expected["allowed_facets"])
        or any(
            type(f) is not str or f not in expected["allowed_facets"]
            for f in value["requested_facets"]
        )
        or len(set(value["requested_facets"])) != len(value["requested_facets"])
    ):
        return rejected("objective_not_supplied")
    return dict(
        status="accepted",
        code=None,
        scope=dict(
            entry_ref=entry["entry_ref"],
            entry_id=entry["entry_id"],
            objective_kind=value["objective_kind"],
            metric_name=value["metric_name"],
            requested_facets=deepcopy(value["requested_facets"]),
            binding=expected["binding"],
        ),
    )


def _facets(value: Any, allowed: list[str]) -> bool:
    return (
        type(value) is list
        and len(value) <= len(allowed)
        and all(type(f) is str and f in allowed for f in value)
        and len(value) == len(set(value))
    )


def _scope_binding(value: Any, *, short_refs: bool = False) -> bool:
    """Shape only: owner replay separately re-admits the original frozen entry."""
    if type(value) is not dict:
        return False
    common = {"targets", "source_prefix"}
    record = {"record_sha256", "card_id", "source_refs"}
    route = {"route_binding_sha256"}
    if set(value) not in (common | record, common | route):
        return False
    try:
        require_pin(value["source_prefix"])
        digest = (
            value["record_sha256"] if "record_sha256" in value else value["route_binding_sha256"]
        )
        if short_refs and "record_sha256" in value:
            if type(digest) is not str or re.fullmatch(r"H[1-9][0-9]{0,3}", digest) is None:
                return False
        else:
            require_pin(digest)
    except (ValueError, TypeError):
        return False
    return not (
        "record_sha256" in value
        and (
            type(value["card_id"]) is not str
            or not value["card_id"].strip()
            or type(value["source_refs"]) is not list
            or any(type(ref) is not str or not ref.strip() for ref in value["source_refs"])
        )
    )


def _scoped_row(row: dict[str, Any], *, short_refs: bool = False) -> None:
    if set(row) != QUESTION_FIELDS | {"observable_scope", "remaining_facets"}:
        raise ExchangeBoundaryError("observable_question_scope_shape")
    scope, remaining = row["observable_scope"], row["remaining_facets"]
    if scope is None:
        if remaining is not None:
            raise ExchangeBoundaryError("remaining_facets_outside_request")
        return
    if (
        type(scope) is not dict
        or set(scope) != SCOPE_FIELDS
        or type(scope["entry_ref"]) is not str
        or not scope["entry_ref"].startswith("A")
        or type(scope["objective_kind"]) is not str
        or scope["objective_kind"] not in FACETS
        or (
            scope["objective_kind"] == "metric_summary"
            and (type(scope["metric_name"]) is not str or not scope["metric_name"].strip())
        )
        or (scope["objective_kind"] != "metric_summary" and scope["metric_name"] is not None)
        or not _facets(scope["requested_facets"], FACETS[scope["objective_kind"]])
        or not scope["requested_facets"]
        or not _scope_binding(scope["binding"], short_refs=short_refs)
        or scope["binding"].get("targets") != row["target_service_keys"]
    ):
        raise ExchangeBoundaryError("observable_question_scope_shape")
    try:
        require_pin(scope["entry_id"])
    except (ValueError, TypeError) as exc:
        raise ExchangeBoundaryError("observable_question_scope_shape") from exc
    if not _facets(remaining, scope["requested_facets"]):
        raise ExchangeBoundaryError("remaining_facets_outside_request")
    if row["state"] == "answered" and remaining:
        raise ExchangeBoundaryError("answered_with_remaining_facets")


def _legacy_state(state: dict[str, Any], *, short_refs: bool = False) -> dict[str, Any]:
    if state.get("schema_version") != SCOPED_QUESTION_SCHEMA:
        raise ExchangeBoundaryError("observable_question_context_required")
    result = deepcopy(state)
    result["schema_version"] = "nurse.v2.sre-question-state/1"
    if type(result.get("questions")) is not list:
        raise ExchangeBoundaryError("observable_question_scope_shape")
    for row in result["questions"]:
        if type(row) is not dict or set(row) != QUESTION_FIELDS | {
            "observable_scope",
            "remaining_facets",
            "first_decision_index",
            "last_update_decision_index",
        }:
            raise ExchangeBoundaryError("observable_question_scope_shape")
        _scoped_row(
            {k: row[k] for k in QUESTION_FIELDS | {"observable_scope", "remaining_facets"}},
            short_refs=short_refs,
        )
        row.pop("observable_scope")
        row.pop("remaining_facets")
    return result


def apply_observable_plan(
    state: dict[str, Any],
    plan: dict[str, Any],
    *,
    decision_index: int,
) -> dict[str, Any]:
    """Reuse all original identity/relationship checks; never rewrite stored iq text."""
    legacy = _legacy_state(state)
    original = {row["question_id"]: row for row in state["questions"]}
    if type(plan) is not dict or type(plan.get("updates")) is not list:
        raise ExchangeBoundaryError("observable_question_scope_shape")
    scopes = {}
    for row in plan["updates"]:
        if type(row) is not dict:
            raise ExchangeBoundaryError("observable_question_scope_shape")
        _scoped_row(row)
        previous = original.get(row["question_id"])
        if previous is not None and previous["observable_scope"] != row["observable_scope"]:
            raise ExchangeBoundaryError("observable_question_scope_changed")
        scopes[row["question_id"]] = (row["observable_scope"], row["remaining_facets"])
    narrow = dict(
        deepcopy(plan),
        updates=[{k: deepcopy(row[k]) for k in QUESTION_FIELDS} for row in plan["updates"]],
    )
    result = apply_question_plan(legacy, narrow, decision_index=decision_index)
    for row in result["questions"]:
        scope, remaining = (
            scopes.get(
                row["question_id"],
                (
                    original[row["question_id"]]["observable_scope"],
                    original[row["question_id"]]["remaining_facets"],
                ),
            )
            if row["question_id"] in original
            else scopes[row["question_id"]]
        )
        row.update(observable_scope=deepcopy(scope), remaining_facets=deepcopy(remaining))
    result["schema_version"] = SCOPED_QUESTION_SCHEMA
    return result


def project_observable_questions(memory: list[dict[str, Any]]) -> dict[str, Any]:
    """The original conversation remains the only history; this is a bounded projection."""
    state = dict(project_question_state([]), schema_version=SCOPED_QUESTION_SCHEMA)
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
        state = apply_observable_plan(state, row["investigation_plan"], decision_index=index)
        if row.get("action_index") is not None:
            state["latest_action_question_id"] = state["current_question_id"]
        assessment = row.get("stop_assessment")
        if assessment is not None:
            validate_stop_assessment(state, assessment, entry_field="entry_ref")
        state["stop_assessment"] = deepcopy(assessment)
    return state


def validate_observable_questions(
    state: dict[str, Any],
    *,
    context: dict[str, Any],
    short_refs: bool = False,
) -> None:
    from nurse_rcl.v2.live.ops_sre_d26_questions import validate_question_state

    legacy = _legacy_state(state, short_refs=short_refs)
    validate_question_state(
        legacy,
        context=dict(
            context, schema_version="nurse.v2.sre-investigation-input/20", question_state=legacy
        ),
    )


def admit_observable_note(value: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
    return _admit_scoped_note(value, context=context)


def _admit_scoped_note(
    value: dict[str, Any], *, context: dict[str, Any], _single_owner: bool = False
) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_action_decision import NOTE_FIELDS, admit_note

    schema = "nurse.v2.sre-investigation-input/22" if _single_owner else OBSERVABLE_INPUT_SCHEMA
    if context.get("schema_version") != schema:
        raise ExchangeBoundaryError("observable_context_required")
    if value["investigation_note"] is None:
        return dict(status="absent", code=None, plan=None)
    try:
        note = value["investigation_note"]
        if type(note) is not dict or set(note) != NOTE_FIELDS | {"observable_test"}:
            raise ExchangeBoundaryError("observable_note_shape")
        if type(value["actions"]) is not list or len(value["actions"]) > 1:
            raise ExchangeBoundaryError("sre_note_action_question")
        objective = admit_observable_test(
            note["observable_test"],
            selection=value["actions"][0] if value["actions"] else None,
            table=context["action_entries"],
            _single_owner=_single_owner,
        )
        if objective["status"] != "accepted":
            raise ExchangeBoundaryError(objective["code"])
        scope = objective["scope"]
        if scope is not None and note["target_service_keys"] != scope["binding"]["targets"]:
            raise ExchangeBoundaryError("objective_not_supplied")
        updates = note["question_updates"]
        if type(updates) is not list or len(updates) > 8:
            raise ExchangeBoundaryError("sre_note_update_shape")
        previous = {r["question_id"]: r for r in context["question_state"]["questions"]}
        extended = {}
        for row in updates:
            if (
                type(row) is not dict
                or set(row)
                != {
                    "question_id",
                    "state",
                    "remaining_gap",
                    "basis_refs",
                    "remaining_facets",
                }
                or type(row["question_id"]) is not str
                or row["question_id"] not in previous
            ):
                raise ExchangeBoundaryError("sre_note_update_shape")
            old = previous[row["question_id"]]
            extended[row["question_id"]] = dict(
                observable_scope=deepcopy(old["observable_scope"]),
                remaining_facets=deepcopy(row["remaining_facets"]),
            )
            _scoped_row(
                dict(
                    {k: deepcopy(old[k]) for k in QUESTION_FIELDS},
                    state=row["state"],
                    **extended[row["question_id"]],
                )
            )
        if _single_owner:
            from nurse_rcl.v2.live.ops_sre_action_decision import compile_note_plan

            # Explicit base-field view, not a legacy decision or synthetic feedback.
            narrow = {k: deepcopy(note[k]) for k in NOTE_FIELDS}
            narrow["question_updates"] = [
                {k: deepcopy(r[k]) for k in ("question_id", "state", "remaining_gap", "basis_refs")}
                for r in updates
            ]
            state = _legacy_state(context["question_state"])
            plan = compile_note_plan(narrow, value["actions"], state, context=context)
            bound = apply_question_plan(state, plan, context=context)
            if not value["actions"]:
                validate_stop_assessment(
                    bound, note["stop_assessment"], context=context, entry_field="entry_ref"
                )
        else:
            legacy_context = dict(
                deepcopy(context),
                schema_version="nurse.v2.sre-investigation-input/20",
                question_state=_legacy_state(context["question_state"]),
            )
            legacy_value = deepcopy(value)
            legacy_value["investigation_note"].pop("observable_test")
            for row in legacy_value["investigation_note"]["question_updates"]:
                row.pop("remaining_facets")
            admission = admit_note(legacy_value, context=legacy_context)
            if admission["status"] != "accepted":
                return admission
            plan = admission["plan"]
        from nurse_rcl.v2.live.ops_sre_registered_field_delivery import already_supplied_facets

        try:
            contradiction = already_supplied_facets(context, updates)
        except ValueError as exc:
            raise ExchangeBoundaryError("registered_field_delivery_context") from exc
        if contradiction:
            raise ExchangeBoundaryError("remaining_facets_already_supplied")
        for row in plan["updates"]:
            row.update(
                extended.get(
                    row["question_id"],
                    dict(
                        observable_scope=deepcopy(scope),
                        remaining_facets=None
                        if scope is None
                        else deepcopy(scope["requested_facets"]),
                    ),
                )
            )
        apply_observable_plan(
            context["question_state"], plan, decision_index=context["decision_index"]
        )
        return dict(status="accepted", code=None, plan=plan)
    except ExchangeBoundaryError as exc:
        return dict(status="rejected", code=exc.code, plan=None)
