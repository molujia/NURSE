"""Narrow answer facets are immutable public scope, not diagnostic conclusions."""

from copy import deepcopy

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from tests.test_v2_live_sre_action_decision import context, decision, note
from tests.test_v2_live_sre_observable_purpose import api, declaration, selected, table


def first():
    result, _ = table()
    entry = selected(result)
    ctx = context(first=True)
    ctx.update(schema_version="nurse.v2.sre-investigation-input/21", action_entries=result)
    assert callable(getattr(api(), "project_observable_questions", None)), (
        "scoped projection missing"
    )
    ctx["question_state"] = api().project_observable_questions([])
    ctx["capabilities"]["investigation_situation"]["candidates"] = [
        dict(candidate_service_key="candidate"),
    ]
    value = decision(first=True)
    value["actions"] = [dict(entry_ref=entry["entry_ref"], question="Read the selected summary.")]
    value["investigation_note"] = dict(
        note(), target_service_keys=["candidate"], basis_refs=[], observable_test=declaration(entry)
    )
    return value, ctx


def second(state="open", remaining=None, answer="partial"):
    v1, c1 = first()
    admitted = api().admit_observable_note(v1, context=c1)
    memory = [
        dict(
            role="agent_statement",
            decision_index=1,
            action_index=1,
            investigation_plan=admitted["plan"],
            stop_assessment=None,
        )
    ]
    c2 = deepcopy(c1)
    c2.update(decision_index=2, question_state=api().project_observable_questions(memory))
    v2 = deepcopy(v1)
    v2["actions"][0]["question"] = "Read another facet and retain global unknown."
    v2["latest_result_update"] = dict(
        action_index=1,
        answer_status=answer,
        summary="Summary read.",
        remaining_question=None if answer == "answered" else "Missing coverage.",
        content_relation="new_content",
    )
    v2["investigation_note"]["question_updates"] = [
        dict(
            question_id="iq_d1",
            state=state,
            remaining_gap="" if state == "answered" else "Missing coverage.",
            remaining_facets=["coverage"] if remaining is None else remaining,
            basis_refs=[],
        )
    ]
    return v2, c2, memory


def test_new_question_keeps_entry_and_requested_facets_without_automatic_answer():
    value, ctx = first()
    result = api().admit_observable_note(value, context=ctx)
    assert result["status"] == "accepted"
    row = result["plan"]["updates"][0]
    assert row["question_id"] == "iq_d1" and row["question"] == value["actions"][0]["question"]
    assert (
        row["observable_scope"]["requested_facets"]
        == declaration(selected(ctx["action_entries"]))["requested_facets"]
    )
    assert row["remaining_facets"] == row["observable_scope"]["requested_facets"]
    assert row["state"] == "open" and row["parent_question_id"] is None


@pytest.mark.parametrize("state", ["open", "blocked", "deferred"])
def test_old_read_gap_accepts_only_original_facet_subset(state):
    value, ctx, _ = second(state=state)
    original = deepcopy(ctx["question_state"]["questions"][0])
    result = api().admit_observable_note(value, context=ctx)
    assert result["status"] == "accepted"
    row = result["plan"]["updates"][0]
    assert row["remaining_facets"] == ["coverage"]
    assert row["observable_scope"] == original["observable_scope"]
    assert row["question"] == original["question"]
    assert value["unresolved"]  # global diagnostic gaps are not removed


@pytest.mark.parametrize(
    "facets", [["mechanism"], ["error_count"], ["coverage", "coverage"], "coverage"]
)
def test_expanding_old_gap_rejects_note_without_rewriting_action(facets):
    value, ctx, _ = second(remaining=facets)
    before = deepcopy((value, ctx))
    assert api().admit_observable_note(value, context=ctx) == dict(
        status="rejected",
        code="remaining_facets_outside_request",
        plan=None,
    )
    assert (value, ctx) == before


def test_explicit_answer_requires_empty_remaining_facets():
    value, ctx, _ = second(state="answered", remaining=["coverage"], answer="answered")
    assert (
        api().admit_observable_note(value, context=ctx)["code"] == "answered_with_remaining_facets"
    )
    value["investigation_note"]["question_updates"][0]["remaining_facets"] = []
    result = api().admit_observable_note(value, context=ctx)
    assert result["status"] == "accepted" and result["plan"]["updates"][0]["state"] == "answered"


def test_partial_feedback_cannot_become_same_question_answered_in_new_scope():
    value, ctx, _ = second(state="answered", remaining=[], answer="partial")
    assert (
        api().admit_observable_note(value, context=ctx)["code"] == "sre_note_latest_question_status"
    )


def test_no_note_does_not_close_old_question_or_admit_a_scope():
    value, ctx, memory = second()
    value["investigation_note"] = None
    assert api().admit_observable_note(value, context=ctx)["status"] == "absent"
    memory.append(
        dict(
            role="agent_statement",
            decision_index=2,
            action_index=2,
            annotation_admission=dict(status="absent", code=None),
        )
    )
    state = api().project_observable_questions(memory)
    assert state["questions"][0]["state"] == "open"
    assert state["current_question_id"] is None and state["latest_action_question_id"] is None


def test_saved_scope_cannot_change_entry_binding_or_original_facets():
    value, ctx, memory = second()
    result = api().admit_observable_note(value, context=ctx)
    changed = deepcopy(result["plan"])
    changed["updates"][0]["observable_scope"]["binding"]["targets"] = ["another-subject"]
    assert callable(getattr(api(), "apply_observable_plan", None)), "scoped restoration missing"
    with pytest.raises(ValueError, match="scope"):
        api().apply_observable_plan(
            api().project_observable_questions(memory), changed, decision_index=2
        )


def test_correct_structure_does_not_grade_or_rewrite_free_text_gap():
    value, ctx, _ = second()
    text = "Mechanisms and other symptoms remain unread."
    value["investigation_note"]["question_updates"][0]["remaining_gap"] = text
    before = canonical_json_bytes(value)
    result = api().admit_observable_note(value, context=ctx)
    assert result["status"] == "accepted"
    assert result["plan"]["updates"][0]["remaining_gap"] == text
    assert canonical_json_bytes(value) == before
    # This deliberately contradictory prose still fails an actual semantic review.


@pytest.mark.parametrize(
    "field,bad",
    [("entry_ref", 42), ("entry_id", "not-a-pin"), ("metric_name", []), ("binding", {})],
)
def test_restored_new_scope_rejects_invalid_binding_shape(field, bad):
    value, ctx = first()
    plan = api().admit_observable_note(value, context=ctx)["plan"]
    plan["updates"][0]["observable_scope"][field] = bad
    with pytest.raises(ValueError, match="scope"):
        api().apply_observable_plan(ctx["question_state"], plan, decision_index=1)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("source_prefix", "not-a-pin"),
        ("record_sha256", 42),
        ("card_id", None),
        ("source_refs", "not-a-list"),
        ("unexpected", "extra"),
    ],
)
def test_restored_scope_rejects_malformed_source_binding(field, bad):
    value, ctx = first()
    plan = api().admit_observable_note(value, context=ctx)["plan"]
    plan["updates"][0]["observable_scope"]["binding"][field] = bad
    with pytest.raises(ValueError, match="scope"):
        api().apply_observable_plan(ctx["question_state"], plan, decision_index=1)


@pytest.mark.parametrize("mutation", ["extra", "missing", "not_row", "not_questions"])
def test_saved_question_rows_are_not_silently_filtered_or_untyped(mutation):
    value, ctx, _ = second()
    state = deepcopy(ctx["question_state"])
    if mutation == "extra":
        state["questions"][0]["unknown_authority"] = "ignore me"
    elif mutation == "missing":
        state["questions"][0].pop("remaining_facets")
    elif mutation == "not_row":
        state["questions"][0] = None
    else:
        state["questions"] = None
    plan = api().admit_observable_note(value, context=ctx)["plan"]
    with pytest.raises(ValueError, match="scope"):
        api().apply_observable_plan(state, plan, decision_index=2)


def test_short_digest_shape_is_transport_only_never_original_scope_authority():
    _, ctx, _ = second()
    state = deepcopy(ctx["question_state"])
    state["questions"][0]["observable_scope"]["binding"]["record_sha256"] = "H1"
    with pytest.raises(ValueError, match="scope"):
        api().validate_observable_questions(state, context=ctx)
    api().validate_observable_questions(state, context=ctx, short_refs=True)
    state["questions"][0]["observable_scope"]["binding"]["record_sha256"] = "H_fake"
    with pytest.raises(ValueError, match="scope"):
        api().validate_observable_questions(state, context=ctx, short_refs=True)
