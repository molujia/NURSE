"""Focused public-status coherence; never grade prose or cancel a legal selection."""

from copy import deepcopy

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_sre_action_decision import admit_note, note_memory
from nurse_rcl.v2.live.ops_sre_entry_decision import admit_entry_note, parse_entry_decision
from tests.test_v2_live_sre_action_decision import context, decision


def proposal(answer_status="partial", question_state="answered"):
    value, ctx = decision(), context()
    ctx["schema_version"] = "nurse.v2.sre-investigation-input/20"
    latest = ctx["question_state"]["latest_action_question_id"]
    ctx["question_state"]["questions"][0]["question"] = "Which values did the record supply?"
    value["actions"] = [dict(entry_ref="A1", question="Inspect another actual symptom.")]
    value["latest_result_update"].update(
        answer_status=answer_status,
        summary="The record was read; the diagnostic fault status remains unknown.",
        remaining_question=None
        if answer_status == "answered"
        else "A requested window is missing.",
    )
    value["investigation_note"]["question_updates"] = [
        dict(
            question_id=latest,
            state=question_state,
            remaining_gap="" if question_state == "answered" else "A requested window is missing.",
            basis_refs=[dict(action_index=1, card_id="card_family")],
        )
    ]
    value["unresolved"] = ["The fault remains unknown; mechanism and onset evidence are unread."]
    return value, ctx


@pytest.mark.parametrize("answer_status", ["partial", "unresolved", "no_observation", "deferred"])
def test_same_question_status_cannot_be_partial_feedback_and_answered_note(answer_status):
    value, ctx = proposal(answer_status)
    before = deepcopy((value, ctx))
    assert parse_entry_decision(canonical_json_bytes(value), context=ctx) == value
    assert admit_entry_note(value, context=ctx) == dict(
        status="rejected", code="sre_note_latest_question_status", plan=None
    )
    memory = note_memory(value, context=ctx)
    assert memory["annotation_admission"]["code"] == "sre_note_latest_question_status"
    assert "investigation_plan" not in memory
    assert (value, ctx) == before


@pytest.mark.parametrize("state", ["open", "blocked", "deferred"])
def test_same_question_status_cannot_be_answered_feedback_and_unresolved_note(state):
    value, ctx = proposal("answered", state)
    assert parse_entry_decision(canonical_json_bytes(value), context=ctx) == value
    assert admit_entry_note(value, context=ctx)["code"] == "sre_note_latest_question_status"


def test_narrow_read_status_answered_does_not_resolve_unknown_fault():
    value, ctx = proposal("answered", "answered")
    original = deepcopy((value, ctx))
    result = admit_entry_note(value, context=ctx)
    assert result["status"] == "accepted"
    assert result["plan"]["updates"][0]["state"] == "answered"
    assert value["unresolved"] and "unknown" in value["unresolved"][0]
    assert (value, ctx) == original


def test_other_question_status_answered_is_not_tied_to_latest_partial():
    value, ctx = proposal()
    other = deepcopy(ctx["question_state"]["questions"][0])
    other.update(question_id="iq_other", question="What did the independent prior read supply?")
    ctx["question_state"]["questions"].append(other)
    value["investigation_note"]["question_updates"][0]["question_id"] = "iq_other"
    assert admit_entry_note(value, context=ctx)["status"] == "accepted"


@pytest.mark.parametrize("state", ["open", "blocked", "deferred"])
def test_partial_question_status_preserves_specific_gap(state):
    value, ctx = proposal("partial", state)
    result = admit_entry_note(value, context=ctx)
    assert result["status"] == "accepted"
    assert result["plan"]["updates"][0]["remaining_gap"] == "A requested window is missing."


def test_omitted_question_status_update_does_not_force_an_answer_or_rewrite():
    value, ctx = proposal()
    value["investigation_note"]["question_updates"] = []
    assert admit_entry_note(value, context=ctx)["status"] == "accepted"
    value["investigation_note"] = None
    assert admit_entry_note(value, context=ctx)["status"] == "absent"


def test_old_action_status_acceptance_domain_is_unchanged():
    value, ctx = proposal()
    ctx["schema_version"] = "nurse.v2.sre-investigation-input/18"
    assert admit_note(value, context=ctx)["status"] == "accepted"


def guidance_surfaces():
    from nurse_rcl.v2.live.ops_sre_entry_exchange import (
        ENTRY_SKILL_GUIDANCE,
        ENTRY_SYSTEM,
        entry_common_guidance,
    )

    return dict(
        system=ENTRY_SYSTEM, common=entry_common_guidance()["guidance"], **ENTRY_SKILL_GUIDANCE
    )


@pytest.mark.parametrize(
    "surface",
    ["system", "common", "boundary", "symptoms", "mechanisms", "dependency", "timing", "scoring"],
)
def test_question_scope_guidance_distinguishes_read_answer_from_fault_unknown(surface):
    text = guidance_surfaces()[surface]
    for requirement in (
        "same stored action question",
        "latest_action_question_id",
        "answered/answered",
        "remaining_question=null",
        "diagnostic unknown in unresolved",
        "partial with open/blocked/deferred",
    ):
        assert requirement in text


@pytest.mark.parametrize(
    "surface",
    ["system", "common", "boundary", "symptoms", "mechanisms", "dependency", "timing", "scoring"],
)
def test_diagnostic_guidance_distinguishes_unknown_from_covered_negative_and_fault_source(surface):
    text = guidance_surfaces()[surface]
    for requirement in (
        "unknown is a gap, not refutation",
        "absence weakens only with adequate coverage",
        "matched subject, window and unit",
        "symptoms alone do not establish a fault source",
        "mechanism and direction or timing evidence",
    ):
        assert requirement in text


def test_diagnostic_guidance_does_not_add_prose_filter_or_quality_retry():
    value, ctx = proposal("answered", "answered")
    # This intentionally weak prose remains paid native text, not a parse retry.
    value["investigation_note"]["discriminating_value"] = "Unknown would weaken the fault source."
    before = deepcopy(value)
    assert parse_entry_decision(canonical_json_bytes(value), context=ctx) == before
    assert admit_entry_note(value, context=ctx)["status"] == "accepted"
    assert value == before  # Real semantic quality still needs an actual-model review.
