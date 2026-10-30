"""B separates public-note admission from the immutable proposed action."""

from copy import deepcopy
from importlib import import_module, util

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from tests.test_v2_live_sre_d26_decision import ctx


def api():
    name = "nurse_rcl.v2.live.ops_sre_action_decision"
    assert util.find_spec(name) is not None, "action-first admission is missing"
    return import_module(name)


def note():
    return dict(
        role="symptoms",
        target_service_keys=["candidate_a"],
        why_now="The score comparison cannot establish the observed fault.",
        discriminating_value="Check whether the measured symptoms support the mechanism.",
        basis_refs=[dict(action_index=1, card_id="card_family")],
        question_updates=[],
        stop_assessment=None,
    )


def decision(first=False, stop=False):
    return dict(
        actions=[]
        if stop
        else [
            dict(
                tool="read_node",
                skill_id="sre-mechanisms/13",
                question="Which measured conditions support or refute the fault?",
                parameters=dict(card_id="entry_mechanism", section="payload", cursor=None),
            )
        ],
        stop_reason="Further evidence remains unchecked." if stop else None,
        public_summary="The fixed ranking is not a confirmed cause.",
        unresolved=["Mechanism and symptoms remain unresolved."],
        responds_to=[] if first else [dict(action_index=1, card_ids=["card_family"])],
        stop_basis="no_useful_check" if stop else None,
        latest_result_update=None
        if first
        else dict(
            action_index=1,
            answer_status="partial",
            summary="No new content was delivered.",
            remaining_question="The mechanism still needs checking.",
            content_relation="no_new_content",
        ),
        investigation_note=None if first or stop else note(),
    )


def context(first=False):
    value = ctx(first)
    value["schema_version"] = "nurse.v2.sre-investigation-input/18"
    return value


@pytest.mark.parametrize("bad", ["gap", "source", "ambiguous", "shape", "target", "future"])
def test_invalid_note_does_not_cancel_independently_valid_action(bad):
    value, c = decision(), context()
    if bad == "gap":
        value["investigation_note"]["question_updates"] = [
            dict(
                question_id="iq_fault",
                state="answered",
                remaining_gap=None,
                basis_refs=[],
            )
        ]
    elif bad == "source":
        value["investigation_note"]["basis_refs"][0]["card_id"] = "overview_only"
    elif bad == "future":
        value["investigation_note"]["basis_refs"][0]["action_index"] = 2
    elif bad == "ambiguous":
        c["delivered_record_index"]["records"].append(
            dict(
                c["delivered_record_index"]["records"][0],
                record_sha256="b" * 64,
            )
        )
    elif bad == "shape":
        value["investigation_note"] = "unsupported public note"
    else:
        value["investigation_note"]["target_service_keys"] = ["other_case"]
    before, original_context = deepcopy(value), deepcopy(c)
    assert api().parse_action_decision(canonical_json_bytes(value), context=c) == before
    admission = api().admit_note(value, context=c)
    assert admission["status"] == "rejected"
    assert admission["plan"] is None
    assert admission["code"]
    assert value == before and c == original_context


def test_program_binds_unique_real_delivery_without_model_hashes():
    value, c = decision(), context()
    assert api().parse_action_decision(canonical_json_bytes(value), context=c) == value
    admission = api().admit_note(value, context=c)
    assert admission["status"] == "accepted"
    row = admission["plan"]["updates"][-1]
    assert row["question_id"] == "iq_d2"
    assert row["question"] == value["actions"][0]["question"]
    assert row["basis_refs"] == [
        dict(
            action_index=1,
            card_id="card_family",
            record_sha256="a" * 64,
            delivery_kind="complete",
        )
    ]
    assert "record_sha256" not in value["investigation_note"]["basis_refs"][0]


@pytest.mark.parametrize("bad", ["feedback", "link", "budget", "shape"])
def test_action_feedback_and_budget_boundaries_still_reject(bad):
    value, c = decision(stop=bad == "budget"), context()
    if bad == "feedback":
        value["latest_result_update"]["content_relation"] = "new_content"
    elif bad == "link":
        value["responds_to"][0]["card_ids"] = ["not_delivered"]
    elif bad == "budget":
        value["stop_basis"] = "budget_exhausted"
    else:
        value["unexpected"] = True
    with pytest.raises(ExchangeBoundaryError):
        api().parse_action_decision(canonical_json_bytes(value), context=c)


def test_missing_note_is_explicit_not_a_fake_answer():
    value, c = decision(first=True), context(first=True)
    assert api().parse_action_decision(canonical_json_bytes(value), context=c) == value
    assert api().admit_note(value, context=c) == dict(status="absent", code=None, plan=None)


def test_question_updates_preserve_identity_and_unresolved_fault():
    value, c = decision(), context()
    value["investigation_note"]["question_updates"] = [
        dict(
            question_id="iq_fault",
            state="blocked",
            remaining_gap="No self-time observation.",
            basis_refs=[dict(action_index=1, card_id="card_family")],
        )
    ]
    result = api().admit_note(value, context=c)
    assert result["status"] == "accepted"
    previous = result["plan"]["updates"][0]
    assert previous["question"] == c["question_state"]["questions"][0]["question"]
    assert previous["state"] == "blocked"
    assert api().note_memory(value, context=c)["annotation_admission"]["status"] == "accepted"


@pytest.mark.parametrize(
    "answer_status", ["answered", "partial", "unresolved", "no_observation", "deferred"]
)
def test_typed_feedback_is_retained_without_granting_new_evidence(answer_status):
    value, c = decision(), context()
    value["latest_result_update"]["answer_status"] = answer_status
    value["investigation_note"]["basis_refs"][0]["card_id"] = "unread_mechanism"
    assert api().parse_action_decision(canonical_json_bytes(value), context=c) == value
    assert api().note_memory(value, context=c)["annotation_admission"]["status"] == "rejected"
    assert "investigation_plan" not in api().note_memory(value, context=c)
