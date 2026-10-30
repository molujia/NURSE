"""Entry-only supply and question-scope contracts, not a model semantic grader."""

from copy import deepcopy

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_sre_entry_decision import admit_entry_note, parse_entry_decision
from tests.test_v2_live_sre_entry_feedback_semantics import guidance_surfaces, proposal

SURFACES = [
    "system", "common", "boundary", "symptoms", "mechanisms", "dependency", "timing", "scoring"
]


@pytest.mark.parametrize("surface", SURFACES)
def test_predicted_discrimination_is_bounded_by_selected_supply(surface):
    text = guidance_surfaces()[surface]
    for requirement in (
        "question and discriminating_value within the selected supply_contract and limits",
        "For a symptom check",
        "specific symptom hypothesis",
        "changes the next check, not source identity",
        "different metrics are not interchangeable negative controls",
        "Zero errors with an unknown denominator or coverage is not covered absence",
    ):
        assert requirement in text


@pytest.mark.parametrize("surface", SURFACES)
def test_read_gap_does_not_silently_expand_into_diagnostic_question(surface):
    text = guidance_surfaces()[surface]
    for requirement in (
        "Keep the stored read question's original scope",
        "only missing parts of that original request belong in its remaining gap",
        "A broader diagnostic question belongs in unresolved",
        "new action question with an explicit public relation to the prior question",
        "Do not invent question IDs or parent links",
    ):
        assert requirement in text


def test_new_symptom_skill_removes_ambiguous_contrast_only_in_entry():
    from nurse_rcl.v2.live.ops_sre_action_exchange import ACTION_SKILL_GUIDANCE
    from nurse_rcl.v2.live.ops_sre_entry_exchange import ENTRY_GUIDANCE, ENTRY_SKILL_GUIDANCE

    ambiguous = "a covered normal contrast can weaken an explanation"
    assert ambiguous in ACTION_SKILL_GUIDANCE["symptoms"]  # Old paid policy is unchanged.
    assert ambiguous not in ENTRY_SKILL_GUIDANCE["symptoms"]
    assert "covered matched contrast can weaken only that symptom hypothesis" in (
        ENTRY_SKILL_GUIDANCE["symptoms"]
    )
    for node, original in ACTION_SKILL_GUIDANCE.items():
        if node != "symptoms":
            assert ENTRY_SKILL_GUIDANCE[node] == original + " " + ENTRY_GUIDANCE


@pytest.mark.parametrize("read_status", ["answered", "partial"])
def test_new_diagnostic_question_keeps_old_read_identity_and_specific_gap(read_status):
    value, ctx = proposal(read_status, "answered" if read_status == "answered" else "open")
    old = deepcopy(ctx["question_state"]["questions"][0])
    old_id = old["question_id"]
    question = f"Following {old_id}, does the selected record show its own measured symptom?"
    value["actions"][0]["question"] = question
    value["investigation_note"]["discriminating_value"] = (
        "A covered matched contrast can test this symptom hypothesis, not source identity."
    )
    before = deepcopy((value, ctx))
    assert parse_entry_decision(canonical_json_bytes(value), context=ctx) == value
    result = admit_entry_note(value, context=ctx)
    assert result["status"] == "accepted"
    rows = {row["question_id"]: row for row in result["plan"]["updates"]}
    assert rows[old_id]["question"] == old["question"]
    assert rows[old_id]["role"] == old["role"]
    assert rows[old_id]["target_service_keys"] == old["target_service_keys"]
    assert rows[old_id]["remaining_gap"] == (
        "" if read_status == "answered" else "A requested window is missing."
    )
    current = rows[result["plan"]["current_question_id"]]
    assert current["question_id"] != old_id and current["question"] == question
    assert current["state"] == "open" and current["parent_question_id"] is None
    assert value["unresolved"]  # Read status does not close the fault investigation.
    assert (value, ctx) == before


def test_overpromised_native_purpose_is_not_a_quality_retry_or_rewritten():
    value, ctx = proposal("answered", "answered")
    # Deliberately weak synthetic prose, not a imported paid trajectory.
    value["investigation_note"]["discriminating_value"] = (
        "Another candidate's symptom will support or weaken this candidate as the source."
    )
    before = deepcopy(value)
    assert parse_entry_decision(canonical_json_bytes(value), context=ctx) == before
    assert admit_entry_note(value, context=ctx)["status"] == "accepted"
    assert value == before  # Actual semantic improvement needs new native-model evidence.


def test_entry_guidance_is_in_policy_hash_and_common_digest():
    from nurse_rcl.v2.live.ops_empirical_policy import MAX_OUTPUT_TOKENS, OUTPUT_POLICY
    from nurse_rcl.v2.live.ops_sre_entry_exchange import ENTRY_SYSTEM, entry_decision_policy

    policy = entry_decision_policy()
    pinned = dict(
        protocol=policy["protocol"],
        system=ENTRY_SYSTEM,
        output_policy=OUTPUT_POLICY,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    assert policy["prompt_sha256"] == sha256_hex(canonical_json_bytes(pinned))
    guidance = policy["guidance"]
    assert guidance["content_sha256"] == sha256_hex(guidance["guidance"].encode())
    assert "changes the next check, not source identity" in pinned["system"]
    assert "original scope" in guidance["guidance"]
