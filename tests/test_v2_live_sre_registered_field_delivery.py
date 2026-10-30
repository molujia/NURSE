"""R2 only: supplied fields are not missing, but delivery is never an answer."""

from copy import deepcopy

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_sre_facet_reading import ERROR_FIELDS, _field
from nurse_rcl.v2.live.ops_sre_observable_scope import (
    admit_observable_note,
    project_observable_questions,
)
from nurse_rcl.v2.live.ops_sre_workbench import _record_metadata
from tests.test_v2_live_sre_facet_reading import context as field_context
from tests.test_v2_live_sre_observable_purpose import declaration, selected, table
from tests.test_v2_live_sre_observable_questions import first


def fixture(facet="coverage", *, legacy=False):
    value, ctx = first()
    entries, _ = table("error_summary", None)
    entry = selected(entries)
    record = deepcopy(field_context()["previous_feedback"]["semantic_record"])
    record["targets"] = ["candidate"]
    record["payload"]["facts"].update(
        observed_error_count=0,
        baseline_known_denominator=49,
        fault_known_denominator=0,
        baseline_fraction=0,
        fault_fraction=None,
    )
    identity = sha256_hex(canonical_json_bytes(record))
    entry["source_binding"] = dict(
        card_id=record["card_id"], record_sha256=identity, source_refs=record["source_refs"]
    )
    entry["observable_contract"]["binding"].update(entry["source_binding"])
    ctx["action_entries"] = entries
    value["actions"] = [dict(entry_ref=entry["entry_ref"], question="What fields are recorded?")]
    value["investigation_note"]["observable_test"] = declaration(entry)
    admitted = admit_observable_note(value, context=ctx)
    assert admitted["status"] == "accepted", admitted
    memory = [
        dict(
            role="agent_statement",
            decision_index=1,
            action_index=1,
            investigation_plan=admitted["plan"],
            stop_assessment=None,
        )
    ]
    ctx.update(decision_index=2, question_state=project_observable_questions(memory))
    ctx["actions"] = [
        dict(action_index=1, status="completed", visible_card_ids=[record["card_id"]])
    ]
    ctx["previous_feedback"] = dict(action_index=1, semantic_record=record, critical_record_ids=[])
    metadata = _record_metadata(record)
    metadata.update(action_indices=[1], latest_delivery="first_complete")
    if not legacy:
        metadata["registered_field_delivery"] = dict(
            schema_version="nurse.v2.sre-registered-field-delivery/1",
            tool_version="ops-symptoms/1",
            kind="error_summary",
            source_refs=deepcopy(record["source_refs"]),
            fields=[
                {k: v for k, v in _field(record, f, p).items() if k != "value"}
                for f, paths in ERROR_FIELDS.items()
                for p in paths
            ],
        )
    ctx["delivered_record_index"] = dict(
        schema_version=f"nurse.v2.sre-delivered-record-index/{1 if legacy else 2}",
        authority="delivery_metadata_not_diagnostic_evidence",
        latest_action_index=1,
        latest_record_refs=[identity],
        records=[metadata],
    )
    ctx["factual_reading_aid"] = dict(
        schema_version=f"nurse.v2.sre-factual-reading-aid/{3 if legacy else 4}"
    )
    value["latest_result_update"] = dict(
        action_index=1,
        answer_status="partial",
        summary="All requested fields supplied.",
        remaining_question="Diagnostic adequacy unknown.",
        content_relation="new_content",
    )
    value["investigation_note"]["question_updates"] = [
        dict(
            question_id="iq_d1",
            state="open",
            remaining_gap="Diagnostic adequacy unknown.",
            remaining_facets=[facet],
            basis_refs=[],
        )
    ]
    return value, ctx, record, memory


@pytest.mark.parametrize("facet", list(ERROR_FIELDS))
def test_already_supplied_facet_rejects_note_without_editing_native(facet):
    value, ctx, _, _ = fixture(facet)
    from nurse_rcl.v2.live.ops_sre_registered_field_delivery import question_delivery_fields

    assert len(question_delivery_fields(ctx)[0]["fields"]) == 7
    before = canonical_json_bytes(dict(value=value, context=ctx))
    assert admit_observable_note(value, context=ctx) == dict(
        status="rejected",
        code="remaining_facets_already_supplied",
        plan=None,
    )
    assert canonical_json_bytes(dict(value=value, context=ctx)) == before


def test_legacy_pair_keeps_original_structure_only_acceptance():
    value, ctx, _, _ = fixture(legacy=True)
    assert admit_observable_note(value, context=ctx)["status"] == "accepted"


def generated(*, missing=None):
    value, ctx, record, memory = fixture()
    if missing:
        record["payload"]["facts"].pop(missing)
        binding = ctx["question_state"]["questions"][0]["observable_scope"]["binding"]
        binding["record_sha256"] = sha256_hex(canonical_json_bytes(record))
    from nurse_rcl.v2.live.ops_sre_workbench import delivered_record_index

    history = [dict(action_index=1, status="completed", view=ctx["previous_feedback"])]
    ctx["delivered_record_index"] = delivered_record_index(
        history, question_state=ctx["question_state"]
    )
    return value, ctx, history, memory


def test_generated_index_is_bound_and_contains_seven_states_not_values():
    _, ctx, _, _ = generated()
    from nurse_rcl.v2.live.ops_sre_registered_field_delivery import question_delivery_fields

    index = ctx["delivered_record_index"]
    assert index["schema_version"] == "nurse.v2.sre-delivered-record-index/2"
    fields = question_delivery_fields(ctx)[0]["fields"]
    assert len(fields) == 7 and all(set(f) == {"facet", "source_field", "state"} for f in fields)
    assert [f["state"] for f in fields] == [
        "supplied_value",
        "supplied_value",
        "supplied_value",
        "supplied_value",
        "supplied_null",
        "supplied_value",
        "supplied_unknown",
    ]


def test_one_missing_denominator_side_remains_a_lawful_open_gap():
    value, ctx, _, _ = generated(missing="fault_known_denominator")
    value["investigation_note"]["question_updates"][0]["remaining_facets"] = ["denominator"]
    admission = admit_observable_note(value, context=ctx)
    assert admission["status"] == "accepted", admission
    assert admission["plan"]["updates"][0]["state"] == "open"
    assert admission["plan"]["updates"][0]["remaining_facets"] == ["denominator"]


@pytest.mark.parametrize(
    "bad", ["digest", "source", "subject", "window", "partial", "version", "failed"]
)
def test_nonmatching_history_never_certifies_original_question_fields(bad):
    _, ctx, history, _ = generated()
    record = history[0]["view"]["semantic_record"]
    if bad == "digest":
        record["payload"]["facts"]["observed_error_count"] = 99
    elif bad == "source":
        record["source_refs"] = ["c" * 64]
    elif bad == "subject":
        record["targets"] = ["other"]
    elif bad == "window":
        record["window_ns"] = [1, 2]
    elif bad == "partial":
        record["representation"] = "overview_not_full_judgment"
    elif bad == "version":
        record["tool_version"] = "ops-symptoms/999"
    else:
        history[0]["status"] = "failed"
    from nurse_rcl.v2.live.ops_sre_workbench import delivered_record_index

    actual = delivered_record_index(history, question_state=ctx["question_state"])
    assert all("registered_field_delivery" not in r for r in actual["records"])


def test_past_question_fields_do_not_disappear_when_latest_is_a_different_record():
    value, ctx, history, _ = generated()
    other = deepcopy(history[0])
    other["action_index"] = 2
    other["view"]["action_index"] = 2
    other["view"]["semantic_record"]["card_id"] = "another-card"
    history.append(other)
    ctx["actions"].append(
        dict(action_index=2, status="completed", visible_card_ids=["another-card"])
    )
    ctx["previous_feedback"] = other["view"]
    ctx["decision_index"] = 3
    from nurse_rcl.v2.live.ops_sre_workbench import delivered_record_index

    ctx["delivered_record_index"] = delivered_record_index(
        history, question_state=ctx["question_state"]
    )
    from nurse_rcl.v2.live.ops_sre_registered_field_delivery import question_delivery_fields

    assert len(question_delivery_fields(ctx)) == 1
    assert admit_observable_note(value, context=ctx)["code"] == "remaining_facets_already_supplied"


@pytest.mark.parametrize("state", ["open", "blocked", "deferred", "answered"])
def test_no_remaining_does_not_force_answer_and_explicit_answer_keeps_scope(state):
    value, ctx, _, _ = generated()
    update = value["investigation_note"]["question_updates"][0]
    update.update(
        state=state,
        remaining_facets=[],
        remaining_gap="" if state == "answered" else "Unknown diagnosis.",
    )
    if state == "answered":
        value["latest_result_update"].update(answer_status="answered", remaining_question=None)
    previous = deepcopy(ctx["question_state"]["questions"][0])
    admitted = admit_observable_note(value, context=ctx)
    assert admitted["status"] == "accepted", admitted
    actual = admitted["plan"]["updates"][0]
    assert actual["state"] == state and actual["question"] == previous["question"]
    assert actual["observable_scope"] == previous["observable_scope"]
    assert ctx["question_state"]["questions"][0] == previous


def test_no_native_update_or_absent_note_never_answers_old_question():
    value, ctx, _, _ = generated()
    before = deepcopy(ctx["question_state"])
    value["investigation_note"]["question_updates"] = []
    actual = admit_observable_note(value, context=ctx)
    assert actual["status"] == "accepted"
    assert all(r["question_id"] != "iq_d1" for r in actual["plan"]["updates"])
    value["investigation_note"] = None
    assert admit_observable_note(value, context=ctx)["status"] == "absent"
    assert ctx["question_state"] == before


@pytest.mark.parametrize("side", ["index", "aid"])
def test_mixed_versions_cannot_silently_disable_new_gate(side):
    value, ctx, _, _ = generated()
    if side == "index":
        ctx["delivered_record_index"]["schema_version"] = "nurse.v2.sre-delivered-record-index/1"
    else:
        ctx["factual_reading_aid"]["schema_version"] = "nurse.v2.sre-factual-reading-aid/3"
    assert admit_observable_note(value, context=ctx)["code"] == "registered_field_delivery_context"


@pytest.mark.parametrize("bad", ["future", "failed", "invisible", "duplicate"])
def test_metadata_must_still_be_unique_past_completed_visible(bad):
    value, ctx, _, _ = generated()
    row = ctx["delivered_record_index"]["records"][0]
    if bad == "future":
        row["action_indices"] = [2]
    elif bad == "failed":
        ctx["actions"][0]["status"] = "failed"
    elif bad == "invisible":
        ctx["actions"][0]["visible_card_ids"] = []
    else:
        ctx["delivered_record_index"]["records"].append(deepcopy(row))
    assert admit_observable_note(value, context=ctx)["code"] == "registered_field_delivery_context"


def test_repeated_same_record_shares_states_and_preserves_actual_action_associations():
    _, ctx, history, _ = generated()
    history.append(dict(deepcopy(history[0]), action_index=2))
    ctx["actions"].append(dict(deepcopy(ctx["actions"][0]), action_index=2))
    from nurse_rcl.v2.live.ops_sre_registered_field_delivery import question_delivery_fields
    from nurse_rcl.v2.live.ops_sre_workbench import delivered_record_index

    ctx["delivered_record_index"] = delivered_record_index(
        history, question_state=ctx["question_state"]
    )
    rows = question_delivery_fields(ctx)
    assert len(rows) == 1 and rows[0]["action_indices"] == [1, 2]
    assert len(rows[0]["fields"]) == 7


def test_latest_actual_states_reject_tampering_even_if_both_displays_are_changed():
    from nurse_rcl.v2.live.ops_sre_factual_reading import (
        factual_reading_aid,
        validate_factual_reading_aid,
    )

    _, ctx, _, _ = generated()
    ctx["capabilities"] = {}
    ctx["factual_reading_aid"] = factual_reading_aid(
        ctx, schema_version="nurse.v2.sre-factual-reading-aid/4"
    )
    validate_factual_reading_aid(ctx)
    ctx["delivered_record_index"]["records"][0]["registered_field_delivery"]["fields"][-1][
        "state"
    ] = "not_supplied"
    ctx["factual_reading_aid"]["question_delivery_fields"][0]["fields"][-1]["state"] = (
        "not_supplied"
    )
    with pytest.raises(ValueError, match="latest actual record"):
        validate_factual_reading_aid(ctx)


def test_new_pair_cannot_be_requested_from_legacy_index_or_old_aid_from_new_index():
    from nurse_rcl.v2.live.ops_sre_factual_reading import factual_reading_aid

    _, ctx, _, _ = generated()
    ctx["capabilities"] = {}
    with pytest.raises(ValueError, match="version pair"):
        factual_reading_aid(ctx)
    ctx["delivered_record_index"]["schema_version"] = "nurse.v2.sre-delivered-record-index/1"
    with pytest.raises(ValueError, match="version pair"):
        factual_reading_aid(ctx, schema_version="nurse.v2.sre-factual-reading-aid/4")


def test_projection_is_limited_to_eight_original_questions_and_56_path_states():
    from nurse_rcl.v2.live.ops_sre_registered_field_delivery import question_delivery_fields

    _, ctx, _, _ = generated()
    question = ctx["question_state"]["questions"][0]
    ctx["question_state"]["questions"] = [
        dict(deepcopy(question), question_id=f"iq_d{i}") for i in range(1, 9)
    ]
    rows = question_delivery_fields(ctx)
    assert len(rows) == 8 and sum(len(r["fields"]) for r in rows) == 56
    ctx["question_state"]["questions"].append(dict(deepcopy(question), question_id="iq_d9"))
    with pytest.raises(ValueError, match="question limit"):
        question_delivery_fields(ctx)


@pytest.mark.parametrize("bad", ["value", "path", "state", "registry"])
def test_state_metadata_cannot_accept_values_unregistered_paths_or_versions(bad):
    from nurse_rcl.v2.live.ops_sre_registered_field_delivery import validate_registered_metadata

    _, ctx, _, _ = generated()
    data = ctx["delivered_record_index"]["records"][0]["registered_field_delivery"]
    if bad == "value":
        data["fields"][0]["value"] = 0
    elif bad == "path":
        data["fields"][0]["source_field"] = "payload.facts.coverage_adequacy"
    elif bad == "state":
        data["fields"][0]["state"] = "answered"
    else:
        data["schema_version"] = "nurse.v2.sre-registered-field-delivery/999"
    with pytest.raises(ValueError, match="registered field delivery"):
        validate_registered_metadata(ctx["delivered_record_index"])


def test_old_index_cannot_smuggle_new_metadata_and_old_compiler_cannot_take_new_index():
    from nurse_rcl.v2.live.ops_sre_action_entries import _delivery_rows
    from nurse_rcl.v2.live.ops_sre_workbench import validate_workbench_context

    _, ctx, _, _ = generated()
    index = ctx["delivered_record_index"]
    assert _delivery_rows(index, 2, observable=True) == index["records"]
    with pytest.raises(ValueError, match="entry delivery index"):
        _delivery_rows(index, 2)
    index["schema_version"] = "nurse.v2.sre-delivered-record-index/1"
    with pytest.raises(ValueError, match="workbench record fields"):
        validate_workbench_context(ctx)


@pytest.mark.parametrize("bad", ["missing_indices", "non_record"])
def test_new_index_validates_base_shape_before_projecting_registered_states(bad):
    from nurse_rcl.v2.live.ops_sre_workbench import validate_workbench_context

    _, ctx, _, _ = generated()
    if bad == "missing_indices":
        ctx["delivered_record_index"]["records"][0].pop("action_indices")
    else:
        ctx["delivered_record_index"]["records"] = [None]
    with pytest.raises(ValueError, match="workbench record fields"):
        validate_workbench_context(ctx)
