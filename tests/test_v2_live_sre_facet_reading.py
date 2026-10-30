"""Registered supplied-field states are not answers, diagnoses or retry authority."""

import importlib
import inspect
from copy import deepcopy

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex


def build(context, **kwargs):
    module = importlib.import_module("nurse_rcl.v2.live.ops_sre_factual_reading")
    function = getattr(module, "latest_question_field_reading_aid", None)
    assert callable(function), "latest narrow question needs registered field delivery states"
    if kwargs:
        assert "short_refs" in inspect.signature(function).parameters, (
            "readable aliases need separate transport validation"
        )
    return function(context, **kwargs)


def context():
    record = dict(
        card_id="card-error",
        node="symptoms",
        tool_version="ops-symptoms/1",
        targets=["svc-error"],
        source_role="core_published",
        source_refs=["a" * 64],
        unit="observed_record_fraction",
        coverage=None,
        truth_value="unknown",
        payload=dict(
            kind="error_summary",
            facts=dict(
                observed_error_count=2,
                fault_known_denominator=5410,
                baseline_known_denominator=6908,
                fault_fraction=0.0003696857670979667,
                baseline_fraction=0.0,
                sampling_coverage="unknown",
            ),
        ),
    )
    value = dict(
        actions=[dict(action_index=1, status="completed")],
        previous_feedback=dict(action_index=1, semantic_record=record, critical_record_ids=[]),
        question_state=dict(
            latest_action_question_id="iq_d1",
            questions=[
                dict(
                    question_id="iq_d1",
                    question="What error summary fields are recorded?",
                    state="open",
                    remaining_facets=["coverage"],
                    target_service_keys=["svc-error"],
                    observable_scope=dict(
                        objective_kind="error_summary",
                        requested_facets=["coverage"],
                        binding={},
                    ),
                )
            ],
        ),
    )
    rebind(value)
    return value


def rebind(value):
    record = value["previous_feedback"]["semantic_record"]
    scope = value["question_state"]["questions"][0]["observable_scope"]
    scope["binding"] = dict(
        card_id=record["card_id"],
        record_sha256=sha256_hex(canonical_json_bytes(record)),
        targets=deepcopy(record["targets"]),
        source_refs=deepcopy(record["source_refs"]),
        source_prefix="b" * 64,
    )


def test_supplied_unknown_answers_record_availability_not_diagnostic_coverage():
    value = context()
    before = canonical_json_bytes(value)
    rows = build(value)
    assert rows == [
        dict(
            authority="field_delivery_only_not_question_answer_or_diagnostic_evidence",
            question_id="iq_d1",
            action_index=1,
            card_id="card-error",
            record_sha256=value["question_state"]["questions"][0]["observable_scope"]["binding"][
                "record_sha256"
            ],
            objective_kind="error_summary",
            fields=[
                dict(
                    facet="coverage",
                    source_field="payload.facts.sampling_coverage",
                    state="supplied_unknown",
                    value="unknown",
                )
            ],
        )
    ]
    assert canonical_json_bytes(value) == before
    assert value["question_state"]["questions"][0]["state"] == "open"


@pytest.mark.parametrize(
    "supplied,state", [(None, "supplied_null"), (0.0, "supplied_value"), ("", "supplied_value")]
)
def test_explicit_null_zero_and_empty_string_are_not_missing(supplied, state):
    value = context()
    value["previous_feedback"]["semantic_record"]["payload"]["facts"]["sampling_coverage"] = (
        supplied
    )
    rebind(value)
    field = build(value)[0]["fields"][0]
    assert field["state"] == state and field["value"] == supplied
    assert "answered" not in build(value)[0] and "remaining_facets" not in build(value)[0]


def test_missing_nested_field_does_not_borrow_null_coverage_header():
    value = context()
    value["previous_feedback"]["semantic_record"]["payload"]["facts"].pop("sampling_coverage")
    rebind(value)
    assert build(value)[0]["fields"] == [
        dict(facet="coverage", source_field="payload.facts.sampling_coverage", state="not_supplied")
    ]


def test_requested_facets_only_preserve_each_denominator_side_and_native_unit():
    value = context()
    question = value["question_state"]["questions"][0]
    question["observable_scope"]["requested_facets"] = [
        "unit",
        "denominator",
        "error_count",
        "fraction",
    ]
    record = value["previous_feedback"]["semantic_record"]
    record["payload"]["facts"].pop("baseline_known_denominator")
    rebind(value)
    fields = build(value)[0]["fields"]
    assert [f["facet"] for f in fields] == [
        "unit",
        "denominator",
        "denominator",
        "error_count",
        "fraction",
        "fraction",
    ]
    assert fields[0] == dict(
        facet="unit", source_field="unit", state="supplied_value", value="observed_record_fraction"
    )
    assert fields[1]["state"] == "not_supplied" and "value" not in fields[1]
    assert fields[2]["value"] == 5410 and fields[3]["value"] == 2
    assert fields[4]["value"] == 0.0 and fields[5]["value"] == 0.0003696857670979667
    assert not any(f["facet"] == "coverage" for f in fields)


@pytest.mark.parametrize("change", ["card_id", "record_sha256", "targets", "source_refs"])
def test_scope_binding_cannot_borrow_another_record_subject_or_source(change):
    value = context()
    value["question_state"]["questions"][0]["observable_scope"]["binding"][change] = "different"
    assert build(value) == []


@pytest.mark.parametrize("change", ["role", "content", "window"])
def test_whole_record_digest_not_a_field_subset_is_required(change):
    value = context()
    record = value["previous_feedback"]["semantic_record"]
    if change == "role":
        record["source_role"] = "supplementary"
    elif change == "content":
        record["payload"]["facts"]["fault_fraction"] = 0.1
    else:
        record["window_ns"] = [1, 2]
    assert build(value) == []


@pytest.mark.parametrize(
    "change",
    [
        "no_action",
        "failed",
        "resource_limit",
        "brief",
        "unknown_version",
        "paid_episode",
        "wrong_node",
        "wrong_objective",
        "no_question",
        "wrong_subject",
        "bad_facet",
    ],
)
def test_unregistered_partial_or_unadmitted_input_never_gains_field_authority(change):
    value = context()
    record = value["previous_feedback"]["semantic_record"]
    question = value["question_state"]["questions"][0]
    if change == "no_action":
        value["actions"] = []
    elif change == "failed":
        value["actions"][-1]["status"] = "rejected"
    elif change == "resource_limit":
        value["previous_feedback"]["status"] = "resource_limit"
    elif change == "brief":
        record["representation"] = "delivered_section_not_full_judgment"
    elif change == "unknown_version":
        record["tool_version"] = "unknown/1"
    elif change == "paid_episode":
        record["payload"]["kind"] = "paid_episode"
    elif change == "wrong_node":
        record["node"] = "scoring"
    elif change == "wrong_objective":
        question["observable_scope"]["objective_kind"] = "record_read"
    elif change == "no_question":
        value["question_state"]["latest_action_question_id"] = None
    elif change == "wrong_subject":
        question["target_service_keys"] = ["another"]
    else:
        question["observable_scope"]["requested_facets"] = ["mechanism"]
    rebind(value)
    assert build(value) == []


def test_latest_feedback_cannot_reuse_an_earlier_action_number():
    value = context()
    value["previous_feedback"]["action_index"] = 2
    with pytest.raises(ValueError, match="latest"):
        build(value)


def test_projection_does_not_create_mutable_aliases_to_original_values():
    value = context()
    fields = build(value)[0]["fields"]
    fields[0]["value"] = "known"
    assert (
        value["previous_feedback"]["semantic_record"]["payload"]["facts"]["sampling_coverage"]
        == "unknown"
    )


def short_context():
    """Explicit model-only aliases, not a second hash of an altered original."""
    value = context()
    record = value["previous_feedback"]["semantic_record"]
    record.update(card_id="C1", targets=["S1"], source_refs=["P1"])
    binding = value["question_state"]["questions"][0]["observable_scope"]["binding"]
    binding.update(card_id="C1", targets=["S1"], source_refs=["P1"], record_sha256="H1")
    value["question_state"]["questions"][0]["target_service_keys"] = ["S1"]
    delivery = dict(
        card_id="C1",
        targets=["S1"],
        source_role="core_published",
        delivery_kind="complete",
        record_sha256="H1",
    )
    value["previous_feedback"]["delivery_feedback"] = dict(action_index=1, records=[delivery])
    index = dict(delivery, node="symptoms", fields_present=["facts", "kind"], action_indices=[1])
    value["delivered_record_index"] = dict(
        latest_action_index=1, latest_record_refs=["H1"], records=[index]
    )
    return value


def test_readable_alias_display_matches_latest_complete_delivery_without_rehashing_aliases():
    value = short_context()
    before = canonical_json_bytes(value)
    rows = build(value, short_refs=True)
    assert rows[0]["record_sha256"] == "H1"
    assert rows[0]["fields"][0]["state"] == "supplied_unknown"
    assert canonical_json_bytes(value) == before
    # Full owner admission must still require the original canonical digest.
    assert build(value) == []


@pytest.mark.parametrize(
    "change",
    ["binding", "delivery", "index", "duplicate", "stale", "partial", "fields", "role", "window"],
)
def test_readable_alias_identity_cannot_borrow_stale_partial_or_different_metadata(change):
    value = short_context()
    delivery = value["previous_feedback"]["delivery_feedback"]
    index = value["delivered_record_index"]
    if change == "binding":
        value["question_state"]["questions"][0]["observable_scope"]["binding"]["record_sha256"] = (
            "H2"
        )
    elif change == "delivery":
        delivery["records"][0]["record_sha256"] = "H2"
    elif change == "index":
        index["records"][0]["record_sha256"] = "H2"
    elif change == "duplicate":
        delivery["records"].append(deepcopy(delivery["records"][0]))
    elif change == "stale":
        delivery["action_index"] = 2
    elif change == "partial":
        delivery["records"][0]["delivery_kind"] = "partial"
    elif change == "fields":
        index["records"][0]["fields_present"] = ["kind"]
    elif change == "role":
        index["records"][0]["source_role"] = "supplementary"
    else:
        index["records"][0]["window_ns"] = [1, 2]
    assert build(value, short_refs=True) == []
