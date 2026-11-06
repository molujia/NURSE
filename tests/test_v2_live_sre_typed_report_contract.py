"""One explicit normal report contract; these tests do not test model efficacy."""

from copy import deepcopy

import pytest

from tests.sre_typed_report_fixtures import SECTIONS, api, empty_native, native, raw


def test_policy_has_one_active_contract_without_inherited_output_override():
    contract = api("contract")
    policy = contract.typed_report_policy()
    assert policy["protocol"] == "nurse.v2.sre-typed-report-exchange/1"
    assert policy["profile"] == "hermes-sre-typed-report/1"
    assert policy["schema"] == "nurse.v2.atomic-typed-report-input/1"
    assert contract.TYPED_PATH_REPORT_RESPONSE in policy["system"]
    assert "OUTPUT OVERRIDE" not in policy["system"]
    assert "nurse.v2.atomic-report-expression-response/3" not in policy["system"]
    assert "nurse.v2.sre-report-response/4" not in policy["system"]
    assert set(policy["response_schema"]["properties"]) == {
        "schema_version", *SECTIONS, "predicted_paths", "path_abstention_reason"
    }
    assert policy["max_tokens"] == 65536


def test_empty_report_is_legal_without_model_echoed_core_or_versions():
    result = api("contract").parse_typed_response(raw(empty_native()))
    assert result.model_dump(mode="json") == empty_native()


def test_surface_and_outcome_strings_are_not_normalized():
    value = native()
    text = value["executive_summary"][0]["surface_text"]
    result = api("contract").parse_typed_response(raw(value))
    assert result.executive_summary[0].surface_text == text


@pytest.mark.parametrize(
    "bad",
    [
        "old_schema",
        "core_digest",
        "row_version",
        "extra_selection",
        "duplicate_claim",
        "duplicate_primary",
        "wrong_field_type",
        "blank_text",
        "partial_outcomes",
        "duplicate_outcome",
    ],
)
def test_native_contract_rejects_legacy_forgery_duplicates_and_incomplete_outcomes(bad):
    value = native(primary=[dict(evidence_ref="f1", field="baseline_denominator")])
    selected = value["executive_summary"][0]
    if bad == "old_schema":
        value["schema_version"] = "nurse.v2.atomic-report-expression-response/3"
    elif bad == "core_digest":
        value["core_result_digest"] = "a" * 64
    elif bad == "row_version":
        selected["row_version"] = "a" * 64
    elif bad == "extra_selection":
        selected["assertions"] = []
    elif bad == "duplicate_claim":
        value["final_synthesis"] = [deepcopy(selected)]
    elif bad == "duplicate_primary":
        selected["primary_fields"] *= 2
    elif bad == "wrong_field_type":
        selected["primary_fields"][0]["field"] = 4
    elif bad == "blank_text":
        selected["surface_text"] = "  \n "
    else:
        branch = dict(
            outcome="unknown",
            interpretation="Unknown is unresolved.",
            condition_effect="condition_unknown",
        )
        selected["outcomes"] = [branch] if bad == "partial_outcomes" else [branch] * 3
    with pytest.raises(ValueError):
        api("contract").parse_typed_response(raw(value))


def test_complete_outcomes_keep_original_order_and_interpretations():
    value = native(
        outcomes=[
            dict(outcome=t, condition_effect=effect, interpretation=f"  {t}  ")
            for t, effect in [
                ("unknown", "condition_unknown"),
                ("true", "necessary_condition_holds"),
                ("false", "necessary_condition_fails"),
            ]
        ]
    )
    result = api("contract").parse_typed_response(raw(value))
    assert result.model_dump(mode="json") == value
