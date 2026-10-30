"""Program address/version closure does not certify model prose."""

import json
from copy import deepcopy

import pytest

from nurse_rcl.artifacts import sha256_hex
from tests.sre_typed_report_fixtures import (
    api,
    empty_native,
    material,
    native,
    raw,
)
from tests.sre_typed_report_fixtures import (
    original_catalog as expression_catalog,
)


def bind(value, supply=None, store=None, **kwargs):
    if supply is None:
        supply, store = material()
    module = api("binding")
    catalog = module.typed_expression_catalog(supply, store)
    request_sha = "a" * 64
    original = raw(value)
    origin = dict(
        request_sha256=request_sha,
        catalog_sha256=sha256_hex(raw(catalog)),
        native_sha256=sha256_hex(original),
    )
    params = dict(
        request_sha256=request_sha, catalog_sha256=sha256_hex(raw(catalog)), response_origin=origin
    )
    params.update(kwargs)
    return module.bind_typed_response(original, supply, store, **params)


@pytest.mark.parametrize("step,target", [("s7", "other-service"), ("s1", "service-new")])
def test_catalog_uses_complete_actual_legal_set_and_all_source_rows(step, target):
    supply, store = material(step, target)
    before = raw(supply), raw(store.to_packet())
    catalog = api("binding").typed_expression_catalog(supply, store)
    original = expression_catalog(supply, store)
    assert [r["expression"] for r in catalog["rows"]] == original["expressions"]
    assert set(catalog["sources"]) == set(supply["provided_fact_refs"])
    assert all(
        s["source"]["fact"]["target_service_keys"] == [target] for s in catalog["sources"].values()
    )
    assert catalog["condition_boundaries"] == original["condition_boundaries"]
    assert (raw(supply), raw(store.to_packet())) == before


def test_units_preserve_missing_null_record_unit_exact_json_types_and_parent_scope():
    supply, store = material()
    catalog = api("binding").typed_expression_catalog(supply, store)
    source = catalog["sources"]["f1"]
    assert source["source"] == store.resolve("f1").model_dump(mode="json")
    fields = {f["field"]: f for f in source["fields"]}
    assert fields["baseline_denominator"]["value"] == 0
    assert fields["baseline_denominator"]["json_type"] == "integer"
    assert fields["baseline_denominator"]["unit"] == dict(
        state="supplied_null", field="baseline_denominator_unit", value=None, json_type="null"
    )
    assert fields["unknown"]["unit"] == dict(state="not_supplied")
    assert source["record_unit"] == dict(
        state="supplied", field="unit", value="cores", json_type="string"
    )
    assert "predicate_kind" not in fields  # child f3 is not parent f1's field.


def test_binding_keeps_raw_surface_and_all_legacy_assertions_distinct_from_primary():
    supply, store = material()
    value = native(
        primary=[dict(evidence_ref="f1", field="baseline_denominator")],
        text="0 counts (a deliberately unsupported unit); do not repair.",
    )
    receipt, checked = bind(value, supply, store)
    expanded = json.loads(checked)
    selected = expanded["executive_summary"][0]
    original = expression_catalog(supply, store)["expressions"][0]["claim"]
    assert selected["assertions"] == original["assertions"]
    assert selected["surface_text"] == value["executive_summary"][0]["surface_text"]
    assert expanded["core_result_digest"] == supply["core_result_digest"]
    assert receipt["semantic_certified"] is False
    row = receipt["bindings"][0]
    assert row["model_primary_fields"] == value["executive_summary"][0]["primary_fields"]
    assert row["legacy_program_assertions"] == original["assertions"]
    assert row["primary_fields_covered"] is True
    assert row["companion_sources"]["f1"]["record_unit"]["value"] == "cores"


def test_empty_primary_fields_and_omissions_are_explicitly_uncovered_not_filled():
    receipt, checked = bind(native())
    assert receipt["bindings"][0]["primary_fields_covered"] is False
    empty, expanded = bind(empty_native())
    assert empty["bindings"] == []
    assert not json.loads(expanded)["executive_summary"]


@pytest.mark.parametrize(
    "bad",
    [
        "foreign_field",
        "child_field",
        "foreign_source",
        "expression",
        "formal_section",
        "unknown_step",
        "wrong_step",
        "outcomes",
        "catalog",
        "origin_request",
        "origin_native",
        "no_origin",
    ],
)
def test_exact_permissions_and_external_request_origin_are_required(bad):
    supply, store = material()
    value = native(primary=[dict(evidence_ref="f1", field="baseline_denominator")])
    selected = value["executive_summary"][0]
    params = {}
    if bad in {"foreign_field", "child_field"}:
        selected["primary_fields"][0]["field"] = (
            "invented" if bad == "foreign_field" else "predicate_kind"
        )
    elif bad == "foreign_source":
        selected["primary_fields"][0]["evidence_ref"] = "f2"
    elif bad == "expression":
        selected["expression_ref"] = "e99999"
    elif bad == "formal_section":
        choice = next(
            r
            for r in expression_catalog(supply, store)["expressions"]
            if r["claim"]["evidence_refs"] == ["f2"]
        )
        value = native(choice["expression_ref"], section="formal_interpretations")
    elif bad in {"unknown_step", "wrong_step"}:
        value = native(section="step_findings", step="s8")
        if bad == "wrong_step":
            supply["steps"].append(
                dict(
                    step_ref="s8",
                    executed=dict(status="completed"),
                    delivered_fact_refs=[],
                    delivered_brief_refs=[],
                )
            )
    elif bad == "outcomes":
        selected["outcomes"] = [
            dict(outcome=t, interpretation=t, condition_effect=e)
            for t, e in [
                ("true", "necessary_condition_holds"),
                ("false", "necessary_condition_fails"),
                ("unknown", "condition_unknown"),
            ]
        ]
    elif bad == "catalog":
        params["catalog_sha256"] = "b" * 64
    elif bad == "no_origin":
        params["response_origin"] = None
    else:
        catalog = api("binding").typed_expression_catalog(supply, store)
        params["response_origin"] = dict(
            request_sha256="a" * 64,
            catalog_sha256=sha256_hex(raw(catalog)),
            native_sha256=sha256_hex(raw(value)),
        )
        params["response_origin"][
            "request_sha256" if bad == "origin_request" else "native_sha256"
        ] = "e" * 64
    with pytest.raises(ValueError):
        bind(value, supply, store, **params)


def test_program_row_versions_change_with_source_core_and_scope_not_prose():
    supply, store = material()
    module = api("binding")
    first = module.typed_expression_catalog(supply, store)
    assert first == module.typed_expression_catalog(supply, store)
    changed = deepcopy(supply)
    changed["core_result_digest"] = "e" * 64
    second = module.typed_expression_catalog(changed, store)
    assert first["rows"][0]["row_version"] != second["rows"][0]["row_version"]


def test_empty_legal_set_is_not_filled_by_research_case_constants():
    supply, store = material()
    supply.update(provided_fact_refs=[], formal_fact_refs=[], steps=[])
    catalog = api("binding").typed_expression_catalog(supply, store)
    assert catalog["rows"] == [] and catalog["sources"] == {}
    receipt, checked = bind(empty_native(), supply, store)
    assert receipt["bindings"] == []
    assert json.loads(checked)["step_findings"] == {}
