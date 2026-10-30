"""G+ is a distinct normal native contract, not a renamed research answer."""

import json
from importlib import import_module, util

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from tests.sre_typed_report_fixtures import material, native


def api():
    name = "nurse_rcl.v2.live.ops_sre_gplus_contract"
    assert util.find_spec(name), "missing normal G+ contract"
    return import_module(name)


def bind(value, *, mutation=None):
    module = api()
    from nurse_rcl.v2.live.ops_sre_gplus_graph import build_gplus_graph
    from nurse_rcl.v2.live.ops_sre_typed_report_binding import typed_expression_catalog

    supply, store = material()
    graph = build_gplus_graph(typed_expression_catalog(supply, store))
    raw = canonical_json_bytes(value)
    params = dict(request_sha256="a" * 64, graph_sha256=sha256_hex(canonical_json_bytes(graph)))
    origin = dict(**params, native_sha256=sha256_hex(raw))
    if mutation == "origin":
        origin["native_sha256"] = "b" * 64
    if mutation == "graph":
        params["graph_sha256"] = "c" * 64
    return module.bind_gplus_response(raw, supply, store, response_origin=origin, **params)


def value():
    result = native(primary=[dict(evidence_ref="f1", field="baseline_denominator")])
    result["schema_version"] = "nurse.v2.atomic-gplus-report-expression-response/1"
    return result


def test_gplus_binds_real_closure_and_keeps_native_text_separate_from_program_facts():
    module = api()
    original = value()
    receipt, checked = bind(original)
    assert (
        module.parse_gplus_response(canonical_json_bytes(original)).model_dump(mode="json")
        == original
    )
    row = receipt["bindings"][0]
    assert row["upstream_closure"]["mandatory_complete"]
    assert row["model_primary_fields"] == original["executive_summary"][0]["primary_fields"]
    assert receipt["semantic_certified"] is False
    assert receipt["native_sha256"] == sha256_hex(canonical_json_bytes(original))
    assert (
        json.loads(checked)["executive_summary"][0]["surface_text"]
        == original["executive_summary"][0]["surface_text"]
    )


@pytest.mark.parametrize(
    "bad",
    ["old_schema", "no_primary", "foreign_field", "absent_step", "section", "origin", "graph"],
)
def test_gplus_rejects_wrong_contract_source_permission_or_version(bad):
    api()
    original = value()
    selected = original["executive_summary"][0]
    if bad == "old_schema":
        original["schema_version"] = "nurse.v2.atomic-typed-report-expression-response/1"
    elif bad == "no_primary":
        selected["primary_fields"] = []
    elif bad == "foreign_field":
        selected["primary_fields"][0]["field"] = "fault_denominator"
    elif bad == "absent_step":
        original["executive_summary"] = []
        original["step_findings"] = {"s8": [selected]}
    elif bad == "section":
        original["executive_summary"] = []
        original["next_checks"] = [selected]
    with pytest.raises(ValueError):
        bind(original, mutation=bad)


def test_gplus_omission_is_legal_uncovered_and_not_repaired():
    api()
    original = value()
    original["executive_summary"] = []
    receipt, checked = bind(original)
    assert receipt["bindings"] == []
    assert not json.loads(checked)["executive_summary"]
