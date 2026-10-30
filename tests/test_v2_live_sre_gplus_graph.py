"""Dynamic normal G+ graph; no research-case assumptions or model calls."""

from copy import deepcopy
from importlib import import_module, util

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from tests.sre_typed_report_fixtures import fact, material


def api():
    name = "nurse_rcl.v2.live.ops_sre_gplus_graph"
    assert util.find_spec(name), "missing normal G+ graph"
    return import_module(name)


def catalog():
    from nurse_rcl.v2.live.ops_sre_typed_report_binding import typed_expression_catalog

    supply, store = material()
    return typed_expression_catalog(supply, store)


def test_graph_keeps_actual_sources_fields_units_permissions_and_sink_choices():
    module = api()
    original = catalog()
    before = canonical_json_bytes(original)
    graph = module.build_gplus_graph(original)
    nodes = module.validate_gplus_graph(graph)
    fields = [n["data"] for n in nodes.values() if n["kind"] == "field"]
    row = next(
        f for f in fields if f["evidence_ref"] == "f1" and f["field"] == "baseline_denominator"
    )
    assert row["value"] == 0 and row["json_type"] == "integer"
    assert row["unit"]["state"] == "supplied_null"
    assert next(f for f in fields if f["field"] == "unknown")["unit"] == {"state": "not_supplied"}
    contexts = [n["data"] for n in nodes.values() if n["kind"] == "context"]
    assert {n["evidence_ref"] for n in contexts} == {"f1", "f2", "f3"}
    assert all(n["source"]["fact"]["target_service_keys"] == ["other-service"] for n in contexts)
    assert any(c["step_ref"] == "s7" for c in graph["choices"])
    assert not any(nodes[e["from"]]["kind"] == "leaf" for e in graph["edges"])
    assert canonical_json_bytes(original) == before
    for choice in graph["choices"]:
        closure = module.gplus_closure(graph, choice["leaf_id"])
        assert closure["mandatory_complete"] and len(closure["nodes"]) <= 64
        assert choice["closure_admitted"]


def test_no_research_menu16_or_fixed_target_step_limit():
    module = api()
    from nurse_rcl.v2.live.ops_sre_fact_store import build_fact_store
    from nurse_rcl.v2.live.ops_sre_typed_report_binding import typed_expression_catalog

    supply, _ = material(step="s1", target="unrelated-service")
    rows = [fact(f"f{i}", {"value": i}, target=f"service-{i}") for i in range(1, 22)]
    supply.update(provided_fact_refs=[r.fact.fact_ref for r in rows], formal_fact_refs=[], steps=[])
    graph = module.build_gplus_graph(typed_expression_catalog(supply, build_fact_store(rows)))
    assert len(graph["choices"]) > 16
    assert len([n for n in graph["nodes"] if n["kind"] == "context"]) == 21


def test_over64_mandatory_closure_is_unavailable_not_trimmed():
    module = api()
    from nurse_rcl.v2.live.ops_sre_fact_store import build_fact_store
    from nurse_rcl.v2.live.ops_sre_typed_report_binding import typed_expression_catalog

    supply, _ = material()
    row = fact("f1", {f"field_{i}": i for i in range(70)})
    supply.update(provided_fact_refs=["f1"], formal_fact_refs=[], steps=[])
    graph = module.build_gplus_graph(typed_expression_catalog(supply, build_fact_store([row])))
    assert graph["choices"] and not any(c["closure_admitted"] for c in graph["choices"])
    assert len([n for n in graph["nodes"] if n["kind"] == "field"]) == 70
    with pytest.raises(module.ClosureBudgetError):
        module.gplus_closure(graph, graph["choices"][0]["leaf_id"])


@pytest.mark.parametrize("mutation", ["missing_parent", "cycle", "duplicate_node", "leaf_child"])
def test_dangling_cyclic_or_non_sink_graph_rejected(mutation):
    module = api()
    graph = deepcopy(module.build_gplus_graph(catalog()))
    if mutation == "missing_parent":
        graph["edges"][0]["from"] = "absent"
    elif mutation == "duplicate_node":
        graph["nodes"].append(deepcopy(graph["nodes"][0]))
    elif mutation == "leaf_child":
        graph["edges"][0]["from"] = graph["choices"][0]["leaf_id"]
    else:
        field = next(n["id"] for n in graph["nodes"] if n["kind"] == "field")
        graph["edges"].append(
            dict(**{"from": field, "to": field}, relation="available_basis", mandatory=True)
        )
    with pytest.raises(ValueError):
        module.validate_gplus_graph(graph)


def test_empty_legal_catalog_stays_empty_and_does_not_invent_evidence():
    module = api()
    value = catalog()
    value.update(rows=[], sources={}, field_scope={}, condition_boundaries={})
    graph = module.build_gplus_graph(value)
    assert graph["nodes"] == [] and graph["choices"] == []
