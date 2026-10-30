"""Lossless explicitly versioned graph tables; no relaxed JSON resource ceiling."""

from copy import deepcopy

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from tests.test_v2_live_sre_gplus_graph import catalog


def test_table_graph_expands_to_exact_v1_sources_permissions_and_every_closure():
    from nurse_rcl.v2.live.ops_sre_gplus_graph import (
        build_gplus_graph,
        expand_gplus_graph,
        gplus_closure,
    )

    source = catalog()
    expanded = build_gplus_graph(source)
    packed = build_gplus_graph(source, representation="tables")
    before = canonical_json_bytes(packed)
    assert packed["schema_version"] == "nurse.v2.sre-gplus-evidence-dag/2"
    assert expand_gplus_graph(packed) == expanded
    assert packed["expanded_sha256"] == sha256_hex(canonical_json_bytes(expanded))
    for choice in expanded["choices"]:
        assert gplus_closure(packed, choice["leaf_id"]) == gplus_closure(
            expanded, choice["leaf_id"]
        )
    assert canonical_json_bytes(packed) == before
    assert len(before) < len(canonical_json_bytes(expanded))


def test_repeated_permission_expressions_and_adjacencies_are_stored_once():
    from nurse_rcl.v2.live.ops_sre_gplus_graph import build_gplus_graph, expand_gplus_graph

    graph = build_gplus_graph(catalog(), representation="tables")
    assert "expression_pool" in graph
    assert all("targets" in group and "rows" in group for group in graph["edge_groups"])
    logical = expand_gplus_graph(graph)
    permissions = [node for node in logical["nodes"] if node["kind"] == "permission"]
    assert len(graph["expression_pool"]) < len(permissions)


def test_shared_expression_amplification_is_rejected_before_full_expansion():
    from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
    from nurse_rcl.v2.live.ops_sre_gplus_graph import build_gplus_graph, expand_gplus_graph

    packet = build_gplus_graph(catalog(), representation="tables")
    expression = deepcopy(packet["expression_pool"][0])
    expression["claim"]["recommendation"] = dict(large_literal="x" * 400000)
    packet["expression_pool"] = [expression]
    table = next(t for t in packet["node_tables"] if t["kind"] == "permission")
    column = 1 + table["columns"].index("expression_index")
    for row in table["rows"]:
        row[column] = 0
    original = deepcopy(table["rows"][0])
    for number in range(200):
        row = deepcopy(original)
        row[0] = len(packet["node_ids"])
        row[column] = 0
        packet["node_ids"].append("permission:" + sha256_hex(str(number).encode()))
        table["rows"].append(row)
    with pytest.raises(ExchangeBoundaryError) as caught:
        expand_gplus_graph(packet)
    assert caught.value.boundary == "logical_bytes"


def test_expression_assertion_values_reference_the_exact_original_source_field_once():
    from nurse_rcl.v2.live.ops_sre_gplus_graph import build_gplus_graph, expand_gplus_graph

    source = catalog()
    packet = build_gplus_graph(source, representation="tables")
    assert all(
        "assertion_fields" in row["claim"] and "assertions" not in row["claim"]
        for row in packet["expression_pool"]
    )
    assert expand_gplus_graph(packet) == build_gplus_graph(source)


def test_builder_bounds_repeated_expression_work_before_deepcopy(monkeypatch):
    import nurse_rcl.v2.live.ops_sre_gplus_graph as module
    from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
    from nurse_rcl.v2.live.ops_sre_gplus_graph_tables import GRAPH_WORK_BYTES

    source = catalog()
    source["rows"] = source["rows"][:1]
    expression = source["rows"][0]["expression"]
    expression["sections"] = ["step_findings"]
    expression["step_refs"] = [f"s{i}" for i in range(180)]
    expression["claim"]["recommendation"] = dict(literal="x" * 400000)
    copied = 0
    original = module.deepcopy

    def bounded_copy(value):
        nonlocal copied
        copied += len(canonical_json_bytes(value))
        assert copied <= GRAPH_WORK_BYTES, "builder copied replicated data before work guard"
        return original(value)

    monkeypatch.setattr(module, "deepcopy", bounded_copy)
    with pytest.raises(ExchangeBoundaryError) as caught:
        module.build_gplus_graph(source, representation="tables")
    assert caught.value.boundary == "logical_bytes"


def test_repeated_large_source_field_is_charged_before_reserializing_thousands(monkeypatch):
    import nurse_rcl.v2.live.ops_sre_gplus_graph_tables as tables
    from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
    from nurse_rcl.v2.live.ops_sre_gplus_graph import build_gplus_graph, expand_gplus_graph

    packet = build_gplus_graph(catalog(), representation="tables")
    table = next(t for t in packet["node_tables"] if t["kind"] == "context")
    source = table["rows"][0][1 + table["columns"].index("source")]["fact"]
    source["values"][0]["value"] = "x" * 1000000
    pair = [source["fact_ref"], source["values"][0]["name"]]
    expression = deepcopy(packet["expression_pool"][0])
    expression["claim"]["assertion_fields"] = [pair] * 5000
    packet["expression_pool"] = [expression]
    permissions = next(t for t in packet["node_tables"] if t["kind"] == "permission")
    column = 1 + permissions["columns"].index("expression_index")
    for row in permissions["rows"]:
        row[column] = 0
    original, large_serializations = tables.canonical_json_bytes, 0

    def counted(value):
        nonlocal large_serializations
        if type(value) is str and len(value) > 500000:
            large_serializations += 1
            assert large_serializations <= 80, "unbounded repeated scalar serialization"
        return original(value)

    monkeypatch.setattr(tables, "canonical_json_bytes", counted)
    with pytest.raises(ExchangeBoundaryError) as caught:
        expand_gplus_graph(packet)
    assert caught.value.boundary == "logical_bytes"


@pytest.mark.parametrize(
    "bad",
    [
        "schema",
        "node_index",
        "duplicate_node",
        "edge_index",
        "bool_index",
        "relation",
        "choice",
        "hash",
        "extra_column",
        "field",
        "unknown_key",
    ],
)
def test_table_graph_rejects_invalid_or_changed_references_and_typed_values(bad):
    from nurse_rcl.v2.live.ops_sre_gplus_graph import build_gplus_graph, expand_gplus_graph

    packet = deepcopy(build_gplus_graph(catalog(), representation="tables"))
    if bad == "schema":
        packet["schema_version"] = "nurse.v2.sre-gplus-evidence-dag/1"
    elif bad == "node_index":
        packet["node_tables"][0]["rows"][0][0] = -1
    elif bad == "duplicate_node":
        packet["node_tables"][0]["rows"].append(deepcopy(packet["node_tables"][0]["rows"][0]))
    elif bad == "edge_index":
        packet["edge_groups"][0]["rows"][0][0] = len(packet["node_ids"])
    elif bad == "bool_index":
        packet["edge_groups"][0]["rows"][0][0] = True
    elif bad == "relation":
        packet["edge_groups"][0]["relation"] = "guessed_cause"
    elif bad == "choice":
        packet["choice_table"]["rows"][0][0] = "foreign"
    elif bad == "hash":
        packet["expanded_sha256"] = "f" * 64
    elif bad == "extra_column":
        packet["node_tables"][0]["columns"].append("guessed")
    elif bad == "field":
        table = next(row for row in packet["node_tables"] if row["kind"] == "field")
        table["rows"][0][-1] = "changed"
    else:
        packet["extra"] = 1
    with pytest.raises(ValueError):
        expand_gplus_graph(packet)
