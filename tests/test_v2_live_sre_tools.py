"""Question-oriented inspection must not acquire measurement or scoring authority."""

# ruff: noqa: F811 -- pytest fixtures intentionally use imported fixture names.

import importlib
import importlib.util
import json
from dataclasses import replace

import pytest

from nurse_rcl.v2.live.ops_nodes import terminal_nodes
from nurse_rcl.v2.live.ops_sre_encoding import expand
from tests.test_v2_live_ops_tools import complete, prepared_cases  # noqa: F401


def api():
    name = "nurse_rcl.v2.live.ops_sre_tools"
    assert importlib.util.find_spec(name), "SRE question-oriented tool profile is not implemented"
    return importlib.import_module(name)


def toolbox(complete, *, published=False):
    data, blocks, evidence, _ = complete
    tool = api().OpsSreToolbox(data=data, block_root=blocks)
    if published:
        tool.admit_core(nodes=terminal_nodes(data, evidence), steps=evidence.session.steps)
    return tool


def inspect(tool, name, **parameters):
    return expand(json.loads(tool.execute(name, {**parameters, "cursor": None}).model_view_json))


def pages(tool, name, **parameters):
    cursor = None
    while True:
        outcome = tool.execute(name, {**parameters, "cursor": cursor})
        view = expand(json.loads(outcome.model_view_json))
        yield view
        cursor = view["page"]["next_cursor"]
        if cursor is None:
            break


def test_sre_profile_and_fixed_parameters_are_separate_from_legacy(complete):
    tool = toolbox(complete)
    capabilities = tool.capabilities()
    assert capabilities["investigation_profile_id"] == "hermes-sre-tools/4"
    assert set(capabilities["tools"]) == {
        "inspect_case", "inspect_candidate", "inspect_dependency", "compare_candidates",
        "browse_index", "read_node", "measure_signal", "compare_timing", "check_mechanism",
    }
    assert capabilities["tools"]["inspect_candidate"] == ["candidate_service_key", "cursor"]
    assert set(capabilities["parameter_contracts"]) == set(capabilities["tools"])


def test_case_view_contains_windows_summary_and_legal_entries_without_measurement(
    complete, monkeypatch,
):
    tool = toolbox(complete)

    def forbidden(*args, **kwargs):
        pytest.fail("aggregate inspection must not measure or rerun core")

    monkeypatch.setattr("nurse_rcl.v2.live.ops_tools._measure_values", forbidden)
    monkeypatch.setattr("nurse_rcl.v2.live.normal_queries.run_normal_queries", forbidden)
    view = inspect(tool, "inspect_case")
    assert view["boundary"]["baseline_window"]
    assert view["boundary"]["fault_window"]
    assert view["boundary"]["candidate_count"] == len(complete[0].catalog.candidate_service_keys)
    assert view["page"]["records"]
    assert view["next_checks"]
    assert view["truth_value"] == "unknown"
    assert view["score_authority"] is False
    assert "points" not in json.dumps(view)


def test_candidate_summary_is_not_an_episode_or_unpublished_score(complete):
    data = complete[0]
    tool = toolbox(complete)
    candidate = data.sources[0].candidate_service_key
    view = inspect(tool, "inspect_candidate", candidate_service_key=candidate)
    assert view["sections_state"][candidate] == {
        "mechanisms": "not_published", "scoring": "not_published",
    }
    summaries = [r for page in pages(tool, "inspect_candidate", candidate_service_key=candidate)
                 for r in page["page"]["records"] if r["source_role"] == "prepared_summary"]
    assert summaries
    row = summaries[0]
    assert row["payload"]["episode_status"] == "not_measured_by_summary"
    assert row["payload"]["baseline"]["valid_point_count"] > 0
    assert row["payload"]["fault"]["valid_point_count"] > 0
    assert row["targets"] == [candidate]
    assert row["unit"] and row["window_ns"] and row["source_refs"]
    for check in view["next_checks"]:
        assert check["tool"] in api().SRE_TOOL_PARAMETERS
        assert set(check["parameters"]) == set(api().SRE_TOOL_PARAMETERS[check["tool"]])


def test_non_top_five_candidate_and_full_mechanism_conditions_are_available(complete):
    tool = toolbox(complete, published=True)
    data, _, formal, _ = complete
    candidate = formal.ranking.ordered_service_keys[-1]
    frozen = formal.model_dump_json()
    found_mechanism = False
    for view in pages(tool, "inspect_candidate", candidate_service_key=candidate):
        assert view["sections_state"][candidate]["scoring"] == "published"
        found_mechanism |= any(r["node"] == "mechanisms" for r in view["page"]["records"])
        for row in view["page"]["records"]:
            if row["card_id"] in view["critical_record_ids"]:
                assert row["payload"] == json.loads(tool.cards[row["card_id"]].payload_json)
                assert row["source_role"] == "core_published"
                assert row["formal_scoring_use"] == "not_asserted_by_publication"
    assert found_mechanism
    assert formal.model_dump_json() == frozen
    assert data.catalog.candidate_service_keys == tool._data.catalog.candidate_service_keys


def test_pair_does_not_expand_individual_fact_scope_or_compute_scores(complete):
    data = complete[0]
    left, right = data.catalog.candidate_service_keys[:2]
    tool = toolbox(complete)
    view = inspect(tool, "compare_candidates",
                   left_candidate_service_key=left, right_candidate_service_key=right)
    assert view["scope"] == [left, right]
    assert view["sections_state"][left]["scoring"] == "not_published"
    assert view["sections_state"][right]["scoring"] == "not_published"
    summaries = [r for page in pages(tool, "compare_candidates",
        left_candidate_service_key=left, right_candidate_service_key=right)
        for r in page["page"]["records"] if r["node"] == "symptoms"]
    assert {tuple(r["targets"]) for r in summaries} == {(left,), (right,)}
    assert "comparison_score" not in view


def test_dependency_retains_actual_edge_roles_and_missing_self_time(complete):
    from tests.test_v2_graph_snapshots import (
        coverages,
        graph_context,
        graphs,
        pair_from_parents,
        source,
    )

    data, blocks, _, _ = complete
    g = graphs()
    contract, universe, frozen, registry = graph_context()
    pair, _ = pair_from_parents(
        g, time_contract=contract, candidate_universe=universe, frozen_candidates=frozen,
        entity_registry=registry, coverage_parents=coverages(g, contract),
        source_parents=(source(g, contract, kind="fault", errors=2),),
    )
    left, right = data.catalog.candidate_service_keys[:2]
    edge = pair.fault_snapshot.edges[0].model_copy(update={
        "caller_service_key": left, "callee_service_key": right,
    })
    graph = data.graph.graph_pair.model_copy(update={
        "fault_snapshot": data.graph.graph_pair.fault_snapshot.model_copy(update={
            "edges": (edge,),
        }),
    })
    data = replace(data, graph=data.graph.model_copy(update={"graph_pair": graph}))
    tool = api().OpsSreToolbox(data=data, block_root=blocks)
    edge = data.graph.graph_pair.fault_snapshot.edges[0]
    view = inspect(tool, "inspect_dependency",
                   caller_service_key=edge.caller_service_key,
                   callee_service_key=edge.callee_service_key)
    assert view["dependency"]["edge_state"] == "observed"
    assert view["dependency"]["callee_self_time"] == "not_available"
    edges = [r["payload"]["edge"] for r in view["page"]["records"] if r["node"] == "dependency"]
    assert edges
    assert all(e["caller_service_key"] == edge.caller_service_key
               and e["callee_service_key"] == edge.callee_service_key for e in edges)
    assert "caller_latency_not_callee_self_time" in view["limitations"]
    reverse = inspect(tool, "inspect_dependency",
                      caller_service_key=edge.callee_service_key,
                      callee_service_key=edge.caller_service_key)
    matching = [e for e in data.graph.graph_pair.fault_snapshot.edges
                if e.caller_service_key == edge.callee_service_key
                and e.callee_service_key == edge.caller_service_key]
    if not matching:
        assert reverse["dependency"]["edge_state"] == "not_observed"
        assert reverse["truth_value"] == "unknown"


@pytest.mark.parametrize("name,params", [
    ("inspect_case", {"path": "/etc/passwd", "cursor": None}),
    ("inspect_case", {"cursor": "made-up"}),
    ("inspect_candidate", {"candidate_service_key": "labels", "cursor": None}),
    ("inspect_candidate", {"candidate_service_key": [], "cursor": None}),
    ("inspect_dependency", {"caller_service_key": "x", "callee_service_key": "y",
                            "cursor": None}),
])
def test_unregistered_domains_rejected(complete, name, params):
    with pytest.raises(ValueError):
        toolbox(complete).execute(name, params)


def test_pair_requires_distinct_candidates(complete):
    candidate = complete[0].catalog.candidate_service_keys[0]
    with pytest.raises(ValueError, match="distinct"):
        inspect(toolbox(complete), "compare_candidates",
                left_candidate_service_key=candidate, right_candidate_service_key=candidate)


def test_pages_keep_complete_records_and_bind_scope_and_publication(complete):
    tool = toolbox(complete, published=True)
    data, _, formal, _ = complete
    first = inspect(tool, "inspect_case")
    assert "page" in first, first
    assert 0 < first["page"]["count"] <= 16
    assert first["page"]["has_more"]
    assert not first["critical_record_ids"]  # Candidate navigation is not full evidence.
    assert all(r["representation"] == "candidate_overview_not_full_judgment"
               for r in first["page"]["records"])
    assert len(json.dumps(first).encode()) < 80000  # Fixture regression, not a runtime quota.
    assert first["critical_overview"]["unexpanded_count"] > 0
    assert first["critical_overview"]["next_cursor"] == first["page"]["next_cursor"]
    seen = {r["candidate_service_key"] for r in first["page"]["records"]}
    cursor = first["page"]["next_cursor"]
    with pytest.raises(ValueError, match="cursor"):
        tool.execute("inspect_candidate", {
            "candidate_service_key": data.catalog.candidate_service_keys[0], "cursor": cursor,
        })
    while cursor is not None:
        page = expand(json.loads(tool.execute("inspect_case", {"cursor": cursor}).model_view_json))
        ids = {r["candidate_service_key"] for r in page["page"]["records"]}
        assert not seen & ids
        assert page["page"]["count"] <= 16
        assert page["publication_prefix"] == first["publication_prefix"]
        seen |= ids
        cursor = page["page"]["next_cursor"]
    assert len(seen) == first["page"]["total"]
    assert formal.ranking.ordered_service_keys


def test_mechanism_expansion_keeps_entire_conditions_not_scalar_pages(complete):
    tool = toolbox(complete, published=True)
    key, card = next((k, c) for k, c in tool.cards.items() if c.node == "mechanisms")
    for section in ("summary", "payload"):
        view = expand(json.loads(tool.execute("read_node", {
            "card_id": key, "section": section, "cursor": None,
        }).model_view_json))
        assert view["semantic_record"]["payload"] == json.loads(card.payload_json)
        assert view["semantic_record"]["targets"]
        assert view["semantic_record"]["evidence_ids"] == list(card.evidence_ids)
        assert view["next_cursor"] is None


def test_case_overview_exposes_true_refutation_even_for_positive_mechanism(complete):
    from nurse_rcl.artifacts import canonical_json_bytes
    from nurse_rcl.v2.live.ops_tools import card_id

    tool = toolbox(complete)
    template = next(c for c in terminal_nodes(complete[0], complete[2]) if c.node == "mechanisms")
    payload = json.loads(template.payload_json)
    payload["conditions"][0].update(truth_value="true", requirement="refuting")
    card = template.model_copy(update={
        "payload_json": canonical_json_bytes(payload).decode(), "truth_value": "true",
    })
    tool.admit_core(nodes=(card,), steps=())
    view = inspect(tool, "inspect_case")
    row = next(r for v in pages(tool, "inspect_case") for c in v["page"]["records"]
               for r in c["mechanisms"]["states"] if r["card_id"] == card_id(card))
    assert not view["critical_record_ids"]
    assert row["judgment_status_counts"]["refuting_true"] == 1
    assert view["critical_overview"]["unexpanded_judgment_status_counts"]["refuting_true"] == 1
    assert "conditions" not in row
    assert row["expansion"]["parameters"]["card_id"] == card_id(card)


def test_missing_mechanism_conditions_offer_real_registered_checks(complete):
    tool = toolbox(complete, published=True)
    data, _, evidence, _ = complete
    request = next(r for r in data.catalog.requests
                   if r.request_key not in evidence.session.submitted_request_keys
                   and any(n.queryable for n in r.needs))
    assert any(c["tool"] == "measure_signal" and c["parameters"] == {
        "request_key": request.request_key}
        for view in pages(tool, "inspect_candidate",
                          candidate_service_key=request.candidate_service_key)
        for c in view["next_checks"])


def test_old_prefix_does_not_gain_new_cards_or_new_publication_role(complete):
    tool = toolbox(complete)
    nodes = terminal_nodes(complete[0], complete[2])
    tool.admit_core(nodes=nodes[:24], steps=())
    candidate = complete[2].ranking.ordered_service_keys[0]
    params = dict(candidate_service_key=candidate)
    before = inspect(tool, "inspect_candidate", **params)
    cursor = before["page"]["next_cursor"]
    assert cursor is not None
    expected = json.loads(tool.execute("inspect_candidate", dict(params, cursor=cursor)).model_view_json)
    tool.admit_core(nodes=nodes[24:], steps=complete[2].session.steps)
    tool.admit_core(nodes=tuple(tool.cards.values()), steps=())
    inspect(tool, "inspect_case")  # Refresh cached current roles.
    actual = json.loads(tool.execute("inspect_candidate", dict(params, cursor=cursor)).model_view_json)
    # History is frozen; execution hints describe what is legal now, not new facts.
    expected.pop("next_checks")
    actual.pop("next_checks")
    expected.pop("fixed_metadata_bytes")  # Includes current executable hints, not evidence.
    actual.pop("fixed_metadata_bytes")
    assert actual == expected


@pytest.mark.parametrize("kind,container", [
    ("fixed_summary", {"facts": {"facts": [
        {"predicate_kind": "local_anomaly_observed", "truth_value": "true"},
        {"predicate_kind": "persistent_pressure", "truth_value": "unknown"},
        {"predicate_kind": "verified_normal", "truth_value": "false"},
    ]}}),
    ("supplementary_episode", {"feedback": {"predicates": [
        {"predicate_kind": "local_anomaly_observed", "truth_value": "true"},
        {"predicate_kind": "persistent_pressure", "truth_value": "unknown"},
        {"predicate_kind": "verified_normal", "truth_value": "false"},
    ]}}),
    ("mechanism", {"conditions": [
        {"operator": "or", "requirement": "necessary", "truth_value": "true",
         "leaves": [{"truth_value": "true"}, {"truth_value": "unknown"},
                    {"truth_value": "false"}]},
    ]}),
])
def test_overview_keeps_nested_judgment_status_without_claiming_complete_evidence(kind, container):
    from nurse_rcl.v2.live.ops_sre_views import critical_overview, critical_record, overview_record

    row = dict(card_id="fixture", node="symptoms", source_role="core_published",
               truth_value="true", payload=dict(kind=kind, **container))
    view = overview_record(row)
    assert view["judgment_status_counts"].get("unknown") == 1
    assert view["judgment_status_counts"].get("false") == 1
    assert view["judgment_status_by_group"]
    assert critical_record(view), "later public memory must retain critical overview status"
    assert view["representation"] == "overview_not_full_judgment"
    assert view["expansion"]["parameters"]["card_id"] == "fixture"
    pending = critical_overview([row], expanded_ids=set(), next_cursor=None)
    assert pending["unexpanded_judgment_status_counts"]["unknown"] == 1
    assert "metadata_counts_not_independent_support" in pending["meaning"]


def test_old_page_does_not_offer_already_measured_request(complete):
    tool = toolbox(complete)
    tool.admit_core(nodes=terminal_nodes(complete[0], complete[2]), steps=())
    candidate = complete[2].ranking.ordered_service_keys[0]
    params = dict(candidate_service_key=candidate)
    first = inspect(tool, "inspect_candidate", **params)
    cursor = first["page"]["next_cursor"]
    while cursor is not None:
        before = expand(json.loads(tool.execute("inspect_candidate", dict(params, cursor=cursor))
                                   .model_view_json))
        request = next((c["parameters"]["request_key"] for c in before["next_checks"]
                        if c["tool"] == "measure_signal"), None)
        if request is not None:
            tool.execute("measure_signal", {"request_key": request})
            after = expand(json.loads(tool.execute("inspect_candidate", dict(params, cursor=cursor))
                                      .model_view_json))
            assert after["page"] == before["page"]
            assert after["publication_prefix"] == before["publication_prefix"]
            assert not any(c["tool"] == "measure_signal" and
                           c["parameters"]["request_key"] == request
                           for c in after["next_checks"])
            return
        cursor = before["page"]["next_cursor"]
    pytest.fail("fixture must have a continuation containing a registered request")


@pytest.mark.parametrize("dataset", ["rcabench", "aiops25"])
def test_sre_inspection_supports_both_adapters(prepared_cases, dataset):
    data, blocks, _ = prepared_cases(dataset)
    tool = api().OpsSreToolbox(data=data, block_root=blocks)
    view = inspect(tool, "inspect_case")
    assert view["page"]["records"]
    assert view["boundary"]["missing_modalities"] == list(
        data.diagnostics.legacy.sources.adapter.time_contract.missing_modalities)
