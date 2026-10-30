"""Narrow supplementary operations cannot acquire formal RCL authority."""

import importlib
import importlib.util
import json

import pytest

from nurse_rcl.v2.live.normal_queries import run_normal_queries
from nurse_rcl.v2.live.normal_scoring import score_normal_case
from nurse_rcl.v2.live.ops_nodes import terminal_nodes
from nurse_rcl.v2.live.repair_measurements import _measure_values
from nurse_rcl.v2.live.repair_queries import _feedback_values
from tests.test_v2_live_normal_queries import prepared_cases  # noqa: F401


@pytest.fixture(scope="module")
def complete(prepared_cases, tmp_path_factory):  # noqa: F811
    # Nine services with eight paid measurements, including a non-Top-K service.
    # The ordinary RCABench fixture intentionally has no eligible paid steps.
    data, blocks, _ = prepared_cases("eight")
    root = tmp_path_factory.mktemp("ops-tool-reference")
    session = run_normal_queries(data=data, ledger_root=root / "queries", block_root=blocks)
    return data, blocks, score_normal_case(data=data, session=session), None


def api():
    name = "nurse_rcl.v2.live.ops_tools"
    assert importlib.util.find_spec(name), "missing isolated investigation tools"
    return importlib.import_module(name)


def toolbox(complete):
    data, blocks, _, _ = complete
    return api().OpsToolbox(data=data, block_root=blocks)


def test_capabilities_include_complete_non_ranked_domain_and_no_raw_points(complete):
    tool = toolbox(complete)
    data, _, _, _ = complete
    capabilities = tool.capabilities()
    assert tuple(capabilities["candidate_service_keys"]) == data.catalog.candidate_service_keys
    assert {r["request_key"] for r in capabilities["requests"]} == {
        r.request_key for r in data.catalog.requests
    }
    assert "points" not in json.dumps(capabilities)
    assert "SECRET_LABEL" not in json.dumps(capabilities)
    assert not {"score", "rank", "execute", "shell", "read_file"} & set(capabilities["tools"])


def test_measurement_is_one_registered_source_and_matches_shared_values(complete, monkeypatch):
    module = api()
    tool = toolbox(complete)
    data, _, _, _ = complete
    request = next(r for r in data.catalog.requests if any(n.queryable for n in r.needs))
    source = next(s for s in data.sources if s.source_sha256 == request.source_sha256)
    measured = _measure_values(source)
    expected = _feedback_values(data.catalog, source, measured)
    calls = []

    def one_source(value):
        calls.append(value.source_sha256)
        return _measure_values(value)

    monkeypatch.setattr(module, "_measure_values", one_source)
    result = tool.execute("measure_signal", {"request_key": request.request_key})
    assert calls == [source.source_sha256]
    assert result.origin == "measured"
    assert result.reused_core_evidence_ids == ()
    body = json.loads(result.evidence.payload_json)
    assert body["measurement_sha256"] == measured.measurement_sha256
    assert body["feedback"] == expected.model_dump(mode="json")
    assert body["score_authority"] is False
    assert result.evidence.unit == source.observation.unit
    assert "points" not in body
    assert tool.capabilities()["onset_ids"]


def test_matching_completed_core_measurement_is_reused_but_not_promoted(complete, monkeypatch):
    module = api()
    tool = toolbox(complete)
    data, _, evidence, _ = complete
    tool.admit_core(nodes=terminal_nodes(data, evidence), steps=evidence.session.steps)
    step = evidence.session.steps[0]

    def forbidden(*args):
        pytest.fail("matching completed core measurement should not be computed twice")

    monkeypatch.setattr(module, "_measure_values", forbidden)
    result = tool.execute("measure_signal", {"request_key": step.feedback.request_key})
    assert result.origin == "core_reuse"
    assert step.measurement.measurement_sha256 in result.reused_core_evidence_ids
    assert json.loads(result.evidence.payload_json)["score_authority"] is False
    wrong = step.model_copy(update={"measurement": step.measurement.model_copy(update={
        "semantics_sha256": "0" * 64,
    })})
    with pytest.raises(ValueError):
        toolbox(complete).admit_core(nodes=(), steps=(wrong,))


@pytest.mark.parametrize("tool_name,params", [
    ("read_file", {"path": "/etc/passwd"}),
    ("measure_signal", {"request_key": "../../labels.json"}),
    ("measure_signal", {"request_key": "case-other/query"}),
    ("measure_signal", {"request_key": "__import__('os').system('id')"}),
    ("measure_signal", {"request_key": "injection_metadata"}),
    ("measure_signal", {"request_key": "evaluator_map"}),
    ("read_node", {"card_id": "unknown", "label": "answer"}),
    ("compare_timing", {"source_onset_id": "unknown", "target_onset_id": "unknown",
                        "source_bounds": [1, 2]}),
    ("measure_signal", {"request_key": ["a", "b"]}),
])
def test_unregistered_or_expansive_parameters_are_rejected(complete, tool_name, params):
    tool = toolbox(complete)
    with pytest.raises(ValueError):
        tool.execute(tool_name, params)


def test_missing_modality_is_unknown_not_healthy(complete):
    tool = toolbox(complete)
    data, _, _, _ = complete
    result = tool.execute("list_observations", {
        "candidate_service_key": data.catalog.candidate_service_keys[0], "modality": "logs",
    })
    assert result.evidence.availability == "unavailable"
    assert result.evidence.truth_value == "unknown"
    assert result.evidence.limitations


def test_present_modality_uses_adapter_names_and_returns_registered_series(complete):
    tool = toolbox(complete)
    data, _, _, _ = complete
    source = data.sources[0]
    result = tool.execute("list_observations", {
        "candidate_service_key": source.candidate_service_key,
        "modality": source.observation.modality,
    })
    assert result.evidence.availability == "available"
    assert json.loads(result.evidence.payload_json)["requests"]


def test_supplementary_measurement_accepts_candidate_outside_formal_top_five(complete):
    tool = toolbox(complete)
    data, _, evidence, _ = complete
    request = next(r for r in data.catalog.requests
                   if r.candidate_service_key not in evidence.ranking.ordered_service_keys[:5])
    result = tool.execute("measure_signal", {"request_key": request.request_key})
    assert result.origin == "measured"
    assert json.loads(result.evidence.payload_json)["candidate_service_key"] == (
        request.candidate_service_key
    )


def test_timing_resolves_only_actual_received_onsets(complete):
    tool = toolbox(complete)
    data, _, evidence, _ = complete
    tool.admit_core(nodes=terminal_nodes(data, evidence), steps=evidence.session.steps)
    ids = tool.capabilities()["onset_ids"]
    assert len(ids) >= 2
    result = tool.execute("compare_timing", {
        "source_onset_id": ids[0], "target_onset_id": ids[1],
    })
    body = json.loads(result.evidence.payload_json)
    assert "source_bounds_ns" in body
    assert body["score_authority"] is False
    assert "timing_and_dependency_do_not_confirm_causality" in result.evidence.limitations


def test_mechanism_recheck_preserves_formal_table_and_unmeasured_unknowns(complete):
    tool = toolbox(complete)
    data, _, evidence, _ = complete
    nodes = terminal_nodes(data, evidence)
    tool.admit_core(nodes=nodes, steps=evidence.session.steps)
    # Hash order does not identify an unmeasured mechanism. Select the actual
    # ninth (unmeasured) candidate/condition instead of an arbitrary first card.
    request = next(r for r in data.catalog.requests
                   if r.request_key not in evidence.session.submitted_request_keys
                   and any(n.queryable and n.predicate_kind == "local_pressure_observed"
                           for n in r.needs))
    mechanism = next(n.mechanism_key for n in request.needs
                     if n.queryable and n.predicate_kind == "local_pressure_observed")
    card_id = next(r["card_id"] for r in tool.capabilities()["nodes"]
                   if r["node"] == "mechanisms" and r["parameters"] == dict(
                       candidate_service_key=request.candidate_service_key,
                       mechanism_key=mechanism))
    formal = tool.execute("read_node", {"card_id": card_id}).evidence
    result = tool.execute("check_mechanism", {"card_id": card_id})
    body = json.loads(result.evidence.payload_json)
    assert body["formal_decision"] == json.loads(formal.payload_json)["decision"]
    assert any(c["predicate_kind"] == "local_pressure_observed" and c["truth_value"] == "unknown"
               for c in body["conditions"])
    assert body["score_authority"] is False
    assert "supplementary_recheck_not_formal_replacement" in result.evidence.limitations
    assert tool.execute("read_node", {"card_id": card_id}).evidence == formal


def test_core_publication_rejects_other_snapshot(complete):
    tool = toolbox(complete)
    card = next(iter(tool.cards.values()))
    wrong = card.model_copy(update={"case_snapshot_id": "0" * 64})
    with pytest.raises(ValueError, match="snapshot"):
        tool.admit_core(nodes=(wrong,), steps=())


def test_supplementary_recheck_updates_conditions_and_next_check_without_changing_formal(complete):
    tool = toolbox(complete)
    data, _, evidence, _ = complete
    tool.admit_core(nodes=terminal_nodes(data, evidence), steps=evidence.session.steps)
    request = next(r for r in data.catalog.requests
                   if r.request_key not in evidence.session.submitted_request_keys
                   and any(n.queryable and n.predicate_kind == "local_pressure_observed"
                           for n in r.needs))
    mechanism = next(n.mechanism_key for n in request.needs
                     if n.queryable and n.predicate_kind == "local_pressure_observed")
    row = next(r for r in tool.capabilities()["nodes"] if r["node"] == "mechanisms"
               and r["parameters"] == dict(candidate_service_key=request.candidate_service_key,
                                           mechanism_key=mechanism))
    key = row["card_id"]
    before = tool.execute("read_node", {"card_id": key}).evidence
    original = json.loads(before.payload_json)
    assert any(request.request_key in c["request_keys"] for c in original["distinguishing_checks"])
    tool.execute("measure_signal", {"request_key": request.request_key})
    checked = json.loads(tool.execute("check_mechanism", {"card_id": key}).evidence.payload_json)
    assert any(c["predicate_kind"] == "local_pressure_observed" and c["truth_value"] == "true"
               for c in checked["conditions"])
    assert all(request.request_key not in c["request_keys"]
               for c in checked["distinguishing_checks"])
    assert tool.execute("read_node", {"card_id": key}).evidence == before


@pytest.mark.parametrize("dataset", ["rcabench", "aiops25"])
def test_tool_catalog_accepts_both_dataset_adapters(prepared_cases, dataset):  # noqa: F811
    data, blocks, _ = prepared_cases(dataset)
    tool = api().OpsToolbox(data=data, block_root=blocks)
    source = next(s for s in data.sources if s.candidate_service_key is not None)
    listed = tool.execute("list_observations", {
        "candidate_service_key": source.candidate_service_key,
        "modality": source.observation.modality,
    })
    assert listed.evidence.availability == "available"
    assert json.loads(listed.evidence.payload_json)["requests"]
