"""Evidence presented to the SRE has a separate source namespace for every past case."""

import json

import pytest

from nurse_rcl.v2.live.ops_sre_historical_cases import HISTORY_GUIDANCE, HistoricalContext


def precedents(query_id, core_digest):
    return HistoricalContext.from_retrieval(
        dict(
            schema_version="nurse.history-retrieval/1",
            query_id=query_id,
            matches=[dict(
                rank=1, case_id="opq_" + "9" * 32, distance=0.125,
                evidence=[dict(text="CPU usage rose.", source_ref="#/evidence/summary/1")],
            )],
        ),
        query_id=query_id,
        core_result_digest=core_digest,
    )


def test_historical_context_retains_only_case_scoped_evidence_and_rejects_self_matches():
    history = precedents("opq_" + "1" * 32, "a" * 64)
    [case] = history.model_dump(mode="json")["cases"]
    assert set(case) == {"case_id", "evidence"}
    assert set(case["evidence"][0]) == {"text", "source_ref"}
    assert case["case_id"] in history.markdown() and "#/evidence/summary/1" in history.markdown()
    with pytest.raises(ValueError, match="current or a duplicate case"):
        precedents(case["case_id"], "a" * 64)


def test_composition_pins_embedding_model_before_the_core_runs():
    from nurse_rcl.v2.live.ops_sre_gplus_case import (
        GplusHistoricalCompositionPlan,
        GplusHistoryMethodPlan,
        check_history_receipt,
        composition_grant,
        composition_sha256,
    )

    settings = dict(
        core_plan_path="/plans/core.json", core_plan_sha256="b" * 64, run_root="/results",
        investigation_decision_limit=2, historical_library_path="/input/historical.json",
        historical_library_sha256="e" * 64, historical_query_group_id="group-1",
    )
    default = GplusHistoricalCompositionPlan(**settings)
    assert composition_grant(default)["history_embedding_model"] == "doubao-embedding-vision-251215"
    alternative = GplusHistoricalCompositionPlan(
        **settings, history_embedding_model="doubao-embedding-text-240715"
    )
    assert composition_sha256(default) != composition_sha256(alternative)
    with pytest.raises(ValueError):
        GplusHistoricalCompositionPlan(**settings, history_embedding_model="arbitrary-model")
    rules = GplusHistoryMethodPlan(**settings, history_method="fixed-rule")
    embedded = GplusHistoryMethodPlan(**settings, history_method="embedding")
    assert composition_sha256(embedded) != composition_sha256(rules)
    assert composition_grant(rules)["history_method"] == "fixed-rule"
    receipt = dict(library_sha256="e" * 64, model=None, method="fixed-rule",
                   rule_id="nurse.typed-idf-jaccard/1",
                   metric="idf-weighted-jaccard-distance")
    check_history_receipt(rules, receipt)
    with pytest.raises(ValueError):
        check_history_receipt(embedded, receipt)


def test_model_and_profile_are_pinned_when_preparing_a_v5_plan(monkeypatch):
    from nurse_rcl.v2.live.ops_sre_gplus_case import (
        GplusAmplitudeQualityPlan,
        check_history_receipt,
        composition_grant,
        composition_sha256,
    )

    settings = dict(core_plan_path="/plans/core.json", core_plan_sha256="b" * 64,
                    run_root="/results", investigation_decision_limit=2,
                    historical_library_path="/input/reviewed.json",
                    historical_library_sha256="e" * 64, historical_query_group_id="query")
    monkeypatch.delenv("NURSE_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("ARK_EMBEDDING_MODEL", raising=False)
    rule = GplusAmplitudeQualityPlan(**settings)
    assert composition_grant(rule)["history_method"] == "fixed-rule"
    monkeypatch.setenv("NURSE_EMBEDDING_MODEL", "text-embedding-3-small")
    embedded = GplusAmplitudeQualityPlan(**settings)
    assert composition_grant(embedded)["history_method"] == "embedding"
    assert composition_sha256(embedded) != composition_sha256(rule)
    assert GplusAmplitudeQualityPlan.model_validate_json(rule.model_dump_json()) == rule
    receipt = dict(library_sha256="e" * 64, model="text-embedding-3-small", method="embedding",
                   evidence_profile="amplitude-quality", metric="euclidean", rule_id=None)
    check_history_receipt(embedded, receipt)
    with pytest.raises(ValueError, match="profile"):
        check_history_receipt(embedded, dict(receipt, evidence_profile="direction-only"))


def test_focused_request_sends_history_as_untrusted_side_information():
    from nurse_rcl.v2.live.ops_sre_brief_exchange import _body, _request
    from nurse_rcl.v2.live.ops_sre_focused_exchange import FOCUSED_PROTOCOL

    current = dict(
        protocol=FOCUSED_PROTOCOL, opaque_case_id="opq_" + "1" * 32,
        case_snapshot_id="b" * 64, context={"selector_choices": ["A1"]},
    )
    original_history = precedents(current["opaque_case_id"], "a" * 64)
    history = original_history.model_view()
    original = _request(current, report=False)
    wire = _request(dict(current, historical_precedents=history), report=False)
    body = _body(wire, report=False)
    assert body["historical_precedents"] == history
    assert original_history.cases[0].case_id.encode() not in wire
    assert HistoricalContext.from_model_view(history) == original_history
    assert body["context"] == _body(original, report=False)["context"]
    assert json.loads(wire)["messages"][0]["content"].endswith(HISTORY_GUIDANCE)
    assert "historical_precedents" not in _body(original, report=False)
    wrong = dict(history, query_id="opq_" + "f" * 32)
    with pytest.raises(ValueError, match="frozen query"):
        _body(_request(dict(current, historical_precedents=wrong), report=False), report=False)


def test_gplus_receives_precedents_without_changing_current_evidence_permissions():
    from nurse_rcl.v2.live.ops_sre_gplus_report import (
        bind_gplus_report,
        gplus_report_body,
        project_gplus_report,
    )
    from tests.test_v2_live_sre_typed_report_wire import context

    report, refs = context()
    original = project_gplus_report(report, references=refs).wire_bytes
    history = precedents(report.metadata.evidence.opaque_case_id, report.core_result_digest)
    wire = project_gplus_report(report, references=refs, historical_precedents=history).wire_bytes
    historical, supplied = gplus_report_body(wire, references=refs)
    baseline, original_store = gplus_report_body(original, references=refs)
    assert historical["historical_precedents"] == history.model_view()
    assert json.loads(wire)["messages"][0]["content"].endswith(HISTORY_GUIDANCE)
    assert historical["graph_sha256"] == baseline["graph_sha256"]
    assert supplied.to_packet() == original_store.to_packet()
    baseline_prompt = bind_gplus_report(original).prompt_schema_sha256
    assert bind_gplus_report(wire).prompt_schema_sha256 != baseline_prompt
    with pytest.raises(ValueError, match="frozen query"):
        project_gplus_report(report, references=refs, historical_precedents=history.model_copy(
            update=dict(core_result_digest="0" * 64)
        ))
