from nurse.context_packager import package_verified_rcl_context
from nurse.schemas import EvidenceGraph, NursePrediction, VerifiedEvidence


def _evidence(evidence_id: str, entity: str) -> VerifiedEvidence:
    return VerifiedEvidence(
        evidence_id=evidence_id,
        case_id="case-1",
        entity=entity,
        signal="cpu",
        kind="metric",
        time_range={"start": 100, "end": 200},
        claim={"metric_role": "source_metric", "evidence_program_weight": 2.0},
        provenance={"source": "metrics.csv", "row": 7},
        observed={"observed": 8.0, "baseline": 1.5},
        verification_rule="metric_window_lookup",
        verified=True,
    )


def _prediction() -> NursePrediction:
    return NursePrediction(
        case_id="case-1",
        ranked=["checkout", "frontend"],
        scores={"checkout": 3.0, "frontend": 1.0},
        score_breakdown={
            "checkout": {"telemetry": 1.0, "counterfactual_drop": 2.0},
            "frontend": {"symptom_penalty": -0.5},
        },
        evidence_ids={"checkout": ["ev-1"], "frontend": ["ev-2"]},
        closure_paths=[{"type": "propagates_to", "src": "checkout", "dst": "frontend"}],
        metadata={"ranker": "evidence_program_v2"},
    )


def test_context_packager_uses_prediction_without_mutating_rank():
    graph = EvidenceGraph(
        case_id="case-1",
        evidences=[_evidence("ev-1", "checkout"), _evidence("ev-2", "frontend")],
        relations=[{"type": "supports_root", "src": "checkout", "dst": "frontend"}],
        metadata={"dataset": "synthetic"},
    )
    prediction = _prediction()

    context = package_verified_rcl_context(
        graph=graph,
        prediction=prediction,
        dataset="synthetic",
        llm_advisory={"rank_repair": [{"promote": "frontend", "to_rank": 1}]},
        rejected_items=[{"evidence_id": "bad-1", "reason": "missing provenance"}],
    )

    assert context.ranked_candidates == prediction.ranked
    assert prediction.ranked == ["checkout", "frontend"]
    assert context.advisory["rank_repair"][0]["promote"] == "frontend"
    assert context.provenance["llm_rank_mutation"] is False
    assert context.provenance["rank_path"] == "evidence_core_v2"


def test_context_packager_keeps_only_verified_context_without_labels():
    graph = EvidenceGraph(
        case_id="case-1",
        evidences=[_evidence("ev-1", "checkout")],
        relations=[],
        metadata={"dataset": "synthetic"},
    )

    context = package_verified_rcl_context(
        graph=graph,
        prediction=_prediction(),
        dataset="synthetic",
    )

    payload = context.to_dict()
    assert payload["evidence_summaries"][0]["evidence_id"] == "ev-1"
    assert payload["evidence_summaries"][0]["entity"] == "checkout"
    assert payload["score_summary"]["checkout"] == 3.0
    assert "root_cause" not in str(payload).lower()
    assert "answer_key" not in str(payload).lower()
