from nurse.history_prior import build_history_record
from nurse.ranker import NurseRanker
from nurse.schemas import EvidenceGraph, VerifiedEvidence


def _evidence(evidence_id, entity, signal, metric_role, observed=7.0, baseline=1.0):
    return VerifiedEvidence(
        evidence_id=evidence_id,
        case_id="case-1",
        entity=entity,
        signal=signal,
        kind="metric",
        time_range={"start": 1, "end": 2},
        claim={"metric_role": metric_role, "anomaly_score": 9.0},
        provenance={"source": "unit-test"},
        observed={"observed": observed, "baseline": baseline, "matching_points": 8},
        verification_rule="unit-test",
        verified=True,
    )


def test_ranker_reports_ephr_components_and_downstream_penalty():
    graph = EvidenceGraph(
        case_id="case-1",
        evidences=[
            _evidence("ev-root", "checkout", "cpu", "source_metric"),
            _evidence("ev-symptom", "frontend", "latency", "symptom_metric"),
        ],
        relations=[
            {"type": "propagates_to", "src": "checkout", "dst": "frontend", "confidence": 1.0},
        ],
        metadata={"service_candidates": ["checkout", "frontend"]},
    )

    prediction = NurseRanker().rank(graph)

    assert prediction.ranked[0] == "checkout"
    assert prediction.score_breakdown["checkout"]["E"] > 0
    assert prediction.score_breakdown["checkout"]["P"] > 0
    assert prediction.score_breakdown["frontend"]["P"] > 0
    assert prediction.score_breakdown["frontend"]["symptom_penalty"] < 0


def test_ranker_applies_history_and_rule_terms():
    semantic_payload = {
        "output_contract": "semantic_rank_only",
        "service_candidates": ["checkout", "frontend"],
        "current_ranking": ["checkout", "frontend"],
        "candidate_cards": [{"service": "checkout"}, {"service": "frontend"}],
    }
    history_index = [
        build_history_record(
            case_id="past-1",
            dataset="synthetic",
            semantic_payload=semantic_payload,
            score_summary={"checkout": 4.0, "frontend": 1.0},
            score_breakdown={"checkout": {"E": 4.0}, "frontend": {"E": 1.0}},
            ranked_candidates=["checkout", "frontend"],
        )
    ]
    graph = EvidenceGraph(
        case_id="case-2",
        evidences=[_evidence("ev-checkout", "checkout", "cpu", "source_metric")],
        relations=[],
        metadata={
            "service_candidates": ["checkout", "frontend"],
            "semantic_rank_payload": semantic_payload,
            "approved_history_index": history_index,
            "sre_rule_scores": {"frontend": 2.0},
        },
    )

    prediction = NurseRanker(history_prior_weight=0.5, rule_score_weight=1.0).rank(graph)

    assert prediction.score_breakdown["checkout"]["H"] > 0
    assert prediction.score_breakdown["frontend"]["R"] > 0
    assert prediction.metadata["history_prior_enabled"] is True
    assert prediction.metadata["sre_rule_enabled"] is True


def test_aiops_retained_nonnode_projects_infra_signal_to_application_candidates():
    graph = EvidenceGraph(
        case_id="case-aiops",
        evidences=[
            _evidence("ev-node", "aiops-k8s-07", "node_cpu_usage_rate", "source_metric"),
        ],
        relations=[
            {"type": "propagates_to", "src": "aiops-k8s-07", "dst": "cart", "confidence": 1.0},
        ],
        metadata={
            "dataset": "aiops25-retained-nonnode",
            "service_candidates": ["cart", "aiops-k8s-07"],
        },
    )

    prediction = NurseRanker().rank(graph)

    assert prediction.ranked[0] == "cart"
    assert prediction.score_breakdown["cart"]["aiops_infra_projection"] > 0
    assert prediction.score_breakdown["cart"]["P"] > 0
    assert prediction.score_breakdown["aiops-k8s-07"]["aiops_infra_penalty"] < 0
    assert prediction.score_breakdown["aiops-k8s-07"]["P"] < 0
