import pytest

from nurse.schemas import EvidenceGraph, VerifiedEvidence


def _verified_metric(evidence_id: str, entity: str, signal: str, role: str) -> VerifiedEvidence:
    return VerifiedEvidence(
        evidence_id=evidence_id,
        case_id="case-1",
        entity=entity,
        signal=signal,
        kind="metric",
        time_range={"start": 100, "end": 200},
        claim={"metric_role": role, "evidence_program_weight": 2.0},
        provenance={"source": "metrics.csv"},
        observed={"observed": 8.0, "baseline": 1.0},
        verification_rule="unit-test",
        verified=True,
    )


def test_component_named_evidence_core_wrappers_are_usable():
    from nurse.evidence_admission import admit_evidence_candidates
    from nurse.evidence_graph_builder import build_verified_evidence_graph
    from nurse.evidence_ranker import EvidenceRanker
    from nurse.incident_prior import compute_incident_prior

    evidence = [_verified_metric("ev-checkout", "checkout", "cpu", "source_metric")]
    graph = build_verified_evidence_graph(
        case_id="case-1",
        evidences=evidence,
        topology_edges=[],
        alert_entity="checkout",
    )
    prediction = EvidenceRanker().rank(graph)
    prior = compute_incident_prior(
        case_id="case-1",
        dataset="synthetic",
        service_candidates=["checkout"],
        history_index=None,
    )

    assert callable(admit_evidence_candidates)
    assert isinstance(graph, EvidenceGraph)
    assert prediction.ranked[0] == "checkout"
    assert prior.active is False
    assert prior.score_contribution == {}


def test_incident_prior_rejects_forbidden_history_payload():
    from nurse.incident_prior import compute_incident_prior

    with pytest.raises(ValueError, match="forbidden inference field"):
        compute_incident_prior(
            case_id="case-1",
            dataset="synthetic",
            service_candidates=["checkout"],
            history_index=[{"incident_id": "h1", "root_cause": "checkout"}],
        )
