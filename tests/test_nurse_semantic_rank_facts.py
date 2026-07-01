from nurse.ranker import NurseRanker
from nurse.schemas import EvidenceGraph, VerifiedEvidence
from nurse.semantic_rank_facts import normalize_and_guard_semantic_facts


def test_guarded_semantic_fact_requires_existing_candidate_and_verified_evidence():
    proposal = {
        "semantic_facts": [
            {
                "service": "checkout-service",
                "fact_type": "resource_metric_deviation",
                "supporting_evidence_ids": ["ev-cpu"],
                "confidence": 0.91,
            },
            {
                "service": "made-up-service",
                "fact_type": "resource_metric_deviation",
                "supporting_evidence_ids": ["ev-cpu"],
                "confidence": 0.99,
            },
        ]
    }

    accepted, rejected = normalize_and_guard_semantic_facts(
        proposal=proposal,
        service_candidates=["checkout-service", "frontend"],
        verified_evidence_ids={"ev-cpu"},
        score_breakdown={"checkout-service": {"source_evidence": 1.0}},
    )

    assert [fact.service for fact in accepted] == ["checkout-service"]
    assert [fact.fact_type for fact in accepted] == ["source_support"]
    assert rejected[0]["reason"] == "outside_candidate_set"


def test_guarded_semantic_fact_rejects_unverified_low_confidence_and_unsupported_types():
    proposal = {
        "semantic_facts": [
            {
                "service": "checkout-service",
                "fact_type": "source_support",
                "supporting_evidence_ids": ["ev-cpu"],
                "confidence": 0.91,
            },
            {
                "service": "frontend",
                "fact_type": "source_support",
                "supporting_evidence_ids": ["ev-missing"],
                "confidence": 0.91,
            },
            {
                "service": "frontend",
                "fact_type": "source_support",
                "supporting_evidence_ids": ["ev-cpu"],
                "confidence": 0.2,
            },
            {
                "service": "frontend",
                "fact_type": "invented_fact",
                "supporting_evidence_ids": ["ev-cpu"],
                "confidence": 0.91,
            },
        ]
    }

    accepted, rejected = normalize_and_guard_semantic_facts(
        proposal=proposal,
        service_candidates=["checkout-service", "frontend"],
        verified_evidence_ids={"ev-cpu"},
        score_breakdown={"checkout-service": {"source_evidence": 1.0}},
    )

    assert [fact.service for fact in accepted] == ["checkout-service"]
    assert [item["reason"] for item in rejected] == [
        "unverified_evidence",
        "low_confidence",
        "unsupported_fact_type",
    ]


def test_guarded_semantic_fact_caps_accepted_facts_at_three():
    proposal = {
        "semantic_facts": [
            {
                "service": "checkout-service",
                "fact_type": "source_support",
                "supporting_evidence_ids": ["ev-1"],
                "confidence": 0.91,
            },
            {
                "service": "frontend",
                "fact_type": "source_support",
                "supporting_evidence_ids": ["ev-2"],
                "confidence": 0.91,
            },
            {
                "service": "ad",
                "fact_type": "source_support",
                "supporting_evidence_ids": ["ev-3"],
                "confidence": 0.91,
            },
            {
                "service": "payment",
                "fact_type": "source_support",
                "supporting_evidence_ids": ["ev-4"],
                "confidence": 0.91,
            },
        ]
    }

    accepted, rejected = normalize_and_guard_semantic_facts(
        proposal=proposal,
        service_candidates=["checkout-service", "frontend", "ad", "payment"],
        verified_evidence_ids={"ev-1", "ev-2", "ev-3", "ev-4"},
        score_breakdown={
            "checkout-service": {"source_evidence": 1.0},
            "frontend": {"source_evidence": 1.0},
            "ad": {"source_evidence": 1.0},
            "payment": {"source_evidence": 1.0},
        },
    )

    assert [fact.service for fact in accepted] == ["checkout-service", "frontend", "ad"]
    assert rejected[-1]["reason"] == "too_many_semantic_facts"


def test_guarded_semantic_fact_participates_in_symbolic_ranking():
    graph = EvidenceGraph(
        case_id="case-1",
        evidences=[
            VerifiedEvidence(
                evidence_id="ev-checkout",
                case_id="case-1",
                entity="checkout",
                signal="cpu",
                kind="metric",
                time_range={"start": 1, "end": 2},
                claim={"metric_role": "source_metric", "evidence_program_weight": 1.0},
                provenance={"source": "metrics.csv"},
                observed={"observed": 7.0, "baseline": 1.0},
                verification_rule="metric_window_lookup",
                verified=True,
            ),
            VerifiedEvidence(
                evidence_id="ev-frontend",
                case_id="case-1",
                entity="frontend",
                signal="cpu",
                kind="metric",
                time_range={"start": 1, "end": 2},
                claim={"metric_role": "source_metric", "evidence_program_weight": 1.0},
                provenance={"source": "metrics.csv"},
                observed={"observed": 7.0, "baseline": 1.0},
                verification_rule="metric_window_lookup",
                verified=True,
            ),
        ],
        relations=[],
        metadata={
            "service_candidates": ["checkout", "frontend"],
            "llm_proposal": {
                "semantic_facts": [
                    {
                        "service": "checkout",
                        "fact_type": "root_support",
                        "supporting_evidence_ids": ["ev-checkout"],
                        "confidence": 0.9,
                    }
                ]
            },
        },
    )

    prediction = NurseRanker(allow_llm_scoring=True, symbolic_score_enabled=True).rank(graph)

    assert prediction.ranked[0] == "checkout"
    assert prediction.metadata["semantic_rank_facts_used"] == 1
    assert prediction.metadata["semantic_rank_facts_rejected"] == 0
    assert prediction.score_breakdown["checkout"]["semantic_fact_evidence"] > 0


def test_aiops_discourages_symptom_prone_entrypoint_semantic_facts():
    proposal = {
        "semantic_facts": [
            {
                "service": "frontend",
                "fact_type": "resource_metric_deviation",
                "supporting_evidence_ids": ["ev-cpu"],
                "confidence": 0.91,
            }
        ]
    }

    accepted, rejected = normalize_and_guard_semantic_facts(
        proposal=proposal,
        service_candidates=["frontend", "ad"],
        verified_evidence_ids={"ev-cpu"},
        score_breakdown={},
        dataset="aiops25-retained-nonnode",
    )

    assert accepted == []
    assert rejected[0]["reason"] == "discouraged_target_role"


def test_rcaeval_discourages_support_storage_semantic_facts():
    proposal = {
        "semantic_facts": [
            {
                "service": "basic-service",
                "fact_type": "resource_metric_deviation",
                "supporting_evidence_ids": ["ev-cpu"],
                "confidence": 0.91,
            }
        ]
    }

    accepted, rejected = normalize_and_guard_semantic_facts(
        proposal=proposal,
        service_candidates=["auth-service", "basic-service"],
        verified_evidence_ids={"ev-cpu"},
        score_breakdown={},
        dataset="rcaeval-tt",
    )

    assert accepted == []
    assert rejected[0]["reason"] == "discouraged_target_role"


def test_rcaeval_primary_services_get_stronger_role_prior_than_support_services():
    primary = NurseRanker._service_role_prior("rcaeval-tt", "auth-service")
    support = NurseRanker._service_role_prior("rcaeval-tt", "basic-service")

    assert primary > support
    assert primary >= 0.7
