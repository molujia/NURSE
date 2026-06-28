from nurse.semantic_rank_facts import normalize_and_guard_semantic_facts


def test_guarded_semantic_fact_requires_existing_candidate_and_verified_evidence():
    proposal = {
        "semantic_facts": [
            {
                "service": "checkout-service",
                "fact_type": "root_support",
                "supporting_evidence_ids": ["ev-cpu"],
                "confidence": 0.91,
            },
            {
                "service": "made-up-service",
                "fact_type": "root_support",
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
    assert rejected[0]["reason"] == "outside_candidate_set"
