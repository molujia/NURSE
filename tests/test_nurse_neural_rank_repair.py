from nurse.neural_rank_repair import (
    apply_primary_service_slot_rescue,
    apply_symptom_family_top3_rescue,
    apply_verified_rank_repair,
)


def test_rank_repair_rejects_unverified_candidate_lift():
    result = apply_verified_rank_repair(
        original_ranking=["checkout", "payment", "cart", "ad"],
        proposals=[
            {
                "promote": "ad",
                "to_rank": 1,
                "supporting_evidence_ids": ["missing-evidence"],
                "confidence": 0.95,
            }
        ],
        verified_evidence_ids={"ev-ad"},
    )

    assert result.ranking == ["checkout", "payment", "cart", "ad"]
    assert result.verified_repairs == []
    assert result.rejected_repairs[0]["reason"] == "unverified_evidence"


def test_rank_repair_promotes_verified_candidate_within_candidate_set():
    result = apply_verified_rank_repair(
        original_ranking=["checkout", "payment", "cart", "ad"],
        proposals=[
            {
                "promote": "ad",
                "to_rank": 1,
                "supporting_evidence_ids": ["ev-ad"],
                "confidence": 0.9,
            }
        ],
        verified_evidence_ids={"ev-ad"},
    )

    assert result.ranking == ["ad", "checkout", "payment", "cart"]
    assert result.verified_repairs[0]["promote"] == "ad"


def test_rank_repair_rejects_services_outside_original_top_k():
    result = apply_verified_rank_repair(
        original_ranking=["checkout", "payment", "cart", "ad"],
        proposals=[
            {
                "promote": "inventory",
                "to_rank": 1,
                "supporting_evidence_ids": ["ev-inventory"],
                "confidence": 1.0,
            }
        ],
        verified_evidence_ids={"ev-inventory"},
        repair_top_k=4,
    )

    assert result.ranking == ["checkout", "payment", "cart", "ad"]
    assert result.rejected_repairs[0]["reason"] == "outside_repair_candidate_set"


def test_rank_repair_rejects_blocked_infra_prefixes():
    result = apply_verified_rank_repair(
        original_ranking=["frontend", "k8s-master2", "ad"],
        proposals=[{
            "promote": "k8s-master2",
            "to_rank": 1,
            "supporting_evidence_ids": ["ev-k8s"],
            "confidence": 0.9,
        }],
        verified_evidence_ids=["ev-k8s"],
        blocked_prefixes=["k8s-master", "aiops-k8s-"],
    )

    assert result.ranking == ["frontend", "k8s-master2", "ad"]
    assert result.verified_repairs == []
    assert result.rejected_repairs[0]["reason"] == "blocked_service"

def test_rank_repair_rejects_blocked_services():
    result = apply_verified_rank_repair(
        original_ranking=["currency", "payment", "ad"],
        proposals=[{
            "promote": "payment",
            "to_rank": 1,
            "supporting_evidence_ids": ["ev-payment"],
            "confidence": 0.9,
        }],
        verified_evidence_ids=["ev-payment"],
        blocked_services=["payment", "currency", "frontend", "shipping"],
    )

    assert result.ranking == ["currency", "payment", "ad"]
    assert result.verified_repairs == []
    assert result.rejected_repairs[0]["reason"] == "blocked_service"


def test_rank_repair_can_be_limited_to_original_top_two():
    result = apply_verified_rank_repair(
        original_ranking=["frontend", "checkout", "redis-cart"],
        proposals=[{
            "promote": "redis-cart",
            "to_rank": 1,
            "supporting_evidence_ids": ["ev-redis"],
            "confidence": 0.9,
        }],
        verified_evidence_ids=["ev-redis"],
        repair_top_k=2,
    )

    assert result.ranking == ["frontend", "checkout", "redis-cart"]
    assert result.verified_repairs == []
    assert result.rejected_repairs[0]["reason"] == "outside_repair_candidate_set"

def test_rank_repair_rejects_large_symbolic_score_gap():
    result = apply_verified_rank_repair(
        original_ranking=["travel-service", "auth-service"],
        proposals=[{
            "promote": "auth-service",
            "to_rank": 1,
            "supporting_evidence_ids": ["ev-auth"],
            "confidence": 0.9,
        }],
        verified_evidence_ids=["ev-auth"],
        scores={"travel-service": 6.2, "auth-service": 5.1},
        max_score_gap_from_top=0.3,
    )

    assert result.ranking == ["travel-service", "auth-service"]
    assert result.verified_repairs == []
    assert result.rejected_repairs[0]["reason"] == "score_gap_too_large"


def test_rank_repair_allows_close_symbolic_score_gap():
    result = apply_verified_rank_repair(
        original_ranking=["food-map-service", "auth-service"],
        proposals=[{
            "promote": "auth-service",
            "to_rank": 1,
            "supporting_evidence_ids": ["ev-auth"],
            "confidence": 0.9,
        }],
        verified_evidence_ids=["ev-auth"],
        scores={"food-map-service": 6.08, "auth-service": 5.95},
        max_score_gap_from_top=0.3,
    )

    assert result.ranking == ["auth-service", "food-map-service"]
    assert len(result.verified_repairs) == 1

def test_rank_repair_rejects_weak_source_symptom_candidate_over_stronger_top1():
    result = apply_verified_rank_repair(
        original_ranking=["frontend", "email", "payment"],
        proposals=[{
            "promote": "email",
            "to_rank": 1,
            "supporting_evidence_ids": ["ev-email"],
            "confidence": 0.9,
        }],
        verified_evidence_ids=["ev-email"],
        scores={"frontend": 11.74, "email": 11.63, "payment": 10.21},
        score_breakdown={
            "frontend": {"symptom_penalty": -1.65, "generic_evidence": 0.3},
            "email": {"source_evidence": 0.7, "symptom_penalty": -1.65},
            "payment": {"symptom_penalty": -2.2},
        },
        min_source_evidence_for_top1_repair=1.0,
        reject_symptom_penalty_at_or_below=-1.0,
    )

    assert result.ranking == ["frontend", "email", "payment"]
    assert result.verified_repairs == []
    assert result.rejected_repairs[0]["reason"] == "weak_source_symptom_candidate"


def test_rank_repair_allows_strong_source_candidate_over_symptom_prone_top1():
    result = apply_verified_rank_repair(
        original_ranking=["frontend", "ad", "pd"],
        proposals=[{
            "promote": "ad",
            "to_rank": 1,
            "supporting_evidence_ids": ["ev-ad"],
            "confidence": 0.9,
        }],
        verified_evidence_ids=["ev-ad"],
        scores={"frontend": 15.54, "ad": 13.75, "pd": 6.8},
        score_breakdown={
            "frontend": {"source_evidence": 0.7, "symptom_penalty": -1.1},
            "ad": {"source_evidence": 2.1, "symptom_penalty": -0.55},
        },
        min_source_evidence_for_top1_repair=1.0,
        reject_symptom_penalty_at_or_below=-1.0,
    )

    assert result.ranking == ["ad", "frontend", "pd"]
    assert len(result.verified_repairs) == 1


def test_rank_repair_bypasses_large_score_gap_for_primary_service_into_non_top1_rank():
    result = apply_verified_rank_repair(
        original_ranking=["payment-service", "preserve-service", "user-service", "auth-service"],
        proposals=[{
            "promote": "auth-service",
            "to_rank": 4,
            "supporting_evidence_ids": ["ev-auth"],
            "confidence": 0.8,
        }],
        verified_evidence_ids=["ev-auth"],
        scores={"payment-service": 6.9, "auth-service": 3.0},
        max_score_gap_from_top=0.3,
        score_gap_bypass_services=["auth-service", "order-service", "route-service", "train-service", "travel-service"],
        score_gap_bypass_protected_top1_services=["auth-service", "order-service", "route-service", "train-service", "travel-service"],
        score_gap_bypass_min_target_rank=2,
    )

    assert result.verified_repairs[0]["promote"] == "auth-service"
    assert result.rejected_repairs == []


def test_rank_repair_does_not_bypass_large_gap_when_top1_is_primary_service():
    result = apply_verified_rank_repair(
        original_ranking=["travel-service", "auth-service", "admin-travel-service"],
        proposals=[{
            "promote": "auth-service",
            "to_rank": 2,
            "supporting_evidence_ids": ["ev-auth"],
            "confidence": 0.8,
        }],
        verified_evidence_ids=["ev-auth"],
        scores={"travel-service": 6.9, "auth-service": 3.0},
        max_score_gap_from_top=0.3,
        score_gap_bypass_services=["auth-service", "order-service", "route-service", "train-service", "travel-service"],
        score_gap_bypass_protected_top1_services=["auth-service", "order-service", "route-service", "train-service", "travel-service"],
        score_gap_bypass_min_target_rank=2,
    )

    assert result.verified_repairs == []
    assert result.rejected_repairs[0]["reason"] == "score_gap_too_large"

def test_primary_service_slot_rescue_moves_primary_candidate_to_rank_five():
    ranking = [
        "security-service",
        "basic-service",
        "consign-price-service",
        "consign-service",
        "ui-dashboard",
        "inside-payment-service",
        "admin-travel-service",
        "route-service",
    ]

    result = apply_primary_service_slot_rescue(
        ranking,
        primary_services={"auth-service", "order-service", "route-service", "train-service", "travel-service"},
        top_k=5,
        search_k=10,
    )

    assert result.applied is True
    assert result.promoted_service == "route-service"
    assert result.ranking[:5] == [
        "security-service",
        "basic-service",
        "consign-price-service",
        "consign-service",
        "route-service",
    ]


def test_primary_service_slot_rescue_noops_when_primary_already_in_top_five():
    ranking = ["travel-service", "admin-travel-service", "auth-service", "food-service", "route-service"]

    result = apply_primary_service_slot_rescue(
        ranking,
        primary_services={"auth-service", "order-service", "route-service", "train-service", "travel-service"},
        top_k=5,
        search_k=10,
    )

    assert result.applied is False
    assert result.ranking == ranking


def test_primary_service_slot_rescue_noops_without_primary_in_search_window():
    ranking = ["payment-service", "preserve-service", "user-service", "price-mongo", "config-mongo", "travel2-mongo"]

    result = apply_primary_service_slot_rescue(
        ranking,
        primary_services={"auth-service", "order-service", "route-service", "train-service", "travel-service"},
        top_k=5,
        search_k=10,
    )

    assert result.applied is False
    assert result.ranking == ranking

def test_symptom_family_top3_rescue_promotes_first_non_symptom_application_candidate():
    result = apply_symptom_family_top3_rescue(
        ["redis-cart", "frontend", "ad", "pd"],
        symptom_family={"frontend", "currency", "payment", "shipping", "redis-cart"},
        blocked_prefixes=("k8s-master", "aiops-k8s-"),
        top_k=3,
    )

    assert result.applied is True
    assert result.promoted_service == "ad"
    assert result.ranking[:4] == ["ad", "redis-cart", "frontend", "pd"]


def test_symptom_family_top3_rescue_noops_when_top1_is_not_symptom_family():
    result = apply_symptom_family_top3_rescue(
        ["ad", "frontend", "payment"],
        symptom_family={"frontend", "currency", "payment", "shipping", "redis-cart"},
        blocked_prefixes=("k8s-master", "aiops-k8s-"),
        top_k=3,
    )

    assert result.applied is False
    assert result.ranking == ["ad", "frontend", "payment"]


def test_symptom_family_top3_rescue_does_not_search_beyond_top_three():
    result = apply_symptom_family_top3_rescue(
        ["frontend", "payment", "k8s-master2", "ad"],
        symptom_family={"frontend", "currency", "payment", "shipping", "redis-cart"},
        blocked_prefixes=("k8s-master", "aiops-k8s-"),
        top_k=3,
    )

    assert result.applied is False
    assert result.ranking == ["frontend", "payment", "k8s-master2", "ad"]
