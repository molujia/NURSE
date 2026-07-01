from nurse.history_prior import (
    build_corrected_history_record,
    build_history_record,
    compute_history_prior,
)


def _semantic_payload():
    return {
        "output_contract": "semantic_rank_only",
        "service_candidates": ["checkout", "frontend"],
        "current_ranking": ["checkout", "frontend"],
        "candidate_cards": [{"service": "checkout"}, {"service": "frontend"}],
    }


def test_corrected_history_record_promotes_ground_truth_labels_to_top():
    corrected = build_corrected_history_record(
        case_id="case-1",
        dataset="synthetic",
        semantic_payload=_semantic_payload(),
        score_summary={"frontend": 3.0, "checkout": 1.0},
        score_breakdown={
            "frontend": {"E": 3.0, "P": 0.2},
            "checkout": {"E": 1.0, "P": 0.1},
        },
        ranked_candidates=["frontend", "checkout"],
        labels=["checkout"],
    )

    assert corrected["ranked_candidates"][0] == "checkout"
    assert corrected["score_summary"]["checkout"] > corrected["score_summary"]["frontend"]
    assert corrected["history_correction"]["labels"] == ["checkout"]
    assert corrected["history_correction"]["corrected_top1"] == "checkout"


def test_history_prior_skips_same_case_and_other_datasets():
    payload = _semantic_payload()
    history_index = [
        build_history_record(
            case_id="case-1",
            dataset="dataset-a",
            semantic_payload=payload,
            score_summary={"checkout": 8.0},
            score_breakdown={"checkout": {"E": 8.0}},
            ranked_candidates=["checkout"],
        ),
        build_history_record(
            case_id="case-2",
            dataset="dataset-b",
            semantic_payload=payload,
            score_summary={"checkout": 7.0},
            score_breakdown={"checkout": {"E": 7.0}},
            ranked_candidates=["checkout"],
        ),
        build_history_record(
            case_id="case-3",
            dataset="dataset-a",
            semantic_payload=payload,
            score_summary={"frontend": 5.0},
            score_breakdown={"frontend": {"E": 5.0}},
            ranked_candidates=["frontend"],
        ),
    ]

    result = compute_history_prior(
        case_id="case-1",
        dataset="dataset-a",
        current_payload=payload,
        history_index=history_index,
        service_candidates=["checkout", "frontend"],
        top_k=3,
        history_weight=1.0,
    )

    assert result.active is True
    assert [item["case_id"] for item in result.matched_history] == ["case-3"]
    assert result.score_contribution == {"frontend": 5.0}
