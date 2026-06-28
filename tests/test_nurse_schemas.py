import pytest

from nurse.schemas import (
    AlarmArtifact,
    EvidenceCandidate,
    EvidenceGraph,
    NursePrediction,
    VerifiedEvidence,
)


def test_alarm_artifact_round_trip_rejects_label_fields():
    alarm = AlarmArtifact(
        case_id="case-1",
        dataset="synthetic",
        trigger_time=120,
        window={"start": 60, "end": 180},
        alert_entity="checkoutservice",
        signals=[{"entity": "checkoutservice", "metric": "error_rate", "score": 9.0}],
        provenance={"source": "unit-test"},
    )

    restored = AlarmArtifact.from_dict(alarm.to_dict())

    assert restored.case_id == "case-1"
    assert restored.window["start"] == 60
    assert restored.alert_entity == "checkoutservice"
    assert restored.signals[0]["metric"] == "error_rate"

    payload = alarm.to_dict()
    payload["root_cause"] = "paymentservice"
    with pytest.raises(ValueError, match="forbidden"):
        AlarmArtifact.from_dict(payload)


def test_evidence_and_prediction_round_trip():
    candidate = EvidenceCandidate(
        evidence_id="cand-1",
        case_id="case-1",
        entity="paymentservice",
        signal="error_rate",
        kind="metric",
        time_range={"start": 100, "end": 130},
        claim={"operator": ">", "value": 0.2},
        provenance={"source": "metrics.json"},
    )
    verified = VerifiedEvidence.from_candidate(
        candidate,
        observed={"value": 0.5, "baseline": 0.0},
        verification_rule="metric_threshold",
    )
    graph = EvidenceGraph(case_id="case-1", evidences=[verified], relations=[])
    prediction = NursePrediction(
        case_id="case-1",
        ranked=["paymentservice", "checkoutservice"],
        scores={"paymentservice": 1.2, "checkoutservice": 0.4},
        score_breakdown={"paymentservice": {"source": 1.0}},
        evidence_ids={"paymentservice": ["cand-1"]},
        closure_paths=[],
    )

    assert VerifiedEvidence.from_dict(verified.to_dict()).verified is True
    assert EvidenceGraph.from_dict(graph.to_dict()).evidences[0].entity == "paymentservice"
    assert NursePrediction.from_dict(prediction.to_dict()).ranked[0] == "paymentservice"
