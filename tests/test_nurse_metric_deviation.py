from nurse.metric_deviation import service_deviation_scores, service_metric_deviation_scores
from nurse.raw_adapters import RawIncident
from nurse.schemas import AlarmArtifact


def test_metric_deviation_uses_shared_canonical_fallback_when_no_pre_window():
    incident = RawIncident(
        dataset="synthetic",
        case_id="case-no-pre-window",
        case_dir=None,
        available_modalities=["metrics"],
        file_index={},
        metric_series={
            "alert.mem": [(10, 1.0), (11, 1001.0)],
            "root.cpu": [(10, 1.0), (11, 2.0)],
        },
        topology_edges=[],
        metadata={},
    )
    alarm = AlarmArtifact(
        case_id="case-no-pre-window",
        dataset="synthetic",
        trigger_time=10,
        window={"start": 0, "end": 11},
        alert_entity="alert",
        signals=[],
        provenance={"source": "test"},
    )

    scores = service_deviation_scores(incident, alarm)

    assert scores["alert"] < 2.0


def test_metric_deviation_exports_per_metric_scores_without_labels():
    incident = RawIncident(
        dataset="synthetic",
        case_id="case-per-metric",
        case_dir=None,
        available_modalities=["metrics"],
        file_index={},
        metric_series={
            "root.cpu": [(0, 1.0), (1, 1.0), (2, 8.0), (3, 9.0)],
            "root.request": [(0, 10.0), (1, 10.0), (2, 12.0), (3, 13.0)],
            "symptom.rrt": [(0, 5.0), (1, 5.0), (2, 15.0), (3, 16.0)],
        },
        topology_edges=[("root", "symptom")],
        metadata={},
    )
    alarm = AlarmArtifact(
        case_id="case-per-metric",
        dataset="synthetic",
        trigger_time=2,
        window={"start": 2, "end": 3},
        alert_entity="root",
        signals=[],
        provenance={"source": "test"},
    )

    scores = service_metric_deviation_scores(
        incident,
        alarm,
        service_candidates=["root", "symptom"],
    )

    serialized = str(scores).lower()
    assert "root_cause" not in serialized
    assert "answer_key" not in serialized
    assert scores["root"]["cpu"] > scores["root"].get("request", 0.0)
    assert scores["symptom"]["rrt"] > 0

def test_metric_deviation_uses_relative_floor_for_large_constant_gauges():
    incident = RawIncident(
        dataset="synthetic",
        case_id="case-relative-floor",
        case_dir=None,
        available_modalities=["metrics"],
        file_index={},
        metric_series={
            "false.memory": [(0, 1_000_000.0), (1, 1_000_000.0), (2, 1_001_000.0)],
            "root.cpu": [(0, 1.0), (1, 1.0), (2, 10.0)],
        },
        topology_edges=[],
        metadata={},
    )
    alarm = AlarmArtifact(
        case_id="case-relative-floor",
        dataset="synthetic",
        trigger_time=2,
        window={"start": 2, "end": 2},
        alert_entity="root",
        signals=[],
        provenance={"source": "test"},
    )

    scores = service_metric_deviation_scores(
        incident,
        alarm,
        service_candidates=["false", "root"],
    )

    assert scores["root"]["cpu"] > scores["false"]["memory"]
    assert scores["false"]["memory"] < 1.0

def test_metric_deviation_can_use_fixed_window_alarm_detector_backend():
    incident = RawIncident(
        dataset="synthetic",
        case_id="case-ad-profile",
        case_dir=None,
        available_modalities=["metrics"],
        file_index={},
        metric_series={
            "root.cpu": [(0, 1.0), (1, 1.2), (2, 0.9), (3, 1.1), (4, 6.0), (5, 5.5)],
            "noise.cpu": [(0, 1.0), (1, 1.1), (2, 0.9), (3, 1.0), (4, 1.2), (5, 0.8)],
        },
        topology_edges=[],
        metadata={},
    )
    alarm = AlarmArtifact(
        case_id="case-ad-profile",
        dataset="synthetic",
        trigger_time=4,
        window={"normal_start": 0, "normal_end": 4, "start": 4, "end": 5},
        alert_entity="root",
        signals=[],
        provenance={"source": "test"},
    )

    scores = service_metric_deviation_scores(
        incident,
        alarm,
        service_candidates=["root", "noise"],
        score_backend="alarm_fixed_window",
    )

    assert scores["root"]["cpu"] > scores["noise"].get("cpu", 0.0)
    assert scores["root"]["cpu"] > 1.0

