from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

from nurse import raw_adapters
from nurse.alarm_service import detect_alarm
from nurse.raw_adapters import RawIncident, load_raw_incident
from nurse.evidence_program import EvidenceFact
from nurse.raw_evidence_compiler import service_candidates_from_incident
from nurse.schemas import AlarmArtifact


def test_aiops25_loader_uses_window_width_seconds_for_raw_window(tmp_path, monkeypatch):
    case_dir = tmp_path / "case-1"
    case_dir.mkdir()
    raw_root = tmp_path / "raw-root"
    raw_root.mkdir()
    start_time = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)
    (case_dir / "case.json").write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "source_layout": "AIOps25",
                "raw_paths": {"root": str(raw_root)},
                "start_time": start_time.isoformat().replace("+00:00", "Z"),
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    def fake_metric_series(root, start, end):
        captured["metric_window"] = (start, end)
        return {}, [], {"status": "stub"}

    def fake_files_for_window(root, start, end, subdir):
        captured.setdefault("file_windows", []).append((subdir, start, end))
        return []

    monkeypatch.setattr(raw_adapters, "_load_aiops25_metric_series", fake_metric_series)
    monkeypatch.setattr(raw_adapters, "_aiops25_files_for_window", fake_files_for_window)
    monkeypatch.setattr(raw_adapters, "date_dirs_for_window", lambda root, start, end: [tmp_path])

    incident = load_raw_incident(case_dir, dataset="aiops25-retained-nonnode", window_width_seconds=120)

    expected_start = start_time - timedelta(seconds=120)
    expected_end = start_time + timedelta(seconds=120)
    assert captured["metric_window"] == (expected_start, expected_end)
    assert captured["file_windows"] == [
        ("log-parquet", expected_start, expected_end),
        ("trace-parquet", expected_start, expected_end),
    ]
    assert incident.metadata["window_width_seconds"] == 120
    assert incident.metadata["fault_window_seconds"] == 120
    assert incident.metadata["normal_window_seconds"] == 120


def test_benchmark_alarm_uses_custom_window_width_seconds():
    incident = RawIncident(
        dataset="aiops25-retained-nonnode",
        case_id="case-1",
        case_dir=None,
        available_modalities=["metrics"],
        file_index={},
        metric_series={"checkout.cpu": [(1000, 1.0), (1010, 5.0), (1020, 1.5)]},
        topology_edges=[],
        metadata={
            "benchmark_trigger_time": 1000,
            "alert_entity": {"entity_name": "checkout"},
        },
    )

    alarm = detect_alarm(incident, window_width_seconds=120)

    assert alarm is not None
    assert alarm.alert_entity == "checkout"
    assert alarm.window["normal_start"] == 880
    assert alarm.window["end"] == 1120
    assert alarm.window["normal_window_seconds"] == 120
    assert alarm.window["fault_window_seconds"] == 120
    assert alarm.window["window_width_seconds"] == 120
    assert alarm.provenance["window_width_seconds"] == 120


def test_candidate_expansion_rounds_bounded_and_deterministic(monkeypatch):
    incident = RawIncident(
        dataset="synthetic",
        case_id="case-1",
        case_dir=None,
        available_modalities=["metrics"],
        file_index={},
        metric_series={},
        topology_edges=[
            ("svc-a", "svc-b"),
            ("svc-b", "svc-c"),
            ("svc-c", "svc-d"),
        ],
        metadata={"alert_entity": {"entity_name": "svc-a"}},
    )
    alarm = AlarmArtifact(
        case_id="case-1",
        dataset="synthetic",
        trigger_time=1,
        window={"start": 0, "end": 2},
        alert_entity="svc-a",
        signals=[],
        provenance={"source": "unit-test"},
    )

    def fake_metric_facts(*args, **kwargs):
        return [
            EvidenceFact("anomaly", "svc-a", "cpu", 10.0, {"robust_z": 9.0}),
            EvidenceFact("anomaly", "svc-a", "mem", 10.0, {"robust_z": 9.0}),
            EvidenceFact("anomaly", "svc-b", "cpu", 3.0, {"robust_z": 5.5}),
            EvidenceFact("anomaly", "svc-c", "cpu", 3.0, {"robust_z": 5.5}),
            EvidenceFact("anomaly", "svc-d", "cpu", 3.0, {"robust_z": 5.5}),
        ]

    monkeypatch.setattr("nurse.raw_evidence_compiler.compile_metric_facts", fake_metric_facts)

    round_0 = service_candidates_from_incident(incident, alarm, candidate_expansion_rounds=0)
    round_1 = service_candidates_from_incident(incident, alarm, candidate_expansion_rounds=1)
    round_2 = service_candidates_from_incident(incident, alarm, candidate_expansion_rounds=2)
    round_3 = service_candidates_from_incident(incident, alarm, candidate_expansion_rounds=3)

    assert round_0 == ["svc-a"]
    assert round_1 == ["svc-a", "svc-b"]
    assert round_2 == ["svc-a", "svc-b", "svc-c"]
    assert round_3 == ["svc-a", "svc-b", "svc-c", "svc-d"]


def test_candidate_expansion_rounds_relax_evidence_gate_without_topology(monkeypatch):
    incident = RawIncident(
        dataset="synthetic",
        case_id="case-2",
        case_dir=None,
        available_modalities=["metrics"],
        file_index={},
        metric_series={},
        topology_edges=[],
        metadata={},
    )
    alarm = AlarmArtifact(
        case_id="case-2",
        dataset="synthetic",
        trigger_time=1,
        window={"start": 0, "end": 2},
        alert_entity="",
        signals=[],
        provenance={"source": "unit-test"},
    )

    def fake_metric_facts(*args, **kwargs):
        return [
            EvidenceFact("anomaly", "svc-a", "cpu", 10.0, {"robust_z": 9.0}),
            EvidenceFact("anomaly", "svc-a", "mem", 10.0, {"robust_z": 9.0}),
            EvidenceFact("anomaly", "svc-b", "cpu", 8.0, {"robust_z": 6.0}),
            EvidenceFact("anomaly", "svc-b", "mem", 8.0, {"robust_z": 6.0}),
            EvidenceFact("anomaly", "svc-c", "cpu", 8.0, {"robust_z": 9.0}),
            EvidenceFact("anomaly", "svc-d", "cpu", 6.0, {"robust_z": 3.5}),
            EvidenceFact("anomaly", "svc-d", "mem", 6.0, {"robust_z": 3.5}),
            EvidenceFact("anomaly", "svc-e", "cpu", 6.0, {"robust_z": 5.5}),
            EvidenceFact("anomaly", "svc-f", "cpu", 6.0, {"robust_z": 3.5}),
        ]

    monkeypatch.setattr("nurse.raw_evidence_compiler.compile_metric_facts", fake_metric_facts)

    expected = [
        ["svc-a"],
        ["svc-a", "svc-b"],
        ["svc-a", "svc-b", "svc-c"],
        ["svc-a", "svc-b", "svc-c", "svc-d"],
        ["svc-a", "svc-b", "svc-c", "svc-d", "svc-e"],
        ["svc-a", "svc-b", "svc-c", "svc-d", "svc-e", "svc-f"],
    ]

    for mj, services in enumerate(expected):
        assert service_candidates_from_incident(incident, alarm, candidate_expansion_rounds=mj) == services
