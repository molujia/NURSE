from nurse.alarm_detector import (
    RobustAlarmDetector,
    detect_alarm_from_series,
    fixed_window_metric_signals,
)
from nurse.raw_adapters import RawIncident
from nurse.runner import _first_alarm


def test_mad_detector_triggers_on_spike():
    series = [(i, 10.0) for i in range(20)] + [(20, 55.0), (21, 60.0)]

    alarm = detect_alarm_from_series(
        case_id="case-1",
        dataset="synthetic",
        entity="paymentservice",
        metric="error_rate",
        series=series,
        detector=RobustAlarmDetector(baseline_points=20, threshold=3.0),
    )

    assert alarm is not None
    assert alarm.trigger_time == 21
    assert alarm.alert_entity == "paymentservice"
    assert alarm.signals[0]["metric"] == "error_rate"


def test_mad_detector_ignores_stable_series():
    series = [(i, 10.0) for i in range(30)]

    alarm = detect_alarm_from_series(
        case_id="case-1",
        dataset="synthetic",
        entity="paymentservice",
        metric="cpu",
        series=series,
        detector=RobustAlarmDetector(baseline_points=20, threshold=3.0),
    )

    assert alarm is None


def test_detector_uses_peak_anomaly_not_first_threshold_crossing():
    series = [(i, 10.0) for i in range(20)]
    series.extend([(20, 14.0), (21, 10.0), (22, 10.0), (23, 60.0)])

    alarm = detect_alarm_from_series(
        case_id="case-1",
        dataset="synthetic",
        entity="paymentservice",
        metric="cpu",
        series=series,
        detector=RobustAlarmDetector(baseline_points=20, threshold=3.0),
    )

    assert alarm is not None
    assert alarm.trigger_time == 23


def test_detector_ignores_non_finite_metric_values():
    series = [(i, 10.0) for i in range(19)]
    series.append((19, float("nan")))
    series.extend([(20, 10.0), (21, float("inf")), (22, 80.0)])

    alarm = detect_alarm_from_series(
        case_id="case-1",
        dataset="synthetic",
        entity="paymentservice",
        metric="cpu",
        series=series,
        detector=RobustAlarmDetector(baseline_points=20, threshold=3.0),
    )

    assert alarm is not None
    assert alarm.trigger_time == 22


def test_alarm_window_radius_scales_for_epoch_microsecond_timestamps():
    trigger = 1_777_092_372_000_000
    series = [(trigger - (20 - i) * 1_000_000, 1.0) for i in range(20)]
    series.append((trigger, 10.0))

    alarm = detect_alarm_from_series(
        case_id="case-1",
        dataset="synthetic",
        entity="paymentservice",
        metric="cpu",
        series=series,
        detector=RobustAlarmDetector(baseline_points=20, threshold=3.0, window_radius=60),
    )

    assert alarm is not None
    assert alarm.window["start"] == trigger - 60_000_000
    assert alarm.window["end"] == trigger + 60_000_000


def test_incident_alarm_selection_uses_strongest_detected_signal():
    incident = RawIncident(
        dataset="synthetic",
        case_id="case-strongest",
        case_dir=None,
        available_modalities=["metrics"],
        file_index={},
        metric_series={
            "aservice.cpu": [(i, 1.0) for i in range(20)] + [(20, 4.1)],
            "zservice.cpu": [(i, 1.0) for i in range(20)] + [(20, 20.0)],
        },
        topology_edges=[],
        metadata={},
    )

    alarm = _first_alarm(incident)

    assert alarm is not None
    assert alarm.alert_entity == "zservice"


def test_fixed_window_metric_signals_breaks_ties_deterministically():
    series = [
        (0, 1.0),
        (1, 2.0),
        (2, 1.0),
        (3, 2.0),
        (4, 1.0),
        (5, 2.0),
        (6, 1.0),
        (7, 2.0),
        (8, 1.0),
        (9, 2.0),
        (10, 50.0),
        (11, 1.0),
        (12, 60.0),
    ]
    metric_series = {
        "bservice.cpu": series,
        "aservice.cpu": list(series),
    }

    signals = fixed_window_metric_signals(
        metric_series,
        start=10,
        end=12,
        normal_start=0,
        normal_end=10,
    )

    assert [(signal["entity"], signal["metric"]) for signal in signals[:2]] == [
        ("aservice", "cpu"),
        ("bservice", "cpu"),
    ]
