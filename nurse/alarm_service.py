"""Single alarm entrypoint used by NURSE."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional

from .alarm_detector import (
    FixedWindowScoreConfig,
    RobustAlarmDetector,
    detect_alarm_from_series,
    fixed_window_metric_signals,
)
from .raw_adapters import RawIncident
from .schemas import AlarmArtifact


REPO_ROOT = Path(__file__).resolve().parents[1]
ALARM_PROTOCOL_VERSION = "benchmark_trigger_fixed_window_v1"
FAULT_WINDOW_SECONDS = 480
NORMAL_WINDOW_SECONDS = 480
DEFAULT_DETECTOR_CONFIG = {
    "protocol": ALARM_PROTOCOL_VERSION,
    "fault_window_seconds": FAULT_WINDOW_SECONDS,
    "normal_window_seconds": NORMAL_WINDOW_SECONDS,
    "window_width_seconds": FAULT_WINDOW_SECONDS,
    "baseline_points": 20,
    "threshold": 3.0,
    "window_radius": 60,
}


def alarm_cache_path(
    incident: RawIncident,
    repo_root: str | Path = REPO_ROOT,
    *,
    window_width_seconds: int = FAULT_WINDOW_SECONDS,
) -> Path:
    width = max(int(window_width_seconds), 1)
    base = Path(repo_root) / "outputs" / "nurse" / "alarms" / incident.dataset
    if width == FAULT_WINDOW_SECONDS:
        return base / f"{incident.case_id}.json"
    return base / f"w{width}" / f"{incident.case_id}.json"


def _is_current_alarm(alarm: AlarmArtifact) -> bool:
    return alarm.detector == ALARM_PROTOCOL_VERSION and alarm.provenance.get("alarm_protocol") == ALARM_PROTOCOL_VERSION


def _load_cached_alarm(path: Path) -> Optional[AlarmArtifact]:
    if not path.exists():
        return None
    alarm = AlarmArtifact.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return alarm if _is_current_alarm(alarm) else None


def _parse_time_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if len(text) >= 5 and (text[-5] in {"+", "-"}) and text[-3] != ":":
        text = f"{text[:-2]}:{text[-2:]}"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if len(text) >= 5 and (text[-5] in {"+", "-"}) and text[-3] != ":":
        text = f"{text[:-2]}:{text[-2:]}"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _add_seconds(value: Any, seconds: int | float) -> Any:
    dt = _parse_datetime(value)
    if dt is not None and not isinstance(value, (int, float)):
        return (dt + timedelta(seconds=float(seconds))).isoformat()
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    magnitude = abs(numeric)
    if magnitude > 1e14:
        return numeric + float(seconds) * 1_000_000
    if magnitude > 1e11:
        return numeric + float(seconds) * 1_000
    result = numeric + float(seconds)
    return int(result) if float(result).is_integer() else result


def _metric_time_scale(incident: RawIncident) -> float:
    for series in incident.metric_series.values():
        for timestamp, _ in series:
            try:
                magnitude = abs(float(timestamp))
            except (TypeError, ValueError):
                continue
            if magnitude > 1e14:
                return 1_000_000.0
            if magnitude > 1e11:
                return 1_000.0
            return 1.0
    return 1.0


def _to_metric_time(value: Any, scale: float) -> float | None:
    parsed = _parse_time_value(value)
    if parsed is None:
        return None
    if abs(parsed) < 10_000_000_000 and scale > 1:
        return parsed * scale
    return parsed


def _read_rcaeval_inject_time(incident: RawIncident) -> int | float | None:
    if not incident.case_dir:
        return None
    path = Path(incident.case_dir) / "inject_time.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    value = float(text)
    return int(value) if value.is_integer() else value


def _alert_entity_from_metadata(incident: RawIncident) -> str:
    metadata_alarm = incident.metadata.get("alert_entity") or {}
    if isinstance(metadata_alarm, dict):
        return str(metadata_alarm.get("entity_name") or "")
    return str(metadata_alarm or "")


def _alarm_score(item: AlarmArtifact) -> float:
    values: List[float] = []
    for signal in item.signals:
        score = float(signal.get("score") or 0.0)
        observed = float(signal.get("value") or 0.0)
        baseline = float(signal.get("baseline") or 0.0)
        tie_break = abs(observed - baseline) * 1e-3
        if math.isinf(score):
            score = 1e9
        values.append(score + tie_break)
    return max(values or [0.0])


def _benchmark_trigger(incident: RawIncident) -> tuple[Any, str] | None:
    if incident.dataset.startswith("aiops25"):
        trigger = incident.metadata.get("benchmark_trigger_time") or incident.metadata.get("start_time")
        if trigger:
            return trigger, "benchmark_start_time"
    if incident.dataset == "rcaeval-tt":
        trigger = _read_rcaeval_inject_time(incident)
        if trigger is not None:
            return trigger, "benchmark_inject_time"
    return None


def _benchmark_trigger_alarm(incident: RawIncident, *, window_width_seconds: int = FAULT_WINDOW_SECONDS) -> Optional[AlarmArtifact]:
    trigger = _benchmark_trigger(incident)
    if trigger is None:
        return None
    trigger_time, source = trigger
    width_seconds = max(int(window_width_seconds), 1)
    window = {
        "normal_start": _add_seconds(trigger_time, -width_seconds),
        "normal_end": trigger_time,
        "start": trigger_time,
        "end": _add_seconds(trigger_time, width_seconds),
        "normal_window_seconds": width_seconds,
        "fault_window_seconds": width_seconds,
        "window_width_seconds": width_seconds,
    }

    scale = _metric_time_scale(incident)
    metric_start = _to_metric_time(window["start"], scale)
    metric_end = _to_metric_time(window["end"], scale)
    metric_normal_start = _to_metric_time(window["normal_start"], scale)
    metric_normal_end = _to_metric_time(window["normal_end"], scale)
    signals = []
    if None not in {metric_start, metric_end, metric_normal_start, metric_normal_end}:
        signals = fixed_window_metric_signals(
            incident.metric_series,
            start=float(metric_start),
            end=float(metric_end),
            normal_start=float(metric_normal_start),
            normal_end=float(metric_normal_end),
            config=FixedWindowScoreConfig(
                fault_window_seconds=width_seconds,
                normal_window_seconds=width_seconds,
            ),
        )

    alert_entity = _alert_entity_from_metadata(incident)
    if not alert_entity and signals:
        alert_entity = str(signals[0].get("entity") or "")
    if not alert_entity:
        alert_entity = "unknown"

    return AlarmArtifact(
        case_id=incident.case_id,
        dataset=incident.dataset,
        trigger_time=trigger_time,
        window=window,
        alert_entity=alert_entity,
        signals=signals,
        provenance={
            "source": source,
            "alarm_protocol": ALARM_PROTOCOL_VERSION,
            "fault_window_seconds": width_seconds,
            "normal_window_seconds": width_seconds,
            "window_width_seconds": width_seconds,
            "metric_scoring": "residual_shewhart_ks_iqr",
            "alert_entity_source": "raw_metric_iqr_score",
            "label_usage": "none; benchmark trigger only defines the shared RCL start boundary",
        },
        detector=ALARM_PROTOCOL_VERSION,
    )


def detect_alarm(
    incident: RawIncident,
    *,
    window_width_seconds: int = FAULT_WINDOW_SECONDS,
) -> Optional[AlarmArtifact]:
    width_seconds = max(int(window_width_seconds), 1)
    benchmark_alarm = _benchmark_trigger_alarm(incident, window_width_seconds=width_seconds)
    if benchmark_alarm is not None:
        return benchmark_alarm

    detector_window_radius = int(DEFAULT_DETECTOR_CONFIG["window_radius"])
    if width_seconds != FAULT_WINDOW_SECONDS:
        detector_window_radius = max(1, width_seconds // 2)
    detector = RobustAlarmDetector(
        baseline_points=int(DEFAULT_DETECTOR_CONFIG["baseline_points"]),
        threshold=float(DEFAULT_DETECTOR_CONFIG["threshold"]),
        window_radius=detector_window_radius,
    )
    alarms: List[AlarmArtifact] = []
    for key, series in sorted(incident.metric_series.items()):
        entity, metric = key.rsplit(".", 1) if "." in key else (key, "value")
        alarm = detect_alarm_from_series(
            case_id=incident.case_id,
            dataset=incident.dataset,
            entity=entity,
            metric=metric,
            series=series,
            detector=detector,
        )
        if alarm is not None:
            alarms.append(alarm)
    if alarms:
        return max(alarms, key=_alarm_score)
    return None


def get_or_create_alarm(
    incident: RawIncident,
    *,
    repo_root: str | Path = REPO_ROOT,
    write_cache: bool = True,
    window_width_seconds: int = FAULT_WINDOW_SECONDS,
) -> Optional[AlarmArtifact]:
    path = alarm_cache_path(incident, repo_root, window_width_seconds=window_width_seconds)
    cached = _load_cached_alarm(path)
    if cached is None and max(int(window_width_seconds), 1) == FAULT_WINDOW_SECONDS:
        legacy_path = Path(repo_root) / "outputs" / "nurse" / "alarms" / incident.dataset / f"{incident.case_id}.json"
        if legacy_path != path:
            cached = _load_cached_alarm(legacy_path)
    if cached is not None:
        return cached
    alarm = detect_alarm(incident, window_width_seconds=window_width_seconds)
    if alarm is not None and write_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(alarm.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return alarm
