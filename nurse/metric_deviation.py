"""Raw metric deviation scoring for Nurse evidence closure."""

from __future__ import annotations

import math
from statistics import median
from collections import defaultdict
from typing import Iterable, Sequence

from .alarm_detector import score_fixed_window_metric
from .generated_tools.sliding_3sigma_window import score_metric as score_sliding_3sigma_metric
from .matching import canonical_entity
from .raw_adapters import RawIncident
from .raw_evidence_compiler import METRIC_SCORE_BACKENDS, coerce_window_to_series
from .schemas import AlarmArtifact


def _mad(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    center = median(values)
    return median([abs(value - center) for value in values])


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = sum(values) / len(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def _finite_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isfinite(score):
        return score
    return 1_000_000.0 if score > 0 else 0.0


def _split_metric_key(key: str) -> tuple[str, str]:
    if "." in key:
        entity, metric = key.rsplit(".", 1)
    else:
        entity, metric = key, "value"
    return entity, metric


def _allowed_services(service_candidates: Iterable[str] | None) -> set[str]:
    return {
        canonical_entity(str(service))
        for service in (service_candidates or [])
        if canonical_entity(str(service))
    }


def _metric_window_rows(
    incident: RawIncident,
    alarm: AlarmArtifact,
    *,
    service_candidates: Iterable[str] | None = None,
) -> tuple[
    dict[tuple[str, str], list[float]],
    dict[tuple[str, str], list[float]],
]:
    allowed = _allowed_services(service_candidates)
    rows: list[tuple[float, str, str, float]] = []
    all_times: list[float] = []
    for key, series in sorted(incident.metric_series.items()):
        service_raw, metric = _split_metric_key(str(key))
        service = canonical_entity(service_raw)
        if not service or (allowed and service not in allowed):
            continue
        for timestamp, value in series:
            try:
                ts = float(timestamp)
                val = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(ts) or not math.isfinite(val):
                continue
            rows.append((ts, service, metric, val))
            all_times.append(ts)
    if not rows:
        return {}, {}

    start, end = coerce_window_to_series(alarm.window, all_times)
    if start is None:
        ordered_times = sorted(all_times)
        start = ordered_times[max(min(len(ordered_times) // 2, len(ordered_times) - 1), 0)]
    if end is None:
        end = max(all_times)

    normal_rows = [row for row in rows if row[0] < start]
    anomalous_rows = [row for row in rows if start <= row[0] <= end]
    if not normal_rows:
        normal_rows = rows[: max(1, min(len(rows) // 2, len(rows)))]
    if not anomalous_rows:
        anomalous_rows = rows[-max(1, min(len(rows) // 2, len(rows))) :]

    normal_by_metric: dict[tuple[str, str], list[float]] = defaultdict(list)
    anomalous_by_metric: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _, service, metric, value in normal_rows:
        normal_by_metric[(service, metric)].append(value)
    for _, service, metric, value in anomalous_rows:
        anomalous_by_metric[(service, metric)].append(value)
    return normal_by_metric, anomalous_by_metric


def _sliding_3sigma_metric_scores(
    incident: RawIncident,
    alarm: AlarmArtifact,
    *,
    service_candidates: Iterable[str] | None = None,
) -> dict[str, dict[str, float]]:
    allowed = _allowed_services(service_candidates)
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for key, raw_series in sorted(incident.metric_series.items()):
        service_raw, metric = _split_metric_key(str(key))
        service = canonical_entity(service_raw)
        if not service or (allowed and service not in allowed):
            continue
        series: list[tuple[float, float]] = []
        for timestamp, value in raw_series:
            try:
                ts = float(timestamp)
                val = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(ts) and math.isfinite(val):
                series.append((ts, val))
        if not series:
            continue
        series.sort(key=lambda item: item[0])
        times = [timestamp for timestamp, _ in series]
        start, end = coerce_window_to_series(alarm.window, times)
        normal_start, normal_end = coerce_window_to_series(
            {
                "start": alarm.window.get("normal_start"),
                "end": alarm.window.get("normal_end"),
            },
            times,
        )
        if start is None:
            start = times[max(min(len(times) // 2, len(times) - 1), 0)]
        if end is None:
            end = max(times)
        components = score_sliding_3sigma_metric(
            series,
            start=float(start),
            end=float(end),
            normal_start=float(normal_start) if normal_start is not None else None,
            normal_end=float(normal_end) if normal_end is not None else None,
        )
        deviation = _finite_score(components.get("score", 0.0))
        current = scores[service].get(metric, 0.0)
        scores[service][metric] = max(current, deviation)
    return {service: dict(metric_scores) for service, metric_scores in scores.items()}

def _alarm_fixed_window_metric_scores(
    incident: RawIncident,
    alarm: AlarmArtifact,
    *,
    service_candidates: Iterable[str] | None = None,
) -> dict[str, dict[str, float]]:
    allowed = _allowed_services(service_candidates)
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for key, raw_series in sorted(incident.metric_series.items()):
        service_raw, metric = _split_metric_key(str(key))
        service = canonical_entity(service_raw)
        if not service or (allowed and service not in allowed):
            continue
        series: list[tuple[float, float]] = []
        for timestamp, value in raw_series:
            try:
                ts = float(timestamp)
                val = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(ts) and math.isfinite(val):
                series.append((ts, val))
        if not series:
            continue
        series.sort(key=lambda item: item[0])
        times = [timestamp for timestamp, _ in series]
        start, end = coerce_window_to_series(alarm.window, times)
        normal_start, normal_end = coerce_window_to_series(
            {
                "start": alarm.window.get("normal_start"),
                "end": alarm.window.get("normal_end"),
            },
            times,
        )
        if start is None:
            start = times[max(min(len(times) // 2, len(times) - 1), 0)]
        if end is None:
            end = max(times)
        if normal_start is None:
            normal_start = min(times)
        if normal_end is None:
            normal_end = start
        if normal_start > normal_end:
            normal_start, normal_end = normal_end, normal_start
        components = score_fixed_window_metric(
            series,
            start=float(start),
            end=float(end),
            normal_start=float(normal_start),
            normal_end=float(normal_end),
        )
        deviation = _finite_score(components.get("score", 0.0))
        current = scores[service].get(metric, 0.0)
        scores[service][metric] = max(current, deviation)
    return {service: dict(metric_scores) for service, metric_scores in scores.items()}


def service_metric_deviation_scores(
    incident: RawIncident,
    alarm: AlarmArtifact,
    *,
    service_candidates: Iterable[str] | None = None,
    score_backend: str = "evidence_robust",
) -> dict[str, dict[str, float]]:
    """Return uncapped metric deviation by canonical entity and metric.

    The output is inference-safe and label-free. ``evidence_robust`` preserves
    the original Nurse robust-deviation profile. ``alarm_fixed_window`` reuses
    the shared fixed-window alarm scoring components over the same alarm window.
    """

    if score_backend not in METRIC_SCORE_BACKENDS:
        raise ValueError(f"unsupported metric score backend: {score_backend}")
    if score_backend == "alarm_fixed_window":
        return _alarm_fixed_window_metric_scores(
            incident,
            alarm,
            service_candidates=service_candidates,
        )
    if score_backend == "sliding_3sigma_window":
        return _sliding_3sigma_metric_scores(
            incident,
            alarm,
            service_candidates=service_candidates,
        )

    normal_by_metric, anomalous_by_metric = _metric_window_rows(
        incident,
        alarm,
        service_candidates=service_candidates,
    )
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for service, metric in sorted(normal_by_metric):
        normal = normal_by_metric[(service, metric)]
        anomalous = anomalous_by_metric.get((service, metric), [])
        if not normal or not anomalous:
            continue
        center = median(normal)
        relative_floor = max(abs(center) * 0.005, 1.0)
        scale = max(_mad(normal), _std(normal), relative_floor)
        deviation = max(abs(value - center) / scale for value in anomalous)
        current = scores[service].get(metric, 0.0)
        scores[service][metric] = max(current, float(deviation))
    return {service: dict(metric_scores) for service, metric_scores in scores.items()}


def service_deviation_scores(
    incident: RawIncident,
    alarm: AlarmArtifact,
    *,
    service_candidates: Iterable[str] | None = None,
    score_backend: str = "evidence_robust",
) -> dict[str, float]:
    """Return uncapped metric deviation by canonical service.

    This mirrors the canonical baseline window split but stays inside Nurse and
    uses only raw telemetry plus the shared alarm artifact.
    """

    detailed = service_metric_deviation_scores(
        incident,
        alarm,
        service_candidates=service_candidates,
        score_backend=score_backend,
    )
    return {
        service: max(metric_scores.values())
        for service, metric_scores in detailed.items()
        if metric_scores
    }
