"""Compile raw telemetry windows into typed Nurse evidence facts."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from collections import defaultdict
from statistics import median
from typing import Iterable, List, Sequence, Tuple

import pandas as pd

from .alarm_detector import score_fixed_window_metric
from .generated_tools.sliding_3sigma_window import score_metric as score_sliding_3sigma_metric
from .evidence_program import EvidenceFact
from .matching import canonical_entity
from .raw_adapters import RawIncident
from .schemas import AlarmArtifact, EvidenceCandidate


SOURCE_TOKENS = (
    "cpu",
    "memory",
    "mem",
    "disk",
    "error",
    "exception",
    "timeout",
    "restart",
    "process",
    "thread",
)
SYMPTOM_TOKENS = ("latency", "rrt", "duration", "request", "response", "qps", "throughput")
METRIC_SCORE_BACKENDS = {"evidence_robust", "alarm_fixed_window", "sliding_3sigma_window"}

METRIC_WEIGHTS = {
    "cpu": 4.0,
    "memory": 3.5,
    "mem": 3.5,
    "disk": 3.0,
    "process": 3.0,
    "timeout": 3.0,
    "latency": 2.8,
    "rrt": 2.8,
    "error": 2.5,
    "exception": 2.5,
    "request": 0.7,
    "response": 0.7,
    "qps": 0.7,
}
BUSINESS_SERVICES = {
    "ad",
    "cart",
    "checkout",
    "currency",
    "email",
    "frontend",
    "inventory",
    "payment",
    "product-catalog",
    "recommendation",
    "shipping",
}


def metric_role(metric: str) -> str:
    text = str(metric or "").lower()
    if any(token in text for token in SOURCE_TOKENS):
        return "source_metric"
    if any(token in text for token in SYMPTOM_TOKENS):
        return "symptom_metric"
    return "generic_metric"


def metric_weight(metric: str) -> float:
    text = str(metric or "").lower()
    for token, weight in METRIC_WEIGHTS.items():
        if token in text:
            return float(weight)
    return 1.0


def parse_time_value(value) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return None


def _align_to_series_scale(value: float | None, times: Sequence[float]) -> float | None:
    if value is None or not times:
        return value
    reference = median([abs(t) for t in times if math.isfinite(t)] or [0.0])
    aligned = float(value)
    if reference > 1e17 and abs(aligned) < 1e11:
        return aligned * 1_000_000_000.0
    if reference > 1e14 and abs(aligned) < 1e11:
        return aligned * 1_000_000.0
    if reference > 1e11 and abs(aligned) < 1e11:
        return aligned * 1_000.0
    return aligned


def coerce_window_to_series(window: dict, times: Sequence[float]) -> Tuple[float | None, float | None]:
    start = _align_to_series_scale(parse_time_value(window.get("start")), times)
    end = _align_to_series_scale(parse_time_value(window.get("end")), times)
    if start is not None and end is not None and start > end:
        start, end = end, start
    return start, end


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


def _robust_score(baseline: Sequence[float], values: Sequence[float]) -> Tuple[float, float, float]:
    if not baseline or not values:
        return 0.0, 0.0, 0.0
    center = float(median(baseline))
    scale = _mad(baseline)
    if scale > 1e-12:
        scores = [0.6745 * abs(value - center) / scale for value in values]
    else:
        scale = _std(baseline) or 1.0
        scores = [abs(value - center) / scale for value in values]
    peak = max(values, key=lambda value: abs(value - center))
    return float(max(scores)), center, float(peak)



def _coerce_window_value(window: dict, key: str, times: Sequence[float]) -> float | None:
    return _align_to_series_scale(parse_time_value(window.get(key)), times)


def _finite_score(value: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isfinite(score):
        return score
    return 1_000_000.0 if score > 0 else 0.0


def _metric_window_score(
    series: Sequence[Tuple[float, float]],
    window: dict,
    *,
    start: float,
    end: float,
    normal: Sequence[float],
    anomalous: Sequence[float],
    score_backend: str,
) -> Tuple[float, float, float, dict]:
    if score_backend == "evidence_robust":
        score, baseline_center, anomalous_peak = _robust_score(normal, anomalous)
        return score, baseline_center, anomalous_peak, {"score": float(score), "backend": score_backend}
    if score_backend == "sliding_3sigma_window":
        times = [timestamp for timestamp, _ in series]
        normal_start = _coerce_window_value(window, "normal_start", times)
        normal_end = _coerce_window_value(window, "normal_end", times)
        components = score_sliding_3sigma_metric(
            series,
            start=float(start),
            end=float(end),
            normal_start=float(normal_start) if normal_start is not None else None,
            normal_end=float(normal_end) if normal_end is not None else None,
        )
        score = _finite_score(float(components.get("score", 0.0)))
        baseline_center = float(components.get("normal_baseline", median(normal) if normal else 0.0) or 0.0)
        anomalous_peak = float(components.get("anomalous_peak", components.get("fault_mean", median(anomalous) if anomalous else 0.0)) or 0.0)
        components = {key: _finite_score(value) for key, value in components.items()}
        components["backend"] = score_backend
        return score, baseline_center, anomalous_peak, components
    if score_backend != "alarm_fixed_window":
        raise ValueError(f"unsupported metric score backend: {score_backend}")
    times = [timestamp for timestamp, _ in series]
    normal_start = _coerce_window_value(window, "normal_start", times)
    normal_end = _coerce_window_value(window, "normal_end", times)
    if normal_start is None:
        normal_start = min(times) if times else start
    if normal_end is None:
        normal_end = start
    components = score_fixed_window_metric(
        series,
        start=float(start),
        end=float(end),
        normal_start=float(normal_start),
        normal_end=float(normal_end),
    )
    score = _finite_score(float(components.get("score", 0.0)))
    baseline_center = float(components.get("normal_baseline", median(normal) if normal else 0.0) or 0.0)
    anomalous_peak = float(components.get("fault_mean", median(anomalous) if anomalous else 0.0) or 0.0)
    components = {key: _finite_score(value) for key, value in components.items()}
    components["backend"] = score_backend
    return score, baseline_center, anomalous_peak, components

def _allowed_lookup(service_candidates: Iterable[str] | None, services: Iterable[str]) -> dict[str, str]:
    candidates = list(service_candidates or []) or list(services)
    lookup: dict[str, str] = {}
    for candidate in candidates:
        service = canonical_entity(str(candidate))
        if service:
            lookup[service] = service
    return lookup


def _metric_table_from_incident(incident: RawIncident) -> pd.DataFrame:
    rows = []
    for key, series in incident.metric_series.items():
        if "." in key:
            entity, metric = key.rsplit(".", 1)
        else:
            entity, metric = key, "value"
        for timestamp, value in series:
            rows.append(
                {
                    "time": timestamp,
                    "service": entity,
                    "metric": metric,
                    "value": value,
                    "metric_key": key,
                }
            )
    return pd.DataFrame(rows, columns=["time", "service", "metric", "value", "metric_key"])


def compile_metric_facts_from_table(
    metrics: pd.DataFrame,
    alarm: AlarmArtifact,
    *,
    service_candidates: Iterable[str] | None = None,
    min_score: float = 2.5,
    score_backend: str = "evidence_robust",
) -> List[EvidenceFact]:
    if metrics.empty:
        return []
    if score_backend not in METRIC_SCORE_BACKENDS:
        raise ValueError(f"unsupported metric score backend: {score_backend}")
    required = {"time", "service", "metric", "value"}
    if not required.issubset(metrics.columns):
        missing = ", ".join(sorted(required - set(metrics.columns)))
        raise ValueError(f"metric table missing required columns: {missing}")

    frame = metrics.copy()
    frame["time"] = pd.to_numeric(frame["time"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["time", "service", "metric", "value"])
    allowed = _allowed_lookup(service_candidates, frame["service"].astype(str).unique())
    facts: List[EvidenceFact] = []

    for (raw_service, raw_metric), group in frame.groupby(["service", "metric"], dropna=True):
        service = canonical_entity(str(raw_service))
        if not service or service not in allowed:
            continue
        metric = str(raw_metric)
        group = group.sort_values("time")
        times = [float(value) for value in group["time"].tolist()]
        start, end = coerce_window_to_series(alarm.window, times)
        if start is None:
            start = times[max(min(len(times) // 2, len(times) - 1), 0)]
        if end is None:
            end = max(times)
        normal = group[group["time"] < start]
        anomalous = group[(group["time"] >= start) & (group["time"] <= end)]
        if anomalous.empty:
            anomalous = group[group["time"] >= start]
        if normal.empty:
            normal = group.iloc[: max(1, min(20, len(group) // 2))]
        if normal.empty or anomalous.empty:
            continue

        normal_values = [float(value) for value in normal["value"].tolist()]
        anomalous_values = [float(value) for value in anomalous["value"].tolist()]
        series = [(float(row.time), float(row.value)) for row in group.itertuples(index=False)]
        score, baseline_center, anomalous_peak, score_components = _metric_window_score(
            series,
            alarm.window,
            start=float(start),
            end=float(end),
            normal=normal_values,
            anomalous=anomalous_values,
            score_backend=score_backend,
        )
        if score < min_score:
            continue
        role = metric_role(metric)
        weight = score * metric_weight(metric)
        first_seen = None
        for row in anomalous.itertuples(index=False):
            row_score, _, _, _ = _metric_window_score(
                series,
                alarm.window,
                start=float(row.time),
                end=float(row.time),
                normal=normal_values,
                anomalous=[float(row.value)],
                score_backend=score_backend,
            )
            if row_score >= min_score:
                first_seen = float(row.time)
                break
        attrs = {
            "role": role,
            "raw_service": str(raw_service),
            "metric": metric,
            "robust_z": score,
            "score_backend": score_backend,
            "score_components": score_components,
            "metric_weight": metric_weight(metric),
            "baseline_median": baseline_center,
            "anomalous_peak": anomalous_peak,
            "first_seen": first_seen if first_seen is not None else float(anomalous["time"].iloc[0]),
        }
        if "metric_key" in group.columns:
            attrs["metric_key"] = str(group["metric_key"].iloc[0])
        facts.append(EvidenceFact("anomaly", service, metric, float(weight), attrs))

    return sorted(facts, key=lambda fact: (-fact.weight, fact.subject, fact.object))

def compile_metric_facts(
    incident: RawIncident,
    alarm: AlarmArtifact,
    *,
    service_candidates: Iterable[str] | None = None,
    min_score: float = 2.5,
    score_backend: str = "evidence_robust",
) -> List[EvidenceFact]:
    return compile_metric_facts_from_table(
        _metric_table_from_incident(incident),
        alarm,
        service_candidates=service_candidates,
        min_score=min_score,
        score_backend=score_backend,
    )


def service_candidates_from_incident(incident: RawIncident, alarm: AlarmArtifact | None = None) -> List[str]:
    priorities: dict[str, int] = {}

    def add(value: str, priority: int) -> None:
        service = canonical_entity(value)
        if not service:
            return
        if service not in priorities or priority < priorities[service]:
            priorities[service] = priority

    alert_entity = incident.metadata.get("alert_entity") if isinstance(incident.metadata, dict) else {}
    if isinstance(alert_entity, dict) and alert_entity.get("entity_name"):
        add(str(alert_entity.get("entity_name")), 0)
    if alarm and alarm.alert_entity:
        add(alarm.alert_entity, 15)

    for src, dst in incident.topology_edges:
        for raw in (src, dst):
            service = canonical_entity(raw)
            if not service:
                continue
            if "oteldemo." in str(raw).lower() or service in BUSINESS_SERVICES:
                add(raw, 10)
            else:
                add(raw, 40)

    for key in incident.metric_series:
        entity = key.rsplit(".", 1)[0] if "." in key else key
        add(entity, 20)
    return [
        service
        for service, _ in sorted(priorities.items(), key=lambda item: (item[1], item[0]))
        if service
    ]


def compile_metric_candidates(
    incident: RawIncident,
    alarm: AlarmArtifact,
    *,
    max_candidates: int = 80,
    max_per_service: int = 3,
    min_score: float = 2.5,
    score_backend: str = "evidence_robust",
) -> List[EvidenceCandidate]:
    candidates: List[EvidenceCandidate] = []
    services = service_candidates_from_incident(incident, alarm)
    facts = compile_metric_facts(
        incident,
        alarm,
        service_candidates=services,
        min_score=min_score,
        score_backend=score_backend,
    )
    service_counts: dict[str, int] = defaultdict(int)
    selected: List[EvidenceFact] = []
    deferred: List[EvidenceFact] = []
    for fact in facts:
        if service_counts[fact.subject] < max_per_service:
            selected.append(fact)
            service_counts[fact.subject] += 1
        else:
            deferred.append(fact)
        if len(selected) >= max_candidates:
            break
    if len(selected) < max_candidates:
        selected.extend(deferred[: max_candidates - len(selected)])

    for index, fact in enumerate(selected[:max_candidates]):
        attrs = dict(fact.attrs)
        metric_key = str(attrs.get("metric_key") or f"{attrs.get('raw_service', fact.subject)}.{fact.object}")
        candidates.append(
            EvidenceCandidate(
                evidence_id=f"{alarm.case_id}:raw_metric:{index}:{fact.subject}:{fact.object}",
                case_id=alarm.case_id,
                entity=fact.subject,
                signal=fact.object,
                kind="metric",
                time_range=dict(alarm.window),
                claim={
                    "anomaly_score": attrs.get("robust_z"),
                    "observed": attrs.get("anomalous_peak"),
                    "baseline": attrs.get("baseline_median"),
                    "metric_role": attrs.get("role"),
                    "metric_weight": attrs.get("metric_weight"),
                    "evidence_program_weight": fact.weight,
                    "first_seen": attrs.get("first_seen"),
                },
                provenance={
                    "source": "raw_evidence_program",
                    "metric_key": metric_key,
                    "dataset": incident.dataset,
                    "raw_service": attrs.get("raw_service", fact.subject),
                },
                confidence=min(1.0, max(0.1, float(attrs.get("robust_z") or 0.0) / 10.0)),
            )
        )
    return candidates


