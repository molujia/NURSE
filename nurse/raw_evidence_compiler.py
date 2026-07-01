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


def _ordered_unique_services(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for raw in values:
        service = canonical_entity(str(raw))
        if not service or service in seen:
            continue
        seen.add(service)
        ordered.append(service)
    return ordered


def _topology_adjacency(topology_edges: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for raw_src, raw_dst in topology_edges:
        src = canonical_entity(str(raw_src))
        dst = canonical_entity(str(raw_dst))
        if not src or not dst or src == dst:
            continue
        adjacency[src].add(dst)
        adjacency[dst].add(src)
    return adjacency


def _candidate_support_profiles(
    incident: RawIncident,
    alarm: AlarmArtifact | None,
) -> dict[str, dict[str, float]]:
    if alarm is None:
        return {}
    profiles: dict[str, dict[str, float]] = {}
    for fact in compile_metric_facts(incident, alarm):
        service = canonical_entity(str(fact.subject))
        if not service:
            continue
        profile = profiles.setdefault(
            service,
            {
                "moderate_evidence": 0.0,
                "medium_evidence": 0.0,
                "strong_evidence": 0.0,
                "max_score": 0.0,
            },
        )
        score = float(fact.attrs.get("robust_z") or 0.0)
        if score >= 3.0:
            profile["moderate_evidence"] += 1.0
        if score >= 5.0:
            profile["medium_evidence"] += 1.0
        if score >= 8.0:
            profile["strong_evidence"] += 1.0
        profile["max_score"] = max(float(profile["max_score"]), score)
    return profiles


def _candidate_admission_stage(
    profile: dict[str, float],
    *,
    is_alert_entity: bool,
) -> int | None:
    if is_alert_entity:
        return 0
    strong = int(profile.get("strong_evidence", 0.0))
    medium = int(profile.get("medium_evidence", 0.0))
    moderate = int(profile.get("moderate_evidence", 0.0))
    max_score = float(profile.get("max_score", 0.0))
    if strong >= 2 or max_score >= 20.0:
        return 0
    if medium >= 2:
        return 1
    if strong >= 1:
        return 2
    if moderate >= 2:
        return 3
    if medium >= 1:
        return 4
    if moderate >= 1:
        return 5
    return None


def _candidate_profile_sort_key(
    service: str,
    profile: dict[str, float],
    *,
    alert_entity: str,
) -> tuple[float, float, float, float, float, str]:
    return (
        0.0 if service == alert_entity else 1.0,
        -float(profile.get("strong_evidence", 0.0)),
        -float(profile.get("medium_evidence", 0.0)),
        -float(profile.get("moderate_evidence", 0.0)),
        -float(profile.get("max_score", 0.0)),
        service,
    )


def service_candidates_from_incident(
    incident: RawIncident,
    alarm: AlarmArtifact | None = None,
    *,
    candidate_expansion_rounds: int = 2,
) -> List[str]:
    adjacency = _topology_adjacency(incident.topology_edges)
    metric_services = _ordered_unique_services(
        key.rsplit(".", 1)[0] if "." in key else key for key in incident.metric_series
    )

    alert_entity = ""
    if isinstance(incident.metadata, dict):
        metadata_alert = incident.metadata.get("alert_entity") or {}
        if isinstance(metadata_alert, dict):
            alert_entity = canonical_entity(str(metadata_alert.get("entity_name") or ""))
        elif metadata_alert:
            alert_entity = canonical_entity(str(metadata_alert))
    if alarm and alarm.alert_entity:
        alert_entity = canonical_entity(str(alarm.alert_entity)) or alert_entity

    support_profiles = _candidate_support_profiles(incident, alarm)
    staged_profiles: list[tuple[int, str, dict[str, float]]] = []
    for service, profile in support_profiles.items():
        stage = _candidate_admission_stage(
            profile,
            is_alert_entity=service == alert_entity,
        )
        if stage is None:
            continue
        staged_profiles.append((stage, service, profile))
    seed_services = [
        service
        for stage, service, profile in sorted(
            staged_profiles,
            key=lambda item: (
                item[0],
                _candidate_profile_sort_key(
                    item[1],
                    item[2],
                    alert_entity=alert_entity,
                ),
            ),
        )
        if stage <= int(candidate_expansion_rounds)
    ]
    if alert_entity and alert_entity not in seed_services:
        seed_services.insert(0, alert_entity)
    if not seed_services and support_profiles:
        strongest = sorted(
            support_profiles.items(),
            key=lambda item: _candidate_profile_sort_key(
                item[0],
                item[1],
                alert_entity=alert_entity,
            ),
        )[0][0]
        seed_services = [strongest]
    if not seed_services:
        seed_services = _ordered_unique_services(
            ([alert_entity] if alert_entity else [])
            + (sorted(adjacency) if adjacency else [])
            + metric_services
        )

    discovered_depth: dict[str, int] = {}
    discovery_order: dict[str, int] = {}

    def add(service: str, depth: int) -> bool:
        service = canonical_entity(str(service))
        if not service or service in discovered_depth:
            return False
        discovered_depth[service] = int(depth)
        discovery_order[service] = len(discovery_order)
        return True

    frontier: List[str] = []
    for seed in seed_services:
        if add(seed, 0):
            frontier.append(seed)

    expansion_allowed = set(_ordered_unique_services(list(support_profiles) + ([alert_entity] if alert_entity else [])))
    max_rounds = max(int(candidate_expansion_rounds), 0)
    for round_index in range(1, max_rounds + 1):
        next_frontier: List[str] = []
        for service in frontier:
            for neighbor in sorted(adjacency.get(service, set())):
                if expansion_allowed and neighbor not in expansion_allowed:
                    continue
                if add(neighbor, round_index):
                    next_frontier.append(neighbor)
        if not next_frontier:
            break
        frontier = next_frontier

    if not discovered_depth:
        fallback = metric_services or _ordered_unique_services(sorted(adjacency))
        if alert_entity:
            fallback = _ordered_unique_services([alert_entity] + fallback)
        for service in fallback:
            add(service, 0)

    return [
        service
        for service, _ in sorted(
            discovered_depth.items(),
            key=lambda item: (item[1], discovery_order.get(item[0], 0), item[0]),
        )
    ]


def compile_metric_candidates(
    incident: RawIncident,
    alarm: AlarmArtifact,
    *,
    max_candidates: int = 80,
    max_per_service: int = 3,
    min_score: float = 2.5,
    score_backend: str = "evidence_robust",
    service_candidates: Iterable[str] | None = None,
    candidate_expansion_rounds: int = 2,
) -> List[EvidenceCandidate]:
    candidates: List[EvidenceCandidate] = []
    services = (
        _ordered_unique_services(service_candidates)
        if service_candidates is not None
        else service_candidates_from_incident(
            incident,
            alarm,
            candidate_expansion_rounds=candidate_expansion_rounds,
        )
    )
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


