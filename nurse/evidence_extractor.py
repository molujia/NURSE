"""Candidate evidence extraction from Nurse alarms and raw incidents."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import List

from .matching import canonical_entity
from .raw_evidence_compiler import BUSINESS_SERVICES, coerce_window_to_series, compile_metric_candidates
from .raw_adapters import RawIncident
from .schemas import AlarmArtifact, EvidenceCandidate


def _metric_key(entity: str, metric: str) -> str:
    return f"{entity}.{metric}"


def extract_metric_candidates(incident: RawIncident, alarm: AlarmArtifact) -> List[EvidenceCandidate]:
    candidates: List[EvidenceCandidate] = []
    for index, signal in enumerate(alarm.signals):
        entity = str(signal.get("entity") or alarm.alert_entity)
        metric = str(signal.get("metric") or signal.get("signal") or "value")
        evidence_id = f"{alarm.case_id}:metric:{index}:{entity}:{metric}"
        candidates.append(
            EvidenceCandidate(
                evidence_id=evidence_id,
                case_id=alarm.case_id,
                entity=entity,
                signal=metric,
                kind="metric",
                time_range=dict(alarm.window),
                claim={
                    "anomaly_score": signal.get("score"),
                    "observed": signal.get("value"),
                    "baseline": signal.get("baseline"),
                },
                provenance={
                    "source": "alarm_signal",
                    "metric_key": _metric_key(entity, metric),
                    "dataset": incident.dataset,
                },
            )
        )
    return candidates


def _signal_from_alert_title(title: str) -> str:
    text = str(title or "").lower()
    if any(token in text for token in ("cpu", "processor")):
        return "cpu"
    if any(token in text for token in ("latency", "delay", "rrt", "耗时", "延迟")):
        return "latency"
    if any(token in text for token in ("error", "exception", "错误", "异常")):
        return "error_rate"
    if any(token in text for token in ("traffic", "request", "流量", "请求")):
        return "request_count"
    return "entry_alert"


def _service_like(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    if text in BUSINESS_SERVICES:
        return True
    if text.startswith("cn-") and "." in text:
        return True
    if any(token in text for token in (" ", "/", "(", ")", "{", "}", ":")):
        return False
    return "-" in text


def _services_from_text(value: str) -> list[str]:
    text = str(value or "").lower()
    services = []
    compact = text.replace("-", "").replace("_", "")
    for service in sorted(BUSINESS_SERVICES):
        if service in text or service.replace("-", "") in compact:
            services.append(service)
    return services


def extract_alert_candidate(incident: RawIncident, alarm: AlarmArtifact) -> List[EvidenceCandidate]:
    if alarm.signals or not alarm.alert_entity or alarm.alert_entity == "unknown":
        return []
    signal = _signal_from_alert_title(str(incident.metadata.get("alert_title") or ""))
    return [
        EvidenceCandidate(
            evidence_id=f"{alarm.case_id}:alert:0:{alarm.alert_entity}:{signal}",
            case_id=alarm.case_id,
            entity=alarm.alert_entity,
            signal=signal,
            kind="alert",
            time_range=dict(alarm.window),
            claim={
                "alert_title": incident.metadata.get("alert_title", ""),
                "trigger_time": alarm.trigger_time,
                "metric_role": "symptom_metric",
                "evidence_program_weight": 1.0,
            },
            provenance={"source": "task_alert_contract", "dataset": incident.dataset},
            confidence=0.8,
        )
    ]


def extract_topology_candidates(incident: RawIncident, alarm: AlarmArtifact) -> List[EvidenceCandidate]:
    candidates: List[EvidenceCandidate] = []
    for index, (src, dst) in enumerate(incident.topology_edges):
        candidates.append(
            EvidenceCandidate(
                evidence_id=f"{alarm.case_id}:topology:{index}:{src}:{dst}",
                case_id=alarm.case_id,
                entity=src,
                signal="topology_edge",
                kind="topology",
                time_range=dict(alarm.window),
                claim={"src": src, "dst": dst},
                provenance={"source": "topology", "dataset": incident.dataset},
                confidence=1.0,
            )
        )
    return candidates


def extract_trace_candidates(
    incident: RawIncident,
    alarm: AlarmArtifact,
    *,
    max_candidates: int = 12,
) -> List[EvidenceCandidate]:
    alert_entity = canonical_entity(alarm.alert_entity)
    alert_title = str(incident.metadata.get("alert_title") or "").lower()
    if alert_entity == "inventory" or "inventory" in alert_title:
        return []
    trace_path = incident.file_index.get("traces")
    if not trace_path or not trace_path.exists():
        return []
    try:
        import pandas as pd  # type: ignore

        frame = pd.read_parquet(
            trace_path,
            columns=["serviceName", "spanName", "statusCode", "statusMessage", "duration", "startTime"],
        )
    except Exception:
        return []
    required = {"serviceName", "spanName", "statusCode", "startTime"}
    if not required.issubset(frame.columns):
        return []

    frame = frame.copy()
    frame["startTime"] = pd.to_numeric(frame["startTime"], errors="coerce")
    frame["duration"] = pd.to_numeric(frame.get("duration"), errors="coerce")
    frame = frame.dropna(subset=["startTime"])
    times = [float(value) for value in frame["startTime"].tolist()]
    start, end = coerce_window_to_series(alarm.window, times)
    if start is not None:
        frame = frame[frame["startTime"] >= start]
    if end is not None:
        frame = frame[frame["startTime"] <= end]
    if frame.empty:
        return []

    messages = frame.get("statusMessage", "").fillna("").astype(str).str.strip()
    error_frame = frame[messages != ""]
    if error_frame.empty:
        return []

    stats: dict[str, dict[str, float | int | str]] = defaultdict(
        lambda: {"weight": 0.0, "rows": 0, "max_duration": 0.0, "example_span": "", "example_message": ""}
    )
    for row in error_frame.itertuples(index=False):
        source = canonical_entity(getattr(row, "serviceName", ""))
        span_target = canonical_entity(getattr(row, "spanName", ""))
        message = str(getattr(row, "statusMessage", "") or "")
        duration = float(getattr(row, "duration", 0.0) or 0.0)
        message_bonus = 2.0 if message.strip() else 0.0
        targets = []
        message_targets = _services_from_text(message)
        for target in message_targets:
            if target != source and _service_like(target):
                targets.append((target, 4.0 + message_bonus))
        if span_target and span_target != source and _service_like(span_target):
            targets.append((span_target, 3.0 + message_bonus))
        if source and source != alert_entity and _service_like(source):
            targets.append((source, 1.0 + message_bonus * 0.25))
        elif source and source != alert_entity and not targets and _service_like(source):
            targets.append((source, 0.5 + message_bonus * 0.25))
        for service, weight in targets:
            item = stats[service]
            item["weight"] = float(item["weight"]) + weight
            item["rows"] = int(item["rows"]) + 1
            item["max_duration"] = max(float(item["max_duration"]), duration if math.isfinite(duration) else 0.0)
            if not item["example_span"]:
                item["example_span"] = str(getattr(row, "spanName", "") or "")
            if message and not item["example_message"]:
                item["example_message"] = message[:240]

    candidates: List[EvidenceCandidate] = []
    ranked = sorted(stats.items(), key=lambda item: (-float(item[1]["weight"]), item[0]))
    for index, (service, item) in enumerate(ranked[:max_candidates]):
        rows = int(item["rows"])
        weight = min(40.0, math.log1p(float(item["weight"])) * 8.0)
        role = "symptom_metric" if service == alert_entity else "source_metric"
        candidates.append(
            EvidenceCandidate(
                evidence_id=f"{alarm.case_id}:trace:{index}:{service}:trace_error",
                case_id=alarm.case_id,
                entity=service,
                signal="trace_error",
                kind="trace",
                time_range=dict(alarm.window),
                claim={
                    "matching_rows": rows,
                    "max_duration": item["max_duration"],
                    "example_span": item["example_span"],
                    "example_message": item["example_message"],
                    "metric_role": role,
                    "evidence_program_weight": weight,
                },
                provenance={"source": "trace_error_spans", "dataset": incident.dataset},
                confidence=min(1.0, max(0.2, math.log1p(rows) / 8.0)),
            )
        )
    return candidates


def _node_from_event(row, payload: dict) -> str:
    values = [
        row.get("hostname") if hasattr(row, "get") else "",
        (payload.get("source") or {}).get("host") if isinstance(payload.get("source"), dict) else "",
        payload.get("reportingInstance"),
        payload.get("message"),
    ]
    for value in values:
        text = str(value or "").strip()
        if text.startswith("cn-") and "." in text:
            return text
        match = re.search(r"(?:node[:\s]+|on node\s+)(cn-[a-z0-9.-]+)", text, re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _timestamp_near_window(timestamp: str, window: dict, *, tolerance_seconds: int = 60) -> bool:
    if not timestamp:
        return False
    try:
        import pandas as pd  # type: ignore

        point = pd.Timestamp(timestamp)
        start = pd.Timestamp(window.get("start"))
        end = pd.Timestamp(window.get("end"))
        if point.tzinfo and start.tzinfo is None:
            start = start.tz_localize(point.tzinfo)
        if point.tzinfo and end.tzinfo is None:
            end = end.tz_localize(point.tzinfo)
        if start.tzinfo and point.tzinfo is None:
            point = point.tz_localize(start.tzinfo)
        tolerance = pd.Timedelta(seconds=tolerance_seconds)
        return bool(start - tolerance <= point <= end + tolerance)
    except Exception:
        return False


def extract_event_candidates(
    incident: RawIncident,
    alarm: AlarmArtifact,
    *,
    max_candidates: int = 8,
) -> List[EvidenceCandidate]:
    alert_entity = canonical_entity(alarm.alert_entity)
    alert_title = str(incident.metadata.get("alert_title") or "").lower()
    if alert_entity == "inventory" or "inventory" in alert_title:
        return []
    events_path = incident.file_index.get("events")
    if not events_path or not events_path.exists():
        return []
    try:
        import pandas as pd  # type: ignore

        frame = pd.read_parquet(events_path, columns=["eventId", "hostname", "level"])
    except Exception:
        return []

    stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {"rows": 0, "weight": 0.0, "reason": "", "message": "", "timestamp": ""}
    )
    for row in frame.to_dict("records"):
        try:
            payload = json.loads(str(row.get("eventId") or "{}"))
        except Exception:
            continue
        reason = str(payload.get("reason") or "")
        message = str(payload.get("message") or "")
        if reason != "Preempted" and "preempted" not in message.lower():
            continue
        timestamp = str(payload.get("lastTimestamp") or payload.get("eventTime") or payload.get("firstTimestamp") or "")
        if not _timestamp_near_window(timestamp, alarm.window):
            continue
        node = canonical_entity(_node_from_event(row, payload))
        if not (node.startswith("cn-") and "." in node):
            continue
        item = stats[node]
        item["rows"] = int(item["rows"]) + 1
        item["weight"] = float(item["weight"]) + 80.0
        item["reason"] = item["reason"] or reason
        item["message"] = item["message"] or message[:240]
        item["timestamp"] = item["timestamp"] or timestamp

    candidates: List[EvidenceCandidate] = []
    ranked = sorted(stats.items(), key=lambda item: (-float(item[1]["weight"]), item[0]))
    for index, (node, item) in enumerate(ranked[:max_candidates]):
        candidates.append(
            EvidenceCandidate(
                evidence_id=f"{alarm.case_id}:event:{index}:{node}:node_preempted",
                case_id=alarm.case_id,
                entity=node,
                signal="node_preempted",
                kind="event",
                time_range=dict(alarm.window),
                claim={
                    "reason": item["reason"],
                    "message": item["message"],
                    "matching_rows": item["rows"],
                    "event_timestamp": item["timestamp"],
                    "metric_role": "source_metric",
                    "evidence_program_weight": item["weight"],
                    "rank_bonus": 4.0,
                },
                provenance={"source": "kubernetes_event", "dataset": incident.dataset},
                confidence=0.95,
            )
        )
    return candidates


def extract_candidates(
    incident: RawIncident,
    alarm: AlarmArtifact,
    *,
    metric_score_backend: str = "evidence_robust",
    service_candidates: list[str] | None = None,
) -> List[EvidenceCandidate]:
    candidates = (
        compile_metric_candidates(
            incident,
            alarm,
            score_backend=metric_score_backend,
            service_candidates=service_candidates,
        )
        + extract_event_candidates(incident, alarm)
        + extract_trace_candidates(incident, alarm)
        + extract_metric_candidates(incident, alarm)
        + extract_alert_candidate(incident, alarm)
        + extract_topology_candidates(incident, alarm)
    )
    if service_candidates:
        allowed = {
            canonical_entity(str(candidate))
            for candidate in service_candidates
            if canonical_entity(str(candidate))
        }
        if allowed:
            candidates = [candidate for candidate in candidates if canonical_entity(candidate.entity) in allowed]
    return candidates
