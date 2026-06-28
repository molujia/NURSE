"""Deterministic evidence verification against raw telemetry."""

from __future__ import annotations

import json
import re
from statistics import mean
from typing import List, Tuple

from .matching import canonical_entity
from .raw_evidence_compiler import BUSINESS_SERVICES, coerce_window_to_series
from .raw_adapters import RawIncident
from .schemas import EvidenceCandidate, VerifiedEvidence


def _window_points(series: List[Tuple[int | float, float]], start, end) -> List[Tuple[int | float, float]]:
    if start is None:
        start = min((t for t, _ in series), default=0)
    if end is None:
        end = max((t for t, _ in series), default=start)
    return [(t, v) for t, v in series if start <= t <= end]


def _baseline_points(series: List[Tuple[int | float, float]], start) -> List[float]:
    if start is None:
        return []
    return [v for t, v in series if t < start]


def _services_from_text(value: str) -> set[str]:
    text = str(value or "").lower()
    compact = text.replace("-", "").replace("_", "")
    services = set()
    for service in BUSINESS_SERVICES:
        if service in text or service.replace("-", "") in compact:
            services.add(service)
    return services


def verify_candidate(incident: RawIncident, candidate: EvidenceCandidate) -> VerifiedEvidence:
    if candidate.kind == "metric":
        return _verify_metric(incident, candidate)
    if candidate.kind == "alert":
        return _verify_alert(incident, candidate)
    if candidate.kind == "trace":
        return _verify_trace(incident, candidate)
    if candidate.kind == "event":
        return _verify_event(incident, candidate)
    if candidate.kind == "topology":
        return _verify_topology(incident, candidate)
    return VerifiedEvidence.from_candidate(
        candidate,
        observed={},
        verification_rule=f"{candidate.kind}_unsupported",
        verified=False,
        rejection_reason=f"unsupported evidence kind: {candidate.kind}",
    )


def _verify_metric(incident: RawIncident, candidate: EvidenceCandidate) -> VerifiedEvidence:
    key = str(candidate.provenance.get("metric_key") or f"{candidate.entity}.{candidate.signal}")
    series = incident.metric_series.get(key)
    if not series:
        return VerifiedEvidence.from_candidate(
            candidate,
            observed={},
            verification_rule="metric_series_lookup",
            verified=False,
            rejection_reason=f"metric series not found: {key}",
        )
    times = [float(t) for t, _ in series]
    start, end = coerce_window_to_series(candidate.time_range, times)
    points = _window_points(series, start, end)
    if not points:
        return VerifiedEvidence.from_candidate(
            candidate,
            observed={"series": key},
            verification_rule="metric_window_lookup",
            verified=False,
            rejection_reason=f"metric points not found in window: {key}",
        )
    values = [v for _, v in points]
    baseline = _baseline_points(series, start)
    observed = {
        "series": key,
        "matching_points": len(points),
        "max_value": max(values),
        "mean_value": mean(values),
        "baseline_mean": mean(baseline) if baseline else None,
    }
    provenance = dict(candidate.provenance)
    provenance["candidate_source"] = provenance.get("source", "")
    provenance["source"] = "metric_series"
    return VerifiedEvidence.from_candidate(
        EvidenceCandidate(
            **{
                **candidate.to_dict(),
                "provenance": provenance,
            }
        ),
        observed=observed,
        verification_rule="metric_window_lookup",
        verified=True,
    )


def _verify_alert(incident: RawIncident, candidate: EvidenceCandidate) -> VerifiedEvidence:
    alert_entity = incident.metadata.get("alert_entity") or {}
    raw_entity = alert_entity.get("entity_name") if isinstance(alert_entity, dict) else ""
    if candidate.entity and raw_entity and str(candidate.entity) != str(raw_entity):
        return VerifiedEvidence.from_candidate(
            candidate,
            observed={"entity_name": raw_entity},
            verification_rule="task_alert_contract_lookup",
            verified=False,
            rejection_reason=f"alert entity mismatch: {candidate.entity} != {raw_entity}",
        )
    provenance = dict(candidate.provenance)
    provenance["candidate_source"] = provenance.get("source", "")
    provenance["source"] = "task_alert_contract"
    observed = {
        "entity_name": raw_entity or candidate.entity,
        "alert_title": incident.metadata.get("alert_title", ""),
        "alert_window": incident.metadata.get("alert_window", {}),
        "matching_rows": 1,
    }
    return VerifiedEvidence.from_candidate(
        EvidenceCandidate(
            **{
                **candidate.to_dict(),
                "provenance": provenance,
            }
        ),
        observed=observed,
        verification_rule="task_alert_contract_lookup",
        verified=True,
    )


def _verify_trace(incident: RawIncident, candidate: EvidenceCandidate) -> VerifiedEvidence:
    trace_path = incident.file_index.get("traces")
    if not trace_path or not trace_path.exists():
        return VerifiedEvidence.from_candidate(
            candidate,
            observed={},
            verification_rule="trace_file_lookup",
            verified=False,
            rejection_reason="trace file not found",
        )
    try:
        import pandas as pd  # type: ignore

        frame = pd.read_parquet(
            trace_path,
            columns=["serviceName", "spanName", "statusCode", "statusMessage", "duration", "startTime"],
        )
    except Exception as exc:
        return VerifiedEvidence.from_candidate(
            candidate,
            observed={},
            verification_rule="trace_file_lookup",
            verified=False,
            rejection_reason=f"trace file unreadable: {exc}",
        )

    frame = frame.copy()
    frame["startTime"] = pd.to_numeric(frame["startTime"], errors="coerce")
    frame["duration"] = pd.to_numeric(frame.get("duration"), errors="coerce")
    frame = frame.dropna(subset=["startTime"])
    times = [float(t) for t in frame["startTime"].tolist()]
    start, end = coerce_window_to_series(candidate.time_range, times)
    if start is not None:
        frame = frame[frame["startTime"] >= start]
    if end is not None:
        frame = frame[frame["startTime"] <= end]
    messages = frame.get("statusMessage", "").fillna("").astype(str).str.strip()
    frame = frame[messages != ""]
    if frame.empty:
        return VerifiedEvidence.from_candidate(
            candidate,
            observed={},
            verification_rule="trace_window_lookup",
            verified=False,
            rejection_reason="trace error spans not found in window",
        )

    target = canonical_entity(candidate.entity)
    matches = []
    for row in frame.itertuples(index=False):
        source = canonical_entity(getattr(row, "serviceName", ""))
        span_target = canonical_entity(getattr(row, "spanName", ""))
        message_targets = _services_from_text(str(getattr(row, "statusMessage", "") or ""))
        if target and (target in {source, span_target} or target in message_targets):
            matches.append(row)
    if not matches:
        return VerifiedEvidence.from_candidate(
            candidate,
            observed={"candidate_entity": target},
            verification_rule="trace_window_lookup",
            verified=False,
            rejection_reason=f"trace error spans not found for entity: {candidate.entity}",
        )

    durations = []
    for row in matches:
        try:
            durations.append(float(getattr(row, "duration", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
    provenance = dict(candidate.provenance)
    provenance["candidate_source"] = provenance.get("source", "")
    provenance["source"] = "trace_error_spans"
    return VerifiedEvidence.from_candidate(
        EvidenceCandidate(
            **{
                **candidate.to_dict(),
                "provenance": provenance,
            }
        ),
        observed={
            "matching_rows": len(matches),
            "max_duration": max(durations) if durations else None,
        },
        verification_rule="trace_window_lookup",
        verified=True,
    )


def _node_from_event_row(row: dict, payload: dict) -> str:
    values = [
        row.get("hostname"),
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


def _event_timestamp_near_window(timestamp: str, window: dict, *, tolerance_seconds: int = 60) -> bool:
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


def _verify_event(incident: RawIncident, candidate: EvidenceCandidate) -> VerifiedEvidence:
    events_path = incident.file_index.get("events")
    if not events_path or not events_path.exists():
        return VerifiedEvidence.from_candidate(
            candidate,
            observed={},
            verification_rule="kubernetes_event_lookup",
            verified=False,
            rejection_reason="events file not found",
        )
    try:
        import pandas as pd  # type: ignore

        frame = pd.read_parquet(events_path, columns=["eventId", "hostname", "level"])
    except Exception as exc:
        return VerifiedEvidence.from_candidate(
            candidate,
            observed={},
            verification_rule="kubernetes_event_lookup",
            verified=False,
            rejection_reason=f"events file unreadable: {exc}",
        )

    target = canonical_entity(candidate.entity)
    expected_reason = str(candidate.claim.get("reason") or "")
    matches = []
    for row in frame.to_dict("records"):
        try:
            payload = json.loads(str(row.get("eventId") or "{}"))
        except Exception:
            continue
        reason = str(payload.get("reason") or "")
        message = str(payload.get("message") or "")
        if expected_reason and reason != expected_reason:
            continue
        if reason != "Preempted" and "preempted" not in message.lower():
            continue
        timestamp = str(payload.get("lastTimestamp") or payload.get("eventTime") or payload.get("firstTimestamp") or "")
        if not _event_timestamp_near_window(timestamp, candidate.time_range):
            continue
        node = canonical_entity(_node_from_event_row(row, payload))
        if node == target:
            matches.append((payload, timestamp))
    if not matches:
        return VerifiedEvidence.from_candidate(
            candidate,
            observed={"candidate_entity": target},
            verification_rule="kubernetes_event_lookup",
            verified=False,
            rejection_reason=f"kubernetes event not found for node: {candidate.entity}",
        )

    provenance = dict(candidate.provenance)
    provenance["candidate_source"] = provenance.get("source", "")
    provenance["source"] = "kubernetes_event"
    return VerifiedEvidence.from_candidate(
        EvidenceCandidate(
            **{
                **candidate.to_dict(),
                "provenance": provenance,
            }
        ),
        observed={
            "matching_rows": len(matches),
            "reasons": sorted({str(payload.get("reason") or "") for payload, _ in matches}),
            "event_timestamps": [timestamp for _, timestamp in matches[:5]],
        },
        verification_rule="kubernetes_event_lookup",
        verified=True,
    )


def _verify_topology(incident: RawIncident, candidate: EvidenceCandidate) -> VerifiedEvidence:
    src = str(candidate.claim.get("src") or candidate.entity)
    dst = str(candidate.claim.get("dst") or "")
    if (src, dst) not in incident.topology_edges:
        return VerifiedEvidence.from_candidate(
            candidate,
            observed={"src": src, "dst": dst},
            verification_rule="topology_edge_lookup",
            verified=False,
            rejection_reason=f"topology edge not found: {src}->{dst}",
        )
    return VerifiedEvidence.from_candidate(
        candidate,
        observed={"src": src, "dst": dst, "matching_edges": 1},
        verification_rule="topology_edge_lookup",
        verified=True,
    )


def verify_candidates(incident: RawIncident, candidates: List[EvidenceCandidate]) -> List[VerifiedEvidence]:
    return [verify_candidate(incident, candidate) for candidate in candidates]
