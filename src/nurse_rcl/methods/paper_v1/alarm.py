"""Deterministic paper V1 alarm selection over metric deviation evidence."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from itertools import groupby

from pydantic import Field

from nurse_rcl.methods.paper_v1.deviation import MetricDeviation, compute_metric_deviation
from nurse_rcl.schemas import StrictFrozenModel

_IDENTIFIER_SEPARATOR = re.compile(r"[-_\s]+")


class TimedValue(StrictFrozenModel):
    timestamp: int
    value: float | int | None


class SignalSeries(StrictFrozenModel):
    service: str = Field(min_length=1)
    signal: str = Field(min_length=1)
    baseline: tuple[float | int | None, ...]
    fault: tuple[TimedValue, ...]


class AlarmDecision(StrictFrozenModel):
    service: str = Field(min_length=1)
    signal: str = Field(min_length=1)
    crossing_timestamp: int | None
    fallback: bool
    max_deviation: float = Field(ge=0.0)
    anomaly_score: float = Field(ge=0.0, le=100.0)
    evaluated_fault_points: int = Field(ge=5)
    deviation: MetricDeviation


@dataclass(frozen=True)
class _Candidate:
    service: str
    signal: str
    timestamp: int | None
    deviation: MetricDeviation


def canonicalize_service_name(raw: str) -> str:
    """Canonicalize a service identifier for deterministic comparisons."""

    return _canonicalize_identifier(raw, kind="service")


def canonicalize_signal_name(raw: str) -> str:
    """Canonicalize a signal identifier for deterministic comparisons."""

    return _canonicalize_identifier(raw, kind="signal")


def detect_alarm(series_collection: list[SignalSeries]) -> AlarmDecision:
    """Select the first crossing, or the strongest eligible full-window fallback."""

    if not series_collection:
        raise ValueError("alarm detection requires at least one signal series")
    crossings: list[_Candidate] = []
    fallbacks: list[_Candidate] = []
    canonical_pairs: set[tuple[str, str]] = set()

    for series in series_collection:
        service = canonicalize_service_name(series.service)
        signal = canonicalize_signal_name(series.signal)
        pair = (service, signal)
        if pair in canonical_pairs:
            raise ValueError(f"duplicate canonical signal series: {service}/{signal}")
        canonical_pairs.add(pair)

        crossing = _first_crossing(series, service=service, signal=signal)
        if crossing is not None:
            crossings.append(crossing)
        final = compute_metric_deviation(
            baseline=series.baseline,
            fault=(sample.value for sample in series.fault),
        )
        if final.eligible:
            fallbacks.append(
                _Candidate(
                    service=service,
                    signal=signal,
                    timestamp=None,
                    deviation=final,
                )
            )

    if crossings:
        selected = min(
            crossings,
            key=lambda candidate: (
                _required_timestamp(candidate),
                -_required_max_deviation(candidate.deviation),
                candidate.service,
                candidate.signal,
            ),
        )
        return _decision(selected, fallback=False)
    if not fallbacks:
        raise ValueError("no signal series has five valid points in both windows")
    selected = min(
        fallbacks,
        key=lambda candidate: (
            -_required_max_deviation(candidate.deviation),
            candidate.service,
            candidate.signal,
        ),
    )
    return _decision(selected, fallback=True)


def _first_crossing(
    series: SignalSeries,
    *,
    service: str,
    signal: str,
) -> _Candidate | None:
    prefix: list[float | int | None] = []
    ordered = sorted(enumerate(series.fault), key=lambda item: (item[1].timestamp, item[0]))
    for timestamp, group in groupby(ordered, key=lambda item: item[1].timestamp):
        prefix.extend(item[1].value for item in group)
        deviation = compute_metric_deviation(baseline=series.baseline, fault=prefix)
        if deviation.retained:
            return _Candidate(
                service=service,
                signal=signal,
                timestamp=timestamp,
                deviation=deviation,
            )
    return None


def _decision(candidate: _Candidate, *, fallback: bool) -> AlarmDecision:
    deviation = candidate.deviation
    return AlarmDecision(
        service=candidate.service,
        signal=candidate.signal,
        crossing_timestamp=None if fallback else candidate.timestamp,
        fallback=fallback,
        max_deviation=_required_max_deviation(deviation),
        anomaly_score=_required_anomaly_score(deviation),
        evaluated_fault_points=deviation.fault_valid_count,
        deviation=deviation,
    )


def _canonicalize_identifier(raw: str, *, kind: str) -> str:
    normalized = unicodedata.normalize("NFKC", raw).strip().casefold()
    canonical = _IDENTIFIER_SEPARATOR.sub("-", normalized).strip("-")
    if not canonical:
        raise ValueError(f"{kind} name must contain a non-separator character")
    return canonical


def _required_timestamp(candidate: _Candidate) -> int:
    if candidate.timestamp is None:
        raise ValueError("crossing candidate has no timestamp")
    return candidate.timestamp


def _required_max_deviation(deviation: MetricDeviation) -> float:
    if deviation.max_deviation is None:
        raise ValueError("eligible deviation has no maximum")
    return deviation.max_deviation


def _required_anomaly_score(deviation: MetricDeviation) -> float:
    if deviation.anomaly_score is None:
        raise ValueError("eligible deviation has no anomaly score")
    return deviation.anomaly_score
