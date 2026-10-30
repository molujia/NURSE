"""Independent implementation of the fixed paper V1 deviation equations."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from numbers import Real
from typing import Literal

from nurse_rcl.schemas import StrictFrozenModel

MINIMUM_POINTS = 5
MAD_NORMALIZER = 0.6745
EPSILON = 1e-9
EVIDENCE_THRESHOLD = 3.0

ExclusionReason = Literal[
    "insufficient_baseline_points",
    "insufficient_fault_points",
    "below_threshold",
]


class MetricDeviation(StrictFrozenModel):
    """Every input count and numeric term needed to audit one metric series."""

    baseline_input_count: int
    fault_input_count: int
    baseline_valid_count: int
    fault_valid_count: int
    baseline_dropped_count: int
    fault_dropped_count: int
    minimum_points: int = MINIMUM_POINTS
    mad_normalizer: float = MAD_NORMALIZER
    epsilon: float = EPSILON
    threshold: float = EVIDENCE_THRESHOLD
    eligible: bool
    retained: bool
    exclusion_reason: ExclusionReason | None
    baseline_mean: float | None = None
    fault_mean: float | None = None
    baseline_median: float | None = None
    baseline_mad: float | None = None
    scale: float | None = None
    max_fault_distance: float | None = None
    ks_distance: float | None = None
    ks_multiplier: float | None = None
    z_residual: float | None = None
    z_shift: float | None = None
    z_ks: float | None = None
    max_deviation: float | None = None
    anomaly_score: float | None = None


def bounded_anomaly_score(max_deviation: float) -> float:
    """Apply ``100 * (1 - exp(-z / 3))`` with explicit numeric bounds."""

    value = float(max_deviation)
    if math.isnan(value) or value < 0.0:
        raise ValueError("max deviation must be non-negative and not NaN")
    if math.isinf(value):
        return 100.0
    score = -100.0 * math.expm1(-value / 3.0)
    return min(100.0, max(0.0, score))


def compute_metric_deviation(
    *,
    baseline: Iterable[float | int | None],
    fault: Iterable[float | int | None],
) -> MetricDeviation:
    """Compute the fixed residual, shift, scaled-KS, and anomaly terms."""

    baseline_input = tuple(baseline)
    fault_input = tuple(fault)
    baseline_values = _finite_values(baseline_input)
    fault_values = _finite_values(fault_input)
    counts = {
        "baseline_input_count": len(baseline_input),
        "fault_input_count": len(fault_input),
        "baseline_valid_count": len(baseline_values),
        "fault_valid_count": len(fault_values),
        "baseline_dropped_count": len(baseline_input) - len(baseline_values),
        "fault_dropped_count": len(fault_input) - len(fault_values),
    }
    if len(baseline_values) < MINIMUM_POINTS:
        return MetricDeviation(
            **counts,
            eligible=False,
            retained=False,
            exclusion_reason="insufficient_baseline_points",
        )
    if len(fault_values) < MINIMUM_POINTS:
        return MetricDeviation(
            **counts,
            eligible=False,
            retained=False,
            exclusion_reason="insufficient_fault_points",
        )

    baseline_mean = math.fsum(baseline_values) / len(baseline_values)
    fault_mean = math.fsum(fault_values) / len(fault_values)
    baseline_median = float(statistics.median(baseline_values))
    baseline_mad = float(
        statistics.median(abs(value - baseline_median) for value in baseline_values)
    )
    scale = baseline_mad / MAD_NORMALIZER + EPSILON
    max_fault_distance = max(abs(value - baseline_mean) for value in fault_values)
    ks_distance = _empirical_ks_distance(baseline_values, fault_values)
    ks_multiplier = math.sqrt(
        len(baseline_values)
        * len(fault_values)
        / (len(baseline_values) + len(fault_values))
    )
    z_residual = abs(fault_mean - baseline_mean) / scale
    z_shift = max_fault_distance / scale
    z_ks = ks_multiplier * ks_distance
    max_deviation = max(z_residual, z_shift, z_ks)
    retained = max_deviation >= EVIDENCE_THRESHOLD

    return MetricDeviation(
        **counts,
        eligible=True,
        retained=retained,
        exclusion_reason=None if retained else "below_threshold",
        baseline_mean=baseline_mean,
        fault_mean=fault_mean,
        baseline_median=baseline_median,
        baseline_mad=baseline_mad,
        scale=scale,
        max_fault_distance=max_fault_distance,
        ks_distance=ks_distance,
        ks_multiplier=ks_multiplier,
        z_residual=z_residual,
        z_shift=z_shift,
        z_ks=z_ks,
        max_deviation=max_deviation,
        anomaly_score=bounded_anomaly_score(max_deviation),
    )


def _finite_values(values: tuple[float | int | None, ...]) -> tuple[float, ...]:
    output: list[float] = []
    for value in values:
        if value is None or isinstance(value, bool) or not isinstance(value, Real):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            output.append(numeric)
    return tuple(output)


def _empirical_ks_distance(
    baseline: tuple[float, ...],
    fault: tuple[float, ...],
) -> float:
    baseline_sorted = sorted(baseline)
    fault_sorted = sorted(fault)
    support = sorted(set(baseline_sorted) | set(fault_sorted))
    baseline_index = 0
    fault_index = 0
    maximum = 0.0
    for value in support:
        while baseline_index < len(baseline_sorted) and baseline_sorted[baseline_index] <= value:
            baseline_index += 1
        while fault_index < len(fault_sorted) and fault_sorted[fault_index] <= value:
            fault_index += 1
        baseline_cdf = baseline_index / len(baseline_sorted)
        fault_cdf = fault_index / len(fault_sorted)
        maximum = max(maximum, abs(baseline_cdf - fault_cdf))
    return maximum
