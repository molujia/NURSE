"""Sliding-window 3-sigma metric evidence scorer.

This module is intentionally label-free. It only consumes a metric time series and
normal/fault window bounds, then returns auditable scoring components that can be
used by the NURSE metric evidence compiler.
"""

from __future__ import annotations

import math
from statistics import mean, median, pstdev
from typing import Sequence, Tuple


Series = Sequence[Tuple[float, float]]


def _finite_points(series: Series) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for timestamp, value in series:
        try:
            ts = float(timestamp)
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(ts) and math.isfinite(val):
            points.append((ts, val))
    return sorted(points, key=lambda item: item[0])


def _scale(values: Sequence[float], center: float) -> float:
    if len(values) >= 2:
        sigma = float(pstdev(values))
    else:
        sigma = 0.0
    floor = max(abs(center) * 0.005, 1e-9)
    return max(sigma, floor)


def score_metric(
    series: Series,
    *,
    start: float,
    end: float,
    normal_start: float | None = None,
    normal_end: float | None = None,
    window_points: int = 20,
    threshold_sigma: float = 3.0,
) -> dict[str, float]:
    """Return a sliding-window z score and audit components.

    For each point inside the incident window, the scorer compares it with the
    preceding ``window_points`` values. If too few preceding values exist, it
    falls back to values in the provided normal window. Scores below the
    configured 3-sigma threshold are reported as zero so this backend behaves as
    an AD tool rather than a continuous weak-prior scorer.
    """

    points = _finite_points(series)
    if not points:
        return {
            "score": 0.0,
            "raw_max_z": 0.0,
            "threshold_sigma": float(threshold_sigma),
            "normal_baseline": 0.0,
            "fault_mean": 0.0,
            "window_points": float(window_points),
        }

    if normal_start is None:
        normal_start = min(ts for ts, _ in points)
    if normal_end is None:
        normal_end = float(start)
    if normal_start > normal_end:
        normal_start, normal_end = normal_end, normal_start

    normal_values = [value for ts, value in points if float(normal_start) <= ts <= float(normal_end)]
    incident_points = [(ts, value) for ts, value in points if float(start) <= ts <= float(end)]
    if not incident_points:
        incident_points = [(ts, value) for ts, value in points if ts >= float(start)]
    if not incident_points:
        incident_points = points[-1:]

    max_z = 0.0
    max_value = float(incident_points[0][1])
    baseline_at_max = float(mean(normal_values)) if normal_values else float(points[0][1])
    for ts, value in incident_points:
        history = [v for prev_ts, v in points if prev_ts < ts]
        if window_points > 0:
            history = history[-int(window_points):]
        if len(history) < 2 and normal_values:
            history = normal_values
        if not history:
            continue
        center = float(mean(history))
        z = abs(float(value) - center) / _scale(history, center)
        if z > max_z:
            max_z = float(z)
            max_value = float(value)
            baseline_at_max = center

    fault_values = [value for _, value in incident_points]
    score = max_z if max_z >= float(threshold_sigma) else 0.0
    return {
        "score": float(score),
        "raw_max_z": float(max_z),
        "threshold_sigma": float(threshold_sigma),
        "normal_baseline": float(baseline_at_max if normal_values or baseline_at_max else 0.0),
        "fault_mean": float(mean(fault_values)) if fault_values else float(max_value),
        "anomalous_peak": float(max_value),
        "window_points": float(window_points),
    }
