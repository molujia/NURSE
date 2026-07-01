"""Auditable alarm simulation and fixed-window anomaly scoring for Nurse."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List, Optional, Tuple, Union

from .schemas import AlarmArtifact


Point = Tuple[Union[int, float], float]


def _scaled_radius(timestamp: int | float, radius_seconds: int | float) -> int | float:
    try:
        magnitude = abs(float(timestamp))
        radius = float(radius_seconds)
    except (TypeError, ValueError):
        return radius_seconds
    if magnitude > 1e14:
        return radius * 1_000_000
    if magnitude > 1e11:
        return radius * 1_000
    return radius_seconds


@dataclass
class RobustAlarmDetector:
    baseline_points: int = 30
    threshold: float = 3.5
    window_radius: int = 60

    def score(self, baseline: List[float], value: float) -> float:
        baseline = [float(item) for item in baseline if math.isfinite(float(item))]
        if not baseline or not math.isfinite(float(value)):
            return 0.0
        center = median(baseline)
        deviations = [abs(x - center) for x in baseline]
        mad = median(deviations)
        if mad > 1e-12:
            return 0.6745 * abs(value - center) / mad
        std = pstdev(baseline) if len(baseline) > 1 else 0.0
        if std > 1e-12:
            return abs(value - center) / std
        return float("inf") if abs(value - center) > 1e-12 else 0.0

    def detect(self, series: Iterable[Point]) -> Optional[Tuple[Point, float, float]]:
        points = []
        for timestamp, raw_value in series:
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                points.append((timestamp, value))
        points = sorted(points, key=lambda item: item[0])
        if len(points) <= self.baseline_points:
            return None
        baseline = [v for _, v in points[: self.baseline_points]]
        baseline_center = median(baseline)
        best: Optional[Tuple[Point, float, float, float]] = None
        for timestamp, value in points[self.baseline_points :]:
            anomaly_score = self.score(baseline, value)
            if anomaly_score >= self.threshold:
                delta = abs(value - baseline_center)
                candidate = ((timestamp, value), anomaly_score, baseline_center, delta)
                if best is None or (anomaly_score, delta) > (best[1], best[3]):
                    best = candidate
        if best is None:
            return None
        point, score, center, _ = best
        return point, score, center


@dataclass
class FixedWindowScoreConfig:
    fault_window_seconds: int = 480
    normal_window_seconds: int = 480
    min_score: float = 1.0
    iqr_factor: float = 1.5
    max_signals: int = 20


def _is_monotonic(values: List[float]) -> bool:
    if len(values) < 3:
        return False
    diffs = [b - a for a, b in zip(values, values[1:])]
    return all(diff >= 0 for diff in diffs) or all(diff <= 0 for diff in diffs)


def _percentile(values: List[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percent / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _window_values(points: List[Point], start: float, end: float, *, include_end: bool) -> List[float]:
    values: List[float] = []
    for timestamp, raw_value in points:
        try:
            ts = float(timestamp)
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ts) or not math.isfinite(value):
            continue
        inside = start <= ts <= end if include_end else start <= ts < end
        if inside:
            values.append(value)
    return values


def _ks_statistic(left: List[float], right: List[float]) -> float:
    if not left or not right:
        return 0.0
    a = sorted(left)
    b = sorted(right)
    i = j = 0
    best = 0.0
    for value in sorted(set(a + b)):
        while i < len(a) and a[i] <= value:
            i += 1
        while j < len(b) and b[j] <= value:
            j += 1
        best = max(best, abs(i / len(a) - j / len(b)))
    scale = math.sqrt((len(a) * len(b)) / max(len(a) + len(b), 1))
    return best * scale


def score_fixed_window_metric(
    series: Iterable[Point],
    *,
    start: float,
    end: float,
    normal_start: float,
    normal_end: float,
) -> Dict[str, float]:
    points = sorted(list(series), key=lambda item: item[0])
    all_values: List[float] = []
    for _, raw_value in points:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            all_values.append(value)
    if _is_monotonic(all_values):
        return {"residual": 0.0, "shewhart": 0.0, "ks": 0.0, "score": 0.0}

    normal = _window_values(points, normal_start, normal_end, include_end=False)
    fault = _window_values(points, start, end, include_end=True)
    if not normal or not fault:
        return {"residual": 0.0, "shewhart": 0.0, "ks": 0.0, "score": 0.0}

    normal_center = mean(normal)
    normal_std = pstdev(normal) if len(normal) > 1 else 0.0
    value_range = max(all_values) - min(all_values) if all_values else 0.0
    scale = max(normal_std, value_range * 0.04, 1e-6)
    residual = abs(mean(fault) - normal_center) / scale
    shewhart = max(abs(value - normal_center) for value in fault) / scale
    ks = _ks_statistic(normal, fault)
    score = max(residual, shewhart, ks)
    return {
        "residual": float(residual),
        "shewhart": float(shewhart),
        "ks": float(ks),
        "score": float(score),
        "normal_baseline": float(normal_center),
        "fault_mean": float(mean(fault)),
    }


def filter_fixed_window_scores(
    scored_items: Dict[Tuple[str, str], float],
    *,
    min_score: float = 1.0,
    iqr_factor: float = 1.5,
) -> set[Tuple[str, str]]:
    if not scored_items:
        return set()
    scores = [float(score) for score in scored_items.values() if math.isfinite(float(score))]
    if not scores or max(scores) < min_score:
        return set()
    q1 = _percentile(scores, 25)
    q3 = _percentile(scores, 75)
    iqr = q3 - q1
    if iqr > 0:
        threshold = q3 + iqr_factor * iqr
    else:
        center = mean(scores)
        sigma = pstdev(scores) if len(scores) > 1 else 0.0
        threshold = max(center + 2 * sigma, min_score)
    return {
        key
        for key, score in scored_items.items()
        if math.isfinite(float(score)) and float(score) >= threshold and float(score) >= min_score
    }


def fixed_window_metric_signals(
    metric_series: Dict[str, Iterable[Point]],
    *,
    start: float,
    end: float,
    normal_start: float,
    normal_end: float,
    config: FixedWindowScoreConfig | None = None,
) -> List[Dict[str, Union[str, float, bool]]]:
    config = config or FixedWindowScoreConfig()
    components: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, series in sorted(metric_series.items()):
        entity, metric = key.rsplit(".", 1) if "." in key else (key, "value")
        scores = score_fixed_window_metric(
            series,
            start=start,
            end=end,
            normal_start=normal_start,
            normal_end=normal_end,
        )
        components[(entity, metric)] = scores

    scalar_scores = {key: value.get("score", 0.0) for key, value in components.items()}
    selected = filter_fixed_window_scores(
        scalar_scores,
        min_score=config.min_score,
        iqr_factor=config.iqr_factor,
    )
    if not selected and scalar_scores:
        best_key, best_score = max(scalar_scores.items(), key=lambda item: item[1])
        if best_score > 0:
            selected = {best_key}

    ordered_selected = sorted(
        selected,
        key=lambda item: (
            -float(components[item].get("score", 0.0)),
            item[0],
            item[1],
        ),
    )
    signals: List[Dict[str, Union[str, float, bool]]] = []
    for entity, metric in ordered_selected:
        scores = components[(entity, metric)]
        signals.append(
            {
                "entity": entity,
                "metric": metric,
                "score": float(scores.get("score", 0.0)),
                "residual_score": float(scores.get("residual", 0.0)),
                "shewhart_score": float(scores.get("shewhart", 0.0)),
                "ks_score": float(scores.get("ks", 0.0)),
                "baseline": float(scores.get("normal_baseline", 0.0)),
                "value": float(scores.get("fault_mean", 0.0)),
                "filtered_by_iqr": (entity, metric) in selected,
            }
        )
    # Keep the final signal order deterministic when scores tie.
    signals.sort(key=lambda item: (-float(item.get("score") or 0.0), str(item.get("entity") or ""), str(item.get("metric") or "")))
    return signals[: config.max_signals]


def detect_alarm_from_series(
    *,
    case_id: str,
    dataset: str,
    entity: str,
    metric: str,
    series: Iterable[Point],
    detector: RobustAlarmDetector | None = None,
) -> Optional[AlarmArtifact]:
    detector = detector or RobustAlarmDetector()
    detected = detector.detect(series)
    if detected is None:
        return None
    (timestamp, value), score, baseline = detected
    try:
        radius = _scaled_radius(timestamp, detector.window_radius)
        start = timestamp - radius
        end = timestamp + radius
    except TypeError:
        start = timestamp
        end = timestamp
    return AlarmArtifact(
        case_id=case_id,
        dataset=dataset,
        trigger_time=timestamp,
        window={"start": start, "end": end},
        alert_entity=entity,
        signals=[
            {
                "entity": entity,
                "metric": metric,
                "value": value,
                "baseline": baseline,
                "score": score,
            }
        ],
        provenance={
            "source": "raw_metric_series",
            "baseline_points": detector.baseline_points,
            "threshold": detector.threshold,
        },
        detector="robust_mad_or_sigma",
    )
