"""Measure one requested series only after replaying all independently pinned sources."""

from __future__ import annotations

import math
from typing import Protocol, cast

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.observation_schemas import CaseTimeContract
from nurse_rcl.v2.rule_hypotheses import RuleHypothesisEnvelope

from .contracts import (
    Direction,
    DynamicVerificationPolicy,
    DynamicVerificationRequest,
    EpisodeBoundary,
    PointState,
    SignalEpisode,
    SignalEpisodeMeasurement,
    SignalEpisodePoint,
    TruthValue,
    VerificationSourceCatalog,
    VerificationSourceRecord,
)
from .requests import build_dynamic_verification_requests
from .safety import canonical, plain, require_pin, require_unchanged, snapshot
from .source_catalog import require_verification_source_catalog
from .source_store import SourceBlockStore, require_store_unchanged, store_snapshot


class _EpisodeIdentity(Protocol):
    """Only numeric episode identity; it grants no request or source authority."""

    @property
    def canonical_series_key(self) -> str: ...

    @property
    def request_sha256(self) -> str: ...


def _classify_point(
    value: float, median: float | None, scale: float | None, threshold: float
) -> tuple[float | None, PointState]:
    if median is None or scale is None or scale <= 0.0:
        return None, "unknown"
    z = (value - median) / scale
    if not math.isfinite(z):
        return None, "unknown"
    if z >= threshold:
        return z, "up"
    if z <= -threshold:
        return z, "down"
    return (0.0 if z == 0.0 else z), "within_band"


def _episode(
    *,
    points: tuple[SignalEpisodePoint, ...],
    previous: SignalEpisodePoint | None,
    request: _EpisodeIdentity,
    policy: DynamicVerificationPolicy,
    cadence: int,
    start: EpisodeBoundary,
    end: EpisodeBoundary,
) -> SignalEpisode:
    intervals = tuple(
        point.source_interval for point in points if point.source_interval is not None
    )
    if intervals:
        if len(intervals) != len(points):
            raise ValueError("episode cannot mix instantaneous and interval support")
        ordered = sorted(intervals, key=lambda interval: interval.start_unix_ns)
        first, covered_end = ordered[0].start_unix_ns, ordered[0].end_unix_ns
        for interval in ordered[1:]:
            if interval.start_unix_ns > covered_end:
                raise ValueError("episode source coverage must be contiguous")
            covered_end = max(covered_end, interval.end_unix_ns)
        span = covered_end - first
    else:
        span = points[-1].timestamp_unix_ns - points[0].timestamp_unix_ns
    draft = SignalEpisode(
        episode_id="episode_" + "0" * 64,
        canonical_series_key=request.canonical_series_key,
        request_sha256=request.request_sha256,
        policy_sha256=policy.policy_sha256,
        sample_indices=tuple(point.sample_index for point in points),
        direction=cast(Direction, points[0].state),
        support_span_ns=span,
        qualifying=(
            len(points) >= policy.minimum_consecutive_samples
            and span >= policy.minimum_span_cadences * cadence
        ),
        start_reason=start,
        end_reason=end,
        onset_lower_unix_ns=(
            (
                previous.timestamp_unix_ns
                if previous.source_interval is None
                else previous.source_interval.start_unix_ns
            )
            if previous is not None and start == "within_band"
            else None
        ),
        onset_upper_unix_ns=(
            points[0].timestamp_unix_ns
            if points[0].source_interval is None
            else points[0].source_interval.end_unix_ns
        ),
        onset_reason=(
            "observed_bracket"
            if previous is not None and start == "within_band"
            else "window_start"
            if previous is None
            else "gap"
            if start in {"sampling_gap", "source_interval_gap", "unknown_point"}
            else "no_normal_bracket"
        ),
        left_censored=previous is None or start != "within_band",
        right_censored=end == "window_end",
    )
    document = cast(dict[str, object], plain(draft))
    document.pop("episode_id")
    return draft.model_copy(
        update={"episode_id": "episode_" + sha256_hex(canonical_json_bytes(document))}
    )


def _gap_reasons(
    previous: SignalEpisodePoint,
    current: SignalEpisodePoint,
    policy: DynamicVerificationPolicy,
    cadence: int,
) -> tuple[EpisodeBoundary, ...]:
    reasons: list[EpisodeBoundary] = []
    gap = current.timestamp_unix_ns - previous.timestamp_unix_ns
    if policy.maximum_gap_denominator * gap > policy.maximum_gap_numerator * cadence:
        reasons.append("sampling_gap")
    if (
        previous.source_interval is not None
        and current.source_interval is not None
        and current.source_interval.start_unix_ns > previous.source_interval.end_unix_ns
    ):
        reasons.append("source_interval_gap")
    return tuple(reasons)


def _episodes(
    *,
    points: tuple[SignalEpisodePoint, ...],
    request: _EpisodeIdentity,
    policy: DynamicVerificationPolicy,
    cadence: int,
) -> tuple[SignalEpisode, ...]:
    result = []
    current: list[SignalEpisodePoint] = []
    start: EpisodeBoundary = "window_start"
    for point in points:
        previous = points[point.sample_index - 1] if point.sample_index else None
        gaps = () if previous is None else _gap_reasons(previous, point, policy, cadence)
        gap = gaps[0] if gaps else None
        if current and (gap is not None or point.state != current[-1].state):
            end: EpisodeBoundary = gap or (
                "within_band"
                if point.state == "within_band"
                else "unknown_point"
                if point.state == "unknown"
                else "direction_reversal"
            )
            result.append(
                _episode(
                    points=tuple(current),
                    previous=points[current[0].sample_index - 1]
                    if current[0].sample_index
                    else None,
                    request=request,
                    policy=policy,
                    cadence=cadence,
                    start=start,
                    end=end,
                )
            )
            current = []
        if point.state in ("up", "down"):
            if not current:
                start = gap or (
                    "window_start"
                    if previous is None
                    else "within_band"
                    if previous.state == "within_band"
                    else "unknown_point"
                    if previous.state == "unknown"
                    else "direction_reversal"
                )
            current.append(point)
    if current:
        result.append(
            _episode(
                points=tuple(current),
                previous=points[current[0].sample_index - 1] if current[0].sample_index else None,
                request=request,
                policy=policy,
                cadence=cadence,
                start=start,
                end="window_end",
            )
        )
    return tuple(result)


def measure_signal_episodes(
    *,
    envelope: RuleHypothesisEnvelope,
    time_contract: CaseTimeContract,
    source_records: tuple[VerificationSourceRecord, ...],
    expected_source_records_sha256: str,
    request: DynamicVerificationRequest,
    expected_request_sha256: str,
    catalog: VerificationSourceCatalog,
    expected_catalog_sha256: str,
    policy: DynamicVerificationPolicy,
    expected_policy_sha256: str,
    store: SourceBlockStore,
) -> SignalEpisodeMeasurement:
    roots = (
        RuleHypothesisEnvelope,
        CaseTimeContract,
        VerificationSourceRecord,
        DynamicVerificationRequest,
        VerificationSourceCatalog,
        DynamicVerificationPolicy,
    )
    if (
        type(envelope) is not RuleHypothesisEnvelope
        or type(time_contract) is not CaseTimeContract
        or type(source_records) is not tuple
        or any(type(row) is not VerificationSourceRecord for row in source_records)
        or type(request) is not DynamicVerificationRequest
        or type(catalog) is not VerificationSourceCatalog
        or type(policy) is not DynamicVerificationPolicy
    ):
        raise ValueError("dynamic measurement requires exact frozen roots")
    for pin in (
        expected_source_records_sha256,
        expected_request_sha256,
        expected_catalog_sha256,
        expected_policy_sha256,
    ):
        require_pin(pin)
    originals = (envelope, time_contract, source_records, request, catalog, policy)
    before = snapshot(originals, roots)
    state = store_snapshot(store)
    try:
        requests = build_dynamic_verification_requests(
            envelope=envelope,
            time_contract=time_contract,
            catalog=catalog,
            expected_catalog_sha256=expected_catalog_sha256,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
        )
        selected = next(
            (row for row in requests if row.request_sha256 == expected_request_sha256), None
        )
        if selected is None or canonical(selected, (DynamicVerificationRequest,)) != canonical(
            request, (DynamicVerificationRequest,)
        ):
            raise ValueError("dynamic measurement request differs from independently pinned inputs")
        if selected.availability != "available":
            raise ValueError("dynamic measurement request is unavailable")
        sources = require_verification_source_catalog(
            envelope=envelope,
            time_contract=time_contract,
            source_records=source_records,
            expected_source_records_sha256=expected_source_records_sha256,
            catalog=catalog,
            expected_catalog_sha256=expected_catalog_sha256,
            store=store,
        )
        series = next(
            row
            for row in sources.catalog.series
            if row.canonical_series_key == selected.canonical_series_key
        )
        quality = series.quality_result
        component = next(
            row
            for row in series.transform_result.windows[1].components
            if row.name == series.transform_component
        )
        blocked = set()
        if component.aggregation in {"window_count", "exposure_rate"}:
            blocked.add("temporal_resolution_unavailable")
        if quality.baseline_scale is None or quality.baseline_summary.median is None:
            blocked.add("baseline_scale_unavailable")
        if (
            quality.baseline_summary.coverage_fraction
            < quality.configuration.minimum_coverage_fraction
        ):
            blocked.add("insufficient_baseline_coverage")
        reasons = set(quality.quality_flags) | blocked
        points: list[SignalEpisodePoint] = []
        for index, sample in enumerate(component.samples):
            point_state: PointState
            z, point_state = (
                (None, "unknown")
                if blocked
                else _classify_point(
                    sample.value,
                    quality.baseline_summary.median,
                    quality.baseline_scale,
                    quality.configuration.abnormal_z_threshold,
                )
            )
            point_reasons = blocked.copy()
            if not blocked and point_state == "unknown":
                point_reasons.add("non_finite_point_deviation")
            point = SignalEpisodePoint(
                sample_index=index,
                timestamp_unix_ns=sample.timestamp_unix_ns,
                source_interval=sample.source_interval,
                value=sample.value,
                signed_z=z,
                state=point_state,
                reason_codes=tuple(sorted(point_reasons)),
            )
            if points and (
                gaps := _gap_reasons(points[-1], point, policy, series.expected_cadence_ns)
            ):
                reasons.update(gaps)
                point_reasons.update(gap + "_before" for gap in gaps)
                point = point.model_copy(update={"reason_codes": tuple(sorted(point_reasons))})
            points.append(point)
            reasons.update(point_reasons)
        episodes = _episodes(
            points=tuple(points),
            request=selected,
            policy=policy,
            cadence=series.expected_cadence_ns,
        )
        default: TruthValue = (
            "false"
            if (
                points
                and all(point.state != "unknown" for point in points)
                and quality.fault_summary_sufficient
                and quality.fault_summary.coverage_fraction
                >= quality.configuration.minimum_coverage_fraction
                and not {"sampling_gap", "source_interval_gap"}.intersection(reasons)
            )
            else "unknown"
        )
        up: TruthValue = (
            "true" if any(row.qualifying and row.direction == "up" for row in episodes) else default
        )
        down: TruthValue = (
            "true"
            if any(row.qualifying and row.direction == "down" for row in episodes)
            else default
        )
        overall: TruthValue = (
            "true" if "true" in (up, down) else "false" if up == down == "false" else "unknown"
        )
        draft = SignalEpisodeMeasurement(
            request_sha256=selected.request_sha256,
            catalog_sha256=expected_catalog_sha256,
            policy_sha256=expected_policy_sha256,
            canonical_series_key=series.canonical_series_key,
            source_input_sha256=series.source_input_sha256,
            baseline_median=quality.baseline_summary.median,
            baseline_scale=quality.baseline_scale,
            abnormal_z_threshold=quality.configuration.abnormal_z_threshold,
            points=tuple(points),
            episodes=episodes,
            persistent_up=up,
            persistent_down=down,
            persistent_deviation_present=overall,
            quality=quality,
            reason_codes=tuple(sorted(reasons)),
            measurement_sha256="0" * 64,
        )
        document = cast(dict[str, object], plain(draft))
        document.pop("measurement_sha256")
        return draft.model_copy(
            update={"measurement_sha256": sha256_hex(canonical_json_bytes(document))}
        )
    finally:
        require_unchanged(originals, roots, before)
        require_store_unchanged(store, state)


def require_signal_episode_measurement(
    *,
    envelope: RuleHypothesisEnvelope,
    time_contract: CaseTimeContract,
    source_records: tuple[VerificationSourceRecord, ...],
    expected_source_records_sha256: str,
    request: DynamicVerificationRequest,
    expected_request_sha256: str,
    catalog: VerificationSourceCatalog,
    expected_catalog_sha256: str,
    policy: DynamicVerificationPolicy,
    expected_policy_sha256: str,
    store: SourceBlockStore,
    measurement: SignalEpisodeMeasurement,
    expected_measurement_sha256: str,
) -> SignalEpisodeMeasurement:
    """Return fresh evidence rebuilt from complete, independently pinned sources.

    A self-consistent measurement is only a claim. Replay every source gate and
    recompute every point, episode and reason; neither its hash nor verify() is
    a substitute for this comparison.
    """
    if (
        type(envelope) is not RuleHypothesisEnvelope
        or type(time_contract) is not CaseTimeContract
        or type(source_records) is not tuple
        or any(type(row) is not VerificationSourceRecord for row in source_records)
        or type(request) is not DynamicVerificationRequest
        or type(catalog) is not VerificationSourceCatalog
        or type(policy) is not DynamicVerificationPolicy
        or type(measurement) is not SignalEpisodeMeasurement
    ):
        raise ValueError("dynamic measurement replay requires exact frozen roots")
    for pin in (
        expected_source_records_sha256,
        expected_request_sha256,
        expected_catalog_sha256,
        expected_policy_sha256,
        expected_measurement_sha256,
    ):
        require_pin(pin)
    roots = (
        RuleHypothesisEnvelope,
        CaseTimeContract,
        VerificationSourceRecord,
        DynamicVerificationRequest,
        VerificationSourceCatalog,
        DynamicVerificationPolicy,
        SignalEpisodeMeasurement,
    )
    originals = (envelope, time_contract, source_records, request, catalog, policy, measurement)
    before = snapshot(originals, roots)
    state = store_snapshot(store)
    try:
        if (
            measurement.measurement_sha256 != expected_measurement_sha256
            or not measurement.verify()
        ):
            raise ValueError("dynamic measurement differs from independent pin or internal shape")
        rebuilt = measure_signal_episodes(
            envelope=envelope,
            time_contract=time_contract,
            source_records=source_records,
            expected_source_records_sha256=expected_source_records_sha256,
            request=request,
            expected_request_sha256=expected_request_sha256,
            catalog=catalog,
            expected_catalog_sha256=expected_catalog_sha256,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            store=store,
        )
        if rebuilt.measurement_sha256 != expected_measurement_sha256 or canonical(
            rebuilt, (SignalEpisodeMeasurement,)
        ) != canonical(measurement, (SignalEpisodeMeasurement,)):
            raise ValueError(
                "dynamic measurement differs from complete independently pinned sources"
            )
        return rebuilt
    finally:
        require_unchanged(originals, roots, before)
        require_store_unchanged(store, state)
