"""Auditable signal-specific transformations for NURSE V2 observations.

This module only establishes comparable signal semantics.  Robust window
summaries, sufficiency thresholds, anomaly strength, and coverage quality are
deliberately left to the later observation-quality stage.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.observation_schemas import SignalType, TimeInterval

NANOSECONDS_PER_SECOND = 1_000_000_000
SIGNAL_TRANSFORM_RESULT_SCHEMA_VERSION: Literal["nurse.v2.signal-transform-result/3"] = (
    "nurse.v2.signal-transform-result/3"
)
SIGNAL_TRANSFORM_SEMANTICS_VERSION: Literal["nurse-v2-signal-transforms/3"] = (
    "nurse-v2-signal-transforms/3"
)
SIGNAL_TRANSFORM_SOURCE_INPUT_SCHEMA_VERSION: Literal[
    "nurse.v2.signal-transform-source-inputs/2"
] = "nurse.v2.signal-transform-source-inputs/2"

_SIGNAL_TRANSFORM_SEMANTICS: dict[str, object] = {
    "version": SIGNAL_TRANSFORM_SEMANTICS_VERSION,
    "source_input_envelope": SIGNAL_TRANSFORM_SOURCE_INPUT_SCHEMA_VERSION,
    "counter": {
        "aggregation": "cumulative",
        "reset_delta": "post_reset_count",
        "boundary_reset": "explicit_provenance_required",
        "time_base": "actual_elapsed_seconds",
    },
    "gauge": {"aggregation": "instantaneous", "transform": "identity"},
    "latency": {"aggregations": ["p95", "p99", "raw"], "nonnegative": True},
    "error": {
        "aggregation": "cumulative_count",
        "components": ["error_count", "request_count", "error_ratio"],
        "pairing": "identical_source_interval_and_valid_sources",
        "count_integrality": "required",
    },
    "sparse": {
        "aggregation": "event_occurrence",
        "time_base": "verified_exposure_intervals",
    },
    "histogram": {
        "aggregation": "cumulative_buckets",
        "snapshot_transform": "interval_delta",
        "reset_delta": "post_reset_snapshot",
        "quantiles": [0.95, 0.99],
        "interpolation": "linear_within_bucket",
    },
    "sum": {"aggregation": "cumulative_sum"},
    "count": {"aggregation": "cumulative_count", "integrality": "required"},
    "sum_count": {
        "aggregations": ["cumulative_sum", "cumulative_count"],
        "pairing": "identical_source_interval_and_valid_sources",
        "transform": "sum_delta_per_count_delta",
    },
    "unknown": {"formal_scoring_eligible": False, "transform": "undefined"},
}
SIGNAL_TRANSFORM_SEMANTICS_SHA256 = sha256_hex(canonical_json_bytes(_SIGNAL_TRANSFORM_SEMANTICS))

TransformStatus = Literal["valid", "degraded", "unknown"]
WindowName = Literal["baseline", "fault"]


class ScalarSample(StrictFrozenModel):
    """One finite scalar at an exact UTC Unix nanosecond."""

    timestamp_unix_ns: int = Field(ge=0)
    value: float = Field(allow_inf_nan=False)
    source_interval: TimeInterval | None = None


class ScalarWindowInput(StrictFrozenModel):
    """One prefiltered scalar window with explicit aggregation semantics."""

    interval: TimeInterval
    aggregation: str = Field(min_length=1)
    samples: tuple[ScalarSample, ...]
    verified_reset_timestamps: tuple[int, ...] = ()
    verified_boundary_reset: bool = False

    @model_validator(mode="after")
    def require_ordered_bounded_samples(self) -> Self:
        timestamps = tuple(item.timestamp_unix_ns for item in self.samples)
        if timestamps != tuple(sorted(timestamps)) or len(timestamps) != len(set(timestamps)):
            raise ValueError("scalar samples must have strictly increasing timestamps")
        if any(
            not self.interval.start_unix_ns <= timestamp < self.interval.end_unix_ns
            for timestamp in timestamps
        ):
            raise ValueError("scalar sample is outside its declared window")
        if self.verified_reset_timestamps != tuple(sorted(set(self.verified_reset_timestamps))):
            raise ValueError("verified reset timestamps must be sorted and unique")
        if any(timestamp not in set(timestamps) for timestamp in self.verified_reset_timestamps):
            raise ValueError("verified reset must reference an input sample timestamp")
        if timestamps and timestamps[0] in self.verified_reset_timestamps:
            raise ValueError(
                "reset cannot reference the first sample without a predecessor interval"
            )
        if self.verified_boundary_reset and self.aggregation not in {
            "cumulative",
            "cumulative_count",
            "cumulative_sum",
        }:
            raise ValueError("boundary reset requires a cumulative scalar aggregation")
        if any(item.source_interval is not None for item in self.samples):
            raise ValueError("raw scalar samples cannot declare derived source intervals")
        return self


class SparseEventWindowInput(StrictFrozenModel):
    """Sparse events with independently verified observation exposure."""

    interval: TimeInterval
    aggregation: str = Field(min_length=1)
    event_timestamps_unix_ns: tuple[int, ...]
    exposure_intervals: tuple[TimeInterval, ...] = ()
    exposure_provenance_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_ordered_bounded_events(self) -> Self:
        if self.event_timestamps_unix_ns != tuple(sorted(self.event_timestamps_unix_ns)):
            raise ValueError("event timestamps must be ordered")
        if any(
            not self.interval.start_unix_ns <= timestamp < self.interval.end_unix_ns
            for timestamp in self.event_timestamps_unix_ns
        ):
            raise ValueError("event timestamp is outside its declared window")
        if self.exposure_intervals != tuple(
            sorted(self.exposure_intervals, key=lambda item: item.start_unix_ns)
        ):
            raise ValueError("exposure intervals must be ordered")
        if any(
            interval.start_unix_ns < self.interval.start_unix_ns
            or interval.end_unix_ns > self.interval.end_unix_ns
            for interval in self.exposure_intervals
        ):
            raise ValueError("exposure interval is outside its declared window")
        if any(
            left.end_unix_ns > right.start_unix_ns
            for left, right in zip(
                self.exposure_intervals[:-1],
                self.exposure_intervals[1:],
                strict=True,
            )
        ):
            raise ValueError("exposure intervals must not overlap")
        if self.exposure_provenance_ids != tuple(sorted(set(self.exposure_provenance_ids))):
            raise ValueError("exposure provenance IDs must be sorted and unique")
        if bool(self.exposure_intervals) != bool(self.exposure_provenance_ids):
            raise ValueError("known exposure requires intervals and provenance")
        if self.exposure_intervals and any(
            not any(
                interval.start_unix_ns <= timestamp < interval.end_unix_ns
                for interval in self.exposure_intervals
            )
            for timestamp in self.event_timestamps_unix_ns
        ):
            raise ValueError("event timestamp is outside verified exposure")
        return self


class HistogramBucket(StrictFrozenModel):
    """One cumulative bucket; ``None`` denotes the positive-infinity bound."""

    upper_bound: float | None = Field(default=None, allow_inf_nan=False)
    cumulative_count: int = Field(ge=0)


class HistogramSample(StrictFrozenModel):
    """One cumulative histogram snapshot with explicit sum/count semantics."""

    timestamp_unix_ns: int = Field(ge=0)
    buckets: tuple[HistogramBucket, ...]
    count: int = Field(ge=0)
    sum: float = Field(allow_inf_nan=False)


class HistogramWindowInput(StrictFrozenModel):
    """One prefiltered histogram window."""

    interval: TimeInterval
    aggregation: str = Field(min_length=1)
    samples: tuple[HistogramSample, ...]
    verified_reset_timestamps: tuple[int, ...] = ()
    verified_boundary_reset: bool = False

    @model_validator(mode="after")
    def require_ordered_bounded_samples(self) -> Self:
        timestamps = tuple(item.timestamp_unix_ns for item in self.samples)
        if timestamps != tuple(sorted(timestamps)) or len(timestamps) != len(set(timestamps)):
            raise ValueError("histogram samples must have strictly increasing timestamps")
        if any(
            not self.interval.start_unix_ns <= timestamp < self.interval.end_unix_ns
            for timestamp in timestamps
        ):
            raise ValueError("histogram sample is outside its declared window")
        if self.verified_reset_timestamps != tuple(sorted(set(self.verified_reset_timestamps))):
            raise ValueError("verified reset timestamps must be sorted and unique")
        if any(timestamp not in set(timestamps) for timestamp in self.verified_reset_timestamps):
            raise ValueError("verified reset must reference an input sample timestamp")
        if timestamps and timestamps[0] in self.verified_reset_timestamps:
            raise ValueError(
                "reset cannot reference the first sample without a predecessor interval"
            )
        if self.verified_boundary_reset and self.aggregation != "cumulative_buckets":
            raise ValueError("histogram boundary reset requires cumulative buckets")
        return self


SourceWindowInput = ScalarWindowInput | SparseEventWindowInput | HistogramWindowInput


def compute_source_input_sha256(source_inputs: Mapping[str, SourceWindowInput]) -> str:
    """Hash exact named raw windows, including their intervals and reset declarations."""

    if not source_inputs:
        raise ValueError("source input digest requires at least one named window")
    payload = {
        "schema_version": SIGNAL_TRANSFORM_SOURCE_INPUT_SCHEMA_VERSION,
        "inputs": {
            name: {
                "input_kind": type(source).__name__,
                "document": source.model_dump(mode="json"),
            }
            for name, source in sorted(source_inputs.items())
        },
    }
    return sha256_hex(canonical_json_bytes(payload))


class TransformedSeries(StrictFrozenModel):
    """A named derived series whose eligibility is independently auditable."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    unit: str = Field(min_length=1)
    aggregation: str = Field(min_length=1)
    samples: tuple[ScalarSample, ...]
    status: TransformStatus
    formal_scoring_eligible: bool
    quality_flags: tuple[str, ...]

    @model_validator(mode="after")
    def require_consistent_series_state(self) -> Self:
        timestamps = tuple(item.timestamp_unix_ns for item in self.samples)
        if timestamps != tuple(sorted(timestamps)) or len(timestamps) != len(set(timestamps)):
            raise ValueError("transformed samples must have strictly increasing timestamps")
        if self.quality_flags != tuple(sorted(set(self.quality_flags))):
            raise ValueError("series quality flags must be sorted and unique")
        if self.status == "unknown" and (self.samples or self.formal_scoring_eligible):
            raise ValueError("unknown transformed series must be empty and ineligible")
        if self.formal_scoring_eligible and (self.status != "valid" or not self.samples):
            raise ValueError("formal scoring requires a nonempty valid series")
        interval_derived = self.aggregation in {
            "exposure_rate",
            "interval_delta",
            "interval_error_ratio",
            "per_second_rate",
            "sum_delta_per_count_delta",
            "sum_per_count",
            "window_count",
        } or self.name in {"p95", "p99"}
        if interval_derived and any(item.source_interval is None for item in self.samples):
            raise ValueError("interval-derived sample requires source interval provenance")
        if not interval_derived and any(item.source_interval is not None for item in self.samples):
            raise ValueError("instantaneous sample cannot declare source interval provenance")
        return self


class ComponentRawAttribution(StrictFrozenModel):
    """Canonical direct raw-source lineage for one transformed component."""

    component_name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    direct_source_names: tuple[str, ...] = Field(min_length=1)
    raw_record_state: Literal["recorded", "no_record"]
    raw_record_count: int = Field(ge=0)

    @model_validator(mode="after")
    def require_canonical_attribution(self) -> Self:
        if self.direct_source_names != tuple(sorted(set(self.direct_source_names))):
            raise ValueError("direct source names must be sorted and unique")
        if self.raw_record_state == "no_record" and self.raw_record_count != 0:
            raise ValueError("raw record state and count are inconsistent")
        return self


class WindowSignalTransform(StrictFrozenModel):
    """All signal-specific components derived for one observation window."""

    window: WindowName
    interval: TimeInterval
    source_aggregation: str = Field(min_length=1)
    raw_record_state: Literal["recorded", "no_record"]
    raw_record_count: int = Field(ge=0)
    status: TransformStatus
    components: tuple[TransformedSeries, ...]
    component_raw_attributions: tuple[ComponentRawAttribution, ...] = ()
    verified_reset_timestamps: tuple[int, ...]
    verified_boundary_reset_sources: tuple[str, ...] = ()
    exposure_seconds: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    exposure_intervals: tuple[TimeInterval, ...] = ()
    exposure_provenance_ids: tuple[str, ...] = ()
    first_event_unix_ns: int | None = Field(default=None, ge=0)
    quality_flags: tuple[str, ...]

    @model_validator(mode="after")
    def require_canonical_window(self) -> Self:
        if self.raw_record_state == "no_record" and self.raw_record_count != 0:
            raise ValueError("raw record state and count are inconsistent")
        if self.exposure_provenance_ids != tuple(sorted(set(self.exposure_provenance_ids))):
            raise ValueError("exposure provenance IDs must be sorted and unique")
        if bool(self.exposure_intervals) != bool(self.exposure_provenance_ids):
            raise ValueError("known exposure requires intervals and provenance")
        if self.components != tuple(sorted(self.components, key=lambda item: item.name)):
            raise ValueError("transformed components must be canonically ordered")
        names = tuple(item.name for item in self.components)
        if len(names) != len(set(names)):
            raise ValueError("transformed component names must be unique")
        if self.component_raw_attributions and self.component_raw_attributions != tuple(
            sorted(
                self.component_raw_attributions,
                key=lambda item: item.component_name,
            )
        ):
            raise ValueError("component raw attributions must be canonically ordered")
        attribution_names = tuple(item.component_name for item in self.component_raw_attributions)
        if attribution_names and attribution_names != names:
            raise ValueError("component raw attributions must cover every component")
        if self.verified_reset_timestamps != tuple(sorted(set(self.verified_reset_timestamps))):
            raise ValueError("verified reset timestamps must be sorted and unique")
        if self.verified_boundary_reset_sources != tuple(
            sorted(set(self.verified_boundary_reset_sources))
        ):
            raise ValueError("boundary reset sources must be sorted and unique")
        if self.window == "baseline" and self.verified_boundary_reset_sources:
            raise ValueError("baseline window cannot record a boundary reset")
        if self.quality_flags != tuple(sorted(set(self.quality_flags))):
            raise ValueError("window quality flags must be sorted and unique")
        for component in self.components:
            for sample in component.samples:
                if not (
                    self.interval.start_unix_ns
                    <= sample.timestamp_unix_ns
                    < self.interval.end_unix_ns
                ):
                    raise ValueError("component sample is outside its transform window")
                if sample.source_interval is not None and not (
                    self.interval.start_unix_ns
                    <= sample.source_interval.start_unix_ns
                    < sample.source_interval.end_unix_ns
                    <= self.interval.end_unix_ns
                ):
                    raise ValueError("component source interval is outside its transform window")
        if any(
            not self.interval.start_unix_ns <= timestamp < self.interval.end_unix_ns
            for timestamp in self.verified_reset_timestamps
        ):
            raise ValueError("verified reset is outside its transform window")
        if self.first_event_unix_ns is not None and not (
            self.interval.start_unix_ns <= self.first_event_unix_ns < self.interval.end_unix_ns
        ):
            raise ValueError("first event must be inside its declared window")
        component_flags = {
            flag for component in self.components for flag in component.quality_flags
        }
        if set(self.quality_flags) != component_flags:
            raise ValueError("window quality flags must equal component quality flags")
        component_statuses = {item.status for item in self.components}
        if component_statuses == {"unknown"}:
            expected_status: TransformStatus = "unknown"
        elif component_statuses == {"valid"} and not self.quality_flags:
            expected_status = "valid"
        else:
            expected_status = "degraded"
        if self.status != expected_status:
            raise ValueError("window status is inconsistent with its components")
        return self


class SignalTransformResult(StrictFrozenModel):
    """Content-addressed baseline/fault transformation with no anomaly judgment."""

    schema_version: Literal["nurse.v2.signal-transform-result/3"] = (
        SIGNAL_TRANSFORM_RESULT_SCHEMA_VERSION
    )
    semantics_version: Literal["nurse-v2-signal-transforms/3"] = SIGNAL_TRANSFORM_SEMANTICS_VERSION
    signal_type: SignalType
    declared_signal_type: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    transform: str = Field(min_length=1)
    windows: tuple[WindowSignalTransform, WindowSignalTransform]
    status: TransformStatus
    formal_scoring_eligible: bool
    quality_flags: tuple[str, ...]
    source_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_result(self) -> Self:
        if tuple(item.window for item in self.windows) != ("baseline", "fault"):
            raise ValueError("transform result must contain baseline then fault")
        if self.windows[0].interval.end_unix_ns != self.windows[1].interval.start_unix_ns:
            raise ValueError("baseline and fault windows must be contiguous")
        if self.quality_flags != tuple(sorted(set(self.quality_flags))):
            raise ValueError("result quality flags must be sorted and unique")
        expected_flags = {flag for window in self.windows for flag in window.quality_flags}
        if set(self.quality_flags) != expected_flags:
            raise ValueError("result quality flags must equal window quality flags")
        if self.signal_type == "unknown" and (
            self.status != "unknown" or self.formal_scoring_eligible
        ):
            raise ValueError("unknown signal types cannot enter formal scoring")
        window_statuses = {item.status for item in self.windows}
        if window_statuses == {"unknown"} or self.signal_type == "unknown":
            expected_status: TransformStatus = "unknown"
        elif window_statuses == {"valid"}:
            expected_status = "valid"
        else:
            expected_status = "degraded"
        eligible_names = [
            {component.name for component in window.components if component.formal_scoring_eligible}
            for window in self.windows
        ]
        expected_eligible = bool(eligible_names[0] & eligible_names[1])
        if self.signal_type == "unknown":
            expected_eligible = False
        if self.status != expected_status or self.formal_scoring_eligible != expected_eligible:
            raise ValueError("derived transform status or eligibility is inconsistent")
        _require_transform_contract(self)
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"result_sha256"})
        return (
            self.adapter_semantics_sha256 == SIGNAL_TRANSFORM_SEMANTICS_SHA256
            and self.result_sha256 == sha256_hex(canonical_json_bytes(unsigned))
        )

    def verify_source_inputs(self, source_inputs: Mapping[str, SourceWindowInput]) -> bool:
        """Replay exact caller-held raw inputs and compare the complete canonical result."""

        try:
            if self.source_input_sha256 != compute_source_input_sha256(source_inputs):
                return False
            replayed = _replay_transform(self, source_inputs)
        except (KeyError, TypeError, ValueError):
            return False
        return replayed == self


def transform_counter(
    *,
    unit: str,
    baseline: ScalarWindowInput,
    fault: ScalarWindowInput,
) -> SignalTransformResult:
    """Convert a monotonic counter to interval deltas and per-second rates."""

    return _transform_cumulative_pair(
        signal_type="counter",
        declared_signal_type="counter",
        unit=unit,
        baseline=baseline,
        fault=fault,
        expected_aggregation="cumulative",
        delta_name="delta",
        rate_name="rate",
        transform="counter_delta_and_rate",
        require_integral=False,
    )


def transform_sum(
    *,
    unit: str,
    baseline: ScalarWindowInput,
    fault: ScalarWindowInput,
) -> SignalTransformResult:
    """Convert a cumulative sum to interval deltas and actual-time rates."""

    return _transform_cumulative_pair(
        signal_type="sum",
        declared_signal_type="sum",
        unit=unit,
        baseline=baseline,
        fault=fault,
        expected_aggregation="cumulative_sum",
        delta_name="sum_delta",
        rate_name="sum_rate",
        transform="cumulative_sum_delta_and_rate",
        require_integral=False,
    )


def transform_count(
    *,
    unit: str,
    baseline: ScalarWindowInput,
    fault: ScalarWindowInput,
) -> SignalTransformResult:
    """Convert a cumulative count to interval deltas and actual-time rates."""

    return _transform_cumulative_pair(
        signal_type="count",
        declared_signal_type="count",
        unit=unit,
        baseline=baseline,
        fault=fault,
        expected_aggregation="cumulative_count",
        delta_name="count_delta",
        rate_name="count_rate",
        transform="cumulative_count_delta_and_rate",
        require_integral=True,
    )


def transform_gauge(
    *,
    unit: str,
    baseline: ScalarWindowInput,
    fault: ScalarWindowInput,
) -> SignalTransformResult:
    """Preserve instantaneous gauge values without inventing rate semantics."""

    invalid = _aggregation_flags(
        baseline.aggregation,
        fault.aggregation,
        allowed=frozenset({"instantaneous"}),
    )
    if invalid:
        return _unknown_scalar_result(
            signal_type="gauge",
            declared_signal_type="gauge",
            unit=unit,
            transform="gauge_identity",
            baseline=baseline,
            fault=fault,
            component_specs=(("value", unit, "instantaneous"),),
            flags=invalid,
        )
    return _identity_result(
        signal_type="gauge",
        unit=unit,
        transform="gauge_identity",
        component_name="value",
        baseline=baseline,
        fault=fault,
        nonnegative=False,
    )


def transform_latency(
    *,
    unit: str,
    baseline: ScalarWindowInput,
    fault: ScalarWindowInput,
) -> SignalTransformResult:
    """Preserve nonnegative latency values only under one consistent mouth."""

    allowed = frozenset({"raw", "p95", "p99"})
    invalid = _aggregation_flags(
        baseline.aggregation,
        fault.aggregation,
        allowed=allowed,
    )
    if invalid:
        return _unknown_scalar_result(
            signal_type="latency",
            declared_signal_type="latency",
            unit=unit,
            transform="latency_identity",
            baseline=baseline,
            fault=fault,
            component_specs=(("latency", unit, baseline.aggregation),),
            flags=invalid,
        )
    return _identity_result(
        signal_type="latency",
        unit=unit,
        transform="latency_identity",
        component_name="latency",
        baseline=baseline,
        fault=fault,
        nonnegative=True,
    )


def transform_error(
    *,
    baseline_errors: ScalarWindowInput,
    baseline_requests: ScalarWindowInput,
    fault_errors: ScalarWindowInput,
    fault_requests: ScalarWindowInput,
) -> SignalTransformResult:
    """Preserve interval error/request counts and their defined ratios."""

    windows: tuple[tuple[WindowName, ScalarWindowInput, ScalarWindowInput], ...] = (
        ("baseline", baseline_errors, baseline_requests),
        ("fault", fault_errors, fault_requests),
    )
    flags: set[str] = set()
    if any(error.interval != request.interval for _, error, request in windows):
        flags.add("paired_interval_mismatch")
    aggregations = (
        baseline_errors.aggregation,
        baseline_requests.aggregation,
        fault_errors.aggregation,
        fault_requests.aggregation,
    )
    if len(set(aggregations)) != 1:
        flags.add("aggregation_mismatch")
    if any(value != "cumulative_count" for value in aggregations):
        flags.add("unsupported_aggregation")
    if flags:
        return _unknown_error_result(
            baseline=baseline_errors,
            fault=fault_errors,
            source_inputs={
                "baseline_errors": baseline_errors,
                "baseline_requests": baseline_requests,
                "fault_errors": fault_errors,
                "fault_requests": fault_requests,
            },
            flags=flags,
        )

    error_baseline_flags, error_fault_flags = _scalar_boundary_flags(baseline_errors, fault_errors)
    request_baseline_flags, request_fault_flags = _scalar_boundary_flags(
        baseline_requests, fault_requests
    )
    output_windows = (
        _error_window(
            "baseline",
            baseline_errors,
            baseline_requests,
            error_extra_flags=error_baseline_flags,
            request_extra_flags=request_baseline_flags,
        ),
        _error_window(
            "fault",
            fault_errors,
            fault_requests,
            error_extra_flags=error_fault_flags,
            request_extra_flags=request_fault_flags,
            boundary_reset_sources=tuple(
                source
                for source, present in (
                    ("errors", fault_errors.verified_boundary_reset),
                    ("requests", fault_requests.verified_boundary_reset),
                )
                if present
            ),
        ),
    )
    return _build_result(
        signal_type="error",
        declared_signal_type="error",
        unit="ratio",
        transform="error_count_request_ratio",
        windows=output_windows,
        source_inputs={
            "baseline_errors": baseline_errors,
            "baseline_requests": baseline_requests,
            "fault_errors": fault_errors,
            "fault_requests": fault_requests,
        },
    )


def transform_sparse_events(
    *,
    unit: str,
    baseline: SparseEventWindowInput,
    fault: SparseEventWindowInput,
) -> SignalTransformResult:
    """Convert event occurrences to counts/rates over observed exposure time."""

    invalid = _aggregation_flags(
        baseline.aggregation,
        fault.aggregation,
        allowed=frozenset({"event_occurrence"}),
    )
    sources: tuple[tuple[WindowName, SparseEventWindowInput], ...] = (
        ("baseline", baseline),
        ("fault", fault),
    )
    if invalid:
        windows = tuple(
            _unknown_window(
                name=name,
                interval=source.interval,
                aggregation=source.aggregation,
                component_specs=(
                    ("event_count", unit, "window_count"),
                    ("event_rate", f"{unit}/second", "exposure_rate"),
                ),
                flags=invalid,
            )
            for name, source in sources
        )
    else:
        windows = tuple(_sparse_window(name, source, unit) for name, source in sources)
    return _build_result(
        signal_type="sparse",
        declared_signal_type="sparse",
        unit=unit,
        transform="event_exposure_rate",
        windows=windows,
        source_inputs={"baseline": baseline, "fault": fault},
        extra_flags=invalid,
    )


def transform_histogram(
    *,
    unit: str,
    baseline: HistogramWindowInput,
    fault: HistogramWindowInput,
) -> SignalTransformResult:
    """Derive p95/p99 only from complete, monotone cumulative buckets."""

    invalid = _aggregation_flags(
        baseline.aggregation,
        fault.aggregation,
        allowed=frozenset({"cumulative_buckets"}),
    )
    boundary_schemas = {
        _histogram_boundary_signature(sample)
        for source in (baseline, fault)
        for sample in source.samples
    }
    if len(boundary_schemas) > 1:
        invalid.add("histogram_boundary_mismatch")
    sources: tuple[tuple[WindowName, HistogramWindowInput], ...] = (
        ("baseline", baseline),
        ("fault", fault),
    )
    if invalid:
        windows = tuple(
            _unknown_window(
                name=name,
                interval=source.interval,
                aggregation=source.aggregation,
                component_specs=_histogram_component_specs(unit),
                flags=invalid,
            )
            for name, source in sources
        )
    else:
        baseline_flags, fault_flags = _histogram_boundary_flags(baseline, fault)
        windows = (
            _histogram_window(
                "baseline",
                baseline,
                unit,
                extra_flags=baseline_flags,
            ),
            _histogram_window(
                "fault",
                fault,
                unit,
                extra_flags=fault_flags,
                boundary_reset_sources=("histogram",) if fault.verified_boundary_reset else (),
            ),
        )
    return _build_result(
        signal_type="histogram",
        declared_signal_type="histogram",
        unit=unit,
        transform="cumulative_histogram_percentiles",
        windows=windows,
        source_inputs={"baseline": baseline, "fault": fault},
        extra_flags=invalid,
    )


def transform_sum_count(
    *,
    unit: str,
    baseline_sum: ScalarWindowInput,
    baseline_count: ScalarWindowInput,
    fault_sum: ScalarWindowInput,
    fault_count: ScalarWindowInput,
) -> SignalTransformResult:
    """Pair cumulative sum/count deltas and derive a mean only when defined."""

    windows: tuple[tuple[WindowName, ScalarWindowInput, ScalarWindowInput], ...] = (
        ("baseline", baseline_sum, baseline_count),
        ("fault", fault_sum, fault_count),
    )
    flags: set[str] = set()
    if any(total.interval != count.interval for _, total, count in windows):
        flags.add("paired_interval_mismatch")
    if baseline_sum.aggregation != fault_sum.aggregation:
        flags.add("aggregation_mismatch")
    if baseline_count.aggregation != fault_count.aggregation:
        flags.add("aggregation_mismatch")
    if any(source.aggregation != "cumulative_sum" for source in (baseline_sum, fault_sum)):
        flags.add("unsupported_sum_aggregation")
    if any(source.aggregation != "cumulative_count" for source in (baseline_count, fault_count)):
        flags.add("unsupported_count_aggregation")
    if flags:
        component_specs = (
            ("count_delta", "count", "interval_delta"),
            ("mean", unit, "sum_delta_per_count_delta"),
            ("sum_delta", unit, "interval_delta"),
        )
        output_windows = tuple(
            _unknown_window(
                name=name,
                interval=total.interval,
                aggregation=f"{total.aggregation}+{count.aggregation}",
                component_specs=component_specs,
                flags=flags,
            )
            for name, total, count in windows
        )
    else:
        sum_baseline_flags, sum_fault_flags = _scalar_boundary_flags(baseline_sum, fault_sum)
        count_baseline_flags, count_fault_flags = _scalar_boundary_flags(
            baseline_count, fault_count
        )
        output_windows = (
            _sum_count_window(
                "baseline",
                baseline_sum,
                baseline_count,
                unit,
                sum_extra_flags=sum_baseline_flags,
                count_extra_flags=count_baseline_flags,
            ),
            _sum_count_window(
                "fault",
                fault_sum,
                fault_count,
                unit,
                sum_extra_flags=sum_fault_flags,
                count_extra_flags=count_fault_flags,
                boundary_reset_sources=tuple(
                    source
                    for source, present in (
                        ("count", fault_count.verified_boundary_reset),
                        ("sum", fault_sum.verified_boundary_reset),
                    )
                    if present
                ),
            ),
        )
    return _build_result(
        signal_type="sum",
        declared_signal_type="sum+count",
        unit=unit,
        transform="sum_count_mean",
        windows=output_windows,
        source_inputs={
            "baseline_sum": baseline_sum,
            "baseline_count": baseline_count,
            "fault_sum": fault_sum,
            "fault_count": fault_count,
        },
        extra_flags=flags,
    )


def transform_unknown(
    *,
    declared_signal_type: str,
    unit: str,
    baseline: ScalarWindowInput,
    fault: ScalarWindowInput,
) -> SignalTransformResult:
    """Retain an undefined source type as explicit non-scorable evidence."""

    return _unknown_scalar_result(
        signal_type="unknown",
        declared_signal_type=declared_signal_type,
        unit=unit,
        transform="undefined_signal_transform",
        baseline=baseline,
        fault=fault,
        component_specs=(("value", unit, "undefined"),),
        flags={"undefined_signal_type"},
    )


def _transform_cumulative_pair(
    *,
    signal_type: Literal["counter", "sum", "count"],
    declared_signal_type: str,
    unit: str,
    baseline: ScalarWindowInput,
    fault: ScalarWindowInput,
    expected_aggregation: str,
    delta_name: str,
    rate_name: str,
    transform: str,
    require_integral: bool,
) -> SignalTransformResult:
    invalid = _aggregation_flags(
        baseline.aggregation,
        fault.aggregation,
        allowed=frozenset({expected_aggregation}),
    )
    component_specs = (
        (delta_name, unit, "interval_delta"),
        (rate_name, f"{unit}/second", "per_second_rate"),
    )
    if invalid:
        return _unknown_scalar_result(
            signal_type=signal_type,
            declared_signal_type=declared_signal_type,
            unit=unit,
            transform=transform,
            baseline=baseline,
            fault=fault,
            component_specs=component_specs,
            flags=invalid,
        )
    baseline_boundary_flags, fault_boundary_flags = _scalar_boundary_flags(baseline, fault)
    windows = (
        _cumulative_window(
            name="baseline",
            source=baseline,
            unit=unit,
            delta_name=delta_name,
            rate_name=rate_name,
            require_integral=require_integral,
            extra_flags=baseline_boundary_flags,
        ),
        _cumulative_window(
            name="fault",
            source=fault,
            unit=unit,
            delta_name=delta_name,
            rate_name=rate_name,
            require_integral=require_integral,
            extra_flags=fault_boundary_flags,
            boundary_reset_sources=("value",) if fault.verified_boundary_reset else (),
        ),
    )
    return _build_result(
        signal_type=signal_type,
        declared_signal_type=declared_signal_type,
        unit=unit,
        transform=transform,
        windows=windows,
        source_inputs={"baseline": baseline, "fault": fault},
    )


def _identity_result(
    *,
    signal_type: Literal["gauge", "latency"],
    unit: str,
    transform: str,
    component_name: str,
    baseline: ScalarWindowInput,
    fault: ScalarWindowInput,
    nonnegative: bool,
) -> SignalTransformResult:
    sources: tuple[tuple[WindowName, ScalarWindowInput], ...] = (
        ("baseline", baseline),
        ("fault", fault),
    )
    windows = tuple(
        _identity_window(name, source, unit, component_name, nonnegative)
        for name, source in sources
    )
    return _build_result(
        signal_type=signal_type,
        declared_signal_type=signal_type,
        unit=unit,
        transform=transform,
        windows=windows,
        source_inputs={"baseline": baseline, "fault": fault},
    )


def _identity_window(
    name: WindowName,
    source: ScalarWindowInput,
    unit: str,
    component_name: str,
    nonnegative: bool,
) -> WindowSignalTransform:
    flags: set[str] = set()
    if source.verified_reset_timestamps:
        flags.add("unexpected_reset_markers")
    if nonnegative and any(item.value < 0.0 for item in source.samples):
        flags.add("negative_latency_value")
    if not source.samples:
        flags.add("no_samples")
    if flags:
        component = _unknown_component(
            component_name,
            unit,
            source.aggregation,
            flags,
        )
    else:
        component = _component(
            name=component_name,
            unit=unit,
            aggregation=source.aggregation,
            samples=source.samples,
        )
    return _window(
        name=name,
        interval=source.interval,
        aggregation=source.aggregation,
        components=(component,),
        resets=(),
        flags=flags,
    )


def _cumulative_window(
    *,
    name: WindowName,
    source: ScalarWindowInput,
    unit: str,
    delta_name: str,
    rate_name: str,
    require_integral: bool,
    extra_flags: Iterable[str] = (),
    boundary_reset_sources: Iterable[str] = (),
) -> WindowSignalTransform:
    flags = set(extra_flags)
    if any(item.value < 0.0 for item in source.samples):
        flags.add("negative_cumulative_value")
    if require_integral and any(not item.value.is_integer() for item in source.samples):
        flags.add("non_integral_count")
    deltas: list[ScalarSample] = []
    rates: list[ScalarSample] = []
    if len(source.samples) < 2:
        flags.add("insufficient_cumulative_samples")
    if not flags:
        for previous, current in zip(source.samples[:-1], source.samples[1:], strict=True):
            elapsed_seconds = (
                current.timestamp_unix_ns - previous.timestamp_unix_ns
            ) / NANOSECONDS_PER_SECOND
            source_interval = TimeInterval(
                start_unix_ns=previous.timestamp_unix_ns,
                end_unix_ns=current.timestamp_unix_ns,
            )
            if current.timestamp_unix_ns in source.verified_reset_timestamps:
                delta = current.value
            elif current.value < previous.value:
                flags.add("unverified_counter_decrease")
                continue
            else:
                delta = current.value - previous.value
            rate = delta / elapsed_seconds
            if not math.isfinite(delta) or not math.isfinite(rate):
                flags.add("non_finite_derived_statistic")
                deltas.clear()
                rates.clear()
                break
            deltas.append(
                ScalarSample(
                    timestamp_unix_ns=current.timestamp_unix_ns,
                    value=delta,
                    source_interval=source_interval,
                )
            )
            rates.append(
                ScalarSample(
                    timestamp_unix_ns=current.timestamp_unix_ns,
                    value=rate,
                    source_interval=source_interval,
                )
            )
    if not deltas:
        components = (
            _unknown_component(delta_name, unit, "interval_delta", flags),
            _unknown_component(
                rate_name,
                f"{unit}/second",
                "per_second_rate",
                flags,
            ),
        )
    elif flags:
        components = (
            _component(
                name=delta_name,
                unit=unit,
                aggregation="interval_delta",
                samples=deltas,
                status="degraded",
                eligible=False,
                flags=flags,
            ),
            _component(
                name=rate_name,
                unit=f"{unit}/second",
                aggregation="per_second_rate",
                samples=rates,
                status="degraded",
                eligible=False,
                flags=flags,
            ),
        )
    else:
        components = (
            _component(
                name=delta_name,
                unit=unit,
                aggregation="interval_delta",
                samples=deltas,
            ),
            _component(
                name=rate_name,
                unit=f"{unit}/second",
                aggregation="per_second_rate",
                samples=rates,
            ),
        )
    return _window(
        name=name,
        interval=source.interval,
        aggregation=source.aggregation,
        components=components,
        resets=source.verified_reset_timestamps,
        flags=flags,
        boundary_reset_sources=boundary_reset_sources,
    )


def _error_window(
    name: WindowName,
    errors: ScalarWindowInput,
    requests: ScalarWindowInput,
    *,
    error_extra_flags: Iterable[str] = (),
    request_extra_flags: Iterable[str] = (),
    boundary_reset_sources: Iterable[str] = (),
) -> WindowSignalTransform:
    error_window = _cumulative_window(
        name=name,
        source=errors,
        unit="errors",
        delta_name="error_count",
        rate_name="error_rate",
        require_integral=True,
        extra_flags=error_extra_flags,
    )
    request_window = _cumulative_window(
        name=name,
        source=requests,
        unit="requests",
        delta_name="request_count",
        rate_name="request_rate",
        require_integral=True,
        extra_flags=request_extra_flags,
    )
    error_counts = _component_by_name(error_window, "error_count")
    request_counts = _component_by_name(request_window, "request_count")
    flags = set(error_window.quality_flags) | set(request_window.quality_flags)
    ratios: list[ScalarSample] = []
    ratio_flags: set[str] = set()
    if error_counts.status != "valid" or request_counts.status != "valid":
        ratio_flags.add("source_count_series_not_valid")
    else:
        error_by_interval = {_source_interval_key(item): item for item in error_counts.samples}
        request_by_interval = {_source_interval_key(item): item for item in request_counts.samples}
        if set(error_by_interval) != set(request_by_interval):
            ratio_flags.add("unaligned_count_intervals")
        for interval_key in sorted(set(error_by_interval) & set(request_by_interval)):
            error_sample = error_by_interval[interval_key]
            request_sample = request_by_interval[interval_key]
            request_count = request_sample.value
            error_count = error_sample.value
            if request_count == 0.0:
                ratio_flags.add("zero_request_count")
            elif error_count > request_count:
                ratio_flags.add("error_count_exceeds_request_count")
            else:
                ratios.append(
                    ScalarSample(
                        timestamp_unix_ns=error_sample.timestamp_unix_ns,
                        value=error_count / request_count,
                        source_interval=error_sample.source_interval,
                    )
                )
    flags.update(ratio_flags)
    if not ratios:
        ratio = _unknown_component(
            "error_ratio",
            "ratio",
            "interval_error_ratio",
            ratio_flags or flags or {"no_aligned_count_samples"},
        )
    elif ratio_flags:
        ratio = _component(
            name="error_ratio",
            unit="ratio",
            aggregation="interval_error_ratio",
            samples=ratios,
            status="degraded",
            eligible=False,
            flags=ratio_flags,
        )
    else:
        ratio = _component(
            name="error_ratio",
            unit="ratio",
            aggregation="interval_error_ratio",
            samples=ratios,
        )
    return _window(
        name=name,
        interval=errors.interval,
        aggregation="cumulative_count",
        components=(error_counts, request_counts, ratio),
        resets=tuple(
            sorted(
                set(error_window.verified_reset_timestamps)
                | set(request_window.verified_reset_timestamps)
            )
        ),
        flags=flags,
        boundary_reset_sources=boundary_reset_sources,
    )


def _sparse_window(
    name: WindowName,
    source: SparseEventWindowInput,
    unit: str,
) -> WindowSignalTransform:
    if not source.exposure_intervals:
        return _window(
            name=name,
            interval=source.interval,
            aggregation=source.aggregation,
            components=(
                _unknown_component("event_count", unit, "window_count", {"exposure_unknown"}),
                _unknown_component(
                    "event_rate",
                    f"{unit}/second",
                    "exposure_rate",
                    {"exposure_unknown"},
                ),
            ),
            resets=(),
            flags={"exposure_unknown"},
            first_event_unix_ns=(
                source.event_timestamps_unix_ns[0] if source.event_timestamps_unix_ns else None
            ),
        )
    exposure_seconds = (
        sum(interval.end_unix_ns - interval.start_unix_ns for interval in source.exposure_intervals)
        / NANOSECONDS_PER_SECOND
    )
    count = len(source.event_timestamps_unix_ns)
    derived_at = source.interval.end_unix_ns - 1
    source_interval = source.interval
    components = (
        _component(
            name="event_count",
            unit=unit,
            aggregation="window_count",
            samples=(
                ScalarSample(
                    timestamp_unix_ns=derived_at,
                    value=float(count),
                    source_interval=source_interval,
                ),
            ),
        ),
        _component(
            name="event_rate",
            unit=f"{unit}/second",
            aggregation="exposure_rate",
            samples=(
                ScalarSample(
                    timestamp_unix_ns=derived_at,
                    value=count / exposure_seconds,
                    source_interval=source_interval,
                ),
            ),
        ),
    )
    return _window(
        name=name,
        interval=source.interval,
        aggregation=source.aggregation,
        components=components,
        resets=(),
        flags=(),
        exposure_seconds=exposure_seconds,
        exposure_intervals=source.exposure_intervals,
        exposure_provenance_ids=source.exposure_provenance_ids,
        first_event_unix_ns=(
            source.event_timestamps_unix_ns[0] if source.event_timestamps_unix_ns else None
        ),
    )


def _histogram_window(
    name: WindowName,
    source: HistogramWindowInput,
    unit: str,
    *,
    extra_flags: Iterable[str] = (),
    boundary_reset_sources: Iterable[str] = (),
) -> WindowSignalTransform:
    flags = set(extra_flags)
    percentile_points: dict[str, list[ScalarSample]] = {"mean": [], "p95": [], "p99": []}
    if len(source.samples) < 2:
        flags.add("insufficient_histogram_samples")
    for sample in source.samples:
        flags.update(_histogram_flags(sample))
    interval_samples: list[tuple[HistogramSample, TimeInterval]] = []
    if not flags:
        for previous, current in zip(source.samples[:-1], source.samples[1:], strict=True):
            support = TimeInterval(
                start_unix_ns=previous.timestamp_unix_ns,
                end_unix_ns=current.timestamp_unix_ns,
            )
            if current.timestamp_unix_ns in source.verified_reset_timestamps:
                derived = current
            elif (
                current.count < previous.count
                or current.sum < previous.sum
                or any(
                    current_bucket.cumulative_count < previous_bucket.cumulative_count
                    for previous_bucket, current_bucket in zip(
                        previous.buckets,
                        current.buckets,
                        strict=True,
                    )
                )
            ):
                flags.add("unverified_histogram_decrease")
                continue
            else:
                derived = HistogramSample(
                    timestamp_unix_ns=current.timestamp_unix_ns,
                    buckets=tuple(
                        HistogramBucket(
                            upper_bound=current_bucket.upper_bound,
                            cumulative_count=(
                                current_bucket.cumulative_count - previous_bucket.cumulative_count
                            ),
                        )
                        for previous_bucket, current_bucket in zip(
                            previous.buckets,
                            current.buckets,
                            strict=True,
                        )
                    ),
                    count=current.count - previous.count,
                    sum=current.sum - previous.sum,
                )
            derived_flags = _histogram_flags(derived)
            if derived_flags:
                flags.update(derived_flags)
                continue
            interval_samples.append((derived, support))
    for sample, support in interval_samples:
        quantiles = {
            "p95": _histogram_quantile(sample, 0.95),
            "p99": _histogram_quantile(sample, 0.99),
        }
        if any(value is None for value in quantiles.values()):
            flags.add("unbounded_histogram_quantile")
            continue
        percentile_points["mean"].append(
            ScalarSample(
                timestamp_unix_ns=sample.timestamp_unix_ns,
                value=sample.sum / sample.count,
                source_interval=support,
            )
        )
        for component_name, value in quantiles.items():
            if value is None:  # pragma: no cover - guarded above
                raise AssertionError("histogram quantile unexpectedly absent")
            percentile_points[component_name].append(
                ScalarSample(
                    timestamp_unix_ns=sample.timestamp_unix_ns,
                    value=value,
                    source_interval=support,
                )
            )
    if flags:
        components = tuple(
            _unknown_component(component_name, unit, aggregation, flags)
            for component_name, _, aggregation in _histogram_component_specs(unit)
        )
    else:
        components = tuple(
            _component(
                name=component_name,
                unit=unit,
                aggregation=aggregation,
                samples=percentile_points[component_name],
            )
            for component_name, _, aggregation in _histogram_component_specs(unit)
        )
    return _window(
        name=name,
        interval=source.interval,
        aggregation=source.aggregation,
        components=components,
        resets=source.verified_reset_timestamps,
        flags=flags,
        boundary_reset_sources=boundary_reset_sources,
    )


def _histogram_flags(sample: HistogramSample) -> set[str]:
    if not sample.buckets:
        return {"missing_histogram_buckets"}
    finite_bounds = [item.upper_bound for item in sample.buckets if item.upper_bound is not None]
    if (
        finite_bounds != sorted(finite_bounds)
        or len(finite_bounds) != len(set(finite_bounds))
        or any(value < 0.0 for value in finite_bounds)
        or any(
            item.upper_bound is None and index != len(sample.buckets) - 1
            for index, item in enumerate(sample.buckets)
        )
    ):
        return {"invalid_histogram_boundaries"}
    counts = [item.cumulative_count for item in sample.buckets]
    if counts != sorted(counts):
        return {"non_monotone_histogram_counts"}
    if counts[-1] != sample.count:
        return {"histogram_count_mismatch"}
    if sample.count == 0:
        return {"empty_histogram"}
    if sample.sum < 0.0:
        return {"negative_histogram_sum"}
    minimum_sum = 0.0
    maximum_sum = 0.0
    maximum_is_bounded = True
    previous_count = 0
    previous_bound = 0.0
    try:
        for bucket in sample.buckets:
            bucket_count = bucket.cumulative_count - previous_count
            minimum_sum += bucket_count * previous_bound
            if bucket.upper_bound is None:
                if bucket_count:
                    maximum_is_bounded = False
            else:
                maximum_sum += bucket_count * bucket.upper_bound
                previous_bound = bucket.upper_bound
            previous_count = bucket.cumulative_count
    except OverflowError:
        return {"non_finite_derived_statistic"}
    if not all(math.isfinite(value) for value in (minimum_sum, maximum_sum)):
        return {"non_finite_derived_statistic"}
    magnitude = max(abs(sample.sum), abs(minimum_sum), abs(maximum_sum), 1.0)
    tolerance = 1e-12 * magnitude
    if sample.sum < minimum_sum - tolerance or (
        maximum_is_bounded and sample.sum > maximum_sum + tolerance
    ):
        return {"histogram_sum_count_mismatch"}
    return set()


def _histogram_boundary_signature(sample: HistogramSample) -> tuple[float | None, ...]:
    return tuple(bucket.upper_bound for bucket in sample.buckets)


def _histogram_quantile(sample: HistogramSample, quantile: float) -> float | None:
    rank = quantile * sample.count
    previous_count = 0
    previous_bound = 0.0
    for bucket in sample.buckets:
        if bucket.cumulative_count < rank:
            previous_count = bucket.cumulative_count
            if bucket.upper_bound is not None:
                previous_bound = bucket.upper_bound
            continue
        if bucket.upper_bound is None:
            return None
        bucket_count = bucket.cumulative_count - previous_count
        if bucket_count <= 0:  # pragma: no cover - monotone search invariant
            return bucket.upper_bound
        fraction = (rank - previous_count) / bucket_count
        return previous_bound + fraction * (bucket.upper_bound - previous_bound)
    return None


def _sum_count_window(
    name: WindowName,
    total: ScalarWindowInput,
    count: ScalarWindowInput,
    unit: str,
    *,
    sum_extra_flags: Iterable[str] = (),
    count_extra_flags: Iterable[str] = (),
    boundary_reset_sources: Iterable[str] = (),
) -> WindowSignalTransform:
    sum_window = _cumulative_window(
        name=name,
        source=total,
        unit=unit,
        delta_name="sum_delta",
        rate_name="sum_rate",
        require_integral=False,
        extra_flags=sum_extra_flags,
    )
    count_window = _cumulative_window(
        name=name,
        source=count,
        unit="count",
        delta_name="count_delta",
        rate_name="count_rate",
        require_integral=True,
        extra_flags=count_extra_flags,
    )
    sum_deltas = _component_by_name(sum_window, "sum_delta")
    count_deltas = _component_by_name(count_window, "count_delta")
    flags = set(sum_window.quality_flags) | set(count_window.quality_flags)
    means: list[ScalarSample] = []
    mean_flags: set[str] = set()
    if sum_deltas.status != "valid" or count_deltas.status != "valid":
        mean_flags.add("source_sum_or_count_series_not_valid")
    else:
        sum_by_interval = {_source_interval_key(item): item for item in sum_deltas.samples}
        count_by_interval = {_source_interval_key(item): item for item in count_deltas.samples}
        if set(sum_by_interval) != set(count_by_interval):
            mean_flags.add("unaligned_sum_count_intervals")
        for interval_key in sorted(set(sum_by_interval) & set(count_by_interval)):
            sum_sample = sum_by_interval[interval_key]
            count_sample = count_by_interval[interval_key]
            denominator = count_sample.value
            if denominator == 0.0:
                mean_flags.add("zero_count")
            else:
                means.append(
                    ScalarSample(
                        timestamp_unix_ns=sum_sample.timestamp_unix_ns,
                        value=sum_sample.value / denominator,
                        source_interval=sum_sample.source_interval,
                    )
                )
    flags.update(mean_flags)
    if not means:
        mean = _unknown_component(
            "mean",
            unit,
            "sum_delta_per_count_delta",
            mean_flags or flags or {"no_aligned_sum_count_samples"},
        )
    elif mean_flags:
        mean = _component(
            name="mean",
            unit=unit,
            aggregation="sum_delta_per_count_delta",
            samples=means,
            status="degraded",
            eligible=False,
            flags=mean_flags,
        )
    else:
        mean = _component(
            name="mean",
            unit=unit,
            aggregation="sum_delta_per_count_delta",
            samples=means,
        )
    return _window(
        name=name,
        interval=total.interval,
        aggregation="cumulative_sum+cumulative_count",
        components=(sum_deltas, count_deltas, mean),
        resets=tuple(
            sorted(
                set(sum_window.verified_reset_timestamps)
                | set(count_window.verified_reset_timestamps)
            )
        ),
        flags=flags,
        boundary_reset_sources=boundary_reset_sources,
    )


def _scalar_boundary_flags(
    baseline: ScalarWindowInput,
    fault: ScalarWindowInput,
) -> tuple[set[str], set[str]]:
    baseline_flags: set[str] = set()
    fault_flags: set[str] = set()
    if baseline.verified_boundary_reset:
        baseline_flags.add("unexpected_baseline_boundary_reset")
    if fault.verified_boundary_reset and (not baseline.samples or not fault.samples):
        fault_flags.add("unverifiable_boundary_reset")
    elif (
        baseline.samples
        and fault.samples
        and fault.samples[0].value < baseline.samples[-1].value
        and not fault.verified_boundary_reset
    ):
        fault_flags.add("unverified_boundary_decrease")
    return baseline_flags, fault_flags


def _histogram_boundary_flags(
    baseline: HistogramWindowInput,
    fault: HistogramWindowInput,
) -> tuple[set[str], set[str]]:
    baseline_flags: set[str] = set()
    fault_flags: set[str] = set()
    if baseline.verified_boundary_reset:
        baseline_flags.add("unexpected_baseline_boundary_reset")
    if fault.verified_boundary_reset and (not baseline.samples or not fault.samples):
        fault_flags.add("unverifiable_boundary_reset")
    elif baseline.samples and fault.samples and not fault.verified_boundary_reset:
        previous = baseline.samples[-1]
        current = fault.samples[0]
        if (
            current.count < previous.count
            or current.sum < previous.sum
            or any(
                current_bucket.cumulative_count < previous_bucket.cumulative_count
                for previous_bucket, current_bucket in zip(
                    previous.buckets,
                    current.buckets,
                    strict=True,
                )
            )
        ):
            fault_flags.add("unverified_boundary_decrease")
    return baseline_flags, fault_flags


def _aggregation_flags(
    baseline: str,
    fault: str,
    *,
    allowed: frozenset[str],
) -> set[str]:
    flags: set[str] = set()
    if baseline != fault:
        flags.add("aggregation_mismatch")
    if baseline not in allowed or fault not in allowed:
        flags.add("unsupported_aggregation")
    return flags


def _unknown_scalar_result(
    *,
    signal_type: SignalType,
    declared_signal_type: str,
    unit: str,
    transform: str,
    baseline: ScalarWindowInput,
    fault: ScalarWindowInput,
    component_specs: Sequence[tuple[str, str, str]],
    flags: Iterable[str],
) -> SignalTransformResult:
    frozen_flags = set(flags)
    sources: tuple[tuple[WindowName, ScalarWindowInput], ...] = (
        ("baseline", baseline),
        ("fault", fault),
    )
    windows = tuple(
        _unknown_window(
            name=name,
            interval=source.interval,
            aggregation=source.aggregation,
            component_specs=component_specs,
            flags=frozen_flags,
        )
        for name, source in sources
    )
    return _build_result(
        signal_type=signal_type,
        declared_signal_type=declared_signal_type,
        unit=unit,
        transform=transform,
        windows=windows,
        source_inputs={"baseline": baseline, "fault": fault},
        extra_flags=frozen_flags,
    )


def _unknown_error_result(
    *,
    baseline: ScalarWindowInput,
    fault: ScalarWindowInput,
    source_inputs: Mapping[str, SourceWindowInput],
    flags: Iterable[str],
) -> SignalTransformResult:
    frozen_flags = set(flags)
    component_specs = (
        ("error_count", "errors", "interval_delta"),
        ("error_ratio", "ratio", "interval_error_ratio"),
        ("request_count", "requests", "interval_delta"),
    )
    sources: tuple[tuple[WindowName, ScalarWindowInput], ...] = (
        ("baseline", baseline),
        ("fault", fault),
    )
    windows = tuple(
        _unknown_window(
            name=name,
            interval=source.interval,
            aggregation=source.aggregation,
            component_specs=component_specs,
            flags=frozen_flags,
        )
        for name, source in sources
    )
    return _build_result(
        signal_type="error",
        declared_signal_type="error",
        unit="ratio",
        transform="error_count_request_ratio",
        windows=windows,
        source_inputs=source_inputs,
        extra_flags=frozen_flags,
    )


def _unknown_window(
    *,
    name: WindowName,
    interval: TimeInterval,
    aggregation: str,
    component_specs: Sequence[tuple[str, str, str]],
    flags: Iterable[str],
) -> WindowSignalTransform:
    frozen_flags = set(flags)
    return _window(
        name=name,
        interval=interval,
        aggregation=aggregation,
        components=tuple(
            _unknown_component(component_name, unit, component_aggregation, frozen_flags)
            for component_name, unit, component_aggregation in component_specs
        ),
        resets=(),
        flags=frozen_flags,
    )


def _component(
    *,
    name: str,
    unit: str,
    aggregation: str,
    samples: Sequence[ScalarSample],
    status: TransformStatus = "valid",
    eligible: bool = True,
    flags: Iterable[str] = (),
) -> TransformedSeries:
    return TransformedSeries(
        name=name,
        unit=unit,
        aggregation=aggregation,
        samples=tuple(samples),
        status=status,
        formal_scoring_eligible=eligible,
        quality_flags=tuple(sorted(set(flags))),
    )


def _unknown_component(
    name: str,
    unit: str,
    aggregation: str,
    flags: Iterable[str],
) -> TransformedSeries:
    return _component(
        name=name,
        unit=unit,
        aggregation=aggregation,
        samples=(),
        status="unknown",
        eligible=False,
        flags=flags,
    )


def _window(
    *,
    name: WindowName,
    interval: TimeInterval,
    aggregation: str,
    components: Sequence[TransformedSeries],
    resets: Iterable[int],
    flags: Iterable[str],
    boundary_reset_sources: Iterable[str] = (),
    exposure_seconds: float | None = None,
    exposure_intervals: Sequence[TimeInterval] = (),
    exposure_provenance_ids: Iterable[str] = (),
    first_event_unix_ns: int | None = None,
    raw_record_count: int = 0,
) -> WindowSignalTransform:
    ordered = tuple(sorted(components, key=lambda item: item.name))
    statuses = {item.status for item in ordered}
    frozen_flags = set(flags)
    if statuses == {"unknown"}:
        status: TransformStatus = "unknown"
    elif statuses == {"valid"} and not frozen_flags:
        status = "valid"
    else:
        status = "degraded"
    return WindowSignalTransform(
        window=name,
        interval=interval,
        source_aggregation=aggregation,
        raw_record_state="recorded" if raw_record_count else "no_record",
        raw_record_count=raw_record_count,
        status=status,
        components=ordered,
        verified_reset_timestamps=tuple(sorted(set(resets))),
        verified_boundary_reset_sources=tuple(sorted(set(boundary_reset_sources))),
        exposure_seconds=exposure_seconds,
        exposure_intervals=tuple(exposure_intervals),
        exposure_provenance_ids=tuple(sorted(set(exposure_provenance_ids))),
        first_event_unix_ns=first_event_unix_ns,
        quality_flags=tuple(sorted(frozen_flags)),
    )


def _build_result(
    *,
    signal_type: SignalType,
    declared_signal_type: str,
    unit: str,
    transform: str,
    windows: Sequence[WindowSignalTransform],
    source_inputs: Mapping[str, SourceWindowInput],
    extra_flags: Iterable[str] = (),
) -> SignalTransformResult:
    ordered = tuple(windows)
    if len(ordered) != 2:
        raise ValueError("signal transform requires baseline and fault windows")
    source_contract = _component_source_contract(transform)
    window_names: tuple[WindowName, WindowName] = ("baseline", "fault")
    window_source_names = {
        window: tuple(
            sorted(
                {
                    source_name
                    for component_sources in source_contract[window].values()
                    for source_name in component_sources
                }
            )
        )
        for window in window_names
    }
    raw_counts = {
        name: sum(
            _raw_record_count(source_inputs[source_name])
            for source_name in window_source_names[name]
        )
        for name in window_names
    }
    raw_states = {
        name: (
            "recorded"
            if any(
                _raw_record_state(source_inputs[source_name]) == "recorded"
                for source_name in window_source_names[name]
            )
            else "no_record"
        )
        for name in window_names
    }
    typed_windows = tuple(
        WindowSignalTransform.model_validate(
            {
                **window.model_dump(mode="json"),
                "raw_record_state": raw_states[window.window],
                "raw_record_count": raw_counts[window.window],
                "component_raw_attributions": [
                    _build_component_raw_attribution(
                        component_name=component.name,
                        direct_source_names=source_contract[window.window][component.name],
                        source_inputs=source_inputs,
                    ).model_dump(mode="json")
                    for component in window.components
                ],
            }
        )
        for window in ordered
    )
    statuses = {item.status for item in typed_windows}
    if statuses == {"unknown"}:
        status: TransformStatus = "unknown"
    elif statuses == {"valid"}:
        status = "valid"
    else:
        status = "degraded"
    eligible_names = [
        {component.name for component in window.components if component.formal_scoring_eligible}
        for window in typed_windows
    ]
    formally_eligible = bool(eligible_names[0] & eligible_names[1])
    flags = set(extra_flags)
    for window in typed_windows:
        flags.update(window.quality_flags)
        for component in window.components:
            flags.update(component.quality_flags)
    if signal_type == "unknown":
        status = "unknown"
        formally_eligible = False
    unsigned: dict[str, object] = {
        "schema_version": SIGNAL_TRANSFORM_RESULT_SCHEMA_VERSION,
        "semantics_version": SIGNAL_TRANSFORM_SEMANTICS_VERSION,
        "signal_type": signal_type,
        "declared_signal_type": declared_signal_type,
        "unit": unit,
        "transform": transform,
        "windows": [item.model_dump(mode="json") for item in typed_windows],
        "status": status,
        "formal_scoring_eligible": formally_eligible,
        "quality_flags": tuple(sorted(flags)),
        "source_input_sha256": compute_source_input_sha256(source_inputs),
        "adapter_semantics_sha256": SIGNAL_TRANSFORM_SEMANTICS_SHA256,
    }
    result = SignalTransformResult.model_validate(
        {
            **unsigned,
            "result_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("signal transform result verification failed")
    return result


def _raw_record_count(source: SourceWindowInput) -> int:
    if isinstance(source, SparseEventWindowInput):
        return len(source.event_timestamps_unix_ns)
    return len(source.samples)


def _raw_record_state(source: SourceWindowInput) -> Literal["recorded", "no_record"]:
    if isinstance(source, SparseEventWindowInput) and source.exposure_intervals:
        return "recorded"
    return "recorded" if _raw_record_count(source) else "no_record"


def _build_component_raw_attribution(
    *,
    component_name: str,
    direct_source_names: tuple[str, ...],
    source_inputs: Mapping[str, SourceWindowInput],
) -> ComponentRawAttribution:
    direct_sources = tuple(source_inputs[name] for name in direct_source_names)
    count = sum(_raw_record_count(source) for source in direct_sources)
    state: Literal["recorded", "no_record"] = (
        "recorded"
        if any(_raw_record_state(source) == "recorded" for source in direct_sources)
        else "no_record"
    )
    return ComponentRawAttribution(
        component_name=component_name,
        direct_source_names=direct_source_names,
        raw_record_state=state,
        raw_record_count=count,
    )


def _component_source_contract(
    transform: str,
) -> dict[WindowName, dict[str, tuple[str, ...]]]:
    simple_components: dict[str, tuple[str, ...]] = {
        "counter_delta_and_rate": ("delta", "rate"),
        "cumulative_sum_delta_and_rate": ("sum_delta", "sum_rate"),
        "cumulative_count_delta_and_rate": ("count_delta", "count_rate"),
        "gauge_identity": ("value",),
        "latency_identity": ("latency",),
        "event_exposure_rate": ("event_count", "event_rate"),
        "cumulative_histogram_percentiles": ("mean", "p95", "p99"),
        "undefined_signal_transform": ("value",),
    }
    if transform in simple_components:
        components = simple_components[transform]
        return {
            "baseline": {name: ("baseline",) for name in components},
            "fault": {name: ("fault",) for name in components},
        }
    if transform == "error_count_request_ratio":
        return {
            "baseline": {
                "error_count": ("baseline_errors",),
                "error_ratio": ("baseline_errors", "baseline_requests"),
                "request_count": ("baseline_requests",),
            },
            "fault": {
                "error_count": ("fault_errors",),
                "error_ratio": ("fault_errors", "fault_requests"),
                "request_count": ("fault_requests",),
            },
        }
    if transform == "sum_count_mean":
        return {
            "baseline": {
                "count_delta": ("baseline_count",),
                "mean": ("baseline_count", "baseline_sum"),
                "sum_delta": ("baseline_sum",),
            },
            "fault": {
                "count_delta": ("fault_count",),
                "mean": ("fault_count", "fault_sum"),
                "sum_delta": ("fault_sum",),
            },
        }
    raise ValueError("transform has no explicit component source contract")


def _component_by_name(
    window: WindowSignalTransform,
    name: str,
) -> TransformedSeries:
    for component in window.components:
        if component.name == name:
            return component
    raise AssertionError(f"missing transformed component: {name}")


def _source_interval_key(sample: ScalarSample) -> tuple[int, int]:
    if sample.source_interval is None:  # pragma: no cover - construction invariant
        raise AssertionError("interval-derived sample is missing its source interval")
    return (
        sample.source_interval.start_unix_ns,
        sample.source_interval.end_unix_ns,
    )


def _histogram_component_specs(unit: str) -> tuple[tuple[str, str, str], ...]:
    return (
        ("mean", unit, "sum_per_count"),
        ("p95", unit, "p95"),
        ("p99", unit, "p99"),
    )


def _require_transform_contract(result: SignalTransformResult) -> None:
    key = (result.signal_type, result.declared_signal_type, result.transform)
    specs: tuple[tuple[str, str, str], ...]
    expected_source: str | None
    reset_sources: set[str]
    if key == ("counter", "counter", "counter_delta_and_rate"):
        specs = (
            ("delta", result.unit, "interval_delta"),
            ("rate", f"{result.unit}/second", "per_second_rate"),
        )
        expected_source = "cumulative"
        reset_sources = {"value"}
    elif key == ("gauge", "gauge", "gauge_identity"):
        specs = (("value", result.unit, "instantaneous"),)
        expected_source = "instantaneous"
        reset_sources = set()
    elif key == ("latency", "latency", "latency_identity"):
        aggregation = result.windows[0].source_aggregation
        specs = (("latency", result.unit, aggregation),)
        expected_source = aggregation
        reset_sources = set()
    elif key == ("error", "error", "error_count_request_ratio"):
        if result.unit != "ratio":
            raise ValueError("error transform result unit must be ratio")
        specs = (
            ("error_count", "errors", "interval_delta"),
            ("error_ratio", "ratio", "interval_error_ratio"),
            ("request_count", "requests", "interval_delta"),
        )
        expected_source = "cumulative_count"
        reset_sources = {"errors", "requests"}
    elif key == ("sparse", "sparse", "event_exposure_rate"):
        specs = (
            ("event_count", result.unit, "window_count"),
            ("event_rate", f"{result.unit}/second", "exposure_rate"),
        )
        expected_source = "event_occurrence"
        reset_sources = set()
    elif key == ("histogram", "histogram", "cumulative_histogram_percentiles"):
        specs = _histogram_component_specs(result.unit)
        expected_source = "cumulative_buckets"
        reset_sources = {"histogram"}
    elif key == ("sum", "sum", "cumulative_sum_delta_and_rate"):
        specs = (
            ("sum_delta", result.unit, "interval_delta"),
            ("sum_rate", f"{result.unit}/second", "per_second_rate"),
        )
        expected_source = "cumulative_sum"
        reset_sources = {"value"}
    elif key == ("count", "count", "cumulative_count_delta_and_rate"):
        specs = (
            ("count_delta", result.unit, "interval_delta"),
            ("count_rate", f"{result.unit}/second", "per_second_rate"),
        )
        expected_source = "cumulative_count"
        reset_sources = {"value"}
    elif key == ("sum", "sum+count", "sum_count_mean"):
        specs = (
            ("count_delta", "count", "interval_delta"),
            ("mean", result.unit, "sum_delta_per_count_delta"),
            ("sum_delta", result.unit, "interval_delta"),
        )
        expected_source = "cumulative_sum+cumulative_count"
        reset_sources = {"count", "sum"}
    elif result.signal_type == "unknown" and result.transform == "undefined_signal_transform":
        specs = (("value", result.unit, "undefined"),)
        expected_source = None
        reset_sources = set()
    else:
        raise ValueError("signal type, declaration, and transform contract are inconsistent")

    expected_components = tuple(sorted(specs))
    source_contract = _component_source_contract(result.transform)
    for window in result.windows:
        actual_components = tuple(
            (component.name, component.unit, component.aggregation)
            for component in window.components
        )
        if actual_components != expected_components:
            raise ValueError("transformed component contract is inconsistent")
        actual_sources = {
            attribution.component_name: attribution.direct_source_names
            for attribution in window.component_raw_attributions
        }
        if actual_sources != source_contract[window.window]:
            raise ValueError("component raw source attribution is inconsistent")
        if window.status == "valid" and (
            expected_source is None or window.source_aggregation != expected_source
        ):
            raise ValueError("valid window source aggregation is inconsistent")
        if set(window.verified_boundary_reset_sources) - reset_sources:
            raise ValueError("transform cannot record resets for this source")
        if not reset_sources and window.verified_reset_timestamps:
            raise ValueError("transform cannot record resets")
        if result.signal_type == "sparse" and window.status == "valid":
            expected_exposure = (
                sum(
                    interval.end_unix_ns - interval.start_unix_ns
                    for interval in window.exposure_intervals
                )
                / NANOSECONDS_PER_SECOND
            )
            if window.exposure_seconds != expected_exposure:
                raise ValueError("sparse exposure is inconsistent with its window")
            if not window.exposure_intervals or not window.exposure_provenance_ids:
                raise ValueError("valid sparse exposure requires provenance")
        elif result.signal_type != "sparse" and (
            window.exposure_seconds is not None
            or window.exposure_intervals
            or window.exposure_provenance_ids
            or window.first_event_unix_ns is not None
        ):
            raise ValueError("non-sparse transform cannot record sparse-event fields")


def _replay_transform(
    result: SignalTransformResult,
    source_inputs: Mapping[str, SourceWindowInput],
) -> SignalTransformResult:
    if result.transform == "counter_delta_and_rate":
        _require_source_names(source_inputs, {"baseline", "fault"})
        return transform_counter(
            unit=result.unit,
            baseline=_scalar_source(source_inputs, "baseline"),
            fault=_scalar_source(source_inputs, "fault"),
        )
    if result.transform == "cumulative_sum_delta_and_rate":
        _require_source_names(source_inputs, {"baseline", "fault"})
        return transform_sum(
            unit=result.unit,
            baseline=_scalar_source(source_inputs, "baseline"),
            fault=_scalar_source(source_inputs, "fault"),
        )
    if result.transform == "cumulative_count_delta_and_rate":
        _require_source_names(source_inputs, {"baseline", "fault"})
        return transform_count(
            unit=result.unit,
            baseline=_scalar_source(source_inputs, "baseline"),
            fault=_scalar_source(source_inputs, "fault"),
        )
    if result.transform == "gauge_identity":
        _require_source_names(source_inputs, {"baseline", "fault"})
        return transform_gauge(
            unit=result.unit,
            baseline=_scalar_source(source_inputs, "baseline"),
            fault=_scalar_source(source_inputs, "fault"),
        )
    if result.transform == "latency_identity":
        _require_source_names(source_inputs, {"baseline", "fault"})
        return transform_latency(
            unit=result.unit,
            baseline=_scalar_source(source_inputs, "baseline"),
            fault=_scalar_source(source_inputs, "fault"),
        )
    if result.transform == "error_count_request_ratio":
        names = {
            "baseline_errors",
            "baseline_requests",
            "fault_errors",
            "fault_requests",
        }
        _require_source_names(source_inputs, names)
        return transform_error(
            baseline_errors=_scalar_source(source_inputs, "baseline_errors"),
            baseline_requests=_scalar_source(source_inputs, "baseline_requests"),
            fault_errors=_scalar_source(source_inputs, "fault_errors"),
            fault_requests=_scalar_source(source_inputs, "fault_requests"),
        )
    if result.transform == "event_exposure_rate":
        _require_source_names(source_inputs, {"baseline", "fault"})
        return transform_sparse_events(
            unit=result.unit,
            baseline=_sparse_source(source_inputs, "baseline"),
            fault=_sparse_source(source_inputs, "fault"),
        )
    if result.transform == "cumulative_histogram_percentiles":
        _require_source_names(source_inputs, {"baseline", "fault"})
        return transform_histogram(
            unit=result.unit,
            baseline=_histogram_source(source_inputs, "baseline"),
            fault=_histogram_source(source_inputs, "fault"),
        )
    if result.transform == "sum_count_mean":
        names = {"baseline_sum", "baseline_count", "fault_sum", "fault_count"}
        _require_source_names(source_inputs, names)
        return transform_sum_count(
            unit=result.unit,
            baseline_sum=_scalar_source(source_inputs, "baseline_sum"),
            baseline_count=_scalar_source(source_inputs, "baseline_count"),
            fault_sum=_scalar_source(source_inputs, "fault_sum"),
            fault_count=_scalar_source(source_inputs, "fault_count"),
        )
    if result.transform == "undefined_signal_transform":
        _require_source_names(source_inputs, {"baseline", "fault"})
        return transform_unknown(
            declared_signal_type=result.declared_signal_type,
            unit=result.unit,
            baseline=_scalar_source(source_inputs, "baseline"),
            fault=_scalar_source(source_inputs, "fault"),
        )
    raise ValueError("unsupported signal transform replay")


def _require_source_names(
    source_inputs: Mapping[str, SourceWindowInput], expected: set[str]
) -> None:
    if set(source_inputs) != expected:
        raise ValueError("source input names do not match the transform contract")


def _scalar_source(source_inputs: Mapping[str, SourceWindowInput], name: str) -> ScalarWindowInput:
    source = source_inputs[name]
    if not isinstance(source, ScalarWindowInput):
        raise TypeError(f"source input {name} must be scalar")
    return source


def _sparse_source(
    source_inputs: Mapping[str, SourceWindowInput], name: str
) -> SparseEventWindowInput:
    source = source_inputs[name]
    if not isinstance(source, SparseEventWindowInput):
        raise TypeError(f"source input {name} must be sparse events")
    return source


def _histogram_source(
    source_inputs: Mapping[str, SourceWindowInput], name: str
) -> HistogramWindowInput:
    source = source_inputs[name]
    if not isinstance(source, HistogramWindowInput):
        raise TypeError(f"source input {name} must be a histogram")
    return source
