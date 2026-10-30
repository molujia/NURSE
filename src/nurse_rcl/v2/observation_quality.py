"""Robust, auditable observation summaries for NURSE V2.

This layer consumes the comparable signal components produced by
``signal_transforms``.  It does not bind entities, group events, or inspect
labels.  Every statistic and quality discount is deterministic and replayable.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.observation_schemas import (
    EvidenceState,
    SignalType,
    WindowSummary,
)
from nurse_rcl.v2.signal_transforms import (
    ComponentRawAttribution,
    SignalTransformResult,
    SourceWindowInput,
    TransformedSeries,
    WindowSignalTransform,
)

OBSERVATION_QUALITY_RESULT_SCHEMA_VERSION: Literal["nurse.v2.observation-quality-result/3"] = (
    "nurse.v2.observation-quality-result/3"
)
OBSERVATION_QUALITY_SEMANTICS_VERSION: Literal["nurse-v2-observation-quality/5"] = (
    "nurse-v2-observation-quality/5"
)

BASELINE_STEP_MINIMUM_POSITIVE_STEPS = 3
BASELINE_STEP_MINIMUM_SUPPORT = 2
BASELINE_STEP_RELATIVE_TOLERANCE = 1e-6
BASELINE_STEP_ULP_TOLERANCE = 8

_OBSERVATION_QUALITY_SEMANTICS: dict[str, object] = {
    "version": OBSERVATION_QUALITY_SEMANTICS_VERSION,
    "location": "median",
    "scale": {
        "mad_multiplier": 1.4826,
        "relative_median_floor": 0.05,
        "requires_positive_absolute_resolution": True,
        "normal_only_resolution_estimate": (
            "median_positive_adjacent_step_with_three_steps_and_relative_ulp_support"
        ),
        "normal_only_resolution_parameters": {
            "minimum_positive_steps": BASELINE_STEP_MINIMUM_POSITIVE_STEPS,
            "minimum_support": BASELINE_STEP_MINIMUM_SUPPORT,
            "relative_tolerance": BASELINE_STEP_RELATIVE_TOLERANCE,
            "ulp_tolerance": BASELINE_STEP_ULP_TOLERANCE,
        },
    },
    "anomaly": {
        "z": "abs(median_fault-median_baseline)/baseline_scale",
        "mapping": "-expm1(-z/anomaly_mapping_divisor)",
        "default_abnormal_z_threshold": 3.0,
    },
    "derived_value_validation": {
        "comparison": "canonical_json_bytes_after_positive_zero_normalization",
        "fixed_absolute_tolerance": None,
        "negative_zero_policy": "reject_derived_and_emit_positive_zero",
    },
    "coverage": {
        "instantaneous": "union_of_sample_cadence_intervals_over_window",
        "interval": "union_of_source_intervals_capped_per_cadence_over_window",
        "sparse": "declared_window_exposure_over_window",
        "combined": "minimum_baseline_fault_coverage_times_objective_factors",
    },
    "unknown_policy": "never_convert_missing_or_invalid_evidence_to_normal",
}
OBSERVATION_QUALITY_SEMANTICS_SHA256 = sha256_hex(
    canonical_json_bytes(_OBSERVATION_QUALITY_SEMANTICS)
)

ResolutionBasis = Literal["collection_precision", "unit_manifest"]
ResolutionSource = Literal[
    "collection_precision",
    "unit_manifest",
    "discrete_count",
    "baseline_median_positive_step",
    "unresolved",
]
BindingStatus = Literal["bound", "unbound", "conflicted"]
BaselineContamination = Literal["none", "suspected", "confirmed"]


class AbsoluteResolutionDeclaration(StrictFrozenModel):
    """Positive resolution declared by collection or a versioned unit manifest."""

    signal_type: SignalType
    component_name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    unit: str = Field(min_length=1)
    aggregation: str = Field(min_length=1)
    value: float = Field(gt=0.0, allow_inf_nan=False)
    basis: ResolutionBasis
    manifest_id: str = Field(min_length=1)
    manifest_version: str = Field(min_length=1)
    provenance_id: str = Field(min_length=1)


class ObservationQualityContext(StrictFrozenModel):
    """Objective, label-free quality inputs for one transformed signal."""

    expected_cadence_ns: int = Field(gt=0)
    binding_status: BindingStatus
    baseline_contamination: BaselineContamination
    clock_uncertainty_ns: int = Field(ge=0)
    late_data_observed: bool
    adapter_manifest_id: str = Field(min_length=1)
    adapter_manifest_version: str = Field(min_length=1)
    absolute_resolution: AbsoluteResolutionDeclaration | None = None


class ObservationQualityConfiguration(StrictFrozenModel):
    """Versioned development defaults; none of these values are label-fitted."""

    minimum_valid_points: int = Field(default=5, ge=1)
    minimum_coverage_fraction: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    mad_multiplier: float = Field(default=1.4826, gt=0.0, allow_inf_nan=False)
    relative_median_floor: float = Field(
        default=0.05,
        gt=0.0,
        allow_inf_nan=False,
    )
    anomaly_mapping_divisor: float = Field(
        default=3.0,
        gt=0.0,
        allow_inf_nan=False,
    )
    abnormal_z_threshold: float = Field(
        default=3.0,
        gt=0.0,
        allow_inf_nan=False,
    )
    suspected_contamination_factor: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    conflicted_binding_factor: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    late_data_factor: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )


class QualityFactor(StrictFrozenModel):
    """One transparent multiplicative factor in overall coverage quality."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    value: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    reason: str = Field(min_length=1)


@dataclass(frozen=True)
class _CanonicalQualityDerivation:
    """Every deterministic field derived from canonical observation inputs."""

    baseline_summary: WindowSummary
    fault_summary: WindowSummary
    baseline_summary_sufficient: bool
    fault_summary_sufficient: bool
    absolute_resolution: float | None
    absolute_resolution_source: ResolutionSource
    baseline_scale: float | None
    robust_z: float | None
    anomaly_strength: float | None
    evidence_state: EvidenceState
    coverage_quality: float
    minimum_valid_points: int
    formal_scoring_eligible: bool
    quality_factors: tuple[QualityFactor, ...]
    quality_flags: tuple[str, ...]


class ObservationQualityResult(StrictFrozenModel):
    """Content-addressed robust summary and anomaly judgment for one component."""

    schema_version: Literal["nurse.v2.observation-quality-result/3"] = (
        OBSERVATION_QUALITY_RESULT_SCHEMA_VERSION
    )
    semantics_version: Literal["nurse-v2-observation-quality/5"] = (
        OBSERVATION_QUALITY_SEMANTICS_VERSION
    )
    semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    component_name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    signal_type: SignalType
    unit: str = Field(min_length=1)
    aggregation: str = Field(min_length=1)
    transform_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_raw_record_state: Literal["recorded", "no_record"]
    fault_raw_record_state: Literal["recorded", "no_record"]
    baseline_direct_source_names: tuple[str, ...] = Field(min_length=1)
    fault_direct_source_names: tuple[str, ...] = Field(min_length=1)
    baseline_raw_record_count: int = Field(ge=0)
    fault_raw_record_count: int = Field(ge=0)
    upstream_formal_scoring_eligible: bool
    upstream_quality_flags: tuple[str, ...]
    resolution_estimation_nonfinite: bool
    context: ObservationQualityContext
    configuration: ObservationQualityConfiguration
    baseline_summary: WindowSummary
    fault_summary: WindowSummary
    baseline_summary_sufficient: bool
    fault_summary_sufficient: bool
    absolute_resolution: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
    )
    absolute_resolution_source: ResolutionSource
    baseline_scale: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    robust_z: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    anomaly_strength: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    evidence_state: EvidenceState
    coverage_quality: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    minimum_valid_points: int = Field(ge=1)
    formal_scoring_eligible: bool
    quality_factors: tuple[QualityFactor, ...]
    quality_flags: tuple[str, ...]
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_derived_state(self) -> Self:
        if self.semantics_sha256 != OBSERVATION_QUALITY_SEMANTICS_SHA256:
            raise ValueError("observation quality semantics hash is inconsistent")
        if self.quality_factors != tuple(sorted(self.quality_factors, key=lambda item: item.name)):
            raise ValueError("quality factors must be canonically ordered")
        factor_names = tuple(item.name for item in self.quality_factors)
        if len(factor_names) != len(set(factor_names)):
            raise ValueError("quality factor names must be unique")
        if self.quality_flags != tuple(sorted(set(self.quality_flags))):
            raise ValueError("quality flags must be sorted and unique")
        if self.upstream_quality_flags != tuple(sorted(set(self.upstream_quality_flags))):
            raise ValueError("upstream quality flags must be sorted and unique")
        for source_names in (
            self.baseline_direct_source_names,
            self.fault_direct_source_names,
        ):
            if source_names != tuple(sorted(set(source_names))):
                raise ValueError("direct source names must be sorted and unique")
        if (
            self.baseline_raw_record_state == "no_record" and self.baseline_raw_record_count != 0
        ) or (self.fault_raw_record_state == "no_record" and self.fault_raw_record_count != 0):
            raise ValueError("raw record state and count are inconsistent")
        resolved = self.absolute_resolution_source != "unresolved"
        if resolved != (self.absolute_resolution is not None):
            raise ValueError("resolution value and source are inconsistent")
        declaration = self.context.absolute_resolution
        if declaration is not None and (
            self.absolute_resolution != declaration.value
            or self.absolute_resolution_source != declaration.basis
            or self.signal_type != declaration.signal_type
            or self.component_name != declaration.component_name
            or self.unit != declaration.unit
            or self.aggregation != declaration.aggregation
            or self.context.adapter_manifest_id != declaration.manifest_id
            or self.context.adapter_manifest_version != declaration.manifest_version
        ):
            raise ValueError("declared absolute resolution is inconsistent")
        expected = _derive_canonical_quality(
            baseline_summary=self.baseline_summary,
            fault_summary=self.fault_summary,
            baseline_raw_record_state=self.baseline_raw_record_state,
            fault_raw_record_state=self.fault_raw_record_state,
            resolution=self.absolute_resolution,
            resolution_source=self.absolute_resolution_source,
            upstream_eligible=self.upstream_formal_scoring_eligible,
            upstream_flags=self.upstream_quality_flags,
            resolution_estimation_nonfinite=self.resolution_estimation_nonfinite,
            context=self.context,
            configuration=self.configuration,
            signal_type=self.signal_type,
            aggregation=self.aggregation,
        )
        if not _canonical_json_equal(
            self.baseline_summary.model_dump(mode="json"),
            expected.baseline_summary.model_dump(mode="json"),
        ):
            raise ValueError("baseline summary is not canonically derived")
        if not _canonical_json_equal(
            self.fault_summary.model_dump(mode="json"),
            expected.fault_summary.model_dump(mode="json"),
        ):
            raise ValueError("fault summary is not canonically derived")
        if not _canonical_json_equal(
            self.absolute_resolution,
            expected.absolute_resolution,
        ):
            raise ValueError("resolution is not canonically derived")
        if not _canonical_json_equal(
            [item.model_dump(mode="json") for item in self.quality_factors],
            [item.model_dump(mode="json") for item in expected.quality_factors],
        ):
            raise ValueError("quality factors are inconsistent with objective inputs")
        if not _canonical_json_equal(self.coverage_quality, expected.coverage_quality):
            raise ValueError("coverage quality is inconsistent with quality factors")
        if self.minimum_valid_points != expected.minimum_valid_points:
            raise ValueError("minimum valid points are inconsistent with signal semantics")
        if (
            self.baseline_summary_sufficient != expected.baseline_summary_sufficient
            or self.fault_summary_sufficient != expected.fault_summary_sufficient
        ):
            raise ValueError("summary sufficiency is inconsistent with valid point counts")
        if not _canonical_json_equal(self.baseline_scale, expected.baseline_scale):
            raise ValueError("baseline scale is inconsistent with robust summaries")
        if not _canonical_json_equal(self.robust_z, expected.robust_z):
            raise ValueError("robust z is inconsistent with robust summaries")
        if not _canonical_json_equal(self.anomaly_strength, expected.anomaly_strength):
            raise ValueError("anomaly strength is inconsistent with robust z")
        if self.evidence_state != expected.evidence_state:
            raise ValueError("evidence state is inconsistent with canonical inputs")
        if self.formal_scoring_eligible != expected.formal_scoring_eligible:
            raise ValueError("formal scoring eligibility is inconsistent with canonical inputs")
        if self.quality_flags != expected.quality_flags:
            raise ValueError("quality flags are inconsistent with canonical inputs")
        return self

    def verify(self) -> bool:
        try:
            document = self.model_dump(mode="json")
            type(self).model_validate(document)
        except (TypeError, ValueError):
            return False
        unsigned = {key: value for key, value in document.items() if key != "result_sha256"}
        return (
            self.semantics_sha256 == OBSERVATION_QUALITY_SEMANTICS_SHA256
            and self.result_sha256 == sha256_hex(canonical_json_bytes(unsigned))
        )

    def verify_transform(self, *, transform_result: SignalTransformResult) -> bool:
        """Replay from the exact upstream transform and compare every field."""

        try:
            replayed = summarize_transformed_component(
                transform_result=transform_result,
                component_name=self.component_name,
                context=self.context,
                configuration=self.configuration,
            )
        except (KeyError, TypeError, ValueError):
            return False
        return replayed == self

    def verify_source_inputs(
        self,
        *,
        transform_result: SignalTransformResult,
        source_inputs: Mapping[str, SourceWindowInput],
    ) -> bool:
        """Replay raw telemetry through transform and quality boundaries."""

        if (
            self.source_input_sha256 != transform_result.source_input_sha256
            or not transform_result.verify_source_inputs(source_inputs)
        ):
            return False
        return self.verify_transform(transform_result=transform_result)


def summarize_transformed_component(
    *,
    transform_result: SignalTransformResult,
    component_name: str,
    context: ObservationQualityContext,
    configuration: ObservationQualityConfiguration | None = None,
) -> ObservationQualityResult:
    """Summarize one comparable B/F component without labels or LLM confidence."""

    if not transform_result.verify():
        raise ValueError("signal transform result failed content verification")
    config = configuration or ObservationQualityConfiguration()
    baseline_window, fault_window = transform_result.windows
    baseline_component = _component(baseline_window, component_name)
    fault_component = _component(fault_window, component_name)
    baseline_attribution = _raw_attribution(baseline_window, component_name)
    fault_attribution = _raw_attribution(fault_window, component_name)
    if baseline_component.unit != fault_component.unit:
        raise ValueError("baseline and fault component units differ")
    if baseline_component.aggregation != fault_component.aggregation:
        raise ValueError("baseline and fault component aggregations differ")

    baseline_summary = _window_summary(
        window=baseline_window,
        component=baseline_component,
        expected_cadence_ns=context.expected_cadence_ns,
    )
    fault_summary = _window_summary(
        window=fault_window,
        component=fault_component,
        expected_cadence_ns=context.expected_cadence_ns,
    )
    resolution, resolution_source, resolution_estimation_nonfinite = _resolve_absolute_resolution(
        baseline_component=baseline_component,
        signal_type=transform_result.signal_type,
        component_name=component_name,
        context=context,
    )
    upstream_eligible = (
        baseline_component.status == "valid"
        and fault_component.status == "valid"
        and baseline_component.formal_scoring_eligible
        and fault_component.formal_scoring_eligible
    )
    upstream_flags = tuple(
        sorted(set(baseline_component.quality_flags) | set(fault_component.quality_flags))
    )
    derived = _derive_canonical_quality(
        baseline_summary=baseline_summary,
        fault_summary=fault_summary,
        baseline_raw_record_state=baseline_attribution.raw_record_state,
        fault_raw_record_state=fault_attribution.raw_record_state,
        resolution=resolution,
        resolution_source=resolution_source,
        upstream_eligible=upstream_eligible,
        upstream_flags=upstream_flags,
        resolution_estimation_nonfinite=resolution_estimation_nonfinite,
        context=context,
        configuration=config,
        signal_type=transform_result.signal_type,
        aggregation=baseline_component.aggregation,
    )

    unsigned: dict[str, object] = {
        "schema_version": OBSERVATION_QUALITY_RESULT_SCHEMA_VERSION,
        "semantics_version": OBSERVATION_QUALITY_SEMANTICS_VERSION,
        "semantics_sha256": OBSERVATION_QUALITY_SEMANTICS_SHA256,
        "component_name": component_name,
        "signal_type": transform_result.signal_type,
        "unit": baseline_component.unit,
        "aggregation": baseline_component.aggregation,
        "transform_result_sha256": transform_result.result_sha256,
        "source_input_sha256": transform_result.source_input_sha256,
        "baseline_raw_record_state": baseline_attribution.raw_record_state,
        "fault_raw_record_state": fault_attribution.raw_record_state,
        "baseline_direct_source_names": baseline_attribution.direct_source_names,
        "fault_direct_source_names": fault_attribution.direct_source_names,
        "baseline_raw_record_count": baseline_attribution.raw_record_count,
        "fault_raw_record_count": fault_attribution.raw_record_count,
        "upstream_formal_scoring_eligible": upstream_eligible,
        "upstream_quality_flags": upstream_flags,
        "resolution_estimation_nonfinite": resolution_estimation_nonfinite,
        "context": context.model_dump(mode="json"),
        "configuration": config.model_dump(mode="json"),
        "baseline_summary": derived.baseline_summary.model_dump(mode="json"),
        "fault_summary": derived.fault_summary.model_dump(mode="json"),
        "baseline_summary_sufficient": derived.baseline_summary_sufficient,
        "fault_summary_sufficient": derived.fault_summary_sufficient,
        "absolute_resolution": derived.absolute_resolution,
        "absolute_resolution_source": derived.absolute_resolution_source,
        "baseline_scale": derived.baseline_scale,
        "robust_z": derived.robust_z,
        "anomaly_strength": derived.anomaly_strength,
        "evidence_state": derived.evidence_state,
        "coverage_quality": derived.coverage_quality,
        "minimum_valid_points": derived.minimum_valid_points,
        "formal_scoring_eligible": derived.formal_scoring_eligible,
        "quality_factors": [item.model_dump(mode="json") for item in derived.quality_factors],
        "quality_flags": derived.quality_flags,
    }
    result = ObservationQualityResult.model_validate(
        {
            **unsigned,
            "result_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("observation quality result verification failed")
    return result


def _component(window: WindowSignalTransform, name: str) -> TransformedSeries:
    for component in window.components:
        if component.name == name:
            return component
    raise ValueError(f"missing transformed component: {name}")


def _raw_attribution(
    window: WindowSignalTransform,
    component_name: str,
) -> ComponentRawAttribution:
    for attribution in window.component_raw_attributions:
        if attribution.component_name == component_name:
            return attribution
    raise ValueError(f"missing raw attribution for component: {component_name}")


def _minimum_valid_points(*, signal_type: SignalType, aggregation: str, configured: int) -> int:
    if signal_type == "sparse" and aggregation in {"window_count", "exposure_rate"}:
        return 1
    return configured


def _window_summary(
    *,
    window: WindowSignalTransform,
    component: TransformedSeries,
    expected_cadence_ns: int,
) -> WindowSummary:
    if not component.samples:
        return WindowSummary(
            record_state="no_record",
            valid_point_count=0,
            coverage_fraction=0.0,
            median=None,
            mad=None,
            minimum=None,
            maximum=None,
            event_count=None,
        )
    values = tuple(sample.value for sample in component.samples)
    location = float(statistics.median(values))
    deviations = tuple(abs(value - location) for value in values)
    mad = float(statistics.median(deviations))
    statistics_are_finite = (
        math.isfinite(location)
        and math.isfinite(mad)
        and all(math.isfinite(value) for value in deviations)
    )
    event_count: int | None = None
    if component.name == "event_count" and len(values) == 1 and values[0].is_integer():
        event_count = int(values[0])
    return WindowSummary(
        record_state="observed",
        valid_point_count=len(values),
        coverage_fraction=_coverage_fraction(
            window=window,
            component=component,
            expected_cadence_ns=expected_cadence_ns,
        ),
        median=location if statistics_are_finite else None,
        mad=mad if statistics_are_finite else None,
        minimum=min(values) if statistics_are_finite else None,
        maximum=max(values) if statistics_are_finite else None,
        event_count=event_count,
    )


def _coverage_fraction(
    *,
    window: WindowSignalTransform,
    component: TransformedSeries,
    expected_cadence_ns: int,
) -> float:
    width = window.interval.end_unix_ns - window.interval.start_unix_ns
    if window.exposure_seconds is not None and component.aggregation in {
        "exposure_rate",
        "window_count",
    }:
        return min(
            1.0,
            window.exposure_seconds * 1_000_000_000 / width,
        )
    source_intervals = tuple(
        sample.source_interval for sample in component.samples if sample.source_interval is not None
    )
    if source_intervals:
        credited_intervals = tuple(
            (
                interval.start_unix_ns,
                min(
                    interval.end_unix_ns,
                    interval.start_unix_ns + expected_cadence_ns,
                ),
            )
            for interval in source_intervals
        )
        covered = _union_pair_width(credited_intervals)
        return min(1.0, covered / width)
    credited_samples = tuple(
        (
            sample.timestamp_unix_ns,
            min(
                window.interval.end_unix_ns,
                sample.timestamp_unix_ns + expected_cadence_ns,
            ),
        )
        for sample in component.samples
    )
    return min(1.0, _union_pair_width(credited_samples) / width)


def _union_pair_width(intervals: Sequence[tuple[int, int]]) -> int:
    ordered = sorted(intervals, key=lambda item: item[0])
    start, end = ordered[0]
    total = 0
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _resolve_absolute_resolution(
    *,
    baseline_component: TransformedSeries,
    signal_type: SignalType,
    component_name: str,
    context: ObservationQualityContext,
) -> tuple[float | None, ResolutionSource, bool]:
    declaration = context.absolute_resolution
    if declaration is not None:
        if declaration.unit != baseline_component.unit:
            raise ValueError("absolute resolution unit does not match component unit")
        if (
            declaration.signal_type != signal_type
            or declaration.component_name != component_name
            or declaration.aggregation != baseline_component.aggregation
            or declaration.manifest_id != context.adapter_manifest_id
            or declaration.manifest_version != context.adapter_manifest_version
        ):
            raise ValueError("resolution declaration does not exactly match signal binding")
        return declaration.value, declaration.basis, False
    if baseline_component.aggregation in {"window_count"} or baseline_component.name in {
        "count_delta",
        "error_count",
        "event_count",
        "request_count",
    }:
        return 1.0, "discrete_count", False
    values = sorted({sample.value for sample in baseline_component.samples})
    steps = [right - left for left, right in pairwise(values)]
    if any(not math.isfinite(step) for step in steps):
        return None, "unresolved", True
    positive_steps = [step for step in steps if step > 0.0]
    if len(positive_steps) >= BASELINE_STEP_MINIMUM_POSITIVE_STEPS:
        candidate = float(statistics.median(positive_steps))
        support = sum(
            math.isclose(
                step,
                candidate,
                rel_tol=BASELINE_STEP_RELATIVE_TOLERANCE,
                abs_tol=BASELINE_STEP_ULP_TOLERANCE * max(math.ulp(step), math.ulp(candidate)),
            )
            for step in positive_steps
        )
        if support >= BASELINE_STEP_MINIMUM_SUPPORT:
            return candidate, "baseline_median_positive_step", False
    return None, "unresolved", False


def _quality_factors(
    *,
    baseline_summary: WindowSummary,
    fault_summary: WindowSummary,
    context: ObservationQualityContext,
    configuration: ObservationQualityConfiguration,
) -> tuple[QualityFactor, ...]:
    binding = {
        "bound": 1.0,
        "unbound": 0.0,
        "conflicted": configuration.conflicted_binding_factor,
    }[context.binding_status]
    contamination = {
        "none": 1.0,
        "suspected": configuration.suspected_contamination_factor,
        "confirmed": 0.0,
    }[context.baseline_contamination]
    clock = context.expected_cadence_ns / (
        context.expected_cadence_ns + 2 * context.clock_uncertainty_ns
    )
    factors = (
        QualityFactor(
            name="baseline_contamination",
            value=contamination,
            reason=context.baseline_contamination,
        ),
        QualityFactor(name="binding", value=binding, reason=context.binding_status),
        QualityFactor(
            name="clock",
            value=clock,
            reason=f"uncertainty_ns={context.clock_uncertainty_ns}",
        ),
        QualityFactor(
            name="late_data",
            value=configuration.late_data_factor if context.late_data_observed else 1.0,
            reason="observed" if context.late_data_observed else "not_observed",
        ),
        QualityFactor(
            name="window_coverage",
            value=min(
                baseline_summary.coverage_fraction,
                fault_summary.coverage_fraction,
            ),
            reason="minimum_baseline_fault_coverage",
        ),
    )
    return tuple(sorted(factors, key=lambda item: item.name))


def _required_statistic(value: float | None, name: str) -> float:
    if value is None:  # pragma: no cover - guarded by observed summaries
        raise AssertionError(f"missing {name}")
    return value


def _canonical_json_equal(left: object, right: object) -> bool:
    """Compare JSON values without collapsing distinct IEEE-754 encodings."""

    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _positive_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _positive_zero_optional(value: float | None) -> float | None:
    return None if value is None else _positive_zero(value)


def _canonical_summary(summary: WindowSummary) -> WindowSummary:
    return WindowSummary.model_validate(
        {
            **summary.model_dump(mode="json"),
            "coverage_fraction": _positive_zero(summary.coverage_fraction),
            "median": _positive_zero_optional(summary.median),
            "mad": _positive_zero_optional(summary.mad),
            "minimum": _positive_zero_optional(summary.minimum),
            "maximum": _positive_zero_optional(summary.maximum),
        }
    )


def _summaries_are_all_zero(*summaries: WindowSummary) -> bool:
    return bool(summaries) and all(
        summary.record_state == "observed" and summary.minimum == 0.0 and summary.maximum == 0.0
        for summary in summaries
    )


def _derive_canonical_quality(
    *,
    baseline_summary: WindowSummary,
    fault_summary: WindowSummary,
    baseline_raw_record_state: Literal["recorded", "no_record"],
    fault_raw_record_state: Literal["recorded", "no_record"],
    resolution: float | None,
    resolution_source: ResolutionSource,
    upstream_eligible: bool,
    upstream_flags: Sequence[str],
    resolution_estimation_nonfinite: bool,
    context: ObservationQualityContext,
    configuration: ObservationQualityConfiguration,
    signal_type: SignalType,
    aggregation: str,
    allow_unresolved_gauge_scale: bool = False,
) -> _CanonicalQualityDerivation:
    """Derive every replayable quality field from one canonical input set."""

    baseline_summary = _canonical_summary(baseline_summary)
    fault_summary = _canonical_summary(fault_summary)
    resolution = _positive_zero_optional(resolution)
    minimum_points = _minimum_valid_points(
        signal_type=signal_type,
        aggregation=aggregation,
        configured=configuration.minimum_valid_points,
    )
    quality_factors = tuple(
        item.model_copy(update={"value": _positive_zero(item.value)})
        for item in _quality_factors(
            baseline_summary=baseline_summary,
            fault_summary=fault_summary,
            context=context,
            configuration=configuration,
        )
    )
    flags = set(upstream_flags)
    statistical_gauge = allow_unresolved_gauge_scale and signal_type == "gauge"
    if statistical_gauge and resolution is None:
        flags.add("absolute_resolution_unresolved")
    if not upstream_eligible:
        flags.add("upstream_transform_ineligible")
    eligible = False
    baseline_scale: float | None = None
    robust_z: float | None = None
    anomaly_strength: float | None = None
    if resolution_estimation_nonfinite:
        flags.add("non_finite_derived_statistic")
    if baseline_raw_record_state == "no_record" and fault_raw_record_state == "no_record":
        state: EvidenceState = "no_record"
        flags.add("no_record")
    elif baseline_raw_record_state == "no_record" or fault_raw_record_state == "no_record":
        state = "unknown"
        flags.add("missing_comparable_window")
    elif resolution_estimation_nonfinite or _summary_statistics_missing(
        baseline_summary, fault_summary
    ):
        state = "unknown"
        flags.add("non_finite_derived_statistic")
    elif (
        baseline_summary.valid_point_count < minimum_points
        or fault_summary.valid_point_count < minimum_points
    ):
        state = "unknown"
        if baseline_summary.valid_point_count < minimum_points:
            flags.add("insufficient_baseline_points")
        if fault_summary.valid_point_count < minimum_points:
            flags.add("insufficient_fault_points")
    elif not upstream_eligible:
        state = "unknown"
    elif (
        statistical_gauge
        and resolution is None
        and baseline_summary.median == 0.0
        and baseline_summary.mad == 0.0
    ):
        state = "unknown"
        flags.add("statistical_scale_unresolved")
    elif (
        not statistical_gauge
        and _summaries_are_all_zero(baseline_summary, fault_summary)
        and resolution is None
    ):
        state = "zero"
        flags.add("absolute_resolution_unresolved")
    elif resolution is None and not statistical_gauge:
        state = "unknown"
        flags.add("absolute_resolution_unresolved")
    else:
        robust = _compute_robust_statistics(
            baseline_summary=baseline_summary,
            fault_summary=fault_summary,
            resolution=resolution,
            configuration=configuration,
        )
        if robust is None:
            state = "unknown"
            flags.add("non_finite_derived_statistic")
        else:
            baseline_scale, robust_z, anomaly_strength = robust
            if _summaries_are_all_zero(baseline_summary, fault_summary):
                state = "zero"
            elif robust_z >= configuration.abnormal_z_threshold:
                state = "abnormal"
            else:
                state = "verified_normal"
            eligible = True

    if context.baseline_contamination == "confirmed":
        if state != "no_record":
            state = "unknown"
            baseline_scale = None
            robust_z = None
            anomaly_strength = None
            eligible = False
        flags.add("baseline_contamination_confirmed")
    elif context.baseline_contamination == "suspected":
        if state not in {"no_record", "unknown", "zero"}:
            state = "degraded"
        flags.add("baseline_contamination_suspected")
    if context.binding_status != "bound":
        if state not in {"no_record", "unknown", "zero"}:
            state = "degraded"
        eligible = False
        flags.add(f"binding_{context.binding_status}")
    if context.late_data_observed:
        if state not in {"no_record", "unknown", "zero"}:
            state = "degraded"
        flags.add("late_data_observed")
    if context.clock_uncertainty_ns >= context.expected_cadence_ns:
        if state not in {"no_record", "unknown", "zero"}:
            state = "degraded"
        flags.add("clock_uncertainty_at_least_cadence")
    if min(
        baseline_summary.coverage_fraction,
        fault_summary.coverage_fraction,
    ) < configuration.minimum_coverage_fraction and state not in {
        "no_record",
        "unknown",
    }:
        if state != "zero":
            state = "degraded"
        flags.add("insufficient_window_coverage")
    return _CanonicalQualityDerivation(
        baseline_summary=baseline_summary,
        fault_summary=fault_summary,
        baseline_summary_sufficient=(baseline_summary.valid_point_count >= minimum_points),
        fault_summary_sufficient=fault_summary.valid_point_count >= minimum_points,
        absolute_resolution=resolution,
        absolute_resolution_source=resolution_source,
        baseline_scale=_positive_zero_optional(baseline_scale),
        robust_z=_positive_zero_optional(robust_z),
        anomaly_strength=_positive_zero_optional(anomaly_strength),
        evidence_state=state,
        coverage_quality=_positive_zero(math.prod(item.value for item in quality_factors)),
        minimum_valid_points=minimum_points,
        formal_scoring_eligible=eligible,
        quality_factors=quality_factors,
        quality_flags=tuple(sorted(flags)),
    )


def _summary_statistics_missing(*summaries: WindowSummary) -> bool:
    return any(
        summary.valid_point_count > 0
        and any(
            value is None
            for value in (
                summary.median,
                summary.mad,
                summary.minimum,
                summary.maximum,
            )
        )
        for summary in summaries
    )


def _compute_robust_statistics(
    *,
    baseline_summary: WindowSummary,
    fault_summary: WindowSummary,
    resolution: float | None,
    configuration: ObservationQualityConfiguration,
) -> tuple[float, float, float] | None:
    if _summary_statistics_missing(baseline_summary, fault_summary):
        return None
    baseline_median = _required_statistic(baseline_summary.median, "baseline median")
    baseline_mad = _required_statistic(baseline_summary.mad, "baseline MAD")
    fault_median = _required_statistic(fault_summary.median, "fault median")
    scale = max(
        configuration.mad_multiplier * baseline_mad,
        resolution if resolution is not None else 0.0,
        configuration.relative_median_floor * abs(baseline_median),
    )
    delta = abs(fault_median - baseline_median)
    if not math.isfinite(scale) or scale <= 0.0 or not math.isfinite(delta):
        return None
    robust_z = delta / scale
    if not math.isfinite(robust_z):
        return None
    anomaly_strength = -math.expm1(-robust_z / configuration.anomaly_mapping_divisor)
    if not math.isfinite(anomaly_strength):
        return None
    return scale, robust_z, anomaly_strength
