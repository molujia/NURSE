"""Explicit repair quality: statistical scale is not collection precision.

The original quality/5 summary is retained verbatim as the source-summary
envelope. New decisions have a different schema and semantics; a legacy reader
cannot accidentally consume them. Raw-source permission still requires replay
through the original transform, not just a self-consistent summary or digest.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.observation_quality import (
    OBSERVATION_QUALITY_SEMANTICS_SHA256,
    ObservationQualityConfiguration,
    ObservationQualityContext,
    ObservationQualityResult,
    ResolutionSource,
    _derive_canonical_quality,
    summarize_transformed_component,
)
from nurse_rcl.v2.observation_schemas import EvidenceState
from nurse_rcl.v2.signal_transforms import SignalTransformResult, SourceWindowInput

REPAIR_QUALITY_SCHEMA_VERSION: Literal["nurse.v2.live-repair-quality/1"] = (
    "nurse.v2.live-repair-quality/1"
)
REPAIR_QUALITY_SEMANTICS_VERSION: Literal["nurse-v2-live-repair-quality/1"] = (
    "nurse-v2-live-repair-quality/1"
)
REPAIR_QUALITY_SEMANTICS_SHA256 = sha256_hex(
    canonical_json_bytes(
        {
            "version": REPAIR_QUALITY_SEMANTICS_VERSION,
            "inherited_quality_semantics_sha256": OBSERVATION_QUALITY_SEMANTICS_SHA256,
            "scale": "max(1.4826*MAD_B,0.05*abs(median_B),legitimate_resolution_if_available)",
            "unresolved_resolution": (
                "permitted_only_for_known_gauge_positive_finite_baseline_scale"
            ),
            "zero_scale": "unknown_without_epsilon",
            "signed_change": "(median_F-median_B)/baseline_scale",
            "threshold": 3.0,
            "mapping_divisor": 3.0,
            "other_quality_checks": "unchanged",
        }
    )
)


def _repair_fields(source: ObservationQualityResult) -> dict[str, object]:
    """Pure derivation; this helper confers no raw-source trust."""

    derived = _derive_canonical_quality(
        baseline_summary=source.baseline_summary,
        fault_summary=source.fault_summary,
        baseline_raw_record_state=source.baseline_raw_record_state,
        fault_raw_record_state=source.fault_raw_record_state,
        resolution=source.absolute_resolution,
        resolution_source=source.absolute_resolution_source,
        upstream_eligible=source.upstream_formal_scoring_eligible,
        upstream_flags=source.upstream_quality_flags,
        resolution_estimation_nonfinite=source.resolution_estimation_nonfinite,
        context=source.context,
        configuration=source.configuration,
        signal_type=source.signal_type,
        aggregation=source.aggregation,
        allow_unresolved_gauge_scale=True,
    )
    signed = None
    if derived.baseline_scale is not None:
        baseline, fault = source.baseline_summary.median, source.fault_summary.median
        assert baseline is not None and fault is not None
        signed = (fault - baseline) / derived.baseline_scale
        signed = 0.0 if signed == 0.0 else signed
    return {
        "baseline_scale": derived.baseline_scale,
        "signed_standardized_change": signed,
        "robust_z": derived.robust_z,
        "anomaly_strength": derived.anomaly_strength,
        "evidence_state": derived.evidence_state,
        "formal_scoring_eligible": derived.formal_scoring_eligible,
        "quality_flags": derived.quality_flags,
    }


class RepairObservationQualityResult(StrictFrozenModel):
    """New decisions alongside an unchanged, independently replayable old summary."""

    schema_version: Literal["nurse.v2.live-repair-quality/1"] = REPAIR_QUALITY_SCHEMA_VERSION
    semantics_version: Literal["nurse-v2-live-repair-quality/1"] = REPAIR_QUALITY_SEMANTICS_VERSION
    semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_summary: ObservationQualityResult
    baseline_scale: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    signed_standardized_change: float | None = Field(default=None, allow_inf_nan=False)
    robust_z: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    anomaly_strength: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    evidence_state: EvidenceState
    formal_scoring_eligible: bool
    quality_flags: tuple[str, ...]
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def context(self) -> ObservationQualityContext:
        return self.source_summary.context

    @property
    def absolute_resolution(self) -> float | None:
        return self.source_summary.absolute_resolution

    @property
    def absolute_resolution_source(self) -> ResolutionSource:
        return self.source_summary.absolute_resolution_source

    @property
    def coverage_quality(self) -> float:
        return self.source_summary.coverage_quality

    @model_validator(mode="after")
    def require_canonical_derivation(self) -> Self:
        if self.semantics_sha256 != REPAIR_QUALITY_SEMANTICS_SHA256:
            raise ValueError("repair quality semantics mismatch")
        if not self.source_summary.verify():
            raise ValueError("repair source summary failed verification")
        expected = _repair_fields(self.source_summary)
        actual = self.model_dump(mode="json", include=set(expected))
        if canonical_json_bytes(actual) != canonical_json_bytes(expected):
            raise ValueError("repair quality is not canonically derived")
        return self

    def verify(self) -> bool:
        try:
            document = self.model_dump(mode="json")
            type(self).model_validate(document)
            unsigned = {key: value for key, value in document.items() if key != "result_sha256"}
            return self.result_sha256 == sha256_hex(canonical_json_bytes(unsigned))
        except (TypeError, ValueError):
            return False

    def verify_transform(self, *, transform_result: SignalTransformResult) -> bool:
        try:
            replayed = summarize_repair_component(
                transform_result=transform_result,
                component_name=self.source_summary.component_name,
                context=self.context,
                configuration=self.source_summary.configuration,
            )
            return self.verify() and canonical_json_bytes(self.model_dump(mode="json")) == (
                canonical_json_bytes(replayed.model_dump(mode="json"))
            )
        except (KeyError, TypeError, ValueError):
            return False

    def verify_source_inputs(
        self,
        *,
        transform_result: SignalTransformResult,
        source_inputs: Mapping[str, SourceWindowInput],
    ) -> bool:
        return self.source_summary.verify_source_inputs(
            transform_result=transform_result,
            source_inputs=source_inputs,
        ) and self.verify_transform(transform_result=transform_result)


def summarize_repair_component(
    *,
    transform_result: SignalTransformResult,
    component_name: str,
    context: ObservationQualityContext,
    configuration: ObservationQualityConfiguration | None = None,
) -> RepairObservationQualityResult:
    """Compare known gauges even when legitimate collection resolution is unknown."""

    source = summarize_transformed_component(
        transform_result=transform_result,
        component_name=component_name,
        context=context,
        configuration=configuration,
    )
    unsigned = {
        "schema_version": REPAIR_QUALITY_SCHEMA_VERSION,
        "semantics_version": REPAIR_QUALITY_SEMANTICS_VERSION,
        "semantics_sha256": REPAIR_QUALITY_SEMANTICS_SHA256,
        "source_summary": source.model_dump(mode="json"),
        **_repair_fields(source),
    }
    return RepairObservationQualityResult.model_validate(
        {
            **unsigned,
            "result_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
