"""Observed lifecycle states, not inferred continuous outages or service death."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.live.repair_observations import (
    REPAIR_QUALITY_SEMANTICS_SHA256,
    RepairObservationQualityResult,
    summarize_repair_component,
)
from nurse_rcl.v2.observation_quality import ObservationQualityContext
from nurse_rcl.v2.signal_transforms import ScalarSample, SignalTransformResult, SourceWindowInput

LIFECYCLE_SCHEMA_VERSION: Literal["nurse.v2.live-repair-lifecycle/1"] = (
    "nurse.v2.live-repair-lifecycle/1"
)
# These exact instantaneous names already have count semantics in live assembly.
# No name substring, case identity or benchmark label grants availability meaning.
_REGISTERED_MEANINGS = {
    "rcabench": {
        "k8s.container.ready": "readiness_zero",
        "k8s.deployment.available": "available_count_zero",
        "k8s.replicaset.available": "available_count_zero",
        "k8s.statefulset.ready_pods": "ready_count_zero",
        "k8s.deployment.desired": "desired_count_zero",
        "k8s.replicaset.desired": "desired_count_zero",
        "k8s.statefulset.desired_pods": "desired_count_zero",
        "k8s.statefulset.current_pods": "current_count_zero",
        "k8s.statefulset.updated_pods": "updated_count_zero",
    },
    "aiops25": {"pod_processes": "process_count_zero"},
}
_AVAILABILITY_MEANINGS = {"readiness_zero", "available_count_zero", "ready_count_zero"}
LIFECYCLE_SEMANTICS_SHA256 = sha256_hex(
    canonical_json_bytes(
        {
            "version": "nurse-v2-live-repair-lifecycle/1",
            "metrics": _REGISTERED_MEANINGS,
            "quality": REPAIR_QUALITY_SEMANTICS_SHA256,
            "events": "consecutive_observed_zero_points_split_by_nonzero_invalid_or_gap",
            "gap": "delta_timestamp_exceeds_expected_cadence",
            "bounds": "adjacent_valid_positive_sample_plus_clock_uncertainty_otherwise_censored",
            "duration": "never_inferred_continuous",
            "scope": "observed_entity_not_entire_service",
            "existence": "fault_source_and_binding_not_whole_window_median_or_pressure_persistence",
        }
    )
)


class LifecycleZeroEvent(StrictFrozenModel):
    zero_timestamps_ns: tuple[int, ...] = Field(min_length=1)
    onset_bounds_ns: tuple[int, int] | None
    recovery_bounds_ns: tuple[int, int] | None
    left_censored: bool
    right_censored: bool
    continuous_duration_ns: None = None


def _valid_state(sample: ScalarSample, meaning: str) -> bool:
    return (
        sample.value >= 0
        and sample.value.is_integer()
        and (meaning != "readiness_zero" or sample.value <= 1.0)
    )


def _event_fields(
    *,
    transform: SignalTransformResult,
    context: ObservationQualityContext,
    dataset: str,
    metric_name: str,
) -> dict[str, object]:
    meaning = _REGISTERED_MEANINGS.get(dataset, {}).get(metric_name)
    if meaning is None:
        raise ValueError("lifecycle metric is not registered for this dataset")
    if not transform.verify():
        raise ValueError("lifecycle transform failed verification")
    component = next((c for c in transform.windows[1].components if c.name == "value"), None)
    if (
        transform.signal_type != "gauge"
        or component is None
        or component.unit != "count"
        or component.aggregation != "instantaneous"
    ):
        raise ValueError("lifecycle requires registered instantaneous count gauge semantics")
    samples = component.samples
    cadence = context.expected_cadence_ns
    uncertainty = context.clock_uncertainty_ns
    gaps = tuple(
        (a.timestamp_unix_ns, b.timestamp_unix_ns)
        for a, b in pairwise(samples)
        if b.timestamp_unix_ns - a.timestamp_unix_ns > cadence
    )
    invalid = tuple(s.timestamp_unix_ns for s in samples if not _valid_state(s, meaning))
    runs: list[list[int]] = []
    for index, sample in enumerate(samples):
        if sample.value != 0.0:
            continue
        if (
            runs
            and index == runs[-1][-1] + 1
            and sample.timestamp_unix_ns - samples[index - 1].timestamp_unix_ns <= cadence
        ):
            runs[-1].append(index)
        else:
            runs.append([index])
    events = []
    for run in runs:
        first, last = run[0], run[-1]
        onset = recovery = None
        if first > 0:
            previous, current = samples[first - 1], samples[first]
            if (
                _valid_state(previous, meaning)
                and previous.value > 0
                and current.timestamp_unix_ns - previous.timestamp_unix_ns <= cadence
            ):
                onset = (
                    max(0, previous.timestamp_unix_ns - uncertainty),
                    current.timestamp_unix_ns + uncertainty,
                )
        if last + 1 < len(samples):
            current, following = samples[last], samples[last + 1]
            if (
                _valid_state(following, meaning)
                and following.value > 0
                and following.timestamp_unix_ns - current.timestamp_unix_ns <= cadence
            ):
                recovery = (
                    max(0, current.timestamp_unix_ns - uncertainty),
                    following.timestamp_unix_ns + uncertainty,
                )
        events.append(
            LifecycleZeroEvent(
                zero_timestamps_ns=tuple(samples[i].timestamp_unix_ns for i in run),
                onset_bounds_ns=onset,
                recovery_bounds_ns=recovery,
                left_censored=onset is None,
                right_censored=recovery is None,
            )
        )
    eligible = (
        bool(events) and component.formal_scoring_eligible and context.binding_status == "bound"
    )
    return {
        "state_meaning": meaning,
        "observed_zero": bool(events),
        "formal_event_eligible": eligible,
        "availability_support": eligible and meaning in _AVAILABILITY_MEANINGS,
        "events": tuple(event.model_dump(mode="json") for event in events),
        "invalid_state_timestamps_ns": invalid,
        "missing_cadence_intervals_ns": gaps,
    }


class LifecycleObservation(StrictFrozenModel):
    """Point-state evidence plus an unchanged whole-window quality comparison."""

    schema_version: Literal["nurse.v2.live-repair-lifecycle/1"] = LIFECYCLE_SCHEMA_VERSION
    semantics_version: Literal["nurse-v2-live-repair-lifecycle/1"] = (
        "nurse-v2-live-repair-lifecycle/1"
    )
    semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset: Literal["rcabench", "aiops25"]
    metric_name: str = Field(min_length=1)
    scope: Literal["observed_entity_not_entire_service"] = "observed_entity_not_entire_service"
    transform_result: SignalTransformResult
    window_quality: RepairObservationQualityResult
    state_meaning: str = Field(min_length=1)
    observed_zero: bool
    formal_event_eligible: bool
    availability_support: bool
    events: tuple[LifecycleZeroEvent, ...]
    invalid_state_timestamps_ns: tuple[int, ...]
    missing_cadence_intervals_ns: tuple[tuple[int, int], ...]
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_events(self) -> Self:
        if self.semantics_sha256 != LIFECYCLE_SEMANTICS_SHA256:
            raise ValueError("lifecycle semantics mismatch")
        if not self.window_quality.verify_transform(transform_result=self.transform_result):
            raise ValueError("lifecycle window summary is not source-bound")
        expected = _event_fields(
            transform=self.transform_result,
            context=self.window_quality.context,
            dataset=self.dataset,
            metric_name=self.metric_name,
        )
        actual = self.model_dump(mode="json", include=set(expected))
        if canonical_json_bytes(actual) != canonical_json_bytes(expected):
            raise ValueError("lifecycle events are not canonically derived")
        return self

    def verify(self) -> bool:
        try:
            document = self.model_dump(mode="json")
            type(self).model_validate(document)
            unsigned = {key: value for key, value in document.items() if key != "result_sha256"}
            return self.result_sha256 == sha256_hex(canonical_json_bytes(unsigned))
        except (TypeError, ValueError):
            return False

    def verify_transform(
        self, *, transform_result: SignalTransformResult, dataset: str, metric_name: str
    ) -> bool:
        try:
            replayed = summarize_lifecycle_component(
                transform_result=transform_result,
                dataset=dataset,
                metric_name=metric_name,
                context=self.window_quality.context,
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
        dataset: str,
        metric_name: str,
    ) -> bool:
        return transform_result.verify_source_inputs(source_inputs) and self.verify_transform(
            transform_result=transform_result,
            dataset=dataset,
            metric_name=metric_name,
        )


def summarize_lifecycle_component(
    *,
    dataset: str,
    metric_name: str,
    transform_result: SignalTransformResult,
    context: ObservationQualityContext,
) -> LifecycleObservation:
    # Revalidate nested types before deriving any event, including model_copy input.
    transform = SignalTransformResult.model_validate(transform_result.model_dump(mode="json"))
    fields = _event_fields(
        transform=transform, context=context, dataset=dataset, metric_name=metric_name
    )
    quality = summarize_repair_component(
        transform_result=transform, component_name="value", context=context
    )
    unsigned = {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "semantics_version": "nurse-v2-live-repair-lifecycle/1",
        "semantics_sha256": LIFECYCLE_SEMANTICS_SHA256,
        "dataset": dataset,
        "metric_name": metric_name,
        "scope": "observed_entity_not_entire_service",
        "transform_result": transform.model_dump(mode="json"),
        "window_quality": quality.model_dump(mode="json"),
        **fields,
    }
    return LifecycleObservation.model_validate(
        {
            **unsigned,
            "result_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
