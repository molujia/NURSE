"""Explicit live scalar-source bridge; never re-label legacy quality or windows.

Publication binds a code-owned observation to its raw scalar windows. It does
not authenticate arbitrary caller-provided telemetry: the complete live case
must first pass file/adapter replay (3.4), and its source pins must be frozen by
the enclosing case invocation. The consumer requires those independent pins
and reads both original source blocks on every call. No cross-call trust cache.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import fields
from typing import Literal, cast

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.contracts import SourceBlockReference, _SourceModel
from nurse_rcl.v2.dynamic_verification.safety import (
    canonical,
    plain,
    require_pin,
    require_unchanged,
    snapshot,
)
from nurse_rcl.v2.dynamic_verification.source_store import (
    DirectorySourceBlockStore,
    SourceBlockStore,
    read_source_window,
    require_store_unchanged,
    store_snapshot,
)
from nurse_rcl.v2.live.assembly import (
    AIOPS25_SEMANTICS_SHA256,
    AIOPS25_VERSION,
    SEMANTICS_SHA256,
    VERSION,
    LiveSignal,
)
from nurse_rcl.v2.live.repair_assembly import RepairSignalDiagnostic
from nurse_rcl.v2.live.repair_lifecycle import LifecycleObservation, summarize_lifecycle_component
from nurse_rcl.v2.live.repair_observations import RepairObservationQualityResult
from nurse_rcl.v2.observation_quality import ObservationQualityResult
from nurse_rcl.v2.observation_schemas import CaseTimeContract, Observation, TimeInterval
from nurse_rcl.v2.signal_transforms import (
    SIGNAL_TRANSFORM_SEMANTICS_SHA256,
    ScalarWindowInput,
    SignalTransformResult,
)

REPAIR_MEASUREMENT_SOURCE_SCHEMA_VERSION: Literal["nurse.v2.live-repair-measurement-source/1"] = (
    "nurse.v2.live-repair-measurement-source/1"
)
_DATASETS = {
    "rcabench": (VERSION, SEMANTICS_SHA256),
    "aiops25": (AIOPS25_VERSION, AIOPS25_SEMANTICS_SHA256),
}


class RepairMeasurementSource(_SourceModel):
    schema_version: Literal["nurse.v2.live-repair-measurement-source/1"] = (
        REPAIR_MEASUREMENT_SOURCE_SCHEMA_VERSION
    )
    dataset: Literal["rcabench", "aiops25"]
    signal_id: str = Field(pattern=r"^sig_[0-9a-f]{24}$")
    observed_entity_key: str = Field(min_length=1)
    candidate_service_key: str | None
    metric_name: str | None
    source_file_sha256s: tuple[str, ...]
    assembly_flags: tuple[str, ...]
    observation: Observation
    time_contract: CaseTimeContract
    query_interval: TimeInterval
    transform_semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transform_result: SignalTransformResult
    quality: RepairObservationQualityResult
    lifecycle: LifecycleObservation | None
    source_blocks: tuple[SourceBlockReference, ...]
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_distinct_bound_semantics(self) -> Self:
        transform, quality, observation = self.transform_result, self.quality, self.observation
        contract = self.time_contract
        version, semantics = _DATASETS[self.dataset]
        summary = quality.source_summary
        if (
            observation.adapter_semantics_sha256 != semantics
            or summary.context.adapter_manifest_id != version
            or summary.context.adapter_manifest_version != "1"
            or self.transform_semantics_sha256 != SIGNAL_TRANSFORM_SEMANTICS_SHA256
            or transform.adapter_semantics_sha256 != self.transform_semantics_sha256
            or not transform.verify()
            or not quality.verify_transform(transform_result=transform)
        ):
            raise ValueError("repair source dataset/transform/quality semantics mismatch")
        if (
            transform.windows[0].interval != contract.baseline_window
            or transform.windows[1].interval != contract.fault_window
            or observation.interval.start_unix_ns != contract.baseline_window.start_unix_ns
            or observation.interval.end_unix_ns != contract.fault_window.end_unix_ns
            or observation.opaque_case_id != contract.opaque_case_id
            or observation.decision_time_unix_ns != contract.decision_time_unix_ns
            or self.query_interval != contract.fault_window
            or summary.context.expected_cadence_ns != contract.expected_cadence_ns
            or summary.context.clock_uncertainty_ns != contract.clock_uncertainty_ns
            or summary.context.late_data_observed != contract.late_data_observed
        ):
            raise ValueError("repair source B/F observation/query scope mismatch")
        if summary.context.binding_status != (
            "bound" if self.candidate_service_key else "unbound"
        ) or (
            self.candidate_service_key is not None
            and self.candidate_service_key != self.observed_entity_key
        ):
            raise ValueError("repair source entity binding mismatch")
        pins = self.source_file_sha256s
        if not pins or pins != tuple(sorted(set(pins))):
            raise ValueError("repair source file identities must be nonempty ordered unique")
        for pin in pins:
            require_pin(pin)
        fingerprint = sha256_hex(
            canonical_json_bytes([self.signal_id, pins, transform.source_input_sha256])
        )
        if (
            observation.source_fingerprint != fingerprint
            or observation.provenance_ids != tuple("source:" + pin for pin in pins)
            or observation.observation_id
            != "obs_"
            + sha256_hex(canonical_json_bytes([self.signal_id, summary.result_sha256]))[:24]
            or observation.transform != transform.transform
            or self.assembly_flags != tuple(sorted(set(self.assembly_flags)))
        ):
            raise ValueError("repair source original observation identity mismatch")
        for name in (
            "signal_type",
            "unit",
            "aggregation",
            "baseline_summary",
            "fault_summary",
            "anomaly_strength",
            "coverage_quality",
            "minimum_valid_points",
            "formal_scoring_eligible",
        ):
            if getattr(observation, name) != getattr(summary, name):
                raise ValueError(
                    "repair source old observation differs from source summary: " + name
                )
        expected_state = "unknown" if transform.signal_type == "unknown" else summary.evidence_state
        if observation.evidence_state != expected_state or observation.quality_flags != tuple(
            sorted(set(summary.quality_flags) | set(self.assembly_flags))
        ):
            raise ValueError("repair source old observation state or flags mismatch")
        if tuple(block.name for block in self.source_blocks) != ("baseline", "fault") or any(
            block.input_kind != "ScalarWindowInput" for block in self.source_blocks
        ):
            raise ValueError("repair source requires complete scalar B/F blocks")
        expected_lifecycle = None
        if observation.signal_family == "lifecycle" and transform.signal_type == "gauge":
            if self.metric_name is None:
                raise ValueError("lifecycle source requires an exact registered metric")
            expected_lifecycle = summarize_lifecycle_component(
                dataset=self.dataset,
                metric_name=self.metric_name,
                transform_result=transform,
                context=summary.context,
            )
        if canonical(self.lifecycle, (LifecycleObservation,)) != canonical(
            expected_lifecycle, (LifecycleObservation,)
        ):
            raise ValueError("repair lifecycle differs from source transform")
        return self


_INPUT_ROOTS = (
    Observation,
    CaseTimeContract,
    ScalarWindowInput,
    SignalTransformResult,
    ObservationQualityResult,
    RepairObservationQualityResult,
    LifecycleObservation,
)


def _inputs(
    signal: LiveSignal, repair: RepairSignalDiagnostic, contract: CaseTimeContract, dataset: str
) -> tuple[object, ...]:
    if (
        type(signal) is not LiveSignal
        or type(repair) is not RepairSignalDiagnostic
        or type(contract) is not CaseTimeContract
    ):
        raise ValueError("repair source publication requires exact input types")
    if type(dataset) is not str or dataset not in _DATASETS:
        raise ValueError("repair source dataset is not registered")
    return (
        tuple(object.__getattribute__(signal, f.name) for f in fields(LiveSignal)),
        tuple(object.__getattribute__(repair, f.name) for f in fields(RepairSignalDiagnostic)),
        contract,
        dataset,
    )


def build_repair_measurement_source(
    *,
    signal: LiveSignal,
    repair: RepairSignalDiagnostic,
    time_contract: CaseTimeContract,
    dataset: str,
    store: DirectorySourceBlockStore,
) -> RepairMeasurementSource:
    """Publish both windows; caller must bind the full case file-replay identity.

    This per-series adapter never samples candidates or claims a complete case.
    The enclosing case must conserve every original signal and unknown entry.
    """
    if type(store) is not DirectorySourceBlockStore:
        raise ValueError("repair source publication requires the exact directory store")
    before = snapshot(_inputs(signal, repair, time_contract, dataset), _INPUT_ROOTS)
    state = store_snapshot(store)
    try:
        return _measurement_source_values(
            signal=signal, repair=repair, time_contract=time_contract, dataset=dataset, store=store,
        )
    finally:
        require_unchanged(_inputs(signal, repair, time_contract, dataset), _INPUT_ROOTS, before)
        require_store_unchanged(store, state)


def _measurement_source_values(
    *, signal: LiveSignal, repair: RepairSignalDiagnostic, time_contract: CaseTimeContract,
    dataset: str, store: DirectorySourceBlockStore,
) -> RepairMeasurementSource:
    """Shared producer checks/values, with no lifetime monitoring or source replay."""
    _inputs(signal, repair, time_contract, dataset)
    windows = {"baseline": signal.baseline, "fault": signal.fault}
    if (
        repair.signal_id != signal.signal_id
        or canonical(repair.quality.source_summary, _INPUT_ROOTS)
        != canonical(signal.quality, _INPUT_ROOTS)
        or not repair.quality.verify_source_inputs(
            transform_result=signal.transform, source_inputs=windows
        )
    ):
        raise ValueError("repair source raw windows or independent summaries differ")
    blocks = []
    for name, window in windows.items():
        raw = canonical_json_bytes(dict(
            schema_version="nurse.v2.signal-transform-source-inputs/2",
            inputs={name: dict(input_kind="ScalarWindowInput", document=plain(window))},
        ))
        pin = store.put(raw=raw)
        blocks.append(SourceBlockReference(
            name=name, input_kind="ScalarWindowInput", sha256=pin, size_bytes=len(raw),
        ))
    draft = RepairMeasurementSource(
        dataset=cast(Literal["rcabench", "aiops25"], dataset),
        signal_id=signal.signal_id, observed_entity_key=signal.observed_entity_key,
        candidate_service_key=signal.service, metric_name=repair.metric_name,
        source_file_sha256s=signal.source_file_sha256s, assembly_flags=signal.flags,
        observation=signal.observation, time_contract=time_contract,
        query_interval=time_contract.fault_window,
        transform_semantics_sha256=signal.transform.adapter_semantics_sha256,
        transform_result=signal.transform, quality=repair.quality, lifecycle=repair.lifecycle,
        source_blocks=tuple(blocks), source_sha256="0" * 64,
    )
    unsigned = cast(dict[str, object], plain(draft))
    unsigned.pop("source_sha256")
    return draft.model_copy(update={"source_sha256": sha256_hex(canonical_json_bytes(unsigned))})


@contextmanager
def _replayed_repair_measurement_source(
    *,
    source: RepairMeasurementSource,
    expected_source_sha256: str,
    time_contract: CaseTimeContract,
    expected_time_contract_sha256: str,
    store: SourceBlockStore,
) -> Iterator[RepairMeasurementSource]:
    """Own one independent B/F replay through the consuming call's exit."""
    if type(source) is not RepairMeasurementSource or type(time_contract) is not CaseTimeContract:
        raise ValueError("repair source replay requires exact current types")
    require_pin(expected_source_sha256)
    require_pin(expected_time_contract_sha256)
    roots = (RepairMeasurementSource, CaseTimeContract)
    before = snapshot((source, time_contract), roots)
    state = store_snapshot(store)
    with ExitStack() as guards:
        guards.callback(require_store_unchanged, store, state)
        guards.callback(require_unchanged, (source, time_contract), roots, before)
        if sha256_hex(canonical(time_contract, roots)) != expected_time_contract_sha256:
            raise ValueError("repair source independent time pin mismatch")
        fresh = RepairMeasurementSource.model_validate_json(canonical(source, roots))
        fresh_before = snapshot(fresh, roots)
        guards.callback(require_unchanged, fresh, roots, fresh_before)
        # The fresh value remains guarded while the nested consumer uses it.
        unsigned = cast(dict[str, object], plain(fresh))
        unsigned.pop("source_sha256")
        if (
            fresh.source_sha256 != expected_source_sha256
            or sha256_hex(canonical_json_bytes(unsigned)) != expected_source_sha256
            or canonical(fresh.time_contract, roots) != canonical(time_contract, roots)
        ):
            raise ValueError("repair source independent source pin or time mismatch")
        windows = {
            block.name: read_source_window(store=store, block=block)
            for block in fresh.source_blocks
        }
        if not fresh.quality.verify_source_inputs(
            transform_result=fresh.transform_result, source_inputs=windows
        ):
            raise ValueError("repair source blocks cannot reproduce transform and quality")
        if fresh.lifecycle is not None and not fresh.lifecycle.verify_source_inputs(
            transform_result=fresh.transform_result,
            source_inputs=windows,
            dataset=fresh.dataset,
            metric_name=fresh.metric_name or "",
        ):
            raise ValueError("repair source lifecycle cannot be replayed")
        yield fresh


def require_repair_measurement_source(
    *,
    source: RepairMeasurementSource,
    expected_source_sha256: str,
    time_contract: CaseTimeContract,
    expected_time_contract_sha256: str,
    store: SourceBlockStore,
) -> RepairMeasurementSource:
    """Public calls always re-read both blocks and finish all guards before return."""
    with _replayed_repair_measurement_source(
        source=source,
        expected_source_sha256=expected_source_sha256,
        time_contract=time_contract,
        expected_time_contract_sha256=expected_time_contract_sha256,
        store=store,
    ) as fresh:
        return fresh
