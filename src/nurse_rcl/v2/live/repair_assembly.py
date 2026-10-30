"""Whole-case observation diagnostics with explicit repair identities.

This bundle retains the legacy input, not a rewritten quality/5 result. It is
NOT a ranking/agent input: consumers require the later repair source and score
contracts. A descriptor or a self-consistent digest never grants source trust.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa  # type: ignore[import-untyped]

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.aiops25_sources import AIOps25LiveSources, read_aiops25_live_sources
from nurse_rcl.v2.live.assembly import (
    AIOPS25_SEMANTICS_SHA256,
    SEMANTICS_SHA256,
    AIOps25LiveInputs,
    LiveInputs,
    _digest,
    _identity,
    assemble_aiops25_live_inputs,
    assemble_rcabench_live_inputs,
)
from nurse_rcl.v2.live.repair_errors import (
    ERROR_SEMANTICS_SHA256,
    ErrorObservation,
    LogErrorRecord,
    TraceErrorRecord,
    summarize_log_errors,
    summarize_trace_errors,
)
from nurse_rcl.v2.live.repair_lifecycle import (
    LIFECYCLE_SEMANTICS_SHA256,
    LifecycleObservation,
    summarize_lifecycle_component,
)
from nurse_rcl.v2.live.repair_observations import (
    REPAIR_QUALITY_SEMANTICS_SHA256,
    RepairObservationQualityResult,
    summarize_repair_component,
)
from nurse_rcl.v2.live.sources import RCABenchLiveSources, read_rcabench_live_sources

REPAIR_DIAGNOSTIC_SCHEMA_VERSION = "nurse.v2.live-repair-diagnostic-inputs/1"
REPAIR_DIAGNOSTIC_SEMANTICS_SHA256 = _digest(
    dict(
        version="nurse-v2-live-repair-diagnostic-assembly/1",
        legacy=(SEMANTICS_SHA256, AIOPS25_SEMANTICS_SHA256),
        quality=REPAIR_QUALITY_SEMANTICS_SHA256,
        lifecycle=LIFECYCLE_SEMANTICS_SHA256,
        errors=ERROR_SEMANTICS_SHA256,
        metric_identity="exact_legacy_kind_entity_metric_attributes_component",
        source_row="window_kind_index_in_complete_pinned_projection",
        error_windows="separate_baseline_and_fault_no_temporal_inference",
        invalid_error_fields="uninterpretable_type_or_nonfinite_becomes_unknown_no_coercion",
        ranking_ready=False,
    )
)
Dataset = Literal["rcabench", "aiops25"]
Sources = RCABenchLiveSources | AIOps25LiveSources


@dataclass(frozen=True)
class RepairSignalDiagnostic:
    signal_id: str
    metric_name: str | None
    quality: RepairObservationQualityResult
    lifecycle: LifecycleObservation | None
    lifecycle_unavailable_reason: str | None


@dataclass(frozen=True)
class RepairDiagnosticInputs:
    dataset: Dataset
    legacy: LiveInputs = field(repr=False)
    signals: tuple[RepairSignalDiagnostic, ...]
    errors: tuple[ErrorObservation, ...] = field(repr=False)

    def diagnostic_descriptor(self) -> dict[str, Any]:
        # Hash actual retained values, not their caller-controlled result pins.
        return dict(
            schema_version=REPAIR_DIAGNOSTIC_SCHEMA_VERSION,
            semantics_sha256=REPAIR_DIAGNOSTIC_SEMANTICS_SHA256,
            dataset=self.dataset,
            ranking_ready=False,
            legacy=self.legacy.inference_descriptor(),
            signals_sha256=_digest(
                [
                    dict(
                        signal_id=s.signal_id,
                        metric_name=s.metric_name,
                        quality=s.quality.model_dump(mode="json"),
                        lifecycle=s.lifecycle.model_dump(mode="json") if s.lifecycle else None,
                        lifecycle_unavailable_reason=s.lifecycle_unavailable_reason,
                    )
                    for s in self.signals
                ]
            ),
            errors_sha256=tuple(_digest(e.model_dump(mode="json")) for e in self.errors),
        )

    def verify_sources(self, root: Path) -> bool:
        if type(self) is not RepairDiagnosticInputs:
            return False
        sources = self.legacy.sources
        if not (
            (
                self.dataset == "rcabench"
                and type(self.legacy) is LiveInputs
                and type(sources) is RCABenchLiveSources
            )
            or (
                self.dataset == "aiops25"
                and type(self.legacy) is AIOps25LiveInputs
                and type(sources) is AIOps25LiveSources
            )
        ):
            return False
        try:
            before = canonical_json_bytes(self.diagnostic_descriptor())
            source_before = canonical_json_bytes(sources.descriptor())
            fresh: Sources | None = None
            fresh_before: bytes | None = None
            matched = False
            try:
                if type(sources) is RCABenchLiveSources:
                    fresh = read_rcabench_live_sources(
                        root,
                        opaque_case_id=sources.adapter.opaque_case_id,
                        configuration=sources.adapter.configuration,
                    )
                    fresh_before = canonical_json_bytes(fresh.descriptor())
                    rebuilt = assemble_rcabench_repair_diagnostics(
                        fresh,
                        q_base=self.legacy.q_base,
                        q_max=self.legacy.q_max,
                    )
                else:
                    assert isinstance(sources, AIOps25LiveSources)
                    fresh = read_aiops25_live_sources(
                        root,
                        source_case_id=sources.source_case_id,
                        opaque_case_id=sources.adapter.opaque_case_id,
                        authorized_queue=sources.authorized_queue,
                        configuration=sources.adapter.configuration,
                    )
                    fresh_before = canonical_json_bytes(fresh.descriptor())
                    rebuilt = assemble_aiops25_repair_diagnostics(
                        fresh,
                        q_base=self.legacy.q_base,
                        q_max=self.legacy.q_max,
                    )
                matched = fresh_before == source_before and before == canonical_json_bytes(
                    rebuilt.diagnostic_descriptor()
                )
            finally:
                if (
                    source_before != canonical_json_bytes(sources.descriptor())
                    or before != canonical_json_bytes(self.diagnostic_descriptor())
                    or (
                        fresh is not None
                        and fresh_before is not None
                        and fresh_before != canonical_json_bytes(fresh.descriptor())
                    )
                ):
                    raise ValueError("repair diagnostic inputs changed during replay")
            return matched
        except (OSError, TypeError, ValueError, pa.ArrowException):
            return False


def _text(value: Any) -> str | None:
    return value if type(value) is str else None


def _status(value: Any) -> str | int | None:
    return value if type(value) in (str, int) else None


def _number(value: Any) -> int | float | None:
    if type(value) is int or (type(value) is float and math.isfinite(value)):
        return value
    return None


def _assemble(legacy: LiveInputs, dataset: Dataset) -> RepairDiagnosticInputs:
    metric_names: dict[str, str | None] = {}
    traces: dict[str, list[TraceErrorRecord]] = {"baseline": [], "fault": []}
    logs: dict[str, list[LogErrorRecord]] = {"baseline": [], "fault": []}
    for item in legacy.sources.tables:
        window, kind = item.audit.window, item.audit.source_kind
        table = item.table
        table = table.set_column(
            table.column_names.index("time"),
            "time",
            table.column("time").cast(pa.timestamp("ns", tz="UTC")).cast(pa.int64()),
        )
        # Batches bound Python conversion memory, not case coverage: every row
        # contributes to one complete-case result; no modalities/candidates cut.
        row_index = 0
        for batch in table.to_batches(max_chunksize=8192):
            for row in batch.to_pylist():
                _, entity = _identity(row)
                if kind.startswith("metrics"):
                    metric = row.get("metric")
                    attributes = {
                        k: v for k, v in row.items() if k.startswith("attr.") and v is not None
                    }
                    components = (
                        ("count", "sum", "min", "max")
                        if kind == "metrics_histogram"
                        else ("value",)
                    )
                    for component in components:
                        # Exact legacy identity, never infer a name from a hash.
                        sid = "sig_" + _digest([kind, entity, metric, attributes, component])[:24]
                        name = _text(metric)
                        if sid in metric_names and metric_names[sid] != name:
                            raise ValueError("conflicting metric identity")
                        metric_names[sid] = name
                else:
                    common = dict(
                        source_sha256=item.audit.source_sha256,
                        source_ref=f"{window}:{kind}:projected-row:{row_index}",
                        entity_key=entity,
                        timestamp_unix_ns=row["time"],
                    )
                    if kind == "traces":
                        http = row.get("attr.http.response.status_code")
                        traces[window].append(
                            TraceErrorRecord(
                                **common,
                                duration_ns=_number(row.get("duration")),
                                trace_id=_text(row.get("trace_id")),
                                span_id=_text(row.get("span_id")),
                                span_kind=_status(row.get("attr.span_kind")),
                                status_code=_status(row.get("attr.status_code")),
                                http_status_code=http if type(http) is str else _number(http),
                            )
                        )
                    elif kind == "logs":
                        logs[window].append(
                            LogErrorRecord(**common, severity=_text(row.get("level")))
                        )
                    else:
                        raise ValueError("unsupported source modality")
                row_index += 1
    signals = []
    for signal in legacy.signals:
        metric_name = metric_names.get(signal.signal_id)
        quality = summarize_repair_component(
            transform_result=signal.transform,
            component_name=signal.quality.component_name,
            context=signal.quality.context,
            configuration=signal.quality.configuration,
        )
        lifecycle = None
        unavailable: str | None = "not_registered_lifecycle_signal"
        if signal.signal_family == "lifecycle":
            if signal.transform.signal_type == "gauge" and metric_name is not None:
                lifecycle = summarize_lifecycle_component(
                    dataset=dataset,
                    metric_name=metric_name,
                    transform_result=signal.transform,
                    context=signal.quality.context,
                )
                unavailable = None
            else:
                unavailable = "lifecycle_source_transform_ineligible"
        signals.append(
            RepairSignalDiagnostic(signal.signal_id, metric_name, quality, lifecycle, unavailable)
        )
    contract = legacy.sources.adapter.time_contract
    errors = []
    for window, interval in (
        ("baseline", contract.baseline_window),
        ("fault", contract.fault_window),
    ):
        errors.append(
            summarize_trace_errors(records=traces[window], dataset=dataset, interval=interval)
        )
        errors.append(
            summarize_log_errors(records=logs[window], dataset=dataset, interval=interval)
        )
    return RepairDiagnosticInputs(dataset, legacy, tuple(signals), tuple(errors))


def assemble_rcabench_repair_diagnostics(
    sources: RCABenchLiveSources,
    *,
    q_base: int = 8,
    q_max: int = 12,
) -> RepairDiagnosticInputs:
    return _assemble(assemble_rcabench_live_inputs(sources, q_base=q_base, q_max=q_max), "rcabench")


def assemble_aiops25_repair_diagnostics(
    sources: AIOps25LiveSources,
    *,
    q_base: int = 8,
    q_max: int = 12,
) -> RepairDiagnosticInputs:
    return _assemble(assemble_aiops25_live_inputs(sources, q_base=q_base, q_max=q_max), "aiops25")
