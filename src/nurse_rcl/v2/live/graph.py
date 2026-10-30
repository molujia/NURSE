"""Trace-parent service calls, without inventing collector coverage or causality."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa  # type: ignore[import-untyped]
from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.event_grouping import AdapterAssociationSourceField
from nurse_rcl.v2.graph_snapshots import (
    FrozenGraphInputManifest,
    GraphSnapshotPair,
    build_frozen_graph_input_manifest,
    build_graph_adapter_parent_record,
    build_graph_snapshot_pair,
)
from nurse_rcl.v2.live.assembly import SEMANTICS_SHA256, LiveInputs, _identity
from nurse_rcl.v2.observation_schemas import TimeInterval


class LiveTraceCall(StrictFrozenModel):
    caller: str
    callee: str
    request_count: int = Field(gt=0)
    child_duration_p95_seconds: float = Field(ge=0, allow_inf_nan=False)


class LiveTraceExport(StrictFrozenModel):
    """Actual producer export, reconstructed from pinned telemetry at the live boundary.

    This is NOT a collector-provided export ID. The explicit live export format
    binds complete input projection and deterministic measurements. Hashes alone
    do not authorize it: build/verify both replay the underlying source files.
    """

    schema_version: Literal["nurse.v2.live-trace-export/1"] = "nurse.v2.live-trace-export/1"
    derivation: Literal["exact-trace-parent-child-duration/1"] = (
        "exact-trace-parent-child-duration/1"
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    input_descriptor_sha256: str
    observation_semantics_sha256: str = SEMANTICS_SHA256
    window_kind: Literal["normal", "fault"]
    interval: TimeInterval
    trace_projection_pins: tuple[tuple[str, str], ...]
    collection_state: Literal["unknown"] = "unknown"
    calls: tuple[LiveTraceCall, ...]
    accounting: tuple[tuple[str, int], ...]

    @property
    def export_id(self) -> str:
        # Producer-issued document identity, separate from its content digest.
        # Identity is case/window scoped; immutable content is pinned below.
        return (
            f"export:live-trace-v1-{self.opaque_case_id[4:]}-{self.window_kind}"
            f"-{self.interval.start_unix_ns}-{self.interval.end_unix_ns}"
        )

    @property
    def export_sha256(self) -> str:
        return sha256_hex(canonical_json_bytes(self.model_dump(mode="json")))


@dataclass(frozen=True)
class LiveGraph:
    exports: tuple[LiveTraceExport, ...]
    manifest: FrozenGraphInputManifest
    snapshots: GraphSnapshotPair
    accounting: dict[str, int]

    def verify_sources(self, inputs: LiveInputs, root: Path) -> bool:
        try:
            return self == build_live_graph(inputs, source_root=root)
        except (ValueError, OSError, pa.ArrowException):
            return False


def _valid_trace_identifier(value: object, width: int) -> bool:
    return (
        type(value) is str
        and re.fullmatch("[0-9a-f]{" + str(width) + "}", value) is not None
        and value != "0" * width
    )


def build_live_graph(inputs: LiveInputs, *, source_root: Path) -> LiveGraph:
    if type(inputs) is not LiveInputs or not inputs.verify_sources(source_root):
        raise ValueError("live graph input source replay failed")
    return _compile_live_graph(inputs)


def _compile_live_graph(inputs: LiveInputs) -> LiveGraph:
    """Pure construction only; the enclosing caller owns independent file replay."""
    contract = inputs.sources.adapter.time_contract
    contract_pin = inputs.candidate_universe.observation_contract_sha256
    input_pin = inputs.sources.descriptor()["descriptor_sha256"]
    total_counts: Counter[str] = Counter()
    parents = []
    exports = []
    windows: tuple[tuple[str, Literal["normal", "fault"], TimeInterval], ...] = (
        ("baseline", "normal", contract.baseline_window),
        ("fault", "fault", contract.fault_window),
    )
    for window, kind, interval in windows:
        counts: Counter[str] = Counter()
        tables = [
            s
            for s in inputs.sources.tables
            if s.audit.source_kind == "traces" and s.audit.window == window
        ]
        pins = tuple(
            sorted((s.audit.source_sha256, s.descriptor()["projection_sha256"]) for s in tables)
        )
        spans: dict[tuple[str, str], set[tuple[Any, ...]]] = defaultdict(set)
        for source in tables:
            table = source.table.set_column(
                0,
                "time",
                source.table.column("time").cast(pa.timestamp("ns", tz="UTC")).cast(pa.int64()),
            )
            for row in table.to_pylist():
                trace, span = row.get("trace_id"), row.get("span_id")
                service, _ = _identity(row)
                duration, timestamp = row.get("duration"), row["time"]
                parent = row.get("parent_span_id")
                # OTel empty/all-zero parent is a root, not a joinable span.
                if parent in (None, "", "0" * 16):
                    parent = None
                if (
                    not _valid_trace_identifier(trace, 32)
                    or not _valid_trace_identifier(span, 16)
                    or (parent is not None and not _valid_trace_identifier(parent, 16))
                    or type(duration) not in (int, float)
                    or not math.isfinite(duration)
                    or duration < 0
                    or timestamp + duration > interval.end_unix_ns
                ):
                    counts["invalid_or_unfinished_span"] += 1
                    continue
                spans[(trace, span)].add((service, parent, duration, timestamp))
        ambiguous = {key for key, values in spans.items() if len(values) != 1}
        counts["ambiguous_span_identity"] += len(ambiguous)
        resolved = {
            key: next(iter(values)) for key, values in spans.items() if key not in ambiguous
        }
        edges: dict[tuple[str, str], list[float]] = defaultdict(list)
        for (trace, _), (callee, parent, duration, _) in resolved.items():
            if not parent:
                continue
            upstream = resolved.get((trace, parent))
            if upstream is None:
                counts["unmatched_parent"] += 1
                continue
            caller = upstream[0]
            if caller is None or callee is None:
                counts["unbound_endpoint"] += 1
                continue
            if caller != callee:
                edges[(caller, callee)].append(float(duration) / 1e9)
        export = LiveTraceExport(
            opaque_case_id=contract.opaque_case_id,
            input_descriptor_sha256=input_pin,
            window_kind=kind,
            interval=interval,
            trace_projection_pins=pins,
            calls=tuple(
                LiveTraceCall(
                    caller=caller,
                    callee=callee,
                    request_count=len(durations),
                    child_duration_p95_seconds=sorted(durations)[
                        math.ceil(0.95 * len(durations)) - 1
                    ],
                )
                for (caller, callee), durations in sorted(edges.items())
            ),
            accounting=tuple(sorted(counts.items())),
        )
        exports.append(export)
        total_counts.update(counts)
        provenance = (
            AdapterAssociationSourceField(field_kind="export_id", raw_value=export.export_id),
        )
        # Presence of spans is not a measured collection coverage. The actual
        # export says unknown, including when the trace source is absent.
        parents.append(
            build_graph_adapter_parent_record(
                opaque_case_id=contract.opaque_case_id,
                observation_contract_sha256=contract_pin,
                window_kind=kind,
                interval=interval,
                parent_artifact_kind="coverage_samples",
                collection_state=export.collection_state,
                source_identity_fields=provenance,
                provenance_fields=provenance,
            )
        )
        for index, call in enumerate(export.calls):
            # This is the actual row selector inside the retained sorted export,
            # not a prehashed raw trace identifier.
            selector = f"relation:call-{index}"
            parents.append(
                build_graph_adapter_parent_record(
                    opaque_case_id=contract.opaque_case_id,
                    observation_contract_sha256=contract_pin,
                    window_kind=kind,
                    interval=interval,
                    parent_artifact_kind="call_measurement",
                    caller_service_key=call.caller,
                    callee_service_key=call.callee,
                    relation_type="service_call",
                    direction="caller_to_callee",
                    observation_state="observed",
                    request_count=call.request_count,
                    error_count=None,
                    latency_p95_seconds=call.child_duration_p95_seconds,
                    source_identity_fields=(
                        *provenance,
                        AdapterAssociationSourceField(
                            field_kind="source_relation_id", raw_value=selector
                        ),
                    ),
                    provenance_fields=provenance,
                )
            )
    manifest = build_frozen_graph_input_manifest(
        opaque_case_id=contract.opaque_case_id,
        observation_contract_sha256=contract_pin,
        parent_records=tuple(parents),
    )
    pair = build_graph_snapshot_pair(
        time_contract=contract,
        candidate_universe=inputs.candidate_universe,
        frozen_candidates=inputs.frozen_candidates,
        entity_registry=inputs.entity_registry,
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
    )
    return LiveGraph(tuple(exports), manifest, pair, dict(sorted(total_counts.items())))
