"""Current dataset graph context without inventing lost Jaeger reference meaning.

RCABench retains its existing source-replayed graph. The mandated AIOps25
preparer takes the first references[*].spanID but discards refType and traceID;
such a prepared parent field alone cannot license a directed service call.
All underlying rows stay in pinned whole-case sources. This intermediate
artifact is not independent raw-file authority for a final score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal, cast

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.dynamic_verification.safety import plain, require_pin
from nurse_rcl.v2.event_grouping import AdapterAssociationSourceField
from nurse_rcl.v2.graph_snapshots import (
    FrozenGraphInputManifest,
    GraphSnapshotPair,
    build_frozen_graph_input_manifest,
    build_graph_adapter_parent_record,
    build_graph_snapshot_pair,
)
from nurse_rcl.v2.live.assembly import AIOps25LiveInputs, LiveInputs
from nurse_rcl.v2.live.graph import LiveTraceExport, _compile_live_graph
from nurse_rcl.v2.live.repair_safety import RepairModel, canonical, require_unchanged, snapshot
from nurse_rcl.v2.observation_schemas import TimeInterval

REPAIR_GRAPH_SCHEMA_VERSION: Final = "nurse.v2.live-repair-graph/1"
REPAIR_GRAPH_SEMANTICS_SHA256: Final = sha256_hex(
    canonical_json_bytes(
        dict(
            version="nurse-v2-live-repair-graph/1",
            rcabench="unchanged_exact_trace_parent_child_duration_live_graph",
            aiops25="prepared_first_spanID_reference_lacks_refType_and_reference_traceID",
            missing="untyped_reference_not_directed_call_or_verified_absence",
            completeness="all_trace_rows_counted_pinned_full_case_retained_all_C",
            coverage="unknown_not_span_presence",
        )
    )
)


class RepairPreparedReferenceWindow(RepairModel):
    window_kind: Literal["normal", "fault"]
    interval: TimeInterval
    source_projection_sha256s: tuple[str, ...]
    trace_row_count: int = Field(ge=0)
    reference_row_count: int = Field(ge=0)
    relation_state: Literal["unknown"] = "unknown"
    collection_state: Literal["unknown"] = "unknown"
    reason: Literal["prepared_reference_kind_and_trace_binding_unavailable"] = (
        "prepared_reference_kind_and_trace_binding_unavailable"
    )


class RepairLiveGraph(RepairModel):
    schema_version: Literal["nurse.v2.live-repair-graph/1"] = REPAIR_GRAPH_SCHEMA_VERSION
    semantics_sha256: SHA256
    dataset: Literal["rcabench", "aiops25"]
    opaque_case_id: str
    input_descriptor_sha256: SHA256
    manifest: FrozenGraphInputManifest
    graph_pair: GraphSnapshotPair
    rcabench_exports: tuple[LiveTraceExport, ...]
    aiops25_reference_windows: tuple[RepairPreparedReferenceWindow, ...]
    limitations: tuple[str, ...]
    raw_file_authority: Literal[False] = False
    result_sha256: SHA256


_ROOTS = (RepairLiveGraph,)


def _source_descriptor(inputs: LiveInputs) -> bytes:
    if type(inputs) not in (LiveInputs, AIOps25LiveInputs):
        raise ValueError("repair graph requires an exact dataset input type")
    if type(inputs.signals) is not tuple or type(inputs.sources.tables) is not tuple:
        raise ValueError("repair graph input representation changed")
    return canonical_json_bytes(inputs.inference_descriptor())


def _compile_graph(inputs: LiveInputs, source_root: Path, descriptor: bytes) -> RepairLiveGraph:
    legacy = inputs
    contract = inputs.sources.adapter.time_contract
    exports: tuple[LiveTraceExport, ...] = ()
    windows: list[RepairPreparedReferenceWindow] = []
    if type(inputs) is LiveInputs:
        original = _compile_live_graph(inputs)
        manifest, pair, exports = original.manifest, original.snapshots, original.exports
        limitations: tuple[str, ...] = (
            "collection_coverage_unknown",
            "service_call_not_confirmed_failure",
        )
        dataset: Literal["rcabench", "aiops25"] = "rcabench"
    else:
        dataset = "aiops25"
        parents = []
        for window, kind, interval in (
            ("baseline", "normal", contract.baseline_window),
            ("fault", "fault", contract.fault_window),
        ):
            tables = tuple(
                t
                for t in inputs.sources.tables
                if t.audit.source_kind == "traces" and t.audit.window == window
            )
            count = 0
            for item in tables:
                if "parent_span_id" in item.table.column_names:
                    # Read every column value; this is provenance accounting,
                    # not a new measurement, inferred edge or coverage claim.
                    count += sum(
                        type(v) is str and bool(v)
                        for v in item.table.column("parent_span_id").to_pylist()
                    )
            windows.append(
                RepairPreparedReferenceWindow(
                    window_kind=cast(Literal["normal", "fault"], kind),
                    interval=interval,
                    source_projection_sha256s=tuple(
                        sorted(t.descriptor()["projection_sha256"] for t in tables)
                    ),
                    trace_row_count=sum(t.table.num_rows for t in tables),
                    reference_row_count=count,
                )
            )
            provenance = (
                AdapterAssociationSourceField(
                    field_kind="export_id",
                    raw_value=f"export:repair-aiops25-reference-{contract.opaque_case_id}-{kind}",
                ),
            )
            parents.append(
                build_graph_adapter_parent_record(
                    opaque_case_id=contract.opaque_case_id,
                    observation_contract_sha256=legacy.candidate_universe.observation_contract_sha256,
                    window_kind=cast(Literal["normal", "fault"], kind),
                    interval=interval,
                    parent_artifact_kind="coverage_samples",
                    collection_state="unknown",
                    source_identity_fields=provenance,
                    provenance_fields=provenance,
                )
            )
        manifest = build_frozen_graph_input_manifest(
            opaque_case_id=contract.opaque_case_id,
            observation_contract_sha256=legacy.candidate_universe.observation_contract_sha256,
            parent_records=tuple(parents),
        )
        pair = build_graph_snapshot_pair(
            time_contract=contract,
            candidate_universe=legacy.candidate_universe,
            frozen_candidates=legacy.frozen_candidates,
            entity_registry=legacy.entity_registry,
            graph_input_manifest=manifest,
            expected_graph_input_manifest_sha256=manifest.manifest_sha256,
        )
        limitations = (
            "collection_coverage_unknown",
            "prepared_reference_kind_and_trace_binding_unavailable",
            "no_licensed_edges_not_verified_absence",
        )
    result = RepairLiveGraph(
        semantics_sha256=REPAIR_GRAPH_SEMANTICS_SHA256,
        dataset=dataset,
        opaque_case_id=contract.opaque_case_id,
        input_descriptor_sha256=sha256_hex(descriptor),
        manifest=manifest,
        graph_pair=pair,
        rcabench_exports=exports,
        aiops25_reference_windows=tuple(windows),
        limitations=limitations,
        result_sha256="0" * 64,
    )
    unsigned = cast(dict[str, object], plain(result))
    unsigned.pop("result_sha256")
    return result.model_copy(update={"result_sha256": sha256_hex(canonical_json_bytes(unsigned))})


def build_repair_live_graph(*, inputs: LiveInputs, source_root: Path) -> RepairLiveGraph:
    before = _source_descriptor(inputs)
    try:
        if not inputs.verify_sources(source_root):
            raise ValueError("repair graph input source replay failed")
        result = _compile_graph(inputs, source_root, before)
        canonical(result, _ROOTS)
        return result
    finally:
        if before != _source_descriptor(inputs):
            raise ValueError("repair graph input content or representation changed")


def require_repair_live_graph(
    *, graph: RepairLiveGraph, expected_graph_sha256: str, inputs: LiveInputs, source_root: Path
) -> RepairLiveGraph:
    if type(graph) is not RepairLiveGraph:
        raise ValueError("exact current repair graph required")
    before = snapshot(graph, _ROOTS)
    try:
        require_pin(expected_graph_sha256)
        rebuilt = build_repair_live_graph(inputs=inputs, source_root=source_root)
        if rebuilt.result_sha256 != expected_graph_sha256 or canonical(
            rebuilt, _ROOTS
        ) != canonical(graph, _ROOTS):
            raise ValueError("repair graph differs from source replay")
        return rebuilt
    finally:
        require_unchanged(graph, _ROOTS, before)
