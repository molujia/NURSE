"""Small immutable projections from an already-admitted case, not a new audit."""

from __future__ import annotations

from typing import Any, Literal

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.contracts import TruthValue
from nurse_rcl.v2.live.ops_contracts import NodeKind, OpsNodeEvidence
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData
from nurse_rcl.v2.observation_schemas import TimeInterval


def prepared_snapshot_id(data: RepairPreparationData) -> str:
    return sha256_hex(canonical_json_bytes(dict(
        dataset=data.diagnostics.dataset, opaque_case_id=data.catalog.opaque_case_id,
        source_descriptor_sha256=data.source_descriptor_sha256,
        source_files_sha256=data.source_files_sha256,
        candidate_freeze_sha256=data.candidate_freeze_sha256,
        catalog_sha256=data.catalog.catalog_sha256,
        graph_manifest_sha256=data.graph.manifest.manifest_sha256,
    )))


def node_card(
    *, snapshot: str, node: NodeKind, parameters: dict[str, Any], payload: dict[str, Any],
    evidence_ids: tuple[str, ...], source_refs: tuple[str, ...], interval: TimeInterval,
    unit: str | None = None, coverage: float | None = None,
    availability: Literal["available", "partial", "unavailable"] = "available",
    truth: TruthValue = "unknown", limitations: tuple[str, ...] = (),
) -> OpsNodeEvidence:
    return OpsNodeEvidence(
        case_snapshot_id=snapshot, node=node, tool_version="ops-" + node + "/1",
        parameters_json=canonical_json_bytes(parameters).decode("utf-8"),
        payload_json=canonical_json_bytes(payload).decode("utf-8"),
        evidence_ids=tuple(dict.fromkeys(evidence_ids)), source_refs=source_refs,
        unit=unit, window_ns=(interval.start_unix_ns, interval.end_unix_ns),
        coverage=coverage, availability=availability, truth_value=truth,
        limitations=limitations,
    )
