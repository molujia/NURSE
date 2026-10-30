"""Deterministic normal-B0 owner, with no investigation or credential input."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from time import monotonic
from typing import Any

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes
from nurse_rcl.v2.live.normal_queries import _read_bytes, run_normal_queries
from nurse_rcl.v2.live.normal_scoring import score_normal_case
from nurse_rcl.v2.live.ops_contracts import OpsCoreIdentity
from nurse_rcl.v2.live.ops_node_common import prepared_snapshot_id
from nurse_rcl.v2.live.ops_nodes import initial_nodes, terminal_nodes
from nurse_rcl.v2.live.ops_result import OpsCoreResult, seal_core_result
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData
from nurse_rcl.v2.live.repair_safety import REPAIR_MAX_JSON_BYTES
from nurse_rcl.v2.live.staged_case import require_plain_path


def run_ops_core(
    *, data: RepairPreparationData, identity: OpsCoreIdentity, root: Path, block_root: Path,
) -> OpsCoreResult:
    """Run from admitted preparation; only this owner writes core/result.json.

    Server admission/code pins and process capability separation belong to the
    outer runner. This private worker does not accept arbitrary telemetry paths.
    """
    if identity.case_snapshot_id != prepared_snapshot_id(data) or (
        identity.dataset != data.diagnostics.dataset
        or identity.opaque_case_id != data.catalog.opaque_case_id
    ):
        raise ValueError("core preparation identity differs")
    require_plain_path(root)
    core = root / "core"
    core.mkdir(mode=0o700, exist_ok=True)
    require_plain_path(core)
    descriptor = os.open(core / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(data=data, identity=identity, core=core, block_root=block_root)
    finally:
        os.close(descriptor)


def _run_locked(
    *, data: RepairPreparationData, identity: OpsCoreIdentity, core: Path, block_root: Path,
) -> OpsCoreResult:
    binding = canonical_json_bytes(identity.model_dump(mode="json"))
    identity_path = core / "identity.json"
    if identity_path.exists() or identity_path.is_symlink():
        if _read_bytes(identity_path, 16384) != binding:
            raise ValueError("core run identity changed; use a new run")
    else:
        if set(p.name for p in core.iterdir()) - {"lock"}:
            raise ValueError("core identity missing from existing work")
        atomic_write_bytes(identity_path, binding)
    result_path = core / "result.json"
    if result_path.exists() or result_path.is_symlink():
        result = OpsCoreResult.model_validate_json(_read_bytes(result_path, REPAIR_MAX_JSON_BYTES))
        if result.identity != identity:
            raise ValueError("saved core identity mismatch")
        return result
    started = monotonic()
    diagnostic_status: dict[str, Any] = {
        "initial_nodes": {"status": "completed"}, "terminal_nodes": {"status": "pending"},
    }
    try:
        initial = initial_nodes(data)
    except ValueError as exc:
        initial = ()
        diagnostic_status["initial_nodes"] = dict(
            status="unavailable", failure_type=type(exc).__name__, failure_reason=str(exc)[:2048],
        )
    atomic_write_bytes(core / "initial-nodes.json", canonical_json_bytes(dict(
        provisional=True, case_snapshot_id=identity.case_snapshot_id,
        projection_status=diagnostic_status["initial_nodes"],
        nodes=[n.model_dump(mode="json") for n in initial],
    )))
    initial_seconds = monotonic() - started
    query_start = monotonic()
    session = run_normal_queries(data=data, ledger_root=core / "queries", block_root=block_root)
    query_seconds = monotonic() - query_start
    scoring_start = monotonic()
    evidence = score_normal_case(data=data, session=session)
    scoring_seconds = monotonic() - scoring_start
    result = seal_core_result(identity=identity, evidence=evidence)
    diagnostic_status["core_result_digest"] = result.core_result_digest
    atomic_write_bytes(core / "diagnostics-status.json", canonical_json_bytes(diagnostic_status))
    # Freeze localization before optional presentation. A bounded card can fail
    # without discarding complete queries, conditions, terms and scores, all of
    # which remain in this full formal artifact.
    atomic_write_bytes(result_path, canonical_json_bytes(result.model_dump(mode="json")))
    nodes_start = monotonic()
    try:
        nodes = terminal_nodes(data, evidence)
        diagnostic_status["terminal_nodes"] = {"status": "completed"}
    except ValueError as exc:
        nodes = ()
        diagnostic_status["terminal_nodes"] = dict(
            status="unavailable", failure_type=type(exc).__name__, failure_reason=str(exc)[:2048],
        )
    node_seconds = monotonic() - nodes_start
    atomic_write_bytes(core / "diagnostics.json", canonical_json_bytes(dict(
        core_result_digest=result.core_result_digest, provisional=False,
        projection_status=diagnostic_status,
        nodes=[n.model_dump(mode="json") for n in (*initial, *nodes)],
    )))
    atomic_write_bytes(core / "diagnostics-status.json", canonical_json_bytes(diagnostic_status))
    atomic_write_bytes(core / "timing.json", canonical_json_bytes(dict(
        initial_nodes_seconds=initial_seconds, queries_seconds=query_seconds,
        scoring_seconds=scoring_seconds, terminal_nodes_seconds=node_seconds,
        core_elapsed_seconds=monotonic() - started,
    )))
    return result
