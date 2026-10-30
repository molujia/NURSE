"""Bounded read-only import of a pinned completed core, never an old agent trace."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.live.normal_execution import _metadata, _pinned
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_contracts import OpsCoreIdentity, OpsNodeEvidence
from nurse_rcl.v2.live.ops_plan import OpsAdmittedPlan, OpsCaseReference, plain_location
from nurse_rcl.v2.live.ops_result import OpsCoreResult
from nurse_rcl.v2.live.repair_driver import _fsync
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationRun
from nurse_rcl.v2.live.repair_safety import REPAIR_MAX_JSON_BYTES
from nurse_rcl.v2.live.staged_case import require_plain_path

CORE_MEMBERS = (
    "core/identity.json",
    "core/result.json",
    "core/initial-nodes.json",
    "core/diagnostics.json",
    "core/diagnostics-status.json",
    "core/timing.json",
)


class _SourceCoreMetadata(StrictFrozenModel):
    """Read-only provenance, intentionally not an executable/resumable plan."""

    schema_version: Literal["nurse.v2.ops-single-plan/1"]
    code_files: dict[str, SHA256]
    environment: dict[str, str]
    case: OpsCaseReference
    preparation: RepairPreparationRun
    output_root: str
    mode: Literal["core-only", "with-agent"]
    profile_id: Literal[
        "v2-ops-isolated/1",
        "v2-ops-isolated/2",
        "v2-ops-sre/1",
        "v2-ops-sre/2",
        "v2-ops-sre/3",
        "v2-ops-sre/4",
        "v2-ops-sre/5",
        "v2-ops-sre/6",
        "v2-ops-sre/7",
        "v2-ops-sre/8",
        "v2-ops-sre/9",
    ]
    core_method_id: Literal["normal-b0-ops-core/1"]


def _source_metadata(path: Path, pin: str) -> _SourceCoreMetadata:
    raw = _metadata(_pinned(path, pin))
    # No model, transport, output policy or retired protocol is made runnable.
    # Those fields remain pinned in the original plan; only core provenance is read.
    fields = {
        k: raw[k]
        for k in _SourceCoreMetadata.model_fields
        if k not in {"profile_id", "core_method_id"}
    }
    fields.update({k: raw["profile"][k] for k in ("profile_id", "core_method_id")})
    return _SourceCoreMetadata.model_validate_json(canonical_json_bytes(fields))


def source_core_bytes(
    admitted: OpsAdmittedPlan,
    identity: OpsCoreIdentity,
) -> tuple[OpsCoreResult, dict[str, bytes], dict[str, Any]]:
    plan = admitted.plan
    reference = plan.source_core
    if reference is None:
        raise ValueError("source core reference missing")
    source_plan = _source_metadata(Path(reference.plan_path), reference.plan_sha256)
    source_root = require_plain_path(
        Path(source_plan.output_root) / source_plan.case.opaque_case_id
    )
    output = plain_location(Path(plan.output_root))
    if output.is_relative_to(source_root.parent) or source_root.parent.is_relative_to(output):
        raise ValueError("source core output overlaps historical run")
    if source_plan.case != plan.case or (
        source_plan.preparation.key != plan.preparation.key
        or source_plan.environment != plan.environment
        or source_plan.core_method_id != identity.core_method_id
    ):
        raise ValueError("source core plan compatibility differs")
    receipt = _metadata(_pinned(source_root / "case-result.json", reference.case_result_sha256))
    unsigned = {k: v for k, v in receipt.items() if k != "result_sha256"}
    source_identity = dict(
        plan_sha256=reference.plan_sha256,
        code_sha256=sha256_hex(canonical_json_bytes(source_plan.code_files)),
        opaque_case_id=source_plan.case.opaque_case_id,
        profile=source_plan.profile_id,
        mode=source_plan.mode,
    )
    if receipt.get("identity") != source_identity or (
        receipt.get("channels", {}).get("core") != "completed"
        or receipt.get("result_sha256") != sha256_hex(canonical_json_bytes(unsigned))
    ):
        raise ValueError("source core case receipt differs")
    files = receipt.get("files", {})
    if not set(CORE_MEMBERS) <= files.keys():
        raise ValueError("source core publications incomplete")
    blobs = {}
    for name in CORE_MEMBERS:
        raw = _read_bytes(source_root / name, REPAIR_MAX_JSON_BYTES)
        if [len(raw), sha256_hex(raw)] != files[name]:
            raise ValueError("source core member integrity differs")
        blobs[name] = raw
    core = OpsCoreResult.model_validate_json(blobs["core/result.json"])
    if (
        core.identity != identity
        or blobs["core/identity.json"]
        != canonical_json_bytes(
            identity.model_dump(mode="json"),
        )
        or receipt["core_result_digest"] != core.core_result_digest
    ):
        raise ValueError("source core identity differs from prepared case")
    # The diagnostics are a publication, not a second scorer. Inspect only their
    # saved binding and node schemas, without running the projector again.
    import json

    diagnostics = json.loads(blobs["core/diagnostics.json"])
    if diagnostics.get("core_result_digest") != core.core_result_digest or (
        diagnostics.get("provisional") is not False
    ):
        raise ValueError("source core diagnostics binding differs")
    for node in diagnostics["nodes"]:
        if OpsNodeEvidence.model_validate_json(canonical_json_bytes(node)).case_snapshot_id != (
            identity.case_snapshot_id
        ):
            raise ValueError("source core diagnostic snapshot differs")
    provenance = dict(
        schema_version="nurse.v2.sre-source-core/1",
        source=reference.model_dump(mode="json"),
        source_root=str(source_root),
        core_result_digest=core.core_result_digest,
        core_members={n: files[n] for n in CORE_MEMBERS},
        compatibility="same_method_snapshot_configuration_environment_and_preparation_key",
        reuse="core_only_no_investigation_report_or_provider_history",
        timing="historical_core_time_not_new_method_execution",
    )
    return core, blobs, provenance


def import_source_core(admitted: OpsAdmittedPlan, identity: OpsCoreIdentity) -> OpsCoreResult:
    core, blobs, provenance = source_core_bytes(admitted, identity)
    root = admitted.state_root
    directory = root / "core"
    plain_location(directory)
    directory.mkdir(mode=0o700, exist_ok=True)
    require_plain_path(directory)
    if {p.name for p in directory.iterdir()} - {Path(n).name for n in CORE_MEMBERS} - {"lock"}:
        raise ValueError("source core import cannot replace independently started core work")
    # Check all existing targets before writing any missing publication. Crash
    # recovery fills missing exact bytes; no overwrite of mismatched work.
    blobs["source-core.json"] = canonical_json_bytes(provenance)
    for name, raw in blobs.items():
        target = root / name
        if (target.exists() or target.is_symlink()) and (
            _read_bytes(target, REPAIR_MAX_JSON_BYTES) != raw
        ):
            raise ValueError("source core destination differs")
    for name, raw in blobs.items():
        target = root / name
        if not target.exists():
            atomic_write_bytes(target, raw)
    _fsync(directory)
    _fsync(root)
    return core
