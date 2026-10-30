"""Whole-case stage entry: source-bound queries, complete ranking, then Report Agent.

Private adapter specifications are not model payloads. This module never opens
an evaluator map or label. No successful stage is overwritten on resume.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.aiops25_adapter import build_aiops25_adapter_config
from nurse_rcl.v2.aiops25_inventory import AuthorizedAIOps25Queue
from nurse_rcl.v2.live import repair_raw_audit as raw
from nurse_rcl.v2.live.aiops25_sources import read_aiops25_live_sources
from nurse_rcl.v2.live.repair_driver import (
    STAGES,
    publish_bundle,
    publish_json,
    read_bundle,
    read_json,
)
from nurse_rcl.v2.live.repair_query_session import RepairQuerySession
from nurse_rcl.v2.live.repair_report import run_repair_report
from nurse_rcl.v2.live.repair_safety import canonical
from nurse_rcl.v2.live.repair_timing import _preparation_call
from nurse_rcl.v2.live.staged_case import (
    load_staged_rcabench_case,
    read_inference_identity,
    require_plain_path,
)
from nurse_rcl.v2.opaque_ids import InferenceIdentityManifest
from nurse_rcl.v2.rcabench_adapter import build_rcabench_adapter_config


class RepairCaseInput(StrictFrozenModel):
    schema_version: Literal["nurse.v2.live-repair-case-input/1"] = (
        "nurse.v2.live-repair-case-input/1"
    )
    dataset: Literal["rcabench", "aiops25"]
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    source_root: str
    source_descriptor_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_files: tuple[tuple[str, str | None], ...]
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage_sha256: str | None
    source_case_id: str | None
    authorized_queue: AuthorizedAIOps25Queue | None


def stage_identity(execution_sha256: str, input_sha256: str, stage: str) -> str:
    if stage not in STAGES:
        raise ValueError("unregistered repair stage")
    return sha256_hex(canonical_json_bytes([execution_sha256, input_sha256, stage]))


def _read_model(path: Path, model: Any) -> Any:
    source = require_plain_path(path)
    if not source.is_file() or source.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("stage artifact exceeds fixed repair capacity")
    return model.model_validate_json(source.read_bytes())


def _completed_bundle(state: Path, stage: str, identity: str) -> dict[str, Any]:
    for previous in STAGES:
        failure = state / f"failure-{previous}.json"
        if failure.exists() or failure.is_symlink():
            raise ValueError("stage has a retained terminal failure")
        if previous == stage:
            break
    completion = state / f"success-{stage}.json"
    if not completion.exists() and not completion.is_symlink():
        raise ValueError("stage publication is incomplete; owning exits are not confirmed")
    receipt = read_bundle(state / stage, identity=identity)
    if read_json(completion) != dict(identity=identity, stage=stage, receipt=receipt):
        raise ValueError("stage publication completion identity differs")
    return receipt


@_preparation_call("input_validation")
def validate_input(
    *,
    specification: dict[str, Any],
    expected_input_sha256: str,
    identity_manifest_path: Path,
    expected_identity_artifact_sha256: str,
    state_root: Path,
) -> tuple[RepairCaseInput, InferenceIdentityManifest, dict[str, Any]]:
    encoded = canonical_json_bytes(specification)
    if sha256_hex(encoded) != expected_input_sha256:
        raise ValueError("case input differs from independent frozen pin")
    case = RepairCaseInput.model_validate_json(encoded)
    manifest = read_inference_identity(identity_manifest_path, expected_identity_artifact_sha256)
    if not any(
        r.dataset == case.dataset and r.opaque_case_id == case.opaque_case_id
        for r in manifest.records
    ):
        raise ValueError("case is absent from inference identity manifest")
    root = require_plain_path(Path(case.source_root))
    state = require_plain_path(state_root)
    for path in (state / "blocks", state / "query-ledger"):
        if path.exists() or path.is_symlink():
            require_plain_path(path)
    if raw._file_pins(root, case.dataset) != case.source_files:
        raise ValueError("complete source file inventory differs from frozen input")
    if case.dataset == "rcabench":
        config = build_rcabench_adapter_config(window_seconds=240)
        if (
            case.stage_sha256 is None
            or case.source_case_id is not None
            or case.authorized_queue is not None
        ):
            raise ValueError("RCABench needs a pinned opaque stage, not private identity inputs")
        source = load_staged_rcabench_case(
            root,
            expected_stage_sha256=case.stage_sha256,
            inference_manifest_path=identity_manifest_path,
            expected_inference_artifact_sha256=expected_identity_artifact_sha256,
            configuration=config,
        )
        descriptor_pin = source.descriptor()["descriptor_sha256"]
        extra: dict[str, Any] = {}
    else:
        if (
            case.stage_sha256 is not None
            or case.source_case_id is None
            or case.authorized_queue is None
        ):
            raise ValueError("AIOps25 needs a complete authorized prepared source")
        config25 = build_aiops25_adapter_config(window_seconds=480)
        source25 = read_aiops25_live_sources(
            root,
            source_case_id=case.source_case_id,
            opaque_case_id=case.opaque_case_id,
            authorized_queue=case.authorized_queue,
            configuration=config25,
        )
        descriptor_pin = source25.descriptor()["descriptor_sha256"]
        extra = dict(source_case_id=case.source_case_id, authorized_queue=case.authorized_queue)
    configuration = config if case.dataset == "rcabench" else config25
    if (
        configuration.configuration_sha256 != case.configuration_sha256
        or descriptor_pin != case.source_descriptor_sha256
    ):
        raise ValueError("source descriptor or adapter configuration differs from frozen input")
    return (
        case,
        manifest,
        dict(
            source_root=root,
            dataset=case.dataset,
            opaque_case_id=case.opaque_case_id,
            configuration=configuration,
            expected_source_descriptor_sha256=case.source_descriptor_sha256,
            store_root=state / "blocks",
            ledger_root=state / "query-ledger",
            **extra,
        ),
    )


def run_stage(
    *,
    stage: str,
    specification: dict[str, Any],
    expected_input_sha256: str,
    identity_manifest_path: Path,
    expected_identity_artifact_sha256: str,
    state_root: Path,
    execution_sha256: str,
    complete: Callable[[bytes], bytes],
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    emit = progress if progress is not None else lambda _: None
    identity = stage_identity(execution_sha256, expected_input_sha256, stage)
    state = require_plain_path(state_root)
    case, manifest, arguments = validate_input(
        specification=specification,
        expected_input_sha256=expected_input_sha256,
        identity_manifest_path=identity_manifest_path,
        expected_identity_artifact_sha256=expected_identity_artifact_sha256,
        state_root=state,
    )
    emit(
        dict(
            event="case_input_validated",
            stage=stage,
            dataset=case.dataset,
            opaque_case_id=case.opaque_case_id,
        )
    )
    # Changing the execution or source cannot obtain a fresh ledger in this case.
    publish_json(
        state / "identity.json",
        dict(
            execution_sha256=execution_sha256,
            input_sha256=expected_input_sha256,
            opaque_case_id=case.opaque_case_id,
        ),
    )
    failure = state / f"failure-{stage}.json"
    for previous in STAGES:
        retained = state / f"failure-{previous}.json"
        if retained.exists() or retained.is_symlink():
            raise ValueError("stage has a retained terminal failure; no automatic rerun")
        if previous == stage:
            break
    target = state / stage
    if target.exists() or target.is_symlink():
        receipt = _completed_bundle(state, stage, identity)
        emit(dict(event="sealed_stage_reused", stage=stage))
        return receipt
    encoded_input = canonical_json_bytes(specification)

    @_preparation_call("stage_input_exit")
    def unchanged() -> None:
        if (
            canonical_json_bytes(specification) != encoded_input
            or raw._file_pins(Path(case.source_root), case.dataset) != case.source_files
        ):
            raise ValueError("case source or input changed during stage")
        # Stage receipt and manifest are independent input identity as well.
        read_inference_identity(identity_manifest_path, expected_identity_artifact_sha256)

    try:
        files: dict[str, bytes]
        if stage == "queries":
            session = raw.run_repair_queries_from_raw(**arguments, progress=emit)
            files = {"session.json": canonical(session, (RepairQuerySession,))}
            emit(
                dict(
                    event="queries_completed",
                    paid_count=session.paid_count,
                    session_sha256=session.session_sha256,
                )
            )
        else:
            _completed_bundle(
                state,
                "queries",
                stage_identity(execution_sha256, expected_input_sha256, "queries"),
            )
            session = _read_model(state / "queries/session.json", RepairQuerySession)
            if session.failure_reason or not session.stop_reason:
                raise ValueError("downstream stage requires a terminal successful query ledger")
            if stage == "audit":
                audit = raw.build_repair_raw_audit(
                    expected_session_sha256=session.session_sha256, **arguments
                )
                files = {
                    "audit.json": canonical(audit, (raw.RepairRawAudit,)),
                    "summary.json": canonical_json_bytes(
                        dict(
                            audit_sha256=audit.result_sha256,
                            ranking_sha256=audit.ranking.result_sha256,
                            session_sha256=session.session_sha256,
                            candidates=len(audit.ranking.scores),
                        )
                    ),
                }
                emit(
                    dict(
                        event="audit_completed",
                        candidates=len(audit.ranking.scores),
                        audit_sha256=audit.result_sha256,
                    )
                )
            else:
                _completed_bundle(
                    state,
                    "audit",
                    stage_identity(execution_sha256, expected_input_sha256, "audit"),
                )
                audit = _read_model(state / "audit/audit.json", raw.RepairRawAudit)
                result = run_repair_report(
                    audit=audit,
                    expected_audit_sha256=audit.result_sha256,
                    expected_session_sha256=session.session_sha256,
                    identity_manifest=manifest,
                    expected_identity_manifest_sha256=manifest.manifest_sha256,
                    complete=complete,
                    progress=lambda row: emit(dict(event="report_progress", **row)),
                    **arguments,
                )
                files = {"report.json": canonical_json_bytes(result.model_dump(mode="json"))}
                emit(
                    dict(
                        event="report_completed",
                        status=result.status,
                        exchange_status=result.exchange_status,
                        semantic_status="not_established",
                    )
                )
        unchanged()
        receipt = publish_bundle(target, files, identity=identity)
    except BaseException as error:
        publish_json(
            failure,
            dict(identity=identity, stage=stage, status="failed", error_type=type(error).__name__),
        )
        raise
    finally:
        try:
            unchanged()
        except BaseException as error:
            # A published bundle is not success if a post-publication guard fails.
            if not failure.exists():
                publish_json(
                    failure,
                    dict(
                        identity=identity,
                        stage=stage,
                        status="failed",
                        error_type=type(error).__name__,
                    ),
                )
            raise
    # A crash before this durable marker leaves a bundle, never success authority.
    publish_json(
        state / f"success-{stage}.json", dict(identity=identity, stage=stage, receipt=receipt)
    )
    return receipt
