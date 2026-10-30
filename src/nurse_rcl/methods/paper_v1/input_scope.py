"""Freeze the fixed V1 telemetry scope while keeping logs audit-only."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import (
    DatasetName,
    InferenceCase,
    StrictFrozenModel,
    TelemetryModality,
    TelemetryReference,
)

_RANK_BEARING_MODALITIES = ("metrics", "normal_traces", "fault_traces")
_HASH_CHUNK_SIZE = 1024 * 1024


class TelemetryFileAudit(StrictFrozenModel):
    """Content identity and basic completeness statistics for one reference."""

    modality: TelemetryModality
    path: Path
    exists: bool
    non_empty: bool
    byte_size: int = Field(ge=0)
    sha256: str | None


class RankBearingInputScope(StrictFrozenModel):
    """The only telemetry identities allowed to influence fixed V1 ranking."""

    case_id: str
    dataset: DatasetName
    t0: int
    metrics: tuple[TelemetryFileAudit, ...]
    normal_traces: tuple[TelemetryFileAudit, ...]
    fault_traces: tuple[TelemetryFileAudit, ...]
    rank_input_fingerprint: str


class LogModalityAudit(StrictFrozenModel):
    """Audit-only log statistics that are excluded from rank-bearing inputs."""

    files: tuple[TelemetryFileAudit, ...]
    reference_count: int = Field(ge=0)
    present_file_count: int = Field(ge=0)
    non_empty_file_count: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    complete: bool
    log_fingerprint: str


def scope_v1_inputs(case: InferenceCase) -> tuple[RankBearingInputScope, LogModalityAudit]:
    """Return separate rank-bearing telemetry and audit-only log identities."""

    audits = tuple(
        sorted(
            (_audit_reference(reference) for reference in case.telemetry),
            key=lambda audit: (
                audit.modality,
                audit.path.as_posix().casefold(),
                audit.path.as_posix(),
            ),
        )
    )
    by_modality = {
        modality: tuple(audit for audit in audits if audit.modality == modality)
        for modality in _RANK_BEARING_MODALITIES
    }
    rank_payload = {
        "case_id": case.case_id,
        "dataset": case.dataset,
        "t0": case.t0,
        **{
            modality: [audit.model_dump(mode="json") for audit in by_modality[modality]]
            for modality in _RANK_BEARING_MODALITIES
        },
    }
    rank_scope = RankBearingInputScope(
        case_id=case.case_id,
        dataset=case.dataset,
        t0=case.t0,
        metrics=by_modality["metrics"],
        normal_traces=by_modality["normal_traces"],
        fault_traces=by_modality["fault_traces"],
        rank_input_fingerprint=_fingerprint(rank_payload),
    )

    log_files = tuple(audit for audit in audits if audit.modality == "logs")
    present_file_count = sum(audit.exists for audit in log_files)
    non_empty_file_count = sum(audit.non_empty for audit in log_files)
    total_bytes = sum(audit.byte_size for audit in log_files)
    complete = present_file_count == len(log_files) and non_empty_file_count == len(log_files)
    log_payload = {
        "files": [audit.model_dump(mode="json") for audit in log_files],
        "reference_count": len(log_files),
        "present_file_count": present_file_count,
        "non_empty_file_count": non_empty_file_count,
        "total_bytes": total_bytes,
        "complete": complete,
    }
    log_audit = LogModalityAudit(
        files=log_files,
        reference_count=len(log_files),
        present_file_count=present_file_count,
        non_empty_file_count=non_empty_file_count,
        total_bytes=total_bytes,
        complete=complete,
        log_fingerprint=_fingerprint(log_payload),
    )
    return rank_scope, log_audit


def _audit_reference(reference: TelemetryReference) -> TelemetryFileAudit:
    path = reference.path
    if not path.is_file():
        return TelemetryFileAudit(
            modality=reference.modality,
            path=path,
            exists=False,
            non_empty=False,
            byte_size=0,
            sha256=None,
        )

    digest = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_SIZE):
            byte_size += len(chunk)
            digest.update(chunk)
    return TelemetryFileAudit(
        modality=reference.modality,
        path=path,
        exists=True,
        non_empty=byte_size > 0,
        byte_size=byte_size,
        sha256=digest.hexdigest(),
    )


def _fingerprint(payload: object) -> str:
    return sha256_hex(canonical_json_bytes(payload))
