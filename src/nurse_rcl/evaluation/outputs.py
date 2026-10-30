"""Deterministic aggregate and per-case artifacts for the evaluation-only domain."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.evaluation.evaluator import EvaluationResult
from nurse_rcl.models import EvaluationMetrics
from nurse_rcl.schemas import DatasetName, StrictFrozenModel

EVALUATION_OUTPUT_SCHEMA_VERSION: Literal[1] = 1
_AGGREGATE_FIELDS = (
    "dataset",
    "total",
    "hits_at_1",
    "hits_at_3",
    "hits_at_5",
    "acc_at_1",
    "acc_at_3",
    "acc_at_5",
    "mrr",
    "candidate_coverage",
    "failure_rate",
    "prediction_tree_hash",
    "result_hash",
)


class CaseAuditReference(StrictFrozenModel):
    case_id: str = Field(min_length=1)
    path: str = Field(pattern=r"^cases/[A-Za-z0-9][A-Za-z0-9._-]*\.json$")
    audit_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvaluationAggregateArtifact(StrictFrozenModel):
    """JSON aggregate binding metrics to every per-case evaluation audit."""

    schema_version: Literal[1] = EVALUATION_OUTPUT_SCHEMA_VERSION
    dataset: DatasetName
    prediction_tree_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics: EvaluationMetrics
    result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_audits: tuple[CaseAuditReference, ...] = Field(min_length=1)
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if self.case_audits != tuple(
            sorted(self.case_audits, key=lambda item: item.case_id)
        ):
            return False
        case_ids = tuple(item.case_id for item in self.case_audits)
        if len(case_ids) != len(set(case_ids)) or self.metrics.total != len(case_ids):
            return False
        unsigned = self.model_dump(mode="json", exclude={"artifact_hash"})
        return self.artifact_hash == sha256_hex(canonical_json_bytes(unsigned))


class OutputFileDigest(StrictFrozenModel):
    path: str = Field(min_length=1)
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvaluationOutputReceipt(StrictFrozenModel):
    """Content manifest for every aggregate and per-case evaluation file."""

    schema_version: Literal[1] = EVALUATION_OUTPUT_SCHEMA_VERSION
    evaluation_result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_count: int = Field(gt=0)
    files: tuple[OutputFileDigest, ...] = Field(min_length=3)
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            return False
        case_files = tuple(path for path in paths if path.startswith("cases/"))
        if len(case_files) != self.case_count:
            return False
        if not {"aggregate.csv", "aggregate.json"}.issubset(paths):
            return False
        unsigned = self.model_dump(mode="json", exclude={"receipt_hash"})
        return self.receipt_hash == sha256_hex(canonical_json_bytes(unsigned))


def write_evaluation_outputs(
    *,
    result: EvaluationResult,
    prediction_root: Path,
    output_root: Path,
) -> EvaluationOutputReceipt:
    """Persist immutable evaluation artifacts outside the sealed prediction tree."""

    if not result.verify():
        raise ValueError("evaluation result verification failed")
    predictions = prediction_root.resolve(strict=True)
    destination = output_root.resolve(strict=False)
    if destination == predictions or destination.is_relative_to(predictions):
        raise ValueError("evaluation outputs must remain outside the sealed prediction tree")

    case_references: list[CaseAuditReference] = []
    desired: dict[str, bytes] = {}
    for case in result.cases:
        _require_safe_case_id(case.case_id)
        relative_path = f"cases/{case.case_id}.json"
        case_references.append(
            CaseAuditReference(
                case_id=case.case_id,
                path=relative_path,
                audit_hash=case.audit_hash,
            )
        )
        desired[relative_path] = canonical_json_bytes(case.model_dump(mode="json"))

    aggregate = _aggregate_artifact(
        result=result,
        case_references=tuple(case_references),
    )
    desired["aggregate.json"] = canonical_json_bytes(aggregate.model_dump(mode="json"))
    desired["aggregate.csv"] = _aggregate_csv(result)
    file_digests = tuple(
        OutputFileDigest(path=path, size=len(data), sha256=sha256_hex(data))
        for path, data in sorted(desired.items())
    )
    unsigned = {
        "schema_version": EVALUATION_OUTPUT_SCHEMA_VERSION,
        "evaluation_result_hash": result.result_hash,
        "case_count": len(result.cases),
        "files": [item.model_dump(mode="json") for item in file_digests],
    }
    receipt = EvaluationOutputReceipt(
        evaluation_result_hash=result.result_hash,
        case_count=len(result.cases),
        files=file_digests,
        receipt_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not receipt.verify():  # pragma: no cover - construction invariant
        raise AssertionError("evaluation output receipt verification failed")
    desired["output_manifest.json"] = canonical_json_bytes(
        receipt.model_dump(mode="json")
    )
    for relative_path, data in sorted(desired.items()):
        _write_immutable(destination / relative_path, data)
    return receipt


def _aggregate_artifact(
    *,
    result: EvaluationResult,
    case_references: tuple[CaseAuditReference, ...],
) -> EvaluationAggregateArtifact:
    unsigned = {
        "schema_version": EVALUATION_OUTPUT_SCHEMA_VERSION,
        "dataset": result.dataset,
        "prediction_tree_hash": result.prediction_tree_hash,
        "metrics": result.metrics.model_dump(mode="json"),
        "result_hash": result.result_hash,
        "case_audits": [item.model_dump(mode="json") for item in case_references],
    }
    artifact = EvaluationAggregateArtifact(
        dataset=result.dataset,
        prediction_tree_hash=result.prediction_tree_hash,
        metrics=result.metrics,
        result_hash=result.result_hash,
        case_audits=case_references,
        artifact_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not artifact.verify():  # pragma: no cover - construction invariant
        raise AssertionError("evaluation aggregate verification failed")
    return artifact


def _aggregate_csv(result: EvaluationResult) -> bytes:
    metrics = result.metrics
    row: dict[str, str | int | float] = {
        "dataset": result.dataset,
        "total": metrics.total,
        "hits_at_1": metrics.hits_at_1,
        "hits_at_3": metrics.hits_at_3,
        "hits_at_5": metrics.hits_at_5,
        "acc_at_1": metrics.acc_at_1,
        "acc_at_3": metrics.acc_at_3,
        "acc_at_5": metrics.acc_at_5,
        "mrr": metrics.mrr,
        "candidate_coverage": metrics.candidate_coverage,
        "failure_rate": metrics.failure_rate,
        "prediction_tree_hash": result.prediction_tree_hash,
        "result_hash": result.result_hash,
    }
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=_AGGREGATE_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _write_immutable(path: Path, data: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"immutable evaluation output differs: {path.name}")
        return
    atomic_write_bytes(path, data)


def _require_safe_case_id(case_id: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", case_id) is None:
        raise ValueError("evaluation case ID must be a safe path segment")
