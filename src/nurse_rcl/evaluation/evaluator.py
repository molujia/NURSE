"""Isolated multi-root evaluator that reads labels only after seal verification."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.methods.paper_v1.alarm import canonicalize_service_name
from nurse_rcl.models import EvaluationMetrics, PredictionRecord
from nurse_rcl.schemas import DatasetName, EvaluationCase, EvaluationManifest, StrictFrozenModel
from nurse_rcl.sealing import verify_prediction_seal

EVALUATION_SCHEMA_VERSION: Literal[1] = 1
PredictionStatus = Literal["success", "degraded", "failed"]


class CaseEvaluation(StrictFrozenModel):
    """Evaluation-only trace for one included benchmark case."""

    schema_version: Literal[1] = EVALUATION_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    prediction_status: PredictionStatus
    true_services: tuple[str, ...] = Field(min_length=1)
    candidate_services: tuple[str, ...]
    predicted_services: tuple[str, ...]
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    candidate_covered: bool
    failure: bool
    audit_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_consistent_case_metrics(self) -> Self:
        if self.hit_at_1 and not self.hit_at_3:
            raise ValueError("case Top-K hits must be monotonic")
        if self.hit_at_3 and not self.hit_at_5:
            raise ValueError("case Top-K hits must be monotonic")
        if self.failure != (self.prediction_status == "failed"):
            raise ValueError("case failure flag must match prediction status")
        if self.failure and (
            self.candidate_services
            or self.predicted_services
            or self.hit_at_1
            or self.hit_at_3
            or self.hit_at_5
            or self.candidate_covered
            or self.reciprocal_rank != 0.0
        ):
            raise ValueError("failed cases cannot contain successful ranking metrics")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"audit_hash"})
        return self.audit_hash == sha256_hex(canonical_json_bytes(unsigned))


class EvaluationResult(StrictFrozenModel):
    """Content-addressed aggregate over one sealed prediction population."""

    schema_version: Literal[1] = EVALUATION_SCHEMA_VERSION
    dataset: DatasetName
    prediction_tree_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics: EvaluationMetrics
    cases: tuple[CaseEvaluation, ...] = Field(min_length=1)
    result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if any(not item.verify() for item in self.cases):
            return False
        if self.cases != tuple(sorted(self.cases, key=lambda item: item.case_id)):
            return False
        case_ids = tuple(item.case_id for item in self.cases)
        if len(case_ids) != len(set(case_ids)):
            return False
        if self.metrics != _metrics_from_cases(self.cases):
            return False
        unsigned = self.model_dump(mode="json", exclude={"result_hash"})
        return self.result_hash == sha256_hex(canonical_json_bytes(unsigned))


def evaluate_predictions(
    *,
    prediction_root: Path,
    seal_path: Path,
    evaluation_manifest: EvaluationManifest,
) -> EvaluationResult:
    """Verify sealed terminal predictions, then read isolated labels and score them."""

    seal = verify_prediction_seal(prediction_root, seal_path)
    predictions = _load_terminal_predictions(prediction_root)
    included = tuple(case for case in evaluation_manifest.cases if not case.excluded)
    if not included:
        raise ValueError("evaluation manifest has no included cases")
    included_ids = tuple(case.case_id for case in included)
    if len(included_ids) != len(set(included_ids)):
        raise ValueError("evaluation manifest contains duplicate included case IDs")
    if set(predictions) != set(included_ids):
        raise ValueError("terminal case coverage does not match included evaluation cases")

    cases = tuple(
        sorted(
            (
                _evaluate_case(case=case, prediction=predictions[case.case_id])
                for case in included
            ),
            key=lambda item: item.case_id,
        )
    )
    metrics = _metrics_from_cases(cases)
    unsigned = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "dataset": evaluation_manifest.dataset,
        "prediction_tree_hash": seal.tree_hash,
        "metrics": metrics.model_dump(mode="json"),
        "cases": [item.model_dump(mode="json") for item in cases],
    }
    result = EvaluationResult(
        dataset=evaluation_manifest.dataset,
        prediction_tree_hash=seal.tree_hash,
        metrics=metrics,
        cases=cases,
        result_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("evaluation result verification failed")
    return result


def _load_terminal_predictions(prediction_root: Path) -> dict[str, PredictionRecord]:
    root = prediction_root.resolve(strict=True)
    files = tuple(
        sorted(
            (path for path in root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    predictions: dict[str, PredictionRecord] = {}
    for path in files:
        if path.suffix.casefold() != ".json":
            raise ValueError("prediction tree contains a non-JSON terminal artifact")
        try:
            record = PredictionRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(f"terminal prediction is unreadable: {path.name}") from error
        if record.case_id in predictions:
            raise ValueError(f"duplicate terminal record for case: {record.case_id}")
        _validate_terminal_record(record)
        predictions[record.case_id] = record
    return predictions


def _validate_terminal_record(record: PredictionRecord) -> None:
    if record.status == "failed":
        if record.top_k:
            raise ValueError("failed terminal record cannot contain Top-K candidates")
        return
    ranks = tuple(item.rank for item in record.top_k)
    if ranks != tuple(range(1, len(record.top_k) + 1)):
        raise ValueError("terminal Top-K ranks must be contiguous from one")
    services = tuple(canonicalize_service_name(item.service) for item in record.top_k)
    if len(services) != len(set(services)):
        raise ValueError("terminal Top-K canonical services must be unique")
    candidates = tuple(
        canonicalize_service_name(service) for service in record.candidate_services
    )
    if len(candidates) != len(set(candidates)):
        raise ValueError("terminal candidate canonical services must be unique")
    if candidates[: len(services)] != services:
        raise ValueError("terminal Top-K must prefix the full canonical candidates")


def _evaluate_case(
    *,
    case: EvaluationCase,
    prediction: PredictionRecord,
) -> CaseEvaluation:
    true_services = _read_true_services(case)
    candidate_services = tuple(
        canonicalize_service_name(service) for service in prediction.candidate_services
    )
    predicted_services = tuple(
        canonicalize_service_name(item.service) for item in prediction.top_k
    )
    true_set = set(true_services)
    hit_at_1 = bool(true_set & set(predicted_services[:1]))
    hit_at_3 = bool(true_set & set(predicted_services[:3]))
    hit_at_5 = bool(true_set & set(predicted_services[:5]))
    first_rank = next(
        (
            index
            for index, service in enumerate(predicted_services, start=1)
            if service in true_set
        ),
        None,
    )
    reciprocal_rank = 0.0 if first_rank is None else 1.0 / first_rank
    unsigned = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "case_id": case.case_id,
        "prediction_status": prediction.status,
        "true_services": list(true_services),
        "candidate_services": list(candidate_services),
        "predicted_services": list(predicted_services),
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hit_at_5": hit_at_5,
        "reciprocal_rank": reciprocal_rank,
        "candidate_covered": bool(true_set & set(candidate_services)),
        "failure": prediction.status == "failed",
    }
    result = CaseEvaluation(
        case_id=case.case_id,
        prediction_status=prediction.status,
        true_services=true_services,
        candidate_services=candidate_services,
        predicted_services=predicted_services,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        hit_at_5=hit_at_5,
        reciprocal_rank=reciprocal_rank,
        candidate_covered=bool(true_set & set(candidate_services)),
        failure=prediction.status == "failed",
        audit_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("case evaluation verification failed")
    return result


def _read_true_services(case: EvaluationCase) -> tuple[str, ...]:
    if case.label_path is None or case.label_field is None:
        raise ValueError(f"included case has no label reference: {case.case_id}")
    try:
        document: object = json.loads(case.label_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"label document is unreadable: {case.case_id}") from error
    current = document
    for segment in case.label_field.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            raise ValueError(f"label field is missing: {case.case_id}")
        current = current[segment]
    if isinstance(current, str):
        raw_services = (current,)
    elif isinstance(current, list) and all(isinstance(item, str) for item in current):
        raw_services = tuple(current)
    else:
        raise ValueError(f"label field must contain service strings: {case.case_id}")
    if not raw_services:
        raise ValueError(f"label field cannot be empty: {case.case_id}")
    services = tuple(sorted({canonicalize_service_name(item) for item in raw_services}))
    if not services:
        raise ValueError(f"label field cannot be empty: {case.case_id}")
    return services


def _metrics_from_cases(cases: tuple[CaseEvaluation, ...]) -> EvaluationMetrics:
    total = len(cases)
    if total == 0:
        raise ValueError("evaluation requires at least one case")
    hits_at_1 = sum(item.hit_at_1 for item in cases)
    hits_at_3 = sum(item.hit_at_3 for item in cases)
    hits_at_5 = sum(item.hit_at_5 for item in cases)
    metrics = EvaluationMetrics(
        total=total,
        hits_at_1=hits_at_1,
        hits_at_3=hits_at_3,
        hits_at_5=hits_at_5,
        acc_at_1=hits_at_1 / total,
        acc_at_3=hits_at_3 / total,
        acc_at_5=hits_at_5 / total,
        mrr=math.fsum(item.reciprocal_rank for item in cases) / total,
        candidate_coverage=sum(item.candidate_covered for item in cases) / total,
        failure_rate=sum(item.failure for item in cases) / total,
    )
    if not (metrics.acc_at_1 <= metrics.acc_at_3 <= metrics.acc_at_5):
        raise AssertionError("aggregate Top-K metrics are not monotonic")
    return metrics
