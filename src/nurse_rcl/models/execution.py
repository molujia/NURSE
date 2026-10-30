"""Prediction, report, progress, failure, and evaluation contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.models.ranking import RankedCandidate
from nurse_rcl.schemas import StrictFrozenModel


class PredictionRecord(StrictFrozenModel):
    case_id: str = Field(min_length=1)
    status: Literal["success", "degraded", "failed"]
    top_k: tuple[RankedCandidate, ...]
    candidate_services: tuple[str, ...]
    verified_context_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    failure_code: str | None = None

    @model_validator(mode="after")
    def require_status_payload(self) -> Self:
        if self.status in {"success", "degraded"}:
            if (
                not self.top_k
                or not self.candidate_services
                or self.verified_context_hash is None
            ):
                raise ValueError(
                    "successful predictions require candidates, Top-K, and verified context"
                )
            top_k_services = tuple(item.service for item in self.top_k)
            if self.candidate_services[: len(top_k_services)] != top_k_services:
                raise ValueError("prediction Top-K must prefix the full candidate services")
            if len(self.candidate_services) != len(set(self.candidate_services)):
                raise ValueError("prediction candidate services must be unique")
        elif not self.failure_code or self.top_k or self.candidate_services:
            raise ValueError(
                "failed predictions require a failure code and no candidate ranking"
            )
        return self


class ReportRecord(StrictFrozenModel):
    case_id: str = Field(min_length=1)
    verified_context_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    top_k_services: tuple[str, ...] = Field(min_length=1)
    narrative: str = Field(min_length=1)
    degraded: bool


class FailureRecord(StrictFrozenModel):
    case_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    error_code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ProgressSnapshot(StrictFrozenModel):
    run_id: str = Field(min_length=1)
    total: int = Field(ge=0)
    completed: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    degraded: int = Field(ge=0)
    failed: int = Field(ge=0)
    current_group: str | None
    heartbeat_unix: int = Field(ge=0)
    stage_counts: dict[str, int]
    cache_hits: int = Field(ge=0)
    llm_attempts: int = Field(ge=0)
    llm_input_tokens: int = Field(ge=0)
    llm_output_tokens: int = Field(ge=0)
    llm_latency_seconds: float = Field(ge=0.0)
    llm_cost_usd: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.completed != self.succeeded + self.degraded + self.failed:
            raise ValueError("completed count must equal all terminal outcomes")
        if self.completed > self.total:
            raise ValueError("completed count cannot exceed total")
        if any(value < 0 for value in self.stage_counts.values()):
            raise ValueError("stage counts cannot be negative")
        return self


class EvaluationMetrics(StrictFrozenModel):
    total: int = Field(gt=0)
    hits_at_1: int = Field(ge=0)
    hits_at_3: int = Field(ge=0)
    hits_at_5: int = Field(ge=0)
    acc_at_1: float = Field(ge=0.0, le=1.0)
    acc_at_3: float = Field(ge=0.0, le=1.0)
    acc_at_5: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    candidate_coverage: float = Field(ge=0.0, le=1.0)
    failure_rate: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_monotonic_accuracy(self) -> Self:
        if not (self.hits_at_1 <= self.hits_at_3 <= self.hits_at_5 <= self.total):
            raise ValueError("Top-K hit counts must be monotonic")
        if not (self.acc_at_1 <= self.acc_at_3 <= self.acc_at_5):
            raise ValueError("Top-K accuracy must be monotonic")
        return self
