"""Ranking and verified-context contracts."""

from __future__ import annotations

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.models.semantic import GuardDecision
from nurse_rcl.schemas import StrictFrozenModel


class ComponentValues(StrictFrozenModel):
    e: float
    p: float
    h: float
    r: float


class ScoreComponents(StrictFrozenModel):
    raw: ComponentValues
    normalized: ComponentValues
    total: float

    @model_validator(mode="after")
    def require_normalized_bounds(self) -> Self:
        values = self.normalized.model_dump().values()
        if any(value < 0.0 or value > 100.0 for value in values):
            raise ValueError("normalized score components must be in [0, 100]")
        return self


class RankedCandidate(StrictFrozenModel):
    rank: int = Field(ge=1)
    service: str = Field(min_length=1)
    score: float
    components: ScoreComponents
    evidence_ids: tuple[str, ...]


class EvidencePath(StrictFrozenModel):
    source_service: str = Field(min_length=1)
    target_service: str = Field(min_length=1)
    relation_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]


class SCCTrace(StrictFrozenModel):
    component_id: str = Field(min_length=1)
    members: tuple[str, ...] = Field(min_length=1)
    removed_relation_id: str | None = None


class HistoryNeighbor(StrictFrozenModel):
    case_id: str = Field(min_length=1)
    similarity: float = Field(ge=0.0, le=1.0)
    matched_services: tuple[str, ...]


class VerifiedRCLContext(StrictFrozenModel):
    case_id: str = Field(min_length=1)
    top_k: tuple[RankedCandidate, ...] = Field(min_length=1)
    evidence_paths: tuple[EvidencePath, ...]
    guard_decisions: tuple[GuardDecision, ...]
    scc_trace: tuple[SCCTrace, ...]
    history_trace: tuple[HistoryNeighbor, ...]
    warnings: tuple[str, ...]

    @model_validator(mode="after")
    def require_deterministic_ranks(self) -> Self:
        ranks = [candidate.rank for candidate in self.top_k]
        if ranks != list(range(1, len(self.top_k) + 1)):
            raise ValueError("Top-K ranks must be contiguous from one")
        services = [candidate.service for candidate in self.top_k]
        if len(services) != len(set(services)):
            raise ValueError("Top-K services must be unique")
        return self
