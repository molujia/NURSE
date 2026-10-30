"""Fixed component normalization, aggregation, and deterministic Top-K ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.methods.paper_v1.alarm import canonicalize_service_name
from nurse_rcl.methods.paper_v1.causal_history import HistoryScoreArtifact
from nurse_rcl.methods.paper_v1.raw_scoring import RawCandidateEPR, RawEPRArtifact
from nurse_rcl.methods.paper_v1.scc_scoring import SCCScoringArtifact
from nurse_rcl.models import ComponentValues, RankedCandidate, ScoreComponents
from nurse_rcl.schemas import StrictFrozenModel

FINAL_RANKING_VERSION: Literal["final_ranking_v1"] = "final_ranking_v1"
RANKING_TOP_K: Literal[5] = 5
FIXED_COMPONENT_WEIGHTS = ComponentValues(e=1.0, p=1.0, h=1.0, r=1.0)


class CandidateComponentInput(StrictFrozenModel):
    service: str = Field(min_length=1)
    e: float
    p: float
    h: float
    r: float
    evidence_ids: tuple[str, ...]

    @field_validator("e", "p", "h", "r")
    @classmethod
    def require_finite_components(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("ranking components must be finite")
        return value


class RankedComponentArtifact(StrictFrozenModel):
    ranking_version: Literal["final_ranking_v1"] = FINAL_RANKING_VERSION
    component_weights: ComponentValues = FIXED_COMPONENT_WEIGHTS
    candidates: tuple[RankedCandidate, ...] = Field(min_length=1)
    top_k: tuple[RankedCandidate, ...] = Field(min_length=1, max_length=RANKING_TOP_K)
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_complete_ranking_and_top_k_prefix(self) -> Self:
        ranks = tuple(candidate.rank for candidate in self.candidates)
        if ranks != tuple(range(1, len(self.candidates) + 1)):
            raise ValueError("full ranking must have contiguous ranks")
        if self.top_k != self.candidates[:RANKING_TOP_K]:
            raise ValueError("released Top-K must be the fixed ranking prefix")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"artifact_hash"})
        return self.artifact_hash == sha256_hex(canonical_json_bytes(unsigned))


class FinalRankingArtifact(StrictFrozenModel):
    ranking_version: Literal["final_ranking_v1"] = FINAL_RANKING_VERSION
    raw_epr_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scc_scoring_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    history_score_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ranking: RankedComponentArtifact
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if not self.ranking.verify():
            return False
        unsigned = self.model_dump(mode="json", exclude={"artifact_hash"})
        return self.artifact_hash == sha256_hex(canonical_json_bytes(unsigned))


@dataclass(frozen=True, slots=True)
class _ScoredCandidate:
    service: str
    raw: ComponentValues
    normalized: ComponentValues
    total: float
    evidence_ids: tuple[str, ...]


def minmax_normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    """Normalize one component to [0,100], mapping a constant to all zeros."""

    if not values:
        raise ValueError("component normalization requires at least one value")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("component normalization requires finite values")
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return tuple(0.0 for _ in values)
    scale = maximum - minimum
    return tuple(
        min(100.0, max(0.0, 100.0 * (value - minimum) / scale))
        for value in values
    )


def rank_component_inputs(
    *,
    inputs: tuple[CandidateComponentInput, ...],
) -> RankedComponentArtifact:
    """Apply fixed unit weights and the E/P/H/service tie sequence."""

    if not inputs:
        raise ValueError("final ranking requires at least one candidate")
    canonical_inputs = tuple(
        item.model_copy(
            update={
                "service": canonicalize_service_name(item.service),
                "evidence_ids": tuple(sorted(set(item.evidence_ids))),
            }
        )
        for item in inputs
    )
    services = [item.service for item in canonical_inputs]
    if len(services) != len(set(services)):
        raise ValueError("final ranking candidate services must be canonically unique")

    normalized_e = minmax_normalize(tuple(item.e for item in canonical_inputs))
    normalized_p = minmax_normalize(tuple(item.p for item in canonical_inputs))
    normalized_h = minmax_normalize(tuple(item.h for item in canonical_inputs))
    normalized_r = minmax_normalize(tuple(item.r for item in canonical_inputs))
    scored: list[_ScoredCandidate] = []
    for index, item in enumerate(canonical_inputs):
        raw = ComponentValues(e=item.e, p=item.p, h=item.h, r=item.r)
        normalized = ComponentValues(
            e=normalized_e[index],
            p=normalized_p[index],
            h=normalized_h[index],
            r=normalized_r[index],
        )
        total = math.fsum(
            (
                FIXED_COMPONENT_WEIGHTS.e * normalized.e,
                FIXED_COMPONENT_WEIGHTS.p * normalized.p,
                FIXED_COMPONENT_WEIGHTS.h * normalized.h,
                FIXED_COMPONENT_WEIGHTS.r * normalized.r,
            )
        )
        scored.append(
            _ScoredCandidate(
                service=item.service,
                raw=raw,
                normalized=normalized,
                total=total,
                evidence_ids=item.evidence_ids,
            )
        )
    scored.sort(
        key=lambda item: (
            -item.total,
            -item.normalized.e,
            -item.normalized.p,
            -item.normalized.h,
            item.service,
        )
    )
    candidates = tuple(
        RankedCandidate(
            rank=index,
            service=item.service,
            score=item.total,
            components=ScoreComponents(
                raw=item.raw,
                normalized=item.normalized,
                total=item.total,
            ),
            evidence_ids=item.evidence_ids,
        )
        for index, item in enumerate(scored, start=1)
    )
    unsigned = {
        "ranking_version": FINAL_RANKING_VERSION,
        "component_weights": FIXED_COMPONENT_WEIGHTS.model_dump(mode="json"),
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        "top_k": [
            candidate.model_dump(mode="json")
            for candidate in candidates[:RANKING_TOP_K]
        ],
    }
    result = RankedComponentArtifact(
        candidates=candidates,
        top_k=candidates[:RANKING_TOP_K],
        artifact_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("ranked component artifact verification failed")
    return result


def assemble_final_ranking(
    *,
    raw: RawEPRArtifact,
    scc: SCCScoringArtifact,
    history: HistoryScoreArtifact,
) -> FinalRankingArtifact:
    """Verify upstream traces and assemble the fixed V1 released ranking."""

    if not raw.verify():
        raise ValueError("raw E/P/R artifact hash verification failed")
    if not scc.verify():
        raise ValueError("SCC scoring artifact hash verification failed")
    if not history.verify():
        raise ValueError("history score artifact hash verification failed")
    if scc.raw_epr_hash != raw.artifact_hash:
        raise ValueError("SCC scoring does not match raw E/P/R hash")

    raw_by_service = {candidate.service: candidate for candidate in raw.candidates}
    scc_by_service = {candidate.service: candidate for candidate in scc.candidates}
    history_by_service = {
        candidate.service: candidate for candidate in history.candidates
    }
    service_sets = (set(raw_by_service), set(scc_by_service), set(history_by_service))
    if len({frozenset(services) for services in service_sets}) != 1:
        raise ValueError("ranking input candidate services do not match")
    inputs = tuple(
        CandidateComponentInput(
            service=service,
            e=raw_by_service[service].e,
            p=scc_by_service[service].p,
            h=history_by_service[service].h,
            r=raw_by_service[service].r,
            evidence_ids=_candidate_evidence_ids(raw_by_service[service]),
        )
        for service in sorted(raw_by_service)
    )
    ranking = rank_component_inputs(inputs=inputs)
    unsigned = {
        "ranking_version": FINAL_RANKING_VERSION,
        "raw_epr_hash": raw.artifact_hash,
        "scc_scoring_hash": scc.artifact_hash,
        "history_score_hash": history.artifact_hash,
        "ranking": ranking.model_dump(mode="json"),
    }
    result = FinalRankingArtifact(
        raw_epr_hash=raw.artifact_hash,
        scc_scoring_hash=scc.artifact_hash,
        history_score_hash=history.artifact_hash,
        ranking=ranking,
        artifact_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("final ranking artifact verification failed")
    return result


def _candidate_evidence_ids(candidate: RawCandidateEPR) -> tuple[str, ...]:
    evidence_ids = {
        contribution.evidence_id for contribution in candidate.direct_contributions
    }
    for contribution in candidate.propagation_contributions:
        evidence_ids.add(contribution.source_evidence_id)
        evidence_ids.add(contribution.symptom_evidence_id)
    return tuple(sorted(evidence_ids))
