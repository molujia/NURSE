"""Semantic payload, fact, guard, and guarded graph contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.models.evidence import EvidenceRecord, EvidenceRule
from nurse_rcl.schemas import StrictFrozenModel

SemanticFactType = Literal["root_support", "source_support", "symptom_support"]
RelationType = Literal["supports_root", "propagates_to", "symptom_of"]


class CandidateCard(StrictFrozenModel):
    service: str = Field(min_length=1)
    evidence_ids: tuple[str, ...]
    neighbors: tuple[str, ...]


class SemanticPayload(StrictFrozenModel):
    schema_version: Literal[1] = 1
    preliminary_graph_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidates: tuple[CandidateCard, ...] = Field(min_length=1)
    evidence: tuple[EvidenceRecord, ...]
    rules: tuple[EvidenceRule, ...] = ()
    semantic_role_policy: tuple[SemanticFactType, ...]

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        evidence_ids = {record.evidence_id for record in self.evidence}
        rule_ids = {rule.rule_id for rule in self.rules}
        if len(rule_ids) != len(self.rules):
            raise ValueError("semantic payload rule identifiers must be unique")
        for card in self.candidates:
            missing = set(card.evidence_ids) - evidence_ids
            if missing:
                raise ValueError(f"candidate card cites unknown evidence: {sorted(missing)}")
        if self.rules:
            for record in self.evidence:
                if record.verification_rule not in rule_ids:
                    raise ValueError(
                        f"evidence cites unknown payload rule: {record.verification_rule}"
                    )
        return self


class SemanticFact(StrictFrozenModel):
    fact_type: SemanticFactType
    service: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    related_service: Annotated[str, Field(min_length=1)] | None
    related_evidence_id: Annotated[str, Field(min_length=1)] | None

    @model_validator(mode="after")
    def require_relation_shape(self) -> Self:
        related = (self.related_service, self.related_evidence_id)
        if self.fact_type == "root_support":
            if any(value is not None for value in related):
                raise ValueError("root_support related fields must be null")
        elif any(value is None for value in related):
            raise ValueError(f"{self.fact_type} related fields must be non-empty strings")
        return self


class GuardDecision(StrictFrozenModel):
    fact_index: int = Field(ge=0)
    accepted: bool
    code: str = Field(min_length=1)
    fact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class GuardedRelation(StrictFrozenModel):
    relation_id: str = Field(min_length=1)
    relation_type: RelationType
    source_service: str = Field(min_length=1)
    target_service: str | None = None
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    weight: float = Field(ge=0.0, le=100.0)
    fact_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class GuardedEvidenceGraph(StrictFrozenModel):
    preliminary_graph_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    relations: tuple[GuardedRelation, ...]
    decisions: tuple[GuardDecision, ...]
