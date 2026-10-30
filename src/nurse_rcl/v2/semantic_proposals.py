"""Model-authored proposals, never authoritative hypotheses or measurements."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from nurse_rcl.v2.diagnostic_schemas import DiagnosticRole, PredicateRequirement

SEMANTIC_PROPOSAL_RESPONSE_SCHEMA_VERSION: Literal["nurse.v2.semantic-proposal-response/1"] = (
    "nurse.v2.semantic-proposal-response/1"
)

Reference: TypeAlias = Annotated[str, Field(min_length=1, max_length=160)]
CandidateKey: TypeAlias = Annotated[str, Field(min_length=1)]
LocalId: TypeAlias = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")]
References: TypeAlias = Annotated[tuple[Reference, ...], Field(max_length=64)]
Unknowns: TypeAlias = Annotated[
    tuple[Annotated[str, Field(min_length=1, max_length=256)], ...], Field(max_length=16)
]


class ProposalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProposalInterval(ProposalModel):
    start_unix_ns: int = Field(ge=0)
    end_unix_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.start_unix_ns >= self.end_unix_ns:
            raise ValueError("invalid_interval")
        return self


class PredicatePlan(ProposalModel):
    predicate_kind: Reference
    requirement: PredicateRequirement
    observation_ids: References
    relation_ids: References


class SemanticRoleProposal(ProposalModel):
    role: DiagnosticRole
    event_group_id: Reference
    relation_ids: References
    interval: ProposalInterval


class SemanticHypothesisProposal(ProposalModel):
    proposal_id: LocalId
    candidate_service_key: CandidateKey
    mechanism_key: Reference
    observation_ids: References
    event_group_ids: References
    relation_ids: References
    time_intervals: tuple[ProposalInterval, ...] = Field(min_length=1, max_length=8)
    predicate_plans: tuple[PredicatePlan, ...] = Field(max_length=16)
    role_proposals: tuple[SemanticRoleProposal, ...] = Field(max_length=16)
    alternative_proposal_ids: tuple[LocalId, ...] = Field(max_length=32)
    query_ids: tuple[LocalId, ...] = Field(max_length=8)
    unknown_reasons: Unknowns
    rationale: str = Field(min_length=1, max_length=1000)


class SemanticCardQuery(ProposalModel):
    kind: Literal["card"]
    query_id: LocalId
    candidate_service_key: CandidateKey


class SemanticVerificationQuery(ProposalModel):
    kind: Literal["verification"]
    query_id: LocalId
    candidate_service_key: CandidateKey
    operator: Reference
    signal_family: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")] | None
    query_interval: ProposalInterval
    observation_ids: References
    relation_ids: References
    requested_fields: tuple[Reference, ...] = Field(min_length=1, max_length=16)


SemanticQuery: TypeAlias = Annotated[
    SemanticCardQuery | SemanticVerificationQuery, Field(discriminator="kind")
]


class SemanticProposalResponse(ProposalModel):
    schema_version: Literal["nurse.v2.semantic-proposal-response/1"]
    hypotheses: tuple[SemanticHypothesisProposal, ...] = Field(max_length=32)
    queries: tuple[SemanticQuery, ...] = Field(max_length=8)
    unknown_reasons: Unknowns


class ResponseEnvelope(ProposalModel):
    """Internal shape gate; preserve invalid items for partial-response inspection."""

    schema_version: Literal["nurse.v2.semantic-proposal-response/1"]
    hypotheses: tuple[Any, ...] = Field(max_length=32)
    queries: tuple[Any, ...] = Field(max_length=8)
    unknown_reasons: Unknowns


def semantic_output_schema() -> dict[str, Any]:
    """Return a fresh schema; no mutable global schema or provider-specific claims."""
    return SemanticProposalResponse.model_json_schema()
