"""Model-authored report compositions, never verified evidence or action authority."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)
from typing_extensions import Self

from nurse_rcl.v2.exchange_json import (
    ExchangeBoundaryError,
    ExchangeLimits,
    decode_json_object,
    encode_json,
)

REPORT_COMPOSITION_RESPONSE_SCHEMA_VERSION: Literal["nurse.v2.report-composition-response/1"] = (
    "nurse.v2.report-composition-response/1"
)


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("blank_text")
    return value


def _unique_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    if len(set(values)) != len(values):
        raise ValueError("duplicate_value")
    return values


def _exact_scalar(value: object) -> object:
    if type(value) not in (str, int, float, bool, type(None)):
        raise ValueError("invalid_scalar")
    if type(value) is int and not -(10**30) < value < 10**30:
        raise ValueError("integer_limit")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("nonfinite_scalar")
    return value


_FactRef: TypeAlias = Annotated[str, Field(pattern=r"^f[1-9][0-9]{0,5}$")]
_FactRefs: TypeAlias = Annotated[
    tuple[_FactRef, ...], Field(min_length=1, max_length=8), AfterValidator(_unique_strings)
]
_ShortText: TypeAlias = Annotated[
    str, Field(min_length=1, max_length=256), AfterValidator(_nonblank)
]
_AssertionValue: TypeAlias = Annotated[
    Annotated[str, Field(max_length=256)]
    | Annotated[int, Field(gt=-(10**30), lt=10**30)]
    | Annotated[float, Field(allow_inf_nan=False)]
    | bool
    | None,
    BeforeValidator(_exact_scalar),
]


class _ReportProposalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ReportProposalInterval(_ReportProposalModel):
    start_unix_ns: int = Field(ge=0)
    end_unix_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.start_unix_ns >= self.end_unix_ns:
            raise ValueError("invalid_interval")
        return self


class ReportValueAssertion(_ReportProposalModel):
    fact_ref: _FactRef
    field: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    value: _AssertionValue


class ReportRecommendationOutcome(_ReportProposalModel):
    outcome: Literal["false", "true", "unknown", "improves", "no_change", "worsens_or_unknown"]
    interpretation: _ShortText


class ReportRecommendation(_ReportProposalModel):
    """Unexecuted advice; field validation confers no permission or verification."""

    action_class: Literal[
        "read_only_diagnostic_check",
        "reversible_workload_recovery",
        "reversible_resource_adjustment",
        "reversible_traffic_control",
        "reversible_dependency_failover",
        "reversible_fault_containment",
    ]
    precondition_refs: _FactRefs
    possible_outcomes: tuple[ReportRecommendationOutcome, ...] = Field(min_length=3, max_length=3)
    safety_limitations: tuple[_ShortText, ...] = Field(min_length=1, max_length=4)
    status: Literal["proposed"]

    @model_validator(mode="after")
    def unique_outcomes(self) -> Self:
        _unique_strings(tuple(item.outcome for item in self.possible_outcomes))
        return self


class ReportCompositionClaim(_ReportProposalModel):
    """A retained claim proposal; compatibility and source equality need later guards."""

    claim_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")]
    statement_type: Literal["observation", "inference", "recommendation"]
    operation: Literal[
        "describe_observation",
        "explain_priority",
        "compare_candidates",
        "synthesize_evidence",
        "explain_uncertainty",
        "propose_verification",
        "conditional_mitigation",
    ]
    fact_refs: _FactRefs
    target_service_keys: Annotated[
        tuple[_ShortText, ...], Field(max_length=2), AfterValidator(_unique_strings)
    ]
    interval: ReportProposalInterval | None
    requested_claim_strength: Literal[
        "frozen_observation",
        "ranking_inference",
        "conditional_recommendation",
        "confirmed_cause",
        "guaranteed_outcome",
    ]
    assertions: tuple[ReportValueAssertion, ...] = Field(max_length=8)
    recommendation: ReportRecommendation | None
    surface_text: Annotated[str, Field(min_length=1, max_length=512), AfterValidator(_nonblank)]

    @model_validator(mode="after")
    def unique_assertions(self) -> Self:
        keys = {(assertion.fact_ref, assertion.field) for assertion in self.assertions}
        if len(keys) != len(self.assertions):
            raise ValueError("duplicate_assertion")
        return self


class ReportCompositionResponse(_ReportProposalModel):
    schema_version: Literal["nurse.v2.report-composition-response/1"]
    claims: tuple[ReportCompositionClaim, ...] = Field(max_length=24)

    @model_validator(mode="after")
    def unique_claim_ids_and_bounded_recommendations(self) -> Self:
        ids = tuple(claim.claim_id for claim in self.claims)
        _unique_strings(ids)
        if sum(claim.recommendation is not None for claim in self.claims) > 6:
            raise ValueError("recommendation_limit")
        return self


def parse_report_composition_response(raw: bytes) -> ReportCompositionResponse:
    """Parse exact UTF-8 JSON bytes under fixed limits without repairing proposals."""
    limits = ExchangeLimits()
    value = decode_json_object(raw, limits)
    try:
        return ReportCompositionResponse.model_validate_json(
            encode_json(value, limits), strict=True
        )
    except ValidationError:
        raise ExchangeBoundaryError("report_composition_schema") from None


def report_composition_output_schema() -> dict[str, Any]:
    """Return a fresh wire schema; validation does not ground or authorize its content."""
    return ReportCompositionResponse.model_json_schema()
