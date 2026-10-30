"""Strict report proposals; parsing grants neither fact nor ranking authority."""

from __future__ import annotations

from typing import Literal, overload

from pydantic import Field, ValidationError, model_validator
from typing_extensions import Self

from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.exchange_json import (
    ExchangeBoundaryError,
    ExchangeLimits,
    decode_json_object,
    encode_json,
)
from nurse_rcl.v2.live.ops_report_proposals import OpsCompositionClaim
from nurse_rcl.v2.report_composition_proposals import ReportCompositionClaim

REPORT_LIMITS = ExchangeLimits(4_194_304, 64, 100_000)
RESPONSE_LIMITS = ExchangeLimits(65_536, 32, 20_000)
ReportSection = Literal[
    "formal_grounds",
    "supplementary_findings",
    "limitations",
    "conditional_next_steps",
]


class OpsLayeredClaim(StrictFrozenModel):
    section: ReportSection
    claim: ReportCompositionClaim


class OpsLayeredReportResponse(StrictFrozenModel):
    schema_version: Literal["nurse.v2.ops-layered-report-response/1"]
    core_result_digest: SHA256
    claims: tuple[OpsLayeredClaim, ...] = Field(max_length=24)

    @model_validator(mode="after")
    def unique_claims(self) -> Self:
        ids = [item.claim.claim_id for item in self.claims]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate claim IDs")
        if sum(c.claim.recommendation is not None for c in self.claims) > 6:
            raise ValueError("recommendation limit")
        return self


class OpsEmpiricalLayeredClaim(OpsLayeredClaim):
    claim: OpsCompositionClaim


class OpsEmpiricalReportResponse(StrictFrozenModel):
    schema_version: Literal["nurse.v2.ops-layered-report-response/2"]
    core_result_digest: SHA256
    claims: tuple[OpsEmpiricalLayeredClaim, ...]

    @model_validator(mode="after")
    def unique_claims(self) -> Self:
        ids = [item.claim.claim_id for item in self.claims]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate claim IDs")
        return self


class OpsClaimCheck(StrictFrozenModel):
    proposal: OpsLayeredClaim | OpsEmpiricalLayeredClaim
    reason_codes: tuple[str, ...]
    surface_status: Literal["model_unverified", "blocked"]
    semantic_status: Literal["not_established"] = "not_established"


@overload
def parse_ops_report(raw: bytes, *, empirical: Literal[False] = False) -> OpsLayeredReportResponse:
    ...


@overload
def parse_ops_report(raw: bytes, *, empirical: Literal[True]) -> OpsEmpiricalReportResponse:
    ...


@overload
def parse_ops_report(
    raw: bytes, *, empirical: bool,
) -> OpsLayeredReportResponse | OpsEmpiricalReportResponse:
    ...


def parse_ops_report(
    raw: bytes, *, empirical: bool = False,
) -> OpsLayeredReportResponse | OpsEmpiricalReportResponse:
    limits = REPORT_LIMITS if empirical else RESPONSE_LIMITS
    value = decode_json_object(raw, limits)
    try:
        model = OpsEmpiricalReportResponse if empirical else OpsLayeredReportResponse
        return model.model_validate_json(encode_json(value, limits))
    except ValidationError:
        raise ExchangeBoundaryError("ops_report_schema") from None
