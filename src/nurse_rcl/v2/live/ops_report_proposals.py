"""OPS /2 proposal fields: retain structure, remove legacy brevity quotas.

These are subclasses so the existing pure grounding checks accept their values.
They do not change the shared /1 schemas or confer factual authority.
"""

from typing import Annotated, TypeAlias

from pydantic import AfterValidator, BeforeValidator, Field

from nurse_rcl.v2.report_composition_proposals import (
    ReportCompositionClaim,
    ReportRecommendation,
    ReportRecommendationOutcome,
    ReportValueAssertion,
    _exact_scalar,
    _nonblank,
    _unique_strings,
)
from nurse_rcl.v2.report_presentation_schemas import ReportLanguageFieldCheck

Text: TypeAlias = Annotated[str, Field(min_length=1), AfterValidator(_nonblank)]
FactRef: TypeAlias = Annotated[str, Field(pattern=r"^f[1-9][0-9]{0,5}$")]
FactRefs: TypeAlias = Annotated[
    tuple[FactRef, ...], Field(min_length=1), AfterValidator(_unique_strings)
]
Scalar: TypeAlias = Annotated[
    str | Annotated[int, Field(gt=-(10**30), lt=10**30)]
    | Annotated[float, Field(allow_inf_nan=False)] | bool | None,
    BeforeValidator(_exact_scalar),
]


class OpsValueAssertion(ReportValueAssertion):
    value: Scalar


class OpsRecommendationOutcome(ReportRecommendationOutcome):
    interpretation: Text


class OpsRecommendation(ReportRecommendation):
    precondition_refs: FactRefs
    possible_outcomes: tuple[OpsRecommendationOutcome, ...] = Field(min_length=3, max_length=3)
    safety_limitations: tuple[Text, ...] = Field(min_length=1)


class OpsCompositionClaim(ReportCompositionClaim):
    fact_refs: FactRefs
    target_service_keys: Annotated[tuple[Text, ...], AfterValidator(_unique_strings)]
    assertions: tuple[OpsValueAssertion, ...]
    recommendation: OpsRecommendation | None
    surface_text: Text


class OpsReportLanguageFieldCheck(ReportLanguageFieldCheck):
    raw_text: str = Field(min_length=1)
