"""SRE organization is not a second route around the existing claim checks."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AfterValidator, Field, ValidationError, model_validator
from typing_extensions import Self

from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object, encode_json
from nurse_rcl.v2.live.ops_report_checks import check_ops_report
from nurse_rcl.v2.live.ops_report_schema import (
    REPORT_LIMITS,
    OpsClaimCheck,
    OpsEmpiricalLayeredClaim,
    OpsEmpiricalReportResponse,
)
from nurse_rcl.v2.live.ops_sre_report_context import OpsSreReportContext
from nurse_rcl.v2.report_composition_proposals import _unique_strings
from nurse_rcl.v2.report_language_rules import SRE_LANGUAGE_POLICY

ClaimIds = Annotated[tuple[str, ...], AfterValidator(_unique_strings)]


class SreWorkflowStep(StrictFrozenModel):
    step_ref: str = Field(pattern=r"^s[1-8]$")
    observation_claim_ids: ClaimIds = Field(description=(
        "Claims grounded at this completed step: all fact_refs must belong to its "
        "observed_fact_refs. Empty is valid for navigation or no supported finding."))
    synthesis_claim_ids: ClaimIds = Field(description=(
        "Explicit report-time synthesis, possibly using later facts; never assert that "
        "these facts were known earlier. Disjoint from observation_claim_ids here."))


class OpsSreReportResponse(StrictFrozenModel):
    schema_version: Literal["nurse.v2.sre-report-response/1"]
    core_result_digest: SHA256
    claims: tuple[OpsEmpiricalLayeredClaim, ...] = Field(description=(
        "Unique claims: every claim_id must occur in at least one organization list, "
        "including advice. The union of summary/finding/unresolved and workflow claim "
        "IDs must equal this set; do not emit orphan claims or invent referenced IDs."))
    summary_claim_ids: ClaimIds
    finding_claim_ids: ClaimIds
    workflow_steps: tuple[SreWorkflowStep, ...] = Field(max_length=8)
    unresolved_claim_ids: ClaimIds = Field(description=(
        "Claims to show as unresolved issues or proposed next checks; include "
        "conditional_next_steps advice here when not organized elsewhere. "
        "This list does not replace fact_refs or recommendation.precondition_refs."))

    @model_validator(mode="after")
    def reference_closure(self) -> Self:
        ids = [c.claim.claim_id for c in self.claims]
        steps = [s.step_ref for s in self.workflow_steps]
        used = set(self.summary_claim_ids + self.finding_claim_ids + self.unresolved_claim_ids)
        for row in self.workflow_steps:
            if set(row.observation_claim_ids) & set(row.synthesis_claim_ids):
                raise ValueError("claim cannot be both observation and later synthesis")
            used.update(row.observation_claim_ids + row.synthesis_claim_ids)
        if len(set(ids)) != len(ids) or len(set(steps)) != len(steps) or used != set(ids):
            raise ValueError("SRE organization reference closure differs")
        return self


def parse_sre_report(raw: bytes) -> OpsSreReportResponse:
    value = decode_json_object(raw, REPORT_LIMITS)
    try:
        return OpsSreReportResponse.model_validate_json(encode_json(value, REPORT_LIMITS))
    except ValidationError:
        raise ExchangeBoundaryError("sre_report_schema") from None


def check_sre_report(
    response: OpsSreReportResponse, *, context: OpsSreReportContext,
    provided_fact_refs: tuple[str, ...], provided_step_refs: tuple[str, ...],
) -> tuple[OpsClaimCheck, ...]:
    steps = {s["step_ref"]: s for s in context.steps}
    if len(set(provided_step_refs)) != len(provided_step_refs) or (
        not set(provided_step_refs) <= steps.keys()
        or any(s.step_ref not in provided_step_refs for s in response.workflow_steps)
    ):
        raise ValueError("SRE provided step references differ")
    # Explicit in-process adapter ONLY. Old wire parsers still reject SRE responses.
    checked = check_ops_report(OpsEmpiricalReportResponse(
        schema_version="nurse.v2.ops-layered-report-response/2",
        core_result_digest=response.core_result_digest, claims=response.claims),
        context=context.evidence, provided_fact_refs=provided_fact_refs,
        language_policy=SRE_LANGUAGE_POLICY)
    return check_sre_organization(response, checked=checked, steps=steps, profile=context.profile)


def check_sre_organization(
    response: OpsSreReportResponse, *, checked: tuple[OpsClaimCheck, ...],
    steps: dict[str, Any], profile: str,
) -> tuple[OpsClaimCheck, ...]:
    """Same step and frozen-ranking rules for old flat and new native contexts."""
    extra: dict[str, set[str]] = {}
    if profile in {"hermes-sre-tools/3", "hermes-sre-tools/4", "hermes-sre-tools/5"}:
        from nurse_rcl.v2.live.ops_sre_frozen_language import frozen_ranking_fields

        for proposed in response.claims:
            paths = frozen_ranking_fields(proposed.claim)
            if paths:
                extra[proposed.claim.claim_id] = {"frozen_ranking_mutation", *(
                    "frozen_ranking_field:" + path for path in paths)}
    by_id = {c.claim.claim_id: c.claim for c in response.claims}
    for organized in response.workflow_steps:
        step = steps[organized.step_ref]
        delivered = set(step["observed_fact_refs"])
        for key in organized.observation_claim_ids:
            if step["executed"]["status"] != "completed" or (
                not set(by_id[key].fact_refs) <= delivered
            ):
                extra.setdefault(key, set()).add("step_fact_not_delivered")
    return tuple(row.model_copy(update=dict(
        reason_codes=tuple(sorted(set(row.reason_codes) | extra[row.proposal.claim.claim_id])),
        surface_status="blocked")) if row.proposal.claim.claim_id in extra else row
        for row in checked)
