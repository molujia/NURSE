"""D16 explicit step bindings and comparable evidence, never new RCL authority."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, cast

from pydantic import AfterValidator, Field, ValidationError, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object, encode_json
from nurse_rcl.v2.live.ops_report_proposals import OpsRecommendationOutcome, OpsValueAssertion, Text
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_brief_claims import (
    SECTIONS,
    BriefAssertion,
    BriefRecommendation,
    EvidenceRef,
    EvidenceRefs,
    _check_brief_items,
)
from nurse_rcl.v2.live.ops_sre_d22_numbers import FLOAT_ONLY, DecimalValuePolicy
from nurse_rcl.v2.live.ops_sre_fact_store import NativeFactStore
from nurse_rcl.v2.live.repair_module_timing import _module_call
from nurse_rcl.v2.report_composition_proposals import (
    ReportCompositionClaim,
    ReportProposalInterval,
    _unique_strings,
)
from nurse_rcl.v2.report_evidence_schemas import ReportEvidenceFact
from nurse_rcl.v2.report_mechanical_checks import _recommendation_common, _references, _values

StepRef = Annotated[str, Field(pattern=r"^s[1-8]$")]


class ComparableVerification(StrictFrozenModel):
    action_class: Literal["read_only_diagnostic_check"]
    status: Literal["proposed"]
    precondition_refs: EvidenceRefs
    verification_question: Text
    possible_outcomes: tuple[OpsRecommendationOutcome, ...]
    outcome_subject_ref: EvidenceRef | None
    safety_limitations: tuple[Text, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def subject_shape(self) -> Self:
        if not self.possible_outcomes:
            if self.outcome_subject_ref is not None:
                raise ValueError("branchless verification cannot name an outcome subject")
        elif (
            len(self.possible_outcomes) != 3
            or self.outcome_subject_ref is None
            or {v.outcome for v in self.possible_outcomes} != {"true", "false", "unknown"}
        ):
            raise ValueError("verification outcomes require one subject and three distinct values")
        return self


class ComparableClaim(StrictFrozenModel):
    claim_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    statement_type: Literal["observation", "inference", "recommendation"]
    operation: Literal[
        "describe_observation",
        "explain_priority",
        "compare_candidates",
        "compare_mechanisms",
        "synthesize_evidence",
        "explain_uncertainty",
        "propose_verification",
        "conditional_mitigation",
    ]
    evidence_refs: EvidenceRefs
    target_service_keys: Annotated[tuple[Text, ...], AfterValidator(_unique_strings)]
    interval: ReportProposalInterval | None
    requested_claim_strength: Literal[
        "frozen_observation",
        "ranking_inference",
        "conditional_recommendation",
        "confirmed_cause",
        "guaranteed_outcome",
    ]
    assertions: tuple[BriefAssertion, ...]
    recommendation: ComparableVerification | BriefRecommendation | None
    surface_text: Text

    @property
    def fact_refs(self) -> tuple[str, ...]:
        return tuple(ref for ref in self.evidence_refs if ref.startswith("f"))

    @model_validator(mode="after")
    def structure(self) -> Self:
        keys = [(a.evidence_ref, a.field) for a in self.assertions]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate comparable assertion")
        if self.recommendation is not None:
            expected = (
                ComparableVerification
                if self.operation == "propose_verification"
                else BriefRecommendation
            )
            if not isinstance(self.recommendation, expected):
                raise ValueError("recommendation protocol differs from operation")
        return self


class ComparableStepFinding(ComparableClaim):
    step_ref: StepRef


class ComparableReportResponse(StrictFrozenModel):
    schema_version: Literal["nurse.v2.sre-report-response/4"]
    core_result_digest: SHA256
    executive_summary: tuple[ComparableClaim, ...]
    formal_interpretations: tuple[ComparableClaim, ...]
    step_findings: dict[StepRef, tuple[ComparableClaim, ...]] = Field(max_length=8)
    final_synthesis: tuple[ComparableClaim, ...]
    limitations: tuple[ComparableClaim, ...]
    next_checks: tuple[ComparableClaim, ...]

    @model_validator(mode="after")
    def closure(self) -> Self:
        ids = [claim.claim_id for _, claim in comparable_items(self)]
        if len(set(ids)) != len(ids):
            raise ValueError("comparable report reference closure differs")
        return self


class ComparableClaimCheck(StrictFrozenModel):
    section: Literal[
        "executive_summary",
        "formal_interpretations",
        "step_findings",
        "final_synthesis",
        "limitations",
        "next_checks",
    ]
    proposal: ComparableClaim | ComparableStepFinding
    reason_codes: tuple[str, ...]
    surface_status: Literal["model_unverified", "blocked"]
    semantic_status: Literal["not_established"] = "not_established"


def comparable_items(response: ComparableReportResponse) -> list[tuple[str, ComparableClaim]]:
    items: list[tuple[str, ComparableClaim]] = []
    for section in ("executive_summary", *SECTIONS):
        if section == "step_findings":
            for step_ref, claims in response.step_findings.items():
                for claim in claims:
                    bound = ComparableStepFinding.model_validate_json(
                        canonical_json_bytes(
                            dict(claim.model_dump(mode="json"), step_ref=step_ref)
                        ),
                        strict=True,
                    )
                    items.append((section, bound))
        else:
            items.extend((section, claim) for claim in getattr(response, section))
    return items


def parse_comparable_report(raw: bytes) -> ComparableReportResponse:
    value = decode_json_object(raw, REPORT_LIMITS)
    try:
        return ComparableReportResponse.model_validate_json(
            encode_json(value, REPORT_LIMITS), strict=True
        )
    except ValidationError:
        raise ExchangeBoundaryError("sre_comparable_report_schema") from None


@_module_call("sre_comparable_checks")
def check_comparable_report(
    response: ComparableReportResponse,
    *,
    supply: dict[str, Any],
    fact_store: NativeFactStore,
    candidates: tuple[str, ...],
    require_supplied_decimals: bool = False,
    decimal_value_policy: DecimalValuePolicy = FLOAT_ONLY,
) -> tuple[ComparableClaimCheck, ...]:
    if supply["schema_version"] != "nurse.v2.sre-brief-supply/3":
        raise ValueError("comparable report requires its own supply")
    if not set(response.step_findings) <= {s["step_ref"] for s in supply["steps"]}:
        raise ValueError("comparable report names an unexecuted or future step")
    return cast(
        tuple[ComparableClaimCheck, ...],
        _check_brief_items(
            response,
            items=comparable_items(response),
            result_type=ComparableClaimCheck,
            supply=supply,
            fact_store=fact_store,
            candidates=candidates,
            comparable=True,
            require_supplied_decimals=require_supplied_decimals,
            decimal_value_policy=decimal_value_policy,
        ),
    )


@dataclass(frozen=True)
class _ExactComparable:
    """Structural input to pure checks, not a legacy schema or changed operation."""

    operation: str
    fact_refs: tuple[str, ...]
    target_service_keys: tuple[str, ...]
    assertions: tuple[OpsValueAssertion, ...]
    recommendation: ComparableVerification | BriefRecommendation | None


def exact_comparable_claim(claim: ComparableClaim) -> ReportCompositionClaim:
    return cast(
        ReportCompositionClaim,
        _ExactComparable(
            operation=claim.operation,
            fact_refs=claim.fact_refs,
            target_service_keys=claim.target_service_keys,
            assertions=tuple(
                OpsValueAssertion.model_validate_json(
                    canonical_json_bytes(
                        dict(fact_ref=a.evidence_ref, field=a.field, value=a.value)
                    ),
                    strict=True,
                )
                for a in claim.assertions
                if a.evidence_ref.startswith("f")
            ),
            recommendation=claim.recommendation,
        ),
    )


def _mechanism_identity(brief: dict[str, Any]) -> tuple[Any, ...]:
    fields = brief["fields"]
    return (
        fields.get("mechanism_key"),
        fields.get("mechanism_definition_sha256"),
        brief.get("window_ns"),
        brief.get("mechanism_source_kind"),
    )


def comparable_checks(
    claim: ComparableClaim,
    *,
    briefs: dict[str, Any],
    facts: Mapping[str, ReportEvidenceFact],
    b_refs: set[str],
    f_refs: set[str],
) -> set[str]:
    reasons: set[str] = set()
    if claim.operation == "compare_mechanisms":
        cited = [briefs[r] for r in claim.evidence_refs if r in b_refs]
        # Exactly one source per side; ambiguity must be explained separately.
        identities = [_mechanism_identity(b) for b in cited]
        if (
            len(claim.target_service_keys) != 2
            or len(cited) != 2
            or [b["target_service_keys"] for b in cited] != [[t] for t in claim.target_service_keys]
            or any(
                b["granularity"] not in {"mechanism_decision", "formal_mechanism_reference"}
                for b in cited
            )
            or any(v is None for identity in identities for v in identity)
            or (len(identities) == 2 and identities[0] != identities[1])
        ):
            reasons.add("mechanism_comparison_basis_mismatch")
    if claim.operation != "propose_verification" or claim.recommendation is None:
        return reasons
    advice = claim.recommendation
    assert isinstance(advice, ComparableVerification)
    exact = exact_comparable_claim(claim)
    if _references(exact, facts):
        reasons.add("recommendation_precondition_invalid")
    else:
        reasons.update(_recommendation_common(exact, facts, (), None))
    if not claim.target_service_keys:
        reasons.add("recommendation_precondition_invalid")
    ref = advice.outcome_subject_ref
    if ref is None:
        return reasons
    if ref not in claim.evidence_refs or ref not in b_refs | f_refs:
        reasons.add("outcome_subject_not_provided")
        return reasons
    interval = (
        None
        if claim.interval is None
        else [claim.interval.start_unix_ns, claim.interval.end_unix_ns]
    )
    if ref.startswith("b"):
        source = briefs[ref]
        single = source["granularity"] == "mechanism_condition" and source["fields"].get(
            "truth_value"
        ) in {"true", "false", "unknown"}
        scope_ok = tuple(source["target_service_keys"]) == claim.target_service_keys
        window_ok = interval is not None and source["window_ns"] == interval
    else:
        fact = facts[ref]
        values = _values(fact)
        single = (
            fact.kind in {"predicate", "observation"}
            and isinstance(values.get("predicate_kind"), str)
            and values.get("truth_value") in {"true", "false", "unknown"}
        )
        scope_ok = fact.target_service_keys == claim.target_service_keys
        window_ok = interval is not None and any(
            [i.start_unix_ns, i.end_unix_ns] == interval for i in fact.intervals
        )
    if not single:
        reasons.add("outcome_subject_not_single_condition_or_predicate")
    if not scope_ok:
        reasons.add("outcome_subject_scope_mismatch")
    if not window_ok:
        reasons.add("outcome_subject_interval_mismatch")
    return reasons
