"""D13 sectioned proposals, with separate summary and exact-fact permissions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from pydantic import AfterValidator, Field, ValidationError, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object, encode_json
from nurse_rcl.v2.live.ops_report_checks import check_ops_report_records, model_ranking_override
from nurse_rcl.v2.live.ops_report_proposals import (
    OpsCompositionClaim,
    OpsRecommendation,
    Scalar,
    Text,
)
from nurse_rcl.v2.live.ops_report_schema import (
    REPORT_LIMITS,
    OpsEmpiricalLayeredClaim,
    OpsEmpiricalReportResponse,
)
from nurse_rcl.v2.live.ops_sre_d22_numbers import (
    FLOAT_ONLY,
    DecimalValuePolicy,
    validate_decimal_value_policy,
)
from nurse_rcl.v2.live.ops_sre_fact_store import NativeFactStore
from nurse_rcl.v2.live.ops_sre_frozen_language import frozen_ranking_fields
from nurse_rcl.v2.live.ops_sre_native_checks import _EvidenceLookup
from nurse_rcl.v2.live.ops_sre_native_report import NativeFactLookup
from nurse_rcl.v2.report_composition_proposals import (
    ReportCompositionClaim,
    ReportProposalInterval,
    _unique_strings,
)
from nurse_rcl.v2.report_language_rules import (
    SRE_LANGUAGE_POLICY,
    LanguageFactIndex,
    _check_text,
    _LanguageScope,
)
from nurse_rcl.v2.report_mechanical_checks import (
    _assertions,
    _recommendation,
    _references,
    _uncertain,
)
from nurse_rcl.v2.report_presentation_schemas import claim_free_text_fields

EvidenceRef = Annotated[str, Field(pattern=r"^[bf][1-9][0-9]{0,5}$")]
if TYPE_CHECKING:
    from nurse_rcl.v2.live.ops_sre_comparable_claims import (
        ComparableClaimCheck,
        ComparableReportResponse,
    )
EvidenceRefs = Annotated[
    tuple[EvidenceRef, ...], Field(min_length=1), AfterValidator(_unique_strings)
]


class BriefAssertion(StrictFrozenModel):
    evidence_ref: EvidenceRef
    field: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    value: Scalar


class BriefRecommendation(OpsRecommendation):
    precondition_refs: EvidenceRefs


class BriefClaim(StrictFrozenModel):
    claim_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
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
    recommendation: BriefRecommendation | None
    surface_text: Text

    @property
    def fact_refs(self) -> tuple[str, ...]:
        """Only exact f refs for legacy language diagnostics, never virtual b facts."""
        return tuple(ref for ref in self.evidence_refs if ref.startswith("f"))

    @model_validator(mode="after")
    def unique_assertions(self) -> Self:
        keys = [(a.evidence_ref, a.field) for a in self.assertions]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate brief assertion")
        return self


class BriefStepFinding(BriefClaim):
    step_ref: str = Field(pattern=r"^s[1-8]$")


SECTIONS = (
    "formal_interpretations",
    "step_findings",
    "final_synthesis",
    "limitations",
    "next_checks",
)


class BriefReportResponse(StrictFrozenModel):
    schema_version: Literal["nurse.v2.sre-report-response/2"]
    core_result_digest: SHA256
    formal_interpretations: tuple[BriefClaim, ...]
    step_findings: tuple[BriefStepFinding, ...]
    final_synthesis: tuple[BriefClaim, ...]
    limitations: tuple[BriefClaim, ...]
    next_checks: tuple[BriefClaim, ...]
    summary_claim_ids: Annotated[tuple[str, ...], AfterValidator(_unique_strings)]

    @model_validator(mode="after")
    def closure(self) -> Self:
        ids = [c.claim_id for _, c in sectioned_claims(self)]
        if len(set(ids)) != len(ids) or not set(self.summary_claim_ids) <= set(ids):
            raise ValueError("brief report reference closure differs")
        return self


class BriefClaimCheck(StrictFrozenModel):
    section: Literal[
        "executive_summary",
        "formal_interpretations",
        "step_findings",
        "final_synthesis",
        "limitations",
        "next_checks",
    ]
    proposal: BriefClaim | BriefStepFinding
    reason_codes: tuple[str, ...]
    surface_status: Literal["model_unverified", "blocked"]
    semantic_status: Literal["not_established"] = "not_established"


class ConditionReportResponse(StrictFrozenModel):
    schema_version: Literal["nurse.v2.sre-report-response/3"]
    core_result_digest: SHA256
    executive_summary: tuple[BriefClaim, ...]
    formal_interpretations: tuple[BriefClaim, ...]
    step_findings: tuple[BriefStepFinding, ...]
    final_synthesis: tuple[BriefClaim, ...]
    limitations: tuple[BriefClaim, ...]
    next_checks: tuple[BriefClaim, ...]

    @model_validator(mode="after")
    def closure(self) -> Self:
        ids = [c.claim_id for _, c in sectioned_claims(self)]
        if len(set(ids)) != len(ids):
            raise ValueError("condition report reference closure differs")
        return self


def sectioned_claims(
    response: BriefReportResponse | ConditionReportResponse,
) -> list[tuple[str, BriefClaim]]:
    sections = (
        ("executive_summary", *SECTIONS)
        if isinstance(response, ConditionReportResponse)
        else SECTIONS
    )
    return [(section, claim) for section in sections for claim in getattr(response, section)]


def parse_brief_report(
    raw: bytes,
    *,
    condition_explanations: bool = False,
    comparable_evidence: bool = False,
) -> BriefReportResponse | ConditionReportResponse | ComparableReportResponse:
    if comparable_evidence:
        from nurse_rcl.v2.live.ops_sre_comparable_claims import parse_comparable_report

        return parse_comparable_report(raw)
    value = decode_json_object(raw, REPORT_LIMITS)
    try:
        response_type = ConditionReportResponse if condition_explanations else BriefReportResponse
        return response_type.model_validate_json(encode_json(value, REPORT_LIMITS), strict=True)
    except ValidationError:
        raise ExchangeBoundaryError("sre_brief_report_schema") from None


def check_brief_assertion(
    evidence_ref: str,
    field: str,
    value: Any,
    *,
    briefs: dict[str, Any],
    provided_refs: set[str],
) -> tuple[str, ...]:
    if evidence_ref not in provided_refs or evidence_ref not in briefs:
        return ("brief_not_provided",)
    fields = briefs[evidence_ref]["fields"]
    if field not in fields:
        return ("brief_field_not_provided",)
    expected = fields[field]
    if type(expected) not in (str, int, float, bool, type(None)) or (
        type(value) is not type(expected) or value != expected
    ):
        return ("brief_value_mismatch",)
    return ()


def _exact_proposal(claim: BriefClaim) -> OpsCompositionClaim:
    """Adapt only f statements to their unchanged checker, without b substitution."""
    body = claim.model_dump(mode="json")
    body.pop("step_ref", None)
    body.pop("evidence_refs")
    body["fact_refs"] = list(claim.fact_refs)
    body["assertions"] = [
        dict(fact_ref=a.evidence_ref, field=a.field, value=a.value)
        for a in claim.assertions
        if a.evidence_ref.startswith("f")
    ]
    if claim.recommendation is not None and any(
        ref.startswith("b") for ref in claim.recommendation.precondition_refs
    ):
        body["recommendation"] = None  # Already blocked; never turn b into an f precondition.
    return OpsCompositionClaim.model_validate_json(canonical_json_bytes(body), strict=True)


def check_brief_report(
    response: BriefReportResponse | ConditionReportResponse | ComparableReportResponse,
    *,
    supply: dict[str, Any],
    fact_store: NativeFactStore,
    candidates: tuple[str, ...],
    require_supplied_decimals: bool = False,
    decimal_value_policy: DecimalValuePolicy = FLOAT_ONLY,
) -> tuple[BriefClaimCheck, ...] | tuple[ComparableClaimCheck, ...]:
    """Check one admitted supply; callers bind its full context and source identity."""
    from nurse_rcl.v2.live.ops_sre_comparable_claims import (
        ComparableReportResponse,
        check_comparable_report,
    )

    if isinstance(response, ComparableReportResponse):
        return check_comparable_report(
            response, supply=supply, fact_store=fact_store, candidates=candidates,
            require_supplied_decimals=require_supplied_decimals,
            decimal_value_policy=decimal_value_policy,
        )
    return cast(
        tuple[BriefClaimCheck, ...],
        _check_brief_items(
            response,
            items=sectioned_claims(response),
            result_type=BriefClaimCheck,
            supply=supply,
            fact_store=fact_store,
            candidates=candidates,
            require_supplied_decimals=require_supplied_decimals,
            decimal_value_policy=decimal_value_policy,
        ),
    )


def _check_brief_items(
    response: Any,
    *,
    items: list[tuple[str, Any]],
    result_type: type[StrictFrozenModel],
    supply: dict[str, Any],
    fact_store: NativeFactStore,
    candidates: tuple[str, ...],
    comparable: bool = False,
    require_supplied_decimals: bool = False,
    decimal_value_policy: DecimalValuePolicy = FLOAT_ONLY,
) -> tuple[Any, ...]:
    """Shared checks after version-specific parsing, without changing proposals."""
    validate_decimal_value_policy(
        decimal_value_policy, require_supplied_decimals=require_supplied_decimals,
    )
    if response.core_result_digest != supply["core_result_digest"]:
        raise ValueError("brief response core digest differs")
    records = NativeFactLookup(fact_store)
    facts = _EvidenceLookup(records)
    # Full vocabulary is retained even for hidden local facts; only the permission
    # sets below confer assertion rights, not this language-diagnostic index.
    language_index = LanguageFactIndex.build(facts.values())
    briefs = {b["evidence_ref"]: b for b in supply["briefs"]}
    b_refs, f_refs = set(supply["provided_brief_refs"]), set(supply["provided_fact_refs"])
    if (
        b_refs != briefs.keys()
        or len(briefs) != len(supply["briefs"])
        or (not f_refs <= facts.keys())
    ):
        raise ValueError("brief supply alias closure differs")
    steps = {s["step_ref"]: s for s in supply["steps"]}
    results = []
    for section, claim in items:
        reasons: set[str] = set()
        refs = set(claim.evidence_refs)
        if not refs <= b_refs | f_refs:
            reasons.add("evidence_not_provided")
        asserted = {a.evidence_ref for a in claim.assertions}
        if not asserted <= refs:
            reasons.add("hidden_evidence_reference")
        if not claim.assertions:
            reasons.add("explicit_assertion_required")
        b_cited = [briefs[ref] for ref in claim.evidence_refs if ref in briefs and ref in b_refs]
        f_cited = [facts[ref] for ref in claim.fact_refs if ref in facts]
        for assertion in claim.assertions:
            if assertion.evidence_ref.startswith("b"):
                reasons.update(
                    check_brief_assertion(
                        assertion.evidence_ref,
                        assertion.field,
                        assertion.value,
                        briefs=briefs,
                        provided_refs=b_refs,
                    )
                )
        if claim.recommendation is not None:
            preconditions = set(claim.recommendation.precondition_refs)
            if not preconditions <= refs:
                reasons.add("hidden_evidence_reference")
            if any(ref.startswith("b") for ref in preconditions):
                reasons.add("summary_not_observed_precondition")
        if claim.fact_refs:
            new_operation = comparable and claim.operation in {
                "compare_mechanisms",
                "propose_verification",
            }
            if new_operation:
                from nurse_rcl.v2.live.ops_sre_comparable_claims import exact_comparable_claim

                exact = exact_comparable_claim(claim)
            else:
                exact = _exact_proposal(claim)
            reference_errors = _references(exact, facts)
            reasons.update(reference_errors)
            if not reference_errors:
                reasons.update(_assertions(exact, facts))
                if not new_operation:
                    reasons.update(_recommendation(exact, facts, (), None))
            if not b_cited and not new_operation:
                # New organization is code-owned; original f rule set remains intact.
                old_section = (
                    "conditional_next_steps"
                    if section == "next_checks"
                    else "limitations"
                    if section == "limitations"
                    else "formal_grounds"
                    if all(records[f.fact_ref].role == "formal_record" for f in f_cited)
                    else "supplementary_findings"
                )
                original = check_ops_report_records(
                    OpsEmpiricalReportResponse(
                        schema_version="nurse.v2.ops-layered-report-response/2",
                        core_result_digest=response.core_result_digest,
                        claims=(
                            OpsEmpiricalLayeredClaim.model_validate(
                                dict(section=old_section, claim=exact)
                            ),
                        ),
                    ),
                    records=records,
                    facts=facts,
                    candidates=candidates,
                    provided_fact_refs=tuple(supply["provided_fact_refs"]),
                    language_checker=lambda _: (),
                )
                reasons.update(original[0].reason_codes)
        if comparable:
            from nurse_rcl.v2.live.ops_sre_comparable_claims import comparable_checks

            reasons.update(
                comparable_checks(claim, briefs=briefs, facts=facts, b_refs=b_refs, f_refs=f_refs)
            )
        targets = set(claim.target_service_keys)
        scopes = [tuple(b["target_service_keys"]) for b in b_cited] + [
            f.target_service_keys for f in f_cited
        ]
        if (
            not targets <= set(candidates)
            or any(not set(s) <= targets for s in scopes)
            or (targets and not targets <= {t for scope in scopes for t in scope})
            or any(
                (f.kind == "competition" and f.target_service_keys != claim.target_service_keys)
                or (
                    not f.target_service_keys and targets and f.kind not in {"limit", "uncertainty"}
                )
                for f in f_cited
            )
        ):
            reasons.add("target_scope_mismatch")
        if claim.interval is not None:
            interval = [claim.interval.start_unix_ns, claim.interval.end_unix_ns]
            windows = [b["window_ns"] for b in b_cited if b["window_ns"] is not None]
            if (
                any(w != interval for w in windows)
                or any(
                    f.intervals
                    and not any([i.start_unix_ns, i.end_unix_ns] == interval for i in f.intervals)
                    for f in f_cited
                )
                or not (windows or any(f.intervals for f in f_cited))
            ):
                reasons.add("interval_source_mismatch")
        if section == "formal_interpretations" and (
            any(records[f.fact_ref].role != "formal_record" for f in f_cited)
            or any(b["source_role"] not in {"formal_record", "core_published"} for b in b_cited)
        ):
            reasons.add("supplementary_as_formal")
        if section == "step_findings":
            step = steps.get(claim.step_ref)
            if (
                step is None
                or step["executed"]["status"] != "completed"
                or not refs <= set(step["delivered_brief_refs"] + step["delivered_fact_refs"])
            ):
                reasons.add("step_evidence_not_delivered")
        operation = claim.operation
        recommended = operation in {"propose_verification", "conditional_mitigation"}
        expected_type = (
            "recommendation"
            if recommended
            else "observation"
            if operation == "describe_observation"
            else "inference"
        )
        expected_strength = {
            "recommendation": "conditional_recommendation",
            "observation": "frozen_observation",
            "inference": "ranking_inference",
        }
        if claim.statement_type != expected_type:
            reasons.add("operation_type_mismatch")
        if claim.requested_claim_strength != expected_strength[expected_type]:
            reasons.add("operation_strength_mismatch")
        if recommended != (claim.recommendation is not None) or (
            recommended != (section == "next_checks")
        ):
            reasons.add("recommendation_shape_mismatch")
        if b_cited and operation == "describe_observation":
            reasons.add("summary_not_observation")
        if operation in {"explain_priority", "compare_candidates"}:
            kind = "ranking" if operation == "explain_priority" else "competition"
            if not any(
                f.kind == kind and records[f.fact_ref].role == "formal_record" for f in f_cited
            ):
                reasons.add("ranking_basis_missing")
            if len(targets) != (1 if kind == "ranking" else 2):
                reasons.add("operation_target_count")
            if claim.interval is not None:
                reasons.add("interval_forbidden")
        if operation == "synthesize_evidence" and len(b_cited) + len(f_cited) < 2:
            reasons.add("missing_operation_basis")
        if (
            operation == "explain_uncertainty"
            and not any(_uncertain(f) or f.status_codes for f in f_cited)
            and not any(
                b["fields"].get("truth_value") in {"false", "unknown"}
                or b["fields"].get("unknown_count", 0) > 0
                or b["fields"].get("false_count", 0) > 0
                or b["limitations"]
                for b in b_cited
            )
        ):
            reasons.add("missing_operation_basis")
        if operation == "conditional_mitigation" and not any(
            f.kind == "mitigation" for f in f_cited
        ):
            reasons.add("missing_operation_basis")
        # This structural interface exposes only existing f refs to old source
        # diagnostics. b values are checked above; no invented f records or hidden
        # leaf lookups are added to the legacy evidence universe.
        language_claim = cast(ReportCompositionClaim, claim)
        for path, text in claim_free_text_fields(language_claim):
            if model_ranking_override(text):
                reasons.add("model_ranking_override")
            reasons.update(
                _check_text(
                    text,
                    language_claim,
                    _LanguageScope(tuple(f_cited), candidates),
                    conditional_outcome=claim.recommendation is not None
                    and path.startswith("recommendation.possible_outcomes["),
                    language_policy=SRE_LANGUAGE_POLICY,
                    indexed_facts=facts,
                    index=language_index,
                )
            )
        if require_supplied_decimals:
            from nurse_rcl.v2.live.ops_sre_d22_numbers import unsupplied_decimal_in_claim

            if unsupplied_decimal_in_claim(
                claim, briefs=briefs, facts=facts,
                provided_brief_refs=b_refs, provided_fact_refs=f_refs,
                decimal_value_policy=decimal_value_policy,
            ):
                reasons.add("surface_number_not_supplied")
        if frozen_ranking_fields(language_claim):
            reasons.add("frozen_ranking_mutation")
        results.append(
            result_type.model_validate(
                dict(
                    section=section,
                    proposal=claim,
                    reason_codes=tuple(sorted(reasons)),
                    surface_status="blocked" if reasons else "model_unverified",
                )
            )
        )
    return tuple(results)
