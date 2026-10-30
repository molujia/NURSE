"""Role-aware checks on retained values only. No scorer or source replay."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

from nurse_rcl.v2.live.ops_report_context import OpsReportContext, OpsReportFact
from nurse_rcl.v2.live.ops_report_schema import (
    OpsClaimCheck,
    OpsEmpiricalReportResponse,
    OpsLayeredReportResponse,
)
from nurse_rcl.v2.report_composition_proposals import ReportCompositionClaim
from nurse_rcl.v2.report_evidence_schemas import ReportEvidenceFact
from nurse_rcl.v2.report_language_rules import check_fact_claim_language
from nurse_rcl.v2.report_mechanical_checks import (
    _assertions,
    _recommendation,
    _references,
    _uncertain,
)
from nurse_rcl.v2.report_presentation_schemas import (
    ReportLanguageFieldCheck,
    claim_free_text_fields,
)


def model_ranking_override(text: str) -> bool:
    """Original ranking-mutation diagnostic, shared by both report protocols."""
    return (
        re.search(
            r"(?:corrected|revised|new|alternative)\s+(?:top[- ]?1|ranking)|"
            r"(?:修正|改为|调整|新|替代).{0,8}(?:排名|第一|top[- ]?1)",
            text,
            re.IGNORECASE,
        )
        is not None
    )


def check_ops_report(
    response: OpsLayeredReportResponse | OpsEmpiricalReportResponse,
    *,
    context: OpsReportContext,
    provided_fact_refs: tuple[str, ...] | None = None,
    language_policy: str | None = None,
) -> tuple[OpsClaimCheck, ...]:
    if response.core_result_digest != context.core_result_digest:
        raise ValueError("report response core digest differs")
    facts = {f.fact.fact_ref: f.fact for f in context.facts}
    records = {f.fact.fact_ref: f for f in context.facts}
    return check_ops_report_records(
        response,
        facts=facts,
        records=records,
        candidates=tuple(r.candidate_service_key for r in context.ranking),
        provided_fact_refs=provided_fact_refs,
        language_policy=language_policy,
    )


def check_ops_report_records(
    response: OpsLayeredReportResponse | OpsEmpiricalReportResponse,
    *,
    facts: Mapping[str, ReportEvidenceFact],
    records: Mapping[str, OpsReportFact],
    candidates: tuple[str, ...],
    provided_fact_refs: tuple[str, ...] | None,
    language_policy: str | None = None,
    language_checker: Callable[[ReportCompositionClaim], tuple[ReportLanguageFieldCheck, ...]]
    | None = None,
) -> tuple[OpsClaimCheck, ...]:
    """Identical claim rules over admitted records; caller binds the core digest."""
    empirical = isinstance(response, OpsEmpiricalReportResponse)
    if empirical and provided_fact_refs is None:
        raise ValueError("empirical report requires actual provided fact aliases")
    if provided_fact_refs is not None and (
        len(set(provided_fact_refs)) != len(provided_fact_refs)
        or not set(provided_fact_refs) <= facts.keys()
    ):
        raise ValueError("provided report aliases differ from complete facts")
    checks = []
    for proposed in response.claims:
        claim = proposed.claim
        reasons: set[str] = set(_references(claim, facts))
        if provided_fact_refs is not None and not set(claim.fact_refs) <= set(provided_fact_refs):
            reasons.add("fact_not_provided")
        cited = tuple(facts[ref] for ref in claim.fact_refs if ref in facts)
        if not reasons:
            reasons.update(_assertions(claim, facts))
            reasons.update(_recommendation(claim, facts, (), None))
        if not claim.assertions:
            reasons.add("explicit_assertion_required")
        if proposed.section == "formal_grounds" and any(
            records[f.fact_ref].role == "supplementary" for f in cited
        ):
            reasons.add("supplementary_as_formal")
        if proposed.section == "supplementary_findings" and not any(
            records[f.fact_ref].role == "supplementary" for f in cited
        ):
            reasons.add("supplementary_basis_missing")
        targets = set(claim.target_service_keys)
        if (
            not targets <= set(candidates)
            or any(
                not set(f.target_service_keys) <= targets
                or (
                    not f.target_service_keys and targets and f.kind not in {"limit", "uncertainty"}
                )
                or (f.kind == "competition" and f.target_service_keys != claim.target_service_keys)
                for f in cited
            )
            or (targets and not targets <= {t for f in cited for t in f.target_service_keys})
        ):
            reasons.add("target_scope_mismatch")
        if claim.interval is not None:
            interval = (claim.interval.start_unix_ns, claim.interval.end_unix_ns)
            temporal = tuple(f for f in cited if f.intervals)
            if not temporal or any(
                not any((i.start_unix_ns, i.end_unix_ns) == interval for i in f.intervals)
                for f in temporal
            ):
                reasons.add("interval_source_mismatch")
        operation = claim.operation
        recommended = operation in {"propose_verification", "conditional_mitigation"}
        expected_type = (
            "recommendation"
            if recommended
            else ("observation" if operation == "describe_observation" else "inference")
        )
        expected_strength = {
            "recommendation": "conditional_recommendation",
            "observation": "frozen_observation",
            "inference": "ranking_inference",
        }[expected_type]
        if claim.statement_type != expected_type:
            reasons.add("operation_type_mismatch")
        if claim.requested_claim_strength != expected_strength:
            reasons.add("operation_strength_mismatch")
        if recommended != (claim.recommendation is not None):
            reasons.add("recommendation_shape_mismatch")
        if recommended != (proposed.section == "conditional_next_steps"):
            reasons.add("recommendation_section_mismatch")
        if operation == "describe_observation" and (
            claim.interval is None or any(f.scope_basis != "bound_observation" for f in cited)
        ):
            reasons.add("observation_scope_missing")
        if operation in {"explain_priority", "compare_candidates"}:
            required = "ranking" if operation == "explain_priority" else "competition"
            if proposed.section != "formal_grounds" or not any(f.kind == required for f in cited):
                reasons.add("ranking_basis_missing")
            if len(targets) != (1 if operation == "explain_priority" else 2):
                reasons.add("operation_target_count")
            if claim.interval is not None:
                reasons.add("interval_forbidden")
        if operation == "synthesize_evidence" and len(cited) < 2:
            reasons.add("missing_operation_basis")
        if operation == "explain_uncertainty" and not any(
            _uncertain(f)
            or f.status_codes
            or any(
                v.name in {"truth_value", "availability"}
                and v.value in {"unknown", "partial", "unavailable"}
                for v in f.values
            )
            for f in cited
        ):
            reasons.add("missing_operation_basis")
        if operation == "conditional_mitigation" and not any(f.kind == "mitigation" for f in cited):
            reasons.add("missing_operation_basis")
        language = (
            language_checker(claim)
            if language_checker
            else check_fact_claim_language(
                claim,
                facts=tuple(facts.values()),
                candidate_service_keys=candidates,
                ops_empirical=empirical,
                language_policy=language_policy,
            )
        )
        for checked in language:
            reasons.update(checked.reason_codes)
        for _, text in claim_free_text_fields(claim):
            if model_ranking_override(text):
                reasons.add("model_ranking_override")
        checks.append(
            OpsClaimCheck(
                proposal=proposed,
                reason_codes=tuple(sorted(reasons)),
                surface_status="blocked" if reasons else "model_unverified",
            )
        )
    return tuple(checks)
