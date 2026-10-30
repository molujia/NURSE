"""Pure structured report checks; callers must independently replay source authority."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.deterministic_report import DeterministicSREReport
from nurse_rcl.v2.report_composition_proposals import ReportCompositionClaim
from nurse_rcl.v2.report_evidence_schemas import (
    ReportEvidenceBinding,
    ReportEvidenceFact,
    ReportEvidenceView,
)
from nurse_rcl.v2.report_mechanical_schemas import (
    MECHANICAL_STAGES,
    MechanicalReason,
    MechanicalSafetyCode,
    ReportClaimMechanicalValidation,
    ReportMechanicalCheck,
)


def _values(fact: ReportEvidenceFact) -> dict[str, object]:
    return {item.name: item.value for item in fact.values}


def _references(
    claim: ReportCompositionClaim, facts: Mapping[str, ReportEvidenceFact]
) -> set[MechanicalReason]:
    reasons: set[MechanicalReason] = set()
    cited = set(claim.fact_refs)
    if not cited <= facts.keys():
        reasons.add("unknown_fact_reference")
    extra_refs = {item.fact_ref for item in claim.assertions}
    if claim.recommendation is not None:
        extra_refs.update(claim.recommendation.precondition_refs)
    if not extra_refs <= cited:
        reasons.add("hidden_fact_reference")
    return reasons


def _is_context(fact: ReportEvidenceFact, view: ReportEvidenceView) -> bool:
    return fact.fact_ref in view.mandatory_fact_refs and (
        (fact.kind == "candidate_evidence" and fact.scope_basis == "frozen_assessment")
        or (not fact.target_service_keys and fact.kind in {"uncertainty", "limit"})
    )


def _scope(
    claim: ReportCompositionClaim, facts: tuple[ReportEvidenceFact, ...], view: ReportEvidenceView
) -> set[MechanicalReason]:
    reasons: set[MechanicalReason] = set()
    targets = set(claim.target_service_keys)
    if not targets <= set(view.top_k):
        reasons.add("target_outside_top_k")
    for fact in facts:
        scoped = set(fact.target_service_keys)
        if not scoped:
            if targets and fact.kind not in {"uncertainty", "limit"}:
                reasons.add("target_scope_mismatch")
            continue
        shared_declared = (
            len(targets) == 1
            and fact.kind == "predicate"
            and fact.scope_basis == "declared_hypothesis"
            and targets <= scoped
        )
        if not scoped <= targets and not shared_declared:
            reasons.add("target_scope_mismatch")
    if claim.operation in {"explain_priority", "compare_candidates"} and claim.interval is not None:
        reasons.add("interval_forbidden")
    if claim.operation == "describe_observation" and claim.interval is None:
        reasons.add("interval_required")
    if claim.interval is not None:
        requested = (claim.interval.start_unix_ns, claim.interval.end_unix_ns)
        temporal = tuple(f for f in facts if f.target_service_keys and f.intervals)
        if not any(
            requested == (i.start_unix_ns, i.end_unix_ns)
            for fact in temporal
            for i in fact.intervals
        ):
            reasons.add("interval_source_mismatch")
        for fact in temporal:
            if fact.scope_basis == "declared_hypothesis" and len(fact.target_service_keys) > 1:
                reasons.add("interval_scope_ambiguous")
            if not any((i.start_unix_ns, i.end_unix_ns) == requested for i in fact.intervals):
                reasons.add("interval_source_mismatch")
        if claim.operation == "describe_observation":
            for fact in facts:
                if not _is_context(fact, view) and not any(
                    requested == (i.start_unix_ns, i.end_unix_ns) for i in fact.intervals
                ):
                    reasons.add("interval_source_mismatch")
    return reasons


def _positive_count(value: object, minimum: int = 0) -> bool:
    return type(value) is int and value > minimum


def _uncertain(fact: ReportEvidenceFact) -> bool:
    values = _values(fact)
    if fact.kind == "limit":
        return True
    if fact.kind == "observation" and (
        values.get("evidence_state") in {"unknown", "no_record", "degraded"}
        or values.get("baseline_record_state") == "no_record"
        or values.get("fault_record_state") == "no_record"
    ):
        return True
    if fact.kind == "predicate" and (
        values.get("truth_value") == "unknown"
        or (values.get("requirement") == "necessary" and values.get("truth_value") == "false")
        or (values.get("requirement") == "refuting" and values.get("truth_value") == "true")
    ):
        return True
    if any(
        values.get(name) is True
        for name in (
            "has_evidence_tie",
            "has_unknown_evidence",
            "has_conflicting_evidence",
            "has_missing_evidence",
            "evidence_tied",
            "ordering_decided_by_tie_break",
        )
    ):
        return True
    if any(
        _positive_count(values.get(name))
        for name in (
            "unknown_predicate_count",
            "refuting_predicate_count",
            "conflicting_assertion_count",
            "missing_reason_count",
            "necessary_false_count",
            "necessary_unknown_count",
            "optional_unknown_count",
            "refuting_unknown_count",
            "refuting_true_count",
            "verification_actions_truncated_count",
            "conditional_mitigations_truncated_count",
        )
    ):
        return True
    if _positive_count(values.get("tie_group_member_count"), 1):
        return True
    if values.get("evidence_sufficiency") in {"insufficient", "limited", "mixed"}:
        return True
    if values.get("lifecycle_status") == "conflicted":
        return True
    if values.get("run_mode") in {
        "partial_semantic",
        "b0_fallback",
        "deterministic_report_fallback",
        "failure",
    }:
        return True
    # In these source-backed categories codes are respectively missing reasons or
    # degradation reasons. Do not infer uncertainty from arbitrary prose or codes.
    return bool(fact.status_codes) and fact.kind in {"candidate_evidence", "uncertainty"}


def _operation(
    claim: ReportCompositionClaim, facts: tuple[ReportEvidenceFact, ...], view: ReportEvidenceView
) -> set[MechanicalReason]:
    reasons: set[MechanicalReason] = set()
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
    counts = (
        (0, 1)
        if operation == "explain_uncertainty"
        else ((2,) if operation == "compare_candidates" else (1,))
    )
    if len(claim.target_service_keys) not in counts:
        reasons.add("operation_target_count")
    if recommended != (claim.recommendation is not None):
        reasons.add("recommendation_shape_mismatch")
    primary = tuple(f for f in facts if not _is_context(f, view))
    diagnostic = {"observation", "predicate", "hypothesis", "ranking", "candidate_evidence"}
    allowed = {
        "describe_observation": {"observation", "predicate"},
        "explain_priority": diagnostic | {"uncertainty"},
        "compare_candidates": diagnostic | {"competition", "uncertainty"},
        "synthesize_evidence": diagnostic,
        "explain_uncertainty": diagnostic | {"competition", "uncertainty", "limit"},
        "propose_verification": {"observation", "predicate", "hypothesis", "verification"},
        "conditional_mitigation": {"observation", "predicate", "hypothesis", "mitigation"},
    }[operation]
    if any(f.kind not in allowed for f in primary):
        reasons.add("invalid_primary_fact")
    basis = False
    if operation == "describe_observation":

        def observed(fact: ReportEvidenceFact) -> bool:
            return (
                fact.kind in {"observation", "predicate"}
                and fact.scope_basis == "bound_observation"
                and fact.allowed_claim_strength == "frozen_observation"
                and fact.target_service_keys == claim.target_service_keys
            )

        basis = any(observed(f) for f in primary)
        if not all(observed(f) for f in primary):
            reasons.add("invalid_primary_fact")
    elif operation == "explain_priority":
        basis = any(
            f.kind == "ranking" and f.target_service_keys == claim.target_service_keys
            for f in primary
        )
    elif operation == "compare_candidates":
        basis = any(
            f.kind == "competition" and f.target_service_keys == claim.target_service_keys
            for f in primary
        )
        if any(
            f.kind == "competition" and f.target_service_keys != claim.target_service_keys
            for f in primary
        ):
            reasons.add("invalid_primary_fact")
    elif operation == "synthesize_evidence":
        basis = (
            len(
                {
                    f.fact_ref
                    for f in facts
                    if f.kind in diagnostic
                    and set(claim.target_service_keys) & set(f.target_service_keys)
                }
            )
            >= 2
        )
    elif operation == "explain_uncertainty":
        basis = any(_uncertain(f) for f in facts)
    elif operation == "propose_verification":
        basis = any(
            f.kind in {"observation", "predicate", "hypothesis"}
            and set(claim.target_service_keys) <= set(f.target_service_keys)
            and bool(claim.target_service_keys)
            for f in primary
        )
    elif operation == "conditional_mitigation":
        basis = claim.recommendation is not None and any(
            f.kind == "mitigation"
            and f.target_service_keys == claim.target_service_keys
            and _values(f).get("action_class") == claim.recommendation.action_class
            for f in primary
        )
    if not basis:
        reasons.add("missing_operation_basis")
    return reasons


def _assertions(
    claim: ReportCompositionClaim, facts: Mapping[str, ReportEvidenceFact]
) -> set[MechanicalReason]:
    reasons: set[MechanicalReason] = set()
    for assertion in claim.assertions:
        values = _values(facts[assertion.fact_ref])
        if assertion.field not in values:
            reasons.add("assertion_field_missing")
        elif canonical_json_bytes(values[assertion.field]) != canonical_json_bytes(assertion.value):
            reasons.add("assertion_value_mismatch")
    return reasons


def _recommendation(
    claim: ReportCompositionClaim,
    facts: Mapping[str, ReportEvidenceFact],
    bindings: tuple[ReportEvidenceBinding, ...],
    foundation: DeterministicSREReport | None,
) -> set[MechanicalReason]:
    reasons = _recommendation_common(claim, facts, bindings, foundation)
    advice = claim.recommendation
    if advice is not None:
        expected = (
            {"true", "false", "unknown"}
            if claim.operation == "propose_verification"
            else {"improves", "no_change", "worsens_or_unknown"}
        )
        if {item.outcome for item in advice.possible_outcomes} != expected:
            reasons.add("recommendation_outcomes_mismatch")
    return reasons


def _recommendation_common(
    claim: ReportCompositionClaim,
    facts: Mapping[str, ReportEvidenceFact],
    bindings: tuple[ReportEvidenceBinding, ...],
    foundation: DeterministicSREReport | None,
) -> set[MechanicalReason]:
    """Shared status, action, premise and mitigation checks; no outcome invention."""
    advice = claim.recommendation
    if advice is None:
        return set()
    reasons: set[MechanicalReason] = set()
    if advice.status != "proposed":
        reasons.add("recommendation_status_invalid")
    preconditions = tuple(facts[ref] for ref in advice.precondition_refs)
    verification = claim.operation == "propose_verification"
    if verification:
        if advice.action_class != "read_only_diagnostic_check":
            reasons.add("recommendation_action_class_mismatch")
        if not preconditions or any(
            fact.kind not in {"observation", "predicate", "hypothesis"}
            or not set(claim.target_service_keys) <= set(fact.target_service_keys)
            for fact in preconditions
        ):
            reasons.add("recommendation_precondition_invalid")
        return reasons
    if advice.action_class == "read_only_diagnostic_check":
        reasons.add("recommendation_action_class_mismatch")
    if any(
        f.kind != "predicate" or not set(claim.target_service_keys) <= set(f.target_service_keys)
        for f in preconditions
    ):
        reasons.add("recommendation_precondition_invalid")
    if any(_values(f).get("truth_value") != "true" for f in preconditions):
        reasons.add("mitigation_precondition_not_true")
    bound = {item.fact_ref: item for item in bindings}
    candidates = []
    if foundation is not None:
        by_id = {item.mitigation_id: item for item in foundation.conditional_mitigations}
        for ref in claim.fact_refs:
            binding = bound.get(ref)
            if facts[ref].kind != "mitigation" or binding is None:
                continue
            source = by_id.get(binding.source_id)
            if (
                source is not None
                and binding.source_kind == "conditional_mitigation"
                and (source.target_service_key,) == claim.target_service_keys
                and source.action_class == advice.action_class
            ):
                candidates.append(source)
            else:
                reasons.add("mitigation_source_mismatch")
    if not candidates:
        reasons.add("mitigation_source_mismatch")
    provided_ids = {
        bound[f.fact_ref].source_id
        for f in preconditions
        if f.kind == "predicate"
        and f.fact_ref in bound
        and bound[f.fact_ref].source_kind == "predicate"
    }
    # Every cited mitigation must retain its own frozen prerequisites; a sibling
    # template with the same class cannot lend it a different prerequisite set.
    if any(
        not set(source.precondition_predicate_result_ids) <= provided_ids for source in candidates
    ):
        reasons.add("mitigation_preconditions_missing")
    return reasons


def _context_refs(
    claim: ReportCompositionClaim, view: ReportEvidenceView, scope_passed: bool
) -> tuple[str, ...]:
    facts = {fact.fact_ref: fact for fact in view.facts}
    needed = set(claim.fact_refs) & facts.keys()
    for ref in view.mandatory_fact_refs:
        fact = facts[ref]
        if (not fact.target_service_keys and fact.kind in {"uncertainty", "limit"}) or (
            scope_passed
            and fact.kind == "candidate_evidence"
            and fact.scope_basis == "frozen_assessment"
            and set(fact.target_service_keys) & set(claim.target_service_keys)
        ):
            needed.add(ref)
    queue = deque(needed)
    while queue:
        for ref in facts[queue.popleft()].related_fact_refs:
            if ref not in needed:
                needed.add(ref)
                queue.append(ref)
    return tuple(f.fact_ref for f in view.facts if f.fact_ref in needed)


def _binding_overlap(first: ReportEvidenceBinding, second: ReportEvidenceBinding) -> bool:
    def primary(binding: ReportEvidenceBinding) -> tuple[str, str]:
        namespace = binding.source_kind
        if namespace in {"ranked_candidate", "top_one_basis"}:
            namespace = "frozen_score"
        elif namespace == "candidate_evidence":
            namespace = "candidate_summary"
        return namespace, binding.source_id

    # Cross-kind overlap requires an explicit frozen anchor, never coincidental
    # equality of display identifiers such as a candidate summary's service key.
    record_kinds = {
        "observation",
        "predicate",
        "report_hypothesis",
        "role_assertion",
        "verification_action",
        "conditional_mitigation",
        "competition",
        "report_limit",
    }
    return (
        primary(first) == primary(second)
        or bool(set(first.same_source_ids) & set(second.same_source_ids))
        or (first.source_kind in record_kinds and first.source_id in second.same_source_ids)
        or (second.source_kind in record_kinds and second.source_id in first.same_source_ids)
    )


def _shared_groups(
    claim: ReportCompositionClaim,
    view: ReportEvidenceView,
    bindings: tuple[ReportEvidenceBinding, ...],
) -> tuple[tuple[str, ...], ...]:
    by_ref = {b.fact_ref: b for b in bindings}
    refs = [
        f.fact_ref for f in view.facts if f.fact_ref in claim.fact_refs and f.fact_ref in by_ref
    ]
    seen: set[str] = set()
    groups = []
    for ref in refs:
        if ref in seen:
            continue
        component = {ref}
        queue = deque((ref,))
        seen.add(ref)
        while queue:
            current = queue.popleft()
            for other in refs:
                if other not in seen and _binding_overlap(by_ref[current], by_ref[other]):
                    seen.add(other)
                    component.add(other)
                    queue.append(other)
        if len(component) > 1:
            groups.append(tuple(r for r in refs if r in component))
    return tuple(groups)


def _validate_claim(
    claim: ReportCompositionClaim,
    *,
    view: ReportEvidenceView,
    bindings: tuple[ReportEvidenceBinding, ...],
    foundation: DeterministicSREReport | None,
) -> ReportClaimMechanicalValidation:
    """Compute mechanics from already checked inputs; this helper confers no authority."""
    by_ref = {f.fact_ref: f for f in view.facts}
    cited = tuple(by_ref[ref] for ref in claim.fact_refs if ref in by_ref)
    checks = []
    rejected = False
    scope_passed = False
    for stage in MECHANICAL_STAGES:
        reasons: set[MechanicalReason] = set()
        if not rejected:
            if stage == "references":
                reasons = _references(claim, by_ref)
            elif stage == "scope":
                reasons = _scope(claim, cited, view)
                scope_passed = not reasons
            elif stage == "operation":
                reasons = _operation(claim, cited, view)
            elif stage == "assertions":
                reasons = _assertions(claim, by_ref)
            elif stage == "recommendation":
                reasons = _recommendation(claim, by_ref, bindings, foundation)
        checks.append(
            ReportMechanicalCheck(
                stage=stage,
                status="skipped" if rejected else ("rejected" if reasons else "passed"),
                reason_codes=tuple(sorted(reasons)),
            )
        )
        rejected = rejected or bool(reasons)
    safety: set[MechanicalSafetyCode] = set()
    if claim.operation in {"propose_verification", "conditional_mitigation"}:
        safety.update(
            (
                "proposed_only",
                "no_automatic_execution",
                "sealed_ranking_immutable",
                "outcomes_are_conditional",
            )
        )
        if claim.operation == "propose_verification":
            safety.add("read_only")
        else:
            safety.update(("operator_approval_required", "reversible_action_only"))
    if claim.operation == "synthesize_evidence":
        safety.add("distinct_references_do_not_imply_independent_evidence")
    context = _context_refs(claim, view, scope_passed)
    return ReportClaimMechanicalValidation(
        proposal=claim,
        checks=tuple(checks),
        mechanical_status="rejected" if rejected else "passed",
        required_context_refs=context,
        uncited_context_refs=tuple(ref for ref in context if ref not in claim.fact_refs),
        required_safety_codes=tuple(sorted(safety)),
        shared_provenance_groups=_shared_groups(claim, view, bindings),
    )
