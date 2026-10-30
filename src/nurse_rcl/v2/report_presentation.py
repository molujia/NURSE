"""Independently pinned, replayable presentation delivery over frozen report sources."""

from __future__ import annotations

import hmac
from typing import Any, TypeVar

from pydantic import BaseModel

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.case_audit_read import SupportedCaseAudit
from nurse_rcl.v2.opaque_ids import InferenceIdentityManifest
from nurse_rcl.v2.report_composition_proposals import (
    ReportCompositionClaim,
    ReportCompositionResponse,
    ReportRecommendation,
    ReportRecommendationOutcome,
)
from nurse_rcl.v2.report_evidence import _source_types
from nurse_rcl.v2.report_evidence_schemas import ReportEvidencePolicy, ReportEvidenceProjection
from nurse_rcl.v2.report_language_rules import check_report_claim_language
from nurse_rcl.v2.report_mechanical import (
    _guard_report_surfaces,
    _mechanical_local_domains,
    _preflight,
    require_report_composition_mechanics,
)
from nurse_rcl.v2.report_mechanical_schemas import ReportCompositionMechanicalValidation
from nurse_rcl.v2.report_presentation_schemas import (
    ReportClaimPresentationDecision,
    ReportCompositionPresentation,
    ReportLanguageFieldCheck,
    ReportPresentationBlock,
)
from nurse_rcl.v2.report_source_read import SupportedReportContext

_Model = TypeVar("_Model", bound=BaseModel)


def _same_document(left: Any, right: Any) -> bool:
    """Compare detached JSON without serializers or Python's int/float coercion."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(
            _same_document(value, right[name]) for name, value in left.items()
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _same_document(a, b) for a, b in zip(left, right, strict=True)
        )
    if type(left) is float:
        return repr(left) == repr(right)
    return bool(left == right)


def _plain_claim_structure(claim: dict[str, Any]) -> dict[str, Any]:
    result = {name: value for name, value in claim.items() if name != "surface_text"}
    advice = result["recommendation"]
    if advice is not None:
        result["recommendation"] = {
            name: (
                [
                    {key: value for key, value in item.items() if key != "interpretation"}
                    for item in advice["possible_outcomes"]
                ]
                if name == "possible_outcomes"
                else value
            )
            for name, value in advice.items()
            if name != "safety_limitations"
        }
    return result


def _plain_free_text_fields(claim: dict[str, Any]) -> list[tuple[str, str]]:
    fields = [("surface_text", claim["surface_text"])]
    advice = claim["recommendation"]
    if advice is not None:
        fields.extend(
            (f"recommendation.possible_outcomes[{index}].interpretation", item["interpretation"])
            for index, item in enumerate(advice["possible_outcomes"])
        )
        fields.extend(
            (f"recommendation.safety_limitations[{index}]", text)
            for index, text in enumerate(advice["safety_limitations"])
        )
    return fields


def _presentation_local_domains(cls: type[BaseModel], state: dict[str, Any]) -> None:
    """Check only the existing presentation structure before model callbacks."""
    _mechanical_local_domains(cls, state)
    if cls is ReportLanguageFieldCheck:
        reasons = state["reason_codes"]
        if reasons != sorted(set(reasons)):
            raise ValueError("language reasons must be sorted and unique")
    elif cls is ReportPresentationBlock:
        if (state["surface_status"] == "canonical") != (
            state["semantic_status"] == "deterministic_render"
        ):
            raise ValueError("display origin and assurance must agree")
        refs = [fact["fact_ref"] for fact in state["required_context"]]
        if refs != sorted(set(refs), key=lambda ref: int(ref[1:])):
            raise ValueError("display context must be unique in view order")
        if not set(state["claim"]["fact_refs"]) <= set(refs):
            raise ValueError("display context must retain every citation")
        safety = state["required_safety_codes"]
        if safety != sorted(set(safety)):
            raise ValueError("display safety codes must be sorted and unique")
    elif cls is ReportClaimPresentationDecision:
        violation = any(item["reason_codes"] for item in state["field_checks"])
        passed = state["mechanical_status"] == "passed"
        if (state["surface_status"] == "model_unverified") != (passed and not violation):
            raise ValueError("model delivery requires mechanical pass without a known violation")
        if (state["semantic_status"] == "known_violation") != violation:
            raise ValueError("semantic status must retain the original language result")
        replacement = state["replacement"]
        if (replacement is not None) != (passed and violation):
            raise ValueError("only blocked mechanically passed claims receive replacement")
        if replacement is not None and (
            replacement["claim"]["claim_id"] != state["claim_id"]
            or replacement["surface_status"] != "canonical"
        ):
            raise ValueError("replacement must be a canonical block for this claim")
    elif cls is ReportCompositionPresentation:
        mechanical, foundation = state["mechanical_validation"], state["foundation"]
        if (
            foundation["context_sha256"] != mechanical["context_sha256"]
            or foundation["opaque_case_id"] != mechanical["opaque_case_id"]
        ):
            raise ValueError("presentation foundation and mechanical identities differ")
        if len(state["claims"]) != len(mechanical["claims"]):
            raise ValueError("presentation must retain every original claim")
        displayed = iter(state["safe_display_blocks"])
        for decision, result in zip(state["claims"], mechanical["claims"], strict=True):
            proposal = result["proposal"]
            if (
                decision["claim_id"] != proposal["claim_id"]
                or decision["mechanical_status"] != result["mechanical_status"]
            ):
                raise ValueError("presentation decisions differ from original mechanics")
            if [
                (check["field_path"], check["raw_text"]) for check in decision["field_checks"]
            ] != _plain_free_text_fields(proposal):
                raise ValueError("presentation must inspect every original free field in order")
            if result["mechanical_status"] == "rejected":
                continue
            block = next(displayed, None)
            if block is None:
                raise ValueError("display must retain each passed claim or its replacement")
            if decision["replacement"] is not None:
                if not _same_document(block, decision["replacement"]):
                    raise ValueError("display differs from the separate deterministic replacement")
                if not _same_document(
                    _plain_claim_structure(block["claim"]), _plain_claim_structure(proposal)
                ):
                    raise ValueError("replacement must preserve the original structural claim")
            elif block["surface_status"] != "model_unverified" or not _same_document(
                block["claim"], proposal
            ):
                raise ValueError("model display must preserve the exact original proposal")
            if [fact["fact_ref"] for fact in block["required_context"]] != result[
                "required_context_refs"
            ] or block["required_safety_codes"] != result["required_safety_codes"]:
                raise ValueError("display must retain mechanical context and safety disclosures")
        if next(displayed, None) is not None:
            raise ValueError("display contains an extra block")
        delivered = sum(c["surface_status"] == "model_unverified" for c in state["claims"])
        expected = (
            "full"
            if delivered and delivered == len(state["claims"])
            else ("partial" if delivered else "deterministic_fallback")
        )
        if state["status"] != expected:
            raise ValueError("presentation status differs from original model delivery")


def _bytes(value: BaseModel) -> bytes:
    return canonical_json_bytes(value.model_dump(mode="json", warnings="error"))


def _snapshot(value: _Model, original: bytes) -> _Model:
    copied = type(value).model_validate_json(original)
    if not hmac.compare_digest(original, _bytes(copied)):
        raise ValueError("report presentation snapshot changes exact canonical bytes")
    return copied


def _replacement_claim(claim: ReportCompositionClaim) -> ReportCompositionClaim:
    """Construct short program prose; complete cited facts remain in required_context."""
    advice = claim.recommendation
    replacement_advice = None
    if advice is not None:
        explanations = {
            "true": "If true, record support for this predicate; causality remains unestablished.",
            "false": "If false, record contradictory evidence and reassess the hypothesis.",
            "unknown": "If unknown, retain missing evidence and seek an additional bounded check.",
            "improves": "If conditions improve, record this without inferring causality.",
            "no_change": "If there is no change, reassess the conditional proposal.",
            "worsens_or_unknown": "If worse or unknown, stop and request operator reassessment.",
        }
        replacement_advice = ReportRecommendation(
            action_class=advice.action_class,
            precondition_refs=advice.precondition_refs,
            possible_outcomes=tuple(
                ReportRecommendationOutcome(
                    outcome=item.outcome, interpretation=explanations[item.outcome]
                )
                for item in advice.possible_outcomes
            ),
            safety_limitations=(
                "Proposed only; do not execute automatically. "
                "Preserve source preconditions and all outcome branches.",
                "Require the disclosed operator and reversibility safeguards. "
                "No recovery outcome is guaranteed.",
            ),
            status="proposed",
        )
    text = (
        "Proposed conditional action. Read the cited source preconditions, all possible outcomes, "
        "and required safety disclosures below; this action has not been executed."
        if advice is not None
        else "Frozen evidence summary. Read the cited fact values "
        "and required adverse context below; "
        "these references do not establish a causal relationship or independent evidence."
    )
    return claim.model_copy(update={"surface_text": text, "recommendation": replacement_advice})


@_guard_report_surfaces(
    roots=(
        ("validation", ReportCompositionMechanicalValidation),
        ("response", ReportCompositionResponse),
        ("projection", ReportEvidenceProjection),
    ),
    local_domains=_presentation_local_domains,
)
def build_report_composition_presentation(
    *,
    validation: ReportCompositionMechanicalValidation,
    response: ReportCompositionResponse,
    expected_response_sha256: str,
    projection: ReportEvidenceProjection,
    context: SupportedReportContext,
    audit_package: SupportedCaseAudit,
    expected_audit_sha256: str,
    expected_context_sha256: str,
    identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str,
    policy: ReportEvidencePolicy,
    expected_policy_sha256: str,
) -> ReportCompositionPresentation:
    """Replay complete source authority, then deliver checked but unverified model wording."""
    context_type, audit_type = _source_types(context, audit_package)
    roots = (
        (validation, ReportCompositionMechanicalValidation),
        (response, ReportCompositionResponse),
        (projection, ReportEvidenceProjection),
        (context, context_type),
        (audit_package, audit_type),
        (identity_manifest, InferenceIdentityManifest),
        (policy, ReportEvidencePolicy),
    )
    _preflight(
        roots,
        (
            expected_response_sha256,
            expected_audit_sha256,
            expected_context_sha256,
            expected_identity_manifest_sha256,
            expected_policy_sha256,
        ),
    )
    originals = tuple(_bytes(value) for value, _ in roots)
    mechanical_snapshot = _snapshot(validation, originals[0])
    response_snapshot = _snapshot(response, originals[1])
    projection_snapshot = _snapshot(projection, originals[2])
    mechanical = require_report_composition_mechanics(
        mechanical_snapshot,
        response=response_snapshot,
        expected_response_sha256=expected_response_sha256,
        projection=projection_snapshot,
        context=context,
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
        expected_context_sha256=expected_context_sha256,
        identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
    )
    presentation = _present_verified_composition(mechanical, response_snapshot, projection_snapshot)
    _preflight(roots, ())
    if any(
        not hmac.compare_digest(before, _bytes(value))
        for before, (value, _) in zip(originals, roots, strict=True)
    ):
        raise ValueError("report presentation source bytes changed during replay")
    return presentation


def _present_verified_composition(
    mechanical: ReportCompositionMechanicalValidation,
    response_snapshot: ReportCompositionResponse,
    projection_snapshot: ReportEvidenceProjection,
) -> ReportCompositionPresentation:
    """Pure delivery policy; public callers still replay their complete sources."""
    view = projection_snapshot.view
    if view is None:
        raise ValueError("report_evidence_unavailable")
    presentation = ReportCompositionPresentation(
        foundation=projection_snapshot.foundation,
        mechanical_validation=mechanical,
        **_presentation_values(mechanical, response_snapshot, view),
        presentation_sha256="0" * 64,
    )
    digest = sha256_hex(
        canonical_json_bytes(
            presentation.model_dump(mode="json", warnings="error", exclude={"presentation_sha256"})
        )
    )
    return presentation.model_copy(update={"presentation_sha256": digest})


def _presentation_values(
    mechanical: ReportCompositionMechanicalValidation,
    response_snapshot: ReportCompositionResponse,
    view: Any,
) -> dict[str, Any]:
    """Pure common language/claim delivery policy, without any foundation authority."""
    # Complete the raw-language pass before generating any replacement text.
    checks = tuple(
        check_report_claim_language(claim, view=view) for claim in response_snapshot.claims
    )
    decisions = []
    blocks = []
    for result, field_checks in zip(mechanical.claims, checks, strict=True):
        violation = any(check.reason_codes for check in field_checks)
        deliverable = result.mechanical_status == "passed" and not violation
        replacement = None
        if result.mechanical_status == "passed":
            block = ReportPresentationBlock(
                claim=result.proposal if deliverable else _replacement_claim(result.proposal),
                required_context=tuple(
                    fact for fact in view.facts if fact.fact_ref in result.required_context_refs
                ),
                required_safety_codes=result.required_safety_codes,
                surface_status="model_unverified" if deliverable else "canonical",
                semantic_status="not_established" if deliverable else "deterministic_render",
            )
            blocks.append(block)
            if not deliverable:
                replacement = block
        decisions.append(
            ReportClaimPresentationDecision(
                claim_id=result.proposal.claim_id,
                mechanical_status=result.mechanical_status,
                surface_status="model_unverified" if deliverable else "blocked",
                semantic_status="known_violation" if violation else "not_established",
                field_checks=field_checks,
                replacement=replacement,
            )
        )
    delivered = sum(decision.surface_status == "model_unverified" for decision in decisions)
    return dict(
        claims=tuple(decisions),
        safe_display_blocks=tuple(blocks),
        status="full"
        if delivered and delivered == len(decisions)
        else ("partial" if delivered else "deterministic_fallback"),
    )


@_guard_report_surfaces(
    roots=(
        ("presentation", ReportCompositionPresentation),
        ("validation", ReportCompositionMechanicalValidation),
        ("response", ReportCompositionResponse),
        ("projection", ReportEvidenceProjection),
    ),
    local_domains=_presentation_local_domains,
)
def require_report_composition_presentation(
    presentation: ReportCompositionPresentation,
    *,
    validation: ReportCompositionMechanicalValidation,
    response: ReportCompositionResponse,
    expected_response_sha256: str,
    projection: ReportEvidenceProjection,
    context: SupportedReportContext,
    audit_package: SupportedCaseAudit,
    expected_audit_sha256: str,
    expected_context_sha256: str,
    identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str,
    policy: ReportEvidencePolicy,
    expected_policy_sha256: str,
) -> ReportCompositionPresentation:
    """Require original exact integrity and equality with a full independent source replay."""
    context_type, audit_type = _source_types(context, audit_package)
    roots = (
        (presentation, ReportCompositionPresentation),
        (validation, ReportCompositionMechanicalValidation),
        (response, ReportCompositionResponse),
        (projection, ReportEvidenceProjection),
        (context, context_type),
        (audit_package, audit_type),
        (identity_manifest, InferenceIdentityManifest),
        (policy, ReportEvidencePolicy),
    )
    _preflight(
        roots,
        (
            expected_response_sha256,
            expected_audit_sha256,
            expected_context_sha256,
            expected_identity_manifest_sha256,
            expected_policy_sha256,
        ),
    )
    originals = tuple(_bytes(value) for value, _ in roots)
    if not presentation.verify():
        raise ValueError("report presentation failed original integrity verification")
    rebuilt = build_report_composition_presentation(
        validation=validation,
        response=response,
        expected_response_sha256=expected_response_sha256,
        projection=projection,
        context=context,
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
        expected_context_sha256=expected_context_sha256,
        identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
    )
    if not hmac.compare_digest(originals[0], _bytes(rebuilt)):
        raise ValueError("report presentation differs from independently rebuilt delivery")
    _preflight(roots, ())
    if any(
        not hmac.compare_digest(before, _bytes(value))
        for before, (value, _) in zip(originals, roots, strict=True)
    ):
        raise ValueError("report presentation input bytes changed during replay")
    return rebuilt
