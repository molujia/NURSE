"""Private report explainability accounting from source-pinned replay."""

from __future__ import annotations

import hmac
from typing import Any

from pydantic import BaseModel

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.report_explainability_schemas import (
    ReportClaimPopulation,
    ReportCompositionExplainability,
    _bytes,
    _preflight_explainability,
)
from nurse_rcl.v2.report_prompt import ReportSources, _preflight_inputs
from nurse_rcl.v2.report_run_schemas import ReportRunAudit, ReportRunPolicy, _preflight_audit
from nurse_rcl.v2.report_runner import (
    _guard_report_runtime,
    _runtime_local_domains,
    replay_report_composition,
)
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree


def _population_selection(parent: dict[str, Any], child: dict[str, Any], ids: set[str]) -> None:
    if not ids <= set(parent["claim_ids"]) or child["claim_ids"] != [
        identifier for identifier in parent["claim_ids"] if identifier in ids
    ]:
        raise ValueError("report populations must preserve original order and set membership")


def _population_partition(
    parent: dict[str, Any], first: dict[str, Any], second: dict[str, Any]
) -> None:
    first_ids, second_ids = set(first["claim_ids"]), set(second["claim_ids"])
    if first_ids & second_ids or first_ids | second_ids != set(parent["claim_ids"]):
        raise ValueError("report populations must form a disjoint complete partition")
    _population_selection(parent, first, first_ids)
    _population_selection(parent, second, second_ids)


def _explainability_local_domains(cls: type[BaseModel], state: dict[str, Any]) -> None:
    """Original accounting structure over detached data, without integrity callbacks."""
    _runtime_local_domains(cls, state)
    if cls is ReportClaimPopulation:
        ids, count = state["claim_ids"], state["count"]
        if count != len(ids) or len(set(ids)) != count:
            raise ValueError("report claim population requires unique IDs and matching count")
    elif cls is ReportCompositionExplainability:
        generated, unchecked, checked = (
            state["generated_claims"],
            state["unchecked_claims"],
            state["checked_claims"],
        )
        if state["generation_status"] == "unknown":
            if generated is not None or unchecked is not None or checked["count"]:
                raise ValueError("unknown generation cannot invent populations or checks")
        elif generated is None or unchecked is None:
            raise ValueError("known generation requires generated and unchecked populations")
        else:
            _population_partition(generated, checked, unchecked)
        if state["generation_status"] == "not_requested" and (
            generated is None or generated["count"] or checked["count"]
        ):
            raise ValueError("not requested generation must be known empty")
        if state["logical_requests"] != int(state["generation_status"] != "not_requested"):
            raise ValueError("logical request count describes this single run only")
        passed, rejected = (
            state["mechanically_passed_claims"],
            state["mechanically_rejected_claims"],
        )
        violation, unestablished = (
            state["language_violation_claims"],
            state["semantic_not_established_claims"],
        )
        _population_partition(checked, passed, rejected)
        _population_partition(checked, violation, unestablished)
        blocked = set(rejected["claim_ids"]) | set(violation["claim_ids"])
        _population_selection(checked, state["blocked_original_claims"], blocked)
        _population_selection(
            checked, state["displayed_model_claims"], set(checked["claim_ids"]) - blocked
        )
        _population_selection(
            checked,
            state["replacement_claims"],
            set(passed["claim_ids"]) & set(violation["claim_ids"]),
        )
        if not checked["count"] <= state["checked_free_text_fields"] <= 8 * checked["count"]:
            raise ValueError("checked free text counts fields of checked originals only")
        if (
            not violation["count"]
            <= state["violating_free_text_fields"]
            <= min(
                8 * violation["count"], state["checked_free_text_fields"] - unestablished["count"]
            )
        ):
            raise ValueError("violating free text counts distinct violating original fields")
        mechanical, presentation = (
            state["source_mechanical_sha256"],
            state["source_presentation_sha256"],
        )
        if (mechanical is None) != (presentation is None):
            raise ValueError("mechanical and presentation source pins must coexist")
        if mechanical is None and checked["count"]:
            raise ValueError("checked originals require mechanical and presentation source pins")
        if state["generation_status"] != "parsed" and mechanical is not None:
            raise ValueError("unparsed generation cannot contain check source pins")
        displayed = state["displayed_model_claims"]["count"]
        if state["displayed_blocks"] != displayed + state["replacement_claims"]["count"]:
            raise ValueError("displayed blocks count model and replacement blocks only")
        expected_status = (
            "full"
            if displayed and generated is not None and displayed == generated["count"]
            else "partial"
            if displayed
            else "deterministic_fallback"
        )
        if state["report_status"] != expected_status:
            raise ValueError("report status derives from original model delivery, not replacements")


def _population(ids: tuple[str, ...]) -> ReportClaimPopulation:
    return ReportClaimPopulation(claim_ids=ids, count=len(ids))


def _project_replayed_audit(audit: ReportRunAudit) -> ReportCompositionExplainability:
    """Consume only a privately held result of complete, independently pinned replay."""
    response, mechanical, presentation = (
        audit.parsed_response,
        audit.mechanical_validation,
        audit.presentation,
    )
    generated = tuple(c.claim_id for c in response.claims) if response is not None else ()
    decisions = presentation.claims if presentation is not None else ()
    checked = tuple(c.claim_id for c in decisions)
    unknown = audit.exchange_attempted and response is None
    foundation = audit.projection.foundation
    artifact = ReportCompositionExplainability(
        opaque_case_id=audit.opaque_case_id,
        source_run_audit_sha256=audit.audit_sha256,
        source_context_sha256=audit.expected_context_sha256,
        source_top_k_sha256=foundation.source_top_k_sha256,
        source_policy_sha256=audit.expected_policy_sha256,
        source_foundation_sha256=foundation.report_sha256,
        source_mechanical_sha256=mechanical.validation_sha256 if mechanical is not None else None,
        source_presentation_sha256=(
            presentation.presentation_sha256 if presentation is not None else None
        ),
        report_status=audit.status,
        reason_code=audit.reason_code,
        logical_requests=int(audit.exchange_attempted),
        generation_status=(
            "unknown" if unknown else "parsed" if response is not None else "not_requested"
        ),
        generated_claims=None if unknown else _population(generated),
        unchecked_claims=None
        if unknown
        else _population(tuple(i for i in generated if i not in checked)),
        checked_claims=_population(checked),
        mechanically_passed_claims=_population(
            tuple(c.claim_id for c in decisions if c.mechanical_status == "passed")
        ),
        mechanically_rejected_claims=_population(
            tuple(c.claim_id for c in decisions if c.mechanical_status == "rejected")
        ),
        language_violation_claims=_population(
            tuple(c.claim_id for c in decisions if c.semantic_status == "known_violation")
        ),
        semantic_not_established_claims=_population(
            tuple(c.claim_id for c in decisions if c.semantic_status == "not_established")
        ),
        blocked_original_claims=_population(
            tuple(c.claim_id for c in decisions if c.surface_status == "blocked")
        ),
        displayed_model_claims=_population(
            tuple(c.claim_id for c in decisions if c.surface_status == "model_unverified")
        ),
        replacement_claims=_population(
            tuple(c.claim_id for c in decisions if c.replacement is not None)
        ),
        checked_free_text_fields=sum(len(c.field_checks) for c in decisions),
        violating_free_text_fields=sum(
            bool(f.reason_codes) for c in decisions for f in c.field_checks
        ),
        displayed_blocks=len(presentation.safe_display_blocks) if presentation is not None else 0,
        artifact_sha256="0" * 64,
    )
    digest = sha256_hex(
        canonical_json_bytes(
            artifact.model_dump(mode="json", warnings="error", exclude={"artifact_sha256"})
        )
    )
    return artifact.model_copy(update={"artifact_sha256": digest})


@_guard_report_runtime(
    roots=(("audit", ReportRunAudit), ("policy", ReportRunPolicy)),
    pins=("expected_audit_sha256", "expected_policy_sha256"),
)
def build_report_composition_explainability(
    *,
    audit: ReportRunAudit,
    expected_audit_sha256: str,
    sources: ReportSources,
    policy: ReportRunPolicy,
    expected_policy_sha256: str,
) -> ReportCompositionExplainability:
    """Replay source authority and original decisions offline before counting anything."""
    replayed = replay_report_composition(
        audit=audit,
        expected_audit_sha256=expected_audit_sha256,
        sources=sources,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
    )
    return _project_replayed_audit(replayed)


@_guard_report_runtime(
    roots=(
        ("artifact", ReportCompositionExplainability),
        ("audit", ReportRunAudit),
        ("policy", ReportRunPolicy),
    ),
    pins=("expected_artifact_sha256", "expected_audit_sha256", "expected_policy_sha256"),
    local_domains=_explainability_local_domains,
    root_preflights=(("artifact", _preflight_explainability),),
)
def require_report_composition_explainability(
    *,
    artifact: ReportCompositionExplainability,
    expected_artifact_sha256: str,
    audit: ReportRunAudit,
    expected_audit_sha256: str,
    sources: ReportSources,
    policy: ReportRunPolicy,
    expected_policy_sha256: str,
) -> ReportCompositionExplainability:
    """Check the independent artifact pin, then compare complete source-rebuilt bytes."""
    _preflight_explainability(artifact)
    _preflight_inputs(
        sources,
        ((audit, ReportRunAudit), (policy, ReportRunPolicy)),
        (expected_artifact_sha256, expected_audit_sha256, expected_policy_sha256),
    )
    _preflight_audit(audit)
    preflight_exact_pydantic_tree(
        policy, root_model_types=(ReportRunPolicy,), label="report run policy"
    )
    if (
        not hmac.compare_digest(artifact.artifact_sha256, expected_artifact_sha256)
        or not artifact.verify()
    ):
        raise ValueError("report explainability fails exact integrity or independent artifact pin")
    original = _bytes(artifact)
    rebuilt = build_report_composition_explainability(
        audit=audit,
        expected_audit_sha256=expected_audit_sha256,
        sources=sources,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
    )
    _preflight_explainability(artifact)
    if not hmac.compare_digest(original, _bytes(artifact)):
        raise ValueError("report explainability original bytes changed during replay")
    if not hmac.compare_digest(original, _bytes(rebuilt)):
        raise ValueError("report explainability differs from independently source-rebuilt replay")
    return rebuilt
