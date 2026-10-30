"""Independently pinned structural report validation, without prose or display authority."""

from __future__ import annotations

import hmac
import re
from collections.abc import Callable
from functools import wraps
from inspect import signature
from typing import Any, ParamSpec, TypeAlias, TypeVar

from pydantic import BaseModel

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.case_audit_read import SupportedCaseAudit
from nurse_rcl.v2.deterministic_report import _report_input_snapshot
from nurse_rcl.v2.opaque_ids import InferenceIdentityManifest
from nurse_rcl.v2.report_composition_proposals import (
    ReportCompositionClaim,
    ReportCompositionResponse,
    ReportProposalInterval,
    ReportRecommendation,
    ReportRecommendationOutcome,
    _nonblank,
    _unique_strings,
    parse_report_composition_response,
)
from nurse_rcl.v2.report_context_v4 import _report_unchanged
from nurse_rcl.v2.report_evidence import (
    _evidence_local_domains,
    _source_types,
    require_report_evidence_projection,
)
from nurse_rcl.v2.report_evidence_schemas import ReportEvidencePolicy, ReportEvidenceProjection
from nurse_rcl.v2.report_mechanical_checks import _validate_claim
from nurse_rcl.v2.report_mechanical_schemas import (
    MECHANICAL_STAGES,
    ReportClaimMechanicalValidation,
    ReportCompositionMechanicalValidation,
    ReportMechanicalCheck,
    _ordered_aliases,
)
from nurse_rcl.v2.report_source_read import SupportedReportContext, _preflight_report_sources
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")
_SurfaceDomains: TypeAlias = Callable[[type[BaseModel], dict[str, Any]], None]
_INPUT_ERRORS = (AttributeError, KeyError, RecursionError, TypeError, ValueError)


def _mechanical_local_domains(cls: type[BaseModel], state: dict[str, Any]) -> None:
    """Existing proposal and receipt structure over detached exact fields only."""
    _evidence_local_domains(cls, state)
    if cls is ReportProposalInterval:
        if state["start_unix_ns"] >= state["end_unix_ns"]:
            raise ValueError("invalid_interval")
    elif cls is ReportRecommendationOutcome:
        _nonblank(state["interpretation"])
    elif cls is ReportRecommendation:
        _unique_strings(tuple(state["precondition_refs"]))
        _unique_strings(tuple(item["outcome"] for item in state["possible_outcomes"]))
        for text in state["safety_limitations"]:
            _nonblank(text)
    elif cls is ReportCompositionClaim:
        _unique_strings(tuple(state["fact_refs"]))
        _unique_strings(tuple(state["target_service_keys"]))
        for text in (*state["target_service_keys"], state["surface_text"]):
            _nonblank(text)
        keys = {(item["fact_ref"], item["field"]) for item in state["assertions"]}
        if len(keys) != len(state["assertions"]):
            raise ValueError("duplicate_assertion")
    elif cls is ReportCompositionResponse:
        _unique_strings(tuple(claim["claim_id"] for claim in state["claims"]))
        if sum(claim["recommendation"] is not None for claim in state["claims"]) > 6:
            raise ValueError("recommendation_limit")
    elif cls is ReportMechanicalCheck:
        reasons = state["reason_codes"]
        if reasons != sorted(set(reasons)):
            raise ValueError("mechanical reasons must be sorted and unique")
        if (state["status"] == "rejected") != bool(reasons):
            raise ValueError("only rejected checks must contain reasons")
    elif cls is ReportClaimMechanicalValidation:
        if tuple(check["stage"] for check in state["checks"]) != MECHANICAL_STAGES:
            raise ValueError("mechanical checks must have the fixed ordered stages")
        rejected = False
        for check in state["checks"]:
            if rejected:
                if check["status"] != "skipped":
                    raise ValueError("checks after a rejection must be skipped")
            elif check["status"] == "skipped":
                raise ValueError("checks before rejection cannot be skipped")
            elif check["status"] == "rejected":
                rejected = True
        if state["mechanical_status"] != ("rejected" if rejected else "passed"):
            raise ValueError("claim status differs from its checks")
        _ordered_aliases(tuple(state["required_context_refs"]))
        _ordered_aliases(tuple(state["uncited_context_refs"]))
        cited = state["proposal"]["fact_refs"]
        if state["uncited_context_refs"] != [
            ref for ref in state["required_context_refs"] if ref not in cited
        ]:
            raise ValueError("uncited context must be required context minus proposal citations")
        safety = state["required_safety_codes"]
        if safety != sorted(set(safety)):
            raise ValueError("mechanical safety codes must be sorted and unique")
        seen: set[str] = set()
        firsts: list[str] = []
        for group in state["shared_provenance_groups"]:
            _ordered_aliases(tuple(group))
            if len(group) < 2 or seen.intersection(group):
                raise ValueError("shared provenance groups must be disjoint nontrivial groups")
            if not set(group) <= set(cited):
                raise ValueError("shared provenance groups must describe cited references")
            seen.update(group)
            firsts.append(group[0])
        _ordered_aliases(tuple(firsts))
    elif cls is ReportCompositionMechanicalValidation:
        _unique_strings(tuple(claim["proposal"]["claim_id"] for claim in state["claims"]))
        _ordered_aliases(tuple(state["mandatory_fact_refs"]))
        for status, field in (("passed", "passed_count"), ("rejected", "rejected_count")):
            if state[field] != sum(
                claim["mechanical_status"] == status for claim in state["claims"]
            ):
                raise ValueError(f"mechanical {status} count differs from claims")


def _preflight(roots: tuple[tuple[object, type[BaseModel]], ...], pins: tuple[str, ...]) -> None:
    for value, expected_type in roots:
        if type(value) is not expected_type:
            raise ValueError(f"report mechanics requires exact {expected_type.__name__} type")
    for pin in pins:
        if type(pin) is not str or re.fullmatch(r"[0-9a-f]{64}", pin) is None:
            raise ValueError("report mechanics pins require exact lowercase SHA256 strings")
    # Finish every input walk before accessing any input property or serializer.
    for value, expected_type in roots:
        preflight_exact_pydantic_tree(
            value,
            root_model_types=(expected_type,),
            label="report mechanics input",
            max_string_length=None,
        )


def _guard_report_surfaces(
    *,
    roots: tuple[tuple[str, type[BaseModel]], ...],
    local_domains: _SurfaceDomains = _mechanical_local_domains,
) -> Callable[[Callable[_Parameters, _Result]], Callable[_Parameters, _Result]]:
    """Inspect all original inputs before callbacks and retain them on every exit."""

    def decorate(function: Callable[_Parameters, _Result]) -> Callable[_Parameters, _Result]:
        call_signature = signature(function)

        @wraps(function)
        def guarded(*args: _Parameters.args, **kwargs: _Parameters.kwargs) -> _Result:
            bound = call_signature.bind(*args, **kwargs)
            values = bound.arguments
            context, package = values["context"], values["audit_package"]
            context_type, audit_type = _source_types(context, package)
            additional: tuple[tuple[object, type[BaseModel]], ...] = (
                (values["identity_manifest"], InferenceIdentityManifest),
                (values["policy"], ReportEvidencePolicy),
                *((values[name], cls) for name, cls in roots),
            )
            _preflight(
                ((context, context_type), (package, audit_type), *additional),
                tuple(
                    values[name]
                    for name in (
                        "expected_response_sha256",
                        "expected_audit_sha256",
                        "expected_context_sha256",
                        "expected_identity_manifest_sha256",
                        "expected_policy_sha256",
                    )
                ),
            )
            context_original, audit_original, _ = _preflight_report_sources(
                context=context,
                audit_package=package,
                expected_audit_sha256=values["expected_audit_sha256"],
                expected_context_sha256=values["expected_context_sha256"],
            )
            # Report roots retain their original node/depth limits without the
            # sources' 64 MiB JSON budget or a new report/string byte limit.
            originals = tuple(
                _report_input_snapshot(
                    value, model_type=cls, max_nodes=500_000, local_domains=local_domains
                )
                for value, cls in additional
            )
            try:
                return function(*args, **kwargs)
            finally:
                unchanged = True
                for source, original in (
                    (context, context_original),
                    (package, audit_original),
                ):
                    try:
                        _report_unchanged(source, original)
                    except _INPUT_ERRORS:
                        unchanged = False
                for (value, cls), original in zip(additional, originals, strict=True):
                    try:
                        if (
                            _report_input_snapshot(
                                value,
                                model_type=cls,
                                max_nodes=500_000,
                                local_domains=local_domains,
                            )
                            != original
                        ):
                            unchanged = False
                    except _INPUT_ERRORS:
                        unchanged = False
                if not unchanged:
                    raise ValueError(
                        "report surface source bytes changed or original representation "
                        "changed during replay"
                    )

        return guarded

    return decorate


@_guard_report_surfaces(
    roots=(("response", ReportCompositionResponse), ("projection", ReportEvidenceProjection))
)
def validate_report_composition_mechanics(
    *,
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
) -> ReportCompositionMechanicalValidation:
    """Replay all source pins and check proposals, retaining every unchecked surface string.

    The response pin hashes canonical typed response bytes, not a transport's raw wire.
    A passing result is only a private intermediate mechanical validation product.
    """
    original = canonical_json_bytes(response.model_dump(mode="json", warnings="error"))
    snapshot = parse_report_composition_response(original)
    snapshot_bytes = canonical_json_bytes(snapshot.model_dump(mode="json", warnings="error"))
    if not hmac.compare_digest(original, snapshot_bytes):
        raise ValueError("report response snapshot changes exact canonical bytes")
    if not hmac.compare_digest(sha256_hex(snapshot_bytes), expected_response_sha256):
        raise ValueError("report response digest differs from its external pin")
    replayed = require_report_evidence_projection(
        projection,
        context=context,
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
        expected_context_sha256=expected_context_sha256,
        identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
    )
    return _validate_verified_composition(snapshot, replayed, expected_response_sha256)


def _validate_verified_composition(
    snapshot: ReportCompositionResponse, replayed: ReportEvidenceProjection,
    expected_response_sha256: str,
) -> ReportCompositionMechanicalValidation:
    """Pure claim checks; only an enclosing source-backed call may consume this."""
    view = replayed.view
    if view is None:
        raise ValueError("report_evidence_unavailable")
    claims = tuple(
        _validate_claim(
            claim, view=view, bindings=replayed.source_bindings, foundation=replayed.foundation
        )
        for claim in snapshot.claims
    )
    result = ReportCompositionMechanicalValidation(
        opaque_case_id=view.opaque_case_id,
        context_sha256=view.context_sha256,
        projection_sha256=replayed.projection_sha256,
        view_sha256=view.view_sha256,
        policy_sha256=view.policy_sha256,
        response_sha256=expected_response_sha256,
        mandatory_fact_refs=view.mandatory_fact_refs,
        claims=claims,
        passed_count=sum(c.mechanical_status == "passed" for c in claims),
        rejected_count=sum(c.mechanical_status == "rejected" for c in claims),
        validation_sha256="0" * 64,
    )
    digest = sha256_hex(
        canonical_json_bytes(
            result.model_dump(mode="json", warnings="error", exclude={"validation_sha256"})
        )
    )
    return result.model_copy(update={"validation_sha256": digest})


@_guard_report_surfaces(
    roots=(
        ("validation", ReportCompositionMechanicalValidation),
        ("response", ReportCompositionResponse),
        ("projection", ReportEvidenceProjection),
    )
)
def require_report_composition_mechanics(
    validation: ReportCompositionMechanicalValidation,
    *,
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
) -> ReportCompositionMechanicalValidation:
    """Verify original integrity, then compare a fresh full independently pinned replay."""
    if not validation.verify():
        raise ValueError("report mechanics failed original integrity verification")
    provided = canonical_json_bytes(validation.model_dump(mode="json", warnings="error"))
    rebuilt = validate_report_composition_mechanics(
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
    rebuilt_bytes = canonical_json_bytes(rebuilt.model_dump(mode="json", warnings="error"))
    if not hmac.compare_digest(provided, rebuilt_bytes):
        raise ValueError("report mechanics differs from independently rebuilt validation")
    return rebuilt
