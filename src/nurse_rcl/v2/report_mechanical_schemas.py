"""Private report mechanical results; integrity is never semantic or display authority."""

from __future__ import annotations

import hmac
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.report_composition_proposals import ReportCompositionClaim
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

REPORT_COMPOSITION_MECHANICAL_SCHEMA_VERSION: Literal[
    "nurse.v2.report-composition-mechanical-validation/1"
] = "nurse.v2.report-composition-mechanical-validation/1"
REPORT_COMPOSITION_MECHANICAL_RULE_VERSION: Literal[
    "nurse-v2-report-composition-mechanical-rules/1"
] = "nurse-v2-report-composition-mechanical-rules/1"

MechanicalStage: TypeAlias = Literal[
    "references", "scope", "operation", "assertions", "recommendation", "disclosures"
]
MECHANICAL_STAGES: tuple[MechanicalStage, ...] = (
    "references",
    "scope",
    "operation",
    "assertions",
    "recommendation",
    "disclosures",
)
MechanicalReason: TypeAlias = Literal[
    "unknown_fact_reference",
    "hidden_fact_reference",
    "target_outside_top_k",
    "target_scope_mismatch",
    "interval_required",
    "interval_forbidden",
    "interval_source_mismatch",
    "interval_scope_ambiguous",
    "operation_type_mismatch",
    "operation_strength_mismatch",
    "operation_target_count",
    "invalid_primary_fact",
    "missing_operation_basis",
    "recommendation_shape_mismatch",
    "assertion_field_missing",
    "assertion_value_mismatch",
    "recommendation_status_invalid",
    "recommendation_action_class_mismatch",
    "recommendation_outcomes_mismatch",
    "recommendation_precondition_invalid",
    "mitigation_source_mismatch",
    "mitigation_preconditions_missing",
    "mitigation_precondition_not_true",
]
MechanicalSafetyCode: TypeAlias = Literal[
    "proposed_only",
    "no_automatic_execution",
    "sealed_ranking_immutable",
    "outcomes_are_conditional",
    "read_only",
    "operator_approval_required",
    "reversible_action_only",
    "distinct_references_do_not_imply_independent_evidence",
]
_Alias: TypeAlias = Annotated[str, Field(pattern=r"^f[1-9][0-9]{0,5}$")]
_Digest: TypeAlias = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def _ordered_aliases(refs: tuple[str, ...]) -> None:
    # Sent views assign sequential numeric aliases. Lexical sorting puts f10 before f2.
    if refs != tuple(sorted(set(refs), key=lambda ref: int(ref[1:]))):
        raise ValueError("mechanical aliases must be unique in sent-view order")


class _MechanicalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class ReportMechanicalCheck(_MechanicalModel):
    stage: MechanicalStage
    status: Literal["passed", "rejected", "skipped"]
    reason_codes: tuple[MechanicalReason, ...] = ()

    @model_validator(mode="after")
    def consistent_reasons(self) -> Self:
        if self.reason_codes != tuple(sorted(set(self.reason_codes))):
            raise ValueError("mechanical reasons must be sorted and unique")
        if (self.status == "rejected") != bool(self.reason_codes):
            raise ValueError("only rejected checks must contain reasons")
        return self


class ReportClaimMechanicalValidation(_MechanicalModel):
    proposal: ReportCompositionClaim
    checks: tuple[ReportMechanicalCheck, ...] = Field(min_length=6, max_length=6)
    mechanical_status: Literal["passed", "rejected"]
    required_context_refs: tuple[_Alias, ...]
    uncited_context_refs: tuple[_Alias, ...]
    required_safety_codes: tuple[MechanicalSafetyCode, ...]
    shared_provenance_groups: tuple[tuple[_Alias, ...], ...]
    surface_check_performed: Literal[False] = False
    display_authorized: Literal[False] = False
    semantic_status: Literal["not_established"] = "not_established"

    @model_validator(mode="after")
    def consistent_result(self) -> Self:
        if tuple(check.stage for check in self.checks) != MECHANICAL_STAGES:
            raise ValueError("mechanical checks must have the fixed ordered stages")
        rejected = False
        for check in self.checks:
            if rejected:
                if check.status != "skipped":
                    raise ValueError("checks after a rejection must be skipped")
            elif check.status == "skipped":
                raise ValueError("checks before rejection cannot be skipped")
            elif check.status == "rejected":
                rejected = True
        if self.mechanical_status != ("rejected" if rejected else "passed"):
            raise ValueError("claim status differs from its checks")
        _ordered_aliases(self.required_context_refs)
        _ordered_aliases(self.uncited_context_refs)
        if self.uncited_context_refs != tuple(
            ref for ref in self.required_context_refs if ref not in self.proposal.fact_refs
        ):
            raise ValueError("uncited context must be required context minus proposal citations")
        if self.required_safety_codes != tuple(sorted(set(self.required_safety_codes))):
            raise ValueError("mechanical safety codes must be sorted and unique")
        seen: set[str] = set()
        firsts: list[str] = []
        for group in self.shared_provenance_groups:
            _ordered_aliases(group)
            if len(group) < 2 or seen.intersection(group):
                raise ValueError("shared provenance groups must be disjoint nontrivial groups")
            if not set(group) <= set(self.proposal.fact_refs):
                raise ValueError("shared provenance groups must describe cited references")
            seen.update(group)
            firsts.append(group[0])
        _ordered_aliases(tuple(firsts))
        return self


class ReportCompositionMechanicalValidation(_MechanicalModel):
    schema_version: Literal["nurse.v2.report-composition-mechanical-validation/1"] = (
        REPORT_COMPOSITION_MECHANICAL_SCHEMA_VERSION
    )
    rule_version: Literal["nurse-v2-report-composition-mechanical-rules/1"] = (
        REPORT_COMPOSITION_MECHANICAL_RULE_VERSION
    )
    opaque_case_id: Annotated[str, Field(pattern=r"^opq_[0-9a-f]{32}$")]
    context_sha256: _Digest
    projection_sha256: _Digest
    view_sha256: _Digest
    policy_sha256: _Digest
    response_sha256: _Digest
    mandatory_fact_refs: tuple[_Alias, ...] = Field(min_length=1)
    claims: tuple[ReportClaimMechanicalValidation, ...] = Field(max_length=24)
    passed_count: int = Field(ge=0, le=24)
    rejected_count: int = Field(ge=0, le=24)
    surface_check_performed: Literal[False] = False
    display_authorized: Literal[False] = False
    semantic_status: Literal["not_established"] = "not_established"
    validation_sha256: _Digest

    @model_validator(mode="after")
    def consistent_composition(self) -> Self:
        ids = tuple(claim.proposal.claim_id for claim in self.claims)
        if len(ids) != len(set(ids)):
            raise ValueError("mechanical claim ids must be unique")
        _ordered_aliases(self.mandatory_fact_refs)
        if self.passed_count != sum(c.mechanical_status == "passed" for c in self.claims):
            raise ValueError("mechanical passed count differs from claims")
        if self.rejected_count != sum(c.mechanical_status == "rejected" for c in self.claims):
            raise ValueError("mechanical rejected count differs from claims")
        return self

    def verify(self) -> bool:
        """Verify exact canonical integrity only; self-signing never supplies authority."""
        try:
            if type(self) is not ReportCompositionMechanicalValidation:
                return False
            preflight_exact_pydantic_tree(
                self,
                root_model_types=(ReportCompositionMechanicalValidation,),
                label="report mechanical validation",
                max_string_length=None,
            )
            serialized = self.model_dump(mode="json", warnings="error")
            original = canonical_json_bytes(serialized)
            rebuilt = ReportCompositionMechanicalValidation.model_validate_json(original)
            rebuilt_bytes = canonical_json_bytes(rebuilt.model_dump(mode="json", warnings="error"))
            unsigned = {
                key: value for key, value in serialized.items() if key != "validation_sha256"
            }
            return hmac.compare_digest(original, rebuilt_bytes) and hmac.compare_digest(
                serialized["validation_sha256"], sha256_hex(canonical_json_bytes(unsigned))
            )
        except (AttributeError, RecursionError, TypeError, ValueError):
            return False
