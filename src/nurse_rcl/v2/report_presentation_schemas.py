"""Private presentation receipts: exact integrity alone never supplies source authority."""

from __future__ import annotations

import hmac
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.deterministic_report import DeterministicSREReport
from nurse_rcl.v2.report_composition_proposals import ReportCompositionClaim
from nurse_rcl.v2.report_evidence_schemas import ReportEvidenceFact
from nurse_rcl.v2.report_mechanical_schemas import (
    MechanicalSafetyCode,
    ReportCompositionMechanicalValidation,
)
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

REPORT_COMPOSITION_PRESENTATION_SCHEMA_VERSION: Literal[
    "nurse.v2.report-composition-presentation/1"
] = "nurse.v2.report-composition-presentation/1"
REPORT_LANGUAGE_RULE_VERSION: Literal["nurse-v2-report-language-rules/2"] = (
    "nurse-v2-report-language-rules/2"
)
LanguageReason: TypeAlias = Literal[
    "causal_overclaim",
    "score_as_probability",
    "tie_break_as_evidence",
    "missing_evidence_upgraded",
    "unsupported_health_inference",
    "recommendation_executed",
    "recommendation_validated",
    "recommendation_guarantee",
    "prompt_injection",
    "active_content",
    "executable_content",
    "unknown_fact_alias",
    "uncited_fact_alias",
    "source_field_missing",
    "source_value_mismatch",
    "source_unit_mismatch",
    "candidate_value_mismatch",
    "fact_target_mismatch",
    "relation_direction_mismatch",
    "unsupported_relation",
]


def claim_free_text_fields(claim: ReportCompositionClaim) -> tuple[tuple[str, str], ...]:
    """Enumerate every free prose field in fixed proposal order, without rewriting it."""
    fields = [("surface_text", claim.surface_text)]
    if claim.recommendation is not None:
        question = getattr(claim.recommendation, "verification_question", None)
        if question is not None:
            fields.append(("recommendation.verification_question", question))
        fields.extend(
            (f"recommendation.possible_outcomes[{index}].interpretation", item.interpretation)
            for index, item in enumerate(claim.recommendation.possible_outcomes)
        )
        fields.extend(
            (f"recommendation.safety_limitations[{index}]", text)
            for index, text in enumerate(claim.recommendation.safety_limitations)
        )
    return tuple(fields)


class _PresentationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class ReportLanguageFieldCheck(_PresentationModel):
    field_path: str = Field(min_length=1)
    raw_text: str = Field(min_length=1, max_length=512)
    reason_codes: tuple[LanguageReason, ...]

    @model_validator(mode="after")
    def unique_reasons(self) -> Self:
        if self.reason_codes != tuple(sorted(set(self.reason_codes))):
            raise ValueError("language reasons must be sorted and unique")
        return self


class ReportPresentationBlock(_PresentationModel):
    """Only safe model text or program-generated replacement text belongs in this block."""

    claim: ReportCompositionClaim
    required_context: tuple[ReportEvidenceFact, ...]
    required_safety_codes: tuple[MechanicalSafetyCode, ...]
    surface_status: Literal["model_unverified", "canonical"]
    semantic_status: Literal["not_established", "deterministic_render"]

    @model_validator(mode="after")
    def consistent_origin(self) -> Self:
        if (self.surface_status == "canonical") != (self.semantic_status == "deterministic_render"):
            raise ValueError("display origin and assurance must agree")
        refs = tuple(f.fact_ref for f in self.required_context)
        if refs != tuple(sorted(set(refs), key=lambda ref: int(ref[1:]))):
            raise ValueError("display context must be unique in view order")
        if not set(self.claim.fact_refs) <= set(refs):
            raise ValueError("display context must retain every citation")
        if self.required_safety_codes != tuple(sorted(set(self.required_safety_codes))):
            raise ValueError("display safety codes must be sorted and unique")
        return self


class ReportClaimPresentationDecision(_PresentationModel):
    claim_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    mechanical_status: Literal["passed", "rejected"]
    surface_status: Literal["model_unverified", "blocked"]
    semantic_status: Literal["not_established", "known_violation"]
    field_checks: tuple[ReportLanguageFieldCheck, ...] = Field(min_length=1, max_length=8)
    replacement: ReportPresentationBlock | None = None

    @model_validator(mode="after")
    def consistent_decision(self) -> Self:
        violation = any(item.reason_codes for item in self.field_checks)
        deliverable = self.mechanical_status == "passed" and not violation
        if (self.surface_status == "model_unverified") != deliverable:
            raise ValueError("model delivery requires mechanical pass without a known violation")
        if (self.semantic_status == "known_violation") != violation:
            raise ValueError("semantic status must retain the original language result")
        needs_replacement = self.mechanical_status == "passed" and violation
        if (self.replacement is not None) != needs_replacement:
            raise ValueError("only blocked mechanically passed claims receive replacement")
        if self.replacement is not None and (
            self.replacement.claim.claim_id != self.claim_id
            or self.replacement.surface_status != "canonical"
        ):
            raise ValueError("replacement must be a canonical block for this claim")
        return self


def _bytes(value: BaseModel) -> bytes:
    return canonical_json_bytes(value.model_dump(mode="json", warnings="error"))


def _claim_structure(claim: ReportCompositionClaim) -> bytes:
    shape = claim.model_dump(mode="json", warnings="error", exclude={"surface_text"})
    advice = shape["recommendation"]
    if advice is not None:
        del advice["safety_limitations"]
        for outcome in advice["possible_outcomes"]:
            del outcome["interpretation"]
    return canonical_json_bytes(shape)


class ReportCompositionPresentation(_PresentationModel):
    """Private receipt retains original mechanics, decisions and independently replayed display."""

    schema_version: Literal["nurse.v2.report-composition-presentation/1"] = (
        REPORT_COMPOSITION_PRESENTATION_SCHEMA_VERSION
    )
    rule_version: Literal[
        "nurse-v2-report-language-rules/1", "nurse-v2-report-language-rules/2"
    ] = REPORT_LANGUAGE_RULE_VERSION
    foundation: DeterministicSREReport
    foundation_surface_status: Literal["canonical"] = "canonical"
    foundation_semantic_status: Literal["deterministic_render"] = "deterministic_render"
    mechanical_validation: ReportCompositionMechanicalValidation
    claims: tuple[ReportClaimPresentationDecision, ...] = Field(max_length=24)
    safe_display_blocks: tuple[ReportPresentationBlock, ...] = Field(max_length=24)
    status: Literal["full", "partial", "deterministic_fallback"]
    presentation_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def consistent_receipt(self) -> Self:
        mechanical = self.mechanical_validation
        if (
            self.foundation.context_sha256 != mechanical.context_sha256
            or self.foundation.opaque_case_id != mechanical.opaque_case_id
        ):
            raise ValueError("presentation foundation and mechanical identities differ")
        if len(self.claims) != len(mechanical.claims):
            raise ValueError("presentation must retain every original claim")
        displayed = iter(self.safe_display_blocks)
        for decision, result in zip(self.claims, mechanical.claims, strict=True):
            if (
                decision.claim_id != result.proposal.claim_id
                or decision.mechanical_status != result.mechanical_status
            ):
                raise ValueError("presentation decisions differ from original mechanics")
            if tuple(
                (c.field_path, c.raw_text) for c in decision.field_checks
            ) != claim_free_text_fields(result.proposal):
                raise ValueError("presentation must inspect every original free field in order")
            if result.mechanical_status == "rejected":
                continue
            block = next(displayed, None)
            if block is None:
                raise ValueError("display must retain each passed claim or its replacement")
            if decision.replacement is not None:
                if _bytes(block) != _bytes(decision.replacement):
                    raise ValueError("display differs from the separate deterministic replacement")
                if _claim_structure(block.claim) != _claim_structure(result.proposal):
                    raise ValueError("replacement must preserve the original structural claim")
            elif block.surface_status != "model_unverified" or _bytes(block.claim) != _bytes(
                result.proposal
            ):
                raise ValueError("model display must preserve the exact original proposal")
            if (
                tuple(f.fact_ref for f in block.required_context) != result.required_context_refs
                or block.required_safety_codes != result.required_safety_codes
            ):
                raise ValueError("display must retain mechanical context and safety disclosures")
        if next(displayed, None) is not None:
            raise ValueError("display contains an extra block")
        delivered = sum(c.surface_status == "model_unverified" for c in self.claims)
        expected = (
            "full"
            if delivered and delivered == len(self.claims)
            else ("partial" if delivered else "deterministic_fallback")
        )
        if self.status != expected:
            raise ValueError("presentation status differs from original model delivery")
        return self

    def verify(self) -> bool:
        """Check exact integrity and internal consistency; callers still must replay sources."""
        try:
            if type(self) is not ReportCompositionPresentation:
                return False
            preflight_exact_pydantic_tree(
                self,
                root_model_types=(ReportCompositionPresentation,),
                label="report presentation",
                max_string_length=None,
            )
            serialized = self.model_dump(mode="json", warnings="error")
            original = canonical_json_bytes(serialized)
            rebuilt = ReportCompositionPresentation.model_validate_json(original)
            unsigned = {
                key: value for key, value in serialized.items() if key != "presentation_sha256"
            }
            return (
                hmac.compare_digest(original, _bytes(rebuilt))
                and hmac.compare_digest(
                    self.presentation_sha256, sha256_hex(canonical_json_bytes(unsigned))
                )
                and self.foundation.verify()
                and self.mechanical_validation.verify()
            )
        except (AttributeError, RecursionError, TypeError, ValueError):
            return False
