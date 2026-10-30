"""Private report accounting contracts; integrity does not establish source authority."""

from __future__ import annotations

import hmac
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.report_run_schemas import ReportRunReason
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

REPORT_COMPOSITION_EXPLAINABILITY_SCHEMA_VERSION: Literal[
    "nurse.v2.report-composition-explainability/1"
] = "nurse.v2.report-composition-explainability/1"
_SHA: TypeAlias = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_ClaimId: TypeAlias = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")]
_POPULATIONS = (
    "checked_claims",
    "mechanically_passed_claims",
    "mechanically_rejected_claims",
    "language_violation_claims",
    "semantic_not_established_claims",
    "blocked_original_claims",
    "displayed_model_claims",
    "replacement_claims",
)
_INTEGERS = (
    "logical_requests",
    "checked_free_text_fields",
    "violating_free_text_fields",
    "displayed_blocks",
    "deterministic_foundation_reports",
)
_BOOLEANS = (
    "is_llm_internal_transparency",
    "is_human_benefit_evidence",
    "ranking_mutable",
    "adds_ranking_facts",
)
_NULLABLE_STRINGS = ("source_mechanical_sha256", "source_presentation_sha256", "reason_code")


class _ExplainabilityModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class ReportClaimPopulation(_ExplainabilityModel):
    """Distinct original claim IDs, preserving response order; count is not a text unit."""

    claim_ids: tuple[_ClaimId, ...] = Field(max_length=24)
    count: int = Field(ge=0, le=24)

    @field_validator("count", mode="before")
    @classmethod
    def exact_count(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("report claim population requires an exact integer count")
        return value

    @field_validator("claim_ids", mode="before")
    @classmethod
    def exact_claim_ids(cls, value: object, info: ValidationInfo) -> object:
        # JSON arrays arrive as exact lists; Python entry points require exact tuples.
        if type(value) is not tuple and not (info.mode == "json" and type(value) is list):
            raise ValueError("report claim population requires exact tuple claim IDs")
        if len(value) > 24:
            raise ValueError("report claim population exceeds the claim ID limit")
        if any(type(claim_id) is not str for claim_id in value):
            raise ValueError("report claim population requires exact string claim IDs")
        return tuple(value) if type(value) is list else value

    @model_validator(mode="after")
    def consistent_population(self) -> Self:
        if self.count != len(self.claim_ids) or len(set(self.claim_ids)) != self.count:
            raise ValueError("report claim population requires unique IDs and matching count")
        return self


class ReportCompositionExplainability(_ExplainabilityModel):
    """Counts only: checked denominates checks; known generated denominates coverage.

    Unknown and zero denominators define no percentage. The complete deterministic
    foundation is one report, independent of claim blocks and legacy sentence metrics.
    """

    schema_version: Literal["nurse.v2.report-composition-explainability/1"] = (
        REPORT_COMPOSITION_EXPLAINABILITY_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    source_run_audit_sha256: _SHA
    source_context_sha256: _SHA
    source_top_k_sha256: _SHA
    source_policy_sha256: _SHA
    source_foundation_sha256: _SHA
    source_mechanical_sha256: _SHA | None
    source_presentation_sha256: _SHA | None
    objective: Literal["explainability"] = "explainability"
    owner: Literal["report_layer"] = "report_layer"
    is_llm_internal_transparency: Literal[False] = False
    is_human_benefit_evidence: Literal[False] = False
    ranking_mutable: Literal[False] = False
    adds_ranking_facts: Literal[False] = False
    natural_language_fidelity: Literal["not_established"] = "not_established"
    report_status: Literal["full", "partial", "deterministic_fallback"]
    reason_code: ReportRunReason | None
    logical_requests: int = Field(ge=0, le=1)
    generation_status: Literal["not_requested", "parsed", "unknown"]
    generated_claims: ReportClaimPopulation | None
    unchecked_claims: ReportClaimPopulation | None
    checked_claims: ReportClaimPopulation
    mechanically_passed_claims: ReportClaimPopulation
    mechanically_rejected_claims: ReportClaimPopulation
    language_violation_claims: ReportClaimPopulation
    semantic_not_established_claims: ReportClaimPopulation
    blocked_original_claims: ReportClaimPopulation
    displayed_model_claims: ReportClaimPopulation
    replacement_claims: ReportClaimPopulation
    checked_free_text_fields: int = Field(ge=0, le=192)
    violating_free_text_fields: int = Field(ge=0, le=192)
    displayed_blocks: int = Field(ge=0, le=24)
    deterministic_foundation_reports: Literal[1] = 1
    artifact_sha256: _SHA

    @field_validator("*", mode="before")
    @classmethod
    def exact_string_metadata(cls, value: object, info: ValidationInfo) -> object:
        if info.field_name in (
            *_POPULATIONS,
            "generated_claims",
            "unchecked_claims",
            *_INTEGERS,
            *_BOOLEANS,
        ):
            return value
        if value is None and info.field_name in _NULLABLE_STRINGS:
            return value
        if type(value) is not str:
            raise ValueError("report explainability requires exact string metadata")
        return value

    @field_validator(*_INTEGERS, mode="before")
    @classmethod
    def exact_integers(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("report explainability requires exact integer counts")
        return value

    @field_validator(*_BOOLEANS, mode="before")
    @classmethod
    def exact_booleans(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("report explainability requires exact boolean assurance flags")
        return value

    @model_validator(mode="after")
    def consistent_accounting(self) -> Self:
        generated, unchecked = self.generated_claims, self.unchecked_claims
        checked = self.checked_claims
        if self.generation_status == "unknown":
            if generated is not None or unchecked is not None or checked.count:
                raise ValueError("unknown generation cannot invent populations or checks")
        elif generated is None or unchecked is None:
            raise ValueError("known generation requires generated and unchecked populations")
        else:
            _partition(generated, checked, unchecked)
        if self.generation_status == "not_requested" and (
            generated is None or generated.count or checked.count
        ):
            raise ValueError("not requested generation must be known empty")
        if self.logical_requests != int(self.generation_status != "not_requested"):
            raise ValueError("logical request count describes this single run only")
        passed, rejected = self.mechanically_passed_claims, self.mechanically_rejected_claims
        violation, unestablished = (
            self.language_violation_claims,
            self.semantic_not_established_claims,
        )
        _partition(checked, passed, rejected)
        _partition(checked, violation, unestablished)
        blocked = set(rejected.claim_ids) | set(violation.claim_ids)
        _selection(checked, self.blocked_original_claims, blocked)
        _selection(checked, self.displayed_model_claims, set(checked.claim_ids) - blocked)
        _selection(
            checked, self.replacement_claims, set(passed.claim_ids) & set(violation.claim_ids)
        )
        if not checked.count <= self.checked_free_text_fields <= 8 * checked.count:
            raise ValueError("checked free text counts fields of checked originals only")
        if (
            not violation.count
            <= self.violating_free_text_fields
            <= min(8 * violation.count, self.checked_free_text_fields - unestablished.count)
        ):
            raise ValueError("violating free text counts distinct violating original fields")
        if (self.source_mechanical_sha256 is None) != (self.source_presentation_sha256 is None):
            raise ValueError("mechanical and presentation source pins must coexist")
        if self.source_mechanical_sha256 is None and checked.count:
            raise ValueError("checked originals require mechanical and presentation source pins")
        if self.generation_status != "parsed" and self.source_mechanical_sha256 is not None:
            raise ValueError("unparsed generation cannot contain check source pins")
        displayed = self.displayed_model_claims.count
        if self.displayed_blocks != displayed + self.replacement_claims.count:
            raise ValueError("displayed blocks count model and replacement blocks only")
        expected_status = (
            "full"
            if displayed and generated is not None and displayed == generated.count
            else "partial"
            if displayed
            else "deterministic_fallback"
        )
        if self.report_status != expected_status:
            raise ValueError("report status derives from original model delivery, not replacements")
        return self

    def verify(self) -> bool:
        """Check bounded exact integrity only; an independent source replay is still required."""
        try:
            _preflight_explainability(self)
            serialized = self.model_dump(mode="json", warnings="error")
            original = canonical_json_bytes(serialized)
            rebuilt = ReportCompositionExplainability.model_validate_json(original)
            unsigned = {k: v for k, v in serialized.items() if k != "artifact_sha256"}
            return hmac.compare_digest(original, _bytes(rebuilt)) and hmac.compare_digest(
                self.artifact_sha256, sha256_hex(canonical_json_bytes(unsigned))
            )
        except (AttributeError, RecursionError, TypeError, ValueError):
            return False


def _selection(parent: ReportClaimPopulation, child: ReportClaimPopulation, ids: set[str]) -> None:
    if not ids <= set(parent.claim_ids) or child.claim_ids != tuple(
        claim_id for claim_id in parent.claim_ids if claim_id in ids
    ):
        raise ValueError("report populations must preserve original order and set membership")


def _partition(
    parent: ReportClaimPopulation, first: ReportClaimPopulation, second: ReportClaimPopulation
) -> None:
    first_ids, second_ids = set(first.claim_ids), set(second.claim_ids)
    if first_ids & second_ids or first_ids | second_ids != set(parent.claim_ids):
        raise ValueError("report populations must form a disjoint complete partition")
    _selection(parent, first, first_ids)
    _selection(parent, second, second_ids)


def _bytes(value: ReportCompositionExplainability) -> bytes:
    return canonical_json_bytes(value.model_dump(mode="json", warnings="error"))


def _preflight_explainability(artifact: ReportCompositionExplainability) -> None:
    """Reject hostile state and Python/JSON type normalization before any serializer."""
    if type(artifact) is not ReportCompositionExplainability:
        raise ValueError(
            "report explainability requires exact ReportCompositionExplainability type"
        )
    preflight_exact_pydantic_tree(
        artifact,
        root_model_types=(ReportCompositionExplainability,),
        label="report explainability",
        max_depth=16,
        max_nodes=4096,
        max_string_length=256,
    )
    for name in ReportCompositionExplainability.model_fields:
        value = object.__getattribute__(artifact, name)
        if name in (*_POPULATIONS, "generated_claims", "unchecked_claims"):
            if value is None and name in ("generated_claims", "unchecked_claims"):
                continue
            if type(value) is not ReportClaimPopulation:
                raise ValueError("report explainability requires exact claim populations")
            ids = object.__getattribute__(value, "claim_ids")
            count = object.__getattribute__(value, "count")
            if type(ids) is not tuple or len(ids) > 24 or type(count) is not int:
                raise ValueError("report explainability requires bounded exact population types")
            if not 0 <= count <= 24:
                raise ValueError("report explainability population count exceeds bounds")
            if any(type(claim_id) is not str for claim_id in ids):
                raise ValueError("report explainability requires exact string claim IDs")
        elif name in _INTEGERS:
            if type(value) is not int:
                raise ValueError("report explainability requires exact integer counts")
            lower, upper = {
                "logical_requests": (0, 1),
                "checked_free_text_fields": (0, 192),
                "violating_free_text_fields": (0, 192),
                "displayed_blocks": (0, 24),
                "deterministic_foundation_reports": (1, 1),
            }[name]
            if not lower <= value <= upper:
                raise ValueError("report explainability integer count exceeds bounds")
        elif name in _BOOLEANS:
            if type(value) is not bool:
                raise ValueError("report explainability requires exact boolean flags")
        elif not (value is None and name in _NULLABLE_STRINGS) and type(value) is not str:
            raise ValueError("report explainability requires exact string metadata")
