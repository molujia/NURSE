"""Typed report evidence contracts; hashes establish integrity, never authority."""

from __future__ import annotations

import hmac
import math
from typing import Literal, TypeAlias

from pydantic import ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.deterministic_report import DeterministicSREReport
from nurse_rcl.v2.observation_schemas import TimeInterval
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

REPORT_EVIDENCE_POLICY_SCHEMA_VERSION: Literal["nurse.v2.report-evidence-policy/1"] = (
    "nurse.v2.report-evidence-policy/1"
)
REPORT_EVIDENCE_VIEW_SCHEMA_VERSION: Literal["nurse.v2.report-evidence-view/1"] = (
    "nurse.v2.report-evidence-view/1"
)
REPORT_EVIDENCE_PROJECTION_SCHEMA_VERSION: Literal["nurse.v2.report-evidence-projection/1"] = (
    "nurse.v2.report-evidence-projection/1"
)
EvidenceScalar: TypeAlias = str | int | float | bool | None
EvidenceKind: TypeAlias = Literal[
    "ranking",
    "candidate_evidence",
    "competition",
    "uncertainty",
    "limit",
    "observation",
    "hypothesis",
    "predicate",
    "verification",
    "mitigation",
]
ClaimStrength: TypeAlias = Literal[
    "frozen_observation", "ranking_inference", "conditional_recommendation"
]
ScopeBasis: TypeAlias = Literal["frozen_assessment", "bound_observation", "declared_hypothesis"]
OmissionReason: TypeAlias = Literal[
    "detail_byte_budget",
    "detail_count_budget",
    "outside_top_k",
    "unbound_observation",
    "unscoped_predicate",
]


class _EvidenceModel(StrictFrozenModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class ReportEvidenceValue(_EvidenceModel):
    """A named exact JSON scalar, never an arbitrary payload."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    value: EvidenceScalar

    @field_validator("value", mode="before")
    @classmethod
    def require_exact_scalar(cls, value: object) -> object:
        if type(value) not in (str, int, float, bool, type(None)):
            raise ValueError("evidence values require exact JSON scalars")
        if type(value) is float and not math.isfinite(value):
            raise ValueError("evidence values require finite JSON scalars")
        return value


class ReportEvidencePolicy(_EvidenceModel):
    """Versioned byte and detail limits; bytes are not token estimates."""

    schema_version: Literal["nurse.v2.report-evidence-policy/1"] = (
        REPORT_EVIDENCE_POLICY_SCHEMA_VERSION
    )
    configuration_version: str = Field(default="default/1", min_length=1, max_length=128)
    max_view_bytes: int = Field(default=16384, ge=1, le=262144)
    max_detail_facts: int = Field(default=64, ge=0, le=512)

    @model_validator(mode="after")
    def require_versioned_overrides(self) -> Self:
        if self.configuration_version == "default/1" and (
            self.max_view_bytes != 16384 or self.max_detail_facts != 64
        ):
            raise ValueError("policy overrides require a nondefault configuration_version")
        return self

    @property
    def sha256(self) -> str:
        if type(self) is not ReportEvidencePolicy:
            raise ValueError("policy requires exact ReportEvidencePolicy type")
        preflight_exact_pydantic_tree(
            self, root_model_types=(ReportEvidencePolicy,), label="report evidence policy"
        )
        return sha256_hex(canonical_json_bytes(self.model_dump(mode="json", warnings="error")))


class ReportEvidenceFact(_EvidenceModel):
    """One typed fact using local aliases and explicitly bounded claim scope."""

    fact_ref: str = Field(pattern=r"^f[1-9][0-9]{0,5}$")
    kind: EvidenceKind
    target_service_keys: tuple[str, ...]
    intervals: tuple[TimeInterval, ...] = ()
    values: tuple[ReportEvidenceValue, ...] = ()
    status_codes: tuple[str, ...] = ()
    related_fact_refs: tuple[str, ...] = ()
    allowed_claim_strength: ClaimStrength
    scope_basis: ScopeBasis

    @model_validator(mode="after")
    def require_unique_fields(self) -> Self:
        if len(self.target_service_keys) != len(set(self.target_service_keys)):
            raise ValueError("fact targets must be unique")
        if len({item.name for item in self.values}) != len(self.values):
            raise ValueError("fact value names must be unique")
        for values in (self.status_codes, self.related_fact_refs):
            if values != tuple(sorted(set(values))):
                raise ValueError("fact codes and references must be sorted and unique")
        return self


class ReportEvidenceOmission(_EvidenceModel):
    """Deterministically grouped source record omissions, including exclusions."""

    kind: EvidenceKind
    target_service_keys: tuple[str, ...]
    count: int = Field(gt=0)
    reason_code: OmissionReason

    @model_validator(mode="after")
    def require_unique_targets(self) -> Self:
        if len(self.target_service_keys) != len(set(self.target_service_keys)):
            raise ValueError("omission targets must be unique")
        return self


class ReportEvidenceView(_EvidenceModel):
    """Public compact evidence; only replay against independent sources grants trust."""

    schema_version: Literal["nurse.v2.report-evidence-view/1"] = REPORT_EVIDENCE_VIEW_SCHEMA_VERSION
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    top_k: tuple[str, ...] = Field(min_length=1)
    facts: tuple[ReportEvidenceFact, ...] = Field(min_length=1)
    mandatory_fact_refs: tuple[str, ...] = Field(min_length=1)
    omissions: tuple[ReportEvidenceOmission, ...]
    view_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_reference_closure(self) -> Self:
        if len(self.top_k) != len(set(self.top_k)):
            raise ValueError("view Top-K must be unique")
        refs = tuple(item.fact_ref for item in self.facts)
        if refs != tuple(f"f{i}" for i in range(1, len(refs) + 1)):
            raise ValueError("view aliases must be complete and ordered")
        if self.mandatory_fact_refs != refs[: len(self.mandatory_fact_refs)]:
            raise ValueError("mandatory facts must form a nonempty ordered prefix")
        for fact in self.facts:
            if not set(fact.related_fact_refs) <= set(refs):
                raise ValueError("view fact references must have closure")
            if not set(fact.target_service_keys) <= set(self.top_k):
                raise ValueError("view fact targets must be in Top-K")
        omission_keys = tuple(
            (o.kind, o.target_service_keys, o.reason_code) for o in self.omissions
        )
        if omission_keys != tuple(sorted(set(omission_keys))):
            raise ValueError("view omissions must be grouped and sorted")
        return self

    def verify(self) -> bool:
        return _verify_integrity(self, ReportEvidenceView, "view_sha256")


class ReportEvidenceBinding(_EvidenceModel):
    """Private exact source record binding; never include this in model input."""

    fact_ref: str = Field(pattern=r"^f[1-9][0-9]{0,5}$")
    source_kind: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    source_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_fields: tuple[str, ...] = Field(min_length=1)
    same_source_ids: tuple[str, ...]

    @model_validator(mode="after")
    def require_canonical_binding(self) -> Self:
        for values in (self.source_fields, self.same_source_ids):
            if values != tuple(sorted(set(values))):
                raise ValueError("source fields and identities must be sorted and unique")
        return self


class ReportEvidenceProjection(_EvidenceModel):
    """Private replay product retaining the complete deterministic foundation."""

    schema_version: Literal["nurse.v2.report-evidence-projection/1"] = (
        REPORT_EVIDENCE_PROJECTION_SCHEMA_VERSION
    )
    policy: ReportEvidencePolicy
    foundation: DeterministicSREReport
    view: ReportEvidenceView | None
    source_bindings: tuple[ReportEvidenceBinding, ...]
    status: Literal["ready", "mandatory_budget_exhausted"]
    mandatory_view_bytes: int = Field(gt=0)
    projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_consistent_projection(self) -> Self:
        if (self.status == "ready") != (self.view is not None):
            raise ValueError("ready projection must have a view")
        if self.view is None:
            if self.source_bindings or self.mandatory_view_bytes <= self.policy.max_view_bytes:
                raise ValueError("fallback requires exhausted mandatory budget and no bindings")
        else:
            if self.mandatory_view_bytes > self.policy.max_view_bytes:
                raise ValueError("ready projection must fit its mandatory byte budget")
            if tuple(b.fact_ref for b in self.source_bindings) != tuple(
                f.fact_ref for f in self.view.facts
            ):
                raise ValueError("bindings must cover exactly the selected facts")
            if self.view.top_k != tuple(c.service_entity_key for c in self.foundation.top_k):
                raise ValueError("view Top-K differs from foundation")
            if (
                self.view.context_sha256 != self.foundation.context_sha256
                or self.view.opaque_case_id != self.foundation.opaque_case_id
                or self.view.policy_sha256 != self.policy.sha256
            ):
                raise ValueError("view source identity differs from projection")
        return self

    def verify(self) -> bool:
        if not _verify_integrity(self, ReportEvidenceProjection, "projection_sha256"):
            return False
        if not self.foundation.verify():
            return False
        if self.view is None:
            return True
        return (
            self.view.verify()
            and len(canonical_json_bytes(self.view.model_dump(mode="json", warnings="error")))
            <= self.policy.max_view_bytes
        )


def _verify_integrity(value: object, model_type: type[_EvidenceModel], hash_field: str) -> bool:
    try:
        if type(value) is not model_type:
            return False
        preflight_exact_pydantic_tree(
            value,
            root_model_types=(model_type,),
            label="report evidence",
            max_string_length=None,
        )
        assert isinstance(value, _EvidenceModel)
        serialized = value.model_dump(mode="json", warnings="error")
        original_bytes = canonical_json_bytes(serialized)
        rebuilt = model_type.model_validate_json(original_bytes)
        rebuilt_bytes = canonical_json_bytes(rebuilt.model_dump(mode="json", warnings="error"))
        unsigned = {key: item for key, item in serialized.items() if key != hash_field}
        return hmac.compare_digest(original_bytes, rebuilt_bytes) and hmac.compare_digest(
            serialized[hash_field], sha256_hex(canonical_json_bytes(unsigned))
        )
    except (AttributeError, RecursionError, TypeError, ValueError):
        return False
