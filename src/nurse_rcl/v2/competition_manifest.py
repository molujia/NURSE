"""Independent directed competition edges over already frozen core hypotheses.

Declarations and bindings must come from the trusted semantic compiler. A self
hash proves internal integrity only; source authority requires ``verify_inputs``
with that compiler's original declarations, bindings, cores, and pinned hashes.
Nothing in this module modifies a core hypothesis or contributes a score.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Annotated, Any, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.diagnostic_schemas import DiagnosticHypothesis
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree
from nurse_rcl.v2.semantic_proposals import LocalId

COMPETITION_MANIFEST_SCHEMA_VERSION: Literal["nurse.v2.competition-manifest/1"] = (
    "nurse.v2.competition-manifest/1"
)
MAX_COMPETITION_BINDINGS = 32
MAX_COMPETITION_DECLARATIONS = 1024

CompetitionBindingReason: TypeAlias = Literal[
    "compiled",
    "not_compiled",
    "item_schema_invalid",
    "permission_denied",
    "duplicate_local_id",
    "compilation_failed",
]
CompetitionEndpointReason: TypeAlias = CompetitionBindingReason | Literal["missing_binding"]
HypothesisId: TypeAlias = Annotated[str, Field(pattern=r"^hyp_[0-9a-f]{24}$")]
Sha256: TypeAlias = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

_ModelT = TypeVar("_ModelT", bound=BaseModel)
_BOUNDARY_ERRORS = (AttributeError, TypeError, ValueError, RecursionError)


class _CompetitionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, hide_input_in_errors=True)

    @model_validator(mode="before")
    @classmethod
    def preflight_before_validation(cls, value: Any, info: ValidationInfo) -> Any:
        _preflight(value)
        if info.mode == "json" and cls is CompetitionManifest and type(value) is dict:
            # A before validator materializes JSON arrays as Python lists; strict
            # tuple fields need this explicit JSON-only normalization.
            value = dict(value)
            for field in ("declarations", "bindings", "core_hypotheses", "edges"):
                if type(value.get(field)) is list:
                    value[field] = tuple(value[field])
        return value

    @model_validator(mode="after")
    def preflight_existing_state(self) -> Self:
        # Pydantic can skip before validators for an existing model instance.
        # Run before subclass validators invoke any nested record callbacks.
        _preflight(self)
        return self


class CompetitionDeclaration(_CompetitionModel):
    """One directed local-ID edge; duplicate declarations are rejected by the builder."""

    source_local_id: LocalId
    target_local_id: LocalId

    @model_validator(mode="after")
    def reject_self_edge(self) -> Self:
        if self.source_local_id == self.target_local_id:
            raise ValueError("competition_self_edge")
        return self


class CompetitionBinding(_CompetitionModel):
    """A compiler-owned local ID, either compiled or carrying a static failure code."""

    local_id: LocalId
    hypothesis_id: HypothesisId | None
    reason_code: CompetitionBindingReason

    @model_validator(mode="after")
    def require_consistent_binding(self) -> Self:
        if (self.hypothesis_id is not None) != (self.reason_code == "compiled"):
            raise ValueError("competition_binding_reason_mismatch")
        return self


class CompetitionEdge(_CompetitionModel):
    """Resolved or unresolved endpoints, retaining every declared directed edge."""

    source_local_id: LocalId
    target_local_id: LocalId
    source_hypothesis_id: HypothesisId | None
    target_hypothesis_id: HypothesisId | None
    source_reason_code: CompetitionEndpointReason
    target_reason_code: CompetitionEndpointReason
    status: Literal["resolved", "unresolved"]

    @model_validator(mode="after")
    def require_consistent_endpoints(self) -> Self:
        if self.source_local_id == self.target_local_id:
            raise ValueError("competition_self_edge")
        for hypothesis_id, reason in (
            (self.source_hypothesis_id, self.source_reason_code),
            (self.target_hypothesis_id, self.target_reason_code),
        ):
            if (hypothesis_id is not None) != (reason == "compiled"):
                raise ValueError("competition_edge_reason_mismatch")
        resolved = self.source_hypothesis_id is not None and self.target_hypothesis_id is not None
        if (self.status == "resolved") != resolved:
            raise ValueError("competition_edge_status_mismatch")
        return self


class CompetitionManifest(_CompetitionModel):
    """Nonrecursive competition artifact; its hash is never written back to cores."""

    schema_version: Literal["nurse.v2.competition-manifest/1"] = COMPETITION_MANIFEST_SCHEMA_VERSION
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    response_sha256: Sha256
    prompt_sha256: Sha256
    declarations: tuple[CompetitionDeclaration, ...] = Field(
        max_length=MAX_COMPETITION_DECLARATIONS
    )
    bindings: tuple[CompetitionBinding, ...] = Field(max_length=MAX_COMPETITION_BINDINGS)
    core_hypotheses: tuple[DiagnosticHypothesis, ...] = Field(max_length=MAX_COMPETITION_BINDINGS)
    edges: tuple[CompetitionEdge, ...] = Field(max_length=MAX_COMPETITION_DECLARATIONS)
    manifest_sha256: Sha256

    @model_validator(mode="after")
    def require_canonical_manifest(self) -> Self:
        declarations, bindings, cores = _canonical_records(
            opaque_case_id=self.opaque_case_id,
            declarations=self.declarations,
            bindings=self.bindings,
            core_hypotheses=self.core_hypotheses,
        )
        if (
            self.declarations != declarations
            or self.bindings != bindings
            or self.core_hypotheses != cores
            or self.edges != _resolve_edges(declarations, bindings)
        ):
            raise ValueError("competition_manifest_content_mismatch")
        return self

    def verify(self) -> bool:
        """Check bounded exact state, core integrity, mapping closure, edges, and hash."""
        try:
            if type(self) is not CompetitionManifest:
                return False
            _preflight(self)
            _revalidate(self, CompetitionManifest)
            unsigned = self.model_dump(mode="json", exclude={"manifest_sha256"}, warnings="error")
            return self.manifest_sha256 == sha256_hex(canonical_json_bytes(unsigned))
        except _BOUNDARY_ERRORS:
            return False

    def verify_inputs(
        self,
        *,
        opaque_case_id: str,
        response_sha256: str,
        prompt_sha256: str,
        declarations: Sequence[CompetitionDeclaration],
        bindings: Sequence[CompetitionBinding],
        core_hypotheses: Sequence[DiagnosticHypothesis],
    ) -> bool:
        """Rebuild against trusted source inputs, including all pins and local mappings."""
        try:
            if type(self) is not CompetitionManifest:
                return False
            _preflight(self)
            _input_boundary(
                opaque_case_id=opaque_case_id,
                response_sha256=response_sha256,
                prompt_sha256=prompt_sha256,
                declarations=declarations,
                bindings=bindings,
                core_hypotheses=core_hypotheses,
            )
            if not CompetitionManifest.verify(self):
                return False
            rebuilt = build_competition_manifest(
                opaque_case_id=opaque_case_id,
                response_sha256=response_sha256,
                prompt_sha256=prompt_sha256,
                declarations=declarations,
                bindings=bindings,
                core_hypotheses=core_hypotheses,
            )
            return self == rebuilt
        except _BOUNDARY_ERRORS:
            return False


def _preflight(value: object) -> None:
    preflight_exact_pydantic_tree(
        value,
        root_model_types=(CompetitionManifest,),
        label="competition_manifest",
        max_depth=16,
        max_nodes=65_536,
        # Core string/provenance contracts impose no additional length/count cap.
        max_string_length=None,
        provenance_id_limit=None,
    )


def _input_boundary(
    *,
    opaque_case_id: str,
    response_sha256: str,
    prompt_sha256: str,
    declarations: Sequence[CompetitionDeclaration],
    bindings: Sequence[CompetitionBinding],
    core_hypotheses: Sequence[DiagnosticHypothesis],
) -> None:
    for values, limit in (
        (declarations, MAX_COMPETITION_DECLARATIONS),
        (bindings, MAX_COMPETITION_BINDINGS),
        (core_hypotheses, MAX_COMPETITION_BINDINGS),
    ):
        if type(values) not in (list, tuple) or len(values) > limit:
            raise ValueError("competition_sequence_boundary")
    # Preflight the whole input packet before invoking any serializer or verifier.
    _preflight(
        (opaque_case_id, response_sha256, prompt_sha256, declarations, bindings, core_hypotheses)
    )
    for value, pattern in (
        (opaque_case_id, r"opq_[0-9a-f]{32}"),
        (response_sha256, r"[0-9a-f]{64}"),
        (prompt_sha256, r"[0-9a-f]{64}"),
    ):
        if type(value) is not str or re.fullmatch(pattern, value) is None:
            raise ValueError("competition_invalid_source_pin")
    for values, model_type in (
        (declarations, CompetitionDeclaration),
        (bindings, CompetitionBinding),
        (core_hypotheses, DiagnosticHypothesis),
    ):
        if any(type(item) is not model_type for item in values):
            raise ValueError("competition_exact_record_required")


def _revalidate(value: _ModelT, model_type: type[_ModelT]) -> _ModelT:
    if type(value) is not model_type:
        raise ValueError("competition_exact_record_required")
    restored = model_type.model_validate(
        value.model_dump(mode="python", warnings="error"), strict=True
    )
    if restored != value:
        raise ValueError("competition_noncanonical_record")
    return restored


def _canonical_records(
    *,
    opaque_case_id: str,
    declarations: Sequence[CompetitionDeclaration],
    bindings: Sequence[CompetitionBinding],
    core_hypotheses: Sequence[DiagnosticHypothesis],
) -> tuple[
    tuple[CompetitionDeclaration, ...],
    tuple[CompetitionBinding, ...],
    tuple[DiagnosticHypothesis, ...],
]:
    canonical_declarations = tuple(
        sorted(
            (_revalidate(item, CompetitionDeclaration) for item in declarations),
            key=lambda item: (item.source_local_id, item.target_local_id),
        )
    )
    if len(
        {(item.source_local_id, item.target_local_id) for item in canonical_declarations}
    ) != len(canonical_declarations):
        raise ValueError("competition_duplicate_declaration")
    canonical_bindings = tuple(
        sorted(
            (_revalidate(item, CompetitionBinding) for item in bindings),
            key=lambda item: item.local_id,
        )
    )
    if len({item.local_id for item in canonical_bindings}) != len(canonical_bindings):
        raise ValueError("competition_duplicate_local_id")
    mapped_ids = tuple(
        item.hypothesis_id for item in canonical_bindings if item.hypothesis_id is not None
    )
    if len(set(mapped_ids)) != len(mapped_ids):
        raise ValueError("competition_duplicate_hypothesis_alias")
    canonical_cores = tuple(
        sorted(
            (_revalidate(item, DiagnosticHypothesis) for item in core_hypotheses),
            key=lambda item: item.hypothesis_id,
        )
    )
    core_ids = tuple(item.hypothesis_id for item in canonical_cores)
    if len(set(core_ids)) != len(core_ids) or set(core_ids) != set(mapped_ids):
        raise ValueError("competition_core_closure_mismatch")
    for core in canonical_cores:
        if (
            not core.verify()
            or core.opaque_case_id != opaque_case_id
            or core.origin != "semantic_agent"
            or core.lifecycle_status != "proposed"
            or core.alternative_hypothesis_ids
        ):
            raise ValueError("competition_unsupported_core")
    return canonical_declarations, canonical_bindings, canonical_cores


def _resolve_edges(
    declarations: tuple[CompetitionDeclaration, ...], bindings: tuple[CompetitionBinding, ...]
) -> tuple[CompetitionEdge, ...]:
    by_local_id = {item.local_id: item for item in bindings}
    edges = []
    for declaration in declarations:
        source = by_local_id.get(declaration.source_local_id)
        target = by_local_id.get(declaration.target_local_id)
        source_id = None if source is None else source.hypothesis_id
        target_id = None if target is None else target.hypothesis_id
        edges.append(
            CompetitionEdge(
                source_local_id=declaration.source_local_id,
                target_local_id=declaration.target_local_id,
                source_hypothesis_id=source_id,
                target_hypothesis_id=target_id,
                source_reason_code="missing_binding" if source is None else source.reason_code,
                target_reason_code="missing_binding" if target is None else target.reason_code,
                status="resolved"
                if source_id is not None and target_id is not None
                else "unresolved",
            )
        )
    return tuple(edges)


def build_competition_manifest(
    *,
    opaque_case_id: str,
    response_sha256: str,
    prompt_sha256: str,
    declarations: Sequence[CompetitionDeclaration],
    bindings: Sequence[CompetitionBinding],
    core_hypotheses: Sequence[DiagnosticHypothesis],
) -> CompetitionManifest:
    """Canonicalize trusted inputs; reject duplicates and preserve unresolved/cyclic edges.

    Only exact finite lists/tuples and exact code-owned records are accepted.
    Failures expose static codes, never the supplied content or private prose.
    """
    try:
        _input_boundary(
            opaque_case_id=opaque_case_id,
            response_sha256=response_sha256,
            prompt_sha256=prompt_sha256,
            declarations=declarations,
            bindings=bindings,
            core_hypotheses=core_hypotheses,
        )
        canonical_declarations, canonical_bindings, canonical_cores = _canonical_records(
            opaque_case_id=opaque_case_id,
            declarations=declarations,
            bindings=bindings,
            core_hypotheses=core_hypotheses,
        )
        unsigned: dict[str, object] = {
            "schema_version": COMPETITION_MANIFEST_SCHEMA_VERSION,
            "opaque_case_id": opaque_case_id,
            "response_sha256": response_sha256,
            "prompt_sha256": prompt_sha256,
            "declarations": tuple(item.model_dump(mode="json") for item in canonical_declarations),
            "bindings": tuple(item.model_dump(mode="json") for item in canonical_bindings),
            "core_hypotheses": tuple(item.model_dump(mode="python") for item in canonical_cores),
            "edges": tuple(
                item.model_dump(mode="json")
                for item in _resolve_edges(canonical_declarations, canonical_bindings)
            ),
        }
        return CompetitionManifest.model_validate(
            {**unsigned, "manifest_sha256": sha256_hex(canonical_json_bytes(unsigned))}
        )
    except _BOUNDARY_ERRORS:
        raise ValueError("invalid_competition_inputs") from None
