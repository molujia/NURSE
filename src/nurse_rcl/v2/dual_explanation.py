"""Separate, non-compensating explanation artifacts for NURSE V2."""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from typing import Literal, TypeAlias

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.case_audit import (
    CaseAuditLineageIndex,
    CaseAuditLineageNode,
    ImmutableCaseAuditPackage,
    replay_case_audit,
)
from nurse_rcl.v2.deterministic_report import (
    ConditionalMitigation,
    DeterministicSREReport,
    ReportVerificationAction,
    require_deterministic_sre_report,
)
from nurse_rcl.v2.ranking_replay import (
    ModelSensitivityRequest,
    ModelSensitivityResult,
    recompute_model_sensitivity,
    replay_reference_ranking,
)
from nurse_rcl.v2.reference_ranking import ReferenceRankingResult
from nurse_rcl.v2.report_context import FrozenReportContext
from nurse_rcl.v2.report_generation_audit import (
    ReportGenerationAudit,
    ReportGenerationFailureAudit,
    ReportGenerationFailureCode,
    require_report_generation_audit,
    require_report_generation_failure_audit,
)
from nurse_rcl.v2.report_grounding import (
    GroundingViolationCode,
    ReportGroundingCatalog,
    ReportSentenceProposal,
    StatementType,
    build_report_grounding_catalog,
)
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

INTERPRETABILITY_ARTIFACT_SCHEMA_VERSION: Literal["nurse.v2.interpretability-artifact/1"] = (
    "nurse.v2.interpretability-artifact/1"
)
EXPLAINABILITY_ARTIFACT_SCHEMA_VERSION: Literal["nurse.v2.explainability-artifact/1"] = (
    "nurse.v2.explainability-artifact/1"
)
DUAL_EXPLANATION_MANIFEST_SCHEMA_VERSION: Literal["nurse.v2.dual-explanation-manifest/1"] = (
    "nurse.v2.dual-explanation-manifest/1"
)

SensitivityStatus: TypeAlias = Literal["not_requested", "supplied"]
SensitivityPair: TypeAlias = tuple[ModelSensitivityRequest, ModelSensitivityResult]

_MAX_EXPLANATION_DEPTH: Literal[64] = 64
_MAX_EXPLANATION_NODES: Literal[100_000] = 100_000
_MAX_SENSITIVITY_PAIRS: Literal[10_000] = 10_000


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _canonical_ids(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{label} must be sorted and unique")
    return values


class ProvenancePartitions(StrictFrozenModel):
    """Non-compensating accepted/rejected/conflicted/unknown provenance."""

    accepted_provenance_ids: tuple[str, ...]
    rejected_provenance_ids: tuple[str, ...]
    conflicted_provenance_ids: tuple[str, ...]
    unknown_provenance_ids: tuple[str, ...]

    @model_validator(mode="after")
    def require_canonical_partitions(self) -> Self:
        for values, label in (
            (self.accepted_provenance_ids, "accepted provenance IDs"),
            (self.rejected_provenance_ids, "rejected provenance IDs"),
            (self.conflicted_provenance_ids, "conflicted provenance IDs"),
            (self.unknown_provenance_ids, "unknown provenance IDs"),
        ):
            _canonical_ids(values, label=label)
        return self


class SensitivityArtifactReference(StrictFrozenModel):
    """One externally replayed model-behavior sensitivity scenario."""

    scenario_key: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,127}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    top_1_changed: bool
    top_k_membership_changed: bool
    analysis_kind: Literal["model_behavior_only"] = "model_behavior_only"
    is_causal_intervention: Literal[False] = False


class InterpretabilitySensitivitySummary(StrictFrozenModel):
    """Explicitly distinguish no request from supplied sensitivity evidence."""

    status: SensitivityStatus
    references: tuple[SensitivityArtifactReference, ...]
    sensitivity_request_sha256s: tuple[str, ...]
    sensitivity_result_sha256s: tuple[str, ...]
    sensitivity_analysis_kind: Literal["model_behavior_only"] = "model_behavior_only"
    is_causal_intervention: Literal[False] = False

    @model_validator(mode="after")
    def require_status_consistency(self) -> Self:
        if self.references != tuple(sorted(self.references, key=lambda item: item.request_sha256)):
            raise ValueError("sensitivity references must be canonical")
        request_sha256s = tuple(item.request_sha256 for item in self.references)
        result_sha256s = tuple(item.result_sha256 for item in self.references)
        if (
            self.sensitivity_request_sha256s != request_sha256s
            or self.sensitivity_result_sha256s != result_sha256s
            or len(set(request_sha256s)) != len(request_sha256s)
            or len(set(result_sha256s)) != len(result_sha256s)
        ):
            raise ValueError("sensitivity summary does not match its references")
        if (self.status == "not_requested") != (not self.references):
            raise ValueError("sensitivity status contradicts supplied references")
        if len({item.scenario_key for item in self.references}) != len(self.references):
            raise ValueError("sensitivity scenario keys must be unique")
        return self


class CandidateInterpretabilityTrace(StrictFrozenModel):
    """Complete transparent mechanism projection for one ranked candidate."""

    candidate_service_key: str = Field(min_length=1)
    final_rank: int = Field(ge=1, strict=True)
    is_top_k: bool
    top_k_entry_record_id: str | None
    score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quantized_total_score: str = Field(pattern=r"^-?\d+\.\d{12}$")
    quantized_event_evidence_score: str = Field(pattern=r"^\d+\.\d{12}$")
    quantized_propagation_score: str = Field(pattern=r"^\d+\.\d{12}$")
    rule_score: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False, strict=True)
    history_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)
    event_group_ids: tuple[str, ...]
    predicate_result_ids: tuple[str, ...]
    verification_request_ids: tuple[str, ...]
    verification_result_ids: tuple[str, ...]
    hypothesis_ids: tuple[str, ...]
    guard_result_ids: tuple[str, ...]
    role_assertion_ids: tuple[str, ...]
    propagation_path_ids: tuple[str, ...]
    rule_ids: tuple[str, ...]
    history_neighbor_ids: tuple[str, ...]
    provenance: ProvenancePartitions
    backward_record_ids: tuple[str, ...] = Field(min_length=1)
    forward_record_ids: tuple[str, ...] = Field(min_length=1)
    evidence_tie_group_id: str = Field(pattern=r"^tie_[0-9a-f]{24}$")
    stable_tie_break_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tie_break_is_evidence: Literal[False] = False

    @model_validator(mode="after")
    def require_canonical_trace(self) -> Self:
        collections = (
            (self.event_group_ids, "event group IDs"),
            (self.predicate_result_ids, "predicate result IDs"),
            (self.verification_request_ids, "verification request IDs"),
            (self.verification_result_ids, "verification result IDs"),
            (self.hypothesis_ids, "hypothesis IDs"),
            (self.guard_result_ids, "guard result IDs"),
            (self.role_assertion_ids, "role assertion IDs"),
            (self.propagation_path_ids, "propagation path IDs"),
            (self.rule_ids, "rule IDs"),
            (self.history_neighbor_ids, "history neighbor IDs"),
            (self.backward_record_ids, "backward record IDs"),
            (self.forward_record_ids, "forward record IDs"),
        )
        for values, label in collections:
            _canonical_ids(values, label=label)
        if (self.top_k_entry_record_id is not None) != self.is_top_k:
            raise ValueError("Top-K membership contradicts its lineage entry")
        if self.score_sha256 not in self.backward_record_ids:
            raise ValueError("candidate trace must retain its authoritative score")
        return self


class InterpretabilityArtifact(StrictFrozenModel):
    """Small-model mechanism transparency, independent from report quality."""

    schema_version: Literal["nurse.v2.interpretability-artifact/1"] = (
        INTERPRETABILITY_ARTIFACT_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    objective: Literal["interpretability"] = "interpretability"
    owner: Literal["small_model"] = "small_model"
    claim_scope: Literal["mechanism_auditability"] = "mechanism_auditability"
    is_llm_internal_transparency: Literal[False] = False
    is_human_benefit_evidence: Literal[False] = False
    source_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ranking_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ranking_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_top_k_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_lineage_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ranking_replay_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exact_ranking_replay: Literal[True] = True
    candidate_traces: tuple[CandidateInterpretabilityTrace, ...] = Field(min_length=1)
    provenance: ProvenancePartitions
    sensitivity: InterpretabilitySensitivitySummary
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_complete_ordered_candidates(self) -> Self:
        if tuple(item.final_rank for item in self.candidate_traces) != tuple(
            range(1, len(self.candidate_traces) + 1)
        ):
            raise ValueError("interpretability candidate ranks must be complete and ordered")
        if len({item.candidate_service_key for item in self.candidate_traces}) != len(
            self.candidate_traces
        ):
            raise ValueError("interpretability candidates must be unique")
        return self

    def verify(self) -> bool:
        try:
            if type(self) is not InterpretabilityArtifact:
                return False
            preflight_exact_pydantic_tree(
                self,
                root_model_types=(InterpretabilityArtifact,),
                label="interpretability artifact",
                max_depth=_MAX_EXPLANATION_DEPTH,
                max_nodes=_MAX_EXPLANATION_NODES,
            )
            rebuilt = InterpretabilityArtifact.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = rebuilt.model_dump(
                mode="json",
                exclude={"artifact_sha256"},
                warnings="error",
            )
        except (AttributeError, KeyError, RecursionError, TypeError, ValueError):
            return False
        return rebuilt == self and hmac.compare_digest(
            rebuilt.artifact_sha256,
            _digest(unsigned),
        )


class StatementIdPartitions(StrictFrozenModel):
    """Disjoint safe report statements grouped by epistemic type."""

    observation_statement_ids: tuple[str, ...]
    inference_statement_ids: tuple[str, ...]
    recommendation_statement_ids: tuple[str, ...]

    @property
    def all_statement_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                (
                    *self.observation_statement_ids,
                    *self.inference_statement_ids,
                    *self.recommendation_statement_ids,
                )
            )
        )

    @model_validator(mode="after")
    def require_disjoint_canonical_partitions(self) -> Self:
        values = (
            self.observation_statement_ids,
            self.inference_statement_ids,
            self.recommendation_statement_ids,
        )
        for items, label in zip(
            values,
            ("observation", "inference", "recommendation"),
            strict=True,
        ):
            _canonical_ids(items, label=f"{label} statement IDs")
        if any(
            set(left) & set(right)
            for index, left in enumerate(values)
            for right in values[index + 1 :]
        ):
            raise ValueError("statement classes must be disjoint")
        return self


class GenerationViolationReference(StrictFrozenModel):
    """Safe classification reference that deliberately omits raw generated text."""

    position: int = Field(ge=1, strict=True)
    disposition: Literal["weakened", "rejected"]
    violation_codes: tuple[GroundingViolationCode, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_canonical_codes(self) -> Self:
        _canonical_ids(self.violation_codes, label="generation violation codes")
        return self


class ReportGenerationProjection(StrictFrozenModel):
    """Mutually exclusive deterministic, grounded, or fallback report outcome."""

    outcome: Literal[
        "deterministic_baseline",
        "grounded_generation",
        "deterministic_fallback",
    ]
    generation_audit_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    failure_audit_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    failure_code: ReportGenerationFailureCode | None = None
    accepted_count: int | None = Field(default=None, ge=0, strict=True)
    weakened_count: int | None = Field(default=None, ge=0, strict=True)
    rejected_count: int | None = Field(default=None, ge=0, strict=True)
    raw_violation_count: int | None = Field(default=None, ge=0, strict=True)
    grounded_statement_ids: tuple[str, ...]
    violation_references: tuple[GenerationViolationReference, ...]

    @model_validator(mode="after")
    def require_outcome_shape(self) -> Self:
        _canonical_ids(self.grounded_statement_ids, label="grounded statement IDs")
        if self.violation_references != tuple(
            sorted(self.violation_references, key=lambda item: item.position)
        ) or len({item.position for item in self.violation_references}) != len(
            self.violation_references
        ):
            raise ValueError("generation violation references must be ordered and unique")
        counts = (
            self.accepted_count,
            self.weakened_count,
            self.rejected_count,
            self.raw_violation_count,
        )
        if self.outcome == "deterministic_baseline":
            if (
                self.generation_audit_sha256 is not None
                or self.failure_audit_sha256 is not None
                or self.failure_code is not None
                or any(item is not None for item in counts)
                or self.grounded_statement_ids
                or self.violation_references
            ):
                raise ValueError("deterministic baseline cannot claim generation evidence")
        elif self.outcome == "grounded_generation":
            if (
                self.generation_audit_sha256 is None
                or self.failure_audit_sha256 is not None
                or self.failure_code is not None
                or any(item is None for item in counts)
            ):
                raise ValueError("grounded generation requires one complete success audit")
            if not self.grounded_statement_ids:
                raise ValueError("grounded generation must deliver at least one safe statement")
            assert self.accepted_count is not None
            assert self.weakened_count is not None
            assert self.rejected_count is not None
            assert self.raw_violation_count is not None
            if (
                self.accepted_count + self.weakened_count < 1
                or self.raw_violation_count != self.weakened_count + self.rejected_count
                or self.raw_violation_count != len(self.violation_references)
                or self.weakened_count
                != sum(item.disposition == "weakened" for item in self.violation_references)
                or self.rejected_count
                != sum(item.disposition == "rejected" for item in self.violation_references)
            ):
                raise ValueError("grounded generation counts contradict retained decisions")
        elif (
            self.generation_audit_sha256 is not None
            or self.failure_audit_sha256 is None
            or self.failure_code is None
            or any(item is not None for item in counts)
            or self.grounded_statement_ids
            or self.violation_references
        ):
            raise ValueError("deterministic fallback requires only one failure audit")
        return self


class ExplainabilityArtifact(StrictFrozenModel):
    """Frozen-result comprehension and action support, separate from ranking logic."""

    schema_version: Literal["nurse.v2.explainability-artifact/1"] = (
        EXPLAINABILITY_ARTIFACT_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    objective: Literal["explainability"] = "explainability"
    owner: Literal["report_layer"] = "report_layer"
    claim_scope: Literal["frozen_result_comprehension_and_action_support"] = (
        "frozen_result_comprehension_and_action_support"
    )
    is_llm_internal_transparency: Literal[False] = False
    is_human_benefit_evidence: Literal[False] = False
    source_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_top_k_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    top_k_statement_id: str = Field(pattern=r"^stm_[0-9a-f]{24}$")
    top_1_basis_statement_id: str = Field(pattern=r"^stm_[0-9a-f]{24}$")
    competition_statement_ids: tuple[str, ...]
    supporting_reference_ids: tuple[str, ...]
    refuting_reference_ids: tuple[str, ...]
    conflicting_reference_ids: tuple[str, ...]
    missing_reference_ids: tuple[str, ...]
    unknown_reference_ids: tuple[str, ...]
    run_mode_statement_id: str = Field(pattern=r"^stm_[0-9a-f]{24}$")
    degradation_reason_codes: tuple[str, ...]
    limitation_statement_ids: tuple[str, ...]
    available_statements: StatementIdPartitions
    delivered_statements: StatementIdPartitions
    verification_actions: tuple[ReportVerificationAction, ...]
    conditional_mitigations: tuple[ConditionalMitigation, ...]
    generation: ReportGenerationProjection
    ranking_mutable: Literal[False] = False
    adds_ranking_facts: Literal[False] = False
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_closed_report_projection(self) -> Self:
        for values, label in (
            (self.competition_statement_ids, "competition statement IDs"),
            (self.supporting_reference_ids, "supporting reference IDs"),
            (self.refuting_reference_ids, "refuting reference IDs"),
            (self.conflicting_reference_ids, "conflicting reference IDs"),
            (self.missing_reference_ids, "missing reference IDs"),
            (self.unknown_reference_ids, "unknown reference IDs"),
            (self.degradation_reason_codes, "degradation reason codes"),
            (self.limitation_statement_ids, "limitation statement IDs"),
        ):
            _canonical_ids(values, label=label)
        available_ids = set(self.available_statements.all_statement_ids)
        delivered_ids = set(self.delivered_statements.all_statement_ids)
        required_statement_ids = {
            self.top_k_statement_id,
            self.top_1_basis_statement_id,
            self.run_mode_statement_id,
            *self.competition_statement_ids,
            *self.limitation_statement_ids,
        }
        if not available_ids or not required_statement_ids <= available_ids:
            raise ValueError("explainability statement references leave the report catalog")
        if not delivered_ids <= available_ids:
            raise ValueError("delivered statements must be a subset of available statements")
        if self.generation.outcome == "grounded_generation":
            if set(self.generation.grounded_statement_ids) != delivered_ids:
                raise ValueError("grounded statements must equal the delivered projection")
        elif self.available_statements != self.delivered_statements:
            raise ValueError("deterministic reports must deliver their complete safe catalog")
        if self.verification_actions != tuple(
            sorted(self.verification_actions, key=lambda item: item.action_id)
        ) or self.conditional_mitigations != tuple(
            sorted(self.conditional_mitigations, key=lambda item: item.mitigation_id)
        ):
            raise ValueError("report action contracts must be canonical")
        return self

    def verify(self) -> bool:
        try:
            if type(self) is not ExplainabilityArtifact:
                return False
            preflight_exact_pydantic_tree(
                self,
                root_model_types=(ExplainabilityArtifact,),
                label="explainability artifact",
                max_depth=_MAX_EXPLANATION_DEPTH,
                max_nodes=_MAX_EXPLANATION_NODES,
            )
            rebuilt = ExplainabilityArtifact.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if (
                rebuilt != self
                or any(not item.verify() for item in rebuilt.verification_actions)
                or any(not item.verify() for item in rebuilt.conditional_mitigations)
            ):
                return False
            unsigned = rebuilt.model_dump(
                mode="json",
                exclude={"artifact_sha256"},
                warnings="error",
            )
        except (AttributeError, KeyError, RecursionError, TypeError, ValueError):
            return False
        return hmac.compare_digest(rebuilt.artifact_sha256, _digest(unsigned))


class DualExplanationManifest(StrictFrozenModel):
    """Hash-only identity binding for independent explanation layers."""

    schema_version: Literal["nurse.v2.dual-explanation-manifest/1"] = (
        DUAL_EXPLANATION_MANIFEST_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    source_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_top_k_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    interpretability_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    explainability_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    separation_policy: Literal["separate_non_compensating"] = "separate_non_compensating"
    combined_score: None = None
    combined_pass: None = None
    cross_layer_compensation_allowed: Literal[False] = False
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            if type(self) is not DualExplanationManifest:
                return False
            preflight_exact_pydantic_tree(
                self,
                root_model_types=(DualExplanationManifest,),
                label="dual explanation manifest",
                max_depth=_MAX_EXPLANATION_DEPTH,
                max_nodes=_MAX_EXPLANATION_NODES,
            )
            rebuilt = DualExplanationManifest.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if rebuilt != self:
                return False
            unsigned = rebuilt.model_dump(
                mode="json",
                exclude={"manifest_sha256"},
                warnings="error",
            )
        except (AttributeError, KeyError, RecursionError, TypeError, ValueError):
            return False
        return hmac.compare_digest(rebuilt.manifest_sha256, _digest(unsigned))


def _walk_lineage(
    *,
    start_record_id: str,
    node_by_id: Mapping[str, CaseAuditLineageNode],
    direction: Literal["sources", "consumers"],
) -> tuple[str, ...]:
    visited: set[str] = set()
    pending = [start_record_id]
    while pending:
        record_id = pending.pop()
        if record_id in visited:
            continue
        node = node_by_id.get(record_id)
        if node is None:
            raise ValueError(f"interpretability lineage references unknown record: {record_id}")
        visited.add(record_id)
        pending.extend(
            node.source_record_ids if direction == "sources" else node.consumer_record_ids
        )
        if len(visited) + len(pending) > _MAX_EXPLANATION_NODES:
            raise ValueError("interpretability lineage exceeds the node budget")
    return tuple(sorted(visited))


def _ids_by_kind(
    record_ids: tuple[str, ...],
    *,
    node_by_id: Mapping[str, CaseAuditLineageNode],
    kinds: set[str],
) -> tuple[str, ...]:
    return tuple(
        record_id for record_id in record_ids if node_by_id[record_id].record_kind in kinds
    )


def _provenance_partitions(
    *,
    accepted: tuple[str, ...],
    rejected: tuple[str, ...],
    conflicted: tuple[str, ...],
    unknown: tuple[str, ...],
    allowed_ids: set[str] | None = None,
) -> ProvenancePartitions:
    def retained(values: tuple[str, ...]) -> tuple[str, ...]:
        if allowed_ids is None:
            return values
        return tuple(item for item in values if item in allowed_ids)

    return ProvenancePartitions(
        accepted_provenance_ids=retained(accepted),
        rejected_provenance_ids=retained(rejected),
        conflicted_provenance_ids=retained(conflicted),
        unknown_provenance_ids=retained(unknown),
    )


def _stable_tie_break_sha256(*, seed_sha256: str, candidate_service_key: str) -> str:
    return _digest(
        {
            "seed_sha256": seed_sha256,
            "service_entity_key": candidate_service_key,
        }
    )


def _build_sensitivity_summary(
    *,
    ranking: ReferenceRankingResult,
    sensitivity_pairs: tuple[SensitivityPair, ...],
) -> InterpretabilitySensitivitySummary:
    if type(sensitivity_pairs) is not tuple:
        raise ValueError("sensitivity pairs must use an exact tuple")
    if tuple.__len__(sensitivity_pairs) > _MAX_SENSITIVITY_PAIRS:
        raise ValueError("sensitivity pair count exceeds the explanation limit")
    verified: list[tuple[ModelSensitivityRequest, ModelSensitivityResult]] = []
    for pair in sensitivity_pairs:
        if type(pair) is not tuple or tuple.__len__(pair) != 2:
            raise ValueError("each sensitivity pair must use an exact two-item tuple")
        request = tuple.__getitem__(pair, 0)
        result = tuple.__getitem__(pair, 1)
        if (
            type(request) is not ModelSensitivityRequest
            or type(result) is not ModelSensitivityResult
        ):
            raise ValueError("sensitivity pairs require exact request and result models")
        try:
            preflight_exact_pydantic_tree(
                pair,
                root_model_types=(ModelSensitivityRequest, ModelSensitivityResult),
                label="model sensitivity pair",
                max_depth=_MAX_EXPLANATION_DEPTH,
                max_nodes=_MAX_EXPLANATION_NODES,
            )
        except (AttributeError, KeyError, RecursionError, TypeError, ValueError) as error:
            raise ValueError("model sensitivity pair failed safe preflight") from error
        if not request.verify() or not result.verify_against(ranking, request):
            raise ValueError("model sensitivity pair failed external verification")
        recomputed = recompute_model_sensitivity(
            ranking,
            request,
            expected_result_sha256=ranking.result_sha256,
            expected_authoritative_input_sha256=ranking.authoritative_input_sha256,
        )
        if recomputed != result or not hmac.compare_digest(
            recomputed.result_sha256,
            result.result_sha256,
        ):
            raise ValueError("model sensitivity result differs from deterministic recomputation")
        verified.append((request, result))
    ordered = tuple(sorted(verified, key=lambda item: item[0].request_sha256))
    request_digests = tuple(item[0].request_sha256 for item in ordered)
    result_digests = tuple(item[1].result_sha256 for item in ordered)
    scenario_keys = tuple(item[0].scenario_key for item in ordered)
    if (
        len(set(request_digests)) != len(request_digests)
        or len(set(result_digests)) != len(result_digests)
        or len(set(scenario_keys)) != len(scenario_keys)
    ):
        raise ValueError("model sensitivity identities must be unique")
    references = tuple(
        SensitivityArtifactReference(
            scenario_key=request.scenario_key,
            request_sha256=request.request_sha256,
            result_sha256=result.result_sha256,
            top_1_changed=result.top_1_changed,
            top_k_membership_changed=result.top_k_membership_changed,
            analysis_kind="model_behavior_only",
            is_causal_intervention=False,
        )
        for request, result in ordered
    )
    return InterpretabilitySensitivitySummary(
        status="supplied" if references else "not_requested",
        references=references,
        sensitivity_request_sha256s=request_digests,
        sensitivity_result_sha256s=result_digests,
        sensitivity_analysis_kind="model_behavior_only",
        is_causal_intervention=False,
    )


def _candidate_traces(
    *,
    lineage: CaseAuditLineageIndex,
    ranking: ReferenceRankingResult,
    provenance: ProvenancePartitions,
) -> tuple[CandidateInterpretabilityTrace, ...]:
    node_by_id = {item.record_id: item for item in lineage.nodes}
    score_by_candidate = {item.candidate_service_key: item for item in ranking.scores}
    top_k_by_candidate = {item.service_entity_key: item for item in ranking.top_k.entries}
    tie_by_candidate = {
        service_key: tie_group.tie_group_id
        for tie_group in ranking.evidence_tie_groups
        for service_key in tie_group.member_service_keys
    }
    traces: list[CandidateInterpretabilityTrace] = []
    for rank, candidate_service_key in enumerate(
        ranking.ordered_service_keys,
        start=1,
    ):
        score = score_by_candidate[candidate_service_key]
        backward = _walk_lineage(
            start_record_id=score.score_sha256,
            node_by_id=node_by_id,
            direction="sources",
        )
        forward = _walk_lineage(
            start_record_id=score.score_sha256,
            node_by_id=node_by_id,
            direction="consumers",
        )
        top_k_entry = top_k_by_candidate.get(candidate_service_key)
        trace_ids = set(backward) | set(forward)
        traces.append(
            CandidateInterpretabilityTrace(
                candidate_service_key=candidate_service_key,
                final_rank=rank,
                is_top_k=top_k_entry is not None,
                top_k_entry_record_id=(
                    None
                    if top_k_entry is None
                    else f"topk:{top_k_entry.rank}:{candidate_service_key}"
                ),
                score_sha256=score.score_sha256,
                quantized_total_score=score.quantized_total_score,
                quantized_event_evidence_score=score.quantized_event_evidence_score,
                quantized_propagation_score=score.quantized_propagation_score,
                rule_score=score.rule_score,
                history_score=score.history_score,
                event_group_ids=_ids_by_kind(
                    backward,
                    node_by_id=node_by_id,
                    kinds={"event_group"},
                ),
                predicate_result_ids=_ids_by_kind(
                    backward,
                    node_by_id=node_by_id,
                    kinds={"predicate_result"},
                ),
                verification_request_ids=_ids_by_kind(
                    backward,
                    node_by_id=node_by_id,
                    kinds={"verification_request"},
                ),
                verification_result_ids=_ids_by_kind(
                    backward,
                    node_by_id=node_by_id,
                    kinds={"verification_result"},
                ),
                hypothesis_ids=_ids_by_kind(
                    backward,
                    node_by_id=node_by_id,
                    kinds={"diagnostic_hypothesis"},
                ),
                guard_result_ids=_ids_by_kind(
                    backward,
                    node_by_id=node_by_id,
                    kinds={"hypothesis_guard_result"},
                ),
                role_assertion_ids=_ids_by_kind(
                    backward,
                    node_by_id=node_by_id,
                    kinds={"role_assertion"},
                ),
                propagation_path_ids=_ids_by_kind(
                    backward,
                    node_by_id=node_by_id,
                    kinds={"propagation_path_audit"},
                ),
                rule_ids=_ids_by_kind(
                    backward,
                    node_by_id=node_by_id,
                    kinds={"scoring_rule"},
                ),
                history_neighbor_ids=_ids_by_kind(
                    backward,
                    node_by_id=node_by_id,
                    kinds={"history_neighbor_audit"},
                ),
                provenance=_provenance_partitions(
                    accepted=provenance.accepted_provenance_ids,
                    rejected=provenance.rejected_provenance_ids,
                    conflicted=provenance.conflicted_provenance_ids,
                    unknown=provenance.unknown_provenance_ids,
                    allowed_ids=trace_ids,
                ),
                backward_record_ids=backward,
                forward_record_ids=forward,
                evidence_tie_group_id=tie_by_candidate[candidate_service_key],
                stable_tie_break_sha256=_stable_tie_break_sha256(
                    seed_sha256=ranking.config.tie_break_seed_sha256,
                    candidate_service_key=candidate_service_key,
                ),
                tie_break_is_evidence=False,
            )
        )
    return tuple(traces)


def build_interpretability_artifact(
    *,
    audit_package: ImmutableCaseAuditPackage,
    expected_audit_sha256: str,
    sensitivity_pairs: tuple[SensitivityPair, ...],
) -> InterpretabilityArtifact:
    """Build mechanism transparency only from an externally replayed audit."""

    if type(audit_package) is not ImmutableCaseAuditPackage:
        raise ValueError("interpretability requires the exact immutable audit type")
    if type(sensitivity_pairs) is not tuple:
        raise ValueError("sensitivity pairs must use an exact tuple")
    receipt = replay_case_audit(
        audit_package,
        expected_audit_sha256=expected_audit_sha256,
    )
    if receipt.terminal_state != "SEALED" or receipt.ranking_result is None:
        raise ValueError("interpretability requires a successful sealed ranking audit")
    ranking = receipt.ranking_result
    bundle = audit_package.replay_bundle
    lineage = audit_package.lineage_index
    if not lineage.verify_against(bundle):
        raise ValueError("interpretability requires externally replayable lineage")
    ranking_replay = replay_reference_ranking(
        ranking,
        expected_result_sha256=ranking.result_sha256,
        expected_authoritative_input_sha256=ranking.authoritative_input_sha256,
    )
    sealed = audit_package.sealed_audit_package
    provenance = _provenance_partitions(
        accepted=sealed.accepted_provenance_ids,
        rejected=sealed.rejected_provenance_ids,
        conflicted=sealed.conflicted_provenance_ids,
        unknown=sealed.unknown_provenance_ids,
    )
    sensitivity = _build_sensitivity_summary(
        ranking=ranking,
        sensitivity_pairs=sensitivity_pairs,
    )
    unsigned: dict[str, object] = {
        "schema_version": INTERPRETABILITY_ARTIFACT_SCHEMA_VERSION,
        "opaque_case_id": audit_package.opaque_case_id,
        "objective": "interpretability",
        "owner": "small_model",
        "claim_scope": "mechanism_auditability",
        "is_llm_internal_transparency": False,
        "is_human_benefit_evidence": False,
        "source_audit_sha256": audit_package.audit_sha256,
        "source_ranking_bundle_sha256": bundle.bundle_sha256,
        "source_ranking_result_sha256": ranking.result_sha256,
        "source_top_k_sha256": ranking.top_k.ranking_sha256,
        "source_lineage_sha256": lineage.lineage_sha256,
        "ranking_replay_sha256": ranking_replay.replay_sha256,
        "exact_ranking_replay": True,
        "candidate_traces": tuple(
            item.model_dump(mode="json", warnings="error")
            for item in _candidate_traces(
                lineage=lineage,
                ranking=ranking,
                provenance=provenance,
            )
        ),
        "provenance": provenance.model_dump(mode="json", warnings="error"),
        "sensitivity": sensitivity.model_dump(mode="json", warnings="error"),
    }
    result = InterpretabilityArtifact.model_validate(
        {**unsigned, "artifact_sha256": _digest(unsigned)}
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("interpretability artifact verification failed")
    return result


def require_interpretability_artifact(
    artifact: InterpretabilityArtifact,
    *,
    audit_package: ImmutableCaseAuditPackage,
    expected_audit_sha256: str,
    sensitivity_pairs: tuple[SensitivityPair, ...],
) -> InterpretabilityArtifact:
    """Require exact identity after rebuilding from the frozen ranking inputs."""

    if type(artifact) is not InterpretabilityArtifact or not artifact.verify():
        raise ValueError("interpretability artifact failed exact verification")
    expected = build_interpretability_artifact(
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
        sensitivity_pairs=sensitivity_pairs,
    )
    if not hmac.compare_digest(artifact.artifact_sha256, expected.artifact_sha256) or (
        artifact != expected
    ):
        raise ValueError("interpretability artifact differs from rebuilt frozen inputs")
    return InterpretabilityArtifact.model_validate(
        artifact.model_dump(mode="json", warnings="error")
    )


def _statement_partitions(
    catalog: ReportGroundingCatalog,
    *,
    selected_statement_ids: set[str] | None = None,
) -> StatementIdPartitions:
    by_type: dict[StatementType, list[str]] = {
        "observation": [],
        "inference": [],
        "recommendation": [],
    }
    catalog_ids = {item.statement_id for item in catalog.anchors}
    if selected_statement_ids is not None and not selected_statement_ids <= catalog_ids:
        raise ValueError("delivered statement leaves the safe report catalog")
    for anchor in catalog.anchors:
        if selected_statement_ids is None or anchor.statement_id in selected_statement_ids:
            by_type[anchor.statement_type].append(anchor.statement_id)
    return StatementIdPartitions(
        observation_statement_ids=tuple(sorted(by_type["observation"])),
        inference_statement_ids=tuple(sorted(by_type["inference"])),
        recommendation_statement_ids=tuple(sorted(by_type["recommendation"])),
    )


def _canonical_union(values: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    return tuple(sorted({item for collection in values for item in collection}))


def _baseline_generation_projection() -> ReportGenerationProjection:
    return ReportGenerationProjection(
        outcome="deterministic_baseline",
        generation_audit_sha256=None,
        failure_audit_sha256=None,
        failure_code=None,
        accepted_count=None,
        weakened_count=None,
        rejected_count=None,
        raw_violation_count=None,
        grounded_statement_ids=(),
        violation_references=(),
    )


def _grounded_generation_projection(
    audit: ReportGenerationAudit,
) -> ReportGenerationProjection:
    safe_statement_ids = tuple(
        sorted(
            {
                decision.grounded_statement_id
                for decision in audit.grounding_validation.decisions
                if decision.disposition in {"accepted", "weakened"}
                and decision.grounded_statement_id is not None
            }
        )
    )
    if not safe_statement_ids:
        raise ValueError(
            "all generated statements were rejected; grounding_failed fallback is required"
        )
    return ReportGenerationProjection(
        outcome="grounded_generation",
        generation_audit_sha256=audit.audit_sha256,
        failure_audit_sha256=None,
        failure_code=None,
        accepted_count=audit.grounding_validation.accepted_count,
        weakened_count=audit.weakened_count,
        rejected_count=audit.rejected_count,
        raw_violation_count=audit.raw_violation_count,
        grounded_statement_ids=safe_statement_ids,
        violation_references=tuple(
            GenerationViolationReference(
                position=item.position,
                disposition=item.disposition,
                violation_codes=item.violation_codes,
            )
            for item in audit.raw_violations
        ),
    )


def _fallback_generation_projection(
    failure: ReportGenerationFailureAudit,
) -> ReportGenerationProjection:
    return ReportGenerationProjection(
        outcome="deterministic_fallback",
        generation_audit_sha256=None,
        failure_audit_sha256=failure.failure_sha256,
        failure_code=failure.failure_code,
        accepted_count=None,
        weakened_count=None,
        rejected_count=None,
        raw_violation_count=None,
        grounded_statement_ids=(),
        violation_references=(),
    )


def build_explainability_artifact(
    *,
    report: DeterministicSREReport,
    context: FrozenReportContext,
    audit_package: ImmutableCaseAuditPackage,
    expected_audit_sha256: str,
    generation_audit: ReportGenerationAudit | None,
    generation_proposals: tuple[ReportSentenceProposal, ...],
    generation_failure_audit: ReportGenerationFailureAudit | None,
    generation_failure_detail: str | None,
) -> ExplainabilityArtifact:
    """Project frozen report comprehension without changing ranking facts."""

    if type(generation_proposals) is not tuple:
        raise ValueError("report generation proposals must use an exact tuple")
    success_requested = generation_audit is not None or bool(generation_proposals)
    failure_requested = (
        generation_failure_audit is not None or generation_failure_detail is not None
    )
    if success_requested and failure_requested:
        raise ValueError("generation success and failure audits are mutually exclusive")
    if success_requested and (generation_audit is None or not generation_proposals):
        raise ValueError("generation audit and non-empty proposals must be supplied together")
    if failure_requested and (
        generation_failure_audit is None
        or type(generation_failure_detail) is not str
        or not generation_failure_detail
    ):
        raise ValueError("generation failure audit and non-empty detail must be supplied together")
    verified = require_deterministic_sre_report(
        report,
        context=context,
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
    )
    catalog = build_report_grounding_catalog(
        report=verified,
        context=context,
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
    )
    available = _statement_partitions(catalog)
    if not success_requested and not failure_requested:
        generation = _baseline_generation_projection()
        delivered = available
    elif success_requested:
        assert generation_audit is not None
        verified_generation = require_report_generation_audit(
            generation_audit,
            proposals=generation_proposals,
            report=verified,
            context=context,
            audit_package=audit_package,
            expected_audit_sha256=expected_audit_sha256,
        )
        generation = _grounded_generation_projection(verified_generation)
        delivered = _statement_partitions(
            catalog,
            selected_statement_ids=set(generation.grounded_statement_ids),
        )
    else:
        assert generation_failure_audit is not None
        assert generation_failure_detail is not None
        verified_failure = require_report_generation_failure_audit(
            generation_failure_audit,
            report=verified,
            context=context,
            audit_package=audit_package,
            expected_audit_sha256=expected_audit_sha256,
            failure_code=generation_failure_audit.failure_code,
            failure_detail=generation_failure_detail,
        )
        generation = _fallback_generation_projection(verified_failure)
        delivered = available
    evidence = tuple(item.evidence for item in verified.top_k)
    supporting = _canonical_union(
        tuple(
            (
                *item.supporting_event_group_ids,
                *item.supporting_hypothesis_ids,
                *item.supporting_predicate_result_ids,
            )
            for item in evidence
        )
    )
    unsigned: dict[str, object] = {
        "schema_version": EXPLAINABILITY_ARTIFACT_SCHEMA_VERSION,
        "opaque_case_id": verified.opaque_case_id,
        "objective": "explainability",
        "owner": "report_layer",
        "claim_scope": "frozen_result_comprehension_and_action_support",
        "is_llm_internal_transparency": False,
        "is_human_benefit_evidence": False,
        "source_audit_sha256": verified.source_audit_sha256,
        "source_context_sha256": verified.context_sha256,
        "source_report_sha256": verified.report_sha256,
        "source_top_k_sha256": verified.source_top_k_sha256,
        "top_k_statement_id": verified.top_k_statement.statement_id,
        "top_1_basis_statement_id": verified.top_1_basis.statement.statement_id,
        "competition_statement_ids": tuple(
            sorted(item.statement.statement_id for item in verified.competition)
        ),
        "supporting_reference_ids": supporting,
        "refuting_reference_ids": _canonical_union(
            tuple(item.refuting_predicate_result_ids for item in evidence)
        ),
        "conflicting_reference_ids": _canonical_union(
            tuple(item.conflicting_assertion_ids for item in evidence)
        ),
        "missing_reference_ids": _canonical_union(
            tuple(item.missing_reason_codes for item in evidence)
        ),
        "unknown_reference_ids": _canonical_union(
            tuple(item.unknown_predicate_result_ids for item in evidence)
        ),
        "run_mode_statement_id": verified.run_mode_statement.statement_id,
        "degradation_reason_codes": verified.degradation_reason_codes,
        "limitation_statement_ids": tuple(
            sorted(item.statement.statement_id for item in verified.limits)
        ),
        "available_statements": available.model_dump(mode="json", warnings="error"),
        "delivered_statements": delivered.model_dump(mode="json", warnings="error"),
        "verification_actions": tuple(
            item.model_dump(mode="json", warnings="error") for item in verified.verification_actions
        ),
        "conditional_mitigations": tuple(
            item.model_dump(mode="json", warnings="error")
            for item in verified.conditional_mitigations
        ),
        "generation": generation.model_dump(mode="json", warnings="error"),
        "ranking_mutable": False,
        "adds_ranking_facts": False,
    }
    result = ExplainabilityArtifact.model_validate(
        {**unsigned, "artifact_sha256": _digest(unsigned)}
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("explainability artifact verification failed")
    return result


def require_explainability_artifact(
    artifact: ExplainabilityArtifact,
    *,
    report: DeterministicSREReport,
    context: FrozenReportContext,
    audit_package: ImmutableCaseAuditPackage,
    expected_audit_sha256: str,
    generation_audit: ReportGenerationAudit | None,
    generation_proposals: tuple[ReportSentenceProposal, ...],
    generation_failure_audit: ReportGenerationFailureAudit | None,
    generation_failure_detail: str | None,
) -> ExplainabilityArtifact:
    """Require exact identity after rebuilding from frozen report inputs."""

    if type(artifact) is not ExplainabilityArtifact or not artifact.verify():
        raise ValueError("explainability artifact failed exact verification")
    expected = build_explainability_artifact(
        report=report,
        context=context,
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
        generation_audit=generation_audit,
        generation_proposals=generation_proposals,
        generation_failure_audit=generation_failure_audit,
        generation_failure_detail=generation_failure_detail,
    )
    if not hmac.compare_digest(artifact.artifact_sha256, expected.artifact_sha256) or (
        artifact != expected
    ):
        raise ValueError("explainability artifact differs from rebuilt frozen inputs")
    return ExplainabilityArtifact.model_validate(artifact.model_dump(mode="json", warnings="error"))


def build_dual_explanation_manifest(
    *,
    interpretability: InterpretabilityArtifact,
    explainability: ExplainabilityArtifact,
    audit_package: ImmutableCaseAuditPackage,
    expected_audit_sha256: str,
    sensitivity_pairs: tuple[SensitivityPair, ...],
    report: DeterministicSREReport,
    context: FrozenReportContext,
    generation_audit: ReportGenerationAudit | None,
    generation_proposals: tuple[ReportSentenceProposal, ...],
    generation_failure_audit: ReportGenerationFailureAudit | None,
    generation_failure_detail: str | None,
) -> DualExplanationManifest:
    """Bind independent explanation identities without combining their claims."""

    verified_interpretability = require_interpretability_artifact(
        interpretability,
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
        sensitivity_pairs=sensitivity_pairs,
    )
    verified_explainability = require_explainability_artifact(
        explainability,
        report=report,
        context=context,
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
        generation_audit=generation_audit,
        generation_proposals=generation_proposals,
        generation_failure_audit=generation_failure_audit,
        generation_failure_detail=generation_failure_detail,
    )
    anchors = {
        (
            verified_interpretability.opaque_case_id,
            verified_interpretability.source_audit_sha256,
            verified_interpretability.source_top_k_sha256,
        ),
        (
            verified_explainability.opaque_case_id,
            verified_explainability.source_audit_sha256,
            verified_explainability.source_top_k_sha256,
        ),
    }
    if len(anchors) != 1:
        raise ValueError("explanation layers do not share case, audit, and Top-K anchors")
    opaque_case_id, source_audit_sha256, source_top_k_sha256 = anchors.pop()
    unsigned: dict[str, object] = {
        "schema_version": DUAL_EXPLANATION_MANIFEST_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "source_audit_sha256": source_audit_sha256,
        "source_top_k_sha256": source_top_k_sha256,
        "interpretability_sha256": verified_interpretability.artifact_sha256,
        "explainability_sha256": verified_explainability.artifact_sha256,
        "separation_policy": "separate_non_compensating",
        "combined_score": None,
        "combined_pass": None,
        "cross_layer_compensation_allowed": False,
    }
    result = DualExplanationManifest.model_validate(
        {**unsigned, "manifest_sha256": _digest(unsigned)}
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("dual explanation manifest verification failed")
    return result


def require_dual_explanation_manifest(
    manifest: DualExplanationManifest,
    *,
    interpretability: InterpretabilityArtifact,
    explainability: ExplainabilityArtifact,
    audit_package: ImmutableCaseAuditPackage,
    expected_audit_sha256: str,
    sensitivity_pairs: tuple[SensitivityPair, ...],
    report: DeterministicSREReport,
    context: FrozenReportContext,
    generation_audit: ReportGenerationAudit | None,
    generation_proposals: tuple[ReportSentenceProposal, ...],
    generation_failure_audit: ReportGenerationFailureAudit | None,
    generation_failure_detail: str | None,
) -> DualExplanationManifest:
    """Require exact manifest identity after rebuilding both explanation layers."""

    if type(manifest) is not DualExplanationManifest or not manifest.verify():
        raise ValueError("dual explanation manifest failed exact verification")
    expected = build_dual_explanation_manifest(
        interpretability=interpretability,
        explainability=explainability,
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
        sensitivity_pairs=sensitivity_pairs,
        report=report,
        context=context,
        generation_audit=generation_audit,
        generation_proposals=generation_proposals,
        generation_failure_audit=generation_failure_audit,
        generation_failure_detail=generation_failure_detail,
    )
    if not hmac.compare_digest(manifest.manifest_sha256, expected.manifest_sha256) or (
        manifest != expected
    ):
        raise ValueError("dual explanation manifest differs from rebuilt frozen inputs")
    return DualExplanationManifest.model_validate(
        manifest.model_dump(mode="json", warnings="error")
    )


__all__ = [
    "DUAL_EXPLANATION_MANIFEST_SCHEMA_VERSION",
    "EXPLAINABILITY_ARTIFACT_SCHEMA_VERSION",
    "INTERPRETABILITY_ARTIFACT_SCHEMA_VERSION",
    "CandidateInterpretabilityTrace",
    "DualExplanationManifest",
    "ExplainabilityArtifact",
    "GenerationViolationReference",
    "InterpretabilityArtifact",
    "InterpretabilitySensitivitySummary",
    "ProvenancePartitions",
    "ReportGenerationProjection",
    "SensitivityArtifactReference",
    "SensitivityPair",
    "StatementIdPartitions",
    "build_dual_explanation_manifest",
    "build_explainability_artifact",
    "build_interpretability_artifact",
    "require_dual_explanation_manifest",
    "require_explainability_artifact",
    "require_interpretability_artifact",
]
