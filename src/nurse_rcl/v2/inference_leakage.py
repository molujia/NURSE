"""Deterministic, provenance-aware leakage audit for V2 inference artifacts.

This is an in-process inference boundary, not the outbound ARK preflight.  It
accepts only explicitly declared public artifact models or allowlisted mapping
fields, and it binds every case identity to an evaluator-provisioned public
identity manifest.  It deliberately does not scan free-form telemetry values:
label and source isolation is enforced from typed field provenance.
"""

from __future__ import annotations

import hmac
import re
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.aiops25_adapter import (
    AIOps25AdapterConfiguration,
    AIOps25AdapterResult,
    AIOps25SourceRejection,
)
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.case_audit import CaseAuditLineageIndex, CaseAuditLineageNode
from nurse_rcl.v2.context_expansion import (
    CardExpansionBatchResult,
    CardExpansionDecision,
    CardExpansionPolicy,
    CardExpansionRequest,
    CardExpansionRequestCount,
)
from nurse_rcl.v2.coverage_diagnostics import (
    CoverageStageDiagnostic,
    LayeredCandidateCoverageDiagnostics,
)
from nurse_rcl.v2.diagnostic_schemas import (
    DiagnosticHypothesis,
    GuardDecision,
    MeasuredValue,
    MechanismDefinition,
    PredicateResult,
    PredicateThreshold,
    RoleAssertion,
    VerificationRequest,
    VerificationResult,
)
from nurse_rcl.v2.event_grouping import (
    AdapterAssociationIdentity,
    AdapterAssociationSourceField,
    AssociationClaim,
    AssociationDerivationRule,
    AssociationSourceFieldHash,
    AssociationSourceFieldParserSpec,
    CanonicalAssociationSourceRecord,
    EventGroup,
    EventGroupingResult,
    FrozenAssociationPolicy,
    FrozenSignalFamilyMapping,
    GroupingDecision,
    MemberSignalFamilyAssignment,
    ObservationAssociationProof,
    SignalFamilyMappingEntry,
)
from nurse_rcl.v2.event_strength import (
    CandidateEventRoleAdjustment,
    EventGroupBaseStrength,
    EventStrengthResult,
)
from nurse_rcl.v2.family_aggregation import (
    CandidateFamilyAggregation,
    FamilyAggregationResult,
    SignalFamilyMaximum,
    UnboundEventGroupAudit,
)
from nurse_rcl.v2.graph_snapshots import (
    CanonicalGraphCoverageInput,
    CanonicalGraphSourceInput,
    FrozenGraphInputManifest,
    GraphCoverageSummary,
    GraphEdgeChange,
    GraphInputDecision,
    GraphMetricSummary,
    GraphSnapshot,
    GraphSnapshotEdge,
    GraphSnapshotPair,
    GraphSourceRecord,
    GraphWindowCoverage,
    _trusted_graph_snapshot_pair,
)
from nurse_rcl.v2.history_scoring import (
    CandidateHistoryScore,
    HistoryEvaluationResult,
    HistoryHintReference,
    HistoryNeighborAudit,
    HistoryScoreContribution,
)
from nurse_rcl.v2.hypothesis_guard import (
    HypothesisGuardResult,
    ScoreImpactCompilation,
)
from nurse_rcl.v2.mechanism_library import MechanismLibrary, verify_declared_mechanism
from nurse_rcl.v2.observation_schemas import (
    CandidateEntry,
    CandidateUniverse,
    CaseTimeContract,
    EntityBinding,
    EvidenceCard,
    FocusDecision,
    Observation,
    SemanticFocusSet,
    TimeInterval,
    WindowSummary,
)
from nurse_rcl.v2.opaque_ids import (
    EvaluatorIdentityMap,
    EvaluatorIdentityRecord,
    EvaluatorIdentitySource,
    InferenceIdentityManifest,
    InferenceIdentityRecord,
)
from nurse_rcl.v2.propagation_scoring import (
    CandidatePropagationScore,
    PropagationHopAudit,
    PropagationPathAudit,
    PropagationRelationBinding,
    PropagationScoringConfig,
    PropagationScoringResult,
    PropagationTargetScore,
)
from nurse_rcl.v2.query_session import (
    CaseQuerySessionCoordinator,
    CaseQuerySessionResult,
    require_case_query_session_replay,
)
from nurse_rcl.v2.ranking_schemas import (
    CondensedEdge,
    EvidenceTieGroup,
    HistoryContext,
    HistoryNeighbor,
    RuleSet,
    SCCComponent,
    SCCProvenance,
    ScoreDecomposition,
    ScoringRule,
    TopKArtifact,
    TopKEntry,
)
from nurse_rcl.v2.ranking_schemas import (
    FrozenReportContext as LegacyFrozenReportContextV2,
)
from nurse_rcl.v2.rcabench_adapter import (
    RCABenchAdapterConfiguration,
    RCABenchAdapterResult,
    RCABenchSourceAudit,
)
from nurse_rcl.v2.reference_ranking import ReferenceRankingResult, ReferenceScoringConfig
from nurse_rcl.v2.report_context import (
    FrozenReportContext,
    FrozenReportPayload,
    ReportCompetitionDelta,
    ReportHypothesisFact,
)
from nurse_rcl.v2.report_evidence_schemas import (
    ReportEvidenceFact,
    ReportEvidenceOmission,
    ReportEvidenceValue,
    ReportEvidenceView,
)
from nurse_rcl.v2.role_compiler import RoleBindingProposal, RoleCompilationResult
from nurse_rcl.v2.rule_hypotheses import (
    RULE_GENERATOR_VERSION,
    PendingVerificationPlan,
    RuleHypothesisEnvelope,
    RuleHypothesisSet,
)
from nurse_rcl.v2.rule_scoring import CandidateRuleScore, RuleContribution, RuleScoringResult
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree
from nurse_rcl.v2.semantic_context import (
    CompactCandidateDirectory,
    CompactCandidateDirectoryEntry,
    DetailedEvidenceCard,
    SemanticContextBundle,
)
from nurse_rcl.v2.temporal_condensation import (
    FrozenClockResolution,
    OnsetInterval,
    RelationCondensedEdge,
    RelationSCCComponent,
    RelationSCCCondensation,
    TemporalPrecedence,
    build_onset_interval_from_derived_observation,
    case_time_contract_sha256,
    require_frozen_clock_resolution,
)
from nurse_rcl.v2.temporal_entity_registry import (
    EntityMetadataRecord,
    ObservationBindingResolution,
    ObservationEntityInput,
    TemporalEntityRegistry,
)
from nurse_rcl.v2.verification_executor import (
    DerivedVerificationObservation,
    VerificationBatchResult,
    VerificationExecutionDecision,
    VerificationExecutionPolicy,
    VerificationExecutionScope,
    VerificationFieldValue,
    VerificationOperatorPermission,
    VerificationRequestCount,
)

INFERENCE_BOUNDARY_AUDIT_SCHEMA_VERSION: Literal["nurse.v2.inference-boundary-audit/1"] = (
    "nurse.v2.inference-boundary-audit/1"
)

LeakageCode = Literal[
    "invalid_identity_manifest",
    "private_evaluator_artifact",
    "unapproved_model_type",
    "invalid_public_artifact",
    "forbidden_field_provenance",
    "unapproved_mapping_field",
    "unsupported_value_type",
    "untrusted_opaque_identity",
    "collector_only_service",
    "unproven_service",
    "cyclic_artifact",
]


class InferenceBoundaryLeakageError(ValueError):
    """Safe boundary rejection that never includes a submitted value."""

    def __init__(self, code: LeakageCode, location: str) -> None:
        self.code = code
        self.location = location
        super().__init__(f"inference boundary rejected ({code}) at {location}")


class InferenceBoundaryAudit(StrictFrozenModel):
    """Secret-free receipt for one deterministic public-artifact audit."""

    schema_version: Literal["nurse.v2.inference-boundary-audit/1"] = (
        INFERENCE_BOUNDARY_AUDIT_SCHEMA_VERSION
    )
    artifact_count: int = Field(gt=0)
    opaque_case_ids: tuple[str, ...]
    public_model_types: tuple[str, ...]
    identity_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    boundary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        """Verify canonical ordering and the receipt's content digest."""

        try:
            revalidated = InferenceBoundaryAudit.model_validate(
                self.model_dump(mode="python", warnings="error")
            )
            if revalidated != self:
                return False
            if self.opaque_case_ids != tuple(sorted(set(self.opaque_case_ids))):
                return False
            if self.public_model_types != tuple(sorted(set(self.public_model_types))):
                return False
            unsigned = self.model_dump(mode="json", exclude={"audit_sha256"}, warnings="error")
            return self.audit_sha256 == sha256_hex(canonical_json_bytes(unsigned))
        except (AttributeError, TypeError, ValueError):
            return False


# Full grouping lineage is public only inside a replay-verified B0 envelope.
# Do not make these inputs or compact B0 outputs independently consumable roots.
_RULE_ENVELOPE_ONLY_MODEL_TYPES = (
    FrozenCandidateUniverse,
    AdapterAssociationIdentity,
    AdapterAssociationSourceField,
    AssociationClaim,
    AssociationDerivationRule,
    AssociationSourceFieldHash,
    AssociationSourceFieldParserSpec,
    CanonicalAssociationSourceRecord,
    EventGroupingResult,
    FrozenAssociationPolicy,
    FrozenSignalFamilyMapping,
    GroupingDecision,
    ObservationAssociationProof,
    SignalFamilyMappingEntry,
    PendingVerificationPlan,
    RuleHypothesisSet,
)
_REPORT_EVIDENCE_MODEL_TYPES = (
    ReportEvidenceView,
    ReportEvidenceFact,
    ReportEvidenceValue,
    ReportEvidenceOmission,
)
_PUBLIC_MODEL_TYPES = (
    InferenceIdentityRecord,
    TimeInterval,
    CaseTimeContract,
    WindowSummary,
    Observation,
    EntityBinding,
    MemberSignalFamilyAssignment,
    EventGroup,
    CandidateEntry,
    CandidateUniverse,
    FocusDecision,
    SemanticFocusSet,
    EvidenceCard,
    CompactCandidateDirectoryEntry,
    CompactCandidateDirectory,
    DetailedEvidenceCard,
    SemanticContextBundle,
    RCABenchAdapterConfiguration,
    RCABenchSourceAudit,
    RCABenchAdapterResult,
    AIOps25AdapterConfiguration,
    AIOps25SourceRejection,
    AIOps25AdapterResult,
    EntityMetadataRecord,
    ObservationEntityInput,
    ObservationBindingResolution,
    TemporalEntityRegistry,
    MechanismDefinition,
    MechanismLibrary,
    MeasuredValue,
    PredicateThreshold,
    PredicateResult,
    VerificationRequest,
    VerificationResult,
    DiagnosticHypothesis,
    RuleHypothesisEnvelope,
    GuardDecision,
    SCCComponent,
    CondensedEdge,
    SCCProvenance,
    ScoreDecomposition,
    EvidenceTieGroup,
    TopKEntry,
    TopKArtifact,
    ScoringRule,
    RuleSet,
    HistoryNeighbor,
    HistoryContext,
    LegacyFrozenReportContextV2,
    FrozenReportContext,
    *_REPORT_EVIDENCE_MODEL_TYPES,
    *_RULE_ENVELOPE_ONLY_MODEL_TYPES,
)
_PUBLIC_MODEL_FIELD_ALLOWLIST: dict[type[BaseModel], frozenset[str]] = {
    model_type: frozenset(model_type.model_fields) for model_type in _PUBLIC_MODEL_TYPES
}
_REPLAYED_VERIFICATION_MODEL_FIELD_ALLOWLIST: dict[type[BaseModel], frozenset[str]] = {
    model_type: frozenset(model_type.model_fields)
    for model_type in (VerificationFieldValue, DerivedVerificationObservation)
}
_REPLAYED_HYPOTHESIS_GUARD_MODEL_FIELD_ALLOWLIST: dict[type[BaseModel], frozenset[str]] = {
    ScoreImpactCompilation: frozenset(ScoreImpactCompilation.model_fields)
}
_REPLAYED_ROLE_COMPILATION_MODEL_FIELD_ALLOWLIST: dict[type[BaseModel], frozenset[str]] = {
    model_type: frozenset(model_type.model_fields)
    for model_type in (RoleBindingProposal, RoleCompilationResult, RoleAssertion)
}
_REPLAYED_EVENT_STRENGTH_MODEL_FIELD_ALLOWLIST: dict[type[BaseModel], frozenset[str]] = {
    model_type: frozenset(model_type.model_fields)
    for model_type in (
        CandidateEventRoleAdjustment,
        EventGroupBaseStrength,
        EventStrengthResult,
    )
}
_REPLAYED_FAMILY_AGGREGATION_MODEL_FIELD_ALLOWLIST: dict[type[BaseModel], frozenset[str]] = {
    model_type: frozenset(model_type.model_fields)
    for model_type in (
        CandidateFamilyAggregation,
        FamilyAggregationResult,
        SignalFamilyMaximum,
        UnboundEventGroupAudit,
    )
}
_REPLAYED_GRAPH_SNAPSHOT_MODEL_FIELD_ALLOWLIST: dict[type[BaseModel], frozenset[str]] = {
    model_type: frozenset(model_type.model_fields)
    for model_type in (
        GraphCoverageSummary,
        GraphEdgeChange,
        GraphInputDecision,
        GraphMetricSummary,
        GraphSnapshot,
        GraphSnapshotEdge,
        GraphSnapshotPair,
    )
}
_REPLAYED_RELATION_CONDENSATION_MODEL_FIELD_ALLOWLIST: dict[type[BaseModel], frozenset[str]] = {
    model_type: frozenset(model_type.model_fields)
    for model_type in (
        GraphCoverageSummary,
        GraphMetricSummary,
        GraphSnapshotEdge,
        RelationCondensedEdge,
        RelationSCCComponent,
        RelationSCCCondensation,
    )
}
_REPLAYED_ONSET_INTERVAL_MODEL_FIELD_ALLOWLIST: dict[type[BaseModel], frozenset[str]] = {
    OnsetInterval: frozenset(OnsetInterval.model_fields)
}
_REPLAYED_TEMPORAL_PRECEDENCE_MODEL_FIELD_ALLOWLIST: dict[type[BaseModel], frozenset[str]] = {
    TemporalPrecedence: frozenset(TemporalPrecedence.model_fields)
}
_FROZEN_REPORT_PAYLOAD_MODEL_TYPES = (
    FrozenReportPayload,
    ReportHypothesisFact,
    ReportCompetitionDelta,
    CaseAuditLineageIndex,
    CaseAuditLineageNode,
    CardExpansionBatchResult,
    CardExpansionDecision,
    CardExpansionPolicy,
    CardExpansionRequest,
    CardExpansionRequestCount,
    MeasuredValue,
    MechanismDefinition,
    PredicateResult,
    PredicateThreshold,
    RoleAssertion,
    VerificationRequest,
    VerificationResult,
    AdapterAssociationIdentity,
    AdapterAssociationSourceField,
    AssociationClaim,
    AssociationDerivationRule,
    AssociationSourceFieldHash,
    AssociationSourceFieldParserSpec,
    CanonicalAssociationSourceRecord,
    EventGroup,
    EventGroupingResult,
    CandidateEventRoleAdjustment,
    EventGroupBaseStrength,
    EventStrengthResult,
    FrozenAssociationPolicy,
    FrozenSignalFamilyMapping,
    GroupingDecision,
    MemberSignalFamilyAssignment,
    ObservationAssociationProof,
    SignalFamilyMappingEntry,
    CandidateFamilyAggregation,
    FamilyAggregationResult,
    SignalFamilyMaximum,
    UnboundEventGroupAudit,
    GraphCoverageSummary,
    GraphMetricSummary,
    GraphSnapshot,
    GraphSnapshotEdge,
    CandidateHistoryScore,
    HistoryEvaluationResult,
    HistoryHintReference,
    HistoryNeighborAudit,
    HistoryScoreContribution,
    MechanismLibrary,
    CandidateEntry,
    CandidateUniverse,
    CaseTimeContract,
    EntityBinding,
    EvidenceCard,
    FocusDecision,
    Observation,
    SemanticFocusSet,
    TimeInterval,
    WindowSummary,
    CandidatePropagationScore,
    PropagationHopAudit,
    PropagationPathAudit,
    PropagationRelationBinding,
    PropagationScoringConfig,
    PropagationScoringResult,
    PropagationTargetScore,
    CaseQuerySessionResult,
    EvidenceTieGroup,
    RuleSet,
    ScoreDecomposition,
    ScoringRule,
    TopKArtifact,
    TopKEntry,
    ReferenceRankingResult,
    ReferenceScoringConfig,
    RoleBindingProposal,
    RoleCompilationResult,
    ScoreImpactCompilation,
    CandidateRuleScore,
    RuleContribution,
    RuleScoringResult,
    DetailedEvidenceCard,
    OnsetInterval,
    RelationCondensedEdge,
    RelationSCCComponent,
    RelationSCCCondensation,
    TemporalPrecedence,
    EntityMetadataRecord,
    ObservationBindingResolution,
    TemporalEntityRegistry,
    DerivedVerificationObservation,
    VerificationBatchResult,
    VerificationExecutionDecision,
    VerificationExecutionPolicy,
    VerificationExecutionScope,
    VerificationFieldValue,
    VerificationOperatorPermission,
    VerificationRequestCount,
)
_FROZEN_REPORT_PAYLOAD_MODEL_FIELD_ALLOWLIST: dict[type[BaseModel], frozenset[str]] = {
    model_type: frozenset(model_type.model_fields)
    for model_type in _FROZEN_REPORT_PAYLOAD_MODEL_TYPES
}
_IDENTITY_MANIFEST_FIELDS = frozenset(InferenceIdentityManifest.model_fields)
_PRIVATE_EVALUATOR_TYPES = (
    EvaluatorIdentitySource,
    EvaluatorIdentityRecord,
    EvaluatorIdentityMap,
    CoverageStageDiagnostic,
    LayeredCandidateCoverageDiagnostics,
)
_INFERENCE_ENVELOPE_FIELDS = frozenset(
    {
        "candidate_directory",
        "dataset",
        "decision_time",
        "entities",
        "evidence_cards",
        "graph_snapshot",
        "mechanism_library",
        "observations",
        "opaque_case_id",
        "verification_results",
    }
)
_FORBIDDEN_FIELD_TOKENS: tuple[tuple[str, ...], ...] = (
    ("answer", "groundtruth", "targetservice", "rootcause"),
    ("injection", "injectedfault", "faulttype", "faultmetadata", "faultname"),
    (
        "originalcase",
        "originalid",
        "originalname",
        "casename",
        "sourcepath",
        "originalpath",
        "casepath",
        "labelpath",
    ),
)


@dataclass(slots=True)
class _AuditState:
    trusted_dataset_by_id: dict[str, str]
    in_replayed_hypothesis_guard: bool = False
    in_replayed_role_compilation: bool = False
    in_replayed_event_strength: bool = False
    in_replayed_family_aggregation: bool = False
    in_replayed_graph_snapshot: bool = False
    in_replayed_relation_condensation: bool = False
    in_replayed_onset_interval: bool = False
    in_replayed_temporal_precedence: bool = False
    in_replayed_verification: bool = False
    in_rule_envelope: bool = False
    in_frozen_report_payload: bool = False
    in_report_evidence_view: bool = False
    scope_case_id: str | None = None
    scope_dataset: str | None = None
    opaque_case_ids: set[str] = field(default_factory=set)
    public_model_types: set[str] = field(default_factory=set)
    collector_labels: set[str] = field(default_factory=set)
    explicit_service_keys: set[str] = field(default_factory=set)
    candidate_service_keys: set[str] = field(default_factory=set)
    chosen_service_keys: set[str] = field(default_factory=set)


def audit_inference_boundary(
    artifact: object,
    *,
    identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str,
) -> InferenceBoundaryAudit:
    """Audit public inference artifacts without accessing evaluator secrets.

    ``expected_identity_manifest_sha256`` must come from the protected run
    manifest held by the isolated evaluator, never from the current payload.
    Artifact-level ``method`` strings or opaque-ID syntax never grant trust:
    every encountered case ID must be a member of the pinned, verified public
    identity manifest.
    """

    return _audit_inference_boundary(
        artifact,
        identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256,
        in_replayed_hypothesis_guard=False,
        in_replayed_role_compilation=False,
        in_replayed_verification=False,
    )


def audit_replayed_verification_boundary(
    artifact: CaseQuerySessionResult,
    *,
    query_session_coordinator: CaseQuerySessionCoordinator,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str,
    semantic_context: SemanticContextBundle | None = None,
    card_expansion_policy: CardExpansionPolicy | None = None,
) -> InferenceBoundaryAudit:
    """Audit derived values only from the replayed current authoritative session."""

    current_session = query_session_coordinator.require_current(artifact)
    replayed_session = require_case_query_session_replay(
        session=current_session,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        semantic_context=semantic_context,
        card_expansion_policy=card_expansion_policy,
    )
    replayed_batch = replayed_session.verification_result
    if replayed_batch is None:
        raise ValueError("case query session has no verification result")
    if not replayed_batch.derived_observations:
        raise ValueError("verification query ledger has no derived observations")
    return _audit_inference_boundary(
        replayed_batch.derived_observations,
        identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256,
        in_replayed_hypothesis_guard=False,
        in_replayed_role_compilation=False,
        in_replayed_verification=True,
    )


def audit_replayed_hypothesis_guard_boundary(
    artifact: HypothesisGuardResult,
    *,
    submitted_record: object,
    predicates: Sequence[PredicateResult],
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    mechanism_library: MechanismLibrary,
    scope_interval: TimeInterval,
    identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str,
    query_session: CaseQuerySessionResult | None = None,
    query_session_coordinator: CaseQuerySessionCoordinator | None = None,
    semantic_context: SemanticContextBundle | None = None,
    card_expansion_policy: CardExpansionPolicy | None = None,
) -> InferenceBoundaryAudit:
    """Admit a score-impact plan only after exact four-layer guard replay."""

    if type(artifact) is not HypothesisGuardResult or not artifact.verify_inputs(
        submitted_record=submitted_record,
        predicates=predicates,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        mechanism_library=mechanism_library,
        scope_interval=scope_interval,
        query_session=query_session,
        query_session_coordinator=query_session_coordinator,
        semantic_context=semantic_context,
        card_expansion_policy=card_expansion_policy,
    ):
        raise ValueError("hypothesis guard result does not replay from frozen inputs")
    score_impact = artifact.score_impact
    if artifact.final_disposition != "accepted" or score_impact is None:
        raise ValueError("hypothesis guard result has no accepted score impact")
    return _audit_inference_boundary(
        score_impact,
        identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256,
        in_replayed_hypothesis_guard=True,
        in_replayed_role_compilation=False,
        in_replayed_verification=False,
    )


def audit_replayed_role_compilation_boundary(
    artifact: RoleCompilationResult,
    *,
    proposals: Sequence[RoleBindingProposal],
    guard_results: Sequence[HypothesisGuardResult],
    predicates: Sequence[PredicateResult],
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    mechanism_library: MechanismLibrary,
    scope_interval: TimeInterval,
    identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str,
    query_sessions_by_guard_result_id: Mapping[str, CaseQuerySessionResult] | None = None,
    query_session_coordinator: CaseQuerySessionCoordinator | None = None,
    semantic_context: SemanticContextBundle | None = None,
    card_expansion_policy: CardExpansionPolicy | None = None,
) -> InferenceBoundaryAudit:
    """Admit role effects only after exact compilation from frozen inputs."""

    if type(artifact) is not RoleCompilationResult or not artifact.verify_inputs(
        proposals=proposals,
        guard_results=guard_results,
        predicates=predicates,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        mechanism_library=mechanism_library,
        scope_interval=scope_interval,
        query_sessions_by_guard_result_id=query_sessions_by_guard_result_id,
        query_session_coordinator=query_session_coordinator,
        semantic_context=semantic_context,
        card_expansion_policy=card_expansion_policy,
    ):
        raise ValueError("role compilation result does not replay from frozen inputs")
    return _audit_inference_boundary(
        artifact,
        identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256,
        in_replayed_hypothesis_guard=False,
        in_replayed_role_compilation=True,
        in_replayed_verification=False,
    )


def audit_replayed_event_strength_boundary(
    artifact: EventStrengthResult,
    *,
    role_compilation: RoleCompilationResult,
    proposals: Sequence[RoleBindingProposal],
    guard_results: Sequence[HypothesisGuardResult],
    predicates: Sequence[PredicateResult],
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    mechanism_library: MechanismLibrary,
    scope_interval: TimeInterval,
    identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str,
    query_sessions_by_guard_result_id: Mapping[str, CaseQuerySessionResult] | None = None,
    query_session_coordinator: CaseQuerySessionCoordinator | None = None,
    semantic_context: SemanticContextBundle | None = None,
    card_expansion_policy: CardExpansionPolicy | None = None,
) -> InferenceBoundaryAudit:
    """Admit event strengths only after exact grouping and role-result replay."""

    if type(artifact) is not EventStrengthResult or not artifact.verify_inputs(
        role_compilation=role_compilation,
        proposals=proposals,
        guard_results=guard_results,
        predicates=predicates,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        mechanism_library=mechanism_library,
        scope_interval=scope_interval,
        query_sessions_by_guard_result_id=query_sessions_by_guard_result_id,
        query_session_coordinator=query_session_coordinator,
        semantic_context=semantic_context,
        card_expansion_policy=card_expansion_policy,
    ):
        raise ValueError("event strength result does not replay from frozen inputs")
    return _audit_inference_boundary(
        artifact,
        identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256,
        in_replayed_hypothesis_guard=False,
        in_replayed_role_compilation=False,
        in_replayed_verification=False,
        in_replayed_event_strength=True,
    )


def audit_replayed_family_aggregation_boundary(
    artifact: FamilyAggregationResult,
    *,
    event_strength: EventStrengthResult,
    aggregator_version: str,
    role_compilation: RoleCompilationResult,
    proposals: Sequence[RoleBindingProposal],
    guard_results: Sequence[HypothesisGuardResult],
    predicates: Sequence[PredicateResult],
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    mechanism_library: MechanismLibrary,
    scope_interval: TimeInterval,
    identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str,
    query_sessions_by_guard_result_id: Mapping[str, CaseQuerySessionResult] | None = None,
    query_session_coordinator: CaseQuerySessionCoordinator | None = None,
    semantic_context: SemanticContextBundle | None = None,
    card_expansion_policy: CardExpansionPolicy | None = None,
) -> InferenceBoundaryAudit:
    """Admit family aggregation only after exact strength and grouping replay."""

    if type(artifact) is not FamilyAggregationResult or not artifact.verify_inputs(
        event_strength=event_strength,
        aggregator_version=aggregator_version,
        role_compilation=role_compilation,
        proposals=proposals,
        guard_results=guard_results,
        predicates=predicates,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        mechanism_library=mechanism_library,
        scope_interval=scope_interval,
        query_sessions_by_guard_result_id=query_sessions_by_guard_result_id,
        query_session_coordinator=query_session_coordinator,
        semantic_context=semantic_context,
        card_expansion_policy=card_expansion_policy,
    ):
        raise ValueError("family aggregation result does not replay from frozen inputs")
    return _audit_inference_boundary(
        artifact,
        identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256,
        in_replayed_hypothesis_guard=False,
        in_replayed_role_compilation=False,
        in_replayed_verification=False,
        in_replayed_family_aggregation=True,
    )


def _require_replayed_graph_snapshot_pair(
    graph_snapshot_pair: GraphSnapshotPair,
    *,
    time_contract: CaseTimeContract,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    graph_input_manifest: FrozenGraphInputManifest,
    expected_graph_input_manifest_sha256: str,
    expected_minimum_coverage_fraction: float,
    canonical_coverage_inputs: tuple[CanonicalGraphCoverageInput, ...] | None = None,
    canonical_source_inputs: tuple[CanonicalGraphSourceInput, ...] | None = None,
    window_coverages: tuple[GraphWindowCoverage, ...] | None = None,
    source_records: tuple[GraphSourceRecord, ...] | None = None,
) -> GraphSnapshotPair:
    """Return a complete B/F pair only after exact pinned-input replay."""
    try:
        clean_pair = _trusted_graph_snapshot_pair(graph_snapshot_pair)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("graph snapshot pair does not replay from pinned graph inputs") from None
    if not GraphSnapshotPair.verify_inputs(
        clean_pair,
        time_contract=time_contract,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        graph_input_manifest=graph_input_manifest,
        expected_graph_input_manifest_sha256=expected_graph_input_manifest_sha256,
        expected_minimum_coverage_fraction=expected_minimum_coverage_fraction,
        canonical_coverage_inputs=canonical_coverage_inputs,
        canonical_source_inputs=canonical_source_inputs,
        window_coverages=window_coverages,
        source_records=source_records,
    ):
        raise ValueError("graph snapshot pair does not replay from pinned graph inputs")
    return clean_pair


def audit_replayed_graph_snapshot_boundary(
    artifact: GraphSnapshotPair,
    *,
    time_contract: CaseTimeContract,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    graph_input_manifest: FrozenGraphInputManifest,
    expected_graph_input_manifest_sha256: str,
    expected_minimum_coverage_fraction: float,
    canonical_coverage_inputs: tuple[CanonicalGraphCoverageInput, ...] | None = None,
    canonical_source_inputs: tuple[CanonicalGraphSourceInput, ...] | None = None,
    window_coverages: tuple[GraphWindowCoverage, ...] | None = None,
    source_records: tuple[GraphSourceRecord, ...] | None = None,
    identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str,
) -> InferenceBoundaryAudit:
    """Admit graph evidence only as an exactly replayed complete B/F pair."""

    clean_artifact = _require_replayed_graph_snapshot_pair(
        artifact,
        time_contract=time_contract,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        graph_input_manifest=graph_input_manifest,
        expected_graph_input_manifest_sha256=expected_graph_input_manifest_sha256,
        expected_minimum_coverage_fraction=expected_minimum_coverage_fraction,
        canonical_coverage_inputs=canonical_coverage_inputs,
        canonical_source_inputs=canonical_source_inputs,
        window_coverages=window_coverages,
        source_records=source_records,
    )
    return _audit_inference_boundary(
        clean_artifact,
        identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256,
        in_replayed_hypothesis_guard=False,
        in_replayed_role_compilation=False,
        in_replayed_verification=False,
        in_replayed_graph_snapshot=True,
    )


def audit_replayed_relation_scc_condensation_boundary(
    artifact: RelationSCCCondensation,
    *,
    graph_snapshot_pair: GraphSnapshotPair,
    window_kind: Literal["normal", "fault"],
    selected_edge_ids: Sequence[str] | None = None,
    time_contract: CaseTimeContract,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    graph_input_manifest: FrozenGraphInputManifest,
    expected_graph_input_manifest_sha256: str,
    expected_minimum_coverage_fraction: float,
    canonical_coverage_inputs: tuple[CanonicalGraphCoverageInput, ...] | None = None,
    canonical_source_inputs: tuple[CanonicalGraphSourceInput, ...] | None = None,
    window_coverages: tuple[GraphWindowCoverage, ...] | None = None,
    source_records: tuple[GraphSourceRecord, ...] | None = None,
    identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str,
) -> InferenceBoundaryAudit:
    """Admit condensation only from a caller-selected replayed pair window."""

    clean_pair = _require_replayed_graph_snapshot_pair(
        graph_snapshot_pair,
        time_contract=time_contract,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        graph_input_manifest=graph_input_manifest,
        expected_graph_input_manifest_sha256=expected_graph_input_manifest_sha256,
        expected_minimum_coverage_fraction=expected_minimum_coverage_fraction,
        canonical_coverage_inputs=canonical_coverage_inputs,
        canonical_source_inputs=canonical_source_inputs,
        window_coverages=window_coverages,
        source_records=source_records,
    )
    if type(window_kind) is not str or window_kind not in {"normal", "fault"}:
        raise ValueError("graph window kind must be an exact normal or fault string")
    graph_snapshot = (
        clean_pair.normal_snapshot if window_kind == "normal" else clean_pair.fault_snapshot
    )
    if not RelationSCCCondensation.verify_inputs(
        artifact,
        graph_snapshot=graph_snapshot,
        selected_edge_ids=selected_edge_ids,
    ):
        raise ValueError(
            "relation SCC condensation does not replay from authoritative graph snapshot lineage"
        )
    return _audit_inference_boundary(
        artifact,
        identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256,
        in_replayed_hypothesis_guard=False,
        in_replayed_role_compilation=False,
        in_replayed_verification=False,
        in_replayed_relation_condensation=True,
    )


def _require_authoritative_derived_observations(
    *,
    query_session: CaseQuerySessionResult,
    query_session_coordinator: CaseQuerySessionCoordinator,
    derived_observation_ids: tuple[str, ...],
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    semantic_context: SemanticContextBundle | None = None,
    card_expansion_policy: CardExpansionPolicy | None = None,
) -> tuple[DerivedVerificationObservation, ...]:
    """Select exact caller-pinned observations from one replayed current ledger."""

    if any(
        type(item) is not str or re.fullmatch(r"vob_[0-9a-f]{24}", item) is None
        for item in derived_observation_ids
    ):
        raise ValueError("derived observation IDs must use exact canonical strings")
    current_session = query_session_coordinator.require_current(query_session)
    replayed_session = require_case_query_session_replay(
        session=current_session,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        semantic_context=semantic_context,
        card_expansion_policy=card_expansion_policy,
    )
    verification = replayed_session.verification_result
    if verification is None:
        raise ValueError("authoritative verification lineage has no verification result")
    observation_by_id = {
        item.derived_observation_id: item for item in verification.derived_observations
    }
    try:
        return tuple(observation_by_id[item] for item in derived_observation_ids)
    except KeyError:
        raise ValueError(
            "authoritative verification lineage lacks a selected observation"
        ) from None


def audit_replayed_onset_interval_boundary(
    artifact: OnsetInterval,
    *,
    query_session: CaseQuerySessionResult,
    query_session_coordinator: CaseQuerySessionCoordinator,
    derived_observation_id: str,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    time_contract: CaseTimeContract,
    identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str,
    semantic_context: SemanticContextBundle | None = None,
    card_expansion_policy: CardExpansionPolicy | None = None,
) -> InferenceBoundaryAudit:
    """Admit an onset only from caller-selected authoritative verification lineage."""

    try:
        time_contract_sha256 = case_time_contract_sha256(time_contract)
        query_session_coordinator.require_pinned_temporal_context(
            opaque_case_id=time_contract.opaque_case_id,
            time_contract_sha256=time_contract_sha256,
        )
    except (AttributeError, TypeError, ValueError):
        raise ValueError("onset interval inputs do not match pinned temporal context") from None

    (selected_observation,) = _require_authoritative_derived_observations(
        query_session=query_session,
        query_session_coordinator=query_session_coordinator,
        derived_observation_ids=(derived_observation_id,),
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        semantic_context=semantic_context,
        card_expansion_policy=card_expansion_policy,
    )
    if selected_observation.operator != "measure_onset_interval" or not OnsetInterval.verify_inputs(
        artifact,
        derived_observation=selected_observation,
        time_contract=time_contract,
    ):
        raise ValueError("onset interval does not replay from authoritative verification lineage")
    clean_artifact = OnsetInterval.model_validate(
        OnsetInterval.model_dump(artifact, mode="json", warnings="error")
    )
    return _audit_inference_boundary(
        clean_artifact,
        identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256,
        in_replayed_hypothesis_guard=False,
        in_replayed_role_compilation=False,
        in_replayed_verification=False,
        in_replayed_onset_interval=True,
    )


def audit_replayed_temporal_precedence_boundary(
    artifact: TemporalPrecedence,
    *,
    query_session: CaseQuerySessionResult,
    query_session_coordinator: CaseQuerySessionCoordinator,
    source_derived_observation_id: str | None,
    target_derived_observation_id: str | None,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    time_contract: CaseTimeContract,
    clock_resolution: FrozenClockResolution,
    identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str,
    semantic_context: SemanticContextBundle | None = None,
    card_expansion_policy: CardExpansionPolicy | None = None,
) -> InferenceBoundaryAudit:
    """Admit precedence only from caller-selected authoritative onset lineage."""

    try:
        time_contract_sha256 = case_time_contract_sha256(time_contract)
        expected_clock_resolution_sha256 = (
            query_session_coordinator.require_pinned_temporal_context(
                opaque_case_id=time_contract.opaque_case_id,
                time_contract_sha256=time_contract_sha256,
            )
        )
        clean_clock_resolution = require_frozen_clock_resolution(
            clock_resolution=clock_resolution,
            time_contract=time_contract,
            expected_clock_resolution_sha256=expected_clock_resolution_sha256,
        )
    except (AttributeError, TypeError, ValueError):
        raise ValueError(
            "temporal precedence inputs are invalid or do not match pinned temporal context"
        ) from None

    selected_ids = tuple(
        item
        for item in (source_derived_observation_id, target_derived_observation_id)
        if item is not None
    )
    selected_observations = iter(
        _require_authoritative_derived_observations(
            query_session=query_session,
            query_session_coordinator=query_session_coordinator,
            derived_observation_ids=selected_ids,
            candidate_universe=candidate_universe,
            frozen_candidates=frozen_candidates,
            entity_registry=entity_registry,
            event_grouping=event_grouping,
            semantic_context=semantic_context,
            card_expansion_policy=card_expansion_policy,
        )
    )
    source_observation = (
        None if source_derived_observation_id is None else next(selected_observations)
    )
    target_observation = (
        None if target_derived_observation_id is None else next(selected_observations)
    )
    if any(
        item is not None and item.operator != "measure_onset_interval"
        for item in (source_observation, target_observation)
    ):
        raise ValueError(
            "temporal precedence does not replay from authoritative verification lineage"
        )
    source_onset = (
        None
        if source_observation is None
        else build_onset_interval_from_derived_observation(
            derived_observation=source_observation,
            time_contract=time_contract,
        )
    )
    target_onset = (
        None
        if target_observation is None
        else build_onset_interval_from_derived_observation(
            derived_observation=target_observation,
            time_contract=time_contract,
        )
    )
    if not TemporalPrecedence.verify_inputs(
        artifact,
        source_onset=source_onset,
        target_onset=target_onset,
        time_contract=time_contract,
        clock_resolution=clean_clock_resolution,
        expected_clock_resolution_sha256=expected_clock_resolution_sha256,
    ):
        raise ValueError(
            "temporal precedence does not replay from authoritative verification lineage"
        )
    clean_artifact = TemporalPrecedence.model_validate(
        TemporalPrecedence.model_dump(artifact, mode="json", warnings="error")
    )
    return _audit_inference_boundary(
        clean_artifact,
        identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256,
        in_replayed_hypothesis_guard=False,
        in_replayed_role_compilation=False,
        in_replayed_verification=False,
        in_replayed_temporal_precedence=True,
    )


def _audit_inference_boundary(
    artifact: object,
    *,
    identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str,
    in_replayed_hypothesis_guard: bool,
    in_replayed_role_compilation: bool,
    in_replayed_verification: bool,
    in_replayed_event_strength: bool = False,
    in_replayed_family_aggregation: bool = False,
    in_replayed_graph_snapshot: bool = False,
    in_replayed_relation_condensation: bool = False,
    in_replayed_onset_interval: bool = False,
    in_replayed_temporal_precedence: bool = False,
) -> InferenceBoundaryAudit:
    _require_valid_identity_manifest(
        identity_manifest,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256,
    )
    try:
        _reject_private_or_cyclic(artifact, active_ids=set(), location="$artifact")
        _scan_untyped_mapping_keys(artifact, location="$artifact")
    except RecursionError:
        # Malformed model_copy payloads can exhaust the pre-scan recursion limit.
        raise InferenceBoundaryLeakageError("invalid_public_artifact", "$artifact") from None
    state = _AuditState(
        trusted_dataset_by_id={
            record.opaque_case_id: record.dataset for record in identity_manifest.records
        },
        in_replayed_hypothesis_guard=in_replayed_hypothesis_guard,
        in_replayed_role_compilation=in_replayed_role_compilation,
        in_replayed_event_strength=in_replayed_event_strength,
        in_replayed_family_aggregation=in_replayed_family_aggregation,
        in_replayed_graph_snapshot=in_replayed_graph_snapshot,
        in_replayed_relation_condensation=in_replayed_relation_condensation,
        in_replayed_onset_interval=in_replayed_onset_interval,
        in_replayed_temporal_precedence=in_replayed_temporal_precedence,
        in_replayed_verification=in_replayed_verification,
    )
    if isinstance(artifact, Sequence) and not isinstance(artifact, str | bytes):
        for index, item in enumerate(artifact):
            _walk_artifact_root(
                item,
                state=state,
                location=f"$artifact[{index}]",
            )
    else:
        _walk_artifact_root(
            artifact,
            state=state,
            location="$artifact",
        )
    if len(state.opaque_case_ids) != 1:
        raise InferenceBoundaryLeakageError(
            "invalid_public_artifact",
            "$artifact",
        )
    derived_service_keys = state.candidate_service_keys.union(state.chosen_service_keys)
    unproven_service_keys = derived_service_keys.difference(state.explicit_service_keys)
    normalized_unproven_service_keys = {
        _collector_risk_key(service_key) for service_key in unproven_service_keys
    }
    if normalized_unproven_service_keys.intersection(state.collector_labels):
        raise InferenceBoundaryLeakageError(
            "collector_only_service",
            "$artifact",
        )
    if unproven_service_keys:
        raise InferenceBoundaryLeakageError(
            "unproven_service",
            "$artifact",
        )
    artifact_count = (
        len(artifact)
        if isinstance(artifact, Sequence) and not isinstance(artifact, str | bytes)
        else 1
    )
    if artifact_count < 1:
        raise InferenceBoundaryLeakageError(
            "invalid_public_artifact",
            "$artifact",
        )
    boundary_document = {
        "schema_version": INFERENCE_BOUNDARY_AUDIT_SCHEMA_VERSION,
        "identity_manifest_sha256": identity_manifest.manifest_sha256,
        "artifact": _public_json_value(artifact),
    }
    unsigned_audit: dict[str, object] = {
        "schema_version": INFERENCE_BOUNDARY_AUDIT_SCHEMA_VERSION,
        "artifact_count": artifact_count,
        "opaque_case_ids": tuple(sorted(state.opaque_case_ids)),
        "public_model_types": tuple(sorted(state.public_model_types)),
        "identity_manifest_sha256": identity_manifest.manifest_sha256,
        "boundary_sha256": sha256_hex(canonical_json_bytes(boundary_document)),
    }
    audit = InferenceBoundaryAudit.model_validate(
        {
            **unsigned_audit,
            "audit_sha256": sha256_hex(canonical_json_bytes(unsigned_audit)),
        }
    )
    if not audit.verify():  # pragma: no cover - construction invariant
        raise AssertionError("inference boundary audit verification failed")
    return audit


def _require_valid_identity_manifest(
    manifest: object,
    *,
    expected_identity_manifest_sha256: str,
) -> None:
    if type(manifest) is not InferenceIdentityManifest:
        raise InferenceBoundaryLeakageError(
            "invalid_identity_manifest",
            "$identity_manifest",
        )
    actual_manifest_fields = vars(manifest)
    if any(not isinstance(field_name, str) for field_name in actual_manifest_fields) or set(
        actual_manifest_fields
    ) != set(_IDENTITY_MANIFEST_FIELDS):
        raise InferenceBoundaryLeakageError(
            "invalid_identity_manifest",
            "$identity_manifest",
        )
    manifest_sha256 = actual_manifest_fields["manifest_sha256"]
    if (
        not isinstance(expected_identity_manifest_sha256, str)
        or len(expected_identity_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in expected_identity_manifest_sha256
        )
        or not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in manifest_sha256)
        or not hmac.compare_digest(
            manifest_sha256,
            expected_identity_manifest_sha256,
        )
    ):
        raise InferenceBoundaryLeakageError(
            "invalid_identity_manifest",
            "$identity_manifest",
        )
    try:
        revalidated_manifest = InferenceIdentityManifest.model_validate(
            manifest.model_dump(mode="python", warnings="error")
        )
        if revalidated_manifest != manifest:
            raise ValueError
        if manifest.method not in {"random_uuid_v4", "hmac_sha256_128"}:
            raise ValueError
        if not manifest.verify():
            raise ValueError
        for record in manifest.records:
            if type(record) is not InferenceIdentityRecord:
                raise ValueError
            if set(vars(record)) != set(_PUBLIC_MODEL_FIELD_ALLOWLIST[InferenceIdentityRecord]):
                raise ValueError
            revalidated_record = InferenceIdentityRecord.model_validate(
                record.model_dump(mode="python", warnings="error")
            )
            if revalidated_record != record:
                raise ValueError
            if record.method not in {"random_uuid_v4", "hmac_sha256_128"}:
                raise ValueError
            if record.method != manifest.method:
                raise ValueError
            if record.method == "random_uuid_v4":
                parsed = uuid.UUID(hex=record.opaque_case_id.removeprefix("opq_"))
                if parsed.version != 4 or parsed.variant != uuid.RFC_4122:
                    raise ValueError
    except (AttributeError, TypeError, ValueError):
        raise InferenceBoundaryLeakageError(
            "invalid_identity_manifest",
            "$identity_manifest",
        ) from None


def _preflight_report_evidence_model(value: object, *, location: str) -> None:
    # Use the actual MRO, not user-overridable instance/subclass checks or state
    # access. The new types must be safe before even the generic prescan runs.
    runtime_type = type(value)
    if not any(
        base is approved
        for base in type.__getattribute__(runtime_type, "__mro__")
        for approved in _REPORT_EVIDENCE_MODEL_TYPES
    ):
        return
    if not any(runtime_type is approved for approved in _REPORT_EVIDENCE_MODEL_TYPES):
        raise InferenceBoundaryLeakageError("unapproved_model_type", location)
    try:
        preflight_exact_pydantic_tree(
            value,
            root_model_types=(cast(type[BaseModel], runtime_type),),
            label="report evidence public artifact",
            max_string_length=None,
        )
    except (AttributeError, RecursionError, TypeError, ValueError):
        raise InferenceBoundaryLeakageError("invalid_public_artifact", location) from None


def _reject_private_or_cyclic(
    value: object,
    *,
    active_ids: set[int],
    location: str,
) -> None:
    _preflight_report_evidence_model(value, location=location)
    if isinstance(value, _PRIVATE_EVALUATOR_TYPES):
        raise InferenceBoundaryLeakageError(
            "private_evaluator_artifact",
            location,
        )
    children: Sequence[object]
    if isinstance(value, BaseModel):
        children = tuple(vars(value).values())
    elif isinstance(value, Mapping):
        children = tuple(value.keys()) + tuple(value.values())
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        children = value
    else:
        return
    identity = id(value)
    if identity in active_ids:
        raise InferenceBoundaryLeakageError("cyclic_artifact", location)
    active_ids.add(identity)
    try:
        for index, child in enumerate(children):
            _reject_private_or_cyclic(
                child,
                active_ids=active_ids,
                location=f"{location}[{index}]",
            )
    finally:
        active_ids.remove(identity)


def _scan_untyped_mapping_keys(value: object, *, location: str) -> None:
    """Classify forbidden untyped keys before shape validation without scanning values."""

    if isinstance(value, BaseModel):
        return
    if isinstance(value, Mapping):
        for index, (key, item) in enumerate(value.items()):
            if isinstance(key, str) and _is_forbidden_field_name(key):
                raise InferenceBoundaryLeakageError(
                    "forbidden_field_provenance",
                    f"{location}.field[{index}]",
                )
            _scan_untyped_mapping_keys(item, location=f"{location}.value[{index}]")
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, item in enumerate(value):
            _scan_untyped_mapping_keys(item, location=f"{location}[{index}]")


def _walk_public_value(
    value: object,
    *,
    state: _AuditState,
    location: str,
    allow_root_envelope: bool = False,
    allow_typed_mapping: bool = False,
) -> None:
    if isinstance(value, _PRIVATE_EVALUATOR_TYPES):
        raise InferenceBoundaryLeakageError(
            "private_evaluator_artifact",
            location,
        )
    if isinstance(value, BaseModel):
        _walk_public_model(value, state=state, location=location)
        return
    if isinstance(value, Mapping):
        if allow_root_envelope:
            _walk_public_mapping(value, state=state, location=location)
            return
        if allow_typed_mapping:
            for index, item in enumerate(value.values()):
                _walk_public_value(
                    item,
                    state=state,
                    location=f"{location}.value[{index}]",
                    allow_typed_mapping=True,
                )
            return
        raise InferenceBoundaryLeakageError("invalid_public_artifact", location)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, item in enumerate(value):
            _walk_public_value(
                item,
                state=state,
                location=f"{location}[{index}]",
                allow_typed_mapping=allow_typed_mapping,
            )
        return
    if value is None or isinstance(value, str | int | float | bool):
        return
    raise InferenceBoundaryLeakageError("unsupported_value_type", location)


def _walk_artifact_root(
    value: object,
    *,
    state: _AuditState,
    location: str,
) -> None:
    if type(value) in (InferenceIdentityManifest, MechanismDefinition, MechanismLibrary):
        raise InferenceBoundaryLeakageError("invalid_public_artifact", location)
    if not isinstance(value, BaseModel | Mapping):
        raise InferenceBoundaryLeakageError("invalid_public_artifact", location)
    previous_rule_scope = state.in_rule_envelope
    previous_evidence_scope = state.in_report_evidence_view
    state.in_rule_envelope = type(value) is RuleHypothesisEnvelope
    state.in_report_evidence_view = type(value) is ReportEvidenceView
    try:
        # The envelope verifier must succeed before recursive traversal can
        # admit any envelope-only input/output model below this root.
        _walk_public_value(
            value,
            state=state,
            location=location,
            allow_root_envelope=True,
        )
    finally:
        state.in_rule_envelope = previous_rule_scope
        state.in_report_evidence_view = previous_evidence_scope


def _walk_public_model(
    value: BaseModel,
    *,
    state: _AuditState,
    location: str,
) -> None:
    model_type = type(value)
    if model_type in _REPORT_EVIDENCE_MODEL_TYPES and not state.in_report_evidence_view:
        raise InferenceBoundaryLeakageError("invalid_public_artifact", location)
    if (
        model_type in _RULE_ENVELOPE_ONLY_MODEL_TYPES
        and not state.in_rule_envelope
        and not state.in_frozen_report_payload
    ):
        raise InferenceBoundaryLeakageError("invalid_public_artifact", location)
    if state.in_frozen_report_payload:
        allowed_fields = _FROZEN_REPORT_PAYLOAD_MODEL_FIELD_ALLOWLIST.get(model_type)
    else:
        allowed_fields = _PUBLIC_MODEL_FIELD_ALLOWLIST.get(model_type)
        if allowed_fields is None and state.in_replayed_hypothesis_guard:
            allowed_fields = _REPLAYED_HYPOTHESIS_GUARD_MODEL_FIELD_ALLOWLIST.get(model_type)
        if allowed_fields is None and state.in_replayed_role_compilation:
            allowed_fields = _REPLAYED_ROLE_COMPILATION_MODEL_FIELD_ALLOWLIST.get(model_type)
        if allowed_fields is None and state.in_replayed_event_strength:
            allowed_fields = _REPLAYED_EVENT_STRENGTH_MODEL_FIELD_ALLOWLIST.get(model_type)
        if allowed_fields is None and state.in_replayed_family_aggregation:
            allowed_fields = _REPLAYED_FAMILY_AGGREGATION_MODEL_FIELD_ALLOWLIST.get(model_type)
        if allowed_fields is None and state.in_replayed_graph_snapshot:
            allowed_fields = _REPLAYED_GRAPH_SNAPSHOT_MODEL_FIELD_ALLOWLIST.get(model_type)
        if allowed_fields is None and state.in_replayed_relation_condensation:
            allowed_fields = _REPLAYED_RELATION_CONDENSATION_MODEL_FIELD_ALLOWLIST.get(model_type)
        if allowed_fields is None and state.in_replayed_onset_interval:
            allowed_fields = _REPLAYED_ONSET_INTERVAL_MODEL_FIELD_ALLOWLIST.get(model_type)
        if allowed_fields is None and state.in_replayed_temporal_precedence:
            allowed_fields = _REPLAYED_TEMPORAL_PRECEDENCE_MODEL_FIELD_ALLOWLIST.get(model_type)
        if allowed_fields is None and state.in_replayed_verification:
            allowed_fields = _REPLAYED_VERIFICATION_MODEL_FIELD_ALLOWLIST.get(model_type)
    if allowed_fields is None:
        raise InferenceBoundaryLeakageError("unapproved_model_type", location)
    actual_fields = vars(value)
    if any(not isinstance(field_name, str) for field_name in actual_fields):
        raise InferenceBoundaryLeakageError("invalid_public_artifact", location)
    for index, field_name in enumerate(sorted(actual_fields)):
        if field_name not in allowed_fields:
            _reject_mapping_field(field_name, location=f"{location}.field[{index}]")
    if set(actual_fields) != set(allowed_fields):
        raise InferenceBoundaryLeakageError("invalid_public_artifact", location)
    try:
        # Exact view verification preflights every child before parent serialization.
        # This is an integrity/leakage gate, not independent source replay.
        if model_type is ReportEvidenceView and not ReportEvidenceView.verify(
            cast(ReportEvidenceView, value)
        ):
            raise ValueError
        revalidated = model_type.model_validate(value.model_dump(mode="python", warnings="error"))
        if revalidated != value:
            raise ValueError
        if isinstance(value, ScoreImpactCompilation) and not value.verify():
            raise ValueError
        # Repair routing has a private source-replay boundary, not permission to
        # submit its extracted children through an unrelated public envelope.
        if (isinstance(value, DiagnosticHypothesis)
                and value.origin == "deterministic_rule"
                and value.origin_version == "nurse-v2-live-applicable-rule-hypotheses/1"):
            raise ValueError
        # Extracted current-B0 children cannot replace the replay-verified root.
        # Other generators and ordinary predicates retain their existing uses.
        if not state.in_rule_envelope and (
            (
                isinstance(value, DiagnosticHypothesis)
                and value.origin == "deterministic_rule"
                and value.origin_version == RULE_GENERATOR_VERSION
            )
            or (
                isinstance(value, PredicateResult)
                and any(
                    re.fullmatch(r"rule-context:[0-9a-f]{64}", item)
                    for item in value.provenance_ids
                )
            )
        ):
            raise ValueError
        if model_type is MechanismDefinition and not verify_declared_mechanism(value):
            raise ValueError
        verifier = getattr(value, "verify", None)
        if callable(verifier) and verifier() is not True:
            raise ValueError
    except (AttributeError, TypeError, ValueError):
        raise InferenceBoundaryLeakageError(
            "invalid_public_artifact",
            location,
        ) from None
    state.public_model_types.add(model_type.__name__)
    if model_type is ReportEvidenceValue:
        evidence_name = actual_fields["name"]
        # Dynamic names are code-owned provenance fields, not generic telemetry
        # labels. Keep this stricter rule scoped to the new evidence contract.
        if _is_forbidden_field_name(evidence_name) or {"label", "labels", "path", "paths"} & set(
            evidence_name.split("_")
        ):
            raise InferenceBoundaryLeakageError("forbidden_field_provenance", f"{location}.name")
    _record_model_provenance(value, state=state)
    _record_model_identity(value, state=state, location=location)
    for field_name in sorted(allowed_fields):
        field_value = actual_fields[field_name]
        if model_type is FrozenReportContext and field_name == "payload_json":
            previous_report_scope = state.in_frozen_report_payload
            state.in_frozen_report_payload = True
            try:
                _walk_public_value(
                    cast(FrozenReportContext, value).payload,
                    state=state,
                    location=f"{location}.payload",
                )
            finally:
                state.in_frozen_report_payload = previous_report_scope
            continue
        normalized_field_name = _normalized_field_name(field_name)
        if state.in_frozen_report_payload and (
            "provenance" in normalized_field_name
            or "source" in normalized_field_name
            or normalized_field_name in {"origin", "rawvalue"}
        ):
            _reject_forbidden_report_provenance_values(
                field_value,
                location=f"{location}.{field_name}",
            )
        if normalized_field_name in {
            "opaquecaseid",
            "sourceopaquecaseid",
            "currentopaquecaseid",
        } and not (normalized_field_name == "opaquecaseid" and _model_claims_dataset(value)):
            _record_opaque_identity(field_value, state=state, location=location)
        _walk_public_value(
            field_value,
            state=state,
            location=f"{location}.{field_name}",
            allow_typed_mapping=(
                model_type is LegacyFrozenReportContextV2
                and field_name == "graph_snapshot_sha256_by_window"
            ),
        )


def _walk_public_mapping(
    value: Mapping[object, object],
    *,
    state: _AuditState,
    location: str,
) -> None:
    items = tuple(value.items())
    mapping_dataset = value.get("dataset")
    mapping_opaque_id = value.get("opaque_case_id")
    if mapping_opaque_id is None or not isinstance(mapping_dataset, str):
        raise InferenceBoundaryLeakageError("invalid_public_artifact", location)
    if "candidate_directory" in value and "entities" not in value:
        raise InferenceBoundaryLeakageError("invalid_public_artifact", location)
    _record_opaque_identity(
        mapping_opaque_id,
        state=state,
        location=location,
        claimed_dataset=mapping_dataset,
    )
    if not isinstance(mapping_opaque_id, str):  # narrowed by the recorder
        raise InferenceBoundaryLeakageError("invalid_public_artifact", location)
    previous_scope = (state.scope_case_id, state.scope_dataset)
    state.scope_case_id = mapping_opaque_id
    state.scope_dataset = state.trusted_dataset_by_id[mapping_opaque_id]
    try:
        for index, (key, item) in enumerate(items):
            field_location = f"{location}.field[{index}]"
            if not isinstance(key, str):
                raise InferenceBoundaryLeakageError(
                    "unapproved_mapping_field",
                    field_location,
                )
            if key not in _INFERENCE_ENVELOPE_FIELDS:
                _reject_mapping_field(key, location=field_location)
            _walk_envelope_field(
                key,
                item,
                state=state,
                location=f"{location}.{key}",
            )
    finally:
        state.scope_case_id, state.scope_dataset = previous_scope


def _walk_envelope_field(
    field_name: str,
    value: object,
    *,
    state: _AuditState,
    location: str,
) -> None:
    if field_name == "opaque_case_id":
        return
    if field_name == "dataset":
        if not isinstance(value, str) or value not in {"rcabench", "aiops25"}:
            raise InferenceBoundaryLeakageError("invalid_public_artifact", location)
        return
    if field_name == "decision_time":
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InferenceBoundaryLeakageError("invalid_public_artifact", location)
        return
    direct_model_types: dict[str, tuple[type[BaseModel], ...]] = {
        "candidate_directory": (CandidateUniverse, CompactCandidateDirectory),
        "entities": (TemporalEntityRegistry,),
        "mechanism_library": (MechanismLibrary,),
    }
    sequence_model_types: dict[str, tuple[type[BaseModel], ...]] = {
        "evidence_cards": (EvidenceCard, DetailedEvidenceCard),
        "graph_snapshot": (GraphSnapshotPair,),
        "observations": (Observation,),
        "verification_results": (VerificationResult,),
    }
    direct_types = direct_model_types.get(field_name)
    if direct_types is not None:
        if type(value) not in direct_types:
            raise InferenceBoundaryLeakageError("invalid_public_artifact", location)
        _walk_public_value(value, state=state, location=location)
        return
    allowed_item_types = sequence_model_types.get(field_name)
    if allowed_item_types is None:  # pragma: no cover - closed envelope allowlist
        raise InferenceBoundaryLeakageError("invalid_public_artifact", location)
    if type(value) in allowed_item_types:
        _walk_public_value(value, state=state, location=location)
        return
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise InferenceBoundaryLeakageError("invalid_public_artifact", location)
    for index, item in enumerate(value):
        if type(item) not in allowed_item_types:
            raise InferenceBoundaryLeakageError(
                "invalid_public_artifact",
                f"{location}[{index}]",
            )
        _walk_public_value(
            item,
            state=state,
            location=f"{location}[{index}]",
        )


def _reject_mapping_field(field_name: str, *, location: str) -> None:
    if _is_forbidden_field_name(field_name):
        raise InferenceBoundaryLeakageError(
            "forbidden_field_provenance",
            location,
        )
    raise InferenceBoundaryLeakageError("unapproved_mapping_field", location)


def _is_forbidden_field_name(field_name: str) -> bool:
    normalized = _normalized_field_name(field_name)
    return any(
        token in normalized for token_group in _FORBIDDEN_FIELD_TOKENS for token in token_group
    )


def _reject_forbidden_report_provenance_values(
    value: object,
    *,
    location: str,
) -> None:
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFKC", value).strip().casefold()
        looks_like_path = (
            normalized.startswith(("/", "\\\\", "file:"))
            or re.match(r"^[a-z]:[\\/]", normalized) is not None
        )
        looks_like_raw_case_id = (
            re.match(r"^(?:original[\s._:/\\-]*)?case(?:[\s._:/\\-]|$)", normalized) is not None
        )
        if _is_forbidden_field_name(value) or looks_like_path or looks_like_raw_case_id:
            raise InferenceBoundaryLeakageError(
                "forbidden_field_provenance",
                location,
            )
        return
    if isinstance(value, Mapping):
        children: Sequence[object] = tuple(value.keys()) + tuple(value.values())
    elif isinstance(value, Sequence) and not isinstance(value, bytes):
        children = value
    else:
        return
    for index, child in enumerate(children):
        _reject_forbidden_report_provenance_values(
            child,
            location=f"{location}[{index}]",
        )


def _normalized_field_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _record_opaque_identity(
    value: object,
    *,
    state: _AuditState,
    location: str,
    claimed_dataset: str | None = None,
) -> None:
    if not isinstance(value, str):
        raise InferenceBoundaryLeakageError(
            "untrusted_opaque_identity",
            location,
        )
    trusted_dataset = state.trusted_dataset_by_id.get(value)
    if (
        trusted_dataset is None
        or (claimed_dataset is not None and claimed_dataset != trusted_dataset)
        or (bool(state.opaque_case_ids) and value not in state.opaque_case_ids)
        or (state.scope_case_id is not None and value != state.scope_case_id)
        or (
            claimed_dataset is not None
            and state.scope_dataset is not None
            and claimed_dataset != state.scope_dataset
        )
    ):
        raise InferenceBoundaryLeakageError(
            "untrusted_opaque_identity",
            location,
        )
    state.opaque_case_ids.add(value)


def _model_claims_dataset(value: BaseModel) -> bool:
    return isinstance(
        value,
        InferenceIdentityRecord | RCABenchAdapterResult | AIOps25AdapterResult | HistoryContext,
    )


def _record_model_identity(
    value: BaseModel,
    *,
    state: _AuditState,
    location: str,
) -> None:
    if isinstance(value, InferenceIdentityRecord):
        _record_opaque_identity(
            value.opaque_case_id,
            state=state,
            location=location,
            claimed_dataset=value.dataset,
        )
    elif isinstance(value, RCABenchAdapterResult):
        _record_opaque_identity(
            value.opaque_case_id,
            state=state,
            location=location,
            claimed_dataset="rcabench",
        )
    elif isinstance(value, AIOps25AdapterResult):
        _record_opaque_identity(
            value.opaque_case_id,
            state=state,
            location=location,
            claimed_dataset="aiops25",
        )
    elif isinstance(value, HistoryContext):
        _record_opaque_identity(
            value.current_opaque_case_id,
            state=state,
            location=location,
            claimed_dataset=value.dataset,
        )


def _record_model_provenance(value: BaseModel, *, state: _AuditState) -> None:
    if isinstance(value, EntityMetadataRecord):
        if value.relation == "collector_source":
            state.collector_labels.add(_collector_risk_key(value.observed_entity_key))
        elif value.relation == "explicit_service_binding" and value.service_entity_key is not None:
            state.explicit_service_keys.add(value.service_entity_key)
    elif isinstance(value, CandidateEntry | CompactCandidateDirectoryEntry):
        state.candidate_service_keys.add(value.service_entity_key)
    elif (
        isinstance(value, EntityBinding)
        and value.status == "bound"
        and value.service_entity_key is not None
    ):
        state.chosen_service_keys.add(value.service_entity_key)


def _collector_risk_key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _public_json_value(value: object) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", warnings="error")
    if isinstance(value, Mapping):
        return {
            str(key): _public_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_public_json_value(item) for item in value]
    return value
