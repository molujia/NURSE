"""Explicit schema compatibility and migration boundaries for NURSE V2."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.aiops25_adapter import (
    AIOPS25_ADAPTER_CONFIGURATION_SCHEMA_VERSION,
    AIOPS25_ADAPTER_RESULT_SCHEMA_VERSION,
)
from nurse_rcl.v2.b0_rule_binding import B0_RULE_BINDING_SCHEMA_VERSION
from nurse_rcl.v2.candidate_universe import EMPTY_CANDIDATE_FAILURE_SCHEMA_VERSION
from nurse_rcl.v2.case_audit import (
    CASE_AUDIT_LINEAGE_INDEX_SCHEMA_VERSION,
    CASE_EVIDENCE_REPLAY_BUNDLE_SCHEMA_VERSION,
    CASE_RANKING_REPLAY_BUNDLE_SCHEMA_VERSION,
    IMMUTABLE_CASE_AUDIT_SCHEMA_VERSION,
)
from nurse_rcl.v2.case_audit_lineage_v2 import (
    CASE_AUDIT_LINEAGE_INDEX_SCHEMA_VERSION as CASE_AUDIT_LINEAGE_INDEX_V2_SCHEMA_VERSION,
)
from nurse_rcl.v2.case_audit_v2 import (
    CASE_EVIDENCE_REPLAY_BUNDLE_SCHEMA_VERSION as CASE_EVIDENCE_REPLAY_BUNDLE_V2_SCHEMA_VERSION,
)
from nurse_rcl.v2.case_audit_v2 import (
    CASE_RANKING_REPLAY_BUNDLE_SCHEMA_VERSION as CASE_RANKING_REPLAY_BUNDLE_V2_SCHEMA_VERSION,
)
from nurse_rcl.v2.case_audit_v2 import (
    IMMUTABLE_CASE_AUDIT_SCHEMA_VERSION as IMMUTABLE_CASE_AUDIT_V2_SCHEMA_VERSION,
)
from nurse_rcl.v2.case_scoring_selection import CASE_SCORING_SELECTION_SCHEMA_VERSION
from nurse_rcl.v2.case_state import CASE_STATE_ENTRY_SCHEMA_VERSION
from nurse_rcl.v2.competition_manifest import COMPETITION_MANIFEST_SCHEMA_VERSION
from nurse_rcl.v2.competition_roles import COMPETITION_ROLE_COMPILATION_SCHEMA_VERSION
from nurse_rcl.v2.deterministic_report import (
    DETERMINISTIC_SRE_REPORT_SCHEMA_VERSION,
)
from nurse_rcl.v2.diagnostic_schemas import (
    DIAGNOSTIC_HYPOTHESIS_SCHEMA_VERSION,
    GUARD_DECISION_SCHEMA_VERSION,
    LEGACY_VERIFICATION_REQUEST_SCHEMA_VERSION,
    MECHANISM_SCHEMA_VERSION,
    PREDICATE_RESULT_SCHEMA_VERSION,
    ROLE_ASSERTION_SCHEMA_VERSION,
    VERIFICATION_REQUEST_SCHEMA_VERSION,
    VERIFICATION_RESULT_SCHEMA_VERSION,
)
from nurse_rcl.v2.dual_explanation import (
    DUAL_EXPLANATION_MANIFEST_SCHEMA_VERSION,
    EXPLAINABILITY_ARTIFACT_SCHEMA_VERSION,
    INTERPRETABILITY_ARTIFACT_SCHEMA_VERSION,
)
from nurse_rcl.v2.event_grouping import (
    ADAPTER_ASSOCIATION_IDENTITY_SCHEMA_VERSION,
    ASSOCIATION_CLAIM_SCHEMA_VERSION,
    ASSOCIATION_DERIVATION_RULE_SCHEMA_VERSION,
    ASSOCIATION_POLICY_SCHEMA_VERSION,
    ASSOCIATION_SOURCE_RECORD_SCHEMA_VERSION,
    EVENT_GROUP_SCHEMA_VERSION,
    EVENT_GROUPING_RESULT_SCHEMA_VERSION,
    GROUPING_DECISION_SCHEMA_VERSION,
    OBSERVATION_ASSOCIATION_PROOF_SCHEMA_VERSION,
    SIGNAL_FAMILY_MAPPING_ENTRY_SCHEMA_VERSION,
    SIGNAL_FAMILY_MAPPING_SCHEMA_VERSION,
)
from nurse_rcl.v2.event_strength import (
    CANDIDATE_EVENT_ROLE_ADJUSTMENT_SCHEMA_VERSION,
    EVENT_GROUP_BASE_STRENGTH_SCHEMA_VERSION,
    EVENT_STRENGTH_RESULT_SCHEMA_VERSION,
)
from nurse_rcl.v2.family_aggregation import (
    CANDIDATE_FAMILY_AGGREGATION_SCHEMA_VERSION,
    FAMILY_AGGREGATION_RESULT_SCHEMA_VERSION,
    FAMILY_AGGREGATOR_DEFINITION_SCHEMA_VERSION,
    SIGNAL_FAMILY_MAXIMUM_SCHEMA_VERSION,
    UNBOUND_EVENT_GROUP_AUDIT_SCHEMA_VERSION,
)
from nurse_rcl.v2.graph_snapshots import (
    CANONICAL_GRAPH_COVERAGE_INPUT_SCHEMA_VERSION,
    CANONICAL_GRAPH_SOURCE_INPUT_SCHEMA_VERSION,
    FROZEN_GRAPH_INPUT_MANIFEST_SCHEMA_VERSION,
    GRAPH_ADAPTER_PARENT_RECORD_SCHEMA_VERSION,
    GRAPH_INPUT_DECISION_SCHEMA_VERSION,
    GRAPH_SNAPSHOT_PAIR_SCHEMA_VERSION,
    GRAPH_SNAPSHOT_SCHEMA_VERSION,
    GRAPH_SOURCE_RECORD_SCHEMA_VERSION,
    GRAPH_WINDOW_COVERAGE_SCHEMA_VERSION,
)
from nurse_rcl.v2.history_scoring import (
    CANDIDATE_HISTORY_SCORE_SCHEMA_VERSION,
    FROZEN_HISTORY_LIBRARY_SCHEMA_VERSION,
    HISTORY_BASE_CANDIDATE_SCORE_SCHEMA_VERSION,
    HISTORY_EVALUATION_RESULT_SCHEMA_VERSION,
    HISTORY_GENERATION_AUTHORITY_SCHEMA_VERSION,
    HISTORY_HINT_REFERENCE_SCHEMA_VERSION,
    HISTORY_LIBRARY_ENTRY_SCHEMA_VERSION,
    HISTORY_NEIGHBOR_AUDIT_SCHEMA_VERSION,
    HISTORY_QUERY_EMBEDDING_SCHEMA_VERSION,
    HISTORY_SCORE_CONTRIBUTION_SCHEMA_VERSION,
    HISTORY_SCORING_CONFIG_SCHEMA_VERSION,
    HISTORY_SERVICE_MAPPING_SCHEMA_VERSION,
)
from nurse_rcl.v2.hypothesis_guard import (
    HYPOTHESIS_GUARD_RESULT_SCHEMA_VERSION,
    SCORE_IMPACT_COMPILATION_SCHEMA_VERSION,
)
from nurse_rcl.v2.inference_leakage import INFERENCE_BOUNDARY_AUDIT_SCHEMA_VERSION
from nurse_rcl.v2.live.evidence_routing import (
    ROUTED_B0_BINDING_SCHEMA_VERSION,
    ROUTED_RULE_SET_SCHEMA_VERSION,
)
from nurse_rcl.v2.live.evidence_routing_indexed import (
    ROUTED_B0_BINDING_SCHEMA_VERSION as INDEXED_ROUTED_B0_BINDING_SCHEMA_VERSION,
)
from nurse_rcl.v2.live.evidence_routing_indexed import (
    ROUTED_RULE_SET_SCHEMA_VERSION as INDEXED_ROUTED_RULE_SET_SCHEMA_VERSION,
)
from nurse_rcl.v2.live.repair_assembly import REPAIR_DIAGNOSTIC_SCHEMA_VERSION
from nurse_rcl.v2.live.repair_error_references import (
    REPAIR_ERROR_SOURCES_SCHEMA_VERSION,
    REPAIR_REFERENCED_ERROR_FACTS_SCHEMA_VERSION,
)
from nurse_rcl.v2.live.repair_errors import ERROR_OBSERVATION_SCHEMA_VERSION
from nurse_rcl.v2.live.repair_fixed_facts import (
    REPAIR_FIXED_ERROR_SCHEMA_VERSION,
    REPAIR_FIXED_SIGNAL_SCHEMA_VERSION,
)
from nurse_rcl.v2.live.repair_lifecycle import LIFECYCLE_SCHEMA_VERSION
from nurse_rcl.v2.live.repair_measurements import (
    REPAIR_MEASURED_PREDICATE_SCHEMA_VERSION,
    REPAIR_MEASUREMENT_SCHEMA_VERSION,
)
from nurse_rcl.v2.live.repair_mechanisms import REPAIR_MECHANISM_SCHEMA_VERSION
from nurse_rcl.v2.live.repair_observations import REPAIR_QUALITY_SCHEMA_VERSION
from nurse_rcl.v2.live.repair_queries import (
    REPAIR_QUERY_CATALOG_SCHEMA_VERSION,
    REPAIR_QUERY_FEEDBACK_SCHEMA_VERSION,
    REPAIR_QUERY_SELECTION_SCHEMA_VERSION,
)
from nurse_rcl.v2.live.repair_query_session import (
    REPAIR_QUERY_JOURNAL_SCHEMA_VERSION,
    REPAIR_QUERY_SESSION_SCHEMA_VERSION,
)
from nurse_rcl.v2.live.repair_sources import REPAIR_MEASUREMENT_SOURCE_SCHEMA_VERSION
from nurse_rcl.v2.live.repair_strengths import REPAIR_STRENGTH_SCHEMA_VERSION
from nurse_rcl.v2.live.repair_temporal import REPAIR_TEMPORAL_SCHEMA_VERSION
from nurse_rcl.v2.mechanism_library import MECHANISM_LIBRARY_SCHEMA_VERSION
from nurse_rcl.v2.no_role_compilation import NO_ROLE_COMPILATION_RESULT_SCHEMA_VERSION
from nurse_rcl.v2.no_role_evidence import (
    NO_ROLE_EVENT_STRENGTH_RESULT_SCHEMA_VERSION,
    NO_ROLE_FAMILY_AGGREGATION_RESULT_SCHEMA_VERSION,
)
from nurse_rcl.v2.no_role_propagation import NO_ROLE_PROPAGATION_SCORING_RESULT_SCHEMA_VERSION
from nurse_rcl.v2.no_role_ranking import NO_ROLE_REFERENCE_RANKING_RESULT_SCHEMA_VERSION
from nurse_rcl.v2.observation_quality import (
    OBSERVATION_QUALITY_RESULT_SCHEMA_VERSION,
)
from nurse_rcl.v2.observation_schemas import (
    CANDIDATE_UNIVERSE_SCHEMA_VERSION,
    DATASET_MANIFEST_SCHEMA_VERSION,
    ENTITY_BINDING_SCHEMA_VERSION,
    EVIDENCE_CARD_SCHEMA_VERSION,
    FOCUS_SET_SCHEMA_VERSION,
    LEGACY_EVENT_GROUP_SCHEMA_VERSION,
    LEGACY_OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    TIME_CONTRACT_SCHEMA_VERSION,
)
from nurse_rcl.v2.propagation_scoring import (
    CANDIDATE_PROPAGATION_SCORE_SCHEMA_VERSION,
    PROPAGATION_HOP_AUDIT_SCHEMA_VERSION,
    PROPAGATION_PATH_AUDIT_SCHEMA_VERSION,
    PROPAGATION_RELATION_BINDING_SCHEMA_VERSION,
    PROPAGATION_SCORING_CONFIG_SCHEMA_VERSION,
    PROPAGATION_SCORING_RESULT_SCHEMA_VERSION,
    PROPAGATION_TARGET_SCORE_SCHEMA_VERSION,
)
from nurse_rcl.v2.query_session import CASE_QUERY_SESSION_RESULT_SCHEMA_VERSION
from nurse_rcl.v2.ranking_replay import (
    CANDIDATE_RANK_SENSITIVITY_SCHEMA_VERSION,
    MODEL_SENSITIVITY_REQUEST_SCHEMA_VERSION,
    MODEL_SENSITIVITY_RESULT_SCHEMA_VERSION,
    RANKING_REPLAY_RESULT_SCHEMA_VERSION,
    SENSITIVITY_CANDIDATE_TERMS_SCHEMA_VERSION,
)
from nurse_rcl.v2.ranking_schemas import (
    FROZEN_REPORT_CONTEXT_SCHEMA_VERSION as LEGACY_FROZEN_REPORT_CONTEXT_V2_SCHEMA_VERSION,
)
from nurse_rcl.v2.ranking_schemas import (
    HISTORY_CONTEXT_SCHEMA_VERSION,
    LEGACY_FROZEN_REPORT_CONTEXT_SCHEMA_VERSION,
    LEGACY_TOP_K_SCHEMA_VERSION,
    RULE_SET_SCHEMA_VERSION,
    SCC_PROVENANCE_SCHEMA_VERSION,
    SCORE_DECOMPOSITION_SCHEMA_VERSION,
    SCORING_RULE_SCHEMA_VERSION,
    TIE_GROUP_SCHEMA_VERSION,
    TOP_K_SCHEMA_VERSION,
)
from nurse_rcl.v2.rcabench_adapter import (
    RCABENCH_ADAPTER_CONFIGURATION_SCHEMA_VERSION,
    RCABENCH_ADAPTER_RESULT_SCHEMA_VERSION,
)
from nurse_rcl.v2.reference_ranking import (
    REFERENCE_RANKING_RESULT_SCHEMA_VERSION,
    REFERENCE_SCORING_CONFIG_SCHEMA_VERSION,
)
from nurse_rcl.v2.report_composition_proposals import REPORT_COMPOSITION_RESPONSE_SCHEMA_VERSION
from nurse_rcl.v2.report_context import FROZEN_REPORT_CONTEXT_SCHEMA_VERSION
from nurse_rcl.v2.report_context_v4 import (
    FROZEN_REPORT_CONTEXT_SCHEMA_VERSION as FROZEN_REPORT_CONTEXT_V4_SCHEMA_VERSION,
)
from nurse_rcl.v2.report_context_v4 import (
    FROZEN_REPORT_PAYLOAD_SCHEMA_VERSION as FROZEN_REPORT_PAYLOAD_V2_SCHEMA_VERSION,
)
from nurse_rcl.v2.report_evidence_schemas import (
    REPORT_EVIDENCE_POLICY_SCHEMA_VERSION,
    REPORT_EVIDENCE_PROJECTION_SCHEMA_VERSION,
    REPORT_EVIDENCE_VIEW_SCHEMA_VERSION,
)
from nurse_rcl.v2.report_explainability_schemas import (
    REPORT_COMPOSITION_EXPLAINABILITY_SCHEMA_VERSION,
)
from nurse_rcl.v2.report_generation_audit import (
    REPORT_GENERATION_AUDIT_SCHEMA_VERSION,
    REPORT_GENERATION_FAILURE_AUDIT_SCHEMA_VERSION,
)
from nurse_rcl.v2.report_grounding import (
    REPORT_GROUNDING_CATALOG_SCHEMA_VERSION,
    REPORT_GROUNDING_VALIDATION_SCHEMA_VERSION,
)
from nurse_rcl.v2.report_mechanical_schemas import REPORT_COMPOSITION_MECHANICAL_SCHEMA_VERSION
from nurse_rcl.v2.report_presentation_schemas import REPORT_COMPOSITION_PRESENTATION_SCHEMA_VERSION
from nurse_rcl.v2.report_run_schemas import (
    REPORT_RUN_AUDIT_SCHEMA_VERSION,
    REPORT_RUN_POLICY_SCHEMA_VERSION,
)
from nurse_rcl.v2.role_compiler import (
    ROLE_BINDING_PROPOSAL_SCHEMA_VERSION,
    ROLE_COMPILATION_RESULT_SCHEMA_VERSION,
)
from nurse_rcl.v2.rule_hypotheses import (
    PENDING_VERIFICATION_PLAN_SCHEMA_VERSION,
    RULE_HYPOTHESIS_ENVELOPE_SCHEMA_VERSION,
    RULE_HYPOTHESIS_SET_SCHEMA_VERSION,
)
from nurse_rcl.v2.rule_scoring import (
    CANDIDATE_RULE_SCORE_SCHEMA_VERSION,
    RULE_CANDIDATE_ASSIGNMENT_SCHEMA_VERSION,
    RULE_CONTRIBUTION_SCHEMA_VERSION,
    RULE_SCORING_RESULT_SCHEMA_VERSION,
)
from nurse_rcl.v2.runtime_schemas import (
    ATTEMPT_LEDGER_SCHEMA_VERSION,
    CACHE_IDENTITY_SCHEMA_VERSION,
    CASE_CHECKPOINT_SCHEMA_VERSION,
    CASE_EVALUATION_SCHEMA_VERSION,
    CASE_EXPOSURE_SCHEMA_VERSION,
    COST_RECORD_SCHEMA_VERSION,
    SEALED_AUDIT_SCHEMA_VERSION,
    SEMANTIC_CALL_SCHEMA_VERSION,
)
from nurse_rcl.v2.semantic_compilation_artifact import SEMANTIC_COMPILATION_SCHEMA_VERSION
from nurse_rcl.v2.semantic_proposals import SEMANTIC_PROPOSAL_RESPONSE_SCHEMA_VERSION
from nurse_rcl.v2.semantic_run_audit import SEMANTIC_RUN_AUDIT_SCHEMA_VERSION
from nurse_rcl.v2.signal_transforms import SIGNAL_TRANSFORM_RESULT_SCHEMA_VERSION
from nurse_rcl.v2.temporal_condensation import (
    CLOCK_RESOLUTION_SCHEMA_VERSION,
    ONSET_INTERVAL_SCHEMA_VERSION,
    RELATION_CONDENSED_EDGE_SCHEMA_VERSION,
    RELATION_SCC_COMPONENT_SCHEMA_VERSION,
    RELATION_SCC_CONDENSATION_SCHEMA_VERSION,
    TEMPORAL_PRECEDENCE_SCHEMA_VERSION,
)
from nurse_rcl.v2.temporal_entity_registry import (
    ENTITY_METADATA_RECORD_SCHEMA_VERSION,
    OBSERVATION_BINDING_RESOLUTION_SCHEMA_VERSION,
    TEMPORAL_ENTITY_REGISTRY_SCHEMA_VERSION,
)
from nurse_rcl.v2.verification_executor import (
    DERIVED_VERIFICATION_OBSERVATION_SCHEMA_VERSION,
    VERIFICATION_BATCH_RESULT_SCHEMA_VERSION,
    VERIFICATION_EXECUTION_DECISION_SCHEMA_VERSION,
    VERIFICATION_EXECUTION_POLICY_SCHEMA_VERSION,
    VERIFICATION_EXECUTION_SCOPE_SCHEMA_VERSION,
    VERIFICATION_FIELD_VALUE_SCHEMA_VERSION,
    VERIFICATION_OPERATOR_PERMISSION_SCHEMA_VERSION,
    VERIFICATION_REQUEST_COUNT_SCHEMA_VERSION,
)

SCHEMA_COMPATIBILITY_SNAPSHOT_VERSION: Literal["nurse.v2.schema-compatibility-snapshot/1"] = (
    "nurse.v2.schema-compatibility-snapshot/1"
)

CURRENT_SCHEMA_VERSIONS: Mapping[str, str] = MappingProxyType(
    {
        "dataset_manifest": DATASET_MANIFEST_SCHEMA_VERSION,
        "case_time_contract": TIME_CONTRACT_SCHEMA_VERSION,
        "observation": OBSERVATION_SCHEMA_VERSION,
        "observation_quality_result": OBSERVATION_QUALITY_RESULT_SCHEMA_VERSION,
        "entity_binding": ENTITY_BINDING_SCHEMA_VERSION,
        "signal_family_mapping_entry": SIGNAL_FAMILY_MAPPING_ENTRY_SCHEMA_VERSION,
        "signal_family_mapping": SIGNAL_FAMILY_MAPPING_SCHEMA_VERSION,
        "association_derivation_rule": ASSOCIATION_DERIVATION_RULE_SCHEMA_VERSION,
        "association_policy": ASSOCIATION_POLICY_SCHEMA_VERSION,
        "canonical_association_source_record": ASSOCIATION_SOURCE_RECORD_SCHEMA_VERSION,
        "adapter_association_identity": ADAPTER_ASSOCIATION_IDENTITY_SCHEMA_VERSION,
        "observation_association_proof": OBSERVATION_ASSOCIATION_PROOF_SCHEMA_VERSION,
        "event_association_claim": ASSOCIATION_CLAIM_SCHEMA_VERSION,
        "event_group": EVENT_GROUP_SCHEMA_VERSION,
        "event_grouping_decision": GROUPING_DECISION_SCHEMA_VERSION,
        "event_grouping_result": EVENT_GROUPING_RESULT_SCHEMA_VERSION,
        "event_group_base_strength": EVENT_GROUP_BASE_STRENGTH_SCHEMA_VERSION,
        "candidate_event_role_adjustment": CANDIDATE_EVENT_ROLE_ADJUSTMENT_SCHEMA_VERSION,
        "event_strength_result": EVENT_STRENGTH_RESULT_SCHEMA_VERSION,
        "family_aggregator_definition": FAMILY_AGGREGATOR_DEFINITION_SCHEMA_VERSION,
        "signal_family_maximum": SIGNAL_FAMILY_MAXIMUM_SCHEMA_VERSION,
        "candidate_family_aggregation": CANDIDATE_FAMILY_AGGREGATION_SCHEMA_VERSION,
        "family_aggregation_result": FAMILY_AGGREGATION_RESULT_SCHEMA_VERSION,
        "unbound_event_group_audit": UNBOUND_EVENT_GROUP_AUDIT_SCHEMA_VERSION,
        "candidate_universe": CANDIDATE_UNIVERSE_SCHEMA_VERSION,
        "empty_candidate_universe_failure": EMPTY_CANDIDATE_FAILURE_SCHEMA_VERSION,
        "case_evidence_replay_bundle": CASE_EVIDENCE_REPLAY_BUNDLE_SCHEMA_VERSION,
        "case_ranking_replay_bundle": CASE_RANKING_REPLAY_BUNDLE_SCHEMA_VERSION,
        "case_audit_lineage_index": CASE_AUDIT_LINEAGE_INDEX_SCHEMA_VERSION,
        "immutable_case_audit_package": IMMUTABLE_CASE_AUDIT_SCHEMA_VERSION,
        "case_scoring_selection": CASE_SCORING_SELECTION_SCHEMA_VERSION,
        "case_evidence_replay_bundle_v2": CASE_EVIDENCE_REPLAY_BUNDLE_V2_SCHEMA_VERSION,
        "case_ranking_replay_bundle_v2": CASE_RANKING_REPLAY_BUNDLE_V2_SCHEMA_VERSION,
        "case_audit_lineage_index_v2": CASE_AUDIT_LINEAGE_INDEX_V2_SCHEMA_VERSION,
        "immutable_case_audit_package_v2": IMMUTABLE_CASE_AUDIT_V2_SCHEMA_VERSION,
        "semantic_focus_set": FOCUS_SET_SCHEMA_VERSION,
        "evidence_card": EVIDENCE_CARD_SCHEMA_VERSION,
        "mechanism_definition": MECHANISM_SCHEMA_VERSION,
        "mechanism_library": MECHANISM_LIBRARY_SCHEMA_VERSION,
        "predicate_result": PREDICATE_RESULT_SCHEMA_VERSION,
        "verification_request": VERIFICATION_REQUEST_SCHEMA_VERSION,
        "verification_result": VERIFICATION_RESULT_SCHEMA_VERSION,
        "verification_operator_permission": VERIFICATION_OPERATOR_PERMISSION_SCHEMA_VERSION,
        "verification_execution_policy": VERIFICATION_EXECUTION_POLICY_SCHEMA_VERSION,
        "verification_execution_scope": VERIFICATION_EXECUTION_SCOPE_SCHEMA_VERSION,
        "verification_field_value": VERIFICATION_FIELD_VALUE_SCHEMA_VERSION,
        "derived_verification_observation": (DERIVED_VERIFICATION_OBSERVATION_SCHEMA_VERSION),
        "verification_request_count": VERIFICATION_REQUEST_COUNT_SCHEMA_VERSION,
        "verification_execution_decision": VERIFICATION_EXECUTION_DECISION_SCHEMA_VERSION,
        "verification_batch_result": VERIFICATION_BATCH_RESULT_SCHEMA_VERSION,
        "case_query_session_result": CASE_QUERY_SESSION_RESULT_SCHEMA_VERSION,
        "diagnostic_hypothesis": DIAGNOSTIC_HYPOTHESIS_SCHEMA_VERSION,
        "competition_manifest": COMPETITION_MANIFEST_SCHEMA_VERSION,
        "competition_role_compilation": COMPETITION_ROLE_COMPILATION_SCHEMA_VERSION,
        "rule_hypothesis_set": RULE_HYPOTHESIS_SET_SCHEMA_VERSION,
        "b0_rule_binding": B0_RULE_BINDING_SCHEMA_VERSION,
        "live_routed_rule_set": ROUTED_RULE_SET_SCHEMA_VERSION,
        "live_routed_b0_binding": ROUTED_B0_BINDING_SCHEMA_VERSION,
        "live_indexed_routed_rule_set": INDEXED_ROUTED_RULE_SET_SCHEMA_VERSION,
        "live_indexed_routed_b0_binding": INDEXED_ROUTED_B0_BINDING_SCHEMA_VERSION,
        "live_repair_quality": REPAIR_QUALITY_SCHEMA_VERSION,
        "live_repair_lifecycle": LIFECYCLE_SCHEMA_VERSION,
        "live_repair_errors": ERROR_OBSERVATION_SCHEMA_VERSION,
        "live_repair_error_sources": REPAIR_ERROR_SOURCES_SCHEMA_VERSION,
        "live_repair_referenced_error_facts": REPAIR_REFERENCED_ERROR_FACTS_SCHEMA_VERSION,
        "live_repair_fixed_signal_facts": REPAIR_FIXED_SIGNAL_SCHEMA_VERSION,
        "live_repair_fixed_error_facts": REPAIR_FIXED_ERROR_SCHEMA_VERSION,
        "live_repair_mechanism_evidence": REPAIR_MECHANISM_SCHEMA_VERSION,
        "live_repair_strengths": REPAIR_STRENGTH_SCHEMA_VERSION,
        "live_repair_local_scoring": "nurse.v2.live-repair-local-scoring/1",
        "live_repair_propagation": "nurse.v2.live-repair-propagation/1",
        "live_repair_ranking": "nurse.v2.live-repair-ranking/1",
        "live_repair_graph": "nurse.v2.live-repair-graph/1",
        "live_repair_raw_audit": "nurse.v2.live-repair-raw-audit/1",
        "live_repair_case_audit": "nurse.v2.live-repair-case-audit/1",
        "live_repair_case_closure": "nurse.v2.live-repair-case-closure/1",
        "live_repair_case_report": "nurse.v2.live-repair-report/2",
        "live_repair_case_report_projection": "nurse.v2.live-repair-report-projection/2",
        "live_repair_prepared_case_audit": "nurse.v2.live-repair-prepared-case-audit/1",
        "live_repair_prepared_case_closure": "nurse.v2.live-repair-prepared-case-closure/1",
        "live_repair_prepared_report": "nurse.v2.live-repair-prepared-report/1",
        "live_repair_prepared_report_projection": (
            "nurse.v2.live-repair-prepared-report-projection/1"
        ),
        "live_repair_report": "nurse.v2.live-repair-report/1",
        "live_repair_report_projection": "nurse.v2.live-repair-report-projection/1",
        "live_repair_report_presentation": "nurse.v2.live-repair-report-presentation/1",
        "live_repair_diagnostic_inputs": REPAIR_DIAGNOSTIC_SCHEMA_VERSION,
        "live_repair_measurement_source": REPAIR_MEASUREMENT_SOURCE_SCHEMA_VERSION,
        "live_repair_measurement": REPAIR_MEASUREMENT_SCHEMA_VERSION,
        "live_repair_measured_predicate": REPAIR_MEASURED_PREDICATE_SCHEMA_VERSION,
        "live_repair_temporal": REPAIR_TEMPORAL_SCHEMA_VERSION,
        "live_repair_query_catalog": REPAIR_QUERY_CATALOG_SCHEMA_VERSION,
        "live_repair_query_feedback": REPAIR_QUERY_FEEDBACK_SCHEMA_VERSION,
        "live_repair_query_selection": REPAIR_QUERY_SELECTION_SCHEMA_VERSION,
        "live_repair_query_journal": REPAIR_QUERY_JOURNAL_SCHEMA_VERSION,
        "live_repair_query_session": REPAIR_QUERY_SESSION_SCHEMA_VERSION,
        "rule_hypothesis_envelope": RULE_HYPOTHESIS_ENVELOPE_SCHEMA_VERSION,
        "pending_verification_plan": PENDING_VERIFICATION_PLAN_SCHEMA_VERSION,
        "role_assertion": ROLE_ASSERTION_SCHEMA_VERSION,
        "role_binding_proposal": ROLE_BINDING_PROPOSAL_SCHEMA_VERSION,
        "role_compilation_result": ROLE_COMPILATION_RESULT_SCHEMA_VERSION,
        "no_role_compilation_result": NO_ROLE_COMPILATION_RESULT_SCHEMA_VERSION,
        "no_role_event_strength_result": NO_ROLE_EVENT_STRENGTH_RESULT_SCHEMA_VERSION,
        "no_role_family_aggregation_result": NO_ROLE_FAMILY_AGGREGATION_RESULT_SCHEMA_VERSION,
        "no_role_propagation_scoring_result": NO_ROLE_PROPAGATION_SCORING_RESULT_SCHEMA_VERSION,
        "no_role_reference_ranking_result": NO_ROLE_REFERENCE_RANKING_RESULT_SCHEMA_VERSION,
        "guard_decision": GUARD_DECISION_SCHEMA_VERSION,
        "hypothesis_guard_result": HYPOTHESIS_GUARD_RESULT_SCHEMA_VERSION,
        "canonical_graph_coverage_input": CANONICAL_GRAPH_COVERAGE_INPUT_SCHEMA_VERSION,
        "canonical_graph_source_input": CANONICAL_GRAPH_SOURCE_INPUT_SCHEMA_VERSION,
        "frozen_graph_input_manifest": FROZEN_GRAPH_INPUT_MANIFEST_SCHEMA_VERSION,
        "graph_adapter_parent_record": GRAPH_ADAPTER_PARENT_RECORD_SCHEMA_VERSION,
        "graph_input_decision": GRAPH_INPUT_DECISION_SCHEMA_VERSION,
        "graph_snapshot": GRAPH_SNAPSHOT_SCHEMA_VERSION,
        "graph_snapshot_pair": GRAPH_SNAPSHOT_PAIR_SCHEMA_VERSION,
        "graph_source_record": GRAPH_SOURCE_RECORD_SCHEMA_VERSION,
        "graph_window_coverage": GRAPH_WINDOW_COVERAGE_SCHEMA_VERSION,
        "frozen_clock_resolution": CLOCK_RESOLUTION_SCHEMA_VERSION,
        "onset_interval": ONSET_INTERVAL_SCHEMA_VERSION,
        "temporal_precedence": TEMPORAL_PRECEDENCE_SCHEMA_VERSION,
        "relation_scc_component": RELATION_SCC_COMPONENT_SCHEMA_VERSION,
        "relation_condensed_edge": RELATION_CONDENSED_EDGE_SCHEMA_VERSION,
        "relation_scc_condensation": RELATION_SCC_CONDENSATION_SCHEMA_VERSION,
        "propagation_scoring_config": PROPAGATION_SCORING_CONFIG_SCHEMA_VERSION,
        "propagation_relation_binding": PROPAGATION_RELATION_BINDING_SCHEMA_VERSION,
        "propagation_hop_audit": PROPAGATION_HOP_AUDIT_SCHEMA_VERSION,
        "propagation_path_audit": PROPAGATION_PATH_AUDIT_SCHEMA_VERSION,
        "propagation_target_score": PROPAGATION_TARGET_SCORE_SCHEMA_VERSION,
        "candidate_propagation_score": CANDIDATE_PROPAGATION_SCORE_SCHEMA_VERSION,
        "propagation_scoring_result": PROPAGATION_SCORING_RESULT_SCHEMA_VERSION,
        "scc_provenance": SCC_PROVENANCE_SCHEMA_VERSION,
        "score_decomposition": SCORE_DECOMPOSITION_SCHEMA_VERSION,
        "score_impact_compilation": SCORE_IMPACT_COMPILATION_SCHEMA_VERSION,
        "tie_group": TIE_GROUP_SCHEMA_VERSION,
        "top_k": TOP_K_SCHEMA_VERSION,
        "scoring_rule": SCORING_RULE_SCHEMA_VERSION,
        "rule_set": RULE_SET_SCHEMA_VERSION,
        "rule_candidate_assignment": RULE_CANDIDATE_ASSIGNMENT_SCHEMA_VERSION,
        "rule_contribution": RULE_CONTRIBUTION_SCHEMA_VERSION,
        "candidate_rule_score": CANDIDATE_RULE_SCORE_SCHEMA_VERSION,
        "rule_scoring_result": RULE_SCORING_RESULT_SCHEMA_VERSION,
        "history_context": HISTORY_CONTEXT_SCHEMA_VERSION,
        "history_generation_authority": HISTORY_GENERATION_AUTHORITY_SCHEMA_VERSION,
        "history_base_candidate_score": HISTORY_BASE_CANDIDATE_SCORE_SCHEMA_VERSION,
        "history_library_entry": HISTORY_LIBRARY_ENTRY_SCHEMA_VERSION,
        "frozen_history_library": FROZEN_HISTORY_LIBRARY_SCHEMA_VERSION,
        "history_query_embedding": HISTORY_QUERY_EMBEDDING_SCHEMA_VERSION,
        "history_service_mapping": HISTORY_SERVICE_MAPPING_SCHEMA_VERSION,
        "history_scoring_config": HISTORY_SCORING_CONFIG_SCHEMA_VERSION,
        "history_neighbor_audit": HISTORY_NEIGHBOR_AUDIT_SCHEMA_VERSION,
        "history_hint_reference": HISTORY_HINT_REFERENCE_SCHEMA_VERSION,
        "history_score_contribution": HISTORY_SCORE_CONTRIBUTION_SCHEMA_VERSION,
        "candidate_history_score": CANDIDATE_HISTORY_SCORE_SCHEMA_VERSION,
        "history_evaluation_result": HISTORY_EVALUATION_RESULT_SCHEMA_VERSION,
        "reference_scoring_config": REFERENCE_SCORING_CONFIG_SCHEMA_VERSION,
        "reference_ranking_result": REFERENCE_RANKING_RESULT_SCHEMA_VERSION,
        "ranking_replay_result": RANKING_REPLAY_RESULT_SCHEMA_VERSION,
        "sensitivity_candidate_terms": SENSITIVITY_CANDIDATE_TERMS_SCHEMA_VERSION,
        "model_sensitivity_request": MODEL_SENSITIVITY_REQUEST_SCHEMA_VERSION,
        "candidate_rank_sensitivity": CANDIDATE_RANK_SENSITIVITY_SCHEMA_VERSION,
        "model_sensitivity_result": MODEL_SENSITIVITY_RESULT_SCHEMA_VERSION,
        "frozen_report_context": FROZEN_REPORT_CONTEXT_SCHEMA_VERSION,
        "frozen_report_payload_v2": FROZEN_REPORT_PAYLOAD_V2_SCHEMA_VERSION,
        "frozen_report_context_v4": FROZEN_REPORT_CONTEXT_V4_SCHEMA_VERSION,
        "report_composition_response": REPORT_COMPOSITION_RESPONSE_SCHEMA_VERSION,
        "report_composition_mechanical_validation": REPORT_COMPOSITION_MECHANICAL_SCHEMA_VERSION,
        "report_composition_presentation": REPORT_COMPOSITION_PRESENTATION_SCHEMA_VERSION,
        "report_composition_explainability": REPORT_COMPOSITION_EXPLAINABILITY_SCHEMA_VERSION,
        "report_run_policy": REPORT_RUN_POLICY_SCHEMA_VERSION,
        "report_run_audit": REPORT_RUN_AUDIT_SCHEMA_VERSION,
        "report_evidence_policy": REPORT_EVIDENCE_POLICY_SCHEMA_VERSION,
        "report_evidence_view": REPORT_EVIDENCE_VIEW_SCHEMA_VERSION,
        "report_evidence_projection": REPORT_EVIDENCE_PROJECTION_SCHEMA_VERSION,
        "deterministic_sre_report": DETERMINISTIC_SRE_REPORT_SCHEMA_VERSION,
        "report_grounding_catalog": REPORT_GROUNDING_CATALOG_SCHEMA_VERSION,
        "report_grounding_validation": REPORT_GROUNDING_VALIDATION_SCHEMA_VERSION,
        "report_generation_audit": REPORT_GENERATION_AUDIT_SCHEMA_VERSION,
        "report_generation_failure_audit": (REPORT_GENERATION_FAILURE_AUDIT_SCHEMA_VERSION),
        "interpretability_artifact": INTERPRETABILITY_ARTIFACT_SCHEMA_VERSION,
        "explainability_artifact": EXPLAINABILITY_ARTIFACT_SCHEMA_VERSION,
        "dual_explanation_manifest": DUAL_EXPLANATION_MANIFEST_SCHEMA_VERSION,
        "cache_identity": CACHE_IDENTITY_SCHEMA_VERSION,
        "attempt_ledger": ATTEMPT_LEDGER_SCHEMA_VERSION,
        "cost_record": COST_RECORD_SCHEMA_VERSION,
        "semantic_call": SEMANTIC_CALL_SCHEMA_VERSION,
        "semantic_compilation": SEMANTIC_COMPILATION_SCHEMA_VERSION,
        "semantic_run_audit": SEMANTIC_RUN_AUDIT_SCHEMA_VERSION,
        "semantic_proposal_response": SEMANTIC_PROPOSAL_RESPONSE_SCHEMA_VERSION,
        "case_checkpoint": CASE_CHECKPOINT_SCHEMA_VERSION,
        "case_exposure_reference": CASE_EXPOSURE_SCHEMA_VERSION,
        "case_evaluation_output": CASE_EVALUATION_SCHEMA_VERSION,
        "sealed_audit_package": SEALED_AUDIT_SCHEMA_VERSION,
        "case_state_entry": CASE_STATE_ENTRY_SCHEMA_VERSION,
        "aiops25_adapter_configuration": (AIOPS25_ADAPTER_CONFIGURATION_SCHEMA_VERSION),
        "aiops25_adapter_result": AIOPS25_ADAPTER_RESULT_SCHEMA_VERSION,
        "rcabench_adapter_configuration": (RCABENCH_ADAPTER_CONFIGURATION_SCHEMA_VERSION),
        "rcabench_adapter_result": RCABENCH_ADAPTER_RESULT_SCHEMA_VERSION,
        "signal_transform_result": SIGNAL_TRANSFORM_RESULT_SCHEMA_VERSION,
        "entity_metadata_record": ENTITY_METADATA_RECORD_SCHEMA_VERSION,
        "observation_binding_resolution": (OBSERVATION_BINDING_RESOLUTION_SCHEMA_VERSION),
        "temporal_entity_registry": TEMPORAL_ENTITY_REGISTRY_SCHEMA_VERSION,
        "inference_boundary_audit": INFERENCE_BOUNDARY_AUDIT_SCHEMA_VERSION,
    }
)

RECOGNIZED_LEGACY_SCHEMA_VERSIONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "event_group": (LEGACY_EVENT_GROUP_SCHEMA_VERSION,),
        "frozen_report_context": (
            LEGACY_FROZEN_REPORT_CONTEXT_SCHEMA_VERSION,
            LEGACY_FROZEN_REPORT_CONTEXT_V2_SCHEMA_VERSION,
        ),
        "graph_snapshot": (
            "nurse.v2.graph-snapshot/1",
            "nurse.v2.graph-snapshot/2",
            "nurse.v2.graph-snapshot/3",
        ),
        "graph_input_decision": ("nurse.v2.graph-input-decision/1",),
        "canonical_graph_coverage_input": ("nurse.v2.canonical-graph-coverage-input/1",),
        "canonical_graph_source_input": ("nurse.v2.canonical-graph-source-input/1",),
        "graph_snapshot_pair": (
            "nurse.v2.graph-snapshot-pair/1",
            "nurse.v2.graph-snapshot-pair/2",
            "nurse.v2.graph-snapshot-pair/3",
            "nurse.v2.graph-snapshot-pair/4",
        ),
        "graph_source_record": (
            "nurse.v2.graph-source-record/1",
            "nurse.v2.graph-source-record/2",
            "nurse.v2.graph-source-record/3",
        ),
        "graph_window_coverage": (
            "nurse.v2.graph-window-coverage/1",
            "nurse.v2.graph-window-coverage/2",
            "nurse.v2.graph-window-coverage/3",
        ),
        "observation": (LEGACY_OBSERVATION_SCHEMA_VERSION,),
        "onset_interval": ("nurse.v2.onset-interval/1",),
        "predicate_result": ("nurse.v2.predicate-result/1",),
        "verification_request": (LEGACY_VERIFICATION_REQUEST_SCHEMA_VERSION,),
        "rule_hypothesis_envelope": ("nurse.v2.rule-hypothesis-envelope/1",),
        "rule_hypothesis_set": ("nurse.v2.rule-hypothesis-set/1",),
        "top_k": (LEGACY_TOP_K_SCHEMA_VERSION,),
        "temporal_precedence": (
            "nurse.v2.temporal-precedence/1",
            "nurse.v2.temporal-precedence/2",
        ),
    }
)

# No legacy-to-current migration is declared for these exact-version contracts.
# Adding an edge must be accompanied by a named transformer and compatibility fixture.
DECLARED_MIGRATION_EDGES: tuple[tuple[str, str], ...] = ()


class SchemaCompatibilityError(ValueError):
    """Raised when an artifact cannot be read under an exact declared schema."""


class SchemaMigrationError(ValueError):
    """Raised when migration is unsafe, in-place, or not explicitly declared."""


class CompatibleArtifact(StrictFrozenModel):
    """Canonical defensive copy accepted at a declared schema boundary."""

    artifact_kind: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    document_json: str = Field(min_length=2)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sealed: bool

    @property
    def document(self) -> dict[str, Any]:
        value = json.loads(self.document_json)
        if not isinstance(value, dict):  # pragma: no cover - construction invariant
            raise AssertionError("compatible artifact is not an object")
        return value

    def verify(self) -> bool:
        try:
            document = self.document
            data = canonical_json_bytes(document)
        except (TypeError, ValueError):
            return False
        return (
            document.get("schema_version") == self.schema_version
            and data.decode("utf-8") == self.document_json
            and sha256_hex(data) == self.artifact_sha256
        )


class SchemaCompatibilitySnapshot(StrictFrozenModel):
    """Hash-frozen registry used as a static compatibility fixture."""

    schema_version: Literal["nurse.v2.schema-compatibility-snapshot/1"] = (
        SCHEMA_COMPATIBILITY_SNAPSHOT_VERSION
    )
    versions: dict[str, str]
    migration_edges: tuple[tuple[str, str], ...]
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if self.versions != dict(sorted(self.versions.items())):
            return False
        if self.migration_edges != tuple(sorted(set(self.migration_edges))):
            return False
        unsigned = self.model_dump(mode="json", exclude={"snapshot_sha256"})
        return self.snapshot_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_schema_compatibility_snapshot() -> SchemaCompatibilitySnapshot:
    """Freeze the exact schemas and migration edges supported by this build."""

    unsigned: dict[str, object] = {
        "schema_version": SCHEMA_COMPATIBILITY_SNAPSHOT_VERSION,
        "versions": dict(sorted(CURRENT_SCHEMA_VERSIONS.items())),
        "migration_edges": DECLARED_MIGRATION_EDGES,
    }
    snapshot = SchemaCompatibilitySnapshot.model_validate(
        {
            **unsigned,
            "snapshot_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not snapshot.verify():  # pragma: no cover - construction invariant
        raise AssertionError("schema compatibility snapshot verification failed")
    return snapshot


def load_schema_compatibility_snapshot(path: Path) -> SchemaCompatibilitySnapshot:
    """Load only the byte-canonical fixture for the exact running registry."""

    try:
        data = Path(path).read_bytes()
        document = json.loads(data)
        snapshot = SchemaCompatibilitySnapshot.model_validate(document)
    except (OSError, TypeError, ValueError) as error:
        raise SchemaCompatibilityError(
            "schema compatibility snapshot is unreadable or invalid"
        ) from error
    expected = build_schema_compatibility_snapshot()
    if not snapshot.verify() or snapshot != expected:
        raise SchemaCompatibilityError(
            "schema compatibility snapshot does not match the running registry"
        )
    return snapshot


def load_compatible_artifact(
    document: Mapping[str, Any],
    *,
    artifact_kind: str,
    sealed: bool = False,
    expected_artifact_sha256: str | None = None,
) -> CompatibleArtifact:
    """Accept only an exact current schema and return a canonical frozen copy."""

    expected_version = CURRENT_SCHEMA_VERSIONS.get(artifact_kind)
    if expected_version is None:
        raise SchemaCompatibilityError(f"unknown artifact kind: {artifact_kind}")
    try:
        data = canonical_json_bytes(dict(document))
        copied = json.loads(data)
    except (TypeError, ValueError) as error:
        raise SchemaCompatibilityError("artifact is not canonical JSON") from error
    actual_version = copied.get("schema_version")
    if actual_version != expected_version:
        if actual_version in RECOGNIZED_LEGACY_SCHEMA_VERSIONS.get(artifact_kind, ()):
            raise SchemaCompatibilityError(
                f"recognized legacy schema for {artifact_kind}: {actual_version!r}; "
                "an explicit migration is required"
            )
        raise SchemaCompatibilityError(
            f"incompatible schema for {artifact_kind}: "
            f"expected {expected_version}, got {actual_version!r}"
        )
    digest = sha256_hex(data)
    if sealed and expected_artifact_sha256 is None:
        raise SchemaCompatibilityError("sealed artifact requires original digest verification")
    if expected_artifact_sha256 is not None and digest != expected_artifact_sha256:
        raise SchemaCompatibilityError("sealed artifact digest mismatch")
    artifact = CompatibleArtifact(
        artifact_kind=artifact_kind,
        schema_version=expected_version,
        document_json=data.decode("utf-8"),
        artifact_sha256=digest,
        sealed=sealed,
    )
    if not artifact.verify():  # pragma: no cover - construction invariant
        raise AssertionError("compatible artifact verification failed")
    return artifact


def migrate_artifact(
    document: Mapping[str, Any],
    *,
    artifact_kind: str,
    target_schema_version: str,
    sealed: bool,
    expected_source_sha256: str | None = None,
) -> CompatibleArtifact:
    """Deny migration until a named non-mutating edge is explicitly registered."""

    source = load_compatible_artifact(
        document,
        artifact_kind=artifact_kind,
        sealed=sealed,
        expected_artifact_sha256=expected_source_sha256,
    )
    if sealed:
        raise SchemaMigrationError("sealed artifacts cannot be migrated or mutated in place")
    edge = (source.schema_version, target_schema_version)
    if edge not in DECLARED_MIGRATION_EDGES:
        raise SchemaMigrationError(f"no declared migration from {edge[0]} to {edge[1]}")
    raise SchemaMigrationError("declared migration has no registered transformer")
