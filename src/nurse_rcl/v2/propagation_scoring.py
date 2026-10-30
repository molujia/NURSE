"""Bounded, replayable propagation-path scoring for NURSE V2."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, TypeVar, cast

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.context_expansion import CardExpansionPolicy
from nurse_rcl.v2.diagnostic_schemas import PredicateResult, RoleAssertion
from nurse_rcl.v2.event_grouping import EventGroup, EventGroupingResult
from nurse_rcl.v2.graph_snapshots import GraphSnapshotEdge
from nurse_rcl.v2.hypothesis_guard import HypothesisGuardResult
from nurse_rcl.v2.mechanism_library import MechanismLibrary
from nurse_rcl.v2.observation_schemas import CandidateUniverse, TimeInterval
from nurse_rcl.v2.query_session import CaseQuerySessionCoordinator, CaseQuerySessionResult
from nurse_rcl.v2.role_compiler import RoleBindingProposal, RoleCompilationResult
from nurse_rcl.v2.semantic_context import SemanticContextBundle
from nurse_rcl.v2.temporal_condensation import (
    DirectedPropagationRelation,
    OnsetInterval,
    RelationCondensedEdge,
    RelationSCCComponent,
    RelationSCCCondensation,
    TemporalPrecedence,
)
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

PROPAGATION_SCORING_CONFIG_SCHEMA_VERSION: Literal["nurse.v2.propagation-scoring-config/1"] = (
    "nurse.v2.propagation-scoring-config/1"
)
PROPAGATION_HOP_AUDIT_SCHEMA_VERSION: Literal["nurse.v2.propagation-hop-audit/1"] = (
    "nurse.v2.propagation-hop-audit/1"
)
PROPAGATION_PATH_AUDIT_SCHEMA_VERSION: Literal["nurse.v2.propagation-path-audit/1"] = (
    "nurse.v2.propagation-path-audit/1"
)
PROPAGATION_TARGET_SCORE_SCHEMA_VERSION: Literal["nurse.v2.propagation-target-score/1"] = (
    "nurse.v2.propagation-target-score/1"
)
CANDIDATE_PROPAGATION_SCORE_SCHEMA_VERSION: Literal["nurse.v2.candidate-propagation-score/1"] = (
    "nurse.v2.candidate-propagation-score/1"
)
PROPAGATION_SCORING_RESULT_SCHEMA_VERSION: Literal["nurse.v2.propagation-scoring-result/1"] = (
    "nurse.v2.propagation-scoring-result/1"
)
PROPAGATION_RELATION_BINDING_SCHEMA_VERSION: Literal["nurse.v2.propagation-relation-binding/1"] = (
    "nurse.v2.propagation-relation-binding/1"
)

PROPAGATION_POLICY_VERSION: Literal["nurse-v2-bounded-propagation/1"] = (
    "nurse-v2-bounded-propagation/1"
)
COVERAGE_POLICY_VERSION: Literal["nurse-v2-condensed-hop-coverage/1"] = (
    "nurse-v2-condensed-hop-coverage/1"
)
ALLOWED_PROPAGATION_MAX_HOPS = (1, 2, 3, 5)
PROPAGATION_GAMMA: float = 0.25

_MECHANISM_ROLE_RELATION_POLICY: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("incoming_load_pressure", "source_support", ("caller_load_transmission",)),
    ("incoming_load_pressure", "symptom_support", ("caller_load_transmission",)),
    (
        "local_resource_or_execution_pressure",
        "source_support",
        ("received_dependency_failure",),
    ),
    (
        "local_resource_or_execution_pressure",
        "symptom_support",
        ("caller_load_transmission",),
    ),
    (
        "local_storage_or_network_failure",
        "source_support",
        ("received_dependency_failure",),
    ),
    ("received_dependency_failure", "symptom_support", ("received_dependency_failure",)),
    ("workload_unavailability", "source_support", ("received_dependency_failure",)),
)
PROPAGATION_RELATION_POLICY_VERSION: Literal["nurse-v2-propagation-relation-policy/1"] = (
    "nurse-v2-propagation-relation-policy/1"
)
PROPAGATION_RELATION_POLICY_SHA256 = sha256_hex(
    canonical_json_bytes(
        {
            "policy_version": PROPAGATION_RELATION_POLICY_VERSION,
            "bindings": _MECHANISM_ROLE_RELATION_POLICY,
        }
    )
)

PathReason = Literal[
    "credited",
    "hop_limit_exceeded",
    "insufficient_path_coverage",
    "missing_temporal_precedence",
    "same_scc_no_internal_order",
    "source_onset_binding_ambiguous",
    "source_onset_binding_missing",
    "source_role_interval_mismatch",
    "target_onset_binding_ambiguous",
    "target_onset_binding_missing",
    "target_role_interval_mismatch",
    "temporal_equal_or_overlapping",
    "temporal_missing_onset",
    "temporal_onset_outside_time_contract",
    "temporal_unresolved_clock_uncertainty",
]
CandidateReason = Literal[
    "missing_source_support",
    "missing_target_symptom_support",
    "no_credited_path",
    "no_legal_typed_path",
    "propagation_credit_computed",
]
SelectionStatus = Literal["selected", "same_target_not_selected", "rejected"]

_MAX_INPUT_ARTIFACTS = 32_768
_MAX_ENUMERATED_PATHS = 100_000
_MAX_CONDENSATION_COMPONENTS = 32_768
_MAX_CONDENSATION_NODES = 32_768
_MAX_CONDENSATION_EDGES = 131_072
_MAX_SELECTED_ORIGINAL_EDGES = 262_144
_MAX_BUNDLE_PAIRS = 32_768
_MAX_EXPANDED_STATES = 100_000
_MAX_TRAVERSED_EDGES = 500_000
_ModelT = TypeVar("_ModelT", bound=StrictFrozenModel)
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _stable_id(prefix: str, value: object) -> tuple[str, str]:
    digest = _digest(value)
    return f"{prefix}_{digest[:24]}", digest


def _top_two_f_gamma(values: Sequence[float], *, gamma: float) -> float:
    if not values:
        return 0.0
    first = values[0]
    if len(values) == 1:
        return first
    return first + gamma * (1.0 - first) * values[1]


def _same_runtime_shape(value: object, rebuilt: object) -> bool:
    if type(value) is not type(rebuilt):
        return False
    if isinstance(rebuilt, StrictFrozenModel):
        try:
            left = object.__getattribute__(value, "__dict__")
            right = object.__getattribute__(rebuilt, "__dict__")
            left_extra = object.__getattribute__(value, "__pydantic_extra__")
            right_extra = object.__getattribute__(rebuilt, "__pydantic_extra__")
            left_private = object.__getattribute__(value, "__pydantic_private__")
            right_private = object.__getattribute__(rebuilt, "__pydantic_private__")
            left_fields = object.__getattribute__(value, "__pydantic_fields_set__")
            right_fields = object.__getattribute__(rebuilt, "__pydantic_fields_set__")
        except AttributeError:
            return False
        if (
            type(left) is not dict
            or type(right) is not dict
            or sorted(left) != sorted(right)
            or left_extra is not None
            or right_extra is not None
            or left_private is not None
            or right_private is not None
            or type(left_fields) is not set
            or type(right_fields) is not set
            or left_fields != right_fields
        ):
            return False
        return all(
            _same_runtime_shape(dict.__getitem__(left, key), dict.__getitem__(right, key))
            for key in right
        )
    if type(rebuilt) in {tuple, list}:
        left_sequence = cast(Sequence[object], value)
        right_sequence = cast(Sequence[object], rebuilt)
        return len(left_sequence) == len(right_sequence) and all(
            _same_runtime_shape(left_item, right_item)
            for left_item, right_item in zip(left_sequence, right_sequence, strict=True)
        )
    if type(rebuilt) is dict:
        left_mapping = cast(dict[object, object], value)
        right_mapping = cast(dict[object, object], rebuilt)
        return set(left_mapping) == set(right_mapping) and all(
            _same_runtime_shape(left_mapping[key], right_mapping[key]) for key in right_mapping
        )
    return type(value) is type(rebuilt)


def _trusted_model(value: object, model_type: type[_ModelT], *, label: str) -> _ModelT:
    if type(value) is not model_type:
        raise ValueError(f"{label} must use the exact model type")
    try:
        document = model_type.model_dump(value, mode="json", warnings="error")
        rebuilt = model_type.model_validate(document)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{label} is not a canonical model") from error
    if (
        not _same_runtime_shape(value, rebuilt)
        or model_type.model_dump(rebuilt, mode="json", warnings="error") != document
    ):
        raise ValueError(f"{label} is not an exact canonical model")
    return rebuilt


def _plain_models(
    values: object,
    model_type: type[_ModelT],
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[_ModelT, ...]:
    if type(values) not in {tuple, list}:
        raise ValueError(f"{label} must use an exact tuple or list")
    sequence = cast(tuple[object, ...] | list[object], values)
    if len(sequence) > _MAX_INPUT_ARTIFACTS or (not allow_empty and not sequence):
        raise ValueError(f"{label} size is outside the scoring limit")
    return tuple(_trusted_model(item, model_type, label=label) for item in sequence)


def _require_unique_artifact_ids(
    values: Sequence[StrictFrozenModel], *, id_field: str, label: str
) -> None:
    identifiers = tuple(cast(str, getattr(item, id_field)) for item in values)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate {label} are not permitted")


class PropagationScoringConfig(StrictFrozenModel):
    """Frozen Dev configuration for bounded path enumeration and fan-out."""

    schema_version: Literal["nurse.v2.propagation-scoring-config/1"] = (
        PROPAGATION_SCORING_CONFIG_SCHEMA_VERSION
    )
    config_id: str = Field(pattern=r"^pcfg_[0-9a-f]{24}$")
    policy_version: Literal["nurse-v2-bounded-propagation/1"] = PROPAGATION_POLICY_VERSION
    coverage_policy_version: Literal["nurse-v2-condensed-hop-coverage/1"] = COVERAGE_POLICY_VERSION
    propagation_max_hops: Literal[1, 2, 3, 5] = 2
    max_bundle_pairs: int = Field(default=_MAX_BUNDLE_PAIRS, ge=1, le=_MAX_BUNDLE_PAIRS)
    max_expanded_states: int = Field(default=_MAX_EXPANDED_STATES, ge=1, le=_MAX_EXPANDED_STATES)
    max_traversed_edges: int = Field(default=_MAX_TRAVERSED_EDGES, ge=1, le=_MAX_TRAVERSED_EDGES)
    gamma: float = Field(
        default=PROPAGATION_GAMMA,
        ge=PROPAGATION_GAMMA,
        le=PROPAGATION_GAMMA,
        allow_inf_nan=False,
    )
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_content_addressed_config(self) -> Self:
        unsigned = self.model_dump(mode="json", exclude={"config_id", "config_sha256"})
        identifier, digest = _stable_id("pcfg", unsigned)
        if self.config_id != identifier or self.config_sha256 != digest:
            raise ValueError("propagation scoring config is not content-addressed")
        return self

    def verify(self) -> bool:
        try:
            rebuilt = _trusted_model(self, PropagationScoringConfig, label="propagation config")
        except (AttributeError, TypeError, ValueError):
            return False
        return rebuilt == self


def build_propagation_scoring_config(
    *,
    propagation_max_hops: Literal[1, 2, 3, 5] = 2,
    max_bundle_pairs: int = _MAX_BUNDLE_PAIRS,
    max_expanded_states: int = _MAX_EXPANDED_STATES,
    max_traversed_edges: int = _MAX_TRAVERSED_EDGES,
) -> PropagationScoringConfig:
    if type(propagation_max_hops) is not int or propagation_max_hops not in (
        ALLOWED_PROPAGATION_MAX_HOPS
    ):
        raise ValueError("propagation_max_hops must be a predeclared Dev value: 1, 2, 3, or 5")
    for value, maximum, name in (
        (max_bundle_pairs, _MAX_BUNDLE_PAIRS, "max_bundle_pairs"),
        (max_expanded_states, _MAX_EXPANDED_STATES, "max_expanded_states"),
        (max_traversed_edges, _MAX_TRAVERSED_EDGES, "max_traversed_edges"),
    ):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be a positive bounded integer")
    unsigned: dict[str, object] = {
        "schema_version": PROPAGATION_SCORING_CONFIG_SCHEMA_VERSION,
        "policy_version": PROPAGATION_POLICY_VERSION,
        "coverage_policy_version": COVERAGE_POLICY_VERSION,
        "propagation_max_hops": propagation_max_hops,
        "max_bundle_pairs": max_bundle_pairs,
        "max_expanded_states": max_expanded_states,
        "max_traversed_edges": max_traversed_edges,
        "gamma": PROPAGATION_GAMMA,
    }
    identifier, digest = _stable_id("pcfg", unsigned)
    return PropagationScoringConfig.model_validate(
        {**unsigned, "config_id": identifier, "config_sha256": digest}
    )


class PropagationRelationBinding(StrictFrozenModel):
    """Compiler-derived binding from an accepted role to one graph relation type."""

    schema_version: Literal["nurse.v2.propagation-relation-binding/1"] = (
        PROPAGATION_RELATION_BINDING_SCHEMA_VERSION
    )
    binding_id: str = Field(pattern=r"^prbind_[0-9a-f]{24}$")
    role_compilation_id: str = Field(pattern=r"^rcr_[0-9a-f]{24}$")
    role_compilation_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assertion_id: str = Field(pattern=r"^rol_[0-9a-f]{24}$")
    hypothesis_id: str = Field(pattern=r"^hyp_[0-9a-f]{24}$")
    mechanism_key: str = Field(min_length=1)
    role_relation_id: str = Field(min_length=1)
    role: Literal["source_support", "symptom_support"]
    relation_type: DirectedPropagationRelation
    policy_version: Literal["nurse-v2-propagation-relation-policy/1"] = (
        PROPAGATION_RELATION_POLICY_VERSION
    )
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            _trusted_model(self, PropagationRelationBinding, label="propagation relation binding")
            policy = {
                (mechanism_key, role): relation_types
                for mechanism_key, role, relation_types in _MECHANISM_ROLE_RELATION_POLICY
            }
            if (
                self.policy_sha256 != PROPAGATION_RELATION_POLICY_SHA256
                or self.relation_type not in policy.get((self.mechanism_key, self.role), ())
            ):
                return False
            unsigned = self.model_dump(mode="json", exclude={"binding_id", "binding_sha256"})
            identifier, digest = _stable_id("prbind", unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return self.binding_id == identifier and self.binding_sha256 == digest


def _compile_relation_bindings(
    *,
    role_compilation: RoleCompilationResult,
    guard_results: Sequence[HypothesisGuardResult],
) -> tuple[PropagationRelationBinding, ...]:
    guard_by_hypothesis = {
        guard.parsed_hypothesis.hypothesis_id: guard
        for guard in guard_results
        if guard.parsed_hypothesis is not None
    }
    policy = {
        (mechanism_key, role): relation_types
        for mechanism_key, role, relation_types in _MECHANISM_ROLE_RELATION_POLICY
    }
    bindings: list[PropagationRelationBinding] = []
    for assertion in role_compilation.assertions:
        if (
            assertion.lifecycle_status != "accepted"
            or assertion.role not in {"source_support", "symptom_support"}
            or assertion.strength <= 0.0
        ):
            continue
        guard = guard_by_hypothesis.get(assertion.hypothesis_id)
        if guard is None or guard.parsed_hypothesis is None:
            raise ValueError("accepted role has no replayed hypothesis owner")
        mechanism_key = guard.parsed_hypothesis.mechanism_key
        for relation_type in policy.get((mechanism_key, assertion.role), ()):
            unsigned: dict[str, object] = {
                "schema_version": PROPAGATION_RELATION_BINDING_SCHEMA_VERSION,
                "role_compilation_id": role_compilation.role_compilation_id,
                "role_compilation_result_sha256": role_compilation.result_sha256,
                "assertion_id": assertion.assertion_id,
                "hypothesis_id": assertion.hypothesis_id,
                "mechanism_key": mechanism_key,
                "role_relation_id": assertion.relation_id,
                "role": assertion.role,
                "relation_type": relation_type,
                "policy_version": PROPAGATION_RELATION_POLICY_VERSION,
                "policy_sha256": PROPAGATION_RELATION_POLICY_SHA256,
            }
            identifier, digest = _stable_id("prbind", unsigned)
            bindings.append(
                PropagationRelationBinding.model_validate(
                    {**unsigned, "binding_id": identifier, "binding_sha256": digest}
                )
            )
    return tuple(sorted(bindings, key=lambda item: item.binding_id))


class PropagationHopAudit(StrictFrozenModel):
    schema_version: Literal["nurse.v2.propagation-hop-audit/1"] = (
        PROPAGATION_HOP_AUDIT_SCHEMA_VERSION
    )
    hop_audit_id: str = Field(pattern=r"^phop_[0-9a-f]{24}$")
    hop_index: int = Field(ge=1, le=_MAX_CONDENSATION_COMPONENTS)
    condensed_edge_id: str = Field(pattern=r"^rcedge_[0-9a-f]{24}$")
    source_component_id: str = Field(pattern=r"^rscc_[0-9a-f]{24}$")
    target_component_id: str = Field(pattern=r"^rscc_[0-9a-f]{24}$")
    relation_type: DirectedPropagationRelation
    retained_original_edge_ids: tuple[str, ...] = Field(min_length=1)
    sufficient_original_edge_ids: tuple[str, ...]
    selected_original_edge_id: str | None = Field(default=None, pattern=r"^gedge_[0-9a-f]{24}$")
    hop_quality: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    coverage_status: Literal["sufficient", "insufficient"]
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    hop_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            _trusted_model(self, PropagationHopAudit, label="propagation hop audit")
            for values in (
                self.retained_original_edge_ids,
                self.sufficient_original_edge_ids,
                self.provenance_ids,
            ):
                if values != tuple(sorted(set(values))):
                    return False
            if self.coverage_status == "sufficient":
                if self.selected_original_edge_id not in self.sufficient_original_edge_ids:
                    return False
                if self.hop_quality is None:
                    return False
            elif self.selected_original_edge_id is not None or self.hop_quality is not None:
                return False
            unsigned = self.model_dump(mode="json", exclude={"hop_audit_id", "hop_audit_sha256"})
            identifier, digest = _stable_id("phop", unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return self.hop_audit_id == identifier and self.hop_audit_sha256 == digest


class PropagationPathAudit(StrictFrozenModel):
    schema_version: Literal["nurse.v2.propagation-path-audit/1"] = (
        PROPAGATION_PATH_AUDIT_SCHEMA_VERSION
    )
    path_audit_id: str = Field(pattern=r"^ppath_[0-9a-f]{24}$")
    candidate_service_key: str = Field(min_length=1)
    source_event_group_id: str = Field(pattern=r"^evt_[0-9a-f]{24}$")
    target_service_key: str = Field(min_length=1)
    target_event_group_id: str = Field(pattern=r"^evt_[0-9a-f]{24}$")
    relation_type: DirectedPropagationRelation
    condensation_id: str = Field(pattern=r"^rcond_[0-9a-f]{24}$")
    component_ids: tuple[str, ...] = Field(min_length=1)
    condensed_edge_ids: tuple[str, ...]
    hops: tuple[PropagationHopAudit, ...]
    path_length: int = Field(ge=0, le=_MAX_CONDENSATION_COMPONENTS)
    source_assertion_ids: tuple[str, ...] = Field(min_length=1)
    source_contributing_assertion_id: str | None = Field(
        default=None, pattern=r"^rol_[0-9a-f]{24}$"
    )
    target_assertion_ids: tuple[str, ...] = Field(min_length=1)
    target_contributing_assertion_id: str | None = Field(
        default=None, pattern=r"^rol_[0-9a-f]{24}$"
    )
    source_onset_id: str | None = Field(default=None, pattern=r"^onset_[0-9a-f]{24}$")
    target_onset_id: str | None = Field(default=None, pattern=r"^onset_[0-9a-f]{24}$")
    source_onset_option_ids: tuple[str, ...]
    target_onset_option_ids: tuple[str, ...]
    source_binding_observation_ids: tuple[str, ...]
    target_binding_observation_ids: tuple[str, ...]
    precedence_id: str | None = Field(default=None, pattern=r"^precedence_[0-9a-f]{24}$")
    source_support: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    target_symptom_strength: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    time_ok: bool
    path_quality: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    limiting_hop_indices: tuple[int, ...]
    raw_contribution: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    selection_status: SelectionStatus
    reason_codes: tuple[PathReason, ...] = Field(min_length=1)
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    path_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            _trusted_model(self, PropagationPathAudit, label="propagation path audit")
            if any(not PropagationHopAudit.verify(item) for item in self.hops):
                return False
            for values in (
                self.source_assertion_ids,
                self.target_assertion_ids,
                self.source_onset_option_ids,
                self.target_onset_option_ids,
                self.source_binding_observation_ids,
                self.target_binding_observation_ids,
                self.provenance_ids,
            ):
                if values != tuple(sorted(set(values))):
                    return False
            if self.path_length != len(self.condensed_edge_ids) or self.path_length != len(
                self.hops
            ):
                return False
            if len(self.component_ids) != self.path_length + 1:
                return False
            if tuple(item.hop_index for item in self.hops) != tuple(range(1, self.path_length + 1)):
                return False
            if tuple(item.condensed_edge_id for item in self.hops) != self.condensed_edge_ids:
                return False
            if any(
                item.relation_type != self.relation_type
                or item.source_component_id != self.component_ids[index]
                or item.target_component_id != self.component_ids[index + 1]
                for index, item in enumerate(self.hops)
            ):
                return False
            source_interval_mismatch = "source_role_interval_mismatch" in self.reason_codes
            target_interval_mismatch = "target_role_interval_mismatch" in self.reason_codes
            if (self.source_contributing_assertion_id is None) != source_interval_mismatch:
                return False
            if (self.target_contributing_assertion_id is None) != target_interval_mismatch:
                return False
            if (
                self.source_contributing_assertion_id is not None
                and self.source_contributing_assertion_id not in self.source_assertion_ids
            ):
                return False
            if (
                self.target_contributing_assertion_id is not None
                and self.target_contributing_assertion_id not in self.target_assertion_ids
            ):
                return False
            if (self.source_support == 0.0) != source_interval_mismatch:
                return False
            if (self.target_symptom_strength == 0.0) != target_interval_mismatch:
                return False
            if (self.source_onset_id is not None) != (
                self.source_onset_option_ids == (self.source_onset_id,)
            ):
                return False
            if (self.target_onset_id is not None) != (
                self.target_onset_option_ids == (self.target_onset_id,)
            ):
                return False
            credited = self.raw_contribution > 0.0
            if credited != (self.reason_codes == ("credited",)):
                return False
            insufficient_hops = tuple(
                item.hop_index for item in self.hops if item.coverage_status == "insufficient"
            )
            expected_path_quality = (
                0.0
                if insufficient_hops
                else min(
                    (cast(float, item.hop_quality) for item in self.hops),
                    default=0.0,
                )
            )
            if self.path_quality != expected_path_quality:
                return False
            expected_limiting_hops = (
                insufficient_hops
                if insufficient_hops
                else tuple(
                    item.hop_index
                    for item in self.hops
                    if item.hop_quality == expected_path_quality
                )
            )
            if self.limiting_hop_indices != expected_limiting_hops:
                return False
            if credited and (
                not self.time_ok
                or self.path_length == 0
                or any(item.coverage_status != "sufficient" for item in self.hops)
                or self.source_onset_id is None
                or self.target_onset_id is None
                or self.precedence_id is None
                or not self.source_binding_observation_ids
                or not self.target_binding_observation_ids
                or self.raw_contribution
                != self.path_quality
                * min(self.source_support, self.target_symptom_strength)
                / self.path_length
            ):
                return False
            if self.selection_status == "selected" and not credited:
                return False
            if self.selection_status == "rejected" and credited:
                return False
            unsigned = self.model_dump(mode="json", exclude={"path_audit_id", "path_audit_sha256"})
            identifier, digest = _stable_id("ppath", unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return self.path_audit_id == identifier and self.path_audit_sha256 == digest


class PropagationTargetScore(StrictFrozenModel):
    schema_version: Literal["nurse.v2.propagation-target-score/1"] = (
        PROPAGATION_TARGET_SCORE_SCHEMA_VERSION
    )
    target_score_id: str = Field(pattern=r"^ptgt_[0-9a-f]{24}$")
    candidate_service_key: str = Field(min_length=1)
    target_service_key: str = Field(min_length=1)
    target_event_group_id: str = Field(pattern=r"^evt_[0-9a-f]{24}$")
    target_service_keys: tuple[str, ...] = Field(min_length=1)
    retained_path_audit_ids: tuple[str, ...]
    selected_path_audit_id: str | None = Field(default=None, pattern=r"^ppath_[0-9a-f]{24}$")
    contribution: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    status: Literal["selected_maximum", "zero_no_credited_path"]
    target_score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            _trusted_model(self, PropagationTargetScore, label="propagation target score")
            if self.target_service_keys != (self.target_service_key,):
                return False
            if self.target_service_keys != tuple(sorted(set(self.target_service_keys))):
                return False
            if self.retained_path_audit_ids != tuple(sorted(set(self.retained_path_audit_ids))):
                return False
            if (self.selected_path_audit_id is None) != (self.contribution == 0.0):
                return False
            if self.status == "selected_maximum" and self.selected_path_audit_id is None:
                return False
            if self.status == "zero_no_credited_path" and self.selected_path_audit_id is not None:
                return False
            unsigned = self.model_dump(
                mode="json", exclude={"target_score_id", "target_score_sha256"}
            )
            identifier, digest = _stable_id("ptgt", unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return self.target_score_id == identifier and self.target_score_sha256 == digest


class CandidatePropagationScore(StrictFrozenModel):
    schema_version: Literal["nurse.v2.candidate-propagation-score/1"] = (
        CANDIDATE_PROPAGATION_SCORE_SCHEMA_VERSION
    )
    candidate_score_id: str = Field(pattern=r"^pcand_[0-9a-f]{24}$")
    candidate_service_key: str = Field(min_length=1)
    retained_target_score_ids: tuple[str, ...]
    top_target_service_keys: tuple[str, ...] = Field(max_length=2)
    top_target_event_group_ids: tuple[str, ...] = Field(max_length=2)
    top_target_contributions: tuple[UnitFloat, ...] = Field(max_length=2)
    gamma: float = Field(
        default=PROPAGATION_GAMMA,
        ge=PROPAGATION_GAMMA,
        le=PROPAGATION_GAMMA,
        allow_inf_nan=False,
    )
    propagation_score: UnitFloat
    reason_code: CandidateReason
    candidate_score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            _trusted_model(self, CandidatePropagationScore, label="candidate propagation score")
            if self.retained_target_score_ids != tuple(sorted(set(self.retained_target_score_ids))):
                return False
            if not (
                len(self.top_target_service_keys)
                == len(self.top_target_event_group_ids)
                == len(self.top_target_contributions)
            ):
                return False
            if self.top_target_service_keys != tuple(dict.fromkeys(self.top_target_service_keys)):
                return False
            expected = _top_two_f_gamma(self.top_target_contributions, gamma=self.gamma)
            if self.propagation_score != expected:
                return False
            if (self.propagation_score > 0.0) != (
                self.reason_code == "propagation_credit_computed"
            ):
                return False
            unsigned = self.model_dump(
                mode="json", exclude={"candidate_score_id", "candidate_score_sha256"}
            )
            identifier, digest = _stable_id("pcand", unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return self.candidate_score_id == identifier and self.candidate_score_sha256 == digest


class PropagationScoringResult(StrictFrozenModel):
    schema_version: Literal["nurse.v2.propagation-scoring-result/1"] = (
        PROPAGATION_SCORING_RESULT_SCHEMA_VERSION
    )
    scoring_result_id: str = Field(pattern=r"^pres_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    policy_version: Literal["nurse-v2-bounded-propagation/1"] = PROPAGATION_POLICY_VERSION
    config: PropagationScoringConfig
    authoritative_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_service_keys: tuple[str, ...] = Field(min_length=1)
    event_grouping_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    role_compilation_id: str = Field(pattern=r"^rcr_[0-9a-f]{24}$")
    role_compilation_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    relation_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    condensation_sha256s: tuple[str, ...]
    onset_sha256s: tuple[str, ...]
    precedence_sha256s: tuple[str, ...]
    role_assertion_sha256s: tuple[str, ...]
    relation_bindings: tuple[PropagationRelationBinding, ...]
    path_audits: tuple[PropagationPathAudit, ...]
    target_scores: tuple[PropagationTargetScore, ...]
    candidate_scores: tuple[CandidatePropagationScore, ...] = Field(min_length=1)
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify_structure(self) -> bool:
        try:
            _trusted_model(self, PropagationScoringResult, label="propagation scoring result")
            if not PropagationScoringConfig.verify(self.config):
                return False
            if any(not PropagationPathAudit.verify(item) for item in self.path_audits):
                return False
            if any(not PropagationTargetScore.verify(item) for item in self.target_scores):
                return False
            if any(not CandidatePropagationScore.verify(item) for item in self.candidate_scores):
                return False
            if any(not PropagationRelationBinding.verify(item) for item in self.relation_bindings):
                return False
            if self.relation_policy_sha256 != PROPAGATION_RELATION_POLICY_SHA256:
                return False
            if self.relation_bindings != tuple(
                sorted(self.relation_bindings, key=lambda item: item.binding_id)
            ):
                return False
            if any(
                item.role_compilation_id != self.role_compilation_id
                or item.role_compilation_result_sha256 != self.role_compilation_result_sha256
                for item in self.relation_bindings
            ):
                return False
            if self.path_audits != tuple(
                sorted(self.path_audits, key=lambda item: item.path_audit_id)
            ):
                return False
            if self.target_scores != tuple(
                sorted(
                    self.target_scores,
                    key=lambda item: (
                        item.candidate_service_key,
                        item.target_service_key,
                        item.target_event_group_id,
                    ),
                )
            ):
                return False
            if self.candidate_scores != tuple(
                sorted(self.candidate_scores, key=lambda item: item.candidate_service_key)
            ):
                return False
            if self.candidate_service_keys != tuple(sorted(set(self.candidate_service_keys))):
                return False
            if tuple(item.candidate_service_key for item in self.candidate_scores) != (
                self.candidate_service_keys
            ):
                return False
            for values in (
                self.condensation_sha256s,
                self.onset_sha256s,
                self.precedence_sha256s,
                self.role_assertion_sha256s,
            ):
                if values != tuple(sorted(set(values))):
                    return False
            path_by_id = {item.path_audit_id: item for item in self.path_audits}
            target_by_id = {item.target_score_id: item for item in self.target_scores}
            relation_binding_keys = {
                (item.assertion_id, item.role, item.relation_type)
                for item in self.relation_bindings
            }
            for item in self.path_audits:
                if (
                    item.source_contributing_assertion_id is not None
                    and (
                        item.source_contributing_assertion_id,
                        "source_support",
                        item.relation_type,
                    )
                    not in relation_binding_keys
                ):
                    return False
                if (
                    item.target_contributing_assertion_id is not None
                    and (
                        item.target_contributing_assertion_id,
                        "symptom_support",
                        item.relation_type,
                    )
                    not in relation_binding_keys
                ):
                    return False
            if len(path_by_id) != len(self.path_audits) or len(target_by_id) != len(
                self.target_scores
            ):
                return False
            for target in self.target_scores:
                paths = tuple(path_by_id[item] for item in target.retained_path_audit_ids)
                expected_path_ids = tuple(
                    sorted(
                        item.path_audit_id
                        for item in self.path_audits
                        if item.candidate_service_key == target.candidate_service_key
                        and item.target_service_key == target.target_service_key
                        and item.target_event_group_id == target.target_event_group_id
                    )
                )
                if target.retained_path_audit_ids != expected_path_ids or any(
                    item.candidate_service_key != target.candidate_service_key
                    or item.target_service_key != target.target_service_key
                    or item.target_event_group_id != target.target_event_group_id
                    for item in paths
                ):
                    return False
                selected = tuple(item for item in paths if item.selection_status == "selected")
                credited_paths = tuple(item for item in paths if item.raw_contribution > 0.0)
                expected_selected = (
                    None
                    if not credited_paths
                    else min(
                        credited_paths,
                        key=lambda item: (
                            -item.raw_contribution,
                            _digest(
                                item.model_dump(
                                    mode="json",
                                    exclude={
                                        "path_audit_id",
                                        "path_audit_sha256",
                                        "selection_status",
                                    },
                                )
                            ),
                        ),
                    )
                )
                if any(
                    item.selection_status
                    != (
                        "selected"
                        if item is expected_selected
                        else (
                            "same_target_not_selected"
                            if item.raw_contribution > 0.0
                            else "rejected"
                        )
                    )
                    for item in paths
                ):
                    return False
                if expected_selected is None:
                    if (
                        selected
                        or target.selected_path_audit_id is not None
                        or target.contribution != 0.0
                    ):
                        return False
                elif (
                    len(selected) != 1
                    or selected[0] is not expected_selected
                    or target.selected_path_audit_id != expected_selected.path_audit_id
                    or target.contribution != expected_selected.raw_contribution
                ):
                    return False
            for candidate in self.candidate_scores:
                targets = tuple(target_by_id[item] for item in candidate.retained_target_score_ids)
                expected_target_ids = tuple(
                    sorted(
                        item.target_score_id
                        for item in self.target_scores
                        if item.candidate_service_key == candidate.candidate_service_key
                    )
                )
                if candidate.retained_target_score_ids != expected_target_ids or any(
                    item.candidate_service_key != candidate.candidate_service_key
                    for item in targets
                ):
                    return False
                best_by_service: dict[str, PropagationTargetScore] = {}
                for target in targets:
                    current = best_by_service.get(target.target_service_key)
                    if current is None or (-target.contribution, target.target_event_group_id) < (
                        -current.contribution,
                        current.target_event_group_id,
                    ):
                        best_by_service[target.target_service_key] = target
                expected = tuple(
                    item
                    for item in sorted(
                        best_by_service.values(),
                        key=lambda item: (
                            -item.contribution,
                            item.target_service_key,
                            item.target_event_group_id,
                        ),
                    )[:2]
                    if item.contribution > 0.0
                )
                if candidate.top_target_service_keys != tuple(
                    item.target_service_key for item in expected
                ):
                    return False
                if candidate.top_target_event_group_ids != tuple(
                    item.target_event_group_id for item in expected
                ):
                    return False
                if candidate.top_target_contributions != tuple(
                    item.contribution for item in expected
                ):
                    return False
            unsigned = self.model_dump(mode="json", exclude={"scoring_result_id", "result_sha256"})
            identifier, digest = _stable_id("pres", unsigned)
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return self.scoring_result_id == identifier and self.result_sha256 == digest

    def verify(self) -> bool:
        """Compatibility alias for structural, not authoritative, verification."""

        return self.verify_structure()

    def verify_trusted(
        self,
        *,
        expected_result_sha256: str,
        expected_authoritative_input_sha256: str,
        frozen_candidates: FrozenCandidateUniverse,
        condensations: Sequence[RelationSCCCondensation],
        event_grouping: EventGroupingResult,
        onsets: Sequence[OnsetInterval],
        precedences: Sequence[TemporalPrecedence],
        role_compilation: RoleCompilationResult,
        role_proposals: Sequence[RoleBindingProposal],
        role_guard_results: Sequence[HypothesisGuardResult],
        role_predicates: Sequence[PredicateResult],
        candidate_universe: CandidateUniverse,
        entity_registry: TemporalEntityRegistry,
        mechanism_library: MechanismLibrary,
        scope_interval: TimeInterval,
        query_sessions_by_guard_result_id: Mapping[str, CaseQuerySessionResult] | None = None,
        query_session_coordinator: CaseQuerySessionCoordinator | None = None,
        semantic_context: SemanticContextBundle | None = None,
        card_expansion_policy: CardExpansionPolicy | None = None,
        config: PropagationScoringConfig | None = None,
    ) -> bool:
        if (
            type(expected_result_sha256) is not str
            or type(expected_authoritative_input_sha256) is not str
            or self.result_sha256 != expected_result_sha256
            or self.authoritative_input_sha256 != expected_authoritative_input_sha256
            or not self.verify_structure()
        ):
            return False
        try:
            replayed = score_propagation_paths(
                frozen_candidates=frozen_candidates,
                condensations=condensations,
                event_grouping=event_grouping,
                onsets=onsets,
                precedences=precedences,
                role_compilation=role_compilation,
                role_proposals=role_proposals,
                role_guard_results=role_guard_results,
                role_predicates=role_predicates,
                candidate_universe=candidate_universe,
                entity_registry=entity_registry,
                mechanism_library=mechanism_library,
                scope_interval=scope_interval,
                query_sessions_by_guard_result_id=query_sessions_by_guard_result_id,
                query_session_coordinator=query_session_coordinator,
                semantic_context=semantic_context,
                card_expansion_policy=card_expansion_policy,
                config=config,
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return replayed == self

    def verify_inputs(
        self,
        *,
        expected_result_sha256: str,
        expected_authoritative_input_sha256: str,
        frozen_candidates: FrozenCandidateUniverse,
        condensations: Sequence[RelationSCCCondensation],
        event_grouping: EventGroupingResult,
        onsets: Sequence[OnsetInterval],
        precedences: Sequence[TemporalPrecedence],
        role_compilation: RoleCompilationResult,
        role_proposals: Sequence[RoleBindingProposal],
        role_guard_results: Sequence[HypothesisGuardResult],
        role_predicates: Sequence[PredicateResult],
        candidate_universe: CandidateUniverse,
        entity_registry: TemporalEntityRegistry,
        mechanism_library: MechanismLibrary,
        scope_interval: TimeInterval,
        query_sessions_by_guard_result_id: Mapping[str, CaseQuerySessionResult] | None = None,
        query_session_coordinator: CaseQuerySessionCoordinator | None = None,
        semantic_context: SemanticContextBundle | None = None,
        card_expansion_policy: CardExpansionPolicy | None = None,
        config: PropagationScoringConfig | None = None,
    ) -> bool:
        """Compatibility alias for externally anchored full replay verification."""

        return self.verify_trusted(
            expected_result_sha256=expected_result_sha256,
            expected_authoritative_input_sha256=expected_authoritative_input_sha256,
            frozen_candidates=frozen_candidates,
            condensations=condensations,
            event_grouping=event_grouping,
            onsets=onsets,
            precedences=precedences,
            role_compilation=role_compilation,
            role_proposals=role_proposals,
            role_guard_results=role_guard_results,
            role_predicates=role_predicates,
            candidate_universe=candidate_universe,
            entity_registry=entity_registry,
            mechanism_library=mechanism_library,
            scope_interval=scope_interval,
            query_sessions_by_guard_result_id=query_sessions_by_guard_result_id,
            query_session_coordinator=query_session_coordinator,
            semantic_context=semantic_context,
            card_expansion_policy=card_expansion_policy,
            config=config,
        )


@dataclass(frozen=True)
class _RoleBundle:
    service_key: str
    event_group: EventGroup
    assertions: tuple[RoleAssertion, ...]


@dataclass(frozen=True)
class _GraphPath:
    component_ids: tuple[str, ...]
    edges: tuple[RelationCondensedEdge, ...]


@dataclass(frozen=True)
class _CondensationIndex:
    condensation: RelationSCCCondensation
    component_by_service: dict[str, str]
    adjacency: dict[str, tuple[RelationCondensedEdge, ...]]


class PropagationScoringBudgetExceeded(ValueError):
    """Fixed-message rejection carrying a non-sensitive audit reason code."""

    def __init__(self, reason_code: str) -> None:
        super().__init__("propagation scoring budget exceeded")
        self.reason_code = reason_code


def _exact_tuple_field(value: object, *, field_name: str, label: str) -> tuple[object, ...]:
    try:
        state = object.__getattribute__(value, "__dict__")
    except AttributeError as error:
        raise ValueError(f"{label} lacks canonical model state") from error
    if type(state) is not dict or field_name not in state:
        raise ValueError(f"{label} lacks required field {field_name}")
    field_value = dict.__getitem__(state, field_name)
    if type(field_value) is not tuple:
        raise ValueError(f"{label} field {field_name} must use an exact tuple")
    return field_value


def _preflight_condensation_structure(values: object) -> None:
    """Enforce condensation cardinality before model serialization or replay."""

    if type(values) not in {tuple, list}:
        raise ValueError("condensations must use an exact tuple or list")
    sequence = cast(tuple[object, ...] | list[object], values)
    if type(sequence) is tuple:
        sequence_length = tuple.__len__(sequence)
    else:
        sequence_length = list.__len__(cast(list[object], sequence))
    if sequence_length > _MAX_INPUT_ARTIFACTS:
        raise ValueError("condensations size is outside the scoring limit")

    component_count = 0
    node_count = 0
    member_count = 0
    condensed_edge_count = 0
    selected_original_edge_count = 0
    for item in sequence:
        if type(item) is not RelationSCCCondensation:
            raise ValueError("condensations must use the exact model type")
        components = _exact_tuple_field(
            item,
            field_name="components",
            label="relation SCC condensation",
        )
        nodes = _exact_tuple_field(
            item,
            field_name="node_service_keys",
            label="relation SCC condensation",
        )
        condensed_edges = _exact_tuple_field(
            item,
            field_name="condensed_edges",
            label="relation SCC condensation",
        )
        selected_original_edges = _exact_tuple_field(
            item,
            field_name="selected_original_edges",
            label="relation SCC condensation",
        )
        component_count += tuple.__len__(components)
        node_count += tuple.__len__(nodes)
        condensed_edge_count += tuple.__len__(condensed_edges)
        selected_original_edge_count += tuple.__len__(selected_original_edges)
        if (
            component_count > _MAX_CONDENSATION_COMPONENTS
            or node_count > _MAX_CONDENSATION_NODES
            or condensed_edge_count > _MAX_CONDENSATION_EDGES
            or selected_original_edge_count > _MAX_SELECTED_ORIGINAL_EDGES
        ):
            raise PropagationScoringBudgetExceeded("condensation_structure_budget_exceeded")
        for component in components:
            if type(component) is not RelationSCCComponent:
                raise ValueError("condensation components must use the exact model type")
            members = _exact_tuple_field(
                component,
                field_name="member_service_keys",
                label="relation SCC component",
            )
            member_count += tuple.__len__(members)
            if member_count > _MAX_CONDENSATION_NODES:
                raise PropagationScoringBudgetExceeded("condensation_structure_budget_exceeded")


@dataclass
class _ComputationBudget:
    max_bundle_pairs: int
    max_expanded_states: int
    max_traversed_edges: int
    bundle_pairs: int = 0
    expanded_states: int = 0
    traversed_edges: int = 0

    def charge_bundle_pair(self) -> None:
        self.bundle_pairs += 1
        if self.bundle_pairs > self.max_bundle_pairs:
            raise PropagationScoringBudgetExceeded("bundle_pair_budget_exceeded")

    def charge_state(self) -> None:
        self.expanded_states += 1
        if self.expanded_states > self.max_expanded_states:
            raise PropagationScoringBudgetExceeded("expanded_state_budget_exceeded")

    def charge_edge(self) -> None:
        self.traversed_edges += 1
        if self.traversed_edges > self.max_traversed_edges:
            raise PropagationScoringBudgetExceeded("traversed_edge_budget_exceeded")


@dataclass(frozen=True)
class _PathDraft:
    payload: dict[str, object]
    raw_contribution: float
    provisional_sha256: str


@dataclass(frozen=True)
class _OnsetBinding:
    status: Literal["bound", "missing", "ambiguous"]
    onset: OnsetInterval | None
    option_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]


@dataclass(frozen=True)
class _VerifiedInputs:
    frozen_candidates: FrozenCandidateUniverse
    condensations: tuple[RelationSCCCondensation, ...]
    event_grouping: EventGroupingResult
    onsets: tuple[OnsetInterval, ...]
    precedences: tuple[TemporalPrecedence, ...]
    role_compilation: RoleCompilationResult
    role_proposals: tuple[RoleBindingProposal, ...]
    role_guard_results: tuple[HypothesisGuardResult, ...]
    role_predicates: tuple[PredicateResult, ...]
    candidate_universe: CandidateUniverse
    entity_registry: TemporalEntityRegistry
    mechanism_library: MechanismLibrary
    scope_interval: TimeInterval
    relation_bindings: tuple[PropagationRelationBinding, ...]
    config: PropagationScoringConfig


def _verified_inputs(
    *,
    frozen_candidates: FrozenCandidateUniverse,
    condensations: Sequence[RelationSCCCondensation],
    event_grouping: EventGroupingResult,
    onsets: Sequence[OnsetInterval],
    precedences: Sequence[TemporalPrecedence],
    role_compilation: RoleCompilationResult,
    role_proposals: Sequence[RoleBindingProposal],
    role_guard_results: Sequence[HypothesisGuardResult],
    role_predicates: Sequence[PredicateResult],
    candidate_universe: CandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    mechanism_library: MechanismLibrary,
    scope_interval: TimeInterval,
    query_sessions_by_guard_result_id: Mapping[str, CaseQuerySessionResult] | None,
    query_session_coordinator: CaseQuerySessionCoordinator | None,
    semantic_context: SemanticContextBundle | None,
    card_expansion_policy: CardExpansionPolicy | None,
    config: PropagationScoringConfig | None,
) -> _VerifiedInputs:
    _preflight_condensation_structure(condensations)
    frozen = _trusted_model(frozen_candidates, FrozenCandidateUniverse, label="candidate freeze")
    grouping = _trusted_model(event_grouping, EventGroupingResult, label="event grouping")
    condensation_values = _plain_models(
        condensations,
        RelationSCCCondensation,
        label="condensations",
        allow_empty=True,
    )
    onset_values = _plain_models(onsets, OnsetInterval, label="onsets", allow_empty=True)
    precedence_values = _plain_models(
        precedences, TemporalPrecedence, label="precedences", allow_empty=True
    )
    compilation = _trusted_model(role_compilation, RoleCompilationResult, label="role compilation")
    proposal_values = _plain_models(role_proposals, RoleBindingProposal, label="role proposals")
    guard_values = _plain_models(
        role_guard_results, HypothesisGuardResult, label="role guard results"
    )
    predicate_values = _plain_models(role_predicates, PredicateResult, label="role predicates")
    _require_unique_artifact_ids(
        proposal_values,
        id_field="proposal_id",
        label="role proposals",
    )
    _require_unique_artifact_ids(
        guard_values,
        id_field="guard_result_id",
        label="role guard results",
    )
    _require_unique_artifact_ids(
        predicate_values,
        id_field="predicate_result_id",
        label="role predicates",
    )
    universe = _trusted_model(candidate_universe, CandidateUniverse, label="candidate universe")
    registry = _trusted_model(entity_registry, TemporalEntityRegistry, label="entity registry")
    library = _trusted_model(mechanism_library, MechanismLibrary, label="mechanism library")
    scope = _trusted_model(scope_interval, TimeInterval, label="role scope interval")
    config_value = (
        build_propagation_scoring_config()
        if config is None
        else _trusted_model(config, PropagationScoringConfig, label="propagation config")
    )
    if not FrozenCandidateUniverse.verify(frozen):
        raise ValueError("candidate freeze failed content verification")
    if not EventGroupingResult.verify(grouping):
        raise ValueError("event grouping failed exact replay")
    if not PropagationScoringConfig.verify(config_value):
        raise ValueError("propagation config failed content verification")
    if not CandidateUniverse.verify(universe):
        raise ValueError("candidate universe failed content verification")
    if not TemporalEntityRegistry.verify(registry):
        raise ValueError("entity registry failed content verification")
    if not MechanismLibrary.verify(library):
        raise ValueError("mechanism library failed content verification")
    if not compilation.verify_inputs(
        proposals=proposal_values,
        guard_results=guard_values,
        predicates=predicate_values,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        event_grouping=grouping,
        mechanism_library=library,
        scope_interval=scope,
        query_sessions_by_guard_result_id=query_sessions_by_guard_result_id,
        query_session_coordinator=query_session_coordinator,
        semantic_context=semantic_context,
        card_expansion_policy=card_expansion_policy,
    ):
        raise ValueError("role compilation failed authoritative replay")
    assertion_values = _plain_models(
        compilation.assertions, RoleAssertion, label="compiled role assertions"
    )
    relation_bindings = _compile_relation_bindings(
        role_compilation=compilation,
        guard_results=guard_values,
    )
    if any(not RelationSCCCondensation.verify(item) for item in condensation_values):
        raise ValueError("relation condensation failed content verification")
    if any(not OnsetInterval.verify(item) for item in onset_values):
        raise ValueError("onset interval failed content verification")
    if any(not TemporalPrecedence.verify(item) for item in precedence_values):
        raise ValueError("temporal precedence failed content verification")
    if any(not RoleAssertion.verify(item) for item in assertion_values):
        raise ValueError("role assertion failed content verification")
    _require_unique_artifact_ids(onset_values, id_field="onset_id", label="onsets")
    _require_unique_artifact_ids(
        precedence_values,
        id_field="precedence_id",
        label="precedences",
    )
    precedence_pairs = tuple(
        (item.source_onset_id, item.target_onset_id)
        for item in precedence_values
        if item.source_onset_id is not None and item.target_onset_id is not None
    )
    if len(precedence_pairs) != len(set(precedence_pairs)):
        raise ValueError("conflicting temporal precedences are not permitted")
    _require_unique_artifact_ids(
        assertion_values,
        id_field="assertion_id",
        label="role_assertions",
    )
    case_ids = {
        frozen.opaque_case_id,
        grouping.opaque_case_id,
        *(item.opaque_case_id for item in condensation_values),
        *(item.opaque_case_id for item in onset_values),
        *(item.opaque_case_id for item in precedence_values),
    }
    if len(case_ids) != 1:
        raise ValueError("propagation inputs belong to different cases")
    relation_types = [item.relation_type for item in condensation_values]
    if len(relation_types) != len(set(relation_types)):
        raise ValueError("one relation condensation per directed relation is required")
    if any(
        item not in frozen.candidate_service_keys
        for item in grouping_service_keys(assertion_values)
    ):
        raise ValueError("role assertion references a service outside the frozen candidates")
    return _VerifiedInputs(
        frozen_candidates=frozen,
        condensations=tuple(sorted(condensation_values, key=lambda item: item.relation_type)),
        event_grouping=grouping,
        onsets=tuple(sorted(onset_values, key=lambda item: item.onset_id)),
        precedences=tuple(sorted(precedence_values, key=lambda item: item.precedence_id)),
        role_compilation=compilation,
        role_proposals=tuple(sorted(proposal_values, key=lambda item: item.proposal_id)),
        role_guard_results=tuple(sorted(guard_values, key=lambda item: item.guard_result_id)),
        role_predicates=tuple(sorted(predicate_values, key=lambda item: item.predicate_result_id)),
        candidate_universe=universe,
        entity_registry=registry,
        mechanism_library=library,
        scope_interval=scope,
        relation_bindings=relation_bindings,
        config=config_value,
    )


def grouping_service_keys(assertions: Sequence[RoleAssertion]) -> tuple[str, ...]:
    return tuple(sorted({item.candidate_service_key for item in assertions}))


def _authoritative_input_digest(
    verified: _VerifiedInputs,
) -> str:
    return _digest(
        {
            "candidate_universe_sha256": verified.candidate_universe.universe_sha256,
            "candidate_freeze_sha256": verified.frozen_candidates.freeze_sha256,
            "entity_registry_sha256": verified.entity_registry.registry_sha256,
            "condensation_sha256s": tuple(
                item.condensation_sha256 for item in verified.condensations
            ),
            "event_grouping_result_sha256": verified.event_grouping.result_sha256,
            "onset_sha256s": tuple(item.onset_sha256 for item in verified.onsets),
            "precedence_sha256s": tuple(item.precedence_sha256 for item in verified.precedences),
            "role_compilation_result_sha256": verified.role_compilation.result_sha256,
            "role_proposal_sha256s": tuple(
                item.proposal_sha256 for item in verified.role_proposals
            ),
            "role_guard_result_sha256s": tuple(
                item.result_sha256 for item in verified.role_guard_results
            ),
            "role_predicate_sha256s": tuple(
                item.predicate_sha256 for item in verified.role_predicates
            ),
            "mechanism_library_sha256": verified.mechanism_library.library_sha256,
            "scope_interval": verified.scope_interval.model_dump(mode="json"),
            "relation_binding_sha256s": tuple(
                item.binding_sha256 for item in verified.relation_bindings
            ),
            "config_sha256": verified.config.config_sha256,
        }
    )


def _role_bundles(
    *,
    assertions: Sequence[RoleAssertion],
    groups_by_id: dict[str, EventGroup],
    role: Literal["source_support", "symptom_support"],
) -> tuple[_RoleBundle, ...]:
    grouped: dict[tuple[str, str], list[RoleAssertion]] = {}
    for assertion in assertions:
        if (
            assertion.lifecycle_status != "accepted"
            or assertion.role != role
            or assertion.strength <= 0.0
            or (role == "source_support" and assertion.score_direction != "promote")
        ):
            continue
        if assertion.event_group_id not in groups_by_id:
            raise ValueError("role assertion references an unknown event group")
        grouped.setdefault((assertion.candidate_service_key, assertion.event_group_id), []).append(
            assertion
        )
    bundles: list[_RoleBundle] = []
    for (service_key, event_group_id), items in sorted(grouped.items()):
        ordered = tuple(sorted(items, key=lambda item: item.assertion_id))
        bundles.append(
            _RoleBundle(
                service_key=service_key,
                event_group=groups_by_id[event_group_id],
                assertions=ordered,
            )
        )
    return tuple(bundles)


def _binding_observation_ids(onset: OnsetInterval, group: EventGroup) -> tuple[str, ...]:
    group_ids = set(group.observation_ids)
    matches = group_ids & set(onset.provenance_ids)
    return tuple(sorted(matches))


def _event_onset_bindings(
    *, bundles: Sequence[_RoleBundle], onsets: Sequence[OnsetInterval]
) -> dict[tuple[str, str], _OnsetBinding]:
    bindings: dict[tuple[str, str], _OnsetBinding] = {}
    for bundle in bundles:
        key = (bundle.service_key, bundle.event_group.event_group_id)
        if key in bindings:
            continue
        options = tuple(
            (onset, binding_ids)
            for onset in onsets
            if onset.service_key == bundle.service_key
            for binding_ids in (_binding_observation_ids(onset, bundle.event_group),)
            if binding_ids
        )
        option_ids = tuple(sorted(item.onset_id for item, _ in options))
        observation_ids = tuple(sorted({value for _, values in options for value in values}))
        if len(options) == 1:
            bindings[key] = _OnsetBinding(
                status="bound",
                onset=options[0][0],
                option_ids=option_ids,
                observation_ids=observation_ids,
            )
        else:
            bindings[key] = _OnsetBinding(
                status="missing" if not options else "ambiguous",
                onset=None,
                option_ids=option_ids,
                observation_ids=observation_ids,
            )
    return bindings


def _role_contributor_for_onset(
    *,
    bundle: _RoleBundle,
    onset_binding: _OnsetBinding,
) -> RoleAssertion | None:
    eligible = bundle.assertions
    if onset_binding.onset is not None:
        onset = onset_binding.onset
        eligible = tuple(
            item
            for item in eligible
            if item.interval.start_unix_ns <= onset.t_low_unix_ns
            and onset.t_high_unix_ns < item.interval.end_unix_ns
        )
    if not eligible:
        return None
    maximum = max(item.strength for item in eligible)
    return min(
        (item for item in eligible if item.strength == maximum),
        key=lambda item: item.assertion_id,
    )


def _index_condensation(condensation: RelationSCCCondensation) -> _CondensationIndex:
    if (
        len(condensation.components) > _MAX_CONDENSATION_COMPONENTS
        or len(condensation.condensed_edges) > _MAX_CONDENSATION_EDGES
        or len(condensation.selected_original_edges) > _MAX_SELECTED_ORIGINAL_EDGES
    ):
        raise PropagationScoringBudgetExceeded("condensation_structure_budget_exceeded")
    component_by_service = {
        service: component.component_id
        for component in condensation.components
        for service in component.member_service_keys
    }
    adjacency_lists: dict[str, list[RelationCondensedEdge]] = {}
    for edge in condensation.condensed_edges:
        adjacency_lists.setdefault(edge.source_component_id, []).append(edge)
    adjacency = {
        key: tuple(sorted(values, key=lambda item: item.condensed_edge_id))
        for key, values in adjacency_lists.items()
    }
    return _CondensationIndex(
        condensation=condensation,
        component_by_service=component_by_service,
        adjacency=adjacency,
    )


def _component_paths(
    index: _CondensationIndex,
    *,
    source_service_key: str,
    target_service_key: str,
    maximum_hops_to_audit: int,
    budget: _ComputationBudget,
) -> tuple[_GraphPath, ...]:
    source = index.component_by_service.get(source_service_key)
    target = index.component_by_service.get(target_service_key)
    if source is None or target is None:
        return ()
    if source == target:
        return (_GraphPath(component_ids=(source,), edges=()),)
    adjacency = index.adjacency
    found: list[_GraphPath] = []
    stack: list[tuple[str, tuple[str, ...], tuple[RelationCondensedEdge, ...]]] = []
    budget.charge_state()
    stack.append((source, (source,), ()))
    while stack:
        current, component_ids, edges = stack.pop()
        if len(edges) >= maximum_hops_to_audit:
            continue
        for edge in reversed(adjacency.get(current, [])):
            budget.charge_edge()
            next_component = edge.target_component_id
            if next_component in component_ids:  # pragma: no cover - condensation is a DAG
                continue
            next_components = (*component_ids, next_component)
            next_edges = (*edges, edge)
            if next_component == target:
                found.append(_GraphPath(component_ids=next_components, edges=next_edges))
                if len(found) > _MAX_ENUMERATED_PATHS:
                    raise PropagationScoringBudgetExceeded("path_count_budget_exceeded")
                continue
            budget.charge_state()
            stack.append((next_component, next_components, next_edges))
    if not found:
        queue: deque[tuple[str, tuple[str, ...], tuple[RelationCondensedEdge, ...]]] = deque()
        budget.charge_state()
        queue.append((source, (source,), ()))
        visited = {source}
        while queue:
            current, component_ids, edges = queue.popleft()
            for edge in adjacency.get(current, []):
                budget.charge_edge()
                next_component = edge.target_component_id
                if next_component in visited:
                    continue
                next_components = (*component_ids, next_component)
                next_edges = (*edges, edge)
                if next_component == target:
                    return (_GraphPath(component_ids=next_components, edges=next_edges),)
                visited.add(next_component)
                budget.charge_state()
                queue.append((next_component, next_components, next_edges))
    return tuple(
        sorted(
            found,
            key=lambda item: tuple(edge.condensed_edge_id for edge in item.edges),
        )
    )


def _hop_audit(
    *,
    hop_index: int,
    condensed_edge: RelationCondensedEdge,
    originals_by_id: dict[str, GraphSnapshotEdge],
) -> PropagationHopAudit:
    originals = tuple(originals_by_id[item] for item in condensed_edge.original_edge_ids)
    sufficient = tuple(
        item
        for item in originals
        if item.coverage.coverage_state == "sufficient"
        and item.coverage.coverage_fraction is not None
    )
    selected: GraphSnapshotEdge | None = None
    quality: float | None = None
    if sufficient:
        quality = max(cast(float, item.coverage.coverage_fraction) for item in sufficient)
        selected = min(
            (
                item
                for item in sufficient
                if cast(float, item.coverage.coverage_fraction) == quality
            ),
            key=lambda item: item.edge_id,
        )
    unsigned: dict[str, object] = {
        "schema_version": PROPAGATION_HOP_AUDIT_SCHEMA_VERSION,
        "hop_index": hop_index,
        "condensed_edge_id": condensed_edge.condensed_edge_id,
        "source_component_id": condensed_edge.source_component_id,
        "target_component_id": condensed_edge.target_component_id,
        "relation_type": condensed_edge.relation_type,
        "retained_original_edge_ids": condensed_edge.original_edge_ids,
        "sufficient_original_edge_ids": tuple(item.edge_id for item in sufficient),
        "selected_original_edge_id": None if selected is None else selected.edge_id,
        "hop_quality": quality,
        "coverage_status": "sufficient" if selected is not None else "insufficient",
        "provenance_ids": tuple(
            sorted(
                {
                    condensed_edge.condensed_edge_id,
                    *condensed_edge.original_provenance_ids,
                    *(value for item in originals for value in item.provenance_ids),
                    *(value for item in originals for value in item.coverage.provenance_ids),
                }
            )
        ),
    }
    identifier, digest = _stable_id("phop", unsigned)
    return PropagationHopAudit.model_validate(
        {**unsigned, "hop_audit_id": identifier, "hop_audit_sha256": digest}
    )


def _path_reason(
    *,
    graph_path: _GraphPath,
    hops: Sequence[PropagationHopAudit],
    precedence: TemporalPrecedence | None,
    source_binding: _OnsetBinding,
    target_binding: _OnsetBinding,
    source_contributor: RoleAssertion | None,
    target_contributor: RoleAssertion | None,
    maximum_hops: int,
) -> tuple[PathReason, ...]:
    reasons: list[PathReason] = []
    if not graph_path.edges:
        reasons.append("same_scc_no_internal_order")
    if len(graph_path.edges) > maximum_hops:
        reasons.append("hop_limit_exceeded")
    if any(item.coverage_status != "sufficient" for item in hops):
        reasons.append("insufficient_path_coverage")
    if source_binding.status == "missing":
        reasons.append("source_onset_binding_missing")
    elif source_binding.status == "ambiguous":
        reasons.append("source_onset_binding_ambiguous")
    elif source_contributor is None:
        reasons.append("source_role_interval_mismatch")
    if target_binding.status == "missing":
        reasons.append("target_onset_binding_missing")
    elif target_binding.status == "ambiguous":
        reasons.append("target_onset_binding_ambiguous")
    elif target_contributor is None:
        reasons.append("target_role_interval_mismatch")
    if precedence is None:
        reasons.append("missing_temporal_precedence")
    elif not (
        precedence.truth_value == "true"
        and precedence.grants_explicit_support
        and precedence.reason == "strict_precedence"
    ):
        reasons.append(cast(PathReason, f"temporal_{precedence.reason}"))
    return tuple(sorted(set(reasons))) if reasons else ("credited",)


def _path_draft(
    *,
    candidate_service_key: str,
    source_bundle: _RoleBundle,
    target_bundle: _RoleBundle,
    condensation: RelationSCCCondensation,
    graph_path: _GraphPath,
    source_binding: _OnsetBinding,
    target_binding: _OnsetBinding,
    precedence: TemporalPrecedence | None,
    maximum_hops: int,
) -> _PathDraft:
    source_contributor = _role_contributor_for_onset(
        bundle=source_bundle,
        onset_binding=source_binding,
    )
    target_contributor = _role_contributor_for_onset(
        bundle=target_bundle,
        onset_binding=target_binding,
    )
    originals_by_id = {item.edge_id: item for item in condensation.selected_original_edges}
    hops = tuple(
        _hop_audit(
            hop_index=index,
            condensed_edge=edge,
            originals_by_id=originals_by_id,
        )
        for index, edge in enumerate(graph_path.edges, start=1)
    )
    reasons = _path_reason(
        graph_path=graph_path,
        hops=hops,
        precedence=precedence,
        source_binding=source_binding,
        target_binding=target_binding,
        source_contributor=source_contributor,
        target_contributor=target_contributor,
        maximum_hops=maximum_hops,
    )
    insufficient_hops = tuple(
        item.hop_index for item in hops if item.coverage_status == "insufficient"
    )
    path_quality = (
        0.0
        if insufficient_hops
        else min((cast(float, item.hop_quality) for item in hops), default=0.0)
    )
    limiting_hop_indices = (
        insufficient_hops
        if insufficient_hops
        else tuple(item.hop_index for item in hops if item.hop_quality == path_quality)
    )
    credited = reasons == ("credited",)
    raw_contribution = (
        path_quality
        * min(source_contributor.strength, target_contributor.strength)
        / len(graph_path.edges)
        if credited and source_contributor is not None and target_contributor is not None
        else 0.0
    )
    source_onset = source_binding.onset
    target_onset = target_binding.onset
    source_onset_ids = source_binding.option_ids
    target_onset_ids = target_binding.option_ids
    precedence_ids = () if precedence is None else (precedence.precedence_id,)
    source_onset_provenance = () if source_onset is None else source_onset.provenance_ids
    target_onset_provenance = () if target_onset is None else target_onset.provenance_ids
    precedence_provenance = () if precedence is None else precedence.provenance_ids
    payload: dict[str, object] = {
        "schema_version": PROPAGATION_PATH_AUDIT_SCHEMA_VERSION,
        "candidate_service_key": candidate_service_key,
        "source_event_group_id": source_bundle.event_group.event_group_id,
        "target_service_key": target_bundle.service_key,
        "target_event_group_id": target_bundle.event_group.event_group_id,
        "relation_type": condensation.relation_type,
        "condensation_id": condensation.condensation_id,
        "component_ids": graph_path.component_ids,
        "condensed_edge_ids": tuple(item.condensed_edge_id for item in graph_path.edges),
        "hops": tuple(item.model_dump(mode="json") for item in hops),
        "path_length": len(graph_path.edges),
        "source_assertion_ids": tuple(item.assertion_id for item in source_bundle.assertions),
        "source_contributing_assertion_id": (
            None if source_contributor is None else source_contributor.assertion_id
        ),
        "target_assertion_ids": tuple(item.assertion_id for item in target_bundle.assertions),
        "target_contributing_assertion_id": (
            None if target_contributor is None else target_contributor.assertion_id
        ),
        "source_onset_id": None if source_onset is None else source_onset.onset_id,
        "target_onset_id": None if target_onset is None else target_onset.onset_id,
        "source_onset_option_ids": source_binding.option_ids,
        "target_onset_option_ids": target_binding.option_ids,
        "source_binding_observation_ids": source_binding.observation_ids,
        "target_binding_observation_ids": target_binding.observation_ids,
        "precedence_id": None if precedence is None else precedence.precedence_id,
        "source_support": 0.0 if source_contributor is None else source_contributor.strength,
        "target_symptom_strength": (
            0.0 if target_contributor is None else target_contributor.strength
        ),
        "time_ok": precedence is not None
        and precedence.truth_value == "true"
        and precedence.grants_explicit_support
        and precedence.reason == "strict_precedence",
        "path_quality": path_quality,
        "limiting_hop_indices": limiting_hop_indices,
        "raw_contribution": raw_contribution,
        "reason_codes": reasons,
        "provenance_ids": tuple(
            sorted(
                {
                    condensation.condensation_id,
                    *(() if source_contributor is None else (source_contributor.assertion_id,)),
                    *(() if target_contributor is None else (target_contributor.assertion_id,)),
                    *source_onset_ids,
                    *target_onset_ids,
                    *precedence_ids,
                    *source_bundle.event_group.provenance_ids,
                    *target_bundle.event_group.provenance_ids,
                    *source_onset_provenance,
                    *target_onset_provenance,
                    *precedence_provenance,
                    *(item.hop_audit_id for item in hops),
                }
            )
        ),
    }
    return _PathDraft(
        payload=payload,
        raw_contribution=raw_contribution,
        provisional_sha256=_digest(payload),
    )


def _build_path_audits(drafts: Sequence[_PathDraft]) -> tuple[PropagationPathAudit, ...]:
    selected_by_target: dict[tuple[str, str, str], _PathDraft] = {}
    for draft in drafts:
        if draft.raw_contribution <= 0.0:
            continue
        key = (
            cast(str, draft.payload["candidate_service_key"]),
            cast(str, draft.payload["target_service_key"]),
            cast(str, draft.payload["target_event_group_id"]),
        )
        current = selected_by_target.get(key)
        if current is None or (-draft.raw_contribution, draft.provisional_sha256) < (
            -current.raw_contribution,
            current.provisional_sha256,
        ):
            selected_by_target[key] = draft
    audits: list[PropagationPathAudit] = []
    selected_drafts = set(id(item) for item in selected_by_target.values())
    for draft in drafts:
        if draft.raw_contribution <= 0.0:
            selection_status: SelectionStatus = "rejected"
        elif id(draft) in selected_drafts:
            selection_status = "selected"
        else:
            selection_status = "same_target_not_selected"
        unsigned = {**draft.payload, "selection_status": selection_status}
        identifier, digest = _stable_id("ppath", unsigned)
        audits.append(
            PropagationPathAudit.model_validate(
                {**unsigned, "path_audit_id": identifier, "path_audit_sha256": digest}
            )
        )
    return tuple(sorted(audits, key=lambda item: item.path_audit_id))


def _build_target_scores(
    *,
    candidate_keys: Sequence[str],
    target_bundles: Sequence[_RoleBundle],
    path_audits: Sequence[PropagationPathAudit],
) -> tuple[PropagationTargetScore, ...]:
    target_events = {
        (bundle.service_key, bundle.event_group.event_group_id) for bundle in target_bundles
    }
    scores: list[PropagationTargetScore] = []
    for candidate_key in candidate_keys:
        for target_service_key, event_group_id in sorted(target_events):
            paths = tuple(
                item
                for item in path_audits
                if item.candidate_service_key == candidate_key
                and item.target_service_key == target_service_key
                and item.target_event_group_id == event_group_id
            )
            selected = tuple(item for item in paths if item.selection_status == "selected")
            selected_path = selected[0] if selected else None
            unsigned: dict[str, object] = {
                "schema_version": PROPAGATION_TARGET_SCORE_SCHEMA_VERSION,
                "candidate_service_key": candidate_key,
                "target_service_key": target_service_key,
                "target_event_group_id": event_group_id,
                "target_service_keys": (target_service_key,),
                "retained_path_audit_ids": tuple(sorted(item.path_audit_id for item in paths)),
                "selected_path_audit_id": (
                    None if selected_path is None else selected_path.path_audit_id
                ),
                "contribution": 0.0 if selected_path is None else selected_path.raw_contribution,
                "status": (
                    "zero_no_credited_path" if selected_path is None else "selected_maximum"
                ),
            }
            identifier, digest = _stable_id("ptgt", unsigned)
            scores.append(
                PropagationTargetScore.model_validate(
                    {**unsigned, "target_score_id": identifier, "target_score_sha256": digest}
                )
            )
    return tuple(
        sorted(
            scores,
            key=lambda item: (
                item.candidate_service_key,
                item.target_service_key,
                item.target_event_group_id,
            ),
        )
    )


def _build_candidate_scores(
    *,
    candidate_keys: Sequence[str],
    source_bundles: Sequence[_RoleBundle],
    target_bundles: Sequence[_RoleBundle],
    target_scores: Sequence[PropagationTargetScore],
    path_audits: Sequence[PropagationPathAudit],
) -> tuple[CandidatePropagationScore, ...]:
    source_services = {item.service_key for item in source_bundles}
    scores: list[CandidatePropagationScore] = []
    for candidate_key in candidate_keys:
        targets = tuple(
            item for item in target_scores if item.candidate_service_key == candidate_key
        )
        best_by_service: dict[str, PropagationTargetScore] = {}
        for target in targets:
            current = best_by_service.get(target.target_service_key)
            if current is None or (-target.contribution, target.target_event_group_id) < (
                -current.contribution,
                current.target_event_group_id,
            ):
                best_by_service[target.target_service_key] = target
        positive = tuple(
            item
            for item in sorted(
                best_by_service.values(),
                key=lambda item: (
                    -item.contribution,
                    item.target_service_key,
                    item.target_event_group_id,
                ),
            )[:2]
            if item.contribution > 0.0
        )
        contributions = tuple(item.contribution for item in positive)
        propagation_score = _top_two_f_gamma(contributions, gamma=PROPAGATION_GAMMA)
        if propagation_score > 0.0:
            reason: CandidateReason = "propagation_credit_computed"
        elif candidate_key not in source_services:
            reason = "missing_source_support"
        elif not target_bundles:
            reason = "missing_target_symptom_support"
        elif not any(item.candidate_service_key == candidate_key for item in path_audits):
            reason = "no_legal_typed_path"
        else:
            reason = "no_credited_path"
        unsigned: dict[str, object] = {
            "schema_version": CANDIDATE_PROPAGATION_SCORE_SCHEMA_VERSION,
            "candidate_service_key": candidate_key,
            "retained_target_score_ids": tuple(sorted(item.target_score_id for item in targets)),
            "top_target_service_keys": tuple(item.target_service_key for item in positive),
            "top_target_event_group_ids": tuple(item.target_event_group_id for item in positive),
            "top_target_contributions": contributions,
            "gamma": PROPAGATION_GAMMA,
            "propagation_score": propagation_score,
            "reason_code": reason,
        }
        identifier, digest = _stable_id("pcand", unsigned)
        scores.append(
            CandidatePropagationScore.model_validate(
                {**unsigned, "candidate_score_id": identifier, "candidate_score_sha256": digest}
            )
        )
    return tuple(sorted(scores, key=lambda item: item.candidate_service_key))


def score_propagation_paths(
    *,
    frozen_candidates: FrozenCandidateUniverse,
    condensations: Sequence[RelationSCCCondensation],
    event_grouping: EventGroupingResult,
    onsets: Sequence[OnsetInterval],
    precedences: Sequence[TemporalPrecedence],
    role_compilation: RoleCompilationResult | None = None,
    role_proposals: Sequence[RoleBindingProposal] | None = None,
    role_guard_results: Sequence[HypothesisGuardResult] | None = None,
    role_predicates: Sequence[PredicateResult] | None = None,
    candidate_universe: CandidateUniverse | None = None,
    entity_registry: TemporalEntityRegistry | None = None,
    mechanism_library: MechanismLibrary | None = None,
    scope_interval: TimeInterval | None = None,
    role_assertions: Sequence[RoleAssertion] | None = None,
    query_sessions_by_guard_result_id: Mapping[str, CaseQuerySessionResult] | None = None,
    query_session_coordinator: CaseQuerySessionCoordinator | None = None,
    semantic_context: SemanticContextBundle | None = None,
    card_expansion_policy: CardExpansionPolicy | None = None,
    config: PropagationScoringConfig | None = None,
) -> PropagationScoringResult:
    """Score only explicit, covered, typed condensation paths with bounded fan-out."""

    if (
        role_assertions is not None
        or role_compilation is None
        or role_proposals is None
        or role_guard_results is None
        or role_predicates is None
        or candidate_universe is None
        or entity_registry is None
        or mechanism_library is None
        or scope_interval is None
    ):
        raise ValueError("propagation scoring input verification failed") from None
    verified: _VerifiedInputs | None = None
    try:
        verified = _verified_inputs(
            frozen_candidates=frozen_candidates,
            condensations=condensations,
            event_grouping=event_grouping,
            onsets=onsets,
            precedences=precedences,
            role_compilation=role_compilation,
            role_proposals=role_proposals,
            role_guard_results=role_guard_results,
            role_predicates=role_predicates,
            candidate_universe=candidate_universe,
            entity_registry=entity_registry,
            mechanism_library=mechanism_library,
            scope_interval=scope_interval,
            query_sessions_by_guard_result_id=query_sessions_by_guard_result_id,
            query_session_coordinator=query_session_coordinator,
            semantic_context=semantic_context,
            card_expansion_policy=card_expansion_policy,
            config=config,
        )
    except PropagationScoringBudgetExceeded:
        raise
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    if verified is None:
        raise ValueError("propagation scoring input verification failed") from None
    return _score_propagation_from_verified(verified)


def _score_propagation_from_verified(verified: _VerifiedInputs) -> PropagationScoringResult:
    """Unchanged numeric core; only call-local source boundaries may consume it."""
    frozen = verified.frozen_candidates
    condensation_values = verified.condensations
    grouping = verified.event_grouping
    onset_values = verified.onsets
    precedence_values = verified.precedences
    assertion_values = verified.role_compilation.assertions
    relation_bindings = verified.relation_bindings
    config_value = verified.config
    authoritative_input_sha256 = _authoritative_input_digest(verified)
    groups_by_id = {item.event_group_id: item for item in grouping.event_groups}
    bound_assertion_ids = {item.assertion_id for item in relation_bindings}
    source_bundles = _role_bundles(
        assertions=tuple(
            item for item in assertion_values if item.assertion_id in bound_assertion_ids
        ),
        groups_by_id=groups_by_id,
        role="source_support",
    )
    target_bundles = _role_bundles(
        assertions=tuple(
            item for item in assertion_values if item.assertion_id in bound_assertion_ids
        ),
        groups_by_id=groups_by_id,
        role="symptom_support",
    )
    onset_bindings = _event_onset_bindings(
        bundles=(*source_bundles, *target_bundles),
        onsets=onset_values,
    )
    budget = _ComputationBudget(
        max_bundle_pairs=config_value.max_bundle_pairs,
        max_expanded_states=config_value.max_expanded_states,
        max_traversed_edges=config_value.max_traversed_edges,
    )
    condensation_indexes = tuple(_index_condensation(item) for item in condensation_values)
    precedence_by_pair: dict[tuple[str, str], list[TemporalPrecedence]] = {}
    for precedence_value in precedence_values:
        if precedence_value.source_onset_id is None or precedence_value.target_onset_id is None:
            continue
        precedence_by_pair.setdefault(
            (
                precedence_value.source_onset_id,
                precedence_value.target_onset_id,
            ),
            [],
        ).append(precedence_value)
    drafts: list[_PathDraft] = []
    assertions_by_id = {item.assertion_id: item for item in assertion_values}
    for condensation_index in condensation_indexes:
        condensation = condensation_index.condensation
        relation_source_ids = {
            item.assertion_id
            for item in relation_bindings
            if item.relation_type == condensation.relation_type and item.role == "source_support"
        }
        relation_target_ids = {
            item.assertion_id
            for item in relation_bindings
            if item.relation_type == condensation.relation_type and item.role == "symptom_support"
        }
        relation_source_bundles = _role_bundles(
            assertions=tuple(assertions_by_id[item] for item in sorted(relation_source_ids)),
            groups_by_id=groups_by_id,
            role="source_support",
        )
        relation_target_bundles = _role_bundles(
            assertions=tuple(assertions_by_id[item] for item in sorted(relation_target_ids)),
            groups_by_id=groups_by_id,
            role="symptom_support",
        )
        for source_bundle in relation_source_bundles:
            source_binding = onset_bindings[
                (source_bundle.service_key, source_bundle.event_group.event_group_id)
            ]
            for target_bundle in relation_target_bundles:
                budget.charge_bundle_pair()
                target_binding = onset_bindings[
                    (target_bundle.service_key, target_bundle.event_group.event_group_id)
                ]
                graph_paths = _component_paths(
                    condensation_index,
                    source_service_key=source_bundle.service_key,
                    target_service_key=target_bundle.service_key,
                    maximum_hops_to_audit=config_value.propagation_max_hops,
                    budget=budget,
                )
                for graph_path in graph_paths:
                    path_precedence: TemporalPrecedence | None = None
                    if source_binding.onset is not None and target_binding.onset is not None:
                        matches = precedence_by_pair.get(
                            (source_binding.onset.onset_id, target_binding.onset.onset_id), []
                        )
                        path_precedence = matches[0] if matches else None
                    drafts.append(
                        _path_draft(
                            candidate_service_key=source_bundle.service_key,
                            source_bundle=source_bundle,
                            target_bundle=target_bundle,
                            condensation=condensation,
                            graph_path=graph_path,
                            source_binding=source_binding,
                            target_binding=target_binding,
                            precedence=path_precedence,
                            maximum_hops=config_value.propagation_max_hops,
                        )
                    )
                    if len(drafts) > _MAX_ENUMERATED_PATHS:
                        raise PropagationScoringBudgetExceeded("path_count_budget_exceeded")
    path_audits = _build_path_audits(drafts)
    target_scores = _build_target_scores(
        candidate_keys=frozen.candidate_service_keys,
        target_bundles=target_bundles,
        path_audits=path_audits,
    )
    candidate_scores = _build_candidate_scores(
        candidate_keys=frozen.candidate_service_keys,
        source_bundles=source_bundles,
        target_bundles=target_bundles,
        target_scores=target_scores,
        path_audits=path_audits,
    )
    unsigned: dict[str, object] = {
        "schema_version": PROPAGATION_SCORING_RESULT_SCHEMA_VERSION,
        "opaque_case_id": frozen.opaque_case_id,
        "policy_version": PROPAGATION_POLICY_VERSION,
        "config": config_value.model_dump(mode="json"),
        "authoritative_input_sha256": authoritative_input_sha256,
        "candidate_freeze_sha256": frozen.freeze_sha256,
        "candidate_service_keys": frozen.candidate_service_keys,
        "event_grouping_result_sha256": grouping.result_sha256,
        "role_compilation_id": verified.role_compilation.role_compilation_id,
        "role_compilation_result_sha256": verified.role_compilation.result_sha256,
        "relation_policy_sha256": PROPAGATION_RELATION_POLICY_SHA256,
        "condensation_sha256s": tuple(
            sorted(item.condensation_sha256 for item in condensation_values)
        ),
        "onset_sha256s": tuple(sorted(item.onset_sha256 for item in onset_values)),
        "precedence_sha256s": tuple(sorted(item.precedence_sha256 for item in precedence_values)),
        "role_assertion_sha256s": tuple(sorted(item.assertion_sha256 for item in assertion_values)),
        "relation_bindings": tuple(item.model_dump(mode="json") for item in relation_bindings),
        "path_audits": tuple(item.model_dump(mode="json") for item in path_audits),
        "target_scores": tuple(item.model_dump(mode="json") for item in target_scores),
        "candidate_scores": tuple(item.model_dump(mode="json") for item in candidate_scores),
    }
    identifier, digest = _stable_id("pres", unsigned)
    result = PropagationScoringResult.model_validate(
        {**unsigned, "scoring_result_id": identifier, "result_sha256": digest}
    )
    if not PropagationScoringResult.verify(result):  # pragma: no cover - construction invariant
        raise AssertionError("propagation scoring result failed verification")
    return result


__all__ = [
    "ALLOWED_PROPAGATION_MAX_HOPS",
    "CANDIDATE_PROPAGATION_SCORE_SCHEMA_VERSION",
    "COVERAGE_POLICY_VERSION",
    "PROPAGATION_GAMMA",
    "PROPAGATION_HOP_AUDIT_SCHEMA_VERSION",
    "PROPAGATION_PATH_AUDIT_SCHEMA_VERSION",
    "PROPAGATION_POLICY_VERSION",
    "PROPAGATION_RELATION_BINDING_SCHEMA_VERSION",
    "PROPAGATION_RELATION_POLICY_SHA256",
    "PROPAGATION_RELATION_POLICY_VERSION",
    "PROPAGATION_SCORING_CONFIG_SCHEMA_VERSION",
    "PROPAGATION_SCORING_RESULT_SCHEMA_VERSION",
    "PROPAGATION_TARGET_SCORE_SCHEMA_VERSION",
    "CandidatePropagationScore",
    "PropagationHopAudit",
    "PropagationPathAudit",
    "PropagationRelationBinding",
    "PropagationScoringBudgetExceeded",
    "PropagationScoringConfig",
    "PropagationScoringResult",
    "PropagationTargetScore",
    "build_propagation_scoring_config",
    "score_propagation_paths",
]
