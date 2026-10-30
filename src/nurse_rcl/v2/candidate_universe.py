"""Complete, label-free construction of the V2 service candidate universe C."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.case_state import CaseStateEntry, build_case_state_transition
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.observation_schemas import (
    CandidateEntry,
    CandidateEvidenceState,
    CandidateUniverse,
    Observation,
    TimeInterval,
    build_candidate_universe,
)
from nurse_rcl.v2.runtime_schemas import (
    SealedAuditPackage,
    build_sealed_audit_package,
)
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

CANDIDATE_CONSTRUCTION_CONTRACT_VERSION: Literal["nurse.v2.candidate-construction-input/1"] = (
    "nurse.v2.candidate-construction-input/1"
)
EMPTY_CANDIDATE_FAILURE_SCHEMA_VERSION: Literal["nurse.v2.empty-candidate-universe-failure/1"] = (
    "nurse.v2.empty-candidate-universe-failure/1"
)

_STATE_PRIORITY: dict[CandidateEvidenceState, int] = {
    "zero_support": 0,
    "unknown": 1,
    "degraded": 2,
    "supported": 3,
}


@dataclass
class _CandidateAccumulator:
    binding_ids: set[str] = field(default_factory=set)
    observation_ids: set[str] = field(default_factory=set)
    event_group_ids: set[str] = field(default_factory=set)
    states: set[CandidateEvidenceState] = field(default_factory=lambda: {"zero_support"})


class EmptyCandidateUniverseFailure(StrictFrozenModel):
    """Auditable terminal result when no valid in-scope service can form C."""

    schema_version: Literal["nurse.v2.empty-candidate-universe-failure/1"] = (
        EMPTY_CANDIDATE_FAILURE_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    failure_code: Literal["empty_candidate_universe"] = "empty_candidate_universe"
    scope_interval: TimeInterval
    observation_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_record_count: int = Field(ge=0)
    timeline_binding_count: int = Field(ge=0)
    observation_resolution_count: int = Field(ge=0)
    unbound_observation_count: int = Field(ge=0)
    conflicted_binding_count: int = Field(ge=0)
    candidate_count: Literal[0] = 0
    prediction_service_keys: tuple[str, ...] = ()
    semantic_request_permitted: Literal[False] = False
    semantic_call_count: Literal[0] = 0
    empty_semantic_calls_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    diagnostic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_state_entry: CaseStateEntry
    sealed_audit_package: SealedAuditPackage
    outcome_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_terminal_empty_scope(self) -> Self:
        state = self.case_state_entry
        package = self.sealed_audit_package
        if self.prediction_service_keys:
            raise ValueError("empty candidate failure must retain an empty prediction")
        if not state.verify() or state.checkpoint.state != "FAILURE_SEALED":
            raise ValueError("empty candidate failure requires a verified terminal state")
        if state.from_state != "OBSERVATIONS_READY":
            raise ValueError("empty candidate failure must follow OBSERVATIONS_READY")
        if state.failure_code != self.failure_code:
            raise ValueError("empty candidate failure code does not match case state")
        if state.checkpoint.opaque_case_id != self.opaque_case_id:
            raise ValueError("empty candidate failure belongs to a different case state")
        if state.checkpoint.logical_request_ids:
            raise ValueError("empty candidate failure cannot contain semantic requests")
        if not {"entities", "observations"} <= set(state.checkpoint.upstream_hashes):
            raise ValueError("empty candidate failure is missing frozen candidate input lineage")
        if state.checkpoint.produced_artifact_hashes != {
            "candidate_scope_failure": self.diagnostic_sha256
        }:
            raise ValueError("terminal state does not bind the empty candidate diagnostic")
        if self.empty_semantic_calls_sha256 != _empty_semantic_calls_sha256(
            opaque_case_id=self.opaque_case_id
        ):
            raise ValueError("empty calls hash is not the canonical empty semantic call set")
        if not package.verify() or package.terminal_state != "FAILURE_SEALED":
            raise ValueError("empty candidate failure requires a verified audit package")
        if package.opaque_case_id != self.opaque_case_id:
            raise ValueError("empty candidate failure belongs to a different audit package")
        if package.experiment_version != state.checkpoint.experiment_version:
            raise ValueError("empty candidate failure experiment versions differ")
        if package.prediction_service_keys or package.run_mode != "failure":
            raise ValueError("empty candidate audit package must retain an empty prediction")
        if package.artifact_hashes.get("candidate_universe") != self.diagnostic_sha256:
            raise ValueError("audit package does not bind the empty candidate diagnostic")
        if package.artifact_hashes.get("calls") != self.empty_semantic_calls_sha256:
            raise ValueError("audit package does not bind the empty semantic call set")
        conflicting_upstream = tuple(
            sorted(
                key
                for key, value in state.checkpoint.upstream_hashes.items()
                if key in package.artifact_hashes and package.artifact_hashes[key] != value
            )
        )
        if conflicting_upstream:
            raise ValueError(
                "audit package conflicts with terminal upstream artifacts: "
                + ", ".join(conflicting_upstream)
            )
        return self

    def verify(self) -> bool:
        try:
            validated = EmptyCandidateUniverseFailure.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
        except (AttributeError, TypeError, ValueError):
            return False
        diagnostic = _empty_candidate_diagnostic_payload(
            opaque_case_id=validated.opaque_case_id,
            scope_interval=validated.scope_interval,
            observation_contract_sha256=validated.observation_contract_sha256,
            metadata_record_count=validated.metadata_record_count,
            timeline_binding_count=validated.timeline_binding_count,
            observation_resolution_count=validated.observation_resolution_count,
            unbound_observation_count=validated.unbound_observation_count,
            conflicted_binding_count=validated.conflicted_binding_count,
            empty_semantic_calls_sha256=validated.empty_semantic_calls_sha256,
        )
        unsigned = validated.model_dump(mode="json", exclude={"outcome_sha256"})
        return (
            validated == self
            and validated.diagnostic_sha256 == sha256_hex(canonical_json_bytes(diagnostic))
            and validated.outcome_sha256 == sha256_hex(canonical_json_bytes(unsigned))
        )


def _intervals_intersect(left: TimeInterval, right: TimeInterval) -> bool:
    return left.start_unix_ns < right.end_unix_ns and left.end_unix_ns > right.start_unix_ns


def _observation_state(observation: Observation) -> CandidateEvidenceState:
    if observation.evidence_state == "abnormal":
        if (
            observation.formal_scoring_eligible
            and observation.anomaly_strength is not None
            and observation.anomaly_strength > 0.0
        ):
            return "supported"
        return "degraded"
    if observation.evidence_state == "degraded":
        return "degraded"
    if observation.evidence_state in {"unknown", "no_record"}:
        return "unknown"
    return "zero_support"


def _validate_frozen_inputs(
    *,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    scope_interval: TimeInterval,
) -> None:
    try:
        validated_scope = TimeInterval.model_validate(
            scope_interval.model_dump(mode="json", warnings="error")
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("candidate scope is invalid") from error
    if validated_scope != scope_interval:
        raise ValueError("candidate scope is invalid")
    if not entity_registry.verify():
        raise ValueError("entity registry failed frozen verification")
    if not event_grouping.verify():
        raise ValueError("event grouping failed frozen verification")
    if entity_registry.opaque_case_id != event_grouping.opaque_case_id:
        raise ValueError("entity registry and event grouping belong to different cases")
    if entity_registry.decision_time_unix_ns != event_grouping.decision_time_unix_ns:
        raise ValueError("entity registry and event grouping decision times differ")
    if scope_interval.end_unix_ns > entity_registry.decision_time_unix_ns:
        raise ValueError("candidate scope extends beyond decision time")
    registry_observations = tuple(
        resolution.observation for resolution in entity_registry.resolutions
    )
    if registry_observations != event_grouping.observations:
        raise ValueError("entity registry and event grouping observation membership differs")


def _observation_contract_sha256(
    *,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    scope_interval: TimeInterval,
) -> str:
    payload = {
        "schema_version": CANDIDATE_CONSTRUCTION_CONTRACT_VERSION,
        "opaque_case_id": entity_registry.opaque_case_id,
        "decision_time_unix_ns": entity_registry.decision_time_unix_ns,
        "scope_interval": scope_interval.model_dump(mode="json"),
        "entity_registry_sha256": entity_registry.registry_sha256,
        "event_grouping_result_sha256": event_grouping.result_sha256,
    }
    return sha256_hex(canonical_json_bytes(payload))


def _collect_candidate_accumulators(
    *,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    scope_interval: TimeInterval,
) -> dict[str, _CandidateAccumulator]:
    accumulators: dict[str, _CandidateAccumulator] = {}

    def candidate(service_entity_key: str) -> _CandidateAccumulator:
        return accumulators.setdefault(service_entity_key, _CandidateAccumulator())

    for record in entity_registry.metadata_records:
        if (
            record.relation == "explicit_service_binding"
            and record.service_entity_key is not None
            and _intervals_intersect(record.valid_interval, scope_interval)
        ):
            candidate(record.service_entity_key)

    for binding in entity_registry.timeline_bindings:
        if not _intervals_intersect(binding.valid_interval, scope_interval):
            continue
        if binding.status == "bound" and binding.service_entity_key is not None:
            candidate(binding.service_entity_key).binding_ids.add(binding.binding_id)
        elif binding.status == "conflicted":
            for service_entity_key in binding.conflict_service_keys:
                item = candidate(service_entity_key)
                item.binding_ids.add(binding.binding_id)
                item.states.add("unknown")

    event_groups_by_observation: dict[str, set[str]] = {}
    for group in event_grouping.event_groups:
        for observation_id in group.observation_ids:
            event_groups_by_observation.setdefault(observation_id, set()).add(group.event_group_id)

    for resolution in entity_registry.resolutions:
        binding = resolution.binding
        observation = resolution.observation
        if not _intervals_intersect(binding.valid_interval, scope_interval):
            continue
        if binding.status == "conflicted":
            for service_entity_key in binding.conflict_service_keys:
                item = candidate(service_entity_key)
                item.binding_ids.add(binding.binding_id)
                item.states.add("unknown")
            continue
        if binding.status != "bound" or binding.service_entity_key is None:
            continue
        item = candidate(binding.service_entity_key)
        item.binding_ids.add(binding.binding_id)
        if not _intervals_intersect(observation.interval, scope_interval):
            continue
        state = _observation_state(observation)
        item.states.add(state)
        if state != "zero_support":
            item.observation_ids.add(observation.observation_id)
            item.event_group_ids.update(
                event_groups_by_observation.get(observation.observation_id, set())
            )
    return accumulators


def _empty_semantic_calls_sha256(*, opaque_case_id: str) -> str:
    return sha256_hex(
        canonical_json_bytes(
            {
                "opaque_case_id": opaque_case_id,
                "reason_code": "empty_candidate_universe",
                "logical_request_ids": [],
                "semantic_calls": [],
            }
        )
    )


def _empty_candidate_diagnostic_payload(
    *,
    opaque_case_id: str,
    scope_interval: TimeInterval,
    observation_contract_sha256: str,
    metadata_record_count: int,
    timeline_binding_count: int,
    observation_resolution_count: int,
    unbound_observation_count: int,
    conflicted_binding_count: int,
    empty_semantic_calls_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": EMPTY_CANDIDATE_FAILURE_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "failure_code": "empty_candidate_universe",
        "scope_interval": scope_interval.model_dump(mode="json"),
        "observation_contract_sha256": observation_contract_sha256,
        "metadata_record_count": metadata_record_count,
        "timeline_binding_count": timeline_binding_count,
        "observation_resolution_count": observation_resolution_count,
        "unbound_observation_count": unbound_observation_count,
        "conflicted_binding_count": conflicted_binding_count,
        "candidate_count": 0,
        "prediction_service_keys": [],
        "semantic_request_permitted": False,
        "semantic_call_count": 0,
        "empty_semantic_calls_sha256": empty_semantic_calls_sha256,
    }


def build_complete_candidate_universe(
    *,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    scope_interval: TimeInterval,
) -> CandidateUniverse:
    """Build complete C without anomaly, topology, alarm, or context filtering."""

    _validate_frozen_inputs(
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        scope_interval=scope_interval,
    )
    accumulators = _collect_candidate_accumulators(
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        scope_interval=scope_interval,
    )

    entries = tuple(
        CandidateEntry(
            service_entity_key=service_entity_key,
            binding_ids=tuple(sorted(item.binding_ids)),
            observation_ids=tuple(sorted(item.observation_ids)),
            event_group_ids=tuple(sorted(item.event_group_ids)),
            evidence_state=max(item.states, key=_STATE_PRIORITY.__getitem__),
        )
        for service_entity_key, item in sorted(accumulators.items())
    )
    return build_candidate_universe(
        opaque_case_id=entity_registry.opaque_case_id,
        observation_contract_sha256=_observation_contract_sha256(
            entity_registry=entity_registry,
            event_grouping=event_grouping,
            scope_interval=scope_interval,
        ),
        candidates=entries,
    )


def seal_empty_candidate_universe_failure(
    *,
    previous_case_state: CaseStateEntry,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    scope_interval: TimeInterval,
    audit_artifact_hashes: Mapping[str, str],
    created_at_utc: str,
    sealed_at_utc: str,
) -> EmptyCandidateUniverseFailure:
    """Seal an empty-C data-scope failure before any semantic request is possible."""

    _validate_frozen_inputs(
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        scope_interval=scope_interval,
    )
    accumulators = _collect_candidate_accumulators(
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        scope_interval=scope_interval,
    )
    if accumulators:
        raise ValueError("candidate universe is not empty")
    if (
        not previous_case_state.verify()
        or previous_case_state.checkpoint.state != "OBSERVATIONS_READY"
    ):
        raise ValueError("empty candidate failure requires OBSERVATIONS_READY state")
    if previous_case_state.checkpoint.logical_request_ids:
        raise ValueError("empty candidate failure cannot erase a preexisting semantic request")
    if previous_case_state.checkpoint.opaque_case_id != entity_registry.opaque_case_id:
        raise ValueError("empty candidate inputs belong to a different case state")
    if (
        previous_case_state.checkpoint.upstream_hashes.get("entities")
        != entity_registry.registry_sha256
    ):
        raise ValueError("entity registry conflicts with case-state upstream binding")
    if (
        previous_case_state.checkpoint.produced_artifact_hashes.get("observations")
        != event_grouping.result_sha256
    ):
        raise ValueError("OBSERVATIONS_READY state does not bind the frozen observations")
    if audit_artifact_hashes.get("entities") != entity_registry.registry_sha256:
        raise ValueError("entities artifact hash does not bind the frozen entity registry")
    if audit_artifact_hashes.get("observations") != event_grouping.result_sha256:
        raise ValueError("observations artifact hash does not bind the frozen observations")
    terminal_upstream_hashes = dict(previous_case_state.checkpoint.upstream_hashes)
    conflicting_state_lineage = tuple(
        sorted(
            key
            for key, value in previous_case_state.checkpoint.produced_artifact_hashes.items()
            if key in terminal_upstream_hashes and terminal_upstream_hashes[key] != value
        )
    )
    if conflicting_state_lineage:
        raise ValueError(
            "case-state upstream and produced artifact hashes conflict: "
            + ", ".join(conflicting_state_lineage)
        )
    terminal_upstream_hashes.update(previous_case_state.checkpoint.produced_artifact_hashes)
    conflicting_upstream = tuple(
        sorted(
            key
            for key, value in terminal_upstream_hashes.items()
            if key in audit_artifact_hashes and audit_artifact_hashes[key] != value
        )
    )
    if conflicting_upstream:
        raise ValueError(
            "audit artifact hashes conflict with case-state outputs: "
            + ", ".join(conflicting_upstream)
        )

    observation_contract_sha256 = _observation_contract_sha256(
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        scope_interval=scope_interval,
    )
    empty_calls_sha256 = _empty_semantic_calls_sha256(opaque_case_id=entity_registry.opaque_case_id)
    diagnostic = _empty_candidate_diagnostic_payload(
        opaque_case_id=entity_registry.opaque_case_id,
        scope_interval=scope_interval,
        observation_contract_sha256=observation_contract_sha256,
        metadata_record_count=len(entity_registry.metadata_records),
        timeline_binding_count=len(entity_registry.timeline_bindings),
        observation_resolution_count=len(entity_registry.resolutions),
        unbound_observation_count=sum(
            resolution.binding.status == "unbound" for resolution in entity_registry.resolutions
        ),
        conflicted_binding_count=sum(
            binding.status == "conflicted" for binding in entity_registry.timeline_bindings
        ),
        empty_semantic_calls_sha256=empty_calls_sha256,
    )
    diagnostic_sha256 = sha256_hex(canonical_json_bytes(diagnostic))
    terminal_state = build_case_state_transition(
        previous=previous_case_state,
        to_state="FAILURE_SEALED",
        upstream_hashes=terminal_upstream_hashes,
        produced_artifact_hashes={"candidate_scope_failure": diagnostic_sha256},
        logical_request_ids=(),
        created_at_utc=created_at_utc,
        failure_code="empty_candidate_universe",
    )
    artifact_hashes = dict(audit_artifact_hashes)
    artifact_hashes["candidate_universe"] = diagnostic_sha256
    artifact_hashes["calls"] = empty_calls_sha256
    audit_package = build_sealed_audit_package(
        opaque_case_id=entity_registry.opaque_case_id,
        experiment_version=previous_case_state.checkpoint.experiment_version,
        terminal_state="FAILURE_SEALED",
        artifact_hashes=artifact_hashes,
        accepted_provenance_ids=(),
        rejected_provenance_ids=(),
        conflicted_provenance_ids=(),
        unknown_provenance_ids=("candidate_scope:empty_candidate_universe",),
        prediction_service_keys=(),
        run_mode="failure",
        sealed_at_utc=sealed_at_utc,
    )
    unsigned: dict[str, object] = {
        **diagnostic,
        "diagnostic_sha256": diagnostic_sha256,
        "case_state_entry": terminal_state.model_dump(mode="json"),
        "sealed_audit_package": audit_package.model_dump(mode="json"),
    }
    outcome = EmptyCandidateUniverseFailure.model_validate(
        {
            **unsigned,
            "outcome_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not outcome.verify():  # pragma: no cover - construction invariant
        raise AssertionError("empty candidate failure verification failed")
    return outcome
