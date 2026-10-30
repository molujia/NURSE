"""Deterministic, read-only execution of bounded verification requests."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Literal, TypeAlias, TypeVar, cast

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.candidate_freeze import (
    FrozenCandidateUniverse,
    require_frozen_candidate_universe,
)
from nurse_rcl.v2.candidate_universe import build_complete_candidate_universe
from nurse_rcl.v2.context_expansion import CardExpansionBatchResult
from nurse_rcl.v2.diagnostic_schemas import (
    VerificationOperator,
    VerificationRequest,
    VerificationResult,
    build_verification_result,
)
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.observation_schemas import CandidateUniverse, Observation, TimeInterval
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

VERIFICATION_OPERATOR_PERMISSION_SCHEMA_VERSION: Literal[
    "nurse.v2.verification-operator-permission/1"
] = "nurse.v2.verification-operator-permission/1"
VERIFICATION_EXECUTION_POLICY_SCHEMA_VERSION: Literal[
    "nurse.v2.verification-execution-policy/1"
] = "nurse.v2.verification-execution-policy/1"
VERIFICATION_EXECUTION_SCOPE_SCHEMA_VERSION: Literal["nurse.v2.verification-execution-scope/1"] = (
    "nurse.v2.verification-execution-scope/1"
)
VERIFICATION_FIELD_VALUE_SCHEMA_VERSION: Literal["nurse.v2.verification-field-value/1"] = (
    "nurse.v2.verification-field-value/1"
)
DERIVED_VERIFICATION_OBSERVATION_SCHEMA_VERSION: Literal[
    "nurse.v2.derived-verification-observation/1"
] = "nurse.v2.derived-verification-observation/1"
VERIFICATION_REQUEST_COUNT_SCHEMA_VERSION: Literal["nurse.v2.verification-request-count/1"] = (
    "nurse.v2.verification-request-count/1"
)
VERIFICATION_EXECUTION_DECISION_SCHEMA_VERSION: Literal[
    "nurse.v2.verification-execution-decision/1"
] = "nurse.v2.verification-execution-decision/1"
VERIFICATION_BATCH_RESULT_SCHEMA_VERSION: Literal["nurse.v2.verification-batch-result/1"] = (
    "nurse.v2.verification-batch-result/1"
)
VERIFICATION_EXECUTION_POLICY_VERSION: Literal["nurse-v2-verification-execution/1"] = (
    "nurse-v2-verification-execution/1"
)
INITIAL_VERIFICATION_QUERY_LIMIT: Literal[8] = 8

ALLOWED_VERIFICATION_OPERATORS: tuple[VerificationOperator, ...] = (
    "compare_window_summary",
    "inspect_counter_reset",
    "measure_onset_interval",
    "measure_trace_edge_change",
    "resolve_entity_binding",
    "summarize_log_pattern",
)

_OPERATOR_FIELDS: Mapping[VerificationOperator, tuple[str, ...]] = MappingProxyType(
    {
        "compare_window_summary": (
            "anomaly_strength",
            "baseline_coverage_fraction",
            "baseline_median",
            "baseline_sample_count",
            "coverage_quality",
            "evidence_state",
            "fault_coverage_fraction",
            "fault_median",
            "fault_sample_count",
            "median_delta",
            "sample_count",
        ),
        "inspect_counter_reset": ("reset_detected",),
        "measure_onset_interval": (
            "onset_delay_unix_ns",
            "onset_end_unix_ns",
            "onset_start_unix_ns",
        ),
        "measure_trace_edge_change": (
            "baseline_edge_count",
            "edge_count_delta",
            "fault_edge_count",
        ),
        "resolve_entity_binding": ("binding_status", "service_entity_key"),
        "summarize_log_pattern": (
            "baseline_event_count",
            "event_count_delta",
            "fault_event_count",
        ),
    }
)

VerificationReason: TypeAlias = Literal[
    "completed",
    "invalid_request",
    "permission_scope_mismatch",
    "permission_case_mismatch",
    "permission_candidate_mismatch",
    "permission_frozen_universe_mismatch",
    "permission_time_mismatch",
    "permission_observation_denied",
    "permission_relation_denied",
    "permission_operator_denied",
    "permission_field_denied",
    "permission_signal_family_mismatch",
    "permission_observation_candidate_mismatch",
    "query_budget_exhausted",
    "typed_input_unavailable",
]

_Model = TypeVar("_Model", bound=BaseModel)
_MAX_INPUT_DEPTH = 64
_MAX_INPUT_NODES = 20_000


def _digest(unsigned: object) -> str:
    return sha256_hex(canonical_json_bytes(unsigned))


def _stable_id(prefix: str, unsigned: object) -> tuple[str, str]:
    digest = _digest(unsigned)
    return f"{prefix}_{digest[:24]}", digest


def _reject_hidden_cyclic_or_deep(value: object) -> None:
    """Bounded traversal before serialization can omit hidden model state."""

    active: set[int] = set()
    nodes = 0
    stack: list[tuple[object, int, bool]] = [(value, 0, False)]
    while stack:
        item, depth, exiting = stack.pop()
        identity = id(item)
        if exiting:
            active.remove(identity)
            continue
        nodes += 1
        if nodes > _MAX_INPUT_NODES:
            raise ValueError("verification input exceeds structural limits")
        children: tuple[object, ...] | list[object]
        if isinstance(item, BaseModel):
            if set(vars(item)) != set(type(item).model_fields):
                raise ValueError("unexpected fields in verification input")
            children = tuple(vars(item).values())
        elif type(item) is dict:
            if nodes + len(cast(dict[object, object], item)) > _MAX_INPUT_NODES:
                raise ValueError("verification input exceeds structural limits")
            children = list(cast(dict[object, object], item).values())
        elif type(item) is tuple:
            children = cast(tuple[object, ...], item)
        elif type(item) is list:
            children = cast(list[object], item)
        elif isinstance(item, Mapping | Sequence) and not isinstance(item, str | bytes):
            raise ValueError("non-builtin container in verification input")
        else:
            continue
        if depth > _MAX_INPUT_DEPTH or nodes + len(children) > _MAX_INPUT_NODES:
            raise ValueError("verification input exceeds structural limits")
        if identity in active:
            raise ValueError("cyclic verification input")
        active.add(identity)
        stack.append((item, depth, True))
        stack.extend((child, depth + 1, False) for child in reversed(children))


def _validated(value: object, model_type: type[_Model], *, name: str) -> _Model:
    try:
        if type(value) is not model_type or not isinstance(value, BaseModel):
            raise ValueError
        _reject_hidden_cyclic_or_deep(value)
        validated = model_type.model_validate(value.model_dump(mode="python", warnings="error"))
        if validated != value:
            raise ValueError
        return validated
    except (AttributeError, RecursionError, TypeError, ValueError):
        raise ValueError(f"{name} failed exact typed validation") from None


class VerificationOperatorPermission(StrictFrozenModel):
    """One operator-specific subset of the code-owned field allowlist."""

    schema_version: Literal["nurse.v2.verification-operator-permission/1"] = (
        VERIFICATION_OPERATOR_PERMISSION_SCHEMA_VERSION
    )
    operator: VerificationOperator
    requested_fields: tuple[str, ...] = Field(min_length=1)
    permission_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_code_allowlist_subset(self) -> Self:
        if self.operator not in ALLOWED_VERIFICATION_OPERATORS:
            raise ValueError("verification operator is not code-allowlisted")
        if self.requested_fields != tuple(sorted(set(self.requested_fields))):
            raise ValueError("verification fields must be sorted and unique")
        if not set(self.requested_fields) <= set(_OPERATOR_FIELDS[self.operator]):
            raise ValueError("verification field is not code-allowlisted for operator")
        return self

    def verify(self) -> bool:
        try:
            value = VerificationOperatorPermission.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = value.model_dump(mode="json", exclude={"permission_sha256"})
        except (AttributeError, TypeError, ValueError):
            return False
        return value == self and value.permission_sha256 == _digest(unsigned)


def build_verification_operator_permission(
    *,
    operator: VerificationOperator,
    requested_fields: Sequence[str],
) -> VerificationOperatorPermission:
    unsigned: dict[str, object] = {
        "schema_version": VERIFICATION_OPERATOR_PERMISSION_SCHEMA_VERSION,
        "operator": operator,
        "requested_fields": tuple(sorted(set(requested_fields))),
    }
    return VerificationOperatorPermission.model_validate(
        {**unsigned, "permission_sha256": _digest(unsigned)}
    )


class VerificationExecutionPolicy(StrictFrozenModel):
    """Frozen initial allowlist and successful-query budget."""

    schema_version: Literal["nurse.v2.verification-execution-policy/1"] = (
        VERIFICATION_EXECUTION_POLICY_SCHEMA_VERSION
    )
    policy_version: Literal["nurse-v2-verification-execution/1"] = (
        VERIFICATION_EXECUTION_POLICY_VERSION
    )
    query_limit: Literal[8] = INITIAL_VERIFICATION_QUERY_LIMIT
    operator_permissions: tuple[VerificationOperatorPermission, ...]
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_exact_code_policy(self) -> Self:
        expected = tuple(
            (operator, _OPERATOR_FIELDS[operator]) for operator in ALLOWED_VERIFICATION_OPERATORS
        )
        actual = tuple((item.operator, item.requested_fields) for item in self.operator_permissions)
        if actual != expected:
            raise ValueError("verification policy differs from code allowlist")
        return self

    def verify(self) -> bool:
        try:
            value = VerificationExecutionPolicy.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = value.model_dump(mode="json", exclude={"policy_sha256"})
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            value == self
            and all(item.verify() for item in value.operator_permissions)
            and value.policy_sha256 == _digest(unsigned)
        )


def build_verification_execution_policy() -> VerificationExecutionPolicy:
    permissions = tuple(
        build_verification_operator_permission(
            operator=operator,
            requested_fields=_OPERATOR_FIELDS[operator],
        )
        for operator in ALLOWED_VERIFICATION_OPERATORS
    )
    unsigned: dict[str, object] = {
        "schema_version": VERIFICATION_EXECUTION_POLICY_SCHEMA_VERSION,
        "policy_version": VERIFICATION_EXECUTION_POLICY_VERSION,
        "query_limit": INITIAL_VERIFICATION_QUERY_LIMIT,
        "operator_permissions": [item.model_dump(mode="json") for item in permissions],
    }
    return VerificationExecutionPolicy.model_validate(
        {**unsigned, "policy_sha256": _digest(unsigned)}
    )


class VerificationExecutionScope(StrictFrozenModel):
    """Hash-frozen case, time, evidence, relation, and operator permissions."""

    schema_version: Literal["nurse.v2.verification-execution-scope/1"] = (
        VERIFICATION_EXECUTION_SCOPE_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    decision_time_unix_ns: int = Field(ge=0)
    query_scope_interval: TimeInterval
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_universe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entity_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_grouping_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    operator_permissions: tuple[VerificationOperatorPermission, ...]
    scope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_scope(self) -> Self:
        if self.query_scope_interval.end_unix_ns > self.decision_time_unix_ns:
            raise ValueError("verification scope extends beyond decision time")
        if self.observation_ids != tuple(sorted(set(self.observation_ids))):
            raise ValueError("scope observation IDs must be sorted and unique")
        if self.relation_ids != tuple(sorted(set(self.relation_ids))):
            raise ValueError("scope relation IDs must be sorted and unique")
        if self.operator_permissions != tuple(
            sorted(self.operator_permissions, key=lambda item: item.operator)
        ):
            raise ValueError("scope operator permissions must be canonical")
        if len({item.operator for item in self.operator_permissions}) != len(
            self.operator_permissions
        ):
            raise ValueError("scope operator permissions must be unique")
        return self

    def verify(self) -> bool:
        try:
            value = VerificationExecutionScope.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = value.model_dump(mode="json", exclude={"scope_sha256"})
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            value == self
            and all(item.verify() for item in value.operator_permissions)
            and value.scope_sha256 == _digest(unsigned)
        )


def build_verification_execution_scope(
    *,
    policy: VerificationExecutionPolicy,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    observation_ids: Sequence[str],
    relation_ids: Sequence[str],
    operator_permissions: Sequence[VerificationOperatorPermission],
    query_scope_interval: TimeInterval | None = None,
) -> VerificationExecutionScope:
    policy = _validated(policy, VerificationExecutionPolicy, name="verification policy")
    universe = _validated(candidate_universe, CandidateUniverse, name="candidate universe")
    frozen = _validated(frozen_candidates, FrozenCandidateUniverse, name="candidate freeze")
    registry = _validated(entity_registry, TemporalEntityRegistry, name="entity registry")
    grouping = _validated(event_grouping, EventGroupingResult, name="event grouping")
    if not policy.verify() or not registry.verify() or not grouping.verify():
        raise ValueError("verification scope input failed content verification")
    require_frozen_candidate_universe(frozen=frozen, candidate_universe=universe)
    if tuple(item.observation for item in registry.resolutions) != grouping.observations:
        raise ValueError("verification observations differ across frozen inputs")
    if not (
        universe.opaque_case_id
        == frozen.opaque_case_id
        == registry.opaque_case_id
        == grouping.opaque_case_id
    ):
        raise ValueError("verification scope inputs belong to different cases")
    if registry.decision_time_unix_ns != grouping.decision_time_unix_ns:
        raise ValueError("verification scope decision times differ")
    interval = query_scope_interval or TimeInterval(
        start_unix_ns=0,
        end_unix_ns=registry.decision_time_unix_ns,
    )
    rebuilt = build_complete_candidate_universe(
        entity_registry=registry,
        event_grouping=grouping,
        scope_interval=interval,
    )
    if rebuilt != universe:
        raise ValueError("candidate universe is not derived from frozen verification inputs")
    available_observations = {item.observation.observation_id for item in registry.resolutions}
    available_relations = {item.claim_id for item in grouping.association_claims}
    normalized_observations = tuple(sorted(set(observation_ids)))
    normalized_relations = tuple(sorted(set(relation_ids)))
    if not set(normalized_observations) <= available_observations:
        raise ValueError("scope grants unknown observations")
    if not set(normalized_relations) <= available_relations:
        raise ValueError("scope grants unknown relations")
    policy_by_operator = {item.operator: item for item in policy.operator_permissions}
    normalized_permissions = tuple(
        sorted(
            (
                _validated(
                    item,
                    VerificationOperatorPermission,
                    name="operator permission",
                )
                for item in operator_permissions
            ),
            key=lambda item: item.operator,
        )
    )
    if len({item.operator for item in normalized_permissions}) != len(normalized_permissions):
        raise ValueError("duplicate operator permission")
    for permission in normalized_permissions:
        if not permission.verify():
            raise ValueError("operator permission failed verification")
        policy_permission = policy_by_operator.get(permission.operator)
        if policy_permission is None or not set(permission.requested_fields) <= set(
            policy_permission.requested_fields
        ):
            raise ValueError("scope permission exceeds code policy")
    unsigned: dict[str, object] = {
        "schema_version": VERIFICATION_EXECUTION_SCOPE_SCHEMA_VERSION,
        "opaque_case_id": universe.opaque_case_id,
        "decision_time_unix_ns": registry.decision_time_unix_ns,
        "query_scope_interval": interval.model_dump(mode="json"),
        "policy_sha256": policy.policy_sha256,
        "candidate_universe_sha256": universe.universe_sha256,
        "candidate_freeze_sha256": frozen.freeze_sha256,
        "observation_contract_sha256": universe.observation_contract_sha256,
        "entity_registry_sha256": registry.registry_sha256,
        "event_grouping_result_sha256": grouping.result_sha256,
        "observation_ids": normalized_observations,
        "relation_ids": normalized_relations,
        "operator_permissions": [item.model_dump(mode="json") for item in normalized_permissions],
    }
    return VerificationExecutionScope.model_validate(
        {**unsigned, "scope_sha256": _digest(unsigned)}
    )


class VerificationFieldValue(StrictFrozenModel):
    """Canonical scalar returned by a deterministic operator."""

    schema_version: Literal["nurse.v2.verification-field-value/1"] = (
        VERIFICATION_FIELD_VALUE_SCHEMA_VERSION
    )
    field_name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    value_json: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_canonical_scalar(self) -> Self:
        try:
            value = json.loads(self.value_json)
        except (TypeError, ValueError) as error:
            raise ValueError("verification field value is not JSON") from error
        if isinstance(value, list | dict):
            raise ValueError("verification field value must be scalar")
        if canonical_json_bytes(value).decode("utf-8") != self.value_json:
            raise ValueError("verification field value must be canonical JSON")
        return self


def _field_value(field_name: str, value: object) -> VerificationFieldValue:
    return VerificationFieldValue(
        field_name=field_name,
        value_json=canonical_json_bytes(value).decode("utf-8"),
    )


class DerivedVerificationObservation(StrictFrozenModel):
    """Content-addressed observation computed only from authorized frozen sources."""

    schema_version: Literal["nurse.v2.derived-verification-observation/1"] = (
        DERIVED_VERIFICATION_OBSERVATION_SCHEMA_VERSION
    )
    derived_observation_id: str = Field(pattern=r"^vob_[0-9a-f]{24}$")
    verification_request_id: str = Field(pattern=r"^vrq_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_service_key: str = Field(min_length=1)
    operator: VerificationOperator
    query_interval: TimeInterval
    decision_time_unix_ns: int = Field(ge=0)
    values: tuple[VerificationFieldValue, ...] = Field(min_length=1)
    source_observation_ids: tuple[str, ...]
    source_relation_ids: tuple[str, ...]
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    derived_observation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_observation(self) -> Self:
        if self.query_interval.end_unix_ns > self.decision_time_unix_ns:
            raise ValueError("derived observation extends beyond decision time")
        if self.values != tuple(sorted(self.values, key=lambda item: item.field_name)):
            raise ValueError("derived fields must be canonical")
        if len({item.field_name for item in self.values}) != len(self.values):
            raise ValueError("derived fields must be unique")
        for values, name in (
            (self.source_observation_ids, "source observation IDs"),
            (self.source_relation_ids, "source relation IDs"),
            (self.provenance_ids, "provenance IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        return self

    def verify(self) -> bool:
        try:
            value = DerivedVerificationObservation.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = value.model_dump(
                mode="json",
                exclude={"derived_observation_id", "derived_observation_sha256"},
            )
            identifier, digest = _stable_id("vob", unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            value == self
            and value.derived_observation_id == identifier
            and value.derived_observation_sha256 == digest
        )


class VerificationRequestCount(StrictFrozenModel):
    """Cumulative duplicate accounting for one normalized legal request."""

    schema_version: Literal["nurse.v2.verification-request-count/1"] = (
        VERIFICATION_REQUEST_COUNT_SCHEMA_VERSION
    )
    verification_request_id: str = Field(pattern=r"^vrq_[0-9a-f]{24}$")
    submission_count: int = Field(ge=1)


class VerificationExecutionDecision(StrictFrozenModel):
    """Budget and permission outcome for one unique request."""

    schema_version: Literal["nurse.v2.verification-execution-decision/1"] = (
        VERIFICATION_EXECUTION_DECISION_SCHEMA_VERSION
    )
    execution_decision_id: str = Field(pattern=r"^vxd_[0-9a-f]{24}$")
    verification_request_id: str = Field(pattern=r"^vrq_[0-9a-f]{24}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verification_result_id: str = Field(pattern=r"^vrs_[0-9a-f]{24}$")
    reason_code: VerificationReason
    permission_failure: bool
    charged_queries: int = Field(ge=0, le=1)
    remaining_queries: int = Field(ge=0, le=8)
    decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        unsigned = self.model_dump(
            mode="json", exclude={"execution_decision_id", "decision_sha256"}
        )
        identifier, digest = _stable_id("vxd", unsigned)
        return self.execution_decision_id == identifier and self.decision_sha256 == digest


class VerificationBatchResult(StrictFrozenModel):
    """Replayable cumulative ledger for at most eight successful unique queries."""

    schema_version: Literal["nurse.v2.verification-batch-result/1"] = (
        VERIFICATION_BATCH_RESULT_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    policy: VerificationExecutionPolicy
    scope: VerificationExecutionScope
    scope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    submitted_request_count: int = Field(ge=0)
    unique_request_count: int = Field(ge=0)
    duplicate_request_count: int = Field(ge=0)
    requests: tuple[VerificationRequest, ...]
    request_counts: tuple[VerificationRequestCount, ...]
    decisions: tuple[VerificationExecutionDecision, ...]
    results: tuple[VerificationResult, ...]
    derived_observations: tuple[DerivedVerificationObservation, ...]
    preexisting_query_count: int = Field(ge=0, le=8)
    preexisting_query_source_ids: tuple[str, ...]
    charged_query_count: int = Field(ge=0, le=8)
    total_charged_query_count: int = Field(ge=0, le=8)
    remaining_query_count: int = Field(ge=0, le=8)
    batch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            value = VerificationBatchResult.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if value != self or not value.policy.verify() or not value.scope.verify():
                return False
            if (
                value.scope_sha256 != value.scope.scope_sha256
                or value.scope.policy_sha256 != value.policy.policy_sha256
                or value.opaque_case_id != value.scope.opaque_case_id
            ):
                return False
            if value.requests != tuple(
                sorted(value.requests, key=lambda item: item.verification_request_id)
            ):
                return False
            if value.request_counts != tuple(
                sorted(value.request_counts, key=lambda item: item.verification_request_id)
            ):
                return False
            if value.results != tuple(
                sorted(value.results, key=lambda item: item.verification_request_id)
            ):
                return False
            if value.derived_observations != tuple(
                sorted(
                    value.derived_observations,
                    key=lambda item: item.derived_observation_id,
                )
            ):
                return False
            request_ids = {item.verification_request_id for item in value.requests}
            ledger_ids = {item.verification_request_id for item in value.request_counts}
            if len(value.request_counts) != len(ledger_ids):
                return False
            if not request_ids <= ledger_ids:
                return False
            if ledger_ids != {item.verification_request_id for item in value.results}:
                return False
            if ledger_ids != {item.verification_request_id for item in value.decisions}:
                return False
            if value.submitted_request_count != sum(
                item.submission_count for item in value.request_counts
            ):
                return False
            if value.unique_request_count != len(ledger_ids):
                return False
            if value.duplicate_request_count != value.submitted_request_count - len(ledger_ids):
                return False
            if any(not item.verify() for item in value.requests):
                return False
            if any(not item.verify() for item in value.results):
                return False
            if any(not item.verify() for item in value.decisions):
                return False
            if any(not item.verify() for item in value.derived_observations):
                return False
            result_by_id = {item.verification_result_id: item for item in value.results}
            if len(result_by_id) != len(value.results):
                return False
            if len({item.verification_request_id for item in value.decisions}) != len(
                value.decisions
            ):
                return False
            if any(
                decision.verification_result_id not in result_by_id for decision in value.decisions
            ):
                return False
            request_sha_by_id = {
                item.verification_request_id: item.request_sha256 for item in value.requests
            }
            if any(
                decision.verification_request_id in request_sha_by_id
                and decision.request_sha256 != request_sha_by_id[decision.verification_request_id]
                for decision in value.decisions
            ):
                return False
            if any(
                decision.verification_request_id not in request_ids
                and not decision.permission_failure
                for decision in value.decisions
            ):
                return False
            expected_remaining = value.policy.query_limit - value.preexisting_query_count
            for decision in value.decisions:
                result = result_by_id[decision.verification_result_id]
                if result.verification_request_id != decision.verification_request_id:
                    return False
                if decision.permission_failure != (
                    decision.reason_code.startswith("permission_")
                    or decision.reason_code == "invalid_request"
                ):
                    return False
                completed = decision.reason_code == "completed"
                if completed != (decision.charged_queries == 1):
                    return False
                if completed != (result.status == "completed"):
                    return False
                expected_remaining -= decision.charged_queries
                if decision.remaining_queries != expected_remaining:
                    return False
            derived_ids = {item.derived_observation_id for item in value.derived_observations}
            if derived_ids != {
                identifier
                for result in value.results
                for identifier in result.derived_observation_ids
            }:
                return False
            derived_by_id = {
                item.derived_observation_id: item for item in value.derived_observations
            }
            required_provenance = {
                value.policy.policy_sha256,
                value.scope.scope_sha256,
                value.scope.candidate_freeze_sha256,
                value.scope.candidate_universe_sha256,
                value.scope.entity_registry_sha256,
                value.scope.event_grouping_result_sha256,
            }
            for result in value.results:
                for derived_id in result.derived_observation_ids:
                    observation = derived_by_id[derived_id]
                    if (
                        observation.verification_request_id != result.verification_request_id
                        or tuple(result.source_ids) != observation.provenance_ids
                        or not required_provenance <= set(observation.provenance_ids)
                    ):
                        return False
            if value.charged_query_count != sum(item.charged_queries for item in value.decisions):
                return False
            if value.preexisting_query_source_ids != tuple(
                sorted(set(value.preexisting_query_source_ids))
            ):
                return False
            if value.total_charged_query_count != (
                value.preexisting_query_count + value.charged_query_count
            ):
                return False
            if (
                value.remaining_query_count
                != value.policy.query_limit - value.total_charged_query_count
            ):
                return False
            unsigned = value.model_dump(mode="json", exclude={"batch_sha256"})
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return value.batch_sha256 == _digest(unsigned)


def _source_provenance(
    *,
    request: VerificationRequest,
    policy: VerificationExecutionPolicy,
    scope: VerificationExecutionScope,
    frozen_candidates: FrozenCandidateUniverse,
    candidate_universe: CandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    observations: Sequence[Observation],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                *request.observation_ids,
                *request.relation_ids,
                *(item.source_fingerprint for item in observations),
                *(provenance for item in observations for provenance in item.provenance_ids),
                policy.policy_sha256,
                scope.scope_sha256,
                frozen_candidates.freeze_sha256,
                candidate_universe.universe_sha256,
                entity_registry.registry_sha256,
                event_grouping.result_sha256,
            }
        )
    )


def _compare_window_summary_values(observation: Observation) -> dict[str, object]:
    baseline = observation.baseline_summary
    fault = observation.fault_summary
    median_delta = (
        None if baseline.median is None or fault.median is None else fault.median - baseline.median
    )
    return {
        "anomaly_strength": observation.anomaly_strength,
        "baseline_coverage_fraction": baseline.coverage_fraction,
        "baseline_median": baseline.median,
        "baseline_sample_count": baseline.valid_point_count,
        "coverage_quality": observation.coverage_quality,
        "evidence_state": observation.evidence_state,
        "fault_coverage_fraction": fault.coverage_fraction,
        "fault_median": fault.median,
        "fault_sample_count": fault.valid_point_count,
        "median_delta": median_delta,
        "sample_count": baseline.valid_point_count + fault.valid_point_count,
    }


def _measure_onset_interval_values(observation: Observation) -> dict[str, object]:
    """Project one authorized observation interval into closed onset uncertainty bounds.

    ``onset_delay_unix_ns`` is the nonnegative width of that uncertainty
    interval, not a causal lag between services.
    """

    start = observation.interval.start_unix_ns
    end = observation.interval.end_unix_ns
    return {
        "onset_delay_unix_ns": end - start,
        "onset_end_unix_ns": end,
        "onset_start_unix_ns": start,
    }


def _build_derived_observation(
    *,
    request: VerificationRequest,
    provenance_ids: tuple[str, ...],
    values_by_name: Mapping[str, object],
) -> DerivedVerificationObservation:
    values = tuple(
        _field_value(field_name, values_by_name[field_name])
        for field_name in request.requested_fields
    )
    unsigned: dict[str, object] = {
        "schema_version": DERIVED_VERIFICATION_OBSERVATION_SCHEMA_VERSION,
        "verification_request_id": request.verification_request_id,
        "opaque_case_id": request.opaque_case_id,
        "candidate_service_key": request.candidate_service_key,
        "operator": request.operator,
        "query_interval": request.query_interval.model_dump(mode="json"),
        "decision_time_unix_ns": request.decision_time_unix_ns,
        "values": [item.model_dump(mode="json") for item in values],
        "source_observation_ids": request.observation_ids,
        "source_relation_ids": request.relation_ids,
        "provenance_ids": provenance_ids,
    }
    identifier, digest = _stable_id("vob", unsigned)
    return DerivedVerificationObservation.model_validate(
        {
            **unsigned,
            "derived_observation_id": identifier,
            "derived_observation_sha256": digest,
        }
    )


def _permission_reason(
    *,
    request: VerificationRequest,
    policy: VerificationExecutionPolicy,
    scope: VerificationExecutionScope,
    candidate_universe: CandidateUniverse,
    entity_registry: TemporalEntityRegistry,
) -> VerificationReason | None:
    if request.verification_scope_sha256 != scope.scope_sha256:
        return "permission_scope_mismatch"
    if request.opaque_case_id != scope.opaque_case_id:
        return "permission_case_mismatch"
    if request.candidate_freeze_sha256 != scope.candidate_freeze_sha256:
        return "permission_frozen_universe_mismatch"
    if request.decision_time_unix_ns != scope.decision_time_unix_ns:
        return "permission_time_mismatch"
    if (
        request.query_interval.start_unix_ns < scope.query_scope_interval.start_unix_ns
        or request.query_interval.end_unix_ns > scope.query_scope_interval.end_unix_ns
        or request.query_interval.end_unix_ns > request.decision_time_unix_ns
    ):
        return "permission_time_mismatch"
    candidate_by_key = {item.service_entity_key: item for item in candidate_universe.candidates}
    candidate = candidate_by_key.get(request.candidate_service_key)
    if candidate is None:
        return "permission_candidate_mismatch"
    if not set(request.observation_ids) <= set(scope.observation_ids):
        return "permission_observation_denied"
    candidate_observation_ids = {
        item.observation.observation_id
        for item in entity_registry.resolutions
        if item.binding.status == "bound"
        and item.binding.service_entity_key == request.candidate_service_key
    }
    if not set(request.observation_ids) <= candidate_observation_ids:
        return "permission_observation_candidate_mismatch"
    if not set(request.relation_ids) <= set(scope.relation_ids):
        return "permission_relation_denied"
    code_permission = next(
        (item for item in policy.operator_permissions if item.operator == request.operator),
        None,
    )
    scope_permission = next(
        (item for item in scope.operator_permissions if item.operator == request.operator),
        None,
    )
    if (
        request.operator not in ALLOWED_VERIFICATION_OPERATORS
        or code_permission is None
        or scope_permission is None
    ):
        return "permission_operator_denied"
    if not set(request.requested_fields) <= set(code_permission.requested_fields) or not set(
        request.requested_fields
    ) <= set(scope_permission.requested_fields):
        return "permission_field_denied"
    return None


def _expected_completed_artifacts(
    *,
    request: VerificationRequest,
    policy: VerificationExecutionPolicy,
    scope: VerificationExecutionScope,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
) -> tuple[DerivedVerificationObservation, VerificationResult] | None:
    """Replay one legal completed request solely from its frozen typed inputs."""

    if (
        _permission_reason(
            request=request,
            policy=policy,
            scope=scope,
            candidate_universe=candidate_universe,
            entity_registry=entity_registry,
        )
        is not None
    ):
        return None
    observations_by_id = {
        item.observation.observation_id: item.observation for item in entity_registry.resolutions
    }
    resolutions_by_observation_id = {
        item.observation.observation_id: item for item in entity_registry.resolutions
    }
    selected_observations = tuple(
        observations_by_id[item] for item in request.observation_ids if item in observations_by_id
    )
    selected_resolutions = tuple(
        resolutions_by_observation_id[item]
        for item in request.observation_ids
        if item in resolutions_by_observation_id
    )
    if (
        any(item.opaque_case_id != request.opaque_case_id for item in selected_observations)
        or (
            request.signal_family is not None
            and any(item.signal_family != request.signal_family for item in selected_observations)
        )
        or any(item.interval != request.query_interval for item in selected_observations)
    ):
        return None
    provenance = _source_provenance(
        request=request,
        policy=policy,
        scope=scope,
        frozen_candidates=frozen_candidates,
        candidate_universe=candidate_universe,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        observations=selected_observations,
    )
    values_by_name: Mapping[str, object]
    if request.operator == "compare_window_summary" and len(selected_observations) == 1:
        values_by_name = _compare_window_summary_values(selected_observations[0])
    elif request.operator == "measure_onset_interval" and len(selected_observations) == 1:
        values_by_name = _measure_onset_interval_values(selected_observations[0])
    elif request.operator == "resolve_entity_binding" and len(selected_resolutions) == 1:
        binding = selected_resolutions[0].binding
        provenance = tuple(
            sorted(
                {
                    *provenance,
                    binding.binding_id,
                    *binding.source_fingerprints,
                    *binding.provenance_ids,
                }
            )
        )
        values_by_name = {
            "binding_status": binding.status,
            "service_entity_key": binding.service_entity_key,
        }
    else:
        return None
    observation = _build_derived_observation(
        request=request,
        provenance_ids=provenance,
        values_by_name=values_by_name,
    )
    result = build_verification_result(
        verification_request_id=request.verification_request_id,
        status="completed",
        truth_value="unknown",
        derived_observation_ids=(observation.derived_observation_id,),
        source_ids=provenance,
        reason="deterministic_read_only_observation",
    )
    return observation, result


def _expected_valid_request_outcome(
    *,
    request: VerificationRequest,
    remaining_queries: int,
    policy: VerificationExecutionPolicy,
    scope: VerificationExecutionScope,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
) -> tuple[VerificationReason, DerivedVerificationObservation | None, VerificationResult]:
    observations_by_id = {
        item.observation.observation_id: item.observation for item in entity_registry.resolutions
    }
    resolutions_by_observation_id = {
        item.observation.observation_id: item for item in entity_registry.resolutions
    }
    selected_observations = tuple(
        observations_by_id[item] for item in request.observation_ids if item in observations_by_id
    )
    selected_resolutions = tuple(
        resolutions_by_observation_id[item]
        for item in request.observation_ids
        if item in resolutions_by_observation_id
    )
    provenance = _source_provenance(
        request=request,
        policy=policy,
        scope=scope,
        frozen_candidates=frozen_candidates,
        candidate_universe=candidate_universe,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        observations=selected_observations,
    )
    if request.operator == "resolve_entity_binding":
        provenance = tuple(
            sorted(
                {
                    *provenance,
                    *(item.binding.binding_id for item in selected_resolutions),
                    *(
                        source
                        for item in selected_resolutions
                        for source in item.binding.source_fingerprints
                    ),
                    *(
                        source
                        for item in selected_resolutions
                        for source in item.binding.provenance_ids
                    ),
                }
            )
        )
    reason = _permission_reason(
        request=request,
        policy=policy,
        scope=scope,
        candidate_universe=candidate_universe,
        entity_registry=entity_registry,
    )
    expected: tuple[DerivedVerificationObservation, VerificationResult] | None = None
    if reason is None:
        if any(item.opaque_case_id != request.opaque_case_id for item in selected_observations):
            reason = "permission_case_mismatch"
        elif request.signal_family is not None and any(
            item.signal_family != request.signal_family for item in selected_observations
        ):
            reason = "permission_signal_family_mismatch"
        elif any(item.interval != request.query_interval for item in selected_observations):
            reason = "typed_input_unavailable"
        elif remaining_queries < 1:
            reason = "query_budget_exhausted"
        else:
            expected = _expected_completed_artifacts(
                request=request,
                policy=policy,
                scope=scope,
                candidate_universe=candidate_universe,
                frozen_candidates=frozen_candidates,
                entity_registry=entity_registry,
                event_grouping=event_grouping,
            )
            reason = "completed" if expected is not None else "typed_input_unavailable"
    if expected is not None:
        return reason, expected[0], expected[1]
    result = build_verification_result(
        verification_request_id=request.verification_request_id,
        status="rejected",
        truth_value="unknown",
        derived_observation_ids=(),
        source_ids=provenance,
        reason=reason,
    )
    return reason, None, result


def _base_provenance(
    *,
    policy: VerificationExecutionPolicy,
    scope: VerificationExecutionScope,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                policy.policy_sha256,
                scope.scope_sha256,
                frozen_candidates.freeze_sha256,
                candidate_universe.universe_sha256,
                entity_registry.registry_sha256,
                event_grouping.result_sha256,
            }
        )
    )


def _require_prior_completed_results_replay(
    *,
    prior_result: VerificationBatchResult,
    policy: VerificationExecutionPolicy,
    scope: VerificationExecutionScope,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
) -> None:
    request_by_id = {item.verification_request_id: item for item in prior_result.requests}
    result_by_id = {item.verification_request_id: item for item in prior_result.results}
    derived_by_id = {
        item.derived_observation_id: item for item in prior_result.derived_observations
    }
    remaining = policy.query_limit - prior_result.preexisting_query_count
    base_provenance = _base_provenance(
        policy=policy,
        scope=scope,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
    )
    for decision in prior_result.decisions:
        request = request_by_id.get(decision.verification_request_id)
        actual_result = result_by_id.get(decision.verification_request_id)
        if request is None:
            expected_result = build_verification_result(
                verification_request_id=decision.verification_request_id,
                status="rejected",
                truth_value="unknown",
                derived_observation_ids=(),
                source_ids=base_provenance,
                reason="invalid_request",
            )
            expected_decision = _build_decision(
                result=expected_result,
                request_sha256=decision.request_sha256,
                reason_code="invalid_request",
                charged_queries=0,
                remaining_queries=remaining,
            )
            if actual_result != expected_result or decision != expected_decision:
                raise ValueError("prior invalid result does not replay canonically")
            continue
        if actual_result is None:
            raise ValueError("prior completed result does not replay from frozen inputs")
        expected_reason, expected_observation, expected_result = _expected_valid_request_outcome(
            request=request,
            remaining_queries=remaining,
            policy=policy,
            scope=scope,
            candidate_universe=candidate_universe,
            frozen_candidates=frozen_candidates,
            entity_registry=entity_registry,
            event_grouping=event_grouping,
        )
        expected_charged = 1 if expected_reason == "completed" else 0
        expected_remaining = remaining - expected_charged
        expected_decision = _build_decision(
            result=expected_result,
            request_sha256=request.request_sha256,
            reason_code=expected_reason,
            charged_queries=expected_charged,
            remaining_queries=expected_remaining,
        )
        actual_observations = tuple(
            derived_by_id.get(item) for item in actual_result.derived_observation_ids
        )
        expected_observations = () if expected_observation is None else (expected_observation,)
        if (
            decision != expected_decision
            or actual_result != expected_result
            or actual_observations != expected_observations
        ):
            raise ValueError("prior completed result does not replay from frozen inputs")
        remaining = expected_remaining


def _build_decision(
    *,
    result: VerificationResult,
    request_sha256: str,
    reason_code: VerificationReason,
    charged_queries: int,
    remaining_queries: int,
) -> VerificationExecutionDecision:
    permission_failure = reason_code.startswith("permission_") or reason_code == "invalid_request"
    unsigned: dict[str, object] = {
        "schema_version": VERIFICATION_EXECUTION_DECISION_SCHEMA_VERSION,
        "verification_request_id": result.verification_request_id,
        "request_sha256": request_sha256,
        "verification_result_id": result.verification_result_id,
        "reason_code": reason_code,
        "permission_failure": permission_failure,
        "charged_queries": charged_queries,
        "remaining_queries": remaining_queries,
    }
    identifier, digest = _stable_id("vxd", unsigned)
    return VerificationExecutionDecision.model_validate(
        {**unsigned, "execution_decision_id": identifier, "decision_sha256": digest}
    )


def _request_reference(request: object) -> tuple[str, str]:
    request_id = getattr(request, "verification_request_id", None)
    request_sha256 = getattr(request, "request_sha256", None)
    if not isinstance(request_id, str) or re.fullmatch(r"vrq_[0-9a-f]{24}", request_id) is None:
        raise ValueError("invalid verification request ID")
    if not isinstance(request_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", request_sha256) is None:
        raise ValueError("invalid verification request digest")
    return request_id, request_sha256


def _normalize_request(
    request: object,
) -> tuple[VerificationRequest | None, str, str, VerificationReason | None]:
    request_id, request_sha256 = _request_reference(request)
    try:
        _reject_hidden_cyclic_or_deep(request)
    except ValueError:
        return None, request_id, request_sha256, "invalid_request"
    if type(request) is VerificationRequest and isinstance(request, VerificationRequest):
        try:
            content_addressed = request.verify()
        except (AttributeError, RecursionError, TypeError, ValueError):
            content_addressed = False
        if (
            content_addressed
            and type(request.query_interval) is TimeInterval
            and request.query_interval.end_unix_ns > request.decision_time_unix_ns
        ):
            return None, request_id, request_sha256, "invalid_request"
    try:
        validated = _validated(request, VerificationRequest, name="verification request")
    except ValueError:
        return None, request_id, request_sha256, "invalid_request"
    if not validated.verify():
        return None, request_id, request_sha256, "invalid_request"
    return validated, request_id, request_sha256, None


def _validate_execution_inputs(
    *,
    policy: VerificationExecutionPolicy,
    scope: VerificationExecutionScope,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
) -> None:
    policy = _validated(policy, VerificationExecutionPolicy, name="verification policy")
    scope = _validated(scope, VerificationExecutionScope, name="verification scope")
    universe = _validated(candidate_universe, CandidateUniverse, name="candidate universe")
    frozen = _validated(frozen_candidates, FrozenCandidateUniverse, name="candidate freeze")
    registry = _validated(entity_registry, TemporalEntityRegistry, name="entity registry")
    grouping = _validated(event_grouping, EventGroupingResult, name="event grouping")
    if not policy.verify() or not scope.verify() or not registry.verify() or not grouping.verify():
        raise ValueError("verification execution input failed content verification")
    require_frozen_candidate_universe(frozen=frozen, candidate_universe=universe)
    if tuple(item.observation for item in registry.resolutions) != grouping.observations:
        raise ValueError("verification observations differ across frozen inputs")
    if not (
        scope.opaque_case_id
        == universe.opaque_case_id
        == frozen.opaque_case_id
        == registry.opaque_case_id
        == grouping.opaque_case_id
    ):
        raise ValueError("verification execution inputs belong to different cases")
    if not (
        scope.policy_sha256 == policy.policy_sha256
        and scope.candidate_universe_sha256 == universe.universe_sha256
        and scope.candidate_freeze_sha256 == frozen.freeze_sha256
        and scope.observation_contract_sha256 == universe.observation_contract_sha256
        and scope.entity_registry_sha256 == registry.registry_sha256
        and scope.event_grouping_result_sha256 == grouping.result_sha256
    ):
        raise ValueError("verification scope does not match frozen input hashes")
    if scope.decision_time_unix_ns != registry.decision_time_unix_ns:
        raise ValueError("verification scope decision time differs")
    available_observations = {item.observation.observation_id for item in registry.resolutions}
    available_relations = {item.claim_id for item in grouping.association_claims}
    if (
        not set(scope.observation_ids) <= available_observations
        or not set(scope.relation_ids) <= available_relations
    ):
        raise ValueError("verification scope authorization exceeds frozen evidence")
    policy_permissions = {item.operator: item for item in policy.operator_permissions}
    for permission in scope.operator_permissions:
        allowed = policy_permissions.get(permission.operator)
        if (
            not permission.verify()
            or allowed is None
            or not set(permission.requested_fields) <= set(allowed.requested_fields)
        ):
            raise ValueError("verification scope authorization exceeds frozen policy")
    rebuilt = build_complete_candidate_universe(
        entity_registry=registry,
        event_grouping=grouping,
        scope_interval=scope.query_scope_interval,
    )
    if rebuilt != universe:
        raise ValueError("candidate universe is not derived from frozen verification inputs")


def execute_verification_batch(
    *,
    requests: Sequence[VerificationRequest],
    policy: VerificationExecutionPolicy,
    scope: VerificationExecutionScope,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    prior_result: VerificationBatchResult | None = None,
    prior_card_expansion_result: CardExpansionBatchResult | None = None,
) -> VerificationBatchResult:
    """Execute normalized requests with no I/O and no candidate-set mutation."""

    _validate_execution_inputs(
        policy=policy,
        scope=scope,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
    )
    card_query_count = 0
    card_query_source_ids: tuple[str, ...] = ()
    if prior_card_expansion_result is not None:
        card_result = _validated(
            prior_card_expansion_result,
            CardExpansionBatchResult,
            name="prior card expansion result",
        )
        if not card_result.verify():
            raise ValueError("prior card expansion result failed verification")
        if (
            card_result.opaque_case_id != scope.opaque_case_id
            or card_result.candidate_freeze_sha256 != scope.candidate_freeze_sha256
        ):
            raise ValueError("prior card expansion result belongs to another scope")
        if card_result.charged_query_count > policy.query_limit:
            raise ValueError("prior card expansion exceeds the global query budget")
        card_query_count = card_result.charged_query_count
        card_query_source_ids = (card_result.result_sha256,)
    if prior_result is not None:
        prior_result = _validated(
            prior_result,
            VerificationBatchResult,
            name="prior verification result",
        )
        if not prior_result.verify():
            raise ValueError("prior verification result failed verification")
        if (
            prior_result.policy != policy
            or prior_result.scope_sha256 != scope.scope_sha256
            or prior_result.opaque_case_id != scope.opaque_case_id
        ):
            raise ValueError("prior verification result belongs to another scope")
        if (
            prior_result.preexisting_query_count != card_query_count
            or prior_result.preexisting_query_source_ids != card_query_source_ids
        ):
            raise ValueError("prior verification result uses a different preexisting ledger")
        _require_prior_completed_results_replay(
            prior_result=prior_result,
            policy=policy,
            scope=scope,
            candidate_universe=candidate_universe,
            frozen_candidates=frozen_candidates,
            entity_registry=entity_registry,
            event_grouping=event_grouping,
        )

    request_by_id: dict[str, VerificationRequest] = {}
    counts: Counter[str] = Counter()
    invalid_requests: dict[str, tuple[str, VerificationReason]] = {}
    if prior_result is not None:
        request_by_id.update({item.verification_request_id: item for item in prior_result.requests})
        counts.update(
            {
                item.verification_request_id: item.submission_count
                for item in prior_result.request_counts
            }
        )
    prior_request_ids = set(counts)
    prior_verified_request_ids = set(request_by_id)
    prior_request_sha_by_id = (
        {}
        if prior_result is None
        else {item.verification_request_id: item.request_sha256 for item in prior_result.decisions}
    )
    for request in requests:
        validated, request_id, request_sha256, invalid_reason = _normalize_request(request)
        prior_sha256 = prior_request_sha_by_id.get(request_id)
        if prior_sha256 is not None and prior_sha256 != request_sha256:
            raise ValueError("verification request collides with prior result")
        counts[request_id] += 1
        if validated is None:
            if request_id in request_by_id:
                raise ValueError("invalid request collides with a verified request")
            previous_invalid = invalid_requests.setdefault(
                request_id,
                (request_sha256, cast(VerificationReason, invalid_reason)),
            )
            if previous_invalid != (request_sha256, invalid_reason):
                raise ValueError("invalid verification request ID collision")
            continue
        if request_id in invalid_requests or (
            request_id in prior_request_ids and request_id not in prior_verified_request_ids
        ):
            raise ValueError("verified request collides with an invalid request")
        previous = request_by_id.setdefault(validated.verification_request_id, validated)
        if previous != validated:
            raise ValueError("verification request ID collision")

    remaining = (
        policy.query_limit - card_query_count
        if prior_result is None
        else prior_result.remaining_query_count
    )
    decisions = [] if prior_result is None else list(prior_result.decisions)
    results: dict[str, VerificationResult] = (
        {}
        if prior_result is None
        else {item.verification_request_id: item for item in prior_result.results}
    )
    derived: dict[str, DerivedVerificationObservation] = (
        {}
        if prior_result is None
        else {item.derived_observation_id: item for item in prior_result.derived_observations}
    )
    observations_by_id = {
        item.observation.observation_id: item.observation for item in entity_registry.resolutions
    }
    resolutions_by_observation_id = {
        item.observation.observation_id: item for item in entity_registry.resolutions
    }
    base_provenance = _base_provenance(
        policy=policy,
        scope=scope,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
    )
    for request_id in sorted(set(invalid_requests) - prior_request_ids):
        request_sha256, rejection_reason = invalid_requests[request_id]
        result = build_verification_result(
            verification_request_id=request_id,
            status="rejected",
            truth_value="unknown",
            derived_observation_ids=(),
            source_ids=base_provenance,
            reason=rejection_reason,
        )
        results[request_id] = result
        decisions.append(
            _build_decision(
                result=result,
                request_sha256=request_sha256,
                reason_code=rejection_reason,
                charged_queries=0,
                remaining_queries=remaining,
            )
        )
    for request_id in sorted(set(request_by_id) - prior_request_ids):
        request = request_by_id[request_id]
        execution_reason = _permission_reason(
            request=request,
            policy=policy,
            scope=scope,
            candidate_universe=candidate_universe,
            entity_registry=entity_registry,
        )
        selected_observations = tuple(
            observations_by_id[item]
            for item in request.observation_ids
            if item in observations_by_id
        )
        selected_resolutions = tuple(
            resolutions_by_observation_id[item]
            for item in request.observation_ids
            if item in resolutions_by_observation_id
        )
        provenance = _source_provenance(
            request=request,
            policy=policy,
            scope=scope,
            frozen_candidates=frozen_candidates,
            candidate_universe=candidate_universe,
            entity_registry=entity_registry,
            event_grouping=event_grouping,
            observations=selected_observations,
        )
        if request.operator == "resolve_entity_binding":
            provenance = tuple(
                sorted(
                    {
                        *provenance,
                        *(item.binding.binding_id for item in selected_resolutions),
                        *(
                            source
                            for item in selected_resolutions
                            for source in item.binding.source_fingerprints
                        ),
                        *(
                            source
                            for item in selected_resolutions
                            for source in item.binding.provenance_ids
                        ),
                    }
                )
            )
        output: DerivedVerificationObservation | None = None
        completed_result: VerificationResult | None = None
        if execution_reason is None:
            if any(item.opaque_case_id != request.opaque_case_id for item in selected_observations):
                execution_reason = "permission_case_mismatch"
            elif request.signal_family is not None and any(
                item.signal_family != request.signal_family for item in selected_observations
            ):
                execution_reason = "permission_signal_family_mismatch"
            elif any(item.interval != request.query_interval for item in selected_observations):
                execution_reason = "typed_input_unavailable"
            elif remaining < 1:
                execution_reason = "query_budget_exhausted"
            else:
                expected = _expected_completed_artifacts(
                    request=request,
                    policy=policy,
                    scope=scope,
                    candidate_universe=candidate_universe,
                    frozen_candidates=frozen_candidates,
                    entity_registry=entity_registry,
                    event_grouping=event_grouping,
                )
                if expected is None:
                    execution_reason = "typed_input_unavailable"
                else:
                    output, completed_result = expected
                    execution_reason = "completed"
        charged = 1 if execution_reason == "completed" else 0
        remaining -= charged
        if output is not None:
            derived[output.derived_observation_id] = output
            if completed_result is None:  # pragma: no cover - narrowed by dispatcher
                raise AssertionError("completed verification output has no result")
            result = completed_result
        else:
            result = build_verification_result(
                verification_request_id=request.verification_request_id,
                status="rejected",
                truth_value="unknown",
                derived_observation_ids=(),
                source_ids=provenance,
                reason=execution_reason,
            )
        results[request.verification_request_id] = result
        decisions.append(
            _build_decision(
                result=result,
                request_sha256=request.request_sha256,
                reason_code=execution_reason,
                charged_queries=charged,
                remaining_queries=remaining,
            )
        )

    ordered_requests = tuple(
        sorted(request_by_id.values(), key=lambda item: item.verification_request_id)
    )
    request_counts = tuple(
        VerificationRequestCount(
            verification_request_id=request_id,
            submission_count=counts[request_id],
        )
        for request_id in sorted(counts)
    )
    charged_query_count = sum(item.charged_queries for item in decisions)
    unsigned: dict[str, object] = {
        "schema_version": VERIFICATION_BATCH_RESULT_SCHEMA_VERSION,
        "opaque_case_id": scope.opaque_case_id,
        "policy": policy.model_dump(mode="json"),
        "scope": scope.model_dump(mode="json"),
        "scope_sha256": scope.scope_sha256,
        "submitted_request_count": sum(item.submission_count for item in request_counts),
        "unique_request_count": len(request_counts),
        "duplicate_request_count": sum(item.submission_count - 1 for item in request_counts),
        "requests": [item.model_dump(mode="json") for item in ordered_requests],
        "request_counts": [item.model_dump(mode="json") for item in request_counts],
        "decisions": [item.model_dump(mode="json") for item in decisions],
        "results": [
            item.model_dump(mode="json")
            for item in sorted(results.values(), key=lambda item: item.verification_request_id)
        ],
        "derived_observations": [
            item.model_dump(mode="json")
            for item in sorted(derived.values(), key=lambda item: item.derived_observation_id)
        ],
        "preexisting_query_count": card_query_count,
        "preexisting_query_source_ids": card_query_source_ids,
        "charged_query_count": charged_query_count,
        "total_charged_query_count": card_query_count + charged_query_count,
        "remaining_query_count": remaining,
    }
    batch_result = VerificationBatchResult.model_validate(
        {**unsigned, "batch_sha256": _digest(unsigned)}
    )
    if not batch_result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("verification batch result failed verification")
    return batch_result


__all__ = [
    "ALLOWED_VERIFICATION_OPERATORS",
    "DERIVED_VERIFICATION_OBSERVATION_SCHEMA_VERSION",
    "INITIAL_VERIFICATION_QUERY_LIMIT",
    "VERIFICATION_BATCH_RESULT_SCHEMA_VERSION",
    "VERIFICATION_EXECUTION_DECISION_SCHEMA_VERSION",
    "VERIFICATION_EXECUTION_POLICY_SCHEMA_VERSION",
    "VERIFICATION_EXECUTION_SCOPE_SCHEMA_VERSION",
    "VERIFICATION_FIELD_VALUE_SCHEMA_VERSION",
    "VERIFICATION_OPERATOR_PERMISSION_SCHEMA_VERSION",
    "VERIFICATION_REQUEST_COUNT_SCHEMA_VERSION",
    "DerivedVerificationObservation",
    "VerificationBatchResult",
    "VerificationExecutionDecision",
    "VerificationExecutionPolicy",
    "VerificationExecutionScope",
    "VerificationFieldValue",
    "VerificationOperatorPermission",
    "VerificationRequestCount",
    "build_verification_execution_policy",
    "build_verification_execution_scope",
    "build_verification_operator_permission",
    "execute_verification_batch",
]
