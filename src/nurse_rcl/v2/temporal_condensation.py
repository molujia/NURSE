"""Closed onset intervals and relation-scoped graph condensation."""

from __future__ import annotations

import hmac
import json
from collections import deque
from collections.abc import Sequence
from typing import Literal, TypeVar, cast

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.graph_snapshots import (
    GRAPH_SNAPSHOT_SCHEMA_VERSION,
    GraphCoverageSummary,
    GraphMetricSummary,
    GraphRelationType,
    GraphSnapshot,
    GraphSnapshotEdge,
)
from nurse_rcl.v2.observation_schemas import CaseTimeContract, TimeInterval
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree
from nurse_rcl.v2.verification_executor import (
    DerivedVerificationObservation,
    VerificationFieldValue,
)

ONSET_INTERVAL_SCHEMA_VERSION: Literal["nurse.v2.onset-interval/2"] = "nurse.v2.onset-interval/2"
CLOCK_RESOLUTION_SCHEMA_VERSION: Literal["nurse.v2.frozen-clock-resolution/1"] = (
    "nurse.v2.frozen-clock-resolution/1"
)
TEMPORAL_PRECEDENCE_SCHEMA_VERSION: Literal["nurse.v2.temporal-precedence/3"] = (
    "nurse.v2.temporal-precedence/3"
)
RELATION_SCC_COMPONENT_SCHEMA_VERSION: Literal["nurse.v2.relation-scc-component/1"] = (
    "nurse.v2.relation-scc-component/1"
)
RELATION_CONDENSED_EDGE_SCHEMA_VERSION: Literal["nurse.v2.relation-condensed-edge/1"] = (
    "nurse.v2.relation-condensed-edge/1"
)
RELATION_SCC_CONDENSATION_SCHEMA_VERSION: Literal["nurse.v2.relation-scc-condensation/1"] = (
    "nurse.v2.relation-scc-condensation/1"
)

DirectedPropagationRelation = Literal[
    "received_dependency_failure",
    "caller_load_transmission",
]
ClockResolutionStatus = Literal["resolved", "unresolved"]
ClockResolutionBasis = Literal[
    "zero_uncertainty",
    "bounded_uncertainty",
    "unresolved",
]

TemporalPrecedenceReason = Literal[
    "strict_precedence",
    "equal_or_overlapping",
    "missing_onset",
    "onset_outside_time_contract",
    "unresolved_clock_uncertainty",
]


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


_ModelT = TypeVar("_ModelT", bound=StrictFrozenModel)

# These ceilings are deliberately far above the per-case RCABench/AIOps25
# cardinalities while keeping replay and condensation work predictably bounded.
_MAX_CONDENSATION_NODES = 4_096
_MAX_CONDENSATION_SELECTED_EDGES = 32_768
_MAX_CONDENSATION_PROVENANCE_IDS = 131_072
_MAX_SERIALIZED_STRING_LENGTH = 8_192
_SERIALIZATION_PREFLIGHT_MAX_DEPTH = 64
_SERIALIZATION_PREFLIGHT_MAX_NODES = 500_000


class _TemporalResourceLimitError(ValueError):
    """Raised when untrusted temporal replay inputs exceed a fixed ceiling."""


def _serializable_model_types() -> frozenset[type[StrictFrozenModel]]:
    """Return the closed set of exact models serialized by this module."""

    return frozenset(
        {
            CaseTimeContract,
            DerivedVerificationObservation,
            FrozenClockResolution,
            GraphCoverageSummary,
            GraphMetricSummary,
            GraphSnapshot,
            GraphSnapshotEdge,
            OnsetInterval,
            RelationCondensedEdge,
            RelationSCCComponent,
            RelationSCCCondensation,
            TemporalPrecedence,
            TimeInterval,
            VerificationFieldValue,
        }
    )


def _preflight_pydantic_serialization(value: object, *, label: str) -> None:
    """Reject hostile or unbounded nested values before Pydantic sees them."""

    preflight_exact_pydantic_tree(
        value,
        root_model_types=tuple(_serializable_model_types()),
        label=label,
        max_depth=_SERIALIZATION_PREFLIGHT_MAX_DEPTH,
        max_nodes=_SERIALIZATION_PREFLIGHT_MAX_NODES,
        max_string_length=_MAX_SERIALIZED_STRING_LENGTH,
        provenance_id_limit=_MAX_CONDENSATION_PROVENANCE_IDS,
        resource_error_type=_TemporalResourceLimitError,
    )


def _exact_model_tuple_field(value: object, *, field_name: str, label: str) -> tuple[object, ...]:
    state = object.__getattribute__(value, "__dict__")
    if type(state) is not dict or field_name not in state:
        raise ValueError(f"{label} lacks required field {field_name}")
    field_value = dict.__getitem__(state, field_name)
    if type(field_value) is not tuple:
        raise ValueError(f"{label} field {field_name} must use an exact tuple")
    return field_value


def _preflight_model_cardinality(
    value: object,
    *,
    model_type: type[StrictFrozenModel],
    label: str,
) -> None:
    """Apply graph-size ceilings before recursive replay or serialization."""

    condensation_type = globals().get("RelationSCCCondensation")
    if model_type is GraphSnapshot:
        nodes = _exact_model_tuple_field(
            value,
            field_name="node_service_keys",
            label=label,
        )
        edges = _exact_model_tuple_field(value, field_name="edges", label=label)
    elif condensation_type is not None and model_type is condensation_type:
        nodes = _exact_model_tuple_field(
            value,
            field_name="node_service_keys",
            label=label,
        )
        edges = _exact_model_tuple_field(
            value,
            field_name="selected_original_edges",
            label=label,
        )
    else:
        return
    if tuple.__len__(nodes) > _MAX_CONDENSATION_NODES:
        raise _TemporalResourceLimitError(f"{label} exceeds the node limit")
    if tuple.__len__(edges) > _MAX_CONDENSATION_SELECTED_EDGES:
        raise _TemporalResourceLimitError(f"{label} exceeds the selected edge limit")


def _dump_exact_model(
    value: object,
    *,
    model_type: type[_ModelT],
    label: str,
    exclude: set[str] | None = None,
) -> dict[str, object]:
    """Serialize one exact model only after callback-free recursive preflight."""

    if type(value) is not model_type:
        raise ValueError(f"{label} must use the exact model type")
    _preflight_model_cardinality(
        value,
        model_type=model_type,
        label=label,
    )
    _preflight_pydantic_serialization(value, label=label)
    return model_type.model_dump(
        value,
        mode="json",
        warnings="error",
        exclude=exclude,
    )


def _has_same_exact_runtime_shape(value: object, rebuilt: object) -> bool:
    """Compare replay shapes without dispatching to caller-defined methods."""

    if type(value) is not type(rebuilt):
        return False
    if isinstance(rebuilt, StrictFrozenModel):
        try:
            value_state = object.__getattribute__(value, "__dict__")
            rebuilt_state = object.__getattribute__(rebuilt, "__dict__")
        except AttributeError:
            return False
        if type(value_state) is not dict or type(rebuilt_state) is not dict:
            return False
        if any(type(key) is not str for key in (*value_state, *rebuilt_state)):
            return False
        if sorted(value_state) != sorted(rebuilt_state):
            return False
        return all(
            _has_same_exact_runtime_shape(
                dict.__getitem__(value_state, field_name),
                dict.__getitem__(rebuilt_state, field_name),
            )
            for field_name in rebuilt_state
        )
    if type(rebuilt) in {tuple, list}:
        value_sequence = cast(Sequence[object], value)
        rebuilt_sequence = cast(Sequence[object], rebuilt)
        if len(value_sequence) != len(rebuilt_sequence):
            return False
        return all(
            _has_same_exact_runtime_shape(item, clean_item)
            for item, clean_item in zip(value_sequence, rebuilt_sequence, strict=True)
        )
    if type(rebuilt) is dict:
        value_mapping = cast(dict[object, object], value)
        rebuilt_mapping = cast(dict[object, object], rebuilt)
        if len(value_mapping) != len(rebuilt_mapping):
            return False
        if any(type(key) not in {str, int} for key in (*value_mapping, *rebuilt_mapping)):
            return False
        return all(
            key in value_mapping
            and _has_same_exact_runtime_shape(value_mapping[key], rebuilt_mapping[key])
            for key in rebuilt_mapping
        )
    return True


def _trusted_exact_model(
    value: object,
    *,
    model_type: type[_ModelT],
    label: str,
) -> _ModelT:
    """Defensively replay one exact model without calling instance methods."""

    if type(value) is not model_type:
        raise ValueError(f"{label} must use the exact model type")
    document = _dump_exact_model(value, model_type=model_type, label=label)
    rebuilt = model_type.model_validate(document)
    if type(rebuilt) is not model_type or not _has_same_exact_runtime_shape(value, rebuilt):
        raise ValueError(f"{label} is not an exact canonical model")
    if _dump_exact_model(rebuilt, model_type=model_type, label=label) != document:
        raise ValueError(f"{label} is not canonical")
    return rebuilt


def _trusted_graph_snapshot(value: object) -> GraphSnapshot:
    rebuilt = _trusted_exact_model(
        value,
        model_type=GraphSnapshot,
        label="graph snapshot",
    )
    if not GraphSnapshot.verify(rebuilt):
        raise ValueError("graph snapshot failed exact replay")
    return rebuilt


def case_time_contract_sha256(value: CaseTimeContract) -> str:
    """Return the digest of one exact canonical case-time contract."""

    contract = _trusted_exact_model(
        value,
        model_type=CaseTimeContract,
        label="case time contract",
    )
    return _digest(
        _dump_exact_model(
            contract,
            model_type=CaseTimeContract,
            label="case time contract",
        )
    )


def _require_expected_sha256(*, actual: str, expected: str, label: str) -> None:
    if (
        type(expected) is not str
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(f"expected {label} SHA-256 is invalid")
    if not hmac.compare_digest(actual, expected):
        raise ValueError(f"artifact differs from the expected {label}")


class FrozenClockResolution(StrictFrozenModel):
    """Content-addressed clock-order permission bound to one time contract."""

    schema_version: Literal["nurse.v2.frozen-clock-resolution/1"] = CLOCK_RESOLUTION_SCHEMA_VERSION
    resolution_id: str = Field(pattern=r"^clockres_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    time_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    clock_uncertainty_ns: int = Field(ge=0)
    resolution_status: ClockResolutionStatus
    resolution_basis: ClockResolutionBasis
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    resolution_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_resolution(self) -> Self:
        if self.provenance_ids != tuple(sorted(set(self.provenance_ids))):
            raise ValueError("clock-resolution provenance IDs must be sorted and unique")
        expected_basis: ClockResolutionBasis
        if self.resolution_status == "unresolved":
            expected_basis = "unresolved"
        elif self.clock_uncertainty_ns == 0:
            expected_basis = "zero_uncertainty"
        else:
            expected_basis = "bounded_uncertainty"
        if self.resolution_basis != expected_basis:
            raise ValueError("clock-resolution basis does not match its frozen status")
        unsigned = _dump_exact_model(
            self,
            model_type=FrozenClockResolution,
            label="frozen clock resolution",
            exclude={"resolution_id", "resolution_sha256"},
        )
        digest = _digest(unsigned)
        if self.resolution_id != f"clockres_{digest[:24]}" or self.resolution_sha256 != digest:
            raise ValueError("clock resolution is not content-addressed")
        return self

    def verify(self) -> bool:
        try:
            rebuilt = _trusted_exact_model(
                self,
                model_type=FrozenClockResolution,
                label="frozen clock resolution",
            )
            unsigned = _dump_exact_model(
                rebuilt,
                model_type=FrozenClockResolution,
                label="frozen clock resolution",
                exclude={"resolution_id", "resolution_sha256"},
            )
            digest = _digest(unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            rebuilt.resolution_id == f"clockres_{digest[:24]}"
            and rebuilt.resolution_sha256 == digest
        )


def freeze_clock_resolution(
    *,
    time_contract: CaseTimeContract,
    expected_time_contract_sha256: str,
    resolution_status: ClockResolutionStatus,
    provenance_ids: Sequence[str],
) -> FrozenClockResolution:
    """Freeze adapter clock state before any temporal precedence decision."""

    contract = _trusted_exact_model(
        time_contract,
        model_type=CaseTimeContract,
        label="case time contract",
    )
    contract_sha256 = case_time_contract_sha256(contract)
    _require_expected_sha256(
        actual=contract_sha256,
        expected=expected_time_contract_sha256,
        label="case time contract",
    )
    if type(resolution_status) is not str or resolution_status not in {
        "resolved",
        "unresolved",
    }:
        raise ValueError("clock resolution status must be resolved or unresolved")
    if type(provenance_ids) not in {list, tuple} or any(
        type(value) is not str for value in provenance_ids
    ):
        raise ValueError("clock-resolution provenance must use exact strings")
    normalized_provenance = tuple(sorted({contract_sha256, *provenance_ids}))
    basis: ClockResolutionBasis
    if resolution_status == "unresolved":
        basis = "unresolved"
    elif contract.clock_uncertainty_ns == 0:
        basis = "zero_uncertainty"
    else:
        basis = "bounded_uncertainty"
    unsigned: dict[str, object] = {
        "schema_version": CLOCK_RESOLUTION_SCHEMA_VERSION,
        "opaque_case_id": contract.opaque_case_id,
        "time_contract_sha256": contract_sha256,
        "clock_uncertainty_ns": contract.clock_uncertainty_ns,
        "resolution_status": resolution_status,
        "resolution_basis": basis,
        "provenance_ids": normalized_provenance,
    }
    digest = _digest(unsigned)
    return FrozenClockResolution.model_validate(
        {
            **unsigned,
            "resolution_id": f"clockres_{digest[:24]}",
            "resolution_sha256": digest,
        }
    )


def require_frozen_clock_resolution(
    *,
    clock_resolution: FrozenClockResolution,
    time_contract: CaseTimeContract,
    expected_clock_resolution_sha256: str,
) -> FrozenClockResolution:
    """Verify a pinned clock-resolution artifact against its exact time contract."""

    contract = _trusted_exact_model(
        time_contract,
        model_type=CaseTimeContract,
        label="case time contract",
    )
    resolution = _trusted_exact_model(
        clock_resolution,
        model_type=FrozenClockResolution,
        label="frozen clock resolution",
    )
    if not FrozenClockResolution.verify(resolution):
        raise ValueError("frozen clock resolution failed exact content verification")
    _require_expected_sha256(
        actual=resolution.resolution_sha256,
        expected=expected_clock_resolution_sha256,
        label="frozen clock resolution",
    )
    contract_sha256 = case_time_contract_sha256(contract)
    if (
        resolution.opaque_case_id != contract.opaque_case_id
        or resolution.time_contract_sha256 != contract_sha256
        or resolution.clock_uncertainty_ns != contract.clock_uncertainty_ns
    ):
        raise ValueError("frozen clock resolution belongs to another case time contract")
    return resolution


class OnsetInterval(StrictFrozenModel):
    """Content-addressed closed anomaly-onset interval ``[t_low, t_high]``."""

    schema_version: Literal["nurse.v2.onset-interval/2"] = ONSET_INTERVAL_SCHEMA_VERSION
    onset_id: str = Field(pattern=r"^onset_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    time_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    anomaly_id: str = Field(min_length=1)
    service_key: str = Field(min_length=1)
    t_low_unix_ns: int = Field(ge=0)
    t_high_unix_ns: int = Field(ge=0)
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    onset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_closed_canonical_interval(self) -> Self:
        if self.t_high_unix_ns < self.t_low_unix_ns:
            raise ValueError("closed onset upper bound must not precede its lower bound")
        if self.provenance_ids != tuple(sorted(set(self.provenance_ids))):
            raise ValueError("onset provenance IDs must be sorted and unique")
        unsigned = _dump_exact_model(
            self,
            model_type=OnsetInterval,
            label="onset interval",
            exclude={"onset_id", "onset_sha256"},
        )
        digest = _digest(unsigned)
        if self.onset_id != f"onset_{digest[:24]}" or self.onset_sha256 != digest:
            raise ValueError("onset interval is not content-addressed")
        return self

    def verify(self) -> bool:
        try:
            rebuilt = _trusted_exact_model(
                self,
                model_type=OnsetInterval,
                label="onset interval",
            )
        except (AttributeError, TypeError, ValueError):
            return False
        return _dump_exact_model(
            rebuilt,
            model_type=OnsetInterval,
            label="onset interval",
        ) == _dump_exact_model(
            self,
            model_type=OnsetInterval,
            label="onset interval",
        )

    def verify_inputs(
        self,
        *,
        derived_observation: DerivedVerificationObservation,
        time_contract: CaseTimeContract,
    ) -> bool:
        """Replay this onset from one authoritative derived observation."""

        try:
            artifact = _trusted_exact_model(
                self,
                model_type=OnsetInterval,
                label="onset interval",
            )
            replayed = build_onset_interval_from_derived_observation(
                derived_observation=derived_observation,
                time_contract=time_contract,
            )
            return OnsetInterval.verify(artifact) and _dump_exact_model(
                replayed,
                model_type=OnsetInterval,
                label="onset interval",
            ) == _dump_exact_model(
                artifact,
                model_type=OnsetInterval,
                label="onset interval",
            )
        except (AttributeError, TypeError, ValueError):
            return False


def build_onset_interval_from_derived_observation(
    *,
    derived_observation: DerivedVerificationObservation,
    time_contract: CaseTimeContract,
) -> OnsetInterval:
    """Build an onset solely from replay-verified ``measure_onset_interval`` output."""

    observation = _trusted_exact_model(
        derived_observation,
        model_type=DerivedVerificationObservation,
        label="derived onset observation",
    )
    if not DerivedVerificationObservation.verify(observation):
        raise ValueError("derived onset observation failed content verification")
    contract = _trusted_exact_model(
        time_contract,
        model_type=CaseTimeContract,
        label="case time contract",
    )
    if observation.operator != "measure_onset_interval":
        raise ValueError("derived observation is not an onset measurement")
    if (
        observation.opaque_case_id != contract.opaque_case_id
        or observation.decision_time_unix_ns != contract.decision_time_unix_ns
    ):
        raise ValueError("derived onset observation belongs to another time contract")
    values_by_name = {item.field_name: json.loads(item.value_json) for item in observation.values}
    required_names = {"onset_end_unix_ns", "onset_start_unix_ns"}
    if not required_names <= set(values_by_name) or not set(values_by_name) <= {
        *required_names,
        "onset_delay_unix_ns",
    }:
        raise ValueError("derived onset observation has invalid fields")
    start = values_by_name["onset_start_unix_ns"]
    end = values_by_name["onset_end_unix_ns"]
    if type(start) is not int or type(end) is not int:
        raise ValueError("derived onset bounds must use exact integers")
    if (
        start != observation.query_interval.start_unix_ns
        or end != observation.query_interval.end_unix_ns
    ):
        raise ValueError("derived onset bounds differ from the authoritative interval")
    delay = values_by_name.get("onset_delay_unix_ns")
    if delay is not None and (type(delay) is not int or delay != end - start):
        raise ValueError("derived onset delay differs from the authoritative interval width")
    provenance_ids = tuple(
        sorted(
            {
                observation.derived_observation_id,
                observation.derived_observation_sha256,
                observation.verification_request_id,
                *observation.source_observation_ids,
                *observation.source_relation_ids,
                *observation.provenance_ids,
            }
        )
    )
    return build_onset_interval(
        time_contract=contract,
        anomaly_id=observation.derived_observation_id,
        service_key=observation.candidate_service_key,
        t_low_unix_ns=start,
        t_high_unix_ns=end,
        provenance_ids=provenance_ids,
    )


def build_onset_interval(
    *,
    time_contract: CaseTimeContract,
    anomaly_id: str,
    service_key: str,
    t_low_unix_ns: int,
    t_high_unix_ns: int,
    provenance_ids: Sequence[str],
) -> OnsetInterval:
    contract = _trusted_exact_model(
        time_contract,
        model_type=CaseTimeContract,
        label="case time contract",
    )
    if type(t_low_unix_ns) is not int or type(t_high_unix_ns) is not int:
        raise ValueError("onset bounds must use exact integers")
    if (
        t_low_unix_ns < contract.baseline_window.start_unix_ns
        or t_high_unix_ns < contract.baseline_window.start_unix_ns
        or t_low_unix_ns > contract.decision_time_unix_ns
        or t_high_unix_ns > contract.decision_time_unix_ns
    ):
        raise ValueError("onset bounds fall outside the inference time contract")
    if type(provenance_ids) not in {list, tuple}:
        raise ValueError("onset provenance IDs must use an exact list or tuple")
    if any(type(value) is not str for value in provenance_ids):
        raise ValueError("onset provenance IDs must be strings")
    unsigned: dict[str, object] = {
        "schema_version": ONSET_INTERVAL_SCHEMA_VERSION,
        "opaque_case_id": contract.opaque_case_id,
        "time_contract_sha256": _digest(
            _dump_exact_model(
                contract,
                model_type=CaseTimeContract,
                label="case time contract",
            )
        ),
        "anomaly_id": anomaly_id,
        "service_key": service_key,
        "t_low_unix_ns": t_low_unix_ns,
        "t_high_unix_ns": t_high_unix_ns,
        "provenance_ids": tuple(sorted(set(provenance_ids))),
    }
    digest = _digest(unsigned)
    return OnsetInterval.model_validate(
        {**unsigned, "onset_id": f"onset_{digest[:24]}", "onset_sha256": digest}
    )


class TemporalPrecedence(StrictFrozenModel):
    """Three-valued clock-aware order between two anomaly onsets."""

    schema_version: Literal["nurse.v2.temporal-precedence/3"] = TEMPORAL_PRECEDENCE_SCHEMA_VERSION
    precedence_id: str = Field(pattern=r"^precedence_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    time_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_onset_id: str | None = Field(default=None, pattern=r"^onset_[0-9a-f]{24}$")
    target_onset_id: str | None = Field(default=None, pattern=r"^onset_[0-9a-f]{24}$")
    clock_uncertainty_ns: int = Field(ge=0)
    clock_resolution_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    clock_uncertainty_resolved: bool
    source_effective_low_unix_ns: int | None = Field(default=None, ge=0)
    source_effective_high_unix_ns: int | None = Field(default=None, ge=0)
    target_effective_low_unix_ns: int | None = Field(default=None, ge=0)
    target_effective_high_unix_ns: int | None = Field(default=None, ge=0)
    truth_value: Literal["true", "unknown"]
    grants_explicit_support: bool
    reason: TemporalPrecedenceReason
    provenance_ids: tuple[str, ...]
    precedence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            validated = _trusted_exact_model(
                self,
                model_type=TemporalPrecedence,
                label="temporal precedence",
            )
            effective_bounds = (
                validated.source_effective_low_unix_ns,
                validated.source_effective_high_unix_ns,
                validated.target_effective_low_unix_ns,
                validated.target_effective_high_unix_ns,
            )
            if validated.reason == "missing_onset":
                if (
                    validated.source_onset_id is not None and validated.target_onset_id is not None
                ) or any(value is not None for value in effective_bounds):
                    return False
            elif validated.reason == "unresolved_clock_uncertainty":
                if (
                    validated.source_onset_id is None
                    or validated.target_onset_id is None
                    or validated.clock_uncertainty_resolved
                    or any(value is not None for value in effective_bounds)
                ):
                    return False
            elif validated.reason == "onset_outside_time_contract":
                if (
                    validated.source_onset_id is None
                    or validated.target_onset_id is None
                    or not validated.clock_uncertainty_resolved
                    or any(value is not None for value in effective_bounds)
                ):
                    return False
            else:
                if (
                    validated.source_onset_id is None
                    or validated.target_onset_id is None
                    or not validated.clock_uncertainty_resolved
                    or any(value is None for value in effective_bounds)
                ):
                    return False
                source_low = cast(int, effective_bounds[0])
                source_high = cast(int, effective_bounds[1])
                target_low = cast(int, effective_bounds[2])
                target_high = cast(int, effective_bounds[3])
                if source_low > source_high or target_low > target_high:
                    return False
                strict = source_high < target_low
                if strict != (validated.reason == "strict_precedence"):
                    return False
            if validated.truth_value != (
                "true" if validated.reason == "strict_precedence" else "unknown"
            ):
                return False
            if validated.provenance_ids != tuple(sorted(set(validated.provenance_ids))):
                return False
            unsigned = _dump_exact_model(
                validated,
                model_type=TemporalPrecedence,
                label="temporal precedence",
                exclude={"precedence_id", "precedence_sha256"},
            )
            digest = _digest(unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            validated == self
            and self.precedence_id == f"precedence_{digest[:24]}"
            and self.precedence_sha256 == digest
            and self.grants_explicit_support is (self.truth_value == "true")
        )

    def verify_inputs(
        self,
        *,
        source_onset: OnsetInterval | None,
        target_onset: OnsetInterval | None,
        time_contract: CaseTimeContract,
        clock_resolution: FrozenClockResolution,
        expected_clock_resolution_sha256: str,
    ) -> bool:
        """Replay this decision from exact onset and clock inputs."""

        try:
            artifact = _trusted_exact_model(
                self,
                model_type=TemporalPrecedence,
                label="temporal precedence",
            )
            replayed = evaluate_clock_aware_precedence(
                source_onset=source_onset,
                target_onset=target_onset,
                time_contract=time_contract,
                clock_resolution=clock_resolution,
                expected_clock_resolution_sha256=expected_clock_resolution_sha256,
            )
            return TemporalPrecedence.verify(artifact) and _dump_exact_model(
                replayed,
                model_type=TemporalPrecedence,
                label="temporal precedence",
            ) == _dump_exact_model(
                artifact,
                model_type=TemporalPrecedence,
                label="temporal precedence",
            )
        except (AttributeError, TypeError, ValueError):
            return False


def evaluate_clock_aware_precedence(
    *,
    source_onset: OnsetInterval | None,
    target_onset: OnsetInterval | None,
    time_contract: CaseTimeContract,
    clock_resolution: FrozenClockResolution,
    expected_clock_resolution_sha256: str,
) -> TemporalPrecedence:
    contract = _trusted_exact_model(
        time_contract,
        model_type=CaseTimeContract,
        label="case time contract",
    )
    resolution = require_frozen_clock_resolution(
        clock_resolution=clock_resolution,
        time_contract=contract,
        expected_clock_resolution_sha256=expected_clock_resolution_sha256,
    )
    source = (
        None
        if source_onset is None
        else _trusted_exact_model(
            source_onset,
            model_type=OnsetInterval,
            label="source onset",
        )
    )
    target = (
        None
        if target_onset is None
        else _trusted_exact_model(
            target_onset,
            model_type=OnsetInterval,
            label="target onset",
        )
    )
    if source is not None and source.opaque_case_id != contract.opaque_case_id:
        raise ValueError("source onset belongs to another case")
    if target is not None and target.opaque_case_id != contract.opaque_case_id:
        raise ValueError("target onset belongs to another case")
    contract_sha256 = _digest(
        _dump_exact_model(
            contract,
            model_type=CaseTimeContract,
            label="case time contract",
        )
    )
    clock_uncertainty_resolved = resolution.resolution_status == "resolved"
    if source is not None and source.time_contract_sha256 != contract_sha256:
        raise ValueError("source onset belongs to another time contract")
    if target is not None and target.time_contract_sha256 != contract_sha256:
        raise ValueError("target onset belongs to another time contract")
    uncertainty = contract.clock_uncertainty_ns
    effective_bounds: tuple[int | None, int | None, int | None, int | None]
    if source is None or target is None:
        truth_value: Literal["true", "unknown"] = "unknown"
        reason: TemporalPrecedenceReason = "missing_onset"
        effective_bounds = (None, None, None, None)
    elif not clock_uncertainty_resolved:
        truth_value = "unknown"
        reason = "unresolved_clock_uncertainty"
        effective_bounds = (None, None, None, None)
    elif any(
        onset.t_low_unix_ns < contract.baseline_window.start_unix_ns
        or onset.t_high_unix_ns > contract.decision_time_unix_ns
        for onset in (source, target)
    ):
        truth_value = "unknown"
        reason = "onset_outside_time_contract"
        effective_bounds = (None, None, None, None)
    else:
        source_low = max(0, source.t_low_unix_ns - uncertainty)
        source_high = source.t_high_unix_ns + uncertainty
        target_low = max(0, target.t_low_unix_ns - uncertainty)
        target_high = target.t_high_unix_ns + uncertainty
        effective_bounds = (source_low, source_high, target_low, target_high)
        if source_high < target_low:
            truth_value = "true"
            reason = "strict_precedence"
        else:
            truth_value = "unknown"
            reason = "equal_or_overlapping"
    provenance_ids = tuple(
        sorted(
            {
                value
                for onset in (source, target)
                if onset is not None
                for value in onset.provenance_ids
            }
            | set(resolution.provenance_ids)
            | {resolution.resolution_sha256}
        )
    )
    unsigned: dict[str, object] = {
        "schema_version": TEMPORAL_PRECEDENCE_SCHEMA_VERSION,
        "opaque_case_id": contract.opaque_case_id,
        "time_contract_sha256": contract_sha256,
        "source_onset_id": source.onset_id if source is not None else None,
        "target_onset_id": target.onset_id if target is not None else None,
        "clock_uncertainty_ns": uncertainty,
        "clock_resolution_sha256": resolution.resolution_sha256,
        "clock_uncertainty_resolved": clock_uncertainty_resolved,
        "source_effective_low_unix_ns": effective_bounds[0],
        "source_effective_high_unix_ns": effective_bounds[1],
        "target_effective_low_unix_ns": effective_bounds[2],
        "target_effective_high_unix_ns": effective_bounds[3],
        "truth_value": truth_value,
        "grants_explicit_support": truth_value == "true",
        "reason": reason,
        "provenance_ids": provenance_ids,
    }
    digest = _digest(unsigned)
    return TemporalPrecedence.model_validate(
        {
            **unsigned,
            "precedence_id": f"precedence_{digest[:24]}",
            "precedence_sha256": digest,
        }
    )


class RelationSCCComponent(StrictFrozenModel):
    """One relation-specific component with no invented internal source."""

    schema_version: Literal["nurse.v2.relation-scc-component/1"] = (
        RELATION_SCC_COMPONENT_SCHEMA_VERSION
    )
    component_id: str = Field(pattern=r"^rscc_[0-9a-f]{24}$")
    relation_type: DirectedPropagationRelation
    member_service_keys: tuple[str, ...] = Field(min_length=1)
    internal_edge_ids: tuple[str, ...]
    internal_provenance_ids: tuple[str, ...]
    internal_order: Literal["unknown"] = "unknown"
    internal_source_service_key: None = None
    component_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_component(self) -> Self:
        if self.member_service_keys != tuple(sorted(set(self.member_service_keys))):
            raise ValueError("relation SCC members must be sorted and unique")
        if self.internal_edge_ids != tuple(sorted(set(self.internal_edge_ids))):
            raise ValueError("relation SCC internal edge IDs must be sorted and unique")
        if self.internal_provenance_ids != tuple(sorted(set(self.internal_provenance_ids))):
            raise ValueError("relation SCC provenance IDs must be sorted and unique")
        unsigned = _dump_exact_model(
            self,
            model_type=RelationSCCComponent,
            label="relation SCC component",
            exclude={"component_id", "component_sha256"},
        )
        digest = _digest(unsigned)
        if self.component_id != f"rscc_{digest[:24]}" or self.component_sha256 != digest:
            raise ValueError("relation SCC component is not content-addressed")
        return self


class RelationCondensedEdge(StrictFrozenModel):
    """One DAG edge retaining every same-pair original edge and provenance."""

    schema_version: Literal["nurse.v2.relation-condensed-edge/1"] = (
        RELATION_CONDENSED_EDGE_SCHEMA_VERSION
    )
    condensed_edge_id: str = Field(pattern=r"^rcedge_[0-9a-f]{24}$")
    relation_type: DirectedPropagationRelation
    source_component_id: str = Field(pattern=r"^rscc_[0-9a-f]{24}$")
    target_component_id: str = Field(pattern=r"^rscc_[0-9a-f]{24}$")
    original_edge_ids: tuple[str, ...] = Field(min_length=1)
    original_provenance_ids: tuple[str, ...] = Field(min_length=1)
    condensed_edge_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_condensed_edge(self) -> Self:
        if self.source_component_id == self.target_component_id:
            raise ValueError("condensed edge cannot be internal")
        if self.original_edge_ids != tuple(sorted(set(self.original_edge_ids))):
            raise ValueError("condensed original edge IDs must be sorted and unique")
        if self.original_provenance_ids != tuple(sorted(set(self.original_provenance_ids))):
            raise ValueError("condensed provenance IDs must be sorted and unique")
        unsigned = _dump_exact_model(
            self,
            model_type=RelationCondensedEdge,
            label="relation condensed edge",
            exclude={"condensed_edge_id", "condensed_edge_sha256"},
        )
        digest = _digest(unsigned)
        if (
            self.condensed_edge_id != f"rcedge_{digest[:24]}"
            or self.condensed_edge_sha256 != digest
        ):
            raise ValueError("relation condensed edge is not content-addressed")
        return self


class RelationSCCCondensation(StrictFrozenModel):
    """Replayable relation-specific condensation over a current graph snapshot."""

    schema_version: Literal["nurse.v2.relation-scc-condensation/1"] = (
        RELATION_SCC_CONDENSATION_SCHEMA_VERSION
    )
    condensation_id: str = Field(pattern=r"^rcond_[0-9a-f]{24}$")
    graph_schema_version: Literal["nurse.v2.graph-snapshot/4"] = GRAPH_SNAPSHOT_SCHEMA_VERSION
    graph_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    relation_type: DirectedPropagationRelation
    node_service_keys: tuple[str, ...] = Field(min_length=1)
    selected_edge_ids: tuple[str, ...]
    selected_original_edges: tuple[GraphSnapshotEdge, ...]
    components: tuple[RelationSCCComponent, ...] = Field(min_length=1)
    condensed_edges: tuple[RelationCondensedEdge, ...]
    retained_original_edge_ids: tuple[str, ...]
    retained_provenance_ids: tuple[str, ...]
    arbitrary_edge_deletion: Literal[False] = False
    internal_source_invented: Literal[False] = False
    condensation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            validated = _trusted_exact_model(
                self,
                model_type=RelationSCCCondensation,
                label="relation SCC condensation",
            )
            self = validated
            if self.node_service_keys != tuple(sorted(set(self.node_service_keys))):
                return False
            if self.selected_edge_ids != tuple(sorted(set(self.selected_edge_ids))):
                return False
            if tuple(item.edge_id for item in self.selected_original_edges) != (
                self.selected_edge_ids
            ):
                return False
            if self.components != tuple(
                sorted(self.components, key=lambda item: item.member_service_keys)
            ):
                return False
            if any(component.relation_type != self.relation_type for component in self.components):
                return False
            members = tuple(
                member for component in self.components for member in component.member_service_keys
            )
            if len(members) != len(set(members)) or set(members) != set(self.node_service_keys):
                return False
            if self.condensed_edges != tuple(
                sorted(
                    self.condensed_edges,
                    key=lambda item: (
                        item.source_component_id,
                        item.target_component_id,
                    ),
                )
            ):
                return False
            component_by_id = {component.component_id: component for component in self.components}
            if len(component_by_id) != len(self.components):
                return False
            component_id_by_node = {
                member: component.component_id
                for component in self.components
                for member in component.member_service_keys
            }
            selected_by_id = {edge.edge_id: edge for edge in self.selected_original_edges}
            if len(selected_by_id) != len(self.selected_original_edges):
                return False
            if any(
                edge.relation_type != self.relation_type
                or edge.observation_state != "observed"
                or not edge.allows_directed_propagation
                or not edge.verify()
                for edge in self.selected_original_edges
            ):
                return False
            classified_edge_ids = tuple(
                sorted(
                    edge_id
                    for component in self.components
                    for edge_id in component.internal_edge_ids
                )
            ) + tuple(
                sorted(
                    edge_id for edge in self.condensed_edges for edge_id in edge.original_edge_ids
                )
            )
            retained = tuple(sorted(classified_edge_ids))
            if retained != self.retained_original_edge_ids or retained != self.selected_edge_ids:
                return False
            expected_internal_by_component: dict[str, list[GraphSnapshotEdge]] = {
                component_id: [] for component_id in component_by_id
            }
            expected_condensed_by_pair: dict[tuple[str, str], list[GraphSnapshotEdge]] = {}
            for edge in self.selected_original_edges:
                source, target = _directed_endpoints(edge)
                source_component_id = component_id_by_node.get(source)
                target_component_id = component_id_by_node.get(target)
                if source_component_id is None or target_component_id is None:
                    return False
                if source_component_id == target_component_id:
                    expected_internal_by_component[source_component_id].append(edge)
                else:
                    expected_condensed_by_pair.setdefault(
                        (source_component_id, target_component_id), []
                    ).append(edge)
            expected_member_partition = _strongly_connected_components(
                nodes=self.node_service_keys,
                edges=self.selected_original_edges,
            )
            if expected_member_partition != tuple(
                component.member_service_keys for component in self.components
            ):
                return False
            for component_id, component in component_by_id.items():
                expected_edges = expected_internal_by_component[component_id]
                if component.internal_edge_ids != tuple(
                    sorted(edge.edge_id for edge in expected_edges)
                ):
                    return False
                if component.internal_provenance_ids != tuple(
                    sorted({value for edge in expected_edges for value in edge.provenance_ids})
                ):
                    return False
            if len(
                {
                    (edge.source_component_id, edge.target_component_id)
                    for edge in self.condensed_edges
                }
            ) != len(self.condensed_edges):
                return False
            actual_pairs: set[tuple[str, str]] = set()
            for condensed_edge in self.condensed_edges:
                if condensed_edge.relation_type != self.relation_type:
                    return False
                pair = (
                    condensed_edge.source_component_id,
                    condensed_edge.target_component_id,
                )
                original_edges = expected_condensed_by_pair.get(pair)
                if original_edges is None:
                    return False
                actual_pairs.add(pair)
                if condensed_edge.original_edge_ids != tuple(
                    sorted(original.edge_id for original in original_edges)
                ):
                    return False
                if condensed_edge.original_provenance_ids != tuple(
                    sorted(
                        {value for original in original_edges for value in original.provenance_ids}
                    )
                ):
                    return False
            if actual_pairs != set(expected_condensed_by_pair):
                return False
            adjacency: dict[str, set[str]] = {
                component_id: set() for component_id in component_by_id
            }
            indegree = {component_id: 0 for component_id in component_by_id}
            for source_component_id, target_component_id in actual_pairs:
                adjacency[source_component_id].add(target_component_id)
                indegree[target_component_id] += 1
            ready = deque(
                sorted(component_id for component_id, degree in indegree.items() if degree == 0)
            )
            visited_count = 0
            while ready:
                component_id = ready.popleft()
                visited_count += 1
                for target_component_id in sorted(adjacency[component_id]):
                    indegree[target_component_id] -= 1
                    if indegree[target_component_id] == 0:
                        ready.append(target_component_id)
            if visited_count != len(component_by_id):
                return False
            expected_provenance = tuple(
                sorted(
                    {
                        value
                        for edge in self.selected_original_edges
                        for value in edge.provenance_ids
                    }
                )
            )
            if expected_provenance != self.retained_provenance_ids:
                return False
            identity = _dump_exact_model(
                self,
                model_type=RelationSCCCondensation,
                label="relation SCC condensation",
                exclude={"condensation_id", "condensation_sha256"},
            )
            unsigned = _dump_exact_model(
                self,
                model_type=RelationSCCCondensation,
                label="relation SCC condensation",
                exclude={"condensation_sha256"},
            )
            digest = _digest(unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            self.condensation_id == f"rcond_{_digest(identity)[:24]}"
            and self.condensation_sha256 == digest
        )

    def verify_inputs(
        self,
        *,
        graph_snapshot: GraphSnapshot,
        selected_edge_ids: Sequence[str] | None = None,
    ) -> bool:
        """Replay this artifact from one exact current graph snapshot."""

        try:
            artifact = _trusted_exact_model(
                self,
                model_type=RelationSCCCondensation,
                label="relation SCC condensation",
            )
            graph = _trusted_graph_snapshot(graph_snapshot)
            if not RelationSCCCondensation.verify(artifact):
                return False
            replayed = build_relation_scc_condensation(
                graph_snapshot=graph,
                relation_type=artifact.relation_type,
                selected_edge_ids=selected_edge_ids,
            )
            return _dump_exact_model(
                replayed,
                model_type=RelationSCCCondensation,
                label="relation SCC condensation",
            ) == _dump_exact_model(
                artifact,
                model_type=RelationSCCCondensation,
                label="relation SCC condensation",
            )
        except (AttributeError, TypeError, ValueError):
            return False


def _build_component(
    *,
    relation_type: DirectedPropagationRelation,
    member_service_keys: Sequence[str],
    internal_edges: Sequence[GraphSnapshotEdge],
) -> RelationSCCComponent:
    ordered_members = tuple(sorted(set(member_service_keys)))
    ordered_edges = tuple(sorted(internal_edges, key=lambda item: item.edge_id))
    unsigned: dict[str, object] = {
        "schema_version": RELATION_SCC_COMPONENT_SCHEMA_VERSION,
        "relation_type": relation_type,
        "member_service_keys": ordered_members,
        "internal_edge_ids": tuple(item.edge_id for item in ordered_edges),
        "internal_provenance_ids": tuple(
            sorted({value for item in ordered_edges for value in item.provenance_ids})
        ),
        "internal_order": "unknown",
        "internal_source_service_key": None,
    }
    component_digest = _digest(unsigned)
    return RelationSCCComponent.model_validate(
        {
            **unsigned,
            "component_id": f"rscc_{component_digest[:24]}",
            "component_sha256": component_digest,
        }
    )


def _directed_endpoints(edge: GraphSnapshotEdge) -> tuple[str, str]:
    if edge.relation_type == "caller_load_transmission" and edge.direction == "caller_to_callee":
        return edge.caller_service_key, edge.callee_service_key
    if edge.relation_type == "received_dependency_failure" and edge.direction == "callee_to_caller":
        return edge.callee_service_key, edge.caller_service_key
    raise ValueError("propagation relation direction does not match its relation type")


def _build_condensed_edge(
    *,
    relation_type: DirectedPropagationRelation,
    source_component_id: str,
    target_component_id: str,
    original_edges: Sequence[GraphSnapshotEdge],
) -> RelationCondensedEdge:
    ordered_edges = tuple(sorted(original_edges, key=lambda item: item.edge_id))
    unsigned: dict[str, object] = {
        "schema_version": RELATION_CONDENSED_EDGE_SCHEMA_VERSION,
        "relation_type": relation_type,
        "source_component_id": source_component_id,
        "target_component_id": target_component_id,
        "original_edge_ids": tuple(item.edge_id for item in ordered_edges),
        "original_provenance_ids": tuple(
            sorted({value for item in ordered_edges for value in item.provenance_ids})
        ),
    }
    digest = _digest(unsigned)
    return RelationCondensedEdge.model_validate(
        {
            **unsigned,
            "condensed_edge_id": f"rcedge_{digest[:24]}",
            "condensed_edge_sha256": digest,
        }
    )


def _build_condensation_artifact(
    *,
    graph_snapshot: GraphSnapshot,
    relation_type: DirectedPropagationRelation,
    selected_edges: Sequence[GraphSnapshotEdge],
    components: Sequence[RelationSCCComponent],
    condensed_edges: Sequence[RelationCondensedEdge],
) -> RelationSCCCondensation:
    ordered_selected = tuple(sorted(selected_edges, key=lambda item: item.edge_id))
    ordered_components = tuple(sorted(components, key=lambda item: item.member_service_keys))
    ordered_condensed = tuple(
        sorted(
            condensed_edges,
            key=lambda item: (item.source_component_id, item.target_component_id),
        )
    )
    selected_edge_ids = tuple(item.edge_id for item in ordered_selected)
    identity: dict[str, object] = {
        "schema_version": RELATION_SCC_CONDENSATION_SCHEMA_VERSION,
        "graph_schema_version": graph_snapshot.schema_version,
        "graph_snapshot_sha256": graph_snapshot.snapshot_sha256,
        "opaque_case_id": graph_snapshot.opaque_case_id,
        "relation_type": relation_type,
        "node_service_keys": graph_snapshot.node_service_keys,
        "selected_edge_ids": selected_edge_ids,
        "selected_original_edges": tuple(
            _dump_exact_model(
                item,
                model_type=GraphSnapshotEdge,
                label="selected graph edge",
            )
            for item in ordered_selected
        ),
        "components": tuple(
            _dump_exact_model(
                item,
                model_type=RelationSCCComponent,
                label="relation SCC component",
            )
            for item in ordered_components
        ),
        "condensed_edges": tuple(
            _dump_exact_model(
                item,
                model_type=RelationCondensedEdge,
                label="relation condensed edge",
            )
            for item in ordered_condensed
        ),
        "retained_original_edge_ids": selected_edge_ids,
        "retained_provenance_ids": tuple(
            sorted({value for item in ordered_selected for value in item.provenance_ids})
        ),
        "arbitrary_edge_deletion": False,
        "internal_source_invented": False,
    }
    condensation_id = f"rcond_{_digest(identity)[:24]}"
    unsigned = {**identity, "condensation_id": condensation_id}
    return RelationSCCCondensation.model_validate(
        {**unsigned, "condensation_sha256": _digest(unsigned)}
    )


def _strongly_connected_components(
    *,
    nodes: Sequence[str],
    edges: Sequence[GraphSnapshotEdge],
) -> tuple[tuple[str, ...], ...]:
    """Return deterministic SCC members with an iterative Kosaraju traversal."""
    return _scc_from_endpoints(nodes=nodes, endpoints=tuple(_directed_endpoints(e) for e in edges))


def _scc_from_endpoints(
    *, nodes: Sequence[str], endpoints: Sequence[tuple[str, str]]
) -> tuple[tuple[str, ...], ...]:
    """Pure graph arithmetic; endpoint construction carries no relation authority."""

    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    reverse: dict[str, set[str]] = {node: set() for node in nodes}
    for source, target in endpoints:
        adjacency[source].add(target)
        reverse[target].add(source)
    ordered_adjacency = {node: tuple(sorted(targets)) for node, targets in adjacency.items()}
    finish_order: list[str] = []
    visited: set[str] = set()
    for root in sorted(nodes):
        if root in visited:
            continue
        visited.add(root)
        stack: list[tuple[str, int]] = [(root, 0)]
        while stack:
            node, next_index = stack[-1]
            neighbors = ordered_adjacency[node]
            if next_index == len(neighbors):
                finish_order.append(node)
                stack.pop()
                continue
            neighbor = neighbors[next_index]
            stack[-1] = (node, next_index + 1)
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, 0))
    components: list[tuple[str, ...]] = []
    assigned: set[str] = set()
    for root in reversed(finish_order):
        if root in assigned:
            continue
        members: list[str] = []
        assigned.add(root)
        stack = [(root, 0)]
        while stack:
            node, _ = stack.pop()
            members.append(node)
            for neighbor in sorted(reverse[node], reverse=True):
                if neighbor not in assigned:
                    assigned.add(neighbor)
                    stack.append((neighbor, 0))
        components.append(tuple(sorted(members)))
    return tuple(sorted(components))


def build_relation_scc_condensation(
    *,
    graph_snapshot: GraphSnapshot,
    relation_type: GraphRelationType,
    selected_edge_ids: Sequence[str] | None = None,
) -> RelationSCCCondensation:
    if type(relation_type) is not str:
        raise ValueError("relation type must use an exact relation string")
    if relation_type not in {"received_dependency_failure", "caller_load_transmission"}:
        raise ValueError("relation does not allow directed propagation")
    if selected_edge_ids is not None:
        if type(selected_edge_ids) not in {list, tuple}:
            raise ValueError("selected edge IDs must use an exact list or tuple")
        if type(selected_edge_ids) is list:
            selected_edge_count = list.__len__(cast(list[str], selected_edge_ids))
        else:
            selected_edge_count = tuple.__len__(cast(tuple[str, ...], selected_edge_ids))
        if selected_edge_count > _MAX_CONDENSATION_SELECTED_EDGES:
            raise _TemporalResourceLimitError("condensation input exceeds the selected edge limit")
        _preflight_pydantic_serialization(
            selected_edge_ids,
            label="selected edge IDs",
        )
    try:
        graph = _trusted_graph_snapshot(graph_snapshot)
    except _TemporalResourceLimitError:
        raise
    except (AttributeError, TypeError, ValueError):
        raise ValueError("condensation requires an exact verified GraphSnapshot/4") from None
    directed_relation = cast(DirectedPropagationRelation, relation_type)
    eligible = tuple(
        edge
        for edge in graph.edges
        if edge.relation_type == directed_relation
        and edge.observation_state == "observed"
        and edge.allows_directed_propagation
    )
    if selected_edge_ids is None:
        selected = tuple(sorted(eligible, key=lambda item: item.edge_id))
    else:
        graph_edges_by_id = {edge.edge_id: edge for edge in graph.edges}
        selected_values = set(selected_edge_ids)
        selected_members: list[GraphSnapshotEdge] = []
        for edge_id in sorted(selected_values):
            edge = graph_edges_by_id.get(edge_id)
            if edge is None:
                raise ValueError("selected edge does not belong to the graph snapshot")
            if edge.relation_type != directed_relation:
                raise ValueError("selected edge belongs to another relation")
            if edge.observation_state != "observed":
                raise ValueError("selected edge is not an observed accepted graph edge")
            if not edge.allows_directed_propagation:
                raise ValueError("selected edge does not allow directed propagation")
            selected_members.append(edge)
        selected = tuple(selected_members)
        eligible_edge_ids = tuple(sorted(edge.edge_id for edge in eligible))
        if tuple(edge.edge_id for edge in selected) != eligible_edge_ids:
            raise ValueError("selected edge IDs must cover the complete eligible relation edge set")
    member_groups = _strongly_connected_components(
        nodes=graph.node_service_keys,
        edges=selected,
    )
    member_group_by_node = {member: group for group in member_groups for member in group}
    internal_edges_by_group: dict[tuple[str, ...], list[GraphSnapshotEdge]] = {
        group: [] for group in member_groups
    }
    for edge in selected:
        source, target = _directed_endpoints(edge)
        source_group = member_group_by_node[source]
        if source_group == member_group_by_node[target]:
            internal_edges_by_group[source_group].append(edge)
    components = tuple(
        _build_component(
            relation_type=directed_relation,
            member_service_keys=group,
            internal_edges=internal_edges_by_group[group],
        )
        for group in member_groups
    )
    component_id_by_node = {
        member: component.component_id
        for component in components
        for member in component.member_service_keys
    }
    edges_by_component_pair: dict[tuple[str, str], list[GraphSnapshotEdge]] = {}
    for edge in selected:
        source, target = _directed_endpoints(edge)
        pair = (component_id_by_node[source], component_id_by_node[target])
        if pair[0] != pair[1]:
            edges_by_component_pair.setdefault(pair, []).append(edge)
    condensed_edges = tuple(
        _build_condensed_edge(
            relation_type=directed_relation,
            source_component_id=source_component_id,
            target_component_id=target_component_id,
            original_edges=original_edges,
        )
        for (source_component_id, target_component_id), original_edges in sorted(
            edges_by_component_pair.items()
        )
    )
    return _build_condensation_artifact(
        graph_snapshot=graph,
        relation_type=directed_relation,
        selected_edges=selected,
        components=components,
        condensed_edges=condensed_edges,
    )
