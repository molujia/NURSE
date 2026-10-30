"""Replay-gated normal/fault graph snapshot construction."""

from __future__ import annotations

import hmac
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, TypeVar, cast

from pydantic import Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.event_grouping import (
    ASSOCIATION_NORMALIZATION_SEMANTICS,
    AdapterAssociationSourceField,
)
from nurse_rcl.v2.observation_schemas import (
    CandidateUniverse,
    CaseTimeContract,
    TimeInterval,
)
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

GRAPH_ADAPTER_PARENT_RECORD_SCHEMA_VERSION: Literal["nurse.v2.graph-adapter-parent-record/1"] = (
    "nurse.v2.graph-adapter-parent-record/1"
)
FROZEN_GRAPH_INPUT_MANIFEST_SCHEMA_VERSION: Literal["nurse.v2.frozen-graph-input-manifest/1"] = (
    "nurse.v2.frozen-graph-input-manifest/1"
)
CANONICAL_GRAPH_SOURCE_INPUT_SCHEMA_VERSION: Literal["nurse.v2.canonical-graph-source-input/2"] = (
    "nurse.v2.canonical-graph-source-input/2"
)
CANONICAL_GRAPH_COVERAGE_INPUT_SCHEMA_VERSION: Literal[
    "nurse.v2.canonical-graph-coverage-input/2"
] = "nurse.v2.canonical-graph-coverage-input/2"
GRAPH_SOURCE_RECORD_SCHEMA_VERSION: Literal["nurse.v2.graph-source-record/4"] = (
    "nurse.v2.graph-source-record/4"
)
GRAPH_WINDOW_COVERAGE_SCHEMA_VERSION: Literal["nurse.v2.graph-window-coverage/4"] = (
    "nurse.v2.graph-window-coverage/4"
)
GRAPH_SNAPSHOT_SCHEMA_VERSION: Literal["nurse.v2.graph-snapshot/4"] = "nurse.v2.graph-snapshot/4"
GRAPH_SNAPSHOT_PAIR_SCHEMA_VERSION: Literal["nurse.v2.graph-snapshot-pair/5"] = (
    "nurse.v2.graph-snapshot-pair/5"
)
GRAPH_INPUT_DECISION_SCHEMA_VERSION: Literal["nurse.v2.graph-input-decision/2"] = (
    "nurse.v2.graph-input-decision/2"
)

GraphRelationType = Literal[
    "service_call",
    "received_dependency_failure",
    "caller_load_transmission",
    "shared_resource_association",
]
GraphDirection = Literal[
    "caller_to_callee",
    "callee_to_caller",
    "undirected_association",
]
WindowKind = Literal["normal", "fault"]
GraphObservationState = Literal["observed", "no_record", "unknown"]
MetricState = Literal["observed", "zero", "verified_normal", "no_record", "unknown"]
EdgeObservationState = Literal["observed", "not_observed", "unknown"]
CoverageState = Literal["sufficient", "insufficient", "unknown"]
EdgeChangeState = Literal["new", "removed", "increased", "decreased", "stable", "unknown"]
MetricChangeState = Literal["increased", "decreased", "stable", "unknown"]
EdgePresence = Literal[
    "observed",
    "not_observed_sufficient_coverage",
    "not_observed_insufficient_coverage",
    "unknown",
]
GraphInputDisposition = Literal["accepted", "rejected", "unknown"]
GraphInputReason = Literal[
    "accepted",
    "unknown_observation",
    "case_mismatch",
    "observation_contract_mismatch",
    "window_interval_mismatch",
    "post_decision",
    "endpoint_outside_frozen_candidates",
    "endpoint_outside_entity_registry",
    "conflicting_duplicate_source",
]
GraphCollectionState = Literal["observed", "no_record", "unknown"]
GraphParentArtifactKind = Literal["call_measurement", "coverage_samples"]

MAX_EXACT_GRAPH_COUNT = (1 << 53) - 1
MAX_COVERAGE_SAMPLE_COUNT = (1 << 64) - 1

GRAPH_ADAPTER_SEMANTICS_VERSION: Literal["nurse-v2-graph-adapter/1"] = "nurse-v2-graph-adapter/1"
GRAPH_DERIVATION_SEMANTICS_VERSION: Literal["nurse-v2-graph-derivation/2"] = (
    "nurse-v2-graph-derivation/2"
)


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}_{_digest(value)[:24]}"


def _canonical_finite_float(value: object) -> float | None:
    """Return the sole content-addressed representation of a finite float."""

    if value is None:
        return None
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise ValueError("content-addressed graph floats must be int or float values")
    numeric_value = cast(int | float, value)
    try:
        canonical = float(numeric_value)
    except (OverflowError, TypeError) as error:
        raise ValueError("content-addressed graph float cannot be represented") from error
    if not math.isfinite(canonical):
        raise ValueError("content-addressed graph floats must be finite")
    return 0.0 if canonical == 0.0 else canonical


def _canonical_graph_count(value: object, *, field_name: str) -> int | None:
    """Keep graph counts exact through their authoritative float64 summary."""

    if value is None:
        return None
    if isinstance(value, bool) or type(value) is not int:
        raise ValueError(f"{field_name} must be an exact integer graph count")
    count = value
    if not 0 <= count <= MAX_EXACT_GRAPH_COUNT:
        raise ValueError(
            f"{field_name} exceeds the float64 exact integer limit of {MAX_EXACT_GRAPH_COUNT}"
        )
    return count


def _canonical_coverage_sample_count(
    value: object,
    *,
    field_name: str,
    minimum: int,
) -> int | None:
    """Preserve exact adapter coverage counts before validation and hashing."""

    if value is None:
        return None
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field_name} must be an exact integer coverage count at least {minimum}")
    if value > MAX_COVERAGE_SAMPLE_COUNT:
        raise ValueError(
            f"{field_name} exceeds the uint64 coverage count limit of {MAX_COVERAGE_SAMPLE_COUNT}"
        )
    return value


def _coverage_fraction_from_counts(
    *,
    collection_state: GraphCollectionState,
    valid_sample_count: int | None,
    expected_sample_count: int | None,
) -> float | None:
    _require_graph_coverage(
        collection_state=collection_state,
        valid_sample_count=valid_sample_count,
        expected_sample_count=expected_sample_count,
    )
    if collection_state == "unknown":
        return None
    if collection_state == "no_record":
        return 0.0
    return cast(int, valid_sample_count) / cast(int, expected_sample_count)


def _coverage_meets_minimum(
    *,
    valid_sample_count: int,
    expected_sample_count: int,
    minimum_coverage_fraction: float,
) -> bool:
    threshold_numerator, threshold_denominator = Decimal(
        str(minimum_coverage_fraction)
    ).as_integer_ratio()
    return valid_sample_count * threshold_denominator >= (
        expected_sample_count * threshold_numerator
    )


def _has_instance_method_shadow(value: object, *method_names: str) -> bool:
    """Detect methods injected directly onto an otherwise exact model instance."""

    try:
        instance_state = object.__getattribute__(value, "__dict__")
    except AttributeError:
        return False
    if type(instance_state) is not dict:
        return True
    return any(dict.__contains__(instance_state, method_name) for method_name in method_names)


def _expected_direction(relation_type: GraphRelationType) -> GraphDirection:
    if relation_type in {"service_call", "caller_load_transmission"}:
        return "caller_to_callee"
    if relation_type == "received_dependency_failure":
        return "callee_to_caller"
    return "undirected_association"


GRAPH_ADAPTER_SEMANTICS_SHA256 = _digest(
    {
        "version": GRAPH_ADAPTER_SEMANTICS_VERSION,
        "typed_source_fields": "AdapterAssociationSourceField",
        "normalization_semantics": ASSOCIATION_NORMALIZATION_SEMANTICS,
        "source_measurements": (
            "request_count",
            "error_count",
            "latency_p95_seconds",
        ),
        "coverage_measurements": (
            "collection_state",
            "valid_sample_count",
            "expected_sample_count",
        ),
    }
)
GRAPH_DERIVATION_SEMANTICS_SHA256 = _digest(
    {
        "version": GRAPH_DERIVATION_SEMANTICS_VERSION,
        "source_fingerprint": "sha256(canonical typed source identity fields)",
        "source_record_identity": (
            "case+contract+window+source_fingerprint+caller+callee+relation+direction"
        ),
        "provenance": "content-addressed typed provenance fields",
        "coverage_fraction": "valid_sample_count/expected_sample_count",
        "coverage_sample_limit": MAX_COVERAGE_SAMPLE_COUNT,
        "coverage_sufficiency": "decimal threshold ratio compared by integer cross multiplication",
        "normalization_semantics": ASSOCIATION_NORMALIZATION_SEMANTICS,
    }
)

_ModelT = TypeVar("_ModelT", bound=StrictFrozenModel)


def _trusted_exact_model(
    value: object,
    *,
    model_type: type[_ModelT],
    label: str,
    guarded_methods: tuple[str, ...],
) -> _ModelT:
    """Reconstruct one caller model without invoking methods on that instance."""

    if type(value) is not model_type:
        raise ValueError(f"{label} must use the exact model type")
    if _has_instance_method_shadow(value, "model_dump", *guarded_methods):
        raise ValueError(f"{label} contains an instance method shadow")
    document = model_type.model_dump(value, mode="json", warnings="error")
    rebuilt = model_type.model_validate(document)
    if type(rebuilt) is not model_type:
        raise ValueError(f"{label} rebuilt to a non-exact model type")
    if model_type.model_dump(rebuilt, mode="json", warnings="error") != document:
        raise ValueError(f"{label} is not canonical")
    return rebuilt


def _has_same_exact_runtime_shape(value: object, rebuilt: object) -> bool:
    """Compare runtime types recursively without invoking caller equality methods."""

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
            and _has_same_exact_runtime_shape(
                value_mapping[key],
                rebuilt_mapping[key],
            )
            for key in rebuilt_mapping
        )
    return True


def _trusted_time_interval(value: object, *, label: str) -> TimeInterval:
    return _trusted_exact_model(
        value,
        model_type=TimeInterval,
        label=label,
        guarded_methods=(),
    )


def _trusted_case_time_contract(value: object) -> CaseTimeContract:
    if type(value) is CaseTimeContract:
        _trusted_time_interval(
            value.baseline_window,
            label="case time contract baseline time interval",
        )
        _trusted_time_interval(
            value.fault_window,
            label="case time contract fault time interval",
        )
    return _trusted_exact_model(
        value,
        model_type=CaseTimeContract,
        label="case time contract",
        guarded_methods=(),
    )


def _trusted_candidate_universe(value: object) -> CandidateUniverse:
    rebuilt = _trusted_exact_model(
        value,
        model_type=CandidateUniverse,
        label="candidate universe",
        guarded_methods=("verify",),
    )
    if not CandidateUniverse.verify(rebuilt):
        raise ValueError("candidate universe failed verification")
    return rebuilt


def _trusted_frozen_candidate_universe(value: object) -> FrozenCandidateUniverse:
    rebuilt = _trusted_exact_model(
        value,
        model_type=FrozenCandidateUniverse,
        label="frozen candidate universe",
        guarded_methods=("verify",),
    )
    if not FrozenCandidateUniverse.verify(rebuilt):
        raise ValueError("candidate freeze failed verification")
    return rebuilt


def _trusted_entity_registry(value: object) -> TemporalEntityRegistry:
    rebuilt = _trusted_exact_model(
        value,
        model_type=TemporalEntityRegistry,
        label="entity registry",
        guarded_methods=("verify", "_verify_hash_and_structure"),
    )
    if not TemporalEntityRegistry.verify(rebuilt):
        raise ValueError("entity registry failed exact replay")
    return rebuilt


def _validated_source_fields(
    fields: Sequence[AdapterAssociationSourceField],
    *,
    label: str,
) -> tuple[AdapterAssociationSourceField, ...]:
    validated_items = []
    for item in fields:
        if type(item) is not AdapterAssociationSourceField or _has_instance_method_shadow(
            item, "model_dump"
        ):
            raise ValueError(f"{label} must contain exact typed fields")
        validated_items.append(
            AdapterAssociationSourceField.model_validate(
                AdapterAssociationSourceField.model_dump(item, mode="json", warnings="error")
            )
        )
    validated = tuple(validated_items)
    ordered = tuple(sorted(validated, key=lambda item: (item.field_kind, item.raw_value)))
    if not ordered:
        raise ValueError(f"{label} must contain at least one typed field")
    if len({item.field_kind for item in ordered}) != len(ordered):
        raise ValueError(f"{label} field kinds must be unique")
    return ordered


def _source_field_documents(
    fields: Sequence[AdapterAssociationSourceField],
) -> tuple[dict[str, object], ...]:
    return tuple(AdapterAssociationSourceField.model_dump(item, mode="json") for item in fields)


def _derived_source_fingerprint(
    fields: Sequence[AdapterAssociationSourceField],
) -> str:
    return _digest(
        {
            "normalization_semantics": ASSOCIATION_NORMALIZATION_SEMANTICS,
            "source_identity_fields": _source_field_documents(fields),
        }
    )


def _derived_provenance_ids(
    fields: Sequence[AdapterAssociationSourceField],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            _stable_id(
                "gprov",
                {
                    "normalization_semantics": ASSOCIATION_NORMALIZATION_SEMANTICS,
                    "source_field": AdapterAssociationSourceField.model_dump(item, mode="json"),
                },
            )
            for item in fields
        )
    )


def _require_frozen_graph_semantics(
    *, adapter_semantics_sha256: str, derivation_semantics_sha256: str
) -> None:
    if adapter_semantics_sha256 != GRAPH_ADAPTER_SEMANTICS_SHA256:
        raise ValueError("canonical input does not use frozen graph adapter semantics")
    if derivation_semantics_sha256 != GRAPH_DERIVATION_SEMANTICS_SHA256:
        raise ValueError("canonical input does not use frozen graph derivation semantics")


def _require_graph_measurement(
    *,
    caller_service_key: str,
    callee_service_key: str,
    relation_type: GraphRelationType,
    direction: GraphDirection,
    observation_state: GraphObservationState,
    request_count: int | None,
    error_count: int | None,
    latency_p95_seconds: float | None,
) -> None:
    if caller_service_key == callee_service_key:
        raise ValueError("graph source endpoints must be distinct")
    if direction != _expected_direction(relation_type):
        raise ValueError("graph relation direction does not match relation type")
    if relation_type == "shared_resource_association" and caller_service_key > callee_service_key:
        raise ValueError("shared-resource association requires canonical endpoint order")
    if request_count is not None and error_count is not None and error_count > request_count:
        raise ValueError("error count cannot exceed request count")
    numeric = (request_count, error_count, latency_p95_seconds)
    if observation_state != "observed" and any(value is not None for value in numeric):
        raise ValueError("no-record or unknown graph source cannot contain metrics")
    if relation_type == "shared_resource_association" and any(
        value is not None for value in numeric
    ):
        raise ValueError("shared-resource association cannot claim call metrics")


def _require_graph_coverage(
    *,
    collection_state: GraphCollectionState,
    valid_sample_count: int | None,
    expected_sample_count: int | None,
) -> None:
    if collection_state == "unknown":
        if valid_sample_count is not None or expected_sample_count is not None:
            raise ValueError("unknown graph coverage cannot claim sample counts")
    elif valid_sample_count is None or expected_sample_count is None:
        raise ValueError("known graph coverage requires sample counts")
    elif valid_sample_count > expected_sample_count:
        raise ValueError("valid graph coverage samples cannot exceed expected samples")
    elif collection_state == "no_record" and valid_sample_count != 0:
        raise ValueError("no-record graph coverage requires zero valid samples")


class GraphAdapterParentRecord(StrictFrozenModel):
    """One adapter-frozen raw parent artifact bound into a pinned manifest."""

    schema_version: Literal["nurse.v2.graph-adapter-parent-record/1"] = (
        GRAPH_ADAPTER_PARENT_RECORD_SCHEMA_VERSION
    )
    parent_record_id: str = Field(pattern=r"^gpar_[0-9a-f]{24}$")
    parent_selector: str = Field(pattern=r"^gsel_[0-9a-f]{24}$")
    parent_artifact_kind: GraphParentArtifactKind
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    observation_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    window_kind: WindowKind
    interval: TimeInterval
    caller_service_key: str | None = Field(default=None, min_length=1)
    callee_service_key: str | None = Field(default=None, min_length=1)
    relation_type: GraphRelationType | None = None
    direction: GraphDirection | None = None
    observation_state: GraphObservationState | None = None
    request_count: int | None = Field(default=None, ge=0, le=MAX_EXACT_GRAPH_COUNT)
    error_count: int | None = Field(default=None, ge=0, le=MAX_EXACT_GRAPH_COUNT)
    latency_p95_seconds: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    collection_state: GraphCollectionState | None = None
    valid_sample_count: int | None = Field(default=None, ge=0)
    expected_sample_count: int | None = Field(default=None, ge=1)
    source_identity_fields: tuple[AdapterAssociationSourceField, ...] = Field(min_length=1)
    provenance_fields: tuple[AdapterAssociationSourceField, ...] = Field(min_length=1)
    normalization_semantics: Literal["sha256-utf8-nfc/1"] = ASSOCIATION_NORMALIZATION_SEMANTICS
    adapter_semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derivation_semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_artifact_bytes_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("latency_p95_seconds", mode="before")
    @classmethod
    def canonicalize_latency(cls, value: object) -> float | None:
        return _canonical_finite_float(value)

    @field_validator("request_count", "error_count", mode="before")
    @classmethod
    def canonicalize_counts(cls, value: object, info: ValidationInfo) -> int | None:
        return _canonical_graph_count(value, field_name=info.field_name or "graph count")

    @field_validator("valid_sample_count", "expected_sample_count", mode="before")
    @classmethod
    def canonicalize_coverage_sample_counts(cls, value: object, info: ValidationInfo) -> int | None:
        field_name = info.field_name or "coverage sample count"
        minimum = 1 if field_name == "expected_sample_count" else 0
        return _canonical_coverage_sample_count(
            value,
            field_name=field_name,
            minimum=minimum,
        )

    @property
    def derived_source_fingerprint(self) -> str:
        return _derived_source_fingerprint(self.source_identity_fields)

    @property
    def derived_provenance_ids(self) -> tuple[str, ...]:
        return _derived_provenance_ids(self.provenance_fields)

    @property
    def derived_coverage_fraction(self) -> float | None:
        return _coverage_fraction_from_counts(
            collection_state=cast(GraphCollectionState, self.collection_state),
            valid_sample_count=self.valid_sample_count,
            expected_sample_count=self.expected_sample_count,
        )

    @model_validator(mode="after")
    def require_typed_frozen_parent(self) -> Self:
        source_fields = _validated_source_fields(
            self.source_identity_fields, label="graph parent source identity"
        )
        provenance_fields = _validated_source_fields(
            self.provenance_fields, label="graph parent provenance"
        )
        if source_fields != self.source_identity_fields:
            raise ValueError("graph parent source identity fields must be canonically ordered")
        if provenance_fields != self.provenance_fields:
            raise ValueError("graph parent provenance fields must be canonically ordered")
        _require_frozen_graph_semantics(
            adapter_semantics_sha256=self.adapter_semantics_sha256,
            derivation_semantics_sha256=self.derivation_semantics_sha256,
        )
        if self.parent_artifact_kind == "call_measurement":
            if any(
                item is None
                for item in (
                    self.caller_service_key,
                    self.callee_service_key,
                    self.relation_type,
                    self.direction,
                    self.observation_state,
                )
            ):
                raise ValueError("call measurement parent requires typed graph fields")
            if any(
                item is not None
                for item in (
                    self.collection_state,
                    self.valid_sample_count,
                    self.expected_sample_count,
                )
            ):
                raise ValueError("call measurement parent cannot contain coverage samples")
            _require_graph_measurement(
                caller_service_key=cast(str, self.caller_service_key),
                callee_service_key=cast(str, self.callee_service_key),
                relation_type=cast(GraphRelationType, self.relation_type),
                direction=cast(GraphDirection, self.direction),
                observation_state=cast(GraphObservationState, self.observation_state),
                request_count=self.request_count,
                error_count=self.error_count,
                latency_p95_seconds=self.latency_p95_seconds,
            )
        else:
            if self.collection_state is None:
                raise ValueError("coverage parent requires a collection state")
            if any(
                item is not None
                for item in (
                    self.caller_service_key,
                    self.callee_service_key,
                    self.relation_type,
                    self.direction,
                    self.observation_state,
                    self.request_count,
                    self.error_count,
                    self.latency_p95_seconds,
                )
            ):
                raise ValueError("coverage parent cannot contain graph call measurements")
            _require_graph_coverage(
                collection_state=self.collection_state,
                valid_sample_count=self.valid_sample_count,
                expected_sample_count=self.expected_sample_count,
            )
        parent_payload = _graph_parent_artifact_payload(self)
        if self.parent_artifact_bytes_sha256 != _digest(parent_payload):
            raise ValueError("graph parent artifact bytes digest does not match content")
        expected_selector = _stable_id(
            "gsel",
            {
                "parent_artifact_kind": self.parent_artifact_kind,
                "parent_artifact_bytes_sha256": self.parent_artifact_bytes_sha256,
            },
        )
        unsigned = self.model_dump(
            mode="json", exclude={"parent_record_id", "parent_record_sha256"}
        )
        digest = _digest(unsigned)
        if self.parent_selector != expected_selector:
            raise ValueError("graph parent selector is not bound to parent artifact bytes")
        if self.parent_record_id != f"gpar_{digest[:24]}" or self.parent_record_sha256 != digest:
            raise ValueError("graph adapter parent record is not content-addressed")
        return self

    def verify(self) -> bool:
        try:
            return (
                GraphAdapterParentRecord.model_validate(
                    self.model_dump(mode="json", warnings="error")
                )
                == self
            )
        except (AttributeError, TypeError, ValueError):
            return False


def _graph_parent_artifact_payload(record: GraphAdapterParentRecord) -> dict[str, object]:
    return record.model_dump(
        mode="json",
        include={
            "parent_artifact_kind",
            "opaque_case_id",
            "observation_contract_sha256",
            "window_kind",
            "interval",
            "caller_service_key",
            "callee_service_key",
            "relation_type",
            "direction",
            "observation_state",
            "request_count",
            "error_count",
            "latency_p95_seconds",
            "collection_state",
            "valid_sample_count",
            "expected_sample_count",
            "source_identity_fields",
            "provenance_fields",
        },
    )


def build_graph_adapter_parent_record(
    *,
    opaque_case_id: str,
    observation_contract_sha256: str,
    window_kind: WindowKind,
    interval: TimeInterval,
    parent_artifact_kind: GraphParentArtifactKind,
    caller_service_key: str | None = None,
    callee_service_key: str | None = None,
    relation_type: GraphRelationType | None = None,
    direction: GraphDirection | None = None,
    observation_state: GraphObservationState | None = None,
    request_count: int | None = None,
    error_count: int | None = None,
    latency_p95_seconds: float | None = None,
    collection_state: GraphCollectionState | None = None,
    valid_sample_count: int | None = None,
    expected_sample_count: int | None = None,
    source_identity_fields: Sequence[AdapterAssociationSourceField],
    provenance_fields: Sequence[AdapterAssociationSourceField],
) -> GraphAdapterParentRecord:
    interval = _trusted_time_interval(interval, label="graph parent time interval")
    request_count = _canonical_graph_count(request_count, field_name="request_count")
    error_count = _canonical_graph_count(error_count, field_name="error_count")
    latency_p95_seconds = _canonical_finite_float(latency_p95_seconds)
    valid_sample_count = _canonical_coverage_sample_count(
        valid_sample_count,
        field_name="valid_sample_count",
        minimum=0,
    )
    expected_sample_count = _canonical_coverage_sample_count(
        expected_sample_count,
        field_name="expected_sample_count",
        minimum=1,
    )
    source_fields = _validated_source_fields(
        source_identity_fields, label="graph parent source identity"
    )
    provenance = _validated_source_fields(provenance_fields, label="graph parent provenance")
    parent_payload: dict[str, object] = {
        "parent_artifact_kind": parent_artifact_kind,
        "opaque_case_id": opaque_case_id,
        "observation_contract_sha256": observation_contract_sha256,
        "window_kind": window_kind,
        "interval": interval.model_dump(mode="json"),
        "caller_service_key": caller_service_key,
        "callee_service_key": callee_service_key,
        "relation_type": relation_type,
        "direction": direction,
        "observation_state": observation_state,
        "request_count": request_count,
        "error_count": error_count,
        "latency_p95_seconds": latency_p95_seconds,
        "collection_state": collection_state,
        "valid_sample_count": valid_sample_count,
        "expected_sample_count": expected_sample_count,
        "source_identity_fields": _source_field_documents(source_fields),
        "provenance_fields": _source_field_documents(provenance),
    }
    parent_artifact_bytes_sha256 = _digest(parent_payload)
    unsigned: dict[str, object] = {
        "schema_version": GRAPH_ADAPTER_PARENT_RECORD_SCHEMA_VERSION,
        "parent_selector": _stable_id(
            "gsel",
            {
                "parent_artifact_kind": parent_artifact_kind,
                "parent_artifact_bytes_sha256": parent_artifact_bytes_sha256,
            },
        ),
        **parent_payload,
        "normalization_semantics": ASSOCIATION_NORMALIZATION_SEMANTICS,
        "adapter_semantics_sha256": GRAPH_ADAPTER_SEMANTICS_SHA256,
        "derivation_semantics_sha256": GRAPH_DERIVATION_SEMANTICS_SHA256,
        "parent_artifact_bytes_sha256": parent_artifact_bytes_sha256,
    }
    digest = _digest(unsigned)
    return GraphAdapterParentRecord.model_validate(
        {
            **unsigned,
            "parent_record_id": f"gpar_{digest[:24]}",
            "parent_record_sha256": digest,
        }
    )


class FrozenGraphInputManifest(StrictFrozenModel):
    """Adapter-frozen graph inputs whose digest is pinned by the evaluator."""

    schema_version: Literal["nurse.v2.frozen-graph-input-manifest/1"] = (
        FROZEN_GRAPH_INPUT_MANIFEST_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    observation_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalization_semantics: Literal["sha256-utf8-nfc/1"] = ASSOCIATION_NORMALIZATION_SEMANTICS
    adapter_semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derivation_semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_record_count: int = Field(ge=0)
    parent_records: tuple[GraphAdapterParentRecord, ...]
    source_parent_selectors: tuple[str, ...]
    coverage_parent_selectors: tuple[str, ...]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_complete_frozen_manifest(self) -> Self:
        _require_frozen_graph_semantics(
            adapter_semantics_sha256=self.adapter_semantics_sha256,
            derivation_semantics_sha256=self.derivation_semantics_sha256,
        )
        if self.parent_record_count != len(self.parent_records):
            raise ValueError("graph input manifest parent count mismatch")
        if self.parent_records != tuple(
            sorted(self.parent_records, key=lambda item: item.parent_record_sha256)
        ):
            raise ValueError("graph input manifest parents must be canonically ordered")
        if len({item.parent_record_sha256 for item in self.parent_records}) != len(
            self.parent_records
        ):
            raise ValueError("graph input manifest parents must be unique")
        if len({item.parent_selector for item in self.parent_records}) != len(self.parent_records):
            raise ValueError("graph input manifest selectors must be unique")
        if any(not GraphAdapterParentRecord.verify(item) for item in self.parent_records):
            raise ValueError("graph input manifest contains an invalid parent record")
        if self.source_parent_selectors != tuple(sorted(self.source_parent_selectors)):
            raise ValueError("graph source selectors must be canonically ordered")
        if self.coverage_parent_selectors != tuple(sorted(set(self.coverage_parent_selectors))):
            raise ValueError("graph coverage selectors must be sorted and unique")
        source_selectors = {
            item.parent_selector
            for item in self.parent_records
            if item.parent_artifact_kind == "call_measurement"
        }
        coverage_selectors = {
            item.parent_selector
            for item in self.parent_records
            if item.parent_artifact_kind == "coverage_samples"
        }
        if set(self.source_parent_selectors) != source_selectors:
            raise ValueError("graph source selectors do not cover frozen source parents")
        if set(self.coverage_parent_selectors) != coverage_selectors:
            raise ValueError("graph coverage selectors do not cover frozen coverage parents")
        unsigned = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if self.manifest_sha256 != _digest(unsigned):
            raise ValueError("frozen graph input manifest digest does not match content")
        return self

    def verify(self) -> bool:
        try:
            return (
                FrozenGraphInputManifest.model_validate(
                    self.model_dump(mode="json", warnings="error")
                )
                == self
            )
        except (AttributeError, TypeError, ValueError):
            return False


def build_frozen_graph_input_manifest(
    *,
    opaque_case_id: str,
    observation_contract_sha256: str,
    parent_records: Sequence[GraphAdapterParentRecord],
) -> FrozenGraphInputManifest:
    if any(
        type(item) is not GraphAdapterParentRecord
        or _has_instance_method_shadow(item, "model_dump", "verify")
        for item in parent_records
    ):
        raise ValueError("graph input manifest requires exact parent record models")
    validated = tuple(
        GraphAdapterParentRecord.model_validate(
            GraphAdapterParentRecord.model_dump(item, mode="json", warnings="error")
        )
        for item in parent_records
    )
    if any(not GraphAdapterParentRecord.verify(item) for item in validated):
        raise ValueError("graph input manifest parent record failed verification")
    unique_by_sha = {item.parent_record_sha256: item for item in validated}
    unique_parents = tuple(unique_by_sha[key] for key in sorted(unique_by_sha))
    unsigned: dict[str, object] = {
        "schema_version": FROZEN_GRAPH_INPUT_MANIFEST_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "observation_contract_sha256": observation_contract_sha256,
        "normalization_semantics": ASSOCIATION_NORMALIZATION_SEMANTICS,
        "adapter_semantics_sha256": GRAPH_ADAPTER_SEMANTICS_SHA256,
        "derivation_semantics_sha256": GRAPH_DERIVATION_SEMANTICS_SHA256,
        "parent_record_count": len(unique_parents),
        "parent_records": tuple(
            GraphAdapterParentRecord.model_dump(item, mode="json") for item in unique_parents
        ),
        "source_parent_selectors": tuple(
            sorted(
                item.parent_selector
                for item in validated
                if item.parent_artifact_kind == "call_measurement"
            )
        ),
        "coverage_parent_selectors": tuple(
            sorted(
                item.parent_selector
                for item in validated
                if item.parent_artifact_kind == "coverage_samples"
            )
        ),
    }
    return FrozenGraphInputManifest.model_validate(
        {**unsigned, "manifest_sha256": _digest(unsigned)}
    )


def _require_pinned_graph_input_manifest(
    manifest: FrozenGraphInputManifest,
    *,
    expected_graph_input_manifest_sha256: str,
) -> FrozenGraphInputManifest:
    if (
        type(expected_graph_input_manifest_sha256) is not str
        or len(expected_graph_input_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_graph_input_manifest_sha256
        )
    ):
        raise ValueError("expected graph input manifest digest must be lowercase sha256")
    if type(manifest) is not FrozenGraphInputManifest:
        raise ValueError("graph input manifest must use the exact frozen model type")
    if _has_instance_method_shadow(manifest, "model_dump", "verify"):
        raise ValueError("frozen graph input manifest contains an instance method shadow")
    validated = FrozenGraphInputManifest.model_validate(
        FrozenGraphInputManifest.model_dump(manifest, mode="json", warnings="error")
    )
    if (
        not _has_same_exact_runtime_shape(manifest, validated)
        or validated != manifest
        or not FrozenGraphInputManifest.verify(validated)
    ):
        raise ValueError("frozen graph input manifest failed verification")
    if not hmac.compare_digest(
        validated.manifest_sha256,
        expected_graph_input_manifest_sha256,
    ):
        raise ValueError("pinned graph input manifest digest mismatch")
    return validated


def _build_parent_selector_index(
    parent_records: Sequence[GraphAdapterParentRecord],
) -> dict[str, GraphAdapterParentRecord]:
    parents_by_selector: dict[str, GraphAdapterParentRecord] = {}
    for parent in parent_records:
        if parent.parent_selector in parents_by_selector:
            raise ValueError("graph parent index selectors must be unique")
        parents_by_selector[parent.parent_selector] = parent
    if len(parents_by_selector) != len(parent_records):  # pragma: no cover - defensive
        raise ValueError("graph parent index is incomplete")
    return parents_by_selector


@dataclass(frozen=True)
class _VerifiedGraphInputContext:
    manifest: FrozenGraphInputManifest
    parents_by_selector: dict[str, GraphAdapterParentRecord]


@dataclass(frozen=True)
class _VerifiedGraphBaseContext:
    time_contract: CaseTimeContract
    candidate_universe: CandidateUniverse
    frozen_candidates: FrozenCandidateUniverse
    entity_registry: TemporalEntityRegistry


def _require_verified_graph_base_context(
    *,
    time_contract: CaseTimeContract,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
) -> _VerifiedGraphBaseContext:
    clean_time_contract = _trusted_case_time_contract(time_contract)
    clean_candidate_universe = _trusted_candidate_universe(candidate_universe)
    clean_frozen_candidates = _trusted_frozen_candidate_universe(frozen_candidates)
    clean_entity_registry = _trusted_entity_registry(entity_registry)
    candidate_keys = tuple(item.service_entity_key for item in clean_candidate_universe.candidates)
    if (
        clean_candidate_universe.opaque_case_id != clean_frozen_candidates.opaque_case_id
        or clean_candidate_universe.universe_sha256
        != clean_frozen_candidates.candidate_universe_sha256
        or clean_candidate_universe.observation_contract_sha256
        != clean_frozen_candidates.observation_contract_sha256
        or candidate_keys != clean_frozen_candidates.candidate_service_keys
    ):
        raise ValueError("candidate universe does not match candidate freeze")
    return _VerifiedGraphBaseContext(
        time_contract=clean_time_contract,
        candidate_universe=clean_candidate_universe,
        frozen_candidates=clean_frozen_candidates,
        entity_registry=clean_entity_registry,
    )


def _require_verified_graph_input_context(
    manifest: FrozenGraphInputManifest,
    *,
    expected_graph_input_manifest_sha256: str,
) -> _VerifiedGraphInputContext:
    validated = _require_pinned_graph_input_manifest(
        manifest,
        expected_graph_input_manifest_sha256=expected_graph_input_manifest_sha256,
    )
    return _VerifiedGraphInputContext(
        manifest=validated,
        parents_by_selector=_build_parent_selector_index(validated.parent_records),
    )


def _indexed_parent_by_selector(
    context: _VerifiedGraphInputContext,
    parent_selector: str,
) -> GraphAdapterParentRecord:
    try:
        return context.parents_by_selector[parent_selector]
    except KeyError as error:
        raise ValueError("graph parent selector must resolve exactly once") from error


def _parent_by_selector(
    manifest: FrozenGraphInputManifest,
    parent_selector: str,
) -> GraphAdapterParentRecord:
    matches = tuple(
        item for item in manifest.parent_records if item.parent_selector == parent_selector
    )
    if len(matches) != 1:
        raise ValueError("graph parent selector must resolve exactly once")
    return matches[0]


class CanonicalGraphSourceInput(StrictFrozenModel):
    """Manifest-bound selector for one graph source parent record."""

    schema_version: Literal["nurse.v2.canonical-graph-source-input/2"] = (
        CANONICAL_GRAPH_SOURCE_INPUT_SCHEMA_VERSION
    )
    canonical_input_id: str = Field(pattern=r"^cgsi_[0-9a-f]{24}$")
    graph_input_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_selector: str = Field(pattern=r"^gsel_[0-9a-f]{24}$")
    parent_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_content_addressed_reference(self) -> Self:
        unsigned = self.model_dump(
            mode="json", exclude={"canonical_input_id", "canonical_input_sha256"}
        )
        digest = _digest(unsigned)
        if (
            self.canonical_input_id != f"cgsi_{digest[:24]}"
            or self.canonical_input_sha256 != digest
        ):
            raise ValueError("canonical graph source input is not content-addressed")
        return self

    def verify(self) -> bool:
        try:
            return (
                CanonicalGraphSourceInput.model_validate(
                    self.model_dump(mode="json", warnings="error")
                )
                == self
            )
        except (AttributeError, TypeError, ValueError):
            return False


class CanonicalGraphCoverageInput(StrictFrozenModel):
    """Manifest-bound selector for one graph coverage parent record."""

    schema_version: Literal["nurse.v2.canonical-graph-coverage-input/2"] = (
        CANONICAL_GRAPH_COVERAGE_INPUT_SCHEMA_VERSION
    )
    canonical_input_id: str = Field(pattern=r"^cgci_[0-9a-f]{24}$")
    graph_input_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_selector: str = Field(pattern=r"^gsel_[0-9a-f]{24}$")
    parent_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_content_addressed_reference(self) -> Self:
        unsigned = self.model_dump(
            mode="json", exclude={"canonical_input_id", "canonical_input_sha256"}
        )
        digest = _digest(unsigned)
        if (
            self.canonical_input_id != f"cgci_{digest[:24]}"
            or self.canonical_input_sha256 != digest
        ):
            raise ValueError("canonical graph coverage input is not content-addressed")
        return self

    def verify(self) -> bool:
        try:
            return (
                CanonicalGraphCoverageInput.model_validate(
                    self.model_dump(mode="json", warnings="error")
                )
                == self
            )
        except (AttributeError, TypeError, ValueError):
            return False


def _build_canonical_graph_input(
    *,
    context: _VerifiedGraphInputContext,
    parent_selector: str,
    expected_kind: GraphParentArtifactKind,
) -> CanonicalGraphSourceInput | CanonicalGraphCoverageInput:
    manifest = context.manifest
    parent = _indexed_parent_by_selector(context, parent_selector)
    if parent.parent_artifact_kind != expected_kind:
        raise ValueError("graph parent selector has the wrong artifact kind")
    schema_version = (
        CANONICAL_GRAPH_SOURCE_INPUT_SCHEMA_VERSION
        if expected_kind == "call_measurement"
        else CANONICAL_GRAPH_COVERAGE_INPUT_SCHEMA_VERSION
    )
    unsigned: dict[str, object] = {
        "schema_version": schema_version,
        "graph_input_manifest_sha256": manifest.manifest_sha256,
        "parent_selector": parent.parent_selector,
        "parent_record_sha256": parent.parent_record_sha256,
    }
    digest = _digest(unsigned)
    model_type = (
        CanonicalGraphSourceInput
        if expected_kind == "call_measurement"
        else CanonicalGraphCoverageInput
    )
    prefix = "cgsi" if expected_kind == "call_measurement" else "cgci"
    return model_type.model_validate(
        {
            **unsigned,
            "canonical_input_id": f"{prefix}_{digest[:24]}",
            "canonical_input_sha256": digest,
        }
    )


def build_canonical_graph_source_input(
    *,
    graph_input_manifest: FrozenGraphInputManifest,
    expected_graph_input_manifest_sha256: str,
    parent_selector: str,
) -> CanonicalGraphSourceInput:
    context = _require_verified_graph_input_context(
        graph_input_manifest,
        expected_graph_input_manifest_sha256=expected_graph_input_manifest_sha256,
    )
    return cast(
        CanonicalGraphSourceInput,
        _build_canonical_graph_input(
            context=context,
            parent_selector=parent_selector,
            expected_kind="call_measurement",
        ),
    )


def build_canonical_graph_coverage_input(
    *,
    graph_input_manifest: FrozenGraphInputManifest,
    expected_graph_input_manifest_sha256: str,
    parent_selector: str,
) -> CanonicalGraphCoverageInput:
    context = _require_verified_graph_input_context(
        graph_input_manifest,
        expected_graph_input_manifest_sha256=expected_graph_input_manifest_sha256,
    )
    return cast(
        CanonicalGraphCoverageInput,
        _build_canonical_graph_input(
            context=context,
            parent_selector=parent_selector,
            expected_kind="coverage_samples",
        ),
    )


def _canonical_inputs_from_manifest(
    context: _VerifiedGraphInputContext,
) -> tuple[tuple[CanonicalGraphCoverageInput, ...], tuple[CanonicalGraphSourceInput, ...]]:
    manifest = context.manifest
    coverages = tuple(
        cast(
            CanonicalGraphCoverageInput,
            _build_canonical_graph_input(
                context=context,
                parent_selector=selector,
                expected_kind="coverage_samples",
            ),
        )
        for selector in manifest.coverage_parent_selectors
    )
    sources = tuple(
        cast(
            CanonicalGraphSourceInput,
            _build_canonical_graph_input(
                context=context,
                parent_selector=selector,
                expected_kind="call_measurement",
            ),
        )
        for selector in manifest.source_parent_selectors
    )
    return coverages, sources


def _resolve_canonical_parent(
    *,
    canonical_input: CanonicalGraphSourceInput | CanonicalGraphCoverageInput,
    graph_input_manifest: FrozenGraphInputManifest,
    expected_graph_input_manifest_sha256: str,
    expected_kind: GraphParentArtifactKind,
) -> tuple[
    CanonicalGraphSourceInput | CanonicalGraphCoverageInput,
    GraphAdapterParentRecord,
]:
    context = _require_verified_graph_input_context(
        graph_input_manifest,
        expected_graph_input_manifest_sha256=expected_graph_input_manifest_sha256,
    )
    return _resolve_canonical_parent_from_context(
        canonical_input=canonical_input,
        context=context,
        expected_kind=expected_kind,
    )


def _resolve_canonical_parent_from_context(
    *,
    canonical_input: CanonicalGraphSourceInput | CanonicalGraphCoverageInput,
    context: _VerifiedGraphInputContext,
    expected_kind: GraphParentArtifactKind,
) -> tuple[
    CanonicalGraphSourceInput | CanonicalGraphCoverageInput,
    GraphAdapterParentRecord,
]:
    manifest = context.manifest
    model_type = (
        CanonicalGraphSourceInput
        if expected_kind == "call_measurement"
        else CanonicalGraphCoverageInput
    )
    if type(canonical_input) is not model_type:
        raise ValueError("canonical graph input has the wrong exact model type")
    if _has_instance_method_shadow(canonical_input, "model_dump", "verify"):
        raise ValueError("canonical graph input contains an instance method shadow")
    validated = model_type.model_validate(
        model_type.model_dump(canonical_input, mode="json", warnings="error")
    )
    verified = (
        CanonicalGraphSourceInput.verify(cast(CanonicalGraphSourceInput, validated))
        if expected_kind == "call_measurement"
        else CanonicalGraphCoverageInput.verify(cast(CanonicalGraphCoverageInput, validated))
    )
    if validated != canonical_input or not verified:
        raise ValueError("canonical graph input failed verification")
    if validated.graph_input_manifest_sha256 != manifest.manifest_sha256:
        raise ValueError("canonical graph input references a different manifest")
    parent = _indexed_parent_by_selector(context, validated.parent_selector)
    if parent.parent_artifact_kind != expected_kind:
        raise ValueError("canonical graph input references the wrong parent kind")
    if parent.parent_record_sha256 != validated.parent_record_sha256:
        raise ValueError("canonical graph input parent digest mismatch")
    return validated, parent


class GraphWindowCoverage(StrictFrozenModel):
    """Derived coverage evidence for exactly one declared B/F window."""

    schema_version: Literal["nurse.v2.graph-window-coverage/4"] = (
        GRAPH_WINDOW_COVERAGE_SCHEMA_VERSION
    )
    coverage_id: str = Field(pattern=r"^gwc_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    observation_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    window_kind: WindowKind
    interval: TimeInterval
    state: Literal["observed", "no_record", "unknown"]
    valid_sample_count: int | None
    expected_sample_count: int | None
    coverage_fraction: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    canonical_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derivation_semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    coverage_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("coverage_fraction", mode="before")
    @classmethod
    def canonicalize_coverage_fraction(cls, value: object) -> float | None:
        return _canonical_finite_float(value)

    @field_validator("valid_sample_count", "expected_sample_count", mode="before")
    @classmethod
    def canonicalize_coverage_sample_counts(cls, value: object, info: ValidationInfo) -> int | None:
        field_name = info.field_name or "coverage sample count"
        minimum = 1 if field_name == "expected_sample_count" else 0
        return _canonical_coverage_sample_count(
            value,
            field_name=field_name,
            minimum=minimum,
        )

    @model_validator(mode="after")
    def require_consistent_coverage(self) -> Self:
        _require_frozen_graph_semantics(
            adapter_semantics_sha256=self.adapter_semantics_sha256,
            derivation_semantics_sha256=self.derivation_semantics_sha256,
        )
        if self.provenance_ids != tuple(sorted(set(self.provenance_ids))):
            raise ValueError("coverage provenance IDs must be sorted and unique")
        expected_fraction = _coverage_fraction_from_counts(
            collection_state=self.state,
            valid_sample_count=self.valid_sample_count,
            expected_sample_count=self.expected_sample_count,
        )
        if self.coverage_fraction != expected_fraction:
            raise ValueError("coverage fraction does not match exact counts")
        unsigned = self.model_dump(mode="json", exclude={"coverage_id", "coverage_sha256"})
        digest = _digest(unsigned)
        if self.coverage_id != f"gwc_{digest[:24]}" or self.coverage_sha256 != digest:
            raise ValueError("graph window coverage is not content-addressed")
        return self

    def verify(self) -> bool:
        try:
            return (
                GraphWindowCoverage.model_validate(self.model_dump(mode="json", warnings="error"))
                == self
            )
        except (AttributeError, TypeError, ValueError):
            return False


def build_graph_window_coverage(
    *,
    canonical_input: CanonicalGraphCoverageInput,
    graph_input_manifest: FrozenGraphInputManifest,
    expected_graph_input_manifest_sha256: str,
) -> GraphWindowCoverage:
    context = _require_verified_graph_input_context(
        graph_input_manifest,
        expected_graph_input_manifest_sha256=expected_graph_input_manifest_sha256,
    )
    return _build_graph_window_coverage_from_context(
        canonical_input=canonical_input,
        context=context,
    )


def _build_graph_window_coverage_from_context(
    *,
    canonical_input: CanonicalGraphCoverageInput,
    context: _VerifiedGraphInputContext,
) -> GraphWindowCoverage:
    validated_input, parent = _resolve_canonical_parent_from_context(
        canonical_input=canonical_input,
        context=context,
        expected_kind="coverage_samples",
    )
    validated = cast(CanonicalGraphCoverageInput, validated_input)
    unsigned: dict[str, object] = {
        "schema_version": GRAPH_WINDOW_COVERAGE_SCHEMA_VERSION,
        "opaque_case_id": parent.opaque_case_id,
        "observation_contract_sha256": parent.observation_contract_sha256,
        "window_kind": parent.window_kind,
        "interval": parent.interval.model_dump(mode="json"),
        "state": parent.collection_state,
        "valid_sample_count": parent.valid_sample_count,
        "expected_sample_count": parent.expected_sample_count,
        "coverage_fraction": parent.derived_coverage_fraction,
        "canonical_input_sha256": validated.canonical_input_sha256,
        "parent_source_sha256": parent.parent_artifact_bytes_sha256,
        "adapter_semantics_sha256": parent.adapter_semantics_sha256,
        "derivation_semantics_sha256": parent.derivation_semantics_sha256,
        "provenance_ids": parent.derived_provenance_ids,
    }
    digest = _digest(unsigned)
    return GraphWindowCoverage.model_validate(
        {
            **unsigned,
            "coverage_id": f"gwc_{digest[:24]}",
            "coverage_sha256": digest,
        }
    )


class GraphSourceRecord(StrictFrozenModel):
    """One graph observation deterministically derived from canonical input."""

    schema_version: Literal["nurse.v2.graph-source-record/4"] = GRAPH_SOURCE_RECORD_SCHEMA_VERSION
    source_record_id: str = Field(pattern=r"^gsrc_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    observation_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    window_kind: WindowKind
    interval: TimeInterval
    caller_service_key: str = Field(min_length=1)
    callee_service_key: str = Field(min_length=1)
    relation_type: GraphRelationType
    direction: GraphDirection
    observation_state: GraphObservationState
    request_count: int | None = Field(default=None, ge=0, le=MAX_EXACT_GRAPH_COUNT)
    error_count: int | None = Field(default=None, ge=0, le=MAX_EXACT_GRAPH_COUNT)
    latency_p95_seconds: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_input_sha256s: tuple[str, ...] = Field(min_length=1)
    parent_source_sha256s: tuple[str, ...] = Field(min_length=1)
    adapter_semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derivation_semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    source_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("latency_p95_seconds", mode="before")
    @classmethod
    def canonicalize_latency(cls, value: object) -> float | None:
        return _canonical_finite_float(value)

    @field_validator("request_count", "error_count", mode="before")
    @classmethod
    def canonicalize_counts(cls, value: object, info: ValidationInfo) -> int | None:
        return _canonical_graph_count(value, field_name=info.field_name or "graph count")

    @model_validator(mode="after")
    def require_typed_content_addressed_record(self) -> Self:
        _require_graph_measurement(
            caller_service_key=self.caller_service_key,
            callee_service_key=self.callee_service_key,
            relation_type=self.relation_type,
            direction=self.direction,
            observation_state=self.observation_state,
            request_count=self.request_count,
            error_count=self.error_count,
            latency_p95_seconds=self.latency_p95_seconds,
        )
        _require_frozen_graph_semantics(
            adapter_semantics_sha256=self.adapter_semantics_sha256,
            derivation_semantics_sha256=self.derivation_semantics_sha256,
        )
        if self.canonical_input_sha256s != tuple(sorted(set(self.canonical_input_sha256s))):
            raise ValueError("canonical graph input hashes must be sorted and unique")
        if self.parent_source_sha256s != tuple(sorted(set(self.parent_source_sha256s))):
            raise ValueError("graph parent source hashes must be sorted and unique")
        if self.provenance_ids != tuple(sorted(set(self.provenance_ids))):
            raise ValueError("graph source provenance IDs must be sorted and unique")
        identity = {
            "opaque_case_id": self.opaque_case_id,
            "observation_contract_sha256": self.observation_contract_sha256,
            "window_kind": self.window_kind,
            "source_fingerprint": self.source_fingerprint,
            "caller_service_key": self.caller_service_key,
            "callee_service_key": self.callee_service_key,
            "relation_type": self.relation_type,
            "direction": self.direction,
        }
        unsigned = self.model_dump(
            mode="json", exclude={"source_record_id", "source_record_sha256"}
        )
        if self.source_record_id != _stable_id("gsrc", identity):
            raise ValueError("graph source record ID is not content-addressed")
        if self.source_record_sha256 != _digest(unsigned):
            raise ValueError("graph source record digest does not match content")
        return self

    def verify(self) -> bool:
        try:
            return (
                GraphSourceRecord.model_validate(self.model_dump(mode="json", warnings="error"))
                == self
            )
        except (AttributeError, TypeError, ValueError):
            return False


def build_graph_source_record(
    *,
    canonical_input: CanonicalGraphSourceInput,
    graph_input_manifest: FrozenGraphInputManifest,
    expected_graph_input_manifest_sha256: str,
) -> GraphSourceRecord:
    context = _require_verified_graph_input_context(
        graph_input_manifest,
        expected_graph_input_manifest_sha256=expected_graph_input_manifest_sha256,
    )
    return _build_graph_source_record_from_context(
        canonical_input=canonical_input,
        context=context,
    )


def _build_graph_source_record_from_context(
    *,
    canonical_input: CanonicalGraphSourceInput,
    context: _VerifiedGraphInputContext,
) -> GraphSourceRecord:
    validated_input, parent = _resolve_canonical_parent_from_context(
        canonical_input=canonical_input,
        context=context,
        expected_kind="call_measurement",
    )
    validated = cast(CanonicalGraphSourceInput, validated_input)
    return _build_graph_source_record(
        opaque_case_id=parent.opaque_case_id,
        observation_contract_sha256=parent.observation_contract_sha256,
        window_kind=parent.window_kind,
        interval=parent.interval,
        caller_service_key=cast(str, parent.caller_service_key),
        callee_service_key=cast(str, parent.callee_service_key),
        relation_type=cast(GraphRelationType, parent.relation_type),
        direction=cast(GraphDirection, parent.direction),
        observation_state=cast(GraphObservationState, parent.observation_state),
        request_count=parent.request_count,
        error_count=parent.error_count,
        latency_p95_seconds=parent.latency_p95_seconds,
        source_fingerprint=parent.derived_source_fingerprint,
        canonical_input_sha256s=(validated.canonical_input_sha256,),
        parent_source_sha256s=(parent.parent_artifact_bytes_sha256,),
        provenance_ids=parent.derived_provenance_ids,
    )


def _build_graph_source_record(
    *,
    opaque_case_id: str,
    observation_contract_sha256: str,
    window_kind: WindowKind,
    interval: TimeInterval,
    caller_service_key: str,
    callee_service_key: str,
    relation_type: GraphRelationType,
    direction: GraphDirection,
    observation_state: GraphObservationState,
    request_count: int | None,
    error_count: int | None,
    latency_p95_seconds: float | None,
    source_fingerprint: str,
    canonical_input_sha256s: Sequence[str],
    parent_source_sha256s: Sequence[str],
    provenance_ids: Sequence[str],
) -> GraphSourceRecord:
    identity = {
        "opaque_case_id": opaque_case_id,
        "observation_contract_sha256": observation_contract_sha256,
        "window_kind": window_kind,
        "source_fingerprint": source_fingerprint,
        "caller_service_key": caller_service_key,
        "callee_service_key": callee_service_key,
        "relation_type": relation_type,
        "direction": direction,
    }
    unsigned: dict[str, object] = {
        "schema_version": GRAPH_SOURCE_RECORD_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "observation_contract_sha256": observation_contract_sha256,
        "window_kind": window_kind,
        "interval": interval.model_dump(mode="json"),
        "caller_service_key": caller_service_key,
        "callee_service_key": callee_service_key,
        "relation_type": relation_type,
        "direction": direction,
        "observation_state": observation_state,
        "request_count": request_count,
        "error_count": error_count,
        "latency_p95_seconds": latency_p95_seconds,
        "source_fingerprint": source_fingerprint,
        "canonical_input_sha256s": tuple(sorted(set(canonical_input_sha256s))),
        "parent_source_sha256s": tuple(sorted(set(parent_source_sha256s))),
        "adapter_semantics_sha256": GRAPH_ADAPTER_SEMANTICS_SHA256,
        "derivation_semantics_sha256": GRAPH_DERIVATION_SEMANTICS_SHA256,
        "provenance_ids": tuple(sorted(set(provenance_ids))),
    }
    return GraphSourceRecord.model_validate(
        {
            **unsigned,
            "source_record_id": _stable_id("gsrc", identity),
            "source_record_sha256": _digest(unsigned),
        }
    )


def _sorted_coverages(
    records: Sequence[GraphWindowCoverage],
) -> tuple[GraphWindowCoverage, ...]:
    return tuple(sorted(records, key=lambda item: item.coverage_sha256))


def _sorted_source_records(
    records: Sequence[GraphSourceRecord],
) -> tuple[GraphSourceRecord, ...]:
    return tuple(sorted(records, key=lambda item: item.source_record_sha256))


class GraphMetricSummary(StrictFrozenModel):
    state: MetricState
    value: float | None = Field(default=None, allow_inf_nan=False)
    aggregation: Literal[
        "sum_unique_sources",
        "weighted_rate_from_counts",
        "max_source_p95",
        "not_computable",
    ]
    contributing_source_record_ids: tuple[str, ...]

    @field_validator("value", mode="before")
    @classmethod
    def canonicalize_value(cls, value: object) -> float | None:
        return _canonical_finite_float(value)

    @model_validator(mode="after")
    def require_consistent_state(self) -> Self:
        if self.contributing_source_record_ids != tuple(
            sorted(set(self.contributing_source_record_ids))
        ):
            raise ValueError("metric source IDs must be sorted and unique")
        if self.state in {"no_record", "unknown"} and self.value is not None:
            raise ValueError("unavailable metric cannot claim a value")
        if self.state in {"observed", "zero", "verified_normal"} and self.value is None:
            raise ValueError("available metric requires a value")
        if self.state in {"zero", "verified_normal"} and self.value != 0.0:
            raise ValueError("zero or verified-normal metric must equal zero")
        return self


class GraphCoverageSummary(StrictFrozenModel):
    observation_state: Literal["observed", "no_record", "unknown"]
    coverage_state: CoverageState
    valid_sample_count: int | None
    expected_sample_count: int | None
    coverage_fraction: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    minimum_coverage_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    provenance_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("coverage_fraction", "minimum_coverage_fraction", mode="before")
    @classmethod
    def canonicalize_fractions(cls, value: object) -> float | None:
        return _canonical_finite_float(value)

    @field_validator("valid_sample_count", "expected_sample_count", mode="before")
    @classmethod
    def canonicalize_coverage_sample_counts(cls, value: object, info: ValidationInfo) -> int | None:
        field_name = info.field_name or "coverage sample count"
        minimum = 1 if field_name == "expected_sample_count" else 0
        return _canonical_coverage_sample_count(
            value,
            field_name=field_name,
            minimum=minimum,
        )

    @model_validator(mode="after")
    def require_consistent_coverage_state(self) -> Self:
        expected_fraction = _coverage_fraction_from_counts(
            collection_state=self.observation_state,
            valid_sample_count=self.valid_sample_count,
            expected_sample_count=self.expected_sample_count,
        )
        if self.coverage_fraction != expected_fraction:
            raise ValueError("coverage fraction does not match exact counts")
        if self.observation_state == "unknown":
            if self.coverage_state != "unknown":
                raise ValueError("unknown coverage must remain unknown")
            return self
        if self.observation_state == "no_record":
            if self.coverage_state != "insufficient":
                raise ValueError("no-record coverage cannot be sufficient")
            return self
        expected_state: CoverageState = (
            "sufficient"
            if _coverage_meets_minimum(
                valid_sample_count=cast(int, self.valid_sample_count),
                expected_sample_count=cast(int, self.expected_sample_count),
                minimum_coverage_fraction=self.minimum_coverage_fraction,
            )
            else "insufficient"
        )
        if self.coverage_state != expected_state:
            raise ValueError("observed coverage state does not match its threshold")
        return self


class GraphSnapshotEdge(StrictFrozenModel):
    edge_id: str = Field(pattern=r"^gedge_[0-9a-f]{24}$")
    caller_service_key: str = Field(min_length=1)
    callee_service_key: str = Field(min_length=1)
    relation_type: GraphRelationType
    direction: GraphDirection
    allows_directed_propagation: bool
    interval: TimeInterval
    observation_state: EdgeObservationState
    coverage: GraphCoverageSummary
    traffic: GraphMetricSummary
    error_count: GraphMetricSummary
    error_rate: GraphMetricSummary
    latency_p95_seconds: GraphMetricSummary
    source_record_ids: tuple[str, ...] = Field(min_length=1)
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    edge_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            validated = GraphSnapshotEdge.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if validated.allows_directed_propagation != (
                validated.relation_type
                in {"received_dependency_failure", "caller_load_transmission"}
            ):
                return False
            unsigned = validated.model_dump(mode="json", exclude={"edge_id", "edge_sha256"})
            digest = _digest(unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            validated == self
            and self.edge_id == f"gedge_{digest[:24]}"
            and self.edge_sha256 == digest
        )


class GraphSnapshot(StrictFrozenModel):
    """Current graph snapshot schema; v1 remains recognizable in ranking_schemas."""

    schema_version: Literal["nurse.v2.graph-snapshot/4"] = GRAPH_SNAPSHOT_SCHEMA_VERSION
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    window_kind: WindowKind
    interval: TimeInterval
    decision_time_unix_ns: int = Field(ge=0)
    observation_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_universe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entity_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    node_service_keys: tuple[str, ...] = Field(min_length=1)
    coverage: GraphCoverageSummary
    edges: tuple[GraphSnapshotEdge, ...]
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            validated = GraphSnapshot.model_validate(self.model_dump(mode="json", warnings="error"))
            if validated != self:
                return False
            if self.node_service_keys != tuple(sorted(set(self.node_service_keys))):
                return False
            if self.edges != tuple(sorted(self.edges, key=_edge_sort_key)):
                return False
            if any(not item.verify() for item in self.edges):
                return False
            nodes = set(self.node_service_keys)
            if any(
                item.caller_service_key not in nodes or item.callee_service_key not in nodes
                for item in self.edges
            ):
                return False
            unsigned = self.model_dump(mode="json", exclude={"snapshot_sha256"})
            return self.snapshot_sha256 == _digest(unsigned)
        except (AttributeError, TypeError, ValueError):
            return False


class GraphEdgeChange(StrictFrozenModel):
    change_id: str = Field(pattern=r"^gec_[0-9a-f]{24}$")
    caller_service_key: str = Field(min_length=1)
    callee_service_key: str = Field(min_length=1)
    relation_type: GraphRelationType
    direction: GraphDirection
    normal_edge_id: str | None = Field(default=None, pattern=r"^gedge_[0-9a-f]{24}$")
    fault_edge_id: str | None = Field(default=None, pattern=r"^gedge_[0-9a-f]{24}$")
    normal_presence: EdgePresence
    fault_presence: EdgePresence
    change_state: EdgeChangeState
    traffic_change: MetricChangeState
    error_count_change: MetricChangeState
    error_rate_change: MetricChangeState
    latency_change: MetricChangeState
    change_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            validated = GraphEdgeChange.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = validated.model_dump(mode="json", exclude={"change_id", "change_sha256"})
            digest = _digest(unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            validated == self
            and self.change_id == f"gec_{digest[:24]}"
            and self.change_sha256 == digest
        )


class GraphInputDecision(StrictFrozenModel):
    """Closed audit disposition for one unique typed input document."""

    schema_version: Literal["nurse.v2.graph-input-decision/2"] = GRAPH_INPUT_DECISION_SCHEMA_VERSION
    decision_id: str = Field(pattern=r"^gid_[0-9a-f]{24}$")
    source_record_id: str = Field(pattern=r"^gsrc_[0-9a-f]{24}$")
    source_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    occurrence_count: int = Field(ge=1)
    disposition: GraphInputDisposition
    reason_codes: tuple[GraphInputReason, ...] = Field(min_length=1)
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_closed_disposition(self) -> Self:
        if self.reason_codes != tuple(sorted(set(self.reason_codes))):
            raise ValueError("graph input reason codes must be sorted and unique")
        if self.provenance_ids != tuple(sorted(set(self.provenance_ids))):
            raise ValueError("graph input provenance IDs must be sorted and unique")
        if self.disposition == "accepted" and self.reason_codes != ("accepted",):
            raise ValueError("accepted graph input requires the accepted reason")
        if self.disposition == "unknown" and self.reason_codes != ("unknown_observation",):
            raise ValueError("unknown graph input requires the unknown reason")
        if self.disposition == "rejected" and any(
            reason in {"accepted", "unknown_observation"} for reason in self.reason_codes
        ):
            raise ValueError("rejected graph input cannot use a retained reason")
        unsigned = self.model_dump(mode="json", exclude={"decision_id", "decision_sha256"})
        digest = _digest(unsigned)
        if self.decision_id != f"gid_{digest[:24]}" or self.decision_sha256 != digest:
            raise ValueError("graph input decision is not content-addressed")
        return self

    def verify(self) -> bool:
        try:
            return (
                GraphInputDecision.model_validate(self.model_dump(mode="json", warnings="error"))
                == self
            )
        except (AttributeError, TypeError, ValueError):
            return False


def _validated_exact_replay_models(
    values: tuple[StrictFrozenModel, ...],
    *,
    model_type: type[StrictFrozenModel],
) -> tuple[StrictFrozenModel, ...]:
    """Rebuild exact base models before comparing caller-supplied replay evidence."""

    if type(values) is not tuple:
        raise ValueError("caller-supplied graph replay inputs must use an exact tuple")
    validated: list[StrictFrozenModel] = []
    for item in values:
        if type(item) is not model_type:
            raise ValueError("caller-supplied graph replay input has a non-exact model type")
        if _has_instance_method_shadow(item, "model_dump", "verify"):
            raise ValueError("caller-supplied graph replay input contains a method shadow")
        document = model_type.model_dump(item, mode="json", warnings="error")
        rebuilt = model_type.model_validate(document)
        if type(rebuilt) is not model_type:
            raise ValueError("caller-supplied graph replay input rebuilt to a non-exact type")
        if model_type.model_dump(rebuilt, mode="json", warnings="error") != document:
            raise ValueError("caller-supplied graph replay input is not canonical")
        verify = getattr(model_type, "verify", None)
        if not callable(verify) or not verify(rebuilt):
            raise ValueError("caller-supplied graph replay input failed verification")
        validated.append(rebuilt)
    return tuple(validated)


class GraphSnapshotPair(StrictFrozenModel):
    schema_version: Literal["nurse.v2.graph-snapshot-pair/5"] = GRAPH_SNAPSHOT_PAIR_SCHEMA_VERSION
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    graph_input_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    time_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_universe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entity_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    minimum_coverage_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    canonical_coverage_input_sha256s: tuple[str, str]
    canonical_source_input_sha256s: tuple[str, ...]
    window_coverage_sha256s: tuple[str, str]
    source_record_sha256s: tuple[str, ...]
    input_occurrence_count: int = Field(ge=0)
    input_decisions: tuple[GraphInputDecision, ...]
    accepted_input_sha256s: tuple[str, ...]
    rejected_input_sha256s: tuple[str, ...]
    unknown_input_sha256s: tuple[str, ...]
    duplicate_source_record_ids: tuple[str, ...]
    conflicted_source_record_ids: tuple[str, ...]
    normal_snapshot: GraphSnapshot
    fault_snapshot: GraphSnapshot
    edge_changes: tuple[GraphEdgeChange, ...]
    pair_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("minimum_coverage_fraction", mode="before")
    @classmethod
    def canonicalize_minimum_coverage_fraction(cls, value: object) -> float:
        canonical = _canonical_finite_float(value)
        if canonical is None:
            raise ValueError("minimum coverage fraction must be a graph float")
        return canonical

    def verify(self) -> bool:
        try:
            validated = GraphSnapshotPair.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if validated != self:
                return False
            if not self.normal_snapshot.verify() or not self.fault_snapshot.verify():
                return False
            if self.normal_snapshot.window_kind != "normal":
                return False
            if self.fault_snapshot.window_kind != "fault":
                return False
            if any(not item.verify() for item in self.edge_changes):
                return False
            if self.input_decisions != tuple(
                sorted(self.input_decisions, key=lambda item: item.source_record_sha256)
            ) or any(not item.verify() for item in self.input_decisions):
                return False
            if sum(item.occurrence_count for item in self.input_decisions) != (
                self.input_occurrence_count
            ):
                return False
            partitions = {
                disposition: tuple(
                    item.source_record_sha256
                    for item in self.input_decisions
                    if item.disposition == disposition
                )
                for disposition in cast(
                    tuple[GraphInputDisposition, ...],
                    ("accepted", "rejected", "unknown"),
                )
            }
            if (
                partitions["accepted"] != self.accepted_input_sha256s
                or partitions["rejected"] != self.rejected_input_sha256s
                or partitions["unknown"] != self.unknown_input_sha256s
            ):
                return False
            if self.duplicate_source_record_ids != tuple(
                sorted(set(self.duplicate_source_record_ids))
            ):
                return False
            if self.conflicted_source_record_ids != tuple(
                sorted(set(self.conflicted_source_record_ids))
            ):
                return False
            if set(self.duplicate_source_record_ids) & set(self.conflicted_source_record_ids):
                return False
            if self.canonical_source_input_sha256s != tuple(
                sorted(self.canonical_source_input_sha256s)
            ):
                return False
            unsigned = self.model_dump(mode="json", exclude={"pair_sha256"})
            return self.pair_sha256 == _digest(unsigned)
        except (AttributeError, TypeError, ValueError):
            return False

    def verify_inputs(
        self,
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
    ) -> bool:
        try:
            clean_pair = _trusted_graph_snapshot_pair(self)
            canonical_expected_minimum = _canonical_finite_float(expected_minimum_coverage_fraction)
            if canonical_expected_minimum is None or not (0.0 <= canonical_expected_minimum <= 1.0):
                return False
            if canonical_expected_minimum != clean_pair.minimum_coverage_fraction:
                return False
            base_context = _require_verified_graph_base_context(
                time_contract=time_contract,
                candidate_universe=candidate_universe,
                frozen_candidates=frozen_candidates,
                entity_registry=entity_registry,
            )
            context = _require_verified_graph_input_context(
                graph_input_manifest,
                expected_graph_input_manifest_sha256=expected_graph_input_manifest_sha256,
            )
            expected_canonical_coverages, expected_canonical_sources = (
                _canonical_inputs_from_manifest(context)
            )
            expected_coverages = tuple(
                _build_graph_window_coverage_from_context(
                    canonical_input=item,
                    context=context,
                )
                for item in expected_canonical_coverages
            )
            expected_sources = tuple(
                _build_graph_source_record_from_context(
                    canonical_input=item,
                    context=context,
                )
                for item in expected_canonical_sources
            )
            if canonical_coverage_inputs is not None:
                supplied_canonical_coverages = cast(
                    tuple[CanonicalGraphCoverageInput, ...],
                    _validated_exact_replay_models(
                        canonical_coverage_inputs,
                        model_type=CanonicalGraphCoverageInput,
                    ),
                )
                if expected_canonical_coverages != supplied_canonical_coverages:
                    return False
            if canonical_source_inputs is not None:
                supplied_canonical_sources = cast(
                    tuple[CanonicalGraphSourceInput, ...],
                    _validated_exact_replay_models(
                        canonical_source_inputs,
                        model_type=CanonicalGraphSourceInput,
                    ),
                )
                if expected_canonical_sources != supplied_canonical_sources:
                    return False
            if window_coverages is not None:
                supplied_coverages = cast(
                    tuple[GraphWindowCoverage, ...],
                    _validated_exact_replay_models(
                        window_coverages,
                        model_type=GraphWindowCoverage,
                    ),
                )
                if _sorted_coverages(expected_coverages) != _sorted_coverages(supplied_coverages):
                    return False
            if source_records is not None:
                supplied_sources = cast(
                    tuple[GraphSourceRecord, ...],
                    _validated_exact_replay_models(
                        source_records,
                        model_type=GraphSourceRecord,
                    ),
                )
                if _sorted_source_records(expected_sources) != _sorted_source_records(
                    supplied_sources
                ):
                    return False
            expected = _build_graph_snapshot_pair_from_context(
                base_context=base_context,
                context=context,
                minimum_coverage_fraction=canonical_expected_minimum,
            )
        except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
            return False
        expected_document = GraphSnapshotPair.model_dump(
            expected,
            mode="json",
            warnings="error",
        )
        clean_document = GraphSnapshotPair.model_dump(
            clean_pair,
            mode="json",
            warnings="error",
        )
        return hmac.compare_digest(expected.pair_sha256, clean_pair.pair_sha256) and (
            canonical_json_bytes(expected_document) == canonical_json_bytes(clean_document)
        )


def _trusted_graph_snapshot_pair(value: object) -> GraphSnapshotPair:
    preflight_exact_pydantic_tree(
        value,
        root_model_types=(GraphSnapshotPair,),
        label="graph snapshot pair",
    )
    rebuilt = _trusted_exact_model(
        value,
        model_type=GraphSnapshotPair,
        label="graph snapshot pair",
        guarded_methods=("verify", "verify_inputs"),
    )
    if not _has_same_exact_runtime_shape(value, rebuilt):
        raise ValueError("graph snapshot pair contains runtime field poisoning")
    if not GraphSnapshotPair.verify(rebuilt):
        raise ValueError("graph snapshot pair failed verification")
    return rebuilt


def _coverage_summary(record: GraphWindowCoverage, minimum: float) -> GraphCoverageSummary:
    if record.state == "unknown":
        state: CoverageState = "unknown"
    elif record.state == "no_record":
        state = "insufficient"
    elif _coverage_meets_minimum(
        valid_sample_count=cast(int, record.valid_sample_count),
        expected_sample_count=cast(int, record.expected_sample_count),
        minimum_coverage_fraction=minimum,
    ):
        state = "sufficient"
    else:
        state = "insufficient"
    return GraphCoverageSummary(
        observation_state=record.state,
        coverage_state=state,
        valid_sample_count=record.valid_sample_count,
        expected_sample_count=record.expected_sample_count,
        coverage_fraction=record.coverage_fraction,
        minimum_coverage_fraction=minimum,
        provenance_ids=record.provenance_ids,
    )


def _metric_summary(
    records: Sequence[GraphSourceRecord],
    field: Literal["request_count", "error_count", "latency_p95_seconds"],
    *,
    coverage_sufficient: bool,
) -> GraphMetricSummary:
    unavailable = tuple(
        item
        for item in records
        if item.observation_state == "unknown"
        or (item.observation_state == "observed" and getattr(item, field) is None)
    )
    if unavailable:
        return GraphMetricSummary(
            state="unknown",
            value=None,
            aggregation="not_computable",
            contributing_source_record_ids=tuple(sorted(item.source_record_id for item in records)),
        )
    contributing = tuple(item for item in records if getattr(item, field) is not None)
    source_ids = tuple(sorted(item.source_record_id for item in contributing))
    if not contributing:
        if records and all(item.observation_state == "no_record" for item in records):
            state: MetricState = "no_record"
        else:
            state = "unknown"
        return GraphMetricSummary(
            state=state,
            value=None,
            aggregation="not_computable",
            contributing_source_record_ids=(),
        )
    aggregation: Literal["max_source_p95", "sum_unique_sources"]
    if field == "latency_p95_seconds":
        values = tuple(cast(float, item.latency_p95_seconds) for item in contributing)
        value = max(values)
        aggregation = "max_source_p95"
    else:
        count_total = _sum_graph_counts(contributing, field)
        value = cast(float, _canonical_finite_float(count_total))
        aggregation = "sum_unique_sources"
    if value == 0.0:
        state = (
            "verified_normal"
            if field == "error_count"
            and coverage_sufficient
            and any((item.request_count or 0) > 0 for item in contributing)
            else "zero"
        )
    else:
        state = "observed"
    return GraphMetricSummary(
        state=state,
        value=value,
        aggregation=aggregation,
        contributing_source_record_ids=source_ids,
    )


def _error_rate_summary(
    records: Sequence[GraphSourceRecord], *, coverage_sufficient: bool
) -> GraphMetricSummary:
    unavailable = tuple(
        item
        for item in records
        if item.observation_state == "unknown"
        or (
            item.observation_state == "observed"
            and (item.request_count is None or item.error_count is None)
        )
    )
    if unavailable:
        return GraphMetricSummary(
            state="unknown",
            value=None,
            aggregation="not_computable",
            contributing_source_record_ids=tuple(sorted(item.source_record_id for item in records)),
        )
    contributing = tuple(
        item for item in records if item.request_count is not None and item.error_count is not None
    )
    if not contributing:
        state: MetricState = (
            "no_record"
            if records and all(item.observation_state == "no_record" for item in records)
            else "unknown"
        )
        return GraphMetricSummary(
            state=state,
            value=None,
            aggregation="not_computable",
            contributing_source_record_ids=(),
        )
    requests = _sum_graph_counts(contributing, "request_count")
    errors = _sum_graph_counts(contributing, "error_count")
    if requests == 0:
        return GraphMetricSummary(
            state="unknown",
            value=None,
            aggregation="not_computable",
            contributing_source_record_ids=(),
        )
    value = errors / requests
    rate_state: MetricState = (
        "verified_normal" if value == 0.0 and coverage_sufficient else "observed"
    )
    return GraphMetricSummary(
        state=rate_state,
        value=value,
        aggregation="weighted_rate_from_counts",
        contributing_source_record_ids=tuple(
            sorted(item.source_record_id for item in contributing)
        ),
    )


def _sum_graph_counts(
    records: Sequence[GraphSourceRecord],
    field: Literal["request_count", "error_count"],
) -> int:
    total = 0
    for record in records:
        value = cast(int | None, getattr(record, field))
        if value is None:
            continue
        total += value
        if total > MAX_EXACT_GRAPH_COUNT:
            raise ValueError(
                f"aggregate {field} exceeds the float64 exact integer limit "
                f"of {MAX_EXACT_GRAPH_COUNT}"
            )
    return total


def _edge_sort_key(edge: GraphSnapshotEdge) -> tuple[str, str, str]:
    return (edge.relation_type, edge.caller_service_key, edge.callee_service_key)


def _build_edge(
    *,
    records: Sequence[GraphSourceRecord],
    interval: TimeInterval,
    coverage: GraphCoverageSummary,
) -> GraphSnapshotEdge:
    ordered = tuple(sorted(records, key=lambda item: item.source_record_id))
    first = ordered[0]
    observed = tuple(item for item in ordered if item.observation_state == "observed")
    if observed:
        observation_state: EdgeObservationState = "observed"
    elif all(item.observation_state == "no_record" for item in ordered):
        observation_state = "not_observed"
    else:
        observation_state = "unknown"
    unsigned: dict[str, object] = {
        "caller_service_key": first.caller_service_key,
        "callee_service_key": first.callee_service_key,
        "relation_type": first.relation_type,
        "direction": first.direction,
        "allows_directed_propagation": first.relation_type
        in {"received_dependency_failure", "caller_load_transmission"},
        "interval": interval.model_dump(mode="json"),
        "observation_state": observation_state,
        "coverage": coverage.model_dump(mode="json"),
        "traffic": _metric_summary(
            ordered,
            "request_count",
            coverage_sufficient=coverage.coverage_state == "sufficient",
        ).model_dump(mode="json"),
        "error_count": _metric_summary(
            ordered,
            "error_count",
            coverage_sufficient=coverage.coverage_state == "sufficient",
        ).model_dump(mode="json"),
        "error_rate": _error_rate_summary(
            ordered,
            coverage_sufficient=coverage.coverage_state == "sufficient",
        ).model_dump(mode="json"),
        "latency_p95_seconds": _metric_summary(
            ordered,
            "latency_p95_seconds",
            coverage_sufficient=coverage.coverage_state == "sufficient",
        ).model_dump(mode="json"),
        "source_record_ids": tuple(item.source_record_id for item in ordered),
        "provenance_ids": tuple(
            sorted({value for item in ordered for value in item.provenance_ids})
        ),
    }
    digest = _digest(unsigned)
    return GraphSnapshotEdge.model_validate(
        {**unsigned, "edge_id": f"gedge_{digest[:24]}", "edge_sha256": digest}
    )


def _build_snapshot(
    *,
    time_contract: CaseTimeContract,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    window_kind: WindowKind,
    window_coverage: GraphWindowCoverage,
    source_records: Sequence[GraphSourceRecord],
    minimum_coverage_fraction: float,
) -> GraphSnapshot:
    coverage = _coverage_summary(window_coverage, minimum_coverage_fraction)
    grouped: dict[tuple[str, str, str, str], list[GraphSourceRecord]] = defaultdict(list)
    for record in source_records:
        grouped[
            (
                record.relation_type,
                record.direction,
                record.caller_service_key,
                record.callee_service_key,
            )
        ].append(record)
    edges = tuple(
        sorted(
            (
                _build_edge(
                    records=records,
                    interval=window_coverage.interval,
                    coverage=coverage,
                )
                for records in grouped.values()
            ),
            key=_edge_sort_key,
        )
    )
    unsigned: dict[str, object] = {
        "schema_version": GRAPH_SNAPSHOT_SCHEMA_VERSION,
        "opaque_case_id": time_contract.opaque_case_id,
        "window_kind": window_kind,
        "interval": window_coverage.interval.model_dump(mode="json"),
        "decision_time_unix_ns": time_contract.decision_time_unix_ns,
        "observation_contract_sha256": candidate_universe.observation_contract_sha256,
        "candidate_universe_sha256": candidate_universe.universe_sha256,
        "candidate_freeze_sha256": frozen_candidates.freeze_sha256,
        "entity_registry_sha256": entity_registry.registry_sha256,
        "node_service_keys": frozen_candidates.candidate_service_keys,
        "coverage": coverage.model_dump(mode="json"),
        "edges": tuple(item.model_dump(mode="json") for item in edges),
        "provenance_ids": tuple(
            sorted(
                set(window_coverage.provenance_ids)
                | {value for item in edges for value in item.provenance_ids}
            )
        ),
    }
    return GraphSnapshot.model_validate({**unsigned, "snapshot_sha256": _digest(unsigned)})


def _edge_key(edge: GraphSnapshotEdge) -> tuple[str, str, str, str]:
    return (
        edge.relation_type,
        edge.direction,
        edge.caller_service_key,
        edge.callee_service_key,
    )


def _build_change(
    normal: GraphSnapshotEdge | None,
    fault: GraphSnapshotEdge | None,
    *,
    normal_coverage: CoverageState,
    fault_coverage: CoverageState,
) -> GraphEdgeChange:
    edge = normal if normal is not None else cast(GraphSnapshotEdge, fault)
    normal_presence = _edge_presence(normal, normal_coverage)
    fault_presence = _edge_presence(fault, fault_coverage)
    traffic_change = _metric_change(normal, fault, "traffic")
    error_count_change = _metric_change(normal, fault, "error_count")
    error_rate_change = _metric_change(normal, fault, "error_rate")
    latency_change = _metric_change(normal, fault, "latency_p95_seconds")
    if normal_presence != "observed" and fault_presence == "observed":
        state: EdgeChangeState = (
            "new" if normal_presence == "not_observed_sufficient_coverage" else "unknown"
        )
    elif normal_presence == "observed" and fault_presence != "observed":
        state = "removed" if fault_presence == "not_observed_sufficient_coverage" else "unknown"
    elif normal_presence != "observed" or fault_presence != "observed":
        state = "unknown"
    else:
        state = traffic_change
    unsigned: dict[str, object] = {
        "caller_service_key": edge.caller_service_key,
        "callee_service_key": edge.callee_service_key,
        "relation_type": edge.relation_type,
        "direction": edge.direction,
        "normal_edge_id": normal.edge_id if normal is not None else None,
        "fault_edge_id": fault.edge_id if fault is not None else None,
        "normal_presence": normal_presence,
        "fault_presence": fault_presence,
        "change_state": state,
        "traffic_change": traffic_change,
        "error_count_change": error_count_change,
        "error_rate_change": error_rate_change,
        "latency_change": latency_change,
    }
    digest = _digest(unsigned)
    return GraphEdgeChange.model_validate(
        {**unsigned, "change_id": f"gec_{digest[:24]}", "change_sha256": digest}
    )


def _edge_presence(edge: GraphSnapshotEdge | None, coverage: CoverageState) -> EdgePresence:
    if edge is not None and edge.observation_state == "observed":
        return "observed"
    if edge is not None and edge.observation_state == "unknown":
        return "unknown"
    if coverage == "sufficient":
        return "not_observed_sufficient_coverage"
    if coverage == "insufficient":
        return "not_observed_insufficient_coverage"
    return "unknown"


def _metric_change(
    normal: GraphSnapshotEdge | None,
    fault: GraphSnapshotEdge | None,
    field: Literal["traffic", "error_count", "error_rate", "latency_p95_seconds"],
) -> MetricChangeState:
    if normal is None or fault is None:
        return "unknown"
    normal_value = cast(GraphMetricSummary, getattr(normal, field)).value
    fault_value = cast(GraphMetricSummary, getattr(fault, field)).value
    if normal_value is None or fault_value is None:
        return "unknown"
    if fault_value > normal_value:
        return "increased"
    if fault_value < normal_value:
        return "decreased"
    return "stable"


def _validate_upstreams(
    *,
    time_contract: CaseTimeContract,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    window_coverages: Sequence[GraphWindowCoverage],
) -> dict[WindowKind, GraphWindowCoverage]:
    validated_time = CaseTimeContract.model_validate(
        CaseTimeContract.model_dump(time_contract, mode="json", warnings="error")
    )
    if validated_time != time_contract:
        raise ValueError("time contract is not exact")
    if not TemporalEntityRegistry.verify(entity_registry):
        raise ValueError("entity registry failed exact replay")
    if (
        entity_registry.opaque_case_id != time_contract.opaque_case_id
        or entity_registry.decision_time_unix_ns != time_contract.decision_time_unix_ns
    ):
        raise ValueError("entity registry and time contract scope differ")
    coverages: dict[WindowKind, GraphWindowCoverage] = {}
    expected_intervals = {
        "normal": time_contract.baseline_window,
        "fault": time_contract.fault_window,
    }
    for record in window_coverages:
        if type(record) is not GraphWindowCoverage or not GraphWindowCoverage.verify(record):
            raise ValueError("graph window coverage failed verification")
        if record.window_kind in coverages:
            raise ValueError("graph window coverage must contain each window once")
        if (
            record.opaque_case_id != time_contract.opaque_case_id
            or record.observation_contract_sha256 != candidate_universe.observation_contract_sha256
            or record.interval != expected_intervals[record.window_kind]
        ):
            raise ValueError("graph window coverage differs from frozen scope")
        coverages[record.window_kind] = record
    if set(coverages) != {"normal", "fault"}:
        raise ValueError("normal and fault graph coverage are both required")
    return coverages


def _entity_registry_service_keys(
    registry: TemporalEntityRegistry,
) -> set[str]:
    keys = {
        record.service_entity_key
        for record in registry.metadata_records
        if record.service_entity_key is not None
    }
    keys.update(
        binding.service_entity_key
        for binding in registry.timeline_bindings
        if binding.service_entity_key is not None
    )
    for binding in registry.timeline_bindings:
        keys.update(binding.conflict_service_keys)
    return keys


def _build_input_decisions(
    *,
    records: Sequence[GraphSourceRecord],
    time_contract: CaseTimeContract,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
) -> tuple[GraphInputDecision, ...]:
    occurrences: dict[str, tuple[GraphSourceRecord, int]] = {}
    for record in records:
        if type(record) is not GraphSourceRecord or not GraphSourceRecord.verify(record):
            raise ValueError("graph source record failed verification")
        previous = occurrences.get(record.source_record_sha256)
        occurrences[record.source_record_sha256] = (
            record,
            1 if previous is None else previous[1] + 1,
        )
    expected_intervals = {
        "normal": time_contract.baseline_window,
        "fault": time_contract.fault_window,
    }
    candidate_keys = set(frozen_candidates.candidate_service_keys)
    registry_keys = _entity_registry_service_keys(entity_registry)
    measurement_digests_by_source: dict[str, set[str]] = defaultdict(set)
    for record, _ in occurrences.values():
        measurement_digests_by_source[record.source_record_id].add(
            _digest(_measurement_payload(record))
        )
    conflicted_source_ids = {
        source_record_id
        for source_record_id, digests in measurement_digests_by_source.items()
        if len(digests) > 1
    }
    decisions = []
    for record, occurrence_count in occurrences.values():
        reasons: list[GraphInputReason] = []
        if record.opaque_case_id != time_contract.opaque_case_id:
            reasons.append("case_mismatch")
        if record.observation_contract_sha256 != candidate_universe.observation_contract_sha256:
            reasons.append("observation_contract_mismatch")
        if record.interval != expected_intervals[record.window_kind]:
            reasons.append("window_interval_mismatch")
        if record.interval.end_unix_ns > time_contract.decision_time_unix_ns:
            reasons.append("post_decision")
        endpoints = {record.caller_service_key, record.callee_service_key}
        if endpoints - candidate_keys:
            reasons.append("endpoint_outside_frozen_candidates")
        elif endpoints - registry_keys:
            reasons.append("endpoint_outside_entity_registry")
        if record.source_record_id in conflicted_source_ids:
            reasons.append("conflicting_duplicate_source")
        if reasons:
            disposition: GraphInputDisposition = "rejected"
        elif record.observation_state == "unknown":
            disposition = "unknown"
            reasons = ["unknown_observation"]
        else:
            disposition = "accepted"
            reasons = ["accepted"]
        unsigned: dict[str, object] = {
            "schema_version": GRAPH_INPUT_DECISION_SCHEMA_VERSION,
            "source_record_id": record.source_record_id,
            "source_record_sha256": record.source_record_sha256,
            "occurrence_count": occurrence_count,
            "disposition": disposition,
            "reason_codes": tuple(sorted(reasons)),
            "provenance_ids": record.provenance_ids,
        }
        digest = _digest(unsigned)
        decisions.append(
            GraphInputDecision.model_validate(
                {
                    **unsigned,
                    "decision_id": f"gid_{digest[:24]}",
                    "decision_sha256": digest,
                }
            )
        )
    return tuple(sorted(decisions, key=lambda item: item.source_record_sha256))


def _conflicted_source_record_ids(
    records: Sequence[GraphSourceRecord],
) -> tuple[str, ...]:
    measurement_digests_by_source: dict[str, set[str]] = defaultdict(set)
    for record in records:
        measurement_digests_by_source[record.source_record_id].add(
            _digest(_measurement_payload(record))
        )
    return tuple(
        sorted(
            source_record_id
            for source_record_id, digests in measurement_digests_by_source.items()
            if len(digests) > 1
        )
    )


def _measurement_payload(record: GraphSourceRecord) -> dict[str, object]:
    return record.model_dump(
        mode="json",
        exclude={
            "canonical_input_sha256s",
            "parent_source_sha256s",
            "provenance_ids",
            "source_record_sha256",
        },
    )


def _merge_retained_sources(
    records: Sequence[GraphSourceRecord],
) -> tuple[tuple[GraphSourceRecord, ...], tuple[str, ...]]:
    by_source_id: dict[str, list[GraphSourceRecord]] = defaultdict(list)
    for record in records:
        by_source_id[record.source_record_id].append(record)
    merged = []
    duplicates = []
    for source_record_id, members in sorted(by_source_id.items()):
        semantic_payloads = {_digest(_measurement_payload(item)) for item in members}
        if len(semantic_payloads) != 1:
            raise ValueError("conflicting graph records share one source identity")
        first = members[0]
        provenance_ids = tuple(sorted({value for item in members for value in item.provenance_ids}))
        canonical = _build_graph_source_record(
            opaque_case_id=first.opaque_case_id,
            observation_contract_sha256=first.observation_contract_sha256,
            window_kind=first.window_kind,
            interval=first.interval,
            caller_service_key=first.caller_service_key,
            callee_service_key=first.callee_service_key,
            relation_type=first.relation_type,
            direction=first.direction,
            observation_state=first.observation_state,
            request_count=first.request_count,
            error_count=first.error_count,
            latency_p95_seconds=first.latency_p95_seconds,
            source_fingerprint=first.source_fingerprint,
            canonical_input_sha256s=tuple(
                sorted({value for item in members for value in item.canonical_input_sha256s})
            ),
            parent_source_sha256s=tuple(
                sorted({value for item in members for value in item.parent_source_sha256s})
            ),
            provenance_ids=provenance_ids,
        )
        merged.append(canonical)
        if len(members) > 1:
            duplicates.append(source_record_id)
    return tuple(merged), tuple(duplicates)


def build_graph_snapshot_pair(
    *,
    time_contract: CaseTimeContract,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    graph_input_manifest: FrozenGraphInputManifest,
    expected_graph_input_manifest_sha256: str,
    minimum_coverage_fraction: float = 0.8,
) -> GraphSnapshotPair:
    """Build B/F snapshots and their deterministic topology/traffic changes."""

    base_context = _require_verified_graph_base_context(
        time_contract=time_contract,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
    )
    context = _require_verified_graph_input_context(
        graph_input_manifest,
        expected_graph_input_manifest_sha256=expected_graph_input_manifest_sha256,
    )
    return _build_graph_snapshot_pair_from_context(
        base_context=base_context,
        context=context,
        minimum_coverage_fraction=minimum_coverage_fraction,
    )


def _build_graph_snapshot_pair_from_context(
    *,
    base_context: _VerifiedGraphBaseContext,
    context: _VerifiedGraphInputContext,
    minimum_coverage_fraction: float = 0.8,
) -> GraphSnapshotPair:
    time_contract = base_context.time_contract
    candidate_universe = base_context.candidate_universe
    frozen_candidates = base_context.frozen_candidates
    entity_registry = base_context.entity_registry
    canonical_minimum = _canonical_finite_float(minimum_coverage_fraction)
    if canonical_minimum is None:
        raise ValueError("minimum coverage fraction must be a graph float")
    minimum_coverage_fraction = canonical_minimum
    if not 0.0 <= minimum_coverage_fraction <= 1.0:
        raise ValueError("minimum coverage fraction must be within [0, 1]")
    manifest = context.manifest
    case_scope = {
        time_contract.opaque_case_id,
        candidate_universe.opaque_case_id,
        frozen_candidates.opaque_case_id,
        entity_registry.opaque_case_id,
        manifest.opaque_case_id,
    }
    if len(case_scope) != 1:
        raise ValueError("graph replay inputs have mismatched case scope")
    if manifest.opaque_case_id != time_contract.opaque_case_id:
        raise ValueError("graph input manifest case does not match time contract")
    if manifest.observation_contract_sha256 != candidate_universe.observation_contract_sha256:
        raise ValueError("graph input manifest observation contract mismatch")
    canonical_coverage_inputs, canonical_source_inputs = _canonical_inputs_from_manifest(context)
    window_coverages = tuple(
        _build_graph_window_coverage_from_context(
            canonical_input=item,
            context=context,
        )
        for item in canonical_coverage_inputs
    )
    source_records = tuple(
        _build_graph_source_record_from_context(
            canonical_input=item,
            context=context,
        )
        for item in canonical_source_inputs
    )
    coverages = _validate_upstreams(
        time_contract=time_contract,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        window_coverages=window_coverages,
    )
    decisions = _build_input_decisions(
        records=source_records,
        time_contract=time_contract,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
    )
    conflicted_source_record_ids = _conflicted_source_record_ids(source_records)
    retained_sha256s = {
        item.source_record_sha256
        for item in decisions
        if item.disposition in {"accepted", "unknown"}
    }
    retained_sources = tuple(
        sorted(
            (item for item in source_records if item.source_record_sha256 in retained_sha256s),
            key=lambda item: item.source_record_sha256,
        )
    )
    ordered_sources, duplicate_source_record_ids = _merge_retained_sources(retained_sources)
    normal = _build_snapshot(
        time_contract=time_contract,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        window_kind="normal",
        window_coverage=coverages["normal"],
        source_records=tuple(item for item in ordered_sources if item.window_kind == "normal"),
        minimum_coverage_fraction=minimum_coverage_fraction,
    )
    fault = _build_snapshot(
        time_contract=time_contract,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        window_kind="fault",
        window_coverage=coverages["fault"],
        source_records=tuple(item for item in ordered_sources if item.window_kind == "fault"),
        minimum_coverage_fraction=minimum_coverage_fraction,
    )
    normal_by_key = {_edge_key(item): item for item in normal.edges}
    fault_by_key = {_edge_key(item): item for item in fault.edges}
    changes = tuple(
        _build_change(
            normal_by_key.get(key),
            fault_by_key.get(key),
            normal_coverage=normal.coverage.coverage_state,
            fault_coverage=fault.coverage.coverage_state,
        )
        for key in sorted(set(normal_by_key) | set(fault_by_key))
    )
    unsigned: dict[str, object] = {
        "schema_version": GRAPH_SNAPSHOT_PAIR_SCHEMA_VERSION,
        "opaque_case_id": time_contract.opaque_case_id,
        "graph_input_manifest_sha256": manifest.manifest_sha256,
        "time_contract_sha256": _digest(CaseTimeContract.model_dump(time_contract, mode="json")),
        "candidate_universe_sha256": candidate_universe.universe_sha256,
        "candidate_freeze_sha256": frozen_candidates.freeze_sha256,
        "entity_registry_sha256": entity_registry.registry_sha256,
        "minimum_coverage_fraction": minimum_coverage_fraction,
        "canonical_coverage_input_sha256s": (
            coverages["normal"].canonical_input_sha256,
            coverages["fault"].canonical_input_sha256,
        ),
        "canonical_source_input_sha256s": tuple(
            sorted(item.canonical_input_sha256 for item in canonical_source_inputs)
        ),
        "window_coverage_sha256s": (
            coverages["normal"].coverage_sha256,
            coverages["fault"].coverage_sha256,
        ),
        "source_record_sha256s": tuple(
            sorted(item.source_record_sha256 for item in source_records)
        ),
        "input_occurrence_count": len(source_records),
        "input_decisions": tuple(item.model_dump(mode="json") for item in decisions),
        "accepted_input_sha256s": tuple(
            item.source_record_sha256 for item in decisions if item.disposition == "accepted"
        ),
        "rejected_input_sha256s": tuple(
            item.source_record_sha256 for item in decisions if item.disposition == "rejected"
        ),
        "unknown_input_sha256s": tuple(
            item.source_record_sha256 for item in decisions if item.disposition == "unknown"
        ),
        "duplicate_source_record_ids": duplicate_source_record_ids,
        "conflicted_source_record_ids": conflicted_source_record_ids,
        "normal_snapshot": normal.model_dump(mode="json"),
        "fault_snapshot": fault.model_dump(mode="json"),
        "edge_changes": tuple(item.model_dump(mode="json") for item in changes),
    }
    result = GraphSnapshotPair.model_validate({**unsigned, "pair_sha256": _digest(unsigned)})
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("graph snapshot pair verification failed")
    return result
