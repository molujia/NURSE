"""Versioned contracts for mechanisms, hypotheses, verification, and guards."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, TypeGuard

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.observation_schemas import TimeInterval

MECHANISM_SCHEMA_VERSION: Literal["nurse.v2.mechanism-definition/1"] = (
    "nurse.v2.mechanism-definition/1"
)
PREDICATE_RESULT_SCHEMA_VERSION: Literal["nurse.v2.predicate-result/2"] = (
    "nurse.v2.predicate-result/2"
)
LEGACY_VERIFICATION_REQUEST_SCHEMA_VERSION: Literal["nurse.v2.verification-request/1"] = (
    "nurse.v2.verification-request/1"
)
VERIFICATION_REQUEST_SCHEMA_VERSION: Literal["nurse.v2.verification-request/2"] = (
    "nurse.v2.verification-request/2"
)
VERIFICATION_RESULT_SCHEMA_VERSION: Literal["nurse.v2.verification-result/1"] = (
    "nurse.v2.verification-result/1"
)
DIAGNOSTIC_HYPOTHESIS_SCHEMA_VERSION: Literal["nurse.v2.diagnostic-hypothesis/1"] = (
    "nurse.v2.diagnostic-hypothesis/1"
)
ROLE_ASSERTION_SCHEMA_VERSION: Literal["nurse.v2.role-assertion/1"] = "nurse.v2.role-assertion/1"
GUARD_DECISION_SCHEMA_VERSION: Literal["nurse.v2.guard-decision/1"] = "nurse.v2.guard-decision/1"

DiagnosticRole = Literal["root_support", "source_support", "symptom_support"]
TruthValue = Literal["true", "false", "unknown"]
PredicateRequirement = Literal["necessary", "optional", "refuting"]
LifecycleStatus = Literal["proposed", "accepted", "rejected", "conflicted"]
VerificationOperator = Literal[
    "compare_window_summary",
    "inspect_counter_reset",
    "summarize_log_pattern",
    "measure_trace_edge_change",
    "resolve_entity_binding",
    "measure_onset_interval",
]
GuardLayer = Literal[
    "structural_permission",
    "observation_reference",
    "mechanism_contract",
    "score_compilation",
]


class MechanismDefinition(StrictFrozenModel):
    """Label-free mechanism contract independent of benchmark fault types."""

    schema_version: Literal["nurse.v2.mechanism-definition/1"] = MECHANISM_SCHEMA_VERSION
    mechanism_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    mechanism_version: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    necessary_predicate_kinds: tuple[str, ...] = Field(min_length=1)
    optional_predicate_kinds: tuple[str, ...]
    refuting_predicate_kinds: tuple[str, ...]
    allowed_roles: tuple[DiagnosticRole, ...] = Field(min_length=1)
    definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_disjoint_canonical_contract(self) -> Self:
        collections = (
            self.necessary_predicate_kinds,
            self.optional_predicate_kinds,
            self.refuting_predicate_kinds,
            self.allowed_roles,
        )
        if any(values != tuple(sorted(set(values))) for values in collections):
            raise ValueError("mechanism contract fields must be sorted and unique")
        predicate_sets = (
            set(self.necessary_predicate_kinds),
            set(self.optional_predicate_kinds),
            set(self.refuting_predicate_kinds),
        )
        if any(
            predicate_sets[left] & predicate_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            raise ValueError("mechanism predicate roles must be disjoint")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"definition_sha256"})
        return self.definition_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_mechanism_definition(
    *,
    mechanism_key: str,
    mechanism_version: str,
    summary: str,
    necessary_predicate_kinds: Sequence[str],
    optional_predicate_kinds: Sequence[str],
    refuting_predicate_kinds: Sequence[str],
    allowed_roles: Sequence[DiagnosticRole],
) -> MechanismDefinition:
    """Build a canonical content-addressed mechanism definition."""

    unsigned: dict[str, object] = {
        "schema_version": MECHANISM_SCHEMA_VERSION,
        "mechanism_key": mechanism_key,
        "mechanism_version": mechanism_version,
        "summary": summary,
        "necessary_predicate_kinds": tuple(sorted(set(necessary_predicate_kinds))),
        "optional_predicate_kinds": tuple(sorted(set(optional_predicate_kinds))),
        "refuting_predicate_kinds": tuple(sorted(set(refuting_predicate_kinds))),
        "allowed_roles": tuple(sorted(set(allowed_roles))),
    }
    definition = MechanismDefinition.model_validate(
        {
            **unsigned,
            "definition_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not definition.verify():  # pragma: no cover - construction invariant
        raise AssertionError("mechanism definition verification failed")
    return definition


class MeasuredValue(StrictFrozenModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    value: float | int | bool | str
    unit: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def reject_non_finite_number(self) -> Self:
        if type(self.value) is float and not math.isfinite(self.value):
            raise ValueError("measured numeric value must be finite")
        return self


class PredicateThreshold(StrictFrozenModel):
    comparator: Literal["gt", "ge", "lt", "le", "eq", "ne", "exists"]
    value: float | int | bool | str | None = None
    unit: str | None = Field(default=None, min_length=1)
    source: Literal["adapter_manifest", "normal_data", "mechanism_constant"]

    @model_validator(mode="after")
    def require_threshold_value(self) -> Self:
        if self.comparator == "exists" and self.value is not None:
            raise ValueError("exists threshold cannot contain a value")
        if self.comparator != "exists" and self.value is None:
            raise ValueError("comparison threshold requires a value")
        if type(self.value) is float and not math.isfinite(self.value):
            raise ValueError("threshold numeric value must be finite")
        return self


_THRESHOLD_SOURCE_PREFIX = {
    "adapter_manifest": "adapter-manifest:",
    "normal_data": "normal-profile:",
    "mechanism_constant": "mechanism:",
}
_PENDING_REASONS = frozenset({"pending_verification", "no_bound_observations"})
_MAX_PREDICATE_DEPTH = 64
_MAX_PREDICATE_RECORDS = 4096


def derive_atomic_predicate_state(
    *,
    measured_values: Sequence[MeasuredValue],
    threshold: PredicateThreshold | None,
    threshold_source: str | None,
    observation_ids: Sequence[str],
    verification_result_ids: Sequence[str],
) -> tuple[TruthValue, str]:
    """Derive the only valid state for a completed atomic predicate."""

    values = tuple(measured_values)
    if threshold is None:
        return "unknown", "threshold_missing"
    if threshold_source is None:
        return "unknown", "threshold_source_missing"
    expected_prefix = _THRESHOLD_SOURCE_PREFIX[threshold.source]
    if not threshold_source.startswith(expected_prefix):
        return "unknown", "threshold_source_mismatch"
    if not threshold_source[len(expected_prefix) :].strip():
        return "unknown", "threshold_source_malformed"
    if not values:
        return "unknown", "measurement_missing"
    if len(values) != 1:
        return "unknown", "ambiguous_measurement"
    measured = values[0]
    if threshold.comparator != "exists" and measured.unit != threshold.unit:
        return "unknown", "unit_mismatch"
    comparison = _compare_predicate_value(measured.value, threshold)
    if comparison is None:
        return "unknown", "type_mismatch"
    if not observation_ids and not verification_result_ids:
        return "unknown", "evidence_reference_missing"
    truth_value: TruthValue = "true" if comparison else "false"
    return truth_value, f"comparison_{truth_value}:{measured.name}:{threshold.comparator}"


def _compare_predicate_value(
    measured: float | int | bool | str,
    threshold: PredicateThreshold,
) -> bool | None:
    if threshold.comparator == "exists":
        return True
    expected = threshold.value
    if expected is None:  # pragma: no cover - PredicateThreshold invariant
        return None
    if threshold.comparator in {"gt", "ge", "lt", "le"}:
        if not (_is_numeric(measured) and _is_numeric(expected)):
            return None
        if threshold.comparator == "gt":
            return measured > expected
        if threshold.comparator == "ge":
            return measured >= expected
        if threshold.comparator == "lt":
            return measured < expected
        return measured <= expected
    comparable = (
        (_is_numeric(measured) and _is_numeric(expected))
        or (type(measured) is bool and type(expected) is bool)
        or (type(measured) is str and type(expected) is str)
    )
    if not comparable:
        return None
    equal = measured == expected
    return equal if threshold.comparator == "eq" else not equal


def _is_numeric(value: object) -> TypeGuard[float | int]:
    return type(value) in (int, float)


class PredicateResult(StrictFrozenModel):
    """Three-valued predicate with measurements, threshold source, and reason."""

    schema_version: Literal["nurse.v2.predicate-result/2"] = PREDICATE_RESULT_SCHEMA_VERSION
    predicate_result_id: str = Field(pattern=r"^prd_[0-9a-f]{24}$")
    predicate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predicate_kind: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    requirement: PredicateRequirement
    truth_value: TruthValue
    logical_operator: Literal["atomic", "and", "or"]
    operand_states: tuple[PredicateResult, ...]
    observation_ids: tuple[str, ...]
    verification_result_ids: tuple[str, ...]
    measured_values: tuple[MeasuredValue, ...]
    threshold: PredicateThreshold | None
    threshold_source: str | None = Field(default=None, min_length=1)
    reason: str | None = Field(default=None, min_length=1)
    provenance_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_auditable_truth_value(self) -> Self:
        for values, name in (
            (self.observation_ids, "observation IDs"),
            (self.verification_result_ids, "verification result IDs"),
            (self.provenance_ids, "provenance IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        if self.measured_values != tuple(
            sorted(self.measured_values, key=lambda item: item.name)
        ) or len({item.name for item in self.measured_values}) != len(self.measured_values):
            raise ValueError("measured values must be canonically named")
        if self.operand_states != tuple(
            sorted(self.operand_states, key=lambda item: item.predicate_result_id)
        ) or len({item.predicate_result_id for item in self.operand_states}) != len(
            self.operand_states
        ):
            raise ValueError("operand states must be sorted and unique")
        if self.logical_operator == "atomic" and self.operand_states:
            raise ValueError("atomic predicate cannot contain operand states")
        if self.logical_operator != "atomic" and not self.operand_states:
            raise ValueError("logical predicate requires operand states")
        if self.logical_operator != "atomic" and (
            self.measured_values or self.threshold is not None or self.threshold_source is not None
        ):
            raise ValueError("logical predicate evidence is retained through operand states")
        if self.reason is None:
            if self.truth_value == "unknown":
                raise ValueError("unknown predicate requires a reason")
            raise ValueError("predicate requires a reason")
        if self.truth_value != "unknown" and not (
            self.observation_ids or self.verification_result_ids
        ):
            raise ValueError("decided predicate requires evidence references")
        if self.threshold is None and self.threshold_source is not None:
            raise ValueError("threshold source requires a threshold")
        if self.logical_operator == "atomic":
            pending = self.truth_value == "unknown" and self.reason in _PENDING_REASONS
            if pending:
                if (
                    self.measured_values
                    or self.threshold is not None
                    or self.threshold_source is not None
                    or self.verification_result_ids
                    or (self.reason == "pending_verification" and not self.observation_ids)
                    or (self.reason == "no_bound_observations" and self.observation_ids)
                ):
                    raise ValueError("pending predicate contains evaluated state")
            else:
                expected = derive_atomic_predicate_state(
                    measured_values=self.measured_values,
                    threshold=self.threshold,
                    threshold_source=self.threshold_source,
                    observation_ids=self.observation_ids,
                    verification_result_ids=self.verification_result_ids,
                )
                if (self.truth_value, self.reason) != expected:
                    raise ValueError("atomic predicate state is not deterministically derived")
        else:
            operand_truths = {item.truth_value for item in self.operand_states}
            if self.logical_operator == "and":
                expected_truth: TruthValue = (
                    "false"
                    if "false" in operand_truths
                    else "unknown"
                    if "unknown" in operand_truths
                    else "true"
                )
            else:
                expected_truth = (
                    "true"
                    if "true" in operand_truths
                    else "unknown"
                    if "unknown" in operand_truths
                    else "false"
                )
            if self.truth_value != expected_truth or self.reason != (
                f"strong_kleene_{self.logical_operator}:{expected_truth}"
            ):
                raise ValueError("logical predicate violates Strong Kleene semantics")
            expected_observations = tuple(
                sorted(
                    {
                        observation_id
                        for item in self.operand_states
                        for observation_id in item.observation_ids
                    }
                )
            )
            expected_results = tuple(
                sorted(
                    {
                        result_id
                        for item in self.operand_states
                        for result_id in item.verification_result_ids
                    }
                )
            )
            expected_provenance = tuple(
                sorted(
                    {
                        provenance_id
                        for item in self.operand_states
                        for provenance_id in item.provenance_ids
                    }
                )
            )
            if (
                self.observation_ids != expected_observations
                or self.verification_result_ids != expected_results
                or self.provenance_ids != expected_provenance
            ):
                raise ValueError("logical predicate lineage must equal the operand union")
        return self

    def verify(self) -> bool:
        """Verify the complete digest and its deterministic shortened identifier."""

        try:
            return _verify_predicate_tree(self)
        except (AttributeError, RecursionError, TypeError, ValueError):
            return False


def build_predicate_result(
    *,
    predicate_kind: str,
    requirement: PredicateRequirement,
    truth_value: TruthValue,
    logical_operator: Literal["atomic", "and", "or"],
    operand_states: Sequence[PredicateResult],
    observation_ids: Sequence[str],
    verification_result_ids: Sequence[str],
    measured_values: Sequence[MeasuredValue],
    threshold: PredicateThreshold | None,
    threshold_source: str | None,
    reason: str,
    provenance_ids: Sequence[str],
) -> PredicateResult:
    """Build a canonical, fully content-addressed three-valued predicate."""

    if not isinstance(operand_states, Sequence) or isinstance(operand_states, str | bytes):
        raise TypeError("operand_states must be a sequence")
    canonical_operands = tuple(operand_states)
    canonical_measured_values = tuple(measured_values)
    if any(type(item) is not PredicateResult or not item.verify() for item in canonical_operands):
        raise ValueError("predicate operands must be complete verified predicate records")
    if any(not _has_exact_model_fields(item, MeasuredValue) for item in canonical_measured_values):
        raise ValueError("measured values must be exact validated records")
    if threshold is not None and not _has_exact_model_fields(threshold, PredicateThreshold):
        raise ValueError("threshold must be an exact validated record")
    if any(
        not _has_revalidated_model_content(item, MeasuredValue)
        for item in canonical_measured_values
    ):
        raise ValueError("measured values must be exact validated records")
    if threshold is not None and not _has_revalidated_model_content(threshold, PredicateThreshold):
        raise ValueError("threshold must be an exact validated record")
    record_count = 1
    tree_depth = 0
    for operand in canonical_operands:
        operand_count, operand_depth = _predicate_tree_shape(operand)
        record_count += operand_count
        tree_depth = max(tree_depth, operand_depth + 1)
    if record_count > _MAX_PREDICATE_RECORDS or tree_depth > _MAX_PREDICATE_DEPTH:
        raise ValueError("predicate tree capacity exceeded")
    unsigned: dict[str, object] = {
        "schema_version": PREDICATE_RESULT_SCHEMA_VERSION,
        "predicate_kind": predicate_kind,
        "requirement": requirement,
        "truth_value": truth_value,
        "logical_operator": logical_operator,
        "operand_states": tuple(
            item.model_dump(mode="json")
            for item in sorted(canonical_operands, key=lambda item: item.predicate_result_id)
        ),
        "observation_ids": tuple(sorted(set(observation_ids))),
        "verification_result_ids": tuple(sorted(set(verification_result_ids))),
        "measured_values": tuple(
            item.model_dump(mode="json")
            for item in sorted(canonical_measured_values, key=lambda item: item.name)
        ),
        "threshold": None if threshold is None else threshold.model_dump(mode="json"),
        "threshold_source": threshold_source,
        "reason": reason,
        "provenance_ids": tuple(sorted(set(provenance_ids))),
    }
    digest = sha256_hex(canonical_json_bytes(unsigned))
    result = PredicateResult.model_validate(
        {
            **unsigned,
            "predicate_result_id": f"prd_{digest[:24]}",
            "predicate_sha256": digest,
        }
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("predicate result verification failed")
    return result


def _verify_predicate_tree(root: PredicateResult) -> bool:
    """Verify a bounded acyclic tree and every nested record digest."""

    stack: list[tuple[PredicateResult, int, bool]] = [(root, 0, False)]
    active_ids: set[int] = set()
    records: list[PredicateResult] = []
    nested_components: list[tuple[StrictFrozenModel, type[StrictFrozenModel]]] = []
    while stack:
        record, depth, exiting = stack.pop()
        record_identity = id(record)
        if exiting:
            active_ids.remove(record_identity)
            continue
        if (
            type(record) is not PredicateResult
            or depth > _MAX_PREDICATE_DEPTH
            or len(records) >= _MAX_PREDICATE_RECORDS
            or record_identity in active_ids
            or set(vars(record)) != set(type(record).model_fields)
            or type(record.operand_states) is not tuple
            or type(record.measured_values) is not tuple
        ):
            return False
        for measured in record.measured_values:
            if not _has_exact_model_fields(measured, MeasuredValue):
                return False
            nested_components.append((measured, MeasuredValue))
        if record.threshold is not None:
            if not _has_exact_model_fields(record.threshold, PredicateThreshold):
                return False
            nested_components.append((record.threshold, PredicateThreshold))
        active_ids.add(record_identity)
        records.append(record)
        stack.append((record, depth, True))
        stack.extend((item, depth + 1, False) for item in reversed(record.operand_states))

    for component, model_type in nested_components:
        if not _has_revalidated_model_content(component, model_type):
            return False
    serialized = root.model_dump(mode="python", warnings="error")
    revalidated = PredicateResult.model_validate(serialized)
    if revalidated != root:
        return False
    for record in reversed(records):
        unsigned = record.model_dump(
            mode="json",
            exclude={"predicate_result_id", "predicate_sha256"},
            warnings="error",
        )
        digest = sha256_hex(canonical_json_bytes(unsigned))
        if record.predicate_sha256 != digest or record.predicate_result_id != f"prd_{digest[:24]}":
            return False
    return True


def _has_exact_model_fields(
    value: object,
    model_type: type[StrictFrozenModel],
) -> bool:
    return type(value) is model_type and set(vars(value)) == set(model_type.model_fields)


def _has_revalidated_model_content(
    value: StrictFrozenModel,
    model_type: type[StrictFrozenModel],
) -> bool:
    try:
        revalidated = model_type.model_validate(value.model_dump(mode="python", warnings="error"))
    except (TypeError, ValueError):
        return False
    return revalidated == value


def _predicate_tree_shape(root: PredicateResult) -> tuple[int, int]:
    """Return serialized record count and maximum depth for a verified tree."""

    record_count = 0
    maximum_depth = 0
    stack = [(root, 0)]
    while stack:
        record, depth = stack.pop()
        record_count += 1
        maximum_depth = max(maximum_depth, depth)
        stack.extend((item, depth + 1) for item in record.operand_states)
    return record_count, maximum_depth


class VerificationRequest(StrictFrozenModel):
    """Deduplicable read-only query constrained to case, candidate, and time."""

    schema_version: Literal["nurse.v2.verification-request/2"] = VERIFICATION_REQUEST_SCHEMA_VERSION
    verification_request_id: str = Field(pattern=r"^vrq_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verification_scope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_service_key: str = Field(min_length=1)
    operator: VerificationOperator
    signal_family: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )
    query_interval: TimeInterval
    decision_time_unix_ns: int = Field(ge=0)
    observation_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    requested_fields: tuple[str, ...] = Field(min_length=1)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_legal_canonical_query(self) -> Self:
        if self.query_interval.end_unix_ns > self.decision_time_unix_ns:
            raise ValueError("post-decision verification is forbidden")
        for values, name in (
            (self.observation_ids, "observation IDs"),
            (self.relation_ids, "relation IDs"),
            (self.requested_fields, "requested fields"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(
            mode="json",
            exclude={"verification_request_id", "request_sha256"},
        )
        digest = sha256_hex(canonical_json_bytes(unsigned))
        return (
            self.request_sha256 == digest and self.verification_request_id == f"vrq_{digest[:24]}"
        )


def build_verification_request(
    *,
    opaque_case_id: str,
    candidate_freeze_sha256: str,
    verification_scope_sha256: str,
    candidate_service_key: str,
    operator: VerificationOperator,
    signal_family: str | None,
    query_interval: TimeInterval,
    decision_time_unix_ns: int,
    observation_ids: Sequence[str],
    relation_ids: Sequence[str],
    requested_fields: Sequence[str],
) -> VerificationRequest:
    """Normalize a legal query so semantically identical requests share an ID."""

    unsigned: dict[str, object] = {
        "schema_version": VERIFICATION_REQUEST_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "candidate_freeze_sha256": candidate_freeze_sha256,
        "verification_scope_sha256": verification_scope_sha256,
        "candidate_service_key": candidate_service_key,
        "operator": operator,
        "signal_family": signal_family,
        "query_interval": query_interval.model_dump(mode="json"),
        "decision_time_unix_ns": decision_time_unix_ns,
        "observation_ids": tuple(sorted(set(observation_ids))),
        "relation_ids": tuple(sorted(set(relation_ids))),
        "requested_fields": tuple(sorted(set(requested_fields))),
    }
    digest = sha256_hex(canonical_json_bytes(unsigned))
    request = VerificationRequest.model_validate(
        {
            **unsigned,
            "verification_request_id": f"vrq_{digest[:24]}",
            "request_sha256": digest,
        }
    )
    if not request.verify():  # pragma: no cover - construction invariant
        raise AssertionError("verification request verification failed")
    return request


class VerificationResult(StrictFrozenModel):
    """Immutable deterministic result or explicit bounded-query rejection."""

    schema_version: Literal["nurse.v2.verification-result/1"] = VERIFICATION_RESULT_SCHEMA_VERSION
    verification_result_id: str = Field(pattern=r"^vrs_[0-9a-f]{24}$")
    verification_request_id: str = Field(pattern=r"^vrq_[0-9a-f]{24}$")
    status: Literal["completed", "rejected"]
    truth_value: TruthValue
    derived_observation_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    reason: str = Field(min_length=1)
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_consistent_result(self) -> Self:
        for values, name in (
            (self.derived_observation_ids, "derived observation IDs"),
            (self.source_ids, "source IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        if self.status == "rejected" and (
            self.truth_value != "unknown" or self.derived_observation_ids
        ):
            raise ValueError("rejected verification must remain unknown without output")
        if self.status == "completed" and not self.source_ids:
            raise ValueError("completed verification requires source provenance")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(
            mode="json",
            exclude={"verification_result_id", "result_sha256"},
        )
        digest = sha256_hex(canonical_json_bytes(unsigned))
        return self.result_sha256 == digest and self.verification_result_id == f"vrs_{digest[:24]}"


def build_verification_result(
    *,
    verification_request_id: str,
    status: Literal["completed", "rejected"],
    truth_value: TruthValue,
    derived_observation_ids: Sequence[str],
    source_ids: Sequence[str],
    reason: str,
) -> VerificationResult:
    unsigned: dict[str, object] = {
        "schema_version": VERIFICATION_RESULT_SCHEMA_VERSION,
        "verification_request_id": verification_request_id,
        "status": status,
        "truth_value": truth_value,
        "derived_observation_ids": tuple(sorted(set(derived_observation_ids))),
        "source_ids": tuple(sorted(set(source_ids))),
        "reason": reason,
    }
    digest = sha256_hex(canonical_json_bytes(unsigned))
    result = VerificationResult.model_validate(
        {
            **unsigned,
            "verification_result_id": f"vrs_{digest[:24]}",
            "result_sha256": digest,
        }
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("verification result verification failed")
    return result


class DiagnosticHypothesis(StrictFrozenModel):
    """One competing, lifecycle-tracked mechanism for one candidate."""

    schema_version: Literal["nurse.v2.diagnostic-hypothesis/1"] = (
        DIAGNOSTIC_HYPOTHESIS_SCHEMA_VERSION
    )
    hypothesis_id: str = Field(pattern=r"^hyp_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_service_key: str = Field(min_length=1)
    mechanism_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    mechanism_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    origin: Literal["deterministic_rule", "semantic_agent", "history_hint", "operator"]
    origin_version: str = Field(min_length=1)
    observation_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    event_group_ids: tuple[str, ...]
    time_intervals: tuple[TimeInterval, ...] = Field(min_length=1)
    necessary_predicate_result_ids: tuple[str, ...] = Field(min_length=1)
    optional_predicate_result_ids: tuple[str, ...]
    refuting_predicate_result_ids: tuple[str, ...]
    alternative_hypothesis_ids: tuple[str, ...]
    verification_request_ids: tuple[str, ...]
    unknown_reasons: tuple[str, ...]
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    lifecycle_status: LifecycleStatus
    hypothesis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_hypothesis(self) -> Self:
        collections = (
            self.observation_ids,
            self.relation_ids,
            self.event_group_ids,
            self.necessary_predicate_result_ids,
            self.optional_predicate_result_ids,
            self.refuting_predicate_result_ids,
            self.alternative_hypothesis_ids,
            self.verification_request_ids,
            self.unknown_reasons,
            self.provenance_ids,
        )
        if any(values != tuple(sorted(set(values))) for values in collections):
            raise ValueError("hypothesis references must be sorted and unique")
        if self.time_intervals != tuple(
            sorted(
                self.time_intervals,
                key=lambda item: (item.start_unix_ns, item.end_unix_ns),
            )
        ):
            raise ValueError("hypothesis intervals must be canonically ordered")
        predicate_sets = (
            set(self.necessary_predicate_result_ids),
            set(self.optional_predicate_result_ids),
            set(self.refuting_predicate_result_ids),
        )
        if any(
            predicate_sets[left] & predicate_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            raise ValueError("predicate result roles must be disjoint")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(
            mode="json",
            exclude={"hypothesis_id", "hypothesis_sha256"},
        )
        digest = sha256_hex(canonical_json_bytes(unsigned))
        return self.hypothesis_sha256 == digest and self.hypothesis_id == f"hyp_{digest[:24]}"


def build_diagnostic_hypothesis(
    *,
    opaque_case_id: str,
    candidate_service_key: str,
    mechanism_key: str,
    mechanism_definition_sha256: str,
    origin: Literal["deterministic_rule", "semantic_agent", "history_hint", "operator"],
    origin_version: str,
    observation_ids: Sequence[str],
    relation_ids: Sequence[str],
    event_group_ids: Sequence[str],
    time_intervals: Sequence[TimeInterval],
    necessary_predicate_result_ids: Sequence[str],
    optional_predicate_result_ids: Sequence[str],
    refuting_predicate_result_ids: Sequence[str],
    alternative_hypothesis_ids: Sequence[str],
    verification_request_ids: Sequence[str],
    unknown_reasons: Sequence[str],
    provenance_ids: Sequence[str],
    lifecycle_status: LifecycleStatus,
) -> DiagnosticHypothesis:
    unsigned: dict[str, object] = {
        "schema_version": DIAGNOSTIC_HYPOTHESIS_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "candidate_service_key": candidate_service_key,
        "mechanism_key": mechanism_key,
        "mechanism_definition_sha256": mechanism_definition_sha256,
        "origin": origin,
        "origin_version": origin_version,
        "observation_ids": tuple(sorted(set(observation_ids))),
        "relation_ids": tuple(sorted(set(relation_ids))),
        "event_group_ids": tuple(sorted(set(event_group_ids))),
        "time_intervals": tuple(
            item.model_dump(mode="json")
            for item in sorted(
                time_intervals,
                key=lambda item: (item.start_unix_ns, item.end_unix_ns),
            )
        ),
        "necessary_predicate_result_ids": tuple(sorted(set(necessary_predicate_result_ids))),
        "optional_predicate_result_ids": tuple(sorted(set(optional_predicate_result_ids))),
        "refuting_predicate_result_ids": tuple(sorted(set(refuting_predicate_result_ids))),
        "alternative_hypothesis_ids": tuple(sorted(set(alternative_hypothesis_ids))),
        "verification_request_ids": tuple(sorted(set(verification_request_ids))),
        "unknown_reasons": tuple(sorted(set(unknown_reasons))),
        "provenance_ids": tuple(sorted(set(provenance_ids))),
        "lifecycle_status": lifecycle_status,
    }
    digest = sha256_hex(canonical_json_bytes(unsigned))
    hypothesis = DiagnosticHypothesis.model_validate(
        {
            **unsigned,
            "hypothesis_id": f"hyp_{digest[:24]}",
            "hypothesis_sha256": digest,
        }
    )
    if not hypothesis.verify():  # pragma: no cover - construction invariant
        raise AssertionError("diagnostic hypothesis verification failed")
    return hypothesis


class RoleAssertion(StrictFrozenModel):
    """Event-, relation-, and interval-bound diagnostic score role."""

    schema_version: Literal["nurse.v2.role-assertion/1"] = ROLE_ASSERTION_SCHEMA_VERSION
    assertion_id: str = Field(pattern=r"^rol_[0-9a-f]{24}$")
    hypothesis_id: str = Field(pattern=r"^hyp_[0-9a-f]{24}$")
    candidate_service_key: str = Field(min_length=1)
    event_group_id: str = Field(pattern=r"^evt_[0-9a-f]{24}$")
    relation_id: str = Field(min_length=1)
    interval: TimeInterval
    role: DiagnosticRole
    strength: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    score_direction: Literal["promote", "neutral", "demote"]
    predicate_result_ids: tuple[str, ...] = Field(min_length=1)
    lifecycle_status: LifecycleStatus
    reason: str = Field(min_length=1)
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    assertion_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_bounded_role(self) -> Self:
        for values, name in (
            (self.predicate_result_ids, "predicate result IDs"),
            (self.provenance_ids, "provenance IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        if self.role == "symptom_support" and self.score_direction == "demote":
            raise ValueError("symptom role cannot demote a candidate")
        if self.lifecycle_status in {"rejected", "conflicted"} and (
            self.score_direction != "neutral"
        ):
            raise ValueError("rejected or conflicted role must be score-neutral")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(
            mode="json",
            exclude={"assertion_id", "assertion_sha256"},
        )
        digest = sha256_hex(canonical_json_bytes(unsigned))
        return self.assertion_sha256 == digest and self.assertion_id == (f"rol_{digest[:24]}")


def build_role_assertion(
    *,
    hypothesis_id: str,
    candidate_service_key: str,
    event_group_id: str,
    relation_id: str,
    interval: TimeInterval,
    role: DiagnosticRole,
    strength: float,
    score_direction: Literal["promote", "neutral", "demote"],
    predicate_result_ids: Sequence[str],
    lifecycle_status: LifecycleStatus,
    reason: str,
    provenance_ids: Sequence[str],
) -> RoleAssertion:
    unsigned: dict[str, object] = {
        "schema_version": ROLE_ASSERTION_SCHEMA_VERSION,
        "hypothesis_id": hypothesis_id,
        "candidate_service_key": candidate_service_key,
        "event_group_id": event_group_id,
        "relation_id": relation_id,
        "interval": interval.model_dump(mode="json"),
        "role": role,
        "strength": strength,
        "score_direction": score_direction,
        "predicate_result_ids": tuple(sorted(set(predicate_result_ids))),
        "lifecycle_status": lifecycle_status,
        "reason": reason,
        "provenance_ids": tuple(sorted(set(provenance_ids))),
    }
    digest = sha256_hex(canonical_json_bytes(unsigned))
    assertion = RoleAssertion.model_validate(
        {
            **unsigned,
            "assertion_id": f"rol_{digest[:24]}",
            "assertion_sha256": digest,
        }
    )
    if not assertion.verify():  # pragma: no cover - construction invariant
        raise AssertionError("role assertion verification failed")
    return assertion


class GuardDecision(StrictFrozenModel):
    """One immutable decision in the fixed four-layer semantic guard."""

    schema_version: Literal["nurse.v2.guard-decision/1"] = GUARD_DECISION_SCHEMA_VERSION
    guard_decision_id: str = Field(pattern=r"^grd_[0-9a-f]{24}$")
    hypothesis_id: str = Field(pattern=r"^hyp_[0-9a-f]{24}$")
    layer: GuardLayer
    sequence: int = Field(ge=1, le=4)
    disposition: LifecycleStatus
    submitted_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason_code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    prior_accepted_layers: int = Field(ge=0, le=3)
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_fixed_layer_order(self) -> Self:
        expected = {
            "structural_permission": 1,
            "observation_reference": 2,
            "mechanism_contract": 3,
            "score_compilation": 4,
        }[self.layer]
        if self.sequence != expected:
            raise ValueError("guard sequence does not match layer")
        if self.prior_accepted_layers > self.sequence - 1:
            raise ValueError("guard cannot accept nonexistent prior layers")
        if self.disposition == "accepted" and (self.prior_accepted_layers != self.sequence - 1):
            raise ValueError("accepted guard requires every prior layer")
        if self.provenance_ids != tuple(sorted(set(self.provenance_ids))):
            raise ValueError("guard provenance IDs must be sorted and unique")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(
            mode="json",
            exclude={"guard_decision_id", "decision_sha256"},
        )
        digest = sha256_hex(canonical_json_bytes(unsigned))
        return self.decision_sha256 == digest and self.guard_decision_id == (f"grd_{digest[:24]}")


def build_guard_decision(
    *,
    hypothesis_id: str,
    layer: GuardLayer,
    sequence: int,
    disposition: LifecycleStatus,
    submitted_record_sha256: str,
    reason_code: str,
    prior_accepted_layers: int,
    provenance_ids: Sequence[str],
) -> GuardDecision:
    unsigned: dict[str, object] = {
        "schema_version": GUARD_DECISION_SCHEMA_VERSION,
        "hypothesis_id": hypothesis_id,
        "layer": layer,
        "sequence": sequence,
        "disposition": disposition,
        "submitted_record_sha256": submitted_record_sha256,
        "reason_code": reason_code,
        "prior_accepted_layers": prior_accepted_layers,
        "provenance_ids": tuple(sorted(set(provenance_ids))),
    }
    digest = sha256_hex(canonical_json_bytes(unsigned))
    decision = GuardDecision.model_validate(
        {
            **unsigned,
            "guard_decision_id": f"grd_{digest[:24]}",
            "decision_sha256": digest,
        }
    )
    if not decision.verify():  # pragma: no cover - construction invariant
        raise AssertionError("guard decision verification failed")
    return decision
