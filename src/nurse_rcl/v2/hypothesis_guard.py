"""Ordered, replayable guards for score-affecting diagnostic hypotheses.

The guard compiles only a bounded event-role impact authorization.  Numeric
event strength and role correction remain deterministic downstream work; a
caller can never supply a score, weight, vote, or free-text impact here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Literal, NamedTuple, TypeAlias, cast

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.candidate_freeze import (
    FrozenCandidateUniverse,
    require_frozen_candidate_universe,
)
from nurse_rcl.v2.candidate_universe import build_complete_candidate_universe
from nurse_rcl.v2.context_expansion import CardExpansionPolicy
from nurse_rcl.v2.diagnostic_schemas import (
    DiagnosticHypothesis,
    DiagnosticRole,
    GuardDecision,
    GuardLayer,
    MeasuredValue,
    PredicateResult,
    VerificationResult,
    build_guard_decision,
)
from nurse_rcl.v2.event_grouping import EventGroup, EventGroupingResult
from nurse_rcl.v2.mechanism_library import MechanismLibrary
from nurse_rcl.v2.observation_schemas import CandidateUniverse, Observation, TimeInterval
from nurse_rcl.v2.query_session import (
    CaseQuerySessionCoordinator,
    CaseQuerySessionResult,
    require_case_query_session_replay,
)
from nurse_rcl.v2.semantic_context import SemanticContextBundle
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry
from nurse_rcl.v2.verification_executor import (
    DerivedVerificationObservation,
    _reject_hidden_cyclic_or_deep,
)

SCORE_IMPACT_COMPILATION_SCHEMA_VERSION: Literal["nurse.v2.score-impact-compilation/1"] = (
    "nurse.v2.score-impact-compilation/1"
)
HYPOTHESIS_GUARD_RESULT_SCHEMA_VERSION: Literal["nurse.v2.hypothesis-guard-result/1"] = (
    "nurse.v2.hypothesis-guard-result/1"
)

GuardFinalDisposition: TypeAlias = Literal["accepted", "rejected"]
PREDICATE_EVIDENCE_CONTRACT_VERSION: Literal["nurse-v2-predicate-evidence-contract/1"] = (
    "nurse-v2-predicate-evidence-contract/1"
)

_ORDERED_LAYERS: tuple[GuardLayer, ...] = (
    "structural_permission",
    "observation_reference",
    "mechanism_contract",
    "score_compilation",
)
_FORBIDDEN_IMPACT_FIELDS = frozenset(
    {
        "confidence",
        "custom_score",
        "free_text_impact",
        "impact",
        "numeric_weight",
        "probability",
        "score",
        "score_impact",
        "vote",
        "votes",
        "weight",
    }
)


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _stable_id(prefix: str, unsigned: object) -> tuple[str, str]:
    digest = _digest(unsigned)
    return f"{prefix}_{digest[:24]}", digest


class _PredicateEvidenceContract(NamedTuple):
    primary_signal_families: tuple[str, ...]
    source_signal_families: tuple[str, ...]
    measured_name: str
    comparator: str
    threshold_value: float | int
    threshold_unit: str
    threshold_source: str


_ALL_PRIMARY_SIGNAL_FAMILIES = (
    "cpu",
    "dependency_failure",
    "error",
    "latency",
    "lifecycle",
    "memory",
    "network",
    "request_load",
    "storage",
    "unclassified",
)


def _strength_contract(
    families: Sequence[str], source: str, source_families: Sequence[str] = ()
) -> _PredicateEvidenceContract:
    return _PredicateEvidenceContract(
        primary_signal_families=tuple(sorted(families)),
        source_signal_families=tuple(sorted(source_families)),
        measured_name="anomaly_strength",
        comparator="gt",
        threshold_value=0.5,
        threshold_unit="ratio",
        threshold_source=source,
    )


def _normal_contract(
    families: Sequence[str], source: str, source_families: Sequence[str] = ()
) -> _PredicateEvidenceContract:
    return _PredicateEvidenceContract(
        primary_signal_families=tuple(sorted(families)),
        source_signal_families=tuple(sorted(source_families)),
        measured_name="anomaly_strength",
        comparator="le",
        threshold_value=0.5,
        threshold_unit="ratio",
        threshold_source=source,
    )


_PREDICATE_EVIDENCE_CONTRACTS: Mapping[str, _PredicateEvidenceContract] = {
    "dependency_failure_precedes_local_symptom": _PredicateEvidenceContract(
        ("dependency_failure", "error", "latency"),
        (),
        "onset_delay_unix_ns",
        "lt",
        0,
        "ns",
        "mechanism:dependency-failure-precedence/1",
    ),
    "incoming_load_precedes_local_pressure": _PredicateEvidenceContract(
        ("cpu", "memory", "request_load"),
        (),
        "onset_delay_unix_ns",
        "lt",
        0,
        "ns",
        "mechanism:incoming-load-precedence/1",
    ),
    "incoming_request_rate_increase": _strength_contract(
        ("request_load",), "mechanism:incoming-request-rate/1"
    ),
    "incoming_request_rate_verified_normal": _normal_contract(
        ("request_load",), "mechanism:incoming-request-rate/1"
    ),
    "lifecycle_transition_observed": _strength_contract(
        ("lifecycle",), "mechanism:lifecycle-transition/1"
    ),
    "local_anomaly_observed": _strength_contract(
        _ALL_PRIMARY_SIGNAL_FAMILIES, "mechanism:local-anomaly/1"
    ),
    "local_error_increase": _strength_contract(("error",), "mechanism:local-error/1"),
    "local_pressure_observed": _strength_contract(
        ("cpu", "memory"),
        "mechanism:local-pressure/1",
        ("cpu_usage", "execution_pressure", "memory_usage", "process_usage", "thread_usage"),
    ),
    "local_resource_and_execution_verified_normal": _normal_contract(
        ("cpu", "memory"),
        "mechanism:local-pressure/1",
        ("cpu_usage", "execution_pressure", "memory_usage", "process_usage", "thread_usage"),
    ),
    "local_signals_verified_normal": _normal_contract(
        _ALL_PRIMARY_SIGNAL_FAMILIES, "mechanism:local-signals/1"
    ),
    "local_storage_and_network_verified_normal": _normal_contract(
        ("network", "storage"), "mechanism:local-storage-network/1"
    ),
    "local_storage_or_network_failure_observed": _strength_contract(
        ("network", "storage"), "mechanism:local-storage-network/1"
    ),
    "outgoing_dependency_calls_verified_normal": _normal_contract(
        ("dependency_failure",), "mechanism:outgoing-dependency-failure/1"
    ),
    "outgoing_dependency_failure_observed": _strength_contract(
        ("dependency_failure",), "mechanism:outgoing-dependency-failure/1"
    ),
    "request_failure_increase": _strength_contract(("error",), "mechanism:request-failure/1"),
    "request_latency_increase": _strength_contract(("latency",), "mechanism:request-latency/1"),
    "workload_availability_verified_normal": _normal_contract(
        ("lifecycle",), "mechanism:workload-unavailability/1"
    ),
    "workload_unavailability_observed": _strength_contract(
        ("lifecycle",), "mechanism:workload-unavailability/1"
    ),
}
_PREDICATE_EVIDENCE_CONTRACT_SHA256 = _digest(
    {
        "version": PREDICATE_EVIDENCE_CONTRACT_VERSION,
        "predicates": {
            key: contract._asdict()
            for key, contract in sorted(_PREDICATE_EVIDENCE_CONTRACTS.items())
        },
    }
)


class ScoreImpactCompilation(StrictFrozenModel):
    """Exact authorization consumed later by the event-role score compiler."""

    schema_version: Literal["nurse.v2.score-impact-compilation/1"] = (
        SCORE_IMPACT_COMPILATION_SCHEMA_VERSION
    )
    score_impact_id: str = Field(pattern=r"^sci_[0-9a-f]{24}$")
    hypothesis_id: str = Field(pattern=r"^hyp_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_service_key: str = Field(min_length=1)
    mechanism_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    mechanism_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_group_ids: tuple[str, ...] = Field(min_length=1)
    time_intervals: tuple[TimeInterval, ...] = Field(min_length=1)
    necessary_predicate_result_ids: tuple[str, ...] = Field(min_length=1)
    optional_predicate_result_ids: tuple[str, ...]
    refuting_predicate_result_ids: tuple[str, ...]
    verification_request_ids: tuple[str, ...]
    predicate_contract_version: Literal["nurse-v2-predicate-evidence-contract/1"] = (
        PREDICATE_EVIDENCE_CONTRACT_VERSION
    )
    predicate_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    allowed_roles: tuple[DiagnosticRole, ...] = Field(min_length=1)
    effect_policy: Literal["reference_event_role_formula"] = "reference_event_role_formula"
    causal_status: Literal["observable_contract_only"] = "observable_contract_only"
    caller_numeric_weight_allowed: Literal[False] = False
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    compilation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_compilation(self) -> Self:
        if self.predicate_contract_sha256 != _PREDICATE_EVIDENCE_CONTRACT_SHA256:
            raise ValueError("score-impact predicate contract digest differs")
        for values, name in (
            (self.event_group_ids, "event group IDs"),
            (self.necessary_predicate_result_ids, "necessary predicate IDs"),
            (self.optional_predicate_result_ids, "optional predicate IDs"),
            (self.refuting_predicate_result_ids, "refuting predicate IDs"),
            (self.verification_request_ids, "verification request IDs"),
            (self.allowed_roles, "allowed roles"),
            (self.provenance_ids, "provenance IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        if self.time_intervals != tuple(
            sorted(
                self.time_intervals,
                key=lambda item: (item.start_unix_ns, item.end_unix_ns),
            )
        ):
            raise ValueError("score-impact intervals must be canonically ordered")
        return self

    def verify(self) -> bool:
        try:
            value = ScoreImpactCompilation.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = value.model_dump(
                mode="json", exclude={"score_impact_id", "compilation_sha256"}
            )
            identifier, digest = _stable_id("sci", unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            value == self
            and value.score_impact_id == identifier
            and value.compilation_sha256 == digest
        )


class HypothesisGuardResult(StrictFrozenModel):
    """Submitted record, ordered decisions, and optional compiled impact."""

    schema_version: Literal["nurse.v2.hypothesis-guard-result/1"] = (
        HYPOTHESIS_GUARD_RESULT_SCHEMA_VERSION
    )
    guard_result_id: str = Field(pattern=r"^hgr_[0-9a-f]{24}$")
    submitted_record_json: str = Field(min_length=2)
    submitted_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parsed_hypothesis: DiagnosticHypothesis | None
    decisions: tuple[GuardDecision, ...] = Field(min_length=1, max_length=4)
    final_disposition: GuardFinalDisposition
    score_impact: ScoreImpactCompilation | None
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_ordered_closed_result(self) -> Self:
        try:
            submitted = json.loads(self.submitted_record_json)
        except (TypeError, ValueError) as error:
            raise ValueError("submitted hypothesis record is not JSON") from error
        if canonical_json_bytes(submitted).decode("utf-8") != self.submitted_record_json:
            raise ValueError("submitted hypothesis record is not canonical JSON")
        if _digest(submitted) != self.submitted_record_sha256:
            raise ValueError("submitted hypothesis record digest differs")
        if tuple(item.layer for item in self.decisions) != _ORDERED_LAYERS[: len(self.decisions)]:
            raise ValueError("guard decisions are not the ordered layer prefix")
        if any(
            not item.verify()
            or item.submitted_record_sha256 != self.submitted_record_sha256
            or item.prior_accepted_layers != index
            for index, item in enumerate(self.decisions)
        ):
            raise ValueError("guard decision chain is invalid")
        if self.parsed_hypothesis is not None and (
            not self.parsed_hypothesis.verify()
            or any(
                item.hypothesis_id != self.parsed_hypothesis.hypothesis_id
                for item in self.decisions
            )
        ):
            raise ValueError("parsed hypothesis does not match guard decisions")
        if self.final_disposition == "accepted":
            if (
                len(self.decisions) != len(_ORDERED_LAYERS)
                or any(item.disposition != "accepted" for item in self.decisions)
                or self.parsed_hypothesis is None
                or self.score_impact is None
                or not self.score_impact.verify()
                or self.score_impact.hypothesis_id != self.parsed_hypothesis.hypothesis_id
                or self.score_impact.opaque_case_id != self.parsed_hypothesis.opaque_case_id
                or self.score_impact.candidate_service_key
                != self.parsed_hypothesis.candidate_service_key
                or self.score_impact.mechanism_key != self.parsed_hypothesis.mechanism_key
                or self.score_impact.mechanism_definition_sha256
                != self.parsed_hypothesis.mechanism_definition_sha256
                or self.score_impact.event_group_ids != self.parsed_hypothesis.event_group_ids
                or self.score_impact.time_intervals != self.parsed_hypothesis.time_intervals
                or self.score_impact.necessary_predicate_result_ids
                != self.parsed_hypothesis.necessary_predicate_result_ids
                or self.score_impact.optional_predicate_result_ids
                != self.parsed_hypothesis.optional_predicate_result_ids
                or self.score_impact.refuting_predicate_result_ids
                != self.parsed_hypothesis.refuting_predicate_result_ids
                or self.score_impact.verification_request_ids
                != self.parsed_hypothesis.verification_request_ids
            ):
                raise ValueError("accepted guard result is incomplete")
        elif (
            self.decisions[-1].disposition != "rejected"
            or any(item.disposition != "accepted" for item in self.decisions[:-1])
            or self.score_impact is not None
        ):
            raise ValueError("rejected guard result must stop at its first rejection")
        return self

    def verify(self) -> bool:
        try:
            value = HypothesisGuardResult.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = value.model_dump(mode="json", exclude={"guard_result_id", "result_sha256"})
            identifier, digest = _stable_id("hgr", unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        if value.parsed_hypothesis is None and any(
            item.hypothesis_id != f"hyp_{value.submitted_record_sha256[:24]}"
            for item in value.decisions
        ):
            return False
        return (
            value == self and value.guard_result_id == identifier and value.result_sha256 == digest
        )

    def verify_inputs(
        self,
        *,
        submitted_record: object,
        predicates: Sequence[PredicateResult],
        candidate_universe: CandidateUniverse,
        frozen_candidates: FrozenCandidateUniverse,
        entity_registry: TemporalEntityRegistry,
        event_grouping: EventGroupingResult,
        mechanism_library: MechanismLibrary,
        scope_interval: TimeInterval,
        query_session: CaseQuerySessionResult | None = None,
        query_session_coordinator: CaseQuerySessionCoordinator | None = None,
        semantic_context: SemanticContextBundle | None = None,
        card_expansion_policy: CardExpansionPolicy | None = None,
    ) -> bool:
        """Re-execute the guard from the original typed sources."""

        try:
            expected = execute_hypothesis_guard(
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
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return expected == self


def _canonical_submission(submitted_record: object) -> tuple[dict[str, object], str, str]:
    _reject_hidden_cyclic_or_deep(submitted_record)
    if type(submitted_record) is DiagnosticHypothesis:
        if set(vars(submitted_record)) != set(DiagnosticHypothesis.model_fields):
            raise ValueError("submitted hypothesis contains hidden fields")
        raw = submitted_record.model_dump(mode="json", warnings="error")
    elif type(submitted_record) is dict:
        raw = dict(submitted_record)
    else:
        raise ValueError("submitted hypothesis must be a plain record or DiagnosticHypothesis")
    encoded = canonical_json_bytes(raw).decode("utf-8")
    return raw, encoded, _digest(raw)


def _contains_forbidden_impact_field(value: object) -> bool:
    if type(value) is dict:
        record = cast(dict[object, object], value)
        for key, item in record.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_IMPACT_FIELDS:
                return True
            if _contains_forbidden_impact_field(item):
                return True
    elif type(value) in {list, tuple}:
        sequence = cast(list[object] | tuple[object, ...], value)
        return any(_contains_forbidden_impact_field(item) for item in sequence)
    return False


def _safe_predicate_sequence(value: object) -> tuple[PredicateResult, ...]:
    if type(value) not in {tuple, list}:
        raise ValueError("hypothesis guard requires a plain predicate sequence")
    sequence = cast(list[PredicateResult] | tuple[PredicateResult, ...], value)
    if len(sequence) > 4096:
        raise ValueError("hypothesis guard predicate sequence is too large")
    return tuple(sequence)


def _validated_inputs(
    *,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    mechanism_library: MechanismLibrary,
    scope_interval: TimeInterval,
) -> tuple[
    CandidateUniverse,
    FrozenCandidateUniverse,
    TemporalEntityRegistry,
    EventGroupingResult,
    MechanismLibrary,
    TimeInterval,
]:
    if (
        type(candidate_universe) is not CandidateUniverse
        or not candidate_universe.verify()
        or type(frozen_candidates) is not FrozenCandidateUniverse
        or type(entity_registry) is not TemporalEntityRegistry
        or not entity_registry.verify()
        or type(event_grouping) is not EventGroupingResult
        or not event_grouping.verify()
        or type(mechanism_library) is not MechanismLibrary
        or not mechanism_library.verify()
        or type(scope_interval) is not TimeInterval
    ):
        raise ValueError("hypothesis guard requires verified frozen typed inputs")
    require_frozen_candidate_universe(
        frozen=frozen_candidates,
        candidate_universe=candidate_universe,
    )
    rebuilt = build_complete_candidate_universe(
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        scope_interval=scope_interval,
    )
    if rebuilt != candidate_universe:
        raise ValueError("candidate universe does not replay from hypothesis guard inputs")
    if not (
        candidate_universe.opaque_case_id
        == entity_registry.opaque_case_id
        == event_grouping.opaque_case_id
    ):
        raise ValueError("hypothesis guard case scope differs")
    if (
        scope_interval.end_unix_ns > entity_registry.decision_time_unix_ns
        or entity_registry.decision_time_unix_ns != event_grouping.decision_time_unix_ns
    ):
        raise ValueError("hypothesis guard time scope differs")
    return (
        candidate_universe,
        frozen_candidates,
        entity_registry,
        event_grouping,
        mechanism_library,
        scope_interval,
    )


def _decision(
    *,
    hypothesis_id: str,
    layer_index: int,
    disposition: Literal["accepted", "rejected"],
    submitted_sha256: str,
    reason_code: str,
    provenance_ids: Sequence[str],
) -> GuardDecision:
    return build_guard_decision(
        hypothesis_id=hypothesis_id,
        layer=_ORDERED_LAYERS[layer_index],
        sequence=layer_index + 1,
        disposition=disposition,
        submitted_record_sha256=submitted_sha256,
        reason_code=reason_code,
        prior_accepted_layers=layer_index,
        provenance_ids=provenance_ids,
    )


def _build_result(
    *,
    submitted_json: str,
    submitted_sha256: str,
    parsed_hypothesis: DiagnosticHypothesis | None,
    decisions: Sequence[GuardDecision],
    final_disposition: GuardFinalDisposition,
    score_impact: ScoreImpactCompilation | None,
) -> HypothesisGuardResult:
    unsigned: dict[str, object] = {
        "schema_version": HYPOTHESIS_GUARD_RESULT_SCHEMA_VERSION,
        "submitted_record_json": submitted_json,
        "submitted_record_sha256": submitted_sha256,
        "parsed_hypothesis": (
            None if parsed_hypothesis is None else parsed_hypothesis.model_dump(mode="json")
        ),
        "decisions": tuple(item.model_dump(mode="json") for item in decisions),
        "final_disposition": final_disposition,
        "score_impact": None if score_impact is None else score_impact.model_dump(mode="json"),
    }
    identifier, digest = _stable_id("hgr", unsigned)
    result = HypothesisGuardResult.model_validate(
        {**unsigned, "guard_result_id": identifier, "result_sha256": digest}
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("hypothesis guard result verification failed")
    return result


def _overlaps(left: TimeInterval, right: TimeInterval) -> bool:
    return left.start_unix_ns < right.end_unix_ns and left.end_unix_ns > right.start_unix_ns


def _observation_values(observation: Observation) -> Mapping[str, tuple[object, str | None]]:
    baseline = observation.baseline_summary
    fault = observation.fault_summary
    median_delta = (
        None if baseline.median is None or fault.median is None else fault.median - baseline.median
    )
    return {
        "anomaly_strength": (observation.anomaly_strength, "ratio"),
        "baseline_coverage_fraction": (baseline.coverage_fraction, "ratio"),
        "baseline_median": (baseline.median, observation.unit),
        "baseline_sample_count": (baseline.valid_point_count, "count"),
        "coverage_quality": (observation.coverage_quality, "ratio"),
        "evidence_state": (observation.evidence_state, None),
        "fault_coverage_fraction": (fault.coverage_fraction, "ratio"),
        "fault_median": (fault.median, observation.unit),
        "fault_sample_count": (fault.valid_point_count, "count"),
        "median_delta": (median_delta, observation.unit),
        "sample_count": (baseline.valid_point_count + fault.valid_point_count, "count"),
    }


def _measurement_matches(
    measured: MeasuredValue,
    observations: Sequence[Observation],
    derived_observations: Sequence[DerivedVerificationObservation] = (),
    derived_source_observations: Sequence[Observation] = (),
) -> bool:
    if any(
        measured.name in _observation_values(item)
        and _observation_values(item)[measured.name] == (measured.value, measured.unit)
        for item in observations
    ):
        return True
    observations_by_id = {item.observation_id: item for item in derived_source_observations}
    for derived in derived_observations:
        source_observations = tuple(
            observations_by_id[item]
            for item in derived.source_observation_ids
            if item in observations_by_id
        )
        for field in derived.values:
            if field.field_name != measured.name:
                continue
            value = json.loads(field.value_json)
            unit = _derived_field_unit(field.field_name, source_observations)
            if (value, unit) == (measured.value, measured.unit):
                return True
    return False


def _derived_field_unit(
    field_name: str,
    source_observations: Sequence[Observation],
) -> str | None:
    if field_name in {
        "anomaly_strength",
        "baseline_coverage_fraction",
        "coverage_quality",
        "fault_coverage_fraction",
    }:
        return "ratio"
    if field_name in {
        "baseline_sample_count",
        "fault_sample_count",
        "sample_count",
        "baseline_edge_count",
        "fault_edge_count",
        "edge_count_delta",
        "baseline_event_count",
        "fault_event_count",
        "event_count_delta",
    }:
        return "count"
    if field_name in {"onset_delay_unix_ns", "onset_end_unix_ns", "onset_start_unix_ns"}:
        return "ns"
    if field_name in {"baseline_median", "fault_median", "median_delta"}:
        units = {item.unit for item in source_observations}
        return next(iter(units)) if len(units) == 1 else None
    return None


def _predicate_leaves(predicate: PredicateResult) -> tuple[PredicateResult, ...]:
    if predicate.logical_operator == "atomic":
        return (predicate,)
    return tuple(
        leaf for operand in predicate.operand_states for leaf in _predicate_leaves(operand)
    )


def _observation_rejection(
    *,
    hypothesis: DiagnosticHypothesis,
    predicates: Sequence[PredicateResult],
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    query_session: CaseQuerySessionResult | None,
) -> str | None:
    observations_by_id = {item.observation_id: item for item in event_grouping.observations}
    resolutions_by_id = {
        item.observation.observation_id: item for item in entity_registry.resolutions
    }
    verification_batch = None if query_session is None else query_session.verification_result
    derived_by_id = (
        {}
        if verification_batch is None
        else {item.derived_observation_id: item for item in verification_batch.derived_observations}
    )
    verification_result_by_id: dict[str, VerificationResult] = (
        {}
        if verification_batch is None
        else {item.verification_result_id: item for item in verification_batch.results}
    )
    selected_observations: list[Observation] = []
    selected_derived: list[DerivedVerificationObservation] = []
    relation_required_observation_ids: set[str] = set()
    for observation_id in hypothesis.observation_ids:
        observation = observations_by_id.get(observation_id)
        resolution = resolutions_by_id.get(observation_id)
        if observation is None or resolution is None:
            derived = derived_by_id.get(observation_id)
            if derived is None:
                return "unknown_observation_reference"
            if derived.opaque_case_id != hypothesis.opaque_case_id:
                return "observation_case_mismatch"
            if derived.verification_request_id not in hypothesis.verification_request_ids:
                return "verification_observation_request_unlinked"
            if derived.candidate_service_key != hypothesis.candidate_service_key:
                return "observation_binding_invalid"
            if not any(
                _overlaps(derived.query_interval, item) for item in hypothesis.time_intervals
            ):
                return "observation_interval_mismatch"
            selected_derived.append(derived)
            continue
        if observation.opaque_case_id != hypothesis.opaque_case_id:
            return "observation_case_mismatch"
        if (
            resolution.binding.status != "bound"
            or resolution.binding.service_entity_key != hypothesis.candidate_service_key
        ):
            if not hypothesis.relation_ids:
                return "observation_binding_invalid"
            relation_required_observation_ids.add(observation_id)
        if not any(_overlaps(observation.interval, item) for item in hypothesis.time_intervals):
            return "observation_interval_mismatch"
        selected_observations.append(observation)
    groups_by_id: dict[str, EventGroup] = {
        item.event_group_id: item for item in event_grouping.event_groups
    }
    event_link_observation_ids = {
        *hypothesis.observation_ids,
        *(source_id for item in selected_derived for source_id in item.source_observation_ids),
    }
    candidate_bound_observation_ids = {
        observation_id
        for observation_id, resolution in resolutions_by_id.items()
        if resolution.binding.status == "bound"
        and resolution.binding.service_entity_key == hypothesis.candidate_service_key
    }
    selected_claims = tuple(
        item
        for item in event_grouping.association_claims
        if item.claim_id in hypothesis.relation_ids
    )
    included_group_ids = set(hypothesis.event_group_ids)
    eligible_claims = tuple(
        claim
        for claim in selected_claims
        if set(claim.observation_ids).intersection(event_link_observation_ids)
        and set(claim.observation_ids).intersection(candidate_bound_observation_ids)
        and any(
            group.event_group_id in included_group_ids
            and claim.claim_id in group.association_claim_ids
            for group in event_grouping.event_groups
        )
    )
    if len(eligible_claims) != len(selected_claims) or any(
        not any(observation_id in claim.observation_ids for claim in eligible_claims)
        for observation_id in relation_required_observation_ids
    ):
        return "relation_evidence_unlinked"
    for event_group_id in hypothesis.event_group_ids:
        group = groups_by_id.get(event_group_id)
        if group is None:
            return "unknown_event_group_reference"
        if not set(group.observation_ids).intersection(event_link_observation_ids):
            return "event_group_not_linked"
        if not any(_overlaps(group.interval, item) for item in hypothesis.time_intervals):
            return "event_group_interval_mismatch"
    cited_groups = tuple(groups_by_id[item] for item in hypothesis.event_group_ids)
    if (
        hypothesis.event_group_ids
        and hypothesis.observation_ids
        and any(
            not any(
                _overlaps(interval, evidence_interval)
                for evidence_interval in (
                    *(item.interval for item in selected_observations),
                    *(item.query_interval for item in selected_derived),
                )
            )
            or not any(_overlaps(interval, group.interval) for group in cited_groups)
            for interval in hypothesis.time_intervals
        )
    ):
        return "unsupported_hypothesis_interval"
    predicate_by_id: dict[str, PredicateResult] = {}
    for predicate in predicates:
        if type(predicate) is not PredicateResult or not predicate.verify():
            return "invalid_predicate_record"
        if predicate.predicate_result_id in predicate_by_id:
            return "duplicate_predicate_reference"
        predicate_by_id[predicate.predicate_result_id] = predicate
    referenced_ids = (
        *hypothesis.necessary_predicate_result_ids,
        *hypothesis.optional_predicate_result_ids,
        *hypothesis.refuting_predicate_result_ids,
    )
    for predicate_id in referenced_ids:
        referenced_predicate = predicate_by_id.get(predicate_id)
        if referenced_predicate is None:
            return "unknown_predicate_reference"
        for leaf in _predicate_leaves(referenced_predicate):
            if not set(leaf.observation_ids) <= set(hypothesis.observation_ids):
                return "predicate_evidence_reference_denied"
            if not set(leaf.verification_result_ids) <= set(verification_result_by_id):
                return "predicate_verification_reference_denied"
            referenced_verification_results = tuple(
                verification_result_by_id[item] for item in leaf.verification_result_ids
            )
            if leaf.truth_value != "unknown" and any(
                item.status != "completed" for item in referenced_verification_results
            ):
                return "predicate_verification_not_completed"
            if leaf.truth_value != "unknown" and any(
                not set(item.derived_observation_ids) <= set(leaf.observation_ids)
                for item in referenced_verification_results
            ):
                return "predicate_verification_observation_unlinked"
            leaf_original_observations = tuple(
                observations_by_id[item]
                for item in leaf.observation_ids
                if item in observations_by_id
            )
            leaf_derived_observations = tuple(
                derived_by_id[item] for item in leaf.observation_ids if item in derived_by_id
            )
            source_observation_ids = {
                source_id
                for item in leaf_derived_observations
                for source_id in item.source_observation_ids
            }
            leaf_source_observations = tuple(
                observations_by_id[item]
                for item in source_observation_ids
                if item in observations_by_id
            )
            if leaf.truth_value != "unknown" and any(
                not item.formal_scoring_eligible
                or item.evidence_state in {"unknown", "degraded", "no_record"}
                for item in (*leaf_original_observations, *leaf_source_observations)
            ):
                return "observation_sampling_invalid"
            if any(
                not _measurement_matches(
                    measured,
                    leaf_original_observations,
                    leaf_derived_observations,
                    leaf_source_observations,
                )
                for measured in leaf.measured_values
            ):
                return "predicate_measurement_mismatch"
    return None


def _mechanism_rejection(
    *,
    hypothesis: DiagnosticHypothesis,
    predicates: Mapping[str, PredicateResult],
    mechanism_library: MechanismLibrary,
    event_grouping: EventGroupingResult,
    query_session: CaseQuerySessionResult | None,
) -> str | None:
    try:
        mechanism = mechanism_library.lookup(hypothesis.mechanism_key)
    except KeyError:
        return "unknown_mechanism"
    if hypothesis.mechanism_definition_sha256 != mechanism.definition_sha256:
        return "mechanism_definition_mismatch"
    references = (
        ("necessary", hypothesis.necessary_predicate_result_ids),
        ("optional", hypothesis.optional_predicate_result_ids),
        ("refuting", hypothesis.refuting_predicate_result_ids),
    )
    declared = {
        "necessary": set(mechanism.necessary_predicate_kinds),
        "optional": set(mechanism.optional_predicate_kinds),
        "refuting": set(mechanism.refuting_predicate_kinds),
    }
    for requirement, predicate_ids in references:
        kinds: list[str] = []
        for predicate_id in predicate_ids:
            predicate = predicates[predicate_id]
            if predicate.requirement != requirement:
                return "predicate_requirement_mismatch"
            if predicate.predicate_kind not in declared[requirement]:
                return "predicate_kind_denied"
            contract_reason = _predicate_contract_rejection(
                predicate=predicate,
                event_grouping=event_grouping,
                query_session=query_session,
            )
            if contract_reason is not None:
                return contract_reason
            kinds.append(predicate.predicate_kind)
        if len(kinds) != len(set(kinds)):
            return "duplicate_predicate_kind"
        if requirement == "necessary" and set(kinds) != declared[requirement]:
            return "missing_necessary_predicate"
    return None


def _predicate_contract_rejection(
    *,
    predicate: PredicateResult,
    event_grouping: EventGroupingResult,
    query_session: CaseQuerySessionResult | None,
) -> str | None:
    contract = _PREDICATE_EVIDENCE_CONTRACTS.get(predicate.predicate_kind)
    if contract is None:
        return "predicate_evidence_contract_missing"
    family_by_observation_id = {
        observation_id: group.primary_signal_family
        for group in event_grouping.event_groups
        for observation_id in group.observation_ids
    }
    verification_batch = None if query_session is None else query_session.verification_result
    derived_by_id = (
        {}
        if verification_batch is None
        else {item.derived_observation_id: item for item in verification_batch.derived_observations}
    )
    allowed_families = set(contract.primary_signal_families)
    allowed_source_families = set(contract.source_signal_families)
    observations_by_id = {item.observation_id: item for item in event_grouping.observations}
    for leaf in _predicate_leaves(predicate):
        if (
            leaf.predicate_kind != predicate.predicate_kind
            or leaf.requirement != predicate.requirement
        ):
            return "predicate_operand_contract_mismatch"
        source_observation_ids = {
            observation_id
            for observation_id in leaf.observation_ids
            if observation_id in family_by_observation_id
        }
        source_observation_ids.update(
            source_id
            for observation_id in leaf.observation_ids
            for derived in (
                (derived_by_id[observation_id],) if observation_id in derived_by_id else ()
            )
            for source_id in derived.source_observation_ids
        )
        if source_observation_ids and any(
            family_by_observation_id.get(observation_id) not in allowed_families
            and (
                observation_id not in observations_by_id
                or observations_by_id[observation_id].signal_family not in allowed_source_families
            )
            for observation_id in source_observation_ids
        ):
            return "predicate_signal_family_denied"
        if leaf.truth_value == "unknown" and leaf.reason in {
            "pending_verification",
            "no_bound_observations",
        }:
            continue
        if len(leaf.measured_values) != 1 or leaf.measured_values[0].name != contract.measured_name:
            return "predicate_measurement_contract_mismatch"
        threshold = leaf.threshold
        if (
            threshold is None
            or threshold.comparator != contract.comparator
            or threshold.value != contract.threshold_value
            or threshold.unit != contract.threshold_unit
            or threshold.source != "mechanism_constant"
            or leaf.threshold_source != contract.threshold_source
        ):
            return "predicate_threshold_contract_mismatch"
    return None


def _build_score_impact(
    *,
    hypothesis: DiagnosticHypothesis,
    predicates: Mapping[str, PredicateResult],
    mechanism_library: MechanismLibrary,
    provenance_ids: Sequence[str],
) -> tuple[ScoreImpactCompilation | None, str | None]:
    necessary = tuple(predicates[item] for item in hypothesis.necessary_predicate_result_ids)
    if any(item.truth_value == "false" for item in necessary):
        return None, "necessary_predicate_false"
    if any(item.truth_value == "unknown" for item in necessary):
        return None, "necessary_predicate_unknown"
    if not hypothesis.event_group_ids or not hypothesis.observation_ids:
        return None, "score_impact_unbound"
    mechanism = mechanism_library.lookup(hypothesis.mechanism_key)
    unsigned: dict[str, object] = {
        "schema_version": SCORE_IMPACT_COMPILATION_SCHEMA_VERSION,
        "hypothesis_id": hypothesis.hypothesis_id,
        "opaque_case_id": hypothesis.opaque_case_id,
        "candidate_service_key": hypothesis.candidate_service_key,
        "mechanism_key": hypothesis.mechanism_key,
        "mechanism_definition_sha256": hypothesis.mechanism_definition_sha256,
        "event_group_ids": hypothesis.event_group_ids,
        "time_intervals": tuple(item.model_dump(mode="json") for item in hypothesis.time_intervals),
        "necessary_predicate_result_ids": hypothesis.necessary_predicate_result_ids,
        "optional_predicate_result_ids": hypothesis.optional_predicate_result_ids,
        "refuting_predicate_result_ids": hypothesis.refuting_predicate_result_ids,
        "verification_request_ids": hypothesis.verification_request_ids,
        "predicate_contract_version": PREDICATE_EVIDENCE_CONTRACT_VERSION,
        "predicate_contract_sha256": _PREDICATE_EVIDENCE_CONTRACT_SHA256,
        "allowed_roles": mechanism.allowed_roles,
        "effect_policy": "reference_event_role_formula",
        "causal_status": "observable_contract_only",
        "caller_numeric_weight_allowed": False,
        "provenance_ids": tuple(sorted(set(provenance_ids))),
    }
    identifier, digest = _stable_id("sci", unsigned)
    result = ScoreImpactCompilation.model_validate(
        {**unsigned, "score_impact_id": identifier, "compilation_sha256": digest}
    )
    return result, None


def execute_hypothesis_guard(
    *,
    submitted_record: object,
    predicates: Sequence[PredicateResult],
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    mechanism_library: MechanismLibrary,
    scope_interval: TimeInterval,
    query_session: CaseQuerySessionResult | None = None,
    query_session_coordinator: CaseQuerySessionCoordinator | None = None,
    semantic_context: SemanticContextBundle | None = None,
    card_expansion_policy: CardExpansionPolicy | None = None,
) -> HypothesisGuardResult:
    """Apply the fixed guards in order and stop after the first rejection."""

    predicate_values = _safe_predicate_sequence(predicates)
    (
        universe,
        frozen,
        registry,
        grouping,
        library,
        scope,
    ) = _validated_inputs(
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        mechanism_library=mechanism_library,
        scope_interval=scope_interval,
    )
    return _execute_hypothesis_guard_values(
        submitted_record=submitted_record, predicate_values=predicate_values,
        universe=universe, frozen=frozen, registry=registry, grouping=grouping,
        library=library, scope=scope, query_session=query_session,
        query_session_coordinator=query_session_coordinator,
        semantic_context=semantic_context, card_expansion_policy=card_expansion_policy,
    )


def _execute_hypothesis_guard_values(
    *,
    submitted_record: object,
    predicate_values: tuple[PredicateResult, ...],
    universe: CandidateUniverse,
    frozen: FrozenCandidateUniverse,
    registry: TemporalEntityRegistry,
    grouping: EventGroupingResult,
    library: MechanismLibrary,
    scope: TimeInterval,
    query_session: CaseQuerySessionResult | None = None,
    query_session_coordinator: CaseQuerySessionCoordinator | None = None,
    semantic_context: SemanticContextBundle | None = None,
    card_expansion_policy: CardExpansionPolicy | None = None,
) -> HypothesisGuardResult:
    """Unchanged four-layer kernel inside the owning caller's validation scope.

    This private helper is not a source-authentication entry point. The public
    single guard validates on every call; the live B0 batch owns full source
    replay, fresh copies, pins and unchanged-state checks around its whole call.
    """
    raw, submitted_json, submitted_sha256 = _canonical_submission(submitted_record)
    fallback_hypothesis_id = f"hyp_{submitted_sha256[:24]}"
    replayed_session: CaseQuerySessionResult | None = None
    session_reason: str | None = None
    if query_session is not None:
        if query_session_coordinator is None:
            session_reason = "query_session_coordinator_required"
        else:
            try:
                current = query_session_coordinator.require_current(query_session)
                replayed_session = require_case_query_session_replay(
                    session=current,
                    candidate_universe=universe,
                    frozen_candidates=frozen,
                    entity_registry=registry,
                    event_grouping=grouping,
                    semantic_context=semantic_context,
                    card_expansion_policy=card_expansion_policy,
                )
            except (AttributeError, TypeError, ValueError):
                session_reason = "query_session_not_authoritative"
    provenance = tuple(
        sorted(
            {
                submitted_sha256,
                universe.universe_sha256,
                frozen.freeze_sha256,
                registry.registry_sha256,
                grouping.result_sha256,
                library.library_sha256,
                _digest(scope.model_dump(mode="json")),
                *(() if replayed_session is None else (replayed_session.session_sha256,)),
            }
        )
    )
    forbidden_impact_field = _contains_forbidden_impact_field(raw)
    try:
        hypothesis = DiagnosticHypothesis.model_validate(raw)
    except (TypeError, ValueError):
        decision = _decision(
            hypothesis_id=fallback_hypothesis_id,
            layer_index=0,
            disposition="rejected",
            submitted_sha256=submitted_sha256,
            reason_code=(
                "forbidden_score_field"
                if forbidden_impact_field
                else "invalid_schema_or_forbidden_field"
            ),
            provenance_ids=provenance,
        )
        return _build_result(
            submitted_json=submitted_json,
            submitted_sha256=submitted_sha256,
            parsed_hypothesis=None,
            decisions=(decision,),
            final_disposition="rejected",
            score_impact=None,
        )
    structural_reason: str | None = None
    if session_reason is not None:
        structural_reason = session_reason
    elif not hypothesis.verify():
        structural_reason = "invalid_hypothesis_integrity"
    elif hypothesis.lifecycle_status != "proposed":
        structural_reason = "invalid_hypothesis_lifecycle"
    elif hypothesis.opaque_case_id != universe.opaque_case_id:
        structural_reason = "case_scope_mismatch"
    elif hypothesis.candidate_service_key not in frozen.candidate_service_keys:
        structural_reason = "candidate_scope_denied"
    elif any(
        item.start_unix_ns < scope.start_unix_ns
        or item.end_unix_ns > scope.end_unix_ns
        or item.end_unix_ns > registry.decision_time_unix_ns
        for item in hypothesis.time_intervals
    ):
        structural_reason = "time_scope_denied"
    elif hypothesis.verification_request_ids and replayed_session is None:
        structural_reason = "query_session_required"
    elif hypothesis.verification_request_ids and replayed_session is not None:
        verification_batch = replayed_session.verification_result
        known_request_ids = (
            set()
            if verification_batch is None
            else {item.verification_request_id for item in verification_batch.decisions}
        )
        if not set(hypothesis.verification_request_ids) <= known_request_ids:
            structural_reason = "verification_request_scope_denied"
        elif verification_batch is not None and any(
            item.candidate_service_key != hypothesis.candidate_service_key
            for item in verification_batch.requests
            if item.verification_request_id in hypothesis.verification_request_ids
        ):
            structural_reason = "verification_candidate_scope_denied"
    if (
        structural_reason is None
        and hypothesis.relation_ids
        and not set(hypothesis.relation_ids)
        <= {item.claim_id for item in grouping.association_claims}
    ):
        structural_reason = "relation_scope_denied"
    first = _decision(
        hypothesis_id=hypothesis.hypothesis_id,
        layer_index=0,
        disposition="accepted" if structural_reason is None else "rejected",
        submitted_sha256=submitted_sha256,
        reason_code="accepted" if structural_reason is None else structural_reason,
        provenance_ids=provenance,
    )
    if structural_reason is not None:
        return _build_result(
            submitted_json=submitted_json,
            submitted_sha256=submitted_sha256,
            parsed_hypothesis=hypothesis,
            decisions=(first,),
            final_disposition="rejected",
            score_impact=None,
        )
    observation_reason = _observation_rejection(
        hypothesis=hypothesis,
        predicates=predicate_values,
        entity_registry=registry,
        event_grouping=grouping,
        query_session=replayed_session,
    )
    second = _decision(
        hypothesis_id=hypothesis.hypothesis_id,
        layer_index=1,
        disposition="accepted" if observation_reason is None else "rejected",
        submitted_sha256=submitted_sha256,
        reason_code="accepted" if observation_reason is None else observation_reason,
        provenance_ids=provenance,
    )
    if observation_reason is not None:
        return _build_result(
            submitted_json=submitted_json,
            submitted_sha256=submitted_sha256,
            parsed_hypothesis=hypothesis,
            decisions=(first, second),
            final_disposition="rejected",
            score_impact=None,
        )
    predicate_by_id = {item.predicate_result_id: item for item in predicate_values}
    mechanism_reason = _mechanism_rejection(
        hypothesis=hypothesis,
        predicates=predicate_by_id,
        mechanism_library=library,
        event_grouping=grouping,
        query_session=replayed_session,
    )
    third = _decision(
        hypothesis_id=hypothesis.hypothesis_id,
        layer_index=2,
        disposition="accepted" if mechanism_reason is None else "rejected",
        submitted_sha256=submitted_sha256,
        reason_code="accepted" if mechanism_reason is None else mechanism_reason,
        provenance_ids=provenance,
    )
    if mechanism_reason is not None:
        return _build_result(
            submitted_json=submitted_json,
            submitted_sha256=submitted_sha256,
            parsed_hypothesis=hypothesis,
            decisions=(first, second, third),
            final_disposition="rejected",
            score_impact=None,
        )
    score_impact, score_reason = _build_score_impact(
        hypothesis=hypothesis,
        predicates=predicate_by_id,
        mechanism_library=library,
        provenance_ids=provenance,
    )
    fourth = _decision(
        hypothesis_id=hypothesis.hypothesis_id,
        layer_index=3,
        disposition="accepted" if score_reason is None else "rejected",
        submitted_sha256=submitted_sha256,
        reason_code="accepted" if score_reason is None else score_reason,
        provenance_ids=provenance,
    )
    return _build_result(
        submitted_json=submitted_json,
        submitted_sha256=submitted_sha256,
        parsed_hypothesis=hypothesis,
        decisions=(first, second, third, fourth),
        final_disposition="accepted" if score_reason is None else "rejected",
        score_impact=score_impact,
    )
