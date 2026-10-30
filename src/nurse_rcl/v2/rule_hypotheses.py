"""Complete B0 candidate-by-mechanism proposals over frozen input artifacts.

This stage proposes pending predicates and operator plans, not truth values,
verification requests, measurements, or score effects. B0 and B2 can consume
the same frozen inputs and mechanism catalog. No Q selection is accepted here.
The compact result's verify() checks internal consistency; verify_inputs()
must be used to prove its binding to the original frozen upstream artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypeVar

from pydantic import BaseModel, Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.candidate_freeze import (
    FrozenCandidateUniverse,
    require_frozen_candidate_universe,
)
from nurse_rcl.v2.candidate_universe import build_complete_candidate_universe
from nurse_rcl.v2.diagnostic_schemas import (
    DiagnosticHypothesis,
    PredicateRequirement,
    PredicateResult,
    VerificationOperator,
    build_diagnostic_hypothesis,
    build_predicate_result,
)
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.mechanism_library import MechanismLibrary, build_mechanism_library
from nurse_rcl.v2.observation_schemas import CandidateUniverse, TimeInterval
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

RULE_HYPOTHESIS_SET_SCHEMA_VERSION: Literal["nurse.v2.rule-hypothesis-set/2"] = (
    "nurse.v2.rule-hypothesis-set/2"
)
RULE_HYPOTHESIS_ENVELOPE_SCHEMA_VERSION: Literal["nurse.v2.rule-hypothesis-envelope/2"] = (
    "nurse.v2.rule-hypothesis-envelope/2"
)
PENDING_VERIFICATION_PLAN_SCHEMA_VERSION: Literal["nurse.v2.pending-verification-plan/1"] = (
    "nurse.v2.pending-verification-plan/1"
)
RULE_GENERATOR_VERSION: Literal["nurse-v2-b0-rule-hypotheses/2"] = "nurse-v2-b0-rule-hypotheses/2"

_Model = TypeVar("_Model", bound=BaseModel)
_OUTPUT_FIELDS = {"hypotheses", "predicates", "verification_plans", "result_sha256"}
_CandidateReferences = dict[str, tuple[tuple[str, ...], tuple[str, ...]]]


def _reject_hidden_fields(value: object, active: set[int]) -> None:
    """Reject Python model_copy extras before serializers can silently drop them."""

    if isinstance(value, BaseModel):
        if set(vars(value)) != set(type(value).model_fields):
            raise ValueError("unexpected fields in rule input")
        children = tuple(vars(value).values())
    elif isinstance(value, Mapping):
        children = tuple(value.values())
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        children = tuple(value)
    else:
        return
    if id(value) in active:
        raise ValueError("cyclic rule input")
    active.add(id(value))
    try:
        for child in children:
            _reject_hidden_fields(child, active)
    finally:
        active.remove(id(value))


def _validated(value: object, model_type: type[_Model]) -> _Model:
    try:
        if type(value) is not model_type or not isinstance(value, BaseModel):
            raise ValueError
        _reject_hidden_fields(value, set())
        validated = model_type.model_validate(value.model_dump(mode="python", warnings="error"))
        if validated != value:
            raise ValueError
        return validated
    except (AttributeError, RecursionError, TypeError, ValueError):
        raise ValueError("invalid typed rule artifact") from None


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _operators(predicate_kind: str) -> tuple[VerificationOperator, ...]:
    """Versioned planning hints only; availability and truth require later evaluation."""

    operators: set[VerificationOperator] = {"compare_window_summary"}
    if "precedes" in predicate_kind or "transition" in predicate_kind:
        operators.add("measure_onset_interval")
    if "dependency" in predicate_kind or "network" in predicate_kind:
        operators.add("measure_trace_edge_change")
    if "error" in predicate_kind or "failure" in predicate_kind:
        operators.add("summarize_log_pattern")
    if "workload" in predicate_kind or "lifecycle" in predicate_kind:
        operators.add("resolve_entity_binding")
    if "request_rate" in predicate_kind:
        operators.add("inspect_counter_reset")
    return tuple(sorted(operators))


class PendingVerificationPlan(StrictFrozenModel):
    """An unexecuted allowlisted operator plan, not a budgeted query request."""

    schema_version: Literal["nurse.v2.pending-verification-plan/1"] = (
        PENDING_VERIFICATION_PLAN_SCHEMA_VERSION
    )
    plan_id: str = Field(pattern=r"^vpl_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_service_key: str = Field(min_length=1)
    predicate_result_id: str = Field(pattern=r"^prd_[0-9a-f]{24}$")
    operators: tuple[VerificationOperator, ...] = Field(min_length=1)
    observation_ids: tuple[str, ...]
    query_interval: TimeInterval
    decision_time_unix_ns: int = Field(ge=0)
    status: Literal["pending"] = "pending"
    executed: Literal[False] = False
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            value = _validated(self, PendingVerificationPlan)
            if value.operators != tuple(sorted(set(value.operators))):
                return False
            if value.observation_ids != tuple(sorted(set(value.observation_ids))):
                return False
            if value.query_interval.end_unix_ns > value.decision_time_unix_ns:
                return False
            digest = _digest(value.model_dump(mode="json", exclude={"plan_id", "plan_sha256"}))
            return value.plan_sha256 == digest and value.plan_id == f"vpl_{digest[:24]}"
        except (AttributeError, TypeError, ValueError):
            return False


class RuleHypothesisSet(StrictFrozenModel):
    """Compact complete B0 proposals; original inputs remain separately content-addressed."""

    schema_version: Literal["nurse.v2.rule-hypothesis-set/2"] = RULE_HYPOTHESIS_SET_SCHEMA_VERSION
    generator_version: Literal["nurse-v2-b0-rule-hypotheses/2"] = RULE_GENERATOR_VERSION
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    decision_time_unix_ns: int = Field(ge=0)
    scope_interval: TimeInterval
    candidate_universe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entity_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_grouping_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mechanism_library_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_group_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_service_keys: tuple[str, ...] = Field(min_length=1)
    mechanism_keys: tuple[str, ...] = Field(min_length=1)
    observation_ids: tuple[str, ...]
    event_group_ids: tuple[str, ...]
    hypotheses: tuple[DiagnosticHypothesis, ...] = Field(min_length=1)
    predicates: tuple[PredicateResult, ...] = Field(min_length=1)
    verification_plans: tuple[PendingVerificationPlan, ...] = Field(min_length=1)
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        """Check pending semantics and internal closure, not upstream authenticity."""

        try:
            value = _validated(self, RuleHypothesisSet)
            library = build_mechanism_library()
            if value.mechanism_library_sha256 != library.library_sha256:
                return False
            if value.mechanism_keys != tuple(item.mechanism_key for item in library.definitions):
                return False
            if value.scope_interval.end_unix_ns > value.decision_time_unix_ns:
                return False
            for collection in (
                value.candidate_service_keys,
                value.observation_ids,
                value.event_group_ids,
            ):
                if collection != tuple(sorted(set(collection))):
                    return False
            pairs = tuple(
                (item.candidate_service_key, item.mechanism_key) for item in value.hypotheses
            )
            if pairs != tuple(
                (candidate, mechanism)
                for candidate in value.candidate_service_keys
                for mechanism in value.mechanism_keys
            ):
                return False
            references: _CandidateReferences = {}
            for hypothesis in value.hypotheses:
                if not set(hypothesis.observation_ids) <= set(value.observation_ids):
                    return False
                if not set(hypothesis.event_group_ids) <= set(value.event_group_ids):
                    return False
                candidate_refs = (hypothesis.observation_ids, hypothesis.event_group_ids)
                previous = references.setdefault(hypothesis.candidate_service_key, candidate_refs)
                if previous != candidate_refs:
                    return False
            header = value.model_dump(mode="json", exclude=_OUTPUT_FIELDS)
            expected = _propose(
                opaque_case_id=value.opaque_case_id,
                decision_time=value.decision_time_unix_ns,
                scope=value.scope_interval,
                context_sha256=_digest(header),
                references=references,
                library=library,
            )
            if expected != (value.hypotheses, value.predicates, value.verification_plans):
                return False
            unsigned = value.model_dump(mode="json", exclude={"result_sha256"})
            return value.result_sha256 == _digest(unsigned)
        except (AttributeError, KeyError, TypeError, ValueError):
            return False

    def verify_inputs(
        self,
        *,
        candidate_universe: CandidateUniverse,
        frozen_candidates: FrozenCandidateUniverse,
        entity_registry: TemporalEntityRegistry,
        event_grouping: EventGroupingResult,
        scope_interval: TimeInterval,
        mechanism_library: MechanismLibrary,
    ) -> bool:
        """Reproduce the proposals from the original frozen inputs before consumption."""

        try:
            if not self.verify():
                return False
            expected = build_rule_hypothesis_set(
                candidate_universe=candidate_universe,
                frozen_candidates=frozen_candidates,
                entity_registry=entity_registry,
                event_grouping=event_grouping,
                scope_interval=scope_interval,
                mechanism_library=mechanism_library,
            )
            return self == expected
        except (AttributeError, KeyError, TypeError, ValueError):
            return False


class RuleHypothesisEnvelope(StrictFrozenModel):
    """In-process consumption boundary carrying the complete original frozen inputs.

    The compact result alone proves only internal consistency. This envelope's
    mandatory verifier replays the original inputs to prove complete C coverage
    and observation-binding lineage. It is not a semantic/ARK request payload;
    the boundary audit content-addresses the complete envelope.
    """

    schema_version: Literal["nurse.v2.rule-hypothesis-envelope/2"] = (
        RULE_HYPOTHESIS_ENVELOPE_SCHEMA_VERSION
    )
    candidate_universe: CandidateUniverse
    frozen_candidates: FrozenCandidateUniverse
    entity_registry: TemporalEntityRegistry
    event_grouping: EventGroupingResult
    scope_interval: TimeInterval
    mechanism_library: MechanismLibrary
    rule_hypothesis_set: RuleHypothesisSet

    def verify(self) -> bool:
        try:
            value = _validated(self, RuleHypothesisEnvelope)
            return value.rule_hypothesis_set.verify_inputs(
                candidate_universe=value.candidate_universe,
                frozen_candidates=value.frozen_candidates,
                entity_registry=value.entity_registry,
                event_grouping=value.event_grouping,
                scope_interval=value.scope_interval,
                mechanism_library=value.mechanism_library,
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return False


def _propose(
    *,
    opaque_case_id: str,
    decision_time: int,
    scope: TimeInterval,
    context_sha256: str,
    references: _CandidateReferences,
    library: MechanismLibrary,
) -> tuple[
    tuple[DiagnosticHypothesis, ...],
    tuple[PredicateResult, ...],
    tuple[PendingVerificationPlan, ...],
]:
    hypotheses: list[DiagnosticHypothesis] = []
    predicates: list[PredicateResult] = []
    plans: list[PendingVerificationPlan] = []
    for candidate_key, (observation_ids, event_group_ids) in sorted(references.items()):
        reason = "pending_verification" if observation_ids else "no_bound_observations"
        for mechanism in library.definitions:
            provenance = tuple(
                sorted(
                    (
                        f"rule-context:{context_sha256}",
                        f"candidate:{_digest(candidate_key)}",
                        f"mechanism:{mechanism.definition_sha256}",
                    )
                )
            )
            by_requirement: dict[PredicateRequirement, list[str]] = {
                "necessary": [],
                "optional": [],
                "refuting": [],
            }
            requirement_kinds: tuple[tuple[PredicateRequirement, tuple[str, ...]], ...] = (
                ("necessary", mechanism.necessary_predicate_kinds),
                ("optional", mechanism.optional_predicate_kinds),
                ("refuting", mechanism.refuting_predicate_kinds),
            )
            for requirement, kinds in requirement_kinds:
                for kind in kinds:
                    predicate = build_predicate_result(
                        predicate_kind=kind,
                        requirement=requirement,
                        truth_value="unknown",
                        logical_operator="atomic",
                        operand_states=(),
                        observation_ids=observation_ids,
                        verification_result_ids=(),
                        measured_values=(),
                        threshold=None,
                        threshold_source=None,
                        reason=reason,
                        provenance_ids=provenance,
                    )
                    predicates.append(predicate)
                    by_requirement[requirement].append(predicate.predicate_result_id)
                    plan_payload: dict[str, object] = {
                        "schema_version": PENDING_VERIFICATION_PLAN_SCHEMA_VERSION,
                        "opaque_case_id": opaque_case_id,
                        "candidate_service_key": candidate_key,
                        "predicate_result_id": predicate.predicate_result_id,
                        "operators": _operators(kind),
                        "observation_ids": observation_ids,
                        "query_interval": scope.model_dump(mode="json"),
                        "decision_time_unix_ns": decision_time,
                        "status": "pending",
                        "executed": False,
                    }
                    plan_digest = _digest(plan_payload)
                    plans.append(
                        PendingVerificationPlan.model_validate(
                            {
                                **plan_payload,
                                "plan_id": f"vpl_{plan_digest[:24]}",
                                "plan_sha256": plan_digest,
                            }
                        )
                    )
            hypotheses.append(
                build_diagnostic_hypothesis(
                    opaque_case_id=opaque_case_id,
                    candidate_service_key=candidate_key,
                    mechanism_key=mechanism.mechanism_key,
                    mechanism_definition_sha256=mechanism.definition_sha256,
                    origin="deterministic_rule",
                    origin_version=RULE_GENERATOR_VERSION,
                    observation_ids=observation_ids,
                    relation_ids=(),
                    event_group_ids=event_group_ids,
                    time_intervals=(scope,),
                    necessary_predicate_result_ids=by_requirement["necessary"],
                    optional_predicate_result_ids=by_requirement["optional"],
                    refuting_predicate_result_ids=by_requirement["refuting"],
                    alternative_hypothesis_ids=(),
                    verification_request_ids=(),
                    unknown_reasons=(reason,),
                    provenance_ids=provenance,
                    lifecycle_status="proposed",
                )
            )
    return (
        tuple(hypotheses),
        tuple(sorted(predicates, key=lambda item: item.predicate_result_id)),
        tuple(sorted(plans, key=lambda item: item.plan_id)),
    )


def build_rule_hypothesis_set(
    *,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    scope_interval: TimeInterval,
    mechanism_library: MechanismLibrary,
) -> RuleHypothesisSet:
    """Propose every C x mechanism pair, keeping normal, zero, and unknown inputs."""

    universe = _validated(candidate_universe, CandidateUniverse)
    frozen = _validated(frozen_candidates, FrozenCandidateUniverse)
    entities = _validated(entity_registry, TemporalEntityRegistry)
    grouping = _validated(event_grouping, EventGroupingResult)
    scope = _validated(scope_interval, TimeInterval)
    library = _validated(mechanism_library, MechanismLibrary)
    require_frozen_candidate_universe(frozen=frozen, candidate_universe=universe)
    # Rebuilding C closes observation/binding/event membership and time/case scope,
    # even if a caller has rehashed and re-frozen a different candidate projection.
    expected_universe = build_complete_candidate_universe(
        entity_registry=entities, event_grouping=grouping, scope_interval=scope
    )
    if universe != expected_universe:
        raise ValueError("candidate universe does not match frozen rule input lineage")
    references: _CandidateReferences = {}
    for candidate in universe.candidates:
        observation_ids = tuple(
            sorted(
                resolution.observation.observation_id
                for resolution in entities.resolutions
                if resolution.binding.status == "bound"
                and resolution.binding.service_entity_key == candidate.service_entity_key
                and resolution.observation.interval.start_unix_ns < scope.end_unix_ns
                and resolution.observation.interval.end_unix_ns > scope.start_unix_ns
            )
        )
        event_group_ids = tuple(
            sorted(
                group.event_group_id
                for group in grouping.event_groups
                if set(group.observation_ids).intersection(observation_ids)
            )
        )
        references[candidate.service_entity_key] = (observation_ids, event_group_ids)
    header: dict[str, object] = {
        "schema_version": RULE_HYPOTHESIS_SET_SCHEMA_VERSION,
        "generator_version": RULE_GENERATOR_VERSION,
        "opaque_case_id": universe.opaque_case_id,
        "decision_time_unix_ns": entities.decision_time_unix_ns,
        "scope_interval": scope.model_dump(mode="json"),
        "candidate_universe_sha256": universe.universe_sha256,
        "candidate_freeze_sha256": frozen.freeze_sha256,
        "observation_contract_sha256": universe.observation_contract_sha256,
        "entity_registry_sha256": entities.registry_sha256,
        "event_grouping_result_sha256": grouping.result_sha256,
        "mechanism_library_sha256": library.library_sha256,
        "observation_set_sha256": _digest(
            [item.model_dump(mode="json") for item in grouping.observations]
        ),
        "event_group_set_sha256": _digest(
            [item.model_dump(mode="json") for item in grouping.event_groups]
        ),
        "candidate_service_keys": frozen.candidate_service_keys,
        "mechanism_keys": tuple(item.mechanism_key for item in library.definitions),
        "observation_ids": tuple(item.observation_id for item in grouping.observations),
        "event_group_ids": tuple(item.event_group_id for item in grouping.event_groups),
    }
    hypotheses, predicates, plans = _propose(
        opaque_case_id=universe.opaque_case_id,
        decision_time=entities.decision_time_unix_ns,
        scope=scope,
        context_sha256=_digest(header),
        references=references,
        library=library,
    )
    unsigned: dict[str, object] = {
        **header,
        "hypotheses": tuple(item.model_dump(mode="json") for item in hypotheses),
        "predicates": tuple(item.model_dump(mode="json") for item in predicates),
        "verification_plans": tuple(item.model_dump(mode="json") for item in plans),
    }
    result = RuleHypothesisSet.model_validate({**unsigned, "result_sha256": _digest(unsigned)})
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("rule hypothesis set verification failed")
    return result
