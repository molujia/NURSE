"""Replay-gated event-group strength and candidate role adjustment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.context_expansion import CardExpansionPolicy
from nurse_rcl.v2.diagnostic_schemas import PredicateResult, RoleAssertion
from nurse_rcl.v2.event_grouping import EventGroupingResult, replay_event_grouping_result
from nurse_rcl.v2.hypothesis_guard import HypothesisGuardResult
from nurse_rcl.v2.mechanism_library import MechanismLibrary
from nurse_rcl.v2.observation_schemas import CandidateUniverse, TimeInterval
from nurse_rcl.v2.query_session import CaseQuerySessionCoordinator, CaseQuerySessionResult
from nurse_rcl.v2.role_compiler import RoleBindingProposal, RoleCompilationResult
from nurse_rcl.v2.semantic_context import SemanticContextBundle
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

EVENT_GROUP_BASE_STRENGTH_SCHEMA_VERSION: Literal["nurse.v2.event-group-base-strength/1"] = (
    "nurse.v2.event-group-base-strength/1"
)
CANDIDATE_EVENT_ROLE_ADJUSTMENT_SCHEMA_VERSION: Literal[
    "nurse.v2.candidate-event-role-adjustment/1"
] = "nurse.v2.candidate-event-role-adjustment/1"
EVENT_STRENGTH_RESULT_SCHEMA_VERSION: Literal["nurse.v2.event-strength-result/1"] = (
    "nurse.v2.event-strength-result/1"
)
EVENT_STRENGTH_POLICY_VERSION: Literal["nurse-v2-event-strength/1"] = "nurse-v2-event-strength/1"


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


EVENT_STRENGTH_POLICY_SHA256 = _digest(
    {
        "version": EVENT_STRENGTH_POLICY_VERSION,
        "base": "max(objective_coverage_quality * bounded_anomaly_strength)",
        "eligible": "formal_scoring_eligible_with_bounded_anomaly_strength",
        "support": "max(accepted_promote_strength)",
        "refutation": "max(accepted_demote_strength)",
        "role_adjustment": "0.5 + 0.5 * support - 0.5 * refutation",
        "same_event_and_role_effect": "canonical_max_without_accumulation",
    }
)


class EventGroupBaseStrength(StrictFrozenModel):
    """One case-level ``v_g`` with all grouped source references retained."""

    schema_version: Literal["nurse.v2.event-group-base-strength/1"] = (
        EVENT_GROUP_BASE_STRENGTH_SCHEMA_VERSION
    )
    strength_id: str = Field(pattern=r"^egs_[0-9a-f]{24}$")
    event_group_id: str = Field(pattern=r"^evt_[0-9a-f]{24}$")
    status: Literal["computed", "unknown_no_eligible_observation"]
    raw_strength: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    contributing_observation_ids: tuple[str, ...]
    retained_observation_ids: tuple[str, ...] = Field(min_length=1)
    deduplicated_observation_ids: tuple[str, ...]
    source_fingerprints: tuple[str, ...] = Field(min_length=1)
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    strength_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_strength(self) -> Self:
        for values, name in (
            (self.contributing_observation_ids, "contributing observation IDs"),
            (self.retained_observation_ids, "retained observation IDs"),
            (self.deduplicated_observation_ids, "deduplicated observation IDs"),
            (self.source_fingerprints, "source fingerprints"),
            (self.provenance_ids, "provenance IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        if not set(self.contributing_observation_ids) <= set(self.retained_observation_ids):
            raise ValueError("contributing observations must be retained")
        if not set(self.deduplicated_observation_ids) <= set(self.retained_observation_ids):
            raise ValueError("deduplicated observations must be retained")
        if set(self.contributing_observation_ids) & set(self.deduplicated_observation_ids):
            raise ValueError("an observation cannot both contribute and be deduplicated")
        if self.status == "computed":
            if self.raw_strength is None or len(self.contributing_observation_ids) != 1:
                raise ValueError("computed strength requires one canonical maximum contributor")
        elif self.raw_strength is not None or self.contributing_observation_ids:
            raise ValueError("unknown strength cannot claim a numeric contributor")
        return self

    def verify(self) -> bool:
        try:
            validated = EventGroupBaseStrength.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = validated.model_dump(mode="json", exclude={"strength_id", "strength_sha256"})
            digest = _digest(unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            validated == self
            and validated.strength_id == f"egs_{digest[:24]}"
            and validated.strength_sha256 == digest
        )


class CandidateEventRoleAdjustment(StrictFrozenModel):
    """Candidate-bound ``u_g``, ``c_g``, ``r_g``, and ``x_g``."""

    schema_version: Literal["nurse.v2.candidate-event-role-adjustment/1"] = (
        CANDIDATE_EVENT_ROLE_ADJUSTMENT_SCHEMA_VERSION
    )
    adjustment_id: str = Field(pattern=r"^era_[0-9a-f]{24}$")
    candidate_service_key: str = Field(min_length=1)
    event_group_id: str = Field(pattern=r"^evt_[0-9a-f]{24}$")
    base_strength_id: str = Field(pattern=r"^egs_[0-9a-f]{24}$")
    support_strength: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    refutation_strength: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    role_adjustment: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    adjusted_strength: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    contributing_assertion_ids: tuple[str, ...]
    retained_assertion_ids: tuple[str, ...]
    deduplicated_assertion_ids: tuple[str, ...]
    provenance_ids: tuple[str, ...]
    adjustment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_bounded_formula(self) -> Self:
        for values, name in (
            (self.contributing_assertion_ids, "contributing assertion IDs"),
            (self.retained_assertion_ids, "retained assertion IDs"),
            (self.deduplicated_assertion_ids, "deduplicated assertion IDs"),
            (self.provenance_ids, "provenance IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        if not set(self.contributing_assertion_ids) <= set(self.retained_assertion_ids):
            raise ValueError("contributing assertions must be retained")
        if not set(self.deduplicated_assertion_ids) <= set(self.retained_assertion_ids):
            raise ValueError("deduplicated assertions must be retained")
        if set(self.contributing_assertion_ids) & set(self.deduplicated_assertion_ids):
            raise ValueError("contributing and deduplicated assertions must be disjoint")
        expected = 0.5 + 0.5 * self.support_strength - 0.5 * self.refutation_strength
        if self.role_adjustment != expected:
            raise ValueError("role adjustment does not match verified maxima")
        return self

    def verify(self) -> bool:
        try:
            validated = CandidateEventRoleAdjustment.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = validated.model_dump(
                mode="json", exclude={"adjustment_id", "adjustment_sha256"}
            )
            digest = _digest(unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            validated == self
            and validated.adjustment_id == f"era_{digest[:24]}"
            and validated.adjustment_sha256 == digest
        )


class EventStrengthResult(StrictFrozenModel):
    """Case-atomic strength result with one base per group and candidate-bound roles."""

    schema_version: Literal["nurse.v2.event-strength-result/1"] = (
        EVENT_STRENGTH_RESULT_SCHEMA_VERSION
    )
    event_strength_result_id: str = Field(pattern=r"^esr_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    policy_version: Literal["nurse-v2-event-strength/1"] = EVENT_STRENGTH_POLICY_VERSION
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_grouping_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    role_compilation_id: str = Field(pattern=r"^rcr_[0-9a-f]{24}$")
    role_compilation_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    group_strengths: tuple[EventGroupBaseStrength, ...]
    candidate_event_adjustments: tuple[CandidateEventRoleAdjustment, ...]
    retained_assertion_ids: tuple[str, ...]
    unbound_assertion_ids: tuple[str, ...]
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_closed_canonical_result(self) -> Self:
        if self.policy_sha256 != EVENT_STRENGTH_POLICY_SHA256:
            raise ValueError("event-strength policy digest differs")
        if self.group_strengths != tuple(
            sorted(self.group_strengths, key=lambda item: item.event_group_id)
        ):
            raise ValueError("event-group base strengths must be canonically ordered")
        if self.candidate_event_adjustments != tuple(
            sorted(
                self.candidate_event_adjustments,
                key=lambda item: (item.candidate_service_key, item.event_group_id),
            )
        ):
            raise ValueError("candidate-event adjustments must be canonically ordered")
        if len({item.event_group_id for item in self.group_strengths}) != len(self.group_strengths):
            raise ValueError("event-group base strengths must be unique")
        if len(
            {
                (item.candidate_service_key, item.event_group_id)
                for item in self.candidate_event_adjustments
            }
        ) != len(self.candidate_event_adjustments):
            raise ValueError("candidate-event adjustments must be unique")
        for values, name in (
            (self.retained_assertion_ids, "retained assertion IDs"),
            (self.unbound_assertion_ids, "unbound assertion IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        if not set(self.unbound_assertion_ids) <= set(self.retained_assertion_ids):
            raise ValueError("unbound assertions must be retained")
        bound_assertion_ids = tuple(
            assertion_id
            for adjustment in self.candidate_event_adjustments
            for assertion_id in adjustment.retained_assertion_ids
        )
        if len(bound_assertion_ids) != len(set(bound_assertion_ids)):
            raise ValueError("an assertion cannot belong to multiple candidate-event adjustments")
        bound_assertion_id_set = set(bound_assertion_ids)
        unbound_assertion_id_set = set(self.unbound_assertion_ids)
        if bound_assertion_id_set & unbound_assertion_id_set:
            raise ValueError("an assertion cannot be both bound and unbound")
        if bound_assertion_id_set | unbound_assertion_id_set != set(self.retained_assertion_ids):
            raise ValueError("bound and unbound assertions must partition retained assertions")
        return self

    def verify(self) -> bool:
        try:
            validated = EventStrengthResult.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if (
                validated != self
                or any(not item.verify() for item in validated.group_strengths)
                or any(not item.verify() for item in validated.candidate_event_adjustments)
            ):
                return False
            base_by_id = {item.strength_id: item for item in validated.group_strengths}
            referenced_assertion_ids = set(validated.unbound_assertion_ids)
            for adjustment in validated.candidate_event_adjustments:
                base = base_by_id.get(adjustment.base_strength_id)
                if base is None or adjustment.event_group_id != base.event_group_id:
                    return False
                expected_adjusted = (
                    None
                    if base.raw_strength is None
                    else base.raw_strength * adjustment.role_adjustment
                )
                if adjustment.adjusted_strength != expected_adjusted:
                    return False
                referenced_assertion_ids.update(adjustment.retained_assertion_ids)
            if referenced_assertion_ids != set(validated.retained_assertion_ids):
                return False
            unsigned = validated.model_dump(
                mode="json", exclude={"event_strength_result_id", "result_sha256"}
            )
            digest = _digest(unsigned)
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return (
            validated.event_strength_result_id == f"esr_{digest[:24]}"
            and validated.result_sha256 == digest
        )

    def verify_inputs(
        self,
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
        query_sessions_by_guard_result_id: Mapping[str, CaseQuerySessionResult] | None = None,
        query_session_coordinator: CaseQuerySessionCoordinator | None = None,
        semantic_context: SemanticContextBundle | None = None,
        card_expansion_policy: CardExpansionPolicy | None = None,
    ) -> bool:
        try:
            expected = compile_event_strengths(
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
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return expected == self


def compile_event_group_base_strengths(
    *, event_grouping: EventGroupingResult
) -> tuple[EventGroupBaseStrength, ...]:
    """Compute each ``v_g`` only after exact deterministic-grouping replay."""

    if type(event_grouping) is not EventGroupingResult:
        raise ValueError("event strengths require an exact event grouping result")
    replayed = replay_event_grouping_result(event_grouping)
    return _compile_base_strengths(replayed)


def _compile_base_strengths(replayed: EventGroupingResult) -> tuple[EventGroupBaseStrength, ...]:
    """Same numeric core; the calling boundary must replay the full grouping."""
    observations = {item.observation_id: item for item in replayed.observations}
    strengths: list[EventGroupBaseStrength] = []
    for group in replayed.event_groups:
        contributing_ids: tuple[str, ...]
        deduplicated_ids: tuple[str, ...]
        eligible = tuple(
            (
                observation.observation_id,
                observation.coverage_quality * observation.anomaly_strength,
            )
            for observation_id in group.observation_ids
            for observation in (observations[observation_id],)
            if observation.formal_scoring_eligible and observation.anomaly_strength is not None
        )
        if eligible:
            raw_strength = max(value for _, value in eligible)
            contributor = min(
                observation_id for observation_id, value in eligible if value == raw_strength
            )
            status = "computed"
            contributing_ids = (contributor,)
            deduplicated_ids = tuple(
                sorted(
                    observation_id
                    for observation_id, _ in eligible
                    if observation_id != contributor
                )
            )
        else:
            raw_strength = None
            status = "unknown_no_eligible_observation"
            contributing_ids = ()
            deduplicated_ids = ()
        unsigned: dict[str, object] = {
            "schema_version": EVENT_GROUP_BASE_STRENGTH_SCHEMA_VERSION,
            "event_group_id": group.event_group_id,
            "status": status,
            "raw_strength": raw_strength,
            "contributing_observation_ids": contributing_ids,
            "retained_observation_ids": group.observation_ids,
            "deduplicated_observation_ids": deduplicated_ids,
            "source_fingerprints": group.source_fingerprints,
            "provenance_ids": group.provenance_ids,
        }
        digest = _digest(unsigned)
        strengths.append(
            EventGroupBaseStrength.model_validate(
                {**unsigned, "strength_id": f"egs_{digest[:24]}", "strength_sha256": digest}
            )
        )
    return tuple(sorted(strengths, key=lambda item: item.event_group_id))


def _build_candidate_event_adjustment(
    *,
    candidate_service_key: str,
    base_strength: EventGroupBaseStrength,
    assertions: Sequence[RoleAssertion],
) -> CandidateEventRoleAdjustment:
    """Build one adjustment from assertions already admitted by the outer replay gate."""

    if type(base_strength) is not EventGroupBaseStrength or not base_strength.verify():
        raise ValueError("candidate adjustment requires a verified base strength")
    unique: dict[str, RoleAssertion] = {}
    for assertion in assertions:
        if type(assertion) is not RoleAssertion or not assertion.verify():
            raise ValueError("candidate adjustment received an invalid role assertion")
        if (
            assertion.candidate_service_key != candidate_service_key
            or assertion.event_group_id != base_strength.event_group_id
        ):
            raise ValueError("role assertion differs from the candidate-event binding")
        previous = unique.setdefault(assertion.assertion_id, assertion)
        if previous != assertion:
            raise ValueError("role assertion ID collision")
    retained = tuple(sorted(unique.values(), key=lambda item: item.assertion_id))
    supports = tuple(
        item
        for item in retained
        if item.lifecycle_status == "accepted" and item.score_direction == "promote"
    )
    refutations = tuple(
        item
        for item in retained
        if item.lifecycle_status == "accepted" and item.score_direction == "demote"
    )
    support_strength = max((item.strength for item in supports), default=0.0)
    refutation_strength = max((item.strength for item in refutations), default=0.0)
    support_contributor = min(
        (item.assertion_id for item in supports if item.strength == support_strength),
        default=None,
    )
    refutation_contributor = min(
        (item.assertion_id for item in refutations if item.strength == refutation_strength),
        default=None,
    )
    contributing_ids = tuple(
        sorted(item for item in (support_contributor, refutation_contributor) if item is not None)
    )
    directional_ids = {item.assertion_id for item in (*supports, *refutations)}
    deduplicated_ids = tuple(sorted(directional_ids - set(contributing_ids)))
    role_adjustment = 0.5 + 0.5 * support_strength - 0.5 * refutation_strength
    adjusted_strength = (
        None if base_strength.raw_strength is None else base_strength.raw_strength * role_adjustment
    )
    unsigned: dict[str, object] = {
        "schema_version": CANDIDATE_EVENT_ROLE_ADJUSTMENT_SCHEMA_VERSION,
        "candidate_service_key": candidate_service_key,
        "event_group_id": base_strength.event_group_id,
        "base_strength_id": base_strength.strength_id,
        "support_strength": support_strength,
        "refutation_strength": refutation_strength,
        "role_adjustment": role_adjustment,
        "adjusted_strength": adjusted_strength,
        "contributing_assertion_ids": contributing_ids,
        "retained_assertion_ids": tuple(item.assertion_id for item in retained),
        "deduplicated_assertion_ids": deduplicated_ids,
        "provenance_ids": tuple(
            sorted({value for item in retained for value in item.provenance_ids})
        ),
    }
    digest = _digest(unsigned)
    return CandidateEventRoleAdjustment.model_validate(
        {**unsigned, "adjustment_id": f"era_{digest[:24]}", "adjustment_sha256": digest}
    )


def compile_event_strengths(
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
    query_sessions_by_guard_result_id: Mapping[str, CaseQuerySessionResult] | None = None,
    query_session_coordinator: CaseQuerySessionCoordinator | None = None,
    semantic_context: SemanticContextBundle | None = None,
    card_expansion_policy: CardExpansionPolicy | None = None,
) -> EventStrengthResult:
    """Compile strengths only from exact grouping and role-compilation replay roots."""

    if type(event_grouping) is not EventGroupingResult:
        raise ValueError("event strengths require an exact event grouping result")
    replayed_grouping = replay_event_grouping_result(event_grouping)
    if type(role_compilation) is not RoleCompilationResult:
        raise ValueError("event strengths require a role compilation result")
    if not role_compilation.verify_inputs(
        proposals=proposals,
        guard_results=guard_results,
        predicates=predicates,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=replayed_grouping,
        mechanism_library=mechanism_library,
        scope_interval=scope_interval,
        query_sessions_by_guard_result_id=query_sessions_by_guard_result_id,
        query_session_coordinator=query_session_coordinator,
        semantic_context=semantic_context,
        card_expansion_policy=card_expansion_policy,
    ):
        raise ValueError("role compilation result does not replay from frozen inputs")
    if role_compilation.opaque_case_id != replayed_grouping.opaque_case_id:
        raise ValueError("event grouping and role compilation case scopes differ")

    bases = compile_event_group_base_strengths(event_grouping=replayed_grouping)
    adjustments, unbound_assertion_ids = _compile_event_adjustments(
        bases=bases, candidate_universe=candidate_universe, assertions=role_compilation.assertions,
    )
    unsigned: dict[str, object] = {
        "schema_version": EVENT_STRENGTH_RESULT_SCHEMA_VERSION,
        "opaque_case_id": replayed_grouping.opaque_case_id,
        "policy_version": EVENT_STRENGTH_POLICY_VERSION,
        "policy_sha256": EVENT_STRENGTH_POLICY_SHA256,
        "event_grouping_result_sha256": replayed_grouping.result_sha256,
        "role_compilation_id": role_compilation.role_compilation_id,
        "role_compilation_result_sha256": role_compilation.result_sha256,
        "group_strengths": tuple(item.model_dump(mode="json") for item in bases),
        "candidate_event_adjustments": tuple(item.model_dump(mode="json") for item in adjustments),
        "retained_assertion_ids": tuple(
            sorted(item.assertion_id for item in role_compilation.assertions)
        ),
        "unbound_assertion_ids": tuple(sorted(unbound_assertion_ids)),
    }
    digest = _digest(unsigned)
    result = EventStrengthResult.model_validate(
        {
            **unsigned,
            "event_strength_result_id": f"esr_{digest[:24]}",
            "result_sha256": digest,
        }
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("event-strength result verification failed")
    return result


def _compile_event_adjustments(
    *, bases: tuple[EventGroupBaseStrength, ...], candidate_universe: CandidateUniverse,
    assertions: tuple[RoleAssertion, ...],
) -> tuple[tuple[CandidateEventRoleAdjustment, ...], tuple[str, ...]]:
    """Preserve binding checks and the shared neutral/support/refutation formula."""
    base_by_event = {item.event_group_id: item for item in bases}
    legal_bindings = {
        (candidate.service_entity_key, event_group_id)
        for candidate in candidate_universe.candidates
        for event_group_id in candidate.event_group_ids
    }
    if any(event_group_id not in base_by_event for _, event_group_id in legal_bindings):
        raise ValueError("candidate universe references an unknown event group")
    assertions_by_binding: dict[tuple[str, str], list[RoleAssertion]] = {}
    unbound_assertion_ids: list[str] = []
    for assertion in assertions:
        binding = (assertion.candidate_service_key, assertion.event_group_id)
        if binding not in legal_bindings:
            unbound_assertion_ids.append(assertion.assertion_id)
            continue
        assertions_by_binding.setdefault(binding, []).append(assertion)
    adjustments = tuple(
        _build_candidate_event_adjustment(
            candidate_service_key=candidate_service_key,
            base_strength=base_by_event[event_group_id],
            assertions=assertions_by_binding.get((candidate_service_key, event_group_id), ()),
        )
        for candidate_service_key, event_group_id in sorted(legal_bindings)
    )
    return adjustments, tuple(sorted(unbound_assertion_ids))
