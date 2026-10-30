"""Deterministic compilation of event-, relation-, and interval-bound roles."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Literal, cast

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.candidate_freeze import (
    FrozenCandidateUniverse,
    require_frozen_candidate_universe,
)
from nurse_rcl.v2.context_expansion import CardExpansionPolicy
from nurse_rcl.v2.diagnostic_schemas import (
    DiagnosticRole,
    PredicateResult,
    RoleAssertion,
    build_role_assertion,
)
from nurse_rcl.v2.event_grouping import AssociationClaim, EventGroup, EventGroupingResult
from nurse_rcl.v2.hypothesis_guard import HypothesisGuardResult
from nurse_rcl.v2.mechanism_library import MechanismLibrary
from nurse_rcl.v2.observation_schemas import CandidateUniverse, Observation, TimeInterval
from nurse_rcl.v2.query_session import CaseQuerySessionCoordinator, CaseQuerySessionResult
from nurse_rcl.v2.semantic_context import SemanticContextBundle
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

ROLE_BINDING_PROPOSAL_SCHEMA_VERSION: Literal["nurse.v2.role-binding-proposal/1"] = (
    "nurse.v2.role-binding-proposal/1"
)
ROLE_COMPILATION_RESULT_SCHEMA_VERSION: Literal["nurse.v2.role-compilation-result/1"] = (
    "nurse.v2.role-compilation-result/1"
)
ROLE_COMPILATION_POLICY_VERSION: Literal["nurse-v2-role-compilation/1"] = (
    "nurse-v2-role-compilation/1"
)

_MAX_GUARDS = 4096
_MAX_PREDICATES = 4096
_MAX_PROPOSALS = 16384


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _stable_id(prefix: str, value: object) -> tuple[str, str]:
    digest = _digest(value)
    return f"{prefix}_{digest[:24]}", digest


_ROLE_POLICY_DOCUMENT = {
    "version": ROLE_COMPILATION_POLICY_VERSION,
    "proposal_numeric_fields_allowed": False,
    "support_strength": "minimum_necessary_predicate_source_coverage",
    "symptom_direction": "neutral",
    "local_relation": "content_addressed_candidate_event_binding",
    "negative_correction": ("complete_true_refutation_and_distinct_verified_alternative_support"),
    "conflict": "explicit_alternative_or_same_hypothesis_incompatible_role",
}
ROLE_COMPILATION_POLICY_SHA256 = _digest(_ROLE_POLICY_DOCUMENT)


class RoleBindingProposal(StrictFrozenModel):
    """A score-free role choice whose numeric effect is compiler-owned."""

    schema_version: Literal["nurse.v2.role-binding-proposal/1"] = (
        ROLE_BINDING_PROPOSAL_SCHEMA_VERSION
    )
    proposal_id: str = Field(pattern=r"^rbp_[0-9a-f]{24}$")
    guard_result_id: str = Field(pattern=r"^hgr_[0-9a-f]{24}$")
    hypothesis_id: str = Field(pattern=r"^hyp_[0-9a-f]{24}$")
    event_group_id: str = Field(pattern=r"^evt_[0-9a-f]{24}$")
    relation_id: str = Field(pattern=r"^(?:assoc|lrel)_[0-9a-f]{24}$")
    interval: TimeInterval
    role: DiagnosticRole
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_proposal(self) -> Self:
        if self.provenance_ids != tuple(sorted(set(self.provenance_ids))):
            raise ValueError("role proposal provenance IDs must be sorted and unique")
        return self

    def verify(self) -> bool:
        try:
            value = RoleBindingProposal.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = value.model_dump(mode="json", exclude={"proposal_id", "proposal_sha256"})
            identifier, digest = _stable_id("rbp", unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return value == self and value.proposal_id == identifier and value.proposal_sha256 == digest


def build_role_binding_proposal(
    *,
    guard_result_id: str,
    hypothesis_id: str,
    event_group_id: str,
    relation_id: str,
    interval: TimeInterval,
    role: DiagnosticRole,
    provenance_ids: Sequence[str],
) -> RoleBindingProposal:
    """Build a canonical proposal without score, weight, vote, or confidence."""

    unsigned: dict[str, object] = {
        "schema_version": ROLE_BINDING_PROPOSAL_SCHEMA_VERSION,
        "guard_result_id": guard_result_id,
        "hypothesis_id": hypothesis_id,
        "event_group_id": event_group_id,
        "relation_id": relation_id,
        "interval": interval.model_dump(mode="json"),
        "role": role,
        "provenance_ids": tuple(sorted(set(provenance_ids))),
    }
    identifier, digest = _stable_id("rbp", unsigned)
    return RoleBindingProposal.model_validate(
        {**unsigned, "proposal_id": identifier, "proposal_sha256": digest}
    )


def local_event_relation_id(*, candidate_service_key: str, event_group_id: str) -> str:
    """Name the only legal relation for purely local event evidence."""

    identifier, _ = _stable_id(
        "lrel",
        {
            "policy_version": ROLE_COMPILATION_POLICY_VERSION,
            "candidate_service_key": candidate_service_key,
            "event_group_id": event_group_id,
        },
    )
    return identifier


def build_default_role_binding_proposals(
    *,
    guard_results: Sequence[HypothesisGuardResult],
    event_grouping: EventGroupingResult,
) -> tuple[RoleBindingProposal, ...]:
    """Derive the transparent B0 role choice without assigning numeric impact."""

    guards = _plain_sequence(guard_results, limit=_MAX_GUARDS, name="hypothesis guard results")
    if type(event_grouping) is not EventGroupingResult or not event_grouping.verify():
        raise ValueError("default role derivation requires verified event grouping")
    return _default_role_binding_proposals(guards=guards, event_grouping=event_grouping)


def _default_role_binding_proposals(
    *, guards: Sequence[object], event_grouping: EventGroupingResult
) -> tuple[RoleBindingProposal, ...]:
    """Numeric-free core; caller owns complete grouping replay in this invocation."""
    groups_by_id = {item.event_group_id: item for item in event_grouping.event_groups}
    proposals: dict[str, RoleBindingProposal] = {}
    for value in guards:
        if (
            type(value) is not HypothesisGuardResult
            or not value.verify()
            or value.final_disposition != "accepted"
            or value.parsed_hypothesis is None
            or value.score_impact is None
        ):
            raise ValueError("default role derivation requires accepted guard results")
        hypothesis = value.parsed_hypothesis
        impact = value.score_impact
        if hypothesis.opaque_case_id != event_grouping.opaque_case_id:
            raise ValueError("default role derivation case scope differs")
        allowed = set(impact.allowed_roles)
        for event_group_id in impact.event_group_ids:
            group = groups_by_id.get(event_group_id)
            if group is None:
                raise ValueError("default role derivation references an unknown event group")
            relation_ids = tuple(
                sorted(set(hypothesis.relation_ids) & set(group.association_claim_ids))
            )
            if allowed == {"symptom_support"}:
                role: DiagnosticRole = "symptom_support"
            elif relation_ids and "source_support" in allowed:
                role = "source_support"
            elif "root_support" in allowed:
                role = "root_support"
            elif "symptom_support" in allowed:
                role = "symptom_support"
            else:
                role = "source_support"
            if not relation_ids:
                relation_ids = (
                    local_event_relation_id(
                        candidate_service_key=hypothesis.candidate_service_key,
                        event_group_id=event_group_id,
                    ),
                )
            for interval in impact.time_intervals:
                if not _overlaps(interval, group.interval):
                    continue
                for relation_id in relation_ids:
                    proposal = build_role_binding_proposal(
                        guard_result_id=value.guard_result_id,
                        hypothesis_id=hypothesis.hypothesis_id,
                        event_group_id=event_group_id,
                        relation_id=relation_id,
                        interval=interval,
                        role=role,
                        provenance_ids=(
                            value.guard_result_id,
                            impact.score_impact_id,
                            ROLE_COMPILATION_POLICY_SHA256,
                            "role-policy:default",
                        ),
                    )
                    previous = proposals.setdefault(proposal.proposal_id, proposal)
                    if previous != proposal:  # pragma: no cover - content ID invariant
                        raise ValueError("default role proposal ID collision")
    if not proposals:
        raise ValueError("default role derivation produced no bound proposal")
    return tuple(sorted(proposals.values(), key=lambda item: item.proposal_id))


class RoleCompilationResult(StrictFrozenModel):
    """Closed, content-addressed output of role and correction compilation."""

    schema_version: Literal["nurse.v2.role-compilation-result/1"] = (
        ROLE_COMPILATION_RESULT_SCHEMA_VERSION
    )
    role_compilation_id: str = Field(pattern=r"^rcr_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    policy_version: Literal["nurse-v2-role-compilation/1"] = ROLE_COMPILATION_POLICY_VERSION
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    guard_result_ids: tuple[str, ...] = Field(min_length=1)
    proposals: tuple[RoleBindingProposal, ...] = Field(min_length=1)
    assertions: tuple[RoleAssertion, ...] = Field(min_length=1)
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_closed_canonical_result(self) -> Self:
        if self.policy_sha256 != ROLE_COMPILATION_POLICY_SHA256:
            raise ValueError("role compilation policy digest differs")
        if self.guard_result_ids != tuple(sorted(set(self.guard_result_ids))):
            raise ValueError("guard result IDs must be sorted and unique")
        if self.proposals != tuple(sorted(self.proposals, key=lambda item: item.proposal_id)):
            raise ValueError("role proposals must be canonically ordered")
        if self.assertions != tuple(sorted(self.assertions, key=lambda item: item.assertion_id)):
            raise ValueError("role assertions must be canonically ordered")
        if len({item.proposal_id for item in self.proposals}) != len(self.proposals):
            raise ValueError("role proposal IDs must be unique")
        if len({item.assertion_id for item in self.assertions}) != len(self.assertions):
            raise ValueError("role assertion IDs must be unique")
        if any(not item.verify() for item in self.proposals) or any(
            not item.verify() for item in self.assertions
        ):
            raise ValueError("role compilation contains an invalid child")
        if not {item.guard_result_id for item in self.proposals} <= set(self.guard_result_ids):
            raise ValueError("role proposal references an unretained guard result")
        return self

    def verify(self) -> bool:
        try:
            value = RoleCompilationResult.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = value.model_dump(
                mode="json", exclude={"role_compilation_id", "result_sha256"}
            )
            identifier, digest = _stable_id("rcr", unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            value == self
            and value.role_compilation_id == identifier
            and value.result_sha256 == digest
        )

    def verify_inputs(
        self,
        *,
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
            expected = compile_event_roles(
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


def _plain_sequence(value: object, *, limit: int, name: str) -> tuple[object, ...]:
    if type(value) not in {tuple, list}:
        raise ValueError(f"{name} must be a plain sequence")
    sequence = cast(tuple[object, ...] | list[object], value)
    if not sequence or len(sequence) > limit:
        raise ValueError(f"{name} size is outside the compilation limit")
    return tuple(sequence)


def _overlaps(left: TimeInterval, right: TimeInterval) -> bool:
    return max(left.start_unix_ns, right.start_unix_ns) < min(left.end_unix_ns, right.end_unix_ns)


def _proposal_effect_key(
    proposal: RoleBindingProposal,
    guard: HypothesisGuardResult,
) -> tuple[object, ...]:
    """Identify the compiler-owned effect while excluding caller audit metadata."""

    hypothesis = guard.parsed_hypothesis
    impact = guard.score_impact
    if hypothesis is None or impact is None:  # pragma: no cover - accepted guard invariant
        raise AssertionError("accepted guard has no compiled hypothesis")
    return (
        proposal.guard_result_id,
        proposal.hypothesis_id,
        hypothesis.candidate_service_key,
        proposal.event_group_id,
        proposal.relation_id,
        proposal.interval.start_unix_ns,
        proposal.interval.end_unix_ns,
        proposal.role,
        impact.score_impact_id,
        impact.necessary_predicate_result_ids,
    )


def _canonical_role_proposals(
    *,
    proposals_by_id: Mapping[str, RoleBindingProposal],
    guards_by_id: Mapping[str, HypothesisGuardResult],
) -> tuple[dict[str, RoleBindingProposal], dict[str, tuple[str, ...]]]:
    """Collapse equal effects while retaining every proposal and caller audit ID."""

    proposals_by_effect: dict[tuple[object, ...], list[RoleBindingProposal]] = {}
    for proposal in sorted(proposals_by_id.values(), key=lambda item: item.proposal_id):
        guard = guards_by_id[proposal.guard_result_id]
        proposals_by_effect.setdefault(_proposal_effect_key(proposal, guard), []).append(proposal)
    canonical_by_id: dict[str, RoleBindingProposal] = {}
    audit_ids_by_canonical_id: dict[str, tuple[str, ...]] = {}
    for equivalent_proposals in proposals_by_effect.values():
        canonical = equivalent_proposals[0]
        canonical_by_id[canonical.proposal_id] = canonical
        audit_ids_by_canonical_id[canonical.proposal_id] = tuple(
            sorted(
                {
                    audit_id
                    for proposal in equivalent_proposals
                    for audit_id in (proposal.proposal_id, *proposal.provenance_ids)
                }
            )
        )
    return canonical_by_id, audit_ids_by_canonical_id


def _exact_binding_key(assertion: RoleAssertion) -> tuple[str, str, str, int, int]:
    return (
        assertion.candidate_service_key,
        assertion.event_group_id,
        assertion.relation_id,
        assertion.interval.start_unix_ns,
        assertion.interval.end_unix_ns,
    )


def _cross_candidate_binding_key(assertion: RoleAssertion) -> tuple[str, str, int, int]:
    return (
        assertion.event_group_id,
        assertion.relation_id,
        assertion.interval.start_unix_ns,
        assertion.interval.end_unix_ns,
    )


def _predicate_leaves(predicate: PredicateResult) -> tuple[PredicateResult, ...]:
    if predicate.logical_operator == "atomic":
        return (predicate,)
    return tuple(
        leaf for operand in predicate.operand_states for leaf in _predicate_leaves(operand)
    )


def _source_observation_ids(
    *,
    predicate_ids: Sequence[str],
    predicates_by_id: Mapping[str, PredicateResult],
    query_session: CaseQuerySessionResult | None,
) -> tuple[str, ...]:
    derived_by_id = (
        {}
        if query_session is None or query_session.verification_result is None
        else {
            item.derived_observation_id: item
            for item in query_session.verification_result.derived_observations
        }
    )
    source_ids: set[str] = set()
    for predicate_id in predicate_ids:
        predicate = predicates_by_id[predicate_id]
        for leaf in _predicate_leaves(predicate):
            for observation_id in leaf.observation_ids:
                derived = derived_by_id.get(observation_id)
                if derived is None:
                    source_ids.add(observation_id)
                else:
                    source_ids.update(derived.source_observation_ids)
    return tuple(sorted(source_ids))


def _proposal_rejection(
    *,
    proposal: RoleBindingProposal,
    guard: HypothesisGuardResult,
    event_groups_by_id: Mapping[str, EventGroup],
    association_claims_by_id: Mapping[str, AssociationClaim],
    observations_by_id: Mapping[str, Observation],
    candidate_bound_observation_ids: set[str],
    necessary_source_ids: set[str],
) -> str | None:
    hypothesis = guard.parsed_hypothesis
    impact = guard.score_impact
    if hypothesis is None or impact is None:  # pragma: no cover - accepted guard invariant
        return "guard_has_no_score_impact"
    if proposal.hypothesis_id != hypothesis.hypothesis_id:
        return "hypothesis_guard_mismatch"
    if proposal.role not in impact.allowed_roles:
        return "role_not_permitted"
    if proposal.event_group_id not in impact.event_group_ids:
        return "event_group_not_authorized"
    group = event_groups_by_id.get(proposal.event_group_id)
    if group is None:
        return "unknown_event_group"
    if proposal.interval not in impact.time_intervals:
        return "interval_not_authorized"
    if not _overlaps(proposal.interval, group.interval):
        return "event_interval_unlinked"
    event_source_ids = set(group.observation_ids).intersection(necessary_source_ids)
    if not event_source_ids:
        return "event_predicate_unlinked"
    interval_source_ids = {
        source_id
        for source_id in event_source_ids
        if source_id in observations_by_id
        and _overlaps(observations_by_id[source_id].interval, proposal.interval)
    }
    if not interval_source_ids:
        return "interval_predicate_unlinked"
    local_relation = local_event_relation_id(
        candidate_service_key=hypothesis.candidate_service_key,
        event_group_id=proposal.event_group_id,
    )
    if proposal.relation_id == local_relation:
        if proposal.role == "source_support":
            return "source_relation_unproven"
        if not (candidate_bound_observation_ids & interval_source_ids):
            return "local_relation_unproven"
        return None
    if proposal.relation_id not in hypothesis.relation_ids:
        return "relation_not_authorized"
    if proposal.relation_id not in group.association_claim_ids:
        return "relation_event_unlinked"
    relation = association_claims_by_id.get(proposal.relation_id)
    if relation is None:  # pragma: no cover - verified grouping invariant
        return "relation_event_unlinked"
    candidate_endpoint_ids = (
        set(relation.observation_ids) & candidate_bound_observation_ids & interval_source_ids
    )
    if not candidate_endpoint_ids:
        return {
            "root_support": "root_relation_unproven",
            "source_support": "source_relation_unproven",
            "symptom_support": "symptom_relation_unproven",
        }[proposal.role]
    return None


def _rejected_assertion(
    *,
    proposal: RoleBindingProposal,
    guard: HypothesisGuardResult,
    predicate_result_ids: Sequence[str],
    reason: str,
    proposal_audit_ids: Sequence[str],
) -> RoleAssertion:
    hypothesis = guard.parsed_hypothesis
    if hypothesis is None:  # pragma: no cover - accepted guard invariant
        raise ValueError("role proposal references a guard without a hypothesis")
    return build_role_assertion(
        hypothesis_id=hypothesis.hypothesis_id,
        candidate_service_key=hypothesis.candidate_service_key,
        event_group_id=proposal.event_group_id,
        relation_id=proposal.relation_id,
        interval=proposal.interval,
        role=proposal.role,
        strength=0.0,
        score_direction="neutral",
        predicate_result_ids=predicate_result_ids,
        lifecycle_status="rejected",
        reason=reason,
        provenance_ids=(guard.guard_result_id, *proposal_audit_ids),
    )


def _conflict_neutralized_assertions(
    *,
    assertions: Sequence[RoleAssertion],
    proposal_id_by_assertion_id: Mapping[str, str],
    proposals_by_id: Mapping[str, RoleBindingProposal],
    guards_by_id: Mapping[str, HypothesisGuardResult],
    alternatives_by_hypothesis_id: Mapping[str, Sequence[str]] | None = None,
) -> tuple[tuple[RoleAssertion, ...], dict[str, str]]:
    """Withdraw only explicit, binding-identical incompatible alternatives."""

    conflicts_by_assertion_id: dict[str, set[str]] = {}
    values = tuple(assertions)
    assertions_by_binding_and_hypothesis: dict[
        tuple[str, str, str, int, int], dict[str, list[RoleAssertion]]
    ] = {}
    for assertion in values:
        if assertion.lifecycle_status != "accepted":
            continue
        owner_proposal_id = proposal_id_by_assertion_id.get(assertion.assertion_id)
        if owner_proposal_id is None:
            continue
        proposal = proposals_by_id[owner_proposal_id]
        guard = guards_by_id[proposal.guard_result_id]
        if guard.parsed_hypothesis is None:  # pragma: no cover - accepted guard invariant
            continue
        assertions_by_binding_and_hypothesis.setdefault(
            _exact_binding_key(assertion), {}
        ).setdefault(assertion.hypothesis_id, []).append(assertion)

    for assertions_by_hypothesis in assertions_by_binding_and_hypothesis.values():
        audit_ids_by_hypothesis_and_role: dict[str, dict[DiagnosticRole, set[str]]] = {}
        for hypothesis_id, hypothesis_assertions in assertions_by_hypothesis.items():
            audit_ids_by_role = audit_ids_by_hypothesis_and_role.setdefault(hypothesis_id, {})
            for assertion in hypothesis_assertions:
                owner_proposal_id = proposal_id_by_assertion_id[assertion.assertion_id]
                audit_ids_by_role.setdefault(assertion.role, set()).update(
                    {assertion.assertion_id, owner_proposal_id}
                )

        def register_conflicts(
            source_assertions: Sequence[RoleAssertion],
            target_audit_ids_by_role: Mapping[DiagnosticRole, set[str]],
        ) -> None:
            for assertion in source_assertions:
                conflicting_audit_ids = {
                    audit_id
                    for role, audit_ids in target_audit_ids_by_role.items()
                    if role != assertion.role
                    for audit_id in audit_ids
                }
                if not conflicting_audit_ids:
                    continue
                owner_proposal_id = proposal_id_by_assertion_id[assertion.assertion_id]
                conflicts_by_assertion_id.setdefault(assertion.assertion_id, set()).update(
                    {owner_proposal_id, *conflicting_audit_ids}
                )

        for hypothesis_id, hypothesis_assertions in assertions_by_hypothesis.items():
            local_audit_ids_by_role = audit_ids_by_hypothesis_and_role[hypothesis_id]
            register_conflicts(hypothesis_assertions, local_audit_ids_by_role)
            owner_proposal_id = proposal_id_by_assertion_id[hypothesis_assertions[0].assertion_id]
            owner_proposal = proposals_by_id[owner_proposal_id]
            hypothesis = guards_by_id[owner_proposal.guard_result_id].parsed_hypothesis
            if hypothesis is None:  # pragma: no cover - accepted guard invariant
                continue
            alternative_ids = (
                hypothesis.alternative_hypothesis_ids
                if alternatives_by_hypothesis_id is None
                else alternatives_by_hypothesis_id.get(hypothesis.hypothesis_id, ())
            )
            for alternative_hypothesis_id in alternative_ids:
                alternative_assertions = assertions_by_hypothesis.get(alternative_hypothesis_id)
                if alternative_assertions is None:
                    continue
                alternative_audit_ids_by_role = audit_ids_by_hypothesis_and_role[
                    alternative_hypothesis_id
                ]
                register_conflicts(hypothesis_assertions, alternative_audit_ids_by_role)
                register_conflicts(alternative_assertions, local_audit_ids_by_role)
    neutralized: list[RoleAssertion] = []
    neutralized_proposal_id_by_assertion_id: dict[str, str] = {}
    for assertion in values:
        owner_proposal_id = proposal_id_by_assertion_id.get(assertion.assertion_id)
        conflicts = conflicts_by_assertion_id.get(assertion.assertion_id)
        if not conflicts:
            neutralized.append(assertion)
            if owner_proposal_id is not None:
                neutralized_proposal_id_by_assertion_id[assertion.assertion_id] = owner_proposal_id
            continue
        neutralized_assertion = build_role_assertion(
            hypothesis_id=assertion.hypothesis_id,
            candidate_service_key=assertion.candidate_service_key,
            event_group_id=assertion.event_group_id,
            relation_id=assertion.relation_id,
            interval=assertion.interval,
            role=assertion.role,
            strength=assertion.strength,
            score_direction="neutral",
            predicate_result_ids=assertion.predicate_result_ids,
            lifecycle_status="conflicted",
            reason="explicit_alternative_role_conflict",
            provenance_ids=tuple(
                sorted(
                    {
                        *assertion.provenance_ids,
                        assertion.assertion_id,
                        *conflicts,
                    }
                )
            ),
        )
        neutralized.append(neutralized_assertion)
        if owner_proposal_id is not None:
            neutralized_proposal_id_by_assertion_id[neutralized_assertion.assertion_id] = (
                owner_proposal_id
            )
    return tuple(neutralized), neutralized_proposal_id_by_assertion_id


def _predicate_quality(
    *,
    predicate_ids: Sequence[str],
    predicates_by_id: Mapping[str, PredicateResult],
    observations_by_id: Mapping[str, object],
    query_session: CaseQuerySessionResult | None,
) -> float | None:
    source_ids = _source_observation_ids(
        predicate_ids=predicate_ids,
        predicates_by_id=predicates_by_id,
        query_session=query_session,
    )
    if not source_ids:
        return None
    qualities: list[float] = []
    for source_id in source_ids:
        observation = observations_by_id.get(source_id)
        quality = getattr(observation, "coverage_quality", None)
        if type(quality) is not float or not 0.0 <= quality <= 1.0:
            return None
        qualities.append(quality)
    return min(qualities)


def _predicate_sources_match_binding(
    *,
    predicate_ids: Sequence[str],
    candidate_service_key: str,
    event_group_id: str,
    relation_id: str,
    interval: TimeInterval,
    predicates_by_id: Mapping[str, PredicateResult],
    observations_by_id: Mapping[str, Observation],
    event_groups_by_id: Mapping[str, EventGroup],
    association_claims_by_id: Mapping[str, AssociationClaim],
    candidate_bound_ids_by_service: Mapping[str, set[str]],
    query_session: CaseQuerySessionResult | None,
) -> bool:
    """Require every original source behind predicates to match one exact role binding."""

    source_ids = set(
        _source_observation_ids(
            predicate_ids=predicate_ids,
            predicates_by_id=predicates_by_id,
            query_session=query_session,
        )
    )
    group = event_groups_by_id.get(event_group_id)
    if not source_ids or group is None:
        return False
    if not source_ids <= set(group.observation_ids):
        return False
    if not source_ids <= candidate_bound_ids_by_service.get(candidate_service_key, set()):
        return False
    if any(
        source_id not in observations_by_id
        or not _overlaps(observations_by_id[source_id].interval, interval)
        for source_id in source_ids
    ):
        return False
    local_relation = local_event_relation_id(
        candidate_service_key=candidate_service_key,
        event_group_id=event_group_id,
    )
    if relation_id == local_relation:
        return not group.association_claim_ids
    relation = association_claims_by_id.get(relation_id)
    return (
        relation is not None
        and relation_id in group.association_claim_ids
        and source_ids <= set(relation.observation_ids)
    )


def _merge_equivalent_assertions(
    assertions: Sequence[RoleAssertion],
) -> tuple[RoleAssertion, ...]:
    assertions_by_effect: dict[tuple[object, ...], list[RoleAssertion]] = {}
    for assertion in assertions:
        effect_key = (
            assertion.hypothesis_id,
            assertion.candidate_service_key,
            assertion.event_group_id,
            assertion.relation_id,
            assertion.interval.start_unix_ns,
            assertion.interval.end_unix_ns,
            assertion.role,
            assertion.strength,
            assertion.score_direction,
            assertion.predicate_result_ids,
            assertion.lifecycle_status,
            assertion.reason,
        )
        assertions_by_effect.setdefault(effect_key, []).append(assertion)
    merged: list[RoleAssertion] = []
    for equivalent_assertions in assertions_by_effect.values():
        representative = equivalent_assertions[0]
        if len(equivalent_assertions) == 1:
            merged.append(representative)
            continue
        merged.append(
            build_role_assertion(
                hypothesis_id=representative.hypothesis_id,
                candidate_service_key=representative.candidate_service_key,
                event_group_id=representative.event_group_id,
                relation_id=representative.relation_id,
                interval=representative.interval,
                role=representative.role,
                strength=representative.strength,
                score_direction=representative.score_direction,
                predicate_result_ids=representative.predicate_result_ids,
                lifecycle_status=representative.lifecycle_status,
                reason=representative.reason,
                provenance_ids=tuple(
                    audit_id
                    for assertion in equivalent_assertions
                    for audit_id in assertion.provenance_ids
                ),
            )
        )
    return tuple(merged)


def _negative_correction_assertions(
    *,
    assertions: Sequence[RoleAssertion],
    proposal_id_by_assertion_id: Mapping[str, str],
    proposals_by_id: Mapping[str, RoleBindingProposal],
    proposal_audit_ids_by_id: Mapping[str, Sequence[str]],
    guards_by_id: Mapping[str, HypothesisGuardResult],
    predicates_by_id: Mapping[str, PredicateResult],
    observations_by_id: Mapping[str, Observation],
    event_groups_by_id: Mapping[str, EventGroup],
    association_claims_by_id: Mapping[str, AssociationClaim],
    candidate_bound_ids_by_service: Mapping[str, set[str]],
    mechanism_library: MechanismLibrary,
    sessions: Mapping[str, CaseQuerySessionResult],
    alternatives_by_hypothesis_id: Mapping[str, Sequence[str]] | None = None,
) -> tuple[RoleAssertion, ...]:
    """Compile a demotion only when every frozen correction premise is true."""

    guard_by_hypothesis_id = {
        guard.parsed_hypothesis.hypothesis_id: guard
        for guard in guards_by_id.values()
        if guard.parsed_hypothesis is not None
    }
    assertion_by_proposal_id: dict[str, RoleAssertion] = {}
    for assertion in assertions:
        owner_proposal_id = proposal_id_by_assertion_id.get(assertion.assertion_id)
        if owner_proposal_id is not None:
            assertion_by_proposal_id[owner_proposal_id] = assertion
    proposal_audit_ids_by_guard_id: dict[str, set[str]] = {}
    assertion_entries_by_guard_and_binding: dict[
        tuple[str, str, str, int, int], list[tuple[RoleBindingProposal, RoleAssertion]]
    ] = {}
    for proposal in proposals_by_id.values():
        proposal_audit_ids_by_guard_id.setdefault(proposal.guard_result_id, set()).update(
            proposal_audit_ids_by_id[proposal.proposal_id]
        )
        indexed_assertion = assertion_by_proposal_id.get(proposal.proposal_id)
        if indexed_assertion is None:
            continue
        assertion_entries_by_guard_and_binding.setdefault(
            (proposal.guard_result_id, *_cross_candidate_binding_key(indexed_assertion)), []
        ).append((proposal, indexed_assertion))
    corrections: list[RoleAssertion] = []
    for local_proposal_id, local_proposal in sorted(proposals_by_id.items()):
        local_assertion = assertion_by_proposal_id.get(local_proposal_id)
        local_guard = guards_by_id[local_proposal.guard_result_id]
        local_hypothesis = local_guard.parsed_hypothesis
        local_impact = local_guard.score_impact
        if local_hypothesis is None or local_impact is None:  # pragma: no cover
            continue
        alternative_ids = (
            local_hypothesis.alternative_hypothesis_ids
            if alternatives_by_hypothesis_id is None
            else alternatives_by_hypothesis_id.get(local_hypothesis.hypothesis_id, ())
        )
        if not (
            local_assertion is not None
            and local_assertion.lifecycle_status == "accepted"
            and local_assertion.score_direction == "promote"
            and local_assertion.role in {"root_support", "source_support"}
            and (local_hypothesis.refuting_predicate_result_ids or alternative_ids)
        ):
            continue
        mechanism = mechanism_library.lookup(local_hypothesis.mechanism_key)
        refuting = tuple(
            predicates_by_id[item] for item in local_hypothesis.refuting_predicate_result_ids
        )
        complete_refutation = (
            bool(mechanism.refuting_predicate_kinds)
            and {item.predicate_kind for item in refuting}
            == set(mechanism.refuting_predicate_kinds)
            and all(item.requirement == "refuting" for item in refuting)
        )
        true_refutation = complete_refutation and all(
            item.truth_value == "true" for item in refuting
        )
        refutation_bound = true_refutation and _predicate_sources_match_binding(
            predicate_ids=local_hypothesis.refuting_predicate_result_ids,
            candidate_service_key=local_assertion.candidate_service_key,
            event_group_id=local_assertion.event_group_id,
            relation_id=local_assertion.relation_id,
            interval=local_assertion.interval,
            predicates_by_id=predicates_by_id,
            observations_by_id=observations_by_id,
            event_groups_by_id=event_groups_by_id,
            association_claims_by_id=association_claims_by_id,
            candidate_bound_ids_by_service=candidate_bound_ids_by_service,
            query_session=sessions.get(local_guard.guard_result_id),
        )
        refutation_quality = (
            _predicate_quality(
                predicate_ids=local_hypothesis.refuting_predicate_result_ids,
                predicates_by_id=predicates_by_id,
                observations_by_id=observations_by_id,
                query_session=sessions.get(local_guard.guard_result_id),
            )
            if refutation_bound
            else None
        )
        alternatives: list[tuple[RoleBindingProposal, RoleAssertion, HypothesisGuardResult]] = []
        alternative_audit_ids: set[str] = set(alternative_ids)
        for alternative_hypothesis_id in alternative_ids:
            alternative_guard = guard_by_hypothesis_id.get(alternative_hypothesis_id)
            if alternative_guard is None or alternative_guard.parsed_hypothesis is None:
                continue
            alternative_audit_ids.add(alternative_guard.guard_result_id)
            alternative_impact = alternative_guard.score_impact
            if alternative_impact is not None:
                alternative_audit_ids.add(alternative_impact.score_impact_id)
                alternative_audit_ids.update(alternative_impact.necessary_predicate_result_ids)
            alternative_audit_ids.update(
                proposal_audit_ids_by_guard_id.get(alternative_guard.guard_result_id, set())
            )
            if (
                alternative_guard.parsed_hypothesis.candidate_service_key
                == local_hypothesis.candidate_service_key
            ):
                continue
            alternative_binding_key = (
                alternative_guard.guard_result_id,
                *_cross_candidate_binding_key(local_assertion),
            )
            for (
                alternative_proposal,
                alternative_assertion,
            ) in assertion_entries_by_guard_and_binding.get(alternative_binding_key, ()):
                if not (
                    alternative_assertion.lifecycle_status == "accepted"
                    and alternative_assertion.score_direction == "promote"
                    and alternative_assertion.role in {"root_support", "source_support"}
                    and alternative_impact is not None
                    and _predicate_sources_match_binding(
                        predicate_ids=alternative_impact.necessary_predicate_result_ids,
                        candidate_service_key=alternative_assertion.candidate_service_key,
                        event_group_id=alternative_assertion.event_group_id,
                        relation_id=alternative_assertion.relation_id,
                        interval=alternative_assertion.interval,
                        predicates_by_id=predicates_by_id,
                        observations_by_id=observations_by_id,
                        event_groups_by_id=event_groups_by_id,
                        association_claims_by_id=association_claims_by_id,
                        candidate_bound_ids_by_service=candidate_bound_ids_by_service,
                        query_session=sessions.get(alternative_guard.guard_result_id),
                    )
                ):
                    continue
                alternatives.append(
                    (alternative_proposal, alternative_assertion, alternative_guard)
                )
        predicate_ids = tuple(
            sorted(
                {
                    *local_hypothesis.refuting_predicate_result_ids,
                    *(
                        local_hypothesis.necessary_predicate_result_ids
                        if not local_hypothesis.refuting_predicate_result_ids
                        else ()
                    ),
                }
            )
        )
        if not refutation_bound or refutation_quality is None or not alternatives:
            reason = (
                "negative_correction_refutation_incomplete"
                if not complete_refutation
                else "negative_correction_refutation_not_true"
                if not true_refutation
                else "negative_correction_refutation_unbound"
                if not refutation_bound
                else "negative_correction_refutation_quality_unknown"
                if refutation_quality is None
                else "negative_correction_alternative_unverified"
            )
            corrections.append(
                build_role_assertion(
                    hypothesis_id=local_assertion.hypothesis_id,
                    candidate_service_key=local_assertion.candidate_service_key,
                    event_group_id=local_assertion.event_group_id,
                    relation_id=local_assertion.relation_id,
                    interval=local_assertion.interval,
                    role=local_assertion.role,
                    strength=0.0,
                    score_direction="neutral",
                    predicate_result_ids=predicate_ids,
                    lifecycle_status="rejected",
                    reason=reason,
                    provenance_ids=(
                        *proposal_audit_ids_by_id[local_proposal.proposal_id],
                        local_guard.guard_result_id,
                        local_impact.score_impact_id,
                        ROLE_COMPILATION_POLICY_SHA256,
                        *predicate_ids,
                        *alternative_audit_ids,
                    ),
                )
            )
            continue
        for alternative_proposal, alternative_assertion, alternative_guard in alternatives:
            alternative_impact = alternative_guard.score_impact
            if alternative_impact is None:  # pragma: no cover - accepted guard invariant
                continue
            correction_predicate_ids = tuple(
                sorted(
                    {
                        *local_hypothesis.refuting_predicate_result_ids,
                        *alternative_impact.necessary_predicate_result_ids,
                    }
                )
            )
            corrections.append(
                build_role_assertion(
                    hypothesis_id=local_assertion.hypothesis_id,
                    candidate_service_key=local_assertion.candidate_service_key,
                    event_group_id=local_assertion.event_group_id,
                    relation_id=local_assertion.relation_id,
                    interval=local_assertion.interval,
                    role=local_assertion.role,
                    strength=min(refutation_quality, alternative_assertion.strength),
                    score_direction="demote",
                    predicate_result_ids=correction_predicate_ids,
                    lifecycle_status="accepted",
                    reason="constrained_negative_correction",
                    provenance_ids=tuple(
                        sorted(
                            {
                                *proposal_audit_ids_by_id[local_proposal.proposal_id],
                                *proposal_audit_ids_by_id[alternative_proposal.proposal_id],
                                local_guard.guard_result_id,
                                alternative_guard.guard_result_id,
                                local_impact.score_impact_id,
                                alternative_impact.score_impact_id,
                                ROLE_COMPILATION_POLICY_SHA256,
                                *correction_predicate_ids,
                            }
                        )
                    ),
                )
            )
    return _merge_equivalent_assertions(corrections)


def _compile_event_role_records(
    *,
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
    alternatives_by_hypothesis_id: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    """Shared record engine; public callers must validate/pin any external graph.

    With no graph override this produces the unchanged version-1 unsigned record.
    Graph-aware callers must seal it under their own graph-bound result version.
    """

    proposal_values = _plain_sequence(proposals, limit=_MAX_PROPOSALS, name="role proposals")
    guard_values = _plain_sequence(
        guard_results, limit=_MAX_GUARDS, name="hypothesis guard results"
    )
    predicate_values = _plain_sequence(predicates, limit=_MAX_PREDICATES, name="predicate results")
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
        raise ValueError("role compiler requires verified frozen typed inputs")
    require_frozen_candidate_universe(
        frozen=frozen_candidates,
        candidate_universe=candidate_universe,
    )
    if not (
        candidate_universe.opaque_case_id
        == entity_registry.opaque_case_id
        == event_grouping.opaque_case_id
    ):
        raise ValueError("role compiler case scope differs")
    sessions = (
        {} if query_sessions_by_guard_result_id is None else dict(query_sessions_by_guard_result_id)
    )
    predicates_by_id: dict[str, PredicateResult] = {}
    for predicate_record in predicate_values:
        if type(predicate_record) is not PredicateResult or not predicate_record.verify():
            raise ValueError("role compiler received an invalid predicate")
        previous_predicate = predicates_by_id.setdefault(
            predicate_record.predicate_result_id, predicate_record
        )
        if previous_predicate != predicate_record:
            raise ValueError("role compiler predicate ID collision")
    guards_by_id: dict[str, HypothesisGuardResult] = {}
    for guard_record in guard_values:
        if type(guard_record) is not HypothesisGuardResult:
            raise ValueError("role compiler received an invalid guard result")
        query_session = sessions.get(guard_record.guard_result_id)
        try:
            submitted_record = json.loads(guard_record.submitted_record_json)
        except (TypeError, ValueError) as error:
            raise ValueError("role compiler guard submission is invalid") from error
        if (
            guard_record.final_disposition != "accepted"
            or guard_record.score_impact is None
            or not guard_record.verify_inputs(
                submitted_record=submitted_record,
                predicates=tuple(predicates_by_id.values()),
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
        ):
            raise ValueError("role compiler requires a replayed accepted guard result")
        previous_guard = guards_by_id.setdefault(guard_record.guard_result_id, guard_record)
        if previous_guard != guard_record:
            raise ValueError("role compiler guard result ID collision")
    return _compile_role_records_core(
        proposal_values=proposal_values, guards_by_id=guards_by_id,
        predicates_by_id=predicates_by_id, candidate_universe=candidate_universe,
        entity_registry=entity_registry, event_grouping=event_grouping,
        mechanism_library=mechanism_library, sessions=sessions,
        alternatives_by_hypothesis_id=alternatives_by_hypothesis_id,
    )


def _compile_role_records_core(
    *, proposal_values: Sequence[object], guards_by_id: dict[str, HypothesisGuardResult],
    predicates_by_id: dict[str, PredicateResult], candidate_universe: CandidateUniverse,
    entity_registry: TemporalEntityRegistry, event_grouping: EventGroupingResult,
    mechanism_library: MechanismLibrary, sessions: dict[str, CaseQuerySessionResult],
    alternatives_by_hypothesis_id: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    """Unchanged role semantics over call-local, source-replayed guard inputs.

    This private engine is not an authorization boundary. Legacy callers above
    retain every source/guard replay; the live owner provides its full batch gate
    and protects source and intermediate state through both exits.
    """
    if set(sessions) - set(guards_by_id):
        raise ValueError("role compiler received an unrelated query session")
    proposals_by_id: dict[str, RoleBindingProposal] = {}
    for proposal_record in proposal_values:
        if type(proposal_record) is not RoleBindingProposal or not proposal_record.verify():
            raise ValueError("role compiler received an invalid role proposal")
        previous_proposal = proposals_by_id.setdefault(proposal_record.proposal_id, proposal_record)
        if previous_proposal != proposal_record:
            raise ValueError("role compiler proposal ID collision")
        if proposal_record.guard_result_id not in guards_by_id:
            raise ValueError("role proposal references an unknown guard result")
    canonical_proposals_by_id, proposal_audit_ids_by_id = _canonical_role_proposals(
        proposals_by_id=proposals_by_id,
        guards_by_id=guards_by_id,
    )
    event_groups_by_id = {item.event_group_id: item for item in event_grouping.event_groups}
    association_claims_by_id = {item.claim_id: item for item in event_grouping.association_claims}
    observations_by_id = {item.observation_id: item for item in event_grouping.observations}
    candidate_bound_ids_by_service: dict[str, set[str]] = {}
    for resolution in entity_registry.resolutions:
        if (
            resolution.binding.status == "bound"
            and resolution.binding.service_entity_key is not None
        ):
            candidate_bound_ids_by_service.setdefault(
                resolution.binding.service_entity_key, set()
            ).add(resolution.observation.observation_id)
    assertions: list[RoleAssertion] = []
    proposal_id_by_assertion_id: dict[str, str] = {}
    for proposal in sorted(canonical_proposals_by_id.values(), key=lambda item: item.proposal_id):
        guard = guards_by_id[proposal.guard_result_id]
        hypothesis = guard.parsed_hypothesis
        impact = guard.score_impact
        if hypothesis is None or impact is None:  # pragma: no cover - accepted guard invariant
            raise AssertionError("accepted guard has no compiled hypothesis")
        query_session = sessions.get(guard.guard_result_id)
        necessary_source_ids = set(
            _source_observation_ids(
                predicate_ids=impact.necessary_predicate_result_ids,
                predicates_by_id=predicates_by_id,
                query_session=query_session,
            )
        )
        candidate_bound_ids = candidate_bound_ids_by_service.get(
            hypothesis.candidate_service_key, set()
        )
        reason = _proposal_rejection(
            proposal=proposal,
            guard=guard,
            event_groups_by_id=event_groups_by_id,
            association_claims_by_id=association_claims_by_id,
            observations_by_id=observations_by_id,
            candidate_bound_observation_ids=candidate_bound_ids,
            necessary_source_ids=necessary_source_ids,
        )
        qualities = tuple(
            observations_by_id[item].coverage_quality
            for item in sorted(necessary_source_ids)
            if item in observations_by_id
        )
        if reason is None and len(qualities) != len(necessary_source_ids):
            reason = "role_strength_unknown"
        if reason is not None:
            assertions.append(
                _rejected_assertion(
                    proposal=proposal,
                    guard=guard,
                    predicate_result_ids=impact.necessary_predicate_result_ids,
                    reason=reason,
                    proposal_audit_ids=proposal_audit_ids_by_id[proposal.proposal_id],
                )
            )
            continue
        strength = min(qualities)
        direction: Literal["promote", "neutral", "demote"] = (
            "neutral" if proposal.role == "symptom_support" else "promote"
        )
        assertion = build_role_assertion(
            hypothesis_id=hypothesis.hypothesis_id,
            candidate_service_key=hypothesis.candidate_service_key,
            event_group_id=proposal.event_group_id,
            relation_id=proposal.relation_id,
            interval=proposal.interval,
            role=proposal.role,
            strength=strength,
            score_direction=direction,
            predicate_result_ids=impact.necessary_predicate_result_ids,
            lifecycle_status="accepted",
            reason=f"verified_{proposal.role}",
            provenance_ids=tuple(
                sorted(
                    {
                        proposal.proposal_id,
                        guard.guard_result_id,
                        impact.score_impact_id,
                        ROLE_COMPILATION_POLICY_SHA256,
                        *proposal_audit_ids_by_id[proposal.proposal_id],
                        *impact.necessary_predicate_result_ids,
                    }
                )
            ),
        )
        assertions.append(assertion)
        proposal_id_by_assertion_id[assertion.assertion_id] = proposal.proposal_id
    neutralized_assertions, proposal_id_by_assertion_id = _conflict_neutralized_assertions(
        assertions=assertions,
        proposal_id_by_assertion_id=proposal_id_by_assertion_id,
        proposals_by_id=canonical_proposals_by_id,
        guards_by_id=guards_by_id,
        alternatives_by_hypothesis_id=alternatives_by_hypothesis_id,
    )
    assertions = list(neutralized_assertions)
    assertions.extend(
        _negative_correction_assertions(
            assertions=assertions,
            proposal_id_by_assertion_id=proposal_id_by_assertion_id,
            proposals_by_id=canonical_proposals_by_id,
            proposal_audit_ids_by_id=proposal_audit_ids_by_id,
            guards_by_id=guards_by_id,
            predicates_by_id=predicates_by_id,
            observations_by_id=observations_by_id,
            event_groups_by_id=event_groups_by_id,
            association_claims_by_id=association_claims_by_id,
            candidate_bound_ids_by_service=candidate_bound_ids_by_service,
            mechanism_library=mechanism_library,
            sessions=sessions,
            alternatives_by_hypothesis_id=alternatives_by_hypothesis_id,
        )
    )
    ordered_proposals = tuple(sorted(proposals_by_id.values(), key=lambda item: item.proposal_id))
    ordered_assertions = tuple(sorted(assertions, key=lambda item: item.assertion_id))
    unsigned: dict[str, object] = {
        "schema_version": ROLE_COMPILATION_RESULT_SCHEMA_VERSION,
        "opaque_case_id": candidate_universe.opaque_case_id,
        "policy_version": ROLE_COMPILATION_POLICY_VERSION,
        "policy_sha256": ROLE_COMPILATION_POLICY_SHA256,
        "guard_result_ids": tuple(sorted(guards_by_id)),
        "proposals": tuple(item.model_dump(mode="json") for item in ordered_proposals),
        "assertions": tuple(item.model_dump(mode="json") for item in ordered_assertions),
    }
    return unsigned


def compile_event_roles(
    *,
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
) -> RoleCompilationResult:
    """Unchanged legacy path: only embedded alternatives affect version-1 output."""
    unsigned = _compile_event_role_records(
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
    identifier, digest = _stable_id("rcr", unsigned)
    return RoleCompilationResult.model_validate(
        {**unsigned, "role_compilation_id": identifier, "result_sha256": digest}
    )
