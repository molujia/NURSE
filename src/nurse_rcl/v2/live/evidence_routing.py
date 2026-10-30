"""Versioned predicate domains, independently replayed from complete old sources.

This is an internal producer, not a scoring permission or an outbound payload.
Selection reads contracts, bindings and scope, never anomaly magnitude or truth.
Old proposal/envelope bytes and their original verifier remain unchanged.
"""

from __future__ import annotations

import hmac
import re
from contextlib import ExitStack
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2 import predicate_plan_evaluator
from nurse_rcl.v2.b0_rule_binding import _bind_rule_values, _bytes, _require_unchanged, _snapshot
from nurse_rcl.v2.diagnostic_schemas import (
    DiagnosticHypothesis,
    PredicateRequirement,
    PredicateResult,
    build_diagnostic_hypothesis,
    build_predicate_result,
)
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.hypothesis_guard import (
    _PREDICATE_EVIDENCE_CONTRACT_SHA256,
    _PREDICATE_EVIDENCE_CONTRACTS,
    PREDICATE_EVIDENCE_CONTRACT_VERSION,
)
from nurse_rcl.v2.live_capacity import binding_capacity
from nurse_rcl.v2.observation_schemas import TimeInterval
from nurse_rcl.v2.rule_hypotheses import PendingVerificationPlan, RuleHypothesisEnvelope

ROUTED_RULE_GENERATOR_VERSION = "nurse-v2-live-applicable-rule-hypotheses/1"
ROUTED_RULE_SET_SCHEMA_VERSION = "nurse.v2.live-routed-rule-set/1"
ROUTED_B0_BINDING_SCHEMA_VERSION = "nurse.v2.live-routed-b0-binding/1"


class RoutedObservationBinding(StrictFrozenModel):
    observation_id: str = Field(pattern=r"^obs_[0-9a-f]{24}$")
    observation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    entity_binding_id: str | None
    interval: TimeInterval
    unit: str | None
    source_signal_family: str
    primary_signal_family: str | None
    adapter_semantics_sha256: str | None


class NonApplicableObservation(StrictFrozenModel):
    observation_id: str = Field(pattern=r"^obs_[0-9a-f]{24}$")
    reason: Literal["predicate_signal_family_denied"] = "predicate_signal_family_denied"


class PredicateApplicability(StrictFrozenModel):
    candidate_service_key: str
    mechanism_key: str
    predicate_kind: str
    requirement: PredicateRequirement
    scope_interval: TimeInterval
    original_predicate_result_id: str = Field(pattern=r"^prd_[0-9a-f]{24}$")
    predicate_result_id: str = Field(pattern=r"^prd_[0-9a-f]{24}$")
    original_observation_ids: tuple[str, ...]
    applicable_observation_ids: tuple[str, ...]
    non_applicable: tuple[NonApplicableObservation, ...]
    observation_bindings: tuple[RoutedObservationBinding, ...]
    domain_reason: Literal["applicable_evidence_retained", "no_applicable_evidence"]


class RoutedRuleSet(StrictFrozenModel):
    schema_version: Literal["nurse.v2.live-routed-rule-set/1"] = "nurse.v2.live-routed-rule-set/1"
    generator_version: Literal["nurse-v2-live-applicable-rule-hypotheses/1"] = (
        "nurse-v2-live-applicable-rule-hypotheses/1"
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    source_rule_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_contract_version: Literal["nurse-v2-predicate-evidence-contract/1"]
    evidence_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_service_keys: tuple[str, ...]
    mechanism_keys: tuple[str, ...]
    routes: tuple[PredicateApplicability, ...]
    hypotheses: tuple[DiagnosticHypothesis, ...]
    predicates: tuple[PredicateResult, ...]
    verification_plans: tuple[PendingVerificationPlan, ...]
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self, *, capacity_profile: str | None = None) -> bool:
        """Internal consistency only; require_routed_rule_set grants source binding."""
        try:
            with binding_capacity(capacity_profile):
                if type(self) is not RoutedRuleSet:
                    return False
                raw = _snapshot(self, RoutedRuleSet)[0]
                fresh = RoutedRuleSet.model_validate_json(raw)
                if raw != _bytes(fresh) or fresh.result_sha256 != _digest(
                    fresh.model_dump(mode="json", exclude={"result_sha256"})
                ):
                    return False
                if fresh.evidence_contract_sha256 != _PREDICATE_EVIDENCE_CONTRACT_SHA256:
                    return False
                predicates = {p.predicate_result_id: p for p in fresh.predicates}
                if len(predicates) != len(fresh.predicates):
                    return False
                if not all(p.verify() for p in fresh.predicates):
                    return False
                if not all(p.verify() for p in fresh.verification_plans):
                    return False
                if not all(
                    h.verify() and h.origin_version == ROUTED_RULE_GENERATOR_VERSION
                    for h in fresh.hypotheses
                ):
                    return False
                if tuple(
                    (h.candidate_service_key, h.mechanism_key) for h in fresh.hypotheses
                ) != tuple(
                    (c, m) for c in fresh.candidate_service_keys for m in fresh.mechanism_keys
                ):
                    return False
                if len({r.original_predicate_result_id for r in fresh.routes}) != len(fresh.routes):
                    return False
                if {r.predicate_result_id for r in fresh.routes} != set(predicates):
                    return False
                for route in fresh.routes:
                    original = set(route.original_observation_ids)
                    applicable = set(route.applicable_observation_ids)
                    excluded = {r.observation_id for r in route.non_applicable}
                    if applicable & excluded or applicable | excluded != original:
                        return False
                    if (
                        tuple(b.observation_id for b in route.observation_bindings)
                        != route.original_observation_ids
                    ):
                        return False
                    if (
                        predicates[route.predicate_result_id].observation_ids
                        != route.applicable_observation_ids
                    ):
                        return False
                return True
        except (AttributeError, KeyError, RecursionError, TypeError, ValueError):
            return False


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _pin(value: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("evidence routing requires independent exact SHA256 pins")


def _routing_values(source: RuleHypothesisEnvelope) -> RoutedRuleSet:
    """Pure projection inside the caller's original/fresh replay scope."""
    result = RoutedRuleSet(**_routing_payload(source, indexed=False), result_sha256="0" * 64)
    return result.model_copy(
        update={"result_sha256": _digest(result.model_dump(mode="json", exclude={"result_sha256"}))}
    )


def _routing_payload(source: RuleHypothesisEnvelope, *, indexed: bool) -> dict[str, Any]:
    """Pure values only: representation choice grants no validation authority.

    The /1 caller still constructs expanded bindings. The /2 caller constructs
    each table value once directly, without materializing an expanded document.
    Every public producer/consumer separately enforces its existing replay scope.
    """
    seeds = source.rule_hypothesis_set
    observations = {o.observation_id: o for o in source.event_grouping.observations}
    primary = {
        i: g.primary_signal_family
        for g in source.event_grouping.event_groups
        for i in g.observation_ids
    }
    seed_predicates = {p.predicate_result_id: p for p in seeds.predicates}
    seed_plans = {p.predicate_result_id: p for p in seeds.verification_plans}
    predicates: dict[str, PredicateResult] = {}
    plans: list[PendingVerificationPlan] = []
    routes: list[dict[str, Any]] = []
    bindings: dict[str, RoutedObservationBinding] = {}
    hypotheses: list[DiagnosticHypothesis] = []

    def observation_binding(i: str) -> RoutedObservationBinding:
        return RoutedObservationBinding(
            observation_id=i,
            observation_sha256=_digest(observations[i].model_dump(mode="json")),
            source_fingerprint=observations[i].source_fingerprint,
            entity_binding_id=observations[i].entity_binding_id,
            interval=observations[i].interval,
            unit=observations[i].unit,
            source_signal_family=observations[i].signal_family,
            primary_signal_family=primary.get(i),
            adapter_semantics_sha256=observations[i].adapter_semantics_sha256,
        )

    for hypothesis in seeds.hypotheses:
        mapped: dict[PredicateRequirement, list[str]] = {
            "necessary": [],
            "optional": [],
            "refuting": [],
        }
        for _requirement, ids in (
            ("necessary", hypothesis.necessary_predicate_result_ids),
            ("optional", hypothesis.optional_predicate_result_ids),
            ("refuting", hypothesis.refuting_predicate_result_ids),
        ):
            for seed_id in ids:
                seed = seed_predicates[seed_id]
                contract = _PREDICATE_EVIDENCE_CONTRACTS[seed.predicate_kind]
                applicable = tuple(
                    i
                    for i in seed.observation_ids
                    if primary.get(i) in contract.primary_signal_families
                    or observations[i].signal_family in contract.source_signal_families
                )
                payload = seed.model_dump(
                    mode="python",
                    exclude={"schema_version", "predicate_result_id", "predicate_sha256"},
                )
                payload.update(
                    observation_ids=applicable,
                    reason="pending_verification" if applicable else "no_bound_observations",
                    provenance_ids=tuple(
                        sorted(
                            (
                                *seed.provenance_ids,
                                f"evidence-routing:{ROUTED_RULE_GENERATOR_VERSION}",
                                f"evidence-contract:{_PREDICATE_EVIDENCE_CONTRACT_SHA256}",
                            )
                        )
                    ),
                )
                predicate = build_predicate_result(**payload)
                predicates[predicate.predicate_result_id] = predicate
                mapped[seed.requirement].append(predicate.predicate_result_id)
                plan_payload = seed_plans[seed_id].model_dump(
                    mode="json", exclude={"plan_id", "plan_sha256"}
                )
                plan_payload.update(
                    observation_ids=applicable, predicate_result_id=predicate.predicate_result_id
                )
                plan_digest = _digest(plan_payload)
                plans.append(
                    PendingVerificationPlan(
                        **plan_payload, plan_id=f"vpl_{plan_digest[:24]}", plan_sha256=plan_digest
                    )
                )
                route = dict(
                    candidate_service_key=hypothesis.candidate_service_key,
                    mechanism_key=hypothesis.mechanism_key,
                    predicate_kind=seed.predicate_kind,
                    requirement=seed.requirement,
                    scope_interval=source.scope_interval,
                    original_predicate_result_id=seed_id,
                    predicate_result_id=predicate.predicate_result_id,
                    original_observation_ids=seed.observation_ids,
                    applicable_observation_ids=applicable,
                    non_applicable=tuple(
                        NonApplicableObservation(observation_id=i)
                        for i in seed.observation_ids
                        if i not in applicable
                    ),
                    domain_reason="applicable_evidence_retained"
                    if applicable
                    else "no_applicable_evidence",
                )
                if indexed:
                    for i in seed.observation_ids:
                        if i not in bindings:
                            bindings[i] = observation_binding(i)
                else:
                    route["observation_bindings"] = tuple(
                        observation_binding(i) for i in seed.observation_ids
                    )
                routes.append(route)
        hypothesis_payload = hypothesis.model_dump(
            mode="python", exclude={"schema_version", "hypothesis_id", "hypothesis_sha256"}
        )
        hypothesis_payload.update(
            origin_version=ROUTED_RULE_GENERATOR_VERSION,
            time_intervals=hypothesis.time_intervals,
            necessary_predicate_result_ids=tuple(mapped["necessary"]),
            optional_predicate_result_ids=tuple(mapped["optional"]),
            refuting_predicate_result_ids=tuple(mapped["refuting"]),
            provenance_ids=tuple(
                sorted(
                    (
                        *hypothesis.provenance_ids,
                        f"evidence-routing:{ROUTED_RULE_GENERATOR_VERSION}",
                    )
                )
            ),
        )
        hypotheses.append(build_diagnostic_hypothesis(**hypothesis_payload))
    payload = dict(
        opaque_case_id=seeds.opaque_case_id,
        source_rule_set_sha256=seeds.result_sha256,
        evidence_contract_version=PREDICATE_EVIDENCE_CONTRACT_VERSION,
        evidence_contract_sha256=_PREDICATE_EVIDENCE_CONTRACT_SHA256,
        candidate_service_keys=seeds.candidate_service_keys,
        mechanism_keys=seeds.mechanism_keys,
        routes=tuple(sorted(routes, key=lambda r: r["original_predicate_result_id"])),
        hypotheses=tuple(hypotheses),
        predicates=tuple(predicates[k] for k in sorted(predicates)),
        verification_plans=tuple(sorted(plans, key=lambda p: p.plan_id)),
    )
    if indexed:
        payload["observation_bindings"] = tuple(bindings[i] for i in sorted(bindings))
    return payload


def build_routed_rule_set(
    *,
    envelope: RuleHypothesisEnvelope,
    expected_rule_set_sha256: str,
    capacity_profile: str | None = None,
) -> RoutedRuleSet:
    with binding_capacity(capacity_profile):
        if type(envelope) is not RuleHypothesisEnvelope:
            raise ValueError("evidence routing requires an exact original envelope")
        _pin(expected_rule_set_sha256)
        before = _snapshot(envelope, RuleHypothesisEnvelope)
        with ExitStack() as guards:
            guards.callback(_require_unchanged, envelope, before, RuleHypothesisEnvelope)
            fresh = RuleHypothesisEnvelope.model_validate_json(before[0])
            fresh_before = _snapshot(fresh, RuleHypothesisEnvelope)
            guards.callback(_require_unchanged, fresh, fresh_before, RuleHypothesisEnvelope)
            if (
                before[0] != fresh_before[0]
                or not hmac.compare_digest(
                    fresh.rule_hypothesis_set.result_sha256, expected_rule_set_sha256
                )
                or not fresh.verify()
            ):
                raise ValueError("evidence routing source fails independent complete replay")
            result = _routing_values(fresh)
        return result


def require_routed_rule_set(
    *,
    result: RoutedRuleSet,
    expected_result_sha256: str,
    envelope: RuleHypothesisEnvelope,
    expected_rule_set_sha256: str,
    capacity_profile: str | None = None,
) -> RoutedRuleSet:
    with binding_capacity(capacity_profile):
        if type(result) is not RoutedRuleSet or type(envelope) is not RuleHypothesisEnvelope:
            raise ValueError("evidence routing requires exact registered artifact types")
        _pin(expected_result_sha256)
        _pin(expected_rule_set_sha256)
        before = _snapshot(result, RoutedRuleSet)
        source_before = _snapshot(envelope, RuleHypothesisEnvelope)
        with ExitStack() as guards:
            guards.callback(_require_unchanged, result, before, RoutedRuleSet)
            guards.callback(_require_unchanged, envelope, source_before, RuleHypothesisEnvelope)
            if not hmac.compare_digest(result.result_sha256, expected_result_sha256):
                raise ValueError("evidence routing fails independent result pin")
            rebuilt = build_routed_rule_set(
                envelope=envelope,
                expected_rule_set_sha256=expected_rule_set_sha256,
                capacity_profile=capacity_profile,
            )
            if not hmac.compare_digest(before[0], _bytes(rebuilt)):
                raise ValueError("evidence routing differs from independent complete source replay")
        return rebuilt


class RoutedB0Binding(StrictFrozenModel):
    """New binding identity: neither this digest nor verify() proves source authority."""

    schema_version: Literal["nurse.v2.live-routed-b0-binding/1"] = (
        "nurse.v2.live-routed-b0-binding/1"
    )
    policy_version: Literal["live-applicable-observation-rule-binding/1"] = (
        "live-applicable-observation-rule-binding/1"
    )
    predicate_evaluation_policy_version: Literal["original-observation-predicate-plan/1"] = (
        "original-observation-predicate-plan/1"
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    source_rule_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    routing_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predicate_id_mapping: tuple[tuple[str, str], ...]
    hypothesis_id_mapping: tuple[tuple[str, str], ...]
    predicates: tuple[PredicateResult, ...]
    hypotheses: tuple[DiagnosticHypothesis, ...]
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self, *, capacity_profile: str | None = None) -> bool:
        try:
            with binding_capacity(capacity_profile):
                if type(self) is not RoutedB0Binding:
                    return False
                raw = _snapshot(self, RoutedB0Binding)[0]
                fresh = RoutedB0Binding.model_validate_json(raw)
                return (
                    raw == _bytes(fresh)
                    and fresh.binding_sha256
                    == _digest(fresh.model_dump(mode="json", exclude={"binding_sha256"}))
                    and all(p.verify() for p in fresh.predicates)
                    and all(
                        h.verify() and h.origin_version == ROUTED_RULE_GENERATOR_VERSION
                        for h in fresh.hypotheses
                    )
                )
        except (AttributeError, KeyError, RecursionError, TypeError, ValueError):
            return False


def build_routed_b0_binding(
    *,
    envelope: RuleHypothesisEnvelope,
    routing: RoutedRuleSet,
    expected_rule_set_sha256: str,
    expected_routing_sha256: str,
    capacity_profile: str | None = None,
) -> RoutedB0Binding:
    with binding_capacity(capacity_profile):
        if type(envelope) is not RuleHypothesisEnvelope or type(routing) is not RoutedRuleSet:
            raise ValueError("routed binding requires exact registered sources")
        _pin(expected_rule_set_sha256)
        _pin(expected_routing_sha256)
        before = _snapshot(envelope, RuleHypothesisEnvelope)
        routing_before = _snapshot(routing, RoutedRuleSet)
        with ExitStack() as guards:
            guards.callback(_require_unchanged, envelope, before, RuleHypothesisEnvelope)
            guards.callback(_require_unchanged, routing, routing_before, RoutedRuleSet)
            fresh = RuleHypothesisEnvelope.model_validate_json(before[0])
            guards.callback(
                _require_unchanged,
                fresh,
                _snapshot(fresh, RuleHypothesisEnvelope),
                RuleHypothesisEnvelope,
            )
            seeds = require_routed_rule_set(
                result=routing,
                expected_result_sha256=expected_routing_sha256,
                envelope=fresh,
                expected_rule_set_sha256=expected_rule_set_sha256,
                capacity_profile=capacity_profile,
            )
            guards.callback(
                _require_unchanged, seeds, _snapshot(seeds, RoutedRuleSet), RoutedRuleSet
            )
            grouping = predicate_plan_evaluator._prepare_predicate_grouping(fresh.event_grouping)
            guards.callback(
                _require_unchanged,
                grouping,
                _snapshot(grouping, EventGroupingResult),
                EventGroupingResult,
            )
            pm, hm, predicates, hypotheses = _bind_rule_values(
                seed_predicates=seeds.predicates,
                seed_hypotheses=seeds.hypotheses,
                grouping=grouping,
            )
            result = RoutedB0Binding(
                opaque_case_id=seeds.opaque_case_id,
                source_rule_set_sha256=expected_rule_set_sha256,
                routing_result_sha256=seeds.result_sha256,
                predicate_id_mapping=pm,
                hypothesis_id_mapping=hm,
                predicates=predicates,
                hypotheses=hypotheses,
                binding_sha256="0" * 64,
            )
            result = result.model_copy(
                update={
                    "binding_sha256": _digest(
                        result.model_dump(mode="json", exclude={"binding_sha256"})
                    )
                }
            )
        return result


def require_routed_b0_binding(
    *,
    binding: RoutedB0Binding,
    expected_binding_sha256: str,
    envelope: RuleHypothesisEnvelope,
    routing: RoutedRuleSet,
    expected_rule_set_sha256: str,
    expected_routing_sha256: str,
    capacity_profile: str | None = None,
) -> RoutedB0Binding:
    with binding_capacity(capacity_profile):
        if (
            type(binding) is not RoutedB0Binding
            or type(envelope) is not RuleHypothesisEnvelope
            or type(routing) is not RoutedRuleSet
        ):
            raise ValueError("routed binding requires exact registered artifacts")
        _pin(expected_binding_sha256)
        _pin(expected_rule_set_sha256)
        _pin(expected_routing_sha256)
        with ExitStack() as guards:
            for value, kind in (
                (binding, RoutedB0Binding),
                (envelope, RuleHypothesisEnvelope),
                (routing, RoutedRuleSet),
            ):
                guards.callback(_require_unchanged, value, _snapshot(value, kind), kind)
            original_bytes = _bytes(binding)
            if not hmac.compare_digest(binding.binding_sha256, expected_binding_sha256):
                raise ValueError("routed binding fails independent binding pin")
            rebuilt = build_routed_b0_binding(
                envelope=envelope,
                routing=routing,
                expected_rule_set_sha256=expected_rule_set_sha256,
                expected_routing_sha256=expected_routing_sha256,
                capacity_profile=capacity_profile,
            )
            if not hmac.compare_digest(original_bytes, _bytes(rebuilt)):
                raise ValueError("routed binding differs from independent complete source replay")
        return rebuilt
