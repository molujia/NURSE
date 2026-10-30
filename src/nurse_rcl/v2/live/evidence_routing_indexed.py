"""Indexed /2 routing representation with unchanged source/exit boundaries.

Bindings are values, not authority tokens. Every call reconstructs them from
the complete original source. No expanded /1 routing object is constructed on
this path and no verified context or cross-call cache is introduced.
"""

from __future__ import annotations

import hmac
from contextlib import ExitStack
from typing import Literal

from pydantic import Field

from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2 import predicate_plan_evaluator
from nurse_rcl.v2.b0_rule_binding import _bind_rule_values, _bytes, _require_unchanged, _snapshot
from nurse_rcl.v2.diagnostic_schemas import (
    DiagnosticHypothesis,
    PredicateRequirement,
    PredicateResult,
)
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.hypothesis_guard import _PREDICATE_EVIDENCE_CONTRACT_SHA256
from nurse_rcl.v2.live.evidence_routing import (
    ROUTED_RULE_GENERATOR_VERSION,
    NonApplicableObservation,
    RoutedObservationBinding,
    _digest,
    _pin,
    _routing_payload,
)
from nurse_rcl.v2.live_capacity import binding_capacity
from nurse_rcl.v2.observation_schemas import TimeInterval
from nurse_rcl.v2.rule_hypotheses import PendingVerificationPlan, RuleHypothesisEnvelope

ROUTED_RULE_SET_SCHEMA_VERSION = "nurse.v2.live-routed-rule-set/2"
ROUTED_B0_BINDING_SCHEMA_VERSION = "nurse.v2.live-routed-b0-binding/2"


class IndexedPredicateApplicability(StrictFrozenModel):
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
    domain_reason: Literal["applicable_evidence_retained", "no_applicable_evidence"]


class IndexedRoutedRuleSet(StrictFrozenModel):
    schema_version: Literal["nurse.v2.live-routed-rule-set/2"] = "nurse.v2.live-routed-rule-set/2"
    generator_version: Literal["nurse-v2-live-applicable-rule-hypotheses/1"] = (
        "nurse-v2-live-applicable-rule-hypotheses/1"
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    source_rule_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_contract_version: Literal["nurse-v2-predicate-evidence-contract/1"]
    evidence_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_service_keys: tuple[str, ...]
    mechanism_keys: tuple[str, ...]
    observation_bindings: tuple[RoutedObservationBinding, ...]
    routes: tuple[IndexedPredicateApplicability, ...]
    hypotheses: tuple[DiagnosticHypothesis, ...]
    predicates: tuple[PredicateResult, ...]
    verification_plans: tuple[PendingVerificationPlan, ...]
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self, *, capacity_profile: str | None = None) -> bool:
        """Internal consistency only, never source authority."""
        try:
            with binding_capacity(capacity_profile):
                if type(self) is not IndexedRoutedRuleSet:
                    return False
                raw = _snapshot(self, IndexedRoutedRuleSet)[0]
                fresh = IndexedRoutedRuleSet.model_validate_json(raw)
                if raw != _bytes(fresh) or fresh.result_sha256 != _digest(
                    fresh.model_dump(mode="json", exclude={"result_sha256"})
                ):
                    return False
                if fresh.evidence_contract_sha256 != _PREDICATE_EVIDENCE_CONTRACT_SHA256:
                    return False
                ids = tuple(b.observation_id for b in fresh.observation_bindings)
                if ids != tuple(sorted(set(ids))) or set(ids) != {
                    i for r in fresh.routes for i in r.original_observation_ids
                }:
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
                    if (
                        len(original) != len(route.original_observation_ids)
                        or len(applicable) != len(route.applicable_observation_ids)
                        or len(excluded) != len(route.non_applicable)
                        or applicable & excluded
                        or applicable | excluded != original
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


def _routing_values(source: RuleHypothesisEnvelope) -> IndexedRoutedRuleSet:
    result = IndexedRoutedRuleSet(**_routing_payload(source, indexed=True), result_sha256="0" * 64)
    return result.model_copy(
        update={"result_sha256": _digest(result.model_dump(mode="json", exclude={"result_sha256"}))}
    )


def build_routed_rule_set(
    *,
    envelope: RuleHypothesisEnvelope,
    expected_rule_set_sha256: str,
    capacity_profile: str | None = None,
) -> IndexedRoutedRuleSet:
    with binding_capacity(capacity_profile):
        if type(envelope) is not RuleHypothesisEnvelope:
            raise ValueError("indexed evidence routing requires an exact original envelope")
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
                raise ValueError(
                    "indexed evidence routing source fails independent complete replay"
                )
            result = _routing_values(fresh)
        return result


def require_routed_rule_set(
    *,
    result: IndexedRoutedRuleSet,
    expected_result_sha256: str,
    envelope: RuleHypothesisEnvelope,
    expected_rule_set_sha256: str,
    capacity_profile: str | None = None,
) -> IndexedRoutedRuleSet:
    with binding_capacity(capacity_profile):
        if type(result) is not IndexedRoutedRuleSet or type(envelope) is not RuleHypothesisEnvelope:
            raise ValueError("indexed evidence routing requires exact registered artifact types")
        _pin(expected_result_sha256)
        _pin(expected_rule_set_sha256)
        before = _snapshot(result, IndexedRoutedRuleSet)
        source_before = _snapshot(envelope, RuleHypothesisEnvelope)
        with ExitStack() as guards:
            guards.callback(_require_unchanged, result, before, IndexedRoutedRuleSet)
            guards.callback(_require_unchanged, envelope, source_before, RuleHypothesisEnvelope)
            if not hmac.compare_digest(result.result_sha256, expected_result_sha256):
                raise ValueError("indexed evidence routing fails independent result pin")
            rebuilt = build_routed_rule_set(
                envelope=envelope,
                expected_rule_set_sha256=expected_rule_set_sha256,
                capacity_profile=capacity_profile,
            )
            if not hmac.compare_digest(before[0], _bytes(rebuilt)):
                raise ValueError(
                    "indexed evidence routing differs from independent complete source replay"
                )
        return rebuilt


class IndexedRoutedB0Binding(StrictFrozenModel):
    schema_version: Literal["nurse.v2.live-routed-b0-binding/2"] = (
        "nurse.v2.live-routed-b0-binding/2"
    )
    policy_version: Literal["live-applicable-observation-rule-binding/2"] = (
        "live-applicable-observation-rule-binding/2"
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
                if type(self) is not IndexedRoutedB0Binding:
                    return False
                raw = _snapshot(self, IndexedRoutedB0Binding)[0]
                fresh = IndexedRoutedB0Binding.model_validate_json(raw)
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
    routing: IndexedRoutedRuleSet,
    expected_rule_set_sha256: str,
    expected_routing_sha256: str,
    capacity_profile: str | None = None,
) -> IndexedRoutedB0Binding:
    with binding_capacity(capacity_profile):
        if (
            type(envelope) is not RuleHypothesisEnvelope
            or type(routing) is not IndexedRoutedRuleSet
        ):
            raise ValueError("indexed routed binding requires exact registered sources")
        _pin(expected_rule_set_sha256)
        _pin(expected_routing_sha256)
        before = _snapshot(envelope, RuleHypothesisEnvelope)
        routing_before = _snapshot(routing, IndexedRoutedRuleSet)
        with ExitStack() as guards:
            guards.callback(_require_unchanged, envelope, before, RuleHypothesisEnvelope)
            guards.callback(_require_unchanged, routing, routing_before, IndexedRoutedRuleSet)
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
                _require_unchanged,
                seeds,
                _snapshot(seeds, IndexedRoutedRuleSet),
                IndexedRoutedRuleSet,
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
            result = IndexedRoutedB0Binding(
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
    binding: IndexedRoutedB0Binding,
    expected_binding_sha256: str,
    envelope: RuleHypothesisEnvelope,
    routing: IndexedRoutedRuleSet,
    expected_rule_set_sha256: str,
    expected_routing_sha256: str,
    capacity_profile: str | None = None,
) -> IndexedRoutedB0Binding:
    with binding_capacity(capacity_profile):
        if (
            type(binding) is not IndexedRoutedB0Binding
            or type(envelope) is not RuleHypothesisEnvelope
            or type(routing) is not IndexedRoutedRuleSet
        ):
            raise ValueError("indexed routed binding requires exact registered artifacts")
        for pin in (expected_binding_sha256, expected_rule_set_sha256, expected_routing_sha256):
            _pin(pin)
        with ExitStack() as guards:
            for value, kind in (
                (binding, IndexedRoutedB0Binding),
                (envelope, RuleHypothesisEnvelope),
                (routing, IndexedRoutedRuleSet),
            ):
                guards.callback(_require_unchanged, value, _snapshot(value, kind), kind)
            original_bytes = _bytes(binding)
            if not hmac.compare_digest(binding.binding_sha256, expected_binding_sha256):
                raise ValueError("indexed routed binding fails independent binding pin")
            rebuilt = build_routed_b0_binding(
                envelope=envelope,
                routing=routing,
                expected_rule_set_sha256=expected_rule_set_sha256,
                expected_routing_sha256=expected_routing_sha256,
                capacity_profile=capacity_profile,
            )
            if not hmac.compare_digest(original_bytes, _bytes(rebuilt)):
                raise ValueError(
                    "indexed routed binding differs from independent complete source replay"
                )
        return rebuilt
