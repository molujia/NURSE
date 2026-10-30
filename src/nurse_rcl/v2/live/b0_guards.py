"""Invocation-local B0 guards: shared source replay, unchanged four-layer logic.

No prepared context escapes this function. A later invocation replays again.
Progress is diagnostic only: no guard result is returned until every original
and working input has passed its normal/exceptional unchanged-state boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from typing import Any

from nurse_rcl.v2 import hypothesis_guard as guard
from nurse_rcl.v2.b0_rule_binding import (
    B0RuleBinding,
    _preflight_inputs,
    _require_unchanged,
    _snapshot,
    require_b0_rule_binding,
)
from nurse_rcl.v2.hypothesis_guard import HypothesisGuardResult
from nurse_rcl.v2.live_capacity import LIVE_B0_CAPACITY_PROFILE, binding_capacity
from nurse_rcl.v2.rule_hypotheses import RuleHypothesisEnvelope
from nurse_rcl.v2.live.evidence_routing import (
    RoutedB0Binding, RoutedRuleSet, _pin, require_routed_b0_binding,
)
from nurse_rcl.v2.live import evidence_routing_indexed as indexed


def execute_live_b0_guards(
    *,
    envelope: RuleHypothesisEnvelope,
    binding: B0RuleBinding,
    expected_rule_set_sha256: str,
    expected_binding_sha256: str,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[HypothesisGuardResult, ...]:
    """Guard the complete pinned B0 set without accepting semantic/query state."""
    with binding_capacity(LIVE_B0_CAPACITY_PROFILE):
        _preflight_inputs(envelope, expected_rule_set_sha256,
                          binding, expected_binding_sha256)
        original_source = _snapshot(envelope, RuleHypothesisEnvelope)
        original_binding = _snapshot(binding, B0RuleBinding)
        try:
            fresh = RuleHypothesisEnvelope.model_validate_json(original_source[0])
            fresh_before = _snapshot(fresh, RuleHypothesisEnvelope)
            try:
                # This performs complete independent B0 source and binding replay;
                # neither the input hash nor the incoming receipt self-authorizes.
                working = require_b0_rule_binding(
                    binding=binding, expected_binding_sha256=expected_binding_sha256,
                    envelope=fresh, expected_rule_set_sha256=expected_rule_set_sha256,
                    capacity_profile=LIVE_B0_CAPACITY_PROFILE,
                )
                working_before = _snapshot(working, B0RuleBinding)
                try:
                    universe, frozen, registry, grouping, library, scope = guard._validated_inputs(
                        candidate_universe=fresh.candidate_universe,
                        frozen_candidates=fresh.frozen_candidates,
                        entity_registry=fresh.entity_registry,
                        event_grouping=fresh.event_grouping,
                        mechanism_library=fresh.mechanism_library,
                        scope_interval=fresh.scope_interval,
                    )
                    predicates = {p.predicate_result_id: p for p in working.predicates}
                    results = []
                    for hypothesis in working.hypotheses:
                        ids = (*hypothesis.necessary_predicate_result_ids,
                               *hypothesis.optional_predicate_result_ids,
                               *hypothesis.refuting_predicate_result_ids)
                        result = guard._execute_hypothesis_guard_values(
                            submitted_record=hypothesis.model_dump(mode='json'),
                            predicate_values=guard._safe_predicate_sequence(
                                tuple(predicates[key] for key in sorted(set(ids)))),
                            universe=universe, frozen=frozen, registry=registry,
                            grouping=grouping, library=library, scope=scope,
                        )
                        results.append(result)
                        if progress is not None:
                            progress(dict(completed=len(results), total=len(working.hypotheses),
                                          disposition=result.final_disposition,
                                          reason_codes=[d.reason_code for d in result.decisions]))
                    return tuple(results)
                finally:
                    _require_unchanged(working, working_before, B0RuleBinding)
            finally:
                _require_unchanged(fresh, fresh_before, RuleHypothesisEnvelope)
        finally:
            try:
                _require_unchanged(envelope, original_source, RuleHypothesisEnvelope)
            finally:
                _require_unchanged(binding, original_binding, B0RuleBinding)


def execute_live_routed_b0_guards(
    *, envelope: RuleHypothesisEnvelope, routing: RoutedRuleSet, binding: RoutedB0Binding,
    expected_rule_set_sha256: str, expected_routing_sha256: str, expected_binding_sha256: str,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[HypothesisGuardResult, ...]:
    """New routed source replay; the same four strict guard layers still execute."""
    with binding_capacity(LIVE_B0_CAPACITY_PROFILE):
        if (type(envelope) is not RuleHypothesisEnvelope or type(routing) is not RoutedRuleSet
                or type(binding) is not RoutedB0Binding):
            raise ValueError('routed live guards require exact registered sources')
        for pin in (expected_rule_set_sha256, expected_routing_sha256, expected_binding_sha256):
            _pin(pin)
        original_source = _snapshot(envelope, RuleHypothesisEnvelope)
        original_routing = _snapshot(routing, RoutedRuleSet)
        original_binding = _snapshot(binding, RoutedB0Binding)
        with ExitStack() as guards:
            guards.callback(_require_unchanged, envelope, original_source, RuleHypothesisEnvelope)
            guards.callback(_require_unchanged, routing, original_routing, RoutedRuleSet)
            guards.callback(_require_unchanged, binding, original_binding, RoutedB0Binding)
            fresh = RuleHypothesisEnvelope.model_validate_json(original_source[0])
            guards.callback(_require_unchanged, fresh, _snapshot(fresh, RuleHypothesisEnvelope), RuleHypothesisEnvelope)
            working = require_routed_b0_binding(
                envelope=fresh, routing=routing, binding=binding,
                expected_rule_set_sha256=expected_rule_set_sha256,
                expected_routing_sha256=expected_routing_sha256,
                expected_binding_sha256=expected_binding_sha256,
                capacity_profile=LIVE_B0_CAPACITY_PROFILE)
            guards.callback(_require_unchanged, working, _snapshot(working, RoutedB0Binding), RoutedB0Binding)
            universe, frozen, registry, grouping, library, scope = guard._validated_inputs(
                candidate_universe=fresh.candidate_universe, frozen_candidates=fresh.frozen_candidates,
                entity_registry=fresh.entity_registry, event_grouping=fresh.event_grouping,
                mechanism_library=fresh.mechanism_library, scope_interval=fresh.scope_interval)
            for value in (universe, frozen, registry, grouping, library, scope):
                guards.callback(_require_unchanged, value, _snapshot(value, type(value)), type(value))
            predicates = {p.predicate_result_id: p for p in working.predicates}
            results = []
            for hypothesis in working.hypotheses:
                ids = (*hypothesis.necessary_predicate_result_ids, *hypothesis.optional_predicate_result_ids,
                       *hypothesis.refuting_predicate_result_ids)
                result = guard._execute_hypothesis_guard_values(
                    submitted_record=hypothesis.model_dump(mode='json'),
                    predicate_values=guard._safe_predicate_sequence(tuple(predicates[k] for k in sorted(set(ids)))),
                    universe=universe, frozen=frozen, registry=registry, grouping=grouping, library=library, scope=scope)
                results.append(result)
                if progress is not None:
                    progress(dict(completed=len(results), total=len(working.hypotheses),
                        disposition=result.final_disposition,
                        reason_codes=[d.reason_code for d in result.decisions]))
        return tuple(results)


def execute_live_indexed_routed_b0_guards(
    *, envelope: RuleHypothesisEnvelope, routing: indexed.IndexedRoutedRuleSet,
    binding: indexed.IndexedRoutedB0Binding, expected_rule_set_sha256: str,
    expected_routing_sha256: str, expected_binding_sha256: str,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[HypothesisGuardResult, ...]:
    """Indexed /2 audit, with the /1 replay and all exit boundaries retained."""
    with binding_capacity(LIVE_B0_CAPACITY_PROFILE):
        if (type(envelope) is not RuleHypothesisEnvelope
                or type(routing) is not indexed.IndexedRoutedRuleSet
                or type(binding) is not indexed.IndexedRoutedB0Binding):
            raise ValueError('indexed routed live guards require exact registered sources')
        for pin in (expected_rule_set_sha256, expected_routing_sha256, expected_binding_sha256):
            _pin(pin)
        original_source = _snapshot(envelope, RuleHypothesisEnvelope)
        original_routing = _snapshot(routing, indexed.IndexedRoutedRuleSet)
        original_binding = _snapshot(binding, indexed.IndexedRoutedB0Binding)
        with ExitStack() as guards:
            guards.callback(_require_unchanged, envelope, original_source, RuleHypothesisEnvelope)
            guards.callback(_require_unchanged, routing, original_routing, indexed.IndexedRoutedRuleSet)
            guards.callback(_require_unchanged, binding, original_binding, indexed.IndexedRoutedB0Binding)
            fresh = RuleHypothesisEnvelope.model_validate_json(original_source[0])
            guards.callback(_require_unchanged, fresh, _snapshot(fresh, RuleHypothesisEnvelope), RuleHypothesisEnvelope)
            working = indexed.require_routed_b0_binding(
                envelope=fresh, routing=routing, binding=binding,
                expected_rule_set_sha256=expected_rule_set_sha256,
                expected_routing_sha256=expected_routing_sha256,
                expected_binding_sha256=expected_binding_sha256,
                capacity_profile=LIVE_B0_CAPACITY_PROFILE)
            guards.callback(_require_unchanged, working, _snapshot(working, indexed.IndexedRoutedB0Binding), indexed.IndexedRoutedB0Binding)
            universe, frozen, registry, grouping, library, scope = guard._validated_inputs(
                candidate_universe=fresh.candidate_universe, frozen_candidates=fresh.frozen_candidates,
                entity_registry=fresh.entity_registry, event_grouping=fresh.event_grouping,
                mechanism_library=fresh.mechanism_library, scope_interval=fresh.scope_interval)
            for value in (universe, frozen, registry, grouping, library, scope):
                guards.callback(_require_unchanged, value, _snapshot(value, type(value)), type(value))
            predicates = {p.predicate_result_id: p for p in working.predicates}
            results = []
            for hypothesis in working.hypotheses:
                ids = (*hypothesis.necessary_predicate_result_ids, *hypothesis.optional_predicate_result_ids,
                       *hypothesis.refuting_predicate_result_ids)
                result = guard._execute_hypothesis_guard_values(
                    submitted_record=hypothesis.model_dump(mode='json'),
                    predicate_values=guard._safe_predicate_sequence(tuple(predicates[k] for k in sorted(set(ids)))),
                    universe=universe, frozen=frozen, registry=registry, grouping=grouping, library=library, scope=scope)
                results.append(result)
                if progress is not None:
                    progress(dict(completed=len(results), total=len(working.hypotheses),
                        disposition=result.final_disposition,
                        reason_codes=[d.reason_code for d in result.decisions]))
        return tuple(results)
