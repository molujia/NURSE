"""Original-observation B0 rules over independently replayed live inputs."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from nurse_rcl.v2.b0_rule_binding import B0RuleBinding, build_b0_rule_binding
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.case_scoring_selection import CaseGuardReplayInputs
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.live.assembly import LiveInputs
from nurse_rcl.v2.live.b0_guards import execute_live_b0_guards
from nurse_rcl.v2.live.b0_guards import execute_live_routed_b0_guards
from nurse_rcl.v2.live.evidence_routing import (
    RoutedB0Binding, RoutedRuleSet, build_routed_b0_binding, build_routed_rule_set,
)
from nurse_rcl.v2.live import evidence_routing_indexed as indexed
from nurse_rcl.v2.live.b0_guards import execute_live_indexed_routed_b0_guards
from nurse_rcl.v2.live_capacity import LIVE_B0_CAPACITY_PROFILE, LIVE_B0_CAPACITY_SHA256
from nurse_rcl.v2.mechanism_library import MechanismLibrary, build_mechanism_library
from nurse_rcl.v2.observation_schemas import CandidateUniverse, TimeInterval
from nurse_rcl.v2.rule_hypotheses import RuleHypothesisEnvelope, build_rule_hypothesis_set
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry


class _RuleSources(TypedDict):
    candidate_universe: CandidateUniverse
    frozen_candidates: FrozenCandidateUniverse
    entity_registry: TemporalEntityRegistry
    event_grouping: EventGroupingResult
    scope_interval: TimeInterval
    mechanism_library: MechanismLibrary


@dataclass(frozen=True)
class LiveRuleState:
    envelope: RuleHypothesisEnvelope
    binding: B0RuleBinding
    guard_inputs: CaseGuardReplayInputs
    capacity_profile: str = LIVE_B0_CAPACITY_PROFILE
    capacity_contract_sha256: str = LIVE_B0_CAPACITY_SHA256


def build_live_rule_state(inputs: LiveInputs, *, source_root: Path,
                          progress: Callable[[dict[str, Any]], None] | None = None) -> LiveRuleState:
    """Do not give source authority to a self-consistent input digest.

    No dynamic query, model, manual predicate, or label can participate here.
    All C × initial mechanism pairs are generated and guarded, including
    rejected/unknown hypotheses; Q is not a filter on this path.
    """
    started = time.monotonic()

    def emit(stage: str, **values: Any) -> None:
        if progress is not None:
            progress(dict(stage=stage, elapsed_seconds=time.monotonic() - started, **values))

    emit('input_replay_started')
    if type(inputs) is not LiveInputs or not inputs.verify_sources(source_root):
        raise ValueError('live input source replay failed')
    emit('input_replay_completed')
    sources: _RuleSources = dict(candidate_universe=inputs.candidate_universe,
                   frozen_candidates=inputs.frozen_candidates,
                   entity_registry=inputs.entity_registry, event_grouping=inputs.event_grouping,
                   scope_interval=inputs.scope, mechanism_library=build_mechanism_library())
    emit('rule_proposal_started')
    envelope = RuleHypothesisEnvelope(**sources, rule_hypothesis_set=build_rule_hypothesis_set(**sources))
    emit('rule_proposal_completed', hypotheses=len(envelope.rule_hypothesis_set.hypotheses))
    binding = build_b0_rule_binding(envelope=envelope,
        expected_rule_set_sha256=envelope.rule_hypothesis_set.result_sha256,
        capacity_profile=LIVE_B0_CAPACITY_PROFILE)
    emit('rule_binding_completed', predicates=len(binding.predicates))
    emit('rule_guard_batch_started')
    guards = execute_live_b0_guards(envelope=envelope, binding=binding,
        expected_rule_set_sha256=envelope.rule_hypothesis_set.result_sha256,
        expected_binding_sha256=binding.binding_sha256,
        progress=lambda row: emit('rule_guard', **row))
    guard_inputs = CaseGuardReplayInputs(**sources, time_contract=inputs.sources.adapter.time_contract,
        focus_set=inputs.focus.focus_set, semantic_context=None, card_expansion_policy=None,
        query_sessions=(), role_predicates=binding.predicates,
        role_guard_results=tuple(sorted(guards, key=lambda row: row.guard_result_id)))
    emit('rule_guards_completed', completed=len(guards), total=len(binding.hypotheses))
    return LiveRuleState(envelope, binding, guard_inputs)


@dataclass(frozen=True)
class LiveRoutedRuleState:
    envelope: RuleHypothesisEnvelope
    routing: RoutedRuleSet
    binding: RoutedB0Binding
    guard_inputs: CaseGuardReplayInputs
    capacity_profile: str = LIVE_B0_CAPACITY_PROFILE
    capacity_contract_sha256: str = LIVE_B0_CAPACITY_SHA256


def build_live_routed_rule_state(inputs: LiveInputs, *, source_root: Path,
        progress: Callable[[dict[str, Any]], None] | None = None) -> LiveRoutedRuleState:
    """Routing-only identity; no observation, direction, weight or query changes."""
    started = time.monotonic()

    def emit(stage: str, **values: Any) -> None:
        if progress is not None:
            progress(dict(stage=stage, elapsed_seconds=time.monotonic() - started, **values))

    emit('input_replay_started')
    if type(inputs) is not LiveInputs or not inputs.verify_sources(source_root):
        raise ValueError('live input source replay failed')
    emit('input_replay_completed')
    sources: _RuleSources = dict(candidate_universe=inputs.candidate_universe,
        frozen_candidates=inputs.frozen_candidates, entity_registry=inputs.entity_registry,
        event_grouping=inputs.event_grouping, scope_interval=inputs.scope,
        mechanism_library=build_mechanism_library())
    emit('rule_proposal_started')
    envelope = RuleHypothesisEnvelope(**sources, rule_hypothesis_set=build_rule_hypothesis_set(**sources))
    pin = envelope.rule_hypothesis_set.result_sha256
    emit('rule_proposal_completed', hypotheses=len(envelope.rule_hypothesis_set.hypotheses))
    routing = build_routed_rule_set(envelope=envelope, expected_rule_set_sha256=pin,
        capacity_profile=LIVE_B0_CAPACITY_PROFILE)
    emit('rule_routing_completed', routes=len(routing.routes))
    binding = build_routed_b0_binding(envelope=envelope, routing=routing,
        expected_rule_set_sha256=pin, expected_routing_sha256=routing.result_sha256,
        capacity_profile=LIVE_B0_CAPACITY_PROFILE)
    emit('rule_binding_completed', predicates=len(binding.predicates))
    emit('rule_guard_batch_started')
    guards = execute_live_routed_b0_guards(envelope=envelope, routing=routing, binding=binding,
        expected_rule_set_sha256=pin, expected_routing_sha256=routing.result_sha256,
        expected_binding_sha256=binding.binding_sha256, progress=lambda row: emit('rule_guard', **row))
    guard_inputs = CaseGuardReplayInputs(**sources, time_contract=inputs.sources.adapter.time_contract,
        focus_set=inputs.focus.focus_set, semantic_context=None, card_expansion_policy=None,
        query_sessions=(), role_predicates=binding.predicates,
        role_guard_results=tuple(sorted(guards, key=lambda row: row.guard_result_id)))
    emit('rule_guards_completed', completed=len(guards), total=len(binding.hypotheses))
    return LiveRoutedRuleState(envelope, routing, binding, guard_inputs)


@dataclass(frozen=True)
class LiveIndexedRoutedRuleState:
    envelope: RuleHypothesisEnvelope
    routing: indexed.IndexedRoutedRuleSet
    binding: indexed.IndexedRoutedB0Binding
    guard_inputs: CaseGuardReplayInputs
    capacity_profile: str = LIVE_B0_CAPACITY_PROFILE
    capacity_contract_sha256: str = LIVE_B0_CAPACITY_SHA256


def build_live_indexed_routed_rule_state(inputs: LiveInputs, *, source_root: Path,
        progress: Callable[[dict[str, Any]], None] | None = None) -> LiveIndexedRoutedRuleState:
    """Explicit /2 representation; original observations and guard logic unchanged."""
    started = time.monotonic()

    def emit(stage: str, **values: Any) -> None:
        if progress is not None:
            progress(dict(stage=stage, elapsed_seconds=time.monotonic() - started, **values))

    emit('input_replay_started')
    if type(inputs) is not LiveInputs or not inputs.verify_sources(source_root):
        raise ValueError('live input source replay failed')
    emit('input_replay_completed')
    sources: _RuleSources = dict(candidate_universe=inputs.candidate_universe,
        frozen_candidates=inputs.frozen_candidates, entity_registry=inputs.entity_registry,
        event_grouping=inputs.event_grouping, scope_interval=inputs.scope,
        mechanism_library=build_mechanism_library())
    emit('rule_proposal_started')
    envelope = RuleHypothesisEnvelope(**sources, rule_hypothesis_set=build_rule_hypothesis_set(**sources))
    pin = envelope.rule_hypothesis_set.result_sha256
    emit('rule_proposal_completed', hypotheses=len(envelope.rule_hypothesis_set.hypotheses))
    routing = indexed.build_routed_rule_set(envelope=envelope, expected_rule_set_sha256=pin,
        capacity_profile=LIVE_B0_CAPACITY_PROFILE)
    emit('rule_routing_completed', routes=len(routing.routes),
         unique_observation_bindings=len(routing.observation_bindings))
    binding = indexed.build_routed_b0_binding(envelope=envelope, routing=routing,
        expected_rule_set_sha256=pin, expected_routing_sha256=routing.result_sha256,
        capacity_profile=LIVE_B0_CAPACITY_PROFILE)
    emit('rule_binding_completed', predicates=len(binding.predicates))
    emit('rule_guard_batch_started')
    guards = execute_live_indexed_routed_b0_guards(envelope=envelope, routing=routing, binding=binding,
        expected_rule_set_sha256=pin, expected_routing_sha256=routing.result_sha256,
        expected_binding_sha256=binding.binding_sha256, progress=lambda row: emit('rule_guard', **row))
    guard_inputs = CaseGuardReplayInputs(**sources, time_contract=inputs.sources.adapter.time_contract,
        focus_set=inputs.focus.focus_set, semantic_context=None, card_expansion_policy=None,
        query_sessions=(), role_predicates=binding.predicates,
        role_guard_results=tuple(sorted(guards, key=lambda row: row.guard_result_id)))
    emit('rule_guards_completed', completed=len(guards), total=len(binding.hypotheses))
    return LiveIndexedRoutedRuleState(envelope, routing, binding, guard_inputs)
