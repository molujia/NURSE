"""Current live evidence consumes one source replay, never a reusable trust token."""
import copy
import importlib

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2 import event_strength, family_aggregation, hypothesis_guard, role_compiler
from nurse_rcl.v2 import case_scoring_selection as legacy
from nurse_rcl.v2 import no_role_compilation, no_role_evidence
from nurse_rcl.v2.live import scoring_selection as selection
from tests.test_v2_live_assembly import assemble
from tests.test_v2_live_rules import build
from tests.test_v2_live_sources import T0, case, load, write


def module():
    try:
        return importlib.import_module('nurse_rcl.v2.live.scoring')
    except ModuleNotFoundError:
        pytest.fail('missing invocation-local live scoring boundary', pytrace=False)


@pytest.fixture(scope='module', params=('accepted', 'no_role'))
def frozen_case(request, tmp_path_factory):
    root = case(tmp_path_factory.mktemp('scoring') / 'case')
    metric = 'k8s.deployment.available' if request.param == 'accepted' else 'undefined_metric'
    for window, start, values in [('normal', T0 - 480, [10.] * 16), ('abnormal', T0, [0.] * 16)]:
        write(root, f'{window}_metrics.parquet', [start + i * 30 for i in range(16)],
              service_name=['svc'] * 16, metric=[metric] * 16, value=values)
    state = build(assemble(load(root)), root)
    source = selection.LiveScoringSources(envelope=state.envelope, binding=state.binding,
        time_contract=state.guard_inputs.time_contract, focus_set=state.guard_inputs.focus_set,
        guard_results=state.guard_inputs.role_guard_results)
    pins = dict(expected_source_records_sha256=selection.live_scoring_source_sha256(sources=source),
        expected_rule_set_sha256=state.envelope.rule_hypothesis_set.result_sha256,
        expected_binding_sha256=state.binding.binding_sha256)
    receipt = selection.build_live_scoring_selection(sources=source, **pins)
    return dict(sources=source, selection=receipt, **pins,
                expected_selection_sha256=receipt.selection_sha256)


@pytest.fixture
def args(frozen_case):
    return copy.deepcopy(frozen_case)


def old_evidence(args):
    inputs = selection._guard_inputs(args['sources'])
    pin = legacy.case_guard_replay_inputs_sha256(inputs=inputs)
    receipt = legacy.build_case_scoring_selection(inputs=inputs, expected_source_records_sha256=pin)
    replay = legacy.require_case_scoring_selection(inputs=inputs, selection=receipt,
        expected_source_records_sha256=pin, expected_selection_sha256=receipt.selection_sha256)
    if receipt.eligible_guard_result_ids:
        proposals = role_compiler.build_default_role_binding_proposals(
            guard_results=replay.eligible_guards, event_grouping=inputs.event_grouping)
        common = dict(proposals=proposals, guard_results=replay.eligible_guards,
            predicates=inputs.role_predicates, candidate_universe=inputs.candidate_universe,
            frozen_candidates=inputs.frozen_candidates, entity_registry=inputs.entity_registry,
            event_grouping=inputs.event_grouping, mechanism_library=inputs.mechanism_library,
            scope_interval=inputs.scope_interval)
        roles = role_compiler.compile_event_roles(**common)
        event = event_strength.compile_event_strengths(role_compilation=roles, **common)
        family = family_aggregation.compile_family_aggregation(role_compilation=roles,
            event_strength=event, aggregator_version=family_aggregation.REFERENCE_KEY, **common)
    else:
        common = dict(inputs=inputs, selection=receipt, expected_source_records_sha256=pin,
                      expected_selection_sha256=receipt.selection_sha256)
        roles = no_role_compilation.build_no_role_compilation(**common)
        common.update(role_compilation=roles, expected_role_compilation_sha256=roles.result_sha256)
        event = no_role_evidence.compile_no_role_event_strengths(**common)
        family = no_role_evidence.compile_no_role_family_aggregation(**common,
            event_strength=event, expected_event_strength_sha256=event.result_sha256,
            aggregator_version=family_aggregation.REFERENCE_KEY)
    return roles, event, family


def test_all_numeric_evidence_matches_legacy_and_full_candidates(args):
    live = module()
    result = live.build_live_scoring_evidence(**args)
    roles, event, family = old_evidence(args)
    assert result.schema_version == 'nurse.v2.live-scoring-evidence/1'
    assert result.scoring_path == ('accepted_roles' if args['selection'].eligible_guard_result_ids
                                   else 'no_eligible_roles')
    assert result.role_proposals == roles.proposals
    assert result.role_assertions == roles.assertions
    assert result.group_strengths == event.group_strengths
    assert result.candidate_event_adjustments == event.candidate_event_adjustments
    assert result.retained_assertion_ids == event.retained_assertion_ids
    assert result.unbound_assertion_ids == event.unbound_assertion_ids
    assert result.family_candidates == family.candidates
    assert result.unbound_event_group_audits == family.unbound_event_group_audits
    assert result.retained_event_group_ids == family.retained_event_group_ids
    assert result.aggregated_event_group_ids == family.aggregated_event_group_ids
    assert result.unbound_event_group_ids == family.unbound_event_group_ids
    assert result.provenance_ids == family.provenance_ids
    assert tuple(c.candidate_service_key for c in result.family_candidates) == (
        args['sources'].envelope.frozen_candidates.candidate_service_keys)
    assert result.selection == args['selection']
    assert sha256_hex(canonical_json_bytes(result.model_dump(mode='json', exclude={'result_sha256'}))) == result.result_sha256


def test_replay_each_call_and_only_once_per_call(args, monkeypatch):
    live = module()
    seen = []
    actual = hypothesis_guard._validated_inputs
    def observe(**kw):
        seen.append(kw['event_grouping'])
        return actual(**kw)
    monkeypatch.setattr(hypothesis_guard, '_validated_inputs', observe)
    first = live.build_live_scoring_evidence(**args)
    assert len(seen) == 1
    second = live.build_live_scoring_evidence(**args)
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert first == second


@pytest.mark.parametrize('pin', ('expected_source_records_sha256', 'expected_rule_set_sha256',
                               'expected_binding_sha256', 'expected_selection_sha256'))
def test_independent_pins_cannot_be_replaced(args, pin):
    live = module()
    args[pin] = 'f' * 64
    with pytest.raises(ValueError):
        live.build_live_scoring_evidence(**args)


@pytest.mark.parametrize('kind', ('content', 'container', 'marker', 'slot'))
@pytest.mark.parametrize('fail', (False, True))
@pytest.mark.parametrize('owner', ('original', 'fresh'))
def test_original_and_working_state_checked_after_consumer(args, kind, fail, owner, monkeypatch):
    live = module()
    original = live._compile_evidence
    def mutate(fresh, replay):
        value = original(fresh, replay)
        for source in (args['sources'] if owner == 'original' else fresh,):
            if kind == 'content':
                object.__setattr__(source.time_contract, 'timezone', 'changed')
            elif kind == 'container':
                object.__setattr__(source, 'guard_results', list(source.guard_results))
            elif kind == 'marker':
                source.time_contract.__pydantic_fields_set__.clear()
            else:
                object.__setattr__(source.time_contract, '__pydantic_private__', {})
        if fail:
            raise RuntimeError('controlled error')
        return value
    monkeypatch.setattr(live, '_compile_evidence', mutate)
    with pytest.raises(ValueError, match='changed'):
        live.build_live_scoring_evidence(**args)


@pytest.mark.parametrize('kind', ('content', 'container', 'marker', 'slot'))
@pytest.mark.parametrize('fail', (False, True))
def test_intermediate_mutation_rejects_even_on_downstream_error(args, kind, fail, monkeypatch):
    live = module()
    original = live._compile_families
    def mutate(**kw):
        victim = kw['candidate_event_adjustments'][0]
        if kind == 'content':
            object.__setattr__(victim, 'role_adjustment', 0.99)
        elif kind == 'container':
            object.__setattr__(victim, 'retained_assertion_ids', list(victim.retained_assertion_ids))
        elif kind == 'marker':
            victim.__pydantic_fields_set__.clear()
        else:
            object.__setattr__(victim, '__pydantic_private__', {})
        if fail:
            raise RuntimeError('controlled downstream error')
        return original(**kw)
    monkeypatch.setattr(live, '_compile_families', mutate)
    with pytest.raises(ValueError, match='changed'):
        live.build_live_scoring_evidence(**args)


def test_unchanged_downstream_exception_is_preserved(args, monkeypatch):
    live = module()
    def fail(**kw):
        raise RuntimeError('unchanged downstream failure')
    monkeypatch.setattr(live, '_compile_families', fail)
    with pytest.raises(RuntimeError, match='unchanged downstream failure'):
        live.build_live_scoring_evidence(**args)


def test_rehashed_selection_still_rejected(args):
    live = module()
    values = args['selection'].model_dump(mode='json', exclude={'selection_sha256'})
    values['eligible_guard_result_ids'], values['excluded_guard_result_ids'] = (
        values['excluded_guard_result_ids'], values['eligible_guard_result_ids'])
    args['selection'] = selection.LiveScoringSelection.model_validate({**values,
        'selection_sha256': sha256_hex(canonical_json_bytes(values))})
    args['expected_selection_sha256'] = args['selection'].selection_sha256
    with pytest.raises(ValueError):
        live.build_live_scoring_evidence(**args)


def test_build_requires_exact_live_receipt(args):
    live = module()
    args['selection'] = None
    with pytest.raises(ValueError):
        live.build_live_scoring_evidence(**args)


@pytest.mark.parametrize('field', ('retained_event_group_ids', 'aggregated_event_group_ids',
                                  'provenance_ids'))
def test_new_parent_retains_family_partition_checks(args, field, monkeypatch):
    live = module()
    original = live._compile_families
    def forge(**kw):
        parts = original(**kw)
        parts[field] = () if parts[field] else ('forged',)
        return parts
    monkeypatch.setattr(live, '_compile_families', forge)
    with pytest.raises(ValueError):
        live.build_live_scoring_evidence(**args)
