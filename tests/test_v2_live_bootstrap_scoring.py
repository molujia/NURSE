"""Bootstrap graph, bounded propagation and full-C ranking over live evidence."""
import copy
import importlib

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2 import graph_snapshots, propagation_scoring as propagation, reference_ranking as ranking
from nurse_rcl.v2 import case_scoring_selection as legacy
from nurse_rcl.v2 import no_role_compilation, no_role_propagation
from nurse_rcl.v2.live import scoring_selection as selection
from tests.test_v2_live_assembly import assemble
from tests.test_v2_live_graph import graph, TRACE, PARENT, CHILD
from tests.test_v2_live_rules import build
from tests.test_v2_live_sources import T0, case, load, write
from tests.test_v2_live_scoring import old_evidence


def module():
    try:
        return importlib.import_module('nurse_rcl.v2.live.bootstrap_scoring')
    except ModuleNotFoundError:
        pytest.fail('missing source-bound live bootstrap ranker', pytrace=False)


@pytest.fixture(scope='module', params=('accepted', 'no_role'))
def bootstrap_case(request, tmp_path_factory):
    root = case(tmp_path_factory.mktemp('bootstrap') / 'case')
    metric = 'k8s.deployment.available' if request.param == 'accepted' else 'undefined_metric'
    for window, start, values in [('normal', T0 - 480, [10.] * 16), ('abnormal', T0, [0.] * 16)]:
        write(root, f'{window}_metrics.parquet', [start + i * 30 for i in range(16)],
              service_name=['svc'] * 16, metric=[metric] * 16, value=values)
        write(root, f'{window}_traces.parquet', [start + 1, start + 2],
              service_name=['client', 'svc'], trace_id=[TRACE] * 2,
              span_id=[PARENT, CHILD], parent_span_id=[None, PARENT], duration=[10, 20])
    inputs = assemble(load(root))
    state = build(inputs, root)
    source = selection.LiveScoringSources(envelope=state.envelope, binding=state.binding,
        time_contract=state.guard_inputs.time_contract, focus_set=state.guard_inputs.focus_set,
        guard_results=state.guard_inputs.role_guard_results)
    pins = dict(expected_source_records_sha256=selection.live_scoring_source_sha256(sources=source),
        expected_rule_set_sha256=state.envelope.rule_hypothesis_set.result_sha256,
        expected_binding_sha256=state.binding.binding_sha256)
    receipt = selection.build_live_scoring_selection(sources=source, **pins)
    return dict(sources=source, selection=receipt, **pins,
        expected_selection_sha256=receipt.selection_sha256), graph(inputs, root)


@pytest.fixture
def args(bootstrap_case):
    return copy.deepcopy(bootstrap_case)


def prepare(args):
    live = module()
    result, graph_state = args
    context = live.build_live_bootstrap_context(graph_input_manifest=graph_state.manifest,
        expected_graph_input_manifest_sha256=graph_state.manifest.manifest_sha256)
    result.update(context=context,
        expected_context_sha256=sha256_hex(live.live_bootstrap_context_bytes(context)))
    return result, graph_state


def old_propagation(args, result, roles):
    inputs = selection._guard_inputs(args['sources'])
    if args['selection'].eligible_guard_result_ids:
        return propagation.score_propagation_paths(
            frozen_candidates=inputs.frozen_candidates, condensations=result.condensations,
            event_grouping=inputs.event_grouping, onsets=(), precedences=(),
            role_compilation=roles, role_proposals=roles.proposals,
            role_guard_results=tuple(g for g in inputs.role_guard_results if g.final_disposition == 'accepted'),
            role_predicates=inputs.role_predicates, candidate_universe=inputs.candidate_universe,
            entity_registry=inputs.entity_registry, mechanism_library=inputs.mechanism_library,
            scope_interval=inputs.scope_interval, config=args['context'].propagation_config)
    source_pin = legacy.case_guard_replay_inputs_sha256(inputs=inputs)
    receipt = legacy.build_case_scoring_selection(inputs=inputs, expected_source_records_sha256=source_pin)
    roles = no_role_compilation.build_no_role_compilation(inputs=inputs, selection=receipt,
        expected_source_records_sha256=source_pin, expected_selection_sha256=receipt.selection_sha256)
    kw = dict(inputs=inputs, selection=receipt, expected_source_records_sha256=source_pin,
        expected_selection_sha256=receipt.selection_sha256, role_compilation=roles,
        expected_role_compilation_sha256=roles.result_sha256,
        graph_snapshots=(result.graph_pair.normal_snapshot, result.graph_pair.fault_snapshot),
        condensations=result.condensations, onsets=(), precedences=(),
        config=args['context'].propagation_config)
    return no_role_propagation.score_no_role_propagation(**kw,
        expected_authoritative_input_sha256=no_role_propagation.no_role_propagation_input_sha256(**kw))


def test_full_graph_propagation_scores_ties_and_top_k_match_original_cores(args):
    kw, graph_state = prepare(args)
    live = module()
    result = live.build_live_bootstrap_scoring(**kw)
    assert result.schema_version == 'nurse.v2.live-bootstrap-scoring/1'
    assert result.graph_pair == graph_state.snapshots
    assert len(result.graph_pair.normal_snapshot.edges) == 1
    assert result.onsets == () and result.precedences == ()
    assert result.temporal_status == 'unknown_no_verified_onset'
    old_roles, old_events, old_family = old_evidence(kw)
    prop = old_propagation(kw, result, old_roles)
    assert result.propagation_scores == prop.candidate_scores
    assert result.propagation_path_audits == prop.path_audits
    assert result.propagation_target_scores == prop.target_scores
    config = kw['context'].scoring_config
    expected_scores = tuple(ranking._build_candidate_decomposition(
        candidate_service_key=f.candidate_service_key, event_evidence_score=f.evidence_strength,
        propagation_score=p.propagation_score, rule_score=r.rule_score, history_score=h.history_score,
        explicit_rules_enabled=False, config=config,
        event_group_ids=tuple(key for maximum in f.family_maxima for key in maximum.retained_event_group_ids),
        propagation_path_ids=tuple(path.path_audit_id for path in prop.path_audits
                                  if path.candidate_service_key == f.candidate_service_key),
        rule_ids=tuple(x.contribution_id for x in r.contributions),
        history_neighbor_ids=tuple(x.selected_neighbor_id for x in h.contributions))
        for f, p, r, h in zip(old_family.candidates, prop.candidate_scores,
            result.rule_scoring.candidates, result.history_evaluation.candidate_scores, strict=True))
    assert result.scores == expected_scores
    assert result.evidence_tie_groups == ranking._build_evidence_tie_groups(expected_scores)
    assert result.ordered_service_keys == ranking._ordered_service_keys(
        scores=expected_scores, tie_break_seed_sha256=config.tie_break_seed_sha256)
    assert tuple(row.service_entity_key for row in result.top_k.entries) == result.ordered_service_keys[:config.top_k]
    assert set(result.ordered_service_keys) == set(kw['sources'].envelope.frozen_candidates.candidate_service_keys)


def test_every_call_replays_common_source_once_and_checks_context(args, monkeypatch):
    from nurse_rcl.v2 import hypothesis_guard
    kw, _ = prepare(args)
    seen = []
    actual = hypothesis_guard._validated_inputs
    def observe(**values):
        seen.append(values['event_grouping'])
        return actual(**values)
    monkeypatch.setattr(hypothesis_guard, '_validated_inputs', observe)
    first = module().build_live_bootstrap_scoring(**kw)
    second = module().build_live_bootstrap_scoring(**kw)
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert first == second
    kw['expected_context_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='pin'):
        module().build_live_bootstrap_scoring(**kw)


@pytest.mark.parametrize('kind', ('content', 'marker', 'slot'))
@pytest.mark.parametrize('fail', (False, True))
@pytest.mark.parametrize('owner', ('original_context', 'fresh_context', 'evidence'))
def test_context_and_rank_intermediates_preserved_on_both_exits(args, kind, fail, owner, monkeypatch):
    kw, _ = prepare(args)
    live = module()
    original = live._rank
    def mutate(**values):
        result = original(**values)
        victim = {'original_context': kw['context'], 'fresh_context': values['context'],
                  'evidence': values['evidence']}[owner]
        if kind == 'content':
            object.__setattr__(victim, 'schema_version', 'changed')
        elif kind == 'marker':
            victim.__pydantic_fields_set__.clear()
        else:
            object.__setattr__(victim, '__pydantic_private__', {})
        if fail:
            raise RuntimeError('controlled rank failure')
        return result
    monkeypatch.setattr(live, '_rank', mutate)
    with pytest.raises(ValueError, match='changed'):
        live.build_live_bootstrap_scoring(**kw)


@pytest.mark.parametrize('field', ('dynamic_verification', 'pre_ranking_semantics', 'history_mode'))
def test_bootstrap_rejects_dynamic_or_history_effects(args, field):
    kw, _ = prepare(args)
    live = module()
    changed = kw['context'].model_copy(update={field: 'score' if field == 'history_mode' else True})
    kw['context'] = changed
    with pytest.raises(ValueError):
        live.build_live_bootstrap_scoring(**kw)


def test_wrong_graph_pin_cannot_create_context(bootstrap_case):
    live = module()
    _, state = bootstrap_case
    with pytest.raises(ValueError, match='pin|digest'):
        live.build_live_bootstrap_context(graph_input_manifest=state.manifest,
            expected_graph_input_manifest_sha256='f' * 64)
