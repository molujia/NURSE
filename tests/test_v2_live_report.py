"""Live sources must be replayed before the shared Report Agent can send bytes."""
import copy
import importlib
import json
from dataclasses import replace

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live import bootstrap_audit, bootstrap_scoring
from nurse_rcl.v2.report_evidence_schemas import ReportEvidencePolicy
from nurse_rcl.v2.report_run_schemas import ReportRunPolicy
from tests.test_v2_inference_leakage import report_identity_manifest
from tests.test_v2_live_bootstrap_audit import arguments, pins
from tests.test_v2_live_bootstrap_scoring import bootstrap_case  # noqa: F401


def module():
    try:
        return importlib.import_module('nurse_rcl.v2.live.report')
    except ModuleNotFoundError:
        pytest.fail('missing source-replayed live Report Agent adapter', pytrace=False)


@pytest.fixture(scope='module')
def report_case(bootstrap_case):
    args = arguments(bootstrap_case)
    audit = bootstrap_audit.build_live_bootstrap_audit(**args)
    manifest = report_identity_manifest(audit.result.opaque_case_id)
    policy = ReportEvidencePolicy(configuration_version='live-test/1', max_view_bytes=90000)
    run_policy = ReportRunPolicy(configuration_version='live-test/1',
        input_token_limit=100000, output_token_limit=4096)
    return dict(audit=audit, **pins(args),
        expected_audit_sha256=sha256_hex(bootstrap_audit.live_bootstrap_audit_bytes(audit)),
        identity_manifest=manifest, expected_identity_manifest_sha256=manifest.manifest_sha256,
        evidence_policy=policy, expected_evidence_policy_sha256=policy.sha256,
        run_policy=run_policy, expected_run_policy_sha256=run_policy.sha256)


def response_for(payload, *, unsafe=False):
    data = json.loads(json.loads(payload)['messages'][1]['content'])
    view = data['evidence_view']
    if view['schema_version'] == 'nurse.v2.live-report-wire/1':
        from nurse_rcl.v2.live.report_compact import decode_compact_view
        view = decode_compact_view(canonical_json_bytes(view),
            expected_view_sha256=view['view_sha256']).model_dump(mode='json')
    fact = next(f for f in view['facts'] if f['kind'] == 'ranking')
    response = dict(schema_version='nurse.v2.report-composition-response/1',
        claims=[dict(
            claim_id='claim_1', operation='explain_priority', statement_type='inference',
            requested_claim_strength='ranking_inference', fact_refs=[fact['fact_ref']],
            target_service_keys=list(fact['target_service_keys']),
            surface_text=('This is the confirmed root cause.' if unsafe else
                          'The frozen ranking prioritizes this candidate; the score is not a probability.'),
            assertions=[], interval=None, recommendation=None)])
    return canonical_json_bytes({'choices': [{'index': 0, 'finish_reason': 'stop',
        'message': {'role': 'assistant', 'content': json.dumps(response)}}]})


def test_live_report_preserves_full_ranking_and_uses_shared_public_evidence(report_case):
    live, args = module(), copy.deepcopy(report_case)
    before = bootstrap_audit.live_bootstrap_audit_bytes(args['audit'])
    sent = []
    def complete(payload):
        sent.append(payload)
        assert b'submitted_record_json' not in payload and b'source_bindings' not in payload
        return response_for(payload)
    result = live.run_live_bootstrap_report(**args, complete=complete)
    assert len(sent) == 1
    assert result.schema_version == 'nurse.v2.live-bootstrap-report/2'
    assert result.profile == 'v2-live-bootstrap'
    assert result.dynamic_verification is False and result.pre_ranking_semantics is False
    assert result.source_ranking_result_sha256 == args['audit'].result.result_sha256
    assert result.projection.foundation.source_top_k_sha256 == args['audit'].result.top_k.ranking_sha256
    assert result.projection.foundation.verify()
    assert result.projection.verify()
    assert result.structural_status == 'valid'
    assert result.presentation is not None
    assert result.presentation.verify()
    assert result.status == 'full'
    assert result.interpretability_status == 'source_and_score_replayed'
    assert result.explainability_semantic_status == 'not_established'
    assert result.raw_response_hex and result.request_sha256 == sha256_hex(sent[0])
    assert result.usage_status == 'missing'
    assert result.stage_seconds['source_replay'] >= 0
    assert result.stage_seconds['report_exchange'] >= 0
    assert bootstrap_audit.live_bootstrap_audit_bytes(args['audit']) == before


@pytest.mark.parametrize('pin', ('expected_audit_sha256', 'expected_source_records_sha256',
    'expected_rule_set_sha256', 'expected_binding_sha256', 'expected_selection_sha256',
    'expected_context_sha256', 'expected_identity_manifest_sha256',
    'expected_evidence_policy_sha256', 'expected_run_policy_sha256'))
def test_all_independent_pins_reject_before_network(report_case, pin):
    live, args = module(), copy.deepcopy(report_case)
    args[pin] = 'f' * 64
    def forbidden(_):
        pytest.fail('invalid source must never reach transport')
    with pytest.raises(ValueError):
        live.run_live_bootstrap_report(**args, complete=forbidden)


def test_rehashed_nonleader_score_forgery_is_not_report_authority(report_case):
    live, args = module(), copy.deepcopy(report_case)
    result = args['audit'].result
    row = result.scores[-1].model_copy(update={'quantized_total_score': '0.123456789012'})
    row = row.model_copy(update={'score_sha256': sha256_hex(canonical_json_bytes(
        row.model_dump(mode='json', exclude={'score_sha256'})))})
    result = result.model_copy(update={'scores': (*result.scores[:-1], row)})
    result = result.model_copy(update={'result_sha256': sha256_hex(canonical_json_bytes(
        result.model_dump(mode='json', exclude={'result_sha256'})))})
    args['audit'] = replace(args['audit'], result=result)
    args['expected_audit_sha256'] = sha256_hex(bootstrap_audit.live_bootstrap_audit_bytes(args['audit']))
    with pytest.raises(ValueError, match='replay|differs'):
        live.run_live_bootstrap_report(**args, complete=lambda _: pytest.fail('forgery sent'))


@pytest.mark.parametrize('fail', (False, True))
@pytest.mark.parametrize('target', ('original', 'fresh', 'policy', 'projection'))
def test_normal_and_exception_exits_reject_content_and_representation_mutation(
        report_case, monkeypatch, fail, target):
    live, args = module(), copy.deepcopy(report_case)
    captured = {}
    original = live._compose
    def observe(**values):
        captured.update(values)
        return original(**values)
    monkeypatch.setattr(live, '_compose', observe)
    def mutate(payload):
        if target == 'original':
            value = args['audit'].result
        elif target == 'fresh':
            value = captured['ranking']
        elif target == 'policy':
            value = args['run_policy']
        else:
            value = captured['projection']
        value.__pydantic_fields_set__.clear()
        if fail:
            raise RuntimeError('controlled downstream exception')
        return response_for(payload)
    with pytest.raises(ValueError, match='changed'):
        live.run_live_bootstrap_report(**args, complete=mutate)


def test_valid_but_empty_response_is_retained_without_redraw(report_case):
    live, args = module(), copy.deepcopy(report_case)
    calls = []
    def complete(payload):
        calls.append(payload)
        return canonical_json_bytes({'choices': [{'index': 0, 'finish_reason': 'stop',
            'message': {'role': 'assistant', 'content': '{}'}}]})
    result = live.run_live_bootstrap_report(**args, complete=complete)
    assert len(calls) == 1 and result.structural_status == 'invalid'
    assert result.status == 'deterministic_fallback' and result.raw_response_hex


def test_mandatory_budget_exhaustion_never_hides_evidence_or_sends(report_case):
    live, args = module(), copy.deepcopy(report_case)
    policy = ReportEvidencePolicy(configuration_version='tiny/1', max_view_bytes=1)
    args.update(evidence_policy=policy, expected_evidence_policy_sha256=policy.sha256)
    result = live.run_live_bootstrap_report(**args, complete=lambda _: pytest.fail('overbudget sent'))
    assert result.projection.status == 'mandatory_budget_exhausted'
    assert result.status == 'deterministic_fallback' and result.request_sha256 is None


def test_unsafe_prose_cannot_overrule_frozen_ranking(report_case):
    live, args = module(), copy.deepcopy(report_case)
    result = live.run_live_bootstrap_report(**args, complete=lambda p: response_for(p, unsafe=True))
    assert result.structural_status == 'valid'
    assert result.presentation is not None
    assert all(d.surface_status == 'blocked' for d in result.presentation.claims)
    assert result.status == 'deterministic_fallback'


def test_every_report_invocation_replays_source_authority(report_case, monkeypatch):
    from nurse_rcl.v2 import hypothesis_guard
    live, args = module(), copy.deepcopy(report_case)
    seen = []
    original = hypothesis_guard._validated_inputs
    def observe(**values):
        seen.append(values['event_grouping'])
        return original(**values)
    monkeypatch.setattr(hypothesis_guard, '_validated_inputs', observe)
    first = live.run_live_bootstrap_report(**args, complete=response_for)
    second = live.run_live_bootstrap_report(**args, complete=response_for)
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert first.request_hex == second.request_hex
    assert first.presentation == second.presentation


def test_fresh_source_mutation_at_transport_is_rejected(report_case, monkeypatch):
    live, args = module(), copy.deepcopy(report_case)
    captured = []
    original = bootstrap_scoring._compile_bootstrap
    def observe(**values):
        captured.append(values['fresh'])
        return original(**values)
    monkeypatch.setattr(bootstrap_scoring, '_compile_bootstrap', observe)
    def mutate(payload):
        captured[0].envelope.__pydantic_fields_set__.clear()
        return response_for(payload)
    with pytest.raises(ValueError, match='changed'):
        live.run_live_bootstrap_report(**args, complete=mutate)


def test_live_uses_explicit_grouped_projection_and_compact_wire(report_case):
    result = module().run_live_bootstrap_report(**copy.deepcopy(report_case), complete=response_for)
    assert result.schema_version == 'nurse.v2.live-bootstrap-report/2'
    projection = result.projection
    assert projection.schema_version == 'nurse.v2.live-grouped-report-projection/1'
    assert projection.verify()
    assert projection.original_fact_count == sum(len(g.members) for g in projection.groups)
    request = json.loads(bytes.fromhex(result.request_hex))
    public = json.loads(request['messages'][1]['content'])['evidence_view']
    assert public['schema_version'] == 'nurse.v2.live-report-wire/1'
    assert len(canonical_json_bytes(public)) <= projection.policy.max_view_bytes
    assert 'groups' not in public and 'source_bindings' not in public


@pytest.mark.parametrize('fail', (False, True))
def test_group_membership_mutation_is_rejected_on_both_exits(report_case, monkeypatch, fail):
    live = module()
    captured = []
    original = live._compose
    def observe(**values):
        captured.append(values['projection'])
        return original(**values)
    monkeypatch.setattr(live, '_compose', observe)
    def mutate(payload):
        captured[0].groups[0].members[0].__pydantic_fields_set__.clear()
        if fail: raise RuntimeError('controlled downstream exception')
        return response_for(payload)
    with pytest.raises(ValueError, match='changed'):
        live.run_live_bootstrap_report(**copy.deepcopy(report_case), complete=mutate)


def disabled_case(report_case):
    args = copy.deepcopy(report_case)
    policy = ReportEvidencePolicy(configuration_version='live-bootstrap-report/3', max_view_bytes=80000)
    runtime = ReportRunPolicy(configuration_version='live-bootstrap-report/3',
                              input_token_limit=100000, output_token_limit=4096)
    args.update(evidence_policy=policy, expected_evidence_policy_sha256=policy.sha256,
                run_policy=runtime, expected_run_policy_sha256=runtime.sha256)
    return args


def test_disabled_version_source_report_preserves_ranking_and_request_identity(report_case):
    args = disabled_case(report_case)
    before = bootstrap_audit.live_bootstrap_audit_bytes(args['audit'])
    calls = []
    def complete(payload):
        calls.append(payload)
        assert json.loads(payload)['thinking'] == {'type': 'disabled'}
        return response_for(payload)
    result = module().run_live_bootstrap_report(**args, complete=complete)
    assert result.schema_version == 'nurse.v2.live-bootstrap-report/3'
    assert result.exchange_protocol == 'nurse.v2.live-report-disabled-exchange/1'
    assert len(calls) == 1 and result.request_sha256 == sha256_hex(calls[0])
    assert result.status == 'full' and result.structural_status == 'valid'
    assert result.source_ranking_result_sha256 == args['audit'].result.result_sha256
    assert bootstrap_audit.live_bootstrap_audit_bytes(args['audit']) == before
    assert result.interpretability_status == 'source_and_score_replayed'
    assert result.explainability_semantic_status == 'not_established'
    with pytest.raises(ValueError):
        module().LiveGroupedReportRun.model_validate(result.model_dump())


@pytest.mark.parametrize('fail', (False, True))
def test_disabled_source_and_group_mutation_still_fail_both_exits(report_case, monkeypatch, fail):
    args = disabled_case(report_case)
    live = module()
    captured = []
    original = live._compose
    def observe(**values):
        captured.append(values['projection'])
        return original(**values)
    monkeypatch.setattr(live, '_compose', observe)
    def mutate(payload):
        assert json.loads(payload)['thinking'] == {'type': 'disabled'}
        captured[0].groups[0].members[0].__pydantic_fields_set__.clear()
        if fail:
            raise RuntimeError('controlled failure')
        return response_for(payload)
    with pytest.raises(ValueError, match='changed'):
        live.run_live_bootstrap_report(**args, complete=mutate)


def test_disabled_report_rejects_mismatched_evidence_version_before_send(report_case):
    args = disabled_case(report_case)
    args['evidence_policy'] = report_case['evidence_policy']
    args['expected_evidence_policy_sha256'] = report_case['expected_evidence_policy_sha256']
    with pytest.raises(ValueError, match='configuration'):
        module().run_live_bootstrap_report(**args, complete=lambda _: pytest.fail('version drift sent'))


@pytest.mark.parametrize('mutation', (None, 'normal', 'exception'))
@pytest.mark.parametrize('version', (4, 5))
def test_guided_report_preserves_rank_and_both_exit_checks(report_case, monkeypatch, mutation, version):
    args = disabled_case(report_case)
    for key, pin in (('run_policy', 'expected_run_policy_sha256'), ('evidence_policy', 'expected_evidence_policy_sha256')):
        args[key] = args[key].model_copy(update={'configuration_version': f'live-bootstrap-report/{version}'})
        args[pin] = args[key].sha256
    before = bootstrap_audit.live_bootstrap_audit_bytes(args['audit'])
    calls = []
    captured = []
    live = module()
    original = live._compose
    def observe(**values):
        captured.append(values['projection'])
        return original(**values)
    monkeypatch.setattr(live, '_compose', observe)
    def complete(payload):
        request = json.loads(payload)
        assert request['thinking'] == {'type': 'disabled'}
        public = json.loads(request['messages'][1]['content'])
        assert public['prompt_version'] == f'nurse-v2-live-compact-report-prompt/{version-2}'
        if version == 5:
            assert public['readable_index'] and len(canonical_json_bytes(public)) <= 80000
        calls.append(payload)
        if mutation:
            captured[0].groups[0].members[0].__pydantic_fields_set__.clear()
        if mutation == 'exception':
            raise RuntimeError('controlled failure')
        return response_for(payload)
    if mutation:
        with pytest.raises(ValueError, match='changed'):
            live.run_live_bootstrap_report(**args, complete=complete)
    else:
        result = live.run_live_bootstrap_report(**args, complete=complete)
        assert result.schema_version == f'nurse.v2.live-bootstrap-report/{version}'
        assert result.exchange_protocol == f'nurse.v2.live-report-disabled-exchange/{version-2}'
        assert len(calls) == 1 and result.request_sha256 == sha256_hex(calls[0])
        assert result.status == 'full' and result.structural_status == 'valid'
        assert result.source_ranking_result_sha256 == args['audit'].result.result_sha256
        assert bootstrap_audit.live_bootstrap_audit_bytes(args['audit']) == before
        if version == 5:
            with pytest.raises(ValueError):
                live.LiveGuidedReportRun.model_validate(result.model_dump())


def test_indexed_configuration_cannot_silently_use_old_evidence_policy(report_case):
    args = disabled_case(report_case)
    args['run_policy'] = args['run_policy'].model_copy(update={'configuration_version': 'live-bootstrap-report/5'})
    args['expected_run_policy_sha256'] = args['run_policy'].sha256
    with pytest.raises(ValueError, match='configuration'):
        module().run_live_bootstrap_report(**args, complete=lambda _: pytest.fail('version drift sent'))
