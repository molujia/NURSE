"""Persistent bridge tests at the HTTP boundary; all requests are synthetic."""

import importlib
import json
from dataclasses import replace

import httpx
import pytest

from nurse_rcl.agents.ark import ARK_MODEL, load_ark_settings
from nurse_rcl.artifacts import sha256_hex
from nurse_rcl.v2.hermes_exchange import prepare_exchange

CASE = 'opq_' + 'd' * 32


def module():
    return importlib.import_module('nurse_rcl.v2.live.ark_bridge')


def request(model=ARK_MODEL):
    return prepare_exchange(model=model,
                            messages=[{'role': 'user', 'content': json.dumps({'opaque_case_id': CASE})}],
                            max_output_tokens=64).payload_bytes


def response(content='{}'):
    return {'choices': [{'index': 0, 'finish_reason': 'stop',
                         'message': {'role': 'assistant', 'content': content}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}}


def bridge(tmp_path, handler, **changes):
    arguments = dict(root=tmp_path / 'run', lock_path=tmp_path / 'key2.lock',
                     settings=load_ark_settings({'ARK_API_KEY2': 'only-key-two'}),
                     code_sha256='a' * 64, config_sha256='b' * 64,
                     authorized_case_ids=(CASE,), transport=httpx.MockTransport(handler),
                     sleep=lambda seconds: None)
    arguments.update(changes)
    return module().LiveArkBridge(**arguments)


def binding(payload=None, **changes):
    payload = payload or request()
    fields = dict(opaque_case_id=CASE, variant='v2-live-bootstrap', phase='report',
                  input_sha256='c' * 64, context_sha256='e' * 64,
                  prompt_schema_sha256='f' * 64, request_sha256=sha256_hex(payload))
    fields.update(changes)
    return module().LiveRequestBinding(**fields)


def test_fixed_endpoint_key_and_valid_response_reuse(tmp_path):
    calls = []
    def http(req):
        calls.append(req)
        assert str(req.url) == 'https://ark.cn-beijing.volces.com/api/v3/chat/completions'
        assert req.headers['Authorization'] == 'Bearer only-key-two'
        return httpx.Response(200, json=response())
    with bridge(tmp_path, http) as client:
        value = client.complete(binding(), request())
    with bridge(tmp_path, http) as restarted:
        assert restarted.complete(binding(), request()) == value
        receipt = restarted.receipt(binding())
    assert len(calls) == 1
    assert receipt['attempt_count'] == 1
    assert receipt['response_reuses'] == 1
    assert 'only-key-two' not in ''.join(p.read_text() for p in (tmp_path / 'run').glob('*.json'))


def test_five_attempts_total_across_process_interruption(tmp_path):
    calls = []
    def http(req):
        calls.append(req)
        raise KeyboardInterrupt()
    for _ in range(5):
        with bridge(tmp_path, http) as client, pytest.raises(KeyboardInterrupt):
            client.complete(binding(), request())
    with bridge(tmp_path, http) as client:
        with pytest.raises(module().LiveTransportError, match='exhausted'):
            client.complete(binding(), request())
        receipt = client.receipt(binding())
    assert len(calls) == receipt['attempt_count'] == 5
    assert receipt['outcome_unknown_count'] == 5


def test_structural_failure_retries_but_valid_domain_rejection_does_not(tmp_path):
    calls = []
    def http(req):
        calls.append(req)
        return httpx.Response(200, json=response('not JSON' if len(calls) == 1 else '{}'))
    with bridge(tmp_path, http) as client:
        # {} is structurally valid but not a domain proposal. The bridge MUST
        # return it to the domain guard, never draw again to improve the answer.
        assert json.loads(client.complete(binding(), request())) == response()
        assert client.receipt(binding())['attempt_count'] == 2
    assert len(calls) == 2


@pytest.mark.parametrize('status, expected', [(401, 1), (429, 5), (503, 5)])
def test_http_failures_have_bounded_terminal_receipts(tmp_path, status, expected):
    calls = []
    def http(req):
        calls.append(req)
        return httpx.Response(status, json={'error': 'untrusted text'})
    with bridge(tmp_path, http) as client:
        with pytest.raises(module().LiveTransportError):
            client.complete(binding(), request())
    with bridge(tmp_path, http) as restarted:
        with pytest.raises(module().LiveTransportError):
            restarted.complete(binding(), request())
    assert len(calls) == expected


def test_changed_payload_model_or_unauthorized_case_cannot_send(tmp_path):
    calls = []
    def http(req):
        calls.append(req)
        return httpx.Response(200, json=response())
    with bridge(tmp_path, http) as client:
        for bind, payload in ((binding(), request() + b' '),
                              (binding(opaque_case_id='opq_' + '0' * 32), request()),
                              (binding(request('wrong-model')), request('wrong-model'))):
            with pytest.raises(ValueError):
                client.complete(bind, payload)
    assert calls == []


def test_same_slot_cannot_reset_budget_with_new_context_or_request(tmp_path):
    def http(req):
        return httpx.Response(401, json={})
    with bridge(tmp_path, http) as client:
        with pytest.raises(module().LiveTransportError):
            client.complete(binding(), request())
        with pytest.raises(ValueError, match='binding'):
            client.complete(binding(context_sha256='1' * 64), request())
    with pytest.raises(ValueError, match='run identity'):
        bridge(tmp_path, http, config_sha256='2' * 64)


def test_cross_process_network_lock_blocks_without_consuming_attempt(tmp_path):
    import fcntl
    def http(req):
        pytest.fail('must not send while global lock is owned')
    with bridge(tmp_path, http) as client, (tmp_path / 'key2.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(module().LiveTransportError, match='in_flight'):
            client.complete(binding(), request())
    assert not list((tmp_path / 'run').glob('request-*.json'))


def test_variants_have_separate_report_slots_but_no_third_semantic_round(tmp_path):
    with pytest.raises(ValueError):
        binding(phase='semantic_third')
    with pytest.raises(ValueError):
        binding(variant='B1', phase='semantic_first')
    with bridge(tmp_path, lambda req: httpx.Response(200, json=response())) as client:
        client.complete(binding(variant='B1'), request())
        client.complete(binding(variant='B2'), request())
        assert client.receipt(binding(variant='B1'))['attempt_count'] == 1
        assert client.receipt(binding(variant='B2'))['attempt_count'] == 1


@pytest.mark.parametrize('user_data', [
    {'opaque_case_id': 'opq_' + 'e' * 32},
    {'evidence_view': {'opaque_case_id': 'opq_' + 'e' * 32}},
    {'opaque_case_id': CASE, 'nested': {'opaque_case_id': 'opq_' + 'e' * 32}},
    {'no_identity': True},
])
def test_payload_case_must_match_binding_even_when_both_are_authorized(tmp_path, user_data):
    calls = []
    def http(req):
        calls.append(req)
        return httpx.Response(200, json=response())
    payload = prepare_exchange(model=ARK_MODEL,
        messages=[{'role': 'user', 'content': json.dumps(user_data)}],
        max_output_tokens=64).payload_bytes
    with bridge(tmp_path, http, authorized_case_ids=(CASE, 'opq_' + 'e' * 32)) as client:
        with pytest.raises(ValueError, match='payload case'):
            client.complete(binding(payload), payload)
    assert calls == []
    assert not list((tmp_path / 'run').glob('request-*.json'))


def test_durable_response_survives_crash_before_final_bookkeeping(tmp_path, monkeypatch):
    calls = []
    def http(req):
        calls.append(req)
        return httpx.Response(200, json=response())
    with bridge(tmp_path, http) as client:
        save = client._save
        def crash_before_final_write(ledger):
            if ledger.attempts and ledger.attempts[-1].record is not None:
                raise KeyboardInterrupt()
            save(ledger)
        monkeypatch.setattr(client, '_save', crash_before_final_write)
        with pytest.raises(KeyboardInterrupt):
            client.complete(binding(), request())
    with bridge(tmp_path, http) as client:
        assert json.loads(client.complete(binding(), request())) == response()
        receipt = client.receipt(binding())
    assert len(calls) == receipt['attempt_count'] == 1
    assert receipt['response_reuses'] == 1
    assert receipt['ledger']['attempts'][0]['interrupted'] is True
    assert receipt['ledger']['attempts'][0]['latency_known'] is False
