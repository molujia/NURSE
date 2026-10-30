"""Live audit never mistakes a rehashed ranking for a source replay."""
import copy
import importlib
from dataclasses import replace

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2 import hypothesis_guard
from nurse_rcl.v2.case_audit_read import read_case_audit
from nurse_rcl.v2.live import bootstrap_scoring as scoring
from tests.test_v2_live_bootstrap_scoring import bootstrap_case, prepare  # noqa: F401


def module():
    try:
        return importlib.import_module('nurse_rcl.v2.live.bootstrap_audit')
    except ModuleNotFoundError:
        pytest.fail('missing explicit live source and complete score audit', pytrace=False)


def arguments(bootstrap_case):
    return prepare(copy.deepcopy(bootstrap_case))[0]


def pins(args):
    return {key: value for key, value in args.items() if key.startswith('expected_')}


def test_complete_live_wire_roundtrip_and_independent_source_replay(bootstrap_case, monkeypatch):
    live, args = module(), arguments(bootstrap_case)
    seen = []
    original = hypothesis_guard._validated_inputs
    def observe(**values):
        seen.append(values['event_grouping'])
        return original(**values)
    monkeypatch.setattr(hypothesis_guard, '_validated_inputs', observe)
    audit = live.build_live_bootstrap_audit(**args)
    assert audit.schema_version == 'nurse.v2.live-bootstrap-audit/1'
    raw = live.live_bootstrap_audit_bytes(audit)
    digest = sha256_hex(raw)
    loaded = live.load_live_bootstrap_audit(raw, expected_audit_sha256=digest)
    assert loaded == audit
    result = live.replay_live_bootstrap_audit(audit=loaded, expected_audit_sha256=digest, **pins(args))
    assert result == audit.result
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert raw.count(b'"nurse.v2.live-b0-scoring-sources/1"') == 1
    with pytest.raises(ValueError):
        read_case_audit(audit_package=audit, expected_audit_sha256=digest)


@pytest.mark.parametrize('field', ('scores', 'evidence', 'ordered_service_keys'))
def test_rehashed_forgery_rejected_by_complete_replay(bootstrap_case, field):
    live, args = module(), arguments(bootstrap_case)
    audit = live.build_live_bootstrap_audit(**args)
    result = audit.result
    if field == 'scores':
        # Change a non-leading decomposition without changing the claimed Top-K.
        row = result.scores[-1]
        changed = row.model_copy(update={'quantized_total_score': '0.123456789012'})
        changed = changed.model_copy(update={'score_sha256': sha256_hex(canonical_json_bytes(
            changed.model_dump(mode='json', exclude={'score_sha256'})))})
        result = result.model_copy(update={'scores': (*result.scores[:-1], changed)})
    elif field == 'evidence':
        evidence = result.evidence.model_copy(update={'unbound_assertion_ids': ('forged',)})
        evidence = evidence.model_copy(update={'result_sha256': sha256_hex(canonical_json_bytes(
            evidence.model_dump(mode='json', exclude={'result_sha256'})))})
        result = result.model_copy(update={'evidence': evidence})
    else:
        result = result.model_copy(update={'ordered_service_keys': tuple(reversed(result.ordered_service_keys))})
    result = result.model_copy(update={'result_sha256': sha256_hex(canonical_json_bytes(
        result.model_dump(mode='json', exclude={'result_sha256'})))})
    audit = replace(audit, result=result)
    raw = live.live_bootstrap_audit_bytes(audit)
    with pytest.raises(ValueError, match='replay|differs'):
        live.replay_live_bootstrap_audit(audit=audit,
            expected_audit_sha256=sha256_hex(raw), **pins(args))


@pytest.mark.parametrize('pin', ('expected_source_records_sha256', 'expected_rule_set_sha256',
    'expected_binding_sha256', 'expected_selection_sha256', 'expected_context_sha256',
    'expected_audit_sha256'))
def test_each_independent_pin_required(bootstrap_case, pin):
    live, args = module(), arguments(bootstrap_case)
    audit = live.build_live_bootstrap_audit(**args)
    values = dict(**pins(args), expected_audit_sha256=sha256_hex(live.live_bootstrap_audit_bytes(audit)))
    values[pin] = 'f' * 64
    with pytest.raises(ValueError):
        live.replay_live_bootstrap_audit(audit=audit, **values)


@pytest.mark.parametrize('fail', (False, True))
def test_audit_original_mutation_rejected_on_normal_and_exception_exit(bootstrap_case, fail, monkeypatch):
    live, args = module(), arguments(bootstrap_case)
    audit = live.build_live_bootstrap_audit(**args)
    digest = sha256_hex(live.live_bootstrap_audit_bytes(audit))
    original = scoring._rank
    def mutate(**values):
        result = original(**values)
        audit.result.__pydantic_fields_set__.clear()
        if fail:
            raise RuntimeError('controlled audit failure')
        return result
    monkeypatch.setattr(scoring, '_rank', mutate)
    with pytest.raises(ValueError, match='changed'):
        live.replay_live_bootstrap_audit(audit=audit, expected_audit_sha256=digest, **pins(args))


def test_audit_json_rejects_noncanonical_shape_before_replay(bootstrap_case):
    live, args = module(), arguments(bootstrap_case)
    audit = live.build_live_bootstrap_audit(**args)
    raw = b' ' + live.live_bootstrap_audit_bytes(audit)
    with pytest.raises(ValueError, match='canonical'):
        live.load_live_bootstrap_audit(raw, expected_audit_sha256=sha256_hex(raw))
