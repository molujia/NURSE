import importlib
from dataclasses import replace

import pytest

from tests.test_v2_live_assembly import assemble
from tests.test_v2_live_sources import T0, case, load, write


def build(inputs, root, progress=None):
    module = importlib.import_module('nurse_rcl.v2.live.rules')
    return module.build_live_rule_state(inputs, source_root=root, progress=progress)


def test_real_parquet_observations_drive_complete_b0_rule_guard_set(tmp_path):
    root = case(tmp_path / 'case')
    for window, start, values in [('normal', T0 - 480, [10.] * 16),
                                   ('abnormal', T0, [0.] * 16)]:
        write(root, f'{window}_metrics.parquet', [start + i * 30 for i in range(16)],
              service_name=['svc'] * 16, metric=['k8s.deployment.available'] * 16, value=values)
    inputs = assemble(load(root))
    stages = []
    result = build(inputs, root, stages.append)
    assert len(result.binding.hypotheses) == 6
    assert len(result.guard_inputs.role_guard_results) == 6
    assert any(g.final_disposition == 'accepted' for g in result.guard_inputs.role_guard_results)
    assert any(g.final_disposition != 'accepted' for g in result.guard_inputs.role_guard_results)
    assert result.guard_inputs.frozen_candidates == inputs.frozen_candidates
    assert result.guard_inputs.query_sessions == ()
    assert result.guard_inputs.semantic_context is None
    assert stages[-1]['stage'] == 'rule_guards_completed'


def test_tampered_sources_cannot_enter_live_rule_binding(tmp_path):
    root = case(tmp_path / 'case')
    write(root, 'normal_metrics.parquet', [T0 - 1], service_name=['svc'], metric=['unknown'], value=[1.])
    inputs = assemble(load(root))
    tampered = replace(inputs, signals=())
    with pytest.raises(ValueError, match='source replay'):
        build(tampered, root)
