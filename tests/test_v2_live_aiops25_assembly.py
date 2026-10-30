import importlib
import json
from dataclasses import fields, replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nurse_rcl.v2.signal_transforms import ScalarSample
from tests.test_v2_aiops25_adapter import ALL_FILES, T0, write_prepared_case
from tests.test_v2_live_aiops25_sources import load


def measured_case(tmp_path):
    metrics = ('pod_cpu_usage', 'pod_memory_working_set_bytes', 'pod_processes',
               'error_ratio', 'pod_network_receive_bytes', 'unregistered_cpu')
    files = {}
    for name in ALL_FILES:
        start = T0 - 480 if name.startswith('normal') else T0
        times = [start + n * 30 for n in range(16)]
        files[name] = times * len(metrics) if 'metrics' in name else times
    root = write_prepared_case(tmp_path, files=files, env_update={
        'ABNORMAL_END': T0 + 600, 'ORIGINAL_ABNORMAL_END': T0 + 1,
    })
    for name, times in files.items():
        baseline = name.startswith('normal')
        kind = name.split('_', 1)[1].removesuffix('.parquet')
        values = {'time': pa.array([t * 10**9 for t in times], type=pa.timestamp('ns', tz='UTC')),
                  'service_name': [f'{kind}-only-private'] * len(times)}
        if kind == 'metrics':
            values.update(metric=[m for m in metrics for _ in range(16)],
                          value=[((1. if baseline else 4.) + n) / (100 if m == 'error_ratio' else 1)
                                 for m in metrics for n in range(16)])
        elif kind == 'logs':
            values.update(level=['INFO' if baseline else 'ERROR'] * len(times),
                          message=['raw private prose'] * len(times))
        else:
            values.update(trace_id=['t'] * len(times), span_id=['s'] * len(times),
                          parent_span_id=[None] * len(times),
                          duration=[1000 if baseline else 4000] * len(times))
        pq.write_table(pa.table(values), root / name)
    return root


def assemble(sources, **kwargs):
    return importlib.import_module('nurse_rcl.v2.live.assembly').assemble_aiops25_live_inputs(sources, **kwargs)


def test_aiops_values_use_distinct_semantics_and_complete_candidates(tmp_path):
    root = measured_case(tmp_path)
    result = assemble(load(root), q_base=1, q_max=1)
    assert result.candidate_universe.candidate_count == 3
    assert len(result.focus.focus_set.selected_service_keys) == 1
    assert len(result.signals) == 8
    by_family = {s.signal_family: s for s in result.signals if s.signal_family != 'unclassified'}
    assert {'cpu', 'memory', 'lifecycle', 'error', 'latency'} <= by_family.keys()
    cpu = by_family['cpu']
    assert cpu.transform.signal_type == 'gauge'
    assert cpu.quality.formal_scoring_eligible
    assert cpu.observation.baseline_summary.median == 8.5
    assert cpu.observation.fault_summary.median == 11.5
    unknown = [s for s in result.signals if s.signal_family == 'unclassified']
    assert len(unknown) == 2
    assert all(s.transform.signal_type == 'unknown' for s in unknown)
    trace = by_family['latency']
    assert trace.observation.unit == 'ns'
    assert len(trace.fault.samples) == 16
    descriptor = result.inference_descriptor()
    assert descriptor['schema_version'] == 'nurse.v2.aiops25-live-inputs/1'
    module = importlib.import_module('nurse_rcl.v2.live.assembly')
    assert descriptor['semantics_sha256'] != module.SEMANTICS_SHA256
    public = json.dumps(descriptor)
    assert 'only-private' not in public and 'raw private prose' not in public
    assert '009be6db-313' not in public
    assert result.verify_sources(root)


def test_flat_native_metric_does_not_invent_absolute_resolution(tmp_path):
    root = measured_case(tmp_path)
    for window in ('normal', 'abnormal'):
        path = root / f'{window}_metrics.parquet'
        table = pq.read_table(path)
        pq.write_table(table.set_column(table.column_names.index('value'), 'value',
            pa.array([1. if window == 'normal' else 4.] * table.num_rows)), path)
    cpu = next(s for s in assemble(load(root)).signals if s.signal_family == 'cpu')
    assert not cpu.quality.formal_scoring_eligible
    assert cpu.observation.evidence_state == 'unknown'
    assert 'absolute_resolution_unresolved' in cpu.quality.quality_flags


def test_aiops_input_replay_rejects_changed_retained_sample(tmp_path):
    root = measured_case(tmp_path)
    result = assemble(load(root))
    signal = result.signals[0]
    sample = signal.baseline.samples[0]
    changed = signal.baseline.model_copy(update={'samples': (
        ScalarSample(timestamp_unix_ns=sample.timestamp_unix_ns, value=999.),
        *signal.baseline.samples[1:],
    )})
    altered = replace(result, signals=(replace(signal, baseline=changed), *result.signals[1:]))
    assert not altered.verify_sources(root)


def test_aiops_cannot_be_relabelled_as_legacy_live_input(tmp_path):
    root = measured_case(tmp_path)
    result = assemble(load(root))
    module = importlib.import_module('nurse_rcl.v2.live.assembly')
    assert type(result) is module.AIOps25LiveInputs
    legacy = module.LiveInputs(**{f.name: getattr(result, f.name) for f in fields(result)})
    assert not legacy.verify_sources(root)


def test_dataset_specific_entrypoints_refuse_swapped_source_types(tmp_path):
    from tests.test_v2_live_sources import T0 as RCAB_T0, case, load as load_rcab, write
    root = case(tmp_path / 'rcab')
    write(root, 'normal_metrics.parquet', [RCAB_T0 - 1], metric=['k8s.container.ready'],
          value=[1.], service_name=['svc'])
    with pytest.raises(ValueError, match='source adapter'):
        assemble(load_rcab(root))
    aiops_root = measured_case(tmp_path / 'aiops')
    module = importlib.import_module('nurse_rcl.v2.live.assembly')
    with pytest.raises(ValueError, match='source adapter'):
        module.assemble_rcabench_live_inputs(load(aiops_root))
