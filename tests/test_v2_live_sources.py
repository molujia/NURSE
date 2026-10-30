"""Real Parquet boundary tests; no mocked adapter or safety check."""

import importlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nurse_rcl.v2.rcabench_adapter import build_rcabench_adapter_config

T0 = 1_755_000_000
OPAQUE = 'opq_' + 'd' * 32


def case(root: Path) -> Path:
    root.mkdir()
    (root / 'env.json').write_text(json.dumps({
        'NORMAL_START': str(T0 - 480), 'NORMAL_END': str(T0),
        'ABNORMAL_START': str(T0), 'ABNORMAL_END': str(T0 + 480),
        'TIMEZONE': 'Asia/Shanghai',
    }))
    (root / 'injection.json').write_text('DO_NOT_READ_LABEL')
    (root / 'conclusion.parquet').write_bytes(b'DO_NOT_READ_LABEL')
    return root


def write(root, name, seconds, **columns):
    times = pa.array([t * 1_000_000_000 for t in seconds], type=pa.timestamp('ns', tz='UTC'))
    pq.write_table(pa.table({'time': times, **columns}), root / name)


def load(root):
    module = importlib.import_module('nurse_rcl.v2.live.sources')
    return module.read_rcabench_live_sources(
        root, opaque_case_id=OPAQUE,
        configuration=build_rcabench_adapter_config(
            window_seconds=480, expected_cadence_seconds=30, clock_uncertainty_seconds=0,
        ),
    )


def test_real_rows_are_bounded_projected_and_whole_case(tmp_path):
    root = case(tmp_path / 'private-answer-bearing-case-name')
    write(root, 'normal_metrics.parquet', [T0 - 481, T0 - 30, T0, T0 + 480],
          metric=['cpu'] * 4, value=[90., 1., 99., 100.], service_name=['svc'] * 4,
          answer=['POISON'] * 4)
    write(root, 'abnormal_metrics.parquet', [T0 - 1, T0, T0 + 479, T0 + 480],
          metric=['cpu'] * 4, value=[0., 4., 5., 999.], service_name=['svc'] * 4)
    write(root, 'abnormal_logs.parquet', [T0 + 1], service_name=['only-log-service'],
          level=['ERROR'], message=['untrusted log'], answer=['POISON'])
    write(root, 'abnormal_traces.parquet', [T0 + 2], service_name=['only-trace-service'],
          trace_id=['t'], span_id=['s'], parent_span_id=[None], duration=[123])
    result = load(root)
    by_name = {table.audit.source_file: table for table in result.tables}
    assert by_name['normal_metrics.parquet'].table.column('value').to_pylist() == [1.]
    assert by_name['abnormal_metrics.parquet'].table.column('value').to_pylist() == [4., 5.]
    assert all('answer' not in table.table.column_names for table in result.tables)
    assert set(result.observed_service_names) == {'svc', 'only-log-service', 'only-trace-service'}
    assert len(result.tables) == 4
    assert result.adapter.verify()
    assert result.verify_sources(root)
    # The inference-safe descriptor has source kinds/pins/counts, not raw names/messages/paths.
    descriptor = json.dumps(result.descriptor())
    assert 'private-answer-bearing' not in descriptor
    assert 'POISON' not in descriptor
    assert 'untrusted log' not in descriptor


def test_label_file_mutation_has_no_inference_effect(tmp_path):
    root = case(tmp_path / 'case')
    write(root, 'normal_metrics.parquet', [T0 - 1], metric=['cpu'], value=[1.], service_name=['svc'])
    before = load(root).descriptor()
    (root / 'injection.json').write_text('another answer')
    (root / 'conclusion.parquet').write_bytes(b'another answer')
    assert load(root).descriptor() == before


def test_source_replay_rejects_changed_values_not_only_self_consistent_hash(tmp_path):
    root = case(tmp_path / 'case')
    write(root, 'normal_metrics.parquet', [T0 - 1], metric=['cpu'], value=[1.], service_name=['svc'])
    before = load(root)
    write(root, 'normal_metrics.parquet', [T0 - 1], metric=['cpu'], value=[2.], service_name=['svc'])
    assert not before.verify_sources(root)


def test_symlink_cannot_escape_case_to_another_source(tmp_path):
    root = case(tmp_path / 'case')
    write(tmp_path, 'outside.parquet', [T0 - 1], metric=['cpu'], value=[1.], service_name=['svc'])
    (root / 'normal_metrics.parquet').symlink_to(tmp_path / 'outside.parquet')
    with pytest.raises(ValueError, match='symlink'):
        load(root)


def test_bad_modality_retained_as_unknown_not_invented_normal(tmp_path):
    root = case(tmp_path / 'case')
    (root / 'normal_logs.parquet').write_bytes(b'broken')
    write(root, 'normal_metrics.parquet', [T0 - 1], service_name=['metric-identity-only'])
    result = load(root)
    assert result.rejections == (('normal_logs.parquet', 'unreadable_parquet'),)
    assert result.observed_service_names == ('metric-identity-only',)
    assert result.tables[0].missing_columns == ('metric', 'value')


def test_no_sampling_or_candidate_cap(tmp_path):
    root = case(tmp_path / 'case')
    names = [f'service-{index}' for index in range(120)]
    write(root, 'abnormal_logs.parquet', [T0] * len(names), service_name=names,
          level=['INFO'] * len(names), message=['present'] * len(names))
    result = load(root)
    assert len(result.observed_service_names) == 120
    assert result.tables[0].table.num_rows == 120
