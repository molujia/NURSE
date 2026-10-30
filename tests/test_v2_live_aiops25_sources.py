"""Prepared real-format Parquet boundaries, without inference fixture sources."""

import importlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nurse_rcl.v2.aiops25_adapter import build_aiops25_adapter_config
from tests.test_v2_aiops25_adapter import (
    ALL_FILES, OPAQUE_CASE_ID, SOURCE_CASE_ID, T0, authorized_queue,
    write_prepared_case,
)


def prepared(root):
    times = {
        name: ([T0 - 481, T0 - 480, T0 - 1] if name.startswith('normal')
               else [T0, T0 + 479, T0 + 480])
        for name in ALL_FILES
    }
    case = write_prepared_case(root, files=times, env_update={
        'ABNORMAL_END': T0 + 600, 'ORIGINAL_ABNORMAL_END': T0 + 1,
    })
    for name, seconds in times.items():
        kind = name.split('_', 1)[1].removesuffix('.parquet')
        columns = {'time': pa.array([t * 10**9 for t in seconds],
                                   type=pa.timestamp('ns', tz='UTC')),
                   'service_name': [f'{kind}-service'] * 3,
                   'answer': ['POISON'] * 3}
        if kind == 'metrics':
            columns.update(metric=['pod_cpu_usage'] * 3, value=[1., 2., 999.])
        elif kind == 'logs':
            columns.update(level=['ERROR'] * 3, message=['untrusted prose'] * 3)
        else:
            columns.update(trace_id=['t'] * 3, span_id=['s'] * 3,
                           parent_span_id=[None] * 3, duration=[1000] * 3)
        pq.write_table(pa.table(columns), case / name)
    return case


def load(root, queue=None):
    module = importlib.import_module('nurse_rcl.v2.live.aiops25_sources')
    return module.read_aiops25_live_sources(
        root, source_case_id=SOURCE_CASE_ID, opaque_case_id=OPAQUE_CASE_ID,
        authorized_queue=queue if queue is not None else authorized_queue(),
        configuration=build_aiops25_adapter_config(),
    )


def test_six_complete_sources_keep_decision_gate_and_whitelist(tmp_path):
    root = prepared(tmp_path)
    result = load(root)
    assert result.adapter.verify()
    assert len(result.tables) == 6
    assert all(item.table.num_rows == 2 for item in result.tables)
    assert all('answer' not in item.table.column_names for item in result.tables)
    assert all(item.missing_columns == () for item in result.tables)
    assert result.observed_service_names == ('logs-service', 'metrics-service', 'traces-service')
    assert result.descriptor()['schema_version'] == 'nurse.v2.aiops25-live-sources/1'
    assert result.verify_sources(root)
    assert 'POISON' not in json.dumps(result.descriptor())
    assert 'untrusted prose' not in json.dumps(result.descriptor())
    (root / 'injection.json').write_text('other forbidden answer')
    assert result.verify_sources(root)


@pytest.mark.parametrize('queue', [
    authorized_queue(destination='ark'), authorized_queue(case_ids=('excluded',)),
])
def test_wrong_authorization_fails_before_source_read(tmp_path, queue, monkeypatch):
    root = prepared(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('telemetry must not be read')
    monkeypatch.setattr(pq, 'ParquetFile', forbidden)
    with pytest.raises(ValueError, match='inference|authorized'):
        load(root, queue)


def test_modified_source_or_time_contract_is_not_replayed_as_trusted(tmp_path):
    root = prepared(tmp_path)
    result = load(root)
    path = root / 'abnormal_metrics.parquet'
    table = pq.read_table(path)
    pq.write_table(table.set_column(table.column_names.index('value'), 'value',
                                   pa.array([50., 60., 999.])), path)
    assert not result.verify_sources(root)


@pytest.mark.parametrize('name', ['env.json', '.prepared.json', 'normal_metrics.parquet'])
def test_symlink_rejected(tmp_path, name):
    root = prepared(tmp_path)
    path = root / name
    outside = tmp_path / ('outside-' + name)
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        load(root)


@pytest.mark.parametrize('raises', [False, True])
@pytest.mark.parametrize('name', ['env.json', '.prepared.json'])
def test_source_mutation_on_normal_and_exceptional_read_is_rejected(
    tmp_path, monkeypatch, raises, name,
):
    root = prepared(tmp_path)
    original = pq.ParquetFile.read
    def mutate(parquet, *args, **kwargs):
        table = original(parquet, *args, **kwargs)
        path = root / name
        path.write_bytes(path.read_bytes() + b' ')
        if raises:
            raise RuntimeError('downstream failure')
        return table
    monkeypatch.setattr(pq.ParquetFile, 'read', mutate)
    with pytest.raises(ValueError, match='changed'):
        load(root)


def test_incomplete_modality_is_recorded_not_invented(tmp_path):
    root = prepared(tmp_path)
    (root / 'normal_logs.parquet').write_bytes(b'broken parquet')
    result = load(root)
    assert len(result.tables) == 5
    assert ('normal_logs.parquet', 'unreadable_parquet') in result.rejections
    assert result.verify_sources(root)


def test_no_source_or_service_sampling(tmp_path):
    count = 120
    root = write_prepared_case(tmp_path, files={'abnormal_logs.parquet': [T0] * count})
    pq.write_table(pa.table({
        'time': pa.array([T0 * 10**9] * count, type=pa.timestamp('ns', tz='UTC')),
        'service_name': [f'only-log-{n}' for n in range(count)],
        'level': ['INFO'] * count, 'message': ['present'] * count,
    }), root / 'abnormal_logs.parquet')
    result = load(root)
    assert result.tables[0].table.num_rows == count
    assert len(result.observed_service_names) == count
