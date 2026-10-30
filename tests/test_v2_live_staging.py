"""Provisioning is separate from inference; real files, no model or labels in loader."""

import importlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.opaque_ids import (
    EvaluatorIdentitySource, build_identity_artifacts,
    write_evaluator_identity_map, write_inference_identity_manifest,
)
from nurse_rcl.v2.rcabench_adapter import build_rcabench_adapter_config
from tests.test_v2_live_sources import T0, case, write


def setup_case(tmp_path):
    private = tmp_path / 'private'
    private.mkdir()
    raw = case(private / 'answer-bearing-case')
    # Explicit poison in unrelated metadata must not enter inference storage.
    env = json.loads((raw / 'env.json').read_text())
    env.update(answer='SECRET_LABEL', original_path='/private/answer-bearing-case')
    (raw / 'env.json').write_text(json.dumps(env))
    write(raw, 'normal_metrics.parquet', [T0 - 30], metric=['container.cpu.usage'],
          value=[1.], service_name=['telemetry-service'], answer=['SECRET_LABEL'])
    write(raw, 'abnormal_metrics.parquet', [T0, T0 + 480],
          metric=['container.cpu.usage'] * 2, value=[3., 900.],
          service_name=['telemetry-service'] * 2, answer=['SECRET_LABEL'] * 2)
    table = pq.read_table(raw / 'normal_metrics.parquet')
    table = table.replace_schema_metadata({b'answer': b'SECRET_LABEL'})
    pq.write_table(table, raw / 'normal_metrics.parquet')
    mapping, manifest = build_identity_artifacts((EvaluatorIdentitySource(
        dataset='rcabench', original_case_id=raw.name,
        incident_group_id='grp_' + 'a' * 24, answer_services=('SECRET_LABEL',),
    ),), method='random_uuid_v4')
    evaluator_path, inference_path = private / 'map.json', tmp_path / 'identities.json'
    write_evaluator_identity_map(evaluator_path, mapping)
    write_inference_identity_manifest(inference_path, manifest)
    stage_root = tmp_path / 'staged'
    stage_root.mkdir(mode=0o700)
    args = dict(source_root=raw, evaluator_map_path=evaluator_path,
                expected_evaluator_artifact_sha256=sha256_hex(evaluator_path.read_bytes()),
                inference_manifest_path=inference_path,
                expected_inference_artifact_sha256=sha256_hex(inference_path.read_bytes()),
                stage_root=stage_root)
    return args, manifest.records[0].opaque_case_id


def provision(args):
    module = importlib.import_module('nurse_rcl.v2.provision_live_case')
    return module.provision_rcabench_live_case(**args)


def load(args, opaque, pin):
    module = importlib.import_module('nurse_rcl.v2.live.staged_case')
    return module.load_staged_rcabench_case(
        args['stage_root'] / opaque, expected_stage_sha256=pin,
        inference_manifest_path=args['inference_manifest_path'],
        expected_inference_artifact_sha256=args['expected_inference_artifact_sha256'],
        configuration=build_rcabench_adapter_config(),
    )


def test_provision_projects_identity_env_columns_and_metadata_without_label_reads(tmp_path):
    args, opaque = setup_case(tmp_path)
    # Unreadable garbage evaluator files in the raw case must never be opened.
    result = provision(args)
    assert result.opaque_case_id == opaque
    root = args['stage_root'] / opaque
    assert root.resolve().name == opaque and not root.is_symlink()
    assert {p.name for p in root.iterdir()} == {
        'env.json', 'normal_metrics.parquet', 'abnormal_metrics.parquet', 'stage.json',
    }
    for path in root.iterdir():
        data = path.read_bytes()
        assert b'SECRET_LABEL' not in data
        assert b'answer-bearing-case' not in data
        assert b'original_path' not in data
        assert not path.is_symlink()
    table = pq.read_table(root / 'normal_metrics.parquet')
    assert table.schema.metadata is None
    assert table.column_names == ['time', 'service_name', 'metric', 'value']
    assert table.column('value').to_pylist() == [1.]
    assert pq.read_table(root / 'abnormal_metrics.parquet').num_rows == 2
    sources = load(args, opaque, result.stage_sha256)
    assert sources.adapter.opaque_case_id == opaque
    assert sources.tables[0].table.column('value').to_pylist() == [3.]
    assert sources.adapter.time_contract.late_data_observed
    if os.name == 'posix':
        assert root.stat().st_mode & 0o777 == 0o700
        assert all(p.stat().st_mode & 0o777 == 0o600 for p in root.iterdir())


@pytest.mark.parametrize('key', ['expected_evaluator_artifact_sha256', 'expected_inference_artifact_sha256'])
def test_provision_requires_independent_identity_pins(tmp_path, key):
    args, _ = setup_case(tmp_path)
    args[key] = '0' * 64
    with pytest.raises(ValueError, match='identity'):
        provision(args)
    assert not list(args['stage_root'].iterdir())


def test_unauthorized_case_not_routed_by_supplied_opaque_id(tmp_path):
    args, _ = setup_case(tmp_path)
    args['source_root'] = case(args['source_root'].parent / 'unregistered')
    with pytest.raises(ValueError, match='authorized'):
        provision(args)
    assert not list(args['stage_root'].iterdir())


@pytest.mark.parametrize('target', ['case', 'telemetry', 'parent'])
def test_symlinks_never_smuggle_raw_names_or_sources_into_inference(tmp_path, target):
    args, opaque = setup_case(tmp_path)
    if target == 'case':
        link = tmp_path / 'alias'
        link.symlink_to(args['source_root'], target_is_directory=True)
        args['source_root'] = link
    elif target == 'parent':
        link = tmp_path / 'alias-parent'
        link.symlink_to(args['source_root'].parent, target_is_directory=True)
        args['source_root'] = link / args['source_root'].name
    else:
        source = args['source_root'] / 'normal_metrics.parquet'
        moved = args['source_root'] / 'outside.parquet'
        source.rename(moved)
        source.symlink_to(moved)
    with pytest.raises(ValueError, match='symlink'):
        provision(args)
    assert not (args['stage_root'] / opaque).exists()


def test_existing_stage_never_overwritten(tmp_path):
    args, opaque = setup_case(tmp_path)
    result = provision(args)
    pin = sha256_hex((args['stage_root'] / opaque / 'stage.json').read_bytes())
    with pytest.raises(ValueError, match='exists'):
        provision(args)
    assert sha256_hex((args['stage_root'] / opaque / 'stage.json').read_bytes()) == pin
    assert load(args, opaque, result.stage_sha256).adapter.verify()


@pytest.mark.parametrize('corruption', ['file', 'extra_file', 'extra_field', 'stage_pin', 'manifest_pin'])
def test_inference_loader_rejects_corruption_or_identity_substitution(tmp_path, corruption):
    args, opaque = setup_case(tmp_path)
    result = provision(args)
    root = args['stage_root'] / opaque
    pin = result.stage_sha256
    if corruption == 'file':
        (root / 'normal_metrics.parquet').write_bytes(b'changed')
    elif corruption == 'extra_file':
        (root / 'injection.json').write_bytes(b'SECRET_LABEL')
    elif corruption == 'extra_field':
        payload = json.loads((root / 'stage.json').read_bytes())
        payload['answer'] = 'SECRET_LABEL'
        payload['stage_sha256'] = sha256_hex(canonical_json_bytes({
            k: v for k, v in payload.items() if k != 'stage_sha256'}))
        pin = payload['stage_sha256']
        (root / 'stage.json').write_bytes(canonical_json_bytes(payload))
    elif corruption == 'manifest_pin':
        args['expected_inference_artifact_sha256'] = '0' * 64
    else:
        pin = '0' * 64
    with pytest.raises(ValueError):
        load(args, opaque, pin)


def test_inference_can_run_after_private_inputs_are_unavailable(tmp_path):
    args, opaque = setup_case(tmp_path)
    result = provision(args)
    args['evaluator_map_path'].rename(args['evaluator_map_path'].with_suffix('.offline'))
    args['source_root'].rename(args['source_root'].with_name('offline'))
    assert load(args, opaque, result.stage_sha256).adapter.verify()


def test_entire_case_retains_identity_only_and_all_modalities(tmp_path):
    args, opaque = setup_case(tmp_path)
    raw = args['source_root']
    names = [f'log-only-{index}' for index in range(120)]
    write(raw, 'abnormal_logs.parquet', [T0] * 120, service_name=names,
          level=['INFO'] * 120, message=['observed'] * 120)
    write(raw, 'abnormal_traces.parquet', [T0], service_name=['trace-only'],
          trace_id=['1' * 32], span_id=['2' * 16], parent_span_id=[None], duration=[10])
    result = provision(args)
    sources = load(args, opaque, result.stage_sha256)
    assert len(sources.tables) == 4
    assert len(sources.observed_service_names) == 122
