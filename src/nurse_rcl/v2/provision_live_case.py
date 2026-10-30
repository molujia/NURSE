"""Private provisioning command; run and EXIT before starting live inference.

Only this process may load the pinned evaluator identity map. Answers are not
used, logged or projected. It exports actual telemetry under the existing opaque
ID, with allowlisted columns and time metadata. Never call it from a model runner.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.sources import _ENTITY_COLUMNS, _VALUE_COLUMNS, file_sha256
from nurse_rcl.v2.live.staged_case import (
    ENV_FIELDS, TELEMETRY_FILES, StagedCase, StagedFile,
    read_inference_identity, require_plain_path,
)
from nurse_rcl.v2.opaque_ids import EvaluatorIdentityMap, inference_manifest_from_evaluator
from nurse_rcl.v2.rcabench_adapter import _read_env_contract


def _write_private(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def provision_rcabench_live_case(
    *, source_root: Path, evaluator_map_path: Path, expected_evaluator_artifact_sha256: str,
    inference_manifest_path: Path, expected_inference_artifact_sha256: str, stage_root: Path,
) -> StagedCase:
    """Create one new immutable directory; no label-driven choices or inference."""
    source = require_plain_path(source_root)
    destination = require_plain_path(stage_root)
    private_map = require_plain_path(evaluator_map_path)
    if (os.name == 'posix' and stat.S_IMODE(private_map.stat().st_mode) != 0o600):
        raise ValueError('evaluator identity map must be private')
    if private_map.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('evaluator identity map exceeds bounded size')
    data = private_map.read_bytes()
    if sha256_hex(data) != expected_evaluator_artifact_sha256:
        raise ValueError('evaluator identity artifact pin mismatch')
    mapping = EvaluatorIdentityMap.model_validate_json(data)
    manifest = read_inference_identity(inference_manifest_path, expected_inference_artifact_sha256)
    if not mapping.verify() or inference_manifest_from_evaluator(mapping) != manifest:
        raise ValueError('private and inference identity artifacts disagree')
    matches = [record for record in mapping.records
               if record.dataset == 'rcabench' and record.original_case_id == source.name]
    if len(matches) != 1:
        raise ValueError('case is not authorized by the pinned identity manifests')
    opaque = matches[0].opaque_case_id
    return _export_rcabench_live_case(
        source=source, destination=destination, opaque=opaque,
        expected_inference_artifact_sha256=expected_inference_artifact_sha256,
    )


def provision_selected_rcabench_case(
    *, source_root: Path, selected_case_id: str, public_manifest_path: Path,
    expected_public_manifest_sha256: str, inference_manifest_path: Path,
    expected_inference_artifact_sha256: str, stage_root: Path,
) -> StagedCase:
    """Already selected public-inference case; never opens an evaluator identity map.

    This is explicit administrative input admission, not a model tool, sampler or
    authority inferred from a case name. Caller owns the independent selection pin.
    The one-record opaque inference manifest binds only this fresh case export.
    """
    from nurse_rcl.v2.exchange_json import ExchangeLimits, decode_json_object

    source, destination = require_plain_path(source_root), require_plain_path(stage_root)
    public = require_plain_path(public_manifest_path)
    if public.stat().st_size > 4 * 1024**2:
        raise ValueError('public inference manifest exceeds capacity')
    raw = public.read_bytes()
    if sha256_hex(raw) != expected_public_manifest_sha256:
        raise ValueError('public inference manifest external pin differs')
    value = decode_json_object(raw, ExchangeLimits(4 * 1024**2, 64, 100_000))
    if value.get('dataset') != 'rcabench' or source.name != selected_case_id:
        raise ValueError('selected public case differs from source')
    matches = [row for row in value.get('cases', []) if row.get('case_id') == selected_case_id]
    if len(matches) != 1 or matches[0].get('dataset') != 'rcabench':
        raise ValueError('selected case absent or ambiguous in pinned public inference population')
    telemetry = matches[0].get('telemetry')
    if type(telemetry) is not list:
        raise ValueError('selected public telemetry inventory missing')
    declared = []
    for row in telemetry:
        path = Path(row['path'])
        if path.parent != source or path.name not in TELEMETRY_FILES:
            raise ValueError('public telemetry path is outside the selected source')
        declared.append(require_plain_path(path))
    actual = {require_plain_path(source / name) for name in TELEMETRY_FILES if (source / name).exists()}
    if len(declared) != len(set(declared)) or set(declared) != actual:
        raise ValueError('selected public whole available telemetry inventory differs')
    manifest = read_inference_identity(inference_manifest_path, expected_inference_artifact_sha256)
    if manifest.case_count != 1 or manifest.rcabench_case_count != 1:
        raise ValueError('selected case export requires one fresh opaque inference identity')
    return _export_rcabench_live_case(
        source=source, destination=destination, opaque=manifest.records[0].opaque_case_id,
        expected_inference_artifact_sha256=expected_inference_artifact_sha256,
    )


def _export_rcabench_live_case(
    *, source: Path, destination: Path, opaque: str, expected_inference_artifact_sha256: str,
) -> StagedCase:
    """Shared unchanged whole-case column/time projection, after explicit admission."""
    target = destination / opaque
    if target.exists() or target.is_symlink():
        raise ValueError('immutable staged case already exists')
    paths = [require_plain_path(source / 'env.json')]
    for name in TELEMETRY_FILES:
        path = source / name
        if path.exists() or path.is_symlink():
            paths.append(require_plain_path(path))
    # Retain the exact original source pins without retaining answer-bearing names.
    source_pins = {path.name: file_sha256(path) for path in paths}
    env_data, _, _, _ = _read_env_contract(source / 'env.json')
    env = json.loads(env_data)
    safe_env = {name: env[name] if name == 'TIMEZONE' else int(env[name]) for name in ENV_FIELDS}
    # A failed projection remains an explicitly incomplete directory, never a sealed case.
    temporary = Path(tempfile.mkdtemp(prefix='.incomplete-', dir=destination))
    _write_private(temporary / 'env.json', canonical_json_bytes(safe_env))
    files = [StagedFile(name='env.json', source_sha256=source_pins['env.json'],
                        export_sha256=file_sha256(temporary / 'env.json'),
                        size_bytes=(temporary / 'env.json').stat().st_size)]
    for path in paths[1:]:
        kind = path.name.split('_', 1)[1].removesuffix('.parquet')
        parquet = pq.ParquetFile(path)
        columns = [name for name in ('time', *_ENTITY_COLUMNS, *_VALUE_COLUMNS[kind])
                   if name in parquet.schema_arrow.names]
        # Row counts/order, nulls, values and all modalities are retained. The
        # unchanged adapter enforces B/F and future bounds on consumption.
        schema = pa.schema([pa.field(name, parquet.schema_arrow.field(name).type,
                                     nullable=parquet.schema_arrow.field(name).nullable)
                            for name in columns])
        output = temporary / path.name
        # Exclusive 0600 creation before ParquetWriter opens its file descriptor.
        _write_private(output, b'')
        rows = 0
        with pq.ParquetWriter(output, schema) as writer:
            for batch in parquet.iter_batches(batch_size=65536, columns=columns):
                clean = pa.RecordBatch.from_arrays(batch.columns, schema=schema)
                writer.write_batch(clean)
                rows += batch.num_rows
        with output.open('rb') as stream:
            os.fsync(stream.fileno())
        if rows != parquet.metadata.num_rows:
            raise ValueError('projection changed whole-case row count')
        files.append(StagedFile(name=path.name, source_sha256=source_pins[path.name],
                                export_sha256=file_sha256(output), size_bytes=output.stat().st_size,
                                rows=rows))
    for path in paths:
        if require_plain_path(path) != path or file_sha256(path) != source_pins[path.name]:
            raise ValueError('original source changed during projection')
    current = {name for name in TELEMETRY_FILES if (source / name).exists()}
    if current != set(source_pins) - {'env.json'}:
        raise ValueError('whole-case source inventory changed during projection')
    body = dict(schema_version='nurse.v2.live-staged-case/1', dataset='rcabench',
                opaque_case_id=opaque, inference_artifact_sha256=expected_inference_artifact_sha256,
                files=[item.model_dump(mode='json') for item in sorted(files, key=lambda item: item.name)],
                missing_source_files=sorted(set(TELEMETRY_FILES) - set(source_pins)))
    result = StagedCase.model_validate({**body, 'stage_sha256': sha256_hex(canonical_json_bytes(body))})
    if not result.verify():
        raise ValueError('invalid stage receipt')
    _write_private(temporary / 'stage.json', canonical_json_bytes(result.model_dump(mode='json')))
    # Run ownership is single-writer; rename never replaces a populated prior stage.
    if target.exists() or target.is_symlink():
        raise ValueError('immutable staged case already exists')
    temporary.rename(target)
    if os.name == 'posix':
        descriptor = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'evaluator-map-path', 'inference-manifest-path', 'stage-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('expected-evaluator-artifact-sha256', 'expected-inference-artifact-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    try:
        result = provision_rcabench_live_case(**vars(args))
    except Exception as error:
        # Do not expose private filenames, parsed records or values in CLI errors.
        print(json.dumps({'stage': 'provision_failed', 'error_type': type(error).__name__}), flush=True)
        return 1
    print(canonical_json_bytes(result.model_dump(mode='json')).decode(), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
