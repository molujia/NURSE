"""Whole-case real source reading, before numerical assembly or model projection.

Tables are private adapter inputs, NOT model payloads. Free text and raw entity
identities must pass the downstream typed projection before any external use.
Only the descriptor is safe for progress logs. No evaluation files are opened.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.rcabench_adapter import (
    RCABenchAdapterConfiguration,
    RCABenchAdapterResult,
    RCABenchSourceAudit,
    _adapt_rcabench_case_inputs,
    adapt_rcabench_case,
)

_ENTITY_COLUMNS = (
    'service_name', 'attr.k8s.node.name', 'attr.k8s.namespace.name',
    'attr.k8s.statefulset.name', 'attr.k8s.deployment.name',
    'attr.k8s.replicaset.name', 'attr.k8s.pod.name', 'attr.k8s.container.name',
    'attr.k8s.service.name', 'attr.destination_workload', 'attr.source_workload',
    'attr.destination', 'attr.source',
)
_VALUE_COLUMNS = {
    'metrics': ('metric', 'value'),
    'metrics_sum': ('metric', 'value'),
    # These exports contain summaries, NOT explicit histogram buckets.
    'metrics_histogram': ('metric', 'count', 'sum', 'min', 'max',
                          'attr.jvm.gc.action', 'attr.jvm.gc.name'),
    'logs': ('trace_id', 'span_id', 'level', 'message'),
    'traces': ('trace_id', 'span_id', 'parent_span_id', 'span_name',
               'attr.span_kind', 'duration', 'attr.status_code',
               'attr.http.response.status_code', 'attr.http.request.method'),
}
_REQUIRED = {
    'metrics': ('metric', 'value'), 'metrics_sum': ('metric', 'value'),
    'metrics_histogram': ('metric', 'count', 'sum', 'min', 'max'),
    'logs': ('level', 'message'),
    'traces': ('trace_id', 'span_id', 'parent_span_id', 'duration'),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _table_sha256(table: pa.Table) -> str:
    # Exact ordered column values and types, including nulls/nonfinite values,
    # use Arrow IPC instead of JSON conversion of raw telemetry.
    sink = pa.BufferOutputStream()
    normalized = table.combine_chunks().replace_schema_metadata(None)
    with pa.ipc.new_stream(sink, normalized.schema) as writer:
        writer.write_table(normalized)
    return sha256_hex(sink.getvalue().to_pybytes())


@dataclass(frozen=True)
class LiveSourceTable:
    audit: RCABenchSourceAudit
    table: pa.Table = field(repr=False)
    missing_columns: tuple[str, ...]

    def descriptor(self) -> dict[str, Any]:
        return dict(source_file=self.audit.source_file, source_sha256=self.audit.source_sha256,
                    window=self.audit.window, source_kind=self.audit.source_kind,
                    accepted_rows=self.table.num_rows, missing_columns=self.missing_columns,
                    projection_sha256=_table_sha256(self.table))


@dataclass(frozen=True)
class RCABenchLiveSources:
    adapter: RCABenchAdapterResult
    tables: tuple[LiveSourceTable, ...] = field(repr=False)
    rejections: tuple[tuple[str, str], ...]

    @property
    def observed_service_names(self) -> tuple[str, ...]:
        # Private raw identities only; never a model-facing candidate directory.
        return tuple(sorted({value for item in self.tables
                             if 'service_name' in item.table.column_names
                             for value in item.table.column('service_name').unique().to_pylist()
                             if isinstance(value, str) and value.strip()}))

    def descriptor(self) -> dict[str, Any]:
        body = dict(schema_version='nurse.v2.live-source-descriptor/1',
                    adapter=self.adapter.model_dump(mode='json'),
                    tables=[item.descriptor() for item in self.tables], rejections=self.rejections)
        return {**body, 'descriptor_sha256': sha256_hex(canonical_json_bytes(body))}

    def verify_sources(self, root: Path) -> bool:
        """Re-read actual source files, not only hashes embedded in an artifact."""
        try:
            rebuilt = read_rcabench_live_sources(root, opaque_case_id=self.adapter.opaque_case_id,
                                                 configuration=self.adapter.configuration)
            return rebuilt.descriptor() == self.descriptor()
        except (OSError, ValueError, pa.ArrowException):
            return False


def read_rcabench_live_sources(
    root: Path, *, opaque_case_id: str, configuration: RCABenchAdapterConfiguration,
) -> RCABenchLiveSources:
    """Read every available authorized modality, without sampling or C/Q cuts."""
    return _read_rcabench_source_inputs(
        root, opaque_case_id=opaque_case_id, configuration=configuration,
    )


def _read_rcabench_source_inputs(
    root: Path, *, opaque_case_id: str, configuration: RCABenchAdapterConfiguration,
    input_bytes: Mapping[str, bytes] | None = None,
) -> RCABenchLiveSources:
    """Same projection rules; private pinned bytes never re-open external telemetry."""
    source_root = Path(root)
    if source_root.is_symlink():
        raise ValueError('case symlink is not a declared source')
    source_root = source_root.resolve(strict=True)
    allowed = tuple(f'{window}_{kind}.parquet'
                    for window in ('normal', 'abnormal') for kind in _VALUE_COLUMNS)
    for name in ('env.json', *allowed):
        if (source_root / name).is_symlink():
            raise ValueError('telemetry symlink is not a declared source')
    adapter = (adapt_rcabench_case(source_root, opaque_case_id=opaque_case_id,
                                  configuration=configuration)
               if input_bytes is None else _adapt_rcabench_case_inputs(
                   source_root, opaque_case_id=opaque_case_id, configuration=configuration,
                   input_bytes=input_bytes))
    tables = []
    rejections = []
    for audit in adapter.files:
        if audit.status != 'valid':
            rejections.append((audit.source_file, str(audit.error_code)))
            continue
        path = source_root / audit.source_file
        if input_bytes is None and file_sha256(path) != audit.source_sha256:
            raise ValueError('telemetry source changed after adapter audit')
        parquet = pq.ParquetFile(
            path if input_bytes is None else pa.BufferReader(input_bytes[audit.source_file]))
        fields = parquet.schema_arrow.names
        columns = tuple(c for c in ('time', *_ENTITY_COLUMNS, *_VALUE_COLUMNS[audit.source_kind])
                        if c in fields)
        # Batch reading with an explicit projection prevents arbitrary columns
        # (labels, injected metadata) crossing the ingestion boundary.
        projected = parquet.read(columns=list(columns))
        timestamps = projected.column('time').cast(pa.timestamp('ns', tz='UTC')).cast(pa.int64())
        interval = (adapter.time_contract.baseline_window if audit.window == 'baseline'
                    else adapter.time_contract.fault_window)
        mask = pc.and_(pc.greater_equal(timestamps, interval.start_unix_ns),
                       pc.less(timestamps, interval.end_unix_ns))
        table = projected.filter(mask)
        if table.num_rows != audit.accepted_rows or (
                input_bytes is None and file_sha256(path) != audit.source_sha256):
            raise ValueError('telemetry projection differs from source audit')
        missing = tuple(sorted(set(_REQUIRED[audit.source_kind]) - set(fields)))
        tables.append(LiveSourceTable(audit, table, missing))
    if input_bytes is None and file_sha256(source_root / 'env.json') != adapter.env_sha256:
        raise ValueError('time contract changed during source read')
    return RCABenchLiveSources(adapter, tuple(tables), tuple(rejections))
