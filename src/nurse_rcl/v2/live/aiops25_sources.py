"""Private, decision-bounded AIOps25 prepared sources; never a model payload.

The adapter's raw service directory and these Arrow tables remain inside the
ingestion boundary. Only a downstream opaque projection may reach an agent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.aiops25_adapter import (
    AIOps25AdapterConfiguration,
    AIOps25AdapterResult,
    _adapt_aiops25_case_inputs,
    adapt_aiops25_case,
)
from nurse_rcl.v2.aiops25_inventory import (
    AIOPS25_PARQUET_FILES,
    AuthorizedAIOps25Queue,
)
from nurse_rcl.v2.live.sources import LiveSourceTable, file_sha256

_COLUMNS = {
    'metrics': ('metric', 'value'),
    'logs': ('trace_id', 'span_id', 'level', 'message'),
    'traces': ('trace_id', 'span_id', 'parent_span_id', 'span_name', 'duration',
               'attr.status_code', 'attr.http.response.status_code',
               'attr.http.request.method'),
}
_REQUIRED = {
    'metrics': ('metric', 'value'), 'logs': ('level', 'message'),
    'traces': ('trace_id', 'span_id', 'parent_span_id', 'duration'),
}
_FILES = ('.prepared.json', 'env.json', *AIOPS25_PARQUET_FILES)


def _snapshot(root: Path) -> tuple[tuple[str, str | None], ...]:
    if root.is_symlink():
        raise ValueError('case symlink is not a declared source')
    result = []
    for name in _FILES:
        path = root / name
        if path.is_symlink():
            raise ValueError('telemetry symlink is not a declared source')
        result.append((name, file_sha256(path) if path.is_file() else None))
    return tuple(result)


@dataclass(frozen=True)
class AIOps25LiveSources:
    adapter: AIOps25AdapterResult
    tables: tuple[LiveSourceTable, ...] = field(repr=False)
    rejections: tuple[tuple[str, str], ...]
    source_case_id: str = field(repr=False)
    authorized_queue: AuthorizedAIOps25Queue = field(repr=False)
    source_files: tuple[tuple[str, str | None], ...]

    @property
    def observed_service_names(self) -> tuple[str, ...]:
        return tuple(sorted({
            value for item in self.tables if 'service_name' in item.table.column_names
            for value in item.table.column('service_name').unique().to_pylist()
            if isinstance(value, str) and value.strip()
        }))

    def descriptor(self) -> dict[str, Any]:
        """Private replay identity, not an inference-safe public projection."""
        body = dict(
            schema_version='nurse.v2.aiops25-live-sources/1',
            adapter=self.adapter.model_dump(mode='json'),
            queue_sha256=self.authorized_queue.artifact_hash,
            source_files=self.source_files,
            tables=[item.descriptor() for item in self.tables],
            rejections=self.rejections,
        )
        return {**body, 'descriptor_sha256': sha256_hex(canonical_json_bytes(body))}

    def verify_sources(self, root: Path) -> bool:
        try:
            rebuilt = read_aiops25_live_sources(
                root, source_case_id=self.source_case_id,
                opaque_case_id=self.adapter.opaque_case_id,
                authorized_queue=self.authorized_queue,
                configuration=self.adapter.configuration,
            )
            return rebuilt.descriptor() == self.descriptor()
        except (OSError, ValueError, pa.ArrowException):
            return False


def read_aiops25_live_sources(
    root: Path, *, source_case_id: str, opaque_case_id: str,
    authorized_queue: AuthorizedAIOps25Queue,
    configuration: AIOps25AdapterConfiguration,
) -> AIOps25LiveSources:
    """Read all six available sources, with immutable file and queue checks."""
    return _read_aiops25_source_inputs(
        root, source_case_id=source_case_id, opaque_case_id=opaque_case_id,
        authorized_queue=authorized_queue, configuration=configuration,
    )


def _read_aiops25_source_inputs(
    root: Path, *, source_case_id: str, opaque_case_id: str,
    authorized_queue: AuthorizedAIOps25Queue, configuration: AIOps25AdapterConfiguration,
    input_bytes: Mapping[str, bytes] | None = None,
) -> AIOps25LiveSources:
    """Shared projection; normal ingress supplies the actual pinned bytes once."""
    if (type(authorized_queue) is not AuthorizedAIOps25Queue
            or not authorized_queue.verify()
            or authorized_queue.destination != 'inference'
            or source_case_id not in authorized_queue.case_ids):
        raise ValueError('verified authorized inference queue is required')
    original_root = Path(root)
    if original_root.is_symlink():
        raise ValueError('case symlink is not a declared source')
    source_root = original_root.resolve(strict=True)
    before = (_snapshot(source_root) if input_bytes is None else tuple(
        (name, sha256_hex(input_bytes[name]) if name in input_bytes else None) for name in _FILES))
    queue_before = canonical_json_bytes(authorized_queue.model_dump(mode='json'))
    config_before = canonical_json_bytes(configuration.model_dump(mode='json'))
    try:
        adapter = adapt_aiops25_case(
            source_root, source_case_id=source_case_id, opaque_case_id=opaque_case_id,
            authorized_queue=authorized_queue, configuration=configuration,
        ) if input_bytes is None else _adapt_aiops25_case_inputs(
            source_root, source_case_id=source_case_id, opaque_case_id=opaque_case_id,
            authorized_queue=authorized_queue, configuration=configuration, input_bytes=input_bytes,
        )
        tables = []
        rejections: list[tuple[str, str]] = [
            (item.source_file, item.reason) for item in adapter.source_rejections]
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
            columns = [name for name in ('time', 'service_name', *_COLUMNS[audit.source_kind])
                       if name in fields]
            projected = parquet.read(columns=columns)
            timestamps = (
                projected.column('time').cast(pa.timestamp('ns', tz='UTC')).cast(pa.int64())
            )
            interval = (adapter.time_contract.baseline_window if audit.window == 'baseline'
                        else adapter.time_contract.fault_window)
            table = projected.filter(pc.and_(
                pc.greater_equal(timestamps, interval.start_unix_ns),
                pc.less(timestamps, interval.end_unix_ns),
            ))
            if table.num_rows != audit.accepted_rows or (
                    input_bytes is None and file_sha256(path) != audit.source_sha256):
                raise ValueError('telemetry projection changed from source audit')
            missing = tuple(sorted(set(_REQUIRED[audit.source_kind]) - set(fields)))
            tables.append(LiveSourceTable(audit, table, missing))
        return AIOps25LiveSources(adapter, tuple(tables), tuple(sorted(rejections)),
                                 source_case_id, authorized_queue, before)
    finally:
        if ((input_bytes is None and _snapshot(source_root) != before)
                or original_root.is_symlink()
                or canonical_json_bytes(authorized_queue.model_dump(mode='json')) != queue_before
                or canonical_json_bytes(configuration.model_dump(mode='json')) != config_before):
            raise ValueError('AIOps25 sources or authorization changed during read')
