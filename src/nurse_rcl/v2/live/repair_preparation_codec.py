"""Finite, explicit JSON/Arrow preparation encoding. Never deserialize objects."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, TypeVar

import pyarrow as pa  # type: ignore[import-untyped]
from pydantic import BaseModel

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.aiops25_adapter import AIOps25AdapterResult
from nurse_rcl.v2.aiops25_inventory import AuthorizedAIOps25Queue
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.focus_selection import FocusSelectionResult
from nurse_rcl.v2.live.aiops25_sources import AIOps25LiveSources
from nurse_rcl.v2.live.assembly import AIOps25LiveInputs, LiveInputs, LiveSignal
from nurse_rcl.v2.live.repair_assembly import RepairDiagnosticInputs, RepairSignalDiagnostic
from nurse_rcl.v2.live.repair_error_references import RepairErrorSources
from nurse_rcl.v2.live.repair_errors import ErrorObservation
from nurse_rcl.v2.live.repair_graph import RepairLiveGraph
from nurse_rcl.v2.live.repair_lifecycle import LifecycleObservation
from nurse_rcl.v2.live.repair_observations import RepairObservationQualityResult
from nurse_rcl.v2.live.repair_preparation_artifacts import (
    RepairPreparationData,
    RepairPreparationManifest,
)
from nurse_rcl.v2.live.repair_queries import RepairQueryCatalog
from nurse_rcl.v2.live.repair_safety import REPAIR_MAX_JSON_BYTES, REPAIR_MAX_NODES, canonical
from nurse_rcl.v2.live.repair_sources import RepairMeasurementSource
from nurse_rcl.v2.live.sources import LiveSourceTable, RCABenchLiveSources
from nurse_rcl.v2.live.staged_case import require_plain_path
from nurse_rcl.v2.observation_quality import ObservationQualityResult
from nurse_rcl.v2.observation_schemas import (
    CandidateUniverse,
    CaseTimeContract,
    Observation,
    TimeInterval,
)
from nurse_rcl.v2.rcabench_adapter import RCABenchAdapterResult, RCABenchSourceAudit
from nurse_rcl.v2.signal_transforms import ScalarWindowInput, SignalTransformResult
from nurse_rcl.v2.temporal_condensation import FrozenClockResolution
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

_MODEL = TypeVar("_MODEL", bound=BaseModel)
_ROOTS = (
    AIOps25AdapterResult,
    AuthorizedAIOps25Queue,
    FrozenCandidateUniverse,
    EventGroupingResult,
    FocusSelectionResult,
    ErrorObservation,
    RepairErrorSources,
    RepairLiveGraph,
    LifecycleObservation,
    RepairObservationQualityResult,
    RepairQueryCatalog,
    RepairMeasurementSource,
    ObservationQualityResult,
    CandidateUniverse,
    Observation,
    TimeInterval,
    CaseTimeContract,
    RCABenchAdapterResult,
    RCABenchSourceAudit,
    ScalarWindowInput,
    SignalTransformResult,
    FrozenClockResolution,
    TemporalEntityRegistry,
)


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


@contextmanager
def _member_stream(path: Path, *, limit: int | None = None) -> Iterator[BinaryIO]:
    path = require_plain_path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or (limit is not None and before.st_size > limit):
        raise ValueError("preparation member is not a bounded regular file")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode) or _file_identity(opened) != _file_identity(before):
            raise ValueError("preparation member identity changed")
        try:
            yield handle
        finally:
            after = os.fstat(handle.fileno())
            current = require_plain_path(path).stat()
            if _file_identity(opened) != _file_identity(after) or _file_identity(
                after
            ) != _file_identity(current):
                raise ValueError("preparation member changed during read")


def read_member(path: Path, *, limit: int) -> bytes:
    with _member_stream(path, limit=limit) as handle:
        value = handle.read(limit + 1)
        if len(value) > limit:
            raise ValueError("preparation member exceeds read capacity")
    return value


def member_digest(path: Path) -> tuple[int, str]:
    digest, size = hashlib.sha256(), 0
    with _member_stream(path) as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _save(root: Path, name: str, value: object) -> None:
    path = root / name
    path.parent.mkdir(mode=0o700, exist_ok=True)
    raw = canonical(value, _ROOTS)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _read(root: Path, name: str) -> dict[str, Any]:
    raw = read_member(root / name, limit=REPAIR_MAX_JSON_BYTES)
    return _decode_json_member(raw)


def _decode_json_member(raw: bytes) -> dict[str, Any]:
    # This is a private whole-case archive, not an LLM exchange. Preserve the
    # small exchange limits; lexically bound this registered preparation root
    # before the JSON parser allocates its tree. Full exact-type/node/atom
    # preflight still runs below before any typed restoration.
    quoted = escaped = False
    depth = nodes = 0
    for char in raw:
        if quoted:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                quoted = False
        elif char == 34:
            quoted = True
        elif char in (123, 91):
            depth += 1
            nodes += 1
        elif char in (125, 93):
            depth -= 1
        elif char == 44:
            nodes += 1
        if depth > 64 or nodes > REPAIR_MAX_NODES:
            raise ValueError("preparation JSON exceeds resource capacity")
    value = json.loads(raw)
    if type(value) is not dict or raw != canonical(value, _ROOTS):
        raise ValueError("preparation JSON member is not canonical")
    return value


def _shape(row: dict[str, Any], names: tuple[str, ...]) -> None:
    if type(row) is not dict or set(row) != set(names):
        raise ValueError("preparation member fields differ")


def _model(kind: type[_MODEL], value: object) -> _MODEL:
    # The kind is a fixed code constant at each call, never selected from data.
    return kind.model_validate_json(canonical_json_bytes(value))


def _json_members(data: RepairPreparationData) -> Iterator[tuple[str, object]]:
    diagnostics, legacy = data.diagnostics, data.diagnostics.legacy
    source = legacy.sources
    yield (
        "core.json",
        dict(
            dataset=diagnostics.dataset,
            adapter=source.adapter,
            rejections=source.rejections,
            source_case_id=source.source_case_id if type(source) is AIOps25LiveSources else None,
            authorized_queue=source.authorized_queue
            if type(source) is AIOps25LiveSources
            else None,
            source_files=source.source_files if type(source) is AIOps25LiveSources else None,
            entity_registry=legacy.entity_registry,
            event_grouping=legacy.event_grouping,
            candidate_universe=legacy.candidate_universe,
            frozen_candidates=legacy.frozen_candidates,
            focus=legacy.focus,
            scope=legacy.scope,
            source_accounting=legacy.source_accounting,
            q_base=legacy.q_base,
            q_max=legacy.q_max,
            table_count=len(source.tables),
            signal_count=len(legacy.signals),
            diagnostic_count=len(diagnostics.signals),
            error_count=len(diagnostics.errors),
            measurement_source_count=len(data.sources),
            error_observation_sha256s=data.error_observation_sha256s,
            source_descriptor_sha256=data.source_descriptor_sha256,
            source_files_sha256=data.source_files_sha256,
            candidate_freeze_sha256=data.candidate_freeze_sha256,
        ),
    )
    for i, item in enumerate(source.tables):
        yield (
            f"tables/{i:06d}.json",
            dict(
                audit=item.audit,
                missing_columns=item.missing_columns,
            ),
        )
    for i, signal in enumerate(legacy.signals):
        yield (
            f"signals/{i:06d}.json",
            dict(
                signal_id=signal.signal_id,
                service=signal.service,
                observed_entity_key=signal.observed_entity_key,
                modality=signal.modality,
                signal_family=signal.signal_family,
                component=signal.component,
                baseline=signal.baseline,
                fault=signal.fault,
                transform=signal.transform,
                quality=signal.quality,
                observation=signal.observation,
                source_file_sha256s=signal.source_file_sha256s,
                flags=signal.flags,
            ),
        )
    for i, repair in enumerate(diagnostics.signals):
        yield (
            f"diagnostics/{i:06d}.json",
            dict(
                signal_id=repair.signal_id,
                metric_name=repair.metric_name,
                quality=repair.quality,
                lifecycle=repair.lifecycle,
                lifecycle_unavailable_reason=repair.lifecycle_unavailable_reason,
            ),
        )
    for i, error in enumerate(diagnostics.errors):
        yield f"errors/{i:06d}.json", error
    for i, value in enumerate(data.sources):
        yield f"sources/{i:06d}.json", value
    for name, model in (
        ("graph", data.graph),
        ("clock", data.clock),
        ("catalog", data.catalog),
        ("error-sources", data.error_sources),
    ):
        yield name + ".json", model


def write_preparation_data(root: Path, data: RepairPreparationData) -> None:
    for name, value in _json_members(data):
        _save(root, name, value)
    (root / "tables").mkdir(mode=0o700, exist_ok=True)
    for i, item in enumerate(data.diagnostics.legacy.sources.tables):
        fd = os.open(
            root / f"tables/{i:06d}.arrow",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(fd, "wb") as handle:
            with pa.ipc.new_stream(handle, item.table.schema) as writer:
                writer.write_table(item.table)
            handle.flush()
            os.fsync(handle.fileno())


class _ArrowDigest(io.RawIOBase):
    """Streaming Arrow fingerprint only; never retain a second complete table."""

    def __init__(self) -> None:
        self.digest = hashlib.sha256()
        self.size = 0

    def writable(self) -> bool:
        return True

    def write(self, value: Any) -> int:
        self.digest.update(value)
        self.size += len(value)
        return len(value)


def require_serialized_preparation(
    data: RepairPreparationData, manifest: RepairPreparationManifest
) -> None:
    """Bind values to pinned members before establishing a fresh method baseline.

    Re-encoding typed fixed data is not re-assembling telemetry or generating
    a graph/catalog. Each JSON root retains its original capacity boundary.
    """
    pinned = {item.name: (item.size_bytes, item.sha256) for item in manifest.members}
    restored = set()
    for name, value in _json_members(data):
        encoded = canonical(value, _ROOTS)
        if pinned.get(name) != (len(encoded), hashlib.sha256(encoded).hexdigest()):
            raise ValueError("preparation serialized member payload differs from independent pin")
        restored.add(name)
    for i, item in enumerate(data.diagnostics.legacy.sources.tables):
        stream = _ArrowDigest()
        with pa.ipc.new_stream(stream, item.table.schema) as writer:
            writer.write_table(item.table)
        name = f"tables/{i:06d}.arrow"
        if pinned.get(name) != (stream.size, stream.digest.hexdigest()):
            raise ValueError("preparation serialized Arrow member differs from independent pin")
        restored.add(name)
    if restored != {name for name in pinned if not name.startswith("blocks/")}:
        raise ValueError("preparation serialized member collection differs")


def _decode_preparation_data(
    read_json: Callable[[str], dict[str, Any]], read_arrow: Callable[[str], Any]
) -> RepairPreparationData:
    """Shared finite decoder; readers are internal code, never deserialized callbacks."""
    core = read_json("core.json")
    _shape(
        core,
        (
            "dataset",
            "adapter",
            "rejections",
            "source_case_id",
            "authorized_queue",
            "source_files",
            "entity_registry",
            "event_grouping",
            "candidate_universe",
            "frozen_candidates",
            "focus",
            "scope",
            "source_accounting",
            "q_base",
            "q_max",
            "table_count",
            "signal_count",
            "diagnostic_count",
            "error_count",
            "measurement_source_count",
            "error_observation_sha256s",
            "source_descriptor_sha256",
            "source_files_sha256",
            "candidate_freeze_sha256",
        ),
    )
    if core["dataset"] not in ("rcabench", "aiops25"):
        raise ValueError("preparation dataset differs")
    for name in (
        "table_count",
        "signal_count",
        "diagnostic_count",
        "error_count",
        "measurement_source_count",
    ):
        if type(core[name]) is not int or not 0 <= core[name] <= REPAIR_MAX_NODES:
            raise ValueError("preparation collection count exceeds capacity")
    tables = []
    for i in range(core["table_count"]):
        row = read_json(f"tables/{i:06d}.json")
        _shape(row, ("audit", "missing_columns"))
        table = read_arrow(f"tables/{i:06d}.arrow")
        tables.append(
            LiveSourceTable(
                audit=_model(RCABenchSourceAudit, row["audit"]),
                table=table,
                missing_columns=tuple(row["missing_columns"]),
            )
        )
    source: RCABenchLiveSources | AIOps25LiveSources
    if core["dataset"] == "rcabench":
        if any(core[n] is not None for n in ("source_case_id", "authorized_queue", "source_files")):
            raise ValueError("RCABench preparation carries foreign dataset fields")
        source = RCABenchLiveSources(
            adapter=_model(RCABenchAdapterResult, core["adapter"]),
            tables=tuple(tables),
            rejections=tuple(tuple(v) for v in core["rejections"]),
        )
    else:
        source = AIOps25LiveSources(
            adapter=_model(AIOps25AdapterResult, core["adapter"]),
            tables=tuple(tables),
            rejections=tuple(tuple(v) for v in core["rejections"]),
            source_case_id=core["source_case_id"],
            authorized_queue=_model(AuthorizedAIOps25Queue, core["authorized_queue"]),
            source_files=tuple(tuple(v) for v in core["source_files"]),
        )
    signals = []
    for i in range(core["signal_count"]):
        row = read_json(f"signals/{i:06d}.json")
        _shape(
            row,
            (
                "signal_id",
                "service",
                "observed_entity_key",
                "modality",
                "signal_family",
                "component",
                "baseline",
                "fault",
                "transform",
                "quality",
                "observation",
                "source_file_sha256s",
                "flags",
            ),
        )
        signals.append(
            LiveSignal(
                signal_id=row["signal_id"],
                service=row["service"],
                observed_entity_key=row["observed_entity_key"],
                modality=row["modality"],
                signal_family=row["signal_family"],
                component=row["component"],
                baseline=_model(ScalarWindowInput, row["baseline"]),
                fault=_model(ScalarWindowInput, row["fault"]),
                transform=_model(SignalTransformResult, row["transform"]),
                quality=_model(ObservationQualityResult, row["quality"]),
                observation=_model(Observation, row["observation"]),
                source_file_sha256s=tuple(row["source_file_sha256s"]),
                flags=tuple(row["flags"]),
            )
        )
    cls = LiveInputs if core["dataset"] == "rcabench" else AIOps25LiveInputs
    legacy = cls(
        sources=source,
        signals=tuple(signals),
        entity_registry=_model(TemporalEntityRegistry, core["entity_registry"]),
        event_grouping=_model(EventGroupingResult, core["event_grouping"]),
        candidate_universe=_model(CandidateUniverse, core["candidate_universe"]),
        frozen_candidates=_model(FrozenCandidateUniverse, core["frozen_candidates"]),
        focus=_model(FocusSelectionResult, core["focus"]),
        scope=_model(TimeInterval, core["scope"]),
        source_accounting=core["source_accounting"],
        q_base=core["q_base"],
        q_max=core["q_max"],
    )
    repairs = []
    for i in range(core["diagnostic_count"]):
        row = read_json(f"diagnostics/{i:06d}.json")
        _shape(
            row,
            ("signal_id", "metric_name", "quality", "lifecycle", "lifecycle_unavailable_reason"),
        )
        repairs.append(
            RepairSignalDiagnostic(
                signal_id=row["signal_id"],
                metric_name=row["metric_name"],
                quality=_model(RepairObservationQualityResult, row["quality"]),
                lifecycle=None
                if row["lifecycle"] is None
                else _model(LifecycleObservation, row["lifecycle"]),
                lifecycle_unavailable_reason=row["lifecycle_unavailable_reason"],
            )
        )
    diagnostics = RepairDiagnosticInputs(
        dataset=core["dataset"],
        legacy=legacy,
        signals=tuple(repairs),
        errors=tuple(
            _model(ErrorObservation, read_json(f"errors/{i:06d}.json"))
            for i in range(core["error_count"])
        ),
    )
    return RepairPreparationData(
        diagnostics=diagnostics,
        sources=tuple(
            _model(RepairMeasurementSource, read_json(f"sources/{i:06d}.json"))
            for i in range(core["measurement_source_count"])
        ),
        catalog=_model(RepairQueryCatalog, read_json("catalog.json")),
        graph=_model(RepairLiveGraph, read_json("graph.json")),
        clock=_model(FrozenClockResolution, read_json("clock.json")),
        error_sources=_model(RepairErrorSources, read_json("error-sources.json")),
        error_observation_sha256s=tuple(core["error_observation_sha256s"]),
        source_descriptor_sha256=core["source_descriptor_sha256"],
        source_files_sha256=core["source_files_sha256"],
        candidate_freeze_sha256=core["candidate_freeze_sha256"],
    )


def read_preparation_data(root: Path) -> RepairPreparationData:
    """Legacy decoder; its caller retains complete member replay and exit guards."""

    def arrow(name: str) -> Any:
        with _member_stream(root / name) as handle, pa.ipc.open_stream(handle) as reader:
            return reader.read_all()

    return _decode_preparation_data(lambda name: _read(root, name), arrow)
