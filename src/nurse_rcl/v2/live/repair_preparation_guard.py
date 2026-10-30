"""Private lifetime guards for code-owned whole-case assembly, not trust tokens.

Keep the existing per-series and expanded-projection capacities. The complete
error-source bundle still passes its own whole-case gate before consumption.
Only compact snapshot digests are retained; every check recomputes actual values.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import fields
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]

from nurse_rcl.artifacts import sha256_hex
from nurse_rcl.v2.aiops25_adapter import AIOps25AdapterResult
from nurse_rcl.v2.aiops25_inventory import AuthorizedAIOps25Queue
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.dynamic_verification.safety import snapshot as source_snapshot
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.focus_selection import FocusSelectionResult
from nurse_rcl.v2.live.aiops25_sources import AIOps25LiveSources
from nurse_rcl.v2.live.assembly import (
    AIOps25LiveInputs,
    LiveInputs,
    LiveSignal,
)
from nurse_rcl.v2.live.repair_assembly import RepairDiagnosticInputs, RepairSignalDiagnostic
from nurse_rcl.v2.live.repair_errors import ErrorObservation
from nurse_rcl.v2.live.repair_safety import snapshot
from nurse_rcl.v2.live.repair_sources import _INPUT_ROOTS, _inputs
from nurse_rcl.v2.live.sources import LiveSourceTable, RCABenchLiveSources, _table_sha256
from nurse_rcl.v2.observation_schemas import CandidateUniverse, TimeInterval
from nurse_rcl.v2.rcabench_adapter import RCABenchAdapterResult, RCABenchSourceAudit
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

_SOURCE_ROOTS = (
    RCABenchAdapterResult,
    AIOps25AdapterResult,
    AuthorizedAIOps25Queue,
    RCABenchSourceAudit,
)
_CONTEXT_ROOTS = (
    CandidateUniverse,
    FrozenCandidateUniverse,
    EventGroupingResult,
    FocusSelectionResult,
    TemporalEntityRegistry,
    TimeInterval,
)


def _fields(value: Any, expected: type) -> tuple[tuple[str, Any], ...]:
    names = tuple(f.name for f in fields(expected))
    if type(value) is not expected or tuple(object.__getattribute__(value, "__dict__")) != names:
        raise ValueError("repair assembly content or representation changed")
    return tuple((name, object.__getattribute__(value, name)) for name in names)


def _sources_value(source: RCABenchLiveSources | AIOps25LiveSources) -> object:
    cls = RCABenchLiveSources if type(source) is RCABenchLiveSources else AIOps25LiveSources
    values = _fields(source, cls)
    if type(source.tables) is not tuple:
        raise ValueError("repair source table representation changed")
    tables = []
    for item in source.tables:
        _fields(item, LiveSourceTable)
        table = item.table
        if type(table) is not pa.Table:
            raise ValueError("repair Arrow table representation changed")
        tables.append(
            (
                id(item),
                item.audit,
                item.missing_columns,
                id(table),
                _table_sha256(table),
                sha256_hex(table.schema.serialize().to_pybytes()),
                tuple(tuple(len(chunk) for chunk in column.chunks) for column in table.columns),
            )
        )
    return tuple((name, tuple(tables) if name == "tables" else value) for name, value in values)


def _protect(
    guards: ExitStack,
    current: Callable[[], object],
    roots: tuple[type, ...],
    *,
    per_source: bool = False,
) -> Callable[[], None]:
    capture = source_snapshot if per_source else snapshot

    def state() -> tuple[str, str]:
        raw, representation = capture(current(), roots)
        return sha256_hex(raw), representation

    before = state()

    def unchanged() -> None:
        try:
            same = state() == before
        except (AttributeError, KeyError, TypeError, ValueError):
            same = False
        if not same:
            raise ValueError("repair assembly content or representation changed")

    guards.callback(unchanged)
    return unchanged


def _check_all(callbacks: tuple[Callable[[], None], ...]) -> None:
    """Attempt every established responsibility, including exceptional exits."""
    with ExitStack() as checks:
        for check in callbacks:
            checks.callback(check)


@contextmanager
def _guard_live_sources(
    source: RCABenchLiveSources | AIOps25LiveSources,
) -> Iterator[Callable[[], None]]:
    with ExitStack() as guards:
        check = _protect(guards, lambda: _sources_value(source), _SOURCE_ROOTS)
        yield check


@contextmanager
def _guard_repair_assembly(data: RepairDiagnosticInputs) -> Iterator[Callable[[], None]]:
    with ExitStack() as guards:
        callbacks: list[Callable[[], None]] = []
        legacy = data.legacy
        cls = LiveInputs if data.dataset == "rcabench" else AIOps25LiveInputs

        def context() -> object:
            _fields(data, RepairDiagnosticInputs)
            values = _fields(legacy, cls)
            if any(type(v) is not tuple for v in (data.signals, data.errors, legacy.signals)):
                raise ValueError("repair assembly collection representation changed")
            return (
                data.dataset,
                id(data.legacy),
                id(legacy.sources),
                tuple(id(s) for s in legacy.signals),
                tuple(id(s) for s in data.signals),
                tuple(id(e) for e in data.errors),
                tuple((k, v) for k, v in values if k not in {"sources", "signals"}),
            )

        callbacks.append(_protect(guards, context, _CONTEXT_ROOTS))
        repairs = {r.signal_id: r for r in data.signals}
        if len(repairs) != len(data.signals) or set(repairs) != {
            s.signal_id for s in legacy.signals
        }:
            raise ValueError("repair assembly source collection differs")
        for signal in legacy.signals:
            repair = repairs[signal.signal_id]

            def series(s: LiveSignal = signal, r: RepairSignalDiagnostic = repair) -> object:
                _fields(s, LiveSignal)
                _fields(r, RepairSignalDiagnostic)
                return _inputs(s, r, legacy.sources.adapter.time_contract, data.dataset)

            callbacks.append(_protect(guards, series, _INPUT_ROOTS, per_source=True))
        # These are transient expanded projections, not four separate source bundles.
        for error in data.errors:
            def projection(value: ErrorObservation = error) -> object:
                return value

            callbacks.append(_protect(guards, projection, (ErrorObservation,)))
        retained = tuple(callbacks)
        yield lambda: _check_all(retained)
