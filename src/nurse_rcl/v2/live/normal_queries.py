"""Single-worker incremental queries, with one bounded admission on recovery.

Only project-owned preparation enters this private runner. The journal is a
durability boundary, not permission to replay measurements or reconstruct sources.
"""

from __future__ import annotations

import fcntl
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.normal_artifacts import NormalQueryJournalEntry, NormalQuerySession
from nurse_rcl.v2.live.repair_driver import _fsync
from nurse_rcl.v2.live.repair_measurements import (
    REPAIR_MEASUREMENT_SEMANTICS_SHA256,
    RepairSignalMeasurement,
    _measure_values,
    _numeric_identity,
)
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData
from nurse_rcl.v2.live.repair_preparation_codec import _decode_json_member, read_member
from nurse_rcl.v2.live.repair_queries import (
    REPAIR_QUERY_SEMANTICS_SHA256,
    RepairQueryCatalog,
    RepairQueryFeedback,
    RepairQuerySelection,
    _feedback_values,
    _select_values,
)
from nurse_rcl.v2.live.repair_query_events import query_step
from nurse_rcl.v2.live.repair_query_session import RepairQueryCompletedStep
from nurse_rcl.v2.live.repair_safety import REPAIR_MAX_JSON_BYTES
from nurse_rcl.v2.live.repair_sources import RepairMeasurementSource
from nurse_rcl.v2.live.staged_case import require_plain_path
from nurse_rcl.v2.report_audit_json import MAX_REPORT_AUDIT_BYTES

_M = TypeVar("_M", bound=BaseModel)
_Phase = Literal["initial", "reserved", "measured", "completed", "stopped", "failed"]


def _seal(value: _M, field: str) -> _M:
    pin = sha256_hex(canonical_json_bytes(value.model_dump(mode="json", exclude={field})))
    return value.model_copy(update={field: pin})


def _read_bytes(path: Path, limit: int) -> bytes:
    # Do not allocate the full 256-MiB limit for each small journal entry.
    size = require_plain_path(path).stat().st_size
    if size > limit:
        raise ValueError("normal query record exceeds capacity")
    return read_member(path, limit=size)


def _publish(path: Path, entry: NormalQueryJournalEntry) -> None:
    raw = canonical_json_bytes(entry.model_dump(mode="json"))
    if len(raw) > REPAIR_MAX_JSON_BYTES:
        raise ValueError("normal query publication exceeds capacity")
    descriptor, name = tempfile.mkstemp(prefix=".query-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_write_bytes(path.parent / "head", canonical_json_bytes(
        [entry.sequence, entry.entry_sha256],
    ))
    _fsync(path.parent)


@dataclass(frozen=True)
class _State:
    count: int = 0
    previous: str | None = None
    steps: tuple[RepairQueryCompletedStep, ...] = ()
    submitted: tuple[str, ...] = ()
    pending: RepairQuerySelection | None = None
    measured: RepairSignalMeasurement | None = None
    stop: Any = None
    failure: Any = None


def _measurement_binding(
    measured: RepairSignalMeasurement, source: RepairMeasurementSource,
    catalog: RepairQueryCatalog,
) -> None:
    identity = _numeric_identity(source)
    if (
        measured.source_sha256 != source.source_sha256
        or measured.semantics_sha256 != REPAIR_MEASUREMENT_SEMANTICS_SHA256
        or measured.time_contract_sha256 != catalog.time_contract_sha256
        or measured.request_sha256 != identity.request_sha256
        or measured.canonical_series_key != identity.canonical_series_key
        or measured.interval != source.query_interval
        or measured.quality != source.quality
        or measured.lifecycle != source.lifecycle
    ):
        raise ValueError("normal query measurement source/window binding differs")


def _feedback_binding(
    feedback: RepairQueryFeedback, measured: RepairSignalMeasurement,
    source: RepairMeasurementSource, catalog: RepairQueryCatalog, request_key: str,
) -> None:
    request = next(r for r in catalog.requests if r.request_key == request_key)
    if (
        feedback.catalog_sha256 != catalog.catalog_sha256
        or feedback.semantics_sha256 != REPAIR_QUERY_SEMANTICS_SHA256
        or feedback.request_key != request_key
        or feedback.measurement_sha256 != measured.measurement_sha256
        or len(feedback.predicates) != len(request.needs)
        or len(feedback.onsets) != len(request.needs)
    ):
        raise ValueError("normal query feedback receipt binding differs")
    for predicate, onset, need in zip(
        feedback.predicates, feedback.onsets, request.needs, strict=True,
    ):
        if (
            predicate.source_sha256 != source.source_sha256
            or predicate.measurement_sha256 != measured.measurement_sha256
            or predicate.semantics_sha256 != REPAIR_MEASUREMENT_SEMANTICS_SHA256
            or predicate.candidate_service_key != request.candidate_service_key
            or predicate.observation_id != request.observation_id
            or predicate.mechanism_key != need.mechanism_key
            or predicate.predicate_kind != need.predicate_kind
            or onset.need_key != need.need_key
            or onset.predicate_sha256 != predicate.result_sha256
        ):
            raise ValueError("normal query predicate/onset reference differs")


class _Journal:
    def __init__(self, data: RepairPreparationData, root: Path):
        self.catalog, self.root = data.catalog, root
        self.requests = {r.request_key: r for r in self.catalog.requests}
        self.sources = {s.source_sha256: s for s in data.sources}
        self.state = _State()

    def source(self, selection: RepairQuerySelection) -> RepairMeasurementSource:
        request = self.requests.get(selection.selected_request_key or "")
        if request is None or request.source_sha256 not in self.sources:
            raise ValueError("normal query selection has no admitted source")
        return self.sources[request.source_sha256]

    def transition(self, entry: NormalQueryJournalEntry) -> _State:
        state, catalog = self.state, self.catalog
        if (
            entry.sequence != state.count or entry.previous_sha256 != state.previous
            or entry.catalog_sha256 != catalog.catalog_sha256
            or state.stop is not None or state.failure is not None
        ):
            raise ValueError("normal query journal sequence/identity/terminal mismatch")
        allowed = {
            "initial": (), "reserved": ("selection",), "measured": ("measurement",),
            "completed": ("feedback",), "stopped": ("selection",),
            "failed": ("failure_reason",),
        }[entry.phase]
        if any((getattr(entry, name) is not None) != (name in allowed)
               for name in ("selection", "measurement", "feedback", "failure_reason")):
            raise ValueError("normal query journal phase payload differs")
        changes: dict[str, Any] = {}
        if entry.phase == "initial":
            if state.count != 0:
                raise ValueError("normal query journal repeats initial state")
        elif state.count == 0:
            raise ValueError("normal query journal lacks initial identity")
        elif entry.phase in ("reserved", "stopped"):
            choice = entry.selection
            if choice is None or state.pending is not None or (
                choice.catalog_sha256 != catalog.catalog_sha256
                or choice.semantics_sha256 != REPAIR_QUERY_SEMANTICS_SHA256
                or choice.feedback_sha256s != tuple(s.feedback.result_sha256 for s in state.steps)
                or choice.submitted_request_keys != state.submitted
                or choice.paid_count != len(state.steps)
                or choice.remaining_budget != 8 - len(state.submitted)
                or tuple(r.request_key for r in choice.requests) != tuple(self.requests)
            ):
                raise ValueError("normal query choice prefix differs")
            selected = choice.selected_request_key
            if entry.phase == "reserved":
                if selected not in self.requests or selected in state.submitted or (
                    len(state.submitted) >= 8 or choice.stop_reason is not None
                ):
                    raise ValueError("normal query reservation duplicate/budget mismatch")
                row = next(r for r in choice.requests if r.request_key == selected)
                if not row.eligible or row.candidate_service_key != (
                    self.requests[selected].candidate_service_key
                ) or row.candidate_service_key is None:
                    raise ValueError("normal query reservation is not eligible")
                changes.update(
                    pending=choice, submitted=tuple(sorted((*state.submitted, selected))),
                )
            else:
                if selected is not None or choice.stop_reason is None or any(
                    r.eligible for r in choice.requests
                ) or (choice.stop_reason == "query_budget_exhausted") != (len(state.steps) == 8):
                    raise ValueError("normal query stop is inconsistent")
                changes["stop"] = choice.stop_reason
        elif entry.phase == "measured":
            if state.pending is None or state.measured is not None or entry.measurement is None:
                raise ValueError("normal query measurement lacks a unique reservation")
            _measurement_binding(entry.measurement, self.source(state.pending), catalog)
            changes["measured"] = entry.measurement
        elif entry.phase == "completed":
            if state.pending is None or state.measured is None or entry.feedback is None:
                raise ValueError("normal query completion lacks a measured reservation")
            _feedback_binding(entry.feedback, state.measured, self.source(state.pending),
                              catalog, state.pending.selected_request_key or "")
            changes.update(steps=(*state.steps, RepairQueryCompletedStep(
                selection=state.pending, measurement=state.measured, feedback=entry.feedback,
            )), pending=None, measured=None)
        else:
            if state.pending is None:
                raise ValueError("normal query failure lacks a reservation")
            changes["failure"] = entry.failure_reason
        return replace(state, **changes, count=state.count + 1, previous=entry.entry_sha256)

    def append(self, phase: _Phase, **payload: Any) -> None:
        entry = _seal(NormalQueryJournalEntry(
            sequence=self.state.count, previous_sha256=self.state.previous,
            catalog_sha256=self.catalog.catalog_sha256, phase=phase,
            entry_sha256="0" * 64, **payload,
        ), "entry_sha256")
        successor = self.transition(entry)
        with query_step("append_publish", sequence=entry.sequence, journal_phase=entry.phase):
            _publish(self.root / f"{entry.sequence:03d}.json", entry)
        self.state = successor

    def restore(self) -> None:
        files = tuple(sorted(self.root.glob("*.json")))
        if not 1 <= len(files) <= 27 or tuple(p.name for p in files) != tuple(
            f"{i:03d}.json" for i in range(len(files))
        ):
            raise ValueError("normal query journal is empty/noncontiguous/overcapacity")
        names = {p.name for p in self.root.iterdir()}
        if names - {"lock", "head", *(p.name for p in files)}:
            raise ValueError("normal query journal has unexpected members")
        tails = []
        for path in files:
            raw = _read_bytes(path, REPAIR_MAX_JSON_BYTES)
            _decode_json_member(raw)
            entry = NormalQueryJournalEntry.model_validate_json(raw)
            if raw != canonical_json_bytes(entry.model_dump(mode="json")) or (
                _seal(entry, "entry_sha256") != entry
            ):
                raise ValueError("normal query journal pin mismatch")
            # Check saved numerical receipts' identities, not recompute their values.
            for value, field in ((entry.selection, "result_sha256"),
                                 (entry.measurement, "measurement_sha256"),
                                 (entry.feedback, "result_sha256")):
                if value is not None and _seal(value, field) != value:
                    raise ValueError("normal query saved receipt pin mismatch")
            if entry.feedback is not None and any(
                _seal(p, "result_sha256") != p for p in entry.feedback.predicates
            ):
                raise ValueError("normal query saved predicate pin mismatch")
            self.state = self.transition(entry)
            tails.append(canonical_json_bytes([entry.sequence, entry.entry_sha256]))
        head = self.root / "head"
        if head.exists() or head.is_symlink():
            if _read_bytes(head, 256) not in tails[-2:]:
                raise ValueError("normal query head indicates rollback/corruption")
        elif len(files) != 1:
            raise ValueError("normal query head missing")

    def result(self) -> NormalQuerySession:
        state = self.state
        if state.previous is None:
            raise ValueError("normal query journal has no durable identity")
        return _seal(NormalQuerySession(
            semantics_sha256=REPAIR_QUERY_SEMANTICS_SHA256,
            catalog_sha256=self.catalog.catalog_sha256, journal_sha256=state.previous,
            steps=state.steps, submitted_request_keys=state.submitted,
            pending_selection=state.pending, pending_measurement=state.measured,
            paid_count=len(state.steps), stop_reason=state.stop, failure_reason=state.failure,
            session_sha256="0" * 64,
        ), "session_sha256")


def _consume_blocks(
    source: RepairMeasurementSource, data: RepairPreparationData,
    root: Path, admitted: set[str],
) -> None:
    signal = next(s for s in data.diagnostics.legacy.signals if s.signal_id == source.signal_id)
    for block in source.source_blocks:
        if block.sha256 in admitted:
            continue
        if block.size_bytes > MAX_REPORT_AUDIT_BYTES:
            raise ValueError("normal query block exceeds capacity")
        raw = _read_bytes(root / (block.sha256 + ".json"), block.size_bytes)
        if len(raw) != block.size_bytes or sha256_hex(raw) != block.sha256:
            raise ValueError("normal query source block digest/size differs")
        body = _decode_json_member(raw)
        window = signal.baseline if block.name == "baseline" else signal.fault
        if body != dict(schema_version=block.schema_version, inputs={block.name: dict(
            input_kind=block.input_kind, document=window.model_dump(mode="json"),
        )}):
            raise ValueError("normal query block differs from admitted fixed window")
        admitted.add(block.sha256)


def run_normal_queries(
    *, data: RepairPreparationData, ledger_root: Path, block_root: Path,
) -> NormalQuerySession:
    """Run once; recover durable receipts once, never repeat a paid measurement."""
    require_plain_path(ledger_root.parent)
    require_plain_path(block_root)
    try:
        ledger_root.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        created = False
    require_plain_path(ledger_root)
    lock = os.open(ledger_root / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = _Journal(data, ledger_root)
        if created:
            journal.append("initial")
        else:
            journal.restore()
        state = journal.state
        if state.failure is not None:
            raise ValueError("normal query session failed; retained for diagnosis")
        if state.pending is not None and state.measured is None:
            raise ValueError("normal query pending reservation without a measurement receipt")
        admitted: set[str] = set()
        for _ in range(9):
            state = journal.state
            if state.stop is not None:
                return journal.result()
            if state.pending is None:
                with query_step("choice"):
                    choice = _select_values(data.catalog, tuple(s.feedback for s in state.steps),
                                            state.submitted)
                if choice.selected_request_key is None:
                    journal.append("stopped", selection=choice)
                    return journal.result()
                journal.append("reserved", selection=choice)
            pending = journal.state.pending
            if pending is None:
                raise ValueError("normal query reservation absent")
            source = journal.source(pending)
            measured = journal.state.measured
            if measured is None:
                try:
                    with query_step("measurement"):
                        _consume_blocks(source, data, block_root, admitted)
                        measured = _measure_values(source)
                except Exception:
                    journal.append("failed", failure_reason="measurement_integrity_error")
                    raise
                journal.append("measured", measurement=measured)
            try:
                with query_step("feedback"):
                    feedback = _feedback_values(data.catalog, source, measured)
            except Exception:
                journal.append("failed", failure_reason="completion_integrity_error")
                raise
            journal.append("completed", feedback=feedback)
        raise ValueError("normal query execution did not terminate within eight queries")
    finally:
        os.close(lock)
