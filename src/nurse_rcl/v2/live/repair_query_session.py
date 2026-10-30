"""Server-only durable adapter for current-source, feedback-first verification.

This is not an agent/runtime replacement or a new query policy. It preserves
the existing reserve/measure/append semantics with an append-only journal.
Complete case/file replay and the fixed ledger location belong to the driver.
No prior serialized feedback is accepted without raw-source replay.
"""

from __future__ import annotations

import os
import stat
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import Field

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.dynamic_verification.source_store import require_store_unchanged, store_snapshot
from nurse_rcl.v2.live import repair_queries as _query_values
from nurse_rcl.v2.live.repair_measurements import RepairSignalMeasurement, measure_repair_signal
from nurse_rcl.v2.live.repair_queries import (
    _ROOTS as QUERY_ROOTS,
)
from nurse_rcl.v2.live.repair_queries import (
    REPAIR_QUERY_SEMANTICS_SHA256,
    RepairQueryCatalog,
    RepairQueryFeedback,
    RepairQuerySelection,
    _seal,
    build_repair_query_catalog,
    build_repair_query_feedback,
    select_repair_query,
)
from nurse_rcl.v2.live.repair_query_events import query_step
from nurse_rcl.v2.live.repair_safety import (
    REPAIR_MAX_JSON_BYTES,
    RepairModel,
    canonical,
    require_unchanged,
    snapshot,
)

if TYPE_CHECKING:
    from nurse_rcl.v2.live.repair_case_guards import _CaseInputs

REPAIR_QUERY_JOURNAL_SCHEMA_VERSION: Final = "nurse.v2.live-repair-query-journal/1"
REPAIR_QUERY_SESSION_SCHEMA_VERSION: Final = "nurse.v2.live-repair-query-session/1"


class RepairQueryJournalEntry(RepairModel):
    schema_version: Literal["nurse.v2.live-repair-query-journal/1"] = (
        REPAIR_QUERY_JOURNAL_SCHEMA_VERSION
    )
    sequence: int = Field(ge=0, le=26)
    previous_sha256: SHA256 | None
    catalog_sha256: SHA256
    phase: Literal["initial", "reserved", "measured", "completed", "stopped", "failed"]
    selection: RepairQuerySelection | None = None
    measurement: RepairSignalMeasurement | None = None
    feedback: RepairQueryFeedback | None = None
    failure_reason: Literal["measurement_integrity_error", "completion_integrity_error"] | None = (
        None
    )
    entry_sha256: SHA256


class RepairQueryCompletedStep(RepairModel):
    selection: RepairQuerySelection
    measurement: RepairSignalMeasurement
    feedback: RepairQueryFeedback


class RepairQuerySession(RepairModel):
    schema_version: Literal["nurse.v2.live-repair-query-session/1"] = (
        REPAIR_QUERY_SESSION_SCHEMA_VERSION
    )
    semantics_sha256: SHA256
    catalog_sha256: SHA256
    journal_sha256: SHA256
    steps: tuple[RepairQueryCompletedStep, ...]
    submitted_request_keys: tuple[str, ...]
    pending_selection: RepairQuerySelection | None
    pending_measurement: RepairSignalMeasurement | None
    paid_count: int = Field(ge=0, le=8)
    stop_reason: Literal["query_budget_exhausted", "no_admissible_progress"] | None
    failure_reason: Literal["measurement_integrity_error", "completion_integrity_error"] | None
    session_sha256: SHA256


_ROOTS = (*QUERY_ROOTS, RepairQueryJournalEntry, RepairQuerySession)
_MAX_RECORD_BYTES = REPAIR_MAX_JSON_BYTES


def _fsync_directory(root: Path) -> None:
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(path: Path, entry: RepairQueryJournalEntry) -> None:
    """An exclusive hard-link publishes a complete record, never replacing history."""
    raw = canonical(entry, _ROOTS)
    if _seal(entry, "entry_sha256") != entry:
        raise ValueError("query journal entry digest mismatch")
    temporary = path.parent / (".query-" + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
        atomic_write_bytes(
            path.parent / "head", canonical_json_bytes([entry.sequence, entry.entry_sha256])
        )
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_bytes(path: Path, limit: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
            raise ValueError("query journal record is not a bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("query journal record exceeds capacity")
        return raw
    finally:
        os.close(descriptor)


def _read_journal(root: Path) -> tuple[RepairQueryJournalEntry, ...]:
    files = tuple(sorted(root.glob("*.json")))
    if (
        not files
        or len(files) > 27
        or tuple(p.name for p in files) != tuple(f"{i:03d}.json" for i in range(len(files)))
    ):
        raise ValueError("query journal is empty, noncontiguous or exceeds eight-query capacity")
    entries: list[RepairQueryJournalEntry] = []
    for i, path in enumerate(files):
        raw = _read_bytes(path, _MAX_RECORD_BYTES)
        entry = RepairQueryJournalEntry.model_validate_json(raw)
        if (
            raw != canonical(entry, _ROOTS)
            or entry.sequence != i
            or _seal(entry, "entry_sha256") != entry
            or entry.previous_sha256 != (entries[-1].entry_sha256 if entries else None)
        ):
            raise ValueError("query journal identity/chain mismatch")
        entries.append(entry)
    head = root / "head"
    if head.exists() or head.is_symlink():
        raw_head = _read_bytes(head, 256)
        # One durable record may precede its head write after a crash. A head
        # ahead of the retained journal, or any other rollback, is rejected.
        if raw_head not in (
            canonical_json_bytes([e.sequence, e.entry_sha256]) for e in entries[-2:]
        ):
            raise ValueError("query journal head indicates rollback or corruption")
    elif len(entries) != 1:
        raise ValueError("query journal head is missing")
    return tuple(entries)


class RepairQueryRuntime:
    """One current-version step at a time; all in-memory views are disposable.

    catalog_arguments are forwarded to the explicitly typed catalog builder;
    unexpected arguments are rejected there, never projected or ignored. The
    full source list is retained, not reduced to selected candidates/modalities.
    """

    def __init__(
        self,
        *,
        catalog: RepairQueryCatalog,
        expected_catalog_sha256: str,
        ledger_root: Path,
        **catalog_arguments: Any,
    ) -> None:
        if type(catalog) is not RepairQueryCatalog:
            raise ValueError("query runtime needs an exact current catalog")
        self.catalog = catalog
        self.pin = require_pin(expected_catalog_sha256)
        self.root = Path(ledger_root)
        if self.root.is_symlink():
            raise ValueError("query ledger root may not be a symlink")
        self.arguments = dict(catalog_arguments)

    @contextmanager
    def _protected_inputs(self) -> Iterator[Callable[[], None]]:
        sources = self.arguments.get("sources")
        if type(sources) is not tuple:
            raise ValueError("query runtime requires the complete source tuple")
        store = self.arguments["store"]
        state = store_snapshot(store)
        # Guard each raw series individually, preserving existing per-root caps.
        source_before = tuple(snapshot(s, _ROOTS) for s in sources)

        def current_context() -> tuple[object, ...]:
            return (
                self.catalog,
                self.pin,
                str(self.root),
                tuple(
                    (k, v)
                    for k, v in sorted(self.arguments.items())
                    if k not in {"sources", "store"}
                ),
            )

        context = current_context()
        before = snapshot(context, _ROOTS)
        with ExitStack() as guards:

            def unchanged() -> None:
                def handles() -> None:
                    if (
                        self.arguments.get("sources") is not sources
                        or self.arguments.get("store") is not store
                    ):
                        raise ValueError("query runtime source handles changed")

                with ExitStack() as checks:
                    checks.callback(require_store_unchanged, store, state)
                    for source, saved in zip(sources, source_before, strict=True):
                        checks.callback(require_unchanged, source, _ROOTS, saved)
                    checks.callback(handles)
                    checks.callback(lambda: require_unchanged(current_context(), _ROOTS, before))

            guards.callback(unchanged)
            yield unchanged

    @contextmanager
    def _operation(self) -> Iterator[tuple[RepairQueryCatalog, Callable[[], None]]]:
        with self._protected_inputs() as unchanged, ExitStack() as guards:
            fresh = build_repair_query_catalog(**self.arguments)
            guards.callback(require_unchanged, fresh, _ROOTS, snapshot(fresh, _ROOTS))
            if fresh.catalog_sha256 != self.pin or canonical(fresh, _ROOTS) != canonical(
                self.catalog, _ROOTS
            ):
                raise ValueError(
                    "query runtime catalog differs from complete current source replay"
                )
            yield fresh, unchanged

    def _parameters(
        self, directory: RepairQueryCatalog, choice: RepairQuerySelection
    ) -> dict[str, Any]:
        request = next(
            (r for r in directory.requests if r.request_key == choice.selected_request_key), None
        )
        if request is None:
            raise ValueError("query execution lacks a selected source")
        source = next(
            s for s in self.arguments["sources"] if s.source_sha256 == request.source_sha256
        )
        return dict(
            source=source,
            expected_source_sha256=request.source_sha256,
            time_contract=self.arguments["time_contract"],
            expected_time_contract_sha256=self.arguments["expected_time_contract_sha256"],
            store=self.arguments["store"],
        )

    @staticmethod
    def _choice(
        directory: RepairQueryCatalog,
        steps: tuple[RepairQueryCompletedStep, ...],
        submitted: tuple[str, ...],
    ) -> RepairQuerySelection:
        with query_step("choice"):
            return select_repair_query(
                catalog=directory,
                expected_catalog_sha256=directory.catalog_sha256,
                feedback=tuple(s.feedback for s in steps),
                expected_feedback_sha256s=tuple(s.feedback.result_sha256 for s in steps),
                submitted_request_keys=submitted,
            )

    def _feedback(
        self,
        directory: RepairQueryCatalog,
        selection: RepairQuerySelection,
        measurement: RepairSignalMeasurement,
    ) -> RepairQueryFeedback:
        with query_step("feedback"):
            return build_repair_query_feedback(
                **self._parameters(directory, selection),
                catalog=directory,
                expected_catalog_sha256=directory.catalog_sha256,
                measurement=measurement,
                expected_measurement_sha256=measurement.measurement_sha256,
            )

    def _replay(
        self, directory: RepairQueryCatalog, entries: tuple[RepairQueryJournalEntry, ...]
    ) -> RepairQuerySession:
        with query_step("replay"):
            before = snapshot(entries, _ROOTS)
            try:
                return self._replay_values(directory, entries)
            finally:
                require_unchanged(entries, _ROOTS, before)

    def _replay_values(
        self, directory: RepairQueryCatalog, entries: tuple[RepairQueryJournalEntry, ...]
    ) -> RepairQuerySession:
        return self._replay_with_pending(directory, entries)[0]

    def _replay_with_pending(
        self, directory: RepairQueryCatalog, entries: tuple[RepairQueryJournalEntry, ...]
    ) -> tuple[RepairQuerySession, RepairQueryFeedback | None]:
        """Same complete replay; retain its pending feedback only for this call."""
        steps: tuple[RepairQueryCompletedStep, ...] = ()
        submitted: tuple[str, ...] = ()
        pending, measured, feedback, stop, failure = None, None, None, None, None
        for index, entry in enumerate(entries):
            if (
                entry.catalog_sha256 != directory.catalog_sha256
                or stop is not None
                or failure is not None
            ):
                raise ValueError(
                    "query journal changes source identity or extends a terminal state"
                )
            allowed = {
                "initial": (),
                "reserved": ("selection",),
                "measured": ("measurement",),
                "completed": ("feedback",),
                "stopped": ("selection",),
                "failed": ("failure_reason",),
            }[entry.phase]
            if any(
                (getattr(entry, name) is not None) != (name in allowed)
                for name in ("selection", "measurement", "feedback", "failure_reason")
            ):
                raise ValueError("query journal phase payload mismatch")
            if index == 0:
                if entry.phase != "initial":
                    raise ValueError("query journal must start with its initial identity")
            elif entry.phase == "reserved":
                expected = self._choice(directory, steps, submitted)
                if (
                    pending is not None
                    or expected.selected_request_key is None
                    or entry.selection != expected
                ):
                    raise ValueError("query reservation differs from the actual feedback prefix")
                pending = expected
                submitted = tuple(sorted((*submitted, expected.selected_request_key)))
            elif entry.phase == "measured":
                if pending is None or measured is not None or entry.measurement is None:
                    raise ValueError("query measurement has no unique pending reservation")
                measured = entry.measurement
                feedback = self._feedback(directory, pending, measured)
            elif entry.phase == "completed":
                if (
                    pending is None
                    or measured is None
                    or feedback is None
                    or entry.feedback != feedback
                ):
                    raise ValueError("query completion differs from raw-source feedback replay")
                steps += (
                    RepairQueryCompletedStep(
                        selection=pending, measurement=measured, feedback=feedback
                    ),
                )
                pending, measured, feedback = None, None, None
            elif entry.phase == "stopped":
                expected = self._choice(directory, steps, submitted)
                if (
                    pending is not None
                    or expected.selected_request_key is not None
                    or entry.selection != expected
                ):
                    raise ValueError("query stop differs from actual feedback prefix")
                stop = expected.stop_reason
            elif entry.phase == "failed":
                if pending is None:
                    raise ValueError("query failure has no outstanding reservation")
                failure = entry.failure_reason
            else:
                raise ValueError("query journal repeats an initial state")
        result = _seal(
            RepairQuerySession(
                semantics_sha256=REPAIR_QUERY_SEMANTICS_SHA256,
                catalog_sha256=directory.catalog_sha256,
                journal_sha256=entries[-1].entry_sha256,
                steps=steps,
                submitted_request_keys=submitted,
                pending_selection=pending,
                pending_measurement=measured,
                paid_count=len(steps),
                stop_reason=stop,
                failure_reason=failure,
                session_sha256="0" * 64,
            ),
            "session_sha256",
        )
        return result, feedback

    def inspect(self) -> RepairQuerySession:
        """Read-only source replay of an atomically published prefix, no execution."""
        with self._operation() as (directory, _):
            return self._replay(directory, _read_journal(self.root))

    @contextmanager
    def _locked_ledger(self) -> Iterator[bool]:
        import fcntl

        if self.root.is_symlink():
            raise ValueError("query ledger root may not be a symlink")
        created = False
        try:
            self.root.mkdir()
            created = True
        except FileExistsError:
            if not self.root.is_dir():
                raise ValueError("query ledger root is not a directory") from None
        descriptor = os.open(self.root / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield created
        finally:
            os.close(descriptor)

    def _advance_values(
        self,
        directory: RepairQueryCatalog,
        unchanged: Callable[[], None],
        *,
        created: bool,
    ) -> RepairQuerySession:
        """Private single-step body; its caller must own replay guards and ledger lock."""
        entries: tuple[RepairQueryJournalEntry, ...] = ()

        def append(phase: str, **payload: object) -> None:
            nonlocal entries
            entry = RepairQueryJournalEntry.model_validate(
                dict(
                    sequence=len(entries),
                    previous_sha256=entries[-1].entry_sha256 if entries else None,
                    catalog_sha256=directory.catalog_sha256,
                    phase=phase,
                    entry_sha256="0" * 64,
                    **payload,
                )
            )
            entry = _seal(entry, "entry_sha256")
            with query_step("append_guard", sequence=entry.sequence, journal_phase=entry.phase):
                unchanged()
            with query_step("append_publish", sequence=entry.sequence, journal_phase=entry.phase):
                _publish(self.root / f"{entry.sequence:03d}.json", entry)
            entries += (entry,)

        if created:
            append("initial")
        else:
            entries = _read_journal(self.root)
        state = self._replay(directory, entries)
        if state.failure_reason is not None:
            raise ValueError("query session failed integrity; retained for diagnosis")
        if state.stop_reason is not None:
            return state
        pending = state.pending_selection
        measured = state.pending_measurement
        if pending is not None and measured is None:
            raise ValueError(
                "query session has a pending reservation without a measurement receipt"
            )
        if pending is None:
            pending = self._choice(directory, state.steps, state.submitted_request_keys)
            if pending.selected_request_key is None:
                append("stopped", selection=pending)
                return self._replay(directory, entries)
            append("reserved", selection=pending)
            try:
                with query_step("measurement"):
                    measured = measure_repair_signal(**self._parameters(directory, pending))
            except Exception:
                append("failed", failure_reason="measurement_integrity_error")
                raise
            append("measured", measurement=measured)
        if measured is None:
            raise ValueError("query receipt is absent")
        try:
            feedback = self._feedback(directory, pending, measured)
        except Exception:
            append("failed", failure_reason="completion_integrity_error")
            raise
        append("completed", feedback=feedback)
        return self._replay(directory, entries)

    def advance(self) -> RepairQuerySession:
        with self._locked_ledger() as created, self._operation() as (directory, unchanged):
            return self._advance_values(directory, unchanged, created=created)


def _run_owned_queries(inputs: _CaseInputs, *, created: bool) -> RepairQuerySession:
    """Private, lock-held incremental body. The caller retains all raw exits.

    Existing history is semantically replayed once; each newly computed choice
    and feedback is checked before publication, not recomputed for every prefix.
    Complete final independent semantic replay belongs to the case verifier.
    """
    if type(created) is not bool:
        raise ValueError("owned query ledger creation state must be exact")
    inputs.check_decision()
    runtime = inputs.prepared.runtime
    directory = runtime.catalog
    root = runtime.root

    def disk_identity() -> tuple[int, int, int, int]:
        metadata, lock = root.lstat(), (root / "lock").lstat()
        if not stat.S_ISDIR(metadata.st_mode) or not stat.S_ISREG(lock.st_mode):
            raise ValueError("owned journal directory or lock identity changed")
        return metadata.st_dev, metadata.st_ino, lock.st_dev, lock.st_ino

    disk_before = disk_identity()
    entries: tuple[RepairQueryJournalEntry, ...] = ()
    retained: tuple[bytes, ...] = ()
    state: RepairQuerySession | None = None
    state_before: tuple[bytes, str] | None = None
    current_check: Callable[[], None] | None = None

    def check_state() -> None:
        with ExitStack() as checks:
            if state is not None and state_before is not None:
                checks.callback(require_unchanged, state, _ROOTS, state_before)
            if current_check is not None:
                checks.callback(current_check)
            checks.callback(inputs.check_decision)

    def prefix(*, published: bool = False) -> tuple[RepairQueryJournalEntry, ...]:
        if disk_identity() != disk_before:
            raise ValueError("owned journal directory or lock binding changed")
        names = {path.name for path in root.iterdir()}
        expected_names = {"lock", "head", *(f"{index:03d}.json" for index in range(27))}
        if not names <= expected_names:
            raise ValueError("owned journal has unexpected files")
        if not entries and created and not published:
            if names != {"lock"}:
                raise ValueError("new owned journal is not empty")
            return ()
        result = _read_journal(root)
        if published and _read_bytes(root / "head", 256) != canonical_json_bytes(
            [result[-1].sequence, result[-1].entry_sha256]
        ):
            raise ValueError("owned publication head did not advance to the durable tail")
        return result

    def append(phase: str, **payload: object) -> None:
        nonlocal entries, retained
        entry = _seal(
            RepairQueryJournalEntry.model_validate(
                dict(
                    sequence=len(entries),
                    previous_sha256=entries[-1].entry_sha256 if entries else None,
                    catalog_sha256=directory.catalog_sha256,
                    phase=phase,
                    entry_sha256="0" * 64,
                    **payload,
                )
            ),
            "entry_sha256",
        )
        before = snapshot(entry, _ROOTS)
        with query_step("append_guard", sequence=entry.sequence, journal_phase=entry.phase):
            if tuple(canonical(row, _ROOTS) for row in prefix()) != retained:
                raise ValueError("owned journal prefix changed before append")
            check_state()
        with query_step("append_publish", sequence=entry.sequence, journal_phase=entry.phase):
            try:
                _publish(root / f"{entry.sequence:03d}.json", entry)
            finally:
                with ExitStack() as checks:
                    checks.callback(check_state)
                    checks.callback(require_unchanged, entry, _ROOTS, before)
            published = prefix(published=True)
            expected = (*retained, before[0])
            if tuple(canonical(row, _ROOTS) for row in published) != expected:
                raise ValueError("owned journal publication differs from the new transition")
            if disk_identity() != disk_before:
                raise ValueError("owned journal binding changed after publication")
        # Do not advance the private prefix until durable read-back and guards pass.
        entries += (entry,)
        retained = expected

    def set_state(**changes: Any) -> None:
        nonlocal state, state_before
        check_state()
        values = (
            dict(
                semantics_sha256=REPAIR_QUERY_SEMANTICS_SHA256,
                catalog_sha256=directory.catalog_sha256,
                steps=(),
                submitted_request_keys=(),
                pending_selection=None,
                pending_measurement=None,
                paid_count=0,
                stop_reason=None,
                failure_reason=None,
            )
            if state is None
            else {name: getattr(state, name) for name in RepairQuerySession.model_fields}
        )
        values.update(changes, journal_sha256=entries[-1].entry_sha256, session_sha256="0" * 64)
        state = _seal(RepairQuerySession.model_validate(values), "session_sha256")
        state_before = snapshot(state, _ROOTS)

    with ExitStack() as exits:
        exits.callback(check_state)
        pending_feedback = None
        if created:
            append("initial")
            set_state()
        else:
            entries = prefix()
            retained = tuple(canonical(row, _ROOTS) for row in entries)
            before = snapshot(entries, _ROOTS)
            try:
                with query_step("replay"):
                    state, pending_feedback = runtime._replay_with_pending(directory, entries)
            finally:
                require_unchanged(entries, _ROOTS, before)
            state_before = snapshot(state, _ROOTS)
        if state is None:
            raise ValueError("owned query lacks its initial state")
        if state.failure_reason is not None:
            raise ValueError("query session failed integrity; retained for diagnosis")
        if state.pending_selection is not None and state.pending_measurement is None:
            raise ValueError(
                "query session has a pending reservation without a measurement receipt"
            )
        for _ in range(9):
            check_state()
            if state.stop_reason is not None:
                return state
            pending, measured = state.pending_selection, state.pending_measurement
            if pending is None:
                with query_step("choice"):
                    try:
                        pending = _query_values._select_values(
                            directory,
                            tuple(step.feedback for step in state.steps),
                            state.submitted_request_keys,
                        )
                    finally:
                        check_state()
                if pending.selected_request_key is None:
                    append("stopped", selection=pending)
                    set_state(stop_reason=pending.stop_reason)
                    return state
                append("reserved", selection=pending)
                set_state(
                    pending_selection=pending,
                    submitted_request_keys=tuple(
                        sorted((*state.submitted_request_keys, pending.selected_request_key))
                    ),
                )
            parameters = runtime._parameters(directory, pending)
            source = parameters["source"]
            source_before = snapshot(source, _ROOTS)

            current_check = partial(require_unchanged, source, _ROOTS, source_before)
            if measured is None:
                try:
                    with query_step("measurement"):
                        try:
                            measured = measure_repair_signal(**parameters)
                        finally:
                            check_state()
                except Exception:
                    append("failed", failure_reason="measurement_integrity_error")
                    raise
                append("measured", measurement=measured)
                set_state(pending_measurement=measured)
            try:
                # Receipt recovery reuses the feedback just independently rebuilt
                # by this invocation's full prefix replay, not a serialized claim.
                if pending_feedback is None:
                    try:
                        feedback = runtime._feedback(directory, pending, measured)
                    finally:
                        check_state()
                else:
                    feedback = pending_feedback
                snapshot(feedback, _ROOTS)
            except Exception:
                append("failed", failure_reason="completion_integrity_error")
                raise
            append("completed", feedback=feedback)
            set_state(
                steps=(
                    *state.steps,
                    RepairQueryCompletedStep(
                        selection=pending, measurement=measured, feedback=feedback
                    ),
                ),
                pending_selection=None,
                pending_measurement=None,
                paid_count=state.paid_count + 1,
            )
            current_check = None
            pending_feedback = None
        raise ValueError("whole-case verification did not terminate within eight queries")
