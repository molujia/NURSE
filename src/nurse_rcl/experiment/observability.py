"""Flushed run telemetry, atomic progress, heartbeat, and terminal markers."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Literal

from hermes_runtime.redaction import redact_sensitive_text
from pydantic import Field

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.models import ProgressSnapshot
from nurse_rcl.schemas import StrictFrozenModel

OBSERVABILITY_SCHEMA_VERSION: Literal[1] = 1
HEARTBEAT_INTERVAL_SECONDS = 30.0
PROGRESS_REPLACE_ATTEMPTS = 5
PROGRESS_READ_ATTEMPTS = 5
PROGRESS_REPLACE_DELAY_SECONDS = 0.01
EventLevel = Literal["debug", "info", "warning", "error", "fatal"]


class UsageTotals(StrictFrozenModel):
    """Cumulative non-secret LLM usage, latency, and known provider cost."""

    attempts: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_seconds: float = Field(default=0.0, ge=0.0)
    cost_usd: float = Field(default=0.0, ge=0.0)


class RunEvent(StrictFrozenModel):
    """One content-addressed structured event mirrored to the human run log."""

    schema_version: Literal[1] = OBSERVABILITY_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    timestamp_unix: int = Field(ge=0)
    level: EventLevel
    kind: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)
    message: str = Field(min_length=1)
    case_id: str | None = None
    stage: str | None = None
    usage: UsageTotals
    event_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"event_hash"})
        return self.event_hash == sha256_hex(canonical_json_bytes(unsigned))


class CompletionMarker(StrictFrozenModel):
    """The single successful terminal marker for a run."""

    schema_version: Literal[1] = OBSERVABILITY_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    completed_at_unix: int = Field(ge=0)
    message: str = Field(min_length=1)
    marker_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"marker_hash"})
        return self.marker_hash == sha256_hex(canonical_json_bytes(unsigned))


class FatalMarker(StrictFrozenModel):
    """The single failed terminal marker with redacted diagnostics."""

    schema_version: Literal[1] = OBSERVABILITY_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    failed_at_unix: int = Field(ge=0)
    error_code: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)
    message: str = Field(min_length=1)
    marker_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"marker_hash"})
        return self.marker_hash == sha256_hex(canonical_json_bytes(unsigned))


class RunObserver:
    """Thread-safe owner of one run's observable, restart-safe telemetry files."""

    def __init__(
        self,
        *,
        root: Path,
        run_id: str,
        total: int,
        clock: Callable[[], float] = time.time,
        secrets: Iterable[str] = (),
    ) -> None:
        if not run_id:
            raise ValueError("run_id cannot be empty")
        if total < 0:
            raise ValueError("total cannot be negative")
        self._root = root.resolve(strict=False)
        self._root.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id
        self._total = total
        self._clock = clock
        self._secrets = tuple(item for item in secrets if item)
        self._lock = threading.RLock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_error: Exception | None = None
        self._event_sequence = self._recover_event_sequence()
        self._initialize_progress()

    @property
    def heartbeat_running(self) -> bool:
        thread = self._heartbeat_thread
        return thread is not None and thread.is_alive()

    def emit_event(
        self,
        *,
        kind: str,
        level: EventLevel,
        message: str,
        case_id: str | None = None,
        stage: str | None = None,
        usage: UsageTotals | None = None,
    ) -> RunEvent:
        with self._lock:
            return self._emit_event_locked(
                kind=kind,
                level=level,
                message=message,
                case_id=case_id,
                stage=stage,
                usage=usage or UsageTotals(),
            )

    def update_progress(
        self,
        *,
        completed: int,
        succeeded: int,
        degraded: int,
        failed: int,
        current_group: str | None,
        stage_counts: Mapping[str, int],
        cache_hits: int,
        usage: UsageTotals,
    ) -> ProgressSnapshot:
        with self._lock:
            progress = ProgressSnapshot(
                run_id=self._run_id,
                total=self._total,
                completed=completed,
                succeeded=succeeded,
                degraded=degraded,
                failed=failed,
                current_group=current_group,
                heartbeat_unix=self._now(),
                stage_counts=dict(stage_counts),
                cache_hits=cache_hits,
                llm_attempts=usage.attempts,
                llm_input_tokens=usage.input_tokens,
                llm_output_tokens=usage.output_tokens,
                llm_latency_seconds=usage.latency_seconds,
                llm_cost_usd=usage.cost_usd,
            )
            self._write_progress(progress)
            return progress

    def heartbeat(self) -> ProgressSnapshot:
        """Refresh only heartbeat time and record the liveness event."""

        with self._lock:
            current = self._read_progress()
            progress = current.model_copy(update={"heartbeat_unix": self._now()})
            self._write_progress(progress)
            self._emit_event_locked(
                kind="heartbeat",
                level="debug",
                message="run heartbeat",
                case_id=None,
                stage=None,
                usage=_usage_from_progress(progress),
            )
            return progress

    def start_heartbeat(self) -> None:
        with self._lock:
            if self.heartbeat_running:
                return
            if self._terminal_marker_exists():
                raise ValueError("cannot start heartbeat after a terminal marker")
            self._heartbeat_stop.clear()
            self._heartbeat_error = None
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"nurse-heartbeat-{self._run_id}",
                daemon=True,
            )
            self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        thread = self._heartbeat_thread
        if thread is None:
            return
        self._heartbeat_stop.set()
        thread.join(timeout=5.0)
        if thread.is_alive():  # pragma: no cover - defensive OS/thread failure
            raise RuntimeError("heartbeat worker did not stop")
        self._heartbeat_thread = None
        if self._heartbeat_error is not None:
            error = self._heartbeat_error
            self._heartbeat_error = None
            raise RuntimeError("heartbeat worker failed") from error

    def complete(self, *, message: str) -> CompletionMarker:
        self.stop_heartbeat()
        with self._lock:
            existing = self._read_existing_terminal_marker()
            if isinstance(existing, CompletionMarker):
                return existing
            if existing is not None:
                raise ValueError("a conflicting terminal marker already exists")
            progress = self._read_progress()
            if progress.completed != progress.total:
                raise ValueError("completion marker requires all cases terminal")
            redacted = self._redact(message)
            timestamp = self._now()
            self._emit_event_locked(
                kind="run_completed",
                level="info",
                message=redacted,
                case_id=None,
                stage=None,
                usage=_usage_from_progress(progress),
            )
            unsigned = {
                "schema_version": OBSERVABILITY_SCHEMA_VERSION,
                "run_id": self._run_id,
                "completed_at_unix": timestamp,
                "message": redacted,
            }
            marker = CompletionMarker(
                run_id=self._run_id,
                completed_at_unix=timestamp,
                message=redacted,
                marker_hash=sha256_hex(canonical_json_bytes(unsigned)),
            )
            atomic_write_bytes(
                self._completion_path,
                canonical_json_bytes(marker.model_dump(mode="json")),
            )
            return marker

    def fail(self, *, error_code: str, message: str) -> FatalMarker:
        self.stop_heartbeat()
        with self._lock:
            existing = self._read_existing_terminal_marker()
            if isinstance(existing, FatalMarker):
                return existing
            if existing is not None:
                raise ValueError("a conflicting terminal marker already exists")
            progress = self._read_progress()
            redacted = self._redact(message)
            timestamp = self._now()
            self._emit_event_locked(
                kind="run_failed",
                level="fatal",
                message=redacted,
                case_id=None,
                stage=None,
                usage=_usage_from_progress(progress),
            )
            unsigned = {
                "schema_version": OBSERVABILITY_SCHEMA_VERSION,
                "run_id": self._run_id,
                "failed_at_unix": timestamp,
                "error_code": error_code,
                "message": redacted,
            }
            marker = FatalMarker(
                run_id=self._run_id,
                failed_at_unix=timestamp,
                error_code=error_code,
                message=redacted,
                marker_hash=sha256_hex(canonical_json_bytes(unsigned)),
            )
            atomic_write_bytes(
                self._fatal_path,
                canonical_json_bytes(marker.model_dump(mode="json")),
            )
            return marker

    @property
    def _progress_path(self) -> Path:
        return self._root / "progress.json"

    @property
    def _completion_path(self) -> Path:
        return self._root / "RUN_COMPLETE"

    @property
    def _fatal_path(self) -> Path:
        return self._root / "fatal.json"

    def _initialize_progress(self) -> None:
        if self._progress_path.exists():
            progress = self._read_progress()
            if progress.run_id != self._run_id or progress.total != self._total:
                raise ValueError("existing progress does not match this run")
            return
        self._write_progress(
            ProgressSnapshot(
                run_id=self._run_id,
                total=self._total,
                completed=0,
                succeeded=0,
                degraded=0,
                failed=0,
                current_group=None,
                heartbeat_unix=self._now(),
                stage_counts={},
                cache_hits=0,
                llm_attempts=0,
                llm_input_tokens=0,
                llm_output_tokens=0,
                llm_latency_seconds=0.0,
                llm_cost_usd=0.0,
            )
        )

    def _heartbeat_loop(self) -> None:
        try:
            while not self._heartbeat_stop.wait(HEARTBEAT_INTERVAL_SECONDS):
                self.heartbeat()
        except Exception as error:  # pragma: no cover - surfaced by stop_heartbeat
            self._heartbeat_error = error

    def _emit_event_locked(
        self,
        *,
        kind: str,
        level: EventLevel,
        message: str,
        case_id: str | None,
        stage: str | None,
        usage: UsageTotals,
    ) -> RunEvent:
        sequence = self._event_sequence + 1
        timestamp = self._now()
        redacted = self._redact(message)
        unsigned = {
            "schema_version": OBSERVABILITY_SCHEMA_VERSION,
            "run_id": self._run_id,
            "sequence": sequence,
            "timestamp_unix": timestamp,
            "level": level,
            "kind": kind,
            "message": redacted,
            "case_id": case_id,
            "stage": stage,
            "usage": usage.model_dump(mode="json"),
        }
        event = RunEvent(
            run_id=self._run_id,
            sequence=sequence,
            timestamp_unix=timestamp,
            level=level,
            kind=kind,
            message=redacted,
            case_id=case_id,
            stage=stage,
            usage=usage,
            event_hash=sha256_hex(canonical_json_bytes(unsigned)),
        )
        _append_and_flush(
            self._root / "events.jsonl",
            canonical_json_bytes(event.model_dump(mode="json")).decode("utf-8") + "\n",
        )
        identifiers = " ".join(
            item
            for item in (
                f"case={case_id}" if case_id is not None else "",
                f"stage={stage}" if stage is not None else "",
            )
            if item
        )
        suffix = f" {identifiers}" if identifiers else ""
        _append_and_flush(
            self._root / "run.log",
            f"{event.timestamp_unix} [{level.upper()}] {kind}{suffix}: {event.message}\n",
        )
        self._event_sequence = sequence
        return event

    def _recover_event_sequence(self) -> int:
        path = self._root / "events.jsonl"
        if not path.exists():
            return 0
        expected = 1
        for line in path.read_text(encoding="utf-8").splitlines():
            event = RunEvent.model_validate_json(line)
            if (
                not event.verify()
                or event.run_id != self._run_id
                or event.sequence != expected
            ):
                raise ValueError("existing event chain is invalid")
            expected += 1
        return expected - 1

    def _write_progress(self, progress: ProgressSnapshot) -> None:
        data = canonical_json_bytes(progress.model_dump(mode="json"))
        for attempt in range(1, PROGRESS_REPLACE_ATTEMPTS + 1):
            try:
                atomic_write_bytes(self._progress_path, data)
            except PermissionError:
                if attempt == PROGRESS_REPLACE_ATTEMPTS:
                    raise
                time.sleep(PROGRESS_REPLACE_DELAY_SECONDS)
            else:
                return
        raise AssertionError("bounded progress replacement did not terminate")

    def _read_progress(self) -> ProgressSnapshot:
        return read_progress_snapshot(self._root)

    def _terminal_marker_exists(self) -> bool:
        return self._completion_path.exists() or self._fatal_path.exists()

    def _read_existing_terminal_marker(
        self,
    ) -> CompletionMarker | FatalMarker | None:
        if self._completion_path.exists() and self._fatal_path.exists():
            raise ValueError("multiple terminal markers exist")
        if self._completion_path.exists():
            completion_marker = CompletionMarker.model_validate_json(
                self._completion_path.read_text(encoding="utf-8")
            )
            if (
                not completion_marker.verify()
                or completion_marker.run_id != self._run_id
            ):
                raise ValueError("completion terminal marker is invalid")
            return completion_marker
        if self._fatal_path.exists():
            fatal_marker = FatalMarker.model_validate_json(
                self._fatal_path.read_text(encoding="utf-8")
            )
            if not fatal_marker.verify() or fatal_marker.run_id != self._run_id:
                raise ValueError("fatal terminal marker is invalid")
            return fatal_marker
        return None

    def _redact(self, value: str) -> str:
        return redact_sensitive_text(value, secrets=self._secrets)

    def _now(self) -> int:
        value = int(self._clock())
        if value < 0:
            raise ValueError("clock returned a negative timestamp")
        return value


def _usage_from_progress(progress: ProgressSnapshot) -> UsageTotals:
    return UsageTotals(
        attempts=progress.llm_attempts,
        input_tokens=progress.llm_input_tokens,
        output_tokens=progress.llm_output_tokens,
        latency_seconds=progress.llm_latency_seconds,
        cost_usd=progress.llm_cost_usd,
    )


def read_progress_snapshot(root: Path) -> ProgressSnapshot:
    """Read atomic progress with bounded retries for Windows sharing windows."""

    path = root.resolve(strict=False) / "progress.json"
    for attempt in range(1, PROGRESS_READ_ATTEMPTS + 1):
        try:
            return ProgressSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == PROGRESS_READ_ATTEMPTS:
                raise
            time.sleep(PROGRESS_REPLACE_DELAY_SECONDS)
    raise AssertionError("bounded progress read did not terminate")


def _append_and_flush(path: Path, value: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
