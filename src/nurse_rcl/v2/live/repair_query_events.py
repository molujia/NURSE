"""Fixed method-step events: no source data, callbacks, or timing authority."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.repair_module_timing import _module_span

_OPERATIONS = (
    "runtime_inputs",
    "input_guard",
    "choice",
    "append_guard",
    "append_publish",
    "measurement",
    "feedback",
    "replay",
    "production_scoring",
    "verification_replay",
    "report",
    "owner_exit",
)
_PHASES = ("initial", "reserved", "measured", "completed", "stopped", "failed")
_SPAN_MODULES = {
    "runtime_inputs": "input_guard",
    "input_guard": "input_guard",
    "choice": "choice",
    "append_guard": "input_guard",
    "append_publish": "reservation_publish",
    "measurement": "measurement",
    "feedback": "feedback",
    "replay": "query_replay",
    "production_scoring": "production_scoring",
    "report": "report",
    "owner_exit": "original_exit",
}


@contextmanager
def query_step(
    operation: str, *, sequence: int | None = None, journal_phase: str | None = None
) -> Iterator[None]:
    if (
        type(operation) is not str
        or operation not in _OPERATIONS
        or (sequence is not None and (type(sequence) is not int or not 0 <= sequence <= 26))
        or (
            journal_phase is not None
            and (type(journal_phase) is not str or journal_phase not in _PHASES)
        )
    ):
        raise ValueError("query diagnostic requires bounded exact metadata")

    def emit(status: str, elapsed: float) -> None:
        packet = (
            canonical_json_bytes(
                dict(
                    event="query_method_step",
                    operation=operation,
                    status=status,
                    sequence=sequence,
                    journal_phase=journal_phase,
                    elapsed_seconds=elapsed,
                )
            )
            + b"\n"
        )
        # No arbitrary consumer callback after a completed append guard.
        if os.write(1, packet) != len(packet):
            raise OSError("incomplete query diagnostic write")

    start = time.monotonic()
    emit("started", 0.0)
    completed = False
    try:
        module = _SPAN_MODULES.get(operation)
        with (
            _module_span(module, role="verification" if operation == "replay" else None)
            if module
            else nullcontext()
        ):
            yield
        completed = True
    finally:
        emit("completed" if completed else "failed", time.monotonic() - start)
