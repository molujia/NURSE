"""Private diagnostic spans; never source authority or permission to switch accounts."""

from __future__ import annotations

import json
import os
import stat
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from itertools import pairwise
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.repair_timing import CasePhaseClock, TimingJournal, _time
from nurse_rcl.v2.live.staged_case import require_plain_path

_SCHEMA = "nurse.v2.live-repair-module-span/1"
_MODULES = frozenset(
    (
        "worker_supervision",
        "case_worker",
        "ops_core",
        "ops_investigation",
        "preparation_build",
        "preparation_load",
        "preparation_validate",
        "preparation_publish",
        "preparation_read",
        "preparation_assembly",
        "preparation_guard",
        "preparation_graph",
        "preparation_sources",
        "preparation_catalog",
        "preparation_errors",
        "preparation_encode",
        "input_guard",
        "choice",
        "reservation_publish",
        "measurement",
        "feedback",
        "query_replay",
        "production_scoring",
        "fixed_facts",
        "error_facts",
        "local_scoring",
        "mechanisms",
        "strengths",
        "propagation",
        "ranking",
        "independent_replay",
        "report",
        "report_projection",
        "sre_score_lineage",
        "fixed_core_import",
        "frozen_source_publication",
        "sre_comparison_view",
        "sre_comparable_checks",
        "request_build",
        "ark_attempt",
        "retry_wait",
        "explanation_checks",
        "semantic_fixed_projection",
        "semantic_first",
        "semantic_tools",
        "semantic_feedback",
        "semantic_followup",
        "original_exit",
        "fresh_exit",
        "artifact_publish",
        "parent_finalize",
    )
)
_ROLES = frozenset(("production", "verification", "orchestration"))
_MAX_EVENTS = 8192
_MAX_BYTES = _MAX_EVENTS * 2048
_SESSION: ContextVar[_Recorder | None] = ContextVar("repair_module_recorder", default=None)
_STACK: ContextVar[tuple[dict[str, Any], ...]] = ContextVar("repair_module_stack", default=())


def _hex(value: object, length: int) -> bool:
    return (
        type(value) is str and len(value) == length and all(c in "0123456789abcdef" for c in value)
    )


class _Recorder:
    def __init__(self, root: Path, identity: str, boot_id: str, parent_id: str | None):
        if not _hex(identity, 64) or type(boot_id) is not str or not 0 < len(boot_id) <= 128:
            raise ValueError("invalid module timing identity/epoch")
        if parent_id is not None and not _hex(parent_id, 32):
            raise ValueError("invalid parent module span")
        require_plain_path(root.parent)
        root.mkdir(mode=0o700, exist_ok=True)
        require_plain_path(root)
        self.identity, self.boot_id, self.parent_id = identity, boot_id, parent_id
        self.process_id = uuid.uuid4().hex
        self.path = root / (self.process_id + ".jsonl")
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.sequence, self.previous, self.last = 0, "0" * 64, 0.0

    @property
    def current_span_id(self) -> str | None:
        stack = _STACK.get()
        return stack[-1]["span_id"] if stack else self.parent_id

    def emit(self, span: dict[str, Any], action: str) -> None:
        now = _time(time.monotonic())
        if now < self.last or self.sequence >= _MAX_EVENTS:
            raise ValueError("module timing clock/capacity exceeded")
        body = dict(
            schema_version=_SCHEMA,
            identity=self.identity,
            boot_id=self.boot_id,
            process_id=self.process_id,
            sequence=self.sequence,
            previous=self.previous,
            action=action,
            now=now,
            **span,
        )
        digest = sha256_hex(canonical_json_bytes(body))
        packet = canonical_json_bytes({**body, "sha256": digest}) + b"\n"
        if len(packet) > 2048 or os.write(self.fd, packet) != len(packet):
            raise OSError("incomplete module timing write")
        os.fsync(self.fd)
        self.sequence, self.previous, self.last = self.sequence + 1, digest, now

    def observe(self) -> None:
        # Parent heartbeat observations provide a lower bound, never a fake end.
        stack = _STACK.get()
        if stack:
            self.emit(stack[-1], "observed")


@contextmanager
def _module_timing_scope(
    root: Path,
    *,
    identity: str,
    boot_id: str,
    parent_id: str | None = None,
) -> Iterator[_Recorder]:
    if _SESSION.get() is not None:
        raise ValueError("cannot replace an active module timing owner")
    recorder = _Recorder(root, identity, boot_id, parent_id)
    token = _SESSION.set(recorder)
    primary: BaseException | None = None
    try:
        yield recorder
    except BaseException as exc:
        primary = exc
        raise
    finally:
        _SESSION.reset(token)
        try:
            os.close(recorder.fd)
        except BaseException as exc:
            if primary is None:
                raise
            from nurse_rcl.v2.live.ops_failure import retain_secondary

            retain_secondary(primary, "timing_close", exc)


@contextmanager
def _module_span(module: str, *, role: str | None = None) -> Iterator[None]:
    if (
        type(module) is not str
        or module not in _MODULES
        or (role is not None and (type(role) is not str or role not in _ROLES))
    ):
        raise ValueError("unknown module timing metadata")
    recorder = _SESSION.get()
    if recorder is None:
        yield
        return
    stack = _STACK.get()
    span = dict(
        span_id=uuid.uuid4().hex,
        parent_id=recorder.current_span_id,
        module=module,
        role=role or (stack[-1]["role"] if stack else "production"),
    )
    recorder.emit(span, "started")
    started = recorder.last

    def progress(status: str) -> None:
        if module.startswith("preparation_"):
            packet = (
                canonical_json_bytes(
                    dict(
                        event="preparation_step",
                        module=module,
                        status=status,
                        elapsed_seconds=recorder.last - started,
                    )
                )
                + b"\n"
            )
            if os.write(1, packet) != len(packet):
                raise OSError("incomplete preparation progress write")

    progress("started")
    token = _STACK.set((*stack, span))
    status = "failed"
    primary: BaseException | None = None
    try:
        yield
        status = "completed"
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            recorder.emit(span, status)
            progress(status)
        except BaseException as exc:
            if primary is None:
                raise
            from nurse_rcl.v2.live.ops_failure import retain_secondary

            retain_secondary(primary, "timing_finish", exc)
        finally:
            _STACK.reset(token)


_P = ParamSpec("_P")
_T = TypeVar("_T")


def _module_call(
    module: str, *, role: str | None = None
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def decorate(function: Callable[_P, _T]) -> Callable[_P, _T]:
        @wraps(function)
        def invoke(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            with _module_span(module, role=role):
                return function(*args, **kwargs)

        return invoke

    return decorate


def _module_scope(
    enter: str,
    exit: str,
    *,
    role: str | None = None,
) -> Callable[[Callable[_P, AbstractContextManager[_T]]], Callable[_P, AbstractContextManager[_T]]]:
    """Measure actual enter/exit, with outer fallback if diagnostic output fails."""

    def decorate(
        factory: Callable[_P, AbstractContextManager[_T]],
    ) -> Callable[_P, AbstractContextManager[_T]]:
        @wraps(factory)
        @contextmanager
        def invoke(*args: _P.args, **kwargs: _P.kwargs) -> Iterator[_T]:
            with ExitStack() as owners:
                with _module_span(enter, role=role):
                    value = owners.enter_context(factory(*args, **kwargs))
                try:
                    yield value
                finally:
                    exception = sys.exc_info()
                    with _module_span(exit, role=role):
                        owners.__exit__(*exception)

        return invoke

    return decorate


def _read_log(path: Path) -> tuple[dict[str, Any], ...]:
    require_plain_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_BYTES:
            raise ValueError("module timing file type/capacity differs")
        data = stream.read(_MAX_BYTES + 1)
        after = os.fstat(stream.fileno())
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or len(data) != before.st_size:
            raise ValueError("module timing changed while read")
    lines = data.splitlines(keepends=True)
    if not lines or len(lines) > _MAX_EVENTS:
        raise ValueError("module timing file is empty/over capacity")
    result = []
    for line in lines:
        if len(line) > 2048 or not line.endswith(b"\n"):
            raise ValueError("partial module timing event")
        row = json.loads(line)
        if type(row) is not dict or canonical_json_bytes(row) + b"\n" != line:
            raise ValueError("noncanonical module timing event")
        result.append(row)
    return tuple(result)


def summarize_module_timing(
    *,
    event_paths: tuple[Path, ...],
    timing_root: Path,
    expected_timing_identity: str,
    expected_policy: str,
    checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Read-only reconciliation at a persisted parent checkpoint, not wall time now."""
    from nurse_rcl.v2.live.repair_driver import read_json

    if not event_paths or len(event_paths) > 64 or len(set(event_paths)) != len(event_paths):
        raise ValueError("invalid module timing process collection")
    first = read_json(timing_root / "event-00000000.json", limit=4096)
    timing = TimingJournal._read_existing(
        timing_root,
        identity=expected_timing_identity,
        boot_id=first["boot_id"],
        expected_policy=expected_policy,
    )
    clock_rows = [read_json(p, limit=4096) for p in sorted(timing_root.iterdir())]
    if checkpoint_sha256 is not None:
        indexes = [i for i, row in enumerate(clock_rows) if row["sha256"] == checkpoint_sha256]
        if len(indexes) != 1:
            raise ValueError("module summary checkpoint absent from parent chain")
        clock_rows = clock_rows[: indexes[0] + 1]
        timing = TimingJournal(
            timing.root,
            timing.identity,
            timing.boot_id,
            CasePhaseClock.start(now=clock_rows[0]["now"], policy=expected_policy),
        )
        for row in clock_rows[1:]:
            timing._apply(row["action"], row["account"], row["reason"], row["now"])
        timing.previous, timing.sequence = clock_rows[-1]["sha256"], len(clock_rows)
    intervals = [
        (a["now"], b["now"], a["account"])
        for a, b in pairwise(clock_rows)
        if a["action"] != "pause"
    ]
    spans: dict[str, dict[str, Any]] = {}
    event_pins: dict[str, str] = {}
    process_last: dict[str, float] = {}
    fields = {
        "schema_version",
        "identity",
        "boot_id",
        "process_id",
        "sequence",
        "previous",
        "action",
        "now",
        "span_id",
        "parent_id",
        "module",
        "role",
        "sha256",
    }
    for path in event_paths:
        previous, last = "0" * 64, 0.0
        stack: list[str] = []
        log_rows = _read_log(path)
        if path.name in event_pins:
            raise ValueError("duplicate module timing process identity")
        event_pins[path.name] = sha256_hex(
            b"".join(canonical_json_bytes(r) + b"\n" for r in log_rows)
        )
        for sequence, row in enumerate(log_rows):
            body = {k: v for k, v in row.items() if k != "sha256"}
            if (
                set(row) != fields
                or row["schema_version"] != _SCHEMA
                or row["identity"] != expected_timing_identity
                or row["boot_id"] != timing.boot_id
                or not _hex(row["process_id"], 32)
                or path.name != row["process_id"] + ".jsonl"
                or type(row["sequence"]) is not int
                or row["sequence"] != sequence
                or row["previous"] != previous
                or row["sha256"] != sha256_hex(canonical_json_bytes(body))
                or not _hex(row["span_id"], 32)
                or (row["parent_id"] is not None and not _hex(row["parent_id"], 32))
                or type(row["module"]) is not str
                or row["module"] not in _MODULES
                or type(row["role"]) is not str
                or row["role"] not in _ROLES
            ):
                raise ValueError("module timing identity/shape/chain differs")
            now = _time(row["now"])
            if now < last or not timing.clock.started <= now <= timing.clock.last:
                raise ValueError("module timing lies outside parent account or moves backwards")
            key = row["span_id"]
            if row["action"] == "started":
                if (
                    key in spans
                    or key == row["parent_id"]
                    or (stack and row["parent_id"] != stack[-1])
                ):
                    raise ValueError("module span parent/sequence differs")
                spans[key] = {**row, "start": now, "end": None, "status": "incomplete"}
                stack.append(key)
            elif row["action"] in ("observed", "completed", "failed"):
                if not stack or stack[-1] != key:
                    raise ValueError("module span ended/observed out of order")
                span = spans[key]
                if any(span[k] != row[k] for k in ("parent_id", "module", "role", "process_id")):
                    raise ValueError("module span metadata changed")
                if row["action"] != "observed":
                    span.update(end=now, status=row["action"])
                    stack.pop()
            else:
                raise ValueError("unknown module timing action")
            previous, last = row["sha256"], now
            process_last[row["process_id"]] = now
    for span in spans.values():
        if span["end"] is None:
            span["end"] = process_last[span["process_id"]]
    children: dict[str | None, list[dict[str, Any]]] = {}
    for key, span in spans.items():
        parent = span["parent_id"]
        if parent is not None:
            if parent not in spans:
                raise ValueError("module span lacks independently recorded parent")
            owner = spans[parent]
            if not owner["start"] <= span["start"] <= span["end"] <= owner["end"]:
                raise ValueError("module child lies outside parent span")
            chain = {key}
            cursor = parent
            while cursor is not None:
                if cursor in chain or cursor not in spans:
                    raise ValueError("module timing parent cycle/missing parent")
                chain.add(cursor)
                cursor = spans[cursor]["parent_id"]
        children.setdefault(parent, []).append(span)
    for siblings in children.values():
        ordered = sorted(siblings, key=lambda s: (s["start"], s["end"]))
        if any(a["end"] > b["start"] for a, b in pairwise(ordered)):
            raise ValueError("overlapping sibling module spans")
    totals = {"method": 0.0, "preparation": 0.0}
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, span in spans.items():
        for account in totals:

            def overlap(start: float, end: float, _account: str = account) -> float:
                return float(
                    sum(
                        max(0.0, min(end, b) - max(start, a))
                        for a, b, name in intervals
                        if name == _account
                    )
                )

            inclusive = overlap(span["start"], span["end"])
            exclusive = inclusive - sum(
                overlap(c["start"], c["end"]) for c in children.get(key, ())
            )
            if exclusive < -1e-8:
                raise ValueError("negative exclusive module time")
            if inclusive == 0:
                continue
            group = span["module"], span["role"], account
            row = rows.setdefault(
                group,
                dict(
                    module=group[0],
                    role=group[1],
                    account=account,
                    inclusive_seconds=0.0,
                    exclusive_seconds=0.0,
                    calls=0,
                    completed=0,
                    failed=0,
                    incomplete=0,
                ),
            )
            row["inclusive_seconds"] += inclusive
            row["exclusive_seconds"] += exclusive
            row["calls"] += 1
            row[span["status"]] += 1
            totals[account] += exclusive
    residual = {name: timing.clock.used_seconds[name] - value for name, value in totals.items()}
    if any(value < -1e-8 for value in residual.values()):
        raise ValueError("module timing over-accounts parent time")
    return dict(
        schema_version="nurse.v2.live-repair-module-timing/1",
        timing_identity_sha256=expected_timing_identity,
        policy=expected_policy,
        timing_checkpoint_sha256=timing.previous,
        timing=timing.clock.summary(),
        complete=all(s["status"] != "incomplete" for s in spans.values()),
        modules=[rows[k] for k in sorted(rows)],
        unattributed_seconds=residual,
        paid_measurement_calls=sum(
            s["module"] == "measurement" and s["role"] == "production" for s in spans.values()
        ),
        ark_attempt_calls=sum(s["module"] == "ark_attempt" for s in spans.values()),
        event_files=event_pins,
    )
