"""Durable B0/B2 tool accounting, not measurement or scoring authority.

The owning normal runner must admit the B0 session before opening this journal.
The inherited prefix is immutable. Card expansion and verification share only
the remaining slots; an unfinished reservation stays charged on restart. A
result digest refers to a separately saved receipt, never proof of its content.
"""

from __future__ import annotations

import fcntl
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, TypeAdapter

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.live.normal_artifacts import NormalQuerySession
from nurse_rcl.v2.live.normal_queries import _read_bytes, _seal
from nurse_rcl.v2.live.repair_driver import _fsync
from nurse_rcl.v2.live.repair_preparation_codec import _decode_json_member
from nurse_rcl.v2.live.repair_queries import REPAIR_QUERY_SEMANTICS_SHA256
from nurse_rcl.v2.live.staged_case import require_plain_path

ToolKind = Literal["card", "verification"]
_Key = tuple[ToolKind, str]


class NormalToolKey(StrictFrozenModel):
    kind: ToolKind
    query_key: str = Field(pattern=r"^[a-zA-Z0-9_:-]{1,160}$")


class NormalToolBudgetEntry(StrictFrozenModel):
    schema_version: Literal["nurse.v2.normal-tool-budget-entry/1"] = (
        "nurse.v2.normal-tool-budget-entry/1"
    )
    authority: Literal["accounting_only"] = "accounting_only"
    sequence: int = Field(ge=0, le=16, strict=True)
    previous_sha256: SHA256 | None
    base_identity_sha256: SHA256
    phase: Literal["initial", "reserved", "completed"]
    query: NormalToolKey | None = None
    result_sha256: SHA256 | None = None
    entry_sha256: SHA256


@dataclass(frozen=True, slots=True)
class NormalToolReservation:
    status: Literal["reserved", "reused", "pending", "budget_exhausted"]
    origin: Literal["base", "semantic"]
    result_sha256: str | None


def _base_prefix(base: NormalQuerySession, pin: str) -> tuple[dict[_Key, str], str]:
    if type(base) is not NormalQuerySession or base.session_sha256 != pin or (
        base.semantics_sha256 != REPAIR_QUERY_SEMANTICS_SHA256
        or base.pending_selection is not None or base.pending_measurement is not None
        or base.failure_reason is not None or base.stop_reason is None
        or not 0 <= base.paid_count <= 8 or base.paid_count != len(base.steps)
        or len(base.submitted_request_keys) != base.paid_count
        or (base.stop_reason == "query_budget_exhausted") != (base.paid_count == 8)
        or _seal(base, "session_sha256") != base
    ):
        raise ValueError("normal tool budget requires the pinned complete base session")
    results: dict[_Key, str] = {}
    previous: tuple[str, ...] = ()
    for step in base.steps:
        key = step.feedback.request_key
        NormalToolKey(kind="verification", query_key=key)
        if (
            step.feedback.catalog_sha256 != base.catalog_sha256
            or step.selection.catalog_sha256 != base.catalog_sha256
            or step.selection.selected_request_key != key
            or step.selection.submitted_request_keys != previous
            or step.feedback.measurement_sha256 != step.measurement.measurement_sha256
            or key != "request_" + step.measurement.request_sha256
            or ("verification", key) in results
        ):
            raise ValueError("normal tool base prefix reference/count mismatch")
        results[("verification", key)] = step.feedback.result_sha256
        previous = tuple(sorted((*previous, key)))
    if previous != base.submitted_request_keys:
        raise ValueError("normal tool base submitted prefix differs")
    identity = sha256_hex(canonical_json_bytes(dict(
        base_session_sha256=pin, catalog_sha256=base.catalog_sha256,
        query_limit=8, base_results=sorted((k[0], k[1], v) for k, v in results.items()),
    )))
    return results, identity


class NormalToolBudget:
    """Single-owner accounting. Only a new `reserved` outcome permits execution."""

    def __init__(self, root: Path, base_results: dict[_Key, str], identity: str):
        self._root, self._base, self._identity = root, base_results, identity
        self._extra: dict[_Key, str | None] = {}
        self._sequence = 0
        self._previous: str | None = None
        self._closed = False

    @property
    def used_count(self) -> int:
        return len(self._base) + len(self._extra)

    @property
    def remaining_count(self) -> int:
        return 8 - self.used_count

    @property
    def records(self) -> tuple[tuple[ToolKind, str, str | None], ...]:
        """This branch's durable reservations, in consumption order."""
        return tuple((kind, key, result) for (kind, key), result in self._extra.items())

    def _alive(self) -> None:
        if self._closed:
            raise ValueError("normal tool budget is closed or poisoned")

    def _transition(self, entry: NormalToolBudgetEntry) -> dict[_Key, str | None]:
        if entry.base_identity_sha256 != self._identity:
            raise ValueError("normal tool budget base identity differs")
        if entry.sequence != self._sequence or entry.previous_sha256 != self._previous:
            raise ValueError("normal tool budget sequence differs")
        extra = dict(self._extra)
        if entry.phase == "initial":
            if self._sequence != 0 or entry.query is not None or entry.result_sha256 is not None:
                raise ValueError("normal tool budget invalid initial entry")
        else:
            if self._sequence == 0 or entry.query is None:
                raise ValueError("normal tool budget lacks initial identity/query")
            key = (entry.query.kind, entry.query.query_key)
            if key in self._base:
                raise ValueError("normal tool budget cannot overwrite a base result")
            if entry.phase == "reserved":
                if key in extra or self.used_count >= 8 or entry.result_sha256 is not None:
                    raise ValueError("normal tool duplicate reservation or exhausted budget")
                extra[key] = None
            else:
                if key not in extra or extra[key] is not None or entry.result_sha256 is None:
                    raise ValueError("normal tool completion lacks an unfinished reservation")
                extra[key] = entry.result_sha256
        return extra

    def _append(self, phase: Literal["initial", "reserved", "completed"], *,
                query: NormalToolKey | None = None, result_sha256: str | None = None) -> None:
        self._alive()
        entry = _seal(NormalToolBudgetEntry(
            sequence=self._sequence, previous_sha256=self._previous,
            base_identity_sha256=self._identity, phase=phase, query=query,
            result_sha256=result_sha256, entry_sha256="0" * 64,
        ), "entry_sha256")
        next_state = self._transition(entry)
        raw = canonical_json_bytes(entry.model_dump(mode="json"))
        descriptor, name = tempfile.mkstemp(prefix=".tool-", dir=self._root)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, self._root / f"{entry.sequence:03d}.json")
            _fsync(self._root)
            atomic_write_bytes(self._root / "head", canonical_json_bytes(
                [entry.sequence, entry.entry_sha256],
            ))
            _fsync(self._root)
        except BaseException:
            # Publication may already be durable. Reopen before doing any work.
            self._closed = True
            raise
        finally:
            temporary.unlink(missing_ok=True)
        self._extra, self._previous = next_state, entry.entry_sha256
        self._sequence += 1

    def _restore(self) -> None:
        files = tuple(sorted(self._root.glob("*.json")))
        if not 1 <= len(files) <= 17 or tuple(p.name for p in files) != tuple(
            f"{i:03d}.json" for i in range(len(files))
        ) or {p.name for p in self._root.iterdir()} - {
            "lock", "head", *(p.name for p in files),
        }:
            raise ValueError("normal tool journal is missing/noncontiguous/overcapacity")
        tails = []
        for path in files:
            raw = _read_bytes(path, 32768)
            _decode_json_member(raw)
            entry = NormalToolBudgetEntry.model_validate_json(raw)
            if raw != canonical_json_bytes(entry.model_dump(mode="json")) or (
                _seal(entry, "entry_sha256") != entry
            ):
                raise ValueError("normal tool journal pin differs")
            self._extra = self._transition(entry)
            self._previous = entry.entry_sha256
            self._sequence += 1
            tails.append(canonical_json_bytes([entry.sequence, entry.entry_sha256]))
        head = self._root / "head"
        if head.exists() or head.is_symlink():
            if _read_bytes(head, 256) not in tails[-2:]:
                raise ValueError("normal tool head indicates rollback/corruption")
        elif len(files) != 1:
            raise ValueError("normal tool head missing")

    def reserve(self, *, kind: ToolKind, query_key: str) -> NormalToolReservation:
        self._alive()
        query = NormalToolKey(kind=kind, query_key=query_key)
        key = (query.kind, query.query_key)
        if key in self._base:
            return NormalToolReservation("reused", "base", self._base[key])
        if key in self._extra:
            result = self._extra[key]
            return NormalToolReservation("pending" if result is None else "reused",
                                         "semantic", result)
        if self.remaining_count == 0:
            return NormalToolReservation("budget_exhausted", "semantic", None)
        self._append("reserved", query=query)
        return NormalToolReservation("reserved", "semantic", None)

    def complete(self, *, kind: ToolKind, query_key: str, result_sha256: str) -> None:
        self._alive()
        query = NormalToolKey(kind=kind, query_key=query_key)
        key = (query.kind, query.query_key)
        if key in self._extra and self._extra[key] is not None:
            if self._extra[key] != result_sha256:
                raise ValueError("normal tool completed result differs")
            return
        self._append("completed", query=query, result_sha256=result_sha256)


@contextmanager
def open_normal_tool_budget(
    *, root: Path, base_session: NormalQuerySession, expected_base_session_sha256: str,
) -> Iterator[NormalToolBudget]:
    """Admit one saved identity; no measurement/source/score recomputation or ARK."""
    results, identity = _base_prefix(base_session, expected_base_session_sha256)
    with _open_budget(root, results, identity) as ledger:
        yield ledger


@contextmanager
def open_normal_branch_budget(
    *, root: Path, paired_base_session: NormalQuerySession,
    expected_base_session_sha256: str, branch_identity_sha256: str,
) -> Iterator[NormalToolBudget]:
    """Independent eight-slot branch; a selected B0 reuse must still be reserved.

    Pairing fixes the comparison identity, but exposes neither inherited results
    nor free queries. The owner checks selected measurements before completing.
    """
    _, base_identity = _base_prefix(paired_base_session, expected_base_session_sha256)
    TypeAdapter(SHA256).validate_python(branch_identity_sha256)
    identity = sha256_hex(canonical_json_bytes(dict(
        schema_version="normal-early-semantic-branch-budget/1",
        paired_base_identity_sha256=base_identity,
        branch_identity_sha256=branch_identity_sha256, query_limit=8,
    )))
    with _open_budget(root, {}, identity) as ledger:
        yield ledger


@contextmanager
def _open_budget(root: Path, results: dict[_Key, str], identity: str) -> Iterator[NormalToolBudget]:
    require_plain_path(root.parent)
    try:
        root.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        created = False
    require_plain_path(root)
    lock = os.open(root / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    ledger = NormalToolBudget(root, results, identity)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if created:
            ledger._append("initial")
        else:
            ledger._restore()
        yield ledger
    finally:
        ledger._closed = True
        os.close(lock)
