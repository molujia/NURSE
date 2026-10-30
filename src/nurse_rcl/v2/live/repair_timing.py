"""Explicit case timing policies, not source authority; default work is method."""

from __future__ import annotations

import math
import socket
import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, Protocol, TypeVar

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.staged_case import require_plain_path

TIMING_POLICY = "live-repair-phase-budgets/1"
DIAGNOSTIC_TIMING_POLICY = "live-repair-diagnostic-timing/1"
DEV_TIMING_POLICY = "live-dev-shared-case-timing/1"
PREPARATION_SECONDS = 3600
METHOD_SECONDS = 600
CASE_SECONDS = PREPARATION_SECONDS + METHOD_SECONDS
_REASONS = {
    "input_validation": "preparation",
    "raw_input_prepare": "preparation",
    "raw_input_exit": "preparation",
    "catalog_prepare": "preparation",
    "catalog_exit": "preparation",
    "stage_input_exit": "preparation",
    "orchestration": "method",
    "query_execute": "method",
    "score_execute": "method",
    "report_execute": "method",
}


class CaseBudgetExpired(TimeoutError):
    """The active case account or overall wall-clock budget is exhausted."""


def _limits(policy: str) -> tuple[int, int | None, int | None]:
    if type(policy) is not str:
        raise ValueError("invalid timing policy")
    if policy == TIMING_POLICY:
        return PREPARATION_SECONDS, METHOD_SECONDS, CASE_SECONDS
    if policy in (DIAGNOSTIC_TIMING_POLICY, DEV_TIMING_POLICY):
        return PREPARATION_SECONDS, None, None
    raise ValueError("unsupported timing policy")


def _time(now: float) -> float:
    if type(now) is not float or not math.isfinite(now) or now < 0:
        raise ValueError("invalid phase clock time")
    return now


def _reason(account: str, reason: str) -> None:
    if type(account) is not str or type(reason) is not str or _REASONS.get(reason) != account:
        raise ValueError("invalid timing account or reason")


class CasePhaseClock:
    def __init__(self, now: float, *, policy: str = TIMING_POLICY):
        _limits(policy)
        self.policy = policy
        self.started = self.last = _time(now)
        self.account = "method"
        self.reason = "orchestration"
        self._used = {"preparation": 0.0, "method": 0.0}
        self.active = True

    @classmethod
    def start(cls, *, now: float, policy: str = TIMING_POLICY) -> CasePhaseClock:
        return cls(now, policy=policy)

    @property
    def used_seconds(self) -> dict[str, float]:
        return dict(self._used)

    @property
    def remaining_seconds(self) -> float | None:
        preparation, method, whole = _limits(self.policy)
        limit = preparation if self.account == "preparation" else method
        remaining = []
        if limit is not None:
            remaining.append(limit - self._used[self.account])
        if whole is not None:
            remaining.append(whole - (self.last - self.started))
        return max(0.0, min(remaining)) if remaining else None

    def checkpoint(self, *, now: float) -> None:
        value = _time(now)
        if value < self.last:
            raise ValueError("phase clock moved backwards")
        if self.active:
            self._used[self.account] += value - self.last
        self.last = value
        preparation, method, whole = _limits(self.policy)
        for account, limit in (("preparation", preparation), ("method", method)):
            if limit is not None and self._used[account] >= limit:
                raise CaseBudgetExpired(account + "_timeout")
        if whole is not None and value - self.started >= whole:
            raise CaseBudgetExpired("case_timeout")

    def switch(self, account: str, *, reason: str, now: float) -> None:
        _reason(account, reason)
        self.checkpoint(now=now)
        self.account, self.reason = account, reason

    def summary(self) -> dict[str, Any]:
        preparation, method, _ = _limits(self.policy)
        result = dict(
            policy=self.policy,
            account=self.account,
            phase_reason=self.reason,
            preparation_seconds=self._used["preparation"],
            method_seconds=self._used["method"],
            end_to_end_seconds=self.last - self.started,
            preparation_remaining=max(0.0, preparation - self._used["preparation"]),
            method_remaining=None if method is None else max(0.0, method - self._used["method"]),
        )
        if self.policy == DEV_TIMING_POLICY:
            result.update(active=self.active, method_warning=self._used["method"] > 600,
                          end_to_end_seconds=sum(self._used.values()))
        return result


class TimingJournal:
    """Immutable parent-clock events. Same-boot recovery charges the open interval.

    The Linux monotonic epoch survives a process restart, not a machine reboot.
    Unknown epochs fail closed. No caller-provided elapsed value is trusted.
    """

    def __init__(self, root: Path, identity: str, boot_id: str, clock: CasePhaseClock):
        self.root, self.identity, self.boot_id, self.clock = root, identity, boot_id, clock
        self.sequence = 0
        self.previous = "0" * 64

    @classmethod
    def open(
        cls, root: Path, *, identity: str, now: float, boot_id: str,
        expected_policy: str = TIMING_POLICY,
    ) -> TimingJournal:
        _limits(expected_policy)
        if (
            type(identity) is not str
            or len(identity) != 64
            or any(c not in "0123456789abcdef" for c in identity)
            or type(boot_id) is not str
            or not boot_id
            or len(boot_id) > 128
        ):
            raise ValueError("invalid timing identity or clock epoch")
        _time(now)
        require_plain_path(root.parent)
        if root.is_symlink():
            raise ValueError("timing journal may not be a symlink")
        if root.exists() and not root.is_dir():
            raise ValueError("timing journal must be a directory")
        root.mkdir(mode=0o700, exist_ok=True)
        root = require_plain_path(root)
        paths = sorted(root.iterdir())
        if not paths:
            fresh = cls(
                root, identity, boot_id, CasePhaseClock.start(now=now, policy=expected_policy)
            )
            fresh._append("start", "method", "orchestration", now)
            return fresh
        result = cls._read_existing(
            root, identity=identity, boot_id=boot_id, expected_policy=expected_policy
        )
        result.checkpoint(now=now)
        return result

    @classmethod
    def _read_existing(
        cls, root: Path, *, identity: str, boot_id: str, expected_policy: str = TIMING_POLICY,
    ) -> TimingJournal:
        """Read-only prefix reconstruction; no new balance or clock event."""
        return cls._reconstruct(
            root, identity=identity, boot_id=boot_id, retain_expired=False,
            expected_policy=expected_policy,
        )[0]

    @classmethod
    def _inspect_expired(
        cls, root: Path, *, identity: str, boot_id: str, expected_policy: str = TIMING_POLICY,
    ) -> dict[str, Any]:
        """Historical evidence only; returns no resumable clock or new allowance."""
        result, expired = cls._reconstruct(
            root, identity=identity, boot_id=boot_id, retain_expired=True,
            expected_policy=expected_policy,
        )
        if expired is None:
            raise ValueError("historical failed clock has no exhausted budget")
        return dict(
            identity=result.identity,
            final_head=result.previous,
            events=result.sequence,
            expired_reason=expired,
            timing=result.clock.summary(),
        )

    @classmethod
    def _reconstruct(
        cls, root: Path, *, identity: str, boot_id: str, retain_expired: bool,
        expected_policy: str = TIMING_POLICY,
    ) -> tuple[TimingJournal, str | None]:
        from nurse_rcl.v2.live.repair_driver import read_json

        _limits(expected_policy)
        paths = sorted(require_plain_path(root).iterdir())
        result: TimingJournal | None = None
        expired: str | None = None
        for sequence, path in enumerate(paths):
            if path.name != f"event-{sequence:08d}.json" or path.is_symlink():
                raise ValueError("timing journal sequence or representation differs")
            row = read_json(path, limit=4096)
            if retain_expired and path.read_bytes() != canonical_json_bytes(row):
                raise ValueError("historical timing record is not canonical")
            body = {k: v for k, v in row.items() if k != "sha256"}
            if (
                set(row)
                != {
                    "identity",
                    "boot_id",
                    "policy",
                    "sequence",
                    "previous",
                    "action",
                    "account",
                    "reason",
                    "now",
                    "sha256",
                }
                or row["identity"] != identity
                or row["boot_id"] != boot_id
                or row["policy"] != expected_policy
                or type(row["sequence"]) is not int
                or row["sequence"] != sequence
                or row["sha256"] != sha256_hex(canonical_json_bytes(body))
            ):
                raise ValueError("timing journal identity or integrity differs")
            if result is None:
                if (row["action"], row["account"], row["reason"], row["previous"]) != (
                    "start",
                    "method",
                    "orchestration",
                    "0" * 64,
                ):
                    raise ValueError("invalid timing journal start")
                result = cls(
                    root, identity, boot_id,
                    CasePhaseClock.start(now=row["now"], policy=expected_policy),
                )
            else:
                if row["previous"] != result.previous:
                    raise ValueError("timing journal chain differs")
                if expired is not None and row["action"] != "checkpoint":
                    raise ValueError("historical clock switches after budget exhaustion")
                try:
                    result._apply(row["action"], row["account"], row["reason"], row["now"])
                except CaseBudgetExpired as error:
                    if not retain_expired:
                        raise
                    expired = str(error)
            result.previous, result.sequence = row["sha256"], sequence + 1
        if result is None:
            raise ValueError("timing journal is incomplete")
        return result, expired

    def _apply(self, action: str, account: str, reason: str, now: float) -> None:
        if self.clock.policy == DEV_TIMING_POLICY and action in ("pause", "resume"):
            if (account, reason) != ("method", "orchestration") or (
                self.clock.account != "method" or self.clock.active != (action == "pause")
            ):
                raise ValueError("invalid Dev clock pause/resume")
            self.clock.checkpoint(now=now)
            self.clock.active = action == "resume"
        elif self.clock.policy == DEV_TIMING_POLICY and not self.clock.active:
            raise ValueError("Dev clock must resume before charging work")
        elif action == "switch":
            self.clock.switch(account, reason=reason, now=now)
        elif action == "checkpoint" and (account, reason) == (
            self.clock.account,
            self.clock.reason,
        ):
            self.clock.checkpoint(now=now)
        else:
            raise ValueError("invalid timing journal transition")

    def _append(self, action: str, account: str, reason: str, now: float) -> None:
        from nurse_rcl.v2.live.repair_driver import publish_json

        body = dict(
            identity=self.identity,
            boot_id=self.boot_id,
            policy=self.clock.policy,
            sequence=self.sequence,
            previous=self.previous,
            action=action,
            account=account,
            reason=reason,
            now=now,
        )
        digest = sha256_hex(canonical_json_bytes(body))
        publish_json(self.root / f"event-{self.sequence:08d}.json", {**body, "sha256": digest})
        self.previous, self.sequence = digest, self.sequence + 1

    def _record(self, action: str, account: str, reason: str, now: float) -> None:
        try:
            self._apply(action, account, reason, now)
        except CaseBudgetExpired:
            self._append(action, account, reason, now)
            raise
        self._append(action, account, reason, now)

    def checkpoint(self, *, now: float) -> None:
        self._record("checkpoint", self.clock.account, self.clock.reason, now)

    def switch(self, account: str, *, reason: str, now: float) -> None:
        self._record("switch", account, reason, now)


class _PhaseSink(Protocol):
    account: str
    reason: str

    def switch(self, account: str, *, reason: str) -> None: ...


class PhaseClient:
    """Child endpoint for a private parent-owned SOCK_SEQPACKET pair.

    This carries accounting events only. No input object or source trust crosses
    it, and arbitrary stdout/model text is never parsed as control messages.
    """

    def __init__(self, fd: int):
        self.channel = socket.socket(fileno=fd)
        self.account, self.reason = "method", "orchestration"
        self.sequence = 0
        self.finished = False

    def _exchange(self, action: str, account: str, reason: str) -> None:
        from nurse_rcl.v2.exchange_json import ExchangeLimits, decode_json_object

        if self.finished:
            raise ValueError("timing channel already finished")
        _reason(account, reason)
        request = dict(sequence=self.sequence, action=action, account=account, reason=reason)
        self.channel.sendall(canonical_json_bytes(request))
        reply = self.channel.recv(1025)
        if len(reply) > 1024 or decode_json_object(reply, ExchangeLimits(1024, 4, 32)) != {
            "sequence": self.sequence
        }:
            raise ValueError("timing acknowledgement differs")
        self.sequence += 1
        self.account, self.reason = account, reason

    def switch(self, account: str, *, reason: str) -> None:
        self._exchange("switch", account, reason)

    def finish(self) -> None:
        if self.account != "method":
            raise ValueError("cannot finish inside an input scope")
        self._exchange("finish", "method", "orchestration")
        self.finished = True
        self.channel.close()


_CLIENT: ContextVar[_PhaseSink | None] = ContextVar("repair_case_phase_client", default=None)


@contextmanager
def _use_phase_client(client: _PhaseSink) -> Iterator[None]:
    if _CLIENT.get() is not None:
        raise ValueError("cannot replace an active timing owner")
    token = _CLIENT.set(client)
    try:
        yield
    finally:
        _CLIENT.reset(token)


@contextmanager
def _phase(account: str, reason: str) -> Iterator[None]:
    _reason(account, reason)
    client = _CLIENT.get()
    if client is None:
        # Standalone public APIs retain their original behavior and checks.
        # A supervised worker installs its mandatory endpoint before calling them.
        yield
        return
    previous = client.account, client.reason
    client.switch(account, reason=reason)
    primary: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            client.switch(previous[0], reason=previous[1])
        except BaseException as exc:
            if primary is None:
                raise
            from nurse_rcl.v2.live.ops_failure import retain_secondary

            retain_secondary(primary, "timing_finish", exc)


_P = ParamSpec("_P")
_T = TypeVar("_T")


def _preparation_call(reason: str) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    _reason("preparation", reason)

    def decorate(function: Callable[_P, _T]) -> Callable[_P, _T]:
        @wraps(function)
        def invoke(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            pending = True

            def cleanup() -> None:
                if pending and reason == "stage_input_exit":
                    function(*args, **kwargs)

            with ExitStack() as fallback:
                fallback.callback(cleanup)
                with _phase("preparation", reason):
                    pending = False
                    return function(*args, **kwargs)

        return invoke

    return decorate


def _preparation_scope(
    enter_reason: str,
    exit_reason: str,
) -> Callable[[Callable[_P, AbstractContextManager[_T]]], Callable[_P, AbstractContextManager[_T]]]:
    """Time a context's actual enter/exit, never its yielded consumer body.

    ExitStack remains the outer fallback: failure of timing notification cannot
    prevent established input guards from running on exceptional exit.
    """
    _reason("preparation", enter_reason)
    _reason("preparation", exit_reason)

    def decorate(
        factory: Callable[_P, AbstractContextManager[_T]],
    ) -> Callable[_P, AbstractContextManager[_T]]:
        @wraps(factory)
        @contextmanager
        def invoke(*args: _P.args, **kwargs: _P.kwargs) -> Iterator[_T]:
            with ExitStack() as owners:
                with _phase("preparation", enter_reason):
                    value = owners.enter_context(factory(*args, **kwargs))
                try:
                    yield value
                finally:
                    exception = sys.exc_info()
                    with _phase("preparation", exit_reason):
                        owners.__exit__(*exception)

        return invoke

    return decorate
