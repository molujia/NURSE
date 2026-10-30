"""Opt-in /2 per-case INPUT ledger. No tokenizer, credentials or network access.

The trusted outbound owner must supply a verified input upper bound and bind its
counter to code/config identity. A counter hash is identity, NOT proof of accuracy.
This component is not live-enabled until the counter and new wire are admitted.
Lock order for integration: global single-in-flight lock, then this case lock.
Keep the case lock across reserve/send/receipt/settlement; never acquire global
transport locks from inside this component. Saved response recovery needs no key.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.exchange_json import ExchangeLimits, encode_json
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.repair_driver import _fsync
from nurse_rcl.v2.live.staged_case import require_plain_path

INPUT_CEILING = 100000
REPORT_INPUT_RESERVE = 36864


class InputBudgetError(RuntimeError):
    scope = "input"

    def __init__(self, code: str):
        self.code = code
        super().__init__(code + ":input")


class _Model(StrictFrozenModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class OpsInputRequest(_Model):
    phase: Literal["investigation", "report"]
    decision_index: int | None = Field(default=None, ge=1, le=8)
    protocol: Literal["nurse.v2.ops-decision-exchange/2", "nurse.v2.ops-report-exchange/2"]
    payload_sha256: SHA256

    @model_validator(mode="after")
    def consistent(self) -> Self:
        decision = self.phase == "investigation"
        if decision != (self.decision_index is not None) or self.protocol != (
            "nurse.v2.ops-decision-exchange/2" if decision else "nurse.v2.ops-report-exchange/2"
        ):
            raise ValueError("input request protocol or slot differs")
        return self

    @property
    def slot(self) -> tuple[str, int | None]:
        return self.phase, self.decision_index


class _Identity(_Model):
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    case_snapshot_id: SHA256
    execution_sha256: SHA256
    code_sha256: SHA256
    counter_sha256: SHA256


class OpsInputAttempt(_Model):
    request: OpsInputRequest
    number: int = Field(ge=1, le=5)
    input_upper_bound: int = Field(ge=1)
    state: Literal["reserved", "unknown", "settled"] = "reserved"
    response_sha256: SHA256 | None = None
    usage_json: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    usage_issues: tuple[str, ...] = ()

    @property
    def occupied(self) -> int:
        return self.input_upper_bound if self.input_tokens is None else self.input_tokens


class _State(_Model):
    schema_version: Literal["nurse.v2.ops-input-ledger/2"] = "nurse.v2.ops-input-ledger/2"
    identity: _Identity
    stage: Literal["investigation", "report"] = "investigation"
    attempts: tuple[OpsInputAttempt, ...] = Field(default=(), max_length=45)
    last_denial: Literal["token_budget", "input_counter_contract"] | None = None
    state_sha256: SHA256


def _usage_fields(value: object) -> dict[str, Any]:
    """Independently parse usage parts; do not infer input from output/total.

Cached tokens are an input subset, not an extra debit. Conflicting input aliases
or an impossible cache subset cannot release a reservation. Invalid output or
total does not poison an independently valid input count.
"""
    usage = value if type(value) is dict else {}
    issues: list[str] = []

    def part(mapping: dict[str, Any], names: tuple[str, ...], label: str) -> int | None:
        values = [mapping[k] for k in names if k in mapping]
        if not values:
            issues.append(label + "_missing")
        elif any(type(v) is not int or v < 0 for v in values):
            issues.append(label + "_invalid")
        elif len(set(values)) != 1:
            issues.append(label + "_conflict")
        else:
            return int(values[0])
        return None

    input_count = part(usage, ("prompt_tokens", "input_tokens"), "input")
    output_count = part(usage, ("completion_tokens", "output_tokens"), "output")
    total_count = part(usage, ("total_tokens",), "total")
    cached_values = {}
    for key in ("prompt_tokens_details", "input_tokens_details"):
        if key not in usage:
            continue
        details = usage[key]
        if type(details) is not dict:
            issues.append("cached_invalid")
        elif "cached_tokens" in details:
            cached_values[key] = details["cached_tokens"]
    cached_count = part(cached_values, tuple(cached_values), "cached")
    if input_count is not None and cached_count is not None and cached_count > input_count:
        issues.append("cached_exceeds_input")
    if any(code in issues for code in (
        "cached_invalid", "cached_conflict", "cached_exceeds_input",
    )):
        input_count = None
    if input_count is not None and output_count is not None and total_count is not None and (
        total_count != input_count + output_count
    ):
        issues.append("total_inconsistent")
    return dict(input_tokens=input_count, output_tokens=output_count, total_tokens=total_count,
                cached_tokens=cached_count, usage_issues=tuple(sorted(set(issues))))


class OpsInputBudget:
    """Durable case balance, independent from action/attempt success semantics.

reserve() is permission for one NEW attempt, never an idempotent send ticket.
After interruption a reserved attempt becomes unknown and remains occupied;
only an independently saved response can settle it. Callers must reconcile their
request ledger before reserving a later number. This class cannot reset a run.
"""

    def __init__(
        self, *, root: Path, opaque_case_id: str, case_snapshot_id: str,
        execution_sha256: str, code_sha256: str, counter_sha256: str,
    ):
        self._parent = root
        self._root = root / "input-budget"
        self._identity = _Identity(
            opaque_case_id=opaque_case_id, case_snapshot_id=case_snapshot_id,
            execution_sha256=execution_sha256, code_sha256=code_sha256,
            counter_sha256=counter_sha256,
        )
        self._state: _State | None = None
        self._lock: int | None = None

    def __enter__(self) -> OpsInputBudget:
        if self._lock is not None:
            raise ValueError("input budget already open")
        require_plain_path(self._parent)
        created = False
        try:
            self._root.mkdir(mode=0o700)
            created = True
        except FileExistsError:
            pass
        require_plain_path(self._root)
        self._lock = os.open(self._root / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if created:
                # Persist the new directory entry before any send can be admitted.
                _fsync(self._parent)
            path = self._root / "state.json"
            if path.exists() or path.is_symlink():
                raw = _read_bytes(path, 4 * 1024 * 1024)
                state = _State.model_validate_json(raw)
                if state.identity != self._identity:
                    raise ValueError("input budget identity differs")
                unsigned = state.model_dump(mode="json", exclude={"state_sha256"})
                if sha256_hex(canonical_json_bytes(unsigned)) != state.state_sha256 or (
                    canonical_json_bytes(state.model_dump(mode="json")) != raw
                ):
                    raise ValueError("input budget content digest differs")
                self._validate_state(state)
                self._state = state
                if any(a.state == "reserved" for a in state.attempts):
                    self._persist(state.model_copy(update={"attempts": tuple(
                        a.model_copy(update={"state": "unknown"}) if a.state == "reserved" else a
                        for a in state.attempts
                    )}))
            elif created:
                self._persist(_State(identity=self._identity, state_sha256="0" * 64))
            else:
                raise ValueError("existing input budget state missing; cannot reset")
            return self
        except BaseException:
            os.close(self._lock)
            self._lock, self._state = None, None
            raise

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._lock is not None:
            os.close(self._lock)
        self._lock, self._state = None, None

    def _current(self) -> _State:
        if self._lock is None or self._state is None:
            raise ValueError("input budget is closed")
        return self._state

    @property
    def attempts(self) -> tuple[OpsInputAttempt, ...]:
        return self._current().attempts

    @staticmethod
    def _validate_state(state: _State) -> None:
        seen: dict[tuple[str, int | None], list[OpsInputAttempt]] = {}
        for row in state.attempts:
            earlier = seen.setdefault(row.request.slot, [])
            if row.number != len(earlier) + 1 or any(a.request != row.request for a in earlier):
                raise ValueError("input attempt sequence or request binding differs")
            earlier.append(row)
            if row.request.phase == "report" and state.stage != "report":
                raise ValueError("report reservation before transition")
            if row.state == "settled":
                if row.response_sha256 is None or row.usage_json is None:
                    raise ValueError("input settlement receipt missing")
                fields = _usage_fields(json.loads(row.usage_json))
                if any(getattr(row, k) != v for k, v in fields.items()):
                    raise ValueError("input settlement usage differs")
            elif any(getattr(row, k) is not None for k in (
                "response_sha256", "usage_json", "input_tokens", "output_tokens",
                "total_tokens", "cached_tokens",
            )) or row.usage_issues:
                raise ValueError("unsettled input has usage")
        if sum(a.state == "reserved" for a in state.attempts) > 1:
            raise ValueError("multiple in-flight input reservations")

    def _persist(self, state: _State) -> None:
        if self._lock is None:
            raise ValueError("input budget is closed")
        self._validate_state(state)
        unsigned = state.model_dump(mode="json", exclude={"state_sha256"})
        digest = sha256_hex(canonical_json_bytes(unsigned))
        state = state.model_copy(update={"state_sha256": digest})
        raw = encode_json(
            state.model_dump(mode="json"), ExchangeLimits(4 * 1024 * 1024, 64, 100000),
        )
        try:
            atomic_write_bytes(self._root / "state.json", raw)
            _fsync(self._root)
        except BaseException:
            # Replacement may already have succeeded. Never overwrite that state
            # from stale memory; keep the lock until exit and require reopening.
            self._state = None
            raise
        self._state = state

    def summary(self) -> dict[str, Any]:
        state = self._current()
        rows = state.attempts
        occupied = sum(a.occupied for a in rows)
        reserve = REPORT_INPUT_RESERVE if state.stage == "investigation" else 0
        failed = any(
            a.input_tokens is not None and a.input_tokens > a.input_upper_bound for a in rows
        )
        return dict(
            scope="input", input_ceiling=INPUT_CEILING, stage=state.stage, attempts=len(rows),
            occupied_input_tokens=occupied,
            reported_input_tokens=sum(a.input_tokens for a in rows if a.input_tokens is not None),
            unknown_input_reserved_tokens=sum(a.input_upper_bound for a in rows
                if a.state != "reserved" and a.input_tokens is None),
            in_flight_input_reserved_tokens=sum(a.input_upper_bound for a in rows
                                                if a.state == "reserved"),
            report_input_reserve=reserve,
            available_input_tokens=max(0, INPUT_CEILING - occupied - reserve),
            input_counter_contract_failed=failed, last_denial=state.last_denial,
            reported_output_tokens=sum(
                a.output_tokens for a in rows if a.output_tokens is not None
            ),
            unknown_output_attempts=sum(a.output_tokens is None for a in rows),
            reported_total_tokens=sum(a.total_tokens for a in rows if a.total_tokens is not None),
            unknown_total_attempts=sum(a.total_tokens is None for a in rows),
        )

    def begin_report(self) -> None:
        state = self._current()
        if any(a.state == "reserved" for a in state.attempts):
            raise ValueError("cannot transition with in-flight input")
        if state.stage != "report":
            self._persist(state.model_copy(update={"stage": "report", "last_denial": None}))

    def _rows(self, request: OpsInputRequest) -> list[OpsInputAttempt]:
        if type(request) is not OpsInputRequest:
            raise ValueError("invalid input request")
        OpsInputRequest.model_validate_json(request.model_dump_json())
        rows = [a for a in self._current().attempts if a.request.slot == request.slot]
        if any(a.request != request for a in rows):
            raise ValueError("input request binding differs")
        return rows

    def reserve(self, request: OpsInputRequest, *, number: int, input_upper_bound: int) -> None:
        state = self._current()
        rows = self._rows(request)
        if type(number) is not int or not 1 <= number <= 5:
            raise ValueError("invalid input attempt number")
        if type(input_upper_bound) is not int or input_upper_bound <= 0:
            raise ValueError("invalid input bound")
        if number <= len(rows):
            raise ValueError("input attempt already reserved; cannot resend")
        if number != len(rows) + 1:
            raise ValueError("input attempt sequence differs")
        if request.phase == "investigation" and state.stage == "report":
            raise ValueError("investigation closed")
        if request.phase == "report" and state.stage != "report":
            raise ValueError("report not opened")
        if any(a.state == "reserved" for a in state.attempts):
            raise ValueError("another input attempt is in flight")
        summary = self.summary()
        reason = "input_counter_contract" if summary["input_counter_contract_failed"] else (
            "token_budget" if input_upper_bound > summary["available_input_tokens"] else None
        )
        if reason is not None:
            self._persist(state.model_copy(update={"last_denial": reason}))
            raise InputBudgetError(reason)
        row = OpsInputAttempt(request=request, number=number, input_upper_bound=input_upper_bound)
        self._persist(state.model_copy(update={"attempts": (*state.attempts, row),
                                              "last_denial": None}))

    def _find(self, request: OpsInputRequest, number: int) -> OpsInputAttempt:
        rows = self._rows(request)
        if type(number) is not int or not 1 <= number <= len(rows):
            raise ValueError("input attempt reservation missing")
        return rows[number - 1]

    def _replace(self, old: OpsInputAttempt, new: OpsInputAttempt) -> None:
        state = self._current()
        self._persist(state.model_copy(update={"attempts": tuple(
            new if a.request.slot == old.request.slot and a.number == old.number else a
            for a in state.attempts
        )}))

    def mark_unknown(self, request: OpsInputRequest, *, number: int) -> None:
        row = self._find(request, number)
        if row.state == "settled":
            raise ValueError("cannot forget settled input usage")
        if row.state == "reserved":
            self._replace(row, row.model_copy(update={"state": "unknown"}))

    def settle(
        self, request: OpsInputRequest, *, number: int, usage: object, response_sha256: str,
    ) -> None:
        row = self._find(request, number)
        # Bound usage METADATA only, not provider-generated text or output tokens.
        raw = encode_json(usage, ExchangeLimits(max_bytes=65536)).decode()
        changed = OpsInputAttempt.model_validate_json(canonical_json_bytes({
            **row.model_dump(mode="json"), **_usage_fields(usage), "state": "settled",
            "response_sha256": response_sha256, "usage_json": raw,
        }))
        if row.state == "settled":
            if changed != row:
                raise ValueError("input settlement differs")
            return
        self._replace(row, changed)
