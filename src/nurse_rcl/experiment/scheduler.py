"""Bounded two-stage causal-wave scheduling with atomic group barriers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.agents.retry import CircuitBreaker
from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel

SCHEDULER_SCHEMA_VERSION: Literal[1] = 1
Sha256Hex = str
TerminalStatus = Literal["success", "degraded", "failed"]
ShutdownReason = Literal["completed", "interrupted", "circuit_open"]


class ScheduledCase(StrictFrozenModel):
    """The identity and causal timestamp needed by the scheduler."""

    case_id: str = Field(min_length=1)
    t0: int = Field(ge=0)


class StageAResult(StrictFrozenModel):
    """Case-local output that is safe to carry into one causal group."""

    case_id: str = Field(min_length=1)
    t0: int = Field(ge=0)
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class TerminalCaseResult(StrictFrozenModel):
    """Terminal Stage-B output bound to the exact history snapshot used."""

    case_id: str = Field(min_length=1)
    t0: int = Field(ge=0)
    history_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: TerminalStatus
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class GroupCommitReceipt(StrictFrozenModel):
    """Content-addressed proof that one complete timestamp group was committed."""

    schema_version: Literal[1] = SCHEDULER_SCHEMA_VERSION
    t0: int = Field(ge=0)
    history_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    terminal_artifact_hashes: tuple[str, ...]
    commit_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"receipt_hash"})
        return self.receipt_hash == sha256_hex(canonical_json_bytes(unsigned))


class SchedulingResult(StrictFrozenModel):
    """Immutable schedule result suitable for checkpoint and shutdown auditing."""

    schema_version: Literal[1] = SCHEDULER_SCHEMA_VERSION
    stage_a_results: tuple[StageAResult, ...]
    terminal_results: tuple[TerminalCaseResult, ...]
    group_receipts: tuple[GroupCommitReceipt, ...]
    shutdown_reason: ShutdownReason
    circuit_reason: str | None = None
    result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_consistent_shutdown_reason(self) -> Self:
        if self.shutdown_reason == "circuit_open" and not self.circuit_reason:
            raise ValueError("circuit shutdown requires a circuit reason")
        if self.shutdown_reason != "circuit_open" and self.circuit_reason is not None:
            raise ValueError("circuit reason is valid only for circuit shutdown")
        return self

    def verify(self) -> bool:
        if any(not receipt.verify() for receipt in self.group_receipts):
            return False
        if self.stage_a_results != tuple(
            sorted(self.stage_a_results, key=_case_order)
        ):
            return False
        if self.terminal_results != tuple(
            sorted(self.terminal_results, key=_case_order)
        ):
            return False
        if tuple(receipt.t0 for receipt in self.group_receipts) != tuple(
            sorted(receipt.t0 for receipt in self.group_receipts)
        ):
            return False
        for receipt in self.group_receipts:
            group = tuple(
                item for item in self.terminal_results if item.t0 == receipt.t0
            )
            if (
                not group
                or any(
                    item.history_snapshot_hash != receipt.history_snapshot_hash
                    for item in group
                )
                or tuple(item.artifact_hash for item in group)
                != receipt.terminal_artifact_hashes
            ):
                return False
        unsigned = self.model_dump(mode="json", exclude={"result_hash"})
        return self.result_hash == sha256_hex(canonical_json_bytes(unsigned))


def run_causal_wave_schedule(
    *,
    cases: Sequence[ScheduledCase],
    case_workers: int,
    stage_a: Callable[[ScheduledCase], StageAResult],
    history_snapshot: Callable[[int], str],
    stage_b: Callable[[StageAResult, str], TerminalCaseResult],
    commit_group: Callable[
        [int, str, tuple[TerminalCaseResult, ...]],
        str,
    ],
    circuit_breaker: CircuitBreaker,
    stop_requested: Callable[[], bool] = lambda: False,
) -> SchedulingResult:
    """Run case-local work, then chronological history-isolated timestamp groups."""

    if case_workers < 1:
        raise ValueError("case_workers must be at least one")
    scheduled = tuple(cases)
    _require_unique_case_ids(scheduled)

    with ThreadPoolExecutor(max_workers=case_workers) as executor:
        stage_a_results = tuple(executor.map(stage_a, scheduled))
    _require_stage_a_identity(scheduled=scheduled, results=stage_a_results)

    return run_causal_wave_stage_b_schedule(
        stage_a_results=stage_a_results,
        case_workers=case_workers,
        history_snapshot=history_snapshot,
        stage_b=stage_b,
        commit_group=commit_group,
        circuit_breaker=circuit_breaker,
        stop_requested=stop_requested,
    )


def run_causal_wave_stage_b_schedule(
    *,
    stage_a_results: Sequence[StageAResult],
    case_workers: int,
    history_snapshot: Callable[[int], str],
    stage_b: Callable[[StageAResult, str], TerminalCaseResult],
    commit_group: Callable[
        [int, str, tuple[TerminalCaseResult, ...]],
        str,
    ],
    circuit_breaker: CircuitBreaker,
    stop_requested: Callable[[], bool] = lambda: False,
) -> SchedulingResult:
    """Run only chronological Stage B from validated, precomputed Stage A results."""

    if case_workers < 1:
        raise ValueError("case_workers must be at least one")
    stage_a_values = tuple(stage_a_results)
    _require_unique_stage_a_ids(stage_a_values)
    ordered_stage_a = tuple(sorted(stage_a_values, key=_case_order))

    terminal_results: list[TerminalCaseResult] = []
    receipts: list[GroupCommitReceipt] = []
    shutdown_reason: ShutdownReason = "completed"
    circuit_reason: str | None = None

    for t0 in sorted({item.t0 for item in ordered_stage_a}):
        if circuit_breaker.is_open:
            shutdown_reason = "circuit_open"
            circuit_reason = circuit_breaker.reason
            break
        if stop_requested():
            shutdown_reason = "interrupted"
            break

        history_hash = history_snapshot(t0)
        group = tuple(item for item in ordered_stage_a if item.t0 == t0)
        with ThreadPoolExecutor(max_workers=case_workers) as executor:
            futures = tuple(
                executor.submit(stage_b, item, history_hash) for item in group
            )
            terminal_group = tuple(future.result() for future in futures)
        _require_terminal_identity(
            stage_a_group=group,
            terminal_group=terminal_group,
            history_hash=history_hash,
        )
        ordered_terminal_group = tuple(sorted(terminal_group, key=_case_order))
        commit_hash = commit_group(t0, history_hash, ordered_terminal_group)
        receipt = _group_receipt(
            t0=t0,
            history_hash=history_hash,
            terminal_group=ordered_terminal_group,
            commit_hash=commit_hash,
        )
        terminal_results.extend(ordered_terminal_group)
        receipts.append(receipt)

        if circuit_breaker.is_open:
            shutdown_reason = "circuit_open"
            circuit_reason = circuit_breaker.reason
            break
        if stop_requested():
            shutdown_reason = "interrupted"
            break

    return _scheduling_result(
        stage_a_results=ordered_stage_a,
        terminal_results=tuple(terminal_results),
        group_receipts=tuple(receipts),
        shutdown_reason=shutdown_reason,
        circuit_reason=circuit_reason,
    )


def _group_receipt(
    *,
    t0: int,
    history_hash: str,
    terminal_group: tuple[TerminalCaseResult, ...],
    commit_hash: str,
) -> GroupCommitReceipt:
    unsigned = {
        "schema_version": SCHEDULER_SCHEMA_VERSION,
        "t0": t0,
        "history_snapshot_hash": history_hash,
        "terminal_artifact_hashes": [item.artifact_hash for item in terminal_group],
        "commit_hash": commit_hash,
    }
    receipt = GroupCommitReceipt(
        t0=t0,
        history_snapshot_hash=history_hash,
        terminal_artifact_hashes=tuple(
            item.artifact_hash for item in terminal_group
        ),
        commit_hash=commit_hash,
        receipt_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not receipt.verify():  # pragma: no cover - construction invariant
        raise AssertionError("group commit receipt verification failed")
    return receipt


def _scheduling_result(
    *,
    stage_a_results: tuple[StageAResult, ...],
    terminal_results: tuple[TerminalCaseResult, ...],
    group_receipts: tuple[GroupCommitReceipt, ...],
    shutdown_reason: ShutdownReason,
    circuit_reason: str | None,
) -> SchedulingResult:
    unsigned = {
        "schema_version": SCHEDULER_SCHEMA_VERSION,
        "stage_a_results": [item.model_dump(mode="json") for item in stage_a_results],
        "terminal_results": [
            item.model_dump(mode="json") for item in terminal_results
        ],
        "group_receipts": [
            item.model_dump(mode="json") for item in group_receipts
        ],
        "shutdown_reason": shutdown_reason,
        "circuit_reason": circuit_reason,
    }
    result = SchedulingResult(
        stage_a_results=stage_a_results,
        terminal_results=terminal_results,
        group_receipts=group_receipts,
        shutdown_reason=shutdown_reason,
        circuit_reason=circuit_reason,
        result_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("scheduling result verification failed")
    return result


def _require_unique_case_ids(cases: tuple[ScheduledCase, ...]) -> None:
    case_ids = tuple(item.case_id for item in cases)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("scheduled case identities must be unique")


def _require_unique_stage_a_ids(results: tuple[StageAResult, ...]) -> None:
    case_ids = tuple(item.case_id for item in results)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Stage A result identities must be unique")


def _require_stage_a_identity(
    *,
    scheduled: tuple[ScheduledCase, ...],
    results: tuple[StageAResult, ...],
) -> None:
    expected = {(item.case_id, item.t0) for item in scheduled}
    observed = {(item.case_id, item.t0) for item in results}
    if len(results) != len(scheduled) or observed != expected:
        raise ValueError("Stage A results do not match the scheduled cases")


def _require_terminal_identity(
    *,
    stage_a_group: tuple[StageAResult, ...],
    terminal_group: tuple[TerminalCaseResult, ...],
    history_hash: str,
) -> None:
    expected = {(item.case_id, item.t0) for item in stage_a_group}
    observed = {(item.case_id, item.t0) for item in terminal_group}
    if len(terminal_group) != len(stage_a_group) or observed != expected:
        raise ValueError("terminal results do not match the current timestamp group")
    if any(item.history_snapshot_hash != history_hash for item in terminal_group):
        raise ValueError("terminal result is bound to a different history snapshot")


def _case_order(item: StageAResult | TerminalCaseResult) -> tuple[int, str]:
    return item.t0, item.case_id
