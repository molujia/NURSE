"""Content-addressed stage checkpoints, replay receipts, and failure tombstones."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.agents.retry import AttemptRecord, AttemptUsage
from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.experiment.manifest import RunManifestArtifact
from nurse_rcl.schemas import StrictFrozenModel

CHECKPOINT_SCHEMA_VERSION: Literal[1] = 1
LLM_CACHE_SCHEMA_VERSION: Literal[2] = 2
LLM_ATTEMPT_RESERVATION_SCHEMA_VERSION: Literal[1] = 1
LLM_ATTEMPT_RESULT_SCHEMA_VERSION: Literal[1] = 1
LLM_ATTEMPT_LEDGER_SCHEMA_VERSION: Literal[1] = 1
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ReplayCode = Literal[
    "same_run_exact_hit",
    "stage_not_found",
    "reject_input_drift",
    "reject_history_snapshot_drift",
]
LLMReplayCode = Literal[
    "same_run_exact_hit",
    "reject_request_drift",
    "reject_cross_run_llm_cache",
]


class CheckpointCorruptionError(ValueError):
    """Raised when a persisted checkpoint or cache hash cannot be trusted."""


class StageCheckpoint(StrictFrozenModel):
    schema_version: Literal[1] = CHECKPOINT_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    stage: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    input_hashes: tuple[Sha256Hex, ...] = Field(min_length=1)
    history_snapshot_hash: Sha256Hex | None = None
    output_file: str = Field(min_length=1)
    output_hash: Sha256Hex
    previous_checkpoint_hash: Sha256Hex | None = None
    checkpoint_hash: Sha256Hex

    @model_validator(mode="after")
    def require_rank_history_binding(self) -> Self:
        if self.stage == "ranking" and self.history_snapshot_hash is None:
            raise ValueError("ranking checkpoints require a history snapshot hash")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"checkpoint_hash"})
        return self.checkpoint_hash == sha256_hex(canonical_json_bytes(unsigned))


class FailureTombstone(StrictFrozenModel):
    schema_version: Literal[1] = CHECKPOINT_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    stage: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    failure_code: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    previous_checkpoint_hash: Sha256Hex | None = None
    tombstone_hash: Sha256Hex

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"tombstone_hash"})
        return self.tombstone_hash == sha256_hex(canonical_json_bytes(unsigned))


class CaseRecovery(StrictFrozenModel):
    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    checkpoints: tuple[StageCheckpoint, ...]
    latest_stage: str | None = None
    failure: FailureTombstone | None = None
    recovery_hash: Sha256Hex

    def verify(self) -> bool:
        previous_hash: str | None = None
        for sequence, checkpoint in enumerate(self.checkpoints, start=1):
            if not checkpoint.verify():
                return False
            if (
                checkpoint.run_id != self.run_id
                or checkpoint.case_id != self.case_id
                or checkpoint.sequence != sequence
                or checkpoint.previous_checkpoint_hash != previous_hash
            ):
                return False
            previous_hash = checkpoint.checkpoint_hash
        expected_stage = self.checkpoints[-1].stage if self.checkpoints else None
        if self.latest_stage != expected_stage:
            return False
        if self.failure is not None and (
            not self.failure.verify()
            or self.failure.run_id != self.run_id
            or self.failure.case_id != self.case_id
            or self.failure.previous_checkpoint_hash != previous_hash
        ):
            return False
        unsigned = self.model_dump(mode="json", exclude={"recovery_hash"})
        return self.recovery_hash == sha256_hex(canonical_json_bytes(unsigned))


class StageReplayReceipt(StrictFrozenModel):
    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    hit: bool
    code: ReplayCode
    checkpoint_hash: Sha256Hex | None = None
    output_hash: Sha256Hex | None = None
    output_json: str | None = None
    receipt_hash: Sha256Hex

    @property
    def output(self) -> dict[str, Any] | None:
        if self.output_json is None:
            return None
        value = json.loads(self.output_json)
        return value if isinstance(value, dict) else None

    def verify(self) -> bool:
        if self.hit:
            if (
                self.code != "same_run_exact_hit"
                or self.checkpoint_hash is None
                or self.output_hash is None
                or self.output_json is None
                or sha256_hex(self.output_json.encode("utf-8")) != self.output_hash
            ):
                return False
        elif self.output_hash is not None or self.output_json is not None:
            return False
        unsigned = self.model_dump(mode="json", exclude={"receipt_hash"})
        return self.receipt_hash == sha256_hex(canonical_json_bytes(unsigned))


class LLMCumulativeUsage(StrictFrozenModel):
    """Monotonic provider totals frozen atomically with one successful response."""

    attempts: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_seconds: float = Field(ge=0.0, allow_inf_nan=False)


class LLMAttemptReservation(StrictFrozenModel):
    """Durable pre-provider reservation for one logical-request attempt."""

    schema_version: Literal[1] = LLM_ATTEMPT_RESERVATION_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    request_hash: Sha256Hex
    stage: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    attempt_number: int = Field(ge=1, le=5)
    cumulative_usage: LLMCumulativeUsage
    reservation_hash: Sha256Hex

    @model_validator(mode="after")
    def require_reserved_attempt_in_usage(self) -> Self:
        if self.cumulative_usage.attempts < self.attempt_number:
            raise ValueError("attempt reservation usage cannot omit the reserved attempt")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"reservation_hash"})
        return self.reservation_hash == sha256_hex(canonical_json_bytes(unsigned))


class LLMAttemptResultReceipt(StrictFrozenModel):
    """Durable result and usage for one completed provider attempt."""

    schema_version: Literal[1] = LLM_ATTEMPT_RESULT_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    request_hash: Sha256Hex
    stage: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    attempt: AttemptRecord
    cumulative_usage: LLMCumulativeUsage
    response_json: str | None
    response_hash: Sha256Hex | None
    result_hash: Sha256Hex

    @model_validator(mode="after")
    def require_response_only_for_success(self) -> Self:
        if self.cumulative_usage.attempts < self.attempt.attempt_number:
            raise ValueError("attempt result usage cannot omit the completed attempt")
        if self.attempt.outcome == "success":
            if self.response_json is None or self.response_hash is None:
                raise ValueError("successful attempt result requires its response")
        elif self.response_json is not None or self.response_hash is not None:
            raise ValueError("failed attempt result cannot retain a response")
        return self

    @property
    def response(self) -> dict[str, Any] | None:
        if self.response_json is None:
            return None
        value = json.loads(self.response_json)
        return value if isinstance(value, dict) else None

    def verify(self) -> bool:
        if self.response_json is not None:
            if (
                self.response_hash is None
                or sha256_hex(self.response_json.encode("utf-8")) != self.response_hash
            ):
                return False
            try:
                value = json.loads(self.response_json)
            except (TypeError, ValueError):
                return False
            if (
                not isinstance(value, dict)
                or canonical_json_bytes(value).decode("utf-8") != self.response_json
            ):
                return False
        elif self.response_hash is not None:
            return False
        unsigned = self.model_dump(mode="json", exclude={"result_hash"})
        return self.result_hash == sha256_hex(canonical_json_bytes(unsigned))


class LLMAttemptLedger(StrictFrozenModel):
    """Atomic request ledger anchoring reservations and completed results."""

    schema_version: Literal[1] = LLM_ATTEMPT_LEDGER_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    request_hash: Sha256Hex
    stage: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    reservations: tuple[LLMAttemptReservation, ...] = Field(min_length=1, max_length=5)
    results: tuple[LLMAttemptResultReceipt, ...] = Field(default=(), max_length=5)
    ledger_hash: Sha256Hex

    @model_validator(mode="after")
    def require_one_consistent_attempt_trace(self) -> Self:
        if tuple(item.attempt_number for item in self.reservations) != tuple(
            range(1, len(self.reservations) + 1)
        ):
            raise ValueError("LLM attempt ledger reservations must be contiguous")
        if tuple(item.attempt.attempt_number for item in self.results) != tuple(
            range(1, len(self.results) + 1)
        ):
            raise ValueError("LLM attempt ledger results must be contiguous")
        if not 0 <= len(self.reservations) - len(self.results) <= 1:
            raise ValueError("LLM attempt ledger may have only one dangling reservation")
        bindings = (self.run_id, self.case_id, self.request_hash, self.stage)
        reservations_valid = all(
            (
                item.run_id,
                item.case_id,
                item.request_hash,
                item.stage,
            )
            == bindings
            and item.verify()
            for item in self.reservations
        )
        results_valid = all(
            (
                item.run_id,
                item.case_id,
                item.request_hash,
                item.stage,
            )
            == bindings
            and item.verify()
            for item in self.results
        )
        if not reservations_valid or not results_valid:
            raise ValueError("LLM attempt ledger contains an invalid bound receipt")
        for reservation, result in zip(
            self.reservations,
            self.results,
            strict=False,
        ):
            if result.cumulative_usage.attempts != reservation.cumulative_usage.attempts:
                raise ValueError("attempt result and reservation counts differ")
            if not _usage_is_monotonic(
                reservation.cumulative_usage,
                result.cumulative_usage,
            ):
                raise ValueError("attempt result usage precedes its reservation")
        for previous, current in zip(self.results, self.reservations[1:], strict=False):
            if not _usage_is_monotonic(
                previous.cumulative_usage,
                current.cumulative_usage,
            ):
                raise ValueError("LLM attempt ledger usage is not monotonic")
        if any(item.attempt.outcome != "retry" for item in self.results[:-1]):
            raise ValueError("non-final attempt results must be retries")
        if self.results:
            final = self.results[-1].attempt
            if final.attempt_number == 5 and final.outcome == "retry":
                raise ValueError("the fifth attempt result cannot be retried")
            if final.outcome != "retry" and len(self.results) != len(self.reservations):
                raise ValueError("terminal attempt result cannot have a later reservation")
        return self

    def verify(self) -> bool:
        if any(not item.verify() for item in self.reservations) or any(
            not item.verify() for item in self.results
        ):
            return False
        unsigned = self.model_dump(mode="json", exclude={"ledger_hash"})
        return self.ledger_hash == sha256_hex(canonical_json_bytes(unsigned))


class LLMCacheEntry(StrictFrozenModel):
    schema_version: Literal[2]
    run_id: str = Field(min_length=1)
    request_hash: Sha256Hex
    response_json: str = Field(min_length=2)
    response_hash: Sha256Hex
    attempts: tuple[AttemptRecord, ...] = Field(min_length=1, max_length=5)
    cumulative_usage: LLMCumulativeUsage
    entry_hash: Sha256Hex

    @model_validator(mode="after")
    def require_complete_successful_attempt_trace(self) -> Self:
        if tuple(item.attempt_number for item in self.attempts) != tuple(
            range(1, len(self.attempts) + 1)
        ):
            raise ValueError("LLM cache attempts must be contiguous and one-based")
        if any(item.outcome != "retry" for item in self.attempts[:-1]):
            raise ValueError("non-final cached attempts must have retry outcomes")
        final = self.attempts[-1]
        if final.outcome != "success" or final.category != "success":
            raise ValueError("LLM cache attempt trace must end in success")
        if self.cumulative_usage.attempts < len(self.attempts):
            raise ValueError("cumulative usage cannot omit cached attempts")
        return self

    @property
    def response(self) -> dict[str, Any]:
        value = json.loads(self.response_json)
        return value if isinstance(value, dict) else {}

    def verify(self) -> bool:
        if sha256_hex(self.response_json.encode("utf-8")) != self.response_hash:
            return False
        try:
            value = json.loads(self.response_json)
        except (TypeError, ValueError):
            return False
        if not isinstance(value, dict):
            return False
        if canonical_json_bytes(value).decode("utf-8") != self.response_json:
            return False
        unsigned = self.model_dump(mode="json", exclude={"entry_hash"})
        return self.entry_hash == sha256_hex(canonical_json_bytes(unsigned))


class LLMReplayReceipt(StrictFrozenModel):
    run_id: str = Field(min_length=1)
    source_run_id: str = Field(min_length=1)
    request_hash: Sha256Hex
    hit: bool
    code: LLMReplayCode
    entry_hash: Sha256Hex
    response_hash: Sha256Hex | None = None
    response_json: str | None = None
    receipt_hash: Sha256Hex

    @property
    def response(self) -> dict[str, Any] | None:
        if self.response_json is None:
            return None
        value = json.loads(self.response_json)
        return value if isinstance(value, dict) else None

    def verify(self) -> bool:
        if self.hit:
            if (
                self.code != "same_run_exact_hit"
                or self.response_hash is None
                or self.response_json is None
                or sha256_hex(self.response_json.encode("utf-8")) != self.response_hash
            ):
                return False
        elif self.response_hash is not None or self.response_json is not None:
            return False
        unsigned = self.model_dump(mode="json", exclude={"receipt_hash"})
        return self.receipt_hash == sha256_hex(canonical_json_bytes(unsigned))


class CheckpointStore:
    """One manifest-bound run store with immutable case and LLM artifacts."""

    def __init__(self, *, root: Path, manifest: RunManifestArtifact) -> None:
        if not manifest.verify():
            raise ValueError("run manifest verification failed")
        self._root = root.resolve(strict=False)
        self._manifest = manifest
        self._bind_manifest()

    def write_stage(
        self,
        *,
        case_id: str,
        stage: str,
        input_hashes: tuple[str, ...],
        output: Mapping[str, Any],
        history_snapshot_hash: str | None = None,
    ) -> StageCheckpoint:
        _require_safe_segment(case_id, name="case_id")
        _require_stage(stage)
        recovery = self.recover_case(case_id=case_id)
        if recovery.failure is not None:
            raise ValueError("failure tombstone is terminal for the case")
        output_bytes = canonical_json_bytes(dict(output))
        output_hash = sha256_hex(output_bytes)
        existing = next(
            (item for item in recovery.checkpoints if item.stage == stage),
            None,
        )
        if existing is not None:
            if (
                existing.input_hashes == input_hashes
                and existing.history_snapshot_hash == history_snapshot_hash
                and existing.output_hash == output_hash
            ):
                return existing
            raise ValueError("stage checkpoint is immutable")

        sequence = len(recovery.checkpoints) + 1
        output_file = self._output_relative_path(case_id, sequence, stage).as_posix()
        previous_hash = (
            recovery.checkpoints[-1].checkpoint_hash if recovery.checkpoints else None
        )
        unsigned = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "case_id": case_id,
            "sequence": sequence,
            "stage": stage,
            "input_hashes": list(input_hashes),
            "history_snapshot_hash": history_snapshot_hash,
            "output_file": output_file,
            "output_hash": output_hash,
            "previous_checkpoint_hash": previous_hash,
        }
        checkpoint = StageCheckpoint(
            run_id=self.run_id,
            case_id=case_id,
            sequence=sequence,
            stage=stage,
            input_hashes=input_hashes,
            history_snapshot_hash=history_snapshot_hash,
            output_file=output_file,
            output_hash=output_hash,
            previous_checkpoint_hash=previous_hash,
            checkpoint_hash=sha256_hex(canonical_json_bytes(unsigned)),
        )
        atomic_write_bytes(self._root / output_file, output_bytes)
        atomic_write_bytes(
            self.checkpoint_path(checkpoint),
            canonical_json_bytes(checkpoint.model_dump(mode="json")),
        )
        return checkpoint

    def replay_stage(
        self,
        *,
        case_id: str,
        stage: str,
        input_hashes: tuple[str, ...],
        history_snapshot_hash: str | None = None,
    ) -> StageReplayReceipt:
        _require_safe_segment(case_id, name="case_id")
        _require_stage(stage)
        recovery = self.recover_case(case_id=case_id)
        checkpoint = next(
            (item for item in recovery.checkpoints if item.stage == stage),
            None,
        )
        if checkpoint is None:
            return _stage_receipt(
                run_id=self.run_id,
                case_id=case_id,
                stage=stage,
                code="stage_not_found",
                checkpoint=None,
                output_json=None,
            )
        if checkpoint.input_hashes != input_hashes:
            return _stage_receipt(
                run_id=self.run_id,
                case_id=case_id,
                stage=stage,
                code="reject_input_drift",
                checkpoint=checkpoint,
                output_json=None,
            )
        if checkpoint.history_snapshot_hash != history_snapshot_hash:
            return _stage_receipt(
                run_id=self.run_id,
                case_id=case_id,
                stage=stage,
                code="reject_history_snapshot_drift",
                checkpoint=checkpoint,
                output_json=None,
            )
        output_json = self._read_output(checkpoint).decode("utf-8")
        return _stage_receipt(
            run_id=self.run_id,
            case_id=case_id,
            stage=stage,
            code="same_run_exact_hit",
            checkpoint=checkpoint,
            output_json=output_json,
        )

    def recover_case(self, *, case_id: str) -> CaseRecovery:
        _require_safe_segment(case_id, name="case_id")
        checkpoint_dir = self._case_root(case_id) / "checkpoints"
        paths = tuple(sorted(checkpoint_dir.glob("*.json"))) if checkpoint_dir.exists() else ()
        checkpoints: list[StageCheckpoint] = []
        previous_hash: str | None = None
        for sequence, path in enumerate(paths, start=1):
            checkpoint = self._read_checkpoint(path)
            expected_path = self._checkpoint_path(
                checkpoint.case_id,
                checkpoint.sequence,
                checkpoint.stage,
            )
            if (
                checkpoint.run_id != self.run_id
                or checkpoint.case_id != case_id
                or checkpoint.sequence != sequence
                or path.resolve(strict=False) != expected_path
                or checkpoint.previous_checkpoint_hash != previous_hash
            ):
                raise CheckpointCorruptionError("checkpoint chain identity is invalid")
            expected_output = self._output_relative_path(
                case_id,
                sequence,
                checkpoint.stage,
            ).as_posix()
            if checkpoint.output_file != expected_output:
                raise CheckpointCorruptionError("checkpoint chain output path is invalid")
            self._read_output(checkpoint)
            checkpoints.append(checkpoint)
            previous_hash = checkpoint.checkpoint_hash

        failure = self._read_failure(case_id=case_id)
        if failure is not None and failure.previous_checkpoint_hash != previous_hash:
            raise CheckpointCorruptionError("failure tombstone chain is invalid")
        latest_stage = checkpoints[-1].stage if checkpoints else None
        unsigned = {
            "run_id": self.run_id,
            "case_id": case_id,
            "checkpoints": [item.model_dump(mode="json") for item in checkpoints],
            "latest_stage": latest_stage,
            "failure": failure.model_dump(mode="json") if failure is not None else None,
        }
        result = CaseRecovery(
            run_id=self.run_id,
            case_id=case_id,
            checkpoints=tuple(checkpoints),
            latest_stage=latest_stage,
            failure=failure,
            recovery_hash=sha256_hex(canonical_json_bytes(unsigned)),
        )
        if not result.verify():  # pragma: no cover - construction invariant
            raise AssertionError("case recovery verification failed")
        return result

    def write_failure_tombstone(
        self,
        *,
        case_id: str,
        stage: str,
        failure_code: str,
    ) -> FailureTombstone:
        _require_safe_segment(case_id, name="case_id")
        _require_stage(stage)
        recovery = self.recover_case(case_id=case_id)
        previous_hash = (
            recovery.checkpoints[-1].checkpoint_hash if recovery.checkpoints else None
        )
        unsigned = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "case_id": case_id,
            "stage": stage,
            "failure_code": failure_code,
            "previous_checkpoint_hash": previous_hash,
        }
        tombstone = FailureTombstone(
            run_id=self.run_id,
            case_id=case_id,
            stage=stage,
            failure_code=failure_code,
            previous_checkpoint_hash=previous_hash,
            tombstone_hash=sha256_hex(canonical_json_bytes(unsigned)),
        )
        path = self._failure_path(case_id)
        if recovery.failure is not None:
            if recovery.failure == tombstone:
                return recovery.failure
            raise ValueError("failure tombstone is immutable")
        atomic_write_bytes(path, canonical_json_bytes(tombstone.model_dump(mode="json")))
        return tombstone

    def reserve_llm_attempt(
        self,
        *,
        case_id: str,
        request_hash: str,
        stage: str,
        attempt_number: int,
        cumulative_usage: LLMCumulativeUsage,
    ) -> LLMAttemptReservation:
        _require_safe_segment(case_id, name="case_id")
        _require_stage(stage)
        if re.fullmatch(r"[0-9a-f]{64}", request_hash) is None:
            raise ValueError("request_hash must be lowercase SHA-256")
        ledger = self.llm_attempt_ledger(
            request_hash=request_hash,
            case_id=case_id,
            stage=stage,
        )
        reservations = ledger.reservations if ledger is not None else ()
        results = ledger.results if ledger is not None else ()
        if len(reservations) != len(results):
            raise ValueError("dangling LLM attempt reservation must be reconciled")
        if results and results[-1].attempt.outcome != "retry":
            raise ValueError("terminal LLM attempt ledger cannot reserve another attempt")
        expected_attempt = len(reservations) + 1
        if attempt_number != expected_attempt:
            raise ValueError("LLM attempt reservations must be contiguous")
        if results and not _usage_is_monotonic(
            results[-1].cumulative_usage,
            cumulative_usage,
        ):
            raise ValueError("LLM attempt reservation usage must be monotonic")
        unsigned = {
            "schema_version": LLM_ATTEMPT_RESERVATION_SCHEMA_VERSION,
            "run_id": self.run_id,
            "case_id": case_id,
            "request_hash": request_hash,
            "stage": stage,
            "attempt_number": attempt_number,
            "cumulative_usage": cumulative_usage.model_dump(mode="json"),
        }
        reservation = LLMAttemptReservation(
            run_id=self.run_id,
            case_id=case_id,
            request_hash=request_hash,
            stage=stage,
            attempt_number=attempt_number,
            cumulative_usage=cumulative_usage,
            reservation_hash=sha256_hex(canonical_json_bytes(unsigned)),
        )
        updated = _build_llm_attempt_ledger(
            run_id=self.run_id,
            case_id=case_id,
            request_hash=request_hash,
            stage=stage,
            reservations=(*reservations, reservation),
            results=results,
        )
        atomic_write_bytes(
            self._llm_attempt_ledger_path(request_hash),
            canonical_json_bytes(updated.model_dump(mode="json")),
        )
        return reservation

    def write_llm_attempt_result(
        self,
        *,
        case_id: str,
        request_hash: str,
        stage: str,
        attempt: AttemptRecord,
        cumulative_usage: LLMCumulativeUsage,
        response: Mapping[str, Any] | None,
    ) -> LLMAttemptResultReceipt:
        ledger = self.llm_attempt_ledger(
            request_hash=request_hash,
            case_id=case_id,
            stage=stage,
        )
        if ledger is None:
            raise ValueError("LLM attempt result requires a reservation")
        expected_attempt = len(ledger.results) + 1
        if (
            attempt.attempt_number != expected_attempt
            or len(ledger.reservations) != expected_attempt
        ):
            raise ValueError("LLM attempt result does not match the dangling reservation")
        reservation = ledger.reservations[-1]
        if cumulative_usage.attempts != reservation.cumulative_usage.attempts:
            raise ValueError("LLM attempt result count differs from its reservation")
        if not _usage_is_monotonic(reservation.cumulative_usage, cumulative_usage):
            raise ValueError("LLM attempt result usage must follow its reservation")
        response_json = (
            canonical_json_bytes(dict(response)).decode("utf-8")
            if response is not None
            else None
        )
        response_hash = (
            sha256_hex(response_json.encode("utf-8"))
            if response_json is not None
            else None
        )
        unsigned = {
            "schema_version": LLM_ATTEMPT_RESULT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "case_id": case_id,
            "request_hash": request_hash,
            "stage": stage,
            "attempt": asdict(attempt),
            "cumulative_usage": cumulative_usage.model_dump(mode="json"),
            "response_json": response_json,
            "response_hash": response_hash,
        }
        receipt = LLMAttemptResultReceipt(
            run_id=self.run_id,
            case_id=case_id,
            request_hash=request_hash,
            stage=stage,
            attempt=attempt,
            cumulative_usage=cumulative_usage,
            response_json=response_json,
            response_hash=response_hash,
            result_hash=sha256_hex(canonical_json_bytes(unsigned)),
        )
        updated = _build_llm_attempt_ledger(
            run_id=self.run_id,
            case_id=case_id,
            request_hash=request_hash,
            stage=stage,
            reservations=ledger.reservations,
            results=(*ledger.results, receipt),
        )
        atomic_write_bytes(
            self._llm_attempt_ledger_path(request_hash),
            canonical_json_bytes(updated.model_dump(mode="json")),
        )
        return receipt

    def llm_attempt_ledger(
        self,
        *,
        request_hash: str,
        case_id: str | None = None,
        stage: str | None = None,
    ) -> LLMAttemptLedger | None:
        if re.fullmatch(r"[0-9a-f]{64}", request_hash) is None:
            raise ValueError("request_hash must be lowercase SHA-256")
        if case_id is not None:
            _require_safe_segment(case_id, name="case_id")
        if stage is not None:
            _require_stage(stage)
        path = self._llm_attempt_ledger_path(request_hash)
        if not path.exists():
            return None
        ledger = self._read_llm_attempt_ledger(path)
        if (
            ledger.request_hash != request_hash
            or (case_id is not None and ledger.case_id != case_id)
            or (stage is not None and ledger.stage != stage)
        ):
            raise CheckpointCorruptionError("LLM attempt ledger binding is invalid")
        return ledger

    def llm_attempt_reservations(
        self,
        *,
        request_hash: str,
        case_id: str | None = None,
        stage: str | None = None,
    ) -> tuple[LLMAttemptReservation, ...]:
        ledger = self.llm_attempt_ledger(
            request_hash=request_hash,
            case_id=case_id,
            stage=stage,
        )
        return ledger.reservations if ledger is not None else ()

    def llm_attempt_results(
        self,
        *,
        request_hash: str,
        case_id: str | None = None,
        stage: str | None = None,
    ) -> tuple[LLMAttemptResultReceipt, ...]:
        ledger = self.llm_attempt_ledger(
            request_hash=request_hash,
            case_id=case_id,
            stage=stage,
        )
        return ledger.results if ledger is not None else ()

    def verified_llm_attempt_ledgers(self) -> tuple[LLMAttemptLedger, ...]:
        root = self._root / "llm_attempts"
        if not root.exists():
            return ()
        ledgers: list[LLMAttemptLedger] = []
        for path in sorted(root.iterdir()):
            if (
                not path.is_file()
                or path.suffix != ".json"
                or re.fullmatch(r"[0-9a-f]{64}", path.stem) is None
            ):
                raise CheckpointCorruptionError("LLM attempt ledger path is invalid")
            ledger = self._read_llm_attempt_ledger(path)
            if ledger.request_hash != path.stem:
                raise CheckpointCorruptionError("LLM attempt ledger path is invalid")
            ledgers.append(ledger)
        reservations = sorted(
            (
                reservation
                for ledger in ledgers
                for reservation in ledger.reservations
            ),
            key=lambda item: item.cumulative_usage.attempts,
        )
        if any(
            current.cumulative_usage.attempts
            <= previous.cumulative_usage.attempts
            for previous, current in pairwise(reservations)
        ):
            raise CheckpointCorruptionError(
                "global LLM attempt reservation sequence is invalid"
            )
        if any(
            not _usage_is_monotonic(
                previous.cumulative_usage,
                current.cumulative_usage,
            )
            for previous, current in pairwise(reservations)
        ):
            raise CheckpointCorruptionError(
                "global LLM attempt reservation usage is not monotonic"
            )
        return tuple(ledgers)

    def verified_llm_attempt_reservations(self) -> tuple[LLMAttemptReservation, ...]:
        return tuple(
            reservation
            for ledger in self.verified_llm_attempt_ledgers()
            for reservation in ledger.reservations
        )

    def verified_llm_attempt_results(self) -> tuple[LLMAttemptResultReceipt, ...]:
        return tuple(
            result
            for ledger in self.verified_llm_attempt_ledgers()
            for result in ledger.results
        )

    def reconcile_interrupted_llm_attempts(self) -> tuple[LLMAttemptLedger, ...]:
        for ledger in self.verified_llm_attempt_ledgers():
            if len(ledger.reservations) == len(ledger.results):
                continue
            reservation = ledger.reservations[-1]
            attempt_number = reservation.attempt_number
            self.write_llm_attempt_result(
                case_id=ledger.case_id,
                request_hash=ledger.request_hash,
                stage=ledger.stage,
                attempt=AttemptRecord(
                    attempt_number=attempt_number,
                    category="transport",
                    outcome="exhausted" if attempt_number == 5 else "retry",
                    latency_seconds=0.0,
                    usage=AttemptUsage.unavailable(),
                    delay_seconds=0.0 if attempt_number < 5 else None,
                    error_type="InterruptedAttemptReservation",
                ),
                cumulative_usage=reservation.cumulative_usage,
                response=None,
            )
        return self.verified_llm_attempt_ledgers()

    def write_llm_response(
        self,
        *,
        request_hash: str,
        response: Mapping[str, Any],
        attempts: tuple[AttemptRecord, ...],
        cumulative_usage: LLMCumulativeUsage,
    ) -> LLMCacheEntry:
        response_json = canonical_json_bytes(dict(response)).decode("utf-8")
        response_hash = sha256_hex(response_json.encode("utf-8"))
        unsigned = {
            "schema_version": LLM_CACHE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "request_hash": request_hash,
            "response_json": response_json,
            "response_hash": response_hash,
            "attempts": [asdict(item) for item in attempts],
            "cumulative_usage": cumulative_usage.model_dump(mode="json"),
        }
        entry = LLMCacheEntry(
            schema_version=LLM_CACHE_SCHEMA_VERSION,
            run_id=self.run_id,
            request_hash=request_hash,
            response_json=response_json,
            response_hash=response_hash,
            attempts=attempts,
            cumulative_usage=cumulative_usage,
            entry_hash=sha256_hex(canonical_json_bytes(unsigned)),
        )
        path = self._root / "llm_cache" / f"{request_hash}.json"
        if path.exists():
            existing = self._read_llm_entry(path)
            if existing != entry:
                raise ValueError("same-run LLM cache entry is immutable")
            return existing
        atomic_write_bytes(path, canonical_json_bytes(entry.model_dump(mode="json")))
        return entry

    def verified_llm_cache_entries(self) -> tuple[LLMCacheEntry, ...]:
        """Return every run-bound v2 cache entry after path and hash validation."""

        root = self._root / "llm_cache"
        if not root.exists():
            return ()
        entries: list[LLMCacheEntry] = []
        for path in sorted(root.glob("*.json")):
            entry = self._read_llm_entry(path)
            if path.stem != entry.request_hash:
                raise CheckpointCorruptionError("LLM cache path does not match request hash")
            entries.append(entry)
        return tuple(entries)

    def replay_llm_response(
        self,
        *,
        entry: LLMCacheEntry,
        request_hash: str,
    ) -> LLMReplayReceipt:
        if not entry.verify():
            raise CheckpointCorruptionError("LLM cache entry hash is invalid")
        if entry.run_id != self.run_id:
            code: LLMReplayCode = "reject_cross_run_llm_cache"
            response_json = None
        elif entry.request_hash != request_hash:
            code = "reject_request_drift"
            response_json = None
        else:
            code = "same_run_exact_hit"
            response_json = entry.response_json
        return _llm_receipt(
            run_id=self.run_id,
            entry=entry,
            request_hash=request_hash,
            code=code,
            response_json=response_json,
        )

    def checkpoint_path(self, checkpoint: StageCheckpoint) -> Path:
        return self._checkpoint_path(
            checkpoint.case_id,
            checkpoint.sequence,
            checkpoint.stage,
        )

    @property
    def run_id(self) -> str:
        return self._manifest.manifest.run_id

    def _bind_manifest(self) -> None:
        path = self._root / "run_manifest.json"
        if path.exists():
            try:
                existing = RunManifestArtifact.model_validate(_read_object(path))
            except (TypeError, ValueError) as error:
                raise CheckpointCorruptionError("bound run manifest is unreadable") from error
            if not existing.verify() or existing != self._manifest:
                raise ValueError("checkpoint store run manifest drift detected")
            return
        atomic_write_bytes(
            path,
            canonical_json_bytes(self._manifest.model_dump(mode="json")),
        )

    def _read_checkpoint(self, path: Path) -> StageCheckpoint:
        try:
            checkpoint = StageCheckpoint.model_validate(_read_object(path))
        except (TypeError, ValueError) as error:
            raise CheckpointCorruptionError("checkpoint chain document is unreadable") from error
        if not checkpoint.verify():
            raise CheckpointCorruptionError("checkpoint chain hash is invalid")
        return checkpoint

    def _read_output(self, checkpoint: StageCheckpoint) -> bytes:
        path = self._root / checkpoint.output_file
        try:
            data = path.read_bytes()
            value = json.loads(data)
        except (OSError, TypeError, ValueError) as error:
            raise CheckpointCorruptionError("checkpoint chain output is unreadable") from error
        if (
            not isinstance(value, dict)
            or canonical_json_bytes(value) != data
            or sha256_hex(data) != checkpoint.output_hash
        ):
            raise CheckpointCorruptionError("checkpoint chain output hash is invalid")
        return data

    def _read_failure(self, *, case_id: str) -> FailureTombstone | None:
        path = self._failure_path(case_id)
        if not path.exists():
            return None
        try:
            failure = FailureTombstone.model_validate(_read_object(path))
        except (TypeError, ValueError) as error:
            raise CheckpointCorruptionError("failure tombstone is unreadable") from error
        if (
            not failure.verify()
            or failure.run_id != self.run_id
            or failure.case_id != case_id
        ):
            raise CheckpointCorruptionError("failure tombstone chain is invalid")
        return failure

    def _read_llm_entry(self, path: Path) -> LLMCacheEntry:
        try:
            entry = LLMCacheEntry.model_validate(_read_object(path))
        except (TypeError, ValueError) as error:
            raise CheckpointCorruptionError("LLM cache entry is unreadable") from error
        if not entry.verify() or entry.run_id != self.run_id:
            raise CheckpointCorruptionError("LLM cache entry hash is invalid")
        return entry

    def _read_llm_attempt_ledger(self, path: Path) -> LLMAttemptLedger:
        try:
            ledger = LLMAttemptLedger.model_validate(_read_object(path))
        except (TypeError, ValueError) as error:
            raise CheckpointCorruptionError("LLM attempt ledger is unreadable") from error
        if not ledger.verify() or ledger.run_id != self.run_id:
            raise CheckpointCorruptionError("LLM attempt ledger hash is invalid")
        return ledger

    def _case_root(self, case_id: str) -> Path:
        return self._root / "cases" / case_id

    def _checkpoint_path(self, case_id: str, sequence: int, stage: str) -> Path:
        return self._case_root(case_id) / "checkpoints" / f"{sequence:04d}-{stage}.json"

    def _output_relative_path(self, case_id: str, sequence: int, stage: str) -> Path:
        return Path("cases") / case_id / "outputs" / f"{sequence:04d}-{stage}.json"

    def _failure_path(self, case_id: str) -> Path:
        return self._case_root(case_id) / "failure.json"

    def _llm_attempt_ledger_path(self, request_hash: str) -> Path:
        return (self._root / "llm_attempts" / f"{request_hash}.json").resolve(
            strict=False
        )


def _build_llm_attempt_ledger(
    *,
    run_id: str,
    case_id: str,
    request_hash: str,
    stage: str,
    reservations: tuple[LLMAttemptReservation, ...],
    results: tuple[LLMAttemptResultReceipt, ...],
) -> LLMAttemptLedger:
    unsigned = {
        "schema_version": LLM_ATTEMPT_LEDGER_SCHEMA_VERSION,
        "run_id": run_id,
        "case_id": case_id,
        "request_hash": request_hash,
        "stage": stage,
        "reservations": [item.model_dump(mode="json") for item in reservations],
        "results": [item.model_dump(mode="json") for item in results],
    }
    ledger = LLMAttemptLedger(
        run_id=run_id,
        case_id=case_id,
        request_hash=request_hash,
        stage=stage,
        reservations=reservations,
        results=results,
        ledger_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not ledger.verify():  # pragma: no cover - construction invariant
        raise AssertionError("LLM attempt ledger verification failed")
    return ledger


def _stage_receipt(
    *,
    run_id: str,
    case_id: str,
    stage: str,
    code: ReplayCode,
    checkpoint: StageCheckpoint | None,
    output_json: str | None,
) -> StageReplayReceipt:
    hit = code == "same_run_exact_hit"
    output_hash = checkpoint.output_hash if hit and checkpoint is not None else None
    unsigned = {
        "run_id": run_id,
        "case_id": case_id,
        "stage": stage,
        "hit": hit,
        "code": code,
        "checkpoint_hash": checkpoint.checkpoint_hash if checkpoint is not None else None,
        "output_hash": output_hash,
        "output_json": output_json,
    }
    result = StageReplayReceipt(
        run_id=run_id,
        case_id=case_id,
        stage=stage,
        hit=hit,
        code=code,
        checkpoint_hash=checkpoint.checkpoint_hash if checkpoint is not None else None,
        output_hash=output_hash,
        output_json=output_json,
        receipt_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("stage replay receipt verification failed")
    return result


def _llm_receipt(
    *,
    run_id: str,
    entry: LLMCacheEntry,
    request_hash: str,
    code: LLMReplayCode,
    response_json: str | None,
) -> LLMReplayReceipt:
    hit = code == "same_run_exact_hit"
    response_hash = entry.response_hash if hit else None
    unsigned = {
        "run_id": run_id,
        "source_run_id": entry.run_id,
        "request_hash": request_hash,
        "hit": hit,
        "code": code,
        "entry_hash": entry.entry_hash,
        "response_hash": response_hash,
        "response_json": response_json,
    }
    result = LLMReplayReceipt(
        run_id=run_id,
        source_run_id=entry.run_id,
        request_hash=request_hash,
        hit=hit,
        code=code,
        entry_hash=entry.entry_hash,
        response_hash=response_hash,
        response_json=response_json,
        receipt_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("LLM replay receipt verification failed")
    return result


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("artifact must be a JSON object")
    return value


def _require_safe_segment(value: str, *, name: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None:
        raise ValueError(f"{name} must be a safe path segment")


def _require_stage(stage: str) -> None:
    if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", stage) is None:
        raise ValueError("stage must be a stable lowercase identifier")


def _usage_is_monotonic(
    previous: LLMCumulativeUsage,
    current: LLMCumulativeUsage,
) -> bool:
    return (
        current.attempts >= previous.attempts
        and current.input_tokens >= previous.input_tokens
        and current.output_tokens >= previous.output_tokens
        and current.latency_seconds >= previous.latency_seconds
    )
