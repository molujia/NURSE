"""Durable, independent eight-action budget. This module has no core write API."""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_contracts import (
    OpsInvestigationAction,
    OpsInvestigationEvidence,
    OpsNodeEvidence,
)
from nurse_rcl.v2.live.ops_tools import SKILL_TOOLS, OpsToolbox, OpsToolResult
from nurse_rcl.v2.live.repair_driver import _fsync
from nurse_rcl.v2.live.staged_case import require_plain_path


class OpsActionRecord(StrictFrozenModel):
    index: int = Field(ge=1, le=8, strict=True)
    input_sha256: SHA256
    action: OpsInvestigationAction | None = None
    status: Literal["reserved", "completed", "rejected", "failed", "interrupted"] = "reserved"
    result: OpsInvestigationEvidence | None = None
    failure_type: str | None = None
    elapsed_seconds: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)


class OpsLedgerStateBase(StrictFrozenModel):
    case_snapshot_id: SHA256
    code_sha256: SHA256
    records: tuple[OpsActionRecord, ...] = Field(max_length=8)
    state_sha256: SHA256


class OpsLedgerState(OpsLedgerStateBase):
    schema_version: Literal["nurse.v2.ops-investigation-ledger/1"] = (
        "nurse.v2.ops-investigation-ledger/1"
    )
    investigation_profile_id: Literal["hermes-readonly-ops/1"] = "hermes-readonly-ops/1"


class OpsInvestigationLedger:
    state_type: type[OpsLedgerStateBase] = OpsLedgerState
    record_type: type[OpsActionRecord] = OpsActionRecord
    action_type: type[OpsInvestigationAction] = OpsInvestigationAction
    proposal_bytes = 16384
    state_bytes = 12 * 1024 * 1024
    skill_tools = SKILL_TOOLS

    def __init__(self, *, root: Path, toolbox: OpsToolbox, code_sha256: str):
        self._root = root / "investigation"
        self._toolbox = toolbox
        self._code = code_sha256
        self._lock: int | None = None
        self._records: tuple[OpsActionRecord, ...] = ()

    @property
    def records(self) -> tuple[OpsActionRecord, ...]:
        return self._records

    @property
    def used(self) -> int:
        return len(self._records)

    def __enter__(self) -> OpsInvestigationLedger:
        require_plain_path(self._root.parent)
        self._root.mkdir(mode=0o700, exist_ok=True)
        require_plain_path(self._root)
        if self._lock is not None:
            raise ValueError("investigation ledger is already open")
        self._lock = os.open(self._root / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self._root / "state.json"
            if path.exists() or path.is_symlink():
                raw = self._read_state_bytes(path)
                state = self.state_type.model_validate_json(raw)
                unsigned = state.model_dump(mode="json", exclude={"state_sha256"})
                if state.case_snapshot_id != self._toolbox.case_snapshot_id or (
                    state.code_sha256 != self._code
                ):
                    raise ValueError("investigation ledger identity differs")
                if state.state_sha256 != sha256_hex(canonical_json_bytes(unsigned)) or (
                    raw != canonical_json_bytes(state.model_dump(mode="json"))
                ):
                    raise ValueError("investigation ledger content digest differs")
                self._validate_state_capacity(state, raw)
                self._restore_tool_state(state)
                self._records = state.records
                for index, record in enumerate(self._records, 1):
                    if (
                        record.index != index
                        or (record.action is not None and record.action.action_index != index)
                        or (
                            (record.result is not None)
                            != (record.status in {"completed", "rejected", "failed"})
                        )
                        or (record.status == "reserved" and index != self.used)
                    ):
                        raise ValueError("investigation ledger sequence/receipt differs")
                    if record.result is not None:
                        if record.result.action_index != index:
                            raise ValueError("investigation evidence action index differs")
                        self._toolbox.restore_evidence(record.result)
                if self._records and self._records[-1].status == "reserved":
                    self._replace(
                        self._records[-1].model_copy(
                            update={
                                "status": "interrupted",
                                "failure_type": "completion_unknown_no_refund",
                            }
                        )
                    )
            else:
                self._save()
            return self
        except BaseException:
            os.close(self._lock)
            self._lock = None
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._lock is not None:
            os.close(self._lock)
            self._lock = None

    def _restore_tool_state(self, state: OpsLedgerStateBase) -> None:
        """Version-specific owner metadata. Legacy protocol has none."""

    def _read_state_bytes(self, path: Path) -> bytes:
        return _read_bytes(path, self.state_bytes)

    def _validate_state_capacity(self, state: OpsLedgerStateBase, raw: bytes) -> None:
        if len(raw) > self.state_bytes:
            raise ValueError("investigation checkpoint exceeds read/write capacity")

    def _state_values(self) -> dict[str, Any]:
        return {}

    def _finished_record(
        self,
        record: OpsActionRecord,
        outcome: OpsToolResult | None,
    ) -> OpsActionRecord:
        return record

    def _failed_record(self, record: OpsActionRecord, exc: Exception) -> OpsActionRecord:
        """Version-specific local diagnostics; legacy failure receipts are unchanged."""
        return record

    def _require_recordable_failure(self, exc: Exception) -> None:
        """Legacy failures remain recordable; checkpoint owners may fail closed."""

    def _invoke_tool(self, action: OpsInvestigationAction) -> OpsToolResult:
        """Only new versioned ledgers may specialize a completed tool dispatch."""
        return self._toolbox.execute(action.tool, json.loads(action.parameters_json))

    def _save(self) -> None:
        state = self.state_type(
            case_snapshot_id=self._toolbox.case_snapshot_id,
            code_sha256=self._code,
            records=self._records,
            state_sha256="0" * 64,
            **self._state_values(),
        )
        digest = sha256_hex(
            canonical_json_bytes(
                state.model_dump(
                    mode="json",
                    exclude={"state_sha256"},
                )
            )
        )
        state = state.model_copy(update={"state_sha256": digest})
        raw = canonical_json_bytes(state.model_dump(mode="json"))
        self._validate_state_capacity(state, raw)
        atomic_write_bytes(self._root / "state.json", raw)
        _fsync(self._root)

    def _replace(self, record: OpsActionRecord) -> None:
        self._records = (*self._records[:-1], record)
        self._save()

    def _admit_proposal(
        self,
        proposals: object,
        raw: bytes,
        record: OpsActionRecord,
    ) -> OpsActionRecord:
        """Version hook after the durable debit; legacy four-field contract unchanged."""
        if len(raw) > self.proposal_bytes or type(proposals) is not list or len(proposals) != 1:
            raise ValueError("exactly one bounded action required")
        proposal = proposals[0]
        if type(proposal) is not dict or set(proposal) != {
            "skill_id",
            "tool",
            "parameters",
            "question",
        }:
            raise ValueError("action fields differ from registered contract")
        action = self.action_type(
            action_index=record.index,
            skill_id=proposal["skill_id"],
            tool=proposal["tool"],
            parameters_json=canonical_json_bytes(proposal["parameters"]).decode("utf-8"),
            question=proposal["question"],
        )
        if action.tool not in self.skill_tools.get(action.skill_id, ()):
            raise ValueError("skill/tool combination is not registered")
        return record.model_copy(update={"action": action})

    def dispatch(self, proposals: object) -> OpsInvestigationEvidence:
        if self._lock is None:
            raise ValueError("investigation ledger is closed")
        if self.used >= 8:
            raise ValueError("investigation action budget exhausted")
        if self._records and self._records[-1].status == "reserved":
            raise ValueError("unresolved reservation; reopen to record interruption")
        started = time.monotonic()
        try:
            raw = canonical_json_bytes(proposals)
        except (TypeError, ValueError, RecursionError):
            raw = b"invalid-non-json-proposal"
        record = self.record_type(index=self.used + 1, input_sha256=sha256_hex(raw))
        self._records = (*self._records, record)
        self._save()  # Debit before parsing or invoking any requested tool.
        executing = False
        outcome: OpsToolResult | None = None
        try:
            record = self._admit_proposal(proposals, raw, record)
            self._replace(record)
            action = record.action
            assert action is not None
            executing = True
            outcome = self._invoke_tool(action)
            result = OpsInvestigationEvidence(
                action_index=record.index,
                evidence=outcome.evidence,
                origin=outcome.origin,
                reused_core_evidence_ids=outcome.reused_core_evidence_ids,
            )
            status: Literal["completed", "rejected", "failed"] = "completed"
            failure = None
        except Exception as exc:
            self._require_recordable_failure(exc)
            record = self._records[-1]  # Preserve versioned admission metadata already committed.
            record = self._failed_record(record, exc)
            status = "failed" if executing and not isinstance(exc, ValueError) else "rejected"
            failure = type(exc).__name__
            result = OpsInvestigationEvidence(
                action_index=record.index,
                origin="rejected",
                reused_core_evidence_ids=(),
                evidence=OpsNodeEvidence(
                    case_snapshot_id=self._toolbox.case_snapshot_id,
                    node="boundary",
                    tool_version="ops-investigation-failure/1",
                    parameters_json="{}",
                    evidence_ids=(record.input_sha256,),
                    source_refs=(),
                    unit=None,
                    window_ns=None,
                    coverage=None,
                    availability="unavailable",
                    truth_value="unknown",
                    limitations=("tool_" + status, "no_health_or_causal_conclusion"),
                    payload_json=canonical_json_bytes(
                        dict(
                            status=status,
                            failure_type=failure,
                            raw_input_not_echoed=True,
                        )
                    ).decode("utf-8"),
                ),
            )
        record = record.model_copy(
            update=dict(
                status=status,
                result=result,
                failure_type=failure,
                elapsed_seconds=time.monotonic() - started,
            )
        )
        self._replace(self._finished_record(record, outcome))
        self._toolbox.restore_evidence(result)
        return result
