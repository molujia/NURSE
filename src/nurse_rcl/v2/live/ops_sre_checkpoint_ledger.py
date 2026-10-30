"""Version sixteen checkpoints: a failed commit can never become a tool rejection."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Literal, NoReturn

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.live.ops_contracts import OpsInvestigationEvidence
from nurse_rcl.v2.live.ops_failure import retain_secondary
from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord, OpsLedgerStateBase
from nurse_rcl.v2.live.ops_sre_checkpoint_grouped import (
    pack_grouped_source_navigation,
    unpack_grouped_source_navigation,
    validate_grouped_source_envelope,
)
from nurse_rcl.v2.live.ops_sre_d26_tools import (
    OpsSreQuestionInvestigationLedger,
    OpsSreQuestionLedgerState,
)
from nurse_rcl.v2.live.ops_tools import OpsToolResult
from nurse_rcl.v2.live.repair_driver import _fsync
from nurse_rcl.v2.live.staged_case import require_plain_path


class CheckpointCommitError(RuntimeError):
    def __init__(self, cause: BaseException, *, replaced: bool = False):
        super().__init__("checkpoint_commit_unavailable")
        self.trusted_cause = cause
        self.replaced = replaced


class CheckpointReceiptError(CheckpointCommitError):
    pass


class CheckpointProjectionError(CheckpointCommitError):
    pass


class CheckpointSerializationError(CheckpointCommitError):
    pass


class CheckpointCapacityError(CheckpointCommitError):
    pass


class CheckpointWriteError(CheckpointCommitError):
    pass


class CheckpointDurabilityError(CheckpointCommitError):
    pass


def _write_checkpoint(path: Path, raw: bytes) -> None:
    replaced = False
    primary: CheckpointCommitError | None = None
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        require_plain_path(path.parent)
        if path.exists() or path.is_symlink():
            require_plain_path(path)
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        replaced = True
        _fsync(path.parent)
    except Exception as exc:
        cls = CheckpointDurabilityError if replaced else CheckpointWriteError
        primary = cls(exc, replaced=replaced)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except Exception as exc:
            if primary is not None:
                retain_secondary(primary, "investigation_tool", exc)
            else:
                cls = CheckpointDurabilityError if replaced else CheckpointWriteError
                primary = cls(exc, replaced=replaced)
    if primary is not None:
        raise primary from primary.trusted_cause


class OpsSreCheckpointLedgerState(OpsSreQuestionLedgerState):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/16"] = (
        "nurse.v2.sre-investigation-ledger/16"  # type: ignore[assignment]
    )


class OpsSreCheckpointInvestigationLedger(OpsSreQuestionInvestigationLedger):
    state_type: type[OpsLedgerStateBase] = OpsSreCheckpointLedgerState

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._commit_failure: CheckpointCommitError | None = None
        self._confirmed: tuple[OpsActionRecord, ...] = ()

    def _require_usable(self) -> None:
        if self._commit_failure is not None:
            raise self._commit_failure

    def _abort(self, exc: CheckpointCommitError) -> NoReturn:
        self._commit_failure = exc
        self._records = self._confirmed
        raise exc from exc.trusted_cause

    def _require_recordable_failure(self, exc: Exception) -> None:
        if isinstance(exc, CheckpointCommitError):
            raise exc

    def _restore_tool_state(self, state: OpsLedgerStateBase) -> None:
        super()._restore_tool_state(state)
        self._confirmed = state.records

    def _state_values(self) -> dict[str, Any]:
        self._indexed._publish()
        source = self._indexed.checkpoint_source
        return dict(
            navigation=pack_grouped_source_navigation(
                self._indexed.export_navigation(
                    exclude_card_ids=frozenset(source.cards) if source else frozenset(),
                ),
                source=source,
            )
        )

    def _decode_navigation(self, navigation: dict[str, Any]) -> dict[str, Any]:
        validate_grouped_source_envelope(navigation)
        if navigation["source"] is not None and self._indexed.checkpoint_source is None:
            restore = self._indexed._restore_score_source
            if restore is not None:
                restore()
        return unpack_grouped_source_navigation(navigation, source=self._indexed.checkpoint_source)

    def _finished_record(
        self, record: OpsActionRecord, outcome: OpsToolResult | None
    ) -> OpsActionRecord:
        self._require_usable()
        try:
            return super()._finished_record(record, outcome)
        except Exception as exc:
            self._abort(CheckpointReceiptError(exc))

    def capabilities(self) -> dict[str, Any]:
        self._require_usable()
        return super().capabilities()

    def dispatch(self, proposals: object) -> OpsInvestigationEvidence:
        self._require_usable()
        return super().dispatch(proposals)

    def _save(self) -> None:
        self._require_usable()
        try:
            values = self._state_values()
        except Exception as exc:
            cls = (
                CheckpointCapacityError
                if isinstance(exc, ExchangeBoundaryError)
                else (CheckpointProjectionError)
            )
            self._abort(cls(exc))
        try:
            state = self.state_type(
                case_snapshot_id=self._toolbox.case_snapshot_id,
                code_sha256=self._code,
                records=self._records,
                state_sha256="0" * 64,
                **values,
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
        except Exception as exc:
            self._abort(CheckpointSerializationError(exc))
        try:
            self._validate_state_capacity(state, raw)
        except Exception as exc:
            self._abort(CheckpointCapacityError(exc))
        try:
            _write_checkpoint(self._root / "state.json", raw)
        except CheckpointCommitError as exc:
            self._abort(exc)
        self._confirmed = self._records
