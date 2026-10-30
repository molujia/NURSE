"""Deterministic User Manager state authorization and checkpoint replay."""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import Enum
from typing import Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel

USER_MANAGER_STATE_SCHEMA_VERSION: Literal[1] = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class UserManagerState(str, Enum):
    CREATED = "CREATED"
    CONTROLS_FROZEN = "CONTROLS_FROZEN"
    SCOPE_FIXED = "SCOPE_FIXED"
    CANDIDATES_READY = "CANDIDATES_READY"
    PRELIM_GRAPH_READY = "PRELIM_GRAPH_READY"
    PAYLOAD_READY = "PAYLOAD_READY"
    FACTS_VALIDATED = "FACTS_VALIDATED"
    RANKING_FROZEN = "RANKING_FROZEN"
    REPORT_WRITTEN = "REPORT_WRITTEN"


class UserManagerTool(str, Enum):
    FREEZE_CONTROLS = "freeze_controls"
    ALARM_CONTEXT_ADAPTER = "alarm_context_adapter"
    CANDIDATE_BUILDER = "candidate_builder"
    EVIDENCE_INITIALIZER = "evidence_initializer"
    CONTEXT_DISTILLER = "context_distiller"
    VALIDATE_SEMANTIC_FACTS = "validate_semantic_facts"
    EVIDENCE_RANKER = "evidence_ranker"
    WRITE_REPORT = "write_report"


DETERMINISTIC_TOOLKITS = (
    UserManagerTool.ALARM_CONTEXT_ADAPTER.value,
    UserManagerTool.CANDIDATE_BUILDER.value,
    UserManagerTool.EVIDENCE_INITIALIZER.value,
    UserManagerTool.CONTEXT_DISTILLER.value,
    UserManagerTool.EVIDENCE_RANKER.value,
)

_LEGAL_TRANSITIONS = {
    (UserManagerState.CREATED, UserManagerTool.FREEZE_CONTROLS): (
        UserManagerState.CONTROLS_FROZEN
    ),
    (UserManagerState.CONTROLS_FROZEN, UserManagerTool.ALARM_CONTEXT_ADAPTER): (
        UserManagerState.SCOPE_FIXED
    ),
    (UserManagerState.SCOPE_FIXED, UserManagerTool.CANDIDATE_BUILDER): (
        UserManagerState.CANDIDATES_READY
    ),
    (UserManagerState.CANDIDATES_READY, UserManagerTool.EVIDENCE_INITIALIZER): (
        UserManagerState.PRELIM_GRAPH_READY
    ),
    (UserManagerState.PRELIM_GRAPH_READY, UserManagerTool.CONTEXT_DISTILLER): (
        UserManagerState.PAYLOAD_READY
    ),
    (UserManagerState.PAYLOAD_READY, UserManagerTool.VALIDATE_SEMANTIC_FACTS): (
        UserManagerState.FACTS_VALIDATED
    ),
    (UserManagerState.FACTS_VALIDATED, UserManagerTool.EVIDENCE_RANKER): (
        UserManagerState.RANKING_FROZEN
    ),
    (UserManagerState.RANKING_FROZEN, UserManagerTool.WRITE_REPORT): (
        UserManagerState.REPORT_WRITTEN
    ),
}


class TransitionDecision(StrictFrozenModel):
    from_state: UserManagerState
    tool: UserManagerTool
    accepted: bool
    code: Literal["transition_allowed", "reject_tool_for_state"]
    to_state: UserManagerState


class TransitionReceipt(StrictFrozenModel):
    schema_version: Literal[1] = USER_MANAGER_STATE_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    from_state: UserManagerState
    tool: UserManagerTool
    accepted: bool
    code: Literal["transition_allowed", "reject_tool_for_state"]
    to_state: UserManagerState
    output_artifact_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    previous_receipt_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"receipt_hash"})
        return self.receipt_hash == sha256_hex(canonical_json_bytes(unsigned))


class UserManagerCheckpoint(StrictFrozenModel):
    schema_version: Literal[1] = USER_MANAGER_STATE_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    state: UserManagerState
    receipts: tuple[TransitionReceipt, ...]
    checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        replay_state = UserManagerState.CREATED
        previous_receipt_hash: str | None = None
        for sequence, receipt in enumerate(self.receipts, start=1):
            if not receipt.verify():
                return False
            if receipt.case_id != self.case_id or receipt.sequence != sequence:
                return False
            if receipt.previous_receipt_hash != previous_receipt_hash:
                return False
            decision = authorize_user_manager_tool(
                state=replay_state,
                tool=receipt.tool,
            )
            if (
                receipt.from_state != decision.from_state
                or receipt.accepted != decision.accepted
                or receipt.code != decision.code
                or receipt.to_state != decision.to_state
            ):
                return False
            if receipt.accepted != (receipt.output_artifact_hash is not None):
                return False
            replay_state = receipt.to_state
            previous_receipt_hash = receipt.receipt_hash
        if self.state != replay_state:
            return False
        unsigned = self.model_dump(mode="json", exclude={"checkpoint_hash"})
        return self.checkpoint_hash == sha256_hex(canonical_json_bytes(unsigned))


def authorize_user_manager_tool(
    *,
    state: UserManagerState | str,
    tool: UserManagerTool | str,
) -> TransitionDecision:
    """Return the only permitted transition, leaving rejected calls in place."""

    current = UserManagerState(state)
    requested = UserManagerTool(tool)
    next_state = _LEGAL_TRANSITIONS.get((current, requested))
    if next_state is None:
        return TransitionDecision(
            from_state=current,
            tool=requested,
            accepted=False,
            code="reject_tool_for_state",
            to_state=current,
        )
    return TransitionDecision(
        from_state=current,
        tool=requested,
        accepted=True,
        code="transition_allowed",
        to_state=next_state,
    )


def create_user_manager_checkpoint(*, case_id: str) -> UserManagerCheckpoint:
    """Create the canonical empty checkpoint for a new case."""

    return _build_checkpoint(
        case_id=case_id,
        state=UserManagerState.CREATED,
        receipts=(),
    )


def apply_user_manager_tool(
    *,
    checkpoint: UserManagerCheckpoint,
    tool: UserManagerTool | str,
    output_artifact_hash: str,
) -> UserManagerCheckpoint:
    """Audit a tool request and advance only when the current state authorizes it."""

    if not checkpoint.verify():
        raise ValueError("checkpoint verification failed before transition")
    decision = authorize_user_manager_tool(state=checkpoint.state, tool=tool)
    accepted_output_hash: str | None = None
    if decision.accepted:
        if not _SHA256_PATTERN.fullmatch(output_artifact_hash):
            raise ValueError("accepted tool output must be a lowercase SHA-256 digest")
        accepted_output_hash = output_artifact_hash
    previous_receipt_hash = (
        checkpoint.receipts[-1].receipt_hash if checkpoint.receipts else None
    )
    unsigned_receipt = {
        "schema_version": USER_MANAGER_STATE_SCHEMA_VERSION,
        "case_id": checkpoint.case_id,
        "sequence": len(checkpoint.receipts) + 1,
        "from_state": decision.from_state.value,
        "tool": decision.tool.value,
        "accepted": decision.accepted,
        "code": decision.code,
        "to_state": decision.to_state.value,
        "output_artifact_hash": accepted_output_hash,
        "previous_receipt_hash": previous_receipt_hash,
    }
    receipt = TransitionReceipt(
        case_id=checkpoint.case_id,
        sequence=len(checkpoint.receipts) + 1,
        from_state=decision.from_state,
        tool=decision.tool,
        accepted=decision.accepted,
        code=decision.code,
        to_state=decision.to_state,
        output_artifact_hash=accepted_output_hash,
        previous_receipt_hash=previous_receipt_hash,
        receipt_hash=sha256_hex(canonical_json_bytes(unsigned_receipt)),
    )
    if not receipt.verify():  # pragma: no cover - construction invariant
        raise AssertionError("User Manager transition receipt verification failed")
    return _build_checkpoint(
        case_id=checkpoint.case_id,
        state=decision.to_state,
        receipts=(*checkpoint.receipts, receipt),
    )


def restore_user_manager_checkpoint(
    document: Mapping[str, object],
) -> UserManagerCheckpoint:
    """Restore only a schema-valid checkpoint whose receipt chain replays exactly."""

    try:
        checkpoint = UserManagerCheckpoint.model_validate(document)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint verification failed") from error
    if not checkpoint.verify():
        raise ValueError("checkpoint verification failed")
    return checkpoint


def _build_checkpoint(
    *,
    case_id: str,
    state: UserManagerState,
    receipts: tuple[TransitionReceipt, ...],
) -> UserManagerCheckpoint:
    unsigned = {
        "schema_version": USER_MANAGER_STATE_SCHEMA_VERSION,
        "case_id": case_id,
        "state": state.value,
        "receipts": [receipt.model_dump(mode="json") for receipt in receipts],
    }
    checkpoint = UserManagerCheckpoint(
        case_id=case_id,
        state=state,
        receipts=receipts,
        checkpoint_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not checkpoint.verify():  # pragma: no cover - construction invariant
        raise AssertionError("User Manager checkpoint verification failed")
    return checkpoint
