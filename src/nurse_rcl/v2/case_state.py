"""Auditable NURSE V2 case state transitions and atomic persistence."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.runtime_schemas import (
    CaseCheckpoint,
    CaseState,
    build_case_checkpoint,
)

CASE_STATE_ENTRY_SCHEMA_VERSION: Literal[
    "nurse.v2.case-state-entry/1"
] = "nurse.v2.case-state-entry/1"

PIPELINE_STATES: tuple[CaseState, ...] = (
    "DISCOVERED",
    "OPAQUE_BOUND",
    "OBSERVATIONS_READY",
    "CANDIDATES_FROZEN",
    "RULE_HYPOTHESES_READY",
    "SEMANTIC_PENDING",
    "HYPOTHESES_VERIFIED",
    "RANKED",
    "CONTEXT_FROZEN",
    "EXPLAINED",
    "SEALED",
)
TERMINAL_STATES = frozenset({"SEALED", "FAILURE_SEALED"})
SKIPPABLE_STATES = frozenset({"SEMANTIC_PENDING", "EXPLAINED"})

TransitionKind = Literal["initial", "advance", "skip", "failure"]


class IllegalCaseTransition(ValueError):
    """Raised when a requested transition violates the declared state graph."""


class CaseStateIntegrityError(ValueError):
    """Raised when a checkpoint chain or its upstream binding cannot be trusted."""


class SkippedStageAnnotation(StrictFrozenModel):
    """Explicit provenance for a declared semantic or report-stage bypass."""

    state: CaseState
    reason_code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    detail: str = Field(min_length=1, max_length=512)


class CaseStateEntry(StrictFrozenModel):
    """One hash-chained case checkpoint plus its transition provenance."""

    schema_version: Literal["nurse.v2.case-state-entry/1"] = (
        CASE_STATE_ENTRY_SCHEMA_VERSION
    )
    checkpoint: CaseCheckpoint
    from_state: CaseState | None
    transition_kind: TransitionKind
    skipped_stages: tuple[SkippedStageAnnotation, ...]
    failure_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )
    previous_entry_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    entry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if not self.checkpoint.verify():
            return False
        try:
            if self.transition_kind == "initial":
                if (
                    self.checkpoint.state != "DISCOVERED"
                    or self.checkpoint.sequence != 1
                    or self.from_state is not None
                    or self.previous_entry_sha256 is not None
                    or self.skipped_stages
                    or self.failure_code is not None
                ):
                    return False
            else:
                if self.from_state is None or self.previous_entry_sha256 is None:
                    return False
                kind = _transition_kind(
                    from_state=self.from_state,
                    to_state=self.checkpoint.state,
                    skipped_stages=self.skipped_stages,
                    failure_code=self.failure_code,
                )
                if kind != self.transition_kind:
                    return False
        except IllegalCaseTransition:
            return False
        unsigned = self.model_dump(mode="json", exclude={"entry_sha256"})
        return self.entry_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_initial_case_state(
    *,
    opaque_case_id: str,
    experiment_version: str,
    upstream_hashes: Mapping[str, str],
    produced_artifact_hashes: Mapping[str, str],
    created_at_utc: str,
) -> CaseStateEntry:
    """Create the first verified `DISCOVERED` entry for one case."""

    checkpoint = build_case_checkpoint(
        opaque_case_id=opaque_case_id,
        experiment_version=experiment_version,
        state="DISCOVERED",
        sequence=1,
        upstream_hashes=upstream_hashes,
        produced_artifact_hashes=produced_artifact_hashes,
        logical_request_ids=(),
        created_at_utc=created_at_utc,
    )
    return _build_entry(
        checkpoint=checkpoint,
        from_state=None,
        transition_kind="initial",
        skipped_stages=(),
        failure_code=None,
        previous_entry_sha256=None,
    )


def build_case_state_transition(
    *,
    previous: CaseStateEntry,
    to_state: CaseState,
    upstream_hashes: Mapping[str, str],
    produced_artifact_hashes: Mapping[str, str],
    logical_request_ids: Sequence[str],
    created_at_utc: str,
    skipped_stages: Sequence[SkippedStageAnnotation] = (),
    failure_code: str | None = None,
) -> CaseStateEntry:
    """Advance one verified entry while binding every immediate upstream output."""

    if not previous.verify():
        raise CaseStateIntegrityError("previous case-state entry is invalid")
    missing_or_changed = {
        key: value
        for key, value in previous.checkpoint.produced_artifact_hashes.items()
        if upstream_hashes.get(key) != value
    }
    if missing_or_changed:
        raise CaseStateIntegrityError(
            "transition upstream hashes do not bind previous produced artifacts"
        )
    frozen_skips = tuple(skipped_stages)
    kind = _transition_kind(
        from_state=previous.checkpoint.state,
        to_state=to_state,
        skipped_stages=frozen_skips,
        failure_code=failure_code,
    )
    checkpoint = build_case_checkpoint(
        opaque_case_id=previous.checkpoint.opaque_case_id,
        experiment_version=previous.checkpoint.experiment_version,
        state=to_state,
        sequence=previous.checkpoint.sequence + 1,
        upstream_hashes=upstream_hashes,
        produced_artifact_hashes=produced_artifact_hashes,
        logical_request_ids=logical_request_ids,
        created_at_utc=created_at_utc,
    )
    return _build_entry(
        checkpoint=checkpoint,
        from_state=previous.checkpoint.state,
        transition_kind=kind,
        skipped_stages=frozen_skips,
        failure_code=failure_code,
        previous_entry_sha256=previous.entry_sha256,
    )


class CaseStateStore:
    """Persist and restore an immutable, canonical case-state chain."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve(strict=False)

    def write(self, entry: CaseStateEntry) -> Path:
        """Atomically append one entry; identical rewrites are idempotent."""

        if not entry.verify():
            raise CaseStateIntegrityError("cannot persist invalid entry")
        chain = self.restore_chain(entry.checkpoint.opaque_case_id)
        path = self._entry_path(entry)
        data = canonical_json_bytes(entry.model_dump(mode="json"))
        if path.exists():
            try:
                existing = path.read_bytes()
            except OSError as error:
                raise CaseStateIntegrityError(
                    "persisted case state is unreadable or invalid"
                ) from error
            if existing != data:
                raise CaseStateIntegrityError("case-state entry is immutable")
            return path
        if chain:
            self._verify_link(chain[-1], entry)
        elif entry.transition_kind != "initial":
            raise CaseStateIntegrityError("case-state chain must begin at DISCOVERED")
        atomic_write_bytes(path, data)
        return path

    def restore_chain(self, opaque_case_id: str) -> tuple[CaseStateEntry, ...]:
        """Load a complete chain only after replaying every hash and transition."""

        self._require_opaque_case_id(opaque_case_id)
        root = self._case_root(opaque_case_id)
        if not root.exists():
            return ()
        try:
            paths = tuple(sorted(root.iterdir()))
            if any(not path.is_file() or path.suffix != ".json" for path in paths):
                raise ValueError("unexpected checkpoint path")
            entries: list[CaseStateEntry] = []
            for sequence, path in enumerate(paths, start=1):
                document = json.loads(path.read_bytes())
                entry = CaseStateEntry.model_validate(document)
                if (
                    not entry.verify()
                    or entry.checkpoint.opaque_case_id != opaque_case_id
                    or entry.checkpoint.sequence != sequence
                    or path != self._entry_path(entry)
                ):
                    raise ValueError("checkpoint identity or hash is invalid")
                if entries:
                    self._verify_link(entries[-1], entry)
                elif entry.transition_kind != "initial":
                    raise ValueError("checkpoint chain lacks initial state")
                entries.append(entry)
        except (OSError, TypeError, ValueError) as error:
            raise CaseStateIntegrityError(
                "persisted case state is unreadable or invalid"
            ) from error
        return tuple(entries)

    def restore_latest(
        self,
        opaque_case_id: str,
        *,
        expected_upstream_hashes: Mapping[str, str],
    ) -> CaseStateEntry:
        """Restore the latest state only when its complete input binding matches."""

        chain = self.restore_chain(opaque_case_id)
        if not chain:
            raise CaseStateIntegrityError("case-state chain does not exist")
        latest = chain[-1]
        if latest.checkpoint.upstream_hashes != dict(expected_upstream_hashes):
            raise CaseStateIntegrityError("case-state upstream hash drift detected")
        return latest

    def _verify_link(self, previous: CaseStateEntry, current: CaseStateEntry) -> None:
        if (
            current.previous_entry_sha256 != previous.entry_sha256
            or current.from_state != previous.checkpoint.state
            or current.checkpoint.sequence != previous.checkpoint.sequence + 1
            or current.checkpoint.opaque_case_id
            != previous.checkpoint.opaque_case_id
            or current.checkpoint.experiment_version
            != previous.checkpoint.experiment_version
        ):
            raise CaseStateIntegrityError("case-state checkpoint chain is invalid")
        missing_or_changed = {
            key: value
            for key, value in previous.checkpoint.produced_artifact_hashes.items()
            if current.checkpoint.upstream_hashes.get(key) != value
        }
        if missing_or_changed:
            raise CaseStateIntegrityError("case-state upstream hash chain is invalid")
        if not current.verify():
            raise CaseStateIntegrityError("case-state transition is invalid")

    def _case_root(self, opaque_case_id: str) -> Path:
        self._require_opaque_case_id(opaque_case_id)
        return self._root / opaque_case_id / "checkpoints"

    def _entry_path(self, entry: CaseStateEntry) -> Path:
        checkpoint = entry.checkpoint
        return self._case_root(checkpoint.opaque_case_id) / (
            f"{checkpoint.sequence:04d}-{checkpoint.state}.json"
        )

    @staticmethod
    def _require_opaque_case_id(opaque_case_id: str) -> None:
        if re.fullmatch(r"opq_[0-9a-f]{32}", opaque_case_id) is None:
            raise CaseStateIntegrityError("invalid opaque case identifier")


def _build_entry(
    *,
    checkpoint: CaseCheckpoint,
    from_state: CaseState | None,
    transition_kind: TransitionKind,
    skipped_stages: tuple[SkippedStageAnnotation, ...],
    failure_code: str | None,
    previous_entry_sha256: str | None,
) -> CaseStateEntry:
    unsigned: dict[str, object] = {
        "schema_version": CASE_STATE_ENTRY_SCHEMA_VERSION,
        "checkpoint": checkpoint.model_dump(mode="json"),
        "from_state": from_state,
        "transition_kind": transition_kind,
        "skipped_stages": [item.model_dump(mode="json") for item in skipped_stages],
        "failure_code": failure_code,
        "previous_entry_sha256": previous_entry_sha256,
    }
    entry = CaseStateEntry.model_validate(
        {**unsigned, "entry_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    if not entry.verify():  # pragma: no cover - construction invariant
        raise AssertionError("case-state entry verification failed")
    return entry


def _transition_kind(
    *,
    from_state: CaseState,
    to_state: CaseState,
    skipped_stages: Sequence[SkippedStageAnnotation],
    failure_code: str | None,
) -> TransitionKind:
    if from_state in TERMINAL_STATES:
        raise IllegalCaseTransition(
            f"terminal state {from_state} cannot advance"
        )
    if to_state == "FAILURE_SEALED":
        if failure_code is None or skipped_stages:
            raise IllegalCaseTransition(
                "FAILURE_SEALED requires a failure code and no skip annotations"
            )
        return "failure"
    if failure_code is not None or to_state not in PIPELINE_STATES:
        raise IllegalCaseTransition(
            f"illegal case transition from {from_state} to {to_state}"
        )
    start = PIPELINE_STATES.index(from_state)
    end = PIPELINE_STATES.index(to_state)
    if end <= start:
        raise IllegalCaseTransition(
            f"illegal case transition from {from_state} to {to_state}"
        )
    expected_skips = PIPELINE_STATES[start + 1 : end]
    provided_skips = tuple(item.state for item in skipped_stages)
    if not expected_skips:
        if provided_skips:
            raise IllegalCaseTransition("direct transition cannot carry skip annotations")
        return "advance"
    non_skippable = tuple(
        state for state in expected_skips if state not in SKIPPABLE_STATES
    )
    if non_skippable:
        if any(state in non_skippable for state in provided_skips):
            raise IllegalCaseTransition(
                f"state {non_skippable[0]} is not skippable"
            )
        raise IllegalCaseTransition(
            f"illegal case transition from {from_state} to {to_state}"
        )
    if provided_skips != expected_skips:
        raise IllegalCaseTransition(
            "skip annotations must exactly match bypassed semantic or report stages"
        )
    return "skip"
