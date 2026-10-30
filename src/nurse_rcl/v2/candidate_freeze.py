"""Pre-semantic candidate freezing and immutable membership enforcement."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.observation_schemas import CandidateUniverse

CANDIDATE_FREEZE_SCHEMA_VERSION: Literal["nurse.v2.frozen-candidate-universe/1"] = (
    "nurse.v2.frozen-candidate-universe/1"
)
CANDIDATE_SCOPE_DECISION_SCHEMA_VERSION: Literal["nurse.v2.candidate-scope-decision/1"] = (
    "nurse.v2.candidate-scope-decision/1"
)

CandidateScopeOperation = Literal[
    "semantic_reference",
    "candidate_transform",
    "ranking",
]
CandidateScopeReason = Literal[
    "candidate_scope_accepted",
    "duplicate_candidate_reference",
    "candidate_outside_frozen_universe",
    "candidate_removed_from_frozen_universe",
]


def _scope_reason(
    *,
    duplicate_candidate_keys: Sequence[str],
    added_candidate_keys: Sequence[str],
    missing_candidate_keys: Sequence[str],
) -> CandidateScopeReason:
    if duplicate_candidate_keys:
        return "duplicate_candidate_reference"
    if added_candidate_keys:
        return "candidate_outside_frozen_universe"
    if missing_candidate_keys:
        return "candidate_removed_from_frozen_universe"
    return "candidate_scope_accepted"


class FrozenCandidateUniverse(StrictFrozenModel):
    """Content-addressed token required by semantic and ranking consumers."""

    schema_version: Literal["nurse.v2.frozen-candidate-universe/1"] = (
        CANDIDATE_FREEZE_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_universe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_count: int = Field(ge=1)
    candidate_service_keys: tuple[str, ...] = Field(min_length=1)
    freeze_stage: Literal["pre_semantic"] = "pre_semantic"
    candidate_set_mutable: Literal[False] = False
    freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_candidate_keys(self) -> Self:
        expected = tuple(sorted(set(self.candidate_service_keys)))
        if self.candidate_service_keys != expected:
            raise ValueError("frozen candidate keys must be sorted and unique")
        if self.candidate_count != len(self.candidate_service_keys):
            raise ValueError("frozen candidate count does not match candidate keys")
        return self

    def verify(self) -> bool:
        try:
            validated = FrozenCandidateUniverse.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = validated.model_dump(mode="json", exclude={"freeze_sha256"})
        except (AttributeError, TypeError, ValueError):
            return False
        return validated == self and validated.freeze_sha256 == sha256_hex(
            canonical_json_bytes(unsigned)
        )


class CandidateScopeDecision(StrictFrozenModel):
    """Canonical acceptance or rejection of candidate keys at one consumer."""

    schema_version: Literal["nurse.v2.candidate-scope-decision/1"] = (
        CANDIDATE_SCOPE_DECISION_SCHEMA_VERSION
    )
    decision_id: str = Field(pattern=r"^cgd_[0-9a-f]{24}$")
    candidate_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    operation: CandidateScopeOperation
    requested_candidate_keys: tuple[str, ...]
    duplicate_candidate_keys: tuple[str, ...]
    added_candidate_keys: tuple[str, ...]
    missing_candidate_keys: tuple[str, ...]
    accepted: bool
    reason_code: CandidateScopeReason
    decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_decision(self) -> Self:
        for values, name in (
            (self.requested_candidate_keys, "requested candidate keys"),
            (self.duplicate_candidate_keys, "duplicate candidate keys"),
            (self.added_candidate_keys, "added candidate keys"),
            (self.missing_candidate_keys, "missing candidate keys"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        expected_reason = _scope_reason(
            duplicate_candidate_keys=self.duplicate_candidate_keys,
            added_candidate_keys=self.added_candidate_keys,
            missing_candidate_keys=self.missing_candidate_keys,
        )
        if self.reason_code != expected_reason:
            raise ValueError("candidate decision reason does not describe scope differences")
        if self.accepted != (expected_reason == "candidate_scope_accepted"):
            raise ValueError("candidate decision acceptance does not match scope differences")
        return self

    def verify(self) -> bool:
        try:
            validated = CandidateScopeDecision.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = validated.model_dump(
                mode="json",
                exclude={"decision_id", "decision_sha256"},
            )
            digest = sha256_hex(canonical_json_bytes(unsigned))
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            validated == self
            and validated.decision_sha256 == digest
            and validated.decision_id == f"cgd_{digest[:24]}"
        )


class CandidateUniverseViolation(ValueError):
    """Raised with the immutable decision that rejected a candidate operation."""

    def __init__(self, decision: CandidateScopeDecision) -> None:
        self.decision = decision
        super().__init__(decision.reason_code)


def freeze_candidate_universe(
    candidate_universe: CandidateUniverse,
) -> FrozenCandidateUniverse:
    """Freeze a verified C before any semantic consumer can reference it."""

    try:
        validated = CandidateUniverse.model_validate(
            candidate_universe.model_dump(mode="json", warnings="error")
        )
        verified = validated == candidate_universe and validated.verify()
    except (AttributeError, TypeError, ValueError):
        verified = False
    if not verified:
        raise ValueError("candidate universe failed verification")
    candidate_keys = tuple(item.service_entity_key for item in validated.candidates)
    unsigned: dict[str, object] = {
        "schema_version": CANDIDATE_FREEZE_SCHEMA_VERSION,
        "opaque_case_id": validated.opaque_case_id,
        "candidate_universe_sha256": validated.universe_sha256,
        "observation_contract_sha256": validated.observation_contract_sha256,
        "candidate_count": len(candidate_keys),
        "candidate_service_keys": candidate_keys,
        "freeze_stage": "pre_semantic",
        "candidate_set_mutable": False,
    }
    frozen = FrozenCandidateUniverse.model_validate(
        {
            **unsigned,
            "freeze_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not frozen.verify():  # pragma: no cover - construction invariant
        raise AssertionError("candidate freeze verification failed")
    return frozen


def require_frozen_candidate_universe(
    *,
    frozen: FrozenCandidateUniverse,
    candidate_universe: CandidateUniverse,
) -> CandidateUniverse:
    """Reject a universe that does not exactly match its pre-semantic freeze."""

    if not frozen.verify():
        raise ValueError("candidate freeze failed verification")
    try:
        candidate_keys = tuple(item.service_entity_key for item in candidate_universe.candidates)
        matches = (
            candidate_universe.verify()
            and candidate_universe.opaque_case_id == frozen.opaque_case_id
            and candidate_universe.universe_sha256 == frozen.candidate_universe_sha256
            and candidate_universe.observation_contract_sha256 == frozen.observation_contract_sha256
            and candidate_keys == frozen.candidate_service_keys
        )
    except (AttributeError, TypeError, ValueError):
        matches = False
    if not matches:
        raise ValueError("candidate universe does not match candidate freeze")
    return candidate_universe


def evaluate_candidate_scope(
    *,
    frozen: FrozenCandidateUniverse,
    candidate_service_keys: Sequence[str],
    operation: CandidateScopeOperation,
) -> CandidateScopeDecision:
    """Evaluate a semantic subset or an exact-set transform/ranking request."""

    if not frozen.verify():
        raise ValueError("candidate freeze failed verification")
    counts = Counter(candidate_service_keys)
    requested = tuple(sorted(counts))
    duplicates = tuple(sorted(key for key, count in counts.items() if count > 1))
    frozen_keys = set(frozen.candidate_service_keys)
    requested_keys = set(requested)
    added = tuple(sorted(requested_keys - frozen_keys))
    missing = (
        tuple(sorted(frozen_keys - requested_keys))
        if operation in {"candidate_transform", "ranking"}
        else ()
    )
    reason = _scope_reason(
        duplicate_candidate_keys=duplicates,
        added_candidate_keys=added,
        missing_candidate_keys=missing,
    )
    unsigned: dict[str, object] = {
        "schema_version": CANDIDATE_SCOPE_DECISION_SCHEMA_VERSION,
        "candidate_freeze_sha256": frozen.freeze_sha256,
        "operation": operation,
        "requested_candidate_keys": requested,
        "duplicate_candidate_keys": duplicates,
        "added_candidate_keys": added,
        "missing_candidate_keys": missing,
        "accepted": reason == "candidate_scope_accepted",
        "reason_code": reason,
    }
    digest = sha256_hex(canonical_json_bytes(unsigned))
    decision = CandidateScopeDecision.model_validate(
        {
            **unsigned,
            "decision_id": f"cgd_{digest[:24]}",
            "decision_sha256": digest,
        }
    )
    if not decision.verify():  # pragma: no cover - construction invariant
        raise AssertionError("candidate scope decision verification failed")
    return decision


def require_candidate_scope(
    *,
    frozen: FrozenCandidateUniverse,
    candidate_service_keys: Sequence[str],
    operation: CandidateScopeOperation,
) -> CandidateScopeDecision:
    """Return an accepted decision or raise with the auditable rejection."""

    decision = evaluate_candidate_scope(
        frozen=frozen,
        candidate_service_keys=candidate_service_keys,
        operation=operation,
    )
    if not decision.accepted:
        raise CandidateUniverseViolation(decision)
    return decision
