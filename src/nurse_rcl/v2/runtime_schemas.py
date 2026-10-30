"""Versioned schemas for semantic calls, recovery, evaluation, and sealing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.exposure_registry import ExposureClass
from nurse_rcl.v2.split_allocator import PartitionName

CACHE_IDENTITY_SCHEMA_VERSION: Literal[
    "nurse.v2.cache-identity/1"
] = "nurse.v2.cache-identity/1"
ATTEMPT_LEDGER_SCHEMA_VERSION: Literal[
    "nurse.v2.attempt-ledger/1"
] = "nurse.v2.attempt-ledger/1"
COST_RECORD_SCHEMA_VERSION: Literal[
    "nurse.v2.cost-record/1"
] = "nurse.v2.cost-record/1"
SEMANTIC_CALL_SCHEMA_VERSION: Literal[
    "nurse.v2.semantic-call/1"
] = "nurse.v2.semantic-call/1"
CASE_CHECKPOINT_SCHEMA_VERSION: Literal[
    "nurse.v2.case-checkpoint/1"
] = "nurse.v2.case-checkpoint/1"
CASE_EXPOSURE_SCHEMA_VERSION: Literal[
    "nurse.v2.case-exposure-reference/1"
] = "nurse.v2.case-exposure-reference/1"
CASE_EVALUATION_SCHEMA_VERSION: Literal[
    "nurse.v2.case-evaluation-output/1"
] = "nurse.v2.case-evaluation-output/1"
SEALED_AUDIT_SCHEMA_VERSION: Literal[
    "nurse.v2.sealed-audit-package/1"
] = "nurse.v2.sealed-audit-package/1"

RunMode = Literal[
    "full",
    "partial_semantic",
    "b0_fallback",
    "deterministic_report_fallback",
    "failure",
]
CaseState = Literal[
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
    "FAILURE_SEALED",
]
_UTC_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
_REQUIRED_AUDIT_ARTIFACTS = frozenset(
    {
        "input",
        "configuration",
        "entities",
        "observations",
        "candidate_universe",
        "focus_set",
        "hypotheses",
        "graph",
        "scores",
        "top_k",
        "history",
        "rules",
        "calls",
        "queries",
        "costs",
        "degradation",
    }
)


class CacheIdentity(StrictFrozenModel):
    """Secret-free identity over every input that can change semantic output."""

    schema_version: Literal["nurse.v2.cache-identity/1"] = (
        CACHE_IDENTITY_SCHEMA_VERSION
    )
    endpoint: str = Field(min_length=1)
    model: str = Field(min_length=1)
    generation_settings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mechanism_library_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    experiment_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    history_library_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    cache_key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"cache_key_sha256"})
        return self.cache_key_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_cache_identity(
    *,
    endpoint: str,
    model: str,
    generation_settings_sha256: str,
    payload_sha256: str,
    prompt_sha256: str,
    output_schema_sha256: str,
    mechanism_library_sha256: str,
    adapter_sha256: str,
    experiment_configuration_sha256: str,
    code_sha256: str,
    history_library_sha256: str | None,
) -> CacheIdentity:
    unsigned: dict[str, object] = {
        "schema_version": CACHE_IDENTITY_SCHEMA_VERSION,
        "endpoint": endpoint,
        "model": model,
        "generation_settings_sha256": generation_settings_sha256,
        "payload_sha256": payload_sha256,
        "prompt_sha256": prompt_sha256,
        "output_schema_sha256": output_schema_sha256,
        "mechanism_library_sha256": mechanism_library_sha256,
        "adapter_sha256": adapter_sha256,
        "experiment_configuration_sha256": experiment_configuration_sha256,
        "code_sha256": code_sha256,
        "history_library_sha256": history_library_sha256,
    }
    identity = CacheIdentity.model_validate(
        {**unsigned, "cache_key_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    if not identity.verify():  # pragma: no cover - construction invariant
        raise AssertionError("cache identity verification failed")
    return identity


class AttemptRecord(StrictFrozenModel):
    """One transport attempt; its number is scoped to a persistent logical request."""

    attempt_number: int = Field(ge=1)
    started_at_utc: str = Field(pattern=_UTC_PATTERN)
    completed_at_utc: str = Field(pattern=_UTC_PATTERN)
    outcome: Literal[
        "timeout",
        "transport_error",
        "malformed",
        "structurally_valid",
        "terminal_error",
    ]
    retryable: bool
    response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    error_code: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,63}$")
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def require_consistent_outcome(self) -> Self:
        if self.completed_at_utc < self.started_at_utc:
            raise ValueError("attempt completion precedes start")
        if self.outcome == "structurally_valid":
            if (
                self.response_sha256 is None
                or self.error_code is not None
                or self.retryable
            ):
                raise ValueError("structurally valid attempt is terminal success")
        elif self.error_code is None:
            raise ValueError("failed attempt requires an error code")
        if self.outcome == "terminal_error" and self.retryable:
            raise ValueError("terminal attempt cannot be retryable")
        return self


class AttemptLedger(StrictFrozenModel):
    """Cross-restart lifetime ledger capped at exactly five possible attempts."""

    schema_version: Literal["nurse.v2.attempt-ledger/1"] = (
        ATTEMPT_LEDGER_SCHEMA_VERSION
    )
    logical_request_id: str = Field(pattern=r"^lrq_[0-9a-f]{24}$")
    max_attempts: Literal[5] = 5
    attempts: tuple[AttemptRecord, ...]
    attempt_count: int = Field(ge=0, le=5)
    primary_response_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    exhausted: bool
    ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if self.attempt_count != len(self.attempts) or len(self.attempts) > 5:
            return False
        if tuple(item.attempt_number for item in self.attempts) != tuple(
            range(1, len(self.attempts) + 1)
        ):
            return False
        valid = tuple(
            item for item in self.attempts if item.outcome == "structurally_valid"
        )
        if len(valid) > 1:
            return False
        if valid and valid[0] != self.attempts[-1]:
            return False
        expected_primary = valid[0].response_sha256 if valid else None
        if self.primary_response_sha256 != expected_primary:
            return False
        if self.exhausted != (len(self.attempts) == 5 and not valid):
            return False
        unsigned = self.model_dump(mode="json", exclude={"ledger_sha256"})
        return self.ledger_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_attempt_ledger(
    *,
    logical_request_id: str,
    attempts: Sequence[AttemptRecord],
) -> AttemptLedger:
    frozen = tuple(attempts)
    if len(frozen) > 5 or any(item.attempt_number > 5 for item in frozen):
        raise ValueError("logical request permits at most five lifetime attempts")
    if tuple(item.attempt_number for item in frozen) != tuple(
        range(1, len(frozen) + 1)
    ):
        raise ValueError("attempt numbers must be sequential across restarts")
    valid_indexes = tuple(
        index for index, item in enumerate(frozen) if item.outcome == "structurally_valid"
    )
    if valid_indexes and valid_indexes != (len(frozen) - 1,):
        raise ValueError("attempts cannot continue after structurally valid response")
    primary = frozen[-1].response_sha256 if valid_indexes else None
    unsigned: dict[str, object] = {
        "schema_version": ATTEMPT_LEDGER_SCHEMA_VERSION,
        "logical_request_id": logical_request_id,
        "max_attempts": 5,
        "attempts": [item.model_dump(mode="json") for item in frozen],
        "attempt_count": len(frozen),
        "primary_response_sha256": primary,
        "exhausted": len(frozen) == 5 and not valid_indexes,
    }
    ledger = AttemptLedger.model_validate(
        {**unsigned, "ledger_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    if not ledger.verify():  # pragma: no cover - construction invariant
        raise AssertionError("attempt ledger verification failed")
    return ledger


class CostRecord(StrictFrozenModel):
    schema_version: Literal["nurse.v2.cost-record/1"] = COST_RECORD_SCHEMA_VERSION
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    actual_cost: str = Field(pattern=r"^\d+\.\d{6}$")
    currency: Literal["USD", "CNY"]
    source: Literal["provider_receipt", "estimated"]
    cost_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            return False
        if self.cached_input_tokens > self.input_tokens:
            return False
        unsigned = self.model_dump(mode="json", exclude={"cost_sha256"})
        return self.cost_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_cost_record(
    *,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int,
    actual_cost: str,
    currency: Literal["USD", "CNY"],
    source: Literal["provider_receipt", "estimated"],
) -> CostRecord:
    unsigned: dict[str, object] = {
        "schema_version": COST_RECORD_SCHEMA_VERSION,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "total_tokens": input_tokens + output_tokens,
        "actual_cost": actual_cost,
        "currency": currency,
        "source": source,
    }
    cost = CostRecord.model_validate(
        {**unsigned, "cost_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    if not cost.verify():  # pragma: no cover - construction invariant
        raise AssertionError("cost record verification failed")
    return cost


class SemanticCall(StrictFrozenModel):
    """One logical hypothesis or report request and its entire attempt lifecycle."""

    schema_version: Literal["nurse.v2.semantic-call/1"] = (
        SEMANTIC_CALL_SCHEMA_VERSION
    )
    semantic_call_id: str = Field(pattern=r"^sem_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    stage: Literal["hypothesis", "hypothesis_revision", "report"]
    endpoint: str = Field(min_length=1)
    model: str = Field(min_length=1)
    llm_in_flight: Literal[1] = 1
    input_token_budget: int = Field(gt=0)
    output_token_budget: int = Field(gt=0)
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_ledger: AttemptLedger
    cost: CostRecord
    status: Literal["success", "partial", "fallback", "terminal_failure"]
    primary_response_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    call_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_bound_primary_result(self) -> Self:
        if not self.attempt_ledger.verify() or not self.cost.verify():
            raise ValueError("semantic call embeds invalid ledger or cost")
        if self.primary_response_sha256 != self.attempt_ledger.primary_response_sha256:
            raise ValueError("semantic call primary response does not match ledger")
        if self.status in {"success", "partial"} and self.primary_response_sha256 is None:
            raise ValueError("successful semantic call requires a primary response")
        if self.cost.input_tokens != sum(
            item.input_tokens for item in self.attempt_ledger.attempts
        ) or self.cost.output_tokens != sum(
            item.output_tokens for item in self.attempt_ledger.attempts
        ):
            raise ValueError("semantic call cost tokens do not match attempt ledger")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(
            mode="json",
            exclude={"semantic_call_id", "call_sha256"},
        )
        digest = sha256_hex(canonical_json_bytes(unsigned))
        return self.call_sha256 == digest and self.semantic_call_id == (
            f"sem_{digest[:24]}"
        )


def build_semantic_call(
    *,
    opaque_case_id: str,
    stage: Literal["hypothesis", "hypothesis_revision", "report"],
    endpoint: str,
    model: str,
    input_token_budget: int,
    output_token_budget: int,
    payload_sha256: str,
    prompt_sha256: str,
    output_schema_sha256: str,
    attempt_ledger: AttemptLedger,
    cost: CostRecord,
    status: Literal["success", "partial", "fallback", "terminal_failure"],
) -> SemanticCall:
    unsigned: dict[str, object] = {
        "schema_version": SEMANTIC_CALL_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "stage": stage,
        "endpoint": endpoint,
        "model": model,
        "llm_in_flight": 1,
        "input_token_budget": input_token_budget,
        "output_token_budget": output_token_budget,
        "payload_sha256": payload_sha256,
        "prompt_sha256": prompt_sha256,
        "output_schema_sha256": output_schema_sha256,
        "attempt_ledger": attempt_ledger.model_dump(mode="json"),
        "cost": cost.model_dump(mode="json"),
        "status": status,
        "primary_response_sha256": attempt_ledger.primary_response_sha256,
    }
    digest = sha256_hex(canonical_json_bytes(unsigned))
    call = SemanticCall.model_validate(
        {
            **unsigned,
            "semantic_call_id": f"sem_{digest[:24]}",
            "call_sha256": digest,
        }
    )
    if not call.verify():  # pragma: no cover - construction invariant
        raise AssertionError("semantic call verification failed")
    return call


class CaseCheckpoint(StrictFrozenModel):
    """Hash-bound state snapshot that must be persisted by atomic replacement."""

    schema_version: Literal["nurse.v2.case-checkpoint/1"] = (
        CASE_CHECKPOINT_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    experiment_version: str = Field(min_length=1)
    state: CaseState
    sequence: int = Field(ge=1)
    upstream_hashes: dict[str, str]
    produced_artifact_hashes: dict[str, str]
    logical_request_ids: tuple[str, ...]
    created_at_utc: str = Field(pattern=_UTC_PATTERN)
    atomic_write_required: Literal[True] = True
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if not _valid_hash_mapping(self.upstream_hashes):
            return False
        if not _valid_hash_mapping(self.produced_artifact_hashes):
            return False
        if self.logical_request_ids != tuple(sorted(set(self.logical_request_ids))):
            return False
        unsigned = self.model_dump(mode="json", exclude={"checkpoint_sha256"})
        return self.checkpoint_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_case_checkpoint(
    *,
    opaque_case_id: str,
    experiment_version: str,
    state: CaseState,
    sequence: int,
    upstream_hashes: Mapping[str, str],
    produced_artifact_hashes: Mapping[str, str],
    logical_request_ids: Sequence[str],
    created_at_utc: str,
) -> CaseCheckpoint:
    unsigned: dict[str, object] = {
        "schema_version": CASE_CHECKPOINT_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "experiment_version": experiment_version,
        "state": state,
        "sequence": sequence,
        "upstream_hashes": dict(sorted(upstream_hashes.items())),
        "produced_artifact_hashes": dict(
            sorted(produced_artifact_hashes.items())
        ),
        "logical_request_ids": tuple(sorted(set(logical_request_ids))),
        "created_at_utc": created_at_utc,
        "atomic_write_required": True,
    }
    checkpoint = CaseCheckpoint.model_validate(
        {
            **unsigned,
            "checkpoint_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not checkpoint.verify():
        raise ValueError("checkpoint contains invalid artifact hashes")
    return checkpoint


class CaseExposureRecord(StrictFrozenModel):
    """Evaluator-only binding from opaque case to frozen exposure history."""

    schema_version: Literal["nurse.v2.case-exposure-reference/1"] = (
        CASE_EXPOSURE_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    partition: PartitionName
    exposure_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_event_ids: tuple[str, ...]
    exposure_classes: tuple[ExposureClass, ...]
    evaluator_only: Literal[True] = True
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if self.registry_event_ids != tuple(sorted(set(self.registry_event_ids))):
            return False
        if self.exposure_classes != tuple(sorted(set(self.exposure_classes))):
            return False
        if bool(self.registry_event_ids) != bool(self.exposure_classes):
            return False
        unsigned = self.model_dump(mode="json", exclude={"record_sha256"})
        return self.record_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_case_exposure_record(
    *,
    opaque_case_id: str,
    partition: PartitionName,
    exposure_snapshot_sha256: str,
    registry_event_ids: Sequence[str],
    exposure_classes: Sequence[ExposureClass],
) -> CaseExposureRecord:
    unsigned: dict[str, object] = {
        "schema_version": CASE_EXPOSURE_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "partition": partition,
        "exposure_snapshot_sha256": exposure_snapshot_sha256,
        "registry_event_ids": tuple(sorted(set(registry_event_ids))),
        "exposure_classes": tuple(sorted(set(exposure_classes))),
        "evaluator_only": True,
    }
    record = CaseExposureRecord.model_validate(
        {**unsigned, "record_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    if not record.verify():
        raise ValueError("exposure classes and registry events must both be present")
    return record


class CaseEvaluationOutput(StrictFrozenModel):
    """Evaluator-only per-case metrics that cannot feed inference artifacts."""

    schema_version: Literal["nurse.v2.case-evaluation-output/1"] = (
        CASE_EVALUATION_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    partition: PartitionName
    prediction_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sealed_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_mode: RunMode
    included_in_denominator: Literal[True] = True
    acc_at_1: bool
    acc_at_3: bool
    acc_at_5: bool
    reciprocal_rank: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    discovery_covered: bool
    binding_covered: bool
    candidate_covered: bool
    initial_focus_covered: bool
    final_focus_covered: bool
    evaluator_only: Literal[True] = True
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_consistent_metrics(self) -> Self:
        if (self.acc_at_1 and not self.acc_at_3) or (
            self.acc_at_3 and not self.acc_at_5
        ):
            raise ValueError("accuracy at K must be monotone")
        if self.acc_at_1 and self.reciprocal_rank != 1.0:
            raise ValueError("Top-1 hit requires reciprocal rank one")
        if self.run_mode == "failure" and (
            self.acc_at_1 or self.acc_at_3 or self.acc_at_5 or self.reciprocal_rank
        ):
            raise ValueError("failure evaluation must retain zero localization metrics")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"output_sha256"})
        return self.output_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_case_evaluation_output(
    *,
    opaque_case_id: str,
    partition: PartitionName,
    prediction_artifact_sha256: str,
    sealed_audit_sha256: str,
    run_mode: RunMode,
    acc_at_1: bool,
    acc_at_3: bool,
    acc_at_5: bool,
    reciprocal_rank: float,
    discovery_covered: bool,
    binding_covered: bool,
    candidate_covered: bool,
    initial_focus_covered: bool,
    final_focus_covered: bool,
) -> CaseEvaluationOutput:
    unsigned: dict[str, object] = {
        "schema_version": CASE_EVALUATION_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "partition": partition,
        "prediction_artifact_sha256": prediction_artifact_sha256,
        "sealed_audit_sha256": sealed_audit_sha256,
        "run_mode": run_mode,
        "included_in_denominator": True,
        "acc_at_1": acc_at_1,
        "acc_at_3": acc_at_3,
        "acc_at_5": acc_at_5,
        "reciprocal_rank": reciprocal_rank,
        "discovery_covered": discovery_covered,
        "binding_covered": binding_covered,
        "candidate_covered": candidate_covered,
        "initial_focus_covered": initial_focus_covered,
        "final_focus_covered": final_focus_covered,
        "evaluator_only": True,
    }
    output = CaseEvaluationOutput.model_validate(
        {**unsigned, "output_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    if not output.verify():  # pragma: no cover - construction invariant
        raise AssertionError("evaluation output verification failed")
    return output


class SealedAuditPackage(StrictFrozenModel):
    """Terminal machine-auditable replay chain for success or retained failure."""

    schema_version: Literal["nurse.v2.sealed-audit-package/1"] = (
        SEALED_AUDIT_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    experiment_version: str = Field(min_length=1)
    terminal_state: Literal["SEALED", "FAILURE_SEALED"]
    artifact_hashes: dict[str, str]
    accepted_provenance_ids: tuple[str, ...]
    rejected_provenance_ids: tuple[str, ...]
    conflicted_provenance_ids: tuple[str, ...]
    unknown_provenance_ids: tuple[str, ...]
    prediction_service_keys: tuple[str, ...]
    run_mode: RunMode
    sealed_at_utc: str = Field(pattern=_UTC_PATTERN)
    ranking_mutable: Literal[False] = False
    package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_terminal_consistency(self) -> Self:
        if set(self.artifact_hashes) != _REQUIRED_AUDIT_ARTIFACTS:
            raise ValueError("missing required audit artifacts")
        if not _valid_hash_mapping(self.artifact_hashes):
            raise ValueError("audit artifact hashes are invalid")
        collections = (
            self.accepted_provenance_ids,
            self.rejected_provenance_ids,
            self.conflicted_provenance_ids,
            self.unknown_provenance_ids,
            self.prediction_service_keys,
        )
        if any(values != tuple(sorted(set(values))) for values in collections):
            raise ValueError("audit provenance and predictions must be canonical")
        if self.terminal_state == "FAILURE_SEALED" and self.prediction_service_keys:
            raise ValueError("failure package must have empty prediction")
        if self.terminal_state == "FAILURE_SEALED" and self.run_mode != "failure":
            raise ValueError("failure package must use failure run mode")
        if self.terminal_state == "SEALED" and not self.prediction_service_keys:
            raise ValueError("successful sealed package requires a prediction")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"package_sha256"})
        return self.package_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_sealed_audit_package(
    *,
    opaque_case_id: str,
    experiment_version: str,
    terminal_state: Literal["SEALED", "FAILURE_SEALED"],
    artifact_hashes: Mapping[str, str],
    accepted_provenance_ids: Sequence[str],
    rejected_provenance_ids: Sequence[str],
    conflicted_provenance_ids: Sequence[str],
    unknown_provenance_ids: Sequence[str],
    prediction_service_keys: Sequence[str],
    run_mode: RunMode,
    sealed_at_utc: str,
) -> SealedAuditPackage:
    if set(artifact_hashes) != _REQUIRED_AUDIT_ARTIFACTS:
        missing = sorted(_REQUIRED_AUDIT_ARTIFACTS.difference(artifact_hashes))
        extra = sorted(set(artifact_hashes).difference(_REQUIRED_AUDIT_ARTIFACTS))
        raise ValueError(
            f"missing required audit artifacts: missing={missing}, extra={extra}"
        )
    unsigned: dict[str, object] = {
        "schema_version": SEALED_AUDIT_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "experiment_version": experiment_version,
        "terminal_state": terminal_state,
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "accepted_provenance_ids": tuple(sorted(set(accepted_provenance_ids))),
        "rejected_provenance_ids": tuple(sorted(set(rejected_provenance_ids))),
        "conflicted_provenance_ids": tuple(sorted(set(conflicted_provenance_ids))),
        "unknown_provenance_ids": tuple(sorted(set(unknown_provenance_ids))),
        "prediction_service_keys": tuple(sorted(set(prediction_service_keys))),
        "run_mode": run_mode,
        "sealed_at_utc": sealed_at_utc,
        "ranking_mutable": False,
    }
    package = SealedAuditPackage.model_validate(
        {**unsigned, "package_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    if not package.verify():  # pragma: no cover - construction invariant
        raise AssertionError("sealed audit package verification failed")
    return package


def _valid_hash_mapping(values: Mapping[str, str]) -> bool:
    return bool(values) and all(
        key
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        for key, value in values.items()
    )
