"""Small worker failure receipt, not a provider ledger or an agent trajectory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_contracts import OpsCoreIdentity
from nurse_rcl.v2.live.ops_failure import InvestigationStartupFailure as InvestigationStartupFailure
from nurse_rcl.v2.live.ops_failure import OpsFailureDetail
from nurse_rcl.v2.live.staged_case import require_plain_path

LIMIT = 64 * 1024


class StartupFailure(StrictFrozenModel):
    schema_version: Literal["nurse.v2.investigation-startup-failure/1"] = (
        "nurse.v2.investigation-startup-failure/1"
    )
    identity: OpsCoreIdentity
    code_sha256: SHA256
    execution_sha256: SHA256 | None
    profile: Literal[
        "hermes-sre-tools/9",
        "hermes-sre-tools/10",
        "hermes-sre-tools/11",
        "hermes-sre-tools/12",
        "hermes-sre-tools/13",
        "hermes-sre-tools/14",
        "hermes-sre-tools/15",
        "hermes-sre-tools/16",
        "hermes-sre-tools/17",
    ] = "hermes-sre-tools/9"
    worker_invocation_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    provider_boundary_entered: bool = Field(strict=True)
    prior_state_present: bool = Field(default=False, strict=True)
    primary_failure: OpsFailureDetail
    secondary_failures: tuple[OpsFailureDetail, ...] = Field(max_length=16)
    receipt_sha256: SHA256 = "0" * 64


def save_startup_failure(
    root: Path,
    *,
    identity: OpsCoreIdentity,
    code: str,
    execution: str | None,
    primary: OpsFailureDetail,
    secondary: tuple[OpsFailureDetail, ...],
    provider_boundary_entered: bool,
    invocation_id: str | None = None,
    prior_state_present: bool = False,
    profile: Literal[
        "hermes-sre-tools/9",
        "hermes-sre-tools/10",
        "hermes-sre-tools/11",
        "hermes-sre-tools/12",
        "hermes-sre-tools/13",
        "hermes-sre-tools/14",
        "hermes-sre-tools/15",
        "hermes-sre-tools/16",
        "hermes-sre-tools/17",
    ] = "hermes-sre-tools/9",
) -> StartupFailure:
    directory = root / "investigation"
    require_plain_path(root)
    directory.mkdir(mode=0o700, exist_ok=True)
    require_plain_path(directory)
    receipt = StartupFailure(
        profile=profile,
        identity=identity,
        code_sha256=code,
        execution_sha256=execution,
        worker_invocation_id=invocation_id or uuid4().hex,
        primary_failure=primary,
        secondary_failures=secondary[:16],
        provider_boundary_entered=provider_boundary_entered,
        prior_state_present=prior_state_present,
    )
    pin = sha256_hex(
        canonical_json_bytes(receipt.model_dump(mode="json", exclude={"receipt_sha256"}))
    )
    receipt = receipt.model_copy(update={"receipt_sha256": pin})
    raw = canonical_json_bytes(receipt.model_dump(mode="json"))
    if len(raw) > LIMIT:
        raise ValueError("startup failure receipt exceeds capacity")
    atomic_write_bytes(directory / "startup-failure.json", raw)
    return receipt


def read_startup_failure(
    root: Path,
    *,
    identity: OpsCoreIdentity,
    code: str,
    execution: str | None,
    invocation_id: str | None = None,
    profile: Literal[
        "hermes-sre-tools/9",
        "hermes-sre-tools/10",
        "hermes-sre-tools/11",
        "hermes-sre-tools/12",
        "hermes-sre-tools/13",
        "hermes-sre-tools/14",
        "hermes-sre-tools/15",
        "hermes-sre-tools/16",
        "hermes-sre-tools/17",
    ] = "hermes-sre-tools/9",
) -> StartupFailure:
    raw = _read_bytes(root / "investigation/startup-failure.json", LIMIT)
    receipt = StartupFailure.model_validate_json(raw)
    if receipt.profile != profile:
        raise ValueError("startup failure profile differs")
    if (receipt.identity, receipt.code_sha256, receipt.execution_sha256) != (
        identity,
        code,
        execution,
    ) or (
        canonical_json_bytes(receipt.model_dump(mode="json")) != raw
        or sha256_hex(
            canonical_json_bytes(receipt.model_dump(mode="json", exclude={"receipt_sha256"}))
        )
        != receipt.receipt_sha256
    ):
        raise ValueError("startup failure binding differs")
    if invocation_id is not None and receipt.worker_invocation_id != invocation_id:
        raise ValueError("startup receipt is from another worker invocation")
    return receipt


def no_send_eligible(
    receipt: StartupFailure, *, worker_terminated: bool, report_started: bool, usage: dict[str, Any]
) -> bool:
    """Usage must be the complete, identity-checked provider lifetime inventory."""
    return bool(
        type(receipt) is StartupFailure
        and receipt.execution_sha256 is not None
        and worker_terminated is True
        and report_started is False
        and not receipt.provider_boundary_entered
        and not receipt.prior_state_present
        and usage.get("attempt_usage") == []
        and usage.get("response_reuses") == 0
    )
