"""Local supervisor receipt; not an agent response, retry slot or case result."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_failure import OpsFailureDetail


class WorkerOutcome(StrictFrozenModel):
    schema_version: Literal["nurse.v2.ops-worker-outcome/1"] = "nurse.v2.ops-worker-outcome/1"
    identity: dict[str, str]
    primary_failure: OpsFailureDetail | None = None
    secondary_failures: tuple[OpsFailureDetail, ...] = Field(default=(), max_length=16)
    state_sha256: SHA256 = "0" * 64


class StartupWorkerOutcome(StrictFrozenModel):
    schema_version: Literal["nurse.v2.ops-worker-outcome/2"] = "nurse.v2.ops-worker-outcome/2"
    identity: dict[str, str]
    primary_failure: OpsFailureDetail
    secondary_failures: tuple[OpsFailureDetail, ...] = Field(default=(), max_length=16)
    # Set only by the supervisor after join has proved the child terminated.
    terminated_startup_receipt_sha256: SHA256
    state_sha256: SHA256 = "0" * 64


def write_outcome(
    root: Path,
    identity: dict[str, str],
    primary: OpsFailureDetail | None = None,
    secondary: tuple[OpsFailureDetail, ...] = (),
    *,
    startup_receipt_sha256: str | None = None,
) -> None:
    state: WorkerOutcome | StartupWorkerOutcome
    if startup_receipt_sha256 is not None:
        if primary is None:
            raise ValueError("startup supervisor proof requires first failure")
        state = StartupWorkerOutcome(
            identity=identity,
            primary_failure=primary,
            secondary_failures=secondary,
            terminated_startup_receipt_sha256=startup_receipt_sha256,
        )
    else:
        state = WorkerOutcome(
            identity=identity, primary_failure=primary, secondary_failures=secondary
        )
    pin = sha256_hex(canonical_json_bytes(state.model_dump(mode="json", exclude={"state_sha256"})))
    raw = canonical_json_bytes(
        state.model_copy(update={"state_sha256": pin}).model_dump(mode="json")
    )
    if len(raw) > 16384:
        raise ValueError("worker outcome capacity exceeded")
    atomic_write_bytes(root / "worker-outcome.json", raw)


def read_outcome(root: Path, identity: dict[str, str]) -> WorkerOutcome | StartupWorkerOutcome:
    raw = _read_bytes(root / "worker-outcome.json", 16384)
    import json

    kind = (
        StartupWorkerOutcome
        if json.loads(raw).get("schema_version") == ("nurse.v2.ops-worker-outcome/2")
        else WorkerOutcome
    )
    state = kind.model_validate_json(raw)
    if (
        state.identity != identity
        or raw != canonical_json_bytes(state.model_dump(mode="json"))
        or (
            state.state_sha256
            != sha256_hex(
                canonical_json_bytes(state.model_dump(mode="json", exclude={"state_sha256"}))
            )
        )
    ):
        raise ValueError("worker outcome identity differs")
    return state
