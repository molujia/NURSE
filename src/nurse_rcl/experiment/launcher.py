"""Fail-closed workload-placement preflight for local and server execution."""

from __future__ import annotations

import argparse
import platform
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field, ValidationError, model_validator

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import DatasetName, StrictFrozenModel

PLACEMENT_SCHEMA_VERSION: Literal[1] = 1
LOCAL_DURATION_LIMIT_SECONDS = 20 * 60
RunScope = Literal["synthetic", "sample", "full"]
ExecutionSite = Literal["windows_workstation", "linux_server"]


class RunLaunchRequest(StrictFrozenModel):
    """Immutable placement-relevant controls supplied before any run artifacts."""

    schema_version: Literal[1] = PLACEMENT_SCHEMA_VERSION
    dataset: DatasetName
    scope: RunScope
    estimated_duration_seconds: int = Field(ge=0)
    supervised: bool
    sample_approved: bool
    request_hash: str = Field(default="0" * 64, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def populate_and_verify_request_hash(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        unsigned = {
            "schema_version": data.get(
                "schema_version",
                PLACEMENT_SCHEMA_VERSION,
            ),
            "dataset": data.get("dataset"),
            "scope": data.get("scope"),
            "estimated_duration_seconds": data.get("estimated_duration_seconds"),
            "supervised": data.get("supervised"),
            "sample_approved": data.get("sample_approved"),
        }
        expected = sha256_hex(canonical_json_bytes(unsigned))
        supplied = data.get("request_hash")
        if supplied is not None and supplied != expected:
            raise ValueError("launch request hash verification failed")
        data["request_hash"] = expected
        return data

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"request_hash"})
        return self.request_hash == sha256_hex(canonical_json_bytes(unsigned))


class PlacementDecision(StrictFrozenModel):
    """Content-addressed receipt proving the launch passed placement policy."""

    schema_version: Literal[1] = PLACEMENT_SCHEMA_VERSION
    allowed: Literal[True] = True
    execution_site: ExecutionSite
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: str = Field(min_length=1)
    decision_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"decision_hash"})
        return self.decision_hash == sha256_hex(canonical_json_bytes(unsigned))


class WorkloadPlacementError(RuntimeError):
    """A stable fail-closed launcher denial before any experiment starts."""

    def __init__(self, *, code: str, request_hash: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.request_hash = request_hash


def validate_workload_placement(
    *,
    request: RunLaunchRequest,
    platform_name: str | None = None,
) -> PlacementDecision:
    """Reject unsafe workstation runs and seal permitted execution placement."""

    if not request.verify():
        raise ValueError("launch request verification failed")
    detected = platform.system() if platform_name is None else platform_name
    normalized = detected.strip().casefold()
    if normalized == "windows":
        if request.scope == "full":
            raise _denial(
                request,
                code="full_dataset_requires_server",
                message="full datasets must run on the designated Linux server in tmux",
            )
        if request.estimated_duration_seconds > LOCAL_DURATION_LIMIT_SECONDS:
            raise _denial(
                request,
                code="local_duration_limit_exceeded",
                message="runs estimated over 20 minutes must use the designated Linux server",
            )
        if not request.supervised:
            raise _denial(
                request,
                code="local_execution_requires_supervision",
                message="Windows workstation runs must remain directly supervised",
            )
        if request.scope == "sample" and not request.sample_approved:
            raise _denial(
                request,
                code="local_sample_requires_approval",
                message="Windows sample runs require an explicitly approved small sample",
            )
        return _decision(
            request=request,
            execution_site="windows_workstation",
            reason="supervised local workload is within the 20-minute limit",
        )
    if normalized == "linux":
        return _decision(
            request=request,
            execution_site="linux_server",
            reason="workload is placed on a Linux server",
        )
    raise _denial(
        request,
        code="unsupported_execution_platform",
        message="experiments are supported only on the Windows workstation or Linux server",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate NURSE workload placement")
    parser.add_argument("--dataset", required=True, choices=("rcabench", "aiops25"))
    parser.add_argument(
        "--scope",
        required=True,
        choices=("synthetic", "sample", "full"),
    )
    parser.add_argument("--estimated-seconds", required=True, type=int)
    parser.add_argument("--supervised", action="store_true")
    parser.add_argument("--sample-approved", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        request = RunLaunchRequest(
            dataset=arguments.dataset,
            scope=arguments.scope,
            estimated_duration_seconds=arguments.estimated_seconds,
            supervised=arguments.supervised,
            sample_approved=arguments.sample_approved,
        )
        decision = validate_workload_placement(request=request)
    except WorkloadPlacementError as error:
        output = {
            "allowed": False,
            "code": error.code,
            "message": str(error),
            "request_hash": error.request_hash,
        }
        print(canonical_json_bytes(output).decode("utf-8"))
        return 2
    except ValidationError as error:
        output = {
            "allowed": False,
            "code": "invalid_launch_request",
            "message": str(error),
        }
        print(canonical_json_bytes(output).decode("utf-8"))
        return 2
    print(canonical_json_bytes(decision.model_dump(mode="json")).decode("utf-8"))
    return 0


def _decision(
    *,
    request: RunLaunchRequest,
    execution_site: ExecutionSite,
    reason: str,
) -> PlacementDecision:
    unsigned = {
        "schema_version": PLACEMENT_SCHEMA_VERSION,
        "allowed": True,
        "execution_site": execution_site,
        "request_hash": request.request_hash,
        "reason": reason,
    }
    decision = PlacementDecision(
        execution_site=execution_site,
        request_hash=request.request_hash,
        reason=reason,
        decision_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not decision.verify():  # pragma: no cover - construction invariant
        raise AssertionError("placement decision verification failed")
    return decision


def _denial(
    request: RunLaunchRequest,
    *,
    code: str,
    message: str,
) -> WorkloadPlacementError:
    return WorkloadPlacementError(
        code=code,
        request_hash=request.request_hash,
        message=message,
    )


if __name__ == "__main__":  # pragma: no cover - exercised through console entry point
    raise SystemExit(main())
