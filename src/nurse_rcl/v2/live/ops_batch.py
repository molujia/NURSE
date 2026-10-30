"""Fixed ordered references to complete single-case plans; never discovers cases."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.live.normal_execution import _metadata, _pinned
from nurse_rcl.v2.live.ops_case import plan_identity
from nurse_rcl.v2.live.ops_execution import run_round
from nurse_rcl.v2.live.ops_execution import status as single_status
from nurse_rcl.v2.live.ops_plan import (
    OpsSinglePlan,
    plain_location,
    protected_locations,
    read_ops_plan,
    require_ops_code,
)
from nurse_rcl.v2.live.repair_driver import publish_json
from nurse_rcl.v2.live.repair_execution import install_termination_handlers
from nurse_rcl.v2.live.staged_case import require_plain_path


class OpsPlanReference(StrictFrozenModel):
    plan_path: str
    plan_sha256: SHA256


class OpsBatchPlan(StrictFrozenModel):
    schema_version: Literal["nurse.v2.ops-batch-plan/1"] = "nurse.v2.ops-batch-plan/1"
    cases: tuple[OpsPlanReference, ...] = Field(min_length=1, max_length=1740)
    output_root: str


def _load(path: Path, pin: str) -> tuple[OpsBatchPlan, tuple[OpsSinglePlan, ...]]:
    raw = _pinned(path, pin)
    _metadata(raw)
    batch = OpsBatchPlan.model_validate_json(raw)
    plans = tuple(read_ops_plan(Path(ref.plan_path), ref.plan_sha256) for ref in batch.cases)
    if len({p.case.opaque_case_id for p in plans}) != len(plans):
        raise ValueError("isolated batch contains duplicate cases")
    if (
        len(
            {
                (
                    p.code_root,
                    p.code_sha256,
                    p.environment_sha256,
                    p.mode,
                    p.profile.model_dump_json(),
                )
                for p in plans
            }
        )
        != 1
    ):
        raise ValueError("isolated batch code/environment/profile differs")
    locations = [plain_location(Path(p.output_root)) for p in plans]
    batch_out = require_plain_path(Path(batch.output_root))
    for plan in plans:
        for protected in protected_locations(plan):
            if batch_out.is_relative_to(protected) or protected.is_relative_to(batch_out):
                raise ValueError("isolated batch output overlaps protected input")
    # Batch metadata is separate from all case outputs. No nested aliases.
    locations.append(batch_out)
    for index, location in enumerate(locations):
        for other in locations[index + 1 :]:
            if location.is_relative_to(other) or other.is_relative_to(location):
                raise ValueError("isolated batch outputs overlap")
    return batch, plans


def _summary(pin: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = sum(r.get("channels", {}).get("core") == "completed" for r in results)
    return dict(
        batch_sha256=pin,
        total_cases=len(results),
        completed=completed,
        failed=sum(r.get("status") in {"failed", "interrupted"} for r in results),
        results=results,
    )


def status(path: Path, pin: str) -> dict[str, Any]:
    batch, _ = _load(path, pin)
    return _summary(pin, [single_status(Path(r.plan_path), r.plan_sha256) for r in batch.cases])


def run_batch(
    path: Path,
    pin: str,
    *,
    progress: Callable[[dict[str, Any]], None],
    approved_batch_sha256: str | None = None,
) -> dict[str, Any]:
    batch, plans = _load(path, pin)
    require_ops_code(plans[0])
    if plans[0].mode == "with-agent" and approved_batch_sha256 != pin:
        raise ValueError("isolated batch requires explicit batch approval")
    out = Path(batch.output_root)
    lock = os.open(out / "batch.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish_json(
            out / "identity.json", dict(batch_sha256=pin, plan=batch.model_dump(mode="json"))
        )
        results: list[dict[str, Any]] = []
        for index, reference in enumerate(batch.cases, 1):
            progress(
                dict(
                    event="batch_case_started",
                    case_index=index,
                    total_cases=len(plans),
                    completed=len(results),
                )
            )

            def emit(row: dict[str, Any], current: int = index) -> None:
                progress(dict(row, case_index=current, total_cases=len(plans)))

            result = run_round(
                Path(reference.plan_path),
                reference.plan_sha256,
                progress=emit,
                approved_plan_sha256=reference.plan_sha256
                if approved_batch_sha256 == pin
                else None,
            )
            results.append(result)
            progress(
                dict(
                    event="batch_case_finished",
                    case_index=index,
                    total_cases=len(plans),
                    channels=result.get("channels"),
                    status=result["status"],
                )
            )
            atomic_write_bytes(
                out / "progress.json",
                canonical_json_bytes(
                    dict(batch_sha256=pin, processed=index, total_cases=len(plans), results=results)
                ),
            )
            if result["status"] == "interrupted":
                results.extend(
                    dict(
                        status="pending",
                        identity=plan_identity(p, ref.plan_sha256),
                        channels=dict(core="pending", investigation="pending", report="pending"),
                    )
                    for p, ref in zip(plans[index:], batch.cases[index:], strict=True)
                )
                break
        result = _summary(pin, results)
        atomic_write_bytes(out / "result.json", canonical_json_bytes(result))
        return result
    finally:
        os.close(lock)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--approved-batch-sha256")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    install_termination_handlers()

    def emit(value: dict[str, Any]) -> None:
        print(json.dumps(value, ensure_ascii=False), flush=True)

    try:
        result = (
            status(args.plan, args.plan_sha256)
            if args.status
            else run_batch(
                args.plan,
                args.plan_sha256,
                progress=emit,
                approved_batch_sha256=args.approved_batch_sha256,
            )
        )
        emit(result)
        return (
            130
            if any(r.get("status") == "interrupted" for r in result["results"])
            else int(result["failed"] > 0)
        )
    except KeyboardInterrupt:
        emit(dict(event="ops_batch_interrupted"))
        return 130
    except Exception as exc:
        emit(dict(event="ops_batch_failed", failure_type=type(exc).__name__))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
