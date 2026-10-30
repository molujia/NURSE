"""Dev branch children share the complete case's clock; native identities stay intact."""

from __future__ import annotations

import argparse
import json
import resource
import sys
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Any

from nurse_rcl.artifacts import sha256_hex
from nurse_rcl.v2.live.dev_batch import read_branch
from nurse_rcl.v2.live.dev_budget import DevBudget
from nurse_rcl.v2.live.normal_execution import _metadata, _pinned
from nurse_rcl.v2.live.repair_driver import publish_json, read_json
from nurse_rcl.v2.live.repair_execution import install_termination_handlers, supervise_phases
from nurse_rcl.v2.live.repair_module_timing import _module_span, _module_timing_scope
from nurse_rcl.v2.live.repair_timing import PhaseClient, _phase, _use_phase_client
from nurse_rcl.v2.live.staged_case import require_plain_path


class DevQueuePaused(ValueError):
    """Retain the case; an unclassified/system failure is not a skippable sample."""


class DevPreparationTimeout(TimeoutError):
    """The parent's shared preparation account, not a method deadline, expired."""


def execute_branch(
    path: Path,
    pin: str,
    *,
    branch: str,
    case_root: Path,
    timing: DevBudget,
    progress: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    if branch not in {"B0", "B2"}:
        raise ValueError("invalid Dev branch")
    plan = _metadata(_pinned(path, pin))
    state = Path(plan["output_root"]) / plan["case"]["opaque_case_id"]
    state.mkdir(mode=0o700, exist_ok=True)
    require_plain_path(state)
    record_path = case_root / (branch.lower() + "-execution.json")
    identity = dict(case_timing_sha256=timing.identity, plan_sha256=pin, branch=branch)
    if record_path.exists():
        retained = read_json(record_path)
        if retained.get("identity") != identity:
            raise ValueError("Dev branch resume identity differs")
        view = read_branch(retained["reference"], case=plan["case"], branch=branch)
        if view["cost"].get("provider_fatal_errors"):
            raise DevQueuePaused("fatal provider/configuration error in saved result")
        return retained
    if not timing.clock.active:
        raise DevQueuePaused("Dev branch needs an active shared case clock")
    before = timing.clock.used_seconds
    with (
        _module_timing_scope(
            case_root / "module-spans", identity=timing.identity, boot_id=timing.boot_id
        ) as recorder,
        _module_span("worker_supervision", role="orchestration"),
    ):

        def event(row: dict[str, Any]) -> None:
            recorder.observe()
            progress(dict(row, branch=branch))

        code = supervise_phases(
            [
                sys.executable,
                "-m",
                "nurse_rcl.v2.live.dev_execution",
                "--worker",
                "--plan",
                str(path),
                "--plan-sha256",
                pin,
                "--branch",
                branch,
                "--case-root",
                str(case_root),
                "--case-identity",
                timing.identity,
                "--module-parent",
                str(recorder.current_span_id),
            ],
            log_path=state / "case.log",
            timing=timing,
            progress=event,
        )
    failure_reason = None
    if code == 124 and timing.clock.used_seconds["preparation"] >= 3600:
        raise DevPreparationTimeout("preparation_timeout")
    if code != 0:
        failure = read_json(state / "failure.json") if (state / "failure.json").exists() else {}
        if failure.get("error_type") == "MemoryError":
            failure_reason = "memory_capacity"
        elif failure.get("error_type") == "ExchangeBoundaryError" and (
            failure.get("capacity_reason") == "json_limit"
        ):
            failure_reason = "json_capacity"
        else:
            raise DevQueuePaused(
                "Dev branch worker failed; preserve native records before diagnosis"
            )
    with (
        _module_timing_scope(
            case_root / "module-spans", identity=timing.identity, boot_id=timing.boot_id
        ),
        _module_span(
            "parent_finalize",
            role="verification",
        ),
    ):
        result_path = state / ("failure.json" if failure_reason else "result.json")
        reference = dict(
            plan_path=str(path),
            plan_sha256=pin,
            result_path=str(result_path),
            result_sha256=sha256_hex(result_path.read_bytes()),
        )
        view = read_branch(reference, case=plan["case"], branch=branch)
        if view["cost"].get("provider_fatal_errors"):
            raise DevQueuePaused("fatal provider/configuration error; do not skip case")
    timing.checkpoint(now=monotonic())
    after = timing.clock.used_seconds
    result = dict(
        schema_version="nurse.v2.dev-branch-execution/1",
        identity=identity,
        status=view["status"],
        reference=reference,
        worker_exit_code=code,
        account_delta={k: after[k] - before[k] for k in before},
        timing_checkpoint_sha256=timing.previous,
        failure_reason=failure_reason,
    )
    publish_json(record_path, result)
    progress(dict(event="dev_branch_completed", **result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--branch", choices=("B0", "B2"), required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--case-identity", required=True)
    parser.add_argument("--module-parent", required=True)
    parser.add_argument("--timing-fd", type=int, required=True)
    args = parser.parse_args()
    install_termination_handlers()
    resource.setrlimit(resource.RLIMIT_AS, (8589934592, 8589934592))
    client = PhaseClient(args.timing_fd)
    try:
        with (
            _use_phase_client(client),
            _module_timing_scope(
                args.case_root / "module-spans",
                identity=args.case_identity,
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                parent_id=args.module_parent,
            ),
            _module_span("case_worker"),
        ):
            if args.branch == "B0":
                from nurse_rcl.v2.live.normal_b0_case import run_b0_case
                from nurse_rcl.v2.live.normal_execution import load_execution

                run = run_b0_case
            else:
                from nurse_rcl.v2.live.normal_b2_execution import load_execution, run_b2_case

                run = run_b2_case
            with _phase("preparation", "input_validation"), _module_span("input_guard"):
                plan = load_execution(args.plan, args.plan_sha256)
            result = run(plan=plan, plan_sha256=args.plan_sha256)
        client.finish()
        return int(
            result["status"]
            not in {
                "completed",
                "completed_partial_report",
                "completed_fallback",
            }
        )
    except Exception as error:
        print(
            json.dumps(dict(event="dev_worker_failed", error_type=type(error).__name__)), flush=True
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
