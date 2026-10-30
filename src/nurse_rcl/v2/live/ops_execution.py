"""Supervised isolated single-case entry; no probes, batch discovery or source replay."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import resource
import sys
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Any

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_case import (
    cost_case_view,
    inspect_saved_case,
    inspect_saved_core,
    plan_identity,
    run_ops_case,
)
from nurse_rcl.v2.live.ops_contracts import (
    OpsEmpiricalProfile,
    OpsSreBriefProfile,
    OpsSreComparableProfile,
    OpsSreConditionProfile,
    OpsSreFeedbackFactProfile,
    OpsSreNavigationProfile,
    OpsSreProfileBase,
    OpsSreRecoveryProfile,
    OpsSreWorkbenchProfile,
    OpsSreWorkingProfile,
)
from nurse_rcl.v2.live.ops_failure import (
    FailureStage,
    InvestigationStartupFailure,
    OpsFailureDetail,
    failure_detail,
    secondary_failures,
)
from nurse_rcl.v2.live.ops_plan import OpsSinglePlan, load_ops_plan, read_ops_plan, require_ops_code
from nurse_rcl.v2.live.ops_provider import usage_summary
from nurse_rcl.v2.live.ops_usage import summarize_attempts
from nurse_rcl.v2.live.ops_worker_outcome import read_outcome, write_outcome
from nurse_rcl.v2.live.repair_driver import publish_json, read_json
from nurse_rcl.v2.live.repair_execution import install_termination_handlers, supervise_phases
from nurse_rcl.v2.live.repair_module_timing import (
    _module_span,
    _module_timing_scope,
    summarize_module_timing,
)
from nurse_rcl.v2.live.repair_timing import (
    DEV_TIMING_POLICY,
    CaseBudgetExpired,
    PhaseClient,
    TimingJournal,
    _phase,
    _use_phase_client,
)
from nurse_rcl.v2.live.staged_case import require_plain_path


def status(path: Path, pin: str) -> dict[str, Any]:
    """Read saved status, even if current source code has changed; do not resume."""
    plan = read_ops_plan(path, pin)
    out = Path(plan.output_root)
    root = out / plan.case.opaque_case_id
    if not (out / "terminal.json").exists():
        return dict(status="not_terminal", opaque_case_id=plan.case.opaque_case_id)
    result = read_json(out / "terminal.json")
    if result.get("identity") != plan_identity(plan, pin):
        raise ValueError("isolated terminal identity differs")
    if result.get("channels", {}).get("core") == "completed":
        if result["status"] in {"completed", "completed_partial_report"}:
            inspect_saved_case(root, plan_identity(plan, pin))
        else:
            inspect_saved_core(root, plan)
    return result


def _clock(root: Path, pin: str) -> TimingJournal:
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    directory = root / "timing"
    now = monotonic()
    if directory.exists() and any(directory.iterdir()):
        journal = TimingJournal._read_existing(
            directory, identity=pin, boot_id=boot, expected_policy=DEV_TIMING_POLICY
        )
        if journal.clock.active:
            # We cannot know which part of an unclean gap was active work.
            # Retain and conservatively charge it; never silently reset time.
            gap = dict(
                last_checkpoint=journal.previous,
                unobserved_seconds=now - journal.clock.last,
                timing_interpretation="unknown_gap_conservatively_charged",
            )
            publish_json(root / f"timing-gap-{journal.sequence:08d}.json", gap)
            journal.checkpoint(now=now)
        else:
            journal._record("resume", "method", "orchestration", now)
        return journal
    return TimingJournal.open(
        directory, identity=pin, now=now, boot_id=boot, expected_policy=DEV_TIMING_POLICY
    )


def _modules(root: Path, pin: str) -> dict[str, Any]:
    tables = []
    for path in sorted((root / "module-spans").glob("*.jsonl")):
        table = summarize_module_timing(
            event_paths=(path,),
            timing_root=root / "timing",
            expected_timing_identity=pin,
            expected_policy=DEV_TIMING_POLICY,
        )
        tables.append(dict(process=path.stem, **table))
    return dict(
        aggregation="per_process_not_wall_sum",
        complete=bool(tables) and all(t["complete"] for t in tables),
        processes=tables,
    )


def _publish_terminal(
    out: Path,
    terminal: dict[str, Any],
    progress: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    terminal_pin = sha256_hex(canonical_json_bytes(terminal))
    publish_json(out / f"terminal-{terminal_pin}.json", terminal)
    atomic_write_bytes(out / "terminal.json", canonical_json_bytes(terminal))
    progress(dict(event="ops_terminal", **terminal))
    return terminal


def _invocation_usage(
    plan: OpsSinglePlan,
    pin: str,
    root: Path,
    baseline: dict[str, Any] | None,
    *,
    worker_terminated: bool = False,
) -> dict[str, Any]:
    current = usage_summary(
        root / "provider",
        execution_sha256=pin,
        empirical=isinstance(plan.profile, OpsEmpiricalProfile | OpsSreProfileBase),
        sre=isinstance(plan.profile, OpsSreProfileBase),
        working=isinstance(plan.profile, OpsSreWorkingProfile),
        feedback_fact=isinstance(plan.profile, OpsSreFeedbackFactProfile),
        sre_profile=(
            plan.profile.investigation_profile_id
            if isinstance(plan.profile, OpsSreProfileBase)
            else None
        ),
        confirmed_no_send=plan.mode == "core-only",
    )
    if worker_terminated and isinstance(
        plan.profile, OpsSreComparableProfile | OpsSreNavigationProfile | OpsSreWorkbenchProfile
    ):
        from nurse_rcl.v2.live.ops_startup_failure import no_send_eligible, read_startup_failure
        from nurse_rcl.v2.live.ops_worker_outcome import StartupWorkerOutcome

        try:
            outcome = read_outcome(root, plan_identity(plan, pin))
            if isinstance(outcome, StartupWorkerOutcome):
                core = inspect_saved_core(root, plan)
                receipt = read_startup_failure(
                    root,
                    identity=core.identity,
                    code=plan.code_sha256,
                    execution=pin,
                    profile=plan.profile.investigation_profile_id,
                )
                if receipt.receipt_sha256 != outcome.terminated_startup_receipt_sha256 or (
                    receipt.primary_failure != outcome.primary_failure
                ):
                    raise ValueError("startup supervisor proof differs")
                # Existing provider material is never discarded to obtain zero cost.
                provider_root = root / "provider"
                provider_clear = not provider_root.exists() and not provider_root.is_symlink()
                if provider_clear and no_send_eligible(
                    receipt,
                    worker_terminated=True,
                    report_started=(root / "report").exists() or (root / "report").is_symlink(),
                    usage=current,
                ):
                    current.update(summarize_attempts([], confirmed_no_send=True))
        except (OSError, ValueError):
            # This optional proof can only strengthen unknown usage to known zero.
            # An invalid proof leaves the provider inventory/uncertainty intact;
            # it must not prevent saving the original worker failure terminal.
            pass
    if baseline is None:
        return current
    before = {(r["phase"], r["slot"], r["attempt"]) for r in baseline["attempt_usage"]}
    rows = current["attempt_usage"]
    after = {(r["phase"], r["slot"], r["attempt"]) for r in rows}
    if not before <= after:
        raise ValueError("invocation lost prior attempt identity")
    new = [r for r in rows if (r["phase"], r["slot"], r["attempt"]) not in before]
    added = summarize_attempts(new, confirmed_no_send=not new)
    return dict(
        current,
        incremental_attempt_ids=[list(key) for key in sorted(after - before)],
        incremental_input_tokens=added["input_tokens"],
        incremental_reported_input_tokens=added["reported_input_tokens"],
        incremental_unknown_input_attempts=added["unknown_input_attempts"],
    )


def run_round(
    path: Path,
    pin: str,
    *,
    progress: Callable[[dict[str, Any]], None],
    approved_plan_sha256: str | None = None,
) -> dict[str, Any]:
    plan = read_ops_plan(path, pin)
    if getattr(plan.profile, "investigation_profile_id", None) in {
        "hermes-sre-tools/19",
        "hermes-sre-tools/20",
        "hermes-sre-tools/21",
        "hermes-sre-tools/22",
        "hermes-sre-tools/23",
    }:
        raise ValueError("entry profile requires the separately approved bounded stage")
    require_ops_code(plan)
    out = require_plain_path(Path(plan.output_root))
    root = out / plan.case.opaque_case_id
    root.mkdir(mode=0o700, exist_ok=True)
    require_plain_path(root)
    identity = plan_identity(plan, pin)
    lock = os.open(out / "round.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish_json(out / "identity.json", identity)
        if (out / "terminal.json").exists():
            retained = status(path, pin)
            if (
                retained["status"] in {"completed", "completed_partial_report"}
                or (retained.get("failure_type") == "preparation_timeout")
                or (
                    isinstance(
                        plan.profile,
                        OpsSreWorkingProfile
                        | OpsSreBriefProfile
                        | OpsSreConditionProfile
                        | OpsSreRecoveryProfile
                        | OpsSreComparableProfile
                        | OpsSreNavigationProfile
                        | OpsSreWorkbenchProfile,
                    )
                    and retained.get("terminal_failure") is True
                )
            ):
                if isinstance(plan.profile, OpsEmpiricalProfile | OpsSreProfileBase):
                    # A status read returns the historical artifact. A run call
                    # reusing it exposes this invocation's zero new spend,
                    # without rewriting that artifact or erasing lifetime cost.
                    usage = dict(
                        retained["usage"],
                        incremental_attempt_ids=[],
                        incremental_input_tokens=0,
                        incremental_reported_input_tokens=0,
                        incremental_unknown_input_attempts=0,
                    )
                    view = dict(retained, usage=usage, invocation_kind="completed_result_reuse")
                    view["cost_case"] = cost_case_view(plan, pin, view)
                    return view
                return retained
        if (
            plan.mode == "with-agent"
            and approved_plan_sha256 != pin
            and not (root / "report/context.json").exists()
        ):
            raise ValueError("isolated execution requires explicit plan approval")
        terminal: dict[str, Any] = dict(
            identity=identity,
            status="failed",
            worker_exit_code=None,
            channels=dict(core="pending", investigation="pending", report="pending"),
        )
        baseline = (
            usage_summary(
                root / "provider",
                execution_sha256=pin,
                empirical=True,
                sre=isinstance(plan.profile, OpsSreProfileBase),
                working=isinstance(plan.profile, OpsSreWorkingProfile),
                feedback_fact=isinstance(plan.profile, OpsSreFeedbackFactProfile),
                sre_profile=(
                    plan.profile.investigation_profile_id
                    if isinstance(plan.profile, OpsSreProfileBase)
                    else None
                ),
            )
            if isinstance(plan.profile, OpsEmpiricalProfile | OpsSreProfileBase)
            else None
        )
        try:
            timing = _clock(root, pin)
        except CaseBudgetExpired:
            retained_clock = TimingJournal._inspect_expired(
                root / "timing",
                identity=pin,
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                expected_policy=DEV_TIMING_POLICY,
            )
            terminal.update(
                failure_type="preparation_timeout",
                worker_exit_code=124,
                timing=retained_clock["timing"],
            )
            terminal["channels"]["core"] = "failed"
            if (root / "core/result.json").exists():
                try:
                    saved = inspect_saved_core(root, plan)
                    terminal["channels"]["core"] = "completed"
                    terminal["core_result_digest"] = saved.core_result_digest
                    terminal["case_snapshot_id"] = saved.identity.case_snapshot_id
                    terminal["core_result_saved"] = True
                except ValueError:
                    terminal["core_integrity_failure"] = True
            terminal["usage"] = _invocation_usage(plan, pin, root, baseline)
            if baseline is not None:
                terminal["cost_case"] = cost_case_view(plan, pin, terminal)
            return _publish_terminal(out, terminal, progress)
        with _module_timing_scope(
            root / "module-spans", identity=pin, boot_id=timing.boot_id
        ) as recorder:
            worker_terminated = False
            try:

                def emit(row: dict[str, Any]) -> None:
                    recorder.observe()
                    progress(
                        dict(
                            row,
                            case_index=1,
                            total_cases=1,
                            opaque_case_id=plan.case.opaque_case_id,
                        )
                    )

                command = [
                    sys.executable,
                    "-m",
                    "nurse_rcl.v2.live.ops_execution",
                    "--plan",
                    str(path),
                    "--plan-sha256",
                    pin,
                    "--worker",
                ]
                if approved_plan_sha256 is not None:
                    command.extend(("--approved-plan-sha256", approved_plan_sha256))
                # Invalidate a previous invocation's diagnostic before spawning.
                # Existing core, conversation, report and attempt owners are untouched.
                write_outcome(root, identity)
                with _module_span("worker_supervision", role="orchestration"):
                    code = supervise_phases(
                        command, log_path=root / "case.log", timing=timing, progress=emit
                    )
                    worker_terminated = True
                terminal["worker_exit_code"] = code
                with _module_span("parent_finalize", role="verification"):
                    if code != 0:
                        raise ValueError("isolated worker incomplete")
                    terminal.update(inspect_saved_case(root, identity))
            except BaseException as exc:
                terminal.update(
                    status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    failure_type=type(exc).__name__,
                )
                if (root / "worker-outcome.json").exists():
                    outcome = read_outcome(root, identity)
                    if outcome.primary_failure is not None:
                        terminal.update(
                            failure_type=outcome.primary_failure.failure_type,
                            primary_failure=outcome.primary_failure.model_dump(mode="json"),
                            secondary_failures=[
                                r.model_dump(mode="json") for r in outcome.secondary_failures
                            ],
                        )
                terminal["channels"].update(
                    investigation="disabled" if plan.mode == "core-only" else "unavailable",
                    report="disabled" if plan.mode == "core-only" else "unavailable",
                )
                if (root / "core/result.json").exists():
                    terminal["core_result_saved"] = True
                    try:
                        saved = inspect_saved_core(root, plan)
                        terminal["channels"]["core"] = "completed"
                        terminal["core_result_digest"] = saved.core_result_digest
                        terminal["case_snapshot_id"] = saved.identity.case_snapshot_id
                    except ValueError:
                        terminal["channels"]["core"] = "failed"
                        terminal["core_integrity_failure"] = True
                else:
                    terminal["channels"]["core"] = (
                        "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                    )
                if isinstance(
                    plan.profile,
                    OpsSreWorkingProfile
                    | OpsSreBriefProfile
                    | OpsSreConditionProfile
                    | OpsSreRecoveryProfile
                    | OpsSreComparableProfile
                    | OpsSreNavigationProfile
                    | OpsSreWorkbenchProfile,
                ) and (plan.mode == "with-agent" and terminal["channels"]["core"] == "completed"):
                    from nurse_rcl.v2.live.ops_sre_working_case import inspect_working_progress

                    try:
                        retained = inspect_working_progress(root, plan, pin, saved)
                        outer = [
                            terminal.get("primary_failure"),
                            *terminal.get("secondary_failures", []),
                        ]
                        terminal.update(retained)
                        if terminal["primary_failure"] is not None:
                            terminal["failure_type"] = terminal["primary_failure"]["failure_type"]
                        for detail in outer:
                            if (
                                detail is not None
                                and detail != terminal["primary_failure"]
                                and (detail not in terminal["secondary_failures"])
                            ):
                                if terminal["primary_failure"] is None:
                                    terminal["primary_failure"] = detail
                                else:
                                    terminal["secondary_failures"].append(detail)
                    except (ValueError, OSError) as recovery:
                        terminal["saved_progress_failure_type"] = type(recovery).__name__
        try:
            # Persist exhaustion as a checkpoint before attempting a switch.
            # An expired journal must remain inspectable after a publication crash.
            now = monotonic()
            timing.checkpoint(now=now)
            if timing.clock.account != "method":
                timing.switch("method", reason="orchestration", now=now)
            timing._record("pause", "method", "orchestration", monotonic())
        except Exception as exc:
            detail = failure_detail("timing_checkpoint", exc).model_dump(mode="json")
            if "primary_failure" in terminal:
                terminal.setdefault("secondary_failures", []).append(detail)
            elif isinstance(exc, CaseBudgetExpired):
                terminal.update(status="failed", failure_type="preparation_timeout")
            elif "failure_type" in terminal:
                terminal.setdefault("secondary_failures", []).append(detail)
            else:
                terminal.update(
                    status="failed",
                    failure_type=type(exc).__name__,
                    primary_failure=detail,
                    secondary_failures=[],
                )
        terminal["timing"] = timing.clock.summary()
        terminal["usage"] = _invocation_usage(
            plan, pin, root, baseline, worker_terminated=worker_terminated
        )
        if baseline is not None:
            terminal["cost_case"] = cost_case_view(plan, pin, terminal)
        try:
            table = _modules(root, pin)
            atomic_write_bytes(root / "module-timing.json", canonical_json_bytes(table))
        except (ValueError, CaseBudgetExpired) as exc:
            terminal["module_timing_failure"] = type(exc).__name__
        return _publish_terminal(out, terminal, progress)
    finally:
        os.close(lock)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--timing-fd", type=int)
    args = parser.parse_args()
    install_termination_handlers()
    resource.setrlimit(resource.RLIMIT_AS, (8589934592, 8589934592))

    def emit(row: dict[str, Any]) -> None:
        print(json.dumps(row, ensure_ascii=False), flush=True)

    try:
        if args.worker != (args.timing_fd is not None) or (args.status and args.worker):
            raise ValueError("isolated worker requires private timing channel")
        if args.status:
            emit(status(args.plan, args.plan_sha256))
            return 0
        if not args.worker:
            result = run_round(
                args.plan,
                args.plan_sha256,
                progress=emit,
                approved_plan_sha256=args.approved_plan_sha256,
            )
        else:
            envelope = read_ops_plan(args.plan, args.plan_sha256)
            root = Path(envelope.output_root) / envelope.case.opaque_case_id
            client = PhaseClient(args.timing_fd)
            primary: BaseException | None = None
            primary_detail: OpsFailureDetail | None = None
            secondary: list[OpsFailureDetail] = []

            def retain_failure() -> None:
                try:
                    write_outcome(
                        root,
                        plan_identity(envelope, args.plan_sha256),
                        primary_detail,
                        tuple(secondary),
                        startup_receipt_sha256=(
                            primary.receipt_sha256
                            if isinstance(primary, InvestigationStartupFailure)
                            else None
                        ),
                    )
                except OSError:
                    # A failed disk cannot guarantee persistence. Do not replace
                    # the actual cause with a diagnostic publication failure.
                    emit(dict(event="ops_failure_receipt_unavailable", failure_type="OSError"))

            try:
                with (
                    _use_phase_client(client),
                    _module_timing_scope(
                        root / "module-spans",
                        identity=args.plan_sha256,
                        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    ),
                    _module_span("case_worker"),
                ):
                    with _phase("preparation", "input_validation"), _module_span("input_guard"):
                        admitted = load_ops_plan(args.plan, args.plan_sha256)
                    result = run_ops_case(admitted, approved_plan_sha256=args.approved_plan_sha256)
            except BaseException as exc:
                primary = exc
                primary_detail = failure_detail("case_worker", exc)
                secondary.extend(secondary_failures(exc))
                retain_failure()  # Persist before channel cleanup can fail or be interrupted.
            finally:
                cleanups: tuple[tuple[FailureStage, Callable[[], None]], ...] = (
                    ("timing_finish", client.finish),
                    ("timing_close", client.channel.close),
                )
                for stage, cleanup in cleanups:
                    try:
                        cleanup()
                    except BaseException as exc:
                        detail = failure_detail(stage, exc)
                        if primary is None:
                            primary, primary_detail = exc, detail
                        else:
                            secondary.append(detail)
                        retain_failure()
            if primary is not None:
                raise primary
        return int(
            result["status"]
            not in {"completed", "completed_partial_report", "core_completed_report_unavailable"}
        )
    except KeyboardInterrupt:
        emit(dict(event="ops_interrupted"))
        return 130
    except Exception as exc:
        emit(dict(event="ops_failed", failure_type=type(exc).__name__))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
