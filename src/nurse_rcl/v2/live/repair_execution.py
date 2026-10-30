"""Finite subprocess supervision for the frozen four-case repair run."""

from __future__ import annotations

import math
import os
import select
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.repair_driver import (
    ROUND_SECONDS,
    STAGES,
    open_clock,
    publish_json,
    read_json,
    remaining_seconds,
    require_profile,
    require_queue,
)
from nurse_rcl.v2.live.repair_timing import (
    CaseBudgetExpired,
    PhaseClient,
    TimingJournal,
    _phase,
    _use_phase_client,
)
from nurse_rcl.v2.live.sources import file_sha256
from nurse_rcl.v2.live.staged_case import read_inference_identity, require_plain_path

SINGLE_SCHEMA = "nurse.v2.live-repair-single-plan/1"
FIRST_CASE_ID = "opq_d7573450f2a04691984547e38e51dfc6"


def active_case_indexes(plan: dict[str, Any]) -> tuple[int, ...]:
    if plan["schema_version"] == SINGLE_SCHEMA:
        indexes = plan.get("active_case_indexes")
        if (
            type(indexes) is not list
            or len(indexes) != 1
            or type(indexes[0]) is not int
            or indexes[0] != 0
            or plan["cases"][0]["dataset"] != "rcabench"
            or plan["cases"][0]["opaque_case_id"] != FIRST_CASE_ID
        ):
            raise ValueError("single execution must select only the original first case")
        return (0,)
    if plan["schema_version"] != "nurse.v2.live-repair-execution-plan/2":
        raise ValueError("old mixed-budget plans require their original frozen implementation")
    return (0, 1, 2, 3)


def install_termination_handlers() -> None:
    def interrupted(signum: int, frame: object) -> None:
        # BaseException bypasses request retries but unwinds supervise's finally.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)


def require_code(root: Path, manifest: dict[str, str]) -> None:
    base = require_plain_path(root)
    if type(manifest) is not dict or not manifest:
        raise ValueError("empty frozen code manifest")
    actual = set()
    for directory in (base / "src", base / "third_party/hermes_runtime"):
        require_plain_path(directory)
        for path in directory.rglob("*.py"):
            require_plain_path(path)
            actual.add(path.relative_to(base).as_posix())
    if actual != set(manifest):
        raise ValueError("frozen source inventory differs")
    for name, pin in manifest.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("invalid frozen source path")
        source = require_plain_path(base / path)
        if not source.is_relative_to(base) or file_sha256(source) != pin:
            raise ValueError("frozen code content differs")


def supervise(
    command: list[str],
    *,
    log_path: Path,
    remaining_seconds: float,
    progress: Callable[[dict[str, Any]], None],
) -> int:
    """Append stdout, heartbeat every 30s, TERM/KILL one owned process group.

    The caller supplies the minimum of the original persisted stage/case/round
    deadlines. This helper never refreshes a budget or retries a command.
    """
    if not math.isfinite(remaining_seconds):
        raise ValueError("nonfinite process deadline")
    if remaining_seconds <= 0:
        return 124
    require_plain_path(log_path.parent)
    if log_path.exists() or log_path.is_symlink():
        require_plain_path(log_path)
        if not log_path.is_file():
            raise ValueError("stage log must be a plain file")
    deadline = time.monotonic() + remaining_seconds
    fd = os.open(log_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "ab", buffering=0) as stream:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        progress(dict(event="stage_process_started", pid=child.pid))

        def stop() -> None:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    # Keep one second within the outer 30s TERM/KILL allowance
                    # so the parent can reap a child which ignores TERM.
                    child.wait(timeout=29)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()

        try:
            while child.poll() is None:
                left = deadline - time.monotonic()
                if left <= 0:
                    stop()
                    progress(dict(event="stage_process_exited", exit_code=124, reason="deadline"))
                    return 124
                try:
                    child.wait(timeout=min(30.0, left))
                except subprocess.TimeoutExpired:
                    progress(
                        dict(
                            event="heartbeat",
                            pid=child.pid,
                            remaining_seconds=max(0.0, deadline - time.monotonic()),
                            log_bytes=log_path.stat().st_size,
                        )
                    )
            code = int(child.returncode)
            progress(dict(event="stage_process_exited", exit_code=code))
            return code
        finally:
            stop()
            os.fsync(stream.fileno())


def supervise_phases(
    command: list[str],
    *,
    log_path: Path,
    timing: TimingJournal,
    progress: Callable[[dict[str, Any]], None],
) -> int:
    """Supervise one child while charging the existing case accounts.

    A private packet socket is passed as --timing-fd. Acknowledgement follows
    durable parent-clock accounting. Ordinary stdout is only a log.
    """
    from nurse_rcl.v2.exchange_json import ExchangeLimits, decode_json_object

    try:
        timing.checkpoint(now=time.monotonic())
    except CaseBudgetExpired as error:
        progress(dict(event="stage_process_exited", exit_code=124, reason=str(error)))
        return 124
    require_plain_path(log_path.parent)
    if log_path.exists() or log_path.is_symlink():
        require_plain_path(log_path)
    parent, endpoint = socket.socketpair(type=socket.SOCK_SEQPACKET)
    fd = os.open(log_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    child: subprocess.Popen[bytes] | None = None
    cleanup_seconds = 0.0

    def stop() -> None:
        nonlocal cleanup_seconds
        if child is not None and child.poll() is None:
            tick = time.monotonic()
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=29)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            cleanup_seconds += time.monotonic() - tick

    result, reason = 1, "timing_channel_incomplete"
    finished = False
    try:
        with os.fdopen(fd, "ab", buffering=0) as stream:
            child = subprocess.Popen(
                [*command, "--timing-fd", str(endpoint.fileno())],
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                pass_fds=(endpoint.fileno(),),
                start_new_session=True,
            )
            endpoint.close()
            progress(dict(event="stage_process_started", pid=child.pid, **timing.clock.summary()))
            sequence, finished, closed = 0, False, False
            heartbeat = time.monotonic()
            try:
                while True:
                    timing.clock.checkpoint(now=time.monotonic())
                    readable = [] if closed else [parent]
                    remaining = timing.clock.remaining_seconds
                    ready, _, _ = select.select(
                        readable, [], [], 0.2 if remaining is None else min(0.2, remaining)
                    )
                    if ready:
                        packet = parent.recv(1025)
                        if not packet:
                            closed = True
                            if not finished:
                                raise ValueError("timing channel closed before finish")
                        else:
                            if finished or len(packet) > 1024:
                                raise ValueError("unexpected timing packet")
                            row = decode_json_object(packet, ExchangeLimits(1024, 4, 32))
                            if (
                                set(row) != {"sequence", "action", "account", "reason"}
                                or type(row["sequence"]) is not int
                                or row["sequence"] != sequence
                                or row["action"] not in ("switch", "finish")
                            ):
                                raise ValueError("invalid timing transition sequence")
                            if row["action"] == "finish":
                                if (
                                    timing.clock.account != "method"
                                    or row["account"] != "method"
                                    or row["reason"] != "orchestration"
                                ):
                                    raise ValueError("incomplete timing input scope")
                                finished = True
                            timing.switch(
                                row["account"], reason=row["reason"], now=time.monotonic()
                            )
                            parent.sendall(canonical_json_bytes(dict(sequence=sequence)))
                            sequence += 1
                            progress(dict(event="phase_account", **timing.clock.summary()))
                    if time.monotonic() - heartbeat >= 30:
                        timing.checkpoint(now=time.monotonic())
                        progress(
                            dict(
                                event="heartbeat",
                                pid=child.pid,
                                log_bytes=log_path.stat().st_size,
                                **timing.clock.summary(),
                            )
                        )
                        heartbeat = time.monotonic()
                    if child.poll() is not None:
                        # Drain a queued finish packet before inspecting process exit.
                        if not closed and select.select([parent], [], [], 0)[0]:
                            continue
                        timing.checkpoint(now=time.monotonic())
                        result = int(child.returncode)
                        if result == 0 and not finished:
                            result = 1
                        reason = "completed" if result == 0 else "timing_or_worker_failure"
                        break
            except CaseBudgetExpired as error:
                # Persist the charged endpoint even if the child never returned.
                with suppress(CaseBudgetExpired):
                    timing.checkpoint(now=time.monotonic())
                result, reason = 124, str(error)
            except (ValueError, OSError):
                timing.checkpoint(now=time.monotonic())
                result, reason = 1, "timing_protocol_error"
            finally:
                stop()
                os.fsync(stream.fileno())
    finally:
        stop()
        parent.close()
        endpoint.close()
    progress(
        dict(
            event="stage_process_exited",
            exit_code=result,
            reason=reason,
            cleanup_seconds=cleanup_seconds,
            timing_finished=finished,
            **timing.clock.summary(),
        )
    )
    return result


def method_identity() -> dict[str, str]:
    from nurse_rcl.v2.live.repair_queries import REPAIR_QUERY_SEMANTICS_SHA256
    from nurse_rcl.v2.live.repair_raw_audit import REPAIR_RAW_AUDIT_SEMANTICS_SHA256
    from nurse_rcl.v2.live.repair_report import REPAIR_REPORT_SCHEMA_VERSION
    from nurse_rcl.v2.live.repair_report_projection import POLICY
    from nurse_rcl.v2.live.repair_report_prompt import (
        REPAIR_EXCHANGE_VERSION,
        REPAIR_INSTRUCTIONS,
        REPAIR_PROMPT_VERSION,
    )
    from nurse_rcl.v2.live.repair_report_request import RUN_POLICY

    return dict(
        raw_audit=REPAIR_RAW_AUDIT_SEMANTICS_SHA256,
        query=REPAIR_QUERY_SEMANTICS_SHA256,
        report_schema=REPAIR_REPORT_SCHEMA_VERSION,
        report_exchange=REPAIR_EXCHANGE_VERSION,
        prompt_version=REPAIR_PROMPT_VERSION,
        prompt_instructions_sha256=sha256_hex(REPAIR_INSTRUCTIONS.encode()),
        report_run_policy=RUN_POLICY.sha256,
        report_evidence_policy=POLICY.sha256,
    )


def load_execution(path: Path, expected_sha256: str) -> dict[str, Any]:
    from nurse_rcl.v2.live.repair_stage import RepairCaseInput

    source = require_plain_path(path)
    if file_sha256(source) != expected_sha256:
        raise ValueError("execution plan differs from independent launch pin")
    plan = read_json(source)
    fields = {
        "schema_version",
        "code_root",
        "code_files",
        "configuration_path",
        "configuration_sha256",
        "identity_manifest_path",
        "identity_artifact_sha256",
        "cases",
        "output_root",
        "python_executable",
        "method_identity",
    }
    if plan.get("schema_version") == SINGLE_SCHEMA:
        fields |= {"parent_plan_path", "parent_plan_sha256", "active_case_indexes"}
    if set(plan) != fields:
        raise ValueError("execution plan fields differ")
    if plan["method_identity"] != method_identity() or plan["python_executable"] != sys.executable:
        raise ValueError("frozen method, report or interpreter identity differs")
    require_code(Path(plan["code_root"]), plan["code_files"])
    configuration = require_plain_path(Path(plan["configuration_path"]))
    if file_sha256(configuration) != plan["configuration_sha256"]:
        raise ValueError("frozen configuration differs")
    require_profile(read_json(configuration))
    manifest = read_inference_identity(
        Path(plan["identity_manifest_path"]), plan["identity_artifact_sha256"]
    )
    allowed = {(r.dataset, r.opaque_case_id) for r in manifest.records}
    if type(plan["cases"]) is not list:
        raise ValueError("execution plan needs a fixed queue")
    for item in require_queue(tuple(plan["cases"])):
        if (item["dataset"], item["opaque_case_id"]) not in allowed:
            raise ValueError("queue case not present in inference identity manifest")
        private = require_plain_path(Path(item["input_path"]))
        if file_sha256(private) != item["input_sha256"]:
            raise ValueError("frozen adapter input differs")
        value = RepairCaseInput.model_validate(read_json(private))
        if value.dataset != item["dataset"] or value.opaque_case_id != item["opaque_case_id"]:
            raise ValueError("private adapter input does not match opaque queue")
    active_case_indexes(plan)
    if plan["schema_version"] == SINGLE_SCHEMA:
        parent_path = require_plain_path(Path(plan["parent_plan_path"]))
        if file_sha256(parent_path) != plan["parent_plan_sha256"]:
            raise ValueError("original four-case plan pin differs")
        parent = read_json(parent_path)
        if (
            parent.get("schema_version") != "nurse.v2.live-repair-execution-plan/1"
            or any(
                parent.get(k) != plan[k]
                for k in ("cases", "identity_manifest_path", "identity_artifact_sha256")
            )
            or parent.get("output_root") == plan["output_root"]
        ):
            raise ValueError("single plan differs from original four-case input lineage")
    require_plain_path(Path(plan["output_root"]))
    return plan


def _require_live_closure(state: Path, case: dict[str, Any]) -> None:
    """Check sealed exchange/receipt linkage; semantic review remains separate."""
    from nurse_rcl.v2.live.ark_bridge import _Ledger
    from nurse_rcl.v2.live.repair_report import RepairReportRun

    # Preserve strict nested tuple contracts: serialized JSON uses the JSON
    # validator, never the exact-Python constructor with decoded list values.
    report = RepairReportRun.model_validate_json(
        canonical_json_bytes(read_json(state / "report/report.json"))
    )
    summary = read_json(state / "audit/summary.json")
    if (
        report.status not in ("full", "partial")
        or not report.exchange_attempted
        or report.exchange_status != "response_ready"
        or report.interpretability_status != "source_and_score_replayed"
        or report.source_audit_sha256 != summary["audit_sha256"]
        or report.source_ranking_sha256 != summary["ranking_sha256"]
        or report.presentation is None
        or not report.presentation.safe_display_blocks
        or report.raw_response_hex is None
        or report.response_sha256 is None
        or sha256_hex(bytes.fromhex(report.raw_response_hex)) != report.response_sha256
    ):
        raise ValueError("live report closure incomplete")
    binding = read_json(state / "request-binding.json")
    if (
        binding["opaque_case_id"] != case["opaque_case_id"]
        or binding["request_sha256"] != report.request_sha256
    ):
        raise ValueError("live request receipt binding differs")
    found = False
    for path in state.glob("ark-receipt-*.json"):
        receipt = read_json(path)
        if path.name != f"ark-receipt-{sha256_hex(canonical_json_bytes(receipt))[:16]}.json":
            raise ValueError("ARK receipt pin differs")
        ledger = _Ledger.model_validate(receipt["ledger"])
        if ledger.binding.model_dump(mode="json") != binding or receipt["attempt_count"] != len(
            ledger.attempts
        ):
            raise ValueError("ARK receipt identity differs")
        found |= any(a.response_sha256 == report.response_sha256 for a in ledger.attempts)
    if not found:
        raise ValueError("no actual report response receipt")


def run_round(
    plan_path: Path, plan_sha256: str, *, progress: Callable[[dict[str, Any]], None]
) -> dict[str, Any]:
    import fcntl

    from nurse_rcl.v2.live.repair_stage import _completed_bundle, stage_identity

    plan = load_execution(plan_path, plan_sha256)
    out = require_plain_path(Path(plan["output_root"]))
    fd = os.open(out / "round.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish_json(out / "identity.json", dict(execution_sha256=plan_sha256))
        if (out / "terminal.json").exists():
            return read_json(out / "terminal.json")
        clock = open_clock(
            out / "clock.json", identity=plan_sha256, seconds=ROUND_SECONDS, now=time.time()
        )
        results = []
        active = active_case_indexes(plan)
        total = len(active)
        for position, index in enumerate(active):
            case = plan["cases"][index]
            state = out / case["opaque_case_id"]
            if state.is_symlink():
                raise ValueError("case state cannot be a symlink")
            state.mkdir(mode=0o700, exist_ok=True)
            require_plain_path(state)
            terminal = state / "terminal.json"
            if terminal.exists():
                results.append(read_json(terminal))
                continue
            case_identity = sha256_hex(canonical_json_bytes([plan_sha256, case]))
            result: dict[str, Any] = dict(
                opaque_case_id=case["opaque_case_id"],
                dataset=case["dataset"],
                status="execution_completed",
                source_score_replay_pending=True,
                semantic_review_pending=True,
                execution_sha256=plan_sha256,
            )
            try:
                timing = TimingJournal.open(
                    state / "timing",
                    identity=case_identity,
                    now=time.monotonic(),
                    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                )
            except (ValueError, OSError, CaseBudgetExpired) as error:
                result.update(
                    status="failed",
                    reason="timing_recovery_failed",
                    error_type=type(error).__name__,
                )
                publish_json(terminal, result)
                results.append(result)
                continue
            for stage in STAGES:
                try:
                    timing.switch("method", reason="orchestration", now=time.monotonic())
                except CaseBudgetExpired as error:
                    result.update(status="failed", stage=stage, reason=str(error))
                    break
                if remaining_seconds(clock, now=time.time()) <= 0:
                    result.update(status="failed", stage=stage, reason="round_timeout")
                    break
                identity = stage_identity(plan_sha256, case["input_sha256"], stage)
                if (state / stage).exists():
                    try:
                        _completed_bundle(state, stage, identity)
                    except (OSError, ValueError):
                        result.update(
                            status="failed",
                            stage=stage,
                            reason="stage_publication_integrity",
                        )
                        break
                    continue
                progress(
                    dict(
                        event="case_stage",
                        case_index=position + 1,
                        total_cases=total,
                        dataset=case["dataset"],
                        opaque_case_id=case["opaque_case_id"],
                        stage=stage,
                        **timing.clock.summary(),
                    )
                )
                code = supervise_phases(
                    [
                        sys.executable,
                        "-m",
                        "nurse_rcl.v2.live.repair_execution",
                        "--plan",
                        str(plan_path),
                        "--plan-sha256",
                        plan_sha256,
                        "--case-index",
                        str(index),
                        "--stage",
                        stage,
                    ],
                    log_path=state / f"{stage}.log",
                    timing=timing,
                    progress=progress,
                )
                if code != 0:
                    reason = "worker_failure"
                    if code == 124:
                        reason = timing.clock.account + "_timeout"
                    result.update(status="failed", stage=stage, exit_code=code, reason=reason)
                    break
                try:
                    _completed_bundle(state, stage, identity)
                except (OSError, ValueError):
                    result.update(
                        status="failed",
                        stage=stage,
                        exit_code=code,
                        reason="stage_publication_integrity",
                    )
                    break
            if result["status"] != "failed" and plan["schema_version"] == SINGLE_SCHEMA:
                try:
                    _require_live_closure(state, case)
                    result.update(
                        status="live_closure_completed", source_score_replay_pending=False
                    )
                except (ValueError, OSError, KeyError):
                    result.update(status="failed", reason="live_closure_incomplete")
            if result["status"] != "failed":
                try:
                    timing.checkpoint(now=time.monotonic())
                except CaseBudgetExpired as error:
                    result.update(status="failed", reason=str(error))
            result["timing"] = timing.clock.summary()
            publish_json(terminal, result)
            results.append(result)
            progress(
                dict(
                    event="case_terminal", completed_cases=len(results), total_cases=total, **result
                )
            )
        failed = sum(r["status"] == "failed" for r in results)
        summary = dict(
            execution_sha256=plan_sha256,
            total=total,
            failed=failed,
            completed=total - failed,
            status="execution_completed" if not failed else "completed_with_failures",
            gate_passed=False,
            next_step="independent_source_score_replay_then_isolated_evaluation",
            cases=results,
        )
        publish_json(out / "terminal.json", summary)
        progress(dict(event="round_terminal", total=total, failed=failed, completed=total - failed))
        return summary
    finally:
        os.close(fd)


def run_worker(
    plan: dict[str, Any],
    execution_sha256: str,
    case_index: int,
    stage: str,
    *,
    progress: Callable[[dict[str, Any]], None],
) -> None:
    from nurse_rcl.agents.ark import load_ark_settings_from_key_file
    from nurse_rcl.v2.live.ark_bridge import LiveArkBridge
    from nurse_rcl.v2.live.repair_driver import report_binding
    from nurse_rcl.v2.live.repair_report_prompt import REPAIR_EXCHANGE_VERSION
    from nurse_rcl.v2.live.repair_stage import _completed_bundle, run_stage, stage_identity

    if case_index not in active_case_indexes(plan):
        raise ValueError("worker case is outside the frozen active scope")
    case = plan["cases"][case_index]
    out = require_plain_path(Path(plan["output_root"]))
    state = require_plain_path(out / case["opaque_case_id"])
    spec = read_json(Path(case["input_path"]))

    def check_code() -> None:
        require_code(Path(plan["code_root"]), plan["code_files"])

    def complete(payload: bytes) -> bytes:
        check_code()
        # The source-backed report invokes this only after full replay; the
        # independently pinned stage audit, not request text, supplies the pin.
        _completed_bundle(
            state,
            "audit",
            stage_identity(execution_sha256, case["input_sha256"], "audit"),
        )
        audit = read_json(state / "audit/summary.json")
        binding = report_binding(
            payload,
            opaque_case_id=case["opaque_case_id"],
            source_sha256=spec["source_descriptor_sha256"],
            audit_sha256=audit["audit_sha256"],
        )
        publish_json(state / "request-binding.json", binding.model_dump(mode="json"))
        with LiveArkBridge(
            root=out / "ark",
            lock_path=Path(
                "/home/wangrunzhou/NURSE/development/live-rcl-20260918/locks/ark-in-flight.lock"
            ),
            settings=load_ark_settings_from_key_file(
                Path("/home/wangrunzhou/.config/nurse/ark.env")
            ),
            code_sha256=sha256_hex(canonical_json_bytes(plan["code_files"])),
            config_sha256=execution_sha256,
            authorized_case_ids=tuple(
                plan["cases"][i]["opaque_case_id"] for i in active_case_indexes(plan)
            ),
            request_protocol=REPAIR_EXCHANGE_VERSION,
        ) as bridge:
            progress(
                dict(
                    event="ark_request_started",
                    request_sha256=binding.request_sha256,
                    max_attempts=5,
                    payload_bytes=len(payload),
                )
            )
            try:
                return bridge.complete(binding, payload)
            finally:
                receipt = bridge.receipt(binding)
                pin = sha256_hex(canonical_json_bytes(receipt))
                publish_json(state / f"ark-receipt-{pin[:16]}.json", receipt)
                progress(
                    dict(
                        event="ark_request_finished",
                        attempts=receipt["attempt_count"],
                        response_reuses=receipt["response_reuses"],
                    )
                )
                check_code()

    check_code()
    run_stage(
        stage=stage,
        specification=spec,
        expected_input_sha256=case["input_sha256"],
        identity_manifest_path=Path(plan["identity_manifest_path"]),
        expected_identity_artifact_sha256=plan["identity_artifact_sha256"],
        state_root=state,
        execution_sha256=execution_sha256,
        complete=complete,
        progress=progress,
    )
    check_code()


def main() -> int:
    import argparse
    import json
    import resource

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--case-index", type=int, choices=range(4))
    parser.add_argument("--stage", choices=tuple(STAGES))
    parser.add_argument("--timing-fd", type=int)
    args = parser.parse_args()
    install_termination_handlers()
    started = time.monotonic()

    def emit(row: dict[str, Any]) -> None:
        print(
            json.dumps(
                dict(
                    **row,
                    wall_seconds=time.monotonic() - started,
                    max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                )
            ),
            flush=True,
        )

    try:
        worker = args.stage is not None or args.case_index is not None
        if worker and (args.stage is None or args.case_index is None or args.timing_fd is None):
            raise ValueError("worker needs stage, active case and private timing channel")
        if not worker and args.timing_fd is not None:
            raise ValueError("parent cannot receive a child timing channel")
        plan = load_execution(args.plan, args.plan_sha256)
        if not Path(__file__).resolve().is_relative_to(require_plain_path(Path(plan["code_root"]))):
            raise ValueError("loaded implementation is not the frozen source tree")
        resource.setrlimit(resource.RLIMIT_AS, (8589934592, 8589934592))
        if args.stage is None and args.case_index is None:
            result = run_round(args.plan, args.plan_sha256, progress=emit)
            return 0 if result["failed"] == 0 else 1
        if args.stage is None or args.case_index is None:
            raise ValueError("stage and case index must be specified together")
        client = PhaseClient(args.timing_fd)
        reasons = dict(queries="query_execute", audit="score_execute", report="report_execute")
        with _use_phase_client(client), _phase("method", reasons[args.stage]):
            run_worker(plan, args.plan_sha256, args.case_index, args.stage, progress=emit)
        client.finish()
        return 0
    except KeyboardInterrupt:
        emit(dict(event="execution_interrupted"))
        return 130
    except Exception as error:
        # Paths, provider text and private adapter fields are not console output.
        emit(dict(event="execution_failed", error_type=type(error).__name__))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
