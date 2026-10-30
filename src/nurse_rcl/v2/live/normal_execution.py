"""Single normal RCL worker with bounded admission, phase clock and finite publication."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import resource
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.aiops25_adapter import build_aiops25_adapter_config
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.exchange_json import ExchangeLimits, decode_json_object
from nurse_rcl.v2.live.normal_artifacts import NORMAL_POLICY
from nurse_rcl.v2.live.normal_b0_case import B0_POLICY, inspect_b0_result, run_b0_case
from nurse_rcl.v2.live.normal_case import _identity, inspect_normal_result, run_normal_case
from nurse_rcl.v2.live.normal_preparation import _require_compatible_key
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.repair_case_execution import _profile_fields
from nurse_rcl.v2.live.repair_driver import publish_json, read_json, require_profile_common
from nurse_rcl.v2.live.repair_execution import (
    install_termination_handlers,
    require_code,
    supervise_phases,
)
from nurse_rcl.v2.live.repair_execution import method_identity as numerical_identity
from nurse_rcl.v2.live.repair_module_timing import (
    _module_span,
    _module_timing_scope,
    summarize_module_timing,
)
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationRun
from nurse_rcl.v2.live.repair_stage import RepairCaseInput
from nurse_rcl.v2.live.repair_timing import (
    DIAGNOSTIC_TIMING_POLICY,
    PhaseClient,
    TimingJournal,
    _phase,
    _use_phase_client,
)
from nurse_rcl.v2.live.staged_case import require_plain_path
from nurse_rcl.v2.opaque_ids import InferenceIdentityManifest
from nurse_rcl.v2.rcabench_adapter import build_rcabench_adapter_config

SCHEMA = "nurse.v2.normal-rcl-single-plan/1"
B0_SCHEMA = "nurse.v2.normal-b0-single-plan/1"
_PLAN_FIELDS = {
    "schema_version", "code_root", "code_files", "configuration_path", "configuration_sha256",
    "identity_manifest_path", "identity_artifact_sha256", "case", "output_root",
    "python_executable", "method_identity", "parent_plan_path", "parent_plan_sha256",
    "lineage_path", "lineage_sha256", "preparation",
}


def method_identity() -> dict[str, str]:
    values = numerical_identity()
    values.pop("raw_audit")
    values.update(
        case_execution=NORMAL_POLICY, timing_policy=DIAGNOSTIC_TIMING_POLICY,
        evidence_schema="nurse.v2.normal-rcl-evidence/1",
        report_schema="nurse.v2.normal-rcl-report/1",
        preparation_schema="nurse.v2.live-repair-preparation-manifest/1",
    )
    return values


def b0_method_identity() -> dict[str, str]:
    values = method_identity()
    values.pop("report_schema")
    values.update(case_execution=B0_POLICY, result_schema="nurse.v2.normal-b0-result/1")
    return values


def b0_configuration() -> dict[str, Any]:
    return dict(
        schema_version="nurse.v2.normal-b0-configuration/1", case_execution=B0_POLICY,
        preparation_case_budget_seconds=3600, method_case_budget_seconds=None,
        timing_policy=DIAGNOSTIC_TIMING_POLICY, worker_count=1, memory_limit_bytes=8589934592,
        branch_tool_budget=8, semantic_requests=0, report_requests=0, max_request_attempts=0,
        full_recompute="not_performed",
    )


def require_profile(profile: dict[str, Any]) -> dict[str, int | None]:
    require_profile_common(profile)
    fixed = dict(
        schema_version="nurse.v2.live-evidence-repair-profile/4",
        case_execution_version=NORMAL_POLICY, worker_stage_order=["case"],
        timing_policy_version=DIAGNOSTIC_TIMING_POLICY,
        preparation_case_budget_seconds=3600, method_case_budget_seconds=None,
        case_timeout_seconds=None, round_timeout_seconds=None,
        round_report_logical_requests=1, round_max_network_attempts=5,
        round_launch_requires="task_9_7_frozen_normal_case_and_consumption_confirmation",
    )
    if set(profile) != _profile_fields() or any(
        type(profile.get(k)) is not type(v) or canonical_json_bytes(profile[k])
        != canonical_json_bytes(v) for k, v in fixed.items()
    ):
        raise ValueError("normal profile fields or timing policy differ")
    return dict(preparation=3600, method=None, case=None)


def _pinned(path: Path, pin: str, limit: int = 4 * 1024 * 1024) -> bytes:
    require_pin(pin)
    raw = _read_bytes(path, limit)
    if sha256_hex(raw) != pin:
        raise ValueError("normal boundary bytes differ from independent pin")
    return raw


def _read_plan(path: Path, pin: str) -> dict[str, Any]:
    """Small parent envelope only; the child admits code/config/source metadata once."""
    plan = _metadata(_pinned(path, pin))
    pure = plan.get("schema_version") == B0_SCHEMA
    if set(plan) != _PLAN_FIELDS or plan["schema_version"] not in {SCHEMA, B0_SCHEMA} or (
        plan["method_identity"] != (b0_method_identity() if pure else method_identity())
        or plan["python_executable"] != sys.executable
    ):
        raise ValueError("normal plan fields, method or interpreter differ")
    case = plan["case"]
    if type(case) is not dict or set(case) != {
        "dataset", "opaque_case_id", "input_path", "input_sha256",
    }:
        raise ValueError("normal plan requires exactly one complete case")
    run = RepairPreparationRun.model_validate_json(canonical_json_bytes(plan["preparation"]))
    if (case["dataset"], case["opaque_case_id"], case["input_sha256"]) != (
        run.key.dataset, run.key.opaque_case_id, run.key.input_sha256,
    ) or Path(run.archive_root).name != run.key.key_sha256:
        raise ValueError("normal plan input/preparation identity differs")
    require_plain_path(Path(plan["output_root"]))
    if not Path(__file__).resolve().is_relative_to(require_plain_path(Path(plan["code_root"]))):
        raise ValueError("normal implementation is outside frozen code root")
    return plan


def _metadata(raw: bytes) -> dict[str, Any]:
    # Metadata may be pretty-printed. Its independently pinned *actual* bytes
    # are already checked; archive-only canonical serialization is not required.
    return decode_json_object(raw, ExchangeLimits(4 * 1024 * 1024, 64, 100000))


def _internal(plan: dict[str, Any]) -> dict[str, Any]:
    return dict(
        state_root=Path(plan["output_root"]) / plan["case"]["opaque_case_id"],
        code_sha256=sha256_hex(canonical_json_bytes(plan["code_files"])),
        preparation=RepairPreparationRun.model_validate_json(
            canonical_json_bytes(plan["preparation"]),
        ),
        **({"b0_only": True} if plan["schema_version"] == B0_SCHEMA else {}),
    )


def load_execution(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Admit actual pinned bytes, not separately hashed/reopened raw telemetry."""
    plan = _read_plan(path, expected_sha256)
    require_code(Path(plan["code_root"]), plan["code_files"])
    profile = _metadata(_pinned(
        Path(plan["configuration_path"]), plan["configuration_sha256"],
    ))
    pure = plan["schema_version"] == B0_SCHEMA
    if pure:
        if canonical_json_bytes(profile) != canonical_json_bytes(b0_configuration()):
            raise ValueError("B0 configuration differs")
    else:
        require_profile(profile)
    manifest = InferenceIdentityManifest.model_validate_json(_pinned(
        Path(plan["identity_manifest_path"]), plan["identity_artifact_sha256"],
    ))
    case = plan["case"]
    specification = RepairCaseInput.model_validate_json(_pinned(
        Path(case["input_path"]), case["input_sha256"],
    ))
    if not manifest.verify() or not any(
        (r.dataset, r.opaque_case_id) == (specification.dataset, specification.opaque_case_id)
        for r in manifest.records
    ) or (specification.dataset, specification.opaque_case_id) != (
        case["dataset"], case["opaque_case_id"],
    ):
        raise ValueError("normal case is outside the inference identity manifest")
    parent = _metadata(_pinned(
        Path(plan["parent_plan_path"]), plan["parent_plan_sha256"],
    ))
    if case not in parent.get("cases", []) or parent.get("output_root") == plan["output_root"]:
        raise ValueError("normal case differs from frozen parent input lineage")
    lineage = _metadata(_pinned(Path(plan["lineage_path"]), plan["lineage_sha256"]))
    fixed = dict(
        schema_version=("nurse.v2.normal-b0-lineage/1" if pure else
                        "nurse.v2.normal-rcl-lineage/1"), opaque_case_id=case["opaque_case_id"],
        parent_plan_path=plan["parent_plan_path"], parent_plan_sha256=plan["parent_plan_sha256"],
        budget_relationship="new_development_run_not_resume_or_refund",
        new_query_budget=8, new_report_requests=0 if pure else 1,
        new_max_attempts=0 if pure else 5,
    )
    if set(lineage) != {*fixed, "previous_runs"} or any(
        type(lineage.get(k)) is not type(v) or lineage[k] != v for k, v in fixed.items()
    ) or type(lineage["previous_runs"]) is not list or len(lineage["previous_runs"]) > 16:
        raise ValueError("normal lineage or new consumption budget differs")
    internal = _internal(plan)
    run = internal["preparation"]
    if run.key.source_files != specification.source_files:
        raise ValueError("normal preparation input source inventory differs")
    extra: dict[str, Any] = {}
    configuration: Any
    if specification.dataset == "rcabench":
        configuration = build_rcabench_adapter_config(window_seconds=240)
        if specification.stage_sha256 is None or specification.source_case_id is not None or (
            specification.authorized_queue is not None
        ):
            raise ValueError("normal RCABench requires opaque staged input")
    else:
        configuration = build_aiops25_adapter_config(window_seconds=480)
        if specification.stage_sha256 is not None or specification.source_case_id is None or (
            specification.authorized_queue is None
        ):
            raise ValueError("normal AIOps25 requires complete authorized case metadata")
        extra = dict(source_case_id=specification.source_case_id,
                     authorized_queue=specification.authorized_queue)
    if configuration.configuration_sha256 != specification.configuration_sha256:
        raise ValueError("normal input adapter configuration differs")
    raw = dict(
        source_root=require_plain_path(Path(specification.source_root)),
        dataset=specification.dataset, opaque_case_id=specification.opaque_case_id,
        configuration=configuration,
        expected_source_descriptor_sha256=specification.source_descriptor_sha256,
        store_root=Path(run.archive_root) / "blocks",
        ledger_root=internal["state_root"] / "query-ledger", **extra,
    )
    _require_compatible_key(run.key, raw)
    return dict(**internal, raw_arguments=raw, identity_manifest=manifest)


def run_round(
    path: Path, pin: str, *, progress: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    started = time.monotonic()
    plan = _read_plan(path, pin)
    out = Path(plan["output_root"])
    internal = _internal(plan)
    state: Path = internal["state_root"]
    state.mkdir(mode=0o700, exist_ok=True)
    require_plain_path(state)
    identity = _identity(internal, pin)
    inspect_result = inspect_b0_result if internal.get("b0_only") is True else inspect_normal_result
    descriptor = os.open(out / "round.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish_json(out / "identity.json", identity)
        if (out / "terminal.json").exists():
            retained = read_json(out / "terminal.json")
            if retained.get("identity") != identity:
                raise ValueError("normal retained terminal identity differs")
            if retained["status"] in {"completed", "completed_partial_report"}:
                inspect_result(state, identity)
            return {**retained, "timing": read_json(out / "timing-summary.json")}
        timing = TimingJournal.open(
            state / "timing", identity=pin, now=started,
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            expected_policy=DIAGNOSTIC_TIMING_POLICY,
        )
        terminal: dict[str, Any] = dict(identity=identity, status="failed", worker_exit_code=None)
        with _module_timing_scope(state / "module-spans", identity=pin,
                                  boot_id=timing.boot_id) as recorder:
            try:
                def event(row: dict[str, Any]) -> None:
                    recorder.observe()
                    progress(dict(row, case_index=1, total_cases=1,
                                  opaque_case_id=plan["case"]["opaque_case_id"]))

                with _module_span("worker_supervision", role="orchestration"):
                    code = supervise_phases(
                        [sys.executable, "-m", "nurse_rcl.v2.live.normal_execution",
                         "--plan", str(path), "--plan-sha256", pin, "--worker",
                         "--module-parent", str(recorder.current_span_id)],
                        log_path=state / "case.log", timing=timing, progress=event,
                    )
                terminal["worker_exit_code"] = code
                with _module_span("parent_finalize", role="verification"):
                    if code != 0:
                        raise ValueError("normal worker did not finish")
                    result = inspect_result(state, identity)
                    terminal.update(status=result["status"], paid_count=result["paid_count"],
                                    candidate_count=result["candidate_count"],
                                    evidence_sha256=result["evidence_sha256"])
            except Exception as error:
                terminal.update(status="failed", error_type=type(error).__name__)
        timing.checkpoint(now=time.monotonic())
        table = summarize_module_timing(
            event_paths=tuple(sorted((state / "module-spans").glob("*.jsonl"))),
            timing_root=timing.root, expected_timing_identity=pin,
            expected_policy=DIAGNOSTIC_TIMING_POLICY,
        )
        publish_json(state / "module-timing.json", table)
        publish_json(out / "terminal.json", terminal)
        # Includes worker waits, result inspection and final terminal publication.
        # The tiny timing-summary write itself is not recursively timed.
        timing.checkpoint(now=time.monotonic())
        summary = timing.clock.summary()
        publish_json(out / "timing-summary.json", summary)
        progress(dict(event="normal_terminal", **terminal, timing=summary))
        return {**terminal, "timing": summary}
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--timing-fd", type=int)
    parser.add_argument("--module-parent")
    args = parser.parse_args()
    install_termination_handlers()
    resource.setrlimit(resource.RLIMIT_AS, (8589934592, 8589934592))

    def emit(row: dict[str, Any]) -> None:
        print(json.dumps(row), flush=True)

    try:
        if args.worker != (args.timing_fd is not None and args.module_parent is not None):
            raise ValueError("normal worker requires private parent channels")
        if not args.worker:
            result = run_round(args.plan, args.plan_sha256, progress=emit)
        else:
            # The parent has read only the envelope. All data admission is owned
            # here and charged as preparation, once per worker invocation.
            envelope = _read_plan(args.plan, args.plan_sha256)
            state = _internal(envelope)["state_root"]
            client = PhaseClient(args.timing_fd)
            with _use_phase_client(client), _module_timing_scope(
                state / "module-spans", identity=args.plan_sha256,
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                parent_id=args.module_parent,
            ), _module_span("case_worker"):
                with _phase("preparation", "input_validation"), _module_span("input_guard"):
                    plan = load_execution(args.plan, args.plan_sha256)
                run_case = run_b0_case if plan.get("b0_only") is True else run_normal_case
                result = run_case(plan=plan, plan_sha256=args.plan_sha256)
            client.finish()
        return int(result["status"] not in {"completed", "completed_partial_report"})
    except KeyboardInterrupt:
        emit(dict(event="normal_interrupted"))
        return 130
    except Exception as error:
        emit(dict(event="normal_failed", error_type=type(error).__name__))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
