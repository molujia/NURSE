"""Explicit single-worker case-owner policy; legacy three-stage execution stays intact."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any, cast

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live import repair_execution as legacy
from nurse_rcl.v2.live.repair_case_artifacts import (
    REPAIR_CASE_AUDIT_SEMANTICS_SHA256,
    REPAIR_CASE_POLICY,
    RepairCaseAudit,
    RepairCaseClosure,
)
from nurse_rcl.v2.live.repair_case_owner import _require_response_receipt, _terminal_journal_guard
from nurse_rcl.v2.live.repair_case_report import RepairCaseReportRun
from nurse_rcl.v2.live.repair_case_report import _snapshot as report_snapshot
from nurse_rcl.v2.live.repair_driver import (
    publish_json,
    read_bundle,
    read_json,
    require_queue,
)
from nurse_rcl.v2.live.repair_driver import (
    require_profile as require_legacy_profile,
)
from nurse_rcl.v2.live.repair_execution import require_code, supervise_phases
from nurse_rcl.v2.live.repair_module_timing import (
    _module_span,
    _module_timing_scope,
    summarize_module_timing,
)
from nurse_rcl.v2.live.repair_prepared_case_artifacts import (
    RepairPreparedCaseAudit,
    RepairPreparedCaseClosure,
    RepairPreparedReportRun,
)
from nurse_rcl.v2.live.repair_queries import _seal
from nurse_rcl.v2.live.repair_safety import canonical
from nurse_rcl.v2.live.repair_stage import RepairCaseInput, _read_model
from nurse_rcl.v2.live.repair_timing import (
    DIAGNOSTIC_TIMING_POLICY,
    TIMING_POLICY,
    CaseBudgetExpired,
    PhaseClient,
    TimingJournal,
    _phase,
    _use_phase_client,
)
from nurse_rcl.v2.live.sources import file_sha256
from nurse_rcl.v2.live.staged_case import read_inference_identity, require_plain_path

SINGLE_SCHEMA = "nurse.v2.live-repair-case-owner-single-plan/1"
PREPARED_SCHEMA = "nurse.v2.live-repair-prepared-single-plan/1"
_PENDING = "parent_finalization_pending"


def _profile_fields() -> set[str]:
    return {
        "schema_version",
        "profile",
        "purpose",
        "configuration_state",
        "dynamic_verification",
        "pre_ranking_semantics",
        "history",
        "worker_count",
        "cases_in_flight",
        "llm_in_flight",
        "max_request_attempts",
        "endpoint",
        "model",
        "credential_variable",
        "report_logical_requests_per_case",
        "round_report_logical_requests",
        "round_max_network_attempts",
        "report_thinking",
        "report_input_token_limit",
        "report_output_token_limit",
        "report_temperature",
        "request_timeout_seconds",
        "timing_policy_version",
        "preparation_case_budget_seconds",
        "method_case_budget_seconds",
        "case_timeout_seconds",
        "round_timeout_seconds",
        "worker_stage_order",
        "case_execution_version",
        "report_configuration_version",
        "report_exchange_version",
        "address_space_limit_bytes",
        "terminate_grace_seconds",
        "heartbeat_seconds",
        "rcabench_window_seconds",
        "aiops25_window_seconds",
        "expected_cadence_seconds",
        "max_dynamic_queries_per_case",
        "q_base",
        "q_max",
        "candidate_truncation",
        "case_splitting",
        "modality_filtering",
        "b0_capacity_profile",
        "repair_capacity_profile",
        "repair_max_serialization_nodes",
        "repair_max_serialization_json_bytes",
        "b0_max_serialization_nodes",
        "b0_max_serialization_json_bytes",
        "b0_max_serialization_depth",
        "b0_max_serialization_string_length",
        "statistical_scale",
        "continuous_episode",
        "score_weights",
        "round_launch_requires",
    }


def require_profile(profile: dict[str, Any]) -> dict[str, int]:
    # Legacy finite constraints remain exact; never substitute fake finite values.
    fields = _profile_fields()
    if (
        type(profile) is not dict
        or set(profile) != fields
        or profile["schema_version"] != "nurse.v2.live-evidence-repair-profile/2"
        or profile["case_execution_version"] != REPAIR_CASE_POLICY
        or type(profile["worker_stage_order"]) is not list
        or canonical_json_bytes(profile["worker_stage_order"]) != b'["case"]'
    ):
        raise ValueError("case-owner profile fields or execution controls differ")
    adapted = {
        k: v
        for k, v in profile.items()
        if k not in {"case_execution_version", "worker_stage_order"}
    }
    adapted.update(
        schema_version="nurse.v2.live-evidence-repair-profile/1",
        preparation_execution_version="live-repair-invocation-reuse/1",
        stage_order=["queries", "audit", "report"],
    )
    return require_legacy_profile(adapted)


def method_identity() -> dict[str, str]:
    result = legacy.method_identity()
    result.pop("raw_audit")
    result.update(
        case_execution=REPAIR_CASE_POLICY,
        case_audit=REPAIR_CASE_AUDIT_SEMANTICS_SHA256,
        report_schema="nurse.v2.live-repair-report/2",
        projection_schema="nurse.v2.live-repair-report-projection/2",
        closure_schema="nurse.v2.live-repair-case-closure/1",
    )
    return result


def require_prepared_profile(profile: dict[str, Any]) -> dict[str, int | None]:
    from nurse_rcl.v2.live.repair_driver import require_profile_common
    from nurse_rcl.v2.live.repair_prepared_case_artifacts import REPAIR_PREPARED_POLICY

    require_profile_common(profile)
    fixed = dict(
        schema_version="nurse.v2.live-evidence-repair-profile/3",
        case_execution_version=REPAIR_PREPARED_POLICY,
        worker_stage_order=["case"],
        timing_policy_version=DIAGNOSTIC_TIMING_POLICY,
        preparation_case_budget_seconds=3600,
        method_case_budget_seconds=None,
        case_timeout_seconds=None,
        round_timeout_seconds=None,
        round_report_logical_requests=1,
        round_max_network_attempts=5,
        round_launch_requires="task_8_5_sealed_single_case_preparation_and_diagnostic_timing",
    )
    if set(profile) != _profile_fields() or any(
        type(profile.get(k)) is not type(v)
        or canonical_json_bytes(profile[k]) != canonical_json_bytes(v)
        for k, v in fixed.items()
    ):
        raise ValueError("prepared profile fields or diagnostic controls differ")
    return dict(preparation=3600, method=None, case=None, cleanup=30)


def prepared_method_identity() -> dict[str, str]:
    from nurse_rcl.v2.live.repair_prepared_case_artifacts import (
        REPAIR_PREPARED_AUDIT_SEMANTICS_SHA256,
        REPAIR_PREPARED_POLICY,
    )

    result = method_identity()
    result.update(
        case_execution=REPAIR_PREPARED_POLICY,
        case_audit=REPAIR_PREPARED_AUDIT_SEMANTICS_SHA256,
        report_schema="nurse.v2.live-repair-prepared-report/1",
        projection_schema="nurse.v2.live-repair-prepared-report-projection/1",
        closure_schema="nurse.v2.live-repair-prepared-case-closure/1",
        timing_policy=DIAGNOSTIC_TIMING_POLICY,
        preparation_schema="nurse.v2.live-repair-preparation-manifest/1",
    )
    return result


def _policy(plan: dict[str, Any]) -> str:
    if plan["schema_version"] == PREPARED_SCHEMA:
        return DIAGNOSTIC_TIMING_POLICY
    if plan["schema_version"] == SINGLE_SCHEMA:
        return TIMING_POLICY
    raise ValueError("unsupported case execution policy")


def _prepared_arguments(plan: dict[str, Any]) -> dict[str, Any]:
    """Metadata-only arguments for key checks; no assembly or sibling telemetry."""
    from nurse_rcl.v2.rcabench_adapter import build_rcabench_adapter_config

    case = plan["cases"][0]
    specification = RepairCaseInput.model_validate_json(
        canonical_json_bytes(read_json(Path(case["input_path"])))
    )
    if specification.dataset != "rcabench":
        raise ValueError("diagnostic prepared plan selects only original RCABench case")
    state = Path(plan["output_root"]) / case["opaque_case_id"]
    config = build_rcabench_adapter_config(window_seconds=240)
    if config.configuration_sha256 != specification.configuration_sha256:
        raise ValueError("prepared adapter configuration differs")
    return dict(
        source_root=Path(specification.source_root),
        dataset=specification.dataset,
        opaque_case_id=specification.opaque_case_id,
        configuration=config,
        expected_source_descriptor_sha256=specification.source_descriptor_sha256,
        store_root=state / "blocks",
        ledger_root=state / "query-ledger",
    )


def _require_preparation_plan(plan: dict[str, Any]) -> None:
    from nurse_rcl.v2.live.repair_preparation_archive import (
        _require_key,
        _require_preparation_files,
    )
    from nurse_rcl.v2.live.repair_preparation_artifacts import (
        RepairPreparationBinding,
        RepairPreparationRun,
    )

    run = RepairPreparationRun.model_validate_json(canonical_json_bytes(plan["preparation"]))
    case = plan["cases"][0]
    archive = Path(run.archive_root)
    require_plain_path(archive.parent)
    if (
        archive.name != run.key.key_sha256
        or archive.is_symlink()
        or run.key.input_sha256 != case["input_sha256"]
        or run.key.opaque_case_id != case["opaque_case_id"]
        or (
            run.mode == "build"
            and Path(run.receipt_path)
            != Path(plan["output_root"]) / case["opaque_case_id"] / "preparation-receipt.json"
        )
    ):
        raise ValueError("prepared plan source/archive/receipt binding differs")
    arguments = _prepared_arguments(plan)
    _require_key(run.key, arguments)
    if any(plan["code_files"].get(name) != pin for name, pin in run.key.preparation_code):
        raise ValueError("prepared dependencies differ from frozen execution code")
    if run.mode == "load":
        if run.manifest_sha256 is None or run.receipt_sha256 is None:
            raise ValueError("preparation load lacks independent pins")
        binding = RepairPreparationBinding(
            preparation_key_sha256=run.key.key_sha256,
            manifest_sha256=run.manifest_sha256,
            receipt_sha256=run.receipt_sha256,
        )
        _require_preparation_files(run, binding, arguments)


def active_case_indexes(plan: dict[str, Any]) -> tuple[int, ...]:
    indexes = plan.get("active_case_indexes")
    if (
        plan.get("schema_version") not in (SINGLE_SCHEMA, PREPARED_SCHEMA)
        or type(indexes) is not list
        or len(indexes) != 1
        or type(indexes[0]) is not int
        or indexes[0] != 0
        or plan["cases"][0]["dataset"] != "rcabench"
        or plan["cases"][0]["opaque_case_id"] != legacy.FIRST_CASE_ID
    ):
        raise ValueError("case-owner execution selects only the original first case")
    return (0,)


def load_execution(path: Path, expected_sha256: str) -> dict[str, Any]:
    source = require_plain_path(path)
    if file_sha256(source) != expected_sha256:
        raise ValueError("case-owner plan differs from independent launch pin")
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
        "parent_plan_path",
        "parent_plan_sha256",
        "active_case_indexes",
        "lineage_path",
        "lineage_sha256",
    }
    prepared = plan.get("schema_version") == PREPARED_SCHEMA
    if prepared:
        fields.add("preparation")
    if (
        set(plan) != fields
        or plan["method_identity"]
        != (prepared_method_identity() if prepared else method_identity())
        or plan["python_executable"] != sys.executable
    ):
        raise ValueError("case-owner plan fields, method or interpreter differ")
    require_code(Path(plan["code_root"]), plan["code_files"])
    configuration = require_plain_path(Path(plan["configuration_path"]))
    if file_sha256(configuration) != plan["configuration_sha256"]:
        raise ValueError("case-owner configuration changed")
    (require_prepared_profile if prepared else require_profile)(read_json(configuration))
    manifest = read_inference_identity(
        Path(plan["identity_manifest_path"]), plan["identity_artifact_sha256"]
    )
    allowed = {(r.dataset, r.opaque_case_id) for r in manifest.records}
    if type(plan["cases"]) is not list:
        raise ValueError("case-owner plan needs the original complete four-case metadata")
    for case in require_queue(tuple(plan["cases"])):
        if (case["dataset"], case["opaque_case_id"]) not in allowed:
            raise ValueError("case outside the inference identity manifest")
        item = require_plain_path(Path(case["input_path"]))
        if file_sha256(item) != case["input_sha256"]:
            raise ValueError("case-owner input metadata differs")
        value = RepairCaseInput.model_validate_json(canonical_json_bytes(read_json(item)))
        if (value.dataset, value.opaque_case_id) != (case["dataset"], case["opaque_case_id"]):
            raise ValueError("case-owner metadata queue binding differs")
    active_case_indexes(plan)
    parent_path = require_plain_path(Path(plan["parent_plan_path"]))
    if file_sha256(parent_path) != plan["parent_plan_sha256"]:
        raise ValueError("case-owner original plan pin differs")
    parent = read_json(parent_path)
    if (
        parent.get("schema_version") != "nurse.v2.live-repair-execution-plan/1"
        or any(
            parent.get(k) != plan[k]
            for k in ("cases", "identity_manifest_path", "identity_artifact_sha256")
        )
        or parent.get("output_root") == plan["output_root"]
    ):
        raise ValueError("case-owner plan changes the four-case input lineage")
    lineage_path = require_plain_path(Path(plan["lineage_path"]))
    if file_sha256(lineage_path) != plan["lineage_sha256"]:
        raise ValueError("case-owner development lineage differs")
    lineage = read_json(lineage_path)
    # This is an independently frozen metadata binding, not replay authority.
    # The launch preflight separately reconstructs the complete old ledgers.
    if (
        lineage.get("schema_version") != "nurse.v2.live-repair-case-owner-lineage/1"
        or lineage.get("opaque_case_id") != legacy.FIRST_CASE_ID
        or lineage.get("parent_plan_path") != plan["parent_plan_path"]
        or lineage.get("parent_plan_sha256") != plan["parent_plan_sha256"]
        or lineage.get("budget_relationship") != "new_development_run_not_resume_or_refund"
        or lineage.get("initial_only") is not True
        or lineage.get("old_budget_exhausted") is not True
        or any(
            type(lineage.get(k)) is not int or lineage[k] != 0
            for k in ("query_reservations", "ark_attempts", "measured", "completed")
        )
    ):
        raise ValueError("case-owner development lineage cannot grant a fresh consumed pool")
    require_plain_path(Path(plan["output_root"]))
    if prepared:
        _require_preparation_plan(plan)
    return plan


def entry_file_pins(plan: dict[str, Any], path: Path, pin: str) -> tuple[tuple[str, str], ...]:
    return (
        (str(path), pin),
        (plan["configuration_path"], plan["configuration_sha256"]),
        (plan["parent_plan_path"], plan["parent_plan_sha256"]),
        (plan["lineage_path"], plan["lineage_sha256"]),
        *((case["input_path"], case["input_sha256"]) for case in plan["cases"]),
    )


def require_closure(
    plan: dict[str, Any], pin: str, state: Path
) -> RepairCaseClosure | RepairPreparedCaseClosure:
    """Parent integrity gate; never grants fresh source/score replay authority."""
    prepared = plan["schema_version"] == PREPARED_SCHEMA
    namespace = require_plain_path(state / ("prepared-case-v1" if prepared else "case-owner-v1"))
    if (namespace / "failure.json").exists() or (namespace / "failure.json").is_symlink():
        raise ValueError("case has retained owning failure")
    closure = _read_model(
        namespace / "closure.json", RepairPreparedCaseClosure if prepared else RepairCaseClosure
    )
    audit = _read_model(
        namespace / "result/audit.json", RepairPreparedCaseAudit if prepared else RepairCaseAudit
    )
    report = _read_model(
        namespace / "result/report.json",
        RepairPreparedReportRun if prepared else RepairCaseReportRun,
    )
    case = plan["cases"][0]
    from nurse_rcl.v2.live.repair_raw_audit import _file_pins

    specification = RepairCaseInput.model_validate_json(
        canonical_json_bytes(read_json(Path(case["input_path"])))
    )
    if (
        _file_pins(Path(specification.source_root), specification.dataset)
        != specification.source_files
    ):
        raise ValueError("case closure source files changed after worker exit")
    timing_identity = sha256_hex(canonical_json_bytes([pin, case]))
    receipt = read_bundle(namespace / "result", identity=pin)
    if (
        closure != _seal(closure, "result_sha256")
        or audit != _seal(audit, "result_sha256")
        or closure.execution_sha256 != pin
        or closure.input_sha256 != case["input_sha256"]
        or (closure.dataset, closure.opaque_case_id) != (case["dataset"], case["opaque_case_id"])
        or closure.code_sha256 != sha256_hex(canonical_json_bytes(plan["code_files"]))
        or closure.timing_identity_sha256 != timing_identity
        or closure.audit_sha256 != audit.result_sha256
        or closure.ranking_sha256 != audit.ranking.result_sha256
        or closure.catalog_sha256 != audit.catalog_sha256
        or closure.session_sha256 != audit.session_sha256
        or closure.journal_sha256 != audit.session.journal_sha256
        or closure.source_descriptor_sha256 != audit.source_descriptor_sha256
        or closure.source_files_sha256 != audit.source_files_sha256
        or closure.candidate_freeze_sha256 != audit.candidate_freeze_sha256
        or closure.report_sha256 != report.result_sha256
        or closure.request_sha256 != report.request_sha256
        or closure.response_sha256 != report.response_sha256
        or receipt
        != dict(
            identity=pin,
            files={
                "audit.json": sha256_hex(canonical(audit, (type(audit),))),
                "report.json": sha256_hex(report_snapshot(report)[0]),
                "verification.json": closure.verification_sha256,
                "exit-checks.json": closure.exit_checks_sha256,
            },
        )
        or closure.publication_sha256 != sha256_hex(canonical_json_bytes(receipt))
    ):
        raise ValueError("case closure identity or publication differs")
    _terminal_journal_guard(state / "query-ledger", audit.session)()
    _require_response_receipt(state, report, audit)
    if prepared:
        from nurse_rcl.v2.live.repair_preparation_archive import _require_preparation_files
        from nurse_rcl.v2.live.repair_preparation_artifacts import (
            RepairPreparationReceipt,
            RepairPreparationRun,
        )

        if not (
            type(closure) is RepairPreparedCaseClosure
            and type(audit) is RepairPreparedCaseAudit
            and type(report) is RepairPreparedReportRun
            and closure.preparation
            == audit.preparation
            == report.preparation
            == report.projection.preparation
        ):
            raise ValueError("prepared closure source policy/bindings differ")
        preparation = RepairPreparationRun.model_validate_json(
            canonical_json_bytes(plan["preparation"])
        )
        _require_preparation_files(preparation, closure.preparation, _prepared_arguments(plan))
        build = _read_model(Path(preparation.receipt_path), RepairPreparationReceipt)
        if (
            build.input_sha256 != case["input_sha256"]
            or build.preparation_key_sha256 != closure.preparation.preparation_key_sha256
            or build.manifest_sha256 != closure.preparation.manifest_sha256
            or (
                preparation.mode == "build"
                and (
                    build.execution_sha256 != pin
                    or build.execution_code_sha256 != closure.code_sha256
                )
            )
            or (
                preparation.mode == "load"
                and (
                    preparation.manifest_sha256 != closure.preparation.manifest_sha256
                    or preparation.receipt_sha256 != closure.preparation.receipt_sha256
                )
            )
        ):
            raise ValueError("prepared closure build provenance differs")
    timing_root = require_plain_path(state / "timing")
    first = read_json(timing_root / "event-00000000.json", limit=4096)
    TimingJournal._read_existing(
        timing_root,
        identity=timing_identity,
        boot_id=first["boot_id"],
        expected_policy=_policy(plan),
    )
    if not any(
        read_json(p, limit=4096)["sha256"] == closure.timing_checkpoint_sha256
        for p in timing_root.iterdir()
    ):
        raise ValueError("case closure timing checkpoint absent from the original chain")
    return cast(RepairCaseClosure | RepairPreparedCaseClosure, closure)


def _retained_case_result(plan: dict[str, Any], pin: str, state: Path) -> dict[str, Any]:
    result = read_json(state / "terminal.json")
    case = plan["cases"][0]
    if (result.get("execution_sha256"), result.get("dataset"), result.get("opaque_case_id")) != (
        pin,
        case["dataset"],
        case["opaque_case_id"],
    ):
        raise ValueError("retained terminal case identity differs")
    if result.get("status") == "failed":
        return result  # A failed terminal never launches another worker.
    closure = require_closure(plan, pin, state)
    timing_root = require_plain_path(state / "timing")
    first = read_json(timing_root / "event-00000000.json", limit=4096)
    timing = TimingJournal._read_existing(
        timing_root,
        identity=closure.timing_identity_sha256,
        boot_id=first["boot_id"],
        expected_policy=_policy(plan),
    )
    expected = dict(
        execution_sha256=pin,
        opaque_case_id=case["opaque_case_id"],
        dataset=case["dataset"],
        status=_PENDING,
        semantic_review_pending=True,
        closure_sha256=closure.result_sha256,
        timing_identity_sha256=timing.identity,
        worker_timing_head=result.get("worker_timing_head"),
        worker_exit_code=0,
        source_score_replay_pending=False,
    )
    worker_receipt = dict(
        execution_sha256=pin,
        closure_sha256=closure.result_sha256,
        worker_exit_code=0,
        timing_finished=True,
        timing_identity_sha256=timing.identity,
        worker_timing_head=result.get("worker_timing_head"),
    )
    if (
        canonical_json_bytes(result) != canonical_json_bytes(expected)
        or canonical_json_bytes(read_json(state / "worker-receipt.json"))
        != canonical_json_bytes(worker_receipt)
        or not any(
            read_json(p, limit=4096)["sha256"] == result.get("worker_timing_head")
            for p in timing_root.iterdir()
        )
    ):
        raise ValueError("retained completion terminal or worker receipt differs")
    return result


def _finalized_summary(
    plan: dict[str, Any], pin: str, summary: dict[str, Any], timing: TimingJournal
) -> dict[str, Any]:
    """Read-only final gate. Pending terminals alone never authorize success."""
    out = Path(plan["output_root"])
    state = out / plan["cases"][0]["opaque_case_id"]
    last = read_json(timing.root / f"event-{timing.sequence - 1:08d}.json", limit=4096)
    prepared = plan["schema_version"] == PREPARED_SCHEMA
    previous = last["previous"]
    if prepared:
        boundary = read_json(timing.root / f"event-{timing.sequence - 2:08d}.json", limit=4096)
        previous = boundary["previous"]
    expected = dict(
        execution_sha256=pin,
        case_terminal_sha256=file_sha256(state / "terminal.json"),
        round_terminal_sha256=file_sha256(out / "terminal.json"),
        worker_receipt_sha256=file_sha256(state / "worker-receipt.json"),
        timing_identity_sha256=timing.identity,
        before_finalization_head=previous,
    )
    if (
        read_json(out / "parent-finalization.json") != expected
        or (last["action"], last["account"], last["reason"])
        != ("checkpoint", "method", "orchestration")
        or last["sha256"] != timing.previous
    ):
        raise ValueError("parent publication finalization or final clock is incomplete")
    result = _completed_summary(summary, timing)
    if prepared:
        table, table_pin = _require_module_table(plan, pin, state, timing, last["previous"])
        result = _diagnostic_summary(result, timing, table, table_pin)
    return result


def _require_module_table(
    plan: dict[str, Any],
    pin: str,
    state: Path,
    timing: TimingJournal,
    checkpoint: str,
) -> tuple[dict[str, Any], str]:
    out = Path(plan["output_root"])
    table = read_json(state / "module-timing.json")
    fresh = summarize_module_timing(
        event_paths=tuple(sorted((state / "module-spans").glob("*.jsonl"))),
        timing_root=timing.root,
        expected_timing_identity=timing.identity,
        expected_policy=_policy(plan),
        checkpoint_sha256=checkpoint,
    )
    receipt = dict(
        execution_sha256=pin,
        table_sha256=file_sha256(state / "module-timing.json"),
        parent_finalization_sha256=file_sha256(out / "parent-finalization.json"),
        timing_checkpoint_sha256=checkpoint,
    )
    if (
        canonical_json_bytes(table) != canonical_json_bytes(fresh)
        or not table["complete"]
        or read_json(state / "module-timing-receipt.json") != receipt
    ):
        raise ValueError("module timing or parent finalization receipt differs")
    from nurse_rcl.v2.live.repair_query_session import _read_journal

    paid = sum(entry.phase == "completed" for entry in _read_journal(state / "query-ledger"))
    actual_attempts = max(read_json(p)["attempt_count"] for p in state.glob("ark-receipt-*.json"))
    if table["paid_measurement_calls"] != paid or table["ark_attempt_calls"] != actual_attempts:
        raise ValueError("module timing measurement/ARK counts differ from paid ledgers")

    return table, receipt["table_sha256"]


def _diagnostic_summary(
    result: dict[str, Any],
    timing: TimingJournal,
    table: dict[str, Any],
    table_pin: str,
) -> dict[str, Any]:
    residual = {
        account: timing.clock.used_seconds[account] - table["timing"][account + "_seconds"]
        for account in ("method", "preparation")
    }
    if any(value < 0 for value in residual.values()):
        raise ValueError("negative finalization residual")
    result["cases"][0].update(
        module_timing_sha256=table_pin, finalization_residual_seconds=residual
    )
    return result


def _completed_summary(summary: dict[str, Any], timing: TimingJournal) -> dict[str, Any]:
    row = dict(summary["cases"][0])
    row.update(
        status="live_closure_completed",
        final_timing_head=timing.previous,
        timing=timing.clock.summary(),
    )
    return {**summary, "status": "live_closure_completed", "completed": 1, "cases": [row]}


def run_round(
    plan_path: Path, plan_sha256: str, *, progress: Callable[[dict[str, Any]], None]
) -> dict[str, Any]:
    import fcntl

    plan = load_execution(plan_path, plan_sha256)
    out = require_plain_path(Path(plan["output_root"]))
    fd = os.open(out / "round.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    module_session, final_spans = ExitStack(), ExitStack()
    recorder = None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish_json(out / "identity.json", dict(execution_sha256=plan_sha256))
        if (out / "terminal.json").exists():
            summary = read_json(out / "terminal.json")
            retained = _retained_case_result(
                plan, plan_sha256, out / plan["cases"][0]["opaque_case_id"]
            )
            failed = int(retained["status"] == "failed")
            expected = dict(
                execution_sha256=plan_sha256,
                total=1,
                failed=failed,
                completed=0,
                status="completed_with_failures" if failed else _PENDING,
                gate_passed=False,
                cases=[retained],
            )
            if canonical_json_bytes(summary) != canonical_json_bytes(expected):
                raise ValueError("retained completion summary differs from the verified terminal")
            if failed:
                return summary
            timing_root = out / plan["cases"][0]["opaque_case_id"] / "timing"
            first = read_json(timing_root / "event-00000000.json", limit=4096)
            restored_clock = TimingJournal._read_existing(
                timing_root,
                identity=retained["timing_identity_sha256"],
                boot_id=first["boot_id"],
                expected_policy=_policy(plan),
            )
            return _finalized_summary(plan, plan_sha256, summary, restored_clock)
        case = plan["cases"][0]
        state = out / case["opaque_case_id"]
        state.mkdir(mode=0o700, exist_ok=True)
        require_plain_path(state)
        identity = sha256_hex(canonical_json_bytes([plan_sha256, case]))
        result: dict[str, Any] = dict(
            execution_sha256=plan_sha256,
            opaque_case_id=case["opaque_case_id"],
            dataset=case["dataset"],
            status="failed",
            semantic_review_pending=True,
        )
        timing = None
        if (state / "terminal.json").exists():
            result = _retained_case_result(plan, plan_sha256, state)
            if result["status"] != "failed":
                raise ValueError("incomplete parent publication cannot be promoted on restart")
        else:
            try:
                timing = TimingJournal.open(
                    state / "timing",
                    identity=identity,
                    now=time.monotonic(),
                    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    expected_policy=_policy(plan),
                )
                if plan["schema_version"] == PREPARED_SCHEMA:
                    recorder = module_session.enter_context(
                        _module_timing_scope(
                            state / "module-spans",
                            identity=identity,
                            boot_id=timing.boot_id,
                        )
                    )
                progress(
                    dict(
                        event="case_stage",
                        case_index=1,
                        total_cases=1,
                        stage="case",
                        opaque_case_id=case["opaque_case_id"],
                        **timing.clock.summary(),
                    )
                )
                worker_ending: list[tuple[int, str, bool]] = []

                def supervised_event(row: dict[str, Any]) -> None:
                    # Only the supervisor calls this; stdout is never parsed.
                    if row.get("event") == "stage_process_exited":
                        exit_code, reason = row.get("exit_code"), row.get("reason")
                        if type(exit_code) is not int or type(reason) is not str:
                            raise ValueError("case supervisor completion metadata differs")
                        worker_ending.append(
                            (
                                exit_code,
                                reason,
                                row.get("timing_finished") is True,
                            )
                        )
                    if recorder is not None:
                        recorder.observe()
                    progress(dict(row))

                command = [
                    sys.executable,
                    "-m",
                    "nurse_rcl.v2.live.repair_case_execution",
                    "--plan",
                    str(plan_path),
                    "--plan-sha256",
                    plan_sha256,
                    "--case-index",
                    "0",
                ]
                with (
                    _module_span("worker_supervision", role="orchestration")
                    if recorder is not None
                    else nullcontext()
                ):
                    if recorder is not None:
                        command.extend(["--module-parent", str(recorder.current_span_id)])
                    code = supervise_phases(
                        command,
                        log_path=state / "case.log",
                        timing=timing,
                        progress=supervised_event,
                    )
                if recorder is not None:
                    final_spans.enter_context(_module_span("parent_finalize", role="verification"))
                if code != 0 or worker_ending != [(0, "completed", True)]:
                    raise ValueError("case worker or timing did not finish normally")
                if load_execution(plan_path, plan_sha256) != plan:
                    raise ValueError("case plan changed during worker")
                closure = require_closure(plan, plan_sha256, state)
                timing.checkpoint(now=time.monotonic())
                worker_receipt = dict(
                    execution_sha256=plan_sha256,
                    closure_sha256=closure.result_sha256,
                    worker_exit_code=0,
                    timing_finished=True,
                    timing_identity_sha256=identity,
                    worker_timing_head=timing.previous,
                )
                publish_json(state / "worker-receipt.json", worker_receipt)
                if canonical_json_bytes(
                    read_json(state / "worker-receipt.json")
                ) != canonical_json_bytes(worker_receipt):
                    raise ValueError("parent worker receipt publication differs")
                result.update(
                    status=_PENDING,
                    closure_sha256=closure.result_sha256,
                    timing_identity_sha256=identity,
                    worker_timing_head=timing.previous,
                    worker_exit_code=0,
                    source_score_replay_pending=False,
                )
            except (ValueError, OSError, KeyError, CaseBudgetExpired) as error:
                result.update(
                    status="failed",
                    reason="case_owner_or_timing_incomplete",
                    error_type=type(error).__name__,
                )
            if timing is not None and result["status"] == "failed":
                result["timing"] = timing.clock.summary()
            publish_json(state / "terminal.json", result)
            if canonical_json_bytes(read_json(state / "terminal.json")) != canonical_json_bytes(
                result
            ):
                raise ValueError("parent case terminal publication differs")
        failed = int(result["status"] == "failed")
        summary = dict(
            execution_sha256=plan_sha256,
            total=1,
            failed=failed,
            completed=0,
            status="completed_with_failures" if failed else _PENDING,
            gate_passed=False,
            cases=[result],
        )
        publish_json(out / "terminal.json", summary)
        if canonical_json_bytes(read_json(out / "terminal.json")) != canonical_json_bytes(summary):
            raise ValueError("parent round terminal publication differs")
        progress(
            dict(
                event="round_terminal",
                total=1,
                failed=failed,
                completed=0,
                status=summary["status"],
            )
        )
        if not failed:
            if timing is None:
                raise ValueError("parent publication lost its active timing owner")
            timing.checkpoint(now=time.monotonic())
            finalization = dict(
                execution_sha256=plan_sha256,
                case_terminal_sha256=file_sha256(state / "terminal.json"),
                round_terminal_sha256=file_sha256(out / "terminal.json"),
                worker_receipt_sha256=file_sha256(state / "worker-receipt.json"),
                timing_identity_sha256=timing.identity,
                before_finalization_head=timing.previous,
            )
            publish_json(out / "parent-finalization.json", finalization)
            if read_json(out / "parent-finalization.json") != finalization:
                raise ValueError("parent finalization publication differs")
            final_spans.close()
            # Includes all parent receipts, terminal publications and progress.
            # The final marker is invalid without this succeeding checkpoint.
            timing.checkpoint(now=time.monotonic())
            if recorder is not None:
                table = summarize_module_timing(
                    event_paths=tuple(sorted((state / "module-spans").glob("*.jsonl"))),
                    timing_root=timing.root,
                    expected_timing_identity=identity,
                    expected_policy=_policy(plan),
                )
                if not table["complete"]:
                    raise ValueError("module timing incomplete after finished worker")
                publish_json(state / "module-timing.json", table)
                publish_json(
                    state / "module-timing-receipt.json",
                    dict(
                        execution_sha256=plan_sha256,
                        table_sha256=file_sha256(state / "module-timing.json"),
                        parent_finalization_sha256=file_sha256(out / "parent-finalization.json"),
                        timing_checkpoint_sha256=timing.previous,
                    ),
                )
                # All file/ledger diagnostics finish before the final clock.
                table, table_pin = _require_module_table(
                    plan, plan_sha256, state, timing, timing.previous
                )
                timing.checkpoint(now=time.monotonic())
                summary = _diagnostic_summary(
                    _completed_summary(summary, timing), timing, table, table_pin
                )
            else:
                summary = _finalized_summary(plan, plan_sha256, summary, timing)
        elif recorder is not None:
            final_spans.__exit__(ValueError, ValueError("case did not close"), None)
        return summary
    finally:
        with ExitStack() as cleanup:
            cleanup.callback(os.close, fd)
            cleanup.callback(module_session.__exit__, *sys.exc_info())
            cleanup.callback(final_spans.__exit__, *sys.exc_info())


def main() -> int:
    import argparse
    import json
    import resource

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--case-index", type=int, choices=(0,))
    parser.add_argument("--timing-fd", type=int)
    parser.add_argument("--module-parent")
    args = parser.parse_args()
    legacy.install_termination_handlers()

    def emit(row: dict[str, Any]) -> None:
        print(json.dumps(row), flush=True)

    try:
        worker = args.case_index is not None
        if worker != (args.timing_fd is not None):
            raise ValueError("case worker needs its private timing channel")
        plan = load_execution(args.plan, args.plan_sha256)
        prepared = plan["schema_version"] == PREPARED_SCHEMA
        if (prepared and worker) != (args.module_parent is not None):
            raise ValueError("prepared worker needs its own parent module span")
        if not Path(__file__).resolve().is_relative_to(require_plain_path(Path(plan["code_root"]))):
            raise ValueError("case worker implementation is outside the pinned code root")
        resource.setrlimit(resource.RLIMIT_AS, (8589934592, 8589934592))
        if not worker:
            result = run_round(args.plan, args.plan_sha256, progress=emit)
            emit(dict(event="round_verified", **result))
            return int(result["failed"] != 0)
        from nurse_rcl.v2.live.repair_case_owner import (
            run_repair_case_from_preparation,
            run_repair_case_from_raw,
        )

        client = PhaseClient(args.timing_fd)
        with ExitStack() as worker_scope:
            if prepared:
                case = plan["cases"][0]
                worker_scope.enter_context(
                    _module_timing_scope(
                        Path(plan["output_root"]) / case["opaque_case_id"] / "module-spans",
                        identity=sha256_hex(canonical_json_bytes([args.plan_sha256, case])),
                        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                        parent_id=args.module_parent,
                    )
                )
                worker_scope.enter_context(_module_span("case_worker"))
            worker_scope.enter_context(_use_phase_client(client))
            worker_scope.enter_context(_phase("method", "orchestration"))
            (run_repair_case_from_preparation if prepared else run_repair_case_from_raw)(
                plan_path=args.plan,
                expected_plan_sha256=args.plan_sha256,
                case_index=args.case_index,
            )
        client.finish()
        return 0
    except KeyboardInterrupt:
        emit(dict(event="execution_interrupted"))
        return 130
    except Exception as error:
        emit(dict(event="execution_failed", error_type=type(error).__name__))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
