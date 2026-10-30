"""Keyless one-case status: original saved owners only, no resume or mutation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_case import inspect_saved_case, plan_identity
from nurse_rcl.v2.live.ops_plan import read_ops_plan
from nurse_rcl.v2.live.ops_sre_directed_stage import _inspect_stage
from nurse_rcl.v2.live.ops_sre_gplus_case import GplusFocusedCompositionPlan, read_composition
from nurse_rcl.v2.live.ops_sre_gplus_run import inspect_gplus_report


def inspect_gplus_case(path: Path, pin: str) -> dict[str, Any]:
    plan = read_composition(path, pin)
    root = Path(plan.run_root)
    result_path = root / "composition-result.json"
    if not result_path.exists():
        return dict(
            status="started_without_terminal"
            if (root / "composition-started.json").exists()
            else "not_started",
            composition_sha256=pin,
            new_ark_calls=0,
            preparation_load_import_dispatch=0,
            liveness="not_inferred_from_missing_terminal",
        )
    raw = _read_bytes(result_path, 4 * 1024**2)
    result = json.loads(raw)
    unsigned = {key: value for key, value in result.items() if key != "result_sha256"}
    if (
        raw != canonical_json_bytes(result)
        or result.get("composition_sha256") != pin
        or result.get("result_sha256") != sha256_hex(canonical_json_bytes(unsigned))
    ):
        raise ValueError("G+ composition saved terminal identity differs")
    focused_success = type(plan) is GplusFocusedCompositionPlan and result.get("status") in {
        "completed",
        "completed_partial_report",
    }
    if focused_success:
        for name in (
            "stage_plan_sha256",
            "stage_grant_sha256",
            "report_context_sha256",
            "report_state_sha256",
        ):
            require_pin(result.get(name))
        if (
            result["channels"]["core"] != "completed"
            or not (root / "investigation-plan.json").is_file()
        ):
            raise ValueError("successful focused composition original child is absent")
    core = read_ops_plan(Path(plan.core_plan_path), plan.core_plan_sha256)
    core_root = Path(core.output_root) / core.case.opaque_case_id
    if result["channels"]["core"] == "completed":
        retained = inspect_saved_case(core_root, plan_identity(core, plan.core_plan_sha256))
        if retained["core_result_digest"] != result["core_result_digest"]:
            raise ValueError("G+ saved deterministic core differs")
    stage_pin = result.get("stage_plan_sha256")
    if stage_pin is not None and (root / "investigation-plan.json").exists():
        stage = read_ops_plan(root / "investigation-plan.json", stage_pin)
        if stage.profile.profile_id != getattr(plan, "investigation_profile", "v2-ops-sre/14"):
            raise ValueError("G+ saved child investigation profile differs from composition")
        if (
            stage.case != core.case
            or stage.code_files != core.code_files
            or stage.source_core is None
            or stage.source_core.plan_sha256 != plan.core_plan_sha256
            or Path(stage.output_root) != root / "results"
        ):
            raise ValueError("G+ saved child source/case differs")
        stage_root = Path(stage.output_root) / stage.case.opaque_case_id
        if focused_success and any(
            not (stage_root / name).is_file() for name in ("stage-outcome.json", "stage-round.json")
        ):
            raise ValueError("successful focused composition original stage is absent")
        if (stage_root / "stage-outcome.json").exists():
            outcome = _inspect_stage(stage_root, plan_identity(stage, stage_pin))
            if (
                outcome["approved_grant"]["through_decision_index"]
                != plan.investigation_decision_limit
                or outcome["decision_count"] > plan.investigation_decision_limit
            ):
                raise ValueError("G+ saved child exceeds approved ceiling")
        if result.get("report_context_sha256") is not None:
            report = inspect_gplus_report(
                root=stage_root / "gplus-report",
                expected_context_sha256=result["report_context_sha256"],
                expected_code_sha256=core.code_sha256,
                expected_execution_sha256=pin,
            )
            if report["state_sha256"] != result["report_state_sha256"]:
                raise ValueError("G+ saved report owner differs")
    return dict(
        result, status_reader_new_ark_calls=0, status_reader_preparation_load_import_dispatch=0
    )
