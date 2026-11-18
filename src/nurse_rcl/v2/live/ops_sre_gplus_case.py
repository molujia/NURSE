"""Explicit one-case composition: fresh deterministic core, bounded stage, G+.

The opt-in history composition retrieves reviewed precedents after freezing the core.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import resource
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.live.fixed_history_similarity import (
    DEFAULT_HISTORY_METHOD,
    RULE_DISTANCE,
    RULE_ID,
    HistoryMethod,
)
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_case import inspect_ops_case
from nurse_rcl.v2.live.ops_contracts import OpsSreFeedbackFactProfile, OpsSreFocusedProfile
from nurse_rcl.v2.live.ops_execution import run_round
from nurse_rcl.v2.live.ops_plan import (
    OpsAdmittedPlan,
    OpsHistoricalStagePlan,
    OpsSinglePlan,
    load_ops_plan,
    plain_location,
)
from nurse_rcl.v2.live.ops_result import OpsCoreResult
from nurse_rcl.v2.live.ops_sre_directed_provider import stage_grant
from nurse_rcl.v2.live.ops_sre_directed_stage import run_stage_round
from nurse_rcl.v2.live.ops_sre_gplus_contract import GPLUS_PROTOCOL
from nurse_rcl.v2.live.ops_sre_gplus_provider import GplusDelivery, GplusReportProvider
from nurse_rcl.v2.live.ops_sre_gplus_run import run_gplus_report
from nurse_rcl.v2.live.ops_sre_historical_cases import HistoricalContext
from nurse_rcl.v2.live.ops_sre_stage_handoff import (
    retained_sources as _retained_sources,  # noqa: F401 -- compatibility re-export.
)
from nurse_rcl.v2.live.ops_sre_stage_handoff import (
    stage_report_context,
)
from nurse_rcl.v2.live.semantic_history_retrieval import EmbeddingModel, requested_embedding_model
from nurse_rcl.v2.live.staged_case import require_plain_path
from nurse_rcl.v2.live.verified_history import AMPLITUDE_QUALITY_PROFILE, LEGACY_PROFILE


class GplusCompositionPlan(StrictFrozenModel):
    schema_version: Literal["nurse.v2.gplus-one-case-composition/1"] = (
        "nurse.v2.gplus-one-case-composition/1"
    )
    core_plan_path: str
    core_plan_sha256: SHA256
    run_root: str
    investigation_decision_limit: int = Field(ge=1, le=8, strict=True)
    report_protocol: Literal["nurse.v2.sre-gplus-report-exchange/1"] = GPLUS_PROTOCOL


class GplusFocusedCompositionPlan(GplusCompositionPlan):
    schema_version: Literal["nurse.v2.gplus-one-case-composition/2"] = (
        "nurse.v2.gplus-one-case-composition/2"  # type: ignore[assignment]
    )
    investigation_profile: Literal["v2-ops-sre/15"] = "v2-ops-sre/15"


class GplusHistoricalCompositionPlan(GplusFocusedCompositionPlan):
    schema_version: Literal["nurse.v2.gplus-one-case-composition/3"] = (
        "nurse.v2.gplus-one-case-composition/3"  # type: ignore[assignment]
    )
    historical_library_path: str
    historical_library_sha256: SHA256
    historical_query_group_id: str = Field(pattern=r"^[a-zA-Z0-9_.-]{1,128}$", strict=True)
    history_embedding_model: Literal[
        "doubao-embedding-vision-251215", "doubao-embedding-text-240715"
    ] = "doubao-embedding-vision-251215"
    history_top_k: int = Field(default=5, ge=1, strict=True)


class GplusHistoryMethodPlan(GplusHistoricalCompositionPlan):
    schema_version: Literal["nurse.v2.gplus-one-case-composition/4"] = (
        "nurse.v2.gplus-one-case-composition/4"  # type: ignore[assignment]
    )
    history_method: HistoryMethod = DEFAULT_HISTORY_METHOD


class GplusAmplitudeQualityPlan(GplusHistoricalCompositionPlan):
    schema_version: Literal["nurse.v2.gplus-one-case-composition/5"] = (
        "nurse.v2.gplus-one-case-composition/5"  # type: ignore[assignment]
    )
    # Canonical plans include either the resolved model ID or null. Re-reading
    # a pinned plan never depends on the inspecting process's environment.
    history_embedding_model: EmbeddingModel | None = Field(  # type: ignore[assignment]
        default_factory=lambda: requested_embedding_model()
    )


def _history_method(plan: GplusHistoricalCompositionPlan) -> HistoryMethod:
    if isinstance(plan, GplusAmplitudeQualityPlan):
        return "embedding" if plan.history_embedding_model else "fixed-rule"
    # Version 3 pinned a specific Ark model and always used Euclidean distance.
    return plan.history_method if isinstance(plan, GplusHistoryMethodPlan) else "embedding"


def check_history_receipt(plan: GplusHistoricalCompositionPlan, retrieval: dict[str, Any]) -> None:
    method = _history_method(plan)
    expected_model = plan.history_embedding_model if method == "embedding" else None
    if (retrieval["library_sha256"] != plan.historical_library_sha256
            or retrieval["model"] != expected_model):
        raise ValueError("historical retrieval model or library differs from its plan")
    if isinstance(plan, GplusHistoryMethodPlan | GplusAmplitudeQualityPlan) and (
        retrieval.get("method") != method
        or retrieval.get("metric") != ("euclidean" if method == "embedding" else RULE_DISTANCE)
        or retrieval.get("rule_id") != (RULE_ID if method == "fixed-rule" else None)
    ):
        raise ValueError("historical retrieval method differs from its plan")
    if (isinstance(plan, GplusAmplitudeQualityPlan)
            and retrieval.get("evidence_profile") != AMPLITUDE_QUALITY_PROFILE):
        raise ValueError("historical evidence profile differs from its plan")


def composition_sha256(plan: GplusCompositionPlan) -> str:
    return sha256_hex(canonical_json_bytes(plan.model_dump(mode="json")))


def composition_grant(plan: GplusCompositionPlan) -> dict[str, Any]:
    count = plan.investigation_decision_limit
    grant = dict(
        schema_version="nurse.v2.gplus-composition-grant/2"
        if isinstance(plan, GplusFocusedCompositionPlan)
        else "nurse.v2.gplus-composition-grant/1",
        core_plan_sha256=plan.core_plan_sha256,
        investigation_decisions=count,
        investigation_actions=count,
        strict_duplicate_correction_limit=1,
        report_logical_limit=1,
        ordinary_logical_limit=count + 1,
        logical_limit_with_correction=count + 2,
        attempt_limit_per_logical=5,
        attempt_limit_with_correction=5 * (count + 2),
        legal_early_stop=True,
        last_receipt="report_visible_not_a_nonexistent_successor_decision",
    )
    if isinstance(plan, GplusFocusedCompositionPlan):
        grant["investigation_profile"] = plan.investigation_profile
    if isinstance(plan, GplusHistoricalCompositionPlan):
        grant.update(
            schema_version="nurse.v2.gplus-composition-grant/3",
            historical_library_sha256=plan.historical_library_sha256,
            historical_query_group_id=plan.historical_query_group_id,
            history_embedding_model=plan.history_embedding_model,
            history_top_k=plan.history_top_k,
        )
    if isinstance(plan, GplusHistoryMethodPlan):
        grant.update(schema_version="nurse.v2.gplus-composition-grant/4",
                     history_method=plan.history_method)
    if isinstance(plan, GplusAmplitudeQualityPlan):
        grant.update(schema_version="nurse.v2.gplus-composition-grant/5",
                     history_method=_history_method(plan),
                     evidence_profile=AMPLITUDE_QUALITY_PROFILE)
    return grant


def read_composition(path: Path, pin: str) -> GplusCompositionPlan:
    raw = _read_bytes(path, 65536)
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError("G+ composition plan shape differs")
    model = {
        "nurse.v2.gplus-one-case-composition/2": GplusFocusedCompositionPlan,
        "nurse.v2.gplus-one-case-composition/3": GplusHistoricalCompositionPlan,
        "nurse.v2.gplus-one-case-composition/4": GplusHistoryMethodPlan,
        "nurse.v2.gplus-one-case-composition/5": GplusAmplitudeQualityPlan,
    }.get(value.get("schema_version"), GplusCompositionPlan)
    plan = model.model_validate_json(raw)
    if sha256_hex(raw) != pin or raw != canonical_json_bytes(plan.model_dump(mode="json")):
        raise ValueError("G+ composition external plan pin differs")
    return plan


def admit_composition(plan: GplusCompositionPlan) -> OpsAdmittedPlan:
    root = plain_location(Path(plan.run_root))
    core = load_ops_plan(Path(plan.core_plan_path), plan.core_plan_sha256)
    if (
        core.plan.mode != "core-only"
        or core.plan.source_core is not None
        or core.plan.profile.profile_id != "v2-ops-isolated/2"
        or plain_location(Path(core.plan.output_root)) != root / "core-results"
    ):
        raise ValueError("G+ composition requires a fresh isolated deterministic core plan")
    if any(
        root.is_relative_to(path) or path.is_relative_to(root)
        for path in (
            Path(core.plan.code_root),
            Path(core.plan.preparation.archive_root),
            Path(core.plan.case.input_path).parent,
        )
    ):
        raise ValueError("G+ composition overlaps protected code/preparation/input")
    return core


def _derive_stage(
    plan: GplusCompositionPlan, core: OpsAdmittedPlan, history: HistoricalContext | None = None
) -> tuple[Path, str]:
    """Derive the same-case child only after the new deterministic core is sealed."""
    result = inspect_ops_case(core)
    if result["channels"] != dict(core="completed", investigation="disabled", report="disabled"):
        raise ValueError("G+ fresh core has unexpected model channels")
    prep = core.plan.preparation.model_dump(mode="json")
    receipt_raw = _read_bytes(Path(prep["receipt_path"]), 16384)
    receipt = json.loads(receipt_raw)
    prep.update(
        mode="load",
        manifest_sha256=receipt["manifest_sha256"],
        receipt_sha256=sha256_hex(receipt_raw),
    )
    value = core.plan.model_dump(mode="json")
    output = Path(plan.run_root) / "results"
    output.mkdir(mode=0o700, exist_ok=False)
    value.update(
        profile=(
            OpsSreFocusedProfile()
            if isinstance(plan, GplusFocusedCompositionPlan)
            else OpsSreFeedbackFactProfile()
        ).model_dump(mode="json"),
        mode="with-agent",
        output_root=str(output),
        preparation=prep,
        parent_plan_path=plan.core_plan_path,
        parent_plan_sha256=plan.core_plan_sha256,
        source_core=dict(
            plan_path=plan.core_plan_path,
            plan_sha256=plan.core_plan_sha256,
            case_result_sha256=sha256_hex((core.state_root / "case-result.json").read_bytes()),
        ),
    )
    if isinstance(plan, GplusHistoricalCompositionPlan) != (history is not None):
        raise ValueError("child historical context differs from its composition")
    if history is None:
        child = OpsSinglePlan.model_validate_json(canonical_json_bytes(value))
    else:
        value.update(
            schema_version="nurse.v2.ops-single-plan/2",
            historical_precedents=history.model_dump(mode="json"),
        )
        child = OpsHistoricalStagePlan.model_validate_json(canonical_json_bytes(value))
    path = Path(plan.run_root) / "investigation-plan.json"
    raw = canonical_json_bytes(child.model_dump(mode="json"))
    if path.exists():
        raise ValueError("G+ child plan already exists; no new budget on restart")
    atomic_write_bytes(path, raw)
    load_ops_plan(path, sha256_hex(raw))  # Real same-case/source/code metadata admission.
    return path, sha256_hex(raw)


def run_gplus_case(
    path: Path,
    pin: str,
    *,
    approved_composition_sha256: str | None,
    progress: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    if approved_composition_sha256 != pin:
        raise ValueError("G+ composition requires explicit approval")
    plan = read_composition(path, pin)
    core = admit_composition(plan)
    root = require_plain_path(Path(plan.run_root))
    result_path = root / "composition-result.json"
    if result_path.exists():
        raise ValueError("G+ composition already finished; use status, never launch again")
    descriptor = os.open(root / "composition.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "composition-started.json").exists() or core.state_root.exists():
            raise ValueError("G+ composition already started; no blind restart or resend")
        atomic_write_bytes(
            root / "composition-started.json",
            canonical_json_bytes(
                dict(
                    composition_sha256=pin,
                    grant=composition_grant(plan),
                    opaque_case_id=core.plan.case.opaque_case_id,
                    code_sha256=core.plan.code_sha256,
                )
            ),
        )
        started = monotonic()
        result: dict[str, Any] = dict(
            schema_version="nurse.v2.gplus-one-case-result/1",
            composition_sha256=pin,
            status="failed",
            channels=dict(core="pending", investigation="disabled", report="disabled"),
            failure_type=None,
            stage_plan_sha256=None,
            report_execution_sha256=pin,
        )
        try:
            core_round = run_round(
                Path(plan.core_plan_path), plan.core_plan_sha256, progress=progress
            )
            if core_round["channels"]["core"] != "completed":
                raise ValueError("G+ new deterministic core failed")
            core_result = OpsCoreResult.model_validate_json(
                _read_bytes(core.state_root / "core/result.json", 256 * 1024**2)
            )
            result["channels"]["core"] = "completed"
            result["core_result_digest"] = core_result.core_result_digest
            progress(
                dict(
                    event="gplus_core_completed",
                    core_result_digest=core_result.core_result_digest,
                    core_result_path=str(core.state_root / "core/result.json"),
                )
            )
            history = None
            if isinstance(plan, GplusHistoricalCompositionPlan):
                from nurse_rcl.v2.live.verified_history import retrieve_verified_history

                snapshot = core_result.evidence.graph.graph_pair.fault_snapshot
                retrieval = retrieve_verified_history(
                    query_core=core_result.model_dump(mode="json"),
                    library_path=plain_location(Path(plan.historical_library_path)),
                    query_id=core_result.identity.opaque_case_id,
                    query_group_id=plan.historical_query_group_id,
                    decision_time=snapshot.decision_time_unix_ns,
                    top_k=plan.history_top_k,
                    expected_library_sha256=plan.historical_library_sha256,
                    embedding_model=plan.history_embedding_model,
                    method=_history_method(plan),
                    profile=(AMPLITUDE_QUALITY_PROFILE
                             if isinstance(plan, GplusAmplitudeQualityPlan) else LEGACY_PROFILE),
                )
                check_history_receipt(plan, retrieval)
                history = HistoricalContext.from_retrieval(
                    retrieval, query_id=core_result.identity.opaque_case_id,
                    core_result_digest=core_result.core_result_digest,
                )
                retrieval_bytes = canonical_json_bytes(retrieval)
                atomic_write_bytes(root / "history-retrieval.json", retrieval_bytes)
                result["history_retrieval_sha256"] = sha256_hex(retrieval_bytes)
            stage_path, stage_pin = _derive_stage(plan, core, history)
            grant = stage_grant(stage_pin, plan.investigation_decision_limit)
            grant_pin = sha256_hex(canonical_json_bytes(grant))
            atomic_write_bytes(root / "investigation-grant.json", canonical_json_bytes(grant))
            result.update(stage_plan_sha256=stage_pin, stage_grant_sha256=grant_pin)
            stage_round = run_stage_round(
                stage_path,
                stage_pin,
                stage_decision_limit=plan.investigation_decision_limit,
                approved_plan_sha256=stage_pin,
                approved_stage_sha256=grant_pin,
                progress=progress,
            )
            result["channels"]["investigation"] = stage_round["channels"]["investigation"]
            result.update(
                investigation_decision_count=stage_round.get("decision_count"),
                investigation_action_count=stage_round.get("action_count"),
            )
            if stage_round["status"] not in {"stage_paused", "stage_terminal"}:
                raise ValueError("G+ actual investigation stage failed")
            stage = load_ops_plan(stage_path, stage_pin)
            context, refs = stage_report_context(
                stage,
                grant_sha256=grant_pin,
                worker_invocation_id=stage_round["worker_invocation_id"],
            )
            report_root = stage.state_root / "gplus-report"

            def complete(wire: bytes) -> GplusDelivery:
                return GplusReportProvider(
                    root=report_root / "provider",
                    wire=wire,
                    references=refs,
                    code_sha256=stage.plan.code_sha256,
                    execution_sha256=pin,
                    approved_execution_sha256=pin,
                    expected_context_sha256=context.context_sha256,
                ).report(wire)

            report = run_gplus_report(
                root=report_root,
                context=context,
                references=refs,
                code_sha256=stage.plan.code_sha256,
                execution_sha256=pin,
                approved_execution_sha256=pin,
                complete=complete,
                historical_precedents=history,
            )
            result["channels"]["report"] = report["status"]
            result["report_context_sha256"] = context.context_sha256
            result["report_state_sha256"] = report["state_sha256"]
            result["status"] = (
                "completed" if report["status"] == "completed" else "completed_partial_report"
            )
        except Exception as exc:
            result.update(status="failed", failure_type=type(exc).__name__)
        result["runner_elapsed_seconds"] = monotonic() - started
        result["result_sha256"] = sha256_hex(canonical_json_bytes(result))
        atomic_write_bytes(result_path, canonical_json_bytes(result))
        progress(dict(event="gplus_composition_terminal", **result))
        return result
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--approved-composition-sha256")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_STACK, (64 * 1024**2, 64 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
    if args.status:
        from nurse_rcl.v2.live.ops_sre_gplus_status import inspect_gplus_case

        print(json.dumps(inspect_gplus_case(args.plan, args.plan_sha256)))
        return 0
    if args.check_only:
        plan = read_composition(args.plan, args.plan_sha256)
        core = admit_composition(plan)
        print(
            json.dumps(
                dict(
                    status="admitted",
                    grant=composition_grant(plan),
                    code_sha256=core.plan.code_sha256,
                    new_ark_calls=0,
                    preparation_load_import_dispatch=0,
                )
            )
        )
        return 0
    result = run_gplus_case(
        args.plan,
        args.plan_sha256,
        approved_composition_sha256=args.approved_composition_sha256,
        progress=lambda row: print(json.dumps(row, ensure_ascii=False), flush=True),
    )
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
