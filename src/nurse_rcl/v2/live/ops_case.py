"""Single admitted case: one preparation, isolated workers, report-only recovery."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from time import monotonic
from typing import Any

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.v2.inference_leakage import (
    _reject_forbidden_report_provenance_values,
    audit_inference_boundary,
)
from nurse_rcl.v2.live.normal_artifacts import NormalPreparationReceipt
from nurse_rcl.v2.live.normal_preparation import (
    build_normal_preparation,
    load_normal_preparation,
)
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_agent_state import OpsAgentResult
from nurse_rcl.v2.live.ops_contracts import (
    OpsChannelStatus,
    OpsCoreIdentity,
    OpsEmpiricalProfile,
    OpsInvestigationEvidence,
    OpsSreBriefProfile,
    OpsSreComparableProfile,
    OpsSreConditionProfile,
    OpsSreFeedbackFactProfile,
    OpsSreNavigationProfile,
    OpsSreProfileBase,
    OpsSreQuestionProfile,
    OpsSreRecoveryProfile,
    OpsSreWorkbenchProfile,
    OpsSreWorkingProfile,
)
from nurse_rcl.v2.live.ops_core import run_ops_core
from nurse_rcl.v2.live.ops_indexed_tools import OpsEmpiricalLedgerState
from nurse_rcl.v2.live.ops_investigation_ledger import OpsLedgerState
from nurse_rcl.v2.live.ops_node_common import prepared_snapshot_id
from nurse_rcl.v2.live.ops_pair import start_ops_pair as start_ops_pair
from nurse_rcl.v2.live.ops_plan import OpsAdmittedPlan, OpsSinglePlan, require_ops_code
from nurse_rcl.v2.live.ops_provider import OpsLiveProvider, saved_ops_report, usage_summary
from nurse_rcl.v2.live.ops_report import ReportCompletion, run_ops_report
from nurse_rcl.v2.live.ops_report_context import OpsReportContext, build_report_context
from nurse_rcl.v2.live.ops_result import OpsCoreResult
from nurse_rcl.v2.live.ops_sre_brief_report import BriefSreReportContext
from nurse_rcl.v2.live.ops_sre_native_report import NativeSreReportContext
from nurse_rcl.v2.live.ops_sre_report_context import OpsSreReportContext
from nurse_rcl.v2.live.repair_driver import publish_json, read_json
from nurse_rcl.v2.live.repair_module_timing import _module_span
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData
from nurse_rcl.v2.live.repair_preparation_codec import member_digest
from nurse_rcl.v2.live.repair_safety import REPAIR_MAX_JSON_BYTES
from nurse_rcl.v2.live.repair_timing import _phase
from nurse_rcl.v2.live.staged_case import require_plain_path

_MEMBERS = (
    "core/result.json",
    "core/identity.json",
    "core/diagnostics.json",
    "core/diagnostics-status.json",
    "core/initial-nodes.json",
    "core/timing.json",
    "investigation/conversation.json",
    "investigation/state.json",
    "report/context.json",
    "report/result.json",
    "report/report.md",
    "report/report-evidence.md",
    "report/report-delivery.json",
    "report/public-history.json",
    "report/assembly.json",
    "investigation/result.json",
    "investigation/process.json",
    "formal-binding.json",
    "source-core.json",
)


def case_identity(admitted: OpsAdmittedPlan) -> dict[str, Any]:
    return plan_identity(admitted.plan, admitted.plan_sha256)


def plan_identity(plan: OpsSinglePlan, pin: str) -> dict[str, Any]:
    return dict(
        plan_sha256=pin,
        code_sha256=plan.code_sha256,
        opaque_case_id=plan.case.opaque_case_id,
        profile=plan.profile.profile_id,
        mode=plan.mode,
    )


def inspect_ops_case(admitted: OpsAdmittedPlan) -> dict[str, Any]:
    """Finite saved-member integrity only; never prepare, query, score or call ARK."""
    return inspect_saved_case(admitted.state_root, case_identity(admitted))


def inspect_saved_case(root: Path, identity: dict[str, Any]) -> dict[str, Any]:
    result = read_json(root / "case-result.json")
    if result.get("identity") != identity:
        raise ValueError("isolated case result identity differs")
    unsigned = {k: v for k, v in result.items() if k != "result_sha256"}
    files = result.get("files")
    if result.get("result_sha256") != sha256_hex(canonical_json_bytes(unsigned)) or (
        type(files) is not dict
        or not {"core/result.json", "core/identity.json"} <= files.keys()
        or not files.keys() <= set(_MEMBERS)
    ):
        raise ValueError("isolated case result integrity differs")
    for name, entry in files.items():
        if list(member_digest(root / name)) != entry:
            raise ValueError("isolated saved member integrity differs")
    OpsChannelStatus.model_validate(result["channels"])
    return result


def _preparation(admitted: OpsAdmittedPlan) -> RepairPreparationData:
    plan, pin = admitted.plan, admitted.plan_sha256
    prep = plan.preparation
    receipt_path = Path(prep.receipt_path)
    module = (
        "preparation_build"
        if prep.mode == "build" and not receipt_path.exists()
        else ("preparation_load")
    )
    with _phase("preparation", "raw_input_prepare"), _module_span(module):
        arguments: dict[str, Any] = dict(
            archive_root=Path(prep.archive_root),
            expected_key=prep.key,
            raw_arguments=admitted.raw_arguments,
            receipt_path=receipt_path,
        )
        if prep.mode == "load":
            assert prep.manifest_sha256 is not None and prep.receipt_sha256 is not None
            return load_normal_preparation(
                **arguments,
                expected_manifest_sha256=prep.manifest_sha256,
                expected_receipt_sha256=prep.receipt_sha256,
            )
        if receipt_path.exists():
            raw = _read_bytes(receipt_path, 16384)
            receipt = NormalPreparationReceipt.model_validate_json(raw)
            if (receipt.execution_sha256, receipt.execution_code_sha256) != (
                pin,
                plan.code_sha256,
            ):
                raise ValueError("isolated preparation build owner differs")
            return load_normal_preparation(
                **arguments,
                expected_manifest_sha256=receipt.manifest_sha256,
                expected_receipt_sha256=sha256_hex(raw),
            )
        data, _, _ = build_normal_preparation(
            **arguments, execution_sha256=pin, execution_code_sha256=plan.code_sha256
        )
        return data


def _admit_display(data: RepairPreparationData, admitted: OpsAdmittedPlan) -> None:
    legacy = data.diagnostics.legacy
    audit_inference_boundary(
        (legacy.entity_registry, legacy.candidate_universe),
        identity_manifest=admitted.identity_manifest,
        expected_identity_manifest_sha256=admitted.identity_manifest.manifest_sha256,
    )
    _reject_forbidden_report_provenance_values(
        (data.catalog.candidate_service_keys, tuple(s.metric_name for s in data.sources)),
        location="$ops.display_names",
    )


def _provider(admitted: OpsAdmittedPlan, snapshot: str, approved: str | None) -> OpsLiveProvider:
    return OpsLiveProvider(
        root=admitted.state_root / "provider",
        code_sha256=admitted.plan.code_sha256,
        execution_sha256=admitted.plan_sha256,
        approved_execution_sha256=approved,
        opaque_case_id=admitted.plan.case.opaque_case_id,
        case_snapshot_id=snapshot,
        empirical=isinstance(admitted.plan.profile, OpsEmpiricalProfile | OpsSreProfileBase),
        sre=isinstance(admitted.plan.profile, OpsSreProfileBase),
        working=isinstance(admitted.plan.profile, OpsSreWorkingProfile),
        sre_profile=(
            admitted.plan.profile.investigation_profile_id
            if isinstance(admitted.plan.profile, OpsSreProfileBase)
            else None
        ),
    )


def _core(root: Path, admitted: OpsAdmittedPlan) -> OpsCoreResult:
    return inspect_saved_core(root, admitted.plan)


def inspect_saved_core(root: Path, plan: OpsSinglePlan) -> OpsCoreResult:
    core = OpsCoreResult.model_validate_json(
        _read_bytes(root / "core/result.json", REPAIR_MAX_JSON_BYTES)
    )
    if (
        core.identity.dataset,
        core.identity.opaque_case_id,
        core.identity.configuration_sha256,
        core.identity.environment_sha256,
    ) != (
        plan.case.dataset,
        plan.case.opaque_case_id,
        plan.core_configuration_sha256,
        plan.environment_sha256,
    ):
        raise ValueError("isolated saved core identity differs")
    return core


def _formal_binding(
    admitted: OpsAdmittedPlan,
    context: OpsReportContext
    | OpsSreReportContext
    | NativeSreReportContext
    | BriefSreReportContext,
) -> None:
    root = admitted.state_root
    publish_json(
        root / "formal-binding.json",
        dict(
            identity=case_identity(admitted),
            core_result_digest=context.core_result_digest,
            core_member=list(member_digest(root / "core/result.json")),
            context_member=list(member_digest(root / "report/context.json")),
        ),
    )


def _finish(
    admitted: OpsAdmittedPlan,
    core: OpsCoreResult,
    channels: OpsChannelStatus,
    *,
    failure_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = admitted.state_root
    with _module_span("artifact_publish"):
        empirical = isinstance(admitted.plan.profile, OpsEmpiricalProfile | OpsSreProfileBase)
        sre = isinstance(admitted.plan.profile, OpsSreProfileBase)
        working = isinstance(admitted.plan.profile, OpsSreWorkingProfile)
        brief = isinstance(
            admitted.plan.profile,
            OpsSreBriefProfile
            | OpsSreConditionProfile
            | OpsSreRecoveryProfile
            | OpsSreComparableProfile
            | OpsSreNavigationProfile
            | OpsSreWorkbenchProfile,
        )
        if brief and channels.report in {"completed", "partial"}:
            from nurse_rcl.v2.live.ops_sre_brief_render import inspect_brief_report_pair

            inspect_brief_report_pair(
                root / "report",
                condition_explanations=isinstance(
                    admitted.plan.profile,
                    OpsSreConditionProfile
                    | OpsSreRecoveryProfile
                    | OpsSreComparableProfile
                    | OpsSreNavigationProfile
                    | OpsSreWorkbenchProfile,
                ),
                recovery=isinstance(admitted.plan.profile, OpsSreRecoveryProfile),
                comparable=isinstance(admitted.plan.profile, OpsSreComparableProfile),
                navigation=isinstance(admitted.plan.profile, OpsSreNavigationProfile),
                workbench=isinstance(admitted.plan.profile, OpsSreWorkbenchProfile),
                question_led=isinstance(admitted.plan.profile, OpsSreQuestionProfile),
            )
        elif working and channels.report in {"completed", "partial"}:
            from nurse_rcl.v2.live.ops_sre_native_render import inspect_native_report_pair

            inspect_native_report_pair(root / "report")
        elif sre and not (working or brief) and channels.report != "disabled":
            from nurse_rcl.v2.live.ops_sre_report_delivery import inspect_sre_report_pair

            inspect_sre_report_pair(root / "report")
        files = {n: list(member_digest(root / n)) for n in _MEMBERS if (root / n).exists()}
        body = dict(
            schema_version=(
                "nurse.v2.sre-case-result/3"
                if brief
                else "nurse.v2.sre-case-result/2"
                if working
                else "nurse.v2.sre-case-result/1"
                if sre
                else "nurse.v2.ops-case-result/2"
                if empirical
                else "nurse.v2.ops-case-result/1"
            ),
            identity=case_identity(admitted),
            channels=channels.model_dump(),
            core_result_digest=core.core_result_digest,
            candidate_count=len(core.evidence.ranking.scores),
            paid_count=core.evidence.session.paid_count,
            files=files,
            status="completed"
            if channels.report in {"completed", "disabled"}
            else "completed_partial_report"
            if channels.report == "partial"
            else "core_completed_report_unavailable",
        )
        if working or brief:
            body["terminal_failure"] = channels.report == "unavailable"
            if failure_summary is not None:
                body.update(failure_summary)
        if empirical:
            body.update(
                usage_policy_id="ops-empirical-input-cost/1",
                dataset=admitted.plan.case.dataset,
                case_snapshot_id=core.identity.case_snapshot_id,
                usage=usage_summary(
                    root / "provider",
                    execution_sha256=admitted.plan_sha256,
                    empirical=True,
                    sre=sre,
                    working=working,
                    feedback_fact=isinstance(admitted.plan.profile, OpsSreFeedbackFactProfile),
                    sre_profile=(
                        admitted.plan.profile.investigation_profile_id
                        if isinstance(admitted.plan.profile, OpsSreProfileBase)
                        else None
                    ),
                    confirmed_no_send=admitted.plan.mode == "core-only",
                ),
            )
            body["cost_case"] = cost_case_view(admitted.plan, admitted.plan_sha256, body)
        pin = sha256_hex(canonical_json_bytes(body))
        result = dict(**body, result_sha256=pin)
        publish_json(root / f"case-receipt-{pin}.json", result)
        atomic_write_bytes(root / "case-result.json", canonical_json_bytes(result))
        return result


def cost_case_view(plan: OpsSinglePlan, pin: str, result: dict[str, Any]) -> dict[str, Any]:
    """Derived, version-bound cost projection; attempts remain the sole authority."""
    usage = result["usage"]
    value = dict(
        case_id=plan.case.opaque_case_id,
        dataset=plan.case.dataset,
        version=sha256_hex(
            canonical_json_bytes(
                dict(
                    code_sha256=plan.code_sha256,
                    profile=plan.profile.model_dump(mode="json"),
                    mode=plan.mode,
                    environment_sha256=plan.environment_sha256,
                    core_configuration_sha256=plan.core_configuration_sha256,
                )
            )
        ),
        execution_sha256=pin,
        case_snapshot_id=result.get("case_snapshot_id"),
        preparation_key_sha256=plan.preparation.key.key_sha256,
        status=result["status"],
        channels=result["channels"],
        attempts=usage["attempt_usage"],
        confirmed_no_send=usage["attempts"] == 0 and usage["input_usage_complete"],
        response_reuses=usage["response_reuses"],
    )
    if "incremental_attempt_ids" in usage:
        value["incremental_attempt_ids"] = usage["incremental_attempt_ids"]
    return value


def run_ops_case(admitted: OpsAdmittedPlan, *, approved_plan_sha256: str | None) -> dict[str, Any]:
    """Private worker API; public command supplies source/code-admitted metadata."""
    if getattr(admitted.plan.profile, "investigation_profile_id", None) in {
        "hermes-sre-tools/18",
        "hermes-sre-tools/19",
        "hermes-sre-tools/20",
        "hermes-sre-tools/21",
        "hermes-sre-tools/22",
        "hermes-sre-tools/23",
    }:
        raise ValueError("directed profile requires a bounded decision stage, not full case owner")
    require_ops_code(admitted.plan)
    root = admitted.state_root
    root.mkdir(mode=0o700, exist_ok=True)
    require_plain_path(root)
    descriptor = os.open(root / "case.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish_json(root / "identity.json", case_identity(admitted))
        if (root / "case-result.json").exists():
            retained = read_json(root / "case-result.json")
            if retained["channels"]["report"] in {"completed", "partial", "disabled"} or (
                isinstance(
                    admitted.plan.profile,
                    OpsSreWorkingProfile
                    | OpsSreBriefProfile
                    | OpsSreConditionProfile
                    | OpsSreRecoveryProfile
                    | OpsSreComparableProfile
                    | OpsSreNavigationProfile
                    | OpsSreWorkbenchProfile,
                )
                and retained.get("terminal_failure") is True
            ):
                return inspect_ops_case(admitted)
            # A nonterminal report can legitimately change after a failed
            # summary write. Its own journal validates that recovery; only
            # core/context bytes remain immutable across those attempts.
        if (
            admitted.plan.mode == "with-agent"
            and approved_plan_sha256 != admitted.plan_sha256
            and (not (root / "report/context.json").exists())
        ):
            raise ValueError("isolated agent run requires explicit plan approval")
        try:
            return _run(admitted, approved_plan_sha256)
        except BaseException as exc:
            failure = dict(
                failure_type=type(exc).__name__,
                identity=case_identity(admitted),
                monotonic_seconds=monotonic(),
                core_saved=(root / "core/result.json").exists(),
            )
            pin = sha256_hex(canonical_json_bytes(failure))
            publish_json(root / f"case-failure-{pin}.json", failure)
            raise
    finally:
        os.close(descriptor)


def _run(admitted: OpsAdmittedPlan, approved: str | None) -> dict[str, Any]:
    root, plan = admitted.state_root, admitted.plan
    if (
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
        and plan.mode == "with-agent"
    ):
        from nurse_rcl.v2.live.ops_sre_working_case import run_working_case

        return run_working_case(admitted, approved)
    if isinstance(plan.profile, OpsSreProfileBase) and plan.mode == "with-agent":
        from nurse_rcl.v2.live.ops_sre_case import run_sre_case

        return run_sre_case(admitted, approved)
    empirical = isinstance(plan.profile, OpsEmpiricalProfile)
    if plan.mode == "core-only" and (root / "core/result.json").exists():
        return _finish(
            admitted,
            _core(root, admitted),
            OpsChannelStatus(core="completed", investigation="disabled", report="disabled"),
        )
    context_path = root / "report/context.json"
    if context_path.exists():
        # Once report context is frozen, neither preparation nor investigation
        # may alter it. Existing raw responses can finish offline in report owner.
        core = _core(root, admitted)
        context = OpsReportContext.model_validate_json(_read_bytes(context_path, 4_194_304))
        if (context.core_result_digest, context.case_snapshot_id) != (
            core.core_result_digest,
            core.identity.case_snapshot_id,
        ):
            raise ValueError("isolated frozen report/core identity differs")
    else:
        data = _preparation(admitted)
        with _module_span("input_guard"):
            _admit_display(data, admitted)
        identity = OpsCoreIdentity(
            dataset=plan.case.dataset,
            opaque_case_id=plan.case.opaque_case_id,
            case_snapshot_id=prepared_snapshot_id(data),
            configuration_sha256=plan.core_configuration_sha256,
            environment_sha256=plan.environment_sha256,
        )
        if plan.mode == "core-only":
            core = run_ops_core(
                data=data,
                identity=identity,
                root=root,
                block_root=Path(plan.preparation.archive_root) / "blocks",
            )
            return _finish(
                admitted,
                core,
                OpsChannelStatus(core="completed", investigation="disabled", report="disabled"),
            )
        provider = _provider(admitted, identity.case_snapshot_id, approved)
        pair = start_ops_pair(
            data=data,
            identity=identity,
            root=root,
            block_root=Path(plan.preparation.archive_root) / "blocks",
            code_sha256=plan.code_sha256,
            complete=provider.decision,
            empirical=empirical,
        )
        try:
            core = pair.join_core()
            try:
                agent = pair.join_investigation()
            except RuntimeError as exc:
                agent = OpsAgentResult(
                    termination="unavailable",
                    decision_count=0,
                    action_count=0,
                    messages=(),
                    failure_type=type(exc).__name__,
                )
        finally:
            pair.close()
        ledger_path = root / "investigation/state.json"
        evidence: tuple[OpsInvestigationEvidence, ...] = ()
        if ledger_path.exists():
            state = (OpsEmpiricalLedgerState if empirical else OpsLedgerState).model_validate_json(
                _read_bytes(ledger_path, (64 if empirical else 12) * 1024 * 1024)
            )
            if state.code_sha256 != plan.code_sha256 or (
                state.case_snapshot_id != identity.case_snapshot_id
                or state.state_sha256
                != sha256_hex(
                    canonical_json_bytes(state.model_dump(mode="json", exclude={"state_sha256"}))
                )
            ):
                raise ValueError("isolated investigation receipt integrity differs")
            evidence = tuple(r.result for r in state.records if r.result is not None)
        with _module_span("report_projection"):
            context = build_report_context(
                core=core,
                sources=data.sources,
                investigation=evidence,
                agent_result=agent,
                soft_targets=empirical,
            )
    (root / "report").mkdir(mode=0o700, exist_ok=True)
    publish_json(context_path, context.model_dump(mode="json"))
    _formal_binding(admitted, context)
    report_provider = (
        _provider(admitted, core.identity.case_snapshot_id, approved)
        if approved == admitted.plan_sha256
        else None
    )
    saved = (
        None
        if report_provider is not None or empirical
        else saved_ops_report(
            root / "provider",
            context,
            code_sha256=plan.code_sha256,
            execution_sha256=admitted.plan_sha256,
        )
    )
    complete: ReportCompletion | None = (
        report_provider.report if report_provider is not None else None
    )
    if empirical and report_provider is None:

        def recover_empirical(request: OpsReportContext | bytes) -> bytes:
            if type(request) is not bytes:
                raise ValueError("empirical recovery requires frozen report wire")
            retained = saved_ops_report(
                root / "provider",
                context,
                code_sha256=plan.code_sha256,
                execution_sha256=admitted.plan_sha256,
                empirical=True,
                frozen_wire=request,
            )
            if retained is None:
                raise ValueError("saved_report_unavailable")
            return retained

        complete = recover_empirical
    if saved is not None:

        def recover(_: OpsReportContext | bytes) -> bytes:
            assert saved is not None
            return saved

        complete = recover
    with _module_span("report"):
        report = run_ops_report(
            root=root,
            context=context,
            code_sha256=plan.code_sha256,
            complete=complete,
            empirical=empirical,
        )
    require_ops_code(plan)
    return _finish(admitted, core, report.channels)
