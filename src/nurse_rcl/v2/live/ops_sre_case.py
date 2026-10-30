"""Explicit SRE case branch; old profiles keep their readers and wire semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_agent_state import OpsAgentResult, load_conversation
from nurse_rcl.v2.live.ops_contracts import OpsCoreIdentity
from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord
from nurse_rcl.v2.live.ops_node_common import prepared_snapshot_id
from nurse_rcl.v2.live.ops_plan import OpsAdmittedPlan, require_ops_code
from nurse_rcl.v2.live.ops_provider import saved_sre_report, usage_summary
from nurse_rcl.v2.live.ops_sre_ledger import OpsSreLedgerState
from nurse_rcl.v2.live.ops_sre_report_context import build_sre_report_context
from nurse_rcl.v2.live.ops_sre_report_run import (
    SreReportCompletion,
    publish_saved_sre_report,
    read_public_history,
    run_sre_report,
)
from nurse_rcl.v2.live.ops_sre_report_storage import decode_sre_report_storage
from nurse_rcl.v2.live.repair_module_timing import _module_span


def run_sre_case(admitted: OpsAdmittedPlan, approved: str | None) -> dict[str, Any]:
    # Shared private owner boundaries, not a new authority or a second core.
    from nurse_rcl.v2.live import ops_case as owner

    root, plan = admitted.state_root, admitted.plan
    context_path = root / "report/context.json"
    if context_path.exists():
        core = owner._core(root, admitted)
        context = decode_sre_report_storage(_read_bytes(context_path, 4_194_304))
        if (context.core_result_digest, context.evidence.case_snapshot_id,
            context.evidence.opaque_case_id) != (
            core.core_result_digest, core.identity.case_snapshot_id, plan.case.opaque_case_id,
        ):
            raise ValueError("SRE frozen report/core identity differs")
    else:
        data = owner._preparation(admitted)
        with _module_span("input_guard"):
            owner._admit_display(data, admitted)
        identity = OpsCoreIdentity(dataset=plan.case.dataset,
            opaque_case_id=plan.case.opaque_case_id,
            case_snapshot_id=prepared_snapshot_id(data),
            configuration_sha256=plan.core_configuration_sha256,
            environment_sha256=plan.environment_sha256)
        imported = None
        if plan.source_core is not None:
            from nurse_rcl.v2.live.ops_sre_core_import import import_source_core

            with _module_span("input_guard"):
                imported = import_source_core(admitted, identity)
        provider = owner._provider(admitted, identity.case_snapshot_id, approved)
        pair = owner.start_ops_pair(data=data, identity=identity, root=root,
            block_root=Path(plan.preparation.archive_root) / "blocks",
            code_sha256=plan.code_sha256, complete=provider.decision, empirical=True, sre=True,
            frozen_core=imported)
        try:
            core = pair.join_core()
            try:
                agent = pair.join_investigation()
            except RuntimeError as exc:
                agent = OpsAgentResult(termination="unavailable", decision_count=0,
                    action_count=0, messages=(), failure_type=type(exc).__name__)
        finally:
            pair.close()
        conversation = load_conversation(root / "investigation/conversation.json",
            snapshot=identity.case_snapshot_id, code=plan.code_sha256,
            empirical=True, sre=True, opaque_case_id=plan.case.opaque_case_id)
        ledger_path = root / "investigation/state.json"
        records: tuple[OpsActionRecord, ...] = ()
        if ledger_path.exists():
            raw = _read_bytes(ledger_path, 64 * 1024 * 1024)
            state = OpsSreLedgerState.model_validate_json(raw)
            if (state.code_sha256, state.case_snapshot_id) != (
                plan.code_sha256, identity.case_snapshot_id,
            ) or raw != canonical_json_bytes(state.model_dump(mode="json")) or (
                state.state_sha256 != sha256_hex(canonical_json_bytes(state.model_dump(
                    mode="json", exclude={"state_sha256"})))
            ):
                raise ValueError("SRE investigation receipt identity differs")
            records = state.records
        with _module_span("report_projection"):
            context = build_sre_report_context(core=core, sources=data.sources,
                decisions=conversation.decisions, records=records,
                investigation_status="completed" if agent.termination != "unavailable" else
                "partial" if records else "unavailable", termination=agent.termination,
                failure_type=agent.failure_type)
    report_provider = (owner._provider(admitted, core.identity.case_snapshot_id, approved)
                       if approved == admitted.plan_sha256 else None)

    def recover(wire: bytes) -> bytes:
        retained = saved_sre_report(root / "provider", wire, code_sha256=plan.code_sha256,
                                    execution_sha256=admitted.plan_sha256)
        if retained is None:
            raise ValueError("saved_sre_report_unavailable")
        return retained

    completion: SreReportCompletion = (
        report_provider.report if report_provider is not None else recover)
    with _module_span("report"):
        report = run_sre_report(root=root, context=context, code_sha256=plan.code_sha256,
                                complete=completion)
        usage = usage_summary(root / "provider", execution_sha256=admitted.plan_sha256,
                                    empirical=True, sre=True)
        publish_saved_sre_report(root / "report", context, report,
            accounting=dict(usage=usage, provider_seconds=report.provider_seconds,
                            validation_seconds=report.validation_seconds),
            public_history=read_public_history(root / "investigation/conversation.json"))
    owner._formal_binding(admitted, context)
    require_ops_code(plan)
    return owner._finish(admitted, core, report.channels)
