"""D12 case owner: independent core, truthful investigation and one report assembly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_agent_state import (
    OpsBalancedConversation,
    OpsBriefConversation,
    OpsComparableConversation,
    OpsConditionConversation,
    OpsNavigationAgentResult,
    OpsNavigationConversation,
    OpsReceiptReuseConversation,
    OpsRecoveryConversation,
    OpsReplanningConversation,
    OpsUsedSourceConversation,
    OpsWorkbenchConversation,
    OpsWorkingAgentResult,
    OpsWorkingConversation,
    SreConversationFields,
    load_conversation,
)
from nurse_rcl.v2.live.ops_contracts import (
    OpsChannelStatus,
    OpsCoreIdentity,
    OpsSreActionProfile,
    OpsSreBalancedProfile,
    OpsSreBriefProfile,
    OpsSreComparableProfile,
    OpsSreConditionProfile,
    OpsSreNavigationProfile,
    OpsSreQuestionProfile,
    OpsSreReceiptReuseProfile,
    OpsSreRecoveryProfile,
    OpsSreReplanningProfile,
    OpsSreUsedSourceProfile,
    OpsSreWorkbenchProfile,
    OpsSreWorkingProfile,
)
from nurse_rcl.v2.live.ops_failure import failure_detail, secondary_failures
from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord
from nurse_rcl.v2.live.ops_node_common import prepared_snapshot_id
from nurse_rcl.v2.live.ops_plan import OpsAdmittedPlan, OpsSinglePlan, require_ops_code
from nurse_rcl.v2.live.ops_provider import saved_sre_report, usage_summary
from nurse_rcl.v2.live.ops_report_context import build_report_context
from nurse_rcl.v2.live.ops_result import OpsCoreResult
from nurse_rcl.v2.live.ops_sre_action_tools import (
    OpsSreActionInvestigationLedger,
    OpsSreActionLedgerState,
    OpsSreActionPublicationReader,
    OpsSreActionToolbox,
)
from nurse_rcl.v2.live.ops_sre_brief_report import (
    BriefSreReportContext,
    assemble_brief_report_context,
    decode_brief_report_storage,
    encode_brief_report_storage,
)
from nurse_rcl.v2.live.ops_sre_brief_tools import (
    OpsSreBriefInvestigationLedger,
    OpsSreBriefLedgerState,
    OpsSreBriefToolbox,
)
from nurse_rcl.v2.live.ops_sre_checkpoint_pools_ledger import (
    OpsSrePooledInvestigationLedger,
    OpsSrePooledLedgerState,
)
from nurse_rcl.v2.live.ops_sre_comparable_tools import (
    OpsSreComparableInvestigationLedger,
    OpsSreComparableLedgerState,
    OpsSreComparableToolbox,
)
from nurse_rcl.v2.live.ops_sre_condition_tools import (
    OpsSreConditionInvestigationLedger,
    OpsSreConditionLedgerState,
    OpsSreConditionToolbox,
)
from nurse_rcl.v2.live.ops_sre_d21_tools import (
    OpsSreUsedSourceInvestigationLedger,
    OpsSreUsedSourceLedgerState,
    OpsSreUsedSourcePublicationReader,
    OpsSreUsedSourceToolbox,
)
from nurse_rcl.v2.live.ops_sre_d22_tools import (
    OpsSreReceiptReuseInvestigationLedger,
    OpsSreReceiptReuseLedgerState,
    OpsSreReceiptReusePublicationReader,
    OpsSreReceiptReuseToolbox,
)
from nurse_rcl.v2.live.ops_sre_d23_tools import (
    OpsSreBalancedInvestigationLedger,
    OpsSreBalancedLedgerState,
    OpsSreBalancedPublicationReader,
    OpsSreBalancedToolbox,
)
from nurse_rcl.v2.live.ops_sre_d25_tools import (
    OpsSreReplanningInvestigationLedger,
    OpsSreReplanningLedgerState,
    OpsSreReplanningPublicationReader,
    OpsSreReplanningToolbox,
)
from nurse_rcl.v2.live.ops_sre_d26_tools import (
    OpsSreQuestionPublicationReader,
    OpsSreQuestionToolbox,
)
from nurse_rcl.v2.live.ops_sre_native_report import (
    assemble_native_report_context,
    decode_native_report_storage,
    encode_native_report_storage,
)
from nurse_rcl.v2.live.ops_sre_native_run import (
    inspect_saved_brief_run,
    inspect_saved_native_run,
    publish_saved_native_report,
    run_brief_report,
    run_native_report,
)
from nurse_rcl.v2.live.ops_sre_navigation_tools import (
    OpsSreNavigationInvestigationLedger,
    OpsSreNavigationLedgerState,
    OpsSreNavigationPublicationReader,
    OpsSreNavigationToolbox,
)
from nurse_rcl.v2.live.ops_sre_recovery_tools import (
    OpsSreRecoveryInvestigationLedger,
    OpsSreRecoveryLedgerState,
    OpsSreRecoveryToolbox,
)
from nurse_rcl.v2.live.ops_sre_report_run import read_public_history
from nurse_rcl.v2.live.ops_sre_workbench_tools import (
    OpsSreWorkbenchInvestigationLedger,
    OpsSreWorkbenchLedgerState,
    OpsSreWorkbenchPublicationReader,
    OpsSreWorkbenchToolbox,
)
from nurse_rcl.v2.live.ops_sre_working_tools import OpsSreWorkingLedgerState
from nurse_rcl.v2.live.repair_driver import read_json
from nurse_rcl.v2.live.repair_module_timing import _module_span
from nurse_rcl.v2.live.repair_preparation_codec import member_digest


def _investigation(
    admitted: OpsAdmittedPlan,
    core: OpsCoreResult,
) -> tuple[
    OpsWorkingConversation
    | OpsBriefConversation
    | OpsConditionConversation
    | OpsRecoveryConversation
    | OpsComparableConversation
    | OpsNavigationConversation
    | OpsWorkbenchConversation
    | OpsUsedSourceConversation
    | OpsReceiptReuseConversation
    | OpsBalancedConversation
    | OpsReplanningConversation,
    tuple[OpsActionRecord, ...],
]:
    return _read_investigation(admitted.state_root, admitted.plan, core)


def _read_investigation(
    root: Path,
    plan: OpsSinglePlan,
    core: OpsCoreResult,
) -> tuple[
    OpsWorkingConversation
    | OpsBriefConversation
    | OpsConditionConversation
    | OpsRecoveryConversation
    | OpsComparableConversation
    | OpsNavigationConversation
    | OpsWorkbenchConversation
    | OpsUsedSourceConversation
    | OpsReceiptReuseConversation
    | OpsBalancedConversation
    | OpsReplanningConversation,
    tuple[OpsActionRecord, ...],
]:
    assert isinstance(
        plan.profile,
        OpsSreWorkingProfile
        | OpsSreBriefProfile
        | OpsSreConditionProfile
        | OpsSreRecoveryProfile
        | OpsSreComparableProfile
        | OpsSreNavigationProfile
        | OpsSreWorkbenchProfile,
    )
    brief = isinstance(
        plan.profile,
        OpsSreBriefProfile
        | OpsSreConditionProfile
        | OpsSreRecoveryProfile
        | OpsSreComparableProfile
        | OpsSreNavigationProfile
        | OpsSreWorkbenchProfile,
    )
    condition = isinstance(
        plan.profile,
        OpsSreConditionProfile
        | OpsSreRecoveryProfile
        | OpsSreComparableProfile
        | OpsSreNavigationProfile
        | OpsSreWorkbenchProfile,
    )
    comparable = isinstance(
        plan.profile, OpsSreComparableProfile | OpsSreNavigationProfile | OpsSreWorkbenchProfile
    )
    action_first = isinstance(plan.profile, OpsSreActionProfile)
    question_led = isinstance(plan.profile, OpsSreQuestionProfile)
    replanning = isinstance(plan.profile, OpsSreReplanningProfile)
    balanced = isinstance(plan.profile, OpsSreBalancedProfile)
    receipt_reuse = isinstance(plan.profile, OpsSreReceiptReuseProfile)
    used_source = isinstance(plan.profile, OpsSreUsedSourceProfile)
    workbench = isinstance(plan.profile, OpsSreWorkbenchProfile)
    navigation = isinstance(plan.profile, OpsSreNavigationProfile) or workbench
    recovery = isinstance(plan.profile, OpsSreRecoveryProfile)
    conversation = load_conversation(
        root / "investigation/conversation.json",
        snapshot=core.identity.case_snapshot_id,
        code=plan.code_sha256,
        empirical=True,
        sre=True,
        sre_profile=plan.profile.investigation_profile_id,
        opaque_case_id=plan.case.opaque_case_id,
    )
    assert isinstance(
        conversation,
        OpsWorkingConversation
        | OpsBriefConversation
        | OpsConditionConversation
        | OpsRecoveryConversation
        | OpsComparableConversation
        | OpsNavigationConversation
        | OpsWorkbenchConversation
        | OpsUsedSourceConversation
        | OpsReceiptReuseConversation
        | OpsBalancedConversation
        | OpsReplanningConversation,
    )
    records: tuple[OpsActionRecord, ...] = ()
    path = root / "investigation/state.json"
    if path.exists():
        raw = _read_bytes(path, 64 * 1024 * 1024)
        state = (
            OpsSreActionLedgerState
            if action_first
            else OpsSrePooledLedgerState
            if question_led
            else OpsSreReplanningLedgerState
            if replanning
            else OpsSreBalancedLedgerState
            if balanced
            else OpsSreReceiptReuseLedgerState
            if receipt_reuse
            else OpsSreUsedSourceLedgerState
            if used_source
            else OpsSreWorkbenchLedgerState
            if workbench
            else OpsSreNavigationLedgerState
            if navigation
            else OpsSreComparableLedgerState
            if comparable
            else OpsSreRecoveryLedgerState
            if recovery
            else OpsSreConditionLedgerState
            if condition
            else OpsSreBriefLedgerState
            if brief
            else OpsSreWorkingLedgerState
        ).model_validate_json(raw)
        if (
            (state.code_sha256, state.case_snapshot_id)
            != (
                plan.code_sha256,
                core.identity.case_snapshot_id,
            )
            or raw != canonical_json_bytes(state.model_dump(mode="json"))
            or (
                state.state_sha256
                != sha256_hex(
                    canonical_json_bytes(state.model_dump(mode="json", exclude={"state_sha256"}))
                )
            )
        ):
            raise ValueError("working investigation receipt identity differs")
        records = state.records
    result = conversation.result
    if result is None or result.action_count != len(records):
        raise ValueError("working investigation outcome missing or differs")
    declared = [r for r in conversation.decisions if r.action_index is not None]
    if len(declared) - len(records) not in {0, 1}:
        raise ValueError("working declared/consumed action sequence differs")
    for decision, record in zip(declared, records, strict=False):
        assert decision.response_json is not None
        if sha256_hex(canonical_json_bytes(json.loads(decision.response_json)["actions"])) != (
            record.input_sha256
        ):
            raise ValueError("working declared action differs from receipt")
    if result.termination == "unavailable" and result.investigation_status != (
        "partial" if any(r.status == "completed" for r in records) else "unavailable"
    ):
        raise ValueError("working investigation evidence status differs")
    return conversation, records


def _save_assembly(path: Path, body: dict[str, Any]) -> dict[str, Any]:
    unsigned = {k: v for k, v in body.items() if k != "state_sha256"}
    sealed = dict(unsigned, state_sha256=sha256_hex(canonical_json_bytes(unsigned)))
    atomic_write_bytes(path, canonical_json_bytes(sealed))
    return sealed


def _bind_report_replay_visibility(
    toolbox: OpsSreRecoveryToolbox,
    conversation: SreConversationFields,
    records: tuple[OpsActionRecord, ...],
) -> None:
    """Rebuild historical read admission before the unchanged ledger replay.

    The conversation has already passed the same-run owner validation in
    ``load_conversation``. A final-prefix-only bind could retroactively admit a
    card first shown after its saved read, so verify every completed read at
    its own decision before leaving the final prefix bound for D22 replay.
    """
    from nurse_rcl.v2.live.ops_sre_model_view import _prefix
    from nurse_rcl.v2.live.ops_sre_references import SreReferenceBook
    from nurse_rcl.v2.live.ops_sre_views import semantic_record
    from nurse_rcl.v2.live.ops_tools import card_id

    book = SreReferenceBook.from_state(
        conversation.reference_state,
        snapshot=toolbox.case_snapshot_id,
        run_id=conversation.reference_state["run_id"],
    )
    decision_for_action = {
        row.action_index: row.index
        for row in conversation.decisions
        if row.action_index is not None
    }

    def visible_at(index: int, card: object, targets: list[str]) -> bool:
        _, prefix = _prefix(book, index)
        entries = prefix.to_state()["entries"]
        cards = {original for kind, _, original, _ in entries if kind == "card"}
        services = {original for kind, _, original, _ in entries if kind == "service"}
        return type(card) is str and card in cards and all(t in services for t in targets)

    for record in records:
        action = record.action
        if record.status != "completed" or action is None or action.tool != "read_node":
            continue
        decision_index = decision_for_action.get(record.index)
        if decision_index is None:
            raise ValueError("saved report read decision identity differs")
        parameters = json.loads(action.parameters_json)
        card = parameters.get("card_id")
        if record.result is None or card_id(record.result.evidence) != card:
            raise ValueError("saved report read receipt identity differs")
        row = semantic_record(
            record.result.evidence,
            published=False,
            origin=record.result.origin,
        )
        targets = row["targets"]
        if not visible_at(decision_index, card, targets):
            raise ValueError("saved report read was not visible at decision")
        reuse = getattr(record, "receipt_reuse", None)
        if reuse is not None:
            prior_index = reuse.get("prior_action_index")
            prior_decision = decision_for_action.get(prior_index)
            if (
                type(prior_index) is not int
                or prior_index >= record.index
                or prior_decision is None
            ):
                raise ValueError("saved report reuse decision identity differs")
            if not visible_at(prior_decision, card, targets):
                raise ValueError("saved report reuse source was not visible at decision")
    toolbox.bind_visible_references(
        book=book,
        request_index=max(1, len(conversation.decisions)),
    )


def _assembly_identity(admitted: OpsAdmittedPlan, core: OpsCoreResult) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_case import case_identity

    return dict(
        identity=case_identity(admitted),
        core_result_digest=core.core_result_digest,
        source_members={
            name: list(member_digest(admitted.state_root / name))
            for name in (
                "core/result.json",
                "investigation/conversation.json",
                "investigation/state.json",
            )
        },
    )


def _assembly(admitted: OpsAdmittedPlan, core: OpsCoreResult) -> dict[str, Any]:
    body = read_json(admitted.state_root / "report/assembly.json")
    unsigned = {k: v for k, v in body.items() if k != "state_sha256"}
    if body.get("state_sha256") != sha256_hex(canonical_json_bytes(unsigned)) or any(
        body.get(k) != value for k, value in _assembly_identity(admitted, core).items()
    ):
        raise ValueError("native assembly owner or source identity differs")
    return body


def _summary(body: dict[str, Any]) -> dict[str, Any]:
    return {
        key: body[key]
        for key in ("primary_failure", "secondary_failures", "decision_count", "action_count")
    }


def inspect_working_progress(
    root: Path,
    plan: OpsSinglePlan,
    pin: str,
    core: OpsCoreResult,
) -> dict[str, Any]:
    """Recover truthful saved channels after a worker exits; never run an owner."""
    from nurse_rcl.v2.live.ops_case import plan_identity

    brief = isinstance(
        plan.profile,
        OpsSreBriefProfile
        | OpsSreConditionProfile
        | OpsSreRecoveryProfile
        | OpsSreComparableProfile
        | OpsSreNavigationProfile
        | OpsSreWorkbenchProfile,
    )

    conversation, _ = _read_investigation(root, plan, core)
    result = conversation.result
    assert result is not None
    failures = [result.primary_failure] if result.primary_failure else []
    failures.extend(result.secondary_failures)
    channels = dict(
        core="completed", investigation=result.investigation_status, report="unavailable"
    )
    assembly_path = root / "report/assembly.json"
    if assembly_path.exists():
        body = read_json(assembly_path)
        unsigned = {k: v for k, v in body.items() if k != "state_sha256"}
        members = {
            name: list(member_digest(root / name))
            for name in (
                "core/result.json",
                "investigation/conversation.json",
                "investigation/state.json",
            )
        }
        if body.get("state_sha256") != sha256_hex(canonical_json_bytes(unsigned)) or (
            body.get("identity") != plan_identity(plan, pin)
            or body.get("core_result_digest") != core.core_result_digest
            or body.get("source_members") != members
        ):
            raise ValueError("saved assembly source identity differs")
        from nurse_rcl.v2.live.ops_failure import OpsFailureDetail

        for row in [body.get("primary_failure"), *body.get("secondary_failures", [])]:
            if row is not None:
                failures.append(OpsFailureDetail.model_validate(row))
        context_path = root / "report/context.json"
        if (root / "report/result.json").exists():
            if body.get("context_member") != list(member_digest(context_path)):
                raise ValueError("saved assembly context identity differs")
            context = (decode_brief_report_storage if brief else decode_native_report_storage)(
                _read_bytes(context_path, 4_194_304)
            )
            if (
                context.core_result_digest,
                context.metadata.evidence.case_snapshot_id,
                context.metadata.evidence.opaque_case_id,
            ) != (
                core.core_result_digest,
                core.identity.case_snapshot_id,
                plan.case.opaque_case_id,
            ):
                raise ValueError("saved report core identity differs")
            options: dict[str, Any] = dict(
                code=plan.code_sha256,
                references=conversation.reference_state,
                public_history=read_public_history(root / "investigation/conversation.json"),
            )
            report = (
                inspect_saved_brief_run(root / "report", context, **options)
                if isinstance(context, BriefSreReportContext)
                else inspect_saved_native_run(root / "report", context, **options)
            )
            channels["report"] = report.status if report.status != "pending" else "unavailable"
            if report.primary_failure is not None:
                failures.append(report.primary_failure)
            failures.extend(report.secondary_failures)
    failures = list(dict.fromkeys(failures))
    return dict(
        channels=channels,
        decision_count=result.decision_count,
        action_count=result.action_count,
        primary_failure=failures[0].model_dump(mode="json") if failures else None,
        secondary_failures=[r.model_dump(mode="json") for r in failures[1:17]],
    )


def run_working_case(admitted: OpsAdmittedPlan, approved: str | None) -> dict[str, Any]:
    from nurse_rcl.v2.live import ops_case as owner

    root, plan = admitted.state_root, admitted.plan
    assert isinstance(
        plan.profile,
        OpsSreWorkingProfile
        | OpsSreBriefProfile
        | OpsSreConditionProfile
        | OpsSreRecoveryProfile
        | OpsSreComparableProfile
        | OpsSreNavigationProfile
        | OpsSreWorkbenchProfile,
    )
    brief = isinstance(
        plan.profile,
        OpsSreBriefProfile
        | OpsSreConditionProfile
        | OpsSreRecoveryProfile
        | OpsSreComparableProfile
        | OpsSreNavigationProfile
        | OpsSreWorkbenchProfile,
    )
    condition = isinstance(
        plan.profile,
        OpsSreConditionProfile
        | OpsSreRecoveryProfile
        | OpsSreComparableProfile
        | OpsSreNavigationProfile
        | OpsSreWorkbenchProfile,
    )
    comparable = isinstance(
        plan.profile, OpsSreComparableProfile | OpsSreNavigationProfile | OpsSreWorkbenchProfile
    )
    action_first = isinstance(plan.profile, OpsSreActionProfile)
    question_led = isinstance(plan.profile, OpsSreQuestionProfile)
    replanning = isinstance(plan.profile, OpsSreReplanningProfile)
    balanced = isinstance(plan.profile, OpsSreBalancedProfile)
    receipt_reuse = isinstance(plan.profile, OpsSreReceiptReuseProfile)
    used_source = isinstance(plan.profile, OpsSreUsedSourceProfile)
    workbench = isinstance(plan.profile, OpsSreWorkbenchProfile)
    navigation = isinstance(plan.profile, OpsSreNavigationProfile) or workbench
    recovery = isinstance(plan.profile, OpsSreRecoveryProfile)
    selected = plan.profile.investigation_profile_id
    context_path, assembly_path = root / "report/context.json", root / "report/assembly.json"
    if assembly_path.exists() and not context_path.exists():
        # A failed or interrupted local assembly is not a fresh budget on restart.
        core = owner._core(root, admitted)
        body = _assembly(admitted, core)
        if body["status"] in {"started", "completed"}:
            unknown = failure_detail("report_assembly", RuntimeError("assembly_outcome_unknown"))
            body["secondary_failures"].append(unknown.model_dump(mode="json"))
            if body["primary_failure"] is None:
                body["primary_failure"] = body["secondary_failures"].pop()
            body["status"] = "unavailable"
            body = _save_assembly(assembly_path, body)
        if body["status"] != "unavailable":
            raise ValueError("completed native assembly is missing its context")
        return owner._finish(
            admitted,
            core,
            OpsChannelStatus.model_validate(body["channels"]),
            failure_summary=_summary(body),
        )
    if context_path.exists():
        core = owner._core(root, admitted)
        context = (decode_brief_report_storage if brief else decode_native_report_storage)(
            _read_bytes(context_path, 4_194_304)
        )
        header = context.metadata.evidence
        if (context.core_result_digest, header.case_snapshot_id, header.opaque_case_id) != (
            core.core_result_digest,
            core.identity.case_snapshot_id,
            plan.case.opaque_case_id,
        ):
            raise ValueError("working frozen report/core identity differs")
        body = _assembly(admitted, core)
        if body.get("context_member") != list(member_digest(context_path)):
            raise ValueError("working frozen report/assembly identity differs")
        conversation, records = _investigation(admitted, core)
    else:
        data = owner._preparation(admitted)
        with _module_span("input_guard"):
            owner._admit_display(data, admitted)
        identity = OpsCoreIdentity(
            dataset=plan.case.dataset,
            opaque_case_id=plan.case.opaque_case_id,
            case_snapshot_id=prepared_snapshot_id(data),
            configuration_sha256=plan.core_configuration_sha256,
            environment_sha256=plan.environment_sha256,
        )
        imported = None
        if plan.source_core is not None:
            from nurse_rcl.v2.live.ops_sre_core_import import import_source_core

            with _module_span("input_guard"):
                imported = import_source_core(admitted, identity)
        provider = owner._provider(admitted, identity.case_snapshot_id, approved)
        startup_options: dict[str, Any] = (
            dict(execution_sha256=admitted.plan_sha256) if comparable else {}
        )
        pair = owner.start_ops_pair(
            data=data,
            identity=identity,
            root=root,
            block_root=Path(plan.preparation.archive_root) / "blocks",
            code_sha256=plan.code_sha256,
            complete=provider.decision,
            recover_complete=(provider.saved_decision if used_source else None),
            empirical=True,
            sre=True,
            sre_profile=selected,
            frozen_core=imported,
            **startup_options,
        )
        try:
            core = pair.join_core()
            agent = pair.join_investigation()
        finally:
            pair.close()
        assert isinstance(agent, OpsWorkingAgentResult)
        conversation, records = _investigation(admitted, core)
        body = dict(
            schema_version="nurse.v2.sre-report-assembly/1",
            **_assembly_identity(admitted, core),
            status="started",
            channels=dict(
                core="completed", investigation=agent.investigation_status, report="unavailable"
            ),
            primary_failure=(
                agent.primary_failure.model_dump(mode="json") if agent.primary_failure else None
            ),
            secondary_failures=[r.model_dump(mode="json") for r in agent.secondary_failures],
            decision_count=agent.decision_count,
            action_count=agent.action_count,
        )
        (root / "report").mkdir(mode=0o700, exist_ok=True)
        _save_assembly(assembly_path, body)
        try:
            with _module_span("report_projection"):
                formal = build_report_context(
                    core=core, sources=data.sources, investigation=(), soft_targets=True
                )
                assembly_options: dict[str, Any] = dict(
                    formal=formal,
                    decisions=conversation.decisions,
                    records=records,
                    investigation_status=agent.investigation_status,
                    termination=agent.termination,
                    failure_type=agent.failure_type,
                )
                if brief:
                    # Restore saved cards and frozen prefixes only. No tool dispatch,
                    # no preparation reload and no ledger write during report assembly.
                    toolbox = (
                        OpsSreActionToolbox
                        if action_first
                        else OpsSreQuestionToolbox
                        if question_led
                        else OpsSreReplanningToolbox
                        if replanning
                        else OpsSreBalancedToolbox
                        if balanced
                        else OpsSreReceiptReuseToolbox
                        if receipt_reuse
                        else OpsSreUsedSourceToolbox
                        if used_source
                        else OpsSreWorkbenchToolbox
                        if workbench
                        else OpsSreNavigationToolbox
                        if navigation
                        else OpsSreComparableToolbox
                        if comparable
                        else OpsSreRecoveryToolbox
                        if recovery
                        else OpsSreConditionToolbox
                        if condition
                        else OpsSreBriefToolbox
                    )(data=data, block_root=Path(plan.preparation.archive_root) / "blocks")
                    source_raw = _read_bytes(root / "investigation/state.json", 64 * 1024 * 1024)
                    if [len(source_raw), sha256_hex(source_raw)] != (
                        body["source_members"]["investigation/state.json"]
                    ):
                        raise ValueError("brief assembly source member changed")
                    source_state = (
                        OpsSreActionLedgerState
                        if action_first
                        else OpsSrePooledLedgerState
                        if question_led
                        else OpsSreReplanningLedgerState
                        if replanning
                        else OpsSreBalancedLedgerState
                        if balanced
                        else OpsSreReceiptReuseLedgerState
                        if receipt_reuse
                        else OpsSreUsedSourceLedgerState
                        if used_source
                        else OpsSreWorkbenchLedgerState
                        if workbench
                        else OpsSreNavigationLedgerState
                        if navigation
                        else OpsSreComparableLedgerState
                        if comparable
                        else OpsSreRecoveryLedgerState
                        if recovery
                        else OpsSreConditionLedgerState
                        if condition
                        else OpsSreBriefLedgerState
                    ).model_validate_json(source_raw)
                    if source_state.records != records:
                        raise ValueError("brief assembly ledger changed")
                    ledger = (
                        OpsSreActionInvestigationLedger
                        if action_first
                        else OpsSrePooledInvestigationLedger
                        if question_led
                        else OpsSreReplanningInvestigationLedger
                        if replanning
                        else OpsSreBalancedInvestigationLedger
                        if balanced
                        else OpsSreReceiptReuseInvestigationLedger
                        if receipt_reuse
                        else OpsSreUsedSourceInvestigationLedger
                        if used_source
                        else OpsSreWorkbenchInvestigationLedger
                        if workbench
                        else OpsSreNavigationInvestigationLedger
                        if navigation
                        else OpsSreComparableInvestigationLedger
                        if comparable
                        else OpsSreRecoveryInvestigationLedger
                        if recovery
                        else OpsSreConditionInvestigationLedger
                        if condition
                        else OpsSreBriefInvestigationLedger
                    )(root=root, toolbox=toolbox, code_sha256=plan.code_sha256)
                    if comparable:
                        assert isinstance(toolbox, OpsSreComparableToolbox)
                        from nurse_rcl.v2.live.ops_sre_comparable_publication import (
                            OpsSreComparablePublicationReader,
                        )

                        (
                            OpsSreActionPublicationReader
                            if action_first
                            else OpsSreQuestionPublicationReader
                            if question_led
                            else OpsSreReplanningPublicationReader
                            if replanning
                            else OpsSreBalancedPublicationReader
                            if balanced
                            else OpsSreReceiptReusePublicationReader
                            if receipt_reuse
                            else OpsSreUsedSourcePublicationReader
                            if used_source
                            else OpsSreWorkbenchPublicationReader
                            if workbench
                            else OpsSreNavigationPublicationReader
                            if navigation
                            else OpsSreComparablePublicationReader
                        )(
                            root=root,
                            data=data,
                            identity=core.identity,
                        ).bind_checkpoint(toolbox, core=core)
                    if isinstance(toolbox, OpsSreRecoveryToolbox):
                        assert isinstance(conversation, SreConversationFields)
                        _bind_report_replay_visibility(toolbox, conversation, records)
                    ledger._restore_tool_state(source_state)
                    context = assemble_brief_report_context(
                        toolbox=toolbox,
                        investigation_failure=agent.primary_failure
                        if isinstance(agent, OpsNavigationAgentResult)
                        else None,
                        **assembly_options,
                    )
                    raw = encode_brief_report_storage(context)
                else:
                    context = assemble_native_report_context(**assembly_options)
                    raw = encode_native_report_storage(context)
                # Bind the exact context before publishing it, so either crash
                # window is terminal or recoverable without another assembly.
                body.update(status="completed", context_member=[len(raw), sha256_hex(raw)])
                _save_assembly(assembly_path, body)
                atomic_write_bytes(context_path, raw)
        except BaseException as exc:
            detail = failure_detail("report_assembly", exc).model_dump(mode="json")
            if body["primary_failure"] is None:
                body["primary_failure"] = detail
            else:
                body["secondary_failures"].append(detail)
            body["secondary_failures"].extend(
                row.model_dump(mode="json") for row in secondary_failures(exc)
            )
            body["status"] = "unavailable"
            _save_assembly(assembly_path, body)
            if not isinstance(exc, Exception):
                raise
            return owner._finish(
                admitted,
                core,
                OpsChannelStatus.model_validate(body["channels"]),
                failure_summary=_summary(body),
            )
    report_provider = (
        owner._provider(admitted, core.identity.case_snapshot_id, approved)
        if approved == admitted.plan_sha256
        else None
    )

    def recover(wire: bytes) -> bytes | None:
        return saved_sre_report(
            root / "provider",
            wire,
            code_sha256=plan.code_sha256,
            execution_sha256=admitted.plan_sha256,
            sre_profile=selected,
            read_only=True,
        )

    history = read_public_history(root / "investigation/conversation.json")
    with _module_span("report"):
        report_options: dict[str, Any] = dict(
            root=root,
            code_sha256=plan.code_sha256,
            references=conversation.reference_state,
            complete=report_provider.report if report_provider else None,
            public_history=history,
            recover_saved=recover,
        )
        report = (
            run_brief_report(context=context, **report_options)
            if isinstance(context, BriefSreReportContext)
            else run_native_report(context=context, **report_options)
        )
        usage = usage_summary(
            root / "provider",
            execution_sha256=admitted.plan_sha256,
            empirical=True,
            sre=True,
            sre_profile=selected,
        )
        if not report.secondary_failures:
            publish_saved_native_report(
                root / "report",
                context,
                report,
                public_history=history,
                accounting=dict(
                    usage=usage,
                    provider_seconds=report.provider_seconds,
                    validation_seconds=report.validation_seconds,
                ),
            )
    owner._formal_binding(admitted, context)
    require_ops_code(plan)
    failures = _summary(body)
    if report.primary_failure is not None:
        detail = report.primary_failure.model_dump(mode="json")
        if failures["primary_failure"] is None:
            failures["primary_failure"] = detail
        else:
            failures["secondary_failures"].append(detail)
    failures["secondary_failures"].extend(
        r.model_dump(mode="json") for r in report.secondary_failures
    )
    return owner._finish(admitted, core, report.channels, failure_summary=failures)
