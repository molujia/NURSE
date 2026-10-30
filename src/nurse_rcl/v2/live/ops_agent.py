"""Embedded HERMES feedback loop with an independent, resumable action ledger."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Any

from hermes_runtime.contracts import NormalizedResponse, ToolCall
from hermes_runtime.loop import run_conversation

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_agent_decision import context_json, is_stop, parse_decision
from nurse_rcl.v2.live.ops_agent_state import (
    OpsAgentResult,
    OpsD21DecisionRecord,
    OpsDecisionRecord,
    OpsDirectedStageResult,
    OpsEmpiricalDecisionRecord,
    OpsNavigationAgentResult,
    OpsReplanningAgentResult,
    OpsReplanningDecisionRecord,
    OpsSreDecisionRecord,
    OpsWorkingAgentResult,
    OpsWorkingDecisionRecord,
    SreConversationFields,
    load_conversation,
    save_conversation,
)
from nurse_rcl.v2.live.ops_context_view import investigation_input
from nurse_rcl.v2.live.ops_contracts import SreProfileId, resolve_sre_profile
from nurse_rcl.v2.live.ops_exchange import prepare_ops_decision
from nurse_rcl.v2.live.ops_failure import (
    FailureStage,
    OpsFailureDetail,
    failure_detail,
    secondary_failures,
)
from nurse_rcl.v2.live.ops_indexed_tools import OpsIndexedInvestigationLedger, OpsIndexedToolbox
from nurse_rcl.v2.live.ops_investigation_ledger import OpsInvestigationLedger
from nurse_rcl.v2.live.ops_sre_context import parse_sre_decision, sre_investigation_input
from nurse_rcl.v2.live.ops_sre_exchange import prepare_sre_decision
from nurse_rcl.v2.live.ops_sre_ledger import OpsSreInvestigationLedger
from nurse_rcl.v2.live.ops_sre_tools import OpsSreToolbox
from nurse_rcl.v2.live.ops_sre_working_tools import (
    OpsSreWorkingInvestigationLedger,
    OpsSreWorkingToolbox,
)
from nurse_rcl.v2.live.ops_tools import OpsToolbox

OpsCompletion = Callable[[int, dict[str, Any] | bytes], bytes]


class _ProviderUnavailable(Exception):
    pass


class _NoProgress(Exception):
    def __init__(self, value: dict[str, Any], proof: dict[str, Any]) -> None:
        self.value, self.proof = value, proof


class _InputUnavailable(Exception):
    def __init__(self, detail: OpsFailureDetail) -> None:
        self.detail = detail
        super().__init__("investigation_input_unavailable")


def run_ops_agent(
    *,
    root: Path,
    toolbox: OpsToolbox,
    code_sha256: str,
    complete: OpsCompletion,
    recover_complete: Callable[[int, bytes], bytes | None] | None = None,
    refresh: Callable[[], None] | None = None,
    empirical: bool = False,
    opaque_case_id: str | None = None,
    sre: bool = False,
    working: bool = False,
    sre_profile: SreProfileId | None = None,
    stage_decision_limit: int | None = None,
    feedback_fact: bool = False,
) -> OpsAgentResult:
    """Provider receives detached JSON, never toolbox/core objects or paths.

    The ledger lock also owns conversation state. Replaying saved assistant and
    tool messages does not spend model requests or perform tools again. The
    production provider must enforce durable attempts in its own request slot.
    """
    profile = resolve_sre_profile(sre=sre, working=working, profile=sre_profile)
    focused = profile == "hermes-sre-tools/25"
    atomic_selector = profile == "hermes-sre-tools/23" or focused
    if feedback_fact and not (
        empirical and sre and atomic_selector and not focused and stage_decision_limit is not None
    ):
        raise ValueError("feedback presentation requires explicit bounded atomic SRE owner")
    typed_purpose = profile == "hermes-sre-tools/22"
    single_owner = profile == "hermes-sre-tools/21"
    observable = (
        profile == "hermes-sre-tools/20" or single_owner or typed_purpose or atomic_selector
    )
    entry = profile == "hermes-sre-tools/19" or observable
    directed = profile == "hermes-sre-tools/18" or entry
    from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
    from nurse_rcl.v2.live.ops_sre_references import SreReferenceBook

    reference_book_type = SreActionReferenceBook if entry else SreReferenceBook
    if (
        directed and (type(stage_decision_limit) is not int or not 1 <= stage_decision_limit <= 8)
    ) or (not directed and stage_decision_limit is not None):
        raise ValueError("stage decision limit requires explicit directed bounded owner")
    decision_ceiling = 8 if stage_decision_limit is None else stage_decision_limit
    action_first = profile == "hermes-sre-tools/17" or directed
    question_led = profile in {"hermes-sre-tools/16", "hermes-sre-tools/17"} or directed
    replanning = profile == "hermes-sre-tools/15" or question_led
    balanced = profile == "hermes-sre-tools/14" or replanning
    receipt_reuse = profile == "hermes-sre-tools/13" or balanced
    used_source = profile == "hermes-sre-tools/12" or receipt_reuse
    workbench = profile == "hermes-sre-tools/11" or used_source
    navigation = profile == "hermes-sre-tools/10" or workbench
    comparable = profile == "hermes-sre-tools/9" or navigation
    recovery = profile == "hermes-sre-tools/8" or comparable
    condition = profile == "hermes-sre-tools/7" or recovery
    brief = profile in {"hermes-sre-tools/6", "hermes-sre-tools/7"} or recovery
    working = profile == "hermes-sre-tools/5"
    durable = (
        profile in {"hermes-sre-tools/5", "hermes-sre-tools/6", "hermes-sre-tools/7"} or recovery
    )
    if sre and not empirical:
        raise ValueError("SRE investigation requires empirical wire policy")
    if working and not sre:
        raise ValueError("working investigation requires SRE policy")
    owner: OpsInvestigationLedger
    if sre:
        from nurse_rcl.v2.live.ops_sre_action_tools import (
            OpsSreActionInvestigationLedger,
            OpsSreActionToolbox,
        )
        from nurse_rcl.v2.live.ops_sre_atomic_selector_tools import (
            OpsSreAtomicSelectorInvestigationLedger,
            OpsSreAtomicSelectorToolbox,
        )
        from nurse_rcl.v2.live.ops_sre_brief_tools import (
            OpsSreBriefInvestigationLedger,
            OpsSreBriefToolbox,
        )
        from nurse_rcl.v2.live.ops_sre_checkpoint_pools_ledger import (
            OpsSrePooledInvestigationLedger,
        )
        from nurse_rcl.v2.live.ops_sre_comparable_tools import (
            OpsSreComparableInvestigationLedger,
            OpsSreComparableToolbox,
        )
        from nurse_rcl.v2.live.ops_sre_condition_tools import (
            OpsSreConditionInvestigationLedger,
            OpsSreConditionToolbox,
        )
        from nurse_rcl.v2.live.ops_sre_d21_tools import (
            OpsSreUsedSourceInvestigationLedger,
            OpsSreUsedSourceToolbox,
        )
        from nurse_rcl.v2.live.ops_sre_d22_tools import (
            OpsSreReceiptReuseInvestigationLedger,
            OpsSreReceiptReuseToolbox,
        )
        from nurse_rcl.v2.live.ops_sre_d23_tools import (
            OpsSreBalancedInvestigationLedger,
            OpsSreBalancedToolbox,
        )
        from nurse_rcl.v2.live.ops_sre_d25_tools import (
            OpsSreReplanningInvestigationLedger,
            OpsSreReplanningToolbox,
        )
        from nurse_rcl.v2.live.ops_sre_d26_tools import (
            OpsSreQuestionToolbox,
        )
        from nurse_rcl.v2.live.ops_sre_directed_tools import (
            OpsSreDirectedInvestigationLedger,
            OpsSreDirectedToolbox,
        )
        from nurse_rcl.v2.live.ops_sre_entry_tools import (
            OpsSreEntryInvestigationLedger,
            OpsSreEntryToolbox,
        )
        from nurse_rcl.v2.live.ops_sre_navigation_tools import (
            OpsSreNavigationInvestigationLedger,
            OpsSreNavigationToolbox,
            receipt_history,
        )
        from nurse_rcl.v2.live.ops_sre_observable_tools import (
            OpsSreObservableInvestigationLedger,
            OpsSreObservableToolbox,
        )
        from nurse_rcl.v2.live.ops_sre_recovery_tools import (
            OpsSreRecoveryInvestigationLedger,
            OpsSreRecoveryToolbox,
        )
        from nurse_rcl.v2.live.ops_sre_single_owner_tools import (
            OpsSreSingleOwnerInvestigationLedger,
            OpsSreSingleOwnerToolbox,
        )
        from nurse_rcl.v2.live.ops_sre_typed_purpose_tools import (
            OpsSreTypedPurposeInvestigationLedger,
            OpsSreTypedPurposeToolbox,
        )
        from nurse_rcl.v2.live.ops_sre_workbench_tools import (
            OpsSreWorkbenchInvestigationLedger,
            OpsSreWorkbenchToolbox,
        )

        if focused:
            from nurse_rcl.v2.live.ops_sre_focused_tools import (
                OpsSreFocusedInvestigationLedger,
                OpsSreFocusedToolbox,
            )

        toolbox_type = (
            OpsSreFocusedToolbox
            if focused
            else OpsSreAtomicSelectorToolbox
            if atomic_selector
            else OpsSreTypedPurposeToolbox
            if typed_purpose
            else OpsSreSingleOwnerToolbox
            if single_owner
            else OpsSreObservableToolbox
            if observable
            else OpsSreEntryToolbox
            if entry
            else OpsSreDirectedToolbox
            if directed
            else OpsSreActionToolbox
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
            if brief
            else OpsSreWorkingToolbox
            if working
            else OpsSreToolbox
        )
        if type(toolbox) is not toolbox_type or opaque_case_id is None:
            raise ValueError("SRE agent requires SRE tools and case identity")
        ledger_type = (
            OpsSreFocusedInvestigationLedger
            if focused
            else OpsSreAtomicSelectorInvestigationLedger
            if atomic_selector
            else OpsSreTypedPurposeInvestigationLedger
            if typed_purpose
            else OpsSreSingleOwnerInvestigationLedger
            if single_owner
            else OpsSreObservableInvestigationLedger
            if observable
            else OpsSreEntryInvestigationLedger
            if entry
            else OpsSreDirectedInvestigationLedger
            if directed
            else OpsSreActionInvestigationLedger
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
            if brief
            else OpsSreWorkingInvestigationLedger
            if working
            else OpsSreInvestigationLedger
        )
        owner = ledger_type(root=root, toolbox=toolbox, code_sha256=code_sha256)
    elif empirical:
        if (
            not isinstance(toolbox, OpsIndexedToolbox)
            or not toolbox.soft_targets
            or (opaque_case_id is None)
        ):
            raise ValueError("empirical agent requires indexed soft views and case identity")
        owner = OpsIndexedInvestigationLedger(root=root, toolbox=toolbox, code_sha256=code_sha256)
    else:
        owner = OpsInvestigationLedger(root=root, toolbox=toolbox, code_sha256=code_sha256)
    path = root / "investigation" / "conversation.json"
    prebound_state = None
    if receipt_reuse and (path.exists() or path.is_symlink()):
        # D22 restore checks current card admission before the ledger can open.
        # Bind only visibility from the validated, same-run conversation; after
        # acquiring the ledger lock below, detect a concurrent prefix change.
        from nurse_rcl.v2.live.ops_sre_references import SreReferenceBook

        assert opaque_case_id is not None and isinstance(toolbox, OpsSreRecoveryToolbox)
        prebound_state = load_conversation(
            path,
            snapshot=toolbox.case_snapshot_id,
            code=code_sha256,
            empirical=empirical,
            opaque_case_id=opaque_case_id,
            sre=sre,
            working=working,
            sre_profile=profile,
        )
        assert isinstance(prebound_state, SreConversationFields)
        if atomic_selector and not focused:
            from nurse_rcl.v2.live.ops_sre_feedback_fact_entry import require_feedback_history

            require_feedback_history(
                [
                    wire
                    for row in prebound_state.decisions
                    for wire in (
                        getattr(row, "wire_json", None),
                        getattr(row, "correction_wire_json", None),
                    )
                    if wire is not None
                ],
                enabled=feedback_fact,
            )
        toolbox.bind_visible_references(
            book=reference_book_type.from_state(
                prebound_state.reference_state,
                snapshot=toolbox.case_snapshot_id,
                run_id=prebound_state.reference_state["run_id"],
            ),
            request_index=max(1, len(prebound_state.decisions)),
        )
    with owner as ledger:
        if not path.exists() and ledger.used:
            raise ValueError("investigation conversation missing for existing actions")
        state = load_conversation(
            path,
            snapshot=toolbox.case_snapshot_id,
            code=code_sha256,
            empirical=empirical,
            opaque_case_id=opaque_case_id,
            sre=sre,
            working=working,
            sre_profile=profile,
        )
        if prebound_state is not None and state.state_sha256 != prebound_state.state_sha256:
            raise ValueError("D22 conversation changed while binding replay visibility")
        if atomic_selector and not focused:
            from nurse_rcl.v2.live.ops_sre_feedback_fact_entry import require_feedback_history

            require_feedback_history(
                [
                    wire
                    for row in state.decisions
                    for wire in (
                        getattr(row, "wire_json", None),
                        getattr(row, "correction_wire_json", None),
                    )
                    if wire is not None
                ],
                enabled=feedback_fact,
            )
        if directed and len(state.decisions) > decision_ceiling:
            raise ValueError("stage decision limit is below the saved prefix")
        if recovery:
            from nurse_rcl.v2.live.ops_sre_references import SreReferenceBook

            assert isinstance(state, SreConversationFields)
            assert isinstance(toolbox, OpsSreRecoveryToolbox)
            toolbox.bind_visible_references(
                book=reference_book_type.from_state(
                    state.reference_state,
                    snapshot=toolbox.case_snapshot_id,
                    run_id=state.reference_state["run_id"],
                ),
                request_index=max(1, len(state.decisions)),
            )

        def parse_value(raw: bytes, context: str) -> dict[str, Any]:
            if focused:
                from nurse_rcl.v2.live.ops_sre_focused_decision import parse_focused_decision

                return parse_focused_decision(raw, context=json.loads(context))
            if atomic_selector:
                from nurse_rcl.v2.live.ops_sre_atomic_selector_decision import parse_atomic_decision

                return parse_atomic_decision(raw, context=json.loads(context))
            if typed_purpose:
                from nurse_rcl.v2.live.ops_sre_typed_purpose_decision import (
                    parse_typed_purpose_decision,
                )

                return parse_typed_purpose_decision(raw, context=json.loads(context))
            if single_owner:
                from nurse_rcl.v2.live.ops_sre_single_owner_decision import (
                    parse_single_owner_decision,
                )

                return parse_single_owner_decision(raw, context=json.loads(context))
            if observable:
                from nurse_rcl.v2.live.ops_sre_entry_decision import parse_observable_decision

                return parse_observable_decision(raw, context=json.loads(context))
            if entry:
                from nurse_rcl.v2.live.ops_sre_entry_decision import parse_entry_decision

                return parse_entry_decision(raw, context=json.loads(context))
            if question_led:
                from nurse_rcl.v2.live.ops_sre_d26_decision import parse_question_decision

                return parse_question_decision(raw, context=json.loads(context))
            if balanced:
                from nurse_rcl.v2.live.ops_sre_d23_decision import parse_balanced_decision

                return parse_balanced_decision(raw, context=json.loads(context))
            if condition:
                from nurse_rcl.v2.live.ops_sre_feedback_context import parse_feedback_decision

                return parse_feedback_decision(raw, context=json.loads(context))
            if brief:
                from nurse_rcl.v2.live.ops_sre_brief_context import parse_brief_decision

                return parse_brief_decision(raw, context=json.loads(context))
            return (
                parse_sre_decision(raw, context=json.loads(context))
                if sre
                else parse_decision(raw, empirical=empirical)
            )

        declared = [r for r in state.decisions if r.action_index is not None]
        if ledger.used > len(declared) or ledger.used < max(0, len(declared) - 1):
            raise ValueError("investigation conversation/ledger sequence differs")
        for row, record in zip(declared, ledger.records, strict=False):
            assert row.response_json is not None
            actions = parse_value(row.response_json.encode("utf-8"), row.context_json)["actions"]
            if sha256_hex(canonical_json_bytes(actions)) != record.input_sha256:
                raise ValueError("investigation action receipt differs")
        if navigation:
            from nurse_rcl.v2.live.ops_sre_delivered_navigation import verify_navigation_source

            assert isinstance(toolbox, OpsSreNavigationToolbox)
            used = 0
            for row in state.decisions:
                context = json.loads(row.context_json)
                if single_owner or typed_purpose or atomic_selector:
                    from nurse_rcl.v2.live.ops_sre_encoding import expand

                    # New identity authenticates the complete last supplied
                    # receipt, not only its derived index or presentation aids.
                    actual = receipt_history(ledger.records[:used])
                    previous = actual[-1]["view"] if actual else None
                    saved_previous = context.get("previous_feedback")
                    if canonical_json_bytes(
                        None if previous is None else expand(previous)
                    ) != canonical_json_bytes(
                        None if saved_previous is None else expand(saved_previous)
                    ):
                        raise ValueError("saved single-owner feedback source differs")
                expected = verify_navigation_source(
                    receipt_history(ledger.records[:used]), source=toolbox.checkpoint_source
                )
                if context.get("delivered_navigation") != expected:
                    raise ValueError("saved delivered navigation source differs")
                if workbench:
                    # load_conversation already validated the SRE leaf above;
                    # repeat the narrowing for this independently gated branch.
                    assert isinstance(state, SreConversationFields)
                    from nurse_rcl.v2.live.ops_sre_context import _memory
                    from nurse_rcl.v2.live.ops_sre_workbench import (
                        delivered_record_index,
                        select_public_memory,
                    )

                    prior = state.decisions[: row.index - 1]
                    memory = [
                        _memory(
                            r,
                            parse_value(r.response_json.encode(), r.context_json),
                            references=state.reference_state,
                        )
                        for r in prior
                        if r.response_json is not None
                    ]
                    if replanning:
                        assert isinstance(toolbox, OpsSreBalancedToolbox)
                        from nurse_rcl.v2.live.ops_sre_d25_progress import (
                            build_investigation_progress,
                            select_replanning_memory,
                        )

                        selection = select_replanning_memory(memory)
                        expected_progress = build_investigation_progress(
                            context["actions"],
                            receipt_history(ledger.records[:used]),
                            delivered_record_index(receipt_history(ledger.records[:used])),
                            context["post_action_state"],
                            role_source_resolver=toolbox.role_source_members,
                        )
                        if context.get("investigation_progress") != expected_progress:
                            raise ValueError("D25 saved progress source differs")
                    else:
                        selection = select_public_memory(memory)
                    if question_led:
                        from nurse_rcl.v2.live.ops_sre_atomic_selector_questions import (
                            project_atomic_questions,
                        )
                        from nurse_rcl.v2.live.ops_sre_d26_questions import project_question_state
                        from nurse_rcl.v2.live.ops_sre_d26_semantics import diagnostic_semantics
                        from nurse_rcl.v2.live.ops_sre_observable_scope import (
                            project_observable_questions,
                        )
                        from nurse_rcl.v2.live.ops_sre_typed_purpose_questions import (
                            project_typed_questions,
                        )

                        if focused:
                            from nurse_rcl.v2.live.ops_sre_focused_questions import (
                                project_focused_questions,
                            )

                        if context.get("question_state") != (
                            project_focused_questions(memory)
                            if focused
                            else project_atomic_questions(memory)
                            if atomic_selector
                            else project_typed_questions(memory)
                            if typed_purpose
                            else project_observable_questions(memory)
                            if observable
                            else project_question_state(
                                memory,
                                entry_field="entry_ref" if entry else "card_id",
                            )
                        ) or (
                            context.get("diagnostic_semantics")
                            != diagnostic_semantics(receipt_history(ledger.records[:used]))
                        ):
                            raise ValueError("D26 saved question or semantics source differs")
                    if (
                        context.get("delivered_record_index")
                        != delivered_record_index(
                            receipt_history(ledger.records[:used]),
                            question_state=context["question_state"]
                            if observable
                            and context["delivered_record_index"]["schema_version"]
                            == "nurse.v2.sre-delivered-record-index/2"
                            else None,
                        )
                        or context.get("public_working_memory") != selection
                    ):
                        raise ValueError("saved workbench source differs")
                used += row.action_index is not None
        if replanning and state.decisions:
            last = state.decisions[-1]
            if isinstance(last, OpsReplanningDecisionRecord) and (
                last.blocked_read_proof is not None
            ):
                from nurse_rcl.v2.live.ops_sre_d25_progress import exact_replanning_read

                if entry:
                    from nurse_rcl.v2.live.ops_sre_entry_decision import exact_entry_replanning_read

                    assert isinstance(state, SreConversationFields)
                    expected_proof = exact_entry_replanning_read(
                        json.loads(last.response_json or "{}"),
                        ledger.records,
                        toolbox=toolbox,
                        decisions=state.decisions,
                        context=json.loads(last.context_json),
                        book=SreActionReferenceBook.from_state(
                            state.reference_state,
                            snapshot=toolbox.case_snapshot_id,
                            run_id=state.reference_state["run_id"],
                        ),
                    )
                else:
                    expected_proof = exact_replanning_read(
                        json.loads(last.response_json or "{}"),
                        ledger.records,
                        toolbox=toolbox,
                        decisions=state.decisions,
                        context=json.loads(last.context_json),
                    )
                if expected_proof != last.blocked_read_proof:
                    raise ValueError("D25 saved blocked read source differs")
                if state.result is None:
                    restored = OpsReplanningAgentResult(
                        termination="no_progress_after_correction",
                        decision_count=len(state.decisions),
                        action_count=ledger.used,
                        investigation_status="partial",
                        messages=state.initial_messages,
                    )
                    state = state.model_copy(update={"result": restored})
                    save_conversation(path, state)
        if state.result is not None:
            if state.result.action_count != ledger.used:
                raise ValueError("investigation completion/ledger differs")
            if isinstance(state.result, OpsWorkingAgentResult) and (
                state.result.termination == "unavailable"
                and state.result.investigation_status
                != (
                    "partial"
                    if any(r.status == "completed" for r in ledger.records)
                    else "unavailable"
                )
            ):
                raise ValueError("working terminal evidence status differs")
            return state.result
        if state.decisions and refresh is not None:
            # A fresh toolbox has not admitted core cards/receipts referenced
            # by an already frozen, possibly pending request. Restore the
            # read-only publication before replay/dispatch, without rebuilding
            # any saved context or refunding a request/action slot.
            refresh()
        save_conversation(path, state)
        cursor = 0
        trace = state.initial_messages
        stage: FailureStage = "investigation_input"

        def replace_row(row: OpsDecisionRecord) -> None:
            nonlocal state
            rows = list(state.decisions)
            rows[row.index - 1] = row
            state = state.model_copy(update={"decisions": tuple(rows)})
            save_conversation(path, state)

        def next_response(
            messages: tuple[dict[str, Any], ...],
            schemas: tuple[dict[str, Any], ...],
        ) -> NormalizedResponse:
            nonlocal state, cursor, trace, stage
            del schemas
            trace = messages
            stage = "investigation_input"
            row: OpsDecisionRecord
            cursor += 1
            if cursor > 8:
                raise ValueError("decision budget exhausted")
            if cursor > len(state.decisions):
                if refresh is not None:
                    refresh()  # Owner-only publication read, never a model tool.
                if empirical:
                    assert isinstance(ledger, OpsIndexedInvestigationLedger)
                    assert opaque_case_id is not None
                    references = (
                        json.loads(canonical_json_bytes(state.reference_state))
                        if isinstance(state, SreConversationFields)
                        else None
                    )

                    def make_wire(view: dict[str, Any]) -> dict[str, Any]:
                        if focused:
                            from nurse_rcl.v2.live.ops_sre_focused_exchange import (
                                prepare_focused_decision,
                            )

                            return dict(
                                json.loads(
                                    prepare_focused_decision(
                                        context=view,
                                        references=references,
                                        opaque_case_id=opaque_case_id,
                                        case_snapshot_id=toolbox.case_snapshot_id,
                                        decision_index=cursor,
                                    )
                                )
                            )
                        if feedback_fact:
                            from nurse_rcl.v2.live.ops_sre_feedback_fact_entry import (
                                prepare_entry_decision,
                            )

                            assert references is not None and opaque_case_id is not None
                            return dict(
                                json.loads(
                                    prepare_entry_decision(
                                        view,
                                        references=references,
                                        case_id=opaque_case_id,
                                        snapshot=toolbox.case_snapshot_id,
                                        index=cursor,
                                    )
                                )
                            )
                        if brief:
                            from nurse_rcl.v2.live.ops_sre_brief_exchange import (
                                prepare_brief_decision,
                            )

                            return dict(
                                json.loads(
                                    prepare_brief_decision(
                                        opaque_case_id=opaque_case_id,
                                        case_snapshot_id=toolbox.case_snapshot_id,
                                        decision_index=cursor,
                                        context=view,
                                        references=references,
                                    )
                                )
                            )
                        if sre:
                            return dict(
                                json.loads(
                                    prepare_sre_decision(
                                        opaque_case_id=opaque_case_id,
                                        case_snapshot_id=toolbox.case_snapshot_id,
                                        decision_index=cursor,
                                        context=view,
                                        references=references,
                                        working=working,
                                    )
                                )
                            )
                        raw, _ = prepare_ops_decision(
                            opaque_case_id=opaque_case_id,
                            case_snapshot_id=toolbox.case_snapshot_id,
                            decision_index=cursor,
                            context=view,
                            empirical=True,
                        )
                        return dict(json.loads(raw))

                    projection = (
                        sre_investigation_input(
                            capabilities=ledger.capabilities(),
                            records=ledger.records,
                            decisions=state.decisions,
                            decision_index=cursor,
                            make_wire=make_wire,
                            navigation_source=toolbox.checkpoint_source
                            if isinstance(toolbox, OpsSreNavigationToolbox)
                            else None,
                            role_source_resolver=toolbox.role_source_members
                            if isinstance(toolbox, OpsSreBalancedToolbox)
                            else None,
                            entry_references=references if entry else None,
                            entry_case_id=opaque_case_id if entry else None,
                        )
                        if sre
                        else investigation_input(
                            capabilities=ledger.capabilities(),
                            records=ledger.records,
                            decision_index=cursor,
                            make_wire=make_wire,
                            soft_targets=True,
                        )
                    )
                    if projection.wire_bytes is None or projection.view_json is None:
                        if durable:
                            detail = getattr(projection, "failure_detail", None) or failure_detail(
                                "investigation_input", ValueError("input_unavailable")
                            )
                            raise _InputUnavailable(detail)
                        raise ValueError("investigation input " + projection.status)
                    record_type = (
                        OpsReplanningDecisionRecord
                        if replanning
                        else OpsD21DecisionRecord
                        if used_source
                        else OpsWorkingDecisionRecord
                        if durable
                        else OpsSreDecisionRecord
                        if sre
                        else OpsEmpiricalDecisionRecord
                    )
                    row = record_type(
                        index=cursor,
                        context_json=projection.view_json,
                        wire_json=projection.wire_bytes.decode("utf-8"),
                        wire_sha256=sha256_hex(projection.wire_bytes),
                        target_exceeded=projection.target_exceeded,
                        expansion_reasons=projection.expansion_reasons,
                    )
                    if sre:
                        state = state.model_copy(update={"reference_state": references})
                else:
                    row = OpsDecisionRecord(
                        index=cursor,
                        context_json=context_json(
                            messages=messages,
                            capabilities=toolbox.capabilities(),
                            decision_index=cursor,
                            actions_used=ledger.used,
                        ),
                    )
                state = state.model_copy(update={"decisions": (*state.decisions, row)})
                save_conversation(path, state)  # Reserve BEFORE external work.
            row = state.decisions[cursor - 1]
            if row.response_json is None:
                stage = "investigation_provider"
                started = monotonic()
                try:
                    if isinstance(row, OpsWorkingDecisionRecord):
                        if row.call_started and row.native_response_json is None:
                            raise ValueError("saved investigation outcome unknown; no resend")
                        if not row.call_started:
                            row = row.model_copy(update={"call_started": True})
                            replace_row(row)
                    request = (
                        row.wire_json.encode("utf-8")
                        if isinstance(row, OpsEmpiricalDecisionRecord)
                        else json.loads(row.context_json)
                    )
                    native = (
                        row.native_response_json.encode()
                        if isinstance(row, OpsSreDecisionRecord)
                        and row.native_response_json is not None
                        else complete(cursor, request)
                    )
                    if sre:
                        from nurse_rcl.v2.live.ops_sre_model_view import resolve_model_response

                        assert isinstance(state, SreConversationFields)
                        assert isinstance(row, OpsSreDecisionRecord) and opaque_case_id is not None
                        # Save the untouched model JSON before translation or tool execution.
                        if durable:
                            row = row.model_copy(update={"native_response_json": native.decode()})
                            replace_row(row)
                        if entry:
                            from nurse_rcl.v2.live.ops_sre_brief_exchange import (
                                brief_context_from_wire,
                            )
                            from nurse_rcl.v2.live.ops_sre_entry_decision import (
                                parse_entry_decision,
                            )

                            if focused:
                                from nurse_rcl.v2.live.ops_sre_focused_decision import (
                                    parse_focused_decision as parse_entry_decision,
                                )
                            elif atomic_selector:
                                from nurse_rcl.v2.live.ops_sre_atomic_selector_decision import (
                                    parse_atomic_decision as parse_entry_decision,
                                )
                            elif typed_purpose:
                                from nurse_rcl.v2.live.ops_sre_typed_purpose_decision import (
                                    parse_typed_purpose_decision as parse_entry_decision,
                                )
                            elif single_owner:
                                from nurse_rcl.v2.live.ops_sre_single_owner_decision import (
                                    parse_single_owner_decision as parse_entry_decision,
                                )
                            elif observable:
                                from nurse_rcl.v2.live.ops_sre_entry_decision import (
                                    parse_observable_decision as parse_entry_decision,
                                )

                            native_value = parse_entry_decision(
                                native,
                                context=brief_context_from_wire(
                                    request,
                                    opaque_case_id=opaque_case_id,
                                    case_snapshot_id=toolbox.case_snapshot_id,
                                    decision_index=cursor,
                                ),
                            )
                        elif question_led:
                            from nurse_rcl.v2.live.ops_sre_brief_exchange import (
                                brief_context_from_wire,
                            )
                            from nurse_rcl.v2.live.ops_sre_d26_decision import (
                                parse_question_decision,
                            )

                            native_value = parse_question_decision(
                                native,
                                context=(
                                    brief_context_from_wire(
                                        request,
                                        opaque_case_id=opaque_case_id,
                                        case_snapshot_id=toolbox.case_snapshot_id,
                                        decision_index=cursor,
                                    )
                                ),
                            )
                        elif balanced:
                            from nurse_rcl.v2.live.ops_sre_d23_decision import (
                                parse_balanced_decision,
                            )

                            native_value = parse_balanced_decision(native)
                        elif condition:
                            from nurse_rcl.v2.live.ops_sre_feedback_context import (
                                parse_feedback_decision,
                            )

                            native_value = parse_feedback_decision(native)
                        elif brief:
                            from nurse_rcl.v2.live.ops_sre_brief_context import parse_brief_decision

                            native_value = parse_brief_decision(native)
                        else:
                            native_value = parse_sre_decision(native)
                        row = row.model_copy(update={"native_response_json": native.decode()})
                        replace_row(row)
                        packet = json.loads(json.loads(row.wire_json)["messages"][1]["content"])[
                            "context"
                        ]
                        resolved = resolve_model_response(
                            native_value,
                            packet=packet,
                            references=state.reference_state,
                            case_id=opaque_case_id,
                            snapshot=toolbox.case_snapshot_id,
                            index=cursor,
                            report=False,
                        )
                        value = parse_value(canonical_json_bytes(resolved), row.context_json)
                        if used_source and not is_stop(value):
                            from nurse_rcl.v2.live.ops_sre_brief_exchange import (
                                prepare_brief_correction,
                            )
                            from nurse_rcl.v2.live.ops_sre_d21_reuse import (
                                exact_delivered_ledger_read,
                            )

                            assert isinstance(row, OpsD21DecisionRecord)

                            def read_proof(proposal: dict[str, Any]) -> dict[str, Any] | None:
                                if entry:
                                    from nurse_rcl.v2.live.ops_sre_entry_decision import (
                                        exact_entry_replanning_read,
                                    )

                                    assert isinstance(state, SreConversationFields)
                                    return exact_entry_replanning_read(
                                        proposal,
                                        ledger.records,
                                        toolbox=toolbox,
                                        decisions=state.decisions,
                                        context=json.loads(row.context_json),
                                        book=SreActionReferenceBook.from_state(
                                            state.reference_state,
                                            snapshot=toolbox.case_snapshot_id,
                                            run_id=state.reference_state["run_id"],
                                        ),
                                    )
                                if replanning:
                                    from nurse_rcl.v2.live.ops_sre_d25_progress import (
                                        exact_replanning_read,
                                    )

                                    return exact_replanning_read(
                                        proposal,
                                        ledger.records,
                                        toolbox=toolbox,
                                        decisions=state.decisions,
                                        context=json.loads(row.context_json),
                                    )
                                return exact_delivered_ledger_read(
                                    proposal, ledger.records, toolbox=toolbox
                                )

                            proof = read_proof(value)
                            if (
                                proof is not None
                                and receipt_reuse
                                and not replanning
                                and any(
                                    isinstance(prior, OpsD21DecisionRecord)
                                    and prior.proposal_json is not None
                                    for prior in state.decisions[: cursor - 1]
                                )
                            ):
                                # D22 still reserves the proposed action, then replays
                                # its proved old receipt. No second correction is bought.
                                proof = None
                            if proof is not None:
                                if any(
                                    isinstance(prior, OpsD21DecisionRecord)
                                    and prior.proposal_json is not None
                                    for prior in state.decisions[: cursor - 1]
                                ):
                                    if replanning:
                                        raise _NoProgress(value, proof)
                                    raise ValueError("D21 correction already used in this case")
                                proposal_json = canonical_json_bytes(value).decode("utf-8")
                                if row.proposal_json is not None and (
                                    row.proposal_json != proposal_json
                                    or row.duplicate_proof != proof
                                ):
                                    raise ValueError("D21 saved duplicate proof differs")
                                if row.correction_wire_json is None:
                                    correction_wire = prepare_brief_correction(
                                        row.wire_json.encode(),
                                        native_proposal=native,
                                        proof=proof,
                                        opaque_case_id=opaque_case_id,
                                        case_snapshot_id=toolbox.case_snapshot_id,
                                        decision_index=cursor,
                                        references=state.reference_state,
                                    )
                                    row = row.model_copy(
                                        update={
                                            "proposal_json": proposal_json,
                                            "duplicate_proof": proof,
                                            "correction_wire_json": correction_wire.decode(),
                                            "correction_wire_sha256": sha256_hex(correction_wire),
                                        }
                                    )
                                    replace_row(row)
                                else:
                                    correction_wire = row.correction_wire_json.encode()
                                    if row.correction_wire_sha256 != sha256_hex(correction_wire):
                                        raise ValueError("D21 saved correction wire differs")
                                recovered_native = None
                                if row.correction_call_started and (
                                    row.correction_native_response_json is None
                                ):
                                    recovered_native = (
                                        recover_complete(cursor, correction_wire)
                                        if recover_complete is not None
                                        else None
                                    )
                                    if recovered_native is None:
                                        raise ValueError(
                                            "saved correction outcome unknown; no resend"
                                        )
                                if not row.correction_call_started:
                                    row = row.model_copy(update={"correction_call_started": True})
                                    replace_row(row)
                                correction_native = (
                                    row.correction_native_response_json.encode()
                                    if row.correction_native_response_json is not None
                                    else recovered_native
                                    if recovered_native is not None
                                    else complete(cursor, correction_wire)
                                )
                                row = row.model_copy(
                                    update=dict(
                                        correction_native_response_json=correction_native.decode(),
                                    )
                                )
                                replace_row(row)
                                correction_value = (
                                    parse_entry_decision(
                                        correction_native,
                                        context=brief_context_from_wire(
                                            correction_wire,
                                            opaque_case_id=opaque_case_id,
                                            case_snapshot_id=toolbox.case_snapshot_id,
                                            decision_index=cursor,
                                        ),
                                    )
                                    if entry
                                    else parse_question_decision(
                                        correction_native,
                                        context=brief_context_from_wire(
                                            correction_wire,
                                            opaque_case_id=opaque_case_id,
                                            case_snapshot_id=toolbox.case_snapshot_id,
                                            decision_index=cursor,
                                        ),
                                    )
                                    if question_led
                                    else (
                                        parse_balanced_decision(correction_native)
                                        if balanced
                                        else parse_feedback_decision(correction_native)
                                    )
                                )
                                correction_packet = json.loads(
                                    json.loads(correction_wire)["messages"][1]["content"]
                                )["context"]
                                corrected = resolve_model_response(
                                    correction_value,
                                    packet=correction_packet,
                                    references=state.reference_state,
                                    case_id=opaque_case_id,
                                    snapshot=toolbox.case_snapshot_id,
                                    index=cursor,
                                    report=False,
                                )
                                value = parse_value(
                                    canonical_json_bytes(corrected), row.context_json
                                )
                                repeated = read_proof(value)
                                if replanning and repeated is not None:
                                    raise _NoProgress(value, repeated)
                                if not receipt_reuse and repeated is not None:
                                    raise ValueError("D21 correction repeated no-new-content read")
                    else:
                        value = parse_value(native, row.context_json)
                except _NoProgress as exc:
                    assert isinstance(row, OpsReplanningDecisionRecord)
                    replace_row(
                        row.model_copy(
                            update={
                                "response_json": canonical_json_bytes(exc.value).decode(),
                                "blocked_read_proof": exc.proof,
                                "provider_seconds": row.provider_seconds + monotonic() - started,
                            }
                        )
                    )
                    raise
                except BaseException as exc:
                    if not durable and not isinstance(exc, Exception):
                        raise
                    replace_row(
                        row.model_copy(
                            update={
                                "failures": (*row.failures, type(exc).__name__),
                                "provider_seconds": row.provider_seconds + monotonic() - started,
                            }
                        )
                    )
                    if durable:
                        raise
                    raise _ProviderUnavailable(type(exc).__name__) from None
                row = row.model_copy(
                    update={
                        "response_json": canonical_json_bytes(value).decode("utf-8"),
                        "action_index": None if is_stop(value) else ledger.used + 1,
                        "provider_seconds": row.provider_seconds + monotonic() - started,
                    }
                )
                replace_row(row)  # Bind the response before any tool reservation.
            assert row.response_json is not None
            value = parse_value(row.response_json.encode("utf-8"), row.context_json)
            calls = (
                ()
                if is_stop(value)
                else (
                    ToolCall(
                        id=f"ops_decision_{cursor}",
                        name="ops_action",
                        arguments=canonical_json_bytes({"actions": value["actions"]}).decode(
                            "utf-8"
                        ),
                    ),
                )
            )
            return NormalizedResponse(
                content=row.response_json,
                tool_calls=calls,
                finish_reason="tool_calls" if calls else "stop",
            )

        def dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            nonlocal stage
            stage = "investigation_tool"
            row = state.decisions[cursor - 1]
            if name != "ops_action" or row.action_index is None:
                raise ValueError("unregistered investigation dispatch")
            index = row.action_index
            if index > ledger.used:
                if index != ledger.used + 1:
                    raise ValueError("investigation action sequence differs")
                if recovery:
                    assert isinstance(state, SreConversationFields)
                    assert isinstance(toolbox, OpsSreRecoveryToolbox)
                    toolbox.bind_visible_references(
                        book=reference_book_type.from_state(
                            state.reference_state,
                            snapshot=toolbox.case_snapshot_id,
                            run_id=state.reference_state["run_id"],
                        ),
                        request_index=row.index,
                    )
                if entry:
                    assert isinstance(ledger, OpsSreEntryInvestigationLedger)
                    assert isinstance(state, SreConversationFields) and opaque_case_id is not None
                    ledger.freeze_entries(
                        table=json.loads(row.context_json)["action_entries"],
                        book=SreActionReferenceBook.from_state(
                            state.reference_state,
                            snapshot=toolbox.case_snapshot_id,
                            run_id=state.reference_state["run_id"],
                        ),
                        request_index=row.index,
                        case_id=opaque_case_id,
                    )
                ledger.dispatch(arguments["actions"])
            record = ledger.records[index - 1]
            if durable and getattr(record, "failure_detail", None) is not None:
                raise _InputUnavailable(record.failure_detail)  # type: ignore[attr-defined]
            if record.result is not None:
                return record.result.model_dump(mode="json")
            if record.status != "interrupted":
                raise ValueError("investigation completion receipt missing")
            return dict(
                status="interrupted",
                action_index=index,
                limitations=["completion_unknown_no_refund", "no_factual_conclusion"],
            )

        try:
            run = run_conversation(
                messages=state.initial_messages,
                tools=(),
                complete=next_response,
                dispatch=dispatch,
                max_turns=decision_ceiling,
                return_at_turn_limit=True,
            )
        except _NoProgress:
            blocked = OpsReplanningAgentResult(
                termination="no_progress_after_correction",
                decision_count=len(state.decisions),
                action_count=ledger.used,
                investigation_status="partial",
                messages=trace,
            )
            state = state.model_copy(update={"result": blocked})
            save_conversation(path, state)
            return blocked
        except _ProviderUnavailable as exc:
            return OpsAgentResult(
                termination="unavailable",
                decision_count=len(state.decisions),
                action_count=ledger.used,
                messages=trace,
                failure_type=str(exc),
            )
        except BaseException as exc:
            if not durable:
                raise
            detail = (
                exc.detail if isinstance(exc, _InputUnavailable) else failure_detail(stage, exc)
            )
            if navigation:
                from nurse_rcl.v2.live.ops_sre_navigation_failure import navigation_failure

                detail = navigation_failure(
                    stage, exc, decision_index=cursor or None, detail=detail
                )
            failed_type: type[OpsWorkingAgentResult] = (
                OpsReplanningAgentResult
                if replanning
                else OpsNavigationAgentResult
                if navigation
                else OpsWorkingAgentResult
            )
            failed = failed_type(
                termination="unavailable",
                decision_count=len(state.decisions),
                action_count=ledger.used,
                messages=trace,
                failure_type=detail.failure_type,
                primary_failure=detail,
                secondary_failures=secondary_failures(exc),
                investigation_status="partial"
                if any(r.status == "completed" for r in ledger.records)
                else "unavailable",
            )
            state = state.model_copy(update={"result": failed})
            save_conversation(path, state)
            if not isinstance(exc, Exception):
                raise
            return failed
        stopped = state.decisions[-1].action_index is None
        if directed and not stopped and decision_ceiling < 8:
            # All successful requests/actions remain in the same durable prefix.
            # Do not save a terminal result; a later separately approved ceiling
            # replays this prefix without sending or dispatching it again.
            return OpsDirectedStageResult(
                termination="stage_pause",
                investigation_status="partial",
                decision_count=len(state.decisions),
                action_count=ledger.used,
                messages=run.messages,
            )
        result_type = (
            OpsReplanningAgentResult
            if replanning
            else OpsNavigationAgentResult
            if navigation
            else OpsWorkingAgentResult
            if durable
            else OpsAgentResult
        )
        fields: dict[str, Any] = {"investigation_status": "completed"} if durable else {}
        result = result_type(
            termination="model_stop" if stopped else "decision_budget",
            decision_count=len(state.decisions),
            action_count=ledger.used,
            messages=run.messages,
            **fields,
        )
        state = state.model_copy(update={"result": result})
        save_conversation(path, state)
        return result
