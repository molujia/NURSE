"""Small durable conversation receipts; raw telemetry and credentials stay out."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_agent_decision import initial_messages, is_stop, parse_decision
from nurse_rcl.v2.live.ops_contracts import SreProfileId, resolve_sre_profile
from nurse_rcl.v2.live.ops_failure import OpsFailureDetail
from nurse_rcl.v2.live.ops_sre_navigation_failure import OpsNavigationFailureDetail
from nurse_rcl.v2.live.repair_driver import _fsync


class OpsAgentResult(StrictFrozenModel):
    termination: Literal["model_stop", "decision_budget", "unavailable"]
    decision_count: int = Field(ge=0, le=8, strict=True)
    action_count: int = Field(ge=0, le=8, strict=True)
    messages: tuple[dict[str, Any], ...]
    failure_type: str | None = None


class OpsWorkingAgentResult(OpsAgentResult):
    investigation_status: Literal["completed", "partial", "unavailable"]
    primary_failure: OpsFailureDetail | None = None
    secondary_failures: tuple[OpsFailureDetail, ...] = Field(default=(), max_length=16)


class OpsNavigationAgentResult(OpsWorkingAgentResult):
    primary_failure: OpsNavigationFailureDetail | None = None


class OpsReplanningAgentResult(OpsNavigationAgentResult):
    termination: Literal[  # type: ignore[assignment]
        "model_stop", "decision_budget", "unavailable", "no_progress_after_correction"
    ]


class OpsDecisionRecord(StrictFrozenModel):
    index: int = Field(ge=1, le=8, strict=True)
    context_json: str
    response_json: str | None = None
    action_index: int | None = Field(default=None, ge=1, le=8, strict=True)
    provider_seconds: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    failures: tuple[str, ...] = ()


class OpsEmpiricalDecisionRecord(OpsDecisionRecord):
    wire_json: str
    wire_sha256: SHA256
    target_bytes: Literal[16384] = 16384
    target_exceeded: bool
    expansion_reasons: tuple[str, ...]


class OpsSreDecisionRecord(OpsEmpiricalDecisionRecord):
    native_response_json: str | None = None


class OpsWorkingDecisionRecord(OpsSreDecisionRecord):
    call_started: bool = False


class OpsD21DecisionRecord(OpsWorkingDecisionRecord):
    proposal_json: str | None = None
    duplicate_proof: dict[str, Any] | None = None
    correction_wire_json: str | None = None
    correction_wire_sha256: SHA256 | None = None
    correction_call_started: bool = False
    correction_native_response_json: str | None = None


class OpsReplanningDecisionRecord(OpsD21DecisionRecord):
    blocked_read_proof: dict[str, Any] | None = None


class OpsConversationBase(StrictFrozenModel):
    case_snapshot_id: SHA256
    code_sha256: SHA256
    initial_messages: tuple[dict[str, Any], ...]
    decisions: tuple[OpsDecisionRecord, ...] = Field(max_length=8)
    result: OpsAgentResult | None = None
    state_sha256: SHA256 = "0" * 64


class OpsConversation(OpsConversationBase):
    schema_version: Literal["nurse.v2.ops-conversation/1"] = "nurse.v2.ops-conversation/1"


class OpsEmpiricalConversation(OpsConversationBase):
    schema_version: Literal["nurse.v2.ops-conversation/2"] = "nurse.v2.ops-conversation/2"
    usage_policy_id: Literal["ops-empirical-input-cost/1"] = "ops-empirical-input-cost/1"
    opaque_case_id: str
    decisions: tuple[OpsEmpiricalDecisionRecord, ...] = Field(max_length=8)


class SreConversationFields(OpsConversationBase):
    usage_policy_id: Literal["ops-empirical-input-cost/1"] = "ops-empirical-input-cost/1"
    opaque_case_id: str
    decisions: tuple[OpsSreDecisionRecord, ...] = Field(max_length=8)
    reference_state: dict[str, Any]


class OpsSreConversation(SreConversationFields):
    schema_version: Literal["nurse.v2.sre-conversation/2"] = "nurse.v2.sre-conversation/2"
    investigation_profile_id: Literal["hermes-sre-tools/4"] = "hermes-sre-tools/4"


class OpsWorkingConversation(SreConversationFields):
    schema_version: Literal["nurse.v2.sre-conversation/3"] = "nurse.v2.sre-conversation/3"
    investigation_profile_id: Literal["hermes-sre-tools/5"] = "hermes-sre-tools/5"
    decisions: tuple[OpsWorkingDecisionRecord, ...] = Field(max_length=8)
    result: OpsWorkingAgentResult | None = None


class OpsBriefConversation(SreConversationFields):
    schema_version: Literal["nurse.v2.sre-conversation/4"] = "nurse.v2.sre-conversation/4"
    investigation_profile_id: Literal["hermes-sre-tools/6"] = "hermes-sre-tools/6"
    decisions: tuple[OpsWorkingDecisionRecord, ...] = Field(max_length=8)
    result: OpsWorkingAgentResult | None = None


class OpsConditionConversation(SreConversationFields):
    schema_version: Literal["nurse.v2.sre-conversation/5"] = "nurse.v2.sre-conversation/5"
    investigation_profile_id: Literal["hermes-sre-tools/7"] = "hermes-sre-tools/7"
    decisions: tuple[OpsWorkingDecisionRecord, ...] = Field(max_length=8)
    result: OpsWorkingAgentResult | None = None


class OpsRecoveryConversation(SreConversationFields):
    schema_version: Literal["nurse.v2.sre-conversation/6"] = "nurse.v2.sre-conversation/6"
    investigation_profile_id: Literal["hermes-sre-tools/8"] = "hermes-sre-tools/8"
    decisions: tuple[OpsWorkingDecisionRecord, ...] = Field(max_length=8)
    result: OpsWorkingAgentResult | None = None


class OpsComparableConversation(SreConversationFields):
    schema_version: Literal["nurse.v2.sre-conversation/7"] = "nurse.v2.sre-conversation/7"
    investigation_profile_id: Literal["hermes-sre-tools/9"] = "hermes-sre-tools/9"
    decisions: tuple[OpsWorkingDecisionRecord, ...] = Field(max_length=8)
    result: OpsWorkingAgentResult | None = None


class OpsNavigationConversation(SreConversationFields):
    schema_version: Literal["nurse.v2.sre-conversation/8"] = "nurse.v2.sre-conversation/8"
    investigation_profile_id: Literal["hermes-sre-tools/10"] = "hermes-sre-tools/10"
    decisions: tuple[OpsWorkingDecisionRecord, ...] = Field(max_length=8)
    result: OpsNavigationAgentResult | None = None


class OpsWorkbenchConversation(SreConversationFields):
    schema_version: Literal["nurse.v2.sre-conversation/9"] = "nurse.v2.sre-conversation/9"
    investigation_profile_id: Literal["hermes-sre-tools/11"] = "hermes-sre-tools/11"
    decisions: tuple[OpsWorkingDecisionRecord, ...] = Field(max_length=8)
    result: OpsNavigationAgentResult | None = None


class OpsUsedSourceConversation(SreConversationFields):
    schema_version: Literal["nurse.v2.sre-conversation/10"] = "nurse.v2.sre-conversation/10"
    investigation_profile_id: Literal["hermes-sre-tools/12"] = "hermes-sre-tools/12"
    decisions: tuple[OpsD21DecisionRecord, ...] = Field(max_length=8)
    result: OpsNavigationAgentResult | None = None


class OpsReceiptReuseConversation(SreConversationFields):
    schema_version: Literal["nurse.v2.sre-conversation/11"] = "nurse.v2.sre-conversation/11"
    investigation_profile_id: Literal["hermes-sre-tools/13"] = "hermes-sre-tools/13"
    decisions: tuple[OpsD21DecisionRecord, ...] = Field(max_length=8)
    result: OpsNavigationAgentResult | None = None


class OpsBalancedConversation(SreConversationFields):
    schema_version: Literal["nurse.v2.sre-conversation/12"] = "nurse.v2.sre-conversation/12"
    investigation_profile_id: Literal["hermes-sre-tools/14"] = "hermes-sre-tools/14"
    decisions: tuple[OpsD21DecisionRecord, ...] = Field(max_length=8)
    result: OpsNavigationAgentResult | None = None


class OpsReplanningConversation(SreConversationFields):
    schema_version: Literal["nurse.v2.sre-conversation/13"] = "nurse.v2.sre-conversation/13"
    investigation_profile_id: Literal["hermes-sre-tools/15"] = "hermes-sre-tools/15"
    decisions: tuple[OpsReplanningDecisionRecord, ...] = Field(max_length=8)
    result: OpsReplanningAgentResult | None = None


class OpsQuestionConversation(OpsReplanningConversation):
    schema_version: Literal["nurse.v2.sre-conversation/14"] = "nurse.v2.sre-conversation/14"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/16"] = "hermes-sre-tools/16"  # type: ignore[assignment]


class OpsActionConversation(OpsQuestionConversation):
    schema_version: Literal["nurse.v2.sre-conversation/15"] = "nurse.v2.sre-conversation/15"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/17"] = "hermes-sre-tools/17"  # type: ignore[assignment]


class OpsDirectedConversation(OpsActionConversation):
    schema_version: Literal["nurse.v2.sre-conversation/16"] = "nurse.v2.sre-conversation/16"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/18"] = "hermes-sre-tools/18"  # type: ignore[assignment]


class OpsEntryConversation(OpsDirectedConversation):
    schema_version: Literal["nurse.v2.sre-conversation/17"] = "nurse.v2.sre-conversation/17"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/19"] = "hermes-sre-tools/19"  # type: ignore[assignment]


class OpsObservableConversation(OpsEntryConversation):
    schema_version: Literal["nurse.v2.sre-conversation/18"] = "nurse.v2.sre-conversation/18"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/20"] = "hermes-sre-tools/20"  # type: ignore[assignment]


class OpsSingleOwnerConversation(OpsEntryConversation):
    schema_version: Literal["nurse.v2.sre-conversation/19"] = "nurse.v2.sre-conversation/19"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/21"] = "hermes-sre-tools/21"  # type: ignore[assignment]


class OpsAtomicSelectorConversation(OpsEntryConversation):
    schema_version: Literal["nurse.v2.sre-conversation/21"] = "nurse.v2.sre-conversation/21"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/23"] = "hermes-sre-tools/23"  # type: ignore[assignment]


class OpsFocusedConversation(OpsAtomicSelectorConversation):
    schema_version: Literal["nurse.v2.sre-conversation/22"] = "nurse.v2.sre-conversation/22"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/25"] = "hermes-sre-tools/25"  # type: ignore[assignment]


class OpsTypedPurposeConversation(OpsEntryConversation):
    schema_version: Literal["nurse.v2.sre-conversation/20"] = "nurse.v2.sre-conversation/20"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/22"] = "hermes-sre-tools/22"  # type: ignore[assignment]


class OpsDirectedStageResult(OpsReplanningAgentResult):
    """Stage boundary is not a terminal conversation receipt or a model judgement."""

    termination: Literal[  # type: ignore[assignment]
        "model_stop",
        "decision_budget",
        "unavailable",
        "no_progress_after_correction",
        "stage_pause",
    ]


def sre_run_id(root: Path, snapshot: str, code: str) -> str:
    return sha256_hex(
        canonical_json_bytes(
            dict(
                root=str(root.resolve()),
                snapshot=snapshot,
                code=code,
                schema="sre-reference-owner/1",
            )
        )
    )


def save_conversation(path: Path, state: OpsConversationBase) -> None:
    unsigned = state.model_dump(mode="json", exclude={"state_sha256"})
    sealed = state.model_copy(update={"state_sha256": sha256_hex(canonical_json_bytes(unsigned))})
    raw = canonical_json_bytes(sealed.model_dump(mode="json"))
    if isinstance(state, OpsEmpiricalConversation | SreConversationFields) and (
        len(raw) > 128 * 1024 * 1024
    ):
        raise ValueError("conversation storage resource limit")
    atomic_write_bytes(path, raw)
    _fsync(path.parent)


def load_conversation(
    path: Path,
    *,
    snapshot: str,
    code: str,
    empirical: bool = False,
    opaque_case_id: str | None = None,
    sre: bool = False,
    working: bool = False,
    sre_profile: SreProfileId | None = None,
) -> OpsConversationBase:
    profile = resolve_sre_profile(sre=sre, working=working, profile=sre_profile)
    focused = profile == "hermes-sre-tools/25"
    atomic_selector = profile == "hermes-sre-tools/23" or focused
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
    if working and not sre:
        raise ValueError("working conversation requires SRE policy")
    if sre and not empirical:
        raise ValueError("SRE conversation requires empirical wire policy")
    if empirical and opaque_case_id is None:
        raise ValueError("empirical conversation requires case identity")
    state: OpsConversationBase
    if not (path.exists() or path.is_symlink()):
        if sre:
            from nurse_rcl.v2.live.ops_sre_exchange import initial_sre_messages
            from nurse_rcl.v2.live.ops_sre_references import SreReferenceBook

            assert opaque_case_id is not None
            from nurse_rcl.v2.live.ops_sre_brief_exchange import initial_brief_messages

            state_type = (
                OpsFocusedConversation
                if focused
                else OpsAtomicSelectorConversation
                if atomic_selector
                else OpsTypedPurposeConversation
                if typed_purpose
                else OpsSingleOwnerConversation
                if single_owner
                else OpsObservableConversation
                if observable
                else OpsEntryConversation
                if entry
                else OpsDirectedConversation
                if directed
                else OpsActionConversation
                if action_first
                else OpsQuestionConversation
                if question_led
                else OpsReplanningConversation
                if replanning
                else OpsBalancedConversation
                if balanced
                else OpsReceiptReuseConversation
                if receipt_reuse
                else OpsUsedSourceConversation
                if used_source
                else OpsWorkbenchConversation
                if workbench
                else OpsNavigationConversation
                if navigation
                else OpsComparableConversation
                if comparable
                else OpsRecoveryConversation
                if recovery
                else OpsConditionConversation
                if condition
                else OpsBriefConversation
                if brief
                else OpsWorkingConversation
                if working
                else OpsSreConversation
            )
            return state_type(
                case_snapshot_id=snapshot,
                code_sha256=code,
                opaque_case_id=opaque_case_id,
                initial_messages=initial_brief_messages(
                    condition_explanations=condition,
                    recovery=recovery,
                    comparable=comparable,
                    navigation=navigation,
                    workbench=workbench,
                    used_source=used_source,
                    receipt_reuse=receipt_reuse,
                    balanced=balanced,
                    replanning=replanning,
                    question_led=question_led,
                    action_first=action_first,
                    directed=directed,
                    entry=entry,
                    observable=observable,
                    single_owner=single_owner,
                    typed_purpose=typed_purpose,
                    atomic_selector=atomic_selector,
                    focused=focused,
                )
                if brief
                else initial_sre_messages(working=working),
                decisions=(),
                reference_state=reference_book_type(
                    snapshot=snapshot, run_id=sre_run_id(path.parent.parent, snapshot, code)
                ).to_state(),
            )
        if empirical:
            assert opaque_case_id is not None
            return OpsEmpiricalConversation(
                case_snapshot_id=snapshot,
                code_sha256=code,
                opaque_case_id=opaque_case_id,
                initial_messages=initial_messages(empirical=True),
                decisions=(),
            )
        return OpsConversation(
            case_snapshot_id=snapshot,
            code_sha256=code,
            initial_messages=initial_messages(),
            decisions=(),
        )
    raw = _read_bytes(path, (128 if empirical else 48) * 1024 * 1024)
    state = (
        OpsFocusedConversation.model_validate_json(raw)
        if focused
        else OpsAtomicSelectorConversation.model_validate_json(raw)
        if atomic_selector
        else OpsTypedPurposeConversation.model_validate_json(raw)
        if typed_purpose
        else OpsSingleOwnerConversation.model_validate_json(raw)
        if single_owner
        else OpsObservableConversation.model_validate_json(raw)
        if observable
        else OpsEntryConversation.model_validate_json(raw)
        if entry
        else OpsDirectedConversation.model_validate_json(raw)
        if directed
        else OpsActionConversation.model_validate_json(raw)
        if action_first
        else OpsQuestionConversation.model_validate_json(raw)
        if question_led
        else OpsReplanningConversation.model_validate_json(raw)
        if replanning
        else OpsBalancedConversation.model_validate_json(raw)
        if balanced
        else OpsReceiptReuseConversation.model_validate_json(raw)
        if receipt_reuse
        else OpsUsedSourceConversation.model_validate_json(raw)
        if used_source
        else OpsWorkbenchConversation.model_validate_json(raw)
        if workbench
        else OpsNavigationConversation.model_validate_json(raw)
        if navigation
        else OpsComparableConversation.model_validate_json(raw)
        if comparable
        else OpsRecoveryConversation.model_validate_json(raw)
        if recovery
        else OpsConditionConversation.model_validate_json(raw)
        if condition
        else OpsBriefConversation.model_validate_json(raw)
        if brief
        else OpsWorkingConversation.model_validate_json(raw)
        if working
        else OpsSreConversation.model_validate_json(raw)
        if sre
        else OpsEmpiricalConversation.model_validate_json(raw)
        if empirical
        else OpsConversation.model_validate_json(raw)
    )
    if isinstance(state, OpsEmpiricalConversation | SreConversationFields) and (
        state.opaque_case_id != opaque_case_id
    ):
        raise ValueError("investigation conversation case identity differs")
    if state.case_snapshot_id != snapshot or state.code_sha256 != code:
        raise ValueError("investigation conversation identity differs")
    if isinstance(state, SreConversationFields):
        from nurse_rcl.v2.live.ops_sre_references import SreReferenceBook

        reference_book_type.from_state(
            state.reference_state,
            snapshot=snapshot,
            run_id=sre_run_id(path.parent.parent, snapshot, code),
        )
    if raw != canonical_json_bytes(state.model_dump(mode="json")) or (
        state.state_sha256
        != sha256_hex(
            canonical_json_bytes(
                state.model_dump(
                    mode="json",
                    exclude={"state_sha256"},
                )
            )
        )
    ):
        raise ValueError("investigation conversation digest differs")
    action_count = 0
    for index, row in enumerate(state.decisions, 1):
        if row.index != index or (row.response_json is None and index != len(state.decisions)):
            raise ValueError("investigation decision sequence differs")
        if empirical:
            from nurse_rcl.v2.live.ops_exchange import bind_empirical_decision

            assert isinstance(row, OpsEmpiricalDecisionRecord) and opaque_case_id is not None
            wire = row.wire_json.encode("utf-8")
            if sre:
                from nurse_rcl.v2.live.ops_sre_encoding import expand
                from nurse_rcl.v2.live.ops_sre_exchange import sre_context_from_wire

                source_refs = (
                    state.reference_state if isinstance(state, SreConversationFields) else None
                )
                if brief:
                    from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_context_from_wire

                    context = brief_context_from_wire(
                        wire,
                        opaque_case_id=opaque_case_id,
                        case_snapshot_id=snapshot,
                        decision_index=index,
                        references=source_refs,
                    )
                else:
                    context = sre_context_from_wire(
                        wire,
                        opaque_case_id=opaque_case_id,
                        case_snapshot_id=snapshot,
                        decision_index=index,
                        references=source_refs,
                        working=working,
                    )
                request_pin = sha256_hex(wire)
                context_pin = sha256_hex(canonical_json_bytes(context))
                stored_context_pin = sha256_hex(
                    canonical_json_bytes(expand(json.loads(row.context_json)))
                )
            else:
                binding = bind_empirical_decision(
                    wire,
                    opaque_case_id=opaque_case_id,
                    case_snapshot_id=snapshot,
                    decision_index=index,
                )
                request_pin, context_pin = binding.request_sha256, binding.context_sha256
                stored_context_pin = sha256_hex(row.context_json.encode("utf-8"))
            if request_pin != row.wire_sha256 or (
                context_pin != stored_context_pin
                or row.target_exceeded != (len(wire) > row.target_bytes)
                or bool(row.expansion_reasons) != row.target_exceeded
            ):
                raise ValueError("investigation frozen wire or metadata differs")
        if used_source and isinstance(row, OpsD21DecisionRecord):
            if row.proposal_json is None:
                if any(
                    (
                        row.duplicate_proof is not None,
                        row.correction_wire_json is not None,
                        row.correction_wire_sha256 is not None,
                        row.correction_call_started,
                        row.correction_native_response_json is not None,
                    )
                ):
                    raise ValueError("D21 correction lacks original proposal")
            else:
                if (
                    row.native_response_json is None
                    or row.duplicate_proof is None
                    or (row.correction_wire_json is None or row.correction_wire_sha256 is None)
                ):
                    raise ValueError("D21 correction source is incomplete")
                from nurse_rcl.v2.live.ops_sre_brief_exchange import prepare_brief_correction
                from nurse_rcl.v2.live.ops_sre_model_view import resolve_model_response

                if not isinstance(state, SreConversationFields) or opaque_case_id is None:
                    raise ValueError("D21 correction source identity differs")
                original_packet = json.loads(json.loads(row.wire_json)["messages"][1]["content"])[
                    "context"
                ]
                proposed = resolve_model_response(
                    json.loads(row.native_response_json),
                    packet=original_packet,
                    references=state.reference_state,
                    case_id=opaque_case_id,
                    snapshot=snapshot,
                    index=index,
                    report=False,
                )
                expected_correction = prepare_brief_correction(
                    row.wire_json.encode(),
                    native_proposal=row.native_response_json.encode(),
                    proof=row.duplicate_proof,
                    opaque_case_id=opaque_case_id,
                    case_snapshot_id=snapshot,
                    decision_index=index,
                    references=state.reference_state,
                )
                if (
                    canonical_json_bytes(proposed).decode() != row.proposal_json
                    or expected_correction.decode() != row.correction_wire_json
                    or sha256_hex(expected_correction) != row.correction_wire_sha256
                    or (
                        row.correction_native_response_json is not None
                        and not row.correction_call_started
                    )
                ):
                    raise ValueError("D21 correction source binding differs")
        if row.response_json is not None:
            if sre:
                from nurse_rcl.v2.live.ops_sre_context import parse_sre_decision
                from nurse_rcl.v2.live.ops_sre_model_view import resolve_model_response

                assert isinstance(state, SreConversationFields) and opaque_case_id is not None
                assert isinstance(row, OpsSreDecisionRecord)
                if row.native_response_json is None:
                    raise ValueError("SRE native response missing")
                packet = json.loads(json.loads(row.wire_json)["messages"][1]["content"])["context"]
                resolved = resolve_model_response(
                    json.loads(row.native_response_json),
                    packet=packet,
                    references=state.reference_state,
                    case_id=opaque_case_id,
                    snapshot=snapshot,
                    index=index,
                    report=False,
                )
                if (
                    used_source
                    and isinstance(row, OpsD21DecisionRecord)
                    and (row.proposal_json is not None)
                ):
                    if canonical_json_bytes(resolved).decode() != row.proposal_json or (
                        row.duplicate_proof is None
                        or row.correction_wire_json is None
                        or row.correction_wire_sha256
                        != sha256_hex(row.correction_wire_json.encode())
                    ):
                        raise ValueError("D21 original proposal/correction binding differs")
                    from nurse_rcl.v2.live.ops_sre_brief_exchange import bind_brief_decision

                    correction_binding = bind_brief_decision(
                        row.correction_wire_json.encode(),
                        opaque_case_id=opaque_case_id,
                        case_snapshot_id=snapshot,
                        decision_index=index,
                    )
                    if correction_binding.phase != "investigation_correction":
                        raise ValueError("D21 correction phase differs")
                    if row.correction_native_response_json is not None:
                        packet = json.loads(
                            json.loads(row.correction_wire_json)["messages"][1]["content"]
                        )["context"]
                        resolved = resolve_model_response(
                            json.loads(row.correction_native_response_json),
                            packet=packet,
                            references=state.reference_state,
                            case_id=opaque_case_id,
                            snapshot=snapshot,
                            index=index,
                            report=False,
                        )
                    if not row.correction_call_started:
                        raise ValueError("D21 correction lacks started request")
                elif (
                    used_source
                    and isinstance(row, OpsD21DecisionRecord)
                    and (row.correction_wire_json is not None or row.duplicate_proof is not None)
                ):
                    raise ValueError("D21 correction lacks original proposal")
                if canonical_json_bytes(resolved).decode() != row.response_json:
                    raise ValueError("SRE native/resolved response binding differs")

                parser = parse_sre_decision
                if brief:
                    from nurse_rcl.v2.live.ops_sre_brief_context import parse_brief_decision

                    parser = parse_brief_decision
                if condition:
                    from nurse_rcl.v2.live.ops_sre_feedback_context import parse_feedback_decision

                    parser = parse_feedback_decision
                if balanced:
                    from nurse_rcl.v2.live.ops_sre_d23_decision import parse_balanced_decision

                    parser = parse_balanced_decision
                if question_led:
                    from nurse_rcl.v2.live.ops_sre_d26_decision import parse_question_decision

                    parser = parse_question_decision
                if entry:
                    from nurse_rcl.v2.live.ops_sre_entry_decision import parse_entry_decision

                    parser = parse_entry_decision
                    if focused:
                        from nurse_rcl.v2.live.ops_sre_focused_decision import (
                            parse_focused_decision,
                        )

                        parser = parse_focused_decision
                    elif atomic_selector:
                        from nurse_rcl.v2.live.ops_sre_atomic_selector_decision import (
                            parse_atomic_decision,
                        )

                        parser = parse_atomic_decision
                    elif typed_purpose:
                        from nurse_rcl.v2.live.ops_sre_typed_purpose_decision import (
                            parse_typed_purpose_decision,
                        )

                        parser = parse_typed_purpose_decision
                    elif single_owner:
                        from nurse_rcl.v2.live.ops_sre_single_owner_decision import (
                            parse_single_owner_decision,
                        )

                        parser = parse_single_owner_decision
                    elif observable:
                        from nurse_rcl.v2.live.ops_sre_entry_decision import (
                            parse_observable_decision,
                        )

                        parser = parse_observable_decision
                stopped = is_stop(parser(row.response_json.encode("utf-8"), context=context))
            else:
                stopped = is_stop(
                    parse_decision(row.response_json.encode("utf-8"), empirical=empirical)
                )
            blocked = (
                replanning
                and isinstance(row, OpsReplanningDecisionRecord)
                and (row.blocked_read_proof is not None)
            )
            if blocked:
                assert isinstance(row, OpsReplanningDecisionRecord)
                assert row.blocked_read_proof is not None
                if (
                    stopped
                    or index != len(state.decisions)
                    or not any(
                        isinstance(prior, OpsD21DecisionRecord) and prior.correction_call_started
                        for prior in state.decisions
                    )
                ):
                    raise ValueError("D25 blocked read lacks unique correction")
                from nurse_rcl.v2.live.ops_sre_d25_progress import validate_blocked_read

                if focused:
                    from nurse_rcl.v2.live.ops_sre_focused_decision import (
                        validate_focused_blocked_entry_read as validate_blocked_read,
                    )
                elif atomic_selector:
                    from nurse_rcl.v2.live.ops_sre_atomic_selector_decision import (
                        validate_atomic_blocked_entry_read as validate_blocked_read,
                    )
                elif typed_purpose:
                    from nurse_rcl.v2.live.ops_sre_typed_purpose_decision import (
                        validate_typed_blocked_entry_read as validate_blocked_read,
                    )
                elif entry:
                    from nurse_rcl.v2.live.ops_sre_entry_decision import (
                        validate_blocked_entry_read as validate_blocked_read,
                    )

                validate_blocked_read(
                    json.loads(row.response_json),
                    row.blocked_read_proof,
                    context,
                )
            action_count += not stopped and not blocked
            if row.action_index != (None if stopped or blocked else action_count) or (
                stopped and index != len(state.decisions)
            ):
                raise ValueError("investigation action sequence differs")
        elif row.action_index is not None:
            raise ValueError("pending response cannot have an action")
    no_progress = (
        replanning
        and isinstance(state, OpsReplanningConversation)
        and state.result is not None
        and state.result.termination == "no_progress_after_correction"
    )
    blocked_rows = [
        row
        for row in state.decisions
        if isinstance(row, OpsReplanningDecisionRecord) and row.blocked_read_proof is not None
    ]
    if (
        blocked_rows
        and (
            len(blocked_rows) != 1
            or blocked_rows[0] is not state.decisions[-1]
            or blocked_rows[0].response_json is None
            or (state.result is not None and not no_progress)
        )
    ) or (no_progress and not blocked_rows):
        raise ValueError("D25 blocked read terminal differs")
    failed = (
        isinstance(
            state,
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
        and state.result is not None
        and state.result.termination == "unavailable"
    )
    if isinstance(
        state,
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
    ):
        if any(r.response_json is not None and not r.call_started for r in state.decisions):
            raise ValueError("working response lacks a started request")
        if state.result is not None and (
            failed != (state.result.primary_failure is not None)
            or (failed or no_progress) != (state.result.investigation_status != "completed")
        ):
            raise ValueError("working failure termination differs")
    if state.result is not None and (
        state.result.decision_count != len(state.decisions)
        or (
            state.result.action_count != action_count
            and not (
                failed
                and isinstance(state.result, OpsWorkingAgentResult)
                and state.result.action_count == action_count - 1
                and state.result.primary_failure is not None
                and state.result.primary_failure.stage == "investigation_tool"
                and state.decisions[-1].action_index == action_count
            )
        )
        or (not failed and state.result.termination == "unavailable")
        or (not failed and any(r.response_json is None for r in state.decisions))
    ):
        raise ValueError("investigation completion receipt differs")
    return state
