"""D12 report owner. A started request only resumes from saved output, never redraws.

Caller supplies admitted investigation references and history. Live runner wiring
is separate; neither callback authorization nor credentials are inferred here.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Any, Literal

from pydantic import Field, PrivateAttr

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.live.ark_bridge import NativePartialResponseError
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_agent_state import sre_run_id
from nurse_rcl.v2.live.ops_contracts import OpsChannelStatus
from nurse_rcl.v2.live.ops_failure import (
    OpsFailureDetail,
    failure_detail,
    retain_secondary,
    secondary_failures,
)
from nurse_rcl.v2.live.ops_report import OpsFrozenWireFields, OpsFrozenWireReportRun, _save
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS, OpsClaimCheck
from nurse_rcl.v2.live.ops_sre_brief_claims import (
    BriefClaimCheck,
)
from nurse_rcl.v2.live.ops_sre_brief_report import (
    ActionReportMetadata,
    AtomicReportMetadata,
    BalancedReportMetadata,
    BriefSreReportContext,
    ComparableReportMetadata,
    ConditionReportMetadata,
    NavigationReportMetadata,
    QuestionReportMetadata,
    ReceiptReuseReportMetadata,
    RecoveryReportMetadata,
    ReplanningReportMetadata,
    UsedSourceReportMetadata,
    WorkbenchReportMetadata,
    _require_brief_unchanged,
    encode_brief_report_storage,
)
from nurse_rcl.v2.live.ops_sre_comparable_claims import ComparableClaimCheck
from nurse_rcl.v2.live.ops_sre_context import WORKING_INPUT_SCHEMA
from nurse_rcl.v2.live.ops_sre_native_checks import check_native_sre_report
from nurse_rcl.v2.live.ops_sre_native_exchange import native_report_body, native_report_projection
from nurse_rcl.v2.live.ops_sre_native_model import native_model_book
from nurse_rcl.v2.live.ops_sre_native_render import (
    inspect_native_report_pair as inspect_native_report_pair,
)
from nurse_rcl.v2.live.ops_sre_native_render import (
    save_native_report_pair,
)
from nurse_rcl.v2.live.ops_sre_native_report import (
    NativeSreReportContext,
    encode_native_report_storage,
)
from nurse_rcl.v2.live.ops_sre_references import SreReferenceBook, decode_references
from nurse_rcl.v2.live.ops_sre_report import parse_sre_report
from nurse_rcl.v2.live.ops_sre_report_check_policy import (
    _AdmittedBriefChecks,
    admit_brief_checks,
    parse_checked_brief,
    response_admission,
)
from nurse_rcl.v2.live.repair_driver import _fsync
from nurse_rcl.v2.live.staged_case import require_plain_path

Completion = Callable[[bytes], bytes]
SavedCompletion = Callable[[bytes], bytes | None]


class SreReportOwnerFields(OpsFrozenWireFields):
    provided_step_refs: tuple[str, ...] = ()
    reference_state: dict[str, Any]
    investigation_reference_sha256: SHA256
    public_history_sha256: SHA256
    resolved_response_json: str | None = None
    call_started: bool = False
    primary_failure: OpsFailureDetail | None = None
    secondary_failures: tuple[OpsFailureDetail, ...] = Field(default=(), max_length=16)


class NativeSreReportRun(SreReportOwnerFields, OpsFrozenWireReportRun):
    schema_version: Literal["nurse.v2.sre-report-run/8"] = "nurse.v2.sre-report-run/8"
    investigation_profile_id: Literal["hermes-sre-tools/5"] = "hermes-sre-tools/5"


class BriefReportOwnerBase(SreReportOwnerFields):
    provided_brief_refs: tuple[str, ...] = ()


class BriefReportOwnerFields(BriefReportOwnerBase):
    checks: tuple[BriefClaimCheck, ...] = ()


class BriefSreReportRun(BriefReportOwnerFields):
    schema_version: Literal["nurse.v2.sre-report-run/9"] = "nurse.v2.sre-report-run/9"
    investigation_profile_id: Literal["hermes-sre-tools/6"] = "hermes-sre-tools/6"


class ConditionSreReportRun(BriefReportOwnerFields):
    schema_version: Literal["nurse.v2.sre-report-run/10"] = "nurse.v2.sre-report-run/10"
    investigation_profile_id: Literal["hermes-sre-tools/7"] = "hermes-sre-tools/7"


class RecoverySreReportRun(BriefReportOwnerFields):
    schema_version: Literal["nurse.v2.sre-report-run/11"] = "nurse.v2.sre-report-run/11"
    investigation_profile_id: Literal["hermes-sre-tools/8"] = "hermes-sre-tools/8"


class ComparableSreReportRun(BriefReportOwnerBase):
    schema_version: Literal["nurse.v2.sre-report-run/12"] = "nurse.v2.sre-report-run/12"
    investigation_profile_id: Literal["hermes-sre-tools/9"] = "hermes-sre-tools/9"
    checks: tuple[ComparableClaimCheck, ...] = ()


class NavigationSreReportRun(BriefReportOwnerBase):
    schema_version: Literal["nurse.v2.sre-report-run/13"] = "nurse.v2.sre-report-run/13"
    investigation_profile_id: Literal["hermes-sre-tools/10"] = "hermes-sre-tools/10"
    checks: tuple[ComparableClaimCheck, ...] = ()


class WorkbenchSreReportRun(BriefReportOwnerBase):
    schema_version: Literal["nurse.v2.sre-report-run/14"] = "nurse.v2.sre-report-run/14"
    investigation_profile_id: Literal["hermes-sre-tools/11"] = "hermes-sre-tools/11"
    checks: tuple[ComparableClaimCheck, ...] = ()


class UsedSourceSreReportRun(WorkbenchSreReportRun):
    # Runtime Pydantic identity is deliberately stricter than the inherited version.
    schema_version: Literal["nurse.v2.sre-report-run/15"] = "nurse.v2.sre-report-run/15"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/12"] = "hermes-sre-tools/12"  # type: ignore[assignment]


class ReceiptReuseSreReportRun(UsedSourceSreReportRun):
    schema_version: Literal["nurse.v2.sre-report-run/16"] = "nurse.v2.sre-report-run/16"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/13"] = "hermes-sre-tools/13"  # type: ignore[assignment]


class BalancedSreReportRun(ReceiptReuseSreReportRun):
    schema_version: Literal["nurse.v2.sre-report-run/17"] = "nurse.v2.sre-report-run/17"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/14"] = "hermes-sre-tools/14"  # type: ignore[assignment]


class ReplanningSreReportRun(BalancedSreReportRun):
    schema_version: Literal["nurse.v2.sre-report-run/18"] = "nurse.v2.sre-report-run/18"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/15"] = "hermes-sre-tools/15"  # type: ignore[assignment]


class QuestionSreReportRun(ReplanningSreReportRun):
    schema_version: Literal["nurse.v2.sre-report-run/19"] = "nurse.v2.sre-report-run/19"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/16"] = "hermes-sre-tools/16"  # type: ignore[assignment]


class ConsistentQuestionSreReportRun(QuestionSreReportRun):
    schema_version: Literal["nurse.v2.sre-report-run/20"] = "nurse.v2.sre-report-run/20"  # type: ignore[assignment]
    check_policy_json: str | None = None
    checks_binding_sha256: SHA256 | None = None
    _admitted_checks: _AdmittedBriefChecks | None = PrivateAttr(default=None)


class ActionSreReportRun(ConsistentQuestionSreReportRun):
    schema_version: Literal["nurse.v2.sre-report-run/21"] = "nurse.v2.sre-report-run/21"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/17"] = "hermes-sre-tools/17"  # type: ignore[assignment]


class AtomicSreReportRun(ConsistentQuestionSreReportRun):
    schema_version: Literal["nurse.v2.sre-report-run/27"] = "nurse.v2.sre-report-run/27"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/23"] = "hermes-sre-tools/23"  # type: ignore[assignment]
    provider_seconds: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    validation_seconds: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    accounting_history_json: str | None = None
    report_usage_json: str | None = None
    grounding_json: str | None = None


def _brief_state_type(
    context: BriefSreReportContext,
) -> (
    type[BriefSreReportRun]
    | type[ConditionSreReportRun]
    | type[RecoverySreReportRun]
    | type[ComparableSreReportRun]
    | type[NavigationSreReportRun]
    | type[WorkbenchSreReportRun]
    | type[ReceiptReuseSreReportRun]
    | type[BalancedSreReportRun]
    | type[ReplanningSreReportRun]
    | type[QuestionSreReportRun]
):
    if isinstance(
        context.metadata,
        ComparableReportMetadata | NavigationReportMetadata | WorkbenchReportMetadata,
    ):
        return (
            AtomicSreReportRun
            if type(context.metadata) is AtomicReportMetadata
            else ActionSreReportRun
            if isinstance(context.metadata, ActionReportMetadata)
            else ConsistentQuestionSreReportRun
            if isinstance(context.metadata, QuestionReportMetadata)
            else ReplanningSreReportRun
            if isinstance(context.metadata, ReplanningReportMetadata)
            else BalancedSreReportRun
            if isinstance(context.metadata, BalancedReportMetadata)
            else ReceiptReuseSreReportRun
            if isinstance(context.metadata, ReceiptReuseReportMetadata)
            else UsedSourceSreReportRun
            if isinstance(context.metadata, UsedSourceReportMetadata)
            else WorkbenchSreReportRun
            if isinstance(context.metadata, WorkbenchReportMetadata)
            else NavigationSreReportRun
            if isinstance(context.metadata, NavigationReportMetadata)
            else ComparableSreReportRun
        )
    if isinstance(context.metadata, RecoveryReportMetadata):
        return RecoverySreReportRun
    return (
        ConditionSreReportRun
        if isinstance(context.metadata, ConditionReportMetadata)
        else BriefSreReportRun
    )


ReportRun = (
    NativeSreReportRun
    | BriefSreReportRun
    | ConditionSreReportRun
    | RecoverySreReportRun
    | ComparableSreReportRun
    | NavigationSreReportRun
    | WorkbenchSreReportRun
    | ReceiptReuseSreReportRun
    | BalancedSreReportRun
    | ReplanningSreReportRun
    | QuestionSreReportRun
)
ReportContext = NativeSreReportContext | BriefSreReportContext


def _persist(path: Path, state: ReportRun) -> ReportRun:
    saved = _save(path, state)
    assert isinstance(
        saved,
        NativeSreReportRun
        | BriefSreReportRun
        | ConditionSreReportRun
        | RecoverySreReportRun
        | ComparableSreReportRun
        | NavigationSreReportRun
        | WorkbenchSreReportRun
        | ReceiptReuseSreReportRun
        | BalancedSreReportRun
        | ReplanningSreReportRun
        | QuestionSreReportRun,
    )
    return saved


def _check_wire(state: ReportRun, context: ReportContext) -> None:
    if isinstance(state, ConsistentQuestionSreReportRun) and (
        (state.check_policy_json is None) != (state.checks_binding_sha256 is None)
        or (state.resolved_response_json is None and state.checks_binding_sha256 is not None)
        or (state.checks and state.checks_binding_sha256 is None)
    ):
        raise ValueError("report check binding fields differ")
    if isinstance(state, BriefReportOwnerBase) != isinstance(context, BriefSreReportContext):
        raise ValueError("report owner/context protocol differs")
    if isinstance(context, BriefSreReportContext) and (
        state.investigation_profile_id != context.metadata.profile
    ):
        raise ValueError("brief report profile identity differs")
    if state.input_wire_json is None:
        if (
            state.input_wire_sha256 is not None
            or state.provided_fact_refs
            or (
                state.provided_step_refs
                or state.call_started
                or state.raw_response_json is not None
                or (isinstance(state, BriefReportOwnerBase) and state.provided_brief_refs)
            )
        ):
            raise ValueError("native report aliases or response without wire")
        return
    raw = state.input_wire_json.encode()
    if isinstance(context, BriefSreReportContext):
        from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_report_body, brief_wire_parts

        assert isinstance(state, BriefReportOwnerBase)
        body, store = brief_report_body(raw, references=state.reference_state)
        source, supplied = brief_wire_parts(context)
        expected_facts = supplied.refs()
        if state.provided_brief_refs != tuple(context.supply["provided_brief_refs"]):
            raise ValueError("brief frozen report supply binding differs")
    else:
        body, store = native_report_body(raw, references=state.reference_state)
        source = context.metadata.model_dump(mode="json")
        source["schema_version"] = "nurse.v2.sre-report-input/7"
        supplied = context.fact_store
        expected_facts = supplied.refs()
    metadata = body["context"]
    if (
        (sha256_hex(raw), body["context_sha256"], metadata["evidence"]["core_result_digest"])
        != (
            state.input_wire_sha256,
            state.context_sha256,
            context.core_result_digest,
        )
        or canonical_json_bytes(store.to_packet()) != canonical_json_bytes(supplied.to_packet())
        or any(
            canonical_json_bytes(metadata[key]) != canonical_json_bytes(value)
            for key, value in source.items()
        )
    ):
        raise ValueError("native frozen report source binding differs")
    if state.provided_fact_refs != expected_facts or (
        state.provided_step_refs != tuple(s["step_ref"] for s in context.metadata.steps)
        or state.target_exceeded != (len(raw) > state.target_bytes)
        or bool(state.expansion_reasons) != state.target_exceeded
    ):
        raise ValueError("native frozen report delivery differs")


def _write_bound(path: Path, raw: bytes, *, owner_exists: bool, limit: int) -> None:
    if len(raw) > limit:
        raise ValueError("native report artifact resource limit")
    if path.exists() or path.is_symlink():
        if _read_bytes(path, limit) != raw:
            raise ValueError("native report frozen artifact identity changed")
    elif owner_exists:
        raise ValueError("native report frozen artifact missing")
    else:
        atomic_write_bytes(path, raw)
        _fsync(path.parent)


def run_native_report(
    *,
    root: Path,
    context: NativeSreReportContext,
    code_sha256: str,
    references: dict[str, Any],
    complete: Completion | None,
    public_history: list[dict[str, Any]] | None = None,
    recover_saved: SavedCompletion | None = None,
) -> NativeSreReportRun:
    """Freeze with complete=None. A recovery callback must be strictly keyless/read-only."""
    if type(context) is not NativeSreReportContext or (
        context.metadata.investigation_input_schema != WORKING_INPUT_SCHEMA
    ):
        raise ValueError("native owner rejects diagnostic or old investigation material")
    result = _run_owned_report(
        root=root,
        context=context,
        code_sha256=code_sha256,
        references=references,
        complete=complete,
        public_history=public_history,
        recover_saved=recover_saved,
    )
    assert isinstance(result, NativeSreReportRun)
    return result


def run_brief_report(
    *,
    root: Path,
    context: BriefSreReportContext,
    code_sha256: str,
    references: dict[str, Any],
    complete: Completion | None,
    public_history: list[dict[str, Any]] | None = None,
    recover_saved: SavedCompletion | None = None,
) -> (
    BriefSreReportRun
    | ConditionSreReportRun
    | RecoverySreReportRun
    | ComparableSreReportRun
    | NavigationSreReportRun
    | WorkbenchSreReportRun
    | ReceiptReuseSreReportRun
):
    if type(context) is not BriefSreReportContext or (
        context.metadata.investigation_input_schema
        != (
            "nurse.v2.sre-investigation-input/18"
            if isinstance(context.metadata, ActionReportMetadata)
            else "nurse.v2.sre-investigation-input/17"
            if isinstance(context.metadata, QuestionReportMetadata)
            else "nurse.v2.sre-investigation-input/16"
            if isinstance(context.metadata, ReplanningReportMetadata)
            else "nurse.v2.sre-investigation-input/15"
            if isinstance(context.metadata, BalancedReportMetadata)
            else "nurse.v2.sre-investigation-input/14"
            if isinstance(context.metadata, ReceiptReuseReportMetadata)
            else "nurse.v2.sre-investigation-input/13"
            if isinstance(context.metadata, UsedSourceReportMetadata)
            else "nurse.v2.sre-investigation-input/12"
            if isinstance(context.metadata, WorkbenchReportMetadata)
            else "nurse.v2.sre-investigation-input/11"
            if isinstance(context.metadata, NavigationReportMetadata)
            else "nurse.v2.sre-investigation-input/10"
            if isinstance(
                context.metadata,
                ComparableReportMetadata | NavigationReportMetadata | WorkbenchReportMetadata,
            )
            else "nurse.v2.sre-investigation-input/9"
            if isinstance(context.metadata, RecoveryReportMetadata)
            else "nurse.v2.sre-investigation-input/8"
            if isinstance(context.metadata, ConditionReportMetadata)
            else "nurse.v2.sre-investigation-input/7"
        )
    ):
        raise ValueError("brief owner rejects diagnostic or old investigation material")
    result = _run_owned_report(
        root=root,
        context=context,
        code_sha256=code_sha256,
        references=references,
        complete=complete,
        public_history=public_history,
        recover_saved=recover_saved,
    )
    assert isinstance(
        result,
        BriefSreReportRun
        | ConditionSreReportRun
        | RecoverySreReportRun
        | ComparableSreReportRun
        | NavigationSreReportRun
        | WorkbenchSreReportRun
        | ReceiptReuseSreReportRun,
    )
    return result


def _context_bytes(context: ReportContext) -> bytes:
    return (
        encode_brief_report_storage(context)
        if isinstance(context, BriefSreReportContext)
        else encode_native_report_storage(context)
    )


def _run_owned_report(
    *,
    root: Path,
    context: ReportContext,
    code_sha256: str,
    references: dict[str, Any],
    complete: Completion | None,
    public_history: list[dict[str, Any]] | None,
    recover_saved: SavedCompletion | None,
    source_references: dict[str, Any] | None = None,
    accounting_history_json: str | None = None,
    accounting_usage: Callable[[], dict[str, Any]] | None = None,
) -> ReportRun:
    require_pin(code_sha256)
    header = context.metadata.evidence
    require_plain_path(root)
    book = SreReferenceBook.from_state(
        references,
        snapshot=header.case_snapshot_id,
        run_id=sre_run_id(root, header.case_snapshot_id, code_sha256),
    )
    if any(row[3] > 8 for row in book.to_state()["entries"]):
        raise ValueError("native investigation references contain report-only entries")
    raw_context = _context_bytes(context)
    raw_history = canonical_json_bytes(public_history)
    directory = root / "report"
    directory.mkdir(mode=0o700, exist_ok=True)
    require_plain_path(directory)
    lock = os.open(directory / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if source_references is not None:
                if not isinstance(context, BriefSreReportContext) or (
                    type(context.metadata) is not AtomicReportMetadata
                    or sha256_hex(canonical_json_bytes(source_references))
                    != context.metadata.source_identity["reference_state_sha256"]
                ):
                    raise ValueError("report source reference binding differs")
                _write_bound(
                    directory / "source-references.json",
                    canonical_json_bytes(source_references),
                    owner_exists=(directory / "result.json").exists(),
                    limit=REPORT_LIMITS.max_bytes,
                )
            result = _run(
                directory,
                context,
                code_sha256,
                book.to_state(),
                raw_context,
                raw_history,
                complete,
                recover_saved,
                accounting_history_json,
                accounting_usage,
            )
        except BaseException as primary:
            if isinstance(context, BriefSreReportContext):
                try:
                    _require_brief_unchanged(context, raw_context)
                except Exception as secondary:
                    retain_secondary(primary, "report_validation", secondary)
            raise
        if isinstance(context, BriefSreReportContext):
            _require_brief_unchanged(context, raw_context)
        return result
    finally:
        os.close(lock)


def _run(
    directory: Path,
    context: ReportContext,
    code: str,
    references: dict[str, Any],
    raw_context: bytes,
    raw_history: bytes,
    complete: Completion | None,
    recover_saved: SavedCompletion | None,
    accounting_history_json: str | None = None,
    accounting_usage: Callable[[], dict[str, Any]] | None = None,
) -> ReportRun:
    path = directory / "result.json"
    exists = path.exists() or path.is_symlink()
    _write_bound(
        directory / "context.json", raw_context, owner_exists=exists, limit=REPORT_LIMITS.max_bytes
    )
    _write_bound(
        directory / "public-history.json", raw_history, owner_exists=exists, limit=128 * 1024 * 1024
    )
    pins = (
        code,
        sha256_hex(raw_context),
        context.core_result_digest,
        sha256_hex(canonical_json_bytes(references)),
        sha256_hex(raw_history),
    )
    if exists:
        state = _read_state(path, context, pins)
        if isinstance(state, AtomicSreReportRun) and (
            state.accounting_history_json != accounting_history_json
        ):
            raise ValueError("report accounting history changed after freeze")
    else:
        extra: dict[str, Any] = {}
        state_type: (
            type[BriefSreReportRun]
            | type[ConditionSreReportRun]
            | type[RecoverySreReportRun]
            | type[ComparableSreReportRun]
            | type[NavigationSreReportRun]
            | type[WorkbenchSreReportRun]
            | type[ReceiptReuseSreReportRun]
            | type[NativeSreReportRun]
        )
        if isinstance(context, BriefSreReportContext):
            from nurse_rcl.v2.live.ops_sre_brief_exchange import _project_admitted_brief_report

            projection = _project_admitted_brief_report(
                context, retained=json.loads(raw_context), references=references
            )
            state_type = _brief_state_type(context)
            if state_type is AtomicSreReportRun:
                extra["accounting_history_json"] = accounting_history_json
            extra["provided_brief_refs"] = (
                ()
                if projection.wire_bytes is None
                else tuple(context.supply["provided_brief_refs"])
            )
        else:
            projection = native_report_projection(context, references=references)
            state_type = NativeSreReportRun
        wire = projection.wire_bytes
        state = _persist(
            path,
            state_type(
                code_sha256=code,
                context_sha256=pins[1],
                core_result_digest=pins[2],
                investigation_reference_sha256=pins[3],
                public_history_sha256=pins[4],
                reference_state=references,
                channels=OpsChannelStatus(
                    core="completed",
                    investigation=context.metadata.evidence.investigation_status,
                    report="pending",
                ),
                status="pending" if wire is not None else "unavailable",
                input_wire_json=None if wire is None else wire.decode(),
                input_wire_sha256=None if wire is None else sha256_hex(wire),
                provided_fact_refs=projection.provided_fact_refs,
                provided_step_refs=()
                if wire is None
                else tuple(s["step_ref"] for s in context.metadata.steps),
                target_exceeded=projection.target_exceeded,
                expansion_reasons=projection.expansion_reasons,
                primary_failure=getattr(projection, "failure_detail", None),
                failure_types=() if wire is not None else ("input_" + projection.status,),
                **extra,
            ),
        )
    SreReferenceBook.from_state(
        state.reference_state,
        snapshot=context.metadata.evidence.case_snapshot_id,
        run_id=sre_run_id(directory.parent, context.metadata.evidence.case_snapshot_id, code),
    )
    _check_wire(state, context)
    return _execute(
        directory,
        context,
        state,
        complete,
        recover_saved,
        public_history=json.loads(raw_history),
        brief_retained=json.loads(raw_context)
        if isinstance(context, BriefSreReportContext)
        else None,
        accounting_usage=accounting_usage,
    )


def _read_state(
    path: Path,
    context: ReportContext,
    pins: tuple[str, ...],
) -> ReportRun:
    raw = _read_bytes(path, 64 * 1024 * 1024)
    state = (
        _brief_state_type(context).model_validate_json(raw)
        if isinstance(context, BriefSreReportContext)
        else NativeSreReportRun.model_validate_json(raw)
    )
    if (
        state.code_sha256,
        state.context_sha256,
        state.core_result_digest,
        state.investigation_reference_sha256,
        state.public_history_sha256,
    ) != pins or (
        raw != canonical_json_bytes(state.model_dump(mode="json"))
        or state.state_sha256
        != sha256_hex(canonical_json_bytes(state.model_dump(mode="json", exclude={"state_sha256"})))
        or state.channels.core != "completed"
        or state.channels.investigation != context.metadata.evidence.investigation_status
        or state.channels.report != state.status
    ):
        raise ValueError("native report owner identity differs")
    return state


def inspect_saved_native_run(
    directory: Path,
    context: NativeSreReportContext,
    *,
    code: str,
    references: dict[str, Any],
    public_history: list[dict[str, Any]] | None,
) -> NativeSreReportRun:
    """Read a failed worker's saved channel without publishing or resending anything."""
    if type(context) is not NativeSreReportContext:
        raise ValueError("native context required")
    result = _inspect_saved_run(
        directory, context, code=code, references=references, public_history=public_history
    )
    assert isinstance(result, NativeSreReportRun)
    return result


def inspect_saved_brief_run(
    directory: Path,
    context: BriefSreReportContext,
    *,
    code: str,
    references: dict[str, Any],
    public_history: list[dict[str, Any]] | None,
) -> (
    BriefSreReportRun
    | ConditionSreReportRun
    | RecoverySreReportRun
    | ComparableSreReportRun
    | NavigationSreReportRun
    | WorkbenchSreReportRun
    | ReceiptReuseSreReportRun
):
    if type(context) is not BriefSreReportContext:
        raise ValueError("brief context required")
    result = _inspect_saved_run(
        directory, context, code=code, references=references, public_history=public_history
    )
    assert isinstance(
        result,
        BriefSreReportRun
        | ConditionSreReportRun
        | RecoverySreReportRun
        | ComparableSreReportRun
        | NavigationSreReportRun
        | WorkbenchSreReportRun
        | ReceiptReuseSreReportRun,
    )
    return result


def _inspect_saved_run(
    directory: Path,
    context: ReportContext,
    *,
    code: str,
    references: dict[str, Any],
    public_history: list[dict[str, Any]] | None,
) -> ReportRun:
    raw_context = _context_bytes(context)
    raw_history = canonical_json_bytes(public_history)
    if _read_bytes(directory / "context.json", REPORT_LIMITS.max_bytes) != raw_context or (
        _read_bytes(directory / "public-history.json", 128 * 1024 * 1024) != raw_history
    ):
        raise ValueError("native saved source binding differs")
    state = _read_state(
        directory / "result.json",
        context,
        (
            code,
            sha256_hex(raw_context),
            context.core_result_digest,
            sha256_hex(canonical_json_bytes(references)),
            sha256_hex(raw_history),
        ),
    )
    SreReferenceBook.from_state(
        state.reference_state,
        snapshot=context.metadata.evidence.case_snapshot_id,
        run_id=sre_run_id(directory.parent, context.metadata.evidence.case_snapshot_id, code),
    )
    _check_wire(state, context)
    if state.resolved_response_json is not None and (
        state.raw_response_json is None or _resolve(state, context) != state.resolved_response_json
    ):
        raise ValueError("native/resolved saved response binding differs")
    if state.status in {"completed", "partial"}:
        _check_terminal_summary(state, context)
    return state


def _resolve(state: ReportRun, context: ReportContext) -> str:
    assert state.raw_response_json is not None and state.input_wire_json is not None
    packet = json.loads(json.loads(state.input_wire_json)["messages"][1]["content"])["context"]
    header = context.metadata.evidence
    if isinstance(context, BriefSreReportContext):
        from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_model_book
        from nurse_rcl.v2.live.ops_sre_references import _translate

        response_raw = state.raw_response_json.encode()
        if isinstance(state, AtomicSreReportRun):
            from nurse_rcl.v2.live.ops_sre_atomic_report_expressions import (
                resolve_expression_response,
            )
            from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_report_body

            body, short_store = brief_report_body(state.input_wire_json.encode())
            response_raw = resolve_expression_response(
                response_raw, body["context"]["supply"], short_store,
            )
        native = parse_checked_brief(context, response_raw).model_dump(mode="json")
        book = brief_model_book(
            packet, references=state.reference_state, snapshot=header.case_snapshot_id
        )
        assert book is not None
        return canonical_json_bytes(
            _translate(
                native,
                book=book,
                visible_at=9,
                case_id=header.opaque_case_id,
                encode=False,
                limits=REPORT_LIMITS,
                brief_fields=True,
            )
        ).decode()
    native = parse_sre_report(state.raw_response_json.encode()).model_dump(mode="json")
    book = native_model_book(
        packet, references=state.reference_state, snapshot=header.case_snapshot_id
    )
    assert book is not None
    return canonical_json_bytes(
        decode_references(
            native, book=book, visible_at=9, case_id=header.opaque_case_id, limits=REPORT_LIMITS
        )
    ).decode()


def publish_saved_native_report(
    directory: Path,
    context: ReportContext,
    state: ReportRun,
    *,
    public_history: list[dict[str, Any]] | None = None,
    accounting: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _publish_saved_report(
        directory, context, state, public_history=public_history, accounting=accounting
    )


def _publish_saved_report(
    directory: Path,
    context: ReportContext,
    state: ReportRun,
    *,
    public_history: list[dict[str, Any]] | None = None,
    accounting: dict[str, Any] | None = None,
    brief_retained: dict[str, Any] | None = None,
    admitted_checks: _AdmittedBriefChecks | None = None,
) -> dict[str, Any]:
    if isinstance(context, BriefSreReportContext):
        from nurse_rcl.v2.live.ops_sre_brief_render import _save_admitted_brief_report_pair
        from nurse_rcl.v2.live.ops_sre_brief_report import validate_brief_report

        retained = validate_brief_report(context) if brief_retained is None else brief_retained
        _require_brief_unchanged(context, canonical_json_bytes(retained))

        candidate_brief = None
        checker_failed = (
            state.status == "partial"
            and not state.checks
            and state.primary_failure is not None
            and state.primary_failure.stage == "report_validation"
        )
        if state.resolved_response_json is not None and not checker_failed:
            try:
                if admitted_checks is None and isinstance(state, ConsistentQuestionSreReportRun):
                    admitted_checks = state._admitted_checks
                if admitted_checks is None:
                    admitted_checks = _check_terminal_summary(state, context)
                if admitted_checks is not None:
                    candidate_brief = parse_checked_brief(
                        context,
                        state.resolved_response_json.encode(),
                    )
            except ValueError:
                if isinstance(state, ConsistentQuestionSreReportRun):
                    raise
                candidate_brief = None
        if candidate_brief is None:
            admitted_checks = admit_brief_checks(context, None, retained=retained)
        assert admitted_checks is not None
        if candidate_brief is not None:
            _check_admission_summary(state, admitted_checks)
        return _save_admitted_brief_report_pair(
            directory,
            context=context,
            retained=retained,
            response=candidate_brief,
            admitted_checks=admitted_checks,
            raw_response_json=state.raw_response_json,
            references=state.reference_state,
            public_history=public_history,
            accounting=accounting
            or dict(
                report_status=state.status,
                provider_seconds=state.provider_seconds,
                validation_seconds=state.validation_seconds,
            ),
        )
    response = None
    if state.resolved_response_json is not None:
        try:
            candidate = parse_sre_report(state.resolved_response_json.encode())
            check_native_sre_report(
                candidate,
                context=context,
                provided_fact_refs=state.provided_fact_refs,
                provided_step_refs=state.provided_step_refs,
            )
            response = candidate
        except ValueError:
            pass  # Invalid output remains preserved, not promoted to an observation.
    return save_native_report_pair(
        directory,
        context=context,
        response=response,
        raw_response_json=state.raw_response_json,
        references=state.reference_state,
        provided_fact_refs=state.provided_fact_refs,
        provided_step_refs=state.provided_step_refs,
        public_history=public_history,
        accounting=accounting
        or dict(
            report_status=state.status,
            provider_seconds=state.provider_seconds,
            validation_seconds=state.validation_seconds,
        ),
    )


def _response_checks(
    raw: str, state: ReportRun, context: ReportContext
) -> tuple[OpsClaimCheck, ...] | tuple[BriefClaimCheck, ...] | tuple[ComparableClaimCheck, ...]:
    if isinstance(context, BriefSreReportContext):
        return response_admission(context, raw).checks
    return check_native_sre_report(
        parse_sre_report(raw.encode()),
        context=context,
        provided_fact_refs=state.provided_fact_refs,
        provided_step_refs=state.provided_step_refs,
    )


def _check_admission_summary(state: ReportRun, admitted: _AdmittedBriefChecks) -> None:
    if canonical_json_bytes([c.model_dump(mode="json") for c in state.checks]) != (
        canonical_json_bytes([c.model_dump(mode="json") for c in admitted.checks])
    ) or (
        isinstance(state, ConsistentQuestionSreReportRun)
        and (
            (state.check_policy_json, state.checks_binding_sha256)
            != (admitted.policy_json, admitted.binding_sha256)
        )
    ):
        raise ValueError("saved report checks policy or binding differs")


def _check_terminal_summary(
    state: ReportRun,
    context: ReportContext,
) -> _AdmittedBriefChecks | None:
    admitted = None
    checks: (
        tuple[OpsClaimCheck, ...] | tuple[BriefClaimCheck, ...] | tuple[ComparableClaimCheck, ...]
    )
    if state.resolved_response_json is None:
        if isinstance(state, AtomicSreReportRun) and state.grounding_json is not None:
            raise ValueError("grounding diagnostics without a resolved response")
        if state.status == "completed" or state.checks or state.model_delivered_count:
            raise ValueError("native report summary lacks original response")
        if isinstance(state, ConsistentQuestionSreReportRun) and (
            state.check_policy_json is not None or state.checks_binding_sha256 is not None
        ):
            raise ValueError("report checks binding without response")
        return None
    try:
        if isinstance(context, BriefSreReportContext):
            admitted = response_admission(context, state.resolved_response_json)
            checks = admitted.checks
        else:
            checks = _response_checks(state.resolved_response_json, state, context)
    except ValueError as exc:
        if (
            state.status == "partial"
            and not state.checks
            and not state.model_delivered_count
            and (type(exc).__name__ in state.failure_types)
        ):
            return None  # A saved checker failure is not silently upgraded to a valid report.
        raise
    delivered = sum(c.surface_status == "model_unverified" for c in checks)
    expected = "completed" if delivered and delivered == len(checks) else "partial"
    if isinstance(state, AtomicSreReportRun):
        diagnostic = _atomic_grounding(state, context, checks)
        if state.grounding_json != canonical_json_bytes(diagnostic).decode():
            raise ValueError("saved grounding diagnostics differ from native/source/checks")
        if diagnostic["competition"]["status"] == "missing":
            expected = "partial"
    if (
        state.status != expected
        or state.model_delivered_count != delivered
        or (
            canonical_json_bytes([c.model_dump(mode="json") for c in state.checks])
            != canonical_json_bytes([c.model_dump(mode="json") for c in checks])
        )
    ):
        raise ValueError("native report summary differs from original response")
    if admitted is not None:
        _check_admission_summary(state, admitted)
        if isinstance(state, ConsistentQuestionSreReportRun):
            state._admitted_checks = admitted
    return admitted


def _atomic_grounding(
    state: AtomicSreReportRun, context: ReportContext, checks: Any,
) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_report_grounding import grounding_diagnostics

    assert isinstance(context, BriefSreReportContext)
    assert state.raw_response_json is not None and state.resolved_response_json is not None
    return grounding_diagnostics(
        parse_checked_brief(context, state.resolved_response_json.encode()),
        state.raw_response_json, context.supply, context.fact_store, checks)


def _execute(
    directory: Path,
    context: ReportContext,
    state: ReportRun,
    complete: Completion | None,
    recover_saved: SavedCompletion | None,
    *,
    public_history: list[dict[str, Any]] | None,
    brief_retained: dict[str, Any] | None = None,
    accounting_usage: Callable[[], dict[str, Any]] | None = None,
) -> ReportRun:
    path = directory / "result.json"
    admitted_checks = None
    checks: (
        tuple[OpsClaimCheck, ...] | tuple[BriefClaimCheck, ...] | tuple[ComparableClaimCheck, ...]
    )

    def finish() -> ReportRun:
        nonlocal state
        try:
            accounting = None
            if isinstance(state, AtomicSreReportRun):
                from nurse_rcl.v2.live.ops_sre_atomic_report_accounting import render_accounting

                if accounting_usage is not None:
                    raw_usage = canonical_json_bytes(accounting_usage()).decode()
                    if state.report_usage_json not in (None, raw_usage):
                        raise ValueError("saved report accounting ledger changed")
                    if state.report_usage_json is None:
                        state = _persist(
                            path, state.model_copy(update={"report_usage_json": raw_usage})
                        )
                accounting = render_accounting(state)
            _publish_saved_report(
                directory,
                context,
                state,
                public_history=public_history,
                brief_retained=brief_retained,
                admitted_checks=admitted_checks,
                accounting=accounting,
            )
        except Exception as exc:
            if state.primary_failure is None:
                raise
            secondary = failure_detail("report_delivery", exc)
            if secondary not in state.secondary_failures and len(state.secondary_failures) < 16:
                state = _persist(
                    path,
                    state.model_copy(
                        update={"secondary_failures": (*state.secondary_failures, secondary)}
                    ),
                )
        return state

    if state.resolved_response_json is not None and (
        state.raw_response_json is None or _resolve(state, context) != state.resolved_response_json
    ):
        raise ValueError("native/resolved report binding differs")
    if state.status in {"completed", "partial"}:
        admitted_checks = _check_terminal_summary(state, context)
        return finish()
    if state.input_wire_json is None:
        return finish()
    wire = state.input_wire_json.encode()
    if state.raw_response_json is None:
        if state.call_started and recover_saved is None:
            if state.status == "pending":
                state = _persist(
                    path,
                    state.model_copy(
                        update=dict(
                            status="unavailable",
                            failure_types=(*state.failure_types, "outcome_unknown"),
                        )
                    ),
                )
            return finish()
        if not state.call_started and (complete is None or state.status == "unavailable"):
            return finish()
        recovering = state.call_started
        if not recovering:
            state = _persist(path, state.model_copy(update={"call_started": True}))
        started = monotonic()
        try:
            if recovering:
                assert recover_saved is not None
                raw = recover_saved(wire)
                if raw is None:
                    failures = tuple(dict.fromkeys((*state.failure_types, "outcome_unknown")))
                    state = _persist(
                        path,
                        state.model_copy(update=dict(status="unavailable", failure_types=failures)),
                    )
                    return finish()
            else:
                assert complete is not None
                raw = complete(wire)
            if type(raw) is not bytes or len(raw) > REPORT_LIMITS.max_bytes:
                raise ValueError("native report response resource limit")
            text = raw.decode("utf-8", errors="strict")
        except BaseException as exc:
            state = _persist(
                path,
                state.model_copy(
                    update=dict(
                        status="partial"
                        if isinstance(exc, NativePartialResponseError)
                        else "unavailable",
                        primary_failure=state.primary_failure
                        or failure_detail("report_provider", exc),
                        secondary_failures=tuple(
                            dict.fromkeys((*state.secondary_failures, *secondary_failures(exc)))
                        )[:16],
                        failure_types=tuple(
                            dict.fromkeys((*state.failure_types, type(exc).__name__))
                        ),
                        provider_seconds=state.provider_seconds
                        + (0 if recovering else monotonic() - started),
                        interrupted_provider_count=state.interrupted_provider_count
                        + int(not isinstance(exc, Exception)),
                    )
                ),
            )
            finish()
            if not isinstance(exc, Exception):
                raise
            return state
        state = _persist(
            path,
            state.model_copy(
                update=dict(
                    status="pending",
                    raw_response_json=text,
                    provider_seconds=state.provider_seconds
                    + (0 if recovering else monotonic() - started),
                )
            ),
        )
    started = monotonic()
    try:
        if isinstance(context, BriefSreReportContext):
            assert brief_retained is not None
            _require_brief_unchanged(context, canonical_json_bytes(brief_retained))
        resolved = _resolve(state, context)
        state = state.model_copy(update={"resolved_response_json": resolved})
        if isinstance(context, BriefSreReportContext):
            admitted_checks = response_admission(context, resolved, retained=brief_retained)
            checks = admitted_checks.checks
        else:
            checks = _response_checks(resolved, state, context)
        delivered = sum(c.surface_status == "model_unverified" for c in checks)
        state = state.model_copy(
            update=dict(
                checks=checks,
                model_delivered_count=delivered,
                status="completed" if delivered and delivered == len(checks) else "partial",
            )
        )
        if isinstance(state, AtomicSreReportRun):
            diagnostic = _atomic_grounding(state, context, checks)
            state = state.model_copy(update=dict(
                grounding_json=canonical_json_bytes(diagnostic).decode(),
                status=("partial" if diagnostic["competition"]["status"] == "missing"
                        else state.status),
            ))
        if isinstance(state, ConsistentQuestionSreReportRun):
            assert admitted_checks is not None
            state = state.model_copy(
                update=dict(
                    check_policy_json=admitted_checks.policy_json,
                    checks_binding_sha256=admitted_checks.binding_sha256,
                )
            )
            state._admitted_checks = admitted_checks
    except Exception as exc:
        state = state.model_copy(
            update=dict(
                status="partial",
                primary_failure=state.primary_failure or failure_detail("report_validation", exc),
                secondary_failures=tuple(
                    dict.fromkeys((*state.secondary_failures, *secondary_failures(exc)))
                )[:16],
                failure_types=(*state.failure_types, type(exc).__name__),
            )
        )
    state = _persist(
        path,
        state.model_copy(
            update=dict(validation_seconds=state.validation_seconds + monotonic() - started)
        ),
    )
    return finish()
