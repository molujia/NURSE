"""Explicit, pinned report-only handoff from admitted atomic/M3 source material.

No source discovery, loader, core import, tool dispatch or provider activation.
The caller must admit the terminal source first and supply its independently fixed
handoff pin. Content hashes bind that admission; they are not signatures or consent.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.live.ops_agent_state import OpsDecisionRecord, sre_run_id
from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord
from nurse_rcl.v2.live.ops_report_context import OpsReportContext
from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
from nurse_rcl.v2.live.ops_sre_brief_report import (
    AtomicReportMetadata,
    BriefSreReportContext,
    _assemble_brief_sources,
    _retained_supply,
    bind_retained_source,
    question_report_material,
    validate_brief_report,
)

if TYPE_CHECKING:
    from nurse_rcl.v2.live.ops_sre_native_run import AtomicSreReportRun, Completion, SavedCompletion


class AtomicReportSourceIdentity(StrictFrozenModel):
    schema_version: Literal["nurse.v2.atomic-report-source/1"] = "nurse.v2.atomic-report-source/1"
    execution_profile: Literal["v2-ops-sre/14"] = "v2-ops-sre/14"
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/25"] = (
        "nurse.v2.sre-decision-exchange/25"
    )
    source_plan_sha256: SHA256
    source_code_sha256: SHA256
    stage_grant_sha256: SHA256
    stage_round_sha256: SHA256
    worker_invocation_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    opaque_case_id: str
    case_snapshot_id: SHA256
    source_run_id: SHA256
    core_result_digest: SHA256
    formal_context_sha256: SHA256
    reference_state_sha256: SHA256
    decision_pins: tuple[SHA256, ...] = Field(min_length=1, max_length=8)
    action_pins: tuple[SHA256, ...] = Field(max_length=8)
    retained_sources_sha256: SHA256
    termination: Literal[
        "model_stop",
        "decision_budget",
        "unavailable",
        "no_progress_after_correction",
        "stage_pause",
    ]
    investigation_status: Literal["completed", "partial", "unavailable"]


_Source = TypeVar("_Source", bound=AtomicReportSourceIdentity)


def _digest(value: Any) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _source(value: dict[str, Any], expected: str) -> AtomicReportSourceIdentity:
    return _decode_source(value, expected, AtomicReportSourceIdentity)


def _decode_source(value: dict[str, Any], expected: str, source_type: type[_Source]) -> _Source:
    require_pin(expected)
    source = source_type.model_validate_json(canonical_json_bytes(value), strict=True)
    if _digest(source.model_dump(mode="json")) != expected:
        raise ValueError("atomic report external source pin differs")
    if source.termination in {"stage_pause", "no_progress_after_correction"} and (
        source.investigation_status != "partial"
    ):
        raise ValueError("atomic report source cannot turn a partial cut into completion")
    if len(source.action_pins) > len(source.decision_pins):
        raise ValueError("atomic report source prefix counts differ")
    return source


def validate_atomic_report_identity(metadata: AtomicReportMetadata) -> AtomicReportSourceIdentity:
    return _validate_pinned_report_identity(metadata, _source)


def _validate_pinned_report_identity(
    metadata: Any, decode_source: Callable[..., _Source]
) -> _Source:
    source = decode_source(metadata.source_identity, metadata.handoff_sha256)
    header = metadata.evidence
    if (
        (
            source.opaque_case_id,
            source.case_snapshot_id,
            source.core_result_digest,
            source.termination,
            source.investigation_status,
        )
        != (
            header.opaque_case_id,
            header.case_snapshot_id,
            header.core_result_digest,
            metadata.termination,
            header.investigation_status,
        )
        or len(metadata.steps) != len(source.decision_pins)
        or len(metadata.action_seconds) != len(source.action_pins)
        or len(metadata.decision_seconds) != len(source.decision_pins)
    ):
        raise ValueError("atomic report source/context identity differs")
    return source


def assemble_atomic_report_context(
    *,
    formal: OpsReportContext,
    decisions: tuple[OpsDecisionRecord, ...],
    records: tuple[OpsActionRecord, ...],
    references: dict[str, Any],
    retained_sources: tuple[dict[str, Any], ...],
    source_identity: AtomicReportSourceIdentity,
    expected_source_sha256: str,
) -> BriefSreReportContext:
    """Project exact admitted source bytes; no owner restoration or execution occurs here."""
    from nurse_rcl.v2.live.ops_sre_feedback_fact_entry import entry_context_from_wire

    return _assemble_pinned_report_context(
        formal=formal,
        decisions=decisions,
        records=records,
        references=references,
        retained_sources=retained_sources,
        source_identity=source_identity,
        expected_source_sha256=expected_source_sha256,
        source_type=AtomicReportSourceIdentity,
        metadata_type=AtomicReportMetadata,
        decode_source=_source,
        wire_context=entry_context_from_wire,
        input_schema="nurse.v2.sre-investigation-input/24",
        focused=False,
    )


def _assemble_pinned_report_context(
    *,
    formal: OpsReportContext,
    decisions: tuple[OpsDecisionRecord, ...],
    records: tuple[OpsActionRecord, ...],
    references: dict[str, Any],
    retained_sources: tuple[dict[str, Any], ...],
    source_identity: Any,
    expected_source_sha256: str,
    source_type: Any,
    metadata_type: Any,
    decode_source: Callable[..., Any],
    wire_context: Callable[..., Any],
    input_schema: str,
    focused: bool,
) -> BriefSreReportContext:
    from nurse_rcl.v2.live.ops_sre_comparable_briefs import project_comparable_brief
    from nurse_rcl.v2.live.ops_sre_context import investigation_steps
    from nurse_rcl.v2.live.ops_sre_encoding import expand

    if type(source_identity) is not source_type or type(formal) is not OpsReportContext:
        raise ValueError("atomic report source type differs")
    source = decode_source(source_identity.model_dump(mode="json"), expected_source_sha256)
    if (
        formal.context_sha256 != source.formal_context_sha256
        or _digest(references) != source.reference_state_sha256
        or _digest(retained_sources) != source.retained_sources_sha256
        or tuple(_digest(d.model_dump(mode="json")) for d in decisions) != source.decision_pins
        or tuple(_digest(r.model_dump(mode="json")) for r in records) != source.action_pins
    ):
        raise ValueError("atomic report source material differs from pinned prefix")
    SreActionReferenceBook.from_state(
        references, snapshot=source.case_snapshot_id, run_id=source.source_run_id
    )
    for decision in decisions:
        wire = getattr(decision, "wire_json", None)
        if type(wire) is not str or sha256_hex(wire.encode()) != getattr(
            decision, "wire_sha256", None
        ):
            raise ValueError("atomic report source decision wire differs")
        original = wire_context(
            wire.encode(),
            references=references,
            case_id=source.opaque_case_id,
            snapshot=source.case_snapshot_id,
            index=decision.index,
        )
        if canonical_json_bytes(original) != canonical_json_bytes(
            expand(json.loads(decision.context_json))
        ):
            raise ValueError("atomic report source frozen context differs")
    steps = investigation_steps(
        decisions=decisions,
        records=records,
        references=references,
        input_schema=input_schema,
    )
    by_brief = {_digest(project_comparable_brief(row)): row for row in retained_sources}
    if len(by_brief) != len(retained_sources):
        raise ValueError("atomic report source has duplicate retained records")

    def retained(_view: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
        if _digest(row) not in by_brief:
            raise ValueError("atomic report source lacks exact delivered brief")
        return by_brief[_digest(row)]

    store, sources, deliveries, semantic = _assemble_brief_sources(
        formal=formal,
        steps=steps,
        records=records,
        retained_brief_source=retained,
        focused=focused,
    )
    state, meanings = question_report_material(steps, semantic, atomic=not focused, focused=focused)
    header = formal.model_dump(mode="json", exclude={"facts"})
    header["investigation_status"] = source.investigation_status
    metadata = metadata_type.model_validate_json(
        canonical_json_bytes(
            dict(
                evidence=header,
                steps=steps,
                source_bindings=[bind_retained_source(s, store) for s in sources],
                delivery_bindings=deliveries,
                formal_fact_refs=[f.fact.fact_ref for f in formal.facts],
                termination=source.termination,
                action_seconds=[r.elapsed_seconds for r in records],
                decision_seconds=[d.provider_seconds for d in decisions],
                question_state=state,
                diagnostic_semantics=meanings,
                source_identity=source.model_dump(mode="json"),
                handoff_sha256=expected_source_sha256,
            )
        )
    )
    context = BriefSreReportContext(metadata, store, _retained_supply(metadata, store))
    validate_brief_report(context)
    return context


def _report_references(
    *,
    root: Path,
    context: BriefSreReportContext,
    code: str,
    references: dict[str, Any],
    public_history: list[dict[str, Any]],
    expected: str,
) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_references import SreReferenceBook

    if (
        type(context) is not BriefSreReportContext
        or type(context.metadata) is not AtomicReportMetadata
    ):
        raise ValueError("atomic report requires its exact context version")
    source = validate_atomic_report_identity(context.metadata)
    if expected != context.metadata.handoff_sha256 or (
        _digest(references) != source.reference_state_sha256
        or tuple(_digest(d) for d in public_history) != source.decision_pins
    ):
        raise ValueError("atomic report external source/history binding differs")
    original = SreActionReferenceBook.from_state(
        references, snapshot=source.case_snapshot_id, run_id=source.source_run_id
    )
    book = SreReferenceBook(
        snapshot=source.case_snapshot_id, run_id=sre_run_id(root, source.case_snapshot_id, code)
    )
    for kind, handle, value, first in original.to_state()["entries"]:
        if first > len(source.decision_pins):
            raise ValueError("atomic report source contains future references")
        if kind != "action" and book.issue(kind, value, visible_at=first) != handle:
            raise ValueError("atomic report source ordinary handles changed")
    return book.to_state()


def run_atomic_report(
    *,
    root: Path,
    context: BriefSreReportContext,
    code_sha256: str,
    references: dict[str, Any],
    public_history: list[dict[str, Any]],
    expected_source_sha256: str,
    complete: Completion | None = None,
    recover_saved: SavedCompletion | None = None,
    accounting_history: dict[str, Any] | None = None,
    accounting_usage: Callable[[], dict[str, Any]] | None = None,
) -> AtomicSreReportRun:
    """Explicit report-only owner. No default completion and no ARK provider routing."""
    from nurse_rcl.v2.live.ops_sre_atomic_report_accounting import history_json
    from nurse_rcl.v2.live.ops_sre_native_run import AtomicSreReportRun, _run_owned_report

    report_refs = _report_references(
        root=root,
        context=context,
        code=code_sha256,
        references=references,
        public_history=public_history,
        expected=expected_source_sha256,
    )
    result = _run_owned_report(
        root=root,
        context=context,
        code_sha256=code_sha256,
        references=report_refs,
        public_history=public_history,
        complete=complete,
        recover_saved=recover_saved,
        source_references=references,
        accounting_history_json=history_json(accounting_history),
        accounting_usage=accounting_usage,
    )
    assert type(result) is AtomicSreReportRun
    return result


def inspect_atomic_report(
    *,
    root: Path,
    context: BriefSreReportContext,
    code_sha256: str,
    references: dict[str, Any],
    public_history: list[dict[str, Any]],
    expected_source_sha256: str,
    expected_accounting_history_sha256: str | None = None,
) -> AtomicSreReportRun:
    """Fresh-reader only: no callback, directory creation, publishing or retry."""
    result = _inspect_atomic_source(
        root=root,
        context=context,
        code_sha256=code_sha256,
        references=references,
        public_history=public_history,
        expected_source_sha256=expected_source_sha256,
        expected_accounting_history_sha256=expected_accounting_history_sha256,
    )
    _inspect_atomic_delivery(root / "report", context, result, public_history)
    return result


def _inspect_atomic_source(
    *,
    root: Path,
    context: BriefSreReportContext,
    code_sha256: str,
    references: dict[str, Any],
    public_history: list[dict[str, Any]],
    expected_source_sha256: str,
    expected_accounting_history_sha256: str | None,
) -> AtomicSreReportRun:
    """Validate authoritative inputs/state; independent of not-yet-rendered output."""
    from nurse_rcl.v2.live.ops_sre_atomic_report_accounting import require_history_pin
    from nurse_rcl.v2.live.ops_sre_native_run import AtomicSreReportRun, _inspect_saved_run

    report_refs = _report_references(
        root=root,
        context=context,
        code=code_sha256,
        references=references,
        public_history=public_history,
        expected=expected_source_sha256,
    )
    result = _inspect_saved_run(
        root / "report",
        context,
        code=code_sha256,
        references=report_refs,
        public_history=public_history,
    )
    assert type(result) is AtomicSreReportRun
    require_history_pin(result.accounting_history_json, expected_accounting_history_sha256)
    return result


def _inspect_atomic_delivery(
    directory: Path,
    context: BriefSreReportContext,
    state: AtomicSreReportRun,
    public_history: list[dict[str, Any]],
) -> None:
    """Reproduce both derived surfaces from the same admitted checks without writing."""
    from nurse_rcl.v2.live.normal_queries import _read_bytes
    from nurse_rcl.v2.live.ops_sre_atomic_report_accounting import render_accounting
    from nurse_rcl.v2.live.ops_sre_brief_render import (
        _render_admitted_brief_report_pair,
        inspect_brief_report_pair,
    )
    from nurse_rcl.v2.live.ops_sre_brief_report import _brief_storage_packet
    from nurse_rcl.v2.live.ops_sre_report_check_policy import (
        admit_brief_checks,
        parse_checked_brief,
    )

    if state.report_usage_json is not None:
        from nurse_rcl.v2.live.ops_sre_atomic_report_provider import read_report_usage
        from nurse_rcl.v2.live.ops_sre_brief_exchange import bind_brief_report

        usage = json.loads(state.report_usage_json)
        if (
            state.input_wire_json is None
            or canonical_json_bytes(usage).decode() != (state.report_usage_json)
            or read_report_usage(
                root=directory.parent / "provider",
                code=state.code_sha256,
                execution=usage["paid_execution_sha256"],
                case=context.metadata.evidence.opaque_case_id,
                binding=bind_brief_report(state.input_wire_json.encode()),
                state=state,
            )
            != usage
        ):
            raise ValueError("report accounting original ledger binding differs")

    retained = _brief_storage_packet(context)
    admitted = state._admitted_checks
    response = None
    if admitted is not None and state.resolved_response_json is not None:
        response = parse_checked_brief(context, state.resolved_response_json.encode())
    else:
        admitted = admit_brief_checks(context, None, retained=retained)
    pair = _render_admitted_brief_report_pair(
        context=context,
        retained=retained,
        response=response,
        admitted_checks=admitted,
        references=state.reference_state,
        public_history=public_history,
        raw_response_json=state.raw_response_json,
        accounting=render_accounting(state),
    )
    receipt = inspect_brief_report_pair(directory, atomic_report=True)
    if receipt["binding_sha256"] != pair["binding_sha256"] or any(
        _read_bytes(directory / name, 16 * 1024**2) != pair[key].encode()
        for name, key in (("report.md", "report"), ("report-evidence.md", "evidence"))
    ):
        raise ValueError("atomic report rendered source binding differs")


def read_atomic_report(
    *,
    root: Path,
    code_sha256: str,
    expected_source_sha256: str,
    expected_context_sha256: str,
    expected_accounting_history_sha256: str | None = None,
) -> AtomicSreReportRun:
    """Read a complete saved owner with two caller-pinned boundaries and zero writes."""
    context, state, public_history = _read_atomic_source(
        root=root,
        code_sha256=code_sha256,
        expected_source_sha256=expected_source_sha256,
        expected_context_sha256=expected_context_sha256,
        expected_accounting_history_sha256=expected_accounting_history_sha256,
    )
    _inspect_atomic_delivery(root / "report", context, state, public_history)
    return state


def _read_atomic_source(
    *,
    root: Path,
    code_sha256: str,
    expected_source_sha256: str,
    expected_context_sha256: str,
    expected_accounting_history_sha256: str | None,
) -> tuple[BriefSreReportContext, AtomicSreReportRun, list[dict[str, Any]]]:
    from nurse_rcl.v2.live.normal_queries import _read_bytes
    from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
    from nurse_rcl.v2.live.ops_sre_brief_report import decode_brief_report_storage

    require_pin(expected_context_sha256)
    directory = root / "report"
    raw = _read_bytes(directory / "context.json", REPORT_LIMITS.max_bytes)
    if sha256_hex(raw) != expected_context_sha256:
        raise ValueError("atomic report external context pin differs")
    context = decode_brief_report_storage(raw)
    refs = _read_bytes(directory / "source-references.json", REPORT_LIMITS.max_bytes)
    history = _read_bytes(directory / "public-history.json", 128 * 1024**2)
    references, public_history = json.loads(refs), json.loads(history)
    if canonical_json_bytes(references) != refs or canonical_json_bytes(public_history) != history:
        raise ValueError("atomic report source files are not canonical")
    state = _inspect_atomic_source(
        root=root,
        context=context,
        code_sha256=code_sha256,
        references=references,
        public_history=public_history,
        expected_source_sha256=expected_source_sha256,
        expected_accounting_history_sha256=expected_accounting_history_sha256,
    )
    return context, state, public_history
