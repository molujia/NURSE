"""Durable SRE report owner; no preparation, core work, HTTP or credential reads."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Any, Literal

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.live.ark_bridge import NativePartialResponseError
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_agent_state import OpsSreConversation, load_conversation, sre_run_id
from nurse_rcl.v2.live.ops_context_view import expand_report_facts, required_report_refs
from nurse_rcl.v2.live.ops_contracts import OpsChannelStatus
from nurse_rcl.v2.live.ops_report import OpsFrozenWireReportRun, _fact_identity, _save
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from nurse_rcl.v2.live.ops_sre_model_view import resolve_model_response
from nurse_rcl.v2.live.ops_sre_references import SreReferenceBook
from nurse_rcl.v2.live.ops_sre_report import check_sre_report, parse_sre_report
from nurse_rcl.v2.live.ops_sre_report_context import OpsSreReportContext
from nurse_rcl.v2.live.ops_sre_report_delivery import save_sre_report_pair
from nurse_rcl.v2.live.ops_sre_report_exchange import (
    bind_sre_report,
    sre_report_body,
    sre_report_projection,
)
from nurse_rcl.v2.live.ops_sre_report_storage import encode_sre_report_storage
from nurse_rcl.v2.live.repair_driver import _fsync
from nurse_rcl.v2.live.staged_case import require_plain_path

SreReportCompletion = Callable[[bytes], bytes]


def read_public_history(path: Path) -> list[dict[str, Any]] | None:
    """Presentation-only read after owner admission; raw model text grants no authority."""
    if not path.exists():
        return None
    conversation = json.loads(_read_bytes(path, 128 * 1024 * 1024))
    result = []
    for row in conversation["decisions"]:
        raw = row.get("native_response_json") or row.get("response_json")
        result.append(dict(decision_index=row["index"], raw_response_json=raw))
    return result


def publish_saved_sre_report(
    directory: Path, context: OpsSreReportContext, state: OpsSreReportRun,
    *, accounting: dict[str, Any] | None = None,
    public_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    response = None
    if state.resolved_response_json is not None:
        try:
            candidate = parse_sre_report(state.resolved_response_json.encode())
            check_sre_report(candidate, context=context,
                provided_fact_refs=state.provided_fact_refs,
                provided_step_refs=state.provided_step_refs)
            response = candidate
        except ValueError:
            pass  # Invalid output stays in result.json, not a displayed observation.
    return save_sre_report_pair(directory, context=context, response=response,
        references=state.reference_state, provided_fact_refs=state.provided_fact_refs,
        provided_step_refs=state.provided_step_refs, public_history=public_history,
        accounting=accounting or dict(report_status=state.status,
            provider_seconds=state.provider_seconds, validation_seconds=state.validation_seconds))


class OpsSreReportRun(OpsFrozenWireReportRun):
    schema_version: Literal["nurse.v2.sre-report-run/7"] = "nurse.v2.sre-report-run/7"
    investigation_profile_id: Literal["hermes-sre-tools/4"] = "hermes-sre-tools/4"
    provided_step_refs: tuple[str, ...] = ()
    reference_state: dict[str, Any]
    resolved_response_json: str | None = None


def _persist(path: Path, state: OpsSreReportRun) -> OpsSreReportRun:
    saved = _save(path, state)
    assert isinstance(saved, OpsSreReportRun)
    return saved


def _check_wire(state: OpsSreReportRun, context: OpsSreReportContext) -> None:
    if state.input_wire_json is None:
        if state.input_wire_sha256 is not None or (
            state.provided_fact_refs or state.provided_step_refs
        ):
            raise ValueError("SRE aliases without frozen wire")
        return
    raw = state.input_wire_json.encode()
    binding = bind_sre_report(raw)
    view = sre_report_body(raw, references=state.reference_state)["context"]
    evidence = context.evidence
    facts = {f.fact.fact_ref: f for f in evidence.facts}
    provided = expand_report_facts(view)
    refs = tuple(f.fact.fact_ref for f in provided)
    if (binding.input_sha256, binding.context_sha256, binding.request_sha256,
        binding.opaque_case_id, view["case_snapshot_id"]) != (
        context.core_result_digest, context.context_sha256, state.input_wire_sha256,
        evidence.opaque_case_id, evidence.case_snapshot_id,
    ) or set(refs) != set(state.provided_fact_refs) or len(refs) != len(state.provided_fact_refs):
        raise ValueError("SRE frozen report binding or fact aliases differ")
    if any(f.fact.fact_ref not in facts or
           _fact_identity(facts[f.fact.fact_ref]) != _fact_identity(f)
           for f in provided):
        raise ValueError("SRE frozen report facts differ")
    source = context.model_dump(mode="json")
    for field in ("profile", "steps", "record_groups", "critical_fact_refs",
                  "unexpanded_critical_overviews", "termination"):
        # JSON types are part of the binding (not Python's False == 0).
        if canonical_json_bytes(view[field]) != canonical_json_bytes(source[field]):
            raise ValueError("SRE frozen narrative context differs")
    if tuple(s["step_ref"] for s in view["steps"]) != state.provided_step_refs or (
        not set(context.critical_fact_refs) <= set(refs)
        or not required_report_refs(evidence) <= set(refs)
        or set(view["mandatory_fact_refs"]) != required_report_refs(evidence)
        or state.target_exceeded != (len(raw) > state.target_bytes)
        or bool(state.expansion_reasons) != state.target_exceeded
    ):
        raise ValueError("SRE frozen steps or mandatory context differs")


def run_sre_report(
    *, root: Path, context: OpsSreReportContext, code_sha256: str,
    complete: SreReportCompletion | None,
) -> OpsSreReportRun:
    require_pin(code_sha256)
    if type(context) is not OpsSreReportContext or context.profile != "hermes-sre-tools/4":
        raise ValueError("SRE report context profile differs")
    require_plain_path(root)
    directory = root / "report"
    directory.mkdir(mode=0o700, exist_ok=True)
    require_plain_path(directory)
    lock = os.open(directory / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(directory, context, code_sha256, complete)
    finally:
        os.close(lock)


def _run(
    directory: Path, context: OpsSreReportContext, code: str,
    complete: SreReportCompletion | None,
) -> OpsSreReportRun:
    path, context_path = directory / "result.json", directory / "context.json"
    raw_context = encode_sre_report_storage(context)
    if context_path.exists() or context_path.is_symlink():
        if _read_bytes(context_path, REPORT_LIMITS.max_bytes) != raw_context:
            raise ValueError("SRE report context identity changed")
    elif path.exists() or path.is_symlink():
        raise ValueError("SRE report context missing")
    else:
        atomic_write_bytes(context_path, raw_context)
        _fsync(directory)
    if path.exists() or path.is_symlink():
        raw = _read_bytes(path, 64 * 1024 * 1024)
        state = OpsSreReportRun.model_validate_json(raw)
        if (state.code_sha256, state.context_sha256, state.core_result_digest) != (
            code, context.context_sha256, context.core_result_digest,
        ) or raw != canonical_json_bytes(state.model_dump(mode="json")) or (
            state.state_sha256 != sha256_hex(canonical_json_bytes(state.model_dump(
                mode="json", exclude={"state_sha256"})))
        ):
            raise ValueError("SRE report receipt identity differs")
    else:
        conversation = load_conversation(directory.parent / "investigation/conversation.json",
            snapshot=context.evidence.case_snapshot_id, code=code, empirical=True, sre=True,
            opaque_case_id=context.evidence.opaque_case_id)
        assert isinstance(conversation, OpsSreConversation)
        references = json.loads(canonical_json_bytes(conversation.reference_state))
        projection = sre_report_projection(context, references=references)
        wire = projection.wire_bytes
        state = _persist(path, OpsSreReportRun(code_sha256=code, reference_state=references,
            context_sha256=context.context_sha256, core_result_digest=context.core_result_digest,
            channels=OpsChannelStatus(core="completed",
                investigation=context.evidence.investigation_status, report="pending"),
            input_wire_json=None if wire is None else wire.decode(),
            input_wire_sha256=None if wire is None else sha256_hex(wire),
            provided_fact_refs=projection.provided_fact_refs,
            provided_step_refs=() if wire is None else tuple(s["step_ref"] for s in context.steps),
            target_exceeded=projection.target_exceeded,
            expansion_reasons=projection.expansion_reasons,
            failure_types=() if projection.status == "ready" else ("input_" + projection.status,)))
    SreReferenceBook.from_state(state.reference_state, snapshot=context.evidence.case_snapshot_id,
        run_id=sre_run_id(directory.parent, context.evidence.case_snapshot_id, code))
    _check_wire(state, context)

    return _execute_report(directory, context, state, complete,
        public_history=read_public_history(directory.parent / "investigation/conversation.json"))


def _execute_report(
    directory: Path, context: OpsSreReportContext, state: OpsSreReportRun,
    complete: SreReportCompletion | None,
    *, public_history: list[dict[str, Any]] | None = None,
) -> OpsSreReportRun:
    """Shared post-admission report path; each owner checks its own identity first."""
    path = directory / "result.json"

    def resolve_response() -> str:
        assert state.raw_response_json is not None and state.input_wire_json is not None
        native = parse_sre_report(state.raw_response_json.encode()).model_dump(mode="json")
        packet = json.loads(json.loads(state.input_wire_json)["messages"][1]["content"])["context"]
        value = resolve_model_response(native, packet=packet, references=state.reference_state,
            case_id=context.evidence.opaque_case_id, snapshot=context.evidence.case_snapshot_id,
            index=9, report=True)
        return canonical_json_bytes(value).decode()

    if state.resolved_response_json is not None and (
        state.raw_response_json is None or resolve_response() != state.resolved_response_json
    ):
        raise ValueError("SRE native/resolved report binding differs")

    def finish() -> OpsSreReportRun:
        publish_saved_sre_report(directory, context, state, public_history=public_history)
        return state

    if state.status in {"completed", "partial"}:
        return finish()
    if state.input_wire_json is None or (state.raw_response_json is None and complete is None):
        state = _persist(path, state.model_copy(update=dict(status="unavailable")))
        return finish()
    if state.raw_response_json is None:
        assert complete is not None
        started = monotonic()
        try:
            raw_response = complete(state.input_wire_json.encode())
            if type(raw_response) is not bytes or len(raw_response) > REPORT_LIMITS.max_bytes:
                raise ValueError("SRE report response resource limit")
            text = raw_response.decode("utf-8", errors="strict")
        except BaseException as exc:
            state = _persist(path, state.model_copy(update=dict(
                status="partial" if isinstance(exc, NativePartialResponseError) else "unavailable",
                failure_types=(*state.failure_types, type(exc).__name__),
                provider_seconds=state.provider_seconds + monotonic() - started,
                interrupted_provider_count=state.interrupted_provider_count
                + int(not isinstance(exc, Exception)))))
            finish()
            if not isinstance(exc, Exception):
                raise
            return state
        state = _persist(path, state.model_copy(update=dict(
            status="pending", raw_response_json=text,
            provider_seconds=state.provider_seconds + monotonic() - started)))
    started = monotonic()
    try:
        assert state.raw_response_json is not None
        resolved = resolve_response()
        state = state.model_copy(update={"resolved_response_json": resolved})
        checks = check_sre_report(parse_sre_report(resolved.encode()),
            context=context, provided_fact_refs=state.provided_fact_refs,
            provided_step_refs=state.provided_step_refs)
        delivered = sum(c.surface_status == "model_unverified" for c in checks)
        state = state.model_copy(update=dict(checks=checks, model_delivered_count=delivered,
            status="completed" if delivered and delivered == len(checks) else "partial"))
    except Exception as exc:
        state = state.model_copy(update=dict(status="partial",
            failure_types=(*state.failure_types, type(exc).__name__)))
    state = _persist(path, state.model_copy(update=dict(
        validation_seconds=state.validation_seconds + monotonic() - started)))
    return finish()
