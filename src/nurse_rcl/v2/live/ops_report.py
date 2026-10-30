"""Report-only durable owner. No source preparation, core execution or key lookup."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Literal, TypeVar

from pydantic import Field

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.live.ark_bridge import NativePartialResponseError
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_contracts import OpsChannelStatus
from nurse_rcl.v2.live.ops_report_checks import check_ops_report
from nurse_rcl.v2.live.ops_report_context import OpsReportContext, OpsReportFact
from nurse_rcl.v2.live.ops_report_exchange import bind_empirical_report, empirical_report_projection
from nurse_rcl.v2.live.ops_report_render import render_ops_report
from nurse_rcl.v2.live.ops_report_schema import (
    REPORT_LIMITS,
    RESPONSE_LIMITS,
    OpsClaimCheck,
    parse_ops_report,
)
from nurse_rcl.v2.live.repair_driver import _fsync
from nurse_rcl.v2.live.staged_case import require_plain_path

ReportCompletion = Callable[[OpsReportContext | bytes], bytes]


def _fact_identity(fact: OpsReportFact) -> bytes:
    """Compare exact named cells, independent of the table codec's column order."""
    body = fact.model_dump(mode="json")
    body["fact"]["values"] = sorted(body["fact"]["values"], key=lambda cell: cell["name"])
    return canonical_json_bytes(body)


class OpsReportRunFields(StrictFrozenModel):
    code_sha256: SHA256
    context_sha256: SHA256
    core_result_digest: SHA256
    channels: OpsChannelStatus
    status: Literal["pending", "completed", "partial", "unavailable"] = "pending"
    raw_response_json: str | None = None
    model_delivered_count: int = Field(default=0, ge=0)
    failure_types: tuple[str, ...] = ()
    provider_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    validation_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    interrupted_provider_count: int = Field(default=0, ge=0)
    state_sha256: SHA256 = "0" * 64


class OpsReportRunBase(OpsReportRunFields):
    checks: tuple[OpsClaimCheck, ...] = ()


class OpsReportRun(OpsReportRunBase):
    schema_version: Literal["nurse.v2.ops-report-run/1"] = "nurse.v2.ops-report-run/1"
    model_delivered_count: int = Field(default=0, ge=0, le=24)


class OpsFrozenWireFields(OpsReportRunFields):
    usage_policy_id: Literal["ops-empirical-input-cost/1"] = "ops-empirical-input-cost/1"
    input_wire_json: str | None = None
    input_wire_sha256: SHA256 | None = None
    provided_fact_refs: tuple[str, ...] = ()
    target_bytes: Literal[32768] = 32768
    target_exceeded: bool = False
    expansion_reasons: tuple[str, ...] = ()


class OpsFrozenWireReportRun(OpsFrozenWireFields, OpsReportRunBase):
    """Legacy response checks retain their exact original schema."""


class OpsEmpiricalReportRun(OpsFrozenWireReportRun):
    schema_version: Literal["nurse.v2.ops-report-run/2"] = "nurse.v2.ops-report-run/2"


_RunT = TypeVar("_RunT", bound=OpsReportRunFields)


def _save(path: Path, state: _RunT) -> _RunT:
    state = state.model_copy(
        update={
            "channels": state.channels.model_copy(
                update={
                    "report": state.status,
                }
            )
        }
    )
    body = state.model_dump(mode="json", exclude={"state_sha256"})
    state = state.model_copy(update={"state_sha256": sha256_hex(canonical_json_bytes(body))})
    raw = canonical_json_bytes(state.model_dump(mode="json"))
    if isinstance(state, OpsFrozenWireFields) and len(raw) > 64 * 1024 * 1024:
        raise ValueError("report storage resource limit")
    atomic_write_bytes(path, raw)
    _fsync(path.parent)
    return state


def run_ops_report(
    *,
    root: Path,
    context: OpsReportContext,
    code_sha256: str,
    complete: ReportCompletion | None,
    empirical: bool = False,
) -> OpsReportRunBase:
    """Persist context before provider, response before checks, never redraw quality.

    Production complete MUST use the durable single report bridge slot. A retry
    after interruption uses that same frozen context and shared lifetime attempts.
    This private application boundary accepts trusted owner code, not model code.
    """
    require_pin(code_sha256)
    if context.schema_version != (
        "nurse.v2.ops-report-context/2" if empirical else "nurse.v2.ops-report-context/1"
    ):
        raise ValueError("report context protocol differs")
    require_plain_path(root)
    directory = root / "report"
    directory.mkdir(mode=0o700, exist_ok=True)
    require_plain_path(directory)
    lock = os.open(directory / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(directory, context, code_sha256, complete, empirical=empirical)
    finally:
        os.close(lock)


def _run(
    directory: Path,
    context: OpsReportContext,
    code: str,
    complete: ReportCompletion | None,
    *,
    empirical: bool = False,
) -> OpsReportRunBase:
    path, context_path = directory / "result.json", directory / "context.json"
    context_raw = canonical_json_bytes(context.model_dump(mode="json"))
    if context_path.exists() or context_path.is_symlink():
        if _read_bytes(context_path, 4_194_304) != context_raw:
            raise ValueError("report context identity changed")
    elif path.exists() or path.is_symlink():
        raise ValueError("report context identity missing")
    else:
        atomic_write_bytes(context_path, context_raw)
        _fsync(directory)
    if path.exists() or path.is_symlink():
        raw = _read_bytes(path, 64 * 1024 * 1024 if empirical else 4_194_304)
        state: OpsReportRunBase = (
            OpsEmpiricalReportRun.model_validate_json(raw)
            if empirical
            else OpsReportRun.model_validate_json(raw)
        )
        if (state.code_sha256, state.context_sha256, state.core_result_digest) != (
            code,
            context.context_sha256,
            context.core_result_digest,
        ):
            raise ValueError("report run identity changed")
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
            raise ValueError("report receipt digest differs")
    else:
        initial = (OpsEmpiricalReportRun if empirical else OpsReportRun)(
            code_sha256=code,
            context_sha256=context.context_sha256,
            core_result_digest=context.core_result_digest,
            channels=OpsChannelStatus(
                core="completed",
                investigation=context.investigation_status,
                report="pending",
            ),
        )
        if empirical and context.formal_projection_available:
            projection = empirical_report_projection(context)
            initial = initial.model_copy(
                update={
                    "input_wire_json": None
                    if projection.wire_bytes is None
                    else projection.wire_bytes.decode("utf-8"),
                    "input_wire_sha256": None
                    if projection.wire_bytes is None
                    else sha256_hex(projection.wire_bytes),
                    "provided_fact_refs": projection.provided_fact_refs,
                    "target_exceeded": projection.target_exceeded,
                    "expansion_reasons": projection.expansion_reasons,
                    "failure_types": ()
                    if projection.status == "ready"
                    else ("input_" + projection.status,),
                }
            )
        state = _save(path, initial)

    if isinstance(state, OpsEmpiricalReportRun) and state.input_wire_json is not None:
        from nurse_rcl.v2.live.ops_context_view import expand_report_facts

        wire = state.input_wire_json.encode("utf-8")
        binding = bind_empirical_report(wire)
        body = json.loads(json.loads(wire)["messages"][1]["content"])
        provided = expand_report_facts(body["context"])
        facts = {f.fact.fact_ref: f for f in context.facts}
        if (
            (binding.input_sha256, binding.context_sha256, binding.request_sha256)
            != (
                context.core_result_digest,
                context.context_sha256,
                state.input_wire_sha256,
            )
            or set(f.fact.fact_ref for f in provided) != set(state.provided_fact_refs)
            or (
                body["context"]["case_snapshot_id"] != context.case_snapshot_id
                or binding.opaque_case_id != context.opaque_case_id
            )
            or any(
                f.fact.fact_ref not in facts
                or _fact_identity(facts[f.fact.fact_ref]) != _fact_identity(f)
                for f in provided
            )
            or state.target_exceeded != (len(wire) > state.target_bytes)
            or (bool(state.expansion_reasons) != state.target_exceeded)
        ):
            raise ValueError("report frozen wire or alias identity differs")

    def finish() -> OpsReportRunBase:
        atomic_write_bytes(
            directory / "report.md",
            render_ops_report(context=context, checks=state.checks).encode("utf-8"),
        )
        _fsync(directory)
        return state

    if state.status in {"completed", "partial"}:
        return finish()
    if isinstance(state, OpsEmpiricalReportRun) and state.input_wire_json is None:
        state = _save(path, state.model_copy(update={"status": "unavailable"}))
        return finish()
    if state.raw_response_json is None and (
        not context.formal_projection_available or complete is None
    ):
        reason = "provider_unavailable" if complete is None else "formal_projection_unavailable"
        if state.status != "unavailable" or reason not in state.failure_types:
            state = _save(
                path,
                state.model_copy(
                    update={
                        "status": "unavailable",
                        "failure_types": (*state.failure_types, reason),
                    }
                ),
            )
        return finish()
    if state.raw_response_json is None:
        assert complete is not None
        started = monotonic()
        try:
            # Detach nested values before trusted provider construction.
            request: OpsReportContext | bytes = (
                state.input_wire_json.encode("utf-8")
                if isinstance(state, OpsEmpiricalReportRun) and state.input_wire_json is not None
                else OpsReportContext.model_validate_json(context_raw)
            )
            raw_response = complete(request)
            limits = REPORT_LIMITS if empirical else RESPONSE_LIMITS
            if type(raw_response) is not bytes or len(raw_response) > limits.max_bytes:
                raise ValueError("invalid_report_response_boundary")
            text = raw_response.decode("utf-8", errors="strict")
        except BaseException as exc:
            state = _save(
                path,
                state.model_copy(
                    update={
                        "status": "partial"
                        if empirical and isinstance(exc, NativePartialResponseError)
                        else "unavailable",
                        "failure_types": (*state.failure_types, type(exc).__name__),
                        "provider_seconds": state.provider_seconds + monotonic() - started,
                        "interrupted_provider_count": state.interrupted_provider_count
                        + int(not isinstance(exc, Exception)),
                    }
                ),
            )
            finish()
            if not isinstance(exc, Exception):
                raise
            return state
        state = _save(
            path,
            state.model_copy(
                update={
                    "status": "pending",
                    "raw_response_json": text,
                    "provider_seconds": state.provider_seconds + monotonic() - started,
                }
            ),
        )
    started = monotonic()
    try:
        assert state.raw_response_json is not None
        checks = check_ops_report(
            parse_ops_report(state.raw_response_json.encode(), empirical=empirical),
            context=context,
            provided_fact_refs=state.provided_fact_refs
            if isinstance(state, OpsEmpiricalReportRun)
            else None,
        )
        delivered = sum(c.surface_status == "model_unverified" for c in checks)
        state = state.model_copy(
            update={
                "checks": checks,
                "model_delivered_count": delivered,
                "status": "completed" if delivered and delivered == len(checks) else "partial",
            }
        )
    except Exception as exc:
        state = state.model_copy(
            update={
                "status": "partial",
                "failure_types": (*state.failure_types, type(exc).__name__),
            }
        )
    finally:
        state = _save(
            path,
            state.model_copy(
                update={
                    "validation_seconds": state.validation_seconds + monotonic() - started,
                }
            ),
        )
    return finish()
