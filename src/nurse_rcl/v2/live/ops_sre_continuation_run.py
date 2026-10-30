"""Explicit report continuation: no case runner, preparation or investigation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live import ops_provider
from nurse_rcl.v2.live.ark_bridge import (
    LiveTransportError,
    _read,
    live_ark_lock,
    read_attempt_ledger,
)
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_empirical_policy import SRE_REPORT_PROTOCOL, USAGE_POLICY
from nurse_rcl.v2.live.ops_sre_continuation import (
    continuation_lock,
    read_continuation,
    write_once,
)
from nurse_rcl.v2.live.ops_sre_report_exchange import bind_sre_report
from nurse_rcl.v2.live.ops_sre_report_run import (
    OpsSreReportRun,
    _check_wire,
    _execute_report,
    _persist,
    publish_saved_sre_report,
    read_public_history,
)
from nurse_rcl.v2.live.ops_usage import summarize_attempts
from nurse_rcl.v2.live.repair_driver import _fsync
from nurse_rcl.v2.live.staged_case import require_plain_path


def _provider_state(root: Path, receipt: dict[str, Any], pin: str, wire: bytes) -> str:
    """Inspect before ordinary provider recovery can classify an unknown as retry."""
    binding = bind_sre_report(wire)
    with live_ark_lock(ops_provider.LOCK_PATH):
        folder = root / "provider/ark-report"
        paths = sorted(folder.glob("request-*.json"))
        if not paths:
            return "unknown" if (root / "send-started.json").exists() else "unsent"
        if paths != [folder / f"request-{binding.slot}.json"]:
            raise ValueError("continuation provider request inventory differs")
        identity = _read(folder / "run-identity.json", "live_ark_run")
        if (identity["code_sha256"], identity["config_sha256"], identity["request_protocol"],
            identity["authorized_case_ids"], identity["usage_policy_id"],
            identity["max_attempts"], identity["llm_in_flight"]) != (
            receipt["code_sha256"], pin, SRE_REPORT_PROTOCOL,
            [receipt["source"]["opaque_case_id"]], USAGE_POLICY, 5, 1,
        ):
            raise ValueError("continuation provider identity differs")
        ledger = read_attempt_ledger(paths[0], empirical=True)
        if ledger.binding != binding or ledger.run_identity_sha256 != sha256_hex(
            canonical_json_bytes(identity)
        ):
            raise ValueError("continuation provider binding differs")
        if not ledger.attempts:
            return "unknown" if (root / "send-started.json").exists() else "unsent"
        last = ledger.attempts[-1]
        if last.response_json is not None:
            return "saved"
        if last.record is not None and last.record.outcome in {"fatal", "exhausted"}:
            return "terminal"
        if (last.record is None and getattr(last, "http_status_code", None) is None) or (
            last.record is not None and last.record.error_type == "OutcomeUnknownAfterInterruption"
        ):
            return "unknown"
        return "known_failure"


def _load_owner(directory: Path, initial: OpsSreReportRun) -> OpsSreReportRun:
    path = directory / "result.json"
    if not path.exists() and not path.is_symlink():
        return _persist(path, initial)
    raw = _read_bytes(path, 64 * 1024 * 1024)
    state = OpsSreReportRun.model_validate_json(raw)
    if raw != canonical_json_bytes(state.model_dump(mode="json")) or (
        state.state_sha256 != sha256_hex(canonical_json_bytes(
            state.model_dump(mode="json", exclude={"state_sha256"})))
    ):
        raise ValueError("continuation report state digest differs")
    mutable = {"state_sha256", "channels", "status", "raw_response_json", "checks",
        "model_delivered_count", "failure_types", "provider_seconds", "validation_seconds",
        "interrupted_provider_count", "resolved_response_json"}
    if state.model_dump(mode="json", exclude=mutable) != initial.model_dump(
        mode="json", exclude=mutable,
    ) or (state.channels.core, state.channels.investigation) != ("completed", "completed"):
        raise ValueError("continuation frozen owner differs")
    return state


def run_continuation(
    receipt_path: Path, *, receipt_sha256: str, approved_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    if approved_receipt_sha256 is not None and approved_receipt_sha256 != receipt_sha256:
        raise ValueError("continuation approval differs")
    root = require_plain_path(receipt_path).parent
    with continuation_lock(root):
        receipt, context, initial = read_continuation(receipt_path, receipt_sha256)
        directory = root / "report"
        directory.mkdir(exist_ok=True, mode=0o700)
        require_plain_path(directory)
        write_once(directory / "context.json", _read_bytes(root / "context.json", 4_194_304))
        state = _load_owner(directory, initial)
        _check_wire(state, context)
        assert initial.input_wire_json is not None
        wire = initial.input_wire_json.encode()
        transport_state = _provider_state(root, receipt, receipt_sha256, wire)
        marker = dict(receipt_sha256=receipt_sha256, request_sha256=receipt["request_sha256"])
        marker_path = root / "send-started.json"
        if (marker_path.exists() or marker_path.is_symlink()) and (
            _read_bytes(marker_path, 4096) != canonical_json_bytes(marker)
        ):
            raise ValueError("continuation send marker differs")

        def complete(payload: bytes) -> bytes:
            if payload != wire:
                raise ValueError("continuation frozen wire differs")
            if transport_state in {"saved", "terminal"}:
                saved = ops_provider.saved_sre_report(root / "provider", payload,
                    code_sha256=receipt["code_sha256"], execution_sha256=receipt_sha256)
                if saved is None:
                    # Even malformed saved native output is not permission to redraw.
                    raise LiveTransportError("saved_native_report_not_accepted")
                return saved
            if transport_state == "unknown":
                raise LiveTransportError("continuation_outcome_unknown")
            if approved_receipt_sha256 != receipt_sha256:
                raise ValueError("continuation requires explicit approval")
            write_once(marker_path, canonical_json_bytes(marker))
            provider = ops_provider.OpsLiveProvider(root=root / "provider",
                code_sha256=receipt["code_sha256"], execution_sha256=receipt_sha256,
                approved_execution_sha256=approved_receipt_sha256,
                opaque_case_id=context.evidence.opaque_case_id,
                case_snapshot_id=context.evidence.case_snapshot_id, empirical=True, sre=True)
            return provider.report(payload)

        status: str
        if state.status in {"completed", "partial"} or state.raw_response_json is not None:
            state = _execute_report(directory, context, state, None)
            status = state.status
        elif transport_state == "unknown":
            # Re-render only the saved unavailable owner, without a provider.
            # This also prevents accumulating provenance notes on each check.
            state = _execute_report(directory, context, state, None)
            status = "blocked_unknown_outcome"
        elif transport_state in {"saved", "terminal"} or approved_receipt_sha256 is not None:
            state = _execute_report(directory, context, state, complete)
            status = state.status
        else:
            status = "awaiting_approval"

        sent = marker_path.exists()
        report_usage = ops_provider.usage_summary(root / "provider",
            execution_sha256=receipt_sha256, empirical=True, sre=True,
            confirmed_no_send=not sent)
        source_usage = receipt["source"]["usage"]
        rows = [*source_usage["attempt_usage"], *report_usage["attempt_usage"]]
        usage = summarize_attempts(rows, confirmed_no_send=not rows and not sent)
        if sent and not report_usage["attempt_usage"]:
            # A persisted call-start with missing transport evidence is unknown,
            # not a fabricated zero-cost attempt or permission to send again.
            for field in ("input", "output", "total", "cached"):
                usage[field + "_tokens"] = None
            usage.update(usage_complete=False, input_usage_complete=False)
        result = dict(schema_version="nurse.v2.sre-report-continuation-result/1",
            receipt_sha256=receipt_sha256, status=status, report_status=state.status,
            core_result_digest=context.core_result_digest,
            source_execution_sha256=receipt["source"]["plan_sha256"],
            report_execution_sha256=receipt_sha256,
            source_usage=source_usage, report_usage=report_usage, usage=usage,
            untracked_send_unknown=sent and not report_usage["attempt_usage"],
            timing=dict(historical=receipt["source"]["historical_timing"],
                report_provider_seconds=state.provider_seconds,
                report_validation_seconds=state.validation_seconds),
            investigation_reexecuted=False, explanation_acceptance="pending_assistant_and_user")
        if (directory / "report.md").exists():
            publish_saved_sre_report(directory, context, state, accounting=result,
                public_history=read_public_history(
                    Path(receipt["source"]["root"]) / "investigation/conversation.json"))
        atomic_write_bytes(root / "continuation-result.json", canonical_json_bytes(result))
        _fsync(root)
        return result
