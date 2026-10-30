"""Private report-only transport over the existing empirical bridge and ledger.

The trusted caller owns source admission and human execution approval. A matching
hash is a binding, not consent inferred from arbitrary Python callers. Preflight
does not supply approval, load credentials or construct this live transport.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nurse_rcl.agents import ark
from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.live.ark_bridge import (
    LiveArkBridge,
    LiveRequestBinding,
    _parse_protocol_response,
    _read,
    live_ark_lock,
    read_attempt_ledger,
)
from nurse_rcl.v2.live.ops_empirical_policy import (
    RESPONSE_LIMITS,
    SRE_ATOMIC_REPORT_PROTOCOL,
    USAGE_POLICY,
)
from nurse_rcl.v2.live.ops_provider import (
    KEY_PATH,
    LOCK_PATH,
    _require_outbound,
    _saved_wire_locked,
)
from nurse_rcl.v2.live.ops_sre_atomic_report import read_atomic_report
from nurse_rcl.v2.live.ops_sre_brief_exchange import bind_brief_report, brief_report_body
from nurse_rcl.v2.live.ops_usage import summarize_attempts
from nurse_rcl.v2.live.repair_driver import publish_json


class AtomicReportProvider:
    """One immutable owner, one report slot; ordinary tools/23 remains disabled."""

    def __init__(
        self,
        *,
        root: Path,
        code_sha256: str,
        execution_sha256: str,
        expected_source_sha256: str,
        expected_context_sha256: str,
        expected_wire_sha256: str,
        approved_execution_sha256: str | None = None,
        expected_accounting_history_sha256: str | None = None,
    ) -> None:
        for pin in (
            code_sha256,
            execution_sha256,
            expected_source_sha256,
            expected_context_sha256,
            expected_wire_sha256,
        ):
            require_pin(pin)
        if approved_execution_sha256 not in (None, execution_sha256):
            raise ValueError("report execution approval differs")
        self.owner_root, self.root = root, root / "provider"
        self.code, self.execution = code_sha256, execution_sha256
        self.source_pin, self.context_pin = expected_source_sha256, expected_context_sha256
        self.wire_pin, self.approved = expected_wire_sha256, approved_execution_sha256
        self.history_pin = expected_accounting_history_sha256
        state = self._owner()
        self._started_when_bound = state.call_started
        self._send_attempted = False
        if state.input_wire_json is None:
            raise ValueError("report owner lacks a complete wire")
        self.wire = state.input_wire_json.encode()
        if sha256_hex(self.wire) != self.wire_pin:
            raise ValueError("report external wire pin differs")
        self.binding = bind_brief_report(self.wire)
        body, store = brief_report_body(self.wire)
        metadata = body["context"]
        if body["protocol"] != SRE_ATOMIC_REPORT_PROTOCOL or (
            self.binding.variant != "v2-ops-sre-report/1"
            or metadata["handoff_sha256"] != self.source_pin
        ):
            raise ValueError("report-only source/protocol binding differs")
        header = metadata["evidence"]
        self.case, self.snapshot = header["opaque_case_id"], header["case_snapshot_id"]
        _require_outbound(
            dict(metadata=metadata, fact_records=store.to_packet()),
            case=self.case,
            snapshot=self.snapshot,
        )

    def _owner(self) -> Any:
        return read_atomic_report(
            root=self.owner_root,
            code_sha256=self.code,
            expected_source_sha256=self.source_pin,
            expected_context_sha256=self.context_pin,
            expected_accounting_history_sha256=self.history_pin,
        )

    def _validate(self, wire: bytes) -> None:
        if (
            wire != self.wire
            or sha256_hex(wire) != self.wire_pin
            or (bind_brief_report(wire) != self.binding)
        ):
            raise ValueError("report frozen wire differs")

    @staticmethod
    def _output(raw: bytes) -> bytes:
        return _parse_protocol_response(
            raw,
            output_tool_name=None,
            limits=RESPONSE_LIMITS,
            protocol=SRE_ATOMIC_REPORT_PROTOCOL,
        ).output_json

    def saved_report(self, wire: bytes) -> bytes | None:
        """Strictly keyless/read-only recovery of this slot, never a retry."""
        self._validate(wire)
        with live_ark_lock(LOCK_PATH):
            raw = _saved_wire_locked(
                self.root,
                self.binding,
                code=self.code,
                execution=self.execution,
                protocol=SRE_ATOMIC_REPORT_PROTOCOL,
                read_only=True,
                frozen_wire=wire,
            )
        return None if raw is None else self._output(raw)

    def report(self, wire: bytes) -> bytes:
        self._validate(wire)
        if self.approved != self.execution:
            raise ValueError("real report requires separate execution approval")
        saved = self.saved_report(wire)
        if saved is not None:
            return saved
        if self._started_when_bound or self._send_attempted:
            raise ValueError("report already started; only saved recovery is permitted")
        # The normal owner marks call_started after this provider was bound and
        # before its first callback. A newly bound started owner may only recover.
        self._send_attempted = True
        settings = ark.load_ark_settings_from_key_file(KEY_PATH)
        request = json.loads(wire)
        if settings.redact_artifact(request) != request:
            raise ValueError("report wire contains credential material")
        for folder in ("outbound", "receipts"):
            (self.root / folder).mkdir(parents=True, exist_ok=True, mode=0o700)
        publish_json(
            self.root / "outbound" / f"request-{self.binding.slot}.json",
            dict(
                binding=self.binding.model_dump(mode="json"),
                request=request,
                request_protocol=SRE_ATOMIC_REPORT_PROTOCOL,
                payload_bytes=len(wire),
                execution_sha256=self.execution,
            ),
        )
        with LiveArkBridge(
            root=self.root / "ark-report",
            lock_path=LOCK_PATH,
            settings=settings,
            code_sha256=self.code,
            config_sha256=self.execution,
            authorized_case_ids=(self.case,),
            request_protocol=SRE_ATOMIC_REPORT_PROTOCOL,
        ) as bridge:
            try:
                raw = bridge.complete(self.binding, wire)
            finally:
                receipt = bridge.receipt(self.binding)
                pin = sha256_hex(canonical_json_bytes(receipt))
                publish_json(
                    self.root / "receipts" / f"request-{self.binding.slot}-{pin}.json", receipt
                )
        return self._output(raw)

    def usage(self) -> dict[str, Any]:
        """Identity-checked view of the original report ledger, not a second ledger."""
        from nurse_rcl.v2.live.ops_sre_atomic_report import _read_atomic_source

        _, state, _ = _read_atomic_source(
            root=self.owner_root,
            code_sha256=self.code,
            expected_source_sha256=self.source_pin,
            expected_context_sha256=self.context_pin,
            expected_accounting_history_sha256=self.history_pin,
        )
        return read_report_usage(
            root=self.root,
            code=self.code,
            execution=self.execution,
            binding=self.binding,
            case=self.case,
            state=state,
        )


def read_report_usage(
    *,
    root: Path,
    code: str,
    execution: str,
    binding: LiveRequestBinding,
    case: str,
    state: Any,
) -> dict[str, Any]:
    """Original ledger view after authoritative owner validation, before/after rendering."""
    require_pin(execution)
    with live_ark_lock(LOCK_PATH):
        if (root / "ark-investigation").exists():
            raise ValueError("report-only owner cannot contain investigation spend")
        folder = root / "ark-report"
        paths = sorted(folder.glob("request-*.json"))
        if len(paths) > 1:
            raise ValueError("report-only logical slot count differs")
        rows = []
        if paths:
            identity = _read(folder / "run-identity.json", "live_ark_run")
            if (
                identity.get("request_protocol") != SRE_ATOMIC_REPORT_PROTOCOL
                or identity.get("code_sha256") != code
                or identity.get("config_sha256") != execution
                or identity.get("authorized_case_ids") != [case]
                or identity.get("usage_policy_id") != USAGE_POLICY
                or identity.get("max_attempts") != 5
                or identity.get("llm_in_flight") != 1
            ):
                raise ValueError("report-only usage identity differs")
            ledger = read_attempt_ledger(paths[0], empirical=True)
            if ledger.binding != binding or (
                ledger.run_identity_sha256 != sha256_hex(canonical_json_bytes(identity))
                or paths[0].name != f"request-{binding.slot}.json"
            ):
                raise ValueError("report-only usage slot binding differs")
            for attempt in ledger.attempts:
                usage = (
                    None
                    if attempt.response_json is None
                    else (json.loads(attempt.response_json).get("usage"))
                )
                rows.append(dict(phase="report", slot=1, attempt=attempt.number, usage=usage))
        summary = summarize_attempts(
            rows,
            confirmed_no_send=not state.call_started and not (root / "outbound").exists(),
        )
    return dict(
        summary,
        paid_execution_sha256=execution,
        usage_policy_id=USAGE_POLICY,
        spend_scope="independent_report_execution_lifetime",
        cost=None,
        untracked_send_unknown=state.call_started and not rows,
    )
