"""G+ report-only delivery through the original client/retry/global lock.

One capture and one original ledger; complete bad native text is never retried.
Only the trusted case/owner runner supplies human execution approval.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from nurse_rcl.agents import ark
from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.exchange_json import decode_json_object
from nurse_rcl.v2.live.ark_bridge import LiveArkBridge, _read, read_attempt_ledger
from nurse_rcl.v2.live.ops_empirical_policy import RESPONSE_LIMITS
from nurse_rcl.v2.live.ops_provider import KEY_PATH, LOCK_PATH
from nurse_rcl.v2.live.ops_sre_gplus_contract import GPLUS_PROTOCOL
from nurse_rcl.v2.live.ops_sre_gplus_report import bind_gplus_report, gplus_report_body
from nurse_rcl.v2.live.ops_sre_typed_report_provider import (
    TypedReportTransport,
    _durable,
    parse_typed_receipt,
    typed_delivery_origin,
)


@dataclass(frozen=True)
class GplusDelivery:
    native: bytes
    origin: dict[str, Any]
    receipt: dict[str, Any]


def read_gplus_delivery(
    *, root: Path, wire: bytes, code_sha256: str, execution_sha256: str
) -> GplusDelivery:
    """Keyless read of completed actual-byte chain, never opens a sending bridge."""
    binding = bind_gplus_report(wire)
    identity = _read(root / "ledger/run-identity.json", "live_ark_run")
    if (
        identity["code_sha256"] != code_sha256
        or identity["config_sha256"] != execution_sha256
        or identity["authorized_case_ids"] != [binding.opaque_case_id]
        or identity["request_protocol"] != GPLUS_PROTOCOL
    ):
        raise ValueError("G+ saved delivery identity differs")
    ledger = read_attempt_ledger(root / "ledger" / f"request-{binding.slot}.json", empirical=True)
    receipt = decode_json_object((root / "receipt.json").read_bytes(), RESPONSE_LIMITS)
    if (
        ledger.binding != binding
        or ledger.run_identity_sha256 != sha256_hex(canonical_json_bytes(identity))
        or receipt["ledger"] != ledger.model_dump(mode="json")
    ):
        raise ValueError("G+ saved original ledger differs")
    last = ledger.attempts[-1]
    if last.response_json is None:
        raise ValueError("G+ saved outcome unknown")
    raw = last.response_json.encode()
    origin = typed_delivery_origin(
        capture_root=root / "capture",
        receipt=receipt,
        raw=raw,
        binding=binding,
        code_sha256=code_sha256,
        execution_sha256=execution_sha256,
        protocol=GPLUS_PROTOCOL,
    )
    return GplusDelivery(parse_typed_receipt(raw).output_json, origin, receipt)


class GplusReportProvider:
    def __init__(
        self,
        *,
        root: Path,
        code_sha256: str,
        execution_sha256: str,
        approved_execution_sha256: str | None,
        expected_context_sha256: str,
        wire: bytes,
        references: dict[str, Any],
        settings: ark.ArkSettings | None = None,
        transport: httpx.BaseTransport | None = None,
        lock_path: Path = LOCK_PATH,
        sleep: Callable[[float], None] = time.sleep,
    ):
        for pin in (code_sha256, execution_sha256, expected_context_sha256):
            require_pin(pin)
        body, _ = gplus_report_body(wire, references=references)
        if body["context_sha256"] != expected_context_sha256:
            raise ValueError("G+ trusted owner context differs")
        if approved_execution_sha256 not in (None, execution_sha256):
            raise ValueError("G+ execution approval differs")
        self.root, self.wire = Path(root), wire
        self.code, self.execution, self.approved = (
            code_sha256,
            execution_sha256,
            approved_execution_sha256,
        )
        self.binding = bind_gplus_report(wire)
        self.settings, self.transport = settings, transport
        self.lock_path, self.sleep = lock_path, sleep
        self.started = False

    def report(self, wire: bytes) -> GplusDelivery:
        if wire != self.wire or self.approved != self.execution:
            raise ValueError("G+ approved immutable wire differs")
        if self.started or self.root.exists():
            raise ValueError("G+ already started; only keyless saved recovery permitted")
        self.started = True
        self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        _durable(self.root / "request.bin", wire)
        settings = self.settings or ark.load_ark_settings_from_key_file(KEY_PATH)
        if settings.redact_artifact(json.loads(wire)) != json.loads(wire):
            raise ValueError("G+ request contains credential material")
        capture = TypedReportTransport(
            root=self.root / "capture",
            expected_wire=wire,
            code_sha256=self.code,
            execution_sha256=self.execution,
            transport=self.transport,
            protocol=GPLUS_PROTOCOL,
        )
        with LiveArkBridge(
            root=self.root / "ledger",
            lock_path=self.lock_path,
            settings=settings,
            code_sha256=self.code,
            config_sha256=self.execution,
            authorized_case_ids=(self.binding.opaque_case_id,),
            request_protocol=GPLUS_PROTOCOL,
            transport=capture,
            sleep=self.sleep,
        ) as bridge:
            try:
                bridge.complete(self.binding, wire)
            finally:
                _durable(
                    self.root / "receipt.json", canonical_json_bytes(bridge.receipt(self.binding))
                )
        return read_gplus_delivery(
            root=self.root, wire=wire, code_sha256=self.code, execution_sha256=self.execution
        )
