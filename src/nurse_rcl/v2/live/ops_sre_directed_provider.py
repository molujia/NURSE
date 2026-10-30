"""Decision-only stage owner retains all ordinary outbound and native attempt guards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ark_bridge import LiveRequestBinding
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_contracts import SreProfileId
from nurse_rcl.v2.live.ops_empirical_policy import SRE_DIRECTED_DECISION_PROTOCOL
from nurse_rcl.v2.live.ops_provider import OpsLiveProvider, _saved_wire
from nurse_rcl.v2.live.ops_sre_brief_exchange import bind_brief_decision
from nurse_rcl.v2.live.ops_sre_turn_validation import (
    bind_workbench_turn,
    validate_bound_workbench_response,
)


def stage_grant(execution_sha256: str, ceiling: int) -> dict[str, Any]:
    """Grant scope changes without changing the lifetime transport execution identity."""
    return dict(
        schema_version="nurse.v2.sre-stage-grant/1",
        execution_sha256=execution_sha256,
        through_decision_index=ceiling,
        correction_limit=1,
        report_limit=0,
    )


class OpsSreDirectedDecisionProvider(OpsLiveProvider):
    """A separately approved grant binds this ceiling to one stable execution pin."""

    stage_profile: SreProfileId = "hermes-sre-tools/18"
    stage_variant = "v2-ops-sre/8"
    stage_protocol = SRE_DIRECTED_DECISION_PROTOCOL

    def __init__(
        self,
        *,
        root: Path,
        code_sha256: str,
        execution_sha256: str,
        approved_execution_sha256: str | None,
        opaque_case_id: str,
        case_snapshot_id: str,
        stage_decision_limit: int,
        approved_stage_sha256: str | None,
    ) -> None:
        if type(stage_decision_limit) is not int or not 1 <= stage_decision_limit <= 8:
            raise ValueError("stage decision limit differs")
        if approved_stage_sha256 != sha256_hex(
            canonical_json_bytes(
                stage_grant(execution_sha256, stage_decision_limit),
            )
        ):
            raise ValueError("stage approval differs from the requested decision ceiling")
        super().__init__(
            root=root,
            code_sha256=code_sha256,
            execution_sha256=execution_sha256,
            approved_execution_sha256=approved_execution_sha256,
            opaque_case_id=opaque_case_id,
            case_snapshot_id=case_snapshot_id,
            empirical=True,
            sre=True,
            sre_profile=self.stage_profile,
        )
        self.stage_decision_limit = stage_decision_limit

    def _binding(self, index: int, wire: object) -> LiveRequestBinding:
        if type(index) is not int or not 1 <= index <= self.stage_decision_limit:
            raise ValueError("stage decision index exceeds approved prefix")
        if type(wire) is not bytes:
            raise ValueError("stage decision requires owner-frozen wire")
        binding = bind_brief_decision(
            wire, opaque_case_id=self.case, case_snapshot_id=self.snapshot, decision_index=index
        )
        if binding.variant != self.stage_variant:
            raise ValueError("stage decision profile differs")
        return binding

    def decision(self, index: int, context: dict[str, Any] | bytes) -> bytes:
        binding = self._binding(index, context)
        assert isinstance(context, bytes)
        raw = self._send(context, binding, self.stage_protocol, CONTEXT_LIMITS)
        return validate_bound_workbench_response(
            raw,
            bind_workbench_turn(context, binding),
        ).output_json

    def saved_decision(self, index: int, frozen_wire: bytes) -> bytes | None:
        binding = self._binding(index, frozen_wire)
        raw = _saved_wire(
            self.root,
            binding,
            code=self.code,
            execution=self.execution,
            protocol=self.stage_protocol,
            read_only=True,
            frozen_wire=frozen_wire,
        )
        return (
            None
            if raw is None
            else validate_bound_workbench_response(
                raw,
                bind_workbench_turn(frozen_wire, binding),
            ).output_json
        )
