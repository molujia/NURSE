"""New SRE identity over the existing independent action reservation machinery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.v2.live.ops_card_view import serialize_tool_view
from nurse_rcl.v2.live.ops_contracts import OpsEmpiricalInvestigationAction
from nurse_rcl.v2.live.ops_indexed_tools import (
    OpsEmpiricalActionRecord,
    OpsIndexedInvestigationLedger,
)
from nurse_rcl.v2.live.ops_investigation_ledger import (
    OpsActionRecord,
    OpsInvestigationLedger,
    OpsLedgerStateBase,
)
from nurse_rcl.v2.live.ops_sre_checkpoint import pack_navigation, unpack_navigation
from nurse_rcl.v2.live.ops_sre_encoding import serialize_sre_view
from nurse_rcl.v2.live.ops_sre_tools import SRE_SKILL_TOOLS, OpsSreToolbox, sre_parameter_contracts
from nurse_rcl.v2.live.ops_tools import OpsToolResult


class SreLedgerFields(OpsLedgerStateBase):
    usage_policy_id: Literal["ops-empirical-input-cost/1"] = "ops-empirical-input-cost/1"
    records: tuple[OpsEmpiricalActionRecord, ...] = Field(max_length=8)
    navigation: dict[str, Any]


class OpsSreLedgerState(SreLedgerFields):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/2"] = (
        "nurse.v2.sre-investigation-ledger/2")
    investigation_profile_id: Literal["hermes-sre-tools/4"] = "hermes-sre-tools/4"


class OpsSreInvestigationLedger(OpsIndexedInvestigationLedger):
    _indexed: OpsSreToolbox
    state_type: type[OpsLedgerStateBase] = OpsSreLedgerState
    record_type: type[OpsActionRecord] = OpsEmpiricalActionRecord
    action_type = OpsEmpiricalInvestigationAction
    proposal_bytes = 4_194_304
    skill_tools = SRE_SKILL_TOOLS
    toolbox_type: type[OpsSreToolbox] = OpsSreToolbox

    def __init__(self, *, root: Path, toolbox: OpsSreToolbox, code_sha256: str):
        if type(toolbox) is not self.toolbox_type:
            raise ValueError("SRE ledger requires SRE toolbox")
        # Reuse reservation, interruption, elapsed time and durable charging;
        # do not pass through the legacy profile's exact-type constructor.
        OpsInvestigationLedger.__init__(self, root=root, toolbox=toolbox, code_sha256=code_sha256)
        self._indexed = toolbox

    def _state_values(self) -> dict[str, Any]:
        return dict(navigation=pack_navigation(self._indexed.export_navigation()))

    def capabilities(self) -> dict[str, Any]:
        if self._lock is None:
            raise ValueError("investigation ledger is closed")
        value = self._indexed.capabilities(self.records)
        self._save()
        return value

    def _decode_navigation(self, navigation: dict[str, Any]) -> dict[str, Any]:
        return unpack_navigation(navigation)

    def _restore_tool_state(self, state: OpsLedgerStateBase) -> None:
        if not isinstance(state, self.state_type):
            raise ValueError("SRE ledger profile differs")
        if not isinstance(state, SreLedgerFields):
            raise ValueError("SRE ledger profile differs")
        self._indexed.restore_navigation(self._decode_navigation(state.navigation))
        for record in state.records:
            if (record.model_view_json is not None) != (record.result is not None):
                raise ValueError("SRE receipt/view completion differs")
            if record.model_view_json is not None:
                from nurse_rcl.v2.live.ops_sre_context import _semantic_rows

                view = json.loads(record.model_view_json)
                _semantic_rows(view)
                if serialize_sre_view(view) != record.model_view_json or (
                    view.get("action_index") != record.index
                ):
                    raise ValueError("SRE view canonical form or action differs")

    def _finished_record(
        self, record: OpsActionRecord, outcome: OpsToolResult | None,
    ) -> OpsActionRecord:
        finished = super()._finished_record(record, outcome)
        if isinstance(finished, OpsEmpiricalActionRecord) and finished.status == "rejected":
            assert finished.model_view_json is not None
            view = json.loads(finished.model_view_json)
            contracts = sre_parameter_contracts(self._indexed.inspection_parameters)
            if record.action is not None and record.action.tool in contracts:
                view["parameter_feedback"]["contract"] = contracts[record.action.tool]
            finished = finished.model_copy(update={
                "model_view_json": serialize_tool_view(view, soft_targets=True),
            })
        if isinstance(finished, OpsEmpiricalActionRecord) and finished.model_view_json is not None:
            finished = finished.model_copy(update={
                "model_view_json": serialize_sre_view(json.loads(finished.model_view_json),
                                                      diagnose=self._indexed.diagnose_capacity),
            })
        return finished
