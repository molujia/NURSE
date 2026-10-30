"""D12 focus profile, isolated from the existing /4 tools and paid owners."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.live.ops_failure import OpsFailureDetail, failure_detail
from nurse_rcl.v2.live.ops_indexed_tools import INDEXED_TOOL_PARAMETERS, OpsEmpiricalActionRecord
from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord, OpsLedgerStateBase
from nurse_rcl.v2.live.ops_sre_encoding import serialize_sre_view
from nurse_rcl.v2.live.ops_sre_ledger import OpsSreInvestigationLedger, SreLedgerFields
from nurse_rcl.v2.live.ops_sre_tools import (
    INSPECTION_PARAMETERS,
    SRE_SKILL_TOOLS,
    OpsSreToolbox,
)
from nurse_rcl.v2.live.ops_sre_working_memory import working_history
from nurse_rcl.v2.live.ops_tools import OpsToolResult

WORKING_PROFILE = "hermes-sre-tools/5"
WORKING_INSPECTION_PARAMETERS = {
    **INSPECTION_PARAMETERS,
    "inspect_candidate": ("candidate_service_key", "focus", "cursor"),
    "compare_candidates": (
        "left_candidate_service_key",
        "right_candidate_service_key",
        "focus",
        "cursor",
    ),
}
WORKING_TOOL_PARAMETERS = {**INDEXED_TOOL_PARAMETERS, **WORKING_INSPECTION_PARAMETERS}
WORKING_SKILL_TOOLS = {k.removesuffix("/1") + "/2": v for k, v in SRE_SKILL_TOOLS.items()}


class OpsSreWorkingToolbox(OpsSreToolbox):
    profile_id = WORKING_PROFILE
    inspection_parameters = WORKING_INSPECTION_PARAMETERS
    tool_parameters = WORKING_TOOL_PARAMETERS
    skill_tools = WORKING_SKILL_TOOLS
    inspection_version = "sre-inspect/3"
    view_schema = "nurse.v2.sre-tool-view/4"
    focus_enabled = True
    diagnose_capacity = True


class OpsWorkingActionRecord(OpsEmpiricalActionRecord):
    failure_detail: OpsFailureDetail | None = None


class OpsSreWorkingLedgerState(SreLedgerFields):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/3"] = (
        "nurse.v2.sre-investigation-ledger/3"
    )
    investigation_profile_id: Literal["hermes-sre-tools/5"] = "hermes-sre-tools/5"
    records: tuple[OpsWorkingActionRecord, ...] = Field(max_length=8)


def _history(records: tuple[OpsActionRecord, ...]) -> list[dict[str, Any]]:
    from nurse_rcl.v2.live.ops_sre_context import _receipt

    return [dict(action_index=r.index, status=r.status, view=_receipt(r)[1]) for r in records]


class OpsSreWorkingInvestigationLedger(OpsSreInvestigationLedger):
    state_type: type[OpsLedgerStateBase] = OpsSreWorkingLedgerState
    skill_tools = WORKING_SKILL_TOOLS
    toolbox_type = OpsSreWorkingToolbox
    record_type: type[OpsActionRecord] = OpsWorkingActionRecord

    def _delivery_feedback(self, records: tuple[OpsActionRecord, ...]) -> dict[str, Any]:
        return dict(working_history(_history(records))["delivery_feedback"])

    def _failed_record(self, record: OpsActionRecord, exc: Exception) -> OpsActionRecord:
        if isinstance(exc, ExchangeBoundaryError) and exc.code == "json_limit":
            return record.model_copy(
                update={"failure_detail": failure_detail("investigation_tool", exc)}
            )
        return record

    def _finished_record(
        self,
        record: OpsActionRecord,
        outcome: OpsToolResult | None,
    ) -> OpsActionRecord:
        finished = super()._finished_record(record, outcome)
        if isinstance(finished, OpsEmpiricalActionRecord) and finished.model_view_json is not None:
            view = json.loads(finished.model_view_json)
            if view.get("status") != "resource_limit":
                view["delivery_feedback"] = self._delivery_feedback((*self._records[:-1], finished))
                finished = finished.model_copy(
                    update={"model_view_json": serialize_sre_view(view, diagnose=True)}
                )
            assert finished.model_view_json is not None
            projected = json.loads(finished.model_view_json)
            if projected.get("status") == "resource_limit":
                detail = (
                    OpsFailureDetail.model_validate(projected["failure_detail"])
                    if "failure_detail" in projected
                    else failure_detail("investigation_tool", ValueError("view_resource_limit"))
                )
                finished = finished.model_copy(update={"failure_detail": detail})
        return finished

    def _restore_tool_state(self, state: OpsLedgerStateBase) -> None:
        super()._restore_tool_state(state)
        assert isinstance(state, SreLedgerFields)
        for index, row in enumerate(state.records, 1):
            if row.model_view_json is None:
                continue
            view = json.loads(row.model_view_json)
            if view.get("status") != "resource_limit" and view.get("delivery_feedback") != (
                self._delivery_feedback(state.records[:index])
            ):
                raise ValueError("SRE delivery metadata differs from actual receipts")
