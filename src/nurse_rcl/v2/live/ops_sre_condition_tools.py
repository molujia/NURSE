"""D14 tool/ledger identity; fixed conditions explain, never rescore, sources."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_indexed_tools import OpsIndexedToolResult
from nurse_rcl.v2.live.ops_investigation_ledger import OpsLedgerStateBase
from nurse_rcl.v2.live.ops_sre_brief_tools import (
    BRIEF_SKILL_TOOLS,
    OpsSreBriefInvestigationLedger,
    OpsSreBriefToolbox,
)
from nurse_rcl.v2.live.ops_sre_condition_briefs import project_condition_brief
from nurse_rcl.v2.live.ops_sre_encoding import expand
from nurse_rcl.v2.live.ops_sre_feedback_context import CONDITION_PROFILE
from nurse_rcl.v2.live.ops_sre_ledger import SreLedgerFields
from nurse_rcl.v2.live.ops_sre_working_tools import OpsWorkingActionRecord

CONDITION_SKILL_TOOLS = {k.removesuffix("/3") + "/4": v for k, v in BRIEF_SKILL_TOOLS.items()}


class OpsSreConditionToolbox(OpsSreBriefToolbox):
    profile_id = CONDITION_PROFILE
    skill_tools = CONDITION_SKILL_TOOLS
    inspection_version = "sre-inspect/5"
    view_schema = "nurse.v2.sre-tool-view/6"

    def _inspection_record(self, row: dict[str, Any]) -> dict[str, Any]:
        return project_condition_brief(row)

    def _execute(self, tool: str, parameters: object) -> OpsIndexedToolResult:
        result = super()._execute(tool, parameters)
        if tool == "check_mechanism":
            view = expand(json.loads(result.model_view_json))
            if "semantic_record" in view:
                view["semantic_record"] = project_condition_brief(view["semantic_record"])
                view["representation"] = "condition_evidence_brief"
                result = result.model_copy(update={"model_view_json": self._serialize_view(view)})
        return result

    def retained_brief_source(self, view: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
        if view.get("publication_prefix") is not None:
            return super().retained_brief_source(view, row)
        # Direct check_mechanism has an actual saved card, not an inspection prefix.
        self._refresh_semantic()
        source = self._semantic.get(row["card_id"])
        if source is None or row.get("mechanism_source_kind") != "supplementary_recheck":
            raise ValueError("direct brief source differs")
        if canonical_json_bytes(project_condition_brief(source)) != canonical_json_bytes(row):
            raise ValueError("brief differs from retained source")
        return source


class OpsSreConditionLedgerState(SreLedgerFields):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/5"] = (
        "nurse.v2.sre-investigation-ledger/5"
    )
    investigation_profile_id: Literal["hermes-sre-tools/7"] = "hermes-sre-tools/7"
    records: tuple[OpsWorkingActionRecord, ...] = Field(max_length=8)


class OpsSreConditionInvestigationLedger(OpsSreBriefInvestigationLedger):
    state_type: type[OpsLedgerStateBase] = OpsSreConditionLedgerState
    skill_tools = CONDITION_SKILL_TOOLS
    toolbox_type = OpsSreConditionToolbox
