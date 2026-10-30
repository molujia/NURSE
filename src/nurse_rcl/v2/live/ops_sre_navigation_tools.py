"""D17 owner identity over unchanged scoring tools and source checkpoint codec."""

from __future__ import annotations

import json
from typing import Any, ClassVar, Literal

from pydantic import Field

from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord, OpsLedgerStateBase
from nurse_rcl.v2.live.ops_sre_comparable_publication import OpsSreComparablePublicationReader
from nurse_rcl.v2.live.ops_sre_comparable_tools import (
    COMPARABLE_SKILL_TOOLS,
    OpsSreComparableInvestigationLedger,
    OpsSreComparableToolbox,
)
from nurse_rcl.v2.live.ops_sre_condition_tools import OpsSreConditionInvestigationLedger
from nurse_rcl.v2.live.ops_sre_delivered_navigation import verify_navigation_source
from nurse_rcl.v2.live.ops_sre_encoding import expand
from nurse_rcl.v2.live.ops_sre_ledger import SreLedgerFields
from nurse_rcl.v2.live.ops_sre_recovery_tools import OpsRecoveryActionRecord

NAVIGATION_PROFILE = "hermes-sre-tools/10"
NAVIGATION_SKILL_TOOLS = {k.removesuffix("/6") + "/7": v for k, v in COMPARABLE_SKILL_TOOLS.items()}


def receipt_history(records: tuple[Any, ...]) -> list[dict[str, Any]]:
    from nurse_rcl.v2.live.ops_sre_context import _receipt

    return [dict(action_index=r.index, status=r.status, view=_receipt(r)[1]) for r in records]


class OpsSreNavigationToolbox(OpsSreComparableToolbox):
    profile_id = NAVIGATION_PROFILE
    skill_tools = NAVIGATION_SKILL_TOOLS
    view_schema = "nurse.v2.sre-tool-view/9"
    score_record_types: ClassVar[dict[str, str]] = {
        "score_lineage_summary": (
            "rank, components, aggregation configuration and typed detail links; "
            "not term or pool detail"
        ),
        "score_family": (
            "actual primary_family, representative, family aggregation status and "
            "delivered selection chain; unread terms remain unknown"
        ),
        "score_term": (
            "this term's original inputs, pool selection and role_mechanism_ids; not other terms"
        ),
        "score_propagation": (
            "published propagation summary and typed path links; "
            "zero does not prove absence of propagation"
        ),
        "score_path": "this delivered path's inputs and gaps, not unread paths",
    }

    def capabilities(self, records: tuple[OpsActionRecord, ...] = ()) -> dict[str, Any]:
        value = super().capabilities(records)
        value["score_record_types"] = dict(self.score_record_types)
        return value


class OpsSreNavigationLedgerState(SreLedgerFields):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/9"] = (
        "nurse.v2.sre-investigation-ledger/9"
    )
    investigation_profile_id: Literal["hermes-sre-tools/10"] = "hermes-sre-tools/10"
    records: tuple[OpsRecoveryActionRecord, ...] = Field(max_length=8)


class OpsSreNavigationPublicationReader(OpsSreComparablePublicationReader):
    toolbox_type = OpsSreNavigationToolbox


class OpsSreNavigationInvestigationLedger(OpsSreComparableInvestigationLedger):
    _indexed: OpsSreNavigationToolbox
    state_type: type[OpsLedgerStateBase] = OpsSreNavigationLedgerState
    skill_tools = NAVIGATION_SKILL_TOOLS
    toolbox_type = OpsSreNavigationToolbox

    def _restore_tool_state(self, state: OpsLedgerStateBase) -> None:
        OpsSreConditionInvestigationLedger._restore_tool_state(self, state)
        assert isinstance(state, self.state_type)
        for row in state.records:
            assert isinstance(row, OpsRecoveryActionRecord)
            if row.rejection is None or row.model_view_json is None:
                continue
            view = expand(json.loads(row.model_view_json))
            if view.get("status") != "resource_limit" and (
                view.get("parameter_feedback", {}).get("diagnostic")
                != row.rejection.model_dump(mode="json")
                or view.get("route_binding_sha256") != row.request_binding_sha256
            ):
                raise ValueError("saved navigation rejection binding differs")
        verify_navigation_source(
            receipt_history(state.records), source=self._indexed.checkpoint_source
        )
