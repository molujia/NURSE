"""D21 opt-in score contrast; inherited tools and source cards remain authoritative."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from nurse_rcl.v2.live.ops_investigation_ledger import OpsLedgerStateBase
from nurse_rcl.v2.live.ops_sre_ledger import SreLedgerFields
from nurse_rcl.v2.live.ops_sre_recovery_tools import OpsRecoveryActionRecord
from nurse_rcl.v2.live.ops_sre_score_cards import score_pairs
from nurse_rcl.v2.live.ops_sre_workbench_tools import (
    WORKBENCH_SKILL_TOOLS,
    OpsSreWorkbenchInvestigationLedger,
    OpsSreWorkbenchPublicationReader,
    OpsSreWorkbenchToolbox,
)

USED_SOURCE_PROFILE: Literal["hermes-sre-tools/12"] = "hermes-sre-tools/12"
USED_SOURCE_SKILL_TOOLS = {
    key.removesuffix("/8") + "/9": value for key, value in WORKBENCH_SKILL_TOOLS.items()
}


class OpsSreUsedSourceToolbox(OpsSreWorkbenchToolbox):
    profile_id = USED_SOURCE_PROFILE
    skill_tools = USED_SOURCE_SKILL_TOOLS

    @staticmethod
    def _score_pair_units(rows: list[dict[str, Any]], scope: list[str]) -> list[dict[str, Any]]:
        return score_pairs(rows, scope, used_source_contrast=True)


class OpsSreUsedSourceLedgerState(SreLedgerFields):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/11"] = (
        "nurse.v2.sre-investigation-ledger/11"
    )
    investigation_profile_id: Literal["hermes-sre-tools/12"] = USED_SOURCE_PROFILE
    records: tuple[OpsRecoveryActionRecord, ...] = Field(max_length=8)


class OpsSreUsedSourceInvestigationLedger(OpsSreWorkbenchInvestigationLedger):
    state_type: type[OpsLedgerStateBase] = OpsSreUsedSourceLedgerState
    skill_tools = USED_SOURCE_SKILL_TOOLS
    toolbox_type = OpsSreUsedSourceToolbox


class OpsSreUsedSourcePublicationReader(OpsSreWorkbenchPublicationReader):
    toolbox_type = OpsSreUsedSourceToolbox
