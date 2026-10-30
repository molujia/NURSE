"""D25 opt-in source-bound replanning, preserving D23 source tool admission."""

from typing import Literal

from nurse_rcl.v2.live.ops_investigation_ledger import OpsLedgerStateBase
from nurse_rcl.v2.live.ops_sre_d23_tools import (
    BALANCED_SKILL_TOOLS,
    OpsSreBalancedInvestigationLedger,
    OpsSreBalancedLedgerState,
    OpsSreBalancedPublicationReader,
    OpsSreBalancedToolbox,
)

REPLANNING_PROFILE: Literal["hermes-sre-tools/15"] = "hermes-sre-tools/15"
REPLANNING_SKILL_TOOLS = {
    key.removesuffix("/11") + "/12": value for key, value in BALANCED_SKILL_TOOLS.items()
}


class OpsSreReplanningToolbox(OpsSreBalancedToolbox):
    profile_id = REPLANNING_PROFILE  # type: ignore[assignment]
    skill_tools = REPLANNING_SKILL_TOOLS
    view_schema = "nurse.v2.sre-tool-view/11"


class OpsSreReplanningLedgerState(OpsSreBalancedLedgerState):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/14"] = (
        "nurse.v2.sre-investigation-ledger/14"  # type: ignore[assignment]
    )
    investigation_profile_id: Literal["hermes-sre-tools/15"] = REPLANNING_PROFILE  # type: ignore[assignment]


class OpsSreReplanningInvestigationLedger(OpsSreBalancedInvestigationLedger):
    state_type: type[OpsLedgerStateBase] = OpsSreReplanningLedgerState
    skill_tools = REPLANNING_SKILL_TOOLS
    toolbox_type = OpsSreReplanningToolbox


class OpsSreReplanningPublicationReader(OpsSreBalancedPublicationReader):
    toolbox_type = OpsSreReplanningToolbox
