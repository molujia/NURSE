"""D18 identity over unchanged admitted tools, ledger and checkpoint sources."""

from typing import Literal

from pydantic import Field

from nurse_rcl.v2.live.ops_investigation_ledger import OpsLedgerStateBase
from nurse_rcl.v2.live.ops_sre_ledger import SreLedgerFields
from nurse_rcl.v2.live.ops_sre_navigation_tools import (
    NAVIGATION_SKILL_TOOLS,
    OpsSreNavigationInvestigationLedger,
    OpsSreNavigationPublicationReader,
    OpsSreNavigationToolbox,
)
from nurse_rcl.v2.live.ops_sre_recovery_tools import OpsRecoveryActionRecord

WORKBENCH_PROFILE = "hermes-sre-tools/11"
WORKBENCH_SKILL_TOOLS = {k.removesuffix("/7") + "/8": v for k, v in NAVIGATION_SKILL_TOOLS.items()}


class OpsSreWorkbenchToolbox(OpsSreNavigationToolbox):
    profile_id = WORKBENCH_PROFILE
    skill_tools = WORKBENCH_SKILL_TOOLS
    # Tool record semantics are unchanged; profile and skill identities differ.


class OpsSreWorkbenchLedgerState(SreLedgerFields):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/10"] = (
        "nurse.v2.sre-investigation-ledger/10"
    )
    investigation_profile_id: Literal["hermes-sre-tools/11"] = "hermes-sre-tools/11"
    records: tuple[OpsRecoveryActionRecord, ...] = Field(max_length=8)


class OpsSreWorkbenchInvestigationLedger(OpsSreNavigationInvestigationLedger):
    state_type: type[OpsLedgerStateBase] = OpsSreWorkbenchLedgerState
    skill_tools = WORKBENCH_SKILL_TOOLS
    toolbox_type = OpsSreWorkbenchToolbox


class OpsSreWorkbenchPublicationReader(OpsSreNavigationPublicationReader):
    toolbox_type = OpsSreWorkbenchToolbox
