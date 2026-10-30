"""B reuses the fixed tools and column-pool codec with separate run identity."""

from typing import Literal

from nurse_rcl.v2.live.ops_investigation_ledger import OpsLedgerStateBase
from nurse_rcl.v2.live.ops_sre_checkpoint_pools_ledger import (
    OpsSrePooledInvestigationLedger,
    OpsSrePooledLedgerState,
)
from nurse_rcl.v2.live.ops_sre_d26_tools import (
    QUESTION_SKILL_TOOLS,
    OpsSreQuestionPublicationReader,
    OpsSreQuestionToolbox,
)

ACTION_PROFILE = "hermes-sre-tools/17"


class OpsSreActionToolbox(OpsSreQuestionToolbox):
    profile_id = ACTION_PROFILE  # type: ignore[assignment]


class OpsSreActionLedgerState(OpsSrePooledLedgerState):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/18"] = (
        "nurse.v2.sre-investigation-ledger/18"  # type: ignore[assignment]
    )
    investigation_profile_id: Literal["hermes-sre-tools/17"] = ACTION_PROFILE  # type: ignore[assignment]


class OpsSreActionInvestigationLedger(OpsSrePooledInvestigationLedger):
    state_type: type[OpsLedgerStateBase] = OpsSreActionLedgerState
    toolbox_type = OpsSreActionToolbox
    skill_tools = QUESTION_SKILL_TOOLS


class OpsSreActionPublicationReader(OpsSreQuestionPublicationReader):
    toolbox_type = OpsSreActionToolbox
