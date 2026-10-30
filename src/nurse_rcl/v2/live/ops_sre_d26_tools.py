"""D26 independent identity over the unchanged, admitted D25 source tools."""

from typing import Literal

from nurse_rcl.v2.live.ops_investigation_ledger import OpsLedgerStateBase
from nurse_rcl.v2.live.ops_sre_d25_tools import (
    REPLANNING_SKILL_TOOLS,
    OpsSreReplanningInvestigationLedger,
    OpsSreReplanningLedgerState,
    OpsSreReplanningPublicationReader,
    OpsSreReplanningToolbox,
)

QUESTION_PROFILE: Literal["hermes-sre-tools/16"] = "hermes-sre-tools/16"
QUESTION_SKILL_TOOLS = {
    key.removesuffix("/12") + "/13": value for key, value in REPLANNING_SKILL_TOOLS.items()
}


class OpsSreQuestionToolbox(OpsSreReplanningToolbox):
    profile_id = QUESTION_PROFILE  # type: ignore[assignment]
    skill_tools = QUESTION_SKILL_TOOLS
    view_schema = "nurse.v2.sre-tool-view/12"


class OpsSreQuestionLedgerState(OpsSreReplanningLedgerState):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/15"] = (
        "nurse.v2.sre-investigation-ledger/15"  # type: ignore[assignment]
    )
    investigation_profile_id: Literal["hermes-sre-tools/16"] = QUESTION_PROFILE  # type: ignore[assignment]


class OpsSreQuestionInvestigationLedger(OpsSreReplanningInvestigationLedger):
    state_type: type[OpsLedgerStateBase] = OpsSreQuestionLedgerState
    skill_tools = QUESTION_SKILL_TOOLS
    toolbox_type = OpsSreQuestionToolbox


class OpsSreQuestionPublicationReader(OpsSreReplanningPublicationReader):
    toolbox_type = OpsSreQuestionToolbox
