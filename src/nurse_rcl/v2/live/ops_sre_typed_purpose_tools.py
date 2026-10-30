"""New nullable action identity over unchanged source/admission/dispatch guards."""

import json
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.v2.live.ops_contracts import OpsInvestigationAction
from nurse_rcl.v2.live.ops_sre_action_entries import ENTRY_SKILLS
from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
from nurse_rcl.v2.live.ops_sre_entry_decision import validate_action_entries
from nurse_rcl.v2.live.ops_sre_entry_tools import (
    OpsEntryActionRecord,
    OpsSreEntryInvestigationLedger,
    OpsSreEntryLedgerState,
    OpsSreEntryPublicationReader,
)
from nurse_rcl.v2.live.ops_sre_observable_tools import OpsSreObservableToolbox
from nurse_rcl.v2.live.ops_sre_typed_purpose_decision import TYPED_TOOLS, resolve_typed_selection

TYPED_SKILLS = {k.removesuffix("/15") + "/18": v for k, v in ENTRY_SKILLS.items()}


class OpsTypedPurposeAction(OpsInvestigationAction):
    # Pydantic leaf schema intentionally replaces only this new identity's
    # field; legacy model classes and their non-null validation stay unchanged.
    question: str | None = Field(strict=True, min_length=1)  # type: ignore[assignment]


class OpsTypedPurposeActionRecord(OpsEntryActionRecord):
    action: OpsTypedPurposeAction | None = None  # type: ignore[assignment]


class OpsSreTypedPurposeLedgerState(OpsSreEntryLedgerState):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/23"] = (
        "nurse.v2.sre-investigation-ledger/23"  # type: ignore[assignment]
    )
    investigation_profile_id: Literal["hermes-sre-tools/22"] = TYPED_TOOLS  # type: ignore[assignment]
    records: tuple[OpsTypedPurposeActionRecord, ...] = Field(max_length=8)


class OpsSreTypedPurposeToolbox(OpsSreObservableToolbox):
    profile_id = TYPED_TOOLS
    skill_tools = TYPED_SKILLS
    view_schema = "nurse.v2.sre-tool-view/16"


def validate_typed_entries(table: dict[str, Any], **kwargs: Any) -> None:
    validate_action_entries(table, _typed_purpose=True, **kwargs)


class OpsSreTypedPurposeInvestigationLedger(OpsSreEntryInvestigationLedger):
    state_type = OpsSreTypedPurposeLedgerState
    record_type = OpsTypedPurposeActionRecord
    action_type = OpsTypedPurposeAction  # type: ignore[assignment]
    toolbox_type = OpsSreTypedPurposeToolbox
    skill_tools = TYPED_SKILLS
    validate_entries = staticmethod(validate_typed_entries)

    @staticmethod
    def _resolve(selection: dict[str, Any], frozen: str) -> dict[str, Any]:
        context = json.loads(frozen)
        refs = context["references"]
        book = SreActionReferenceBook.from_state(
            refs, snapshot=refs["snapshot"], run_id=refs["run_id"]
        )
        return resolve_typed_selection(
            selection, table=context["table"], book=book, request_index=context["request_index"]
        )


class OpsSreTypedPurposePublicationReader(OpsSreEntryPublicationReader):
    toolbox_type = OpsSreTypedPurposeToolbox
