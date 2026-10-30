"""New owner identity, unchanged registered tools, scopes and source guards."""

from __future__ import annotations

import json
from typing import Any, Literal

from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
from nurse_rcl.v2.live.ops_sre_entry_decision import (
    resolve_entry_selection,
    validate_action_entries,
)
from nurse_rcl.v2.live.ops_sre_entry_tools import (
    OpsSreEntryInvestigationLedger,
    OpsSreEntryLedgerState,
    OpsSreEntryPublicationReader,
)
from nurse_rcl.v2.live.ops_sre_observable_tools import OBSERVABLE_SKILLS, OpsSreObservableToolbox
from nurse_rcl.v2.live.ops_sre_single_owner_decision import SINGLE_OWNER_TOOLS


def validate_single_owner_entries(table: dict[str, Any], **kwargs: Any) -> None:
    validate_action_entries(table, _single_owner=True, **kwargs)


class OpsSreSingleOwnerToolbox(OpsSreObservableToolbox):
    profile_id = SINGLE_OWNER_TOOLS


class OpsSreSingleOwnerLedgerState(OpsSreEntryLedgerState):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/22"] = (
        "nurse.v2.sre-investigation-ledger/22"  # type: ignore[assignment]
    )
    investigation_profile_id: Literal["hermes-sre-tools/21"] = SINGLE_OWNER_TOOLS  # type: ignore[assignment]


class OpsSreSingleOwnerInvestigationLedger(OpsSreEntryInvestigationLedger):
    state_type = OpsSreSingleOwnerLedgerState
    toolbox_type = OpsSreSingleOwnerToolbox
    skill_tools = OBSERVABLE_SKILLS
    validate_entries = staticmethod(validate_single_owner_entries)

    @staticmethod
    def _resolve(selection: dict[str, Any], frozen: str) -> dict[str, Any]:
        context = json.loads(frozen)
        refs = context["references"]
        book = SreActionReferenceBook.from_state(
            refs, snapshot=refs["snapshot"], run_id=refs["run_id"]
        )
        return resolve_entry_selection(
            selection,
            table=context["table"],
            book=book,
            request_index=context["request_index"],
            _single_owner=True,
        )


class OpsSreSingleOwnerPublicationReader(OpsSreEntryPublicationReader):
    toolbox_type = OpsSreSingleOwnerToolbox
