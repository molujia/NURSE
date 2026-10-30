"""Real string native ledger over the original frozen source and dispatch guards."""

import json
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.v2.live.ops_contracts import OpsInvestigationAction
from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord
from nurse_rcl.v2.live.ops_sre_action_entries import ENTRY_SKILLS
from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
from nurse_rcl.v2.live.ops_sre_atomic_selector_decision import (
    ATOMIC_TOOLS,
    resolve_atomic_selection,
    selection_shape,
    validate_atomic_entries,
)
from nurse_rcl.v2.live.ops_sre_entry_tools import (
    OpsEntryActionRecord,
    OpsSreEntryInvestigationLedger,
    OpsSreEntryLedgerState,
    OpsSreEntryPublicationReader,
)
from nurse_rcl.v2.live.ops_sre_observable_tools import OpsSreObservableToolbox

ATOMIC_SKILLS = {k.removesuffix("/15") + "/19": v for k, v in ENTRY_SKILLS.items()}


class OpsAtomicSelectorAction(OpsInvestigationAction):
    question: str | None = Field(strict=True, min_length=1)  # type: ignore[assignment]


class OpsAtomicSelectorActionRecord(OpsEntryActionRecord):
    action: OpsAtomicSelectorAction | None = None  # type: ignore[assignment]


class OpsSreAtomicSelectorLedgerState(OpsSreEntryLedgerState):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/24"] = (
        "nurse.v2.sre-investigation-ledger/24"  # type: ignore[assignment]
    )
    investigation_profile_id: Literal["hermes-sre-tools/23"] = ATOMIC_TOOLS  # type: ignore[assignment]
    records: tuple[OpsAtomicSelectorActionRecord, ...] = Field(max_length=8)


class OpsSreAtomicSelectorToolbox(OpsSreObservableToolbox):
    profile_id = ATOMIC_TOOLS
    skill_tools = ATOMIC_SKILLS
    view_schema = "nurse.v2.sre-tool-view/17"


class OpsSreAtomicSelectorInvestigationLedger(OpsSreEntryInvestigationLedger):
    state_type = OpsSreAtomicSelectorLedgerState
    record_type = OpsAtomicSelectorActionRecord
    action_type = OpsAtomicSelectorAction  # type: ignore[assignment]
    toolbox_type = OpsSreAtomicSelectorToolbox
    skill_tools = ATOMIC_SKILLS
    validate_entries = staticmethod(validate_atomic_entries)

    @staticmethod
    def _resolve(selection: Any, frozen: str) -> dict[str, Any]:
        context = json.loads(frozen)
        refs = context["references"]
        book = SreActionReferenceBook.from_state(
            refs, snapshot=refs["snapshot"], run_id=refs["run_id"]
        )
        return resolve_atomic_selection(
            selection, table=context["table"], book=book, request_index=context["request_index"]
        )

    def _admit_proposal(
        self,
        proposals: object,
        raw: bytes,
        record: OpsActionRecord,
    ) -> OpsActionRecord:
        if (
            len(raw) > self.proposal_bytes
            or type(proposals) is not list
            or len(proposals) != 1
            or not hasattr(self, "_frozen_entry_context_json")
        ):
            raise ValueError("one frozen atomic selection required")
        selection_shape(proposals[0])
        return self._admit_frozen_selection(
            proposals,
            record,
            selected=proposals[0] if type(proposals[0]) is str else None,
        )

    def _resolve_for_restore(
        self,
        selection: Any,
        frozen: str,
        *,
        has_action: bool,
    ) -> dict[str, Any]:
        resolved = self._resolve(selection, frozen)
        if has_action and resolved["status"] != "accepted":
            raise ValueError("saved entry actual action binding differs")
        return resolved


class OpsSreAtomicSelectorPublicationReader(OpsSreEntryPublicationReader):
    toolbox_type = OpsSreAtomicSelectorToolbox
