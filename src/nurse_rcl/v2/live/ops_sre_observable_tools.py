"""Isolated scope identity; all original action, source and capacity guards remain."""

from __future__ import annotations

import json
from typing import Any, Literal

from nurse_rcl.v2.live.ops_sre_action_entries import ENTRY_SKILLS
from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
from nurse_rcl.v2.live.ops_sre_directed_sources import directed_source_navigation
from nurse_rcl.v2.live.ops_sre_entry_decision import (
    resolve_entry_selection,
    validate_action_entries,
)
from nurse_rcl.v2.live.ops_sre_entry_tools import (
    OpsSreEntryInvestigationLedger,
    OpsSreEntryLedgerState,
    OpsSreEntryPublicationReader,
    OpsSreEntryToolbox,
)
from nurse_rcl.v2.live.ops_sre_observable_scope import OBSERVABLE_TOOLS

OBSERVABLE_SKILLS = {k.removesuffix("/15") + "/16": v for k, v in ENTRY_SKILLS.items()}


def validate_observable_entries(table: dict[str, Any], **kwargs: Any) -> None:
    validate_action_entries(table, _observable=True, **kwargs)


def resolve_observable_selection(selection: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return resolve_entry_selection(selection, _observable=True, **kwargs)


class OpsSreObservableToolbox(OpsSreEntryToolbox):
    profile_id = OBSERVABLE_TOOLS
    skill_tools = OBSERVABLE_SKILLS
    view_schema = "nurse.v2.sre-tool-view/15"

    def _directed_navigation(self, scope: list[str]) -> dict[str, Any]:
        self._refresh_semantic()
        return directed_source_navigation(list(self._semantic.values()), scope, observable=True)


class OpsSreObservableLedgerState(OpsSreEntryLedgerState):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/21"] = (
        "nurse.v2.sre-investigation-ledger/21"  # type: ignore[assignment]
    )
    investigation_profile_id: Literal["hermes-sre-tools/20"] = OBSERVABLE_TOOLS  # type: ignore[assignment]


class OpsSreObservableInvestigationLedger(OpsSreEntryInvestigationLedger):
    state_type = OpsSreObservableLedgerState
    toolbox_type = OpsSreObservableToolbox
    skill_tools = OBSERVABLE_SKILLS
    validate_entries = staticmethod(validate_observable_entries)

    @staticmethod
    def _resolve(selection: dict[str, Any], frozen: str) -> dict[str, Any]:
        context = json.loads(frozen)
        refs = context["references"]
        book = SreActionReferenceBook.from_state(
            refs, snapshot=refs["snapshot"], run_id=refs["run_id"]
        )
        return resolve_observable_selection(
            selection,
            table=context["table"],
            book=book,
            request_index=context["request_index"],
        )


class OpsSreObservablePublicationReader(OpsSreEntryPublicationReader):
    toolbox_type = OpsSreObservableToolbox
