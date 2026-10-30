"""New selection ledger over the original reservation, reader and checkpoint chain."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import encode_json
from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord, OpsLedgerStateBase
from nurse_rcl.v2.live.ops_sre_action_entries import ENTRY_SKILLS, ENTRY_TOOLS
from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
from nurse_rcl.v2.live.ops_sre_d22_tools import OpsSreReceiptReuseActionRecord
from nurse_rcl.v2.live.ops_sre_directed_tools import (
    OpsSreDirectedInvestigationLedger,
    OpsSreDirectedLedgerState,
    OpsSreDirectedPublicationReader,
    OpsSreDirectedToolbox,
)
from nurse_rcl.v2.live.ops_sre_entry_decision import (
    resolve_entry_selection,
    validate_action_entries,
)
from nurse_rcl.v2.live.ops_sre_references import VIEW_LIMITS, SreReferenceBook
from nurse_rcl.v2.live.ops_sre_routes import route_binding
from nurse_rcl.v2.live.ops_tool_rejection import OpsToolRejection


class OpsSreEntryToolbox(OpsSreDirectedToolbox):
    profile_id = ENTRY_TOOLS
    skill_tools = ENTRY_SKILLS
    view_schema = "nurse.v2.sre-tool-view/14"

    def bind_visible_references(self, *, book: SreReferenceBook, request_index: int) -> None:
        if type(book) is not SreActionReferenceBook or book.snapshot != self.case_snapshot_id:
            raise ValueError("entry visibility identity differs")
        book.source_prefix(visible_at=request_index)
        self._visible_resources = {
            kind: {row[2] for row in book._rows if row[0] == kind and row[3] <= request_index}
            for kind in book.prefixes
            if kind != "action"
        }


class OpsEntryActionRecord(OpsSreReceiptReuseActionRecord):
    selection_json: str | None = None
    frozen_entry_context_json: str | None = None
    selection_reason_code: (
        Literal["entry_not_available", "navigation_reference_not_available", "invalid_navigation"]
        | None
    ) = None


class OpsSreEntryLedgerState(OpsSreDirectedLedgerState):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/20"] = (
        "nurse.v2.sre-investigation-ledger/20"  # type: ignore[assignment]
    )
    investigation_profile_id: Literal["hermes-sre-tools/19"] = ENTRY_TOOLS  # type: ignore[assignment]
    records: tuple[OpsEntryActionRecord, ...] = Field(max_length=8)


class OpsSreEntryInvestigationLedger(OpsSreDirectedInvestigationLedger):
    state_type: type[OpsLedgerStateBase] = OpsSreEntryLedgerState
    record_type: type[OpsActionRecord] = OpsEntryActionRecord
    toolbox_type = OpsSreEntryToolbox
    skill_tools = ENTRY_SKILLS
    validate_entries = staticmethod(validate_action_entries)

    def freeze_entries(
        self,
        *,
        table: dict[str, Any],
        book: SreActionReferenceBook,
        request_index: int,
        case_id: str,
    ) -> None:
        self._require_usable()
        if (
            book.snapshot != self._indexed.case_snapshot_id
            or case_id != self._indexed._data.catalog.opaque_case_id
            or type(request_index) is not int
            or request_index != self.used + 1
        ):
            raise ValueError("frozen entry owner binding differs")
        self.validate_entries(table, book=book, request_index=request_index)
        self._frozen_entry_context_json = encode_json(
            dict(
                table=table,
                references=book.to_state(),
                request_index=request_index,
                case_id=case_id,
            ),
            VIEW_LIMITS,
        ).decode()
        self._indexed.bind_visible_references(book=book, request_index=request_index)

    @staticmethod
    def _resolve(selection: dict[str, Any], frozen: str) -> dict[str, Any]:
        context = json.loads(frozen)
        refs = context["references"]
        book = SreActionReferenceBook.from_state(
            refs, snapshot=refs["snapshot"], run_id=refs["run_id"]
        )
        return resolve_entry_selection(
            selection, table=context["table"], book=book, request_index=context["request_index"]
        )

    def _admit_proposal(
        self, proposals: object, raw: bytes, record: OpsActionRecord
    ) -> OpsActionRecord:
        if (
            len(raw) > self.proposal_bytes
            or type(proposals) is not list
            or len(proposals) != 1
            or type(proposals[0]) is not dict
            or not hasattr(self, "_frozen_entry_context_json")
        ):
            raise ValueError("one frozen entry selection required")
        return self._admit_frozen_selection(
            proposals, record, selected=proposals[0].get("entry_ref")
        )

    def _admit_frozen_selection(
        self,
        proposals: list[Any],
        record: OpsActionRecord,
        *,
        selected: str | None,
    ) -> OpsActionRecord:
        """Shared source/route guards after each identity validates its true native shape."""
        record = record.model_copy(
            update=dict(
                selection_json=canonical_json_bytes(proposals[0]).decode(),
                frozen_entry_context_json=self._frozen_entry_context_json,
            )
        )
        self._replace(record)  # Retain selection and exact frozen binding even when rejected.
        resolved = self._resolve(proposals[0], self._frozen_entry_context_json)
        if resolved["status"] != "accepted":
            self._replace(
                record.model_copy(update={"selection_reason_code": resolved["reason_code"]})
            )
            raise OpsToolRejection(
                "reference_not_available"
                if resolved["reason_code"] != "invalid_navigation"
                else "invalid_parameters"
            )
        canonical = resolved["action"]
        record = super()._admit_proposal([canonical], canonical_json_bytes([canonical]), record)
        self._replace(record)
        params = canonical["parameters"]
        pin = route_binding(self._indexed, canonical["tool"], params)
        if selected is not None:
            context = json.loads(self._frozen_entry_context_json)
            entry = next(e for e in context["table"]["entries"] if e["entry_ref"] == selected)
            binding = entry["source_binding"]
            if "route_binding_sha256" in binding and binding["route_binding_sha256"] != pin:
                raise OpsToolRejection("request_binding_mismatch")
            if "record_sha256" in binding:
                self._indexed._refresh_semantic()
                actual = self._indexed._semantic.get(binding["card_id"])
                if (
                    actual is None
                    or sha256_hex(canonical_json_bytes(actual)) != binding["record_sha256"]
                ):
                    raise OpsToolRejection("request_binding_mismatch")
        return record

    def _restore_tool_state(self, state: OpsLedgerStateBase) -> None:
        super()._restore_tool_state(state)
        assert isinstance(state, OpsSreEntryLedgerState)
        for row in state.records:
            if row.selection_json is None:
                if row.action is not None or row.frozen_entry_context_json is not None:
                    raise ValueError("saved entry selection binding differs")
                continue
            if row.frozen_entry_context_json is None:
                raise ValueError("saved entry context binding differs")
            frozen = json.loads(row.frozen_entry_context_json)
            if (
                type(frozen) is not dict
                or set(frozen) != {"table", "references", "request_index", "case_id"}
                or frozen["case_id"] != self._indexed._data.catalog.opaque_case_id
                or type(frozen["request_index"]) is not int
                or frozen["request_index"] != row.index
                or frozen["references"]["snapshot"] != state.case_snapshot_id
            ):
                raise ValueError("saved entry owner binding differs")
            selection = json.loads(row.selection_json)
            if canonical_json_bytes(selection).decode() != row.selection_json or (
                sha256_hex(canonical_json_bytes([selection])) != row.input_sha256
            ):
                raise ValueError("saved entry native selection binding differs")
            resolved = self._resolve_for_restore(
                selection, row.frozen_entry_context_json, has_action=row.action is not None
            )
            if row.action is not None:
                expected = dict(
                    action_index=row.index,
                    **{k: v for k, v in resolved["action"].items() if k != "parameters"},
                    parameters_json=canonical_json_bytes(resolved["action"]["parameters"]).decode(),
                )
                if row.action.model_dump(mode="json") != expected:
                    raise ValueError("saved entry actual action binding differs")
            elif (
                row.selection_reason_code is not None
                and row.selection_reason_code != resolved["reason_code"]
            ):
                raise ValueError("saved entry rejection binding differs")

    def _resolve_for_restore(
        self,
        selection: Any,
        frozen: str,
        *,
        has_action: bool,
    ) -> dict[str, Any]:
        """Legacy identities keep their original resolution and error domain."""
        return self._resolve(selection, frozen)


class OpsSreEntryPublicationReader(OpsSreDirectedPublicationReader):
    toolbox_type = OpsSreEntryToolbox
