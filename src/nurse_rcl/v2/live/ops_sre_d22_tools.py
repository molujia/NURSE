"""D22 exact prior-receipt replay over the unchanged D21 deterministic tools."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_contracts import OpsInvestigationEvidence
from nurse_rcl.v2.live.ops_indexed_tools import OpsIndexedToolResult
from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord, OpsLedgerStateBase
from nurse_rcl.v2.live.ops_sre_context import _receipt
from nurse_rcl.v2.live.ops_sre_d21_tools import (
    USED_SOURCE_SKILL_TOOLS,
    OpsSreUsedSourceInvestigationLedger,
    OpsSreUsedSourcePublicationReader,
    OpsSreUsedSourceToolbox,
)
from nurse_rcl.v2.live.ops_sre_d22_reuse import replay_full_receipt
from nurse_rcl.v2.live.ops_sre_encoding import expand
from nurse_rcl.v2.live.ops_sre_ledger import SreLedgerFields
from nurse_rcl.v2.live.ops_sre_recovery_tools import OpsRecoveryActionRecord
from nurse_rcl.v2.live.ops_sre_score_cards import SCORE_CARD_VERSION
from nurse_rcl.v2.live.ops_sre_views import semantic_record
from nurse_rcl.v2.live.ops_tools import OpsToolResult, card_id

RECEIPT_REUSE_PROFILE: Literal["hermes-sre-tools/13"] = "hermes-sre-tools/13"
RECEIPT_REUSE_SKILL_TOOLS = {
    key.removesuffix("/9") + "/10": value for key, value in USED_SOURCE_SKILL_TOOLS.items()
}


class OpsSreReceiptReuseToolbox(OpsSreUsedSourceToolbox):
    profile_id = RECEIPT_REUSE_PROFILE  # type: ignore[assignment]
    skill_tools = RECEIPT_REUSE_SKILL_TOOLS

    def restore_evidence(self, saved: OpsInvestigationEvidence) -> None:
        source = self.checkpoint_source
        key = card_id(saved.evidence)
        if saved.evidence.tool_version == SCORE_CARD_VERSION:
            if (
                saved.origin != "read_card" or source is None or key not in source.cards
                or saved.evidence != source.cards[key]
                or self._cards.get(key) is not source.cards[key]
            ):
                raise ValueError("saved score receipt differs from admitted source")
            # The source registry is already admitted and identity-bound. A
            # read/replay receipt must not replace its card object on reopen.
            return
        super().restore_evidence(saved)


class OpsSreReceiptReuseActionRecord(OpsRecoveryActionRecord):
    receipt_reuse: dict[str, Any] | None = None


class OpsSreReceiptReuseLedgerState(SreLedgerFields):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/12"] = (
        "nurse.v2.sre-investigation-ledger/12"
    )
    investigation_profile_id: Literal["hermes-sre-tools/13"] = RECEIPT_REUSE_PROFILE
    records: tuple[OpsSreReceiptReuseActionRecord, ...] = Field(max_length=8)


class OpsSreReceiptReuseInvestigationLedger(OpsSreUsedSourceInvestigationLedger):
    state_type: type[OpsLedgerStateBase] = OpsSreReceiptReuseLedgerState
    record_type: type[OpsActionRecord] = OpsSreReceiptReuseActionRecord
    skill_tools = RECEIPT_REUSE_SKILL_TOOLS
    toolbox_type = OpsSreReceiptReuseToolbox

    def _current(self, card: str) -> dict[str, Any] | None:
        if self._indexed.object_contract(card) is None:
            return None
        source = self._indexed._cards.get(card)
        if source is None:
            return None
        return semantic_record(
            source, published=card in self._indexed._core_cards,
            origin=self._indexed._origins.get(card),
        )

    def _replay(self, action: Any) -> dict[str, Any] | None:
        if action.tool != "read_node":
            return None
        params = json.loads(action.parameters_json)
        if self._indexed.object_contract(params.get("card_id")) is None:
            return None
        previous = [dict(_receipt(record)[0], view=expand(_receipt(record)[1]))
                    for record in self.records[:-1]]
        return replay_full_receipt(
            dict(actions=[dict(tool=action.tool, parameters=params)]), previous,
            current_record=self._current, snapshot=self._indexed.case_snapshot_id,
        )

    def _invoke_tool(self, action: Any) -> OpsToolResult:
        reused = self._replay(action)
        if reused is None:
            return super()._invoke_tool(action)
        original = self.records[reused["prior_action_index"] - 1]
        assert original.result is not None
        view = reused.pop("view")
        view["receipt_reuse"] = reused
        limitations = list(view.get("limitations", []))
        if "replayed_prior_full_receipt_no_new_evidence" not in limitations:
            limitations.append("replayed_prior_full_receipt_no_new_evidence")
        view["limitations"] = limitations
        return OpsIndexedToolResult(
            evidence=original.result.evidence, origin=original.result.origin,
            reused_core_evidence_ids=original.result.reused_core_evidence_ids,
            model_view_json=canonical_json_bytes(view).decode(),
        )

    def _finished_record(
        self, record: OpsActionRecord, outcome: OpsToolResult | None,
    ) -> OpsActionRecord:
        finished = super()._finished_record(record, outcome)
        assert isinstance(finished, OpsSreReceiptReuseActionRecord)
        if finished.model_view_json is None:
            return finished
        proof = expand(json.loads(finished.model_view_json)).get("receipt_reuse")
        return finished.model_copy(update={"receipt_reuse": proof})

    def _restore_tool_state(self, state: OpsLedgerStateBase) -> None:
        super()._restore_tool_state(state)
        assert isinstance(state, OpsSreReceiptReuseLedgerState)
        for position, row in enumerate(state.records):
            if row.receipt_reuse is None:
                continue
            if row.model_view_json is None or row.action is None or row.status != "completed":
                raise ValueError("saved receipt reuse lacks completed action")
            actual_view = expand(json.loads(row.model_view_json))
            if actual_view.get("receipt_reuse") != row.receipt_reuse:
                raise ValueError("saved receipt reuse/view binding differs")
            previous = [dict(_receipt(prior)[0], view=expand(_receipt(prior)[1]))
                        for prior in state.records[:position]]
            expected = replay_full_receipt(
                dict(actions=[dict(tool=row.action.tool,
                                   parameters=json.loads(row.action.parameters_json))]),
                previous, current_record=self._current,
                snapshot=self._indexed.case_snapshot_id,
            )
            if expected is None:
                raise ValueError("saved receipt reuse source differs")
            source_view = expected.pop("view")
            if expected != row.receipt_reuse:
                raise ValueError("saved receipt reuse proof differs")
            original = state.records[expected["prior_action_index"] - 1]
            if (
                original.result is None
                or row.result is None
                or row.result.evidence != original.result.evidence
                or row.result.origin != original.result.origin
                or row.result.reused_core_evidence_ids
                != original.result.reused_core_evidence_ids
                or original.result.evidence != self._indexed._cards.get(
                    json.loads(row.action.parameters_json)["card_id"]
                )
            ):
                raise ValueError("saved receipt reuse result differs")
            source_view["receipt_reuse"] = expected
            source_view["limitations"] = list(source_view.get("limitations", []))
            if "replayed_prior_full_receipt_no_new_evidence" not in source_view["limitations"]:
                source_view["limitations"].append("replayed_prior_full_receipt_no_new_evidence")
            source_view.update(action_index=row.index, origin=original.result.origin)
            # The ordinary ledger already verifies the new action's derived
            # delivery_feedback. All remaining text and semantic fields must
            # be identical to the first actual receipt plus the replay marker.
            source_view.pop("delivery_feedback", None)
            actual_view.pop("delivery_feedback", None)
            if canonical_json_bytes(actual_view) != canonical_json_bytes(source_view):
                raise ValueError("saved receipt reuse/view content differs")


class OpsSreReceiptReusePublicationReader(OpsSreUsedSourcePublicationReader):
    toolbox_type = OpsSreReceiptReuseToolbox
