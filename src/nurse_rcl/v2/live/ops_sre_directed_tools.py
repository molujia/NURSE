"""Opt-in directed symptom receipts; no live-owner/profile registration yet."""

from __future__ import annotations

import json
from typing import Any, ClassVar, Literal, cast

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_contracts import OpsNodeEvidence
from nurse_rcl.v2.live.ops_indexed_tools import OpsIndexedToolResult
from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord, OpsLedgerStateBase
from nurse_rcl.v2.live.ops_node_common import node_card
from nurse_rcl.v2.live.ops_sre_action_tools import (
    OpsSreActionInvestigationLedger,
    OpsSreActionLedgerState,
    OpsSreActionPublicationReader,
    OpsSreActionToolbox,
)
from nurse_rcl.v2.live.ops_sre_checkpoint_ledger import CheckpointReceiptError
from nurse_rcl.v2.live.ops_sre_directed_sources import (
    directed_source_binding,
    directed_source_navigation,
    directed_source_relation,
)
from nurse_rcl.v2.live.ops_sre_encoding import SEMANTIC_ENCODING, compact_record
from nurse_rcl.v2.live.ops_tools import OpsToolResult, card_id

DIRECTED_PROFILE = "hermes-sre-tools/18"
DIRECTED_SKILL_TOOLS = {
    key.removesuffix("/13") + "/14": tuple(dict.fromkeys(
        (*tools, "inspect_sources") if key.startswith("sre-symptoms/") else tools,
    )) for key, tools in OpsSreActionToolbox.skill_tools.items()
}


class OpsSreDirectedToolbox(OpsSreActionToolbox):
    profile_id = DIRECTED_PROFILE
    skill_tools = DIRECTED_SKILL_TOOLS
    tool_parameters: ClassVar[dict[str, tuple[str, ...]]] = {  # type: ignore[misc]
        **OpsSreActionToolbox.tool_parameters,
        "inspect_sources": ("card_ids", "candidate_service_keys"),
    }
    view_schema = "nurse.v2.sre-tool-view/13"

    def _investigation_routes(self, records: tuple[OpsActionRecord, ...]) -> dict[str, Any]:
        from nurse_rcl.v2.live.ops_sre_directed_recovery import directed_recovery_routes

        return directed_recovery_routes(self, records)

    def _directed_navigation(self, scope: list[str]) -> dict[str, Any]:
        self._refresh_semantic()
        return directed_source_navigation(list(self._semantic.values()), scope)

    def capabilities(self, records: tuple[OpsActionRecord, ...] = ()) -> dict[str, Any]:
        value = super().capabilities(records)
        value["directed_sources"] = self._directed_navigation(sorted(self._by_candidate))
        value["parameter_contracts"]["inspect_sources"] = dict(
            card_ids="one or two distinct issued symptom card handles in this request prefix",
            candidate_service_keys=(
                "one or two distinct issued candidates; all card targets must fit"
            ),
        )
        result: dict[str, Any] = json.loads(canonical_json_bytes(value))
        return result

    def _inspection_view(self, tool: str, parameters: dict[str, Any]) -> dict[str, Any]:
        value = super()._inspection_view(tool, parameters)
        if tool in {"inspect_candidate", "compare_candidates"}:
            value["directed_sources"] = self._directed_navigation(value["scope"])
        return value

    def _directed_rows(self, parameters: object) -> tuple[list[str], list[dict[str, Any]]]:
        self._request_binding_sha256 = None
        if type(parameters) is not dict or set(parameters) != {
            "card_ids", "candidate_service_keys",
        }:
            self._reject_tool("invalid_parameters", "directed source shape differs")
        keys, scope = parameters["card_ids"], parameters["candidate_service_keys"]
        for values in (keys, scope):
            if (type(values) is not list or not 1 <= len(values) <= 2
                or any(type(value) is not str for value in values)
                or len(values) != len(set(values))):
                self._reject_tool("invalid_parameters", "directed source bounds differ")
        self._publish()
        self._refresh_semantic()
        if any(not self._visible("service", value) or value not in self._by_candidate
               for value in scope) or any(
            not self._visible("card", key) or key not in self._cards for key in keys
        ):
            self._reject_tool("reference_not_available", "source prefix differs")
        rows = []
        for key in keys:
            contract = self.object_contract(key)
            if contract is None:
                self._reject_tool("reference_not_available", "source prefix differs")
            if contract["node"] != "symptoms":
                self._reject_tool("card_kind_mismatch", "directed source role differs")
            if len(contract["targets"]) != 1 or not set(contract["targets"]) <= set(scope):
                self._reject_tool("invalid_parameters", "directed source scope differs")
            rows.append(self._semantic[key])
        return scope, rows

    def execute(self, tool: str, parameters: object) -> OpsIndexedToolResult:
        if tool != "inspect_sources":
            return super().execute(tool, parameters)
        scope, rows = self._directed_rows(parameters)
        parameters = cast(dict[str, Any], parameters)
        pin = directed_source_binding(self.case_snapshot_id, parameters, rows)
        self._route_request_bound(pin)
        encoded = [compact_record(row) for row in rows]
        view: dict[str, Any] = dict(
            schema_version=self.view_schema, investigation_profile_id=self.profile_id,
            semantic_encoding=SEMANTIC_ENCODING, case_snapshot_id=self.case_snapshot_id,
            tool=tool, tool_version="sre-directed-source/1", parameters=parameters,
            scope=scope, status="completed", source_role="aggregation_not_independent_evidence",
            role="supplementary", score_authority=False, route_binding_sha256=pin,
            source_relation=directed_source_relation(rows, scope),
            page=dict(records=encoded, count=len(rows), total=len(rows),
                      unit="complete_record", has_more=False, next_cursor=None),
            soft_target_basis="encoded_page_records_only",
            page_body_bytes=len(canonical_json_bytes(encoded)),
            limitations=["aggregation_not_independent_support",
                         "summary_cannot_prove_episode_or_normality",
                         "existing_sources_only_no_implicit_measurement"],
        )
        def make_receipt(source_ids: tuple[str, ...]) -> OpsNodeEvidence:
            return node_card(
                snapshot=self.case_snapshot_id, node="boundary", parameters=dict(parameters),
                payload=dict(tool=tool, record_ids=source_ids, score_authority=False),
                evidence_ids=source_ids or (self._data.catalog.catalog_sha256,),
                source_refs=(self._data.source_descriptor_sha256,),
                interval=self._data.diagnostics.legacy.sources.adapter.time_contract.fault_window,
                availability="partial", limitations=tuple(view["limitations"]),
            ).model_copy(update={"tool_version": "sre-directed-source/1"})

        receipt = make_receipt(tuple(row["card_id"] for row in rows))
        serialized = self._serialize_view(dict(view, card_id=card_id(receipt)))
        if json.loads(serialized).get("status") == "resource_limit":
            receipt = make_receipt(())
            fallback = json.loads(serialized)
            fallback["card_id"] = card_id(receipt)
            serialized = self._serialize_view(fallback)
        return OpsIndexedToolResult(evidence=receipt, origin="read_card",
                                    model_view_json=serialized)


class OpsSreDirectedLedgerState(OpsSreActionLedgerState):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/19"] = (
        "nurse.v2.sre-investigation-ledger/19"  # type: ignore[assignment]
    )
    investigation_profile_id: Literal["hermes-sre-tools/18"] = DIRECTED_PROFILE  # type: ignore[assignment]


class OpsSreDirectedInvestigationLedger(OpsSreActionInvestigationLedger):
    state_type: type[OpsLedgerStateBase] = OpsSreDirectedLedgerState
    toolbox_type = OpsSreDirectedToolbox
    skill_tools = DIRECTED_SKILL_TOOLS

    def _finished_record(
        self, record: OpsActionRecord, outcome: OpsToolResult | None,
    ) -> OpsActionRecord:
        from nurse_rcl.v2.live.ops_sre_directed_recovery import directed_rejection_metadata

        finished = super()._finished_record(record, outcome)
        assert isinstance(self._indexed, OpsSreDirectedToolbox)
        try:
            return directed_rejection_metadata(self._indexed, finished)
        except Exception as exc:
            self._abort(CheckpointReceiptError(exc))


class OpsSreDirectedPublicationReader(OpsSreActionPublicationReader):
    toolbox_type = OpsSreDirectedToolbox
