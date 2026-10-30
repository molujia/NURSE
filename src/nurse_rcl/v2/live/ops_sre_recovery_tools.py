"""D15 opt-in diagnostics over unchanged deterministic tools and action debits."""

from __future__ import annotations

import json
from typing import Any, Literal, NoReturn

from pydantic import Field

from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.live.ops_card_view import SECTIONS
from nurse_rcl.v2.live.ops_failure import OpsFailureDetail, failure_detail
from nurse_rcl.v2.live.ops_indexed_tools import OpsIndexedToolResult
from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord, OpsLedgerStateBase
from nurse_rcl.v2.live.ops_sre_condition_tools import (
    CONDITION_SKILL_TOOLS,
    OpsSreConditionInvestigationLedger,
    OpsSreConditionToolbox,
)
from nurse_rcl.v2.live.ops_sre_encoding import expand, serialize_sre_view
from nurse_rcl.v2.live.ops_sre_ledger import SreLedgerFields
from nurse_rcl.v2.live.ops_sre_model_view import _prefix
from nurse_rcl.v2.live.ops_sre_references import FIELD_KINDS, SreReferenceBook
from nurse_rcl.v2.live.ops_sre_views import semantic_record
from nurse_rcl.v2.live.ops_sre_working_tools import OpsWorkingActionRecord
from nurse_rcl.v2.live.ops_tool_rejection import OpsToolDiagnostic, OpsToolRejection
from nurse_rcl.v2.live.ops_tools import OpsToolResult

RECOVERY_PROFILE: Literal["hermes-sre-tools/8"] = "hermes-sre-tools/8"
RECOVERY_SKILL_TOOLS = {k.removesuffix("/4") + "/5": v for k, v in CONDITION_SKILL_TOOLS.items()}


class OpsSreRecoveryToolbox(OpsSreConditionToolbox):
    profile_id: str = RECOVERY_PROFILE
    skill_tools = RECOVERY_SKILL_TOOLS
    view_schema = "nurse.v2.sre-tool-view/7"
    _request_binding_sha256: str | None = None

    def _reject_tool(self, reason: str, message: str) -> NoReturn:
        # Never copy the legacy message or model parameters into the diagnostic.
        error = OpsToolRejection(reason)
        error.request_binding_sha256 = getattr(self, "_request_binding_sha256", None)
        raise error

    def _route_request_bound(self, pin: str) -> None:
        self._request_binding_sha256 = pin

    def _investigation_routes(self, records: tuple[OpsActionRecord, ...]) -> dict[str, Any]:
        from nurse_rcl.v2.live.ops_sre_recovery_context import recovery_routes

        return recovery_routes(self, records)

    def bind_visible_references(self, *, book: SreReferenceBook, request_index: int) -> None:
        """Trusted owner binds the verified request prefix, never a model tool."""
        if book.snapshot != self.case_snapshot_id:
            raise ValueError("recovery visibility snapshot differs")
        _, prefix = _prefix(book, request_index)
        visible: dict[str, set[str]] = {}
        for kind, _, original, _ in prefix.to_state()["entries"]:
            visible.setdefault(kind, set()).add(original)
        self._visible_resources = visible

    def _visible(self, kind: str, value: object) -> bool:
        return type(value) is str and value in getattr(self, "_visible_resources", {}).get(kind, ())

    def object_contract(self, key: object) -> dict[str, Any] | None:
        """Only previously visible identity metadata, never the card's facts."""
        if type(key) is not str or not self._visible("card", key) or key not in self._cards:
            return None
        card = self._cards[key]
        row = semantic_record(
            card, published=key in self._core_cards, origin=self._origins.get(key)
        )
        if any(not self._visible("service", target) for target in row["targets"]):
            return None  # Do not disclose hidden endpoints or silently drop one.
        allowed = ["read_node"]
        if key in self._core_cards and card.node == "mechanisms" and "decision" in row["payload"]:
            allowed.append("check_mechanism")
        return dict(
            card_id=key,
            node=card.node,
            targets=row["targets"],
            source_role=row["source_role"],
            allowed_tools=allowed,
            sections=list(SECTIONS),
            authority="navigation_not_evidence",
        )

    def execute(self, tool: str, parameters: object) -> OpsIndexedToolResult:
        self._request_binding_sha256 = None
        if type(parameters) is dict:
            # The ordinary request parser already enforces this prefix. Keep the
            # trusted tool boundary conservative when called directly or restored.
            for field in self.tool_parameters.get(tool, ()):
                kind = FIELD_KINDS.get(field)
                if (
                    kind is not None
                    and kind in {"card", "service", "request", "onset"}
                    and field in parameters
                    and not self._visible(kind, parameters[field])
                ):
                    self._reject_tool("reference_not_available", "reference unavailable")
        result = super().execute(tool, parameters)
        if tool in {"read_node", "check_mechanism"} and type(parameters) is dict:
            contract = self.object_contract(parameters.get("card_id"))
            if contract is not None:
                view = json.loads(result.model_view_json)
                if view.get("status") != "resource_limit":
                    view["object_contract"] = contract
                    result = result.model_copy(
                        update={
                            "model_view_json": self._serialize_view(view),
                        }
                    )
        return result


class OpsRecoveryActionRecord(OpsWorkingActionRecord):
    rejection: OpsToolDiagnostic | None = None
    request_binding_sha256: SHA256 | None = None


class OpsSreRecoveryLedgerState(SreLedgerFields):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/6"] = (
        "nurse.v2.sre-investigation-ledger/6"
    )
    investigation_profile_id: Literal["hermes-sre-tools/8"] = RECOVERY_PROFILE
    records: tuple[OpsRecoveryActionRecord, ...] = Field(max_length=8)


class OpsSreRecoveryInvestigationLedger(OpsSreConditionInvestigationLedger):
    state_type: type[OpsLedgerStateBase] = OpsSreRecoveryLedgerState
    skill_tools = RECOVERY_SKILL_TOOLS
    toolbox_type = OpsSreRecoveryToolbox
    record_type: type[OpsActionRecord] = OpsRecoveryActionRecord

    def _failed_record(self, record: OpsActionRecord, exc: Exception) -> OpsActionRecord:
        record = super()._failed_record(record, exc)
        if isinstance(exc, OpsToolRejection):
            return record.model_copy(
                update={
                    "rejection": exc.diagnostic,
                    "request_binding_sha256": exc.request_binding_sha256,
                }
            )
        return record

    def _finished_record(
        self,
        record: OpsActionRecord,
        outcome: OpsToolResult | None,
    ) -> OpsActionRecord:
        finished = super()._finished_record(record, outcome)
        assert isinstance(finished, OpsRecoveryActionRecord)
        if finished.status not in {"rejected", "failed"} or finished.failure_detail is not None:
            return finished
        assert finished.model_view_json is not None
        view = expand(json.loads(finished.model_view_json))
        if view.get("status") == "resource_limit":
            return finished
        diagnostic = finished.rejection or OpsToolDiagnostic(
            reason_code="unclassified_failure",
            retry_hint="unknown",
        )
        feedback = view.setdefault("parameter_feedback", {})
        feedback["diagnostic"] = diagnostic.model_dump(mode="json")
        if finished.request_binding_sha256 is not None:
            view["route_binding_sha256"] = finished.request_binding_sha256
        # A resource/ref failure never discloses the rejected resource's type.
        if finished.action is not None and diagnostic.reason_code in {
            "card_kind_mismatch",
            "invalid_section",
            "request_binding_mismatch",
        }:
            params = json.loads(finished.action.parameters_json)
            assert isinstance(self._indexed, OpsSreRecoveryToolbox)
            contract = self._indexed.object_contract(params.get("card_id"))
            if contract is not None:
                feedback["object_contract"] = contract
        raw = serialize_sre_view(view, diagnose=True)
        projected = json.loads(raw)
        updates: dict[str, Any] = {"model_view_json": raw}
        if projected.get("status") == "resource_limit":
            updates["failure_detail"] = (
                OpsFailureDetail.model_validate(projected["failure_detail"])
                if "failure_detail" in projected
                else failure_detail("investigation_tool", ValueError("view_resource_limit"))
            )
        return finished.model_copy(update=updates)

    def _restore_tool_state(self, state: OpsLedgerStateBase) -> None:
        super()._restore_tool_state(state)
        assert isinstance(state, OpsSreRecoveryLedgerState)
        for row in state.records:
            if row.rejection is None or row.model_view_json is None:
                continue
            view = expand(json.loads(row.model_view_json))
            if view.get("status") != "resource_limit" and view.get("parameter_feedback", {}).get(
                "diagnostic"
            ) != row.rejection.model_dump(mode="json"):
                raise ValueError("saved rejection diagnostic differs")
            if view.get("status") != "resource_limit" and (
                view.get("route_binding_sha256") != row.request_binding_sha256
            ):
                raise ValueError("saved rejection binding differs")
