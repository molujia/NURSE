"""Focused metadata navigation over original cards, reader and debit guards."""

from __future__ import annotations

import json
from typing import Any, Literal

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_indexed_tools import OpsIndexedToolResult
from nurse_rcl.v2.live.ops_node_common import node_card
from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
from nurse_rcl.v2.live.ops_sre_atomic_selector_tools import (
    ATOMIC_SKILLS,
    OpsSreAtomicSelectorInvestigationLedger,
    OpsSreAtomicSelectorLedgerState,
    OpsSreAtomicSelectorPublicationReader,
    OpsSreAtomicSelectorToolbox,
)
from nurse_rcl.v2.live.ops_sre_encoding import compact_record, expand
from nurse_rcl.v2.live.ops_sre_focused_catalog import EvidenceCatalog
from nurse_rcl.v2.live.ops_sre_focused_decision import (
    FOCUSED_TOOLS,
    resolve_focused_selection,
    selection_shape,
    validate_focused_entries,
)
from nurse_rcl.v2.live.ops_tools import card_id

FOCUSED_SKILLS = {k.removesuffix("/19") + "/20": v for k, v in ATOMIC_SKILLS.items()}


class OpsSreFocusedToolbox(OpsSreAtomicSelectorToolbox):
    profile_id = FOCUSED_TOOLS
    skill_tools = FOCUSED_SKILLS
    view_schema = "nurse.v2.sre-tool-view/18"

    def _enrich(self, row: dict[str, Any]) -> dict[str, Any]:
        from nurse_rcl.v2.live.ops_sre_focused_sources import enrich_semantic

        return enrich_semantic(
            row,
            requests=self._data.catalog.requests,
            observations=self._data.diagnostics.errors,
            sources=self._data.sources,
        )

    def _refresh_semantic(self) -> None:
        super()._refresh_semantic()
        receipts = {
            key
            for key in self._semantic
            if self._cards[key].tool_version in {"sre-directed-source/1", "sre-directed-source/2"}
        }
        for key in receipts:
            self._semantic.pop(key)
        for ids in self._by_candidate.values():
            ids.difference_update(receipts)
        for key, row in list(self._semantic.items()):
            if "source_attribution" not in row:
                self._semantic[key] = self._enrich(row)

    def _inspection_record(self, row: dict[str, Any]) -> dict[str, Any]:
        # A frozen prefix supplies its own publication role and targets. Rebind
        # to that exact row rather than retain a later cache's attribution hash.
        source = self._enrich({k: v for k, v in row.items() if k != "source_attribution"})
        return super()._inspection_record(source)

    def _catalog(self) -> EvidenceCatalog:
        self._refresh_semantic()
        requests = {k: v for k, v in self._requests.items() if v.candidate_service_key}
        fresh = EvidenceCatalog(
            snapshot=self.case_snapshot_id,
            rows=list(self._semantic.values()),
            requests=requests,
            candidates=sorted(self._by_candidate),
        )
        previous = getattr(self, "_evidence_catalog", None)
        if previous is not None:
            state = previous.export_state()
            # Publication can append cards, but an already issued page prefix
            # remains bound to its original immutable descriptors.
            fresh._prefixes, fresh._cursors = state["prefixes"], state["cursors"]
        self._evidence_catalog = fresh
        return fresh

    def capabilities(self, records: tuple[Any, ...] = ()) -> dict[str, Any]:
        value = super().capabilities(records)
        value["parameter_contracts"]["browse_index"] = dict(
            kind=["candidates", "signals", "nodes", "onsets", "evidence"],
            evidence_filters=dict(
                candidate_service_keys="one or two distinct currently issued candidates",
                optional_exact=["metric_name", "signal_family", "record_kind"],
            ),
            cursor="copy issued cursor with identical filters; every call costs one action",
        )
        value["evidence_catalog_targets"] = []
        for record in reversed(records):
            if record.status != "completed" or record.model_view_json is None:
                continue
            view = expand(json.loads(record.model_view_json))
            if view.get("page", {}).get("kind") == "evidence":
                value["evidence_catalog_targets"] = view["page"]["items"]
                break
        return value

    def _read_view(self, key: str, section: str, cursor: object) -> dict[str, Any]:
        value = expand(super()._read_view(key, section, cursor))
        if "semantic_record" in value:
            value["semantic_record"] = compact_record(self._enrich(value["semantic_record"]))
        return value

    def _execute(self, tool: str, parameters: object) -> OpsIndexedToolResult:
        result = super()._execute(tool, parameters)
        view = expand(json.loads(result.model_view_json))
        if "semantic_record" in view:
            view["semantic_record"] = compact_record(self._enrich(view["semantic_record"]))
            result = result.model_copy(update={"model_view_json": self._serialize_view(view)})
        return result

    def execute(self, tool: str, parameters: object) -> OpsIndexedToolResult:
        if (
            tool == "browse_index"
            and type(parameters) is dict
            and parameters.get("kind") == "evidence"
        ):
            if set(parameters) != {"kind", "filters", "cursor"}:
                self._reject_tool("invalid_parameters", "evidence catalog shape differs")
            filters = parameters.get("filters")
            if type(filters) is not dict:
                self._reject_tool("invalid_parameters", "evidence filters require an object")
            scope = filters.get("candidate_service_keys", [])
            if type(scope) is not list or any(not self._visible("service", c) for c in scope):
                self._reject_tool("reference_not_available", "evidence scope prefix differs")
            from nurse_rcl.v2.live.ops_sre_routes import route_binding

            self._publish()
            try:
                pin = route_binding(self, tool, parameters)
                page = self._catalog().browse(
                    filters=parameters["filters"], cursor=parameters["cursor"]
                )
            except ValueError:
                self._reject_tool(
                    "request_binding_mismatch" if parameters["cursor"] else "invalid_parameters",
                    "evidence catalog binding differs",
                )
            self._route_request_bound(pin)
            receipt = node_card(
                snapshot=self.case_snapshot_id,
                node="boundary",
                parameters=parameters,
                payload=dict(
                    tool=tool,
                    kind="evidence",
                    publication_prefix=page["publication_prefix"],
                    score_authority=False,
                ),
                evidence_ids=(self._data.catalog.catalog_sha256,),
                source_refs=(self._data.source_descriptor_sha256,),
                interval=self._data.diagnostics.legacy.sources.adapter.time_contract.fault_window,
                availability="partial",
                limitations=("catalog_not_health_evidence",),
            ).model_copy(update={"tool_version": "ops-index-navigation/2"})
            view = dict(
                schema_version=self.view_schema,
                investigation_profile_id=self.profile_id,
                case_snapshot_id=self.case_snapshot_id,
                tool=tool,
                status="completed",
                card_id=card_id(receipt),
                page=page,
                scope=scope,
                route_binding_sha256=pin,
                role="supplementary",
                score_authority=False,
                limitations=["directory_not_delivered_evidence"],
            )
            return OpsIndexedToolResult(
                evidence=receipt, origin="read_card", model_view_json=self._serialize_view(view)
            )
        result = super().execute(tool, parameters)
        if tool == "inspect_sources":
            from nurse_rcl.v2.live.ops_sre_focused_sources import (
                focused_source_binding,
                focused_source_relation,
            )

            assert type(parameters) is dict
            scope, rows = self._directed_rows(parameters)
            pin = focused_source_binding(self.case_snapshot_id, parameters, rows)
            self._route_request_bound(pin)
            view = expand(json.loads(result.model_view_json))
            view["route_binding_sha256"] = pin
            if view.get("status") != "resource_limit":
                view.update(
                    route_binding_sha256=pin,
                    source_relation=focused_source_relation(rows, scope),
                    tool_version="sre-directed-source/2",
                )
            result = result.model_copy(update={"model_view_json": self._serialize_view(view)})
        return result

    def export_navigation(
        self, *, exclude_card_ids: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        from nurse_rcl.v2.live.ops_sre_focused_sources import source_registry

        return dict(
            super().export_navigation(exclude_card_ids=exclude_card_ids),
            evidence_catalog=self._catalog().export_state(),
            source_registry=source_registry(self._data),
        )

    def restore_navigation(self, state: dict[str, Any]) -> None:
        from nurse_rcl.v2.live.ops_sre_focused_sources import (
            source_registry,
            validate_source_registry,
        )

        if "evidence_catalog" not in state or "source_registry" not in state:
            raise ValueError("focused catalog checkpoint is absent")
        registry = validate_source_registry(state["source_registry"])
        if canonical_json_bytes(registry) != canonical_json_bytes(source_registry(self._data)):
            raise ValueError("focused registered source checkpoint differs")
        catalog = state["evidence_catalog"]
        # Every legacy layer validates its exact old field set after unwrapping.
        super().restore_navigation(
            {k: v for k, v in state.items() if k not in {"evidence_catalog", "source_registry"}}
        )
        self._catalog().restore_state(catalog)


class OpsSreFocusedLedgerState(OpsSreAtomicSelectorLedgerState):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/25"] = (
        "nurse.v2.sre-investigation-ledger/25"  # type: ignore[assignment]
    )
    investigation_profile_id: Literal["hermes-sre-tools/25"] = FOCUSED_TOOLS  # type: ignore[assignment]


class OpsSreFocusedInvestigationLedger(OpsSreAtomicSelectorInvestigationLedger):
    _indexed: OpsSreFocusedToolbox
    state_type = OpsSreFocusedLedgerState  # type: ignore[assignment]
    toolbox_type = OpsSreFocusedToolbox
    skill_tools = FOCUSED_SKILLS
    validate_entries = staticmethod(validate_focused_entries)

    def _current(self, card: str) -> dict[str, Any] | None:
        row = super()._current(card)
        return None if row is None else self._indexed._enrich(row)

    def _restore_tool_state(self, state: Any) -> None:
        super()._restore_tool_state(state)
        from nurse_rcl.v2.live.ops_sre_context import _semantic_rows
        from nurse_rcl.v2.live.ops_sre_views import semantic_record

        for saved in state.records:
            if saved.model_view_json is None or saved.status != "completed":
                continue
            for row in _semantic_rows(expand(json.loads(saved.model_view_json))):
                expected = self._indexed._enrich(
                    {k: v for k, v in row.items() if k != "source_attribution"}
                )
                if canonical_json_bytes(expected) != canonical_json_bytes(row):
                    raise ValueError("focused restored source attribution differs")
                if "source_attribution" not in expected:
                    continue
                card = self._indexed._cards.get(row["card_id"])
                if card is None:
                    raise ValueError("focused restored source card is not admitted")
                original = semantic_record(
                    card,
                    published=row["published_by_core"],
                    origin=self._indexed._origins.get(row["card_id"]),
                )
                # Frozen publication metadata may be older; original card bytes
                # and all delivered observational fields still have to match.
                if any(
                    canonical_json_bytes(row.get(field)) != canonical_json_bytes(original[field])
                    for field in (
                        "node",
                        "tool_version",
                        "parameters",
                        "payload",
                        "unit",
                        "window_ns",
                        "coverage",
                        "availability",
                        "truth_value",
                        "evidence_ids",
                        "source_refs",
                        "limitations",
                    )
                ):
                    raise ValueError("focused restored source card contents differ")

    @staticmethod
    def _resolve(selection: Any, frozen: str) -> dict[str, Any]:
        context = json.loads(frozen)
        refs = context["references"]
        book = SreActionReferenceBook.from_state(
            refs, snapshot=refs["snapshot"], run_id=refs["run_id"]
        )
        return resolve_focused_selection(
            selection, table=context["table"], book=book, request_index=context["request_index"]
        )

    def _admit_proposal(self, proposals: object, raw: bytes, record: Any) -> Any:
        if (
            len(raw) > self.proposal_bytes
            or type(proposals) is not list
            or len(proposals) != 1
            or not hasattr(self, "_frozen_entry_context_json")
        ):
            raise ValueError("one frozen focused selection required")
        selection_shape(proposals[0])
        return self._admit_frozen_selection(
            proposals, record, selected=proposals[0] if type(proposals[0]) is str else None
        )


class OpsSreFocusedPublicationReader(OpsSreAtomicSelectorPublicationReader):
    toolbox_type = OpsSreFocusedToolbox
