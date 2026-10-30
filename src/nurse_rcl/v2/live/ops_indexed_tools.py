"""Opt-in /2 tool adapter and atomic navigation/action checkpoint, not yet live.

Only the trusted investigation owner receives these objects. The model sees the
bounded capabilities/view, and each model operation goes through the ledger.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeLimits, encode_json
from nurse_rcl.v2.live.ops_card_view import (
    SECTIONS,
    VIEW_BYTES,
    bounded_view,
    card_header,
    entities,
    section_leaves,
    section_value,
    serialize_tool_view,
    summary_view,
)
from nurse_rcl.v2.live.ops_contracts import (
    OpsEmpiricalInvestigationAction,
    OpsInvestigationEvidence,
    OpsNodeEvidence,
)
from nurse_rcl.v2.live.ops_index import OpsIndex, index_parameter_contract
from nurse_rcl.v2.live.ops_investigation_ledger import (
    OpsActionRecord,
    OpsInvestigationLedger,
    OpsLedgerStateBase,
)
from nurse_rcl.v2.live.ops_node_common import node_card
from nurse_rcl.v2.live.ops_tools import (
    SKILL_TOOLS,
    TOOL_PARAMETERS,
    OpsToolbox,
    OpsToolResult,
    card_id,
)
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData
from nurse_rcl.v2.live.repair_queries import RepairQueryFeedback
from nurse_rcl.v2.live.repair_query_session import RepairQueryCompletedStep

INDEXED_TOOL_PARAMETERS = {
    **{k: v for k, v in TOOL_PARAMETERS.items() if k != "list_observations"},
    "browse_index": ("kind", "filters", "cursor"),
    "read_node": ("card_id", "section", "cursor"),
}
INDEXED_SKILL_TOOLS = {
    key.replace("/1", "/2"): tuple(
        dict.fromkeys(
            ("browse_index", *(t for t in names if t != "list_observations")),
        )
    )
    for key, names in SKILL_TOOLS.items()
}


def indexed_parameter_contracts() -> dict[str, Any]:
    """Code-owned guidance, not a second validator or a source of new tool authority."""
    return dict(
        browse_index=index_parameter_contract(),
        read_node=dict(
            card_id="exact registered card_id from published summary or nodes page",
            section=list(SECTIONS),
            cursor="null initially; copy issued next_cursor for the same card/section; "
            "summary has no continuation",
        ),
        measure_signal=dict(
            request_key="exact registered request_key with bound=true from "
            "signals page; if already measured, read its evidence card"
        ),
        compare_timing=dict(
            source_onset_id="registered measured onset_id from onsets page, source endpoint",
            target_onset_id="registered measured onset_id from onsets page, target endpoint; "
            "missing dependency or onset bounds can return unknown",
        ),
        check_mechanism=dict(
            card_id="core-published mechanisms card_id with a condition table; "
            "read_node confirms publication; no implicit measurement"
        ),
    )


class OpsIndexedToolResult(OpsToolResult):
    model_view_json: str


class OpsIndexedToolbox(OpsToolbox):
    """Separate representation profile, sharing only existing read-only calculations."""

    def __init__(
        self,
        *,
        data: RepairPreparationData,
        block_root: Path,
        soft_targets: bool = False,
    ):
        super().__init__(data=data, block_root=block_root)
        if type(soft_targets) is not bool:
            raise ValueError("input view policy must be explicit")
        self.soft_targets = soft_targets
        self._index = OpsIndex(
            case_snapshot_id=self.case_snapshot_id,
            catalog_version=data.catalog.catalog_sha256,
            rows=self._directory_rows(),
        )
        self._read_cursors: dict[str, dict[str, Any]] = {}

    def _directory_rows(self) -> dict[str, list[dict[str, Any]]]:
        signals = []
        for request in self._requests.values():
            source = self._sources[request.source_sha256]
            signals.append(
                dict(
                    request_key=request.request_key,
                    candidate_service_key=request.candidate_service_key,
                    observation_id=request.observation_id,
                    unit=source.observation.unit,
                    modality=source.observation.modality,
                    signal_family=source.observation.signal_family,
                    metric_name=source.metric_name,
                    bound=request.candidate_service_key is not None,
                    coverage=source.quality.coverage_quality,
                    formally_eligible=source.quality.formal_scoring_eligible,
                )
            )
        return dict(
            candidates=[
                dict(candidate_service_key=k) for k in self._data.catalog.candidate_service_keys
            ],
            signals=signals,
            nodes=self._node_rows(),
            onsets=self._onset_rows(),
        )

    def _node_rows(self) -> list[dict[str, Any]]:
        rows = []
        for key, card in self._cards.items():
            scoped = sorted(
                set(entities(json.loads(card.parameters_json)))
                | set(entities(json.loads(card.payload_json)))
            )
            rows.append(
                dict(
                    card_id=key,
                    node=card.node,
                    candidate_service_key=scoped[0] if len(scoped) == 1 else None,
                    availability=card.availability,
                    truth_value=card.truth_value,
                )
            )
        return rows

    def _onset_rows(self) -> list[dict[str, Any]]:
        return [
            dict(
                onset_id=o.predicate_sha256,
                candidate_service_key=p.candidate_service_key,
                bounds_ns=o.bounds_ns,
            )
            for f in self._feedback.values()
            for p, o in zip(f.predicates, f.onsets, strict=True)
        ]

    def _publish(self) -> None:
        self._index.publish(kind="nodes", rows=self._node_rows())
        self._index.publish(kind="onsets", rows=self._onset_rows())

    def capabilities(self) -> dict[str, Any]:
        """Owner must checkpoint issued entry cursors before exposing this view."""
        self._publish()
        value: dict[str, Any] = dict(
            investigation_profile_id="hermes-readonly-ops/2",
            case_snapshot_id=self.case_snapshot_id,
            catalog_version=self._data.catalog.catalog_sha256,
            counts=dict(
                candidates=len(self._data.catalog.candidate_service_keys),
                signals=len(self._requests),
                nodes=len(self._cards),
                onsets=len(self._onset_rows()),
            ),
            candidate_entry=self._index.browse_index(kind="candidates", filters={}, cursor=None),
            tools=INDEXED_TOOL_PARAMETERS,
            sections=SECTIONS,
            missing_modalities=self._data.diagnostics.legacy.sources.adapter.time_contract.missing_modalities,
            limitations=[
                "catalog_not_health_evidence",
                "known exact IDs can be used directly",
                "shared cards: browse nodes by node type, then read endpoints",
                "every navigation/read/refusal consumes one of eight actions",
            ],
        )
        if self.soft_targets:
            value["parameter_contracts"] = indexed_parameter_contracts()
            selected: dict[str, dict[str, Any]] = {}
            for key in sorted(self._core_cards):
                card = self._cards[key]
                if card.node in selected:
                    continue
                body = json.loads(card.payload_json)
                selected[card.node] = dict(
                    **card_header(card, published=True),
                    facts={
                        k: v
                        for k, v in body.items()
                        if type(v) not in (dict, list) or k in {"baseline_window", "fault_window"}
                    },
                    meaning="publication_summary_not_full_condition_or_direction_basis",
                    omitted_detail="read registered card for nested judgments and exact scope",
                )
            value.update(
                published_node_summaries=list(selected.values()),
                published_summary_omitted_count=len(self._core_cards) - len(selected),
                published_summary_selection="one per node kind by stable card ID, not ranking",
            )
        return dict(
            json.loads(
                encode_json(
                    value, ExchangeLimits(max_bytes=4_194_304 if self.soft_targets else 8192)
                )
            )
        )

    def restore_evidence(self, saved: OpsInvestigationEvidence) -> None:
        # Navigation is a directory receipt, not another clinical/system fact.
        if saved.evidence.tool_version == "ops-index-navigation/2":
            if saved.evidence.case_snapshot_id != self.case_snapshot_id:
                raise ValueError("navigation snapshot differs")
            return
        super().restore_evidence(saved)

    def execute(self, tool: str, parameters: object) -> OpsIndexedToolResult:
        fields = INDEXED_TOOL_PARAMETERS.get(tool)
        if fields is None or type(parameters) is not dict or set(parameters) != set(fields):
            self._reject_tool(
                "invalid_parameters",
                "indexed tool or parameter domain is not registered",
            )
        self._publish()
        if tool == "browse_index":
            page = self._index.browse_index(**parameters)
            card = node_card(
                snapshot=self.case_snapshot_id,
                node="boundary",
                parameters=parameters,
                payload=dict(page=page, score_authority=False),
                evidence_ids=(self._data.catalog.catalog_sha256,),
                source_refs=(self._data.source_descriptor_sha256,),
                interval=self._data.diagnostics.legacy.sources.adapter.time_contract.fault_window,
                limitations=("catalog_not_health_evidence",),
            ).model_copy(update={"tool_version": "ops-index-navigation/2"})
            view = dict(
                schema_version="nurse.v2.ops-tool-view/2",
                role="supplementary",
                origin="read_card",
                score_authority=False,
                page=page,
            )
            return OpsIndexedToolResult(
                evidence=card, origin="read_card", model_view_json=bounded_view(view, card=card)
            )
        if tool == "read_node":
            key, section, cursor = (parameters[k] for k in fields)
            if type(key) is not str or key not in self._cards:
                self._reject_tool("reference_not_available", "node or section not registered")
            if type(section) is not str or section not in SECTIONS:
                self._reject_tool("invalid_section", "node or section not registered")
            card = self._cards[key]
            view = self._read_view(key, section, cursor)
            return OpsIndexedToolResult(
                evidence=card,
                origin="read_card",
                model_view_json=bounded_view(view, card=card, soft_targets=self.soft_targets),
            )
        outcome = super().execute(tool, parameters)
        self._cards[card_id(outcome.evidence)] = outcome.evidence
        view = summary_view(
            outcome.evidence,
            published=card_id(outcome.evidence) in self._core_cards,
            soft_targets=self.soft_targets,
        )
        view.update(
            origin=outcome.origin, reused_core_evidence_ids=outcome.reused_core_evidence_ids
        )
        return OpsIndexedToolResult(
            **outcome.model_dump(),
            model_view_json=bounded_view(
                view, card=outcome.evidence, soft_targets=self.soft_targets
            ),
        )

    def _cursor(self, binding: dict[str, Any]) -> str:
        return "ops_read_" + sha256_hex(
            canonical_json_bytes(
                dict(
                    case_snapshot_id=self.case_snapshot_id,
                    catalog_version=self._data.catalog.catalog_sha256,
                    profile="hermes-readonly-ops/2",
                    binding=binding,
                )
            )
        )

    def _read_view(self, key: str, section: str, cursor: object) -> dict[str, Any]:
        card = self._cards[key]
        if section == "summary":
            if cursor is not None:
                self._reject_tool(
                    "request_binding_mismatch",
                    "summary does not issue a continuation",
                )
            return summary_view(
                card, published=key in self._core_cards, soft_targets=self.soft_targets
            )
        start = 0
        if cursor is not None:
            if type(cursor) is not str or cursor not in self._read_cursors:
                self._reject_tool("request_binding_mismatch", "read cursor was not issued")
            binding = self._read_cursors[cursor]
            if binding["card_id"] != key or binding["section"] != section:
                self._reject_tool("request_binding_mismatch", "read cursor card/section differs")
            start = binding["offset"]
        leaves = list(section_leaves(section_value(card, section)))
        header = card_header(card, published=key in self._core_cards)

        def assemble(items: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
            binding: dict[str, Any] = dict(card_id=key, section=section, offset=start + len(items))
            more = binding["offset"] < len(leaves)
            return dict(
                **header,
                section=section,
                omitted_detail=["other registered sections", "unread section pages"],
                page=dict(
                    total=len(leaves),
                    count=len(items),
                    items=items,
                    has_more=more,
                    next_cursor=self._cursor(binding) if more else None,
                ),
            ), binding

        items: list[dict[str, Any]] = []
        for path, value in leaves[start:]:
            item = dict(path=path, value=value)
            proposed, _ = assemble([*items, item])
            if len(canonical_json_bytes(proposed)) > VIEW_BYTES:
                if items:
                    break
                # An indivisible oversized scalar is not silently cut or dropped.
                item = dict(
                    path=path,
                    representation="oversize",
                    value_sha256=sha256_hex(canonical_json_bytes(value)),
                    omitted_detail="full value retained in registered card section",
                )
            items.append(item)
            if len(canonical_json_bytes(assemble(items)[0])) > VIEW_BYTES:
                break  # bounded_view will explicitly return context_budget.
        view, following = assemble(items)
        if (self.soft_targets or len(canonical_json_bytes(view)) <= VIEW_BYTES) and (
            view["page"]["next_cursor"]
        ):
            self._read_cursors[view["page"]["next_cursor"]] = following
        return view

    def export_navigation(
        self, *, exclude_card_ids: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        self._publish()
        value = dict(
            index=self._index.export_state(),
            read_cursors=self._read_cursors,
            cards={k: c.model_dump(mode="json") for k, c in self._cards.items()
                   if k not in exclude_card_ids},
            core_cards=sorted(self._core_cards),
            core_steps=[s.model_dump(mode="json") for s in self._core_steps.values()],
            feedback=[f.model_dump(mode="json") for f in self._feedback.values()],
        )
        if self.soft_targets:
            value["input_view_policy"] = "ops-soft-targets/1"
        return value

    def restore_navigation(self, state: dict[str, Any]) -> None:
        """Owner-only recovery after the enclosing ledger's identity/digest admission."""
        if state.get("input_view_policy") != ("ops-soft-targets/1" if self.soft_targets else None):
            raise ValueError("navigation input view policy differs")
        fields = {"index", "read_cursors", "cards", "core_cards", "core_steps", "feedback"}
        if self.soft_targets:
            fields.add("input_view_policy")
        if set(state) != fields:
            raise ValueError("navigation state fields differ")
        cards = {k: OpsNodeEvidence.model_validate(v) for k, v in state["cards"].items()}
        if (
            any(
                k != card_id(c) or c.case_snapshot_id != self.case_snapshot_id
                for k, c in cards.items()
            )
            or not self._cards.keys() <= cards.keys()
        ):
            raise ValueError("navigation card binding differs")
        if not set(state["core_cards"]) <= cards.keys():
            raise ValueError("navigation publication binding differs")
        self._cards = cards
        self._core_cards = set(state["core_cards"])
        for row in state["feedback"]:
            self._remember_feedback(
                RepairQueryFeedback.model_validate_json(canonical_json_bytes(row))
            )
        # Completed step receipts, not bare feedback, authorize one-way reuse.
        # Restore through the same existing small binding checks; no raw replay.
        self.admit_core(
            nodes=(),
            steps=tuple(
                RepairQueryCompletedStep.model_validate_json(canonical_json_bytes(row))
                for row in state["core_steps"]
            ),
        )
        restored = OpsIndex.restore(
            state["index"],
            case_snapshot_id=self.case_snapshot_id,
            catalog_version=self._data.catalog.catalog_sha256,
        )
        expected = OpsIndex(
            case_snapshot_id=self.case_snapshot_id,
            catalog_version=self._data.catalog.catalog_sha256,
            rows=self._directory_rows(),
        )
        if restored.export_state()["rows"] != expected.export_state()["rows"]:
            raise ValueError("navigation directory does not match registered cards/catalog")
        read_cursors = state["read_cursors"]
        for token, binding in read_cursors.items():
            if type(binding) is not dict or set(binding) != {"card_id", "section", "offset"}:
                raise ValueError("read cursor binding fields differ")
            key, section, offset = (binding[k] for k in ("card_id", "section", "offset"))
            if (
                key not in cards
                or section not in SECTIONS[1:]
                or type(offset) is not int
                or (not 0 < offset < len(list(section_leaves(section_value(cards[key], section)))))
                or token != self._cursor(binding)
            ):
                raise ValueError("read cursor content binding differs")
        self._index = restored
        self._read_cursors = json.loads(canonical_json_bytes(read_cursors))


class OpsIndexedActionRecord(OpsActionRecord):
    model_view_json: str | None = None


class OpsIndexedLedgerState(OpsLedgerStateBase):
    schema_version: Literal["nurse.v2.ops-investigation-ledger/2"] = (
        "nurse.v2.ops-investigation-ledger/2"
    )
    investigation_profile_id: Literal["hermes-readonly-ops/2"] = "hermes-readonly-ops/2"
    records: tuple[OpsIndexedActionRecord, ...] = Field(max_length=8)
    navigation: dict[str, Any]


class OpsEmpiricalActionRecord(OpsIndexedActionRecord):
    action: OpsEmpiricalInvestigationAction | None = None


class OpsEmpiricalLedgerState(OpsIndexedLedgerState):
    usage_policy_id: Literal["ops-empirical-input-cost/1"] = "ops-empirical-input-cost/1"
    records: tuple[OpsEmpiricalActionRecord, ...] = Field(max_length=8)


class OpsIndexedInvestigationLedger(OpsInvestigationLedger):
    state_type: type[OpsLedgerStateBase] = OpsIndexedLedgerState
    record_type: type[OpsActionRecord] = OpsIndexedActionRecord
    skill_tools = INDEXED_SKILL_TOOLS
    # Storage bound, not model input/output quota. It holds the full metadata
    # directory, published cards and up to eight full receipts in one checkpoint.
    state_bytes = 64 * 1024 * 1024

    def __init__(self, *, root: Path, toolbox: OpsIndexedToolbox, code_sha256: str):
        if type(toolbox) is not OpsIndexedToolbox:
            raise ValueError("indexed ledger requires the indexed toolbox")
        super().__init__(root=root, toolbox=toolbox, code_sha256=code_sha256)
        self._indexed = toolbox
        if toolbox.soft_targets:
            self.state_type = OpsEmpiricalLedgerState
            self.record_type = OpsEmpiricalActionRecord
            self.action_type = OpsEmpiricalInvestigationAction
            self.proposal_bytes = 4_194_304

    def capabilities(self) -> dict[str, Any]:
        if self._lock is None:
            raise ValueError("investigation ledger is closed")
        value = self._indexed.capabilities()
        self._save()  # Entry cursors durable before model input can be frozen/sent.
        return value

    def _state_values(self) -> dict[str, Any]:
        return dict(navigation=self._indexed.export_navigation())

    def _restore_tool_state(self, state: OpsLedgerStateBase) -> None:
        if not isinstance(state, OpsIndexedLedgerState):
            raise ValueError("indexed ledger profile differs")
        self._indexed.restore_navigation(state.navigation)
        for record in state.records:
            if (record.model_view_json is not None) != (record.result is not None):
                raise ValueError("indexed receipt/view completion differs")
            if record.model_view_json is not None:
                raw = serialize_tool_view(
                    json.loads(record.model_view_json), soft_targets=self._indexed.soft_targets
                )
                if raw != record.model_view_json:
                    raise ValueError("indexed view is not canonical")

    def _finished_record(
        self,
        record: OpsActionRecord,
        outcome: OpsToolResult | None,
    ) -> OpsActionRecord:
        assert record.result is not None
        self._indexed.restore_evidence(record.result)
        if isinstance(outcome, OpsIndexedToolResult) and record.status == "completed":
            view = json.loads(outcome.model_view_json)
        else:
            view = json.loads(
                bounded_view(
                    summary_view(
                        record.result.evidence,
                        published=False,
                        soft_targets=self._indexed.soft_targets,
                    ),
                    card=record.result.evidence,
                    soft_targets=self._indexed.soft_targets,
                )
            )
        view.update(action_index=record.index, origin=record.result.origin)
        if self._indexed.soft_targets and record.status == "rejected":
            # Only static project-owned guidance is exposed. Never echo exception
            # strings, raw parameters or a guessed replacement for a submitted ID.
            feedback: dict[str, Any] = dict(
                code="action_rejected",
                retry_consumes_action=True,
                guidance="Check the registered skill/tool combination and exact parameters "
                "against capabilities. Do not repeat an unchanged rejected action. "
                "This is contract guidance, not a diagnosis or permission expansion.",
            )
            contracts = indexed_parameter_contracts()
            if record.action is not None and record.action.tool in contracts:
                feedback["contract"] = contracts[record.action.tool]
            view["parameter_feedback"] = feedback
        raw = serialize_tool_view(
            view,
            soft_targets=self._indexed.soft_targets,
            diagnose=getattr(self._indexed, "diagnose_capacity", False),
        )
        return record.model_copy(update={"model_view_json": raw})
