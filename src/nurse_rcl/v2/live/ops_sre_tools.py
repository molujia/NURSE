"""Question-oriented SRE profile. Aggregation has no raw-query or core write API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, encode_json
from nurse_rcl.v2.live.ops_card_view import card_header
from nurse_rcl.v2.live.ops_contracts import OpsInvestigationEvidence
from nurse_rcl.v2.live.ops_indexed_tools import (
    INDEXED_TOOL_PARAMETERS,
    OpsIndexedToolbox,
    OpsIndexedToolResult,
    indexed_parameter_contracts,
)
from nurse_rcl.v2.live.ops_node_common import node_card
from nurse_rcl.v2.live.ops_sre_encoding import (
    LIMITS,
    SEMANTIC_ENCODING,
    compact_record,
    expand,
    serialize_sre_view,
)
from nurse_rcl.v2.live.ops_sre_overview import candidate_overview, overview_source_ids
from nurse_rcl.v2.live.ops_sre_views import (
    critical_overview,
    critical_record,
    prepared_summaries,
    record_order,
    semantic_record,
)
from nurse_rcl.v2.live.ops_tools import card_id
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData

if TYPE_CHECKING:
    from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord

SRE_PROFILE = "hermes-sre-tools/4"
INSPECTION_PARAMETERS = {
    "inspect_case": ("cursor",),
    "inspect_candidate": ("candidate_service_key", "cursor"),
    "inspect_dependency": ("caller_service_key", "callee_service_key", "cursor"),
    "compare_candidates": ("left_candidate_service_key", "right_candidate_service_key", "cursor"),
}
SRE_TOOL_PARAMETERS = {**INDEXED_TOOL_PARAMETERS, **INSPECTION_PARAMETERS}
SRE_SKILL_TOOLS = {
    "sre-boundary/1": ("inspect_case", "inspect_candidate", "browse_index", "read_node"),
    "sre-symptoms/1": (
        "inspect_candidate",
        "compare_candidates",
        "measure_signal",
        "browse_index",
        "read_node",
    ),
    "sre-dependency/1": (
        "inspect_dependency",
        "inspect_candidate",
        "compare_timing",
        "browse_index",
        "read_node",
    ),
    "sre-timing/1": (
        "inspect_dependency",
        "inspect_candidate",
        "compare_timing",
        "measure_signal",
        "browse_index",
        "read_node",
    ),
    "sre-mechanisms/1": (
        "inspect_candidate",
        "compare_candidates",
        "check_mechanism",
        "measure_signal",
        "browse_index",
        "read_node",
    ),
    "sre-scoring/1": ("inspect_candidate", "compare_candidates", "browse_index", "read_node"),
}


def sre_parameter_contracts(
    parameters: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    result = indexed_parameter_contracts()
    for tool, fields in (parameters or INSPECTION_PARAMETERS).items():
        result[tool] = {
            field: (
                "null for first page; exact issued next_cursor for same tool and scope"
                if field == "cursor"
                else "all, symptoms, mechanisms, or scoring; "
                "continuations must retain the same focus"
                if field == "focus"
                else "exact admitted candidate_service_key; "
                "comparison endpoints must differ; caller/callee order is significant"
            )
            for field in fields
        }
    return result


class OpsSreToolbox(OpsIndexedToolbox):
    """Only owner code holds this object; the agent receives JSON and nine tools."""

    profile_id = SRE_PROFILE
    inspection_parameters = INSPECTION_PARAMETERS
    tool_parameters = SRE_TOOL_PARAMETERS
    skill_tools = SRE_SKILL_TOOLS
    inspection_version = "sre-inspect/2"
    view_schema = "nurse.v2.sre-tool-view/3"
    focus_enabled = False
    diagnose_capacity = False
    brief_delivery = False

    def _candidate_overview(self, candidate: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return candidate_overview(candidate, rows)

    def _inspection_record(self, row: dict[str, Any]) -> dict[str, Any]:
        return row

    def _serialize_view(self, view: dict[str, Any]) -> str:
        return serialize_sre_view(view, diagnose=self.diagnose_capacity)

    def __init__(self, *, data: RepairPreparationData, block_root: Path):
        super().__init__(data=data, block_root=block_root, soft_targets=True)
        for card in prepared_summaries(data):
            self._cards[card_id(card)] = card
        self._semantic: dict[str, dict[str, Any]] = {}
        self._by_candidate: dict[str, set[str]] = {
            key: set() for key in data.catalog.candidate_service_keys
        }
        self._origins: dict[str, str] = {}
        self._inspection_prefixes: dict[str, dict[str, Any]] = {}
        self._inspection_cursors: dict[str, dict[str, Any]] = {}

    def capabilities(self, records: tuple[OpsActionRecord, ...] = ()) -> dict[str, Any]:
        value = super().capabilities()
        value.update(
            investigation_profile_id=self.profile_id,
            tools=self.tool_parameters,
            parameter_contracts=sre_parameter_contracts(self.inspection_parameters),
            skill_tools=self.skill_tools,
            semantic_encoding=SEMANTIC_ENCODING,
        )
        value["limitations"] = [
            "catalog_not_health_evidence",
            "every call including continuation/refusal costs one action",
            "inspect tools aggregate existing evidence only; "
            "use explicit measure_signal for new data",
            "all admitted candidates are accessible; first page is not complete case coverage",
        ]
        value["investigation_routes"] = self._investigation_routes(records)
        return dict(json.loads(canonical_json_bytes(value)))

    def _investigation_routes(self, records: tuple[OpsActionRecord, ...]) -> dict[str, Any]:
        from nurse_rcl.v2.live.ops_sre_routes import investigation_routes

        return investigation_routes(self, records)

    def _route_request_bound(self, pin: str) -> None:
        """Optional owner receipt hook; legacy profiles keep no failure pin."""

    def _refresh_semantic(self) -> None:
        # One lightweight association pass per newly seen immutable card. Onsets
        # carry predicate IDs, so their targets come from actual completed feedback.
        onset_targets = {
            o.predicate_sha256: p.candidate_service_key
            for f in self._feedback.values()
            for p, o in zip(f.predicates, f.onsets, strict=True)
        }
        for key, card in self._cards.items():
            published = key in self._core_cards
            if key in self._semantic and self._semantic[key]["published_by_core"] == published:
                continue
            if card.tool_version in {
                "sre-inspect/1",
                "sre-inspect/2",
                "sre-inspect/3",
                "sre-inspect/4",
                "ops-index-navigation/2",
                "ops-investigation-failure/1",
            }:
                continue
            extra = tuple(
                sorted({onset_targets[e] for e in card.evidence_ids if e in onset_targets})
            )
            row = semantic_record(
                card, published=published, extra_targets=extra, origin=self._origins.get(key)
            )
            self._semantic[key] = row
            for candidate in row["targets"]:
                if candidate in self._by_candidate:
                    self._by_candidate[candidate].add(key)

    def _scope(self, tool: str, parameters: dict[str, Any]) -> list[str]:
        scope = [
            parameters[k] for k in self.inspection_parameters[tool] if k not in {"cursor", "focus"}
        ]
        if any(type(k) is not str or k not in self._by_candidate for k in scope):
            self._reject_tool("reference_not_available", "candidate outside admitted domain")
        if tool == "compare_candidates" and scope[0] == scope[1]:
            self._reject_tool("invalid_parameters", "comparison requires distinct candidates")
        return scope

    def _focus(self, tool: str, parameters: dict[str, Any]) -> str:
        if "focus" not in self.inspection_parameters.get(tool, ()):
            return "all"
        value = parameters.get("focus")
        if type(value) is not str or value not in {"all", "symptoms", "mechanisms", "scoring"}:
            self._reject_tool("invalid_parameters", "inspection focus differs")
        return value

    def _route_parameters(self, tool: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """Default only owner-created routes, never repair a model's parameters."""
        if "focus" in self.inspection_parameters.get(tool, ()) and "focus" not in parameters:
            return dict(parameters, focus="all")
        return parameters

    def _selected(self, tool: str, scope: list[str]) -> list[str]:
        keys = (
            set().union(*(self._by_candidate[k] for k in scope)) if scope else set(self._semantic)
        )
        selected = []
        for key in keys:
            row = self._semantic[key]
            if row["node"] == "boundary":
                continue  # Boundary is compactly rendered, not a huge ID inventory.
            edge = row["payload"].get("edge")
            if (
                tool == "inspect_dependency"
                and edge is not None
                and (edge["caller_service_key"], edge["callee_service_key"]) != tuple(scope)
            ):
                continue
            selected.append(key)
        return sorted(selected, key=lambda key: record_order(self._semantic[key]))

    def _pin(self, value: object) -> str:
        return sha256_hex(
            canonical_json_bytes(
                dict(
                    profile=self.profile_id,
                    snapshot=self.case_snapshot_id,
                    catalog=self._data.catalog.catalog_sha256,
                    encoding=SEMANTIC_ENCODING,
                    value=value,
                )
            )
        )

    def _prefix(self, tool: str, parameters: dict[str, Any]) -> tuple[dict[str, Any], str, int]:
        scope = self._scope(tool, parameters)
        focus = self._focus(tool, parameters)
        cursor = parameters["cursor"]
        if cursor is not None:
            if type(cursor) is not str or cursor not in self._inspection_cursors:
                self._reject_tool("request_binding_mismatch", "inspection cursor was not issued")
            binding = self._inspection_cursors[cursor]
            prefix = self._inspection_prefixes[binding["prefix"]]
            if (
                prefix["tool"] != tool
                or prefix["scope"] != scope
                or (prefix.get("focus", "all") != focus)
            ):
                self._reject_tool(
                    "request_binding_mismatch",
                    "inspection cursor tool/scope differs",
                )
            return prefix, binding["prefix"], binding["offset"]
        prefix = self._current_prefix(tool, scope, focus=focus)
        pin = self._pin(prefix)
        self._inspection_prefixes[pin] = prefix
        return prefix, pin, 0

    def _current_prefix(
        self,
        tool: str,
        scope: list[str],
        *,
        focus: str = "all",
    ) -> dict[str, Any]:
        self._refresh_semantic()
        selected = self._selected(tool, scope)
        result = dict(
            tool=tool,
            scope=scope,
            ids=selected,
            core_cards=sorted(self._core_cards),
            completed_requests=sorted(self._feedback),
            core_requests=sorted(self._core_steps),
            record_roles={k: self._semantic[k]["source_role"] for k in selected},
            record_targets={k: self._semantic[k]["targets"] for k in selected},
        )
        if self.focus_enabled:
            result.update(
                focus=focus,
                overview_ids=selected,
                ids=[k for k in selected if focus == "all" or self._semantic[k]["node"] == focus],
            )
        return result

    def _next_checks(
        self,
        rows: list[dict[str, Any]],
        prefix: dict[str, Any],
        *,
        capacity: bool = False,
    ) -> list[dict[str, Any]]:
        checks: dict[str, dict[str, Any]] = {}

        def add(tool: str, parameters: dict[str, Any], status: str) -> None:
            item = dict(
                tool=tool,
                parameters=self._route_parameters(tool, parameters),
                status=status,
                skill_id=next(s for s, names in self.skill_tools.items() if tool in names),
            )
            checks[self._pin(item)] = item

        completed = prefix["completed_requests"] if capacity else self._feedback
        core_requests = prefix["core_requests"] if capacity else self._core_steps
        for row in rows:
            add(
                "read_node",
                dict(card_id=row["card_id"], section="payload", cursor=None),
                "available",
            )
            request = row["payload"].get("request_key") or row["parameters"].get("request_key")
            if (
                request in self._requests
                and self._requests[request].candidate_service_key
                and (request not in completed or request in core_requests)
            ):
                add(
                    "measure_signal",
                    dict(request_key=request),
                    "core_reusable" if capacity or request in core_requests else "unmeasured",
                )
            if row["node"] == "mechanisms" and row["card_id"] in prefix["core_cards"]:
                add("check_mechanism", dict(card_id=row["card_id"]), "published")
            edge = row["payload"].get("edge")
            if edge is not None:
                add(
                    "inspect_dependency",
                    dict(
                        caller_service_key=edge["caller_service_key"],
                        callee_service_key=edge["callee_service_key"],
                        cursor=None,
                    ),
                    "available",
                )
            for candidate in row["targets"]:
                add(
                    "inspect_candidate",
                    dict(candidate_service_key=candidate, cursor=None),
                    "available",
                )
        return [checks[k] for k in sorted(checks)]

    def _inspection_page(
        self,
        prefix: dict[str, Any],
        pin: str,
        offset: int,
        count: int,
        parameters: dict[str, Any],
        *,
        capacity: bool = False,
    ) -> dict[str, Any]:
        tool = prefix["tool"]
        keys, scope = prefix["ids"], prefix["scope"]
        # Publication role is frozen with the page, not upgraded by a later core publish.
        rows = [
            dict(
                self._semantic[k],
                published_by_core=k in prefix["core_cards"],
                source_role=prefix["record_roles"][k],
                targets=prefix["record_targets"][k],
            )
            for k in prefix.get("overview_ids", keys)
        ]
        overview = tool == "inspect_case"
        candidates = sorted(self._by_candidate) if overview else scope
        selected_candidates = candidates[offset : offset + count] if overview else candidates
        by_candidate: dict[str, list[dict[str, Any]]] = {c: [] for c in selected_candidates}
        for row in rows:
            for candidate in row["targets"]:
                if candidate in by_candidate:
                    by_candidate[candidate].append(row)
        summaries = [self._candidate_overview(c, by_candidate[c]) for c in selected_candidates]
        for summary in summaries:
            entry = summary["expansion"]
            entry["parameters"] = self._route_parameters(entry["tool"], entry["parameters"])
        by_key = {r["card_id"]: r for r in rows}
        page_rows = (
            []
            if overview
            else [self._inspection_record(by_key[k]) for k in keys[offset : offset + count]]
        )
        shown = summaries if overview else page_rows
        total = len(candidates) if overview else len(keys)
        following = offset + len(shown)
        more = following < total
        next_cursor = None
        if more:
            binding = dict(prefix=pin, offset=following, previous_offset=offset)
            next_cursor = "sre_cursor_" + self._pin(binding)
        contract = self._data.diagnostics.legacy.sources.adapter.time_contract
        scored_candidates = {
            c
            for r in rows
            if r["node"] == "scoring" and r["published_by_core"]
            for c in r["targets"]
        }
        sections = {
            c: {
                node: "published"
                if any(
                    r["node"] == node and c in r["targets"] and r["published_by_core"] for r in rows
                )
                else "not_published"
                for node in ("mechanisms", "scoring")
            }
            for c in selected_candidates
        }
        value = dict(
            schema_version=self.view_schema,
            investigation_profile_id=self.profile_id,
            semantic_encoding=SEMANTIC_ENCODING,
            tool=tool,
            tool_version=self.inspection_version,
            parameters=parameters,
            case_snapshot_id=self.case_snapshot_id,
            publication_prefix=pin,
            scope=scope,
            role="supplementary",
            score_authority=False,
            status="completed",
            availability="partial",
            truth_value="unknown",
            source_role="aggregation_not_independent_evidence",
            sections_state=sections,
            boundary=dict(
                baseline_window=contract.baseline_window.model_dump(mode="json"),
                fault_window=contract.fault_window.model_dump(mode="json"),
                candidate_count=len(self._by_candidate),
                missing_modalities=contract.missing_modalities,
                scoring_publication=dict(
                    scored_candidate_count=len(scored_candidates),
                    status="published"
                    if len(scored_candidates) == len(candidates)
                    else "partial"
                    if scored_candidates
                    else "not_published",
                    basis="frozen_query_not_formal_use",
                ),
            ),
            page=dict(
                records=shown,
                count=len(shown),
                total=total,
                unit="candidate_overview"
                if overview
                else ("evidence_record_or_brief" if self.brief_delivery else "complete_record"),
                has_more=more,
                next_cursor=next_cursor,
                omitted_count=total - following,
                selection="candidate_or_complete_record_body_capacity/3",
                omitted_reason=(
                    "unread_candidate_overviews_not_unknown_window"
                    if overview
                    else "unread_complete_records_not_evidence_absence"
                ),
                omitted_scope=scope or "all_admitted_candidates",
            ),
            critical_record_ids=[r["card_id"] for r in page_rows if critical_record(r)],
            critical_overview=critical_overview(
                rows,
                expanded_ids=(set() if overview or self.brief_delivery else set(keys[:following])),
                next_cursor=next_cursor,
            ),
            next_checks=self._next_checks(page_rows, prefix, capacity=capacity),
            next_checks_basis="current_execution_state_not_historical_evidence",
            limitations=[
                "aggregation_not_independent_support",
                "publication_not_formal_use",
                "summary_cannot_prove_episode_or_normality",
                "next_checks_are_options_not_a_mandatory_sequence",
            ],
        )
        value["critical_overview"]["count_basis"] = (
            "frozen_query_complete_records_not_global_investigation_unread"
            if not self.brief_delivery
            else "frozen_query_full_details_not_delivered_by_briefs"
        )
        if overview:
            value["next_checks"] = [
                dict(
                    s["expansion"],
                    status="available",
                    skill_id=next(s for s in self.skill_tools if s.startswith("sre-boundary/")),
                )
                for s in summaries
            ]
        else:
            value["candidate_overviews"] = summaries
        if self.focus_enabled:
            value["focus_coverage"] = dict(
                focus=prefix["focus"],
                not_selected_count=len(rows) - len(keys),
                delivered_targets=sorted({c for r in page_rows for c in r["targets"]}),
                not_delivered_targets=[
                    c for c in scope if not any(c in r["targets"] for r in page_rows)
                ],
                meaning="unselected_categories_are_unexpanded_not_absent_or_normal",
            )
        if tool == "inspect_dependency":
            edges = [r["payload"]["edge"] for r in rows if "edge" in r["payload"]]
            value["dependency"] = dict(
                caller_service_key=scope[0],
                callee_service_key=scope[1],
                edge_state="observed"
                if any(e["observation_state"] == "observed" for e in edges)
                else "not_observed",
                graph_coverage=self._data.graph.graph_pair.fault_snapshot.coverage.model_dump(
                    mode="json"
                ),
                callee_self_time="not_available",
                causal_confirmation="not_established",
            )
            value["limitations"] += [
                "caller_latency_not_callee_self_time",
                "call_direction_not_symptom_direction_or_causality",
            ]
        return compact_record(value)

    def _page_count(self, prefix: dict[str, Any], pin: str, offset: int) -> int:
        fields = [
            k for k in self.inspection_parameters[prefix["tool"]] if k not in {"cursor", "focus"}
        ]
        parameters = dict(zip(fields, prefix["scope"], strict=True), cursor=None)
        if "focus" in self.inspection_parameters[prefix["tool"]]:
            parameters["focus"] = prefix["focus"]
        # Size with the longest possible cursor and fixed publication-state hints.
        # Runtime next_checks may shrink/change, but cannot move a page boundary.
        parameters["cursor"] = "sre_cursor_" + "0" * 64
        available = min(16, self._page_total(prefix) - offset)
        selected = 0
        for count in range(1, available + 1):
            try:
                view = self._inspection_page(prefix, pin, offset, count, parameters, capacity=True)
                envelope = dict(
                    view,
                    card_id="ops_card_" + "0" * 64,
                    action_index=8,
                    origin="read_card",
                    target_bytes=4096,
                    target_exceeded=True,
                    expansion_reasons=["required_evidence_view"],
                )
                expand(envelope)
                encode_json(envelope, LIMITS)
                size = len(canonical_json_bytes(view["page"]["records"]))
            except ExchangeBoundaryError as exc:
                if exc.code != "json_limit":
                    raise
                break
            if selected and size > 4096:
                break
            selected = count
        # One oversized atomic record remains an explicit resource failure, never split.
        return selected or min(1, available)

    def _page_total(self, prefix: dict[str, Any]) -> int:
        return len(self._by_candidate) if prefix["tool"] == "inspect_case" else len(prefix["ids"])

    def _inspection_view(self, tool: str, parameters: dict[str, Any]) -> dict[str, Any]:
        prefix, pin, offset = self._prefix(tool, parameters)
        count = self._page_count(prefix, pin, offset)
        view = self._inspection_page(prefix, pin, offset, count, parameters)
        # The soft target measures only the encoded complete page body. Fixed
        # metadata and the actual final envelope remain subject to hard limits.
        view["page_body_bytes"] = len(canonical_json_bytes(view["page"]["records"]))
        metadata = dict(view, page={k: v for k, v in view["page"].items() if k != "records"})
        metadata.pop("page_body_bytes")
        view["fixed_metadata_bytes"] = len(canonical_json_bytes(metadata))
        view["soft_target_basis"] = "encoded_page_records_only"
        if view["page"]["has_more"]:
            self._inspection_cursors[view["page"]["next_cursor"]] = dict(
                prefix=pin, offset=offset + count, previous_offset=offset
            )
        return view

    def _read_view(self, key: str, section: str, cursor: object) -> dict[str, Any]:
        if section not in {"summary", "payload"}:
            return super()._read_view(key, section, cursor)
        if cursor is not None:
            self._reject_tool(
                "request_binding_mismatch",
                "complete semantic card has no continuation cursor",
            )
        self._refresh_semantic()
        card = self._cards[key]
        row = self._semantic.get(key) or semantic_record(card, published=key in self._core_cards)
        return dict(
            card_header(card, published=key in self._core_cards),
            semantic_record=compact_record(row),
            section=section,
            next_cursor=None,
            semantic_encoding=SEMANTIC_ENCODING,
            representation="complete_semantic_record",
        )

    def execute(self, tool: str, parameters: object) -> OpsIndexedToolResult:
        from nurse_rcl.v2.live.ops_sre_routes import route_binding

        if type(parameters) is not dict:
            self._reject_tool("invalid_parameters", "SRE parameters require an object")
        self._publish()
        try:
            pin = route_binding(self, tool, parameters)
        except (KeyError, TypeError):
            raise ValueError("SRE route uses an unregistered parameter or cursor") from None
        self._route_request_bound(pin)
        result = self._execute(tool, parameters)
        view = dict(json.loads(result.model_view_json), route_binding_sha256=pin)
        return result.model_copy(update={"model_view_json": self._serialize_view(view)})

    def _execute(self, tool: str, parameters: object) -> OpsIndexedToolResult:
        if tool not in self.inspection_parameters:
            result = super().execute(tool, parameters)
            if tool != "browse_index":
                self._origins.setdefault(card_id(result.evidence), result.origin)
            if tool in {"measure_signal", "compare_timing", "check_mechanism"}:
                card = result.evidence
                row = semantic_record(
                    card, published=card_id(card) in self._core_cards, origin=result.origin
                )
                view = dict(
                    card_header(card, published=card_id(card) in self._core_cards),
                    semantic_record=compact_record(row),
                    origin=result.origin,
                    semantic_encoding=SEMANTIC_ENCODING,
                    reused_core_evidence_ids=result.reused_core_evidence_ids,
                    representation="complete_semantic_record",
                    next_cursor=None,
                )
                result = result.model_copy(
                    update={
                        "model_view_json": self._serialize_view(view),
                    }
                )
            elif tool == "read_node":
                result = result.model_copy(
                    update={
                        "model_view_json": self._serialize_view(json.loads(result.model_view_json)),
                    }
                )
            return result
        if type(parameters) is not dict or set(parameters) != set(self.inspection_parameters[tool]):
            self._reject_tool("invalid_parameters", "SRE inspection parameter contract differs")
        view = self._inspection_view(tool, parameters)
        try:
            expanded = expand(view)
            ids = tuple(
                overview_source_ids(expanded)
                if tool == "inspect_case"
                else (r["card_id"] for r in expanded["page"]["records"])
            )
        except ExchangeBoundaryError as exc:
            if exc.code != "json_limit":
                raise
            # The serializer emits an explicit resource-limit receipt. Do not
            # misclassify a valid but oversized complete record as bad parameters.
            ids = ()
        # The receipt references existing evidence; it is not a new diagnostic fact.
        card = node_card(
            snapshot=self.case_snapshot_id,
            node="boundary",
            parameters=parameters,
            payload=dict(
                tool=tool,
                publication_prefix=view["publication_prefix"],
                record_ids=ids,
                score_authority=False,
            ),
            evidence_ids=ids or (self._data.catalog.catalog_sha256,),
            source_refs=(self._data.source_descriptor_sha256,),
            interval=self._data.diagnostics.legacy.sources.adapter.time_contract.fault_window,
            limitations=tuple(view["limitations"]),
            availability="partial",
        ).model_copy(update={"tool_version": self.inspection_version})
        view["card_id"] = card_id(card)  # Ordinary details, not expanded directory pagination.
        return OpsIndexedToolResult(
            evidence=card, origin="read_card", model_view_json=self._serialize_view(view)
        )

    def restore_evidence(self, saved: OpsInvestigationEvidence) -> None:
        if saved.evidence.case_snapshot_id != self.case_snapshot_id:
            raise ValueError("SRE receipt snapshot differs")
        if saved.evidence.tool_version in {
            "sre-inspect/1",
            "sre-inspect/2",
            "sre-inspect/3",
            "ops-index-navigation/2",
        }:
            return  # No self-recursive aggregation or duplicate independent support.
        self._origins.setdefault(card_id(saved.evidence), saved.origin)
        super().restore_evidence(saved)

    def _cursor(self, binding: dict[str, Any]) -> str:
        return "sre_read_" + self._pin(binding)

    def export_navigation(
        self, *, exclude_card_ids: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        return dict(
            profile=self.profile_id,
            semantic_encoding=SEMANTIC_ENCODING,
            indexed=super().export_navigation(exclude_card_ids=exclude_card_ids),
            origins=self._origins,
            inspection_prefixes=self._inspection_prefixes,
            inspection_cursors=self._inspection_cursors,
        )

    def _restored_prefix_extra_fields(self, prefix: dict[str, Any]) -> set[str]:
        return set()

    def restore_navigation(self, state: dict[str, Any]) -> None:
        if set(state) != {
            "profile",
            "semantic_encoding",
            "indexed",
            "origins",
            "inspection_prefixes",
            "inspection_cursors",
        } or (
            state["profile"] != self.profile_id or state["semantic_encoding"] != SEMANTIC_ENCODING
        ):
            raise ValueError("SRE navigation profile differs")
        origins = state["origins"]
        if type(origins) is not dict or any(
            type(k) is not str or v not in {"measured", "core_reuse", "read_card", "rejected"}
            for k, v in origins.items()
        ):
            raise ValueError("SRE navigation origin differs")
        self._origins = dict(origins)
        super().restore_navigation(state["indexed"])
        if not origins.keys() <= self._cards.keys():
            raise ValueError("SRE navigation origin card differs")
        self._semantic.clear()
        for entries in self._by_candidate.values():
            entries.clear()
        self._refresh_semantic()
        prefixes, cursors = state["inspection_prefixes"], state["inspection_cursors"]
        if type(prefixes) is not dict or type(cursors) is not dict:
            raise ValueError("SRE navigation registries differ")
        for pin, prefix in prefixes.items():
            fields_expected = {
                "tool",
                "scope",
                "ids",
                "core_cards",
                "completed_requests",
                "core_requests",
                "record_roles",
                "record_targets",
            }
            if self.focus_enabled:
                fields_expected.update({"focus", "overview_ids"})
            if type(prefix) is not dict:
                raise ValueError("SRE inspection prefix differs")
            fields_expected.update(self._restored_prefix_extra_fields(prefix))
            if (
                set(prefix) != fields_expected
                or (pin != self._pin(prefix) or prefix["tool"] not in self.inspection_parameters)
            ):
                raise ValueError("SRE inspection prefix differs")
            fields = [
                k
                for k in self.inspection_parameters[prefix["tool"]]
                if k not in {"cursor", "focus"}
            ]
            if type(prefix["scope"]) is not list or len(prefix["scope"]) != len(fields):
                raise ValueError("SRE prefix scope differs")
            self._scope(prefix["tool"], dict(zip(fields, prefix["scope"], strict=True)))
            ids = prefix.get("overview_ids", prefix["ids"])
            if (
                type(ids) is not list
                or any(type(k) is not str for k in ids)
                or (
                    len(ids) != len(set(ids))
                    or not set(ids) <= self._semantic.keys()
                    or not set(prefix["core_cards"]) <= self._core_cards
                    or not set(prefix["completed_requests"]) <= self._feedback.keys()
                    or not set(prefix["core_requests"]) <= self._core_steps.keys()
                    or set(prefix["record_roles"]) != set(ids)
                    or set(prefix["record_targets"]) != set(ids)
                )
            ):
                raise ValueError("SRE prefix references differ")
            if self.focus_enabled:
                focus = self._focus(prefix["tool"], dict(focus=prefix["focus"]))
                if focus != prefix["focus"] or prefix["ids"] != [
                    k for k in ids if focus == "all" or self._semantic[k]["node"] == focus
                ]:
                    raise ValueError("SRE prefix focus selection differs")
            for key in ids:
                expected = semantic_record(
                    self._cards[key],
                    published=key in prefix["core_cards"],
                    origin=self._origins.get(key),
                )
                if prefix["record_roles"][key] != expected["source_role"] or not (
                    set(prefix["record_targets"][key]) <= self._by_candidate.keys()
                ):
                    raise ValueError("SRE prefix role/scope differs")
        for token, binding in cursors.items():
            if (
                type(binding) is not dict
                or set(binding)
                != {
                    "prefix",
                    "offset",
                    "previous_offset",
                }
                or (
                    binding["prefix"] not in prefixes
                    or type(binding["offset"]) is not int
                    or not 0 < binding["offset"] < self._page_total(prefixes[binding["prefix"]])
                    or type(binding["previous_offset"]) is not int
                    or not 0 <= binding["previous_offset"] < binding["offset"]
                    or token != "sre_cursor_" + self._pin(binding)
                )
            ):
                raise ValueError("SRE inspection cursor differs")
            start = binding["previous_offset"]
            prefix = prefixes[binding["prefix"]]
            if start and not any(
                b["prefix"] == binding["prefix"] and b["offset"] == start for b in cursors.values()
            ):
                raise ValueError("SRE cursor parent boundary missing")
            if start + self._page_count(prefix, binding["prefix"], start) != binding["offset"]:
                raise ValueError("SRE cursor capacity boundary differs")
        self._inspection_prefixes = json.loads(canonical_json_bytes(prefixes))
        self._inspection_cursors = json.loads(canonical_json_bytes(cursors))
