"""D13 source-retaining tools: brief first, exact detail through explicit actions."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_investigation_ledger import OpsActionRecord, OpsLedgerStateBase
from nurse_rcl.v2.live.ops_sre_briefs import BRIEF_REPRESENTATION, project_evidence_brief
from nurse_rcl.v2.live.ops_sre_encoding import expand
from nurse_rcl.v2.live.ops_sre_ledger import SreLedgerFields
from nurse_rcl.v2.live.ops_sre_working_memory import working_evidence_view
from nurse_rcl.v2.live.ops_sre_working_tools import (
    WORKING_SKILL_TOOLS,
    OpsSreWorkingInvestigationLedger,
    OpsSreWorkingToolbox,
    OpsWorkingActionRecord,
    _history,
)

BRIEF_PROFILE = "hermes-sre-tools/6"
BRIEF_SKILL_TOOLS = {k.removesuffix("/2") + "/3": v for k, v in WORKING_SKILL_TOOLS.items()}


def candidate_situation(candidate: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A compact directory, not a substitute for any mechanism's conditions."""
    categories = Counter(row["node"] for row in rows)
    mechanisms = [r for r in rows if r["node"] == "mechanisms"]
    scores = [
        dict(
            card_id=r["card_id"],
            rank=r["payload"].get("rank"),
            quantized_total_score=r["payload"].get("score", {}).get("quantized_total_score"),
            source_role=r["source_role"],
            source_refs=r["source_refs"],
            window_ns=r["window_ns"],
        )
        for r in rows
        if r["node"] == "scoring" and r["published_by_core"]
    ]
    # Preserve published rank and exact score, no rescore.
    adverse: Counter[str] = Counter()
    for row in mechanisms:
        for condition in row["payload"].get("conditions", []):
            adverse["condition_" + condition["truth_value"]] += 1
            for leaf in condition.get("leaves", []):
                adverse["leaf_" + leaf["truth_value"]] += 1
    return dict(
        candidate_service_key=candidate,
        source_role="candidate_overview_metadata",
        representation="candidate_situation_not_full_judgment",
        categories=dict(sorted(categories.items())),
        mechanisms=dict(
            status="published"
            if any(r["published_by_core"] for r in mechanisms)
            else "not_published",
            states=[],
            condition_presence_counts=dict(adverse),
        ),
        scoring=dict(status="published" if scores else "not_published", records=scores),
        expansion=dict(
            tool="inspect_candidate",
            parameters=dict(candidate_service_key=candidate, focus="all", cursor=None),
        ),
        meaning="directory_order_not_diagnostic_priority; navigation_not_observation; "
        "counts_not_votes; all_mechanism_conditions_via_targeted_inspection",
    )


def brief_delivery_feedback(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify actual deliveries without promoting a brief to full observation."""
    from nurse_rcl.v2.live.ops_sre_context import _delivered_records
    from nurse_rcl.v2.live.ops_sre_overview import overview_source_ids

    deliveries: dict[str, list[int]] = {}
    seen: set[str] = set()
    feedback: dict[str, Any] = {}
    for saved in history:
        index = saved["action_index"]
        view = working_evidence_view(expand(saved["view"]))
        feedback = dict(
            action_index=index, records=[], authority="delivery_metadata_not_diagnostic_evidence"
        )
        if saved["status"] != "completed" or view.get("status") == "resource_limit":
            continue
        for row in _delivered_records(view):
            representation = row.get("representation")
            kind = (
                "brief"
                if representation == BRIEF_REPRESENTATION
                else "partial"
                if representation
                in {"overview_not_full_judgment", "delivered_section_not_full_judgment"}
                else "complete"
            )
            identity = sha256_hex(canonical_json_bytes(row))
            key = row["card_id"]
            prior = deliveries.get(identity, [])
            status = (
                "repeated_" + kind + "_record"
                if prior
                else "first_complete_expansion"
                if key in seen and kind == "complete"
                else "first_delivery"
            )
            feedback["records"].append(
                dict(
                    card_id=key,
                    record_sha256=identity,
                    delivery_kind=kind,
                    status=status,
                    prior_action_indices=list(prior),
                    source_role=row["source_role"],
                    targets=row.get("targets", []),
                    supplied_fields=sorted(row["payload"]),
                )
            )
            deliveries.setdefault(identity, []).append(index)
            seen.add(key)
        seen.update(overview_source_ids(view))
    return feedback


class OpsSreBriefToolbox(OpsSreWorkingToolbox):
    profile_id = BRIEF_PROFILE
    skill_tools = BRIEF_SKILL_TOOLS
    inspection_version = "sre-inspect/4"
    view_schema = "nurse.v2.sre-tool-view/5"
    brief_delivery = True

    def retained_brief_source(self, view: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
        """Owner-only retained source access; never a model tool or observation grant."""
        prefix = self._inspection_prefixes.get(view.get("publication_prefix", ""))
        key = row["card_id"]
        if prefix is None or key not in prefix["ids"]:
            raise ValueError("brief source prefix differs")
        source = dict(
            self._semantic[key],
            source_role=prefix["record_roles"][key],
            published_by_core=key in prefix["core_cards"],
            targets=prefix["record_targets"][key],
        )
        if canonical_json_bytes(self._inspection_record(source)) != canonical_json_bytes(row):
            raise ValueError("brief differs from retained source")
        return source

    def _inspection_record(self, row: dict[str, Any]) -> dict[str, Any]:
        return project_evidence_brief(row)

    def _candidate_overview(self, candidate: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return candidate_situation(candidate, rows)

    def capabilities(self, records: tuple[OpsActionRecord, ...] = ()) -> dict[str, Any]:
        value = super().capabilities(records)
        # Only current published cards; never reach into a terminal core object.
        value["investigation_situation"] = dict(
            authority="navigation_not_full_evidence",
            candidates=[
                candidate_situation(c, [self._semantic[k] for k in sorted(self._by_candidate[c])])
                for c in sorted(self._by_candidate)
            ],
            meaning="all_candidates_accessible; directory_order_not_diagnostic_priority",
        )
        return value


class OpsSreBriefLedgerState(SreLedgerFields):
    schema_version: Literal["nurse.v2.sre-investigation-ledger/4"] = (
        "nurse.v2.sre-investigation-ledger/4"
    )
    investigation_profile_id: Literal["hermes-sre-tools/6"] = "hermes-sre-tools/6"
    records: tuple[OpsWorkingActionRecord, ...] = Field(max_length=8)


class OpsSreBriefInvestigationLedger(OpsSreWorkingInvestigationLedger):
    state_type: type[OpsLedgerStateBase] = OpsSreBriefLedgerState
    skill_tools = BRIEF_SKILL_TOOLS
    toolbox_type = OpsSreBriefToolbox

    def _delivery_feedback(self, records: tuple[OpsActionRecord, ...]) -> dict[str, Any]:
        return brief_delivery_feedback(_history(records))

    def _restore_tool_state(self, state: OpsLedgerStateBase) -> None:
        super()._restore_tool_state(state)
        from nurse_rcl.v2.live.ops_sre_context import _semantic_rows

        assert isinstance(state, SreLedgerFields)
        for record in state.records:
            if record.model_view_json is None:
                continue
            view = expand(json.loads(record.model_view_json))
            for row in _semantic_rows(view):
                if row.get("representation") != BRIEF_REPRESENTATION:
                    continue
                assert isinstance(self._indexed, OpsSreBriefToolbox)
                self._indexed.retained_brief_source(view, row)
