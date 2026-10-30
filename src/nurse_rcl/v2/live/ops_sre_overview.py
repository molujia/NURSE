"""Candidate-level navigation over frozen published records, never new judgments."""

from __future__ import annotations

from collections import Counter
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_sre_views import overview_record


def _entry(row: dict[str, Any]) -> dict[str, Any]:
    return dict(card_id=row["card_id"], targets=row["targets"],
                source_role=row["source_role"], source_refs=row["source_refs"],
                window_ns=row["window_ns"],
                expansion=dict(tool="read_node", parameters=dict(
                    card_id=row["card_id"], section="payload", cursor=None)))


def candidate_overview(candidate: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts denote record occurrences; a representative ID is only a read route.

    The caller supplies the frozen candidate association. Sources with different
    judgments remain separate. Even a true top-level decision retains false,
    unknown and refuting children by structural level, without recalculating OR/AND.
    """
    counts: dict[str, Counter[str]] = {}
    groups: dict[bytes, dict[str, Any]] = {}
    scores = []
    for row in sorted(rows, key=lambda r: r["card_id"]):
        role, body = row["source_role"], row["payload"]
        counter = counts.setdefault(role, Counter())
        counter["record_count"] += 1
        counter["prepared_summary_count"] += int(role == "prepared_summary")
        counter["measured_record_count"] += int(body.get("kind") in {
            "paid_episode", "supplementary_episode"})
        if row["node"] == "mechanisms":
            decision = body.get("decision", {})
            state = dict(mechanism_key=decision.get("mechanism_key",
                                                   row["parameters"].get("mechanism_key")),
                         disposition=decision.get("disposition"),
                         truth_value=row["truth_value"], source_role=role)
            identity = canonical_json_bytes(state)
            group = groups.setdefault(identity, dict(**_entry(row), **{
                k: v for k, v in state.items() if k != "source_role"},
                record_count=0, additional_source_count=0,
                judgment_status_counts={}, judgment_status_by_group={}))
            group["record_count"] += 1
            group["additional_source_count"] = group["record_count"] - 1
            view = overview_record(row)
            for name, value in view["judgment_status_counts"].items():
                group["judgment_status_counts"][name] = (
                    group["judgment_status_counts"].get(name, 0) + value)
            for path, values in view["judgment_status_by_group"].items():
                group_counts = group["judgment_status_by_group"].setdefault(path, {})
                for name, value in values.items():
                    group_counts[name] = group_counts.get(name, 0) + value
        if row["node"] == "scoring" and row["published_by_core"]:
            # Exact original values; no replacement score, difference or sorting by score.
            scores.append(dict(**_entry(row), rank=body.get("rank")))
    return dict(candidate_service_key=candidate,
        source_role="candidate_overview_metadata",
        representation="candidate_overview_not_full_judgment",
        record_counts=[dict(source_role=role, **counts[role]) for role in sorted(counts)],
        mechanisms=dict(status="published" if any(r["node"] == "mechanisms" and
            r["published_by_core"] for r in rows) else "not_published",
            states=list(groups.values())),
        scoring=dict(status="published" if scores else "not_published", records=scores),
        expansion=dict(tool="inspect_candidate",
                       parameters=dict(candidate_service_key=candidate, cursor=None)),
        meaning="counts_not_votes; representative_source_only; all_sources_via_candidate; "
                "navigation_not_full_evidence_delivery")


def overview_source_ids(view: dict[str, Any]) -> list[str]:
    """Original readable IDs, never the candidate metadata identity."""
    summaries = (view.get("page", {}).get("records", [])
                 if view.get("page", {}).get("unit") == "candidate_overview"
                 else view.get("candidate_overviews", []))
    return sorted({r["card_id"] for summary in summaries
                   for r in summary["mechanisms"]["states"] + summary["scoring"]["records"]})
