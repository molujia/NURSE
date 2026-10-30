"""D12 source-bound working synopses; the original receipts remain authoritative.

This is explicit lossy *input selection*, not a lossless encoding of old leaves.
No new observations, mechanism evaluations, source reads or model calls occur.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_sre_encoding import expand
from nurse_rcl.v2.live.repair_mechanisms import (
    RepairMechanismCondition,
    RepairMechanismDecision,
)

_PARTIAL = {"overview_not_full_judgment", "delivered_section_not_full_judgment"}


def working_evidence_view(view: dict[str, Any]) -> dict[str, Any]:
    """Separate D12 process metadata; retain action associations in the ledger.

    A partial-card fallback must neither hash its own derived digest nor treat
    a repeated action's number as new evidence. Do not apply this to old profiles
    or to nested observational fields with the same names.
    """
    return {key: value for key, value in view.items()
            if key not in {"delivery_feedback", "action_index"}}


def historical_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Summarize only the registered complete mechanism schema, without inference."""
    row = json.loads(canonical_json_bytes(row))
    body = row["payload"]
    if row.get("representation") in _PARTIAL or row["node"] != "mechanisms" or (
        set(body) != {"decision", "conditions", "distinguishing_checks", "score_authority"}
    ):
        return row  # Unknown/partial records retain every actual delivered field.
    decision = RepairMechanismDecision.model_validate_json(canonical_json_bytes(body["decision"]))
    conditions = [RepairMechanismCondition.model_validate_json(canonical_json_bytes(c))
                  for c in body["conditions"]]
    ids = [c.condition_id for c in conditions]
    if len(set(ids)) != len(ids) or ids != list(decision.condition_ids) or any(
        c.candidate_service_key != decision.candidate_service_key
        or c.mechanism_key != decision.mechanism_key for c in conditions
    ):
        raise ValueError("working evidence condition identity differs")
    summaries = []
    for condition in conditions:
        item = condition.model_dump(mode="json", exclude={
            "leaves", "non_applicable", "original_evidence_ids"})
        counts: Counter[str] = Counter(leaf.truth_value for leaf in condition.leaves)
        item.update(leaf_truth_counts={key: counts[key] for key in ("true", "false", "unknown")},
                    non_applicable_count=len(condition.non_applicable),
                    original_evidence_count=len(condition.original_evidence_ids))
        summaries.append(item)
    return dict(row, payload=dict(body, conditions=summaries),
        representation="historical_evidence_summary",
        original_representation=row.get("representation", "full_semantic_record"),
        record_sha256=sha256_hex(canonical_json_bytes(row)),
        expansion=dict(tool="read_node", parameters=dict(
            card_id=row["card_id"], section="payload", cursor=None)),
        omitted_detail="leaf values and individual evidence IDs remain in the original receipt; "
                       "counts are not independent support; reread for complete detail")


def working_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Project at most eight admitted receipts; never merge on card ID alone.

    Caller owns snapshot/source admission. Each receipt is expanded under its
    original boundary; the resulting new working input has its own whole limit.
    """
    from nurse_rcl.v2.live.ops_sre_context import _delivered_records
    from nurse_rcl.v2.live.ops_sre_overview import overview_source_ids
    from nurse_rcl.v2.live.ops_sre_views import critical_record

    if len(history) > 8 or [h["action_index"] for h in history] != list(
        range(1, len(history) + 1)
    ):
        raise ValueError("working history action sequence differs")
    retained: dict[str, dict[str, Any]] = {}
    overviews: dict[str, dict[str, Any]] = {}
    deliveries: dict[str, list[int]] = {}
    seen_cards: set[str] = set()
    expanded_cards: set[str] = set()
    last_identities: set[str] = set()
    feedback: dict[str, Any] = dict(action_index=None, records=[],
        authority="delivery_metadata_not_diagnostic_evidence")
    for saved in history:
        index, view = saved["action_index"], expand(saved["view"])
        view = working_evidence_view(view)
        last_identities = set()
        feedback = dict(action_index=index, records=[],
                        authority="delivery_metadata_not_diagnostic_evidence")
        if saved["status"] != "completed" or view.get("status") == "resource_limit":
            continue
        rows = _delivered_records(view)
        for row in rows:
            key = row["card_id"]
            identity = sha256_hex(canonical_json_bytes(row))
            complete = row.get("representation") not in _PARTIAL
            prior = deliveries.get(identity, [])
            status = ("repeated_complete_record" if prior and complete else
                      "repeated_partial_record" if prior else
                      "changed_source_or_content" if key in expanded_cards else
                      "first_complete_expansion" if key in seen_cards and complete else
                      "first_delivery")
            feedback["records"].append(dict(card_id=key, record_sha256=identity,
                node=row["node"], targets=row.get("targets", []),
                mechanism_key=row["payload"].get("decision", {}).get("mechanism_key"),
                source_role=row["source_role"], status=status, prior_action_indices=list(prior)))
            deliveries.setdefault(identity, []).append(index)
            if complete:
                expanded_cards.add(key)
            seen_cards.add(key)
            if critical_record(row) or row.get("representation") in _PARTIAL:
                entry = retained.setdefault(identity, dict(card_id=key,
                    record_sha256=identity, record=row, action_indices=[],
                    record_action_index=index))
                entry["action_indices"].append(index)
                last_identities.add(identity)
        seen_cards.update(overview_source_ids(view))
        if "critical_overview" in view:
            summaries = (view["page"]["records"]
                if view.get("page", {}).get("unit") == "candidate_overview"
                else view.get("candidate_overviews", []))
            entry = dict(publication_prefix=view["publication_prefix"],
                         overview=view["critical_overview"], candidate_overviews=summaries)
            identity = sha256_hex(canonical_json_bytes(entry))
            overviews.setdefault(identity, dict(entry, action_indices=[]))["action_indices"].append(
                index)
    evidence = []
    for identity, entry in retained.items():
        projected = {k: v for k, v in entry.items() if k != "record"}
        if identity in last_identities:
            projected["record_location"] = (
                "previous_feedback evidence record with this digest, excluding process metadata")
        else:
            projected["record"] = historical_summary(entry["record"])
        evidence.append(projected)
    return dict(critical_evidence=evidence, unexpanded_critical_overviews=list(overviews.values()),
                delivery_feedback=feedback)
