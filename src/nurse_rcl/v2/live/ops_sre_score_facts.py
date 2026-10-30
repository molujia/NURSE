"""Registered score-source facts, not observed diagnostic preconditions."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_report_context import OpsReportFact
from nurse_rcl.v2.live.ops_sre_report_context import _iter_record_facts


def score_record_facts(
    row: dict[str, Any],
    *,
    action_index: int,
    candidates: set[str],
) -> Iterator[OpsReportFact]:
    """Use exact existing f grammar; retained-source layout preserves original keys.

    Two names in RepairStrengthCalculation use uppercase B/F. Normalize only
    those registered slots, not arbitrary user keys. Source paths, all values,
    windows and targets remain unchanged; restore uses the original field map.
    """
    adapted = json.loads(canonical_json_bytes(row))
    payload = adapted["payload"]
    if (
        payload.get("kind")
        not in {
            "score_lineage_summary",
            "score_family",
            "score_propagation",
            "score_path",
            "score_term",
        }
        or row["node"] != "scoring"
    ):
        raise ValueError("score fact source layout is not registered")
    for field in ("term", "selected_term"):
        term = payload.get(field)
        if term is None:
            continue
        calculation = term["calculation"]
        for original, normalized in (("median_B", "median_b"), ("median_F", "median_f")):
            if normalized in calculation or original not in calculation:
                raise ValueError("score calculation source layout differs")
            calculation[normalized] = calculation.pop(original)
    for item in _iter_record_facts(adapted, action_index=action_index, candidates=candidates):
        yield item.model_copy(
            update={
                "fact": item.fact.model_copy(
                    update=dict(
                        kind="candidate_evidence",
                        allowed_claim_strength="ranking_inference",
                        scope_basis="frozen_assessment",
                    )
                )
            }
        )
