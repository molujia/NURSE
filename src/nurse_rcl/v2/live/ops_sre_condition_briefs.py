"""D14 registered condition meaning over admitted sources; no new observations.

Formal and supplementary payloads have explicit adapters. A derived logical
explanation is valid only for the same condition/window with known leaves held
fixed. It neither authorizes hidden leaves nor changes any formal result.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Literal, cast

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_sre_briefs import BRIEF_REPRESENTATION
from nurse_rcl.v2.live.repair_mechanisms import (
    RepairMechanismCondition,
    RepairMechanismDecision,
    _compose,
)

CONDITION_BRIEF_SCHEMA = "nurse.v2.sre-evidence-brief/2"
_FORMAL = {"decision", "conditions", "distinguishing_checks", "score_authority"}
_RECHECK = {
    "formal_decision",
    "conditions",
    "supplementary_necessary_support",
    "score_authority",
    "formal_distinguishing_checks",
    "distinguishing_checks",
}
_PARTIAL = {"overview_not_full_judgment", "delivered_section_not_full_judgment"}


def condition_resolution(operator: str, truth: str, counts: dict[str, int]) -> str:
    """Explain a nonempty Kleene condition, treating no applicability as unknown."""
    if operator not in {"or", "and"} or set(counts) != {"true", "false", "unknown"}:
        raise ValueError("condition resolution shape differs")
    if any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("condition resolution counts differ")
    # Presence suffices; do not expand counts into a new leaf array.
    states: tuple[Literal["true", "false", "unknown"], ...] = ("true", "false", "unknown")
    expected = _compose(tuple(k for k in states if counts[k]), cast(Literal["or", "and"], operator))
    if truth != expected:
        raise ValueError("condition truth differs from complete leaves")
    if sum(counts.values()) == 0:
        return "no_applicable_evidence"
    if counts["unknown"] == 0:
        return "fully_observed"
    if (operator == "or" and counts["true"]) or (operator == "and" and counts["false"]):
        return "decisive_known_leaf"
    return "unresolved_applicable_leaves"


def project_condition_brief(row: dict[str, Any]) -> dict[str, Any]:
    """Retain unknown/partial schemas; project only two complete registered shapes."""
    source_bytes = canonical_json_bytes(row)
    projected: dict[str, Any] = json.loads(source_bytes)
    body = projected["payload"]
    if projected.get("representation") in _PARTIAL or projected["node"] != "mechanisms":
        return projected
    shape = set(body)
    if shape not in (_FORMAL, _RECHECK):
        return projected
    recheck = shape == _RECHECK
    field = "formal_decision" if recheck else "decision"
    id_field = "formal_condition_id" if recheck else "condition_id"
    decision = RepairMechanismDecision.model_validate_json(canonical_json_bytes(body[field]))
    conditions = []
    for raw in body["conditions"]:
        adapted = dict(raw)
        if recheck:
            if "condition_id" in adapted or "formal_condition_id" not in adapted:
                raise ValueError("recheck condition identity differs")
            adapted["condition_id"] = adapted.pop("formal_condition_id")
        conditions.append(
            RepairMechanismCondition.model_validate_json(canonical_json_bytes(adapted))
        )
    ids = [c.condition_id for c in conditions]
    if (
        len(set(ids)) != len(ids)
        or ids != list(decision.condition_ids)
        or any(
            c.candidate_service_key != decision.candidate_service_key
            or c.mechanism_key != decision.mechanism_key
            for c in conditions
        )
    ):
        raise ValueError("condition source identity differs")
    if recheck and (
        body["score_authority"] is not False
        or body["supplementary_necessary_support"]
        != _compose(tuple(c.truth_value for c in conditions if c.requirement == "necessary"), "and")
    ):
        raise ValueError("supplementary condition support differs")
    summaries = []
    for condition in conditions:
        item = condition.model_dump(
            mode="json", exclude={"leaves", "non_applicable", "original_evidence_ids"}
        )
        counts: Counter[str] = Counter(leaf.truth_value for leaf in condition.leaves)
        values = {key: counts[key] for key in ("true", "false", "unknown")}
        item.update(
            leaf_truth_counts=values,
            non_applicable_count=len(condition.non_applicable),
            original_evidence_count=len(condition.original_evidence_ids),
            resolution_state=condition_resolution(
                condition.operator, condition.truth_value, values
            ),
            resolution_assumption="known_leaves_unchanged",
            resolution_scope="same_condition_structure_and_window",
        )
        if recheck:
            item[id_field] = item.pop("condition_id")
        summaries.append(item)
    body["conditions"] = summaries
    for original, count in (
        ("condition_ids", "condition_count"),
        ("support_evidence_ids", "support_evidence_count"),
    ):
        body[field][count] = len(body[field].pop(original))
    projected.update(
        schema_version=CONDITION_BRIEF_SCHEMA,
        representation=BRIEF_REPRESENTATION,
        original_representation=row.get("representation", "full_semantic_record"),
        mechanism_source_kind="supplementary_recheck" if recheck else "formal_mechanism",
        record_sha256=sha256_hex(source_bytes),
        expansion=dict(
            tool="read_node",
            parameters=dict(card_id=row["card_id"], section="payload", cursor=None),
        ),
        omitted_detail="Individual leaves and evidence IDs remain in the complete source; "
        "counts are not votes or independent observations. Resolution assumes the same "
        "condition/window and unchanged known leaves; replacement measurements may change it. "
        "Explicit detail reads consume an action. "
        "Supplementary rechecks cannot replace formal RCL.",
    )
    projected["brief_sha256"] = sha256_hex(canonical_json_bytes(projected))
    return projected
