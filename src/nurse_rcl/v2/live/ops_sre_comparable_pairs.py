"""D16 exact same-kind comparison metadata; never a merged observation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_sre_comparable_briefs import (
    COMPARABLE_BRIEF_SCHEMA,
    project_comparable_brief,
)
from nurse_rcl.v2.live.ops_sre_condition_briefs import _FORMAL, _PARTIAL, _RECHECK

PairKey = tuple[str, str, tuple[int, int], str]
ConditionKey = tuple[str, str, str]


def _condition_key(condition: dict[str, Any]) -> ConditionKey:
    return condition["requirement"], condition["predicate_kind"], condition["operator"]


def _meaning_fields(brief: dict[str, Any]) -> dict[str, Any]:
    """Only comparable semantic fields, not different source/target identities."""
    body = brief["payload"]
    decision = body.get("decision", body.get("formal_decision"))
    result = {
        "decision." + k: decision[k]
        for k in ("disposition", "permitted_roles", "reason_codes", "causal_status")
    }
    if "supplementary_necessary_support" in body:
        result["supplementary_necessary_support"] = body["supplementary_necessary_support"]
    # A condition key may have multiple retained sources. Never pick one.
    grouped: dict[ConditionKey, list[dict[str, Any]]] = defaultdict(list)
    for c in body["conditions"]:
        grouped[_condition_key(c)].append(
            {
                k: c[k]
                for k in (
                    "truth_value",
                    "leaf_truth_counts",
                    "non_applicable_count",
                    "resolution_state",
                    "resolution_assumption",
                    "resolution_scope",
                    "condition_meaning",
                    "reason_codes",
                )
            }
        )
    for key, values in grouped.items():
        result["conditions." + canonical_json_bytes(key).decode()] = sorted(
            values, key=canonical_json_bytes
        )
    return result


def mechanism_pairs(rows: list[dict[str, Any]], scope: list[str]) -> list[dict[str, Any]]:
    """Group exact complete sources without Cartesian multiplication or mutation.

    Complete retained records remain elsewhere in the tool receipt. These refs
    describe grouping only and confer no facts about an undelivered source.
    """
    if len(scope) != 2 or scope[0] == scope[1]:
        raise ValueError("mechanism comparison requires two distinct ordered targets")
    grouped: dict[PairKey, dict[str, list[dict[str, Any]]]] = {}
    seen: set[str] = set()
    for row in rows:
        if (
            row["card_id"] in seen
            or row["node"] != "mechanisms"
            or row.get("representation") in _PARTIAL
            or set(row["payload"]) not in (_FORMAL, _RECHECK)
        ):
            raise ValueError("mechanism comparison requires distinct complete registered sources")
        seen.add(row["card_id"])
        brief = project_comparable_brief(row)
        if brief.get("schema_version") != COMPARABLE_BRIEF_SCHEMA:
            raise ValueError("mechanism comparison source is not registered")
        decision = brief["payload"].get("decision", brief["payload"].get("formal_decision"))
        candidate = decision["candidate_service_key"]
        if candidate not in scope or row["targets"] != [candidate]:
            raise ValueError("mechanism comparison source scope differs")
        window = row["window_ns"]
        if len(window) != 2 or any(type(n) is not int for n in window) or window[0] >= window[1]:
            raise ValueError("mechanism comparison window differs")
        key = (
            decision["mechanism_key"],
            decision["mechanism_definition_sha256"],
            (window[0], window[1]),
            brief["mechanism_source_kind"],
        )
        grouped.setdefault(key, {c: [] for c in scope})[candidate].append(brief)
    result = []
    for key, sides in sorted(grouped.items()):
        left, right = (sorted(sides[c], key=lambda r: r["card_id"]) for c in scope)
        status = (
            "missing_side"
            if not left or not right
            else "single_source_each_side"
            if len(left) == len(right) == 1
            else "multiple_sources_inspect_individually"
        )
        same: list[str] = []
        different: list[str] = []
        if status == "single_source_each_side":
            a, b = _meaning_fields(left[0]), _meaning_fields(right[0])
            for field in sorted(a.keys() | b.keys()):
                (
                    same
                    if field in a
                    and field in b
                    and canonical_json_bytes(a[field]) == (canonical_json_bytes(b[field]))
                    else different
                ).append(field)
        conditions: dict[ConditionKey, dict[str, list[dict[str, Any]]]] = {}
        for label, sources in (("left", left), ("right", right)):
            for row in sources:
                for c in row["payload"]["conditions"]:
                    cid = c.get("condition_id", c.get("formal_condition_id"))
                    conditions.setdefault(_condition_key(c), {"left": [], "right": []})[
                        label
                    ].append(dict(card_id=row["card_id"], condition_id=cid))
        reasons = set()
        if status == "missing_side":
            absent = scope[0] if not left else scope[1]
            for other, candidates in grouped.items():
                if candidates[absent]:
                    for i, reason in enumerate(
                        (
                            "different_mechanism",
                            "different_definition",
                            "different_window",
                            "different_source_category",
                        )
                    ):
                        if key[i] != other[i]:
                            reasons.add(reason)
            if not reasons:
                reasons.add("counterpart_not_published")
        result.append(
            dict(
                kind="mechanism_comparison",
                mechanism_key=key[0],
                mechanism_definition_sha256=key[1],
                window_ns=list(key[2]),
                mechanism_source_kind=key[3],
                left=dict(
                    candidate_service_key=scope[0],
                    status="published" if left else "not_published",
                    card_ids=[r["card_id"] for r in left],
                ),
                right=dict(
                    candidate_service_key=scope[1],
                    status="published" if right else "not_published",
                    card_ids=[r["card_id"] for r in right],
                ),
                conditions=[
                    dict(requirement=k[0], predicate_kind=k[1], operator=k[2], **v)
                    for k, v in sorted(conditions.items())
                ],
                comparison_status=status,
                same_fields=same,
                different_fields=different,
                not_directly_comparable_reasons=sorted(reasons),
                score_authority=False,
                meaning="exact_field_comparison_not_independent_evidence_or_score_contribution",
            )
        )
    return result
