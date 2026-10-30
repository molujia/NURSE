"""Bounded, source-bound navigation across both candidates and three evidence roles.

This is a lossy view of an already frozen publication, never another scorer or
an assertion that unexpanded source cards have been read by the agent.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_sre_briefs import BRIEF_REPRESENTATION

ROLE_CONTRAST_SCHEMA = "nurse.v2.sre-role-contrast/1"
ROLE_CONTRAST_SOFT_BYTES = 32 * 1024
_PARTIAL = {"overview_not_full_judgment", "delivered_section_not_full_judgment"}
_TRUTHS = {"true", "false", "unknown"}
_REQUIREMENTS = {"necessary", "refuting", "optional"}


def _source_ref(row: dict[str, Any]) -> dict[str, Any]:
    return dict(
        card_id=row["card_id"],
        record_sha256=sha256_hex(canonical_json_bytes(row)),
        source_role=row["source_role"],
        source_refs=list(row.get("source_refs", [])),
        window_ns=row.get("window_ns"),
        representation="navigation_reference_not_delivered_detail",
    )


def build_role_manifest(
    scope: list[str],
    records: list[dict[str, Any]],
    scoring: list[dict[str, Any]],
    *,
    publication_epoch_sha256: str,
) -> dict[str, Any]:
    """Small set identity shared across focus changes, not a delivery receipt."""
    if type(scope) is not list or len(scope) != 2 or len(set(scope)) != 2:
        raise ValueError("D23 role manifest scope differs")
    if (
        type(publication_epoch_sha256) is not str
        or len(publication_epoch_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in publication_epoch_sha256)
    ):
        raise ValueError("D23 role manifest publication epoch differs")
    result: dict[str, Any] = dict(
        schema_version="nurse.v2.sre-role-source-manifest/1",
        scope=list(scope),
        publication_epoch_sha256=publication_epoch_sha256,
        authority="frozen_publication_set_identity_not_delivered_evidence",
    )
    for label, candidate in zip(("left", "right"), scope, strict=True):
        result[label] = {}
        for node in ("symptoms", "mechanisms", "scoring"):
            source = scoring if node == "scoring" else records
            identities = sorted(
                (row["card_id"], sha256_hex(canonical_json_bytes(row)))
                for row in source
                if row["node"] == node and row["targets"] == [candidate]
            )
            result[label][node] = dict(
                source_set_sha256=sha256_hex(canonical_json_bytes(identities)),
                total_records=len(identities),
                status="published_source_set" if identities else "not_published_in_frozen_prefix",
            )
    return result


def _delivery_count(rows: list[dict[str, Any]], delivered: list[dict[str, Any]]) -> int:
    available = {row["card_id"] for row in rows}
    return len({
        row["card_id"]
        for row in delivered
        if row.get("card_id") in available
        and row.get("representation") not in _PARTIAL | {BRIEF_REPRESENTATION}
    })


def _role(
    node: str,
    rows: list[dict[str, Any]],
    delivered: list[dict[str, Any]],
) -> dict[str, Any]:
    entry = dict(tool="compare_candidates", focus=node, cursor=None)
    if not rows:
        return dict(
            status="not_supplied", total_records="unknown", complete_delivered_records=0,
            unread_records="unknown", entry=entry, source_bindings=[],
            omitted_source_count=0, meaning="missing_source_is_not_normal_or_absent",
        )
    complete = _delivery_count(rows, delivered)
    refs = [_source_ref(row) for row in sorted(rows, key=lambda r: r["card_id"])[:2]]
    result: dict[str, Any] = dict(
        status="source_metadata_only", total_records=len(rows),
        complete_delivered_records=complete, unread_records=len(rows) - complete,
        entry=entry, source_bindings=refs, omitted_source_count=len(rows) - len(refs),
        meaning="counts_and_references_not_independent_evidence_or_read_detail",
    )
    if node == "symptoms":
        result["examples_not_prioritized"] = [
            dict(
                card_id=row["card_id"],
                fields={
                    key: row["payload"][key]
                    for key in (
                        "kind", "signal_family", "metric_name", "modality",
                        "signed_standardized_change", "eligible",
                    )
                    if key in row["payload"]
                },
            )
            for row in sorted(rows, key=lambda r: r["card_id"])[:2]
        ]
    else:
        counts: dict[str, Counter[str]] = {key: Counter() for key in sorted(_REQUIREMENTS)}
        omitted = 0
        for row in rows:
            for condition in row["payload"].get("conditions", []):
                if type(condition) is not dict:
                    omitted += 1
                    continue
                requirement, truth = condition.get("requirement"), condition.get("truth_value")
                if requirement in _REQUIREMENTS and truth in _TRUTHS:
                    counts[requirement][truth] += 1
                else:
                    omitted += 1
        result["condition_status_counts"] = {
            requirement: {truth: count[truth] for truth in sorted(_TRUTHS)}
            for requirement, count in counts.items()
        }
        result["unclassified_condition_count"] = omitted
        result["condition_count_meaning"] = "source_metadata_occurrences_not_votes_or_full_check"
    return result


def build_role_contrast(
    scope: list[str],
    records: list[dict[str, Any]],
    used_source: dict[str, Any],
    *,
    delivered_records: list[dict[str, Any]],
    max_bytes: int = ROLE_CONTRAST_SOFT_BYTES,
) -> dict[str, Any]:
    """Copy published roles for the two exact targets; omit detail explicitly at capacity."""
    if (
        type(scope) is not list or len(scope) != 2 or len(set(scope)) != 2
        or type(records) is not list or type(delivered_records) is not list
        or type(used_source) is not dict
        or used_source.get("schema_version") != "nurse.v2.sre-used-source-contrast/1"
        or type(max_bytes) is not int or max_bytes < 1024
    ):
        raise ValueError("D23 role contrast input differs")
    if any(
        type(row) is not dict
        or type(row.get("targets")) is not list
        or len(row["targets"]) != 1
        or row["targets"][0] not in scope
        or row.get("node") not in {"symptoms", "mechanisms"}
        for row in records
    ):
        raise ValueError("D23 role contrast source differs")
    value: dict[str, Any] = dict(
        schema_version=ROLE_CONTRAST_SCHEMA,
        authority="frozen_source_navigation_not_new_evidence_or_scoring",
        omitted_detail=[],
    )
    for label, candidate in zip(("left", "right"), scope, strict=True):
        source = [row for row in records if row["targets"] == [candidate]]
        value[label] = dict(
            candidate_service_key=candidate,
            formal_used_score=deepcopy(used_source[label]),
            symptoms=_role("symptoms", [r for r in source if r["node"] == "symptoms"],
                           delivered_records),
            mechanisms=_role("mechanisms", [r for r in source if r["node"] == "mechanisms"],
                             delivered_records),
        )
    if len(canonical_json_bytes(value)) > max_bytes:
        for label in ("left", "right"):
            for node in ("symptoms", "mechanisms"):
                role = value[label][node]
                role["source_bindings"] = []
                role.pop("examples_not_prioritized", None)
                value["omitted_detail"].append(dict(
                    side=label, role=node, reason="role_summary_capacity_limit",
                    omitted_source_count=role["total_records"],
                ))
    if len(canonical_json_bytes(value)) > max_bytes:
        for label in ("left", "right"):
            score = value[label]["formal_used_score"]
            omitted_count = len(score.get("used_families", []))
            score["used_families"] = []
            score["source_bindings"] = []
            if omitted_count:
                score["status"] = "partial"
                value["omitted_detail"].append(dict(
                    side=label, role="formal_used_score", reason="role_summary_capacity_limit",
                    omitted_used_families=omitted_count,
                ))
    if len(canonical_json_bytes(value)) > max_bytes:
        for label in ("left", "right"):
            score = value[label]["formal_used_score"]
            if score.get("status") != "not_published":
                value[label]["formal_used_score"] = dict(
                    status="partial", summary_card_id=score["summary_card_id"],
                    rank=score["rank"],
                    total_score=(score.get("score") or {}).get("total_score"),
                    entry=dict(tool="compare_candidates", focus="scoring", cursor=None),
                    omitted="used_family_and_component_detail_at_capacity",
                )
                value["omitted_detail"].append(dict(
                    side=label, role="formal_used_score", reason="component_capacity_limit",
                ))
    if len(canonical_json_bytes(value)) > max_bytes:
        raise ValueError("D23 role contrast essential fields exceed soft capacity")
    return value
