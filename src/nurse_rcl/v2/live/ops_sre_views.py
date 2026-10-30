"""Lossless semantic records over admitted summaries/cards, never raw queries."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from nurse_rcl.v2.live.ops_card_view import critical_judgments, entities
from nurse_rcl.v2.live.ops_contracts import OpsNodeEvidence
from nurse_rcl.v2.live.ops_node_common import node_card, prepared_snapshot_id
from nurse_rcl.v2.live.ops_tools import card_id
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData


def prepared_summaries(data: RepairPreparationData) -> tuple[OpsNodeEvidence, ...]:
    """Use only already-loaded summary fields; do not traverse source point arrays."""
    snapshot = prepared_snapshot_id(data)
    requests = {r.source_sha256: r.request_key for r in data.catalog.requests}
    cards = []
    for source in data.sources:
        quality, obs = source.quality, source.observation
        if source.candidate_service_key is None:
            continue
        cards.append(node_card(
            snapshot=snapshot, node="symptoms",
            parameters=dict(candidate_service_key=source.candidate_service_key,
                            observation_id=obs.observation_id),
            payload=dict(
                kind="prepared_summary", candidate_service_key=source.candidate_service_key,
                observation_id=obs.observation_id, request_key=requests.get(source.source_sha256),
                signal_family=obs.signal_family, modality=obs.modality,
                metric_name=source.metric_name,
                baseline=quality.source_summary.baseline_summary.model_dump(mode="json"),
                fault=quality.source_summary.fault_summary.model_dump(mode="json"),
                baseline_window=source.time_contract.baseline_window.model_dump(mode="json"),
                fault_window=source.time_contract.fault_window.model_dump(mode="json"),
                signed_standardized_change=quality.signed_standardized_change,
                eligible=quality.formal_scoring_eligible,
                episode_status="not_measured_by_summary",
            ),
            evidence_ids=(obs.observation_id,),
            source_refs=(source.source_sha256, *source.source_file_sha256s),
            interval=source.query_interval, unit=obs.unit, coverage=quality.coverage_quality,
            availability="partial",
            limitations=(*quality.quality_flags, "summary_cannot_prove_episode_or_normality"),
        ).model_copy(update={"tool_version": "sre-prepared-summary/1"}))
    return tuple(cards)


def semantic_record(
    card: OpsNodeEvidence, *, published: bool, extra_targets: tuple[str, ...] = (),
    origin: str | None = None,
) -> dict[str, Any]:
    parameters, body = json.loads(card.parameters_json), json.loads(card.payload_json)
    targets = sorted(set(entities(parameters)) | set(entities(body)) | set(extra_targets))
    role = "core_published" if published else {
        "sre-prepared-summary/1": "prepared_summary",
        "ops-dependency/1": "prepared_dependency",
        "ops-boundary/1": "prepared_boundary",
    }.get(card.tool_version, "investigation_observation")
    if not published and origin in {"measured", "core_reuse"}:
        role = "investigation_" + origin
    return dict(
        card_id=card_id(card), node=card.node, tool_version=card.tool_version,
        parameters=parameters, targets=targets, payload=body,
        source_role=role, published_by_core=published,
        formal_scoring_use="not_asserted_by_publication", score_authority=False,
        unit=card.unit, window_ns=card.window_ns, coverage=card.coverage,
        availability=card.availability, truth_value=card.truth_value,
        evidence_ids=card.evidence_ids, source_refs=card.source_refs,
        limitations=card.limitations,
    )


def critical_record(row: dict[str, Any]) -> bool:
    # A mechanism is indivisible: keep its entire AND/OR, requirement roles and
    # supportive siblings rather than detaching a negative leaf from its scope.
    if row["source_role"] in {"prepared_summary", "prepared_boundary"}:
        return False
    if row.get("representation") == "overview_not_full_judgment" and any(
        row.get("judgment_status_counts", {}).get(key, 0) > 0
        for key in ("false", "unknown", "refuting_true")
    ):
        return True  # Retain the caveat summary; this does not establish a full judgment.
    has_critical = next(critical_judgments(row["payload"], preserve_scope=True), None) is not None
    return bool(row["node"] == "mechanisms" or row["truth_value"] in {"false", "unknown"}
                or has_critical)


def record_order(row: dict[str, Any]) -> tuple[int, str]:
    kind = row["payload"].get("kind")
    priority = 0 if kind in {"paid_episode", "supplementary_episode"} else (
        1 if row["node"] == "mechanisms" else 2 if row["node"] == "scoring" else (
            4 if row["source_role"] == "prepared_summary" else 3))
    return priority, row["card_id"]


def _judgment_counts(body: object) -> dict[str, dict[str, int]]:
    """Count recorded judgments by structural level, never recompute their truth.

    An OR parent can be true while a leaf is unknown. Keep both levels visible;
    these are metadata occurrences, not independent evidence or votes.
    """
    groups: dict[str, Counter[str]] = {}

    def visit(value: object, path: str) -> None:
        if type(value) is dict:
            truth = value.get("truth_value")
            if type(truth) is str and truth in {"true", "false", "unknown"}:
                count = groups.setdefault(path, Counter())
                count[truth] += 1
                if truth == "true" and value.get("requirement") == "refuting":
                    count["refuting_true"] += 1
            for key, child in value.items():
                visit(child, path + "." + key)
        elif type(value) is list:
            for child in value:
                visit(child, path + "[]")

    visit(body, "payload")
    return {key: dict(value) for key, value in groups.items()}


def overview_record(row: dict[str, Any]) -> dict[str, Any]:
    """A status/observation overview, explicitly not a complete mechanism judgment."""
    body = row["payload"]
    selected = {k: v for k, v in body.items() if type(v) not in (dict, list)}
    for field in ("baseline", "fault", "baseline_window", "fault_window"):
        if field in body:
            selected[field] = body[field]
    groups = _judgment_counts(body)
    counts: Counter[str] = Counter({"refuting_true": 0})
    for group in groups.values():
        counts.update(group)
    decision = body.get("decision", {})
    if decision:
        selected["mechanism_key"] = decision.get("mechanism_key")
        selected["disposition"] = decision.get("disposition")
    return dict(row, payload=selected, judgment_status_counts=dict(counts),
                judgment_status_by_group=groups,
                judgment_count_meaning="metadata_occurrences_by_level_not_independent_support",
                representation="overview_not_full_judgment",
                expansion=dict(tool="read_node", parameters=dict(
                    card_id=row["card_id"], section="payload", cursor=None)),
                omitted_detail="nested conditions and observations require complete expansion")


def critical_overview(
    rows: list[dict[str, Any]], *, expanded_ids: set[str], next_cursor: str | None,
) -> dict[str, Any]:
    critical = [r for r in rows if critical_record(r)]
    pending = [r for r in critical if r["card_id"] not in expanded_ids]
    judgment_counts: Counter[str] = Counter()
    for row in pending:
        judgment_counts.update(overview_record(row)["judgment_status_counts"])
    return dict(total_count=len(critical), unexpanded_count=len(pending),
                unexpanded_by_node=dict(Counter(r["node"] for r in pending)),
                unexpanded_by_truth=dict(Counter(r["truth_value"] for r in pending)),
                unexpanded_judgment_status_counts=dict(judgment_counts),
                next_cursor=next_cursor,
                meaning="metadata_counts_not_independent_support; "
                        "unexpanded_is_not_normal_or_ruled_out; "
                        "continue_same_query_or_expand_registered_card")
