"""Registered read-only cards from admitted D16 score lineage, never a scorer."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_contracts import OpsNodeEvidence
from nurse_rcl.v2.live.ops_node_common import node_card
from nurse_rcl.v2.live.ops_sre_score_lineage import SreScoreLineage, term_selection
from nurse_rcl.v2.live.ops_tools import card_id
from nurse_rcl.v2.observation_schemas import TimeInterval

SCORE_CARD_VERSION = "sre-score-lineage/1"
SCORE_PAGE_KINDS = {"score_lineage_summary", "score_family", "score_propagation"}
_BRIEF_SCORE_FIELDS = (
    "candidate_service_key",
    "event_evidence_score",
    "propagation_score",
    "local_score_component",
    "propagation_score_component",
    "rule_score_component",
    "history_score_component",
    "total_score",
    "selected_family_keys",
)
_BRIEF_TERM_FIELDS = ("term_id", "metric_name", "modality", "local_strength")
_BRIEF_SELECTION_FIELDS = (
    "pool_selected",
    "family_selected",
    "family_used_in_aggregation",
    "used_in_family_aggregation",
)
_USED_BRIEF_MAX_BYTES = 16 * 1024


def score_lineage_cards(
    projection: SreScoreLineage,
    *,
    interval: TimeInterval,
) -> tuple[OpsNodeEvidence, ...]:
    return tuple(iter_score_lineage_cards(projection, interval=interval))


def iter_score_lineage_cards(
    projection: SreScoreLineage, *, interval: TimeInterval,
) -> Iterator[OpsNodeEvidence]:
    """Copy exact source values; navigation refs do not supply unread detail."""
    terms = {t.term_id: t for t in projection.terms}
    pools = {p.pool_key: p for p in projection.pools}
    paths = {p.path_id: p for p in projection.paths}
    witnesses = {w.witness_id: w for w in projection.witnesses}

    def card(
        candidate: str,
        payload: dict[str, Any],
        evidence: tuple[str, ...],
        *,
        window: TimeInterval = interval,
    ) -> OpsNodeEvidence:
        return node_card(
            snapshot=projection.core_identity.case_snapshot_id,
            node="scoring",
            parameters=dict(candidate_service_key=candidate),
            payload=dict(payload, score_authority=False),
            evidence_ids=evidence,
            source_refs=(projection.projection_sha256, projection.core_result_digest),
            interval=window,
            unit="dimensionless_support_score",
            limitations=(
                "read_only_formal_score_lineage_not_additive_causal_contribution",
                "unread_detail_reference_is_not_supplied_evidence",
                "summary_cannot_prove_episode_or_normality",
            ),
        ).model_copy(update={"tool_version": SCORE_CARD_VERSION})

    for row in projection.candidates:
        candidate = row.score.candidate_service_key
        term_cards = {}
        for term in projection.terms:
            if term.candidate_service_key != candidate:
                continue
            item = card(
                candidate,
                dict(
                    kind="score_term",
                    term=term.model_dump(mode="json"),
                    pool=pools[term.pool_key].model_dump(mode="json"),
                    selection=term_selection(row, term.term_id),
                ),
                (term.term_id, term.evidence_id),
                window=term.interval,
            )
            term_cards[term.term_id] = card_id(item)
            yield item
        family_cards = []
        for family in row.local.families:
            ids = (
                *family.numeric_term_ids,
                *family.dependent_unused_term_ids,
                *family.unknown_term_ids,
            )
            selected = terms.get(family.selected_term_id or "")
            item = card(
                candidate,
                dict(
                    kind="score_family",
                    candidate_service_key=candidate,
                    family=family.model_dump(mode="json"),
                    used_in_aggregation=family.primary_family in row.local.selected_family_keys,
                    selected_term=selected.model_dump(mode="json") if selected else None,
                    selected_term_selection=term_selection(row, selected.term_id)
                    if selected
                    else None,
                    term_details=[
                        dict(term_id=k, card_id=term_cards[k], selection=term_selection(row, k))
                        for k in ids
                    ],
                    selection_meaning=row.selection_meaning,
                ),
                ids or (row.score.score_sha256,),
            )
            family_cards.append(dict(primary_family=family.primary_family, card_id=card_id(item)))
            yield item
        path_cards = []
        for key in row.propagation.retained_path_ids:
            path = paths[key]
            item = card(
                candidate,
                dict(
                    kind="score_path",
                    path=path.model_dump(mode="json"),
                    selected=key in row.propagation.selected_path_ids,
                    witnesses=[
                        witnesses[k].model_dump(mode="json")
                        for k in dict.fromkeys((path.source_witness_id, path.target_witness_id))
                    ],
                ),
                (path.path_id, path.source_term_id, path.target_term_id),
            )
            path_cards.append(dict(path_id=key, card_id=card_id(item)))
            yield item
        prop_card = card(
            candidate,
            dict(
                kind="score_propagation",
                candidate_service_key=candidate,
                propagation=row.propagation.model_dump(mode="json"),
                path_details=path_cards,
                meaning="numeric_zero_is_not_verified_absence_of_propagation",
            ),
            row.propagation.retained_path_ids or (row.score.score_sha256,),
        )
        yield prop_card
        yield card(
                candidate,
                dict(
                    kind="score_lineage_summary",
                    candidate_service_key=candidate,
                    rank=row.rank,
                    score=row.score.model_dump(mode="json"),
                    final_term_ids=list(row.final_term_ids),
                    selection_meaning=row.selection_meaning,
                    scoring_config=projection.scoring_config.model_dump(mode="json"),
                    aggregation_definition=projection.aggregation_definition.model_dump(
                        mode="json"
                    ),
                    source_pins=projection.source_pins.model_dump(mode="json"),
                    family_details=family_cards,
                    propagation_detail=dict(card_id=card_id(prop_card)),
                    tie_break_is_evidence=False,
                    is_root_cause_probability=False,
                ),
                (row.score.score_sha256,),
        )


def _used_source_side(
    summary_rows: list[dict[str, Any]],
    groups: dict[tuple[int, str], dict[str, list[dict[str, Any]]]],
    candidate: str,
) -> dict[str, Any]:
    """Lossy copy of admitted card fields, not child-card delivery or a scorer."""
    if not summary_rows:
        return dict(status="not_published", summary_card_id=None, rank=None, score=None,
                    used_families=[], omitted=[], source_bindings=[])
    summary = summary_rows[0]
    body = summary["payload"]
    score = body["score"]
    selected = score["selected_family_keys"]
    if type(selected) is not list or len(selected) != len(set(selected)):
        raise ValueError("score selected family source differs")
    details = {r["primary_family"]: r["card_id"] for r in body["family_details"]}
    if len(details) != len(body["family_details"]):
        raise ValueError("score family detail source differs")
    side: dict[str, Any] = dict(
        status="complete", summary_card_id=summary["card_id"], rank=body["rank"],
        score={key: score[key] for key in _BRIEF_SCORE_FIELDS},
        used_families=[], omitted=[],
        source_bindings=[dict(card_id=summary["card_id"],
                              record_sha256=sha256_hex(canonical_json_bytes(summary)),
                              source_refs=summary.get("source_refs", []))],
    )
    for key in selected:
        entries = groups.get((1, key), {}).get(candidate, [])
        if not entries:
            side["omitted"].append(dict(primary_family=key, reason="source_not_published"))
            continue
        if len(entries) != 1:
            raise ValueError("score family source conflicts")
        row = entries[0]
        family = row["payload"]
        if (
            row["card_id"] != details.get(key)
            or row.get("source_refs") != summary.get("source_refs")
            or row.get("source_role") != summary.get("source_role")
            or family.get("used_in_aggregation") is not True
            or family.get("family", {}).get("primary_family") != key
        ):
            side["omitted"].append(dict(primary_family=key, reason="source_identity_differs"))
            continue
        term = family.get("selected_term")
        selection = family.get("selected_term_selection")
        if (
            type(term) is not dict or type(selection) is not dict
            or any(field not in term for field in _BRIEF_TERM_FIELDS)
            or any(field not in selection for field in _BRIEF_SELECTION_FIELDS)
            or term["term_id"] != family["family"].get("selected_term_id")
        ):
            side["omitted"].append(dict(primary_family=key, reason="source_incomplete"))
            continue
        copied = dict(
            primary_family=key, value=family["family"]["value"],
            family_card_id=row["card_id"],
            selected_term={field: term[field] for field in _BRIEF_TERM_FIELDS},
            selected_term_selection={field: selection[field] for field in _BRIEF_SELECTION_FIELDS},
        )
        binding = dict(
            card_id=row["card_id"], record_sha256=sha256_hex(canonical_json_bytes(row)),
            source_refs=row.get("source_refs", []),
        )
        proposed = dict(side, used_families=[*side["used_families"], copied],
                        source_bindings=[*side["source_bindings"], binding])
        if len(canonical_json_bytes(proposed)) > _USED_BRIEF_MAX_BYTES:
            side["omitted"].append(dict(primary_family=key, reason="brief_capacity_limit"))
            continue
        side = proposed
    if side["omitted"] or any(family["value"] is None for family in side["used_families"]):
        side["status"] = "partial"
    return side


def score_pairs(
    rows: list[dict[str, Any]], scope: list[str], *, used_source_contrast: bool = False,
) -> list[dict[str, Any]]:
    """Published formal scores first, then exact family sources and propagation."""
    if len(scope) not in (1, 2) or len(set(scope)) != len(scope):
        raise ValueError("score comparison scope differs")
    groups: dict[tuple[int, str], dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        if len(row["targets"]) != 1 or row["targets"][0] not in scope:
            raise ValueError("score comparison source scope differs")
        body = row["payload"]
        kind = body.get("kind")
        if "score" in body:
            key = (0, "score_comparison")
        elif kind == "score_family":
            key = (1, body["family"]["primary_family"])
        elif kind == "score_propagation":
            key = (2, "propagation_comparison")
        else:
            raise ValueError("score comparison source is unregistered")
        groups.setdefault(key, {c: [] for c in scope})[row["targets"][0]].append(row)
    # Even before publication, expose an explicit unready summary, not health.
    groups.setdefault((0, "score_comparison"), {c: [] for c in scope})
    result = []
    for key, sides in sorted(groups.items()):
        if any(len(values) > 1 for values in sides.values()):
            raise ValueError("score comparison has conflicting published sources")
        pair: dict[str, Any] = dict(
            kind="score_family_comparison" if key[0] == 1 else key[1],
            score_authority=False,
            meaning="exact_saved_score_inputs_not_additive_causal_contribution",
        )
        for i, candidate in enumerate(scope):
            values = sides[candidate]
            pair[("left", "right")[i]] = dict(
                candidate_service_key=candidate,
                status="published" if values else "not_published",
                card_ids=[r["card_id"] for r in values],
            )
        if key[0] == 1:
            pair["primary_family"] = key[1]
        if key[0] == 0:
            pair["details_delivery"] = (
                "family_details_require_following_units"
                if all(
                    v and v[0]["payload"].get("kind") == "score_lineage_summary"
                    for v in sides.values()
                )
                else "not_published"
            )
            pair["differences"] = {}
            pair["difference_direction"] = "left_minus_right_not_causal_effect"
            if len(scope) == 2 and all(sides.values()):
                a, b = (sides[c][0]["payload"]["score"] for c in scope)
                pair["differences"] = {
                    k: a[k] - b[k]
                    for k in (
                        "event_evidence_score",
                        "propagation_score",
                        "local_score_component",
                        "propagation_score_component",
                        "rule_score_component",
                        "history_score_component",
                        "total_score",
                    )
                }
            if used_source_contrast:
                pair["used_source_contrast"] = dict(
                    schema_version="nurse.v2.sre-used-source-contrast/1",
                    left=_used_source_side(sides[scope[0]], groups, scope[0]),
                    right=_used_source_side(
                        sides[scope[1]], groups, scope[1]
                    ) if len(scope) == 2 else None,
                    limitations=[
                        "copied_fields_only_unread_source_cards_not_supplied",
                        "same_family_values_do_not_prove_same_mechanism",
                        "not_additive_causal_contribution",
                    ],
                )
        result.append(pair)
    return result
