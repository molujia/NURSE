"""Pure reading aids over current supplied data, never evidence or prose grading.

No file, source reader, scoring, model, retry or dispatch API lives here. Original
records retain their authority; these copies neither repair native text nor license
an unobserved diagnosis. In particular, no timestamp unit is guessed from magnitude.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from nurse_rcl.v2.live.ops_sre_facet_reading import latest_question_field_reading_aid
from nurse_rcl.v2.live.ops_sre_note_reference_view import note_reference_reading_aid


def exact_unix_seconds(ns: int) -> str:
    """Lossless decimal display of an explicitly declared integer ns value."""
    if type(ns) is not int:
        raise ValueError("time reading aid requires an integer ns value")
    whole, fraction = divmod(abs(ns), 1_000_000_000)
    text = str(whole)
    if fraction:
        text += "." + f"{fraction:09d}".rstrip("0")
    return ("-" if ns < 0 else "") + text


def latest_window_reading_aid(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Only exact registered windows in the latest actual completed delivery."""
    from nurse_rcl.v2.live.ops_sre_context import _delivered_records

    actions = context["actions"]
    if not actions or actions[-1]["status"] != "completed":
        return []
    latest = actions[-1]["action_index"]
    feedback = context["previous_feedback"]
    if feedback.get("action_index") != latest:
        raise ValueError("time reading aid latest feedback identity differs")
    if feedback.get("status") == "resource_limit":
        return []
    result = []
    for row in _delivered_records(feedback):
        body = row["payload"]
        registered = (row.get("tool_version"), body.get("kind"))
        if row.get("node") != "symptoms" or row.get("representation") is not None:
            continue
        if registered not in {
            ("ops-symptoms/1", "error_summary"),
            ("ops-symptoms/1", "fixed_summary"),
            ("sre-prepared-summary/1", "prepared_summary"),
        }:
            continue
        windows = [("window_ns", row.get("window_ns"))]
        if registered[1] != "error_summary":
            for field in ("baseline_window", "fault_window"):
                value = body.get(field)
                if type(value) is dict:
                    windows.append(
                        (
                            "payload." + field,
                            [value.get("start_unix_ns"), value.get("end_unix_ns")],
                        )
                    )
        for path, window in windows:
            if (
                type(window) is not list
                or len(window) != 2
                or any(type(ns) is not int for ns in window)
                or window[0] < 0
                or window[0] >= window[1]
            ):
                continue
            start, end = window
            result.append(
                dict(
                    action_index=latest,
                    card_id=row["card_id"],
                    source_field=path,
                    source_unit="Unix ns",
                    exact_window_ns=[start, end],
                    exact_unix_seconds=[exact_unix_seconds(start), exact_unix_seconds(end)],
                    duration_ns=end - start,
                    duration_seconds=exact_unix_seconds(end - start),
                )
            )
    return result


def directory_reading_aid(capabilities: dict[str, Any]) -> dict[str, Any]:
    """Copy the supplied directory in its original order, without a rank election."""
    rows = capabilities.get("investigation_situation", {}).get("candidates", [])
    return dict(
        authority="navigation_not_diagnostic_evidence_or_selection",
        category_unit="published_node_cards_not_measured_edges_or_votes",
        judgment_count_unit="recorded_occurrences_not_independent_support",
        rank_order="smaller_supplied_rank_precedes_larger_supplied_rank",
        comparison_scope="only_supplied_directory_records_no_global_extremum_inferred",
        records=[
            dict(
                candidate_service_key=row["candidate_service_key"],
                category_card_counts=deepcopy(row.get("categories", {})),
                scoring_status=row.get("scoring", {}).get("status", "not_published"),
                scoring_records=[
                    dict(card_id=r["card_id"], rank=r.get("rank"))
                    for r in row.get("scoring", {}).get("records", [])
                ],
            )
            for row in rows
        ],
    )


def factual_reading_aid(
    context: dict[str, Any],
    *,
    short_refs: bool = False,
    schema_version: str = "nurse.v2.sre-factual-reading-aid/3",
) -> dict[str, Any]:
    """Derived presentation only; the original supplied records remain authoritative."""
    if schema_version not in {
        "nurse.v2.sre-factual-reading-aid/2",
        "nurse.v2.sre-factual-reading-aid/3",
        "nurse.v2.sre-factual-reading-aid/4",
    }:
        raise ValueError("factual reading aid version is not registered")
    from nurse_rcl.v2.live.ops_sre_registered_field_delivery import version_pair_enabled

    version_pair_enabled(dict(context, factual_reading_aid=dict(schema_version=schema_version)))
    aid = dict(
        schema_version=schema_version,
        authority="presentation_only_no_new_evidence_or_semantic_acceptance",
        time_windows=latest_window_reading_aid(context),
        candidate_directory=directory_reading_aid(context["capabilities"]),
        latest_question_fields=latest_question_field_reading_aid(context, short_refs=short_refs),
    )
    if schema_version in {
        "nurse.v2.sre-factual-reading-aid/3", "nurse.v2.sre-factual-reading-aid/4"
    }:
        aid["note_references"] = note_reference_reading_aid(context)
    if schema_version == "nurse.v2.sre-factual-reading-aid/4":
        from nurse_rcl.v2.live.ops_sre_registered_field_delivery import (
            question_delivery_fields,
            validate_current_delivery,
        )

        validate_current_delivery(context, short_refs=short_refs)
        aid["question_delivery_fields"] = question_delivery_fields(context)
    return aid


def validate_factual_reading_aid(context: dict[str, Any], *, short_refs: bool = False) -> None:
    """Transport consistency, not interpretation of any model-written statement."""
    aid = context.get("factual_reading_aid")
    if type(aid) is not dict or type(aid.get("schema_version")) is not str:
        raise ValueError("factual reading aid version is not registered")
    if aid != factual_reading_aid(
        context, short_refs=short_refs, schema_version=aid["schema_version"]
    ):
        raise ValueError("factual reading aid differs from current supplied input")
