"""Registered path-presence metadata, never answers or diagnostic evidence.

Only already-admitted deliveries and immutable question scopes are inputs. No IO,
source reader, measurement, provider, dispatch, retry or prose grading lives here.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_sre_facet_reading import ERROR_FIELDS, _field

INDEX_V2 = "nurse.v2.sre-delivered-record-index/2"
AID_V4 = "nurse.v2.sre-factual-reading-aid/4"
FIELD_SCHEMA = "nurse.v2.sre-registered-field-delivery/1"
SUPPLIED = {"supplied_value", "supplied_unknown", "supplied_null"}
AUTHORITY = "field_delivery_only_not_question_answer_or_diagnostic_evidence"


def registered_field_delivery(record: dict[str, Any]) -> dict[str, Any] | None:
    """Copy presence only from the registered complete original record."""
    if (
        record.get("node") != "symptoms"
        or record.get("tool_version") != "ops-symptoms/1"
        or record.get("payload", {}).get("kind") != "error_summary"
        or record.get("representation") is not None
    ):
        return None
    return dict(
        schema_version=FIELD_SCHEMA,
        tool_version=record["tool_version"],
        kind="error_summary",
        source_refs=deepcopy(record["source_refs"]),
        fields=[
            {k: v for k, v in _field(record, facet, path).items() if k != "value"}
            for facet, paths in ERROR_FIELDS.items()
            for path in paths
        ],
    )


def _matches(row: dict[str, Any], question: dict[str, Any]) -> bool:
    scope = question.get("observable_scope")
    if type(scope) is not dict or scope.get("objective_kind") != "error_summary":
        return False
    facets = scope.get("requested_facets")
    binding = scope.get("binding", {})
    delivery = row.get("registered_field_delivery", {})
    return (
        type(facets) is list
        and bool(facets)
        and all(f in ERROR_FIELDS for f in facets)
        and row.get("node") == "symptoms"
        and row.get("delivery_kind") == "complete"
        and all(row.get(k) == binding.get(k) for k in ("card_id", "record_sha256", "targets"))
        and row.get("targets") == question.get("target_service_keys")
        and delivery.get("source_refs") == binding.get("source_refs")
    )


def enhance_delivered_index(
    index: dict[str, Any], history: list[dict[str, Any]], question_state: dict[str, Any]
) -> dict[str, Any]:
    """At most eight original questions, unique completed visible full records."""
    from nurse_rcl.v2.live.ops_sre_context import _delivered_records
    from nurse_rcl.v2.live.ops_sre_encoding import expand
    from nurse_rcl.v2.live.ops_sre_working_memory import working_evidence_view

    questions = question_state["questions"]
    if len(questions) > 8:
        raise ValueError("registered field delivery question limit")
    result = deepcopy(index)
    result["schema_version"] = INDEX_V2
    indexed = {r["record_sha256"]: r for r in result["records"]}
    candidates = {}
    for saved in history:
        if saved["status"] != "completed":
            continue
        view = working_evidence_view(expand(saved["view"]))
        if view.get("status") == "resource_limit":
            continue
        for record in _delivered_records(view):
            identity = sha256_hex(canonical_json_bytes(record))
            row = indexed[identity]
            if not any(
                _matches(
                    dict(
                        row, registered_field_delivery=dict(source_refs=record.get("source_refs"))
                    ),
                    q,
                )
                for q in questions
            ):
                continue
            fields = registered_field_delivery(record)
            if fields is not None:
                candidates[identity] = fields
    for row in result["records"]:
        fields = candidates.get(row["record_sha256"])
        if fields is None:
            continue
        candidate = dict(row, registered_field_delivery=fields)
        if any(_matches(candidate, q) for q in questions) and not any(
            len(
                [
                    r
                    for r in result["records"]
                    if r["card_id"] == row["card_id"] and i in r["action_indices"]
                ]
            )
            != 1
            for i in row["action_indices"]
        ):
            row["registered_field_delivery"] = fields
    return result


def version_pair_enabled(context: dict[str, Any]) -> bool:
    """Old pairs keep their original domain; mixed new/old pairs cannot downgrade."""
    new_index = context.get("delivered_record_index", {}).get("schema_version") == INDEX_V2
    new_aid = context.get("factual_reading_aid", {}).get("schema_version") == AID_V4
    if new_index != new_aid:
        raise ValueError("registered field delivery version pair differs")
    return bool(new_index)


def validate_registered_metadata(index: dict[str, Any]) -> None:
    """Strict bounded transport shape; original historical authentication is owner replay."""
    count = 0
    for row in index["records"]:
        if "registered_field_delivery" not in row:
            continue
        delivery = row["registered_field_delivery"]
        if (
            index.get("schema_version") != INDEX_V2
            or type(delivery) is not dict
            or set(delivery) != {"schema_version", "tool_version", "kind", "source_refs", "fields"}
            or (delivery["schema_version"], delivery["tool_version"], delivery["kind"])
            != (FIELD_SCHEMA, "ops-symptoms/1", "error_summary")
            or row["node"] != "symptoms"
            or row["delivery_kind"] != "complete"
            or type(delivery["source_refs"]) is not list
            or any(type(r) is not str or not r for r in delivery["source_refs"])
            or type(delivery["fields"]) is not list
        ):
            raise ValueError("registered field delivery metadata differs")
        paths = [(f, p) for f, ps in ERROR_FIELDS.items() for p in ps]
        fields = delivery["fields"]
        if len(fields) != len(paths) or any(
            type(field) is not dict
            or set(field) != {"facet", "source_field", "state"}
            or (field["facet"], field["source_field"]) != path
            or type(field["state"]) is not str
            or field["state"] not in SUPPLIED | {"not_supplied"}
            for field, path in zip(fields, paths, strict=False)
        ):
            raise ValueError("registered field delivery paths differ")
        count += len(fields)
    if count > 56:
        raise ValueError("registered field delivery path limit")


def question_delivery_fields(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Match only issued scopes; this projection carries no values or answer state."""
    index = context["delivered_record_index"]
    validate_registered_metadata(index)
    questions = context.get("question_state", {}).get("questions", [])
    if len(questions) > 8:
        raise ValueError("registered field delivery question limit")
    result = []
    matched = set()
    for question in questions:
        rows = [
            r
            for r in index["records"]
            if "registered_field_delivery" in r and _matches(r, question)
        ]
        if len(rows) != 1:
            continue
        row = rows[0]
        indices = row["action_indices"]
        if not indices or any(
            type(i) is not int
            or not 1 <= i <= len(context["actions"])
            or context["actions"][i - 1]["status"] != "completed"
            or row["card_id"] not in context["actions"][i - 1].get("visible_card_ids", [])
            or len(
                [
                    r
                    for r in index["records"]
                    if r["card_id"] == row["card_id"] and i in r["action_indices"]
                ]
            )
            != 1
            for i in indices
        ):
            raise ValueError("registered field delivery visibility differs")
        matched.add(row["record_sha256"])
        requested = question["observable_scope"]["requested_facets"]
        result.append(
            dict(
                authority=AUTHORITY,
                question_id=question["question_id"],
                card_id=row["card_id"],
                record_sha256=row["record_sha256"],
                action_indices=deepcopy(indices),
                objective_kind="error_summary",
                fields=[
                    deepcopy(f)
                    for f in row["registered_field_delivery"]["fields"]
                    if f["facet"] in requested
                ],
            )
        )
    if any(
        "registered_field_delivery" in r and r["record_sha256"] not in matched
        for r in index["records"]
    ):
        raise ValueError("registered field delivery immutable scope differs")
    return result


def already_supplied_facets(context: dict[str, Any], updates: list[dict[str, Any]]) -> bool:
    if not version_pair_enabled(context):
        return False
    fields = {r["question_id"]: r["fields"] for r in question_delivery_fields(context)}
    return any(
        all(
            any(
                f["source_field"] == path and f["state"] in SUPPLIED
                for f in fields.get(row["question_id"], [])
            )
            for path in ERROR_FIELDS[facet]
        )
        for row in updates
        for facet in row["remaining_facets"] or []
        if facet in ERROR_FIELDS
    )


def validate_current_delivery(context: dict[str, Any], *, short_refs: bool = False) -> None:
    """Latest original values check the states; short labels only check display identity."""
    from nurse_rcl.v2.live.ops_sre_context import _delivered_records
    from nurse_rcl.v2.live.ops_sre_facet_reading import _short_identity

    index = context["delivered_record_index"]
    latest = index["latest_action_index"]
    for row in index["records"]:
        if "registered_field_delivery" not in row or latest not in row["action_indices"]:
            continue
        matches = []
        for record in _delivered_records(context["previous_feedback"]):
            fields = registered_field_delivery(record)
            if fields is None or fields != row["registered_field_delivery"]:
                continue
            binding = dict(record_sha256=row["record_sha256"])
            identity = (
                _short_identity(context, record, binding, latest)
                if short_refs
                else sha256_hex(canonical_json_bytes(record))
            )
            if identity == row["record_sha256"]:
                matches.append(record)
        if len(matches) != 1:
            raise ValueError("registered field delivery differs from latest actual record")
