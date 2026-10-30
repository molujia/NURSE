"""Exact field-delivery display for a registered narrow read, never an answer.

Only the latest supplied complete record and its admitted immutable question are
inputs. Unknown/null availability is separate from diagnostic adequacy. This
module cannot update a question, grade prose, send, dispatch or read a source.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex

ERROR_FIELDS = {
    "error_count": ("payload.facts.observed_error_count",),
    "denominator": (
        "payload.facts.baseline_known_denominator",
        "payload.facts.fault_known_denominator",
    ),
    "fraction": ("payload.facts.baseline_fraction", "payload.facts.fault_fraction"),
    "unit": ("unit",),
    "coverage": ("payload.facts.sampling_coverage",),
}


def _field(record: dict[str, Any], facet: str, path: str) -> dict[str, Any]:
    """Presence of a registered path, not validity/adequacy of its supplied value."""
    value: Any = record
    for key in path.split("."):
        if type(value) is not dict or key not in value:
            return dict(facet=facet, source_field=path, state="not_supplied")
        value = value[key]
    state = (
        "supplied_null"
        if value is None
        else "supplied_unknown"
        if type(value) is str and value == "unknown"
        else "supplied_value"
    )
    return dict(facet=facet, source_field=path, state=state, value=deepcopy(value))


def _short_identity(
    context: dict[str, Any], record: dict[str, Any], binding: dict[str, Any], latest: int
) -> str | None:
    """Model-only display consistency, never original canonical SHA admission.

    Aliases cannot be rehashed to recover an original digest. Require that the
    same issued digest labels this latest complete delivery and its index. The
    owner with its reference book still decodes and checks the full record SHA.
    """
    identity = binding.get("record_sha256")
    if type(identity) is not str or re.fullmatch(r"H[1-9][0-9]{0,3}", identity) is None:
        return None
    delivery = context["previous_feedback"].get("delivery_feedback", {})
    index = context.get("delivered_record_index", {})
    if (
        delivery.get("action_index") != latest
        or index.get("latest_action_index") != latest
        or identity not in index.get("latest_record_refs", [])
    ):
        return None
    for metadata, indexed in ((delivery, False), (index, True)):
        rows = [r for r in metadata.get("records", []) if r.get("record_sha256") == identity]
        if len(rows) != 1:
            return None
        row = rows[0]
        if row.get("delivery_kind") != "complete" or any(
            row.get(k) != record.get(k) for k in ("card_id", "targets", "source_role")
        ):
            return None
        if indexed and (
            latest not in row.get("action_indices", [])
            or row.get("node") != record.get("node")
            or row.get("window_ns") != record.get("window_ns")
            or row.get("fields_present") != sorted(record["payload"])
        ):
            return None
    return identity


def latest_question_field_reading_aid(
    context: dict[str, Any], *, short_refs: bool = False
) -> list[dict[str, Any]]:
    """No inference from completed status: match the complete record and original scope."""
    from nurse_rcl.v2.live.ops_sre_context import _delivered_records

    actions = context["actions"]
    if not actions or actions[-1]["status"] != "completed":
        return []
    latest = actions[-1]["action_index"]
    feedback = context["previous_feedback"]
    if feedback.get("action_index") != latest:
        raise ValueError("field reading aid latest feedback identity differs")
    if feedback.get("status") == "resource_limit":
        return []
    state = context.get("question_state", {})
    question_id = state.get("latest_action_question_id")
    if question_id is None:
        return []
    questions = [r for r in state.get("questions", []) if r["question_id"] == question_id]
    if len(questions) != 1:
        return []
    question = questions[0]
    scope = question.get("observable_scope")
    if type(scope) is not dict or scope.get("objective_kind") != "error_summary":
        return []
    facets = scope["requested_facets"]
    if any(f not in ERROR_FIELDS for f in facets):
        return []
    binding = scope["binding"]
    result = []
    for record in _delivered_records(feedback):
        if (
            record.get("node") != "symptoms"
            or record.get("tool_version") != "ops-symptoms/1"
            or record["payload"].get("kind") != "error_summary"
            or record.get("representation") is not None
            or any(record.get(k) != binding.get(k) for k in ("card_id", "targets", "source_refs"))
            or record.get("targets") != question["target_service_keys"]
        ):
            continue
        identity = (
            _short_identity(context, record, binding, latest)
            if short_refs
            else sha256_hex(canonical_json_bytes(record))
        )
        if identity is None or identity != binding.get("record_sha256"):
            continue
        result.append(
            dict(
                authority="field_delivery_only_not_question_answer_or_diagnostic_evidence",
                question_id=question_id,
                action_index=latest,
                card_id=record["card_id"],
                record_sha256=identity,
                objective_kind="error_summary",
                fields=[
                    _field(record, facet, path) for facet in facets for path in ERROR_FIELDS[facet]
                ],
            )
        )
    return result
