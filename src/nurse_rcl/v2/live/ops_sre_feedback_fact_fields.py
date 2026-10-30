"""Pure, registered presentation of supplied JSON; never infer a diagnosis."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from nurse_rcl.v2.exchange_json import encode_json
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS

JSON_TYPES = {
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    type(None): "null",
    dict: "object",
    list: "array",
}
UNKNOWN_FIELDS = {
    "truth_value",
    "sampling_coverage",
    "disposition",
    "support_value",
    "necessary_support",
    "supplementary_necessary_support",
}
FORMAL_FIELDS = {"decision", "conditions", "distinguishing_checks", "score_authority"}
RECHECK_FIELDS = {
    "formal_decision",
    "conditions",
    "supplementary_necessary_support",
    "score_authority",
    "formal_distinguishing_checks",
    "distinguishing_checks",
}
BRIEF_SCHEMAS = {"nurse.v2.sre-evidence-brief/1", "nurse.v2.sre-evidence-brief/2"}


def field_at(record: dict[str, Any], path: tuple[str | int, ...]) -> dict[str, Any]:
    current: Any = record
    for part in path:
        if type(part) is str and type(current) is dict and part in current:
            current = current.get(part)
        elif type(part) is int and type(current) is list and 0 <= part < len(current):
            current = current[part]
        else:
            return dict(path=list(path), present=False, state="not_supplied")
    if type(current) not in JSON_TYPES:
        raise ValueError("feedback field is not registered JSON")
    state = (
        "supplied_null"
        if current is None
        else (
            "supplied_unknown"
            if path and path[-1] in UNKNOWN_FIELDS and type(current) is str and current == "unknown"
            else "supplied_value"
        )
    )
    return dict(
        path=list(path),
        present=True,
        state=state,
        json_type=JSON_TYPES[type(current)],
        value=deepcopy(current),
    )


def visible_fields(record: dict[str, Any], path: tuple[str | int, ...]) -> list[dict[str, Any]]:
    item = field_at(record, path)
    if not item["present"] or item.get("json_type") not in {"object", "array"}:
        return [item]
    value = item["value"]
    keys = list(value) if type(value) is dict else list(range(len(value)))
    if not keys:
        return [item]
    return [field for key in keys for field in visible_fields(record, (*path, key))]


def project_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Copy registered visible subtrees, without loading or expanding their sources."""
    encode_json(record, CONTEXT_LIMITS)
    body = record.get("payload")
    if type(body) is not dict:
        return None
    representation = record.get("representation")
    if representation != "condition_evidence_brief" and record.get("schema_version") is not None:
        return None
    symptom = (
        record.get("node") == "symptoms"
        and record.get("tool_version") == "ops-symptoms/1"
        and body.get("kind") == "error_summary"
        and representation is None
    )
    mechanism = (
        record.get("node") == "mechanisms"
        and record.get("tool_version") == "ops-mechanisms/1"
        and set(body) in (FORMAL_FIELDS, RECHECK_FIELDS)
        and representation in {None, "condition_evidence_brief", "historical_evidence_summary"}
    )
    if representation == "condition_evidence_brief":
        mechanism = mechanism and record.get("schema_version") in BRIEF_SCHEMAS
    if not (symptom or mechanism):
        return None
    required = {"card_id", "node", "tool_version", "targets", "source_role", "source_refs"}
    if (
        not required <= record.keys()
        or any(
            type(record[k]) is not str or not record[k]
            for k in ("card_id", "node", "tool_version", "source_role")
        )
        or any(
            type(record[k]) is not list
            or not record[k]
            or any(type(v) is not str or not v for v in record[k])
            for k in ("targets", "source_refs")
        )
    ):
        raise ValueError("registered feedback record source binding missing")
    if symptom:
        if type(body.get("facts")) is not dict:
            raise ValueError("registered feedback symptom facts differ")
        paths = [
            ("payload", "facts", name)
            for name in (
                "baseline_fraction",
                "fault_fraction",
                "baseline_known_denominator",
                "fault_known_denominator",
                "observed_error_count",
                "sampling_coverage",
            )
        ]
        fields = [field_at(record, path) for path in paths]
        fields += visible_fields(record, ("payload", "facts", "facts"))
    else:
        decision_key = "formal_decision" if "formal_decision" in body else "decision"
        if type(body.get("conditions")) is not list or type(body[decision_key]) is not dict:
            raise ValueError("registered feedback mechanism structure differs")
        fields = visible_fields(record, ("payload", decision_key))
        fields += visible_fields(record, ("payload", "conditions"))
    return dict(
        qualifiers=deepcopy({k: v for k, v in record.items() if k != "payload"}),
        visible_payload=deepcopy(body),
        fields=fields,
    )
