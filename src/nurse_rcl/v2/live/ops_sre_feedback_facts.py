"""Source-bound projection from current carriers, not an evidence/source loader."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import encode_json
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_sre_feedback_fact_fields import project_record
from nurse_rcl.v2.live.ops_sre_feedback_representation import binding_index

CHECK_SCHEMA = "nurse.v2.sre-feedback-fact-check/1"
CHECK_AUTHORITY = "presentation_only_no_new_evidence_or_semantic_acceptance"
CHECK_SCOPE = "currently_visible_registered_records_only_not_all_case_evidence"
_LOCATION = "previous_feedback record with this exact digest"
_DERIVED = {"condition_evidence_brief", "historical_evidence_summary"}


def _indices(value: Any, actions: list[dict[str, Any]]) -> None:
    if (
        type(value) is not list
        or not value
        or any(
            type(i) is not int
            or not 1 <= i <= len(actions)
            or actions[i - 1]["status"] != "completed"
            for i in value
        )
        or value != sorted(set(value))
    ):
        raise ValueError("feedback delivery action indices differ")


def _admit_carriers(
    context: dict[str, Any], short_refs: bool
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    encode_json(context, CONTEXT_LIMITS)
    index = context.get("decision_index")
    actions = context.get("actions")
    if (
        context.get("schema_version") != "nurse.v2.sre-investigation-input/24"
        or type(index) is not int
        or not 1 <= index <= 8
        or type(actions) is not list
        or len(actions) >= index
        or any(
            type(a) is not dict
            or type(a.get("action_index")) is not int
            or a["action_index"] != i
            or type(a.get("status")) is not str
            for i, a in enumerate(actions, 1)
        )
    ):
        raise ValueError("feedback current input/action identity differs")
    feedback = context.get("previous_feedback")
    delivery = context.get("delivered_record_index")
    critical = context.get("critical_evidence")
    latest = len(actions) or None
    if (
        (type(feedback) is not dict and not (not actions and feedback is None))
        or type(delivery) is not dict
        or type(critical) is not list
        or type(delivery.get("records")) is not list
        or delivery.get("latest_action_index") != latest
        or type(delivery.get("latest_record_refs")) is not list
    ):
        raise ValueError("feedback current carriers differ")
    if actions and (
        type(feedback) is not dict
        or type(feedback.get("action_index")) is not int
        or feedback["action_index"] != latest
        or ("status" in feedback and feedback["status"] != actions[-1]["status"])
    ):
        raise ValueError("feedback latest action/result binding differs")
    if not actions and feedback:
        raise ValueError("feedback first turn has a result")
    pattern = r"H[1-9][0-9]{0,3}" if short_refs else r"[0-9a-f]{64}"
    seen: set[str] = set()
    for item in delivery["records"]:
        if (
            type(item) is not dict
            or not {
                "card_id",
                "node",
                "targets",
                "source_role",
                "record_sha256",
                "action_indices",
                "delivery_kind",
            }
            <= item.keys()
            or type(item["record_sha256"]) is not str
            or not re.fullmatch(pattern, item["record_sha256"])
            or item["record_sha256"] in seen
        ):
            raise ValueError("feedback source index binding differs")
        _indices(item["action_indices"], actions)
        seen.add(item["record_sha256"])
    refs = delivery["latest_record_refs"]
    expected = [m["record_sha256"] for m in delivery["records"] if latest in m["action_indices"]]
    if len(refs) != len(set(refs)) or set(refs) != set(expected):
        raise ValueError("feedback latest record index differs")
    return actions, delivery, critical


def _bind(
    record: dict[str, Any],
    *,
    hint: str | None,
    indices: list[int],
    delivery: dict[str, Any],
    short_refs: bool,
    representation_index: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    # Registration validates required source fields even when note admission failed.
    project_record(record)
    digest = sha256_hex(canonical_json_bytes(record))
    declared = record.get("record_sha256") if record.get("representation") in _DERIVED else None
    matches: list[dict[str, Any]] = []
    for item in delivery["records"]:
        if any(
            canonical_json_bytes(record.get(k)) != canonical_json_bytes(item[k])
            for k in ("card_id", "node", "targets", "source_role")
        ):
            continue
        if "window_ns" in item and record.get("window_ns") != item["window_ns"]:
            continue
        if not set(indices) <= set(item["action_indices"]):
            continue
        if hint is not None and hint != item["record_sha256"]:
            continue
        if not short_refs and item["record_sha256"] not in {digest, declared}:
            continue
        if short_refs and declared is not None:
            if representation_index is None:
                if declared != item["record_sha256"]:
                    continue  # Unchanged validation of original packets.
            else:
                relation = representation_index.get(digest)
                if relation is None or relation['source_record_ref'] != declared or (
                    relation['delivered_record_ref'] != item['record_sha256']
                ):
                    continue
        matches.append(item)
    if len(matches) != 1:
        raise ValueError("feedback record has no unique current source binding")
    item = matches[0]
    if record.get("representation") == "condition_evidence_brief":
        seal = record.get("brief_sha256")
        if type(seal) is not str or not re.fullmatch(r"[0-9a-f]{64}", seal):
            raise ValueError("feedback brief seal missing")
        if not short_refs and seal != sha256_hex(
            canonical_json_bytes({k: v for k, v in record.items() if k != "brief_sha256"})
        ):
            raise ValueError("feedback visible brief seal differs")
    return item


def visible_records(context: dict[str, Any], *, short_refs: bool = False,
                    representation_bindings: list[dict[str, Any]] | None = None,
                    ) -> list[dict[str, Any]]:
    """Return copied bodies/metadata; locations resolve exclusively inside this input.

    Original source admission remains the owner's responsibility. A short-ID
    transport can check local correspondence, never authenticate the original SHA.
    """
    from nurse_rcl.v2.live.ops_sre_context import _delivered_records

    actions, delivery, critical = _admit_carriers(context, short_refs)
    relations = binding_index(representation_bindings)
    if relations is not None and not short_refs:
        raise ValueError('short presentation bindings cannot authenticate long sources')
    candidates: list[dict[str, Any]] = []
    latest = len(actions)
    if actions and actions[-1]["status"] == "completed":
        for position, record in enumerate(_delivered_records(context["previous_feedback"])):
            item = _bind(
                record, hint=None, indices=[latest], delivery=delivery, short_refs=short_refs,
                representation_index=relations,
            )
            candidates.append(
                dict(
                    record=record,
                    source_binding=item,
                    location=["previous_feedback", "delivered_records", position],
                )
            )
    latest_candidates = list(candidates)
    if {c["source_binding"]["record_sha256"] for c in candidates} != set(
        delivery["latest_record_refs"]
    ):
        raise ValueError("feedback latest bodies differ from current index")
    for position, item in enumerate(critical):
        if (
            type(item) is not dict
            or ("record" in item) == ("record_location" in item)
            or not {"card_id", "record_sha256", "action_indices", "record_action_index"}
            <= item.keys()
        ):
            raise ValueError("feedback record body/location binding differs")
        _indices(item["action_indices"], actions)
        if type(item["record_action_index"]) is not int or (
            item["record_action_index"] not in item["action_indices"]
        ):
            raise ValueError("feedback record first-delivery binding differs")
        if "record" in item:
            row = item["record"]
            bound = _bind(
                row,
                hint=item["record_sha256"],
                indices=item["action_indices"],
                delivery=delivery,
                short_refs=short_refs,
                representation_index=relations,
            )
            location = ["critical_evidence", position, "record"]
        else:
            found = [
                c
                for c in latest_candidates
                if c["source_binding"]["record_sha256"] == item["record_sha256"]
            ]
            if (
                item["record_location"] != _LOCATION
                or len(found) != 1
                or latest not in item["action_indices"]
            ):
                raise ValueError("feedback unresolved current record location")
            row, bound = found[0]["record"], found[0]["source_binding"]
            location = ["critical_evidence", position, "record_location"]
        if (
            item["card_id"] != row["card_id"]
            or item["action_indices"] != bound["action_indices"]
            or item["record_action_index"] != bound["action_indices"][0]
        ):
            raise ValueError("feedback critical/index source binding differs")
        candidates.append(dict(record=row, source_binding=bound, location=location))
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        row, bound = candidate["record"], candidate["source_binding"]
        identity = (bound["record_sha256"], row.get("representation", "full_semantic_record"))
        if identity in grouped:
            previous = grouped[identity]
            if canonical_json_bytes(row) != canonical_json_bytes(previous["record"]):
                raise ValueError("feedback same source/representation has conflicting bodies")
            previous["source_locations"].append(candidate["location"])
        else:
            grouped[identity] = dict(
                record=deepcopy(row),
                source_binding=deepcopy(bound),
                action_indices=deepcopy(bound["action_indices"]),
                source_locations=[candidate["location"]],
            )
    if relations is not None and set(relations) != {
        sha256_hex(canonical_json_bytes(item['record'])) for item in grouped.values()
        if item['record'].get('representation') in _DERIVED
    }:
        raise ValueError('feedback representation bindings do not cover exactly current carriers')
    return list(grouped.values())


def feedback_fact_check(context: dict[str, Any], *, short_refs: bool = False,
                        representation_bindings: list[dict[str, Any]] | None = None,
                        ) -> dict[str, Any]:
    candidates = visible_records(context, short_refs=short_refs,
                                 representation_bindings=representation_bindings)
    result: dict[str, Any] = dict(
        schema_version=CHECK_SCHEMA,
        authority=CHECK_AUTHORITY,
        scope=CHECK_SCOPE,
        latest_action=None,
        records=[],
        not_projected=[],
    )
    if context["actions"]:
        action = context["actions"][-1]
        result["latest_action"] = dict(
            action_index=action["action_index"],
            status=action["status"],
            result_location=["previous_feedback"],
        )
    for candidate in candidates:
        row = candidate["record"]
        projection = project_record(row)
        if projection is None:
            result["not_projected"].append(
                dict(
                    card_id=row.get("card_id"),
                    node=row.get("node"),
                    source_locations=deepcopy(candidate["source_locations"]),
                    reason="unregistered_record",
                )
            )
        else:
            result["records"].append(
                dict(
                    projection,
                    source_binding=deepcopy(candidate["source_binding"]),
                    source_locations=deepcopy(candidate["source_locations"]),
                    action_indices=deepcopy(candidate["action_indices"]),
                    current_representation_sha256=sha256_hex(canonical_json_bytes(row)),
                )
            )
    encode_json(result, CONTEXT_LIMITS)
    return result


def validate_feedback_fact_check(
    value: dict[str, Any], context: dict[str, Any], *, short_refs: bool = False,
    representation_bindings: list[dict[str, Any]] | None = None,
) -> None:
    expected = feedback_fact_check(context, short_refs=short_refs,
                                   representation_bindings=representation_bindings)
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise ValueError("feedback fact check differs from visible supplied input")
