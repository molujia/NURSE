"""D18 delivery metadata and explicit selection of old public agent statements.

Pure projections of admitted receipts, not source admission or new evidence.
The original conversation and complete receipts remain authoritative.
"""

from __future__ import annotations

import re
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import decode_json_object, encode_json
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_sre_briefs import BRIEF_REPRESENTATION
from nurse_rcl.v2.live.ops_sre_encoding import expand
from nurse_rcl.v2.live.ops_sre_working_memory import working_evidence_view

WORKBENCH_PROFILE = "hermes-sre-tools/11"
WORKBENCH_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/12"
_PARTIAL = {"overview_not_full_judgment", "delivered_section_not_full_judgment"}
_SCORE_KINDS = {
    "score_lineage_summary",
    "score_family",
    "score_term",
    "score_propagation",
    "score_path",
}


def _copy(value: dict[str, Any]) -> dict[str, Any]:
    return decode_json_object(encode_json(value, CONTEXT_LIMITS), CONTEXT_LIMITS)


def _sequence(rows: list[dict[str, Any]], key: str, label: str) -> None:
    if (
        type(rows) is not list
        or len(rows) > 8
        or any(
            type(row) is not dict or type(row.get(key)) is not int or row[key] != index
            for index, row in enumerate(rows, 1)
        )
    ):
        raise ValueError(f"workbench {label} sequence differs")


def select_public_memory(memory: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep latest public text; select older questions, never infer resolution."""
    _sequence(memory, "decision_index", "memory")
    grouped: dict[bytes, dict[str, Any]] = {}
    for row in memory[:-1]:
        if row.get("role") != "agent_statement":
            raise ValueError("workbench memory role differs")
        question = row.get("investigation_question") or {}
        selected = dict(
            questions=row.get("questions", []),
            unresolved=row.get("unresolved", []),
            target_service_keys=question.get("target_service_keys", []),
            card_ids=question.get("card_ids", []),
            responds_to=row.get("responds_to", []),
        )
        identity = canonical_json_bytes(selected)
        entry = grouped.setdefault(
            identity,
            dict(
                selected,
                source_decisions=[],
                role="agent_statement",
                resolution="not_assessed_by_program",
            ),
        )
        entry["source_decisions"].append(
            dict(decision_index=row["decision_index"], action_index=row.get("action_index"))
        )
    if memory and memory[-1].get("role") != "agent_statement":
        raise ValueError("workbench memory role differs")
    return _copy(dict(recent=memory[-1:], question_history=list(grouped.values())))


def _record_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Registered field meaning only; field presence never implies child delivery."""
    body = row["payload"]
    representation = row.get("representation")
    delivery = (
        "brief"
        if representation == BRIEF_REPRESENTATION
        else "partial"
        if representation in _PARTIAL
        else "complete"
    )
    kind = body.get("kind")
    score = (
        kind in _SCORE_KINDS
        and row.get("node") == "scoring"
        and row.get("tool_version") == "sre-score-lineage/1"
    )
    mechanism = row.get("node") == "mechanisms" and (
        type(body.get("decision")) is dict or type(body.get("formal_decision")) is dict
    )
    result = dict(
        card_id=row["card_id"],
        record_sha256=sha256_hex(canonical_json_bytes(row)),
        kind=kind
        if score
        else ("mechanism_brief" if delivery == "brief" else "mechanism_record")
        if mechanism
        else "unregistered",
        node=row["node"],
        source_role=row["source_role"],
        targets=row.get("targets", []),
        delivery_kind=delivery,
        fields_present=sorted(body),
        action_indices=[],
        latest_delivery=None,
        section_fields={},
    )
    if "window_ns" in row:
        result["window_ns"] = row["window_ns"]
    if score:
        result["section_fields"] = {
            key: sorted(body[key]) if type(body[key]) is dict else None
            for key in (
                "family",
                "term",
                "pool",
                "selection",
                "selected_term",
                "selected_term_selection",
                "path",
            )
            if key in body
        }
        child_key = {"score_family": "family", "score_term": "term", "score_path": "path"}.get(kind)
        child = body.get(child_key, {}) if child_key else {}
        for field in ("primary_family", "term_id", "path_id"):
            if field in child:
                result[field] = child[field]
        if kind == "score_term":
            result["numeric_state"] = (
                "not_supplied"
                if "local_strength" not in child
                else "unknown"
                if child["local_strength"] is None
                else "known"
            )
    if mechanism:
        decision = body.get("decision", body.get("formal_decision", {}))
        if "mechanism_key" in decision:
            result["mechanism_key"] = decision["mechanism_key"]
    return result


def delivered_record_index(
    history: list[dict[str, Any]], *, question_state: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Summarize at most eight real deliveries, sharing by exact record identity.

    The owner admits history before calling. No source file, registry, hidden
    child, model statement or semantic inference participates in this index.
    """
    from nurse_rcl.v2.live.ops_sre_context import _delivered_records

    _sequence(history, "action_index", "history")
    entries: dict[str, dict[str, Any]] = {}
    latest: list[str] = []
    for saved in history:
        latest = []
        for entry in entries.values():
            entry["latest_delivery"] = None
        view = working_evidence_view(expand(saved["view"]))
        if saved["status"] != "completed" or view.get("status") == "resource_limit":
            continue
        for row in _delivered_records(view):
            metadata = _record_metadata(row)
            identity = metadata["record_sha256"]
            prior = [e for e in entries.values() if e["card_id"] == metadata["card_id"]]
            repeated = identity in entries
            relation = (
                "repeat_" + metadata["delivery_kind"] + "_no_new_content"
                if repeated
                else "first_complete_expansion"
                if prior
                and metadata["delivery_kind"] == "complete"
                and all(e["delivery_kind"] == "partial" for e in prior)
                else "changed_content_or_representation"
                if prior
                else "first_" + metadata["delivery_kind"]
            )
            entry = entries.setdefault(identity, metadata)
            if saved["action_index"] not in entry["action_indices"]:
                entry["action_indices"].append(saved["action_index"])
            entry["latest_delivery"] = relation
            if identity not in latest:
                latest.append(identity)
    index = dict(
        schema_version="nurse.v2.sre-delivered-record-index/1",
        latest_action_index=history[-1]["action_index"] if history else None,
        latest_record_refs=latest,
        records=list(entries.values()),
        authority="delivery_metadata_not_diagnostic_evidence",
    )
    if question_state is not None:
        from nurse_rcl.v2.live.ops_sre_registered_field_delivery import enhance_delivered_index

        index = enhance_delivered_index(index, history, question_state)
    return _copy(index)


def validate_workbench_context(context: dict[str, Any]) -> None:
    """Structural transport checks only; the owner separately replays source metadata."""
    encode_json(context, CONTEXT_LIMITS)
    actions = context["actions"]
    _sequence(actions, "action_index", "actions")
    latest = len(actions) or None
    index = context["delivered_record_index"]
    if (
        type(index) is not dict
        or set(index)
        != {"schema_version", "latest_action_index", "latest_record_refs", "records", "authority"}
        or (
            index["schema_version"] not in {
                "nurse.v2.sre-delivered-record-index/1", "nurse.v2.sre-delivered-record-index/2"
            }
            or index["authority"] != "delivery_metadata_not_diagnostic_evidence"
            or index["latest_action_index"] != latest
            or type(index["records"]) is not list
        )
    ):
        raise ValueError("workbench record index differs")
    required = {
        "card_id",
        "record_sha256",
        "kind",
        "node",
        "source_role",
        "targets",
        "delivery_kind",
        "fields_present",
        "action_indices",
        "latest_delivery",
        "section_fields",
    }
    optional = {
        "window_ns",
        "primary_family",
        "term_id",
        "path_id",
        "numeric_state",
        "mechanism_key",
    }
    if index["schema_version"] == "nurse.v2.sre-delivered-record-index/2":
        optional.add("registered_field_delivery")
    seen: set[str] = set()
    actual_latest = []
    for row in index["records"]:
        if type(row) is not dict or not required <= row.keys() or set(row) - required - optional:
            raise ValueError("workbench record fields differ")
        digest, indices = row["record_sha256"], row["action_indices"]
        if (
            type(digest) is not str
            or not re.fullmatch(r"(?:[0-9a-f]{64}|H[1-9][0-9]{0,3})", digest)
            or digest in seen
            or type(indices) is not list
            or not indices
            or any(
                type(i) is not int
                or not 1 <= i <= len(actions)
                or actions[i - 1]["status"] != "completed"
                for i in indices
            )
            or indices != sorted(set(indices))
            or row["delivery_kind"] not in {"brief", "partial", "complete"}
            or type(row["fields_present"]) is not list
            or any(type(f) is not str for f in row["fields_present"])
            or row["fields_present"] != sorted(set(row["fields_present"]))
            or type(row["section_fields"]) is not dict
            or not set(row["section_fields"]) <= set(row["fields_present"])
            or row.get("numeric_state", "not_supplied") not in {"known", "unknown", "not_supplied"}
        ):
            raise ValueError("workbench record structure differs")
        seen.add(digest)
        if latest in indices:
            actual_latest.append(digest)
            if row["latest_delivery"] not in {
                "first_complete",
                "first_partial",
                "first_brief",
                "first_complete_expansion",
                "changed_content_or_representation",
                "repeat_complete_no_new_content",
                "repeat_partial_no_new_content",
                "repeat_brief_no_new_content",
            }:
                raise ValueError("workbench latest delivery differs")
        elif row["latest_delivery"] is not None:
            raise ValueError("workbench stale delivery marked latest")
    refs = index["latest_record_refs"]
    if type(refs) is not list or len(refs) != len(set(refs)) or set(refs) != set(actual_latest):
        raise ValueError("workbench latest record references differ")
    if index["schema_version"] == "nurse.v2.sre-delivered-record-index/2":
        from nurse_rcl.v2.live.ops_sre_registered_field_delivery import (
            question_delivery_fields,
            version_pair_enabled,
        )

        version_pair_enabled(context)
        question_delivery_fields(context)
    memory = context["public_working_memory"]
    if type(memory) is not dict or set(memory) != {"recent", "question_history"}:
        raise ValueError("workbench public memory shape differs")
    prior = context["decision_index"] - 1
    recent, older = memory["recent"], memory["question_history"]
    if type(recent) is not list or len(recent) != min(prior, 1) or type(older) is not list:
        raise ValueError("workbench memory source count differs")
    if recent and (
        type(recent[0].get("decision_index")) is not int
        or recent[0]["decision_index"] != prior
        or recent[0].get("role") != "agent_statement"
    ):
        raise ValueError("workbench recent memory source differs")
    sources = []
    for group in older:
        if (
            type(group) is not dict
            or set(group)
            != {
                "questions",
                "unresolved",
                "target_service_keys",
                "card_ids",
                "responds_to",
                "source_decisions",
                "role",
                "resolution",
            }
            or group["role"] != "agent_statement"
            or group["resolution"] != "not_assessed_by_program"
        ):
            raise ValueError("workbench historical memory differs")
        for source in group["source_decisions"]:
            if set(source) != {"decision_index", "action_index"} or (
                type(source["decision_index"]) is not int
                or (
                    source["action_index"] is not None
                    and (
                        type(source["action_index"]) is not int
                        or not 1 <= source["action_index"] <= len(actions)
                    )
                )
            ):
                raise ValueError("workbench question source differs")
            sources.append(source["decision_index"])
    if sorted(sources) != list(range(1, prior)):
        raise ValueError("workbench question history source sequence differs")
