"""Finite full-detail semantic protocol, separate from legacy/report policies.

Shape validation is not permission to send. The owner still binds fixed source,
scope and leakage before passing the exact request to the persistent bridge.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel

from nurse_rcl.agents.ark import ARK_MODEL
from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.exchange_json import (
    ExchangeBoundaryError,
    ExchangeLimits,
    decode_json_object,
    encode_json,
)
from nurse_rcl.v2.hermes_exchange import PreparedExchange, _validate_payload, prepare_exchange
from nurse_rcl.v2.live.normal_semantic_context import (
    NormalSemanticContext,
    SemanticEdge,
    SemanticError,
    SemanticSignal,
)
from nurse_rcl.v2.live.normal_semantic_exchange import _system
from nurse_rcl.v2.live.normal_semantic_wire import (
    EDGE_COLUMNS,
    ERROR_COLUMNS,
    FACT_COLUMNS,
    NEED_COLUMNS,
    SIGNAL_COLUMNS,
    SUMMARY_COLUMNS,
    WIRE_VERSION,
    build_semantic_wire,
    restore_semantic_context,
)
from nurse_rcl.v2.live.repair_fixed_facts import RepairFixedFact
from nurse_rcl.v2.live.repair_queries import RepairQueryNeed
from nurse_rcl.v2.observation_schemas import TimeInterval, WindowSummary
from nurse_rcl.v2.semantic_run_audit import SemanticRankingPolicy

COMPACT_SEMANTIC_EXCHANGE_VERSION = "nurse.v2.normal-semantic-compact-exchange/1"
COMPACT_POLICY = SemanticRankingPolicy(
    configuration_version="normal-semantic-full-detail-400k/1", input_token_limit=400_000,
)
COMPACT_LIMITS = ExchangeLimits(max_bytes=400_000, max_nodes=100_000)
BODY_VERSION = "nurse.v2.normal-semantic-compact-request/1"
_COLUMNS = dict(
    signal_columns=SIGNAL_COLUMNS, summary_columns=SUMMARY_COLUMNS,
    fact_columns=FACT_COLUMNS, need_columns=NEED_COLUMNS, error_columns=ERROR_COLUMNS,
    edge_columns=EDGE_COLUMNS, request_columns=("request_key", "signal_reference", "needs"),
    request_need_columns=("need_key", "definition_index"),
)
_WIRE_KEYS = set(_COLUMNS) | {
    "schema_version", "opaque_case_id", "scope", "score_authority", "omitted_signal_count",
    "candidates", "focus", "baseline_window", "fault_window", "signals", "summaries",
    "fact_sets", "need_definitions", "requests", "errors", "edges", "graph_limitations",
}
COMPACT_GUIDANCE = """
The user wire is a full-detail columnar display, NOT a candidate ranking.
Each *_columns list names fields of rows in the corresponding table. candidates
contains [short C reference, original service key]; focus lists C references only.
Use C references in response candidate_service_key, S/E/G references in evidence_refs,
R references in request_key and N references in need_keys. Never emit the original
long service/source identifiers in those reference fields. Edge caller/callee
are original service keys and can be matched through the complete candidates table.
Each signal's baseline_summary/fault_summary is a zero-based index into summaries;
fact_set indexes fact_sets, whose rows use fact_columns. Each request is
[R reference, S signal reference, [[N reference, need-definition index], ...]].
The indexed need definition uses need_columns. Indices are integers, not scores.
Dictionary reuse only removes repeated text; it does NOT combine independent
evidence or make one field support a different signal. All Q signals are present,
omitted_signal_count is zero. Non-Q candidates remain in C and are NOT healthy by
default. Missing lifecycle counts and unavailable summaries remain unknown.
"""


def _row(columns: tuple[str, ...], value: Any) -> dict[str, Any]:
    if type(value) is not list or len(value) != len(columns):
        raise ValueError("invalid semantic display row")
    return dict(zip(columns, value, strict=True))


def _at(table: Any, index: Any) -> Any:
    if type(table) is not list or type(index) is not int or not 0 <= index < len(table):
        raise ValueError("invalid semantic display index")
    return table[index]


def _exact(model: type[BaseModel], value: dict[str, Any]) -> None:
    raw = canonical_json_bytes(value)
    if canonical_json_bytes(model.model_validate_json(raw).model_dump(mode="json")) != raw:
        raise ValueError("semantic display must not coerce scientific values")


def _ref(value: Any, prefix: str) -> str:
    if type(value) is not str or re.fullmatch(prefix + r"[0-9]+", value) is None:
        raise ValueError("invalid semantic short reference")
    return value


def _validate_wire(wire: dict[str, Any]) -> None:
    """Validate public shape and references only, not reconstruct private sources."""
    if set(wire) != _WIRE_KEYS or wire["schema_version"] != WIRE_VERSION or (
        wire["scope"] != "fixed_preparation_only" or wire["score_authority"] is not False
        or type(wire["omitted_signal_count"]) is not int or wire["omitted_signal_count"] != 0
    ):
        raise ValueError("invalid compact semantic wire")
    for key, columns in _COLUMNS.items():
        if wire[key] != list(columns):
            raise ValueError("semantic display column contract differs")
    for key in ("signals", "errors", "edges", "summaries", "fact_sets", "need_definitions",
                "requests", "candidates", "focus", "graph_limitations"):
        if type(wire[key]) is not list:
            raise ValueError("invalid semantic display table")
    if any(type(v) is not str for v in wire["graph_limitations"]):
        raise ValueError("invalid semantic graph limitation")
    candidates = {}
    for row in wire["candidates"]:
        item = _row(("alias", "service_key"), row)
        alias = _ref(item["alias"], "C")
        if alias in candidates or type(item["service_key"]) is not str:
            raise ValueError("invalid semantic candidate directory")
        candidates[alias] = item["service_key"]
    if len(set(candidates.values())) != len(candidates) or len(set(wire["focus"])) != len(
        wire["focus"],
    ) or not set(wire["focus"]) <= set(candidates):
        raise ValueError("invalid semantic C/Q")
    for key in ("baseline_window", "fault_window"):
        _exact(TimeInterval, wire[key])
    for row in wire["summaries"]:
        _exact(WindowSummary, _row(SUMMARY_COLUMNS, row))
    for group in wire["fact_sets"]:
        if type(group) is not list:
            raise ValueError("invalid semantic fact set")
        for fact in group:
            _exact(RepairFixedFact, _row(FACT_COLUMNS, fact))
    for row in wire["need_definitions"]:
        _exact(RepairQueryNeed, dict(_row(NEED_COLUMNS, row), need_key="display_only",
                                     condition_key="display_only"))

    def facts(index: Any) -> list[dict[str, Any]]:
        return [_row(FACT_COLUMNS, row) for row in _at(wire["fact_sets"], index)]

    signals = set()
    for row in wire["signals"]:
        item = _row(SIGNAL_COLUMNS, row)
        ref = _ref(item["reference_id"], "S")
        if ref in signals or item["candidate"] not in candidates:
            raise ValueError("invalid semantic signal reference")
        signals.add(ref)
        item["candidate_service_key"] = item.pop("candidate")
        for side in ("baseline", "fault"):
            item[side] = _row(SUMMARY_COLUMNS, _at(wire["summaries"], item.pop(side + "_summary")))
        item["facts"] = facts(item.pop("fact_set"))
        # DTO validation only: these placeholders are never sources or emitted.
        item.update(source_sha256="0" * 64, observation_id="display_only")
        _exact(SemanticSignal, item)
    for row in wire["errors"]:
        item = _row(ERROR_COLUMNS, row)
        _ref(item["reference_id"], "E")
        if item["candidate_service_key"] not in candidates:
            raise ValueError("invalid semantic error candidate")
        item["facts"] = facts(item.pop("fact_set"))
        _exact(SemanticError, item)
    for row in wire["edges"]:
        item = _row(EDGE_COLUMNS, row)
        _ref(item["reference_id"], "G")
        _exact(SemanticEdge, item)
    requests, need_refs = set(), set()
    for row in wire["requests"]:
        item = _row(_COLUMNS["request_columns"], row)
        ref = _ref(item["request_key"], "R")
        if ref in requests or item["signal_reference"] not in signals:
            raise ValueError("invalid semantic request signal reference")
        requests.add(ref)
        if type(item["needs"]) is not list:
            raise ValueError("invalid semantic request needs")
        for need in item["needs"]:
            fields = _row(_COLUMNS["request_need_columns"], need)
            name = _ref(fields["need_key"], "N")
            if name in need_refs:
                raise ValueError("duplicate semantic need reference")
            need_refs.add(name)
            _at(wire["need_definitions"], fields["definition_index"])


def _messages(body: dict[str, Any]) -> list[dict[str, str]]:
    return [dict(role="system", content=_system() + COMPACT_GUIDANCE),
            dict(role="user", content=canonical_json_bytes(body).decode())]


def validate_compact_semantic_payload(payload: dict[str, Any]) -> Literal["semantic_first"]:
    raw = encode_json(payload, COMPACT_LIMITS)
    if _validate_payload(payload) is not None or payload["model"] != ARK_MODEL or (
        payload["max_tokens"] != COMPACT_POLICY.output_token_limit or payload["temperature"] != 0
    ):
        raise ExchangeBoundaryError("invalid_compact_semantic_request")
    if len(raw) + 256 > COMPACT_POLICY.input_token_limit:
        raise ExchangeBoundaryError("compact_semantic_input_budget")
    try:
        messages = payload["messages"]
        if len(messages) != 2:
            raise ValueError
        body = decode_json_object(messages[1]["content"].encode(), COMPACT_LIMITS)
        if set(body) != {"schema_version", "phase", "opaque_case_id", "wire"} or (
            body["schema_version"] != BODY_VERSION or body["phase"] != "semantic_first"
            or type(body["wire"]) is not dict
            or body["opaque_case_id"] != body["wire"]["opaque_case_id"]
        ):
            raise ValueError
        _validate_wire(body["wire"])
        if messages != _messages(body):
            raise ValueError
    except (ValueError, KeyError, TypeError, AttributeError):
        raise ExchangeBoundaryError("invalid_compact_semantic_request") from None
    return "semantic_first"


def prepare_compact_semantic_exchange(*, context: NormalSemanticContext) -> PreparedExchange:
    package = build_semantic_wire(context)
    restore_semantic_context(package, context)
    body = dict(schema_version=BODY_VERSION, phase="semantic_first",
                opaque_case_id=context.opaque_case_id,
                wire=decode_json_object(package.wire_json.encode(), COMPACT_LIMITS))
    prepared = prepare_exchange(
        model=ARK_MODEL, messages=_messages(body),
        max_output_tokens=COMPACT_POLICY.output_token_limit,
        temperature=COMPACT_POLICY.temperature, limits=COMPACT_LIMITS,
    )
    validate_compact_semantic_payload(decode_json_object(prepared.payload_bytes, COMPACT_LIMITS))
    return prepared
