"""Readable, lossless object-list tables for SRE receipts, not semantic summaries.

Each table row means common + zip(columns, row). No references, expressions,
missing-value sentinels, type coercion or implicit field-name abbreviations.
Both encoded and expanded objects retain the original JSON resource boundary.
"""
from __future__ import annotations

import json
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, ExchangeLimits, encode_json

SEMANTIC_ENCODING = "sre-lossless-rows/1"
TABLE = "sre_rows_v1"
LIMITS = ExchangeLimits(max_bytes=4_194_304)


def compact(value: dict[str, Any]) -> dict[str, Any]:
    """Share exact common fields in homogeneous lists without changing row order."""
    encode_json(value, LIMITS)

    def visit(item: Any) -> Any:
        if type(item) is dict:
            if TABLE in item:
                raise ValueError("SRE encoding reserved marker collision")
            return {key: visit(child) for key, child in item.items()}
        if type(item) in (list, tuple):
            rows = [visit(child) for child in item]
            if len(rows) < 2 or any(type(row) is not dict or TABLE in row for row in rows):
                return rows
            first = rows[0]
            if any(set(row) != set(first) for row in rows):
                return rows
            common = {key: first[key] for key in sorted(first) if all(
                canonical_json_bytes(row[key]) == canonical_json_bytes(first[key])
                for row in rows[1:])}
            columns = [key for key in sorted(first) if key not in common]
            table = {TABLE: dict(common=common, columns=columns,
                                 rows=[[row[key] for key in columns] for row in rows])}
            return table if len(canonical_json_bytes(table)) < len(
                canonical_json_bytes(rows)) else rows
        return item

    result: dict[str, Any] = visit(value)
    encode_json(result, LIMITS)
    return result


def expand(value: dict[str, Any]) -> dict[str, Any]:
    """Validate before/during expansion, including common-cell multiplication."""
    encode_json(value, LIMITS)
    spent = 0

    def charge(count: int = 1) -> None:
        nonlocal spent
        spent += count
        if spent > LIMITS.max_nodes:
            raise ExchangeBoundaryError("json_limit", boundary="expanded_nodes",
                observed=spent, limit=LIMITS.max_nodes, observed_is_lower_bound=True)

    def visit(item: Any, depth: int) -> Any:
        charge()
        if type(item) in (dict, list, tuple) and depth >= LIMITS.max_depth:
            raise ExchangeBoundaryError("json_limit", boundary="expanded_depth",
                observed=depth + 1, limit=LIMITS.max_depth, observed_is_lower_bound=True)
        if type(item) is dict:
            if TABLE not in item:
                charge(len(item))  # Dictionary keys are nodes too.
                return {key: visit(child, depth + 1) for key, child in item.items()}
            table = item[TABLE]
            if set(item) != {TABLE} or type(table) is not dict or set(table) != {
                "common", "columns", "rows",
            }:
                raise ValueError("SRE table shape differs")
            common, columns, rows = table["common"], table["columns"], table["rows"]
            if type(common) is not dict or type(columns) is not list or (
                type(rows) is not list or not rows
                or any(type(key) is not str for key in columns)
                or len(set(columns)) != len(columns) or set(common) & set(columns)
                or TABLE in common or TABLE in columns
            ):
                raise ValueError("SRE table columns differ")
            result = []
            for cells in rows:
                if type(cells) is not list or len(cells) != len(columns):
                    raise ValueError("SRE table row differs")
                # Build at most one bounded row before the next budget check.
                result.append(visit({**common, **dict(zip(columns, cells, strict=True))},
                                    depth + 1))
            return result
        if type(item) in (list, tuple):
            return [visit(child, depth + 1) for child in item]
        return item

    result: dict[str, Any] = visit(value, 0)
    encode_json(result, LIMITS)
    return result


def serialize_sre_view(value: dict[str, Any], *, diagnose: bool = False) -> str:
    from nurse_rcl.v2.live.ops_card_view import serialize_tool_view

    try:
        expand(value)
        raw = serialize_tool_view(value, soft_targets=True, diagnose=diagnose)
        if value.get("soft_target_basis") == "encoded_page_records_only":
            projected = json.loads(raw)
            if projected.get("status") != "resource_limit":
                exceeded = value["page_body_bytes"] > 4096
                projected.update(target_exceeded=exceeded,
                    expansion_reasons=["indivisible_page_body"] if exceeded else [])
                raw = encode_json(projected, LIMITS).decode()
        expand(json.loads(raw))  # Include actual projection metadata and ledger fields.
        return raw
    except ExchangeBoundaryError as exc:
        if exc.code != "json_limit":
            raise
        details: dict[str, Any] = {}
        if diagnose:
            from nurse_rcl.v2.live.ops_failure import failure_detail

            details["failure_detail"] = failure_detail(
                "investigation_tool", exc).model_dump(mode="json")
        value = dict(schema_version="nurse.v2.ops-tool-view/2", status="resource_limit",
            scope="evidence_view_resource", card_id=value.get("card_id"),
            case_snapshot_id=value.get("case_snapshot_id"),
            action_index=value.get("action_index"), origin=value.get("origin"),
            role="supplementary", score_authority=False,
            limitations=["full evidence retained; expanded JSON resource boundary exceeded",
                         "not observation absence or normality"], **details)
    return serialize_tool_view(value, soft_targets=True)


def compact_record(value: dict[str, Any]) -> dict[str, Any]:
    """Keep oversized owner evidence for the serializer's capacity outcome.

    This is not a resource-limit bypass: the caller must serialize_sre_view,
    which emits only a small resource_limit receipt if the original cannot fit.
    """
    try:
        return compact(value)
    except ExchangeBoundaryError as exc:
        if exc.code != "json_limit":
            raise
        return value
