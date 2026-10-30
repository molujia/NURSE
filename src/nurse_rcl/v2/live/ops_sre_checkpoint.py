"""Lossless, disk-only tables for known SRE navigation checkpoint regions.

No compression, recursive references, external paths or new evidence authority.
Core selection rows are shared by exact JSON bytes; cards share field layouts
and common cells. Model views and the toolbox's admission checks are unchanged.
"""

from __future__ import annotations

from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes

STORAGE_ENCODING = "sre-checkpoint-tables/1"
COMPONENT_BYTES = 64 * 1024 * 1024


def _capacity(rows: list[dict[str, Any]], remaining: int) -> int:
    remaining -= 2 + max(0, len(rows) - 1)
    for row in rows:
        remaining -= len(canonical_json_bytes(row))
        if remaining < 0:
            raise ValueError("checkpoint expanded table exceeds capacity")
    if remaining < 0:
        raise ValueError("checkpoint expanded table exceeds capacity")
    return remaining


def _table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return dict(common={}, columns=[], rows=[])
    keys = sorted(rows[0])
    if any(set(row) != set(keys) for row in rows):
        raise ValueError("checkpoint table field layout differs")
    common = {key: rows[0][key] for key in keys if all(
        canonical_json_bytes(row[key]) == canonical_json_bytes(rows[0][key])
        for row in rows[1:])}
    columns = [key for key in keys if key not in common]
    return dict(common=common, columns=columns,
                rows=[[row[key] for key in columns] for row in rows])


def _table_sizes(table: Any, *, remaining: int) -> list[int]:
    """Check the expanded byte cost before copying any shared common cells."""
    if type(table) is not dict or set(table) != {"common", "columns", "rows"}:
        raise ValueError("checkpoint table shape differs")
    common, columns, cells = (table[k] for k in ("common", "columns", "rows"))
    if type(common) is not dict or type(columns) is not list or type(cells) is not list or (
        any(type(k) is not str for k in [*common, *columns])
        or len(set(columns)) != len(columns) or set(common) & set(columns)
        or any(type(row) is not list or len(row) != len(columns) for row in cells)
        or (not cells and (common or columns))
    ):
        raise ValueError("checkpoint table fields differ")
    keys = [*common, *columns]
    overhead = 2 + max(0, len(keys) - 1) + len(keys)
    overhead += sum(len(canonical_json_bytes(k)) for k in keys)
    overhead += sum(len(canonical_json_bytes(v)) for v in common.values())
    sizes = []
    total = 2 + max(0, len(cells) - 1)
    if total > remaining:
        raise ValueError("checkpoint expanded table exceeds capacity")
    for row in cells:
        size = overhead + sum(len(canonical_json_bytes(v)) for v in row)
        total += size
        if total > remaining:
            raise ValueError("checkpoint expanded table exceeds capacity")
        sizes.append(size)
    return sizes


def _rows(table: Any, *, remaining: int) -> tuple[list[dict[str, Any]], list[int]]:
    sizes = _table_sizes(table, remaining=remaining)
    common, columns, cells = (table[k] for k in ("common", "columns", "rows"))
    return [{**common, **dict(zip(columns, row, strict=True))} for row in cells], sizes


def pack_navigation(value: dict[str, Any]) -> dict[str, Any]:
    """Only transform registered regions; retain every other field verbatim."""
    if "storage_encoding" in value or "selection_rows" in value:
        raise ValueError("checkpoint navigation is already encoded")
    indexed = value["indexed"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for key, card in sorted(indexed["cards"].items()):
        if "card_id" in card:
            raise ValueError("checkpoint card identity field collision")
        groups.setdefault(card["tool_version"], []).append(dict(card_id=key, **card))
    unique: list[dict[str, Any]] = []
    lookup: dict[bytes, int] = {}
    steps = []
    remaining = COMPONENT_BYTES
    if len(indexed["core_steps"]) > 8:
        raise ValueError("checkpoint core step count exceeds capacity")
    for step in indexed["core_steps"]:
        remaining = _capacity(step["selection"]["requests"], remaining)
        references = []
        for row in step["selection"]["requests"]:
            raw = canonical_json_bytes(row)
            if raw not in lookup:
                lookup[raw] = len(unique)
                unique.append(row)
            references.append(lookup[raw])
        steps.append(dict(step, selection=dict(step["selection"], requests=references)))
    prefixes = []
    for key, prefix in sorted(value["inspection_prefixes"].items()):
        if "prefix_sha256" in prefix:
            raise ValueError("checkpoint prefix identity field collision")
        prefixes.append(dict(prefix_sha256=key, **prefix))
    remaining = COMPONENT_BYTES
    for rows in groups.values():
        remaining = _capacity(rows, remaining)
    remaining = COMPONENT_BYTES
    for rows in indexed["index"]["rows"].values():
        remaining = _capacity(rows, remaining)
    _capacity(prefixes, COMPONENT_BYTES)
    return dict(value, storage_encoding=STORAGE_ENCODING, selection_rows=_table(unique),
        indexed=dict(indexed, cards={k: _table(rows) for k, rows in groups.items()},
            core_steps=steps, index=dict(indexed["index"], rows={
                k: _table(rows) for k, rows in indexed["index"]["rows"].items()})),
        inspection_prefixes=_table(prefixes))


def unpack_navigation(value: dict[str, Any]) -> dict[str, Any]:
    """Restore known tables, then let the existing toolbox validate all bindings."""
    if value.get("storage_encoding") != STORAGE_ENCODING:
        raise ValueError("checkpoint storage encoding differs")
    indexed = value["indexed"]
    cards: dict[str, Any] = {}
    remaining = COMPONENT_BYTES
    for tool, table in indexed["cards"].items():
        rows, sizes = _rows(table, remaining=remaining)
        remaining -= sum(sizes) + max(0, len(rows) - 1) + 2
        for row in rows:
            key = row.pop("card_id", None)
            if type(key) is not str or key in cards or row.get("tool_version") != tool:
                raise ValueError("checkpoint card reference differs")
            cards[key] = row
    unique, sizes = _rows(value["selection_rows"], remaining=COMPONENT_BYTES)
    if len(indexed["core_steps"]) > 8:
        raise ValueError("checkpoint core step count exceeds capacity")
    used = set()
    remaining = COMPONENT_BYTES
    steps = []
    for step in indexed["core_steps"]:
        references = step["selection"]["requests"]
        if type(references) is not list or any(type(i) is not int or not 0 <= i < len(unique)
                                              for i in references):
            raise ValueError("checkpoint selection reference differs")
        remaining -= sum(sizes[i] for i in references) + max(0, len(references) - 1) + 2
        if remaining < 0:
            raise ValueError("checkpoint selection expansion exceeds capacity")
        used.update(references)
        steps.append(dict(step, selection=dict(step["selection"],
            requests=[unique[i] for i in references])))
    if used != set(range(len(unique))):
        raise ValueError("checkpoint selection has unused rows")
    directories = {}
    remaining = COMPONENT_BYTES
    for kind, table in indexed["index"]["rows"].items():
        rows, sizes = _rows(table, remaining=remaining)
        remaining -= sum(sizes) + max(0, len(rows) - 1) + 2
        directories[kind] = rows
    rows, _ = _rows(value["inspection_prefixes"], remaining=COMPONENT_BYTES)
    prefixes = {}
    for row in rows:
        key = row.pop("prefix_sha256", None)
        if type(key) is not str or key in prefixes:
            raise ValueError("checkpoint prefix reference differs")
        prefixes[key] = row
    return dict({k: v for k, v in value.items() if k not in {
        "storage_encoding", "selection_rows"}}, inspection_prefixes=prefixes,
        indexed=dict(indexed, cards=cards, core_steps=steps,
                     index=dict(indexed["index"], rows=directories)))
