"""One-level, disk-only column dictionaries for fixed checkpoint tables."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live import ops_sre_checkpoint as legacy
from nurse_rcl.v2.live import ops_sre_checkpoint_grouped as grouped
from nurse_rcl.v2.live import ops_sre_source_checkpoint as limits
from nurse_rcl.v2.live.ops_sre_source_checkpoint import ScoreSource

STORAGE_ENCODING = "sre-checkpoint-tables/3"
SOURCE_ENCODING = "sre-checkpoint-score-source/3"
COMPONENT_BYTES = legacy.COMPONENT_BYTES


def _pool_table(table: dict[str, Any]) -> dict[str, Any]:
    """Pool only profitable exact column values, without rewriting payloads."""
    plain = dict(table, pools={})
    encoded_rows = [list(row) for row in table["rows"]]
    selected_pools = {}
    for position, key in enumerate(table["columns"]):
        lookup: dict[bytes, int] = {}
        values: list[Any] = []
        references = []
        original_cost = 0
        for row in table["rows"]:
            value = row[position]
            raw = canonical_json_bytes(value)
            original_cost += len(raw)
            if raw not in lookup:
                lookup[raw] = len(values)
                values.append(value)
            references.append(lookup[raw])
        # Charge the entire value array, key, colon, and entry separator.
        pooled_cost = sum(len(str(i)) for i in references)
        pooled_cost += len(canonical_json_bytes(values)) + len(canonical_json_bytes(key)) + 2
        if original_cost > pooled_cost:
            selected_pools[key] = values
            for row, reference in zip(encoded_rows, references, strict=True):
                row[position] = reference
    candidate = dict(table, rows=encoded_rows, pools=selected_pools)
    return (
        candidate if len(canonical_json_bytes(candidate)) < len(canonical_json_bytes(plain))
        else plain
    )


def _pooled_table_sizes(table: Any, *, remaining: int) -> list[int]:
    """Validate and count EVERY expanded occurrence before copying shared cells."""
    if type(table) is not dict or set(table) != {"common", "columns", "rows", "pools"}:
        raise ValueError("checkpoint pooled table shape differs")
    common, columns, cells, pools = (table[k] for k in ("common", "columns", "rows", "pools"))
    if type(common) is not dict or type(columns) is not list or type(cells) is not list or (
        type(pools) is not dict
        or any(type(k) is not str for k in [*common, *columns])
        or len(set(columns)) != len(columns) or set(common) & set(columns)
        or any(type(row) is not list or len(row) != len(columns) for row in cells)
        or (not cells and (common or columns or pools))
    ):
        raise ValueError("checkpoint pooled table fields differ")
    if not set(pools) <= set(columns):
        raise ValueError("checkpoint pool column differs")
    pool_sizes: dict[str, list[int]] = {}
    used: dict[str, set[int]] = {}
    for key, values in pools.items():
        if type(values) is not list or not values:
            raise ValueError("checkpoint pool values differ")
        raw_values = [canonical_json_bytes(value) for value in values]
        if len(set(raw_values)) != len(raw_values):
            raise ValueError("checkpoint pool value duplicated")
        pool_sizes[key] = [len(raw) for raw in raw_values]
        used[key] = set()
    keys = [*common, *columns]
    overhead = 2 + max(0, len(keys) - 1) + len(keys)
    overhead += sum(len(canonical_json_bytes(k)) for k in keys)
    overhead += sum(len(canonical_json_bytes(v)) for v in common.values())
    total = 2 + max(0, len(cells) - 1)
    sizes = []
    if total > remaining:
        raise ValueError("checkpoint expanded table exceeds capacity")
    for row in cells:
        size = overhead
        for key, value in zip(columns, row, strict=True):
            if key in pools:
                if type(value) is not int or not 0 <= value < len(pools[key]):
                    raise ValueError("checkpoint pool reference differs")
                used[key].add(value)
                size += pool_sizes[key][value]
            else:
                size += len(canonical_json_bytes(value))
        total += size
        if total > remaining:
            raise ValueError("checkpoint expanded table exceeds capacity")
        sizes.append(size)
    if any(used[key] != set(range(len(values))) for key, values in pools.items()):
        raise ValueError("checkpoint pool has unused values")
    return sizes


def _pooled_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    """Internal expansion, ONLY after aggregate preflight in the caller."""
    common, columns, cells, pools = (table[k] for k in ("common", "columns", "rows", "pools"))
    return [dict(deepcopy(common), **{
        key: deepcopy(pools[key][value] if key in pools else value)
        for key, value in zip(columns, row, strict=True)
    }) for row in cells]


def pack_pooled_navigation(value: dict[str, Any]) -> dict[str, Any]:
    base = grouped.pack_grouped_navigation(value)
    indexed = base["indexed"]
    return dict(base, storage_encoding=STORAGE_ENCODING,
        selection_rows=_pool_table(base["selection_rows"]),
        indexed=dict(indexed,
            cards={k: _pool_table(t) for k, t in indexed["cards"].items()},
            index=dict(indexed["index"], rows={
                k: _pool_table(t) for k, t in indexed["index"]["rows"].items()})))


def unpack_pooled_navigation(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("storage_encoding") != STORAGE_ENCODING:
        raise ValueError("checkpoint storage encoding differs")
    indexed = value["indexed"]
    # Each original partition accumulates across all its tables, never per pool.
    for tables in (indexed["cards"], indexed["index"]["rows"]):
        remaining = COMPONENT_BYTES
        for table in tables.values():
            sizes = _pooled_table_sizes(table, remaining=remaining)
            remaining -= sum(sizes) + max(0, len(sizes) - 1) + 2
    unique_sizes = _pooled_table_sizes(value["selection_rows"], remaining=COMPONENT_BYTES)
    if len(indexed["core_steps"]) > 8:
        raise ValueError("checkpoint core step count exceeds capacity")
    used = set()
    remaining = COMPONENT_BYTES
    for step in indexed["core_steps"]:
        references = step["selection"]["requests"]
        if type(references) is not list or any(
            type(i) is not int or not 0 <= i < len(unique_sizes) for i in references
        ):
            raise ValueError("checkpoint selection reference differs")
        remaining -= sum(unique_sizes[i] for i in references) + max(0, len(references) - 1) + 2
        if remaining < 0:
            raise ValueError("checkpoint selection expansion exceeds capacity")
        used.update(references)
    if used != set(range(len(unique_sizes))):
        raise ValueError("checkpoint selection has unused rows")
    # Reuse /2 solely for prefixes and untouched fields, never give it a pool.
    base = grouped.unpack_grouped_navigation(dict(value,
        storage_encoding=grouped.STORAGE_ENCODING, selection_rows=legacy._table([]),
        indexed=dict(indexed, cards={}, core_steps=[],
            index=dict(indexed["index"], rows={}))))
    cards = {}
    for tool, table in indexed["cards"].items():
        for row in _pooled_rows(table):
            key = row.pop("card_id", None)
            if type(key) is not str or key in cards or row.get("tool_version") != tool:
                raise ValueError("checkpoint card reference differs")
            cards[key] = row
    unique = _pooled_rows(value["selection_rows"])
    steps = [dict(step, selection=dict(step["selection"], requests=[
        deepcopy(unique[i]) for i in step["selection"]["requests"]
    ])) for step in indexed["core_steps"]]
    directories = {k: _pooled_rows(t) for k, t in indexed["index"]["rows"].items()}
    return dict(base, indexed=dict(base["indexed"], cards=cards, core_steps=steps,
        index=dict(base["indexed"]["index"], rows=directories)))


def pack_pooled_source_navigation(
    value: dict[str, Any], *, source: ScoreSource | None,
) -> dict[str, Any]:
    limits.require_capacity(len(source.descriptor_json) if source else 4,
                            limits.DESCRIPTOR_BYTES, "checkpoint_descriptor_bytes")
    limits._check_regions(value, source)
    return dict(storage_encoding=SOURCE_ENCODING, source=source.descriptor if source else None,
                base=pack_pooled_navigation(value))


def validate_pooled_source_envelope(value: dict[str, Any]) -> bytes:
    if set(value) != {"storage_encoding", "source", "base"} or (
        value["storage_encoding"] != SOURCE_ENCODING
        or type(value["base"]) is not dict
        or value["base"].get("storage_encoding") != STORAGE_ENCODING
    ):
        raise ValueError("source checkpoint encoding differs")
    return limits.validate_source_envelope(dict(value, storage_encoding=limits.STORAGE_ENCODING))


def unpack_pooled_source_navigation(
    value: dict[str, Any], *, source: ScoreSource | None,
) -> dict[str, Any]:
    descriptor = validate_pooled_source_envelope(value)
    if descriptor != (source.descriptor_json if source else b"null"):
        raise ValueError("checkpoint source was not independently admitted")
    base = unpack_pooled_navigation(value["base"])
    limits._check_regions(base, source)
    if source:
        base["indexed"]["cards"].update(
            {k: c.model_dump(mode="json") for k, c in source.cards.items()})
    return base
