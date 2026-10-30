"""Disk-only heterogeneous prefix tables; source descriptors remain version one."""

from __future__ import annotations

from typing import Any

from nurse_rcl.v2.live import ops_sre_checkpoint as legacy
from nurse_rcl.v2.live import ops_sre_source_checkpoint as limits
from nurse_rcl.v2.live.ops_sre_source_checkpoint import ScoreSource

STORAGE_ENCODING = "sre-checkpoint-tables/2"
SOURCE_ENCODING = "sre-checkpoint-score-source/2"
COMPONENT_BYTES = legacy.COMPONENT_BYTES


def pack_grouped_navigation(value: dict[str, Any]) -> dict[str, Any]:
    prefixes = []
    for key, prefix in sorted(value["inspection_prefixes"].items()):
        if "prefix_sha256" in prefix:
            raise ValueError("checkpoint prefix identity field collision")
        prefixes.append(dict(prefix_sha256=key, **prefix))
    legacy._capacity(prefixes, COMPONENT_BYTES)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in prefixes:
        groups.setdefault(tuple(sorted(row)), []).append(row)
    base = legacy.pack_navigation(dict(value, inspection_prefixes={}))
    return dict(
        base,
        storage_encoding=STORAGE_ENCODING,
        inspection_prefixes=[
            dict(layout=list(layout), table=legacy._table(rows))
            for layout, rows in sorted(groups.items())
        ],
    )


def unpack_grouped_navigation(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("storage_encoding") != STORAGE_ENCODING:
        raise ValueError("checkpoint storage encoding differs")
    groups = value["inspection_prefixes"]
    if type(groups) is not list:
        raise ValueError("checkpoint prefix groups differ")
    layouts: set[tuple[str, ...]] = set()
    sizes: list[int] = []
    # Complete preflight before any shared cells are expanded, across ALL groups.
    for group in groups:
        if type(group) is not dict or set(group) != {"layout", "table"}:
            raise ValueError("checkpoint prefix group shape differs")
        layout = group["layout"]
        if (
            type(layout) is not list
            or any(type(k) is not str for k in layout)
            or (layout != sorted(set(layout)) or "prefix_sha256" not in layout)
        ):
            raise ValueError("checkpoint prefix layout differs")
        identity = tuple(layout)
        if identity in layouts:
            raise ValueError("checkpoint prefix layout duplicated")
        layouts.add(identity)
        table = group["table"]
        row_sizes = legacy._table_sizes(table, remaining=COMPONENT_BYTES)
        if not row_sizes or set(layout) != set(table["common"]) | set(table["columns"]):
            raise ValueError("checkpoint prefix layout differs")
        sizes.extend(row_sizes)
        if 2 + sum(sizes) + max(0, len(sizes) - 1) > COMPONENT_BYTES:
            raise ValueError("checkpoint expanded table exceeds capacity")
    if 2 + sum(sizes) + max(0, len(sizes) - 1) > COMPONENT_BYTES:
        raise ValueError("checkpoint expanded table exceeds capacity")
    prefixes = {}
    for group in groups:
        rows, _ = legacy._rows(group["table"], remaining=COMPONENT_BYTES)
        for row in rows:
            key = row.pop("prefix_sha256")
            if type(key) is not str or key in prefixes:
                raise ValueError("checkpoint prefix reference differs")
            prefixes[key] = row
    base = legacy.unpack_navigation(
        dict(value, storage_encoding=legacy.STORAGE_ENCODING, inspection_prefixes=legacy._table([]))
    )
    return dict(base, inspection_prefixes=prefixes)


def pack_grouped_source_navigation(
    value: dict[str, Any],
    *,
    source: ScoreSource | None,
) -> dict[str, Any]:
    limits.require_capacity(
        len(source.descriptor_json) if source else 4,
        limits.DESCRIPTOR_BYTES,
        "checkpoint_descriptor_bytes",
    )
    limits._check_regions(value, source)
    return dict(
        storage_encoding=SOURCE_ENCODING,
        source=source.descriptor if source else None,
        base=pack_grouped_navigation(value),
    )


def validate_grouped_source_envelope(value: dict[str, Any]) -> bytes:
    if set(value) != {"storage_encoding", "source", "base"} or (
        value["storage_encoding"] != SOURCE_ENCODING
        or type(value["base"]) is not dict
        or value["base"].get("storage_encoding") != STORAGE_ENCODING
    ):
        raise ValueError("source checkpoint encoding differs")
    return limits.validate_source_envelope(dict(value, storage_encoding=limits.STORAGE_ENCODING))


def unpack_grouped_source_navigation(
    value: dict[str, Any],
    *,
    source: ScoreSource | None,
) -> dict[str, Any]:
    descriptor = validate_grouped_source_envelope(value)
    if descriptor != (source.descriptor_json if source else b"null"):
        raise ValueError("checkpoint source was not independently admitted")
    base = unpack_grouped_navigation(value["base"])
    limits._check_regions(base, source)
    if source:
        base["indexed"]["cards"].update(
            {k: c.model_dump(mode="json") for k, c in source.cards.items()}
        )
    return base
