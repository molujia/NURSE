"""Fixed report-only resource partitions; never changes transport JSON limits."""

from __future__ import annotations

import math
from typing import Any, cast

from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, ExchangeLimits, _string_size
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS

METADATA_LIMITS = ExchangeLimits(max_bytes=4_194_304, max_depth=64, max_nodes=20_000)


class ReportPartitionError(ExchangeBoundaryError):
    """Numeric diagnostics only; observed can be a proved lower bound."""

    def __init__(self, boundary: str, observed: int, limit: int) -> None:
        # Existing projection callers classify this stable code as resource_limit.
        super().__init__("json_limit", boundary=boundary, observed=observed,
                         limit=limit, observed_is_lower_bound=True)


def _check(boundary: str, observed: int, limit: int) -> None:
    if observed > limit:
        raise ReportPartitionError(boundary, observed, limit)


def _measure(value: object, *, node_limit: int, boundary: str) -> tuple[int, int, int]:
    stack = [(value, 0, False)]
    active: set[int] = set()
    nodes, size, deepest = 1, 0, 0
    while stack:
        item, depth, leaving = stack.pop()
        if leaving:
            active.remove(id(item))
            continue
        kind = type(item)
        if kind is str:
            try:
                size += _string_size(cast(str, item), REPORT_LIMITS.max_bytes - size)
            except ExchangeBoundaryError as exc:
                if exc.code == "json_limit":
                    raise ReportPartitionError("logical_bytes", REPORT_LIMITS.max_bytes + 1,
                                               REPORT_LIMITS.max_bytes) from None
                raise
        elif kind is float:
            if not math.isfinite(cast(float, item)):
                raise ExchangeBoundaryError("json_non_finite")
            size += len(repr(item))
        elif kind is int:
            if not -(10**30) < cast(int, item) < 10**30:
                raise ExchangeBoundaryError("json_limit")
            size += len(str(item))
        elif item is None or kind is bool:
            size += 5 if item is False else 4
        elif kind is dict or kind is list:
            deepest = max(deepest, depth + 1)
            _check("logical_depth", deepest, REPORT_LIMITS.max_depth)
            if id(item) in active:
                raise ExchangeBoundaryError("json_cycle")
            length = len(cast(Any, item))
            nodes += length * (2 if kind is dict else 1)
            _check(boundary, nodes, node_limit)  # Before scheduling any children.
            size += 2 + max(0, length - 1) + (length if kind is dict else 0)
            _check("logical_bytes", size, REPORT_LIMITS.max_bytes)
            active.add(id(item))
            stack.append((item, depth, True))
            if kind is dict:
                for key, child in cast(dict[object, object], item).items():
                    if type(key) is not str:
                        raise ExchangeBoundaryError("json_type")
                    stack.extend(((key, depth + 1, False), (child, depth + 1, False)))
            else:
                stack.extend((child, depth + 1, False) for child in cast(list[Any], item))
        else:
            raise ExchangeBoundaryError("json_type")
        _check("logical_bytes", size, REPORT_LIMITS.max_bytes)
    return nodes, size, deepest


def validate_report_parts(
    metadata: dict[str, Any], facts: list[Any], *, layout: str,
) -> dict[str, int]:
    """Count repeated values by expansion; accept only the two fixed layouts."""
    if type(metadata) is not dict or type(facts) is not list or layout not in {"context", "view"}:
        raise ValueError("SRE report partition shape differs")
    parent = metadata.get("evidence") if layout == "context" else metadata
    if type(parent) is not dict or "facts" in parent:
        raise ValueError("SRE report fact position differs")
    mn, mb, md = _measure(metadata, node_limit=20_000, boundary="metadata_nodes")
    fn, fb, fd = _measure(facts, node_limit=100_000, boundary="facts_nodes")
    size = mb + len(b'"facts":') + fb + int(bool(parent))
    depth = max(md, fd + (2 if layout == "context" else 1))
    _check("logical_bytes", size, REPORT_LIMITS.max_bytes)
    _check("logical_depth", depth, REPORT_LIMITS.max_depth)
    return dict(fact_nodes=fn, metadata_nodes=mn, logical_bytes=size, logical_depth=depth)
