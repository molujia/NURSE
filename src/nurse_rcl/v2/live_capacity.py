"""Explicit, finite capacity for the synchronous live B0 boundary only.

This module does not alter shared serialization defaults or grant source trust.
The context exists so Pydantic callbacks in that same invocation see its policy.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import cast

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex

LIVE_B0_CAPACITY_PROFILE = "live-b0-6m-256m/1"
LIVE_B0_MAX_NODES = 6_000_000
LIVE_B0_MAX_JSON_BYTES = 256 * 1024 * 1024
LIVE_B0_MAX_DEPTH = 64
LIVE_B0_MAX_STRING_LENGTH = 8192
LIVE_B0_CAPACITY_SHA256 = sha256_hex(canonical_json_bytes({
    "profile": LIVE_B0_CAPACITY_PROFILE,
    "max_nodes": LIVE_B0_MAX_NODES,
    "max_json_bytes": LIVE_B0_MAX_JSON_BYTES,
    "max_depth": LIVE_B0_MAX_DEPTH,
    "max_string_length": LIVE_B0_MAX_STRING_LENGTH,
}))
_ACTIVE: ContextVar[str | None] = ContextVar("live_b0_capacity", default=None)
_LIMITS = MappingProxyType({
    "live-b0-2m-96m/1": (2_000_000, 96 * 1024 * 1024),
    LIVE_B0_CAPACITY_PROFILE: (LIVE_B0_MAX_NODES, LIVE_B0_MAX_JSON_BYTES),
})


def active_binding_capacity() -> str | None:
    return _ACTIVE.get()


def active_binding_limits() -> tuple[int, int] | None:
    profile = _ACTIVE.get()
    return None if profile is None else _LIMITS[profile]


@contextmanager
def binding_capacity(profile: str | None) -> Iterator[None]:
    if profile is not None and (
        type(profile) is not str or profile not in _LIMITS
    ):
        raise ValueError("unregistered B0 capacity profile")
    token = _ACTIVE.set(profile)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def require_live_json_budget(value: object) -> None:
    """Count expanded canonical JSON bytes AFTER exact-tree shape preflight.

    Only encode one bounded scalar at a time, never the complete tree. Repeated
    references are counted on every occurrence, as in the actual wire encoding.
    Mapping order does not affect byte length. Shape/cycle/depth validation is
    the caller's prerequisite; no user serializer is invoked here.
    """
    limits = active_binding_limits()
    if limits is None:
        raise ValueError("live byte accounting requires an explicit capacity profile")
    max_bytes = limits[1]
    total = 0

    def add(count: int) -> None:
        nonlocal total
        total += count
        if total > max_bytes:
            raise ValueError("B0 rule binding exceeds the serialization byte budget")

    def visit(current: object) -> None:
        kind = type(current)
        if kind is float and not math.isfinite(cast(float, current)):
            raise ValueError("B0 rule binding requires finite values")
        if kind in (str, int, float, bool, type(None)):
            add(len(canonical_json_bytes(current)))
        elif kind in (tuple, list):
            items = cast(tuple[object, ...], current)
            add(2 + max(0, len(items) - 1))
            for child in items:
                visit(child)
        else:
            state = (current if kind is dict
                     else object.__getattribute__(current, "__dict__"))
            mapping = cast(dict[str | int, object], state)
            add(2 + max(0, len(mapping) - 1))
            for key, child in mapping.items():
                add(len(canonical_json_bytes(key if type(key) is str else str(key))) + 1)
                visit(child)

    visit(value)
