"""Bounded, exact-type JSON at the model exchange boundary (no model callbacks)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, cast


class ExchangeBoundaryError(ValueError):
    """A static error code; never include provider or prompt text in the message."""

    def __init__(
        self, code: str, *, boundary: str | None = None,
        observed: int | None = None, limit: int | None = None,
        observed_is_lower_bound: bool | None = None,
    ) -> None:
        self.code = code
        # Optional numeric diagnostics do not change the error type/code or
        # acceptance boundary. Never include input values in the message.
        self.boundary = boundary
        self.observed, self.limit = observed, limit
        self.observed_is_lower_bound = observed_is_lower_bound
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ExchangeLimits:
    max_bytes: int = 262_144
    max_depth: int = 32
    max_nodes: int = 20_000

    def validate(self) -> None:
        for value, ceiling in (
            (self.max_bytes, 4_194_304),
            (self.max_depth, 64),
            (self.max_nodes, 100_000),
        ):
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ExchangeBoundaryError("invalid_limits")

    def __post_init__(self) -> None:
        self.validate()


def require_limits(limits: ExchangeLimits) -> None:
    if type(limits) is not ExchangeLimits:
        raise ExchangeBoundaryError("invalid_limits")
    limits.validate()


def _string_size(value: str, remaining: int) -> int:
    """Count compact ensure_ascii=False JSON bytes without allocating the text."""
    size = 2  # Quotes; each input code point needs at least one output byte.
    if len(value) + size > remaining:
        raise ExchangeBoundaryError("json_limit", boundary="json_bytes",
            observed=len(value) + size, limit=remaining, observed_is_lower_bound=True)
    for character in value:
        point = ord(character)
        if character in '"\\\b\f\n\r\t':
            size += 2
        elif point < 0x20:
            size += 6
        elif point < 0x80:
            size += 1
        elif point < 0x800:
            size += 2
        elif 0xD800 <= point <= 0xDFFF:
            raise ExchangeBoundaryError("json_encoding")
        elif point < 0x10000:
            size += 3
        else:
            size += 4
        if size > remaining:
            raise ExchangeBoundaryError("json_limit", boundary="json_bytes",
                observed=size, limit=remaining, observed_is_lower_bound=True)
    return size


def _preflight(value: object, limits: ExchangeLimits) -> None:
    stack = [(value, 0, False)]
    active: set[int] = set()
    scheduled = 1
    size = 0
    while stack:
        item, depth, leaving = stack.pop()
        if leaving:
            active.remove(id(item))
            continue
        kind = type(item)
        if kind is str:
            try:
                size += _string_size(cast(str, item), limits.max_bytes - size)
            except ExchangeBoundaryError as exc:
                if exc.code == "json_limit" and exc.observed is not None:
                    raise ExchangeBoundaryError("json_limit", boundary="json_bytes",
                        observed=size + exc.observed, limit=limits.max_bytes,
                        observed_is_lower_bound=True) from None
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
        elif kind is dict or kind is list or kind is tuple:
            # depth counts enclosing containers; even an empty container adds
            # one level, matching the decoder's lexical bracket count.
            if depth >= limits.max_depth:
                raise ExchangeBoundaryError("json_limit", boundary="json_depth",
                    observed=depth + 1, limit=limits.max_depth, observed_is_lower_bound=True)
            if id(item) in active:
                raise ExchangeBoundaryError("json_cycle")
            length = len(cast(Any, item))
            count = length * (2 if kind is dict else 1)
            scheduled += count
            size += 2 + max(0, length - 1) + (length if kind is dict else 0)
            if scheduled > limits.max_nodes:
                raise ExchangeBoundaryError("json_limit", boundary="json_nodes",
                    observed=scheduled, limit=limits.max_nodes, observed_is_lower_bound=True)
            if size > limits.max_bytes:
                raise ExchangeBoundaryError("json_limit", boundary="json_bytes",
                    observed=size, limit=limits.max_bytes, observed_is_lower_bound=True)
            active.add(id(item))
            stack.append((item, depth, True))
            if kind is dict:
                for key, child in cast(dict[object, object], item).items():
                    if type(key) is not str:
                        raise ExchangeBoundaryError("json_type")
                    stack.extend(((key, depth + 1, False), (child, depth + 1, False)))
            else:
                stack.extend((child, depth + 1, False) for child in cast(Any, item))
        else:
            raise ExchangeBoundaryError("json_type")
        if size > limits.max_bytes:
            raise ExchangeBoundaryError("json_limit", boundary="json_bytes",
                observed=size, limit=limits.max_bytes, observed_is_lower_bound=True)


def encode_json(value: object, limits: ExchangeLimits) -> bytes:
    require_limits(limits)
    _preflight(value, limits)
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (UnicodeError, ValueError, RecursionError):
        raise ExchangeBoundaryError("json_encoding") from None
    if len(encoded) > limits.max_bytes:
        raise ExchangeBoundaryError("json_limit", boundary="json_bytes",
            observed=len(encoded), limit=limits.max_bytes, observed_is_lower_bound=False)
    return encoded


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ExchangeBoundaryError("json_duplicate_key")
        result[key] = value
    return result


def _integer(value: str) -> int:
    if len(value.lstrip("-")) > 30:
        raise ExchangeBoundaryError("json_limit")
    return int(value)


def _nonfinite(_: str) -> Any:
    raise ExchangeBoundaryError("json_non_finite")


def decode_json_object(raw: bytes, limits: ExchangeLimits) -> dict[str, Any]:
    require_limits(limits)
    if type(raw) is not bytes:
        raise ExchangeBoundaryError("json_type")
    if len(raw) > limits.max_bytes:
        raise ExchangeBoundaryError("json_limit", boundary="json_bytes",
            observed=len(raw), limit=limits.max_bytes, observed_is_lower_bound=False)
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError:
        raise ExchangeBoundaryError("json_encoding") from None
    # Reject excessive nesting before the recursive stdlib parser is entered.
    quoted = escaped = False
    depth = 0
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            if depth > limits.max_depth:
                raise ExchangeBoundaryError("json_limit", boundary="json_depth",
                    observed=depth, limit=limits.max_depth, observed_is_lower_bound=True)
        elif character in "]}":
            depth -= 1
    try:
        value = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_nonfinite, parse_int=_integer
        )
    except (json.JSONDecodeError, RecursionError):
        raise ExchangeBoundaryError("json_syntax") from None
    if type(value) is not dict:
        raise ExchangeBoundaryError("json_object_required")
    _preflight(value, limits)
    # Also reject lone escaped surrogates before downstream normalization.
    encode_json(value, limits)
    return cast(dict[str, Any], value)
