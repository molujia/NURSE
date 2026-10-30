"""Private Report receipt capacity, measured before serializers allocate JSON.

These limits describe the complete retained audit, independently of model wire
limits. Measurement neither grants evidence authority nor calls model properties.
"""

from __future__ import annotations

import math
from typing import cast

from pydantic import BaseModel

from nurse_rcl.v2.safe_serialization import (
    _declared_model_type_closure,
    preflight_exact_pydantic_tree,
)

MAX_REPORT_AUDIT_BYTES = 64 * 1024 * 1024
MAX_REPORT_AUDIT_DEPTH = 64
MAX_REPORT_AUDIT_NODES = 500_000
MAX_REPORT_AUDIT_ATOM_BYTES = 256
MAX_REPORT_AUDIT_STRING_CHARS = 1_048_576
_WHITESPACE = b" \t\r\n"
_TOKEN_END = b' \t\r\n,:{}[]"'
_SHORT_ESCAPES = "\b\f\n\r\t"


def preflight_json_tree(value: object, *, root_model_types: tuple[type[BaseModel], ...]) -> None:
    """Complete exact type/slot/cycle preflight before measuring any value."""
    preflight_exact_pydantic_tree(
        value,
        root_model_types=root_model_types,
        label="report audit",
        max_depth=MAX_REPORT_AUDIT_DEPTH,
        max_nodes=MAX_REPORT_AUDIT_NODES,
        max_string_length=MAX_REPORT_AUDIT_STRING_CHARS,
    )
    # The generic preflight also permits integer keys for other code-owned
    # schemas. Report canonical JSON permits only exact strings, before measure.
    allowed_models = _declared_model_type_closure(root_model_types)
    stack = [value]
    while stack:
        current = stack.pop()
        kind = type(current)
        if kind is dict:
            mapping = cast(dict[object, object], current)
            if any(type(key) is not str for key in mapping):
                raise ValueError("report audit requires exact string mapping keys")
            stack.extend(mapping.values())
        elif kind in allowed_models:
            stack.extend(object.__getattribute__(current, "__dict__").values())
        elif kind is list or kind is tuple:
            stack.extend(cast(list[object] | tuple[object, ...], current))


def _measure_preflighted(
    value: object,
    *,
    root_model_types: tuple[type[BaseModel], ...],
    max_bytes: int | None = None,
) -> int:
    """Measure a fully preflighted tree; callers must not expose this as a bypass."""
    byte_limit = MAX_REPORT_AUDIT_BYTES if max_bytes is None else max_bytes
    if type(byte_limit) is not int or byte_limit not in (MAX_REPORT_AUDIT_BYTES, 256 * 1024 * 1024):
        raise ValueError("unregistered private JSON byte capacity")
    allowed_models = _declared_model_type_closure(root_model_types)
    total = 0

    def add(count: int) -> None:
        nonlocal total
        total += count
        if total > byte_limit:
            raise ValueError("report audit exceeds byte limit")

    def string_size(text: str) -> None:
        add(2)
        # Bounded chunks avoid encoding an entire large string before the cap.
        for start in range(0, len(text), 4096):
            piece = text[start : start + 4096]
            try:
                size = (
                    len(piece) if piece.isascii() else len(piece.encode("utf-8", errors="strict"))
                )
            except UnicodeError:
                raise ValueError("report audit contains invalid Unicode") from None
            size += piece.count('"') + piece.count("\\")
            if not piece.isprintable():
                for code in range(32):
                    char = chr(code)
                    size += piece.count(char) * (1 if char in _SHORT_ESCAPES else 5)
            add(size)

    def atom_size(size: int) -> None:
        if size > MAX_REPORT_AUDIT_ATOM_BYTES:
            raise ValueError("report audit scalar exceeds atom limit")
        add(size)

    stack = [value]
    while stack:
        current = stack.pop()
        kind = type(current)
        if kind is str:
            string_size(cast(str, current))
        elif current is None:
            atom_size(4)
        elif kind is bool:
            atom_size(4 if current else 5)
        elif kind is int or kind is float:
            if kind is int:
                number = cast(int, current)
                # A decimal digit needs fewer than four binary bits. This loose
                # bound rejects enormous ints before str(), then checks exactly.
                if int.bit_length(number) > MAX_REPORT_AUDIT_ATOM_BYTES * 4:
                    raise ValueError("report audit scalar exceeds atom limit")
            elif not math.isfinite(cast(float, current)):
                raise ValueError("report audit contains a nonfinite scalar")
            atom = str(current) if kind is int else repr(current)
            atom_size(len(atom))
        elif kind is list or kind is tuple:
            sequence = cast(list[object] | tuple[object, ...], current)
            add(2 + max(0, len(sequence) - 1))
            stack.extend(reversed(sequence))
        else:
            if kind in allowed_models:
                mapping = cast(dict[str, object], object.__getattribute__(current, "__dict__"))
            elif kind is dict:
                mapping = cast(dict[str, object], current)
            else:
                raise ValueError("report audit contains an unsupported exact runtime type")
            add(2 + max(0, len(mapping) - 1) + len(mapping))
            for key in mapping:
                if type(key) is not str:
                    raise ValueError("report audit requires exact string mapping keys")
                string_size(key)
                stack.append(mapping[key])
    return total


def measure_canonical_json(value: object, *, root_model_types: tuple[type[BaseModel], ...]) -> int:
    """Return exact canonical UTF-8 bytes after a complete callback-free preflight."""
    preflight_json_tree(value, root_model_types=root_model_types)
    return _measure_preflighted(value, root_model_types=root_model_types)


def scan_audit_json(raw: bytes) -> None:
    """Bound lexical resources before either JSON parser; this is not a parser.

    Values have root depth zero; objects and models have identical depth. Object
    keys are not value nodes, and string punctuation/escaped quotes are opaque.
    """
    if type(raw) is not bytes or len(raw) > MAX_REPORT_AUDIT_BYTES:
        raise ValueError("report audit exceeds byte limit")
    if b"\x00" in raw or raw.startswith((b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff")):
        raise ValueError("report audit requires canonical UTF-8 JSON")
    depth = nodes = index = 0

    def value_node() -> None:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_REPORT_AUDIT_NODES or depth > MAX_REPORT_AUDIT_DEPTH:
            raise ValueError("report audit exceeds JSON resource limit")

    while index < len(raw):
        token = raw[index]
        if token == 34:
            index += 1
            escaped = False
            while index < len(raw):
                char = raw[index]
                index += 1
                if escaped:
                    escaped = False
                elif char == 92:
                    escaped = True
                elif char == 34:
                    break
            else:
                raise ValueError("invalid report audit JSON string")
            following = index
            while following < len(raw) and raw[following] in _WHITESPACE:
                following += 1
            if following == len(raw) or raw[following] != 58:
                value_node()
        elif token in (123, 91):
            value_node()
            depth += 1
            index += 1
        elif token in (125, 93):
            depth -= 1
            if depth < 0:
                raise ValueError("invalid report audit JSON nesting")
            index += 1
        elif token in _WHITESPACE or token in (44, 58):
            index += 1
        else:
            value_node()
            start = index
            while index < len(raw) and raw[index] not in _TOKEN_END:
                index += 1
                if index - start > MAX_REPORT_AUDIT_ATOM_BYTES:
                    raise ValueError("report audit scalar exceeds atom limit")
    if depth:
        raise ValueError("invalid report audit JSON nesting")
