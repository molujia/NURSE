"""Private bounded projection after whole-tree exact preflight.

No JSON is allocated until the complete byte-checked projection succeeds.
Only immutable string lengths are reused within this invocation; every
occurrence still consumes its full expanded byte budget.
"""

from __future__ import annotations

import math
from typing import cast

from pydantic import BaseModel

from nurse_rcl.v2.dynamic_verification.safety import exact_field

_BYTE_LIMIT = 256 * 1024 * 1024
_ATOM_LIMIT = 256
_SHORT_ESCAPES = "\b\f\n\r\t"


def _project_preflighted(value: object) -> object:
    # Caller must first run the complete type/slot/cycle/node/depth/provenance
    # preflight. This helper is not a replacement for that public boundary.
    total = 0
    string_sizes: dict[str, int] = {}

    def add(size: int) -> None:
        nonlocal total
        total += size
        if total > _BYTE_LIMIT:
            raise ValueError("report audit exceeds byte limit")

    def string(text: str) -> None:
        known = string_sizes.get(text)
        if known is not None:
            add(known)
            return
        size = 2
        for start in range(0, len(text), 4096):
            piece = text[start : start + 4096]
            try:
                size += len(piece) if piece.isascii() else len(piece.encode("utf-8"))
            except UnicodeError:
                raise ValueError("report audit contains invalid Unicode") from None
            size += piece.count('"') + piece.count("\\")
            if not piece.isprintable():
                for code in range(32):
                    char = chr(code)
                    size += piece.count(char) * (1 if char in _SHORT_ESCAPES else 5)
        add(size)
        if len(string_sizes) < 512:
            string_sizes[text] = size

    def walk(item: object) -> object:
        kind = type(item)
        if kind is str:
            string(cast(str, item))
            return item
        if item is None:
            add(4)
            return item
        if kind is bool:
            add(4 if item else 5)
            return item
        if kind is int or kind is float:
            if kind is int:
                if int.bit_length(cast(int, item)) > _ATOM_LIMIT * 4:
                    raise ValueError("report audit scalar exceeds atom limit")
                atom = str(item)
            else:
                if not math.isfinite(cast(float, item)):
                    raise ValueError("report audit contains a nonfinite scalar")
                atom = repr(item)
            if len(atom) > _ATOM_LIMIT:
                raise ValueError("report audit scalar exceeds atom limit")
            add(len(atom))
            return item
        if kind is tuple or kind is list:
            sequence = cast(tuple[object, ...] | list[object], item)
            add(2 + max(0, len(sequence) - 1))
            return [walk(child) for child in sequence]
        if kind is dict:
            mapping = cast(dict[str, object], item)
        else:
            mapping = cast(dict[str, object], object.__getattribute__(item, "__dict__"))
            fields = cast(type[BaseModel], kind).model_fields
            if any(
                not exact_field(child, fields[key].annotation) for key, child in mapping.items()
            ):
                raise ValueError("dynamic source field requires its exact declared type")
        add(2 + max(0, len(mapping) - 1) + len(mapping))
        result: dict[str, object] = {}
        for key, child in mapping.items():
            if type(key) is not str:
                raise ValueError("report audit requires exact string mapping keys")
            string(key)
            result[key] = walk(child)
        return result

    return walk(value)
