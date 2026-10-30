"""Callback-free, exact-type snapshots for private dynamic verification inputs."""

from __future__ import annotations

import math
import re
from types import UnionType
from typing import Annotated, Literal, Union, cast, get_args, get_origin

from pydantic import BaseModel

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.report_audit_json import measure_canonical_json
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree


def exact_field(value: object, annotation: object, *, json_mode: bool = False) -> bool:
    """Check code-declared field shapes after bounded, callback-free preflight."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return exact_field(value, args[0], json_mode=json_mode)
    if origin is UnionType or origin is Union:
        return any(exact_field(value, item, json_mode=json_mode) for item in args)
    if origin is Literal:
        return any(type(value) is type(item) and value == item for item in args)
    if origin is tuple:
        if type(value) is not (list if json_mode else tuple):
            return False
        items = cast(tuple[object, ...], value)
        if len(args) == 2 and args[1] is Ellipsis:
            return all(exact_field(item, args[0], json_mode=json_mode) for item in items)
        return len(items) == len(args) and all(
            exact_field(item, expected, json_mode=json_mode)
            for item, expected in zip(items, args, strict=True)
        )
    if json_mode and isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if type(value) is not dict:
            return False
        fields = annotation.model_fields
        document = cast(dict[str, object], value)
        return all(
            name in fields and exact_field(item, fields[name].annotation, json_mode=True)
            for name, item in document.items()
        ) and all(name in document or not field.is_required() for name, field in fields.items())
    return type(value) is annotation


def plain(value: object) -> object:
    """Only call after a whole-tree exact preflight; never invoke serializers."""
    kind = type(value)
    if kind is float:
        if not math.isfinite(cast(float, value)):
            raise ValueError("dynamic source contains nonfinite values")
        return value
    if kind in (str, int, bool, type(None)):
        return value
    if kind in (tuple, list):
        return [plain(item) for item in cast(tuple[object, ...], value)]
    if kind is dict:
        return {key: plain(item) for key, item in cast(dict[str, object], value).items()}
    state = cast(dict[str, object], object.__getattribute__(value, "__dict__"))
    fields = cast(type[BaseModel], kind).model_fields
    if any(not exact_field(item, fields[key].annotation) for key, item in state.items()):
        raise ValueError("dynamic source field requires its exact declared type")
    return {key: plain(item) for key, item in state.items()}


def preflight(value: object, roots: tuple[type[BaseModel], ...]) -> None:
    preflight_exact_pydantic_tree(value, root_model_types=roots, label="dynamic source")
    # The existing measurer is callback-free, counts UTF-8 before allocation,
    # and applies the complete audit's 64 MiB cap and bounded numeric atoms.
    measure_canonical_json(value, root_model_types=roots)
    plain(value)


def canonical(value: object, roots: tuple[type[BaseModel], ...]) -> bytes:
    preflight(value, roots)
    return canonical_json_bytes(plain(value))


def require_pin(pin: object) -> str:
    if type(pin) is not str or re.fullmatch(r"[0-9a-f]{64}", pin) is None:
        raise ValueError("dynamic source requires an exact lowercase SHA256 pin")
    return pin


def representation(value: object) -> object:
    """Preserve container kinds, model slots and field markers, not object IDs."""
    kind = type(value)
    if kind in (str, int, float, bool, type(None)):
        return (kind.__name__, value)
    if kind in (tuple, list):
        return (
            kind.__name__,
            tuple(representation(item) for item in cast(tuple[object, ...], value)),
        )
    if kind is dict:
        return (
            "dict",
            tuple(
                (key, representation(item)) for key, item in cast(dict[str, object], value).items()
            ),
        )
    return (
        kind.__module__,
        kind.__name__,
        tuple(sorted(object.__getattribute__(value, "__pydantic_fields_set__"))),
        representation(object.__getattribute__(value, "__dict__")),
    )


def snapshot(value: object, roots: tuple[type[BaseModel], ...]) -> tuple[bytes, str]:
    raw = canonical(value, roots)
    return raw, sha256_hex(canonical_json_bytes(representation(value)))


def require_unchanged(
    value: object, roots: tuple[type[BaseModel], ...], before: tuple[bytes, str]
) -> None:
    try:
        unchanged = snapshot(value, roots) == before
    except (AttributeError, KeyError, TypeError, ValueError):
        unchanged = False
    if not unchanged:
        raise ValueError("dynamic source input content or representation changed during replay")
