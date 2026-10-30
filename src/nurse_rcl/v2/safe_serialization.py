"""Callback-free preflight for exact, code-owned Pydantic model trees."""

from __future__ import annotations

from functools import cache
from typing import TypeVar, cast, get_args

from pydantic import BaseModel

DEFAULT_SERIALIZATION_MAX_DEPTH = 64
DEFAULT_SERIALIZATION_MAX_NODES = 500_000
DEFAULT_SERIALIZED_STRING_LENGTH = 8_192
DEFAULT_PROVENANCE_ID_LIMIT = 131_072

_ModelT = TypeVar("_ModelT", bound=BaseModel)


@cache
def _declared_model_type_closure(
    root_model_types: tuple[type[BaseModel], ...],
) -> tuple[type[BaseModel], ...]:
    """Return exact model types reachable through code-owned field annotations."""

    pending = list(root_model_types)
    discovered: list[type[BaseModel]] = []
    while pending:
        model_type = pending.pop()
        if any(model_type is item for item in discovered):
            continue
        discovered.append(model_type)
        for field in model_type.model_fields.values():
            annotations = [field.annotation]
            while annotations:
                annotation = annotations.pop()
                if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                    if not any(annotation is item for item in discovered):
                        pending.append(annotation)
                    continue
                annotations.extend(get_args(annotation))
    return tuple(discovered)


def preflight_exact_pydantic_tree(
    value: object,
    *,
    root_model_types: tuple[type[BaseModel], ...],
    label: str,
    max_depth: int = DEFAULT_SERIALIZATION_MAX_DEPTH,
    max_nodes: int = DEFAULT_SERIALIZATION_MAX_NODES,
    max_string_length: int | None = DEFAULT_SERIALIZED_STRING_LENGTH,
    provenance_id_limit: int | None = DEFAULT_PROVENANCE_ID_LIMIT,
    resource_error_type: type[ValueError] = ValueError,
) -> None:
    """Reject hostile or unbounded values before serializer or model callbacks run."""

    if not root_model_types or any(not issubclass(item, BaseModel) for item in root_model_types):
        raise TypeError("root model types must contain code-owned Pydantic model classes")
    if max_depth < 0 or max_nodes < 1:
        raise ValueError("serialization limits must be positive")
    if max_string_length is not None and max_string_length < 0:
        raise ValueError("string length limit must be nonnegative")
    if provenance_id_limit is not None and provenance_id_limit < 0:
        raise ValueError("provenance limit must be nonnegative")

    allowed_models = _declared_model_type_closure(root_model_types)
    stack: list[tuple[object, int, bool]] = [(value, 0, False)]
    active_container_ids: set[int] = set()
    visited_nodes = 0
    scheduled_nodes = 1
    provenance_id_count = 0

    while stack:
        current, depth, leaving = stack.pop()
        current_id = id(current)
        if leaving:
            active_container_ids.remove(current_id)
            continue
        visited_nodes += 1
        if visited_nodes > max_nodes:
            raise resource_error_type(f"{label} exceeds the serialization node budget")
        if depth > max_depth:
            raise resource_error_type(f"{label} exceeds the serialization depth budget")

        current_type = type(current)
        if (
            current_type is str
            or current_type is int
            or current_type is float
            or current_type is bool
            or current_type is type(None)
        ):
            if (
                current_type is str
                and max_string_length is not None
                and str.__len__(cast(str, current)) > max_string_length
            ):
                raise resource_error_type(f"{label} exceeds the string length limit")
            continue

        child_count: int
        model_type = next(
            (item for item in allowed_models if current_type is item),
            None,
        )
        if model_type is not None:
            state = object.__getattribute__(current, "__dict__")
            if type(state) is not dict:
                raise ValueError(f"{label} contains a model without an exact state dictionary")
            expected_field_names = tuple(sorted(model_type.model_fields))
            child_count = len(expected_field_names)
            extra_state = object.__getattribute__(current, "__pydantic_extra__")
            private_state = object.__getattribute__(current, "__pydantic_private__")
            fields_set = object.__getattribute__(current, "__pydantic_fields_set__")
            if extra_state is not None or private_state is not None:
                raise ValueError(f"{label} contains non-canonical Pydantic slot state")
            if type(fields_set) is not set:
                raise ValueError(f"{label} contains a non-exact Pydantic fields set")
            for field_name in set.__iter__(fields_set):
                if type(field_name) is not str or field_name not in expected_field_names:
                    raise ValueError(f"{label} contains an undeclared Pydantic field marker")
            if dict.__len__(state) != len(expected_field_names):
                raise ValueError(f"{label} contains a model with a non-canonical field set")
            field_names = tuple(dict.keys(state))
            if any(type(field_name) is not str for field_name in field_names):
                raise ValueError(f"{label} contains a non-string model field")
            if tuple(sorted(field_names)) != expected_field_names:
                raise ValueError(f"{label} contains a model with a non-canonical field set")
            if provenance_id_limit is not None:
                for field_name in expected_field_names:
                    if not str.endswith(field_name, "provenance_ids"):
                        continue
                    provenance_ids = dict.__getitem__(state, field_name)
                    if type(provenance_ids) is tuple:
                        provenance_id_count += tuple.__len__(provenance_ids)
                    elif type(provenance_ids) is list:
                        provenance_id_count += list.__len__(provenance_ids)
                    if provenance_id_count > provenance_id_limit:
                        raise resource_error_type(f"{label} exceeds the provenance limit")
        elif current_type is tuple:
            current_tuple = cast(tuple[object, ...], current)
            child_count = tuple.__len__(current_tuple)
        elif current_type is list:
            current_list = cast(list[object], current)
            child_count = list.__len__(current_list)
        elif current_type is dict:
            current_dict = cast(dict[object, object], current)
            child_count = dict.__len__(current_dict)
        else:
            raise ValueError(f"{label} contains an unsupported exact runtime type")

        if scheduled_nodes + child_count > max_nodes:
            raise resource_error_type(f"{label} exceeds the serialization node budget")
        scheduled_nodes += child_count
        if current_id in active_container_ids:
            raise ValueError(f"{label} contains a serialization cycle")
        active_container_ids.add(current_id)
        stack.append((current, depth, True))
        if model_type is not None:
            stack.extend(
                (dict.__getitem__(state, field_name), depth + 1, False)
                for field_name in reversed(expected_field_names)
            )
        elif current_type is tuple:
            stack.extend(
                (tuple.__getitem__(current_tuple, index), depth + 1, False)
                for index in range(child_count - 1, -1, -1)
            )
        elif current_type is list:
            stack.extend(
                (list.__getitem__(current_list, index), depth + 1, False)
                for index in range(child_count - 1, -1, -1)
            )
        else:
            for key in dict.__iter__(current_dict):
                if type(key) is not str and type(key) is not int:
                    raise ValueError(f"{label} contains a non-canonical mapping key")
                if (
                    max_string_length is not None
                    and type(key) is str
                    and str.__len__(key) > max_string_length
                ):
                    raise resource_error_type(f"{label} exceeds the string length limit")
                stack.append((dict.__getitem__(current_dict, key), depth + 1, False))
