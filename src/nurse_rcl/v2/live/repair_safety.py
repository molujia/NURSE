"""Explicit, finite whole-case repair capacity; no global overrides or trust cache.

Retain dynamic exact-field/representation rules with a live-only bounded
projection. Only this entry point selects the approved live capacity.
Per-source blocks, old dynamic APIs and Report defaults retain their limits.
"""

from __future__ import annotations

from typing import Final, cast

from pydantic import BaseModel, ValidationInfo, model_validator

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.contracts import _SourceModel
from nurse_rcl.v2.dynamic_verification.safety import exact_field, representation
from nurse_rcl.v2.live.repair_projection import _project_preflighted
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

REPAIR_CAPACITY_PROFILE: Final = "live-repair-6m-256m/1"
REPAIR_MAX_NODES: Final = 6_000_000
REPAIR_MAX_JSON_BYTES: Final = 256 * 1024 * 1024
REPAIR_CAPACITY_SHA256: Final = sha256_hex(
    canonical_json_bytes(
        {
            "profile": REPAIR_CAPACITY_PROFILE,
            "max_nodes": REPAIR_MAX_NODES,
            "max_json_bytes": REPAIR_MAX_JSON_BYTES,
            "max_depth": 64,
            "max_string_length": 8192,
            "max_numeric_atom_bytes": 256,
            "max_provenance_ids": 131_072,
        }
    )
)


def _validated_plain(value: object, roots: tuple[type[BaseModel], ...]) -> object:
    preflight_exact_pydantic_tree(
        value,
        root_model_types=roots,
        label="repair live",
        max_nodes=REPAIR_MAX_NODES,
    )
    # Structural preflight still precedes projection. The private walk checks
    # exact fields and the original byte/atom budgets while constructing plain
    # values; no JSON is encoded before the complete walk succeeds.
    return _project_preflighted(value)


def preflight(value: object, roots: tuple[type[BaseModel], ...]) -> None:
    _validated_plain(value, roots)


def canonical(value: object, roots: tuple[type[BaseModel], ...]) -> bytes:
    return canonical_json_bytes(_validated_plain(value, roots))


def snapshot(value: object, roots: tuple[type[BaseModel], ...]) -> tuple[bytes, str]:
    raw = canonical(value, roots)
    return raw, sha256_hex(canonical_json_bytes(representation(value)))


def require_unchanged(
    value: object, roots: tuple[type[BaseModel], ...], before: tuple[bytes, str]
) -> None:
    try:
        same = snapshot(value, roots) == before
    except (AttributeError, KeyError, TypeError, ValueError):
        same = False
    if not same:
        raise ValueError("repair live input content or representation changed during replay")


class RepairModel(_SourceModel):
    """Current aggregate models only; retain exact shapes and JSON-only tuples."""

    @model_validator(mode="before")
    @classmethod
    def exact_python_fields(cls, value: object, info: ValidationInfo) -> object:
        # Override the same-named inherited validator, not a global policy.
        preflight(value, (cls,))
        if info.mode == "json" and not exact_field(value, cls, json_mode=True):
            raise ValueError("repair live JSON requires exact declared field shapes")
        if info.mode == "python" and type(value) is dict:
            for name, item in cast(dict[str, object], value).items():
                if name not in cls.model_fields or not exact_field(
                    item, cls.model_fields[name].annotation
                ):
                    raise ValueError("repair live constructor requires exact typed fields")
        return value
