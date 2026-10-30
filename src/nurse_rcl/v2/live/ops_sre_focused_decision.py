"""Explicit focused profile; legacy navigation shape domains remain unchanged."""

from typing import Any

from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.live.ops_sre_atomic_selector_decision import (
    selection_shape as atomic_shape,
)
from nurse_rcl.v2.live.ops_sre_atomic_selector_decision import (
    validate_atomic_blocked_entry_read,
)
from nurse_rcl.v2.live.ops_sre_single_owner_decision import _parse_receipt_decision

FOCUSED_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/26"
FOCUSED_ENTRY_SCHEMA = "nurse.v2.sre-action-entries/6"
FOCUSED_PROFILE = "v2-ops-sre/15"
FOCUSED_TOOLS = "hermes-sre-tools/25"


def selection_shape(value: Any) -> None:
    nav = value.get("navigation") if type(value) is dict else None
    if (
        type(nav) is not dict
        or nav.get("kind") != "catalog"
        or nav.get("catalog_kind") != "evidence"
    ):
        atomic_shape(value)
        return
    if (
        set(value) != {"navigation", "question"}
        or type(value["question"]) is not str
        or not value["question"].strip()
        or set(nav) != {"kind", "catalog_kind", "filters", "cursor"}
        or (nav["cursor"] is not None and type(nav["cursor"]) is not str)
    ):
        raise ExchangeBoundaryError("sre_entry_selection_shape")
    filters = nav["filters"]
    from nurse_rcl.v2.live.ops_sre_focused_catalog import FILTERS

    if type(filters) is not dict or not set(filters) <= FILTERS:
        raise ExchangeBoundaryError("sre_entry_selection_shape")
    scope = filters.get("candidate_service_keys")
    if (
        type(scope) is not list
        or not 1 <= len(scope) <= 2
        or any(type(c) is not str or not c for c in scope)
        or len(scope) != len(set(scope))
        or any(
            type(v) is not str or not v for k, v in filters.items() if k != "candidate_service_keys"
        )
    ):
        raise ExchangeBoundaryError("sre_entry_selection_shape")


def parse_focused_decision(raw: bytes, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
    if context is None or context.get("schema_version") != FOCUSED_INPUT_SCHEMA:
        raise ExchangeBoundaryError("focused_context_required")
    return _parse_receipt_decision(raw, context=context, selection_validator=selection_shape)


def compile_focused_entries(capabilities: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_action_entries import compile_action_entries

    return compile_action_entries(capabilities, _focused=True, **kwargs)


def validate_focused_entries(table: dict[str, Any], **kwargs: Any) -> None:
    from nurse_rcl.v2.live.ops_sre_entry_decision import validate_action_entries

    validate_action_entries(table, _focused=True, **kwargs)


def resolve_focused_selection(value: Any, **kwargs: Any) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_entry_decision import resolve_entry_selection

    return resolve_entry_selection(value, _focused=True, **kwargs)


def validate_focused_blocked_entry_read(
    value: dict[str, Any], proof: dict[str, Any], context: dict[str, Any]
) -> None:
    # This proof validates only a concrete frozen selection and its actual read.
    validate_atomic_blocked_entry_read(value, proof, context)
