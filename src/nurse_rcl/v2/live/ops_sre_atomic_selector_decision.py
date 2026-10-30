"""Atomic native references; receipt validation never repairs the proposal."""

from typing import Any

from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.live.ops_sre_entry_decision import selection_shape as navigation_shape
from nurse_rcl.v2.live.ops_sre_single_owner_decision import _parse_receipt_decision

ATOMIC_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/24"
ATOMIC_ENTRY_SCHEMA = "nurse.v2.sre-action-entries/5"
ATOMIC_PROFILE = "v2-ops-sre/13"
ATOMIC_TOOLS = "hermes-sre-tools/23"


def selection_shape(value: Any) -> None:
    if type(value) is str:
        if value:
            return
        raise ExchangeBoundaryError("sre_entry_selection_shape")
    if type(value) is not dict or set(value) != {"navigation", "question"}:
        raise ExchangeBoundaryError("sre_entry_selection_shape")
    navigation_shape(value)


def parse_atomic_decision(raw: bytes, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
    if context is None or context.get("schema_version") != ATOMIC_INPUT_SCHEMA:
        raise ExchangeBoundaryError("atomic_selector_context_required")
    return _parse_receipt_decision(raw, context=context, selection_validator=selection_shape)


def compile_atomic_entries(capabilities: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_action_entries import compile_action_entries

    return compile_action_entries(capabilities, _atomic_selector=True, **kwargs)


def validate_atomic_entries(table: dict[str, Any], **kwargs: Any) -> None:
    from nurse_rcl.v2.live.ops_sre_entry_decision import validate_action_entries

    validate_action_entries(table, _atomic_selector=True, **kwargs)


def resolve_atomic_selection(value: Any, **kwargs: Any) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_entry_decision import resolve_entry_selection

    return resolve_entry_selection(value, _atomic_selector=True, **kwargs)


def validate_atomic_blocked_entry_read(
    value: dict[str, Any], proof: dict[str, Any], context: dict[str, Any]
) -> None:
    """Complete-read proof still binds the frozen canonical action, not a string."""
    from nurse_rcl.v2.live.ops_sre_d25_progress import validate_blocked_read

    actions = value.get("actions")
    if type(actions) is not list or len(actions) != 1:
        raise ValueError("blocked atomic selection differs")
    selection_shape(actions[0])
    if type(actions[0]) is not str:
        raise ValueError("blocked atomic selection is not a concrete reference")
    matches = [e for e in context["action_entries"]["entries"] if e["entry_ref"] == actions[0]]
    if len(matches) != 1:
        raise ValueError("blocked atomic entry is not frozen in this request")
    validate_blocked_read(
        dict(value, actions=[dict(matches[0]["action"], question=None)]), proof, context
    )
