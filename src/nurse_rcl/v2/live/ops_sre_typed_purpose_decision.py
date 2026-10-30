"""Independent native selection: typed scope, never a free concrete purpose."""

from typing import Any

from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.live.ops_sre_entry_decision import selection_shape as navigation_shape
from nurse_rcl.v2.live.ops_sre_single_owner_decision import _parse_receipt_decision

TYPED_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/23"
TYPED_ENTRY_SCHEMA = "nurse.v2.sre-action-entries/4"
TYPED_PROFILE = "v2-ops-sre/12"
TYPED_TOOLS = "hermes-sre-tools/22"


def selection_shape(value: Any) -> None:
    if type(value) is dict and set(value) == {"entry_ref"}:
        if type(value["entry_ref"]) is str and value["entry_ref"].strip():
            return
        raise ExchangeBoundaryError("sre_entry_selection_shape")
    if type(value) is not dict or set(value) != {"navigation", "question"}:
        raise ExchangeBoundaryError("sre_entry_selection_shape")
    navigation_shape(value)


def parse_typed_purpose_decision(
    raw: bytes, *, context: dict[str, Any] | None = None
) -> dict[str, Any]:
    if context is None or context.get("schema_version") != TYPED_INPUT_SCHEMA:
        raise ExchangeBoundaryError("typed_purpose_context_required")
    return _parse_receipt_decision(raw, context=context, selection_validator=selection_shape)


def compile_typed_purpose_entries(capabilities: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_action_entries import compile_action_entries

    return compile_action_entries(capabilities, _typed_purpose=True, **kwargs)


def resolve_typed_selection(value: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_entry_decision import resolve_entry_selection

    return resolve_entry_selection(value, _typed_purpose=True, **kwargs)


def validate_typed_blocked_entry_read(
    value: dict[str, Any], proof: dict[str, Any], context: dict[str, Any]
) -> None:
    """Restore only this new native selection against its frozen canonical action."""
    from nurse_rcl.v2.live.ops_sre_d25_progress import validate_blocked_read

    actions = value.get("actions")
    if type(actions) is not list or len(actions) != 1:
        raise ValueError("blocked typed selection differs")
    selection_shape(actions[0])
    matches = [
        e
        for e in context["action_entries"]["entries"]
        if e["entry_ref"] == actions[0].get("entry_ref")
    ]
    if len(matches) != 1:
        raise ValueError("blocked typed entry is not frozen in this request")
    # This is canonical proof validation, not a fabricated legacy native question.
    validate_blocked_read(
        dict(value, actions=[dict(matches[0]["action"], question=None)]), proof, context
    )
