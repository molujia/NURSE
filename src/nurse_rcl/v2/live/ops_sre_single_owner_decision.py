"""Versioned receipt acknowledgment; only explicit question updates own lifecycle."""

from __future__ import annotations

from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_sre_action_decision import ACTION_FIELDS, STOP_BASES
from nurse_rcl.v2.live.ops_sre_d23_decision import CONTENT_RELATIONS, validate_balanced_facts

SINGLE_OWNER_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/22"
SINGLE_OWNER_ENTRY_SCHEMA = "nurse.v2.sre-action-entries/3"
SINGLE_OWNER_PROFILE = "v2-ops-sre/11"
SINGLE_OWNER_TOOLS = "hermes-sre-tools/21"
LATEST_FIELDS = {"action_index", "content_relation", "summary"}


def parse_single_owner_decision(
    raw: bytes, *, context: dict[str, Any] | None = None
) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_entry_decision import selection_shape

    if context is None or context.get("schema_version") != SINGLE_OWNER_INPUT_SCHEMA:
        raise ExchangeBoundaryError("single_owner_context_required")
    return _parse_receipt_decision(raw, context=context, selection_validator=selection_shape)


def _parse_receipt_decision(
    raw: bytes, *, context: dict[str, Any], selection_validator: Any
) -> dict[str, Any]:
    """Shared pure receipt/stop validation; wrappers own version and selection shape."""
    from nurse_rcl.v2.live.ops_sre_context import parse_sre_decision

    value = decode_json_object(raw, CONTEXT_LIMITS)
    if set(value) != ACTION_FIELDS:
        raise ExchangeBoundaryError("sre_action_decision_shape")
    base = {
        k: value[k]
        for k in ("actions", "stop_reason", "public_summary", "unresolved", "responds_to")
    }
    parse_sre_decision(canonical_json_bytes(base), context=context)
    basis = value["stop_basis"]
    if basis is not None and (type(basis) is not str or basis not in STOP_BASES):
        raise ExchangeBoundaryError("sre_stop_basis")
    if value["actions"] and (value["stop_reason"] is not None or basis is not None):
        raise ExchangeBoundaryError("sre_action_continue_shape")
    if not value["actions"] and value["stop_reason"] is None:
        raise ExchangeBoundaryError("sre_action_stop_shape")
    if len(value["actions"]) > 1:
        raise ExchangeBoundaryError("sre_entry_selection_shape")
    for action in value["actions"]:
        selection_validator(action)
    update = value["latest_result_update"]
    if update is not None and (
        type(update) is not dict
        or set(update) != LATEST_FIELDS
        or type(update["action_index"]) is not int
        or not 1 <= update["action_index"] <= 8
        or type(update["content_relation"]) is not str
        or update["content_relation"] not in CONTENT_RELATIONS
        or type(update["summary"]) is not str
        or not update["summary"].strip()
    ):
        raise ExchangeBoundaryError("sre_latest_result_shape")
    latest = context["actions"][-1]["action_index"] if context["actions"] else None
    if latest is None:
        if update is not None:
            raise ExchangeBoundaryError("sre_latest_result_without_action")
    elif (
        update is None
        or update["action_index"] != latest
        or not any(link["action_index"] == latest for link in value["responds_to"])
    ):
        raise ExchangeBoundaryError("sre_latest_result_not_addressed")
    validate_balanced_facts(value, context)
    return value


def admit_single_owner_note(value: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_observable_scope import _admit_scoped_note

    return _admit_scoped_note(value, context=context, _single_owner=True)


def compile_single_owner_entries(capabilities: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_sre_action_entries import compile_action_entries

    return compile_action_entries(capabilities, _single_owner=True, **kwargs)
