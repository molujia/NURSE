"""Directed protocol reuses admission primitives, never masquerades as B input."""

from typing import Any

from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.live.ops_sre_action_decision import _parse_action_decision, admit_note

DIRECTED_INPUT_SCHEMA = "nurse.v2.sre-investigation-input/19"


def parse_directed_decision(
    raw: bytes, *, context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if context is None or context.get("schema_version") != DIRECTED_INPUT_SCHEMA:
        raise ExchangeBoundaryError("sre_directed_context_required")
    return _parse_action_decision(raw, context=context, input_schema=DIRECTED_INPUT_SCHEMA)


def admit_directed_note(value: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
    if context.get("schema_version") != DIRECTED_INPUT_SCHEMA:
        raise ExchangeBoundaryError("sre_directed_context_required")
    return admit_note(value, context=context)
