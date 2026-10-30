"""Focused input wrapper over original explicit question lifecycle admission."""

from typing import Any

from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.live.ops_sre_focused_decision import FOCUSED_INPUT_SCHEMA, selection_shape
from nurse_rcl.v2.live.ops_sre_typed_purpose_questions import (
    admit_purpose_note,
    apply_typed_plan,
    project_typed_questions,
    validate_typed_questions,
)


def _context(context: dict[str, Any]) -> None:
    if context.get("schema_version") != FOCUSED_INPUT_SCHEMA:
        raise ExchangeBoundaryError("focused_context_required")


def admit_focused_note(
    value: dict[str, Any], *, context: dict[str, Any], **kwargs: Any
) -> dict[str, Any]:
    _context(context)
    return admit_purpose_note(
        value,
        context=context,
        selection_validator=selection_shape,
        atomic_selector=True,
        focused=True,
        **kwargs,
    )


def apply_focused_plan(
    state: dict[str, Any],
    plan: dict[str, Any],
    *,
    decision_index: int,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if context is not None:
        _context(context)
    return apply_typed_plan(state, plan, decision_index=decision_index, context=context)


def project_focused_questions(
    memory: list[dict[str, Any]], *, short_refs: bool = False
) -> dict[str, Any]:
    return project_typed_questions(memory, short_refs=short_refs)


def validate_focused_questions(state: Any, *, context: dict[str, Any], **kwargs: Any) -> None:
    _context(context)
    validate_typed_questions(state, context=context, **kwargs)
