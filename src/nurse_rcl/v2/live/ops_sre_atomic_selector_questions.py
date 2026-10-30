"""Atomic selector wrapper over unchanged pure scope and explicit lifecycle rules."""

from typing import Any

from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
from nurse_rcl.v2.live.ops_sre_atomic_selector_decision import ATOMIC_INPUT_SCHEMA, selection_shape
from nurse_rcl.v2.live.ops_sre_typed_purpose_questions import (
    admit_purpose_note,
    apply_typed_plan,
    project_typed_questions,
    validate_typed_questions,
)


def _context(context: dict[str, Any]) -> None:
    if context.get("schema_version") != ATOMIC_INPUT_SCHEMA:
        raise ExchangeBoundaryError("atomic_selector_context_required")


def admit_atomic_note(
    value: dict[str, Any],
    *,
    context: dict[str, Any],
    book: SreActionReferenceBook | None = None,
    short_refs: bool = False,
) -> dict[str, Any]:
    _context(context)
    return admit_purpose_note(
        value,
        context=context,
        book=book,
        short_refs=short_refs,
        selection_validator=selection_shape,
        atomic_selector=True,
    )


def apply_atomic_plan(
    state: dict[str, Any],
    plan: dict[str, Any],
    *,
    decision_index: int,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if context is not None:
        _context(context)
    return apply_typed_plan(state, plan, decision_index=decision_index, context=context)


def project_atomic_questions(
    memory: list[dict[str, Any]], *, short_refs: bool = False,
) -> dict[str, Any]:
    return project_typed_questions(memory, short_refs=short_refs)


def validate_atomic_questions(
    state: Any,
    *,
    context: dict[str, Any],
    book: SreActionReferenceBook | None = None,
    short_refs: bool = False,
) -> None:
    _context(context)
    validate_typed_questions(state, context=context, book=book, short_refs=short_refs)
