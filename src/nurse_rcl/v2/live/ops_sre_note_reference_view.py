"""Copyable proposal selectors over already supplied history, not source authority."""

from __future__ import annotations

from typing import Any

from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.live.ops_sre_action_decision import bind_note_sources


def note_reference_reading_aid(context: dict[str, Any]) -> dict[str, Any]:
    """Use the original binder; do not read sources, bind new facts or answer questions.

    The display is bounded, not the legal selector domain. Older omitted selectors
    remain subject to the same binder. Short model aliases confer no owner SHA rights.
    """
    refs: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    if type(context.get("decision_index")) is int:
        for action in sorted(
            context.get("actions", []), key=lambda r: r["action_index"], reverse=True
        ):
            for card_id in action.get("visible_card_ids", []):
                selector = dict(action_index=action["action_index"], card_id=card_id)
                try:
                    bind_note_sources([selector], context=context)
                except ExchangeBoundaryError:
                    continue
                identity = (selector["action_index"], selector["card_id"])
                if identity not in seen:
                    refs.append(selector)
                    seen.add(identity)
    shown = min(len(refs), 16)
    return dict(
        authority="proposal_selectors_not_evidence_or_question_answers",
        proposal_refs=refs[:shown],
        total=len(refs),
        shown=shown,
        omitted=len(refs) - shown,
        bound_history_usage="read_only_program_bindings_not_the_native_proposal_format",
    )
