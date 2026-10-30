"""Source-exact D21 no-new-content gate; no cross-request cache or tool dispatch."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex

_READ_PARAMETERS = {"card_id", "section", "cursor"}


def exact_delivered_read(
    proposal: dict[str, Any],
    previous: Sequence[dict[str, Any]],
    *,
    current_record: Callable[[str], dict[str, Any] | None],
    snapshot: str,
) -> dict[str, Any] | None:
    """Return a proof only for a prior complete, identical, still-admitted read."""
    actions = proposal.get("actions")
    if type(actions) is not list or len(actions) != 1:
        return None
    action = actions[0]
    if type(action) is not dict or action.get("tool") != "read_node":
        return None
    parameters = action.get("parameters")
    if (
        type(parameters) is not dict
        or set(parameters) != _READ_PARAMETERS
        or type(parameters["card_id"]) is not str
        or parameters["section"] != "payload"
        or parameters["cursor"] is not None
    ):
        return None
    card = parameters["card_id"]
    current = current_record(card)
    if type(current) is not dict or current.get("card_id") != card:
        return None
    for saved in reversed(previous):
        if (
            saved.get("status") != "completed"
            or saved.get("tool") != "read_node"
            or saved.get("parameters") != parameters
        ):
            continue
        view = saved.get("view")
        if type(view) is not dict:
            continue
        delivered = view.get("semantic_record")
        if (
            view.get("case_snapshot_id") != snapshot
            or view.get("card_id") != card
            or view.get("section") != parameters["section"]
            or view.get("representation") != "complete_semantic_record"
            or view.get("next_cursor") is not None
            or type(delivered) is not dict
            or delivered.get("card_id") != card
            or canonical_json_bytes(delivered.get("source_refs"))
            != canonical_json_bytes(current.get("source_refs"))
            or canonical_json_bytes(delivered) != canonical_json_bytes(current)
        ):
            continue
        return dict(
            reason="already_delivered_no_new_content",
            prior_action_index=saved["action_index"],
            card_id=card,
            parameters_sha256=sha256_hex(canonical_json_bytes(parameters)),
            record_sha256=sha256_hex(canonical_json_bytes(delivered)),
            source_refs=delivered["source_refs"],
            representation="complete_semantic_record",
        )
    return None


def exact_delivered_ledger_read(
    proposal: dict[str, Any], records: Sequence[Any], *, toolbox: Any
) -> dict[str, Any] | None:
    """Check the real receipt and currently admitted card, never hidden source files."""
    from nurse_rcl.v2.live.ops_sre_context import _receipt
    from nurse_rcl.v2.live.ops_sre_encoding import expand
    from nurse_rcl.v2.live.ops_sre_views import semantic_record

    def current(card: str) -> dict[str, Any] | None:
        if toolbox.object_contract(card) is None:
            return None
        source = toolbox._cards.get(card)
        if source is None:
            return None
        row = semantic_record(
            source,
            published=card in toolbox._core_cards,
            origin=toolbox._origins.get(card),
        )
        return toolbox._enrich(row) if toolbox.profile_id == "hermes-sre-tools/25" else row

    history = []
    for record in records:
        receipt, view = _receipt(record)
        history.append(dict(receipt, view=expand(view)))
    return exact_delivered_read(
        proposal, history, current_record=current, snapshot=toolbox.case_snapshot_id
    )
