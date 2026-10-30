"""D21 execution-after view over saved receipts, without interpreting agent intent."""

from __future__ import annotations

from typing import Any


def post_action_state(
    actions: list[dict[str, Any]],
    delivered_index: dict[str, Any],
    *,
    actions_available_now: int,
) -> dict[str, Any]:
    """Copy ledger and delivery metadata; never infer diagnostic resolution."""
    if (
        type(actions) is not list
        or len(actions) > 8
        or any(
            type(row) is not dict or row.get("action_index") != i
            for i, row in enumerate(actions, 1)
        )
        or type(actions_available_now) is not int
        or actions_available_now != 8 - len(actions)
        or delivered_index.get("latest_action_index") != (len(actions) or None)
    ):
        raise ValueError("post-action ledger identity differs")
    refs = delivered_index["latest_record_refs"]
    records = {row["record_sha256"]: row for row in delivered_index["records"]}
    if len(refs) != len(set(refs)) or not set(refs) <= records.keys():
        raise ValueError("post-action delivery source differs")
    delivered = [
        {
            key: records[ref][key]
            for key in (
                "card_id", "record_sha256", "source_role", "kind", "node", "targets",
                "delivery_kind", "fields_present", "latest_delivery", "section_fields",
            )
        }
        for ref in refs
    ]
    relation = (
        "none_delivered"
        if not delivered
        else "no_new_content"
        if all(str(row["latest_delivery"]).startswith("repeat_") for row in delivered)
        else "new_content"
    )
    last = actions[-1] if actions else None
    return dict(
        schema_version="nurse.v2.sre-post-action-state/1",
        public_memory_temporal_role="pre_action_intent_not_outstanding_instruction",
        last_actual_action=None
        if last is None
        else dict(
            action_index=last["action_index"], tool=last.get("tool"),
            parameters=last.get("parameters"), status=last["status"],
        ),
        delivered=delivered,
        content_relation=relation,
        actions_available_now=actions_available_now,
        authority="ledger_and_delivered_record_metadata_not_diagnostic_conclusion",
    )
