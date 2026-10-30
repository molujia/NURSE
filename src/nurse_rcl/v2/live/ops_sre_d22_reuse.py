"""D22 source-exact reuse of an already delivered full read, never a new fact."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_sre_d21_reuse import exact_delivered_read


def replay_full_receipt(
    proposal: dict[str, Any],
    previous: Sequence[dict[str, Any]],
    *,
    current_record: Callable[[str], dict[str, Any] | None],
    snapshot: str,
) -> dict[str, Any] | None:
    """Copy the proved old view with its identity; caller reserves a new action.

    The D21 proof checks the current admitted semantic record against the old
    complete read. Returning the old view, not a newly rendered card, prevents
    a no-new-content replay from accidentally acquiring additional fields.
    """
    proof = exact_delivered_read(
        proposal,
        [row for row in previous if type(row.get("view")) is dict
         and "receipt_reuse" not in row["view"]],
        current_record=current_record,
        snapshot=snapshot,
    )
    if proof is None:
        return None
    index = proof["prior_action_index"]
    if type(index) is not int or not 1 <= index <= 8:
        raise ValueError("invalid replay source action index")
    saved = next((row for row in reversed(previous) if row.get("action_index") == index), None)
    if saved is None or type(saved.get("view")) is not dict:
        raise ValueError("replay source receipt missing")
    return dict(
        prior_action_index=index,
        record_sha256=proof["record_sha256"],
        source_refs=proof["source_refs"],
        no_new_content=True,
        view=json.loads(canonical_json_bytes(saved["view"])),
    )
