"""Exact record-to-step permissions, derived only from already admitted receipts."""

from __future__ import annotations

from typing import Any


def workflow_evidence_index(
    groups: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    steps: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    """A guide, never new permission: mixed claims use the intersection of steps.

    All scalar aliases and provenance remain at their original group positions.
    Equal values with different source records cannot lend one another permissions.
    A completed tool returning unknown is an actual observation, not an incomplete action.
    """
    delivered = [(s["step_ref"], set(s["observed_fact_refs"])) for s in steps
                 if s["executed"]["status"] == "completed"]
    return [dict(record_group_index=index,
        fact_refs_location=f"record_groups[{index}].fact_refs",
        source_location=f"record_groups[{index}]",
        reuse_actions_location=f"record_groups[{index}].action_indices",
        observation_step_refs=[step for step, refs in delivered
                               if group["fact_refs"] and set(group["fact_refs"]) <= refs])
        for index, group in enumerate(groups)]
