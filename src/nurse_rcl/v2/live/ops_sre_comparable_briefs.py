"""D16 role meaning over the complete D14 source-bound condition projection."""

from __future__ import annotations

from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_sre_condition_briefs import (
    _FORMAL,
    _PARTIAL,
    _RECHECK,
    CONDITION_BRIEF_SCHEMA,
    project_condition_brief,
)

COMPARABLE_BRIEF_SCHEMA = "nurse.v2.sre-evidence-brief/3"
_ROLE = {
    "necessary": "necessary_condition",
    "optional": "optional_condition",
    "refuting": "refutation",
}
_TRUTH = {"true": "established", "false": "not_established", "unknown": "undetermined"}


def project_comparable_brief(row: dict[str, Any]) -> dict[str, Any]:
    """Interpret each registered condition, never its effect on the final score.

    The old projector still validates complete leaves, identities and K3 truth.
    No change to /2 or to the formal necessary-only disposition algorithm.
    """
    brief = project_condition_brief(row)
    if (
        row.get("representation") in _PARTIAL
        or row["node"] != "mechanisms"
        or set(row["payload"]) not in (_FORMAL, _RECHECK)
        or brief.get("schema_version") != CONDITION_BRIEF_SCHEMA
    ):
        return brief
    for condition in brief["payload"]["conditions"]:
        condition["condition_meaning"] = (
            _ROLE[condition["requirement"]] + "_" + _TRUTH[condition["truth_value"]]
        )
    brief["payload"].update(
        disposition_rule="necessary_conjunction_only",
        score_effect_rule="requires_actual_term_and_family_selection",
    )
    brief["schema_version"] = COMPARABLE_BRIEF_SCHEMA
    brief.pop("brief_sha256")
    brief["brief_sha256"] = sha256_hex(canonical_json_bytes(brief))
    return brief
