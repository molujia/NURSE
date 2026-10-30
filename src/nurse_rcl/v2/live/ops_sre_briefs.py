"""Source-bound, explicitly lossy first-delivery mechanism briefs (D13).

The caller admits and retains the complete source. A digest or expansion link
does not grant assertion authority over any omitted leaf.
"""

from __future__ import annotations

from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_sre_working_memory import historical_summary

BRIEF_SCHEMA = "nurse.v2.sre-evidence-brief/1"
BRIEF_REPRESENTATION = "condition_evidence_brief"


def project_evidence_brief(row: dict[str, Any]) -> dict[str, Any]:
    """Project only a registered complete mechanism, without source mutation."""
    projected = historical_summary(row)
    if projected.get("representation") != "historical_evidence_summary":
        return projected
    decision = projected["payload"]["decision"]
    # These registered lists remain in the original source. Neither their
    # counts nor a source digest are new observations or independent support.
    for field, count in (
        ("condition_ids", "condition_count"),
        ("support_evidence_ids", "support_evidence_count"),
    ):
        decision[count] = len(decision.pop(field))
    projected.update(
        schema_version=BRIEF_SCHEMA,
        representation=BRIEF_REPRESENTATION,
        omitted_detail="Individual leaves and evidence IDs are retained in the complete "
        "source; not supplied by this brief. Counts are not independent "
        "support. Explicit read_node expansion consumes an action.",
    )
    projected["brief_sha256"] = sha256_hex(canonical_json_bytes(projected))
    return projected
