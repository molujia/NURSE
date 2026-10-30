"""Compatibility API for the shared product successful-stage handoff."""

from nurse_rcl.v2.live.ops_sre_stage_handoff import (
    optional_worker_diagnostic,
    retained_sources,
    stage_report_context,
    validate_stage_success,
)

__all__ = [
    "optional_worker_diagnostic",
    "retained_sources",
    "stage_report_context",
    "validate_stage_success",
]
