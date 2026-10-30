"""Staged diagnostic metadata descriptor; live/storage admission comes after local gates."""

from typing import Literal

from nurse_rcl.v2.live.ops_sre_brief_report import ActionReportMetadata


class StagedReportMetadata(ActionReportMetadata):
    schema_version: Literal["nurse.v2.sre-report-context/18"] = "nurse.v2.sre-report-context/18"  # type: ignore[assignment]
    profile: Literal["hermes-sre-tools/18"] = "hermes-sre-tools/18"  # type: ignore[assignment]
    investigation_input_schema: Literal["nurse.v2.sre-investigation-input/19"] = (
        "nurse.v2.sre-investigation-input/19"  # type: ignore[assignment]
    )
