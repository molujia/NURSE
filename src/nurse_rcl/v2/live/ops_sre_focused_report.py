"""Explicit pinned focused-stage source admission for the G+ report successor."""

from typing import Any, Literal

from nurse_rcl.v2.live.ops_sre_atomic_report import (
    AtomicReportSourceIdentity,
    _assemble_pinned_report_context,
    _decode_source,
    _validate_pinned_report_identity,
)
from nurse_rcl.v2.live.ops_sre_brief_report import BriefSreReportContext, FocusedReportMetadata


class FocusedReportSourceIdentity(AtomicReportSourceIdentity):
    schema_version: Literal["nurse.v2.atomic-report-source/2"] = "nurse.v2.atomic-report-source/2"  # type: ignore[assignment]
    execution_profile: Literal["v2-ops-sre/15"] = "v2-ops-sre/15"  # type: ignore[assignment]
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/26"] = (
        "nurse.v2.sre-decision-exchange/26"  # type: ignore[assignment]
    )


def focused_source(value: dict[str, Any], expected: str) -> FocusedReportSourceIdentity:
    return _decode_source(value, expected, FocusedReportSourceIdentity)


def validate_focused_report_identity(
    metadata: FocusedReportMetadata,
) -> FocusedReportSourceIdentity:
    if type(metadata) is not FocusedReportMetadata:
        raise ValueError("focused report metadata type differs")
    return _validate_pinned_report_identity(metadata, focused_source)


def assemble_focused_report_context(**kwargs: Any) -> BriefSreReportContext:
    from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_context_from_wire

    def wire_context(
        raw: bytes, *, references: dict[str, Any], case_id: str, snapshot: str, index: int
    ) -> dict[str, Any]:
        context = brief_context_from_wire(
            raw,
            references=references,
            opaque_case_id=case_id,
            case_snapshot_id=snapshot,
            decision_index=index,
        )
        if context.get("schema_version") != "nurse.v2.sre-investigation-input/26":
            raise ValueError("focused report source wire version differs")
        return context

    return _assemble_pinned_report_context(
        **kwargs,
        source_type=FocusedReportSourceIdentity,
        metadata_type=FocusedReportMetadata,
        decode_source=focused_source,
        wire_context=wire_context,
        input_schema="nurse.v2.sre-investigation-input/26",
        focused=True,
    )
