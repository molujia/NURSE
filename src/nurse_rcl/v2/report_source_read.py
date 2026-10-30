"""Closed paired source reading for report context /3 and /4."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2 import case_audit as old_audit
from nurse_rcl.v2 import report_context as legacy
from nurse_rcl.v2.candidate_universe import EmptyCandidateUniverseFailure
from nurse_rcl.v2.case_audit_lineage_v2 import CaseAuditLineageIndexV2
from nurse_rcl.v2.case_audit_read import SupportedCaseAudit, read_case_audit
from nurse_rcl.v2.case_audit_v2 import (
    CaseRankingReplayBundleV2,
    ImmutableCaseAuditPackageV2,
    _audit_document,
)
from nurse_rcl.v2.report_context_v4 import (
    FrozenReportContextV4,
    FrozenReportPayloadV2,
    _context_domains,
    _report_document,
    _report_snapshot,
    _report_unchanged,
    _source_embedded_domains,
    freeze_report_context_from_audit_v2,
)
from nurse_rcl.v2.runtime_schemas import SealedAuditPackage

SupportedReportContext: TypeAlias = legacy.FrozenReportContext | FrozenReportContextV4


@dataclass(frozen=True, slots=True, init=False)
class ReportSourceReadView:
    """Fresh private values; construction cannot confer public source authority."""

    canonical_audit_bytes: bytes
    canonical_context_bytes: bytes
    source_audit_schema_version: str
    source_context_schema_version: str
    source_payload_schema_version: str
    source_ranking_bundle_schema_version: str
    source_lineage_schema_version: str
    opaque_case_id: str
    audit_sha256: str
    context_sha256: str
    payload_sha256: str
    context: SupportedReportContext
    payload: legacy.FrozenReportPayload | FrozenReportPayloadV2
    replay_bundle: old_audit.CaseRankingReplayBundle | CaseRankingReplayBundleV2
    lineage_index: old_audit.CaseAuditLineageIndex | CaseAuditLineageIndexV2

    def __init__(self) -> None:
        raise TypeError("report views require complete independently pinned paired reading")


def _pin(pin: object, actual: str, *, label: str) -> None:
    if (
        type(pin) is not str
        or len(pin) != 64
        or any(c not in "0123456789abcdef" for c in pin)
        or not hmac.compare_digest(pin, actual)
    ):
        raise ValueError(f"independent {label} digest does not match the original source")


def _preflight_report_sources(
    *,
    context: SupportedReportContext,
    audit_package: SupportedCaseAudit,
    expected_audit_sha256: str,
    expected_context_sha256: str,
) -> tuple[Any, Any, dict[str, dict[str, Any]]]:
    """Pure paired preflight for internal consumers, with no encoder or replay.

    This detached data is not a qualification receipt and is never accepted as a
    public input. Shared report entry points can run this gate before inspecting
    their other roots; this reader always repeats it over the original sources.
    """
    old = (
        type(context) is legacy.FrozenReportContext
        and type(audit_package) is old_audit.ImmutableCaseAuditPackage
    )
    new = (
        type(context) is FrozenReportContextV4
        and type(audit_package) is ImmutableCaseAuditPackageV2
    )
    if not old and not new:
        raise ValueError("report requires audit /1 + context /3 or audit /2 + context /4")
    # Inspect both roots before using any named field, property or serializer.
    context_original = _report_snapshot(context)
    audit_original = _report_snapshot(audit_package)
    context_state = cast(dict[str, Any], context_original[0])
    audit_state = cast(dict[str, Any], audit_original[0])
    _pin(expected_audit_sha256, audit_state["audit_sha256"], label="audit")
    _pin(expected_context_sha256, context_state["context_sha256"], label="context")
    if audit_state["terminal_state"] != "SEALED":
        raise ValueError("failure audits cannot source a report")
    documents: dict[str, dict[str, Any]] = {}
    for name, cls in (
        (
            "replay_bundle_json",
            CaseRankingReplayBundleV2 if new else old_audit.CaseRankingReplayBundle,
        ),
        ("lineage_index_json", CaseAuditLineageIndexV2 if new else old_audit.CaseAuditLineageIndex),
        ("sealed_audit_package_json", SealedAuditPackage),
        ("failure_diagnostic_json", EmptyCandidateUniverseFailure),
    ):
        raw = audit_state[name]
        if raw is not None:
            documents[name] = _audit_document(raw, cls)
    documents["payload_json"] = _report_document(
        context_state["payload_json"], FrozenReportPayloadV2 if new else legacy.FrozenReportPayload
    )
    _context_domains(context_state, documents["payload_json"])
    _source_embedded_domains(documents["replay_bundle_json"])
    sealed = documents["sealed_audit_package_json"]
    if (
        context_state["source_audit_sha256"] != audit_state["audit_sha256"]
        or context_state["opaque_case_id"] != audit_state["opaque_case_id"]
        or context_state["experiment_version"] != sealed["experiment_version"]
        or context_state["run_mode"] != sealed["run_mode"]
    ):
        raise ValueError("report context differs from its original sealed source identity")
    legacy._parse_utc_seconds(context_state["frozen_at_utc"], label="report context freeze time")
    # Budget the complete decoded named content, including the new payload.
    _report_snapshot(documents)
    return context_original, audit_original, documents


def require_report_source_read_view(
    *,
    context: SupportedReportContext,
    audit_package: SupportedCaseAudit,
    expected_audit_sha256: str,
    expected_context_sha256: str,
) -> ReportSourceReadView:
    """Replay the exact version pair under both independent external pins."""
    context_original, audit_original, documents = _preflight_report_sources(
        context=context,
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
        expected_context_sha256=expected_context_sha256,
    )
    try:
        context_state, audit_state = context_original[0], audit_original[0]
        for name, document in documents.items():
            expected = context_state[name] if name == "payload_json" else audit_state[name]
            if canonical_json_bytes(document).decode("utf-8") != expected:
                raise ValueError("report source named JSON must be canonical")
        context_bytes = canonical_json_bytes(context_state)
        audit_bytes = canonical_json_bytes(audit_state)
        fresh_context: SupportedReportContext
        fresh_audit: SupportedCaseAudit
        rebuilt: SupportedReportContext
        if type(context) is legacy.FrozenReportContext:
            fresh_context = legacy.FrozenReportContext.model_validate_json(context_bytes)
            fresh_audit = old_audit.ImmutableCaseAuditPackage.model_validate_json(audit_bytes)
            rebuilt = legacy.require_frozen_report_context(
                fresh_context,
                audit_package=fresh_audit,
                expected_audit_sha256=expected_audit_sha256,
            )
        else:
            fresh_context = FrozenReportContextV4.model_validate_json(context_bytes)
            fresh_audit = ImmutableCaseAuditPackageV2.model_validate_json(audit_bytes)
            if not fresh_context.verify():
                raise ValueError("report context failed exact verification")
            rebuilt = freeze_report_context_from_audit_v2(
                audit_package=fresh_audit,
                expected_audit_sha256=expected_audit_sha256,
                frozen_at_utc=context_state["frozen_at_utc"],
            )
        if (
            not hmac.compare_digest(expected_context_sha256, rebuilt.context_sha256)
            or canonical_json_bytes(_report_snapshot(rebuilt)[0]) != context_bytes
        ):
            raise ValueError("report context differs from the independently pinned complete source")
        audit_view = read_case_audit(
            audit_package=fresh_audit, expected_audit_sha256=expected_audit_sha256
        )
        payload = fresh_context.payload
        bundle, lineage = audit_view.replay_bundle, audit_view.lineage_index
        if bundle is None or lineage is None:
            raise ValueError("report requires a successful ranked audit")
        view = object.__new__(ReportSourceReadView)
        values = dict(
            canonical_audit_bytes=audit_bytes,
            canonical_context_bytes=context_bytes,
            source_audit_schema_version=audit_state["schema_version"],
            source_context_schema_version=context_state["schema_version"],
            source_payload_schema_version=payload.schema_version,
            source_ranking_bundle_schema_version=bundle.schema_version,
            source_lineage_schema_version=lineage.schema_version,
            opaque_case_id=audit_state["opaque_case_id"],
            audit_sha256=expected_audit_sha256,
            context_sha256=expected_context_sha256,
            payload_sha256=payload.payload_sha256,
            context=fresh_context,
            payload=payload,
            replay_bundle=bundle,
            lineage_index=lineage,
        )
        for name, value in values.items():
            object.__setattr__(view, name, value)
        return view
    finally:
        try:
            _report_unchanged(context, context_original)
        finally:
            _report_unchanged(audit_package, audit_original)


__all__ = ["ReportSourceReadView", "SupportedReportContext", "require_report_source_read_view"]
