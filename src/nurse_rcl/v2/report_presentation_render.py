"""Static HTML over an independently replayed presentation, with no executable surfaces."""

from __future__ import annotations

import html
import json

from pydantic import BaseModel

from nurse_rcl.v2.case_audit_read import SupportedCaseAudit
from nurse_rcl.v2.opaque_ids import InferenceIdentityManifest
from nurse_rcl.v2.report_composition_proposals import ReportCompositionResponse
from nurse_rcl.v2.report_evidence_schemas import ReportEvidencePolicy, ReportEvidenceProjection
from nurse_rcl.v2.report_mechanical import _guard_report_surfaces
from nurse_rcl.v2.report_mechanical_schemas import ReportCompositionMechanicalValidation
from nurse_rcl.v2.report_presentation import (
    _presentation_local_domains,
    require_report_composition_presentation,
)
from nurse_rcl.v2.report_presentation_schemas import (
    ReportCompositionPresentation,
    ReportPresentationBlock,
)
from nurse_rcl.v2.report_source_read import SupportedReportContext


def _escaped_json(value: BaseModel) -> str:
    return html.escape(
        json.dumps(value.model_dump(mode="json", warnings="error"), ensure_ascii=False, indent=2),
        quote=True,
    )


def _block_html(block: ReportPresentationBlock) -> str:
    # Only safe blocks reach this function; raw checks and original mechanics are never rendered.
    claim = block.claim
    parts = [
        "<article><h3>",
        html.escape(claim.claim_id, quote=True),
        "</h3><p>",
        html.escape(block.surface_status, quote=True),
        " / ",
        html.escape(block.semantic_status, quote=True),
        "</p><p>",
        html.escape(claim.surface_text, quote=True),
        "</p><pre>",
        _escaped_json(block),
        "</pre></article>",
    ]
    if block.surface_status == "canonical":
        parts.insert(0, "<p>Deterministic replacement for blocked model wording</p>")
    return "".join(parts)


@_guard_report_surfaces(
    roots=(
        ("presentation", ReportCompositionPresentation),
        ("validation", ReportCompositionMechanicalValidation),
        ("response", ReportCompositionResponse),
        ("projection", ReportEvidenceProjection),
    ),
    local_domains=_presentation_local_domains,
)
def render_report_composition_html(
    presentation: ReportCompositionPresentation,
    *,
    validation: ReportCompositionMechanicalValidation,
    response: ReportCompositionResponse,
    expected_response_sha256: str,
    projection: ReportEvidenceProjection,
    context: SupportedReportContext,
    audit_package: SupportedCaseAudit,
    expected_audit_sha256: str,
    expected_context_sha256: str,
    identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str,
    policy: ReportEvidencePolicy,
    expected_policy_sha256: str,
) -> str:
    """Replay before any attribute access and escape every dynamic display field."""
    replayed = require_report_composition_presentation(
        presentation,
        validation=validation,
        response=response,
        expected_response_sha256=expected_response_sha256,
        projection=projection,
        context=context,
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
        expected_context_sha256=expected_context_sha256,
        identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
    )
    parts = [
        '<!doctype html><html><head><meta charset="utf-8">'
        "<title>Frozen report presentation</title></head><body>",
        "<section><h2>Frozen foundation</h2><p>canonical / deterministic_render</p>"
        "<table><thead><tr><th>Rank</th><th>Candidate</th>"
        "<th>Total score</th></tr></thead><tbody>",
    ]
    for candidate in replayed.foundation.top_k:
        parts.extend(
            (
                "<tr><td>",
                html.escape(str(candidate.rank), quote=True),
                "</td><td>",
                html.escape(candidate.service_entity_key, quote=True),
                "</td><td>",
                html.escape(candidate.quantized_total_score, quote=True),
                "</td></tr>",
            )
        )
    parts.extend(
        (
            "</tbody></table><pre>",
            _escaped_json(replayed.foundation),
            "</pre></section><section><h2>Model explanations</h2>",
        )
    )
    parts.extend(
        _block_html(block)
        for block in replayed.safe_display_blocks
        if block.claim.statement_type != "recommendation"
    )
    parts.append("</section><section><h2>Proposed recommendations</h2>")
    parts.extend(
        _block_html(block)
        for block in replayed.safe_display_blocks
        if block.claim.statement_type == "recommendation"
    )
    parts.extend(
        (
            "</section><section><h2>Runtime &amp; assurance limits</h2><p>",
            html.escape(replayed.status, quote=True),
            "</p><p>Run mode: ",
            html.escape(replayed.foundation.run_mode, quote=True),
            "</p><p>Blocked model claims: ",
            html.escape(
                str(sum(c.surface_status == "blocked" for c in replayed.claims)), quote=True
            ),
            "; deterministic replacements: ",
            html.escape(str(sum(c.replacement is not None for c in replayed.claims)), quote=True),
            "</p><p>Finite rules cannot establish all natural-language semantics.</p>"
            "<p>Model text remains model_unverified / not_established. "
            "Recommendations are proposed only; no automatic execution is authorized."
            "</p></section></body></html>",
        )
    )
    return "".join(parts)
