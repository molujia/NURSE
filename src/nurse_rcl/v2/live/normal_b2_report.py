"""B2 identity around the shared normal-mode evidence and language constraints."""

from pathlib import Path
from typing import Literal

from nurse_rcl.v2.live.normal_artifacts import NormalAssurance
from nurse_rcl.v2.live.normal_b2_report_prompt import (
    B2_REPORT_EXCHANGE_VERSION,
    B2_REPORT_POLICY,
    _b2_public_for_view,
    prepare_b2_report_request,
)
from nurse_rcl.v2.live.normal_b2_result import NormalB2Evidence
from nurse_rcl.v2.live.normal_queries import _seal
from nurse_rcl.v2.live.normal_report import NormalReportPresentation, _run_report_values
from nurse_rcl.v2.live.normal_report_context import NormalObservationContext, observation_contexts
from nurse_rcl.v2.live.repair_module_timing import _module_span
from nurse_rcl.v2.live.repair_report import _SharedReportFields
from nurse_rcl.v2.live.repair_report_projection import _projection_values, _SharedProjectionFields
from nurse_rcl.v2.live.repair_sources import RepairMeasurementSource
from nurse_rcl.v2.opaque_ids import InferenceIdentityManifest
from nurse_rcl.v2.report_evidence_schemas import ReportEvidencePolicy


class NormalB2ReportProjection(_SharedProjectionFields, NormalAssurance):
    schema_version: Literal["nurse.v2.normal-b2-report-projection/1"] = (
        "nurse.v2.normal-b2-report-projection/1"
    )
    profile: Literal["v2-live-hybrid-normal"] = "v2-live-hybrid-normal"
    source_evidence_sha256: str
    observation_contexts: tuple[NormalObservationContext, ...]


class NormalB2ReportRun(_SharedReportFields, NormalAssurance):
    schema_version: Literal["nurse.v2.normal-b2-report/1"] = "nurse.v2.normal-b2-report/1"
    profile: Literal["v2-live-hybrid-normal"] = "v2-live-hybrid-normal"
    exchange_protocol: Literal["nurse.v2.normal-b2-report-exchange/1"] = B2_REPORT_EXCHANGE_VERSION
    pre_ranking_semantics: Literal[True] = True
    source_evidence_sha256: str
    projection: NormalB2ReportProjection
    presentation: NormalReportPresentation | None = None


def project_normal_b2_evidence(
    evidence: NormalB2Evidence, *, sources: tuple[RepairMeasurementSource, ...] = (),
) -> NormalB2ReportProjection:
    if type(evidence) is not NormalB2Evidence:
        raise ValueError("B2 report requires exact owned B2 evidence")
    with _module_span("report_projection"):
        contexts = observation_contexts(evidence, sources)
        values = _projection_values(
            ranking=evidence.ranking, graph=evidence.graph,
            opaque_case_id=evidence.opaque_case_id, audit_sha256=evidence.result_sha256,
            audit_fields=tuple(sorted(NormalB2Evidence.model_fields)),
            observation_contexts=contexts,
            policy=ReportEvidencePolicy(configuration_version="normal-b2-report-grounding/1",
                                        max_view_bytes=80000, max_detail_facts=64),
            semantic_limits=dict(
                semantic_scope=evidence.semantic_scope, semantic_status=evidence.semantic_status,
                baseline_fallback=evidence.baseline_fallback,
                semantic_failure_code=evidence.failure_code or "none",
                selected_round=evidence.selected_round or 0,
                selected_proposal_count=len(evidence.selected_proposal_ids),
                logical_tools_used=evidence.logical_tools_used, card_count=evidence.card_count,
                verification_count=evidence.verification_count,
                reused_verifications=evidence.reused_count,
                executed_verifications=evidence.executed_count,
                paired_physical_union_count=evidence.physical_union_count,
                model_numeric_authority=False,
                followup_scope="complete_hypothesis_snapshot_no_additional_tools_or_score_effect",
            ),
            public_for_view=_b2_public_for_view, request_for_view=prepare_b2_report_request,
            input_token_limit=B2_REPORT_POLICY.input_token_limit,
        )
        values["source_evidence_sha256"] = values.pop("source_audit_sha256")
        values["observation_contexts"] = tuple(
            contexts[b.source_id] for b in values["source_bindings"]
            if b.source_kind == "normal_source_bound_observation"
        )
        return _seal(NormalB2ReportProjection(**values), "projection_sha256")


def run_normal_b2_report(
    *, evidence: NormalB2Evidence, identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str, state_root: Path, execution_sha256: str,
    code_sha256: str, sources: tuple[RepairMeasurementSource, ...] = (),
) -> NormalB2ReportRun:
    """Private admitted B2 evidence; no whole-case replays and no ranking mutation."""
    if type(evidence) is not NormalB2Evidence:
        raise ValueError("B2 report independent evidence identity differs")
    values = _run_report_values(
        evidence=evidence, identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256,
        state_root=state_root, execution_sha256=execution_sha256, code_sha256=code_sha256,
        sources=sources, project=project_normal_b2_evidence, prepare=prepare_b2_report_request,
        protocol=B2_REPORT_EXCHANGE_VERSION, run_policy=B2_REPORT_POLICY, hybrid=True,
    )
    return _seal(NormalB2ReportRun(**values), "result_sha256")
