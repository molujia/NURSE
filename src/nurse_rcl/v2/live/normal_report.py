"""Evidence-constrained Report Agent without a whole-case replay barrier."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal

from nurse_rcl.agents import ark
from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, ExchangeLimits, encode_json
from nurse_rcl.v2.hermes_exchange import PreparedExchange
from nurse_rcl.v2.inference_leakage import audit_inference_boundary
from nurse_rcl.v2.live import ark_bridge
from nurse_rcl.v2.live.normal_artifacts import NormalAssurance, NormalEvidence
from nurse_rcl.v2.live.normal_queries import _seal
from nurse_rcl.v2.live.normal_report_context import NormalObservationContext, observation_contexts
from nurse_rcl.v2.live.repair_driver import publish_json, report_binding
from nurse_rcl.v2.live.repair_module_timing import _module_span
from nurse_rcl.v2.live.repair_report import _mechanics, _ReportContentFields
from nurse_rcl.v2.live.repair_report_projection import (
    _projection_values,
    _ProjectionContentFields,
    _SharedProjectionFields,
)
from nurse_rcl.v2.live.repair_report_prompt import REPAIR_EXCHANGE_VERSION
from nurse_rcl.v2.live.repair_report_request import RUN_POLICY, prepare_repair_report_request
from nurse_rcl.v2.live.repair_sources import RepairMeasurementSource
from nurse_rcl.v2.live.report_exchange import execute_report_once
from nurse_rcl.v2.live.staged_case import require_plain_path
from nurse_rcl.v2.opaque_ids import InferenceIdentityManifest
from nurse_rcl.v2.report_composition_proposals import parse_report_composition_response
from nurse_rcl.v2.report_evidence_schemas import ReportEvidencePolicy, ReportEvidenceView
from nurse_rcl.v2.report_presentation import _presentation_values
from nurse_rcl.v2.report_presentation_schemas import (
    REPORT_LANGUAGE_RULE_VERSION,
    ReportClaimPresentationDecision,
    ReportPresentationBlock,
)
from nurse_rcl.v2.report_run_schemas import ReportRunPolicy
from nurse_rcl.v2.report_runner import _usage_fields

if TYPE_CHECKING:
    from nurse_rcl.v2.live.normal_b2_result import NormalB2Evidence


class NormalReportProjection(_ProjectionContentFields, NormalAssurance):
    schema_version: Literal[
        "nurse.v2.normal-rcl-report-projection/1", "nurse.v2.normal-rcl-report-projection/2"
    ] = "nurse.v2.normal-rcl-report-projection/2"
    source_evidence_sha256: str
    observation_contexts: tuple[NormalObservationContext, ...] = ()


class NormalReportPresentation(NormalAssurance):
    schema_version: Literal[
        "nurse.v2.normal-rcl-report-presentation/1", "nurse.v2.normal-rcl-report-presentation/2"
    ] = "nurse.v2.normal-rcl-report-presentation/1"
    # Absent in historical /1 JSON; new construction explicitly supplies /2.
    rule_version: Literal[
        "nurse-v2-report-language-rules/1", "nurse-v2-report-language-rules/2"
    ] = "nurse-v2-report-language-rules/1"
    source_evidence_sha256: str
    source_ranking_sha256: str
    projection_sha256: str
    claims: tuple[ReportClaimPresentationDecision, ...]
    safe_display_blocks: tuple[ReportPresentationBlock, ...]
    status: Literal["full", "partial", "deterministic_fallback"]
    presentation_sha256: str


class NormalReportRun(_ReportContentFields, NormalAssurance):
    schema_version: Literal["nurse.v2.normal-rcl-report/1"] = "nurse.v2.normal-rcl-report/1"
    source_evidence_sha256: str
    projection: NormalReportProjection
    presentation: NormalReportPresentation | None = None


def project_normal_evidence(
    evidence: NormalEvidence,
    *,
    sources: tuple[RepairMeasurementSource, ...] = (),
    ops_soft_targets: bool = False,
) -> NormalReportProjection:
    with _module_span("report_projection"):
        # The OPS fact inventory has an independent existing JSON resource
        # envelope. It does not construct old ARK requests or consult their token
        # policy. The final OPS wire separately applies its 32 KiB soft target.
        def admit_ops_view(view: ReportEvidenceView, mandatory: bool) -> bool:
            try:
                raw = encode_json(view.model_dump(mode="json"), ExchangeLimits(
                    max_bytes=262144, max_depth=64, max_nodes=100000,
                ))
            except ExchangeBoundaryError as error:
                if error.code != "json_limit":
                    raise
                return False
            return mandatory or len(raw) <= 32768

        contexts = observation_contexts(evidence, sources)
        values = _projection_values(
            ranking=evidence.ranking,
            graph=evidence.graph,
            opaque_case_id=evidence.opaque_case_id,
            audit_sha256=evidence.result_sha256,
            audit_fields=tuple(sorted(NormalEvidence.model_fields)),
            observation_contexts=contexts,
            policy=ReportEvidencePolicy(
                configuration_version=("ops-formal-evidence-resource/2" if ops_soft_targets
                                       else "normal-rcl-report-grounding/2"),
                max_view_bytes=262144 if ops_soft_targets else 80000,
                max_detail_facts=64,
            ),
            view_admission=admit_ops_view if ops_soft_targets else None,
        )
        values["source_evidence_sha256"] = values.pop("source_audit_sha256")
        values["observation_contexts"] = tuple(
            contexts[b.source_id]
            for b in values["source_bindings"]
            if b.source_kind == "normal_source_bound_observation"
        )
        return _seal(NormalReportProjection(**values), "projection_sha256")


def run_normal_report(
    *,
    evidence: NormalEvidence,
    identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str,
    state_root: Path,
    execution_sha256: str,
    code_sha256: str,
    sources: tuple[RepairMeasurementSource, ...] = (),
) -> NormalReportRun:
    """Private owned-evidence input; the transport is fixed, not caller supplied."""
    if type(evidence) is not NormalEvidence:
        raise ValueError("normal report independent case/manifest identity differs")
    values = _run_report_values(
        evidence=evidence, identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256,
        state_root=state_root, execution_sha256=execution_sha256, code_sha256=code_sha256,
        sources=sources, project=project_normal_evidence,
    )
    return _seal(NormalReportRun(**values), "result_sha256")


def _run_report_values(
    *, evidence: NormalEvidence | NormalB2Evidence,
    identity_manifest: InferenceIdentityManifest, expected_identity_manifest_sha256: str,
    state_root: Path, execution_sha256: str, code_sha256: str,
    sources: tuple[RepairMeasurementSource, ...],
    project: Callable[..., _SharedProjectionFields],
    prepare: Callable[[ReportEvidenceView], tuple[Any, PreparedExchange]] = (
        prepare_repair_report_request
    ),
    protocol: str = REPAIR_EXCHANGE_VERSION, run_policy: ReportRunPolicy = RUN_POLICY,
    hybrid: bool = False,
) -> dict[str, Any]:
    """Shared private report core. Public wrappers fix evidence type and protocol."""
    require_pin(execution_sha256)
    require_pin(code_sha256)
    require_pin(expected_identity_manifest_sha256)
    require_plain_path(state_root)
    if (
        type(identity_manifest) is not (InferenceIdentityManifest)
        or (
            not identity_manifest.verify()
            or identity_manifest.manifest_sha256 != expected_identity_manifest_sha256
            or state_root.name != evidence.opaque_case_id
            or not any(
                r.opaque_case_id == evidence.opaque_case_id and r.dataset == evidence.dataset
                for r in identity_manifest.records
            )
        )
    ):
        raise ValueError("normal report independent case/manifest identity differs")
    started = monotonic()
    projection = project(evidence, sources=sources)
    timings = {"report_projection": monotonic() - started}
    values: dict[str, Any] = dict(
        source_evidence_sha256=evidence.result_sha256,
        source_ranking_sha256=evidence.ranking.result_sha256,
        identity_manifest_sha256=expected_identity_manifest_sha256,
        projection=projection,
        run_policy=run_policy,
        stage_seconds=timings,
        result_sha256="0" * 64,
    )

    def finish(**changes: Any) -> dict[str, Any]:
        return {**values, **changes}

    if projection.view is None:
        return finish(reason_code="report_evidence_unavailable")
    started = monotonic()
    try:
        with _module_span("request_build"):
            audit_inference_boundary(
                projection.view,
                identity_manifest=identity_manifest,
                expected_identity_manifest_sha256=expected_identity_manifest_sha256,
            )
            prompt, request = prepare(projection.view)
        values.update(prompt_payload_json=prompt.payload_json.decode(), prompt_sha256=prompt.sha256)
    except ExchangeBoundaryError as error:
        if error.code != "json_limit":
            raise
        return finish(reason_code="request_limit")
    estimate = len(request.payload_bytes) + 256
    values.update(
        request_hex=request.payload_bytes.hex(),
        request_sha256=request.sha256,
        estimated_input_tokens=estimate,
    )
    timings["report_prepare"] = monotonic() - started
    if estimate > run_policy.input_token_limit:
        return finish(reason_code="input_budget_exceeded")
    if hybrid:
        binding = ark_bridge.LiveRequestBinding(
            opaque_case_id=evidence.opaque_case_id, variant="B2", phase="report",
            input_sha256=evidence.source_descriptor_sha256,
            context_sha256=evidence.result_sha256, prompt_schema_sha256=prompt.sha256,
            request_sha256=request.sha256,
        )
    else:
        binding = report_binding(
            request.payload_bytes, opaque_case_id=evidence.opaque_case_id,
            source_sha256=evidence.source_descriptor_sha256, audit_sha256=evidence.result_sha256,
        )
    publish_json(state_root / "request-binding.json", binding.model_dump(mode="json"))
    publish_json(
        state_root / "report-input.json",
        dict(
            schema_version=("nurse.v2.normal-b2-report-input/1" if hybrid
                            else "nurse.v2.normal-rcl-report-input/1"),
            source_evidence_sha256=evidence.result_sha256,
            projection=projection.model_dump(mode="json"),
            request_hex=request.payload_bytes.hex(),
            request_sha256=request.sha256,
        ),
    )

    def complete(payload: bytes) -> bytes:
        if payload != request.payload_bytes:
            raise ValueError("normal report outbound differs from owned projection")
        with ark_bridge.LiveArkBridge(
            root=state_root / "ark",
            lock_path=Path(
                "/home/wangrunzhou/NURSE/development/live-rcl-20260918/locks/ark-in-flight.lock"
            ),
            settings=ark.load_ark_settings_from_key_file(
                Path("/home/wangrunzhou/.config/nurse/ark.env")
            ),
            code_sha256=code_sha256,
            config_sha256=execution_sha256,
            authorized_case_ids=(evidence.opaque_case_id,),
            request_protocol=protocol,
        ) as bridge:
            try:
                return bridge.complete(binding, payload)
            finally:
                receipt = bridge.receipt(binding)
                pin = sha256_hex(canonical_json_bytes(receipt))
                publish_json(state_root / f"ark-receipt-{pin[:16]}.json", receipt)

    started = monotonic()
    attempt = execute_report_once(request, complete, protocol=protocol)
    timings["report_exchange"] = monotonic() - started
    values.update(
        exchange_attempted=True,
        exchange_status=attempt.status,
        exchange_failure_code=attempt.failure_code,
        raw_response_hex=attempt.raw_response.hex() if attempt.raw_response is not None else None,
        response_sha256=attempt.response_sha256,
        received_bytes=attempt.received_bytes,
        **_usage_fields(attempt.raw_response, request),
    )
    if attempt.response is None:
        return finish(reason_code=attempt.failure_code)
    started = monotonic()
    try:
        with _module_span("explanation_checks"):
            response = parse_report_composition_response(attempt.response.output_json)
    except ExchangeBoundaryError as error:
        return finish(structural_status="invalid", reason_code=error.code)
    response_pin = sha256_hex(canonical_json_bytes(response.model_dump(mode="json")))
    values.update(
        structural_status="valid", parsed_response=response, parsed_response_sha256=response_pin
    )
    if values["usage_status"] == "reported":
        for field, limit in (
            ("input", run_policy.input_token_limit),
            ("output", run_policy.output_token_limit),
        ):
            if values[f"actual_{field}_tokens"] > limit:
                return finish(reason_code=f"reported_{field}_budget_exceeded")
    with _module_span("explanation_checks"):
        mechanical = _mechanics(response, projection, response_pin)
        if not mechanical.verify():
            raise ValueError("normal report mechanical integrity failed")
        presentation = _seal(
            NormalReportPresentation(
                schema_version="nurse.v2.normal-rcl-report-presentation/2",
                rule_version=REPORT_LANGUAGE_RULE_VERSION,
                source_evidence_sha256=evidence.result_sha256,
                source_ranking_sha256=evidence.ranking.result_sha256,
                projection_sha256=projection.projection_sha256,
                **_presentation_values(mechanical, response, projection.view),
                presentation_sha256="0" * 64,
            ),
            "presentation_sha256",
        )
    timings["report_checks"] = monotonic() - started
    return finish(
        mechanical_validation=mechanical,
        presentation=presentation,
        status=presentation.status,
        reason_code={
            "full": None,
            "partial": "partially_blocked",
            "deterministic_fallback": "no_deliverable_claims",
        }[presentation.status],
    )
