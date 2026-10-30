"""Source/complete-score replay through one frozen-ranking Report Agent exchange.

No serialized report grants authority. Current repair outputs use their own
identities and reuse only the embedded HERMES exchange and pure claim checks.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from time import monotonic
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.safety import exact_field, plain, representation, require_pin
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.inference_leakage import audit_inference_boundary
from nurse_rcl.v2.live import repair_raw_audit as raw
from nurse_rcl.v2.live import repair_safety
from nurse_rcl.v2.live.repair_module_timing import _module_call, _module_span
from nurse_rcl.v2.live.repair_report_projection import (
    POLICY,
    RepairReportProjection,
    _project_verified,
    _RepairProjectionFields,
    _seal,
    _SharedProjectionFields,
)
from nurse_rcl.v2.live.repair_report_prompt import (
    REPAIR_EXCHANGE_VERSION,
)
from nurse_rcl.v2.live.repair_report_request import RUN_POLICY, prepare_repair_report_request
from nurse_rcl.v2.live.report_exchange import execute_report_once
from nurse_rcl.v2.opaque_ids import InferenceIdentityManifest
from nurse_rcl.v2.report_audit_json import _measure_preflighted
from nurse_rcl.v2.report_composition_proposals import (
    ReportCompositionResponse,
    parse_report_composition_response,
)
from nurse_rcl.v2.report_mechanical_checks import _validate_claim
from nurse_rcl.v2.report_mechanical_schemas import ReportCompositionMechanicalValidation
from nurse_rcl.v2.report_presentation import _presentation_values
from nurse_rcl.v2.report_presentation_schemas import (
    ReportClaimPresentationDecision,
    ReportPresentationBlock,
)
from nurse_rcl.v2.report_run_schemas import ReportRunPolicy
from nurse_rcl.v2.report_runner import _usage_fields
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

REPAIR_REPORT_SCHEMA_VERSION: Final = "nurse.v2.live-repair-report/1"


class RepairReportPresentation(StrictFrozenModel):
    schema_version: Literal["nurse.v2.live-repair-report-presentation/1"] = (
        "nurse.v2.live-repair-report-presentation/1"
    )
    source_audit_sha256: str
    source_ranking_sha256: str
    projection_sha256: str
    claims: tuple[ReportClaimPresentationDecision, ...]
    safe_display_blocks: tuple[ReportPresentationBlock, ...]
    status: Literal["full", "partial", "deterministic_fallback"]
    presentation_sha256: str


class _SharedReportFields(StrictFrozenModel):
    dynamic_verification: Literal[True] = True
    source_ranking_sha256: str
    identity_manifest_sha256: str
    run_policy: ReportRunPolicy
    explainability_semantic_status: Literal["not_established"] = "not_established"
    status: Literal["full", "partial", "deterministic_fallback"] = "deterministic_fallback"
    reason_code: str | None = None
    prompt_payload_json: str | None = Field(default=None, max_length=524288)
    prompt_sha256: str | None = None
    request_hex: str | None = Field(default=None, max_length=524288)
    request_sha256: str | None = None
    estimated_input_tokens: int | None = None
    exchange_attempted: bool = False
    exchange_status: str = "not_attempted"
    exchange_failure_code: str | None = None
    raw_response_hex: str | None = Field(default=None, max_length=524288)
    response_sha256: str | None = None
    received_bytes: int | None = None
    usage_status: Literal["missing", "reported", "invalid"] = "missing"
    raw_usage_json: str | None = Field(default=None, max_length=262144)
    actual_input_tokens: int | None = None
    actual_output_tokens: int | None = None
    cost_status: Literal["unknown"] = "unknown"
    structural_status: Literal["not_checked", "valid", "invalid"] = "not_checked"
    parsed_response: ReportCompositionResponse | None = None
    parsed_response_sha256: str | None = None
    mechanical_validation: ReportCompositionMechanicalValidation | None = None
    stage_seconds: dict[str, float]
    result_sha256: str


class _ReportContentFields(_SharedReportFields):
    profile: Literal["v2-live-evidence-repair"] = "v2-live-evidence-repair"
    exchange_protocol: Literal["nurse.v2.live-repair-report-exchange/1"] = REPAIR_EXCHANGE_VERSION
    pre_ranking_semantics: Literal[False] = False


class _RepairReportFields(_ReportContentFields):
    source_audit_sha256: str
    presentation: RepairReportPresentation | None = None


class RepairReportRun(_RepairReportFields):
    schema_version: Literal["nurse.v2.live-repair-report/1"] = REPAIR_REPORT_SCHEMA_VERSION
    projection: RepairReportProjection
    interpretability_status: Literal["source_and_score_replayed"] = "source_and_score_replayed"


_REPORT_ROOTS = (
    RepairReportRun,
    RepairReportPresentation,
    RepairReportProjection,
    ReportCompositionResponse,
    ReportCompositionMechanicalValidation,
    InferenceIdentityManifest,
    type(POLICY),
    ReportRunPolicy,
)


def _snapshot(value: BaseModel) -> tuple[bytes, str]:
    return _snapshot_surface(value, _REPORT_ROOTS, RepairReportRun)


def _snapshot_surface(
    value: BaseModel, roots: tuple[type[BaseModel], ...], run_type: type[BaseModel]
) -> tuple[bytes, str]:
    if type(value) not in roots:
        raise ValueError("exact registered repair report surface required")
    preflight_exact_pydantic_tree(
        value,
        root_model_types=roots,
        label="repair report",
        max_nodes=500000,
        max_depth=64,
        max_string_length=None,
        provenance_id_limit=None,
    )
    _measure_preflighted(value, root_model_types=roots)
    document: object
    if type(value) is run_type:
        state = object.__getattribute__(value, "__dict__")
        for name, item in state.items():
            if name == "stage_seconds":
                if type(item) is not dict or any(
                    type(k) is not str or type(v) is not float for k, v in item.items()
                ):
                    raise ValueError(
                        "repair report stage timings require exact string/float mapping"
                    )
            elif not exact_field(item, run_type.model_fields[name].annotation):
                raise ValueError("repair report field requires exact declared type")
        document = {name: plain(item) for name, item in state.items()}
    else:
        document = plain(value)
    return (canonical_json_bytes(document), sha256_hex(canonical_json_bytes(representation(value))))


def _unchanged(value: BaseModel, before: tuple[bytes, str]) -> None:
    try:
        same = _snapshot(value) == before
    except (AttributeError, KeyError, TypeError, ValueError, RecursionError):
        same = False
    if not same:
        raise ValueError("repair report content or representation changed")


@_module_call("explanation_checks", role="verification")
def _mechanics(
    response: ReportCompositionResponse, projection: _SharedProjectionFields, response_pin: str
) -> ReportCompositionMechanicalValidation:
    view = projection.view
    assert view is not None
    # No mitigation template is authorized; the common checker receives no
    # legacy foundation, so it cannot borrow one from a different profile.
    claims = tuple(
        _validate_claim(c, view=view, bindings=projection.source_bindings, foundation=None)
        for c in response.claims
    )
    return _seal(
        ReportCompositionMechanicalValidation(
            opaque_case_id=view.opaque_case_id,
            context_sha256=view.context_sha256,
            projection_sha256=projection.projection_sha256,
            view_sha256=view.view_sha256,
            policy_sha256=view.policy_sha256,
            response_sha256=response_pin,
            mandatory_fact_refs=view.mandatory_fact_refs,
            claims=claims,
            passed_count=sum(c.mechanical_status == "passed" for c in claims),
            rejected_count=sum(c.mechanical_status == "rejected" for c in claims),
            validation_sha256="0" * 64,
        ),
        "validation_sha256",
    )


def _compose(
    *,
    audit: raw.RepairRawAudit,
    projection: RepairReportProjection,
    identity_manifest: InferenceIdentityManifest,
    identity_pin: str,
    complete: Callable[[bytes], bytes],
    keep: Callable[[Any], Any],
    timings: dict[str, float],
) -> RepairReportRun:
    values = _compose_values(
        audit_sha256=audit.result_sha256,
        ranking_sha256=audit.ranking.result_sha256,
        projection=projection,
        identity_manifest=identity_manifest,
        identity_pin=identity_pin,
        complete=complete,
        keep=keep,
        timings=timings,
    )
    return cast(RepairReportRun, keep(_seal(RepairReportRun(**values), "result_sha256")))


def _compose_values(
    *,
    audit_sha256: str,
    ranking_sha256: str,
    projection: _RepairProjectionFields,
    identity_manifest: InferenceIdentityManifest,
    identity_pin: str,
    complete: Callable[[bytes], bytes],
    keep: Callable[[Any], Any],
    timings: dict[str, float],
) -> dict[str, Any]:
    values: dict[str, Any] = dict(
        source_audit_sha256=audit_sha256,
        source_ranking_sha256=ranking_sha256,
        identity_manifest_sha256=identity_pin,
        projection=projection,
        run_policy=RUN_POLICY,
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
                expected_identity_manifest_sha256=identity_pin,
            )
            prompt, request = prepare_repair_report_request(projection.view)
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
    if estimate > RUN_POLICY.input_token_limit:
        return finish(reason_code="input_budget_exceeded")
    started = monotonic()
    attempt = execute_report_once(request, complete, protocol=REPAIR_EXCHANGE_VERSION)
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
        with _module_span("explanation_checks", role="verification"):
            response = keep(parse_report_composition_response(attempt.response.output_json))
    except ExchangeBoundaryError as error:
        return finish(structural_status="invalid", reason_code=error.code)
    response_pin = sha256_hex(canonical_json_bytes(response.model_dump(mode="json")))
    values.update(
        structural_status="valid", parsed_response=response, parsed_response_sha256=response_pin
    )
    if values["usage_status"] == "reported":
        for field, limit in (
            ("input", RUN_POLICY.input_token_limit),
            ("output", RUN_POLICY.output_token_limit),
        ):
            if values[f"actual_{field}_tokens"] > limit:
                return finish(reason_code=f"reported_{field}_budget_exceeded")
    mechanical = keep(_mechanics(response, projection, response_pin))
    if not mechanical.verify():
        raise ValueError("repair report mechanical integrity failed")
    presentation = keep(
        _seal(
            RepairReportPresentation(
                source_audit_sha256=audit_sha256,
                source_ranking_sha256=ranking_sha256,
                projection_sha256=projection.projection_sha256,
                **_presentation_values(mechanical, response, projection.view),
                presentation_sha256="0" * 64,
            ),
            "presentation_sha256",
        )
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


def run_repair_report(
    *,
    audit: raw.RepairRawAudit,
    expected_audit_sha256: str,
    expected_session_sha256: str,
    identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str,
    complete: Callable[[bytes], bytes],
    progress: Callable[[dict[str, Any]], None] | None = None,
    **raw_arguments: Any,
) -> RepairReportRun:
    """Read all original sources and replay full scores before one logical exchange."""
    started = monotonic()
    if (
        type(audit) is not raw.RepairRawAudit
        or type(identity_manifest) is not InferenceIdentityManifest
    ):
        raise ValueError("repair report requires exact current audit and identity manifest")
    for pin in (expected_audit_sha256, expected_session_sha256, expected_identity_manifest_sha256):
        require_pin(pin)
    with ExitStack() as guards:
        original = repair_safety.snapshot(audit, (raw.RepairRawAudit,))
        guards.callback(repair_safety.require_unchanged, audit, (raw.RepairRawAudit,), original)

        def keep(value: Any) -> Any:
            guards.callback(_unchanged, value, _snapshot(value))
            return value

        keep(identity_manifest)
        keep(POLICY)
        keep(RUN_POLICY)
        manifest = keep(
            InferenceIdentityManifest.model_validate_json(
                canonical_json_bytes(plain(identity_manifest))
            )
        )
        if (
            not manifest.verify()
            or manifest.manifest_sha256 != expected_identity_manifest_sha256
            or not any(
                r.opaque_case_id == audit.opaque_case_id and r.dataset == audit.dataset
                for r in manifest.records
            )
        ):
            raise ValueError("repair report independent identity pin or dataset mismatch")
        with raw._replayed_raw_audit(
            expected_session_sha256=expected_session_sha256, **raw_arguments
        ) as fresh:
            if (
                fresh.result_sha256 != expected_audit_sha256
                or repair_safety.canonical(fresh, (raw.RepairRawAudit,)) != original[0]
            ):
                raise ValueError("repair report differs from raw source and full score replay")
            timings = {"source_replay": monotonic() - started}
            step = monotonic()
            projection = keep(_project_verified(fresh))
            timings["report_projection"] = monotonic() - step
            if progress:
                progress(
                    dict(
                        phase="repair_report_projection_ready",
                        status=projection.status,
                        mandatory_view_bytes=projection.mandatory_view_bytes,
                    )
                )
            result = _compose(
                audit=fresh,
                projection=projection,
                identity_manifest=manifest,
                identity_pin=expected_identity_manifest_sha256,
                complete=complete,
                keep=keep,
                timings=timings,
            )
            finished = monotonic()
    # Exit validation is part of the recorded total, never skipped on exceptions.
    return _seal(
        result.model_copy(
            update={
                "stage_seconds": {
                    **result.stage_seconds,
                    "source_exit_checks": monotonic() - finished,
                    "total": monotonic() - started,
                }
            }
        ),
        "result_sha256",
    )
