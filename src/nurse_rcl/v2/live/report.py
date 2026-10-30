"""Source-replayed bootstrap adapter to the shared Report Agent.

Live descriptors explicitly bind current-version sources, not legacy audit schemas.
Private attribute views only connect pure reporting cores inside the synchronous
source/score replay. They never escape as trust tokens or bypass public readers.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from time import monotonic
from typing import Any, Literal, Protocol, TypeVar, cast

from pydantic import BaseModel

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2 import deterministic_report, report_context, report_evidence
from nurse_rcl.v2 import case_scoring_selection as legacy
from nurse_rcl.v2 import report_mechanical, report_presentation, report_prompt
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.hermes_exchange import execute_once, prepare_exchange
from nurse_rcl.v2.inference_leakage import audit_inference_boundary
from nurse_rcl.v2.live import bootstrap_audit, bootstrap_scoring, scoring_selection
from nurse_rcl.v2.live import report_projection as grouped_projection
from nurse_rcl.v2.live import report_prompt as compact_prompt
from nurse_rcl.v2.live.report_projection import LiveGroupedReportProjection
from nurse_rcl.v2.live.report_exchange import (
    GUIDED_REPORT_CONFIGURATION_VERSION, GUIDED_REPORT_EXCHANGE_VERSION,
    INDEXED_REPORT_CONFIGURATION_VERSION, INDEXED_REPORT_EXCHANGE_VERSION,
    REPORT_CONFIGURATION_VERSION, REPORT_EXCHANGE_VERSION,
    execute_report_once, prepare_report_exchange,
)
from nurse_rcl.v2.live.report_guidance import LiveGuidedPrompt, guide_prompt
from nurse_rcl.v2.live.report_index import LiveIndexedPrompt, index_prompt
from nurse_rcl.v2.live.scoring import LIVE_SCORING_MAX_NODES
from nurse_rcl.v2.no_role_compilation import _representation
from nurse_rcl.v2.opaque_ids import InferenceIdentityManifest
from nurse_rcl.v2.report_composition_proposals import (
    ReportCompositionResponse, parse_report_composition_response,
)
from nurse_rcl.v2.report_evidence_schemas import ReportEvidencePolicy, ReportEvidenceProjection
from nurse_rcl.v2.report_mechanical_schemas import ReportCompositionMechanicalValidation
from nurse_rcl.v2.report_presentation_schemas import ReportCompositionPresentation
from nurse_rcl.v2.report_run_schemas import ReportRunPolicy
from nurse_rcl.v2.report_runner import _runtime_local_domains, _usage_fields
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

_Model = TypeVar('_Model', bound=BaseModel)


class _Keep(Protocol):
    def __call__(self, value: _Model, /) -> _Model: ...


class LiveReportDescriptor(StrictFrozenModel):
    """The actual report payload is a source reference, not a second full case copy."""
    schema_version: Literal['nurse.v2.live-bootstrap-report-source/1'] = 'nurse.v2.live-bootstrap-report-source/1'
    profile: Literal['v2-live-bootstrap'] = 'v2-live-bootstrap'
    dynamic_verification: Literal[False] = False
    pre_ranking_semantics: Literal[False] = False
    opaque_case_id: str
    source_audit_sha256: str
    source_ranking_result_sha256: str
    source_top_k_sha256: str
    independent_pins: dict[str, str]
    degradation_reason_codes: tuple[str, ...] = (
        'bootstrap_dynamic_verification_disabled', 'bootstrap_pre_ranking_semantics_disabled',
        'unknown_no_verified_onset',
    )


class _ReportRunFields(StrictFrozenModel):
    profile: Literal['v2-live-bootstrap'] = 'v2-live-bootstrap'
    dynamic_verification: Literal[False] = False
    pre_ranking_semantics: Literal[False] = False
    descriptor: LiveReportDescriptor
    source_ranking_result_sha256: str
    run_policy: ReportRunPolicy
    status: Literal['full', 'partial', 'deterministic_fallback'] = 'deterministic_fallback'
    reason_code: str | None = None
    interpretability_status: Literal['source_and_score_replayed'] = 'source_and_score_replayed'
    explainability_semantic_status: Literal['not_established'] = 'not_established'
    prompt_payload_json: str | None = None
    prompt_sha256: str | None = None
    request_hex: str | None = None
    request_sha256: str | None = None
    estimated_input_tokens: int | None = None
    exchange_attempted: bool = False
    exchange_status: str = 'not_attempted'
    exchange_failure_code: str | None = None
    raw_response_hex: str | None = None
    response_sha256: str | None = None
    received_bytes: int | None = None
    usage_status: Literal['missing', 'reported', 'invalid'] = 'missing'
    raw_usage_json: str | None = None
    actual_input_tokens: int | None = None
    actual_output_tokens: int | None = None
    cost_status: Literal['unknown'] = 'unknown'
    structural_status: Literal['not_checked', 'valid', 'invalid'] = 'not_checked'
    parsed_response: ReportCompositionResponse | None = None
    parsed_response_sha256: str | None = None
    mechanical_validation: ReportCompositionMechanicalValidation | None = None
    presentation: ReportCompositionPresentation | None = None
    stage_seconds: dict[str, float]


class LiveReportRun(_ReportRunFields):
    """Historical ungrouped record; never relabeled as compact-authorized output."""
    schema_version: Literal['nurse.v2.live-bootstrap-report/1'] = 'nurse.v2.live-bootstrap-report/1'
    projection: ReportEvidenceProjection


class LiveGroupedReportRun(_ReportRunFields):
    schema_version: Literal['nurse.v2.live-bootstrap-report/2'] = 'nurse.v2.live-bootstrap-report/2'
    projection: LiveGroupedReportProjection


class LiveNoThinkingReportRun(_ReportRunFields):
    schema_version: Literal['nurse.v2.live-bootstrap-report/3'] = 'nurse.v2.live-bootstrap-report/3'
    exchange_protocol: Literal['nurse.v2.live-report-disabled-exchange/1'] = 'nurse.v2.live-report-disabled-exchange/1'
    projection: LiveGroupedReportProjection


class LiveGuidedReportRun(_ReportRunFields):
    schema_version: Literal['nurse.v2.live-bootstrap-report/4'] = 'nurse.v2.live-bootstrap-report/4'
    exchange_protocol: Literal['nurse.v2.live-report-disabled-exchange/2'] = 'nurse.v2.live-report-disabled-exchange/2'
    projection: LiveGroupedReportProjection


class LiveIndexedReportRun(_ReportRunFields):
    schema_version: Literal['nurse.v2.live-bootstrap-report/5'] = 'nurse.v2.live-bootstrap-report/5'
    exchange_protocol: Literal['nurse.v2.live-report-disabled-exchange/3'] = 'nurse.v2.live-report-disabled-exchange/3'
    projection: LiveGroupedReportProjection


CurrentReportRun = LiveGroupedReportRun | LiveNoThinkingReportRun | LiveGuidedReportRun | LiveIndexedReportRun


@dataclass(frozen=True, slots=True)
class _PayloadView:
    """Read-only alias map over fresh sources, never a legacy FrozenReportPayload."""
    sources: scoring_selection.LiveScoringSources
    ranking: bootstrap_scoring.LiveBootstrapScoring
    descriptor: LiveReportDescriptor
    hypotheses: tuple[report_context.ReportHypothesisFact, ...]
    competition_deltas: tuple[report_context.ReportCompetitionDelta, ...]
    payload_sha256: str

    @property
    def predicates(self) -> Any:
        return self.sources.binding.predicates

    @property
    def degradation_reason_codes(self) -> tuple[str, ...]:
        return self.descriptor.degradation_reason_codes

    @property
    def role_compilation(self) -> _PayloadView:
        return self

    @property
    def assertions(self) -> Any:
        return self.ranking.evidence.role_assertions

    @property
    def family_aggregation(self) -> _PayloadView:
        return self

    @property
    def candidates(self) -> Any:
        return self.ranking.evidence.family_candidates

    @property
    def scores(self) -> Any:
        return self.ranking.scores

    @property
    def top_k(self) -> Any:
        return self.ranking.top_k

    @property
    def ranking_result(self) -> bootstrap_scoring.LiveBootstrapScoring:
        return self.ranking

    @property
    def entity_registry(self) -> Any:
        return self.sources.envelope.entity_registry

    @property
    def observations(self) -> Any:
        return tuple(row.observation for row in self.entity_registry.resolutions)

    @property
    def event_groups(self) -> Any:
        return self.sources.envelope.event_grouping.event_groups


@dataclass(frozen=True, slots=True)
class _ContextView:
    payload: _PayloadView
    context_sha256: str
    payload_sha256: str
    source_audit_sha256: str
    source_ranking_result_sha256: str
    source_top_k_sha256: str
    opaque_case_id: str
    # Complete execution of this explicitly named profile, NOT a B2 claim.
    run_mode: Literal['full'] = 'full'


def _preflight(value: object, *, max_nodes: int = LIVE_SCORING_MAX_NODES) -> object:
    preflight_exact_pydantic_tree(value, root_model_types=(*scoring_selection._ROOTS,
        bootstrap_scoring.LiveBootstrapContext, bootstrap_scoring.LiveBootstrapScoring,
        InferenceIdentityManifest, ReportEvidencePolicy, ReportRunPolicy, LiveReportRun,
        LiveGroupedReportRun, LiveNoThinkingReportRun, LiveGuidedReportRun, LiveIndexedReportRun, LiveGroupedReportProjection),
        label='live report', max_nodes=max_nodes, max_depth=64,
        max_string_length=None, provenance_id_limit=None)
    scoring_selection._measure(value)
    return legacy._plain(value)


def _small_snapshot(value: BaseModel) -> object:
    if type(value) in (LiveReportRun, LiveGroupedReportRun, LiveNoThinkingReportRun, LiveGuidedReportRun, LiveIndexedReportRun, LiveGroupedReportProjection, LiveReportDescriptor):
        return canonical_json_bytes(_preflight(value, max_nodes=500_000)), _representation(value)
    return deterministic_report._report_input_snapshot(value, model_type=type(value),
        max_nodes=500_000, local_domains=_runtime_local_domains)


def _compose(*, ranking: bootstrap_scoring.LiveBootstrapScoring,
        descriptor: LiveReportDescriptor, projection: LiveGroupedReportProjection,
        identity_manifest: InferenceIdentityManifest, identity_pin: str,
        run_policy: ReportRunPolicy, complete: Callable[[bytes], bytes],
        keep: _Keep, stage_seconds: dict[str, float]) -> CurrentReportRun:
    indexed = run_policy.configuration_version == INDEXED_REPORT_CONFIGURATION_VERSION
    guided = indexed or run_policy.configuration_version == GUIDED_REPORT_CONFIGURATION_VERSION
    disabled = guided or run_policy.configuration_version == REPORT_CONFIGURATION_VERSION
    record_type = LiveIndexedReportRun if indexed else (LiveGuidedReportRun if guided else (
        LiveNoThinkingReportRun if disabled else LiveGroupedReportRun))
    protocol = INDEXED_REPORT_EXCHANGE_VERSION if indexed else (
        GUIDED_REPORT_EXCHANGE_VERSION if guided else REPORT_EXCHANGE_VERSION)
    values: dict[str, Any] = dict(descriptor=descriptor,
        source_ranking_result_sha256=ranking.result_sha256, projection=projection,
        run_policy=run_policy, stage_seconds=stage_seconds)
    if projection.view is None:
        return keep(record_type(**values, reason_code='report_evidence_unavailable'))
    started = monotonic()
    try:
        base_prompt = compact_prompt._prompt_for_verified_projection(projection,
            identity_manifest=identity_manifest, expected_identity_manifest_sha256=identity_pin)
        prompt: compact_prompt.LiveCompactPrompt | LiveGuidedPrompt | LiveIndexedPrompt = (
            index_prompt(guide_prompt(base_prompt)) if indexed else (
                guide_prompt(base_prompt) if guided else base_prompt))
        values.update(prompt_payload_json=prompt.payload_json.decode(), prompt_sha256=prompt.sha256)
        request = (prepare_report_exchange(messages=prompt.messages(), protocol=protocol) if disabled else
            prepare_exchange(model=run_policy.model, messages=prompt.messages(),
                max_output_tokens=run_policy.output_token_limit, temperature=run_policy.temperature))
    except ExchangeBoundaryError as error:
        if error.code != 'json_limit':
            raise
        return keep(record_type(**values, reason_code='request_limit'))
    estimate = len(request.payload_bytes) + 256
    values.update(request_hex=request.payload_bytes.hex(), request_sha256=request.sha256,
        estimated_input_tokens=estimate)
    stage_seconds['report_prepare'] = monotonic() - started
    if estimate > run_policy.input_token_limit:
        return keep(record_type(**values, reason_code='input_budget_exceeded'))
    started = monotonic()
    attempt = (execute_report_once(request, complete, protocol=protocol) if disabled else
        execute_once(request, complete))
    stage_seconds['report_exchange'] = monotonic() - started
    values.update(exchange_attempted=True, exchange_status=attempt.status,
        exchange_failure_code=attempt.failure_code,
        raw_response_hex=attempt.raw_response.hex() if attempt.raw_response is not None else None,
        response_sha256=attempt.response_sha256, received_bytes=attempt.received_bytes)
    values.update(_usage_fields(attempt.raw_response, request))
    if attempt.response is None:
        return keep(record_type(**values, reason_code=attempt.failure_code))
    started = monotonic()
    try:
        response = keep(parse_report_composition_response(attempt.response.output_json))
    except ExchangeBoundaryError as error:
        return keep(record_type(**values, structural_status='invalid', reason_code=error.code))
    response_pin = sha256_hex(report_prompt._bytes(response))
    values.update(structural_status='valid', parsed_response=response,
        parsed_response_sha256=response_pin)
    if values.get('usage_status') == 'reported':
        for count, limit, reason in (
            (values['actual_input_tokens'], run_policy.input_token_limit, 'reported_input_budget_exceeded'),
            (values['actual_output_tokens'], run_policy.output_token_limit, 'reported_output_budget_exceeded'),
        ):
            if count > limit:
                return keep(record_type(**values, reason_code=reason))
    # Same pure rules over the fresh decoded fact interface; old public readers
    # still reject this explicit live projection and cannot consume its authority.
    mechanical = keep(report_mechanical._validate_verified_composition(response, cast(Any, projection), response_pin))
    presentation = keep(report_presentation._present_verified_composition(mechanical, response, cast(Any, projection)))
    if not mechanical.verify() or not presentation.verify():
        raise ValueError('live report computed mechanics or presentation failed integrity')
    stage_seconds['report_checks'] = monotonic() - started
    values.update(mechanical_validation=mechanical, presentation=presentation,
        status=presentation.status, reason_code={
            'full': None, 'partial': 'partially_blocked',
            'deterministic_fallback': 'no_deliverable_claims'}[presentation.status])
    return keep(record_type(**values))


def run_live_bootstrap_report(*, audit: bootstrap_audit.LiveBootstrapAudit,
        expected_audit_sha256: str, expected_source_records_sha256: str,
        expected_rule_set_sha256: str, expected_binding_sha256: str,
        expected_selection_sha256: str, expected_context_sha256: str,
        identity_manifest: InferenceIdentityManifest, expected_identity_manifest_sha256: str,
        evidence_policy: ReportEvidencePolicy, expected_evidence_policy_sha256: str,
        run_policy: ReportRunPolicy, expected_run_policy_sha256: str,
        complete: Callable[[bytes], bytes],
        progress: Callable[[dict[str, Any]], None] | None = None) -> CurrentReportRun:
    """Recompute all source/score authority before one bounded logical report call."""
    started = monotonic()
    pins = dict(expected_audit_sha256=expected_audit_sha256,
        expected_source_records_sha256=expected_source_records_sha256,
        expected_rule_set_sha256=expected_rule_set_sha256,
        expected_binding_sha256=expected_binding_sha256,
        expected_selection_sha256=expected_selection_sha256,
        expected_context_sha256=expected_context_sha256)
    for pin in (*pins.values(), expected_identity_manifest_sha256,
                expected_evidence_policy_sha256, expected_run_policy_sha256):
        scoring_selection._pin(pin)
    supplied = (identity_manifest, evidence_policy, run_policy)
    if tuple(type(item) for item in supplied) != (
            InferenceIdentityManifest, ReportEvidencePolicy, ReportRunPolicy):
        raise ValueError('live report requires exact manifest and report policies')
    # The complete source remains one copy, with small policy roots in the same budget.
    _preflight((bootstrap_audit._values(audit), supplied))
    retained: list[tuple[BaseModel, object]] = []
    def keep(value: Any) -> Any:
        retained.append((value, _small_snapshot(value)))
        return value
    for item in supplied:
        keep(item)
    try:
        manifest, policy, runtime = tuple(keep(type(item).model_validate_json(
            report_prompt._bytes(item))) for item in supplied)
        if (manifest.manifest_sha256 != expected_identity_manifest_sha256
                or policy.sha256 != expected_evidence_policy_sha256
                or runtime.sha256 != expected_run_policy_sha256):
            raise ValueError('live report independent manifest or policy pin mismatch')
        if runtime.configuration_version in (REPORT_CONFIGURATION_VERSION, GUIDED_REPORT_CONFIGURATION_VERSION,
                                             INDEXED_REPORT_CONFIGURATION_VERSION):
            from nurse_rcl.agents.ark import ARK_MODEL
            if (policy.configuration_version != runtime.configuration_version
                    or runtime.model != ARK_MODEL or runtime.temperature != 0
                    or runtime.input_token_limit != 100000 or runtime.output_token_limit != 4096
                    or policy.max_view_bytes != 80000 or policy.max_detail_facts != 64):
                raise ValueError('disabled report configuration differs from registered limits')
        timings: dict[str, float] = {}
        callback_finished = [started]
        def consume(fresh: scoring_selection.LiveScoringSources,
                    ranking: bootstrap_scoring.LiveBootstrapScoring) -> CurrentReportRun:
            timings['source_replay'] = monotonic() - started
            audit_inference_boundary((fresh.envelope.entity_registry, fresh.envelope.candidate_universe),
                identity_manifest=manifest, expected_identity_manifest_sha256=expected_identity_manifest_sha256)
            descriptor = keep(LiveReportDescriptor(opaque_case_id=ranking.opaque_case_id,
                source_audit_sha256=expected_audit_sha256,
                source_ranking_result_sha256=ranking.result_sha256,
                source_top_k_sha256=ranking.top_k.ranking_sha256, independent_pins={**pins,
                    'identity': expected_identity_manifest_sha256,
                    'evidence_policy': expected_evidence_policy_sha256,
                    'run_policy': expected_run_policy_sha256}))
            digest = sha256_hex(report_prompt._bytes(descriptor))
            facts = report_context._report_hypothesis_facts(ranking.opaque_case_id, fresh.guard_results)
            deltas = report_context._build_competition_deltas(cast(Any, ranking), facts)
            for item in (*facts, *deltas):
                keep(item)
            payload = _PayloadView(fresh, ranking, descriptor, facts, deltas, digest)
            context = _ContextView(payload, digest, digest, expected_audit_sha256,
                ranking.result_sha256, ranking.top_k.ranking_sha256, ranking.opaque_case_id)
            aliases = tuple((view, tuple(object.__getattribute__(view, f.name) for f in fields(view)))
                for view in (payload, context))
            try:
                step = monotonic()
                foundation = keep(deterministic_report._build_report_from_verified_context(cast(Any, context)))
                projection = keep(grouped_projection._project_verified(foundation=foundation,
                    payload=cast(Any, payload), policy=policy))
                if not projection.verify():
                    raise ValueError('live report projection integrity failed')
                # Do not use the compact view to narrow C or replace original case evidence.
                _preflight((bootstrap_audit._values(audit), supplied, projection))
                timings['report_projection'] = monotonic() - step
                if progress is not None:
                    progress(dict(phase='report_projection_ready', status=projection.status,
                        mandatory_view_bytes=projection.mandatory_view_bytes,
                        original_fact_count=projection.original_fact_count,
                        grouped_facts=len(projection.groups),
                        public_wire_bytes=None if projection.public_wire_json is None else len(projection.public_wire_json.encode())))
                result = _compose(ranking=ranking, descriptor=descriptor, projection=projection,
                    identity_manifest=manifest, identity_pin=expected_identity_manifest_sha256,
                    run_policy=runtime, complete=complete, keep=keep, stage_seconds=timings)
                _preflight((bootstrap_audit._values(audit), supplied, result))
                callback_finished[0] = monotonic()
                return result
            finally:
                if any(any(object.__getattribute__(view, f.name) is not before
                           for f, before in zip(fields(view), original, strict=True))
                       for view, original in aliases):
                    raise ValueError('live report private alias representation changed')
        result = bootstrap_audit._replay(audit=audit, **pins, progress=progress, _consume=consume)
        final_timings = {**result.stage_seconds, 'source_exit_checks': monotonic() - callback_finished[0],
                         'total': monotonic() - started}
        return cast(CurrentReportRun, keep(result.model_copy(update={'stage_seconds': final_timings})))
    finally:
        changed = False
        for value, before in reversed(retained):
            try:
                if _small_snapshot(value) != before:
                    changed = True
            except (AttributeError, KeyError, TypeError, ValueError, RecursionError):
                changed = True
        if changed:
            raise ValueError('live report input or intermediate content or representation changed')
