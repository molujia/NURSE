"""Paid branch feedback, without replaying sources, scoring or querying again.

The normal owner supplies its admitted preparation and independently pinned tool
session. Public aliases bind to those results, not to paired B0's unseen future.
This projection has no outbound or numeric authority.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import (
    SHA256,
    Direction,
    EpisodeBoundary,
    OnsetReason,
    TruthValue,
)
from nurse_rcl.v2.exchange_json import decode_json_object, encode_json
from nurse_rcl.v2.live.normal_hybrid_queries import _intent_identity
from nurse_rcl.v2.live.normal_hybrid_tools import NormalHybridToolSession, _stop_binding
from nurse_rcl.v2.live.normal_queries import _feedback_binding, _measurement_binding, _seal
from nurse_rcl.v2.live.normal_semantic import (
    CardSuggestion,
    NormalSemanticCompilation,
    NormalSemanticResponse,
    SemanticRejection,
    _compile_values,
    compile_normal_semantics,
)
from nurse_rcl.v2.live.normal_semantic_context import NormalSemanticContext, _project_fixed_context
from nurse_rcl.v2.live.normal_semantic_wire import (
    WIRE_LIMITS,
    SemanticAlias,
    _expand_response_aliases,
    build_semantic_wire,
)
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData
from nurse_rcl.v2.observation_schemas import TimeInterval


class SemanticEpisode(StrictFrozenModel):
    sample_indices: tuple[int, ...]
    direction: Direction
    support_span_ns: int = Field(ge=0)
    qualifying: bool
    start_reason: EpisodeBoundary
    end_reason: EpisodeBoundary
    onset_lower_unix_ns: int | None
    onset_upper_unix_ns: int
    onset_reason: OnsetReason
    left_censored: bool
    right_censored: bool


class SemanticMeasuredPredicate(StrictFrozenModel):
    reference_id: str
    need_key: str
    mechanism_key: str
    predicate_kind: str
    requirement: Literal["necessary", "optional", "refuting"]
    measurement_field: str
    unit: str
    interval: TimeInterval
    truth_value: TruthValue
    disposition: Literal["usable", "unknown"]
    reason_codes: tuple[str, ...]
    onset_bounds_ns: tuple[int, int] | None
    onset_reason_codes: tuple[str, ...]


class SemanticMeasurement(StrictFrozenModel):
    reference_id: str
    request_key: str
    candidate: str
    signal_reference: str
    unit: str | None
    interval: TimeInterval
    point_count: int = Field(ge=0)
    persistent_up: TruthValue
    persistent_down: TruthValue
    persistent_deviation_present: TruthValue
    general_deviation_present: TruthValue
    verified_normal: TruthValue
    reason_codes: tuple[str, ...]
    episodes: tuple[SemanticEpisode, ...]
    predicates: tuple[SemanticMeasuredPredicate, ...]
    limitations: tuple[str, ...] = (
        "point_values_retained_privately_not_displayed",
        "onset_bounds_are_not_causal_proof_or_link_delay",
        "a_measured_predicate_is_not_a_complete_mechanism",
    )


class NormalSemanticFeedback(StrictFrozenModel):
    schema_version: Literal["nurse.v2.normal-semantic-feedback/1"] = (
        "nurse.v2.normal-semantic-feedback/1"
    )
    scope: Literal["branch_paid_feedback_only"] = "branch_paid_feedback_only"
    first_context_sha256: SHA256
    first_compilation_sha256: SHA256
    session_sha256: SHA256
    fixed: NormalSemanticContext
    first_response: NormalSemanticResponse
    card_candidates: tuple[str, ...]
    measurements: tuple[SemanticMeasurement, ...]
    trigger_proposal_ids: tuple[str, ...]
    feedback_sha256: SHA256


class SemanticFeedbackPackage(StrictFrozenModel):
    schema_version: Literal["nurse.v2.normal-semantic-feedback-package/1"] = (
        "nurse.v2.normal-semantic-feedback-package/1"
    )
    feedback_sha256: SHA256
    public_json: str
    public_sha256: SHA256
    aliases: tuple[SemanticAlias, ...]


class SemanticFeedbackCompilation(StrictFrozenModel):
    schema_version: Literal["nurse.v2.normal-semantic-feedback-compilation/1"] = (
        "nurse.v2.normal-semantic-feedback-compilation/1"
    )
    feedback_sha256: SHA256
    context_sha256: SHA256
    response_sha256: SHA256
    round_number: Literal[2] = 2
    accepted_proposal_ids: tuple[str, ...]
    intents: tuple[()] = ()
    card_candidates: tuple[()] = ()
    revision_dependency_proposal_ids: tuple[()] = ()
    rejections: tuple[SemanticRejection, ...]
    score_authority: Literal[False] = False
    executed_queries: Literal[0] = 0
    compilation_sha256: SHA256


def project_semantic_feedback(
    *, data: RepairPreparationData, context: NormalSemanticContext,
    first_response: NormalSemanticResponse, first_compilation: NormalSemanticCompilation,
    session: NormalHybridToolSession, expected_session_sha256: str,
) -> NormalSemanticFeedback:
    """Bind owner receipts once. Do not recalculate any measurement or ranking."""
    compiled = compile_normal_semantics(data=data, context=context, response=first_response)
    if compiled != first_compilation or type(session) is not NormalHybridToolSession or (
        session.session_sha256 != expected_session_sha256
        or _seal(session, "session_sha256") != session
        or session.catalog_sha256 != context.catalog_sha256
        or session.intents_sha256 != _intent_identity(data.catalog, compiled.intents)
        or session.card_count != len(session.cards)
        or session.verification_count != len(session.steps)
        or session.paid_count != len(session.cards) + len(session.steps)
        or tuple(c.card.candidate_service_key for c in session.cards) != compiled.card_candidates
        or session.submitted_request_keys != tuple(sorted({s.feedback.request_key
                                                          for s in session.steps}))
    ):
        raise ValueError("semantic feedback owner identity differs")
    _stop_binding(session.stop, data, session.branch_identity_sha256, session.intents_sha256,
                  session.cards, session.steps)
    requests = {r.request_key: r for r in data.catalog.requests}
    sources = {s.source_sha256: s for s in data.sources}
    signals = {s.reference_id: s for s in context.signals}
    errors = {e.reference_id: e for e in context.errors}
    for receipt in session.cards:
        card = receipt.card
        if receipt.branch_identity_sha256 != session.branch_identity_sha256 or (
            card.catalog_sha256 != context.catalog_sha256
            or card.opaque_case_id != context.opaque_case_id
            or card.baseline_window != context.baseline_window
            or card.fault_window != context.fault_window
            or card.candidate_service_key not in compiled.card_candidates
            or _seal(receipt, "receipt_sha256") != receipt or _seal(card, "card_sha256") != card
        ):
            raise ValueError("semantic feedback card binding differs")
        signals.update({s.reference_id: s for s in card.signals})
        errors.update({e.reference_id: e for e in card.errors})
    # A paid measurement can provide its own fixed summary, not a free full card
    # for that service. Other signals and errors of non-Q services stay private.
    measured_sources = {step.measurement.source_sha256 for step in session.steps}
    selected = {requests[step.feedback.request_key].candidate_service_key for step in session.steps}
    extra = _project_fixed_context(data, {c for c in selected if c is not None})
    signals.update({s.reference_id: s for s in extra.signals
                    if s.source_sha256 in measured_sources})
    visible_sources = {s.source_sha256 for s in signals.values()}
    fixed = _seal(context.model_copy(update=dict(
        signals=tuple(signals.values()), errors=tuple(errors.values()),
        requests=tuple(r for r in data.catalog.requests if r.source_sha256 in visible_sources),
    )), "context_sha256")
    by_source = {s.source_sha256: s for s in fixed.signals}
    rows = []
    for step in session.steps:
        measured, result = step.measurement, step.feedback
        request = requests[result.request_key]
        source = sources[request.source_sha256]
        _measurement_binding(measured, source, data.catalog)
        _feedback_binding(result, measured, source, data.catalog, request.request_key)
        if step.branch_identity_sha256 != session.branch_identity_sha256 or (
            _seal(step, "step_sha256") != step
            or _seal(measured, "measurement_sha256") != measured
            or _seal(result, "result_sha256") != result
        ):
            raise ValueError("semantic feedback measurement receipt differs")
        predicates = tuple(SemanticMeasuredPredicate(
            reference_id="predicate_" + pred.result_sha256, need_key=need.need_key,
            mechanism_key=pred.mechanism_key, predicate_kind=pred.predicate_kind,
            requirement=need.requirement, measurement_field=pred.measurement_field,
            unit=pred.unit, interval=pred.interval, truth_value=pred.truth_value,
            disposition=pred.disposition, reason_codes=pred.reason_codes,
            onset_bounds_ns=onset.bounds_ns, onset_reason_codes=onset.reason_codes,
        ) for pred, onset, need in zip(
            result.predicates, result.onsets, request.needs, strict=True,
        ))
        assert request.candidate_service_key is not None
        rows.append(SemanticMeasurement(
            reference_id="measurement_" + measured.measurement_sha256,
            request_key=request.request_key, candidate=request.candidate_service_key,
            signal_reference=by_source[source.source_sha256].reference_id,
            unit=source.observation.unit, interval=measured.interval,
            point_count=len(measured.points), persistent_up=measured.persistent_up,
            persistent_down=measured.persistent_down,
            persistent_deviation_present=measured.persistent_deviation_present,
            general_deviation_present=measured.general_deviation_present,
            verified_normal=measured.verified_normal, reason_codes=measured.reason_codes,
            episodes=tuple(SemanticEpisode.model_validate_json(canonical_json_bytes(
                e.model_dump(mode="json", exclude={"episode_id", "canonical_series_key",
                                                    "request_sha256", "policy_sha256"}),
            )) for e in measured.episodes), predicates=predicates,
        ))
    paid_cards = {c.card.candidate_service_key for c in session.cards}
    paid_requests = set(session.submitted_request_keys)
    queries = {q.query_id: q for q in first_response.queries}
    def was_paid(key: str) -> bool:
        query = queries[key]
        return (query.candidate_service_key in paid_cards if isinstance(query, CardSuggestion)
                else query.request_key in paid_requests)

    triggers = tuple(h.proposal_id for h in first_response.hypotheses
                     if h.proposal_id in compiled.revision_dependency_proposal_ids and any(
                         was_paid(key) for key in h.query_ids))
    return _seal(NormalSemanticFeedback(
        first_context_sha256=context.context_sha256,
        first_compilation_sha256=compiled.compilation_sha256, session_sha256=session.session_sha256,
        fixed=fixed, first_response=first_response,
        card_candidates=tuple(c.card.candidate_service_key for c in session.cards),
        measurements=tuple(rows), trigger_proposal_ids=triggers, feedback_sha256="0" * 64,
    ), "feedback_sha256")


def build_feedback_package(feedback: NormalSemanticFeedback) -> SemanticFeedbackPackage:
    if type(feedback) is not NormalSemanticFeedback or (
        _seal(feedback, "feedback_sha256") != feedback
    ):
        raise ValueError("semantic feedback content differs")
    fixed = build_semantic_wire(feedback.fixed)
    aliases = list(fixed.aliases)
    for i, row in enumerate(feedback.measurements):
        aliases.append(SemanticAlias(kind="evidence", original=row.reference_id, alias=f"M{i}"))
        for pred in row.predicates:
            aliases.append(SemanticAlias(kind="evidence", original=pred.reference_id,
                                         alias=f"P{len(aliases)}"))
    refs: dict[tuple[str, str], str] = {(a.kind, a.original): a.alias for a in aliases}
    rows = []
    for row in feedback.measurements:
        value = row.model_dump(mode="json")
        for field, kind in (("reference_id", "evidence"), ("request_key", "request"),
                            ("candidate", "candidate"), ("signal_reference", "evidence")):
            value[field] = refs[kind, value[field]]
        for pred in value["predicates"]:
            pred["reference_id"] = refs["evidence", pred["reference_id"]]
            pred["need_key"] = refs["need", pred["need_key"]]
        rows.append(value)
    first = feedback.first_response.model_dump(mode="json")
    # Rejected first-round references are displayed as unavailable, not promoted
    # to public aliases. The raw response remains in the private request receipt.
    for h in first["hypotheses"]:
        h["candidate_service_key"] = refs.get(
            ("candidate", h["candidate_service_key"]), "unavailable",
        )
        h["evidence_refs"] = [refs.get(("evidence", r), "unavailable") for r in h["evidence_refs"]]
    for q in first["queries"]:
        if q["kind"] == "card":
            q["candidate_service_key"] = refs.get(
                ("candidate", q["candidate_service_key"]), "unavailable",
            )
        else:
            q["request_key"] = refs.get(("request", q["request_key"]), "unavailable")
            q["need_keys"] = [refs.get(("need", n), "unavailable") for n in q["need_keys"]]
    public = dict(schema_version="nurse.v2.normal-semantic-feedback-display/1",
                  scope=feedback.scope,
                  fixed_wire=decode_json_object(fixed.wire_json.encode(), WIRE_LIMITS),
                  card_candidates=[refs["candidate", c] for c in feedback.card_candidates],
                  measurements=rows, first_response=first,
                  trigger_proposal_ids=feedback.trigger_proposal_ids,
                  additional_tool_budget=0, score_authority=False)
    raw = encode_json(public, WIRE_LIMITS)
    return SemanticFeedbackPackage(feedback_sha256=feedback.feedback_sha256,
                                   public_json=raw.decode(), public_sha256=sha256_hex(raw),
                                   aliases=tuple(aliases))


def restore_semantic_feedback(
    package: SemanticFeedbackPackage, feedback: NormalSemanticFeedback,
) -> NormalSemanticFeedback:
    if type(package) is not SemanticFeedbackPackage or canonical_json_bytes(
        package.model_dump(mode="json"),
    ) != canonical_json_bytes(build_feedback_package(feedback).model_dump(mode="json")):
        raise ValueError("semantic feedback representation differs from owner")
    return feedback


def compile_semantic_feedback(
    *, data: RepairPreparationData, feedback: NormalSemanticFeedback,
    package: SemanticFeedbackPackage, response: NormalSemanticResponse,
) -> SemanticFeedbackCompilation:
    restore_semantic_feedback(package, feedback)
    expanded = _expand_response_aliases(response, package.aliases)
    refs: dict[str, tuple[str, str | None]] = {}
    for row in feedback.measurements:
        refs[row.reference_id] = (row.candidate, None)
        refs.update({p.reference_id: (row.candidate, p.mechanism_key) for p in row.predicates})
    values = _compile_values(data=data, context=feedback.fixed, response=expanded,
                             round_number=2, feedback_refs=refs)
    return _seal(SemanticFeedbackCompilation(feedback_sha256=feedback.feedback_sha256, **values),
                 "compilation_sha256")
