"""Explicit repair report facts, never relabelled legacy EventGroups or reports.

Only an enclosing raw-file/score replay calls the pure projection. All C scores
remain private and pinned; bounded Top-K display omissions cannot change them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final, Literal, TypeVar

from pydantic import BaseModel

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.hermes_exchange import PreparedExchange
from nurse_rcl.v2.live.repair_graph import RepairLiveGraph
from nurse_rcl.v2.live.repair_ranking import RepairRanking
from nurse_rcl.v2.live.repair_raw_audit import RepairRawAudit
from nurse_rcl.v2.live.repair_report_request import (
    REPAIR_REPORT_CONFIGURATION,
    RUN_POLICY,
    _public_for_view,
    prepare_repair_report_request,
)
from nurse_rcl.v2.report_evidence_schemas import (
    EvidenceKind,
    OmissionReason,
    ReportEvidenceBinding,
    ReportEvidenceFact,
    ReportEvidenceOmission,
    ReportEvidencePolicy,
    ReportEvidenceValue,
    ReportEvidenceView,
)

if TYPE_CHECKING:
    from nurse_rcl.v2.live.normal_report_context import NormalObservationContext

REPAIR_REPORT_PROJECTION_SCHEMA_VERSION: Final = "nurse.v2.live-repair-report-projection/1"
POLICY = ReportEvidencePolicy(
    configuration_version=REPAIR_REPORT_CONFIGURATION, max_view_bytes=80000, max_detail_facts=64
)


class _SharedProjectionFields(StrictFrozenModel):
    opaque_case_id: str
    source_ranking_sha256: str
    ordered_service_keys: tuple[str, ...]
    policy: ReportEvidencePolicy
    view: ReportEvidenceView | None
    source_bindings: tuple[ReportEvidenceBinding, ...]
    status: Literal["ready", "mandatory_budget_exhausted"]
    mandatory_view_bytes: int
    projection_sha256: str


class _ProjectionContentFields(_SharedProjectionFields):
    profile: Literal["v2-live-evidence-repair"] = "v2-live-evidence-repair"


class _RepairProjectionFields(_ProjectionContentFields):
    source_audit_sha256: str


class RepairReportProjection(_RepairProjectionFields):
    schema_version: Literal["nurse.v2.live-repair-report-projection/1"] = (
        REPAIR_REPORT_PROJECTION_SCHEMA_VERSION
    )


_Model = TypeVar("_Model", bound=BaseModel)


def _seal(value: _Model, field: str) -> _Model:
    return value.model_copy(
        update={
            field: sha256_hex(canonical_json_bytes(value.model_dump(mode="json", exclude={field})))
        }
    )


def _project_verified(audit: RepairRawAudit) -> RepairReportProjection:
    """Pure data projection under the enclosing complete source replay guard."""
    return _seal(
        RepairReportProjection(
            **_projection_values(
                ranking=audit.ranking,
                graph=audit.graph,
                opaque_case_id=audit.opaque_case_id,
                audit_sha256=audit.result_sha256,
                audit_fields=tuple(sorted(RepairRawAudit.model_fields)),
            )
        ),
        "projection_sha256",
    )


def _projection_values(
    *,
    ranking: RepairRanking,
    graph: RepairLiveGraph,
    opaque_case_id: str,
    audit_sha256: str,
    audit_fields: tuple[str, ...],
    observation_contexts: dict[str, NormalObservationContext] | None = None,
    policy: ReportEvidencePolicy = POLICY,
    semantic_limits: dict[str, Any] | None = None,
    public_for_view: Callable[[ReportEvidenceView], dict[str, Any]] = _public_for_view,
    request_for_view: Callable[[ReportEvidenceView], tuple[Any, PreparedExchange]] = (
        prepare_repair_report_request
    ),
    input_token_limit: int = RUN_POLICY.input_token_limit,
    view_admission: Callable[[ReportEvidenceView, bool], bool] | None = None,
) -> dict[str, Any]:
    """Shared facts only; the exact enclosing wrapper owns source semantics."""
    strengths = ranking.propagation.local_scoring.strengths
    mechanisms = strengths.mechanisms
    top = ranking.top_k_service_keys
    scores = {s.candidate_service_key: s for s in ranking.scores}
    facts: list[ReportEvidenceFact] = []
    bindings: list[ReportEvidenceBinding] = []

    def add(
        kind: EvidenceKind,
        targets: tuple[str, ...],
        values: dict[str, Any],
        *,
        source: Any,
        source_id: str,
        source_kind: str,
        codes: tuple[str, ...] = (),
        intervals: tuple[Any, ...] = (),
        same_source: tuple[str, ...] = (),
        observed: bool = False,
    ) -> None:
        ref = f"f{len(facts) + 1}"
        facts.append(
            ReportEvidenceFact(
                fact_ref=ref,
                kind=kind,
                target_service_keys=targets,
                intervals=intervals,
                values=tuple(ReportEvidenceValue(name=k, value=v) for k, v in values.items()),
                status_codes=tuple(sorted(set(codes))),
                allowed_claim_strength="frozen_observation" if observed else "ranking_inference",
                scope_basis="bound_observation" if observed else "frozen_assessment",
            )
        )
        source_pin = (
            ranking.result_sha256
            if source is ranking
            else graph.result_sha256
            if source is graph
            else audit_sha256
            if source is None
            else mechanisms.result_sha256
            if source is mechanisms
            else sha256_hex(canonical_json_bytes(source.model_dump(mode="json")))
        )
        bindings.append(
            ReportEvidenceBinding(
                fact_ref=ref,
                source_kind=source_kind,
                source_id=source_id,
                source_sha256=source_pin,
                source_fields=audit_fields
                if source is None
                else tuple(sorted(type(source).model_fields)),
                same_source_ids=tuple(sorted(set(same_source))),
            )
        )

    add(
        "limit",
        (),
        dict(
            profile=("v2-live-evidence-repair" if semantic_limits is None
                     else "v2-live-hybrid-normal"),
            pre_ranking_semantics=semantic_limits is not None,
            dynamic_queries_used=strengths.dynamic_queries_used,
            dynamic_query_limit=8,
            complete_candidate_count=len(ranking.candidate_service_keys),
            score_is_probability=False,
            tie_break_is_evidence=False,
            negative_correction="no_authorized_negative_role",
            causal_status="not_established",
            graph_edge_count=len(graph.graph_pair.fault_snapshot.edges),
            graph_limitations=";".join(graph.limitations),
            interpretation="accepted mechanisms are observable contracts, not confirmed causes",
            **({} if semantic_limits is None else semantic_limits),
        ),
        source=None,
        source_id=audit_sha256,
        source_kind="repair_case_limits" if semantic_limits is None else "normal_b2_case_limits",
        codes=(
            ("pre_ranking_semantics_disabled" if semantic_limits is None
             else "query_priorities_only"),
            "no_confirmed_causality",
            ("omitted_details_remain_in_complete_private_audit" if semantic_limits is None
             else "omitted_details_remain_in_private_evidence"),
            *graph.limitations,
        ),
    )
    for rank, key in enumerate(top, 1):
        score = scores[key]
        tie = next(t for t in ranking.tie_groups if key in t.member_service_keys)
        add(
            "ranking",
            (key,),
            dict(
                rank=rank,
                quantized_total_score=score.quantized_total_score,
                quantized_event_evidence_score=score.quantized_event_evidence_score,
                quantized_propagation_score=score.quantized_propagation_score,
                local_score_component=score.local_score_component,
                propagation_score_component=score.propagation_score_component,
                tie_group_member_count=len(tie.member_service_keys),
                tie_break_is_evidence=False,
            ),
            source=score,
            source_id=score.score_sha256,
            source_kind="repair_score",
        )
        decisions = [m for m in mechanisms.mechanisms if m.candidate_service_key == key]
        conditions = [c for c in mechanisms.conditions if c.candidate_service_key == key]
        witnesses = [w for w in ranking.propagation.witnesses if w.candidate_service_key == key]
        add(
            "candidate_evidence",
            (key,),
            dict(
                accepted_mechanism_count=sum(m.disposition == "accepted" for m in decisions),
                rejected_mechanism_count=sum(m.disposition == "rejected" for m in decisions),
                unknown_mechanism_count=sum(m.disposition == "unknown" for m in decisions),
                unknown_predicate_count=sum(c.truth_value == "unknown" for c in conditions),
                refuting_predicate_count=sum(
                    c.requirement == "refuting" and c.truth_value == "true" for c in conditions
                ),
                unknown_term_count=len(score.unknown_term_ids),
                dependent_unused_term_count=len(score.dependent_unused_term_ids),
                selected_term_count=len(score.selected_term_ids),
                usable_source_onset_count=sum(
                    w.role == "source" and not w.reason_codes for w in witnesses
                ),
                usable_symptom_onset_count=sum(
                    w.role == "symptom" and not w.reason_codes for w in witnesses
                ),
                uncredited_onset_count=sum(bool(w.reason_codes) for w in witnesses),
                onset_intervals_ns=";".join(
                    f"{w.role}:{','.join(str(n) for n in w.onset_bounds_ns)}" for w in witnesses
                ),
                local_evidence_status=score.local_evidence_status,
            ),
            source=ranking,
            source_id=key,
            source_kind="repair_candidate_summary",
            codes=(
                *score.propagation_reason_codes,
                *(r for w in witnesses for r in w.reason_codes),
            ),
        )
        # All six decisions are mandatory, including rejected and unknown, with
        # every condition's actual result; no favorable mechanism-only selection.
        for decision in decisions:
            owned = [c for c in conditions if c.condition_id in decision.condition_ids]
            values: dict[str, Any] = dict(
                mechanism_key=decision.mechanism_key,
                disposition=decision.disposition,
                causal_status=decision.causal_status,
                permitted_roles=";".join(decision.permitted_roles),
            )
            for requirement in ("necessary", "optional", "refuting"):
                for truth in ("true", "false", "unknown"):
                    values[f"{requirement}_{truth}_count"] = sum(
                        c.requirement == requirement and c.truth_value == truth for c in owned
                    )
                values[f"{requirement}_conditions"] = ";".join(
                    f"{c.predicate_kind}={c.truth_value}"
                    for c in owned
                    if c.requirement == requirement
                )
            add(
                "hypothesis",
                (key,),
                values,
                source=mechanisms,
                source_id=decision.mechanism_id,
                source_kind="repair_mechanism",
                codes=decision.reason_codes,
                same_source=decision.support_evidence_ids,
            )
        for path in ranking.propagation.paths:
            if path.path_id not in score.selected_path_ids:
                continue
            add(
                "hypothesis",
                (key,),
                dict(
                    source_service_key=path.source_service_key,
                    target_service_key=path.target_service_key,
                    claim_kind=path.claim_kind,
                    source_precedes_target=path.source_precedes_target,
                    hop_count=len(path.hop_edge_ids),
                    path_quality=path.path_quality,
                    source_support=path.source_support,
                    symptom_support=path.target_symptom_support,
                    contribution=path.contribution,
                ),
                source=path,
                source_id=path.path_id,
                source_kind="repair_dependency_symptom",
                codes=path.reason_codes,
                same_source=(path.source_term_id, path.target_term_id),
            )
    for key in top[1:]:
        first, other = scores[top[0]], scores[key]
        add(
            "competition",
            (top[0], key),
            dict(
                total_score_gap=first.total_score - other.total_score,
                event_evidence_gap=first.event_evidence_score - other.event_evidence_score,
                propagation_gap=first.propagation_score - other.propagation_score,
                evidence_tied=(
                    first.quantized_total_score,
                    first.quantized_event_evidence_score,
                    first.quantized_propagation_score,
                )
                == (
                    other.quantized_total_score,
                    other.quantized_event_evidence_score,
                    other.quantized_propagation_score,
                ),
            ),
            source=ranking,
            source_id=ranking.result_sha256,
            source_kind="repair_competition",
        )
    mandatory_count = len(facts)
    omitted: Counter[tuple[EvidenceKind, tuple[str, ...], OmissionReason]] = Counter()
    for term in strengths.terms:
        if term.candidate_service_key not in top:
            omitted[("observation", (), "outside_top_k")] += 1
            omitted[("candidate_evidence", (), "outside_top_k")] += 1
    for predicate in mechanisms.dynamic_predicates:
        if predicate.candidate_service_key not in top:
            omitted[("predicate", (), "outside_top_k")] += 1
    # Round-robin over candidates, first strongest actual contributors then all
    # other/unknown terms. This only orders optional explanation detail, not C.
    queues = [[t for t in strengths.terms if t.candidate_service_key == key] for key in top]
    for key, queue in zip(top, queues, strict=True):
        queue.sort(
            key=lambda t: (
                t.term_id not in scores[key].selected_term_ids,
                -(t.local_strength or 0),
                t.term_id,
            )
        )
    details_by_candidate: list[list[dict[str, Any]]] = []
    for key, queue in zip(top, queues, strict=True):
        paid = [p for p in mechanisms.dynamic_predicates if p.candidate_service_key == key]
        details: list[dict[str, Any]] = []
        for term in queue:
            bound = [p for p in paid if p.source_sha256 == term.source_sha256]
            units = {p.unit for p in bound}
            # No guessed unit: absent from this source-derived retained contract
            # is disclosed, even if a metric name looks familiar.
            unit = next(iter(units)) if len(units) == 1 else "unavailable_in_retained_fact"
            context = (observation_contexts or {}).get(term.term_id)
            context_values = {} if context is None else dict(
                baseline_start_unix_ns=context.baseline_window.start_unix_ns,
                baseline_end_unix_ns=context.baseline_window.end_unix_ns,
                fault_start_unix_ns=context.fault_window.start_unix_ns,
                fault_end_unix_ns=context.fault_window.end_unix_ns,
                summary_scope="location_summary_not_whole_window_health"
                if term.kind == "window_summary" else "only_retained_event_or_measurement",
            )
            details.append(
                dict(
                    kind="observation",
                    targets=(key,),
                    values=dict(
                        kind=term.kind,
                        metric_name=term.metric_name,
                        modality=term.modality,
                        primary_family=term.primary_family,
                        unit=context.unit if context is not None else unit,
                        **context_values,
                        raw_strength=term.raw_strength,
                        quality_factor=term.quality_factor,
                        evidence_state="unknown" if term.weighted_strength is None else "measured",
                        observed_timestamps_ns=",".join(str(n) for n in term.timestamps_ns),
                        **{
                            k.lower(): v
                            for k, v in term.calculation.model_dump(exclude_none=True).items()
                        },
                    ),
                    source=context if context is not None else term,
                    source_id=term.term_id,
                    source_kind="normal_source_bound_observation"
                    if context is not None else "repair_strength_observation",
                    same_source=(term.source_sha256, term.pool_key, term.evidence_id),
                    intervals=(term.interval,),
                    codes=term.reason_codes,
                    observed=True,
                )
            )
            details.append(
                dict(
                    kind="candidate_evidence",
                    targets=(key,),
                    values=dict(
                        kind=term.kind,
                        local_strength=term.local_strength,
                        role_support=term.role_support,
                        selected_for_local=term.term_id in scores[key].selected_term_ids,
                        role_mechanisms=";".join(
                            m.mechanism_key
                            for m in mechanisms.mechanisms
                            if m.mechanism_id in term.role_mechanism_ids
                        ),
                    ),
                    source=term,
                    source_id=term.term_id,
                    source_kind="repair_term_score_effect",
                    same_source=(term.source_sha256, term.pool_key, term.evidence_id),
                    codes=term.reason_codes,
                )
            )
        # Measured conditions are distinct from declared mechanism conjunctions.
        # Interleave them with event details, retaining all omissions explicitly.
        predicates = [
            dict(
                kind="predicate",
                targets=(key,),
                values=dict(
                    predicate_kind=p.predicate_kind,
                    mechanism_key=p.mechanism_key,
                    measurement_field=p.measurement_field,
                    truth_value=p.truth_value,
                    unit=p.unit,
                ),
                source=p,
                source_id=p.result_sha256,
                source_kind="repair_measured_predicate",
                same_source=(p.source_sha256, p.measurement_sha256),
                intervals=(p.interval,),
                codes=p.reason_codes,
                observed=True,
            )
            for p in paid
        ]
        details_by_candidate.append(
            [
                row
                for i in range(max(len(details), len(predicates)))
                for group in (details, predicates)
                if i < len(group)
                for row in (group[i],)
            ]
        )
    detail = [
        queue[i]
        for i in range(max((len(q) for q in details_by_candidate), default=0))
        for queue in details_by_candidate
        if i < len(queue)
    ]

    def view_now() -> ReportEvidenceView:
        return _seal(
            ReportEvidenceView(
                opaque_case_id=opaque_case_id,
                context_sha256=audit_sha256,
                policy_sha256=policy.sha256,
                top_k=top,
                facts=tuple(facts),
                mandatory_fact_refs=tuple(f.fact_ref for f in facts[:mandatory_count]),
                omissions=tuple(
                    ReportEvidenceOmission(kind=k, target_service_keys=t, count=n, reason_code=r)
                    for (k, t, r), n in sorted(omitted.items())
                    if n
                ),
                view_sha256="0" * 64,
            ),
            "view_sha256",
        )

    def size(view: ReportEvidenceView) -> int:
        return max(
            len(canonical_json_bytes(public_for_view(view))),
            len(canonical_json_bytes(view.model_dump(mode="json"))),
        )

    def fits(view: ReportEvidenceView) -> bool:
        if view_admission is not None:
            # New owner admits its own representation, not a legacy report wire.
            # Pure callback; it cannot send, measure or change the fact builder.
            return view_admission(view, len(view.facts) == mandatory_count)
        if size(view) > policy.max_view_bytes:
            return False
        try:
            _, request = request_for_view(view)
        except ExchangeBoundaryError as error:
            if error.code != "json_limit":
                raise
            return False
        return len(request.payload_bytes) + 256 <= input_token_limit

    outside_omissions = omitted.copy()

    def state(count: int, reason: OmissionReason) -> ReportEvidenceView:
        # Omission reason describes the constraint stopping this ordered prefix,
        # not a claim that every undisplayed row was individually attempted.
        omitted.clear()
        omitted.update(outside_omissions)
        for row in detail[count:]:
            omitted[(row["kind"], row["targets"], reason)] += 1
        return view_now()

    limit = min(len(detail), policy.max_detail_facts)
    mandatory = state(0, "detail_byte_budget" if limit else "detail_count_budget")
    mandatory_bytes = size(mandatory)
    # Keep the longest feasible prefix of the existing round-robin priority.
    # Back off whole optional facts AND bindings; each trial includes all final
    # omission metadata. At count zero it is exactly the mandatory state above.
    for row in detail[:limit]:
        add(**row)
    view: ReportEvidenceView | None = None
    for count in range(limit, -1, -1):
        candidate = state(count, "detail_count_budget" if count == limit else "detail_byte_budget")
        if fits(candidate):
            view = candidate
            break
        if count:
            facts.pop()
            bindings.pop()
    return dict(
        opaque_case_id=opaque_case_id,
        source_audit_sha256=audit_sha256,
        source_ranking_sha256=ranking.result_sha256,
        ordered_service_keys=ranking.ordered_service_keys,
        policy=policy,
        view=view,
        source_bindings=tuple(bindings) if view else (),
        status="ready" if view else "mandatory_budget_exhausted",
        mandatory_view_bytes=mandatory_bytes,
        projection_sha256="0" * 64,
    )
