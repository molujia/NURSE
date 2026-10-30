"""Source-replayed event magnitudes and bounded LOCAL role effects.

This explicit repair artifact is not a legacy quality/event/role receipt or a
final ranking. Complete raw-file replay, temporal roles, family aggregation and
score audit remain enclosing duties. No labels, weights, selected feedback or
caller-supplied trust flags are inputs. Dependence pools are conservative voting
limits, not claims that different families describe one causal event.
"""

from __future__ import annotations

import math
from statistics import median
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from pydantic import Field

if TYPE_CHECKING:
    from nurse_rcl.v2.live.normal_artifacts import NormalQuerySession
    from nurse_rcl.v2.live.normal_hybrid_queries import NormalHybridQuerySession
    from nurse_rcl.v2.live.normal_hybrid_tools import NormalHybridToolSession

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.dynamic_verification.safety import plain, require_pin
from nurse_rcl.v2.live.repair_fixed_facts import RepairErrorFactSummary, RepairFixedSignalFacts
from nurse_rcl.v2.live.repair_lifecycle import _valid_state
from nurse_rcl.v2.live.repair_mechanisms import (
    REPAIR_MECHANISM_SEMANTICS_SHA256,
    RepairMechanismEvidence,
    _replayed_mechanism_context,
)
from nurse_rcl.v2.live.repair_module_timing import _module_call
from nurse_rcl.v2.live.repair_query_session import RepairQuerySession
from nurse_rcl.v2.live.repair_safety import RepairModel, canonical, require_unchanged, snapshot
from nurse_rcl.v2.live.repair_sources import RepairMeasurementSource
from nurse_rcl.v2.observation_schemas import TimeInterval

REPAIR_STRENGTH_SCHEMA_VERSION: Final = "nurse.v2.live-repair-strengths/1"
REPAIR_STRENGTH_SEMANTICS_SHA256: Final = sha256_hex(
    canonical_json_bytes(
        dict(
            version="nurse-v2-live-repair-strengths/1",
            mechanisms=REPAIR_MECHANISM_SEMANTICS_SHA256,
            summary="unchanged_coverage_times_anomaly_neutral_no_borrowed_root_role",
            episode="paid_qualifying_median_abs_z_exp_divisor_3_times_source_quality",
            state="eligible_F_zero_exists_times_B_positive_state_fraction_times_B_coverage",
            error="fully_known_BF_positive_fraction_delta_else_confirmed_error_observed_share_lower_bound",
            error_coverage="population_unknown_not_numeric_one_not_request_rate",
            local_role="necessary_witness_only_min_condition_max_source_quality",
            adjustment="0.5+0.5*support-no_unverified_refutation",
            dependence="scalar_source_or_same_entity_modality_BF_pool_one_max_adjusted_term",
            authority="local_only_not_raw_file_or_complete_ranking",
        )
    )
)


class RepairStrengthCalculation(RepairModel):
    signed_z: float | None = None
    median_B: float | None = None
    median_F: float | None = None
    baseline_positive_fraction: float | None = None
    baseline_sample_count: int | None = None
    state_meaning: str | None = None
    median_abs_z: float | None = None
    direction: str | None = None
    sample_count: int | None = None
    support_span_ns: int | None = None
    baseline_fraction: float | None = None
    fault_fraction: float | None = None
    baseline_known_denominator: int | None = None
    fault_known_denominator: int | None = None
    fault_observed_count: int | None = None
    fault_error_count: int | None = None
    fault_unknown_or_conflict_count: int | None = None
    span_kind: str | None = None


class RepairStrengthTerm(RepairModel):
    term_id: SHA256
    evidence_id: SHA256
    source_sha256: SHA256
    candidate_service_key: str | None
    observed_entity_key: str | None
    modality: Literal["metrics", "traces", "logs"]
    metric_name: str | None
    primary_family: str
    pool_key: SHA256
    interval: TimeInterval
    kind: Literal["window_summary", "state_event", "paid_episode", "error_status"]
    measurement_sha256: SHA256 | None = None
    episode_id: str | None = None
    timestamps_ns: tuple[int, ...] = ()
    predicate_kinds: tuple[str, ...] = ()
    raw_strength: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    quality_factor: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    weighted_strength: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    role_support: float = Field(ge=0, le=1, allow_inf_nan=False)
    role_mechanism_ids: tuple[str, ...]
    local_strength: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    calculation: RepairStrengthCalculation
    reason_codes: tuple[str, ...]


class RepairDependencePool(RepairModel):
    pool_key: SHA256
    candidate_service_key: str | None
    retained_term_ids: tuple[str, ...]
    selected_term_id: SHA256 | None
    primary_family: str | None
    local_strength: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    reason: Literal["one_maximum_dependent_vote_not_event_association"] = (
        "one_maximum_dependent_vote_not_event_association"
    )


class RepairStrengthEvidence(RepairModel):
    schema_version: Literal["nurse.v2.live-repair-strengths/1"] = REPAIR_STRENGTH_SCHEMA_VERSION
    semantics_sha256: SHA256
    opaque_case_id: str
    session_sha256: SHA256
    candidate_service_keys: tuple[str, ...]
    mechanisms: RepairMechanismEvidence
    terms: tuple[RepairStrengthTerm, ...]
    pools: tuple[RepairDependencePool, ...]
    dynamic_queries_used: int = Field(ge=0, le=8)
    ranking_ready: Literal[False] = False
    result_sha256: SHA256


_ROOTS = (RepairStrengthEvidence, RepairStrengthTerm, RepairDependencePool)


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(plain(value)))


def _term(**values: Any) -> RepairStrengthTerm:
    """Construct a value only; authority comes from the enclosing replay."""
    weight = values["weighted_strength"]
    values["calculation"] = RepairStrengthCalculation(**values["calculation"])
    draft = RepairStrengthTerm(
        **values,
        term_id="0" * 64,
        role_support=0.0,
        role_mechanism_ids=(),
        local_strength=None if weight is None else weight * 0.5,
    )
    unsigned = cast(dict[str, object], plain(draft))
    unsigned.pop("term_id")
    return draft.model_copy(update={"term_id": _digest(unsigned)})


def _pool_key(
    dataset: str,
    entity: str | None,
    modality: str,
    interval: TimeInterval,
    signal_id: str | None = None,
) -> str:
    return _digest((dataset, entity, modality, interval, signal_id))


def _scalar_terms(
    source: RepairMeasurementSource, fact: RepairFixedSignalFacts, session: RepairQuerySession
) -> list[RepairStrengthTerm]:
    rows = _fixed_scalar_terms(source, fact)
    return rows + _paid_scalar_terms(
        rows[0], formal_scoring_eligible=source.quality.formal_scoring_eligible, session=session
    )


def _fixed_scalar_terms(
    source: RepairMeasurementSource, fact: RepairFixedSignalFacts
) -> list[RepairStrengthTerm]:
    q = source.quality
    summary = q.source_summary
    bound = source.candidate_service_key is not None
    base = dict(
        evidence_id=fact.result_sha256,
        source_sha256=source.source_sha256,
        candidate_service_key=source.candidate_service_key,
        observed_entity_key=source.observed_entity_key,
        modality=source.observation.modality,
        metric_name=source.metric_name,
        primary_family=source.observation.signal_family,
        interval=source.query_interval,
        pool_key=_pool_key(
            source.dataset,
            source.observed_entity_key,
            source.observation.modality,
            source.query_interval,
            source.signal_id if source.observation.modality == "metrics" else None,
        ),
    )
    raw = q.anomaly_strength if q.formal_scoring_eligible else None
    rows = [
        _term(
            **base,
            kind="window_summary",
            raw_strength=raw,
            quality_factor=q.coverage_quality,
            weighted_strength=raw * q.coverage_quality if raw is not None and bound else None,
            predicate_kinds=("local_anomaly_observed",),
            calculation=dict(
                signed_z=q.signed_standardized_change,
                median_B=summary.baseline_summary.median,
                median_F=summary.fault_summary.median,
            ),
            reason_codes=("unchanged_window_summary_not_event_magnitude", *q.quality_flags),
        )
    ]
    event = source.lifecycle
    if event is not None:
        component = next(
            c for c in source.transform_result.windows[0].components if c.name == "value"
        )
        coverage = summary.baseline_summary.coverage_fraction
        usable = (
            component.formal_scoring_eligible
            and bound
            and len(component.samples) >= summary.configuration.minimum_valid_points
            and coverage >= summary.configuration.minimum_coverage_fraction
            and all(_valid_state(p, event.state_meaning) for p in component.samples)
        )
        positive = (
            sum(p.value > 0 for p in component.samples) / len(component.samples) if usable else None
        )
        state_raw = (
            (positive * float(event.formal_event_eligible)) if positive is not None else None
        )
        reasons = ["state_existence_not_duration_or_service_outage"]
        if not usable:
            reasons.append("baseline_state_reference_unavailable")
        elif positive == 0:
            reasons.append("baseline_zero_not_new_deviation")
        kinds = ["local_anomaly_observed"] if event.formal_event_eligible else []
        if event.availability_support:
            kinds.append("workload_unavailability_observed")
        rows.append(
            _term(
                **base,
                kind="state_event",
                raw_strength=state_raw,
                quality_factor=coverage if usable else None,
                weighted_strength=state_raw * coverage if state_raw is not None else None,
                predicate_kinds=tuple(sorted(kinds)),
                timestamps_ns=fact.event_timestamps_ns,
                calculation=dict(
                    baseline_positive_fraction=positive,
                    baseline_sample_count=len(component.samples),
                    state_meaning=event.state_meaning,
                ),
                reason_codes=tuple(reasons),
            )
        )
    return rows


def _paid_scalar_terms(
    window: RepairStrengthTerm, *, formal_scoring_eligible: bool,
    session: (
        RepairQuerySession | NormalQuerySession | NormalHybridQuerySession | NormalHybridToolSession
    ),
) -> list[RepairStrengthTerm]:
    """Pure paid terms over source-derived values held by the lexical owner."""
    base = {
        name: getattr(window, name)
        for name in (
            "evidence_id",
            "source_sha256",
            "candidate_service_key",
            "observed_entity_key",
            "modality",
            "metric_name",
            "primary_family",
            "interval",
            "pool_key",
        )
    }
    coverage = cast(float, window.quality_factor)
    bound = window.candidate_service_key is not None
    rows = []
    paid = [
        step for step in session.steps if step.measurement.source_sha256 == window.source_sha256
    ]
    if len(paid) > 1:
        raise ValueError("strength source has repeated paid measurements")
    for step in paid:
        m = step.measurement
        for episode in m.episodes:
            points = tuple(m.points[i] for i in episode.sample_indices)
            if any(
                p.sample_index != i for p, i in zip(points, episode.sample_indices, strict=True)
            ):
                raise ValueError("episode point binding mismatch")
            z = (
                median(abs(p.signed_z) for p in points if p.signed_z is not None)
                if episode.qualifying
                else None
            )
            raw = -math.expm1(-z / 3) if z is not None else 0.0
            episode_kinds = {"local_anomaly_observed"} if episode.qualifying else set()
            if episode.qualifying and episode.direction == "up":
                episode_kinds.update(
                    p.predicate_kind
                    for p in step.feedback.predicates
                    if p.truth_value == "true" and p.measurement_field == "persistent_up"
                )
            rows.append(
                _term(
                    **base,
                    kind="paid_episode",
                    raw_strength=raw,
                    quality_factor=coverage,
                    weighted_strength=raw * coverage if bound and formal_scoring_eligible else None,
                    measurement_sha256=m.measurement_sha256,
                    episode_id=episode.episode_id,
                    timestamps_ns=tuple(p.timestamp_unix_ns for p in points),
                    predicate_kinds=tuple(sorted(episode_kinds)),
                    calculation=dict(
                        median_abs_z=z,
                        direction=episode.direction,
                        sample_count=len(points),
                        support_span_ns=episode.support_span_ns,
                    ),
                    reason_codes=(
                        "paid_qualifying_episode_median_not_peak"
                        if episode.qualifying
                        else "episode_not_qualifying",
                        *m.reason_codes,
                    ),
                )
            )
    return rows


def _error_terms(
    mechanisms: RepairMechanismEvidence, summaries: tuple[RepairErrorFactSummary, ...]
) -> list[RepairStrengthTerm]:
    by_pin = {s.result_sha256: s for s in summaries}
    rows = []
    for fact in mechanisms.error_facts.facts:
        baseline, fault = (by_pin[p] for p in fact.observation_sha256s)
        b, f = (
            next(
                (
                    g
                    for g in s.groups
                    if g.entity_key == fact.observed_entity_key and g.span_kind == fact.span_kind
                ),
                None,
            )
            for s in (baseline, fault)
        )
        complete = all(
            g is not None and g.known_denominator > 0 and g.known_denominator == g.observed_count
            for g in (b, f)
        )
        reasons = [
            "population_sampling_coverage_unknown",
            fact.statistic,
            fact.severity_basis,
            "no_root_permission_from_error_presence",
        ]
        if complete:
            assert (
                b is not None
                and f is not None
                and b.error_fraction is not None
                and f.error_fraction is not None
            )
            raw = max(0.0, f.error_fraction - b.error_fraction)
            reasons.append("fully_known_observed_fraction_delta")
        elif f is not None and f.observed_count > 0 and f.known_denominator > 0:
            raw = f.error_count / f.observed_count
            reasons.append("observed_error_share_lower_bound_not_request_rate")
        else:
            raw = None
            reasons.append("interpretable_status_denominator_unavailable")
        modality = "traces" if fact.modality == "trace" else "logs"
        rows.append(
            _term(
                evidence_id=fact.result_sha256,
                source_sha256=mechanisms.error_facts.source_bundle_sha256,
                candidate_service_key=fact.candidate_service_key,
                observed_entity_key=fact.observed_entity_key,
                modality=modality,
                metric_name=None,
                primary_family="error",
                interval=fact.interval,
                pool_key=_pool_key(fact.dataset, fact.observed_entity_key, modality, fact.interval),
                kind="error_status",
                raw_strength=raw,
                quality_factor=None,
                weighted_strength=raw if fact.candidate_service_key is not None else None,
                predicate_kinds=tuple(
                    v.predicate_kind for v in fact.facts if v.truth_value == "true"
                ),
                calculation=dict(
                    baseline_fraction=fact.baseline_fraction,
                    fault_fraction=fact.fault_fraction,
                    baseline_known_denominator=fact.baseline_known_denominator,
                    fault_known_denominator=fact.fault_known_denominator,
                    fault_observed_count=f.observed_count if f else 0,
                    fault_error_count=f.error_count if f else 0,
                    fault_unknown_or_conflict_count=f.observed_count - f.known_denominator
                    if f
                    else 0,
                    span_kind=fact.span_kind,
                ),
                reason_codes=tuple(sorted(set((*reasons, *fact.limitations)))),
            )
        )
    return rows


def _local_roles(
    terms: list[RepairStrengthTerm], mechanisms: RepairMechanismEvidence
) -> tuple[RepairStrengthTerm, ...]:
    """Bind only numerical witnesses of all necessary conditions; never borrow a role."""
    conditions = {c.condition_id: c for c in mechanisms.conditions}
    supports: dict[str, list[tuple[str, float]]] = {}
    for mechanism in mechanisms.mechanisms:
        if mechanism.disposition != "accepted" or "root_support" not in mechanism.permitted_roles:
            continue
        necessary = [
            conditions[c]
            for c in mechanism.condition_ids
            if conditions[c].requirement == "necessary"
        ]
        witnesses = []
        for condition in necessary:
            ids = {leaf.evidence_id for leaf in condition.leaves if leaf.truth_value == "true"}
            witnesses.append(
                [
                    t
                    for t in terms
                    if t.evidence_id in ids
                    and t.candidate_service_key == mechanism.candidate_service_key
                    and condition.predicate_kind in t.predicate_kinds
                    and t.weighted_strength is not None
                    and t.weighted_strength > 0
                    and t.quality_factor is not None
                ]
            )
        if not witnesses or any(not w for w in witnesses):
            continue
        strength = min(max(cast(float, t.quality_factor) for t in group) for group in witnesses)
        for term in (t for group in witnesses for t in group):
            supports.setdefault(term.term_id, []).append((mechanism.mechanism_id, strength))
    result = []
    for term in terms:
        claims = supports.get(term.term_id, [])
        support = max((q for _, q in claims), default=0.0)
        updated = term.model_copy(
            update=dict(
                role_support=support,
                role_mechanism_ids=tuple(sorted({key for key, _ in claims})),
                local_strength=None
                if term.weighted_strength is None
                else term.weighted_strength * (0.5 + 0.5 * support),
            )
        )
        unsigned = cast(dict[str, object], plain(updated))
        unsigned.pop("term_id")
        result.append(updated.model_copy(update={"term_id": _digest(unsigned)}))
    return tuple(sorted(result, key=lambda t: t.term_id))


def _compile_strength_values(
    *,
    mechanisms: RepairMechanismEvidence,
    session: RepairQuerySession,
    sources: tuple[RepairMeasurementSource, ...],
    error_summaries: tuple[RepairErrorFactSummary, ...],
) -> RepairStrengthEvidence:
    """Pure projection while the whole source/ledger/expansion guards remain open."""
    facts = {f.source_sha256: f for f in mechanisms.signal_facts}
    terms = [
        t for source in sources for t in _scalar_terms(source, facts[source.source_sha256], session)
    ]
    return _compile_strength_terms(
        mechanisms=mechanisms,
        session=session,
        scalar_terms=tuple(terms),
        error_summaries=error_summaries,
    )


@_module_call("strengths")
def _compile_strength_terms(
    *,
    mechanisms: RepairMechanismEvidence,
    session: (
        RepairQuerySession | NormalQuerySession | NormalHybridQuerySession | NormalHybridToolSession
    ),
    scalar_terms: tuple[RepairStrengthTerm, ...],
    error_summaries: tuple[RepairErrorFactSummary, ...],
) -> RepairStrengthEvidence:
    """Unchanged role/dependence reduction; inputs require owning guards."""
    terms = [*scalar_terms, *_error_terms(mechanisms, error_summaries)]
    final = _local_roles(terms, mechanisms)
    if len({t.term_id for t in final}) != len(final):
        raise ValueError("strength term identity collision")
    groups: dict[str, list[RepairStrengthTerm]] = {}
    for term in final:
        groups.setdefault(term.pool_key, []).append(term)
    pools = []
    for key, members in sorted(groups.items()):
        if len({t.candidate_service_key for t in members}) != 1:
            raise ValueError("dependence pool crossed candidate binding")
        eligible = [t for t in members if t.local_strength is not None]
        winner = (
            min(eligible, key=lambda t: (-cast(float, t.local_strength), t.term_id))
            if eligible
            else None
        )
        pools.append(
            RepairDependencePool(
                pool_key=key,
                candidate_service_key=members[0].candidate_service_key,
                retained_term_ids=tuple(t.term_id for t in members),
                selected_term_id=winner.term_id if winner else None,
                primary_family=winner.primary_family if winner else None,
                local_strength=winner.local_strength if winner else None,
            )
        )
    draft = RepairStrengthEvidence(
        semantics_sha256=REPAIR_STRENGTH_SEMANTICS_SHA256,
        opaque_case_id=mechanisms.opaque_case_id,
        session_sha256=mechanisms.session_sha256,
        candidate_service_keys=mechanisms.candidate_service_keys,
        mechanisms=mechanisms,
        terms=final,
        pools=tuple(pools),
        dynamic_queries_used=(len(session.steps) if session.schema_version ==
                              "nurse.v2.normal-hybrid-tool-session/1" else session.paid_count),
        result_sha256="0" * 64,
    )
    unsigned = cast(dict[str, object], plain(draft))
    unsigned.pop("result_sha256")
    return draft.model_copy(update={"result_sha256": _digest(unsigned)})


def build_repair_strengths(**source_arguments: Any) -> RepairStrengthEvidence:
    """Replay all current sources and only completed paid measurements, once per invocation."""
    with _replayed_mechanism_context(**source_arguments) as (mechanisms, session, summaries):
        result = _compile_strength_values(
            mechanisms=mechanisms,
            session=session,
            sources=source_arguments["runtime"].arguments["sources"],
            error_summaries=summaries,
        )
        canonical(result, _ROOTS)
        return result


def require_repair_strengths(
    *, strengths: RepairStrengthEvidence, expected_strengths_sha256: str, **source_arguments: Any
) -> RepairStrengthEvidence:
    if type(strengths) is not RepairStrengthEvidence:
        raise ValueError("exact repair strengths artifact required")
    before = snapshot(strengths, _ROOTS)
    try:
        require_pin(expected_strengths_sha256)
        rebuilt = build_repair_strengths(**source_arguments)
        if rebuilt.result_sha256 != expected_strengths_sha256 or canonical(
            rebuilt, _ROOTS
        ) != canonical(strengths, _ROOTS):
            raise ValueError("repair strengths differ from complete independent replay")
        return rebuilt
    finally:
        require_unchanged(strengths, _ROOTS, before)
