"""Paid, mechanism-grounded dependency symptoms; not observed causal failures.

This boundary reconstructs source blocks, the complete ledger and graph pins.
The outer final scorer must additionally authenticate the raw files/adapter.
No caller-authored witnesses, arbitrary feedback, scores or permission flags.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from pydantic import Field, ValidationInfo, field_validator

if TYPE_CHECKING:
    from nurse_rcl.v2.live.normal_artifacts import NormalQuerySession
    from nurse_rcl.v2.live.normal_hybrid_queries import NormalHybridQuerySession
    from nurse_rcl.v2.live.normal_hybrid_tools import NormalHybridToolSession

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.contracts import SHA256, TruthValue
from nurse_rcl.v2.dynamic_verification.predicate_evaluation import _compare_time_bounds
from nurse_rcl.v2.dynamic_verification.safety import plain, require_pin
from nurse_rcl.v2.graph_snapshots import GraphSnapshotPair
from nurse_rcl.v2.live.repair_dependency_graph import DependencyIndex
from nurse_rcl.v2.live.repair_local_scoring import RepairLocalScoring, _compile_local_values
from nurse_rcl.v2.live.repair_mechanisms import _replayed_mechanism_context
from nurse_rcl.v2.live.repair_module_timing import _module_call
from nurse_rcl.v2.live.repair_query_session import RepairQuerySession
from nurse_rcl.v2.live.repair_safety import RepairModel, canonical, require_unchanged, snapshot
from nurse_rcl.v2.live.repair_strengths import (
    REPAIR_STRENGTH_SEMANTICS_SHA256,
    RepairStrengthEvidence,
    _compile_strength_values,
)
from nurse_rcl.v2.live.repair_temporal import _onset_bounds
from nurse_rcl.v2.propagation_scoring import (
    _ComputationBudget,
    _top_two_f_gamma,
    build_propagation_scoring_config,
)
from nurse_rcl.v2.temporal_condensation import FrozenClockResolution

REPAIR_PROPAGATION_SCHEMA_VERSION: Final = "nurse.v2.live-repair-propagation/1"
_SOURCE_MECHANISMS = frozenset(
    (
        "local_resource_or_execution_pressure",
        "local_storage_or_network_failure",
        "workload_unavailability",
    )
)
REPAIR_PROPAGATION_SEMANTICS_SHA256: Final = sha256_hex(
    canonical_json_bytes(
        dict(
            version="nurse-v2-live-dependency-symptom-support/1",
            strengths=REPAIR_STRENGTH_SEMANTICS_SHA256,
            source_mechanisms=sorted(_SOURCE_MECHANISMS),
            source="complete_accepted_necessary_witness_pool_winner_positive_and_paid_onset",
            target="pool_winning_paid_persistent_latency_increase_ns_not_dependency_failure",
            relation="original_service_call_or_dependency_failure_reverse_dependency_structure",
            temporal="unique_bound_onsets_sufficient_graph_and_hop_coverage_strict_clock_order",
            graph="full_SCC_no_internal_order_original_bounded_paths_two_hops",
            scoring=build_propagation_scoring_config().config_sha256,
            claim="conditional_dependency_aligned_symptom_not_confirmed_causality",
        )
    )
)


class RepairPropagationWitness(RepairModel):
    witness_id: SHA256
    candidate_service_key: str
    term_id: SHA256
    pool_key: SHA256
    source_sha256: SHA256
    role: Literal["source", "symptom"]
    mechanism_ids: tuple[str, ...]
    measurement_sha256: SHA256 | None
    predicate_sha256: SHA256 | None
    episode_id: str | None
    support: float = Field(ge=0, le=1, allow_inf_nan=False)
    onset_bounds_ns: tuple[int, ...]
    reason_codes: tuple[str, ...]


class RepairPropagationPath(RepairModel):
    path_id: SHA256
    source_service_key: str
    target_service_key: str
    source_term_id: SHA256
    target_term_id: SHA256
    source_witness_id: SHA256
    target_witness_id: SHA256
    source_mechanism_ids: tuple[str, ...]
    component_members: tuple[tuple[str, ...], ...]
    hop_edge_ids: tuple[tuple[str, ...], ...]
    selected_edge_ids: tuple[str, ...]
    path_quality: float = Field(ge=0, le=1, allow_inf_nan=False)
    source_support: float = Field(ge=0, le=1, allow_inf_nan=False)
    target_symptom_support: float = Field(ge=0, le=1, allow_inf_nan=False)
    source_precedes_target: TruthValue
    contribution: float = Field(ge=0, le=1, allow_inf_nan=False)
    reason_codes: tuple[str, ...]
    claim_kind: Literal["dependency_aligned_symptom_support_not_confirmed_failure"] = (
        "dependency_aligned_symptom_support_not_confirmed_failure"
    )

    @field_validator("component_members", "hop_edge_ids", mode="before")
    @classmethod
    def json_nested_tuples(cls, value: object, info: ValidationInfo) -> object:
        # Exact JSON shapes are checked before restoration. Python input must
        # still supply the declared tuples, including their inner containers.
        if info.mode == "json" and type(value) in (list, tuple):
            return tuple(tuple(row) for row in cast(list[list[str]], value))
        return value


class RepairCandidatePropagation(RepairModel):
    candidate_service_key: str
    selected_target_service_keys: tuple[str, ...]
    selected_path_ids: tuple[str, ...]
    retained_path_ids: tuple[str, ...]
    propagation_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    reason_codes: tuple[str, ...]


class RepairPropagationEvidence(RepairModel):
    schema_version: Literal["nurse.v2.live-repair-propagation/1"] = (
        REPAIR_PROPAGATION_SCHEMA_VERSION
    )
    semantics_sha256: SHA256
    opaque_case_id: str
    local_scoring: RepairLocalScoring
    graph_pair: GraphSnapshotPair
    clock_resolution: FrozenClockResolution
    candidate_service_keys: tuple[str, ...]
    witnesses: tuple[RepairPropagationWitness, ...]
    paths: tuple[RepairPropagationPath, ...]
    candidates: tuple[RepairCandidatePropagation, ...]
    raw_file_authority: Literal[False] = False
    ranking_ready: Literal[False] = False
    result_sha256: SHA256


_ROOTS = (
    RepairPropagationEvidence,
    RepairLocalScoring,
    RepairStrengthEvidence,
    GraphSnapshotPair,
    FrozenClockResolution,
)


def _seal(value: Any, field: str) -> Any:
    raw = cast(dict[str, object], plain(value))
    raw.pop(field)
    return value.model_copy(update={field: sha256_hex(canonical_json_bytes(raw))})


def _witnesses(
    strengths: RepairStrengthEvidence,
    session: (
        RepairQuerySession | NormalQuerySession | NormalHybridQuerySession | NormalHybridToolSession
    ),
) -> tuple[RepairPropagationWitness, ...]:
    winners = {p.selected_term_id for p in strengths.pools}
    mechanisms = {m.mechanism_id: m for m in strengths.mechanisms.mechanisms}
    paid = {step.measurement.source_sha256: step for step in session.steps}
    rows = []
    for term in strengths.terms:
        if (
            term.term_id not in winners
            or term.candidate_service_key is None
            or term.weighted_strength is None
            or term.weighted_strength <= 0
            or term.quality_factor is None
        ):
            continue
        source_ids = tuple(
            i
            for i in term.role_mechanism_ids
            if mechanisms[i].mechanism_key in _SOURCE_MECHANISMS
            and mechanisms[i].disposition == "accepted"
            and "source_support" in mechanisms[i].permitted_roles
        )
        step = paid.get(term.source_sha256)
        predicates = () if step is None else step.feedback.predicates
        for role in ("source", "symptom"):
            if role == "source":
                if not source_ids or term.role_support <= 0:
                    continue
                permitted = {mechanisms[i].mechanism_key for i in source_ids}
                conditions = {
                    c.predicate_kind
                    for c in strengths.mechanisms.conditions
                    if c.mechanism_key in permitted
                    and c.requirement == "necessary"
                    and c.candidate_service_key == term.candidate_service_key
                }
                matches = [
                    p
                    for p in predicates
                    if p.mechanism_key in permitted
                    and p.predicate_kind in conditions
                    and p.predicate_kind in term.predicate_kinds
                    and p.truth_value == "true"
                ]
                quality = term.role_support
            else:
                if (
                    term.kind != "paid_episode"
                    or term.primary_family != "latency"
                    or "request_latency_increase" not in term.predicate_kinds
                ):
                    continue
                matches = [
                    p
                    for p in predicates
                    if p.predicate_kind == "request_latency_increase"
                    and p.truth_value == "true"
                    and p.unit == "ns"
                    and p.measurement_field == "persistent_up"
                ]
                if not matches:
                    continue
                quality = term.quality_factor
            chosen = min(matches, key=lambda p: p.result_sha256) if matches else None
            bounds = None
            reasons: tuple[str, ...] = ("not_measured",)
            if chosen is not None and step is not None:
                measured = step.measurement
                if term.kind == "paid_episode" and (
                    term.measurement_sha256 != measured.measurement_sha256
                    or not any(
                        e.episode_id == term.episode_id and e.qualifying for e in measured.episodes
                    )
                ):
                    raise ValueError("repair propagation episode/measurement binding differs")
                bounds, reasons = _onset_bounds(measured, chosen)
                gaps = tuple(r for r in measured.reason_codes if "gap" in r)
                if gaps:
                    bounds, reasons = None, tuple(sorted(set((*reasons, *gaps))))
            draft = RepairPropagationWitness(
                witness_id="0" * 64,
                candidate_service_key=term.candidate_service_key,
                term_id=term.term_id,
                pool_key=term.pool_key,
                source_sha256=term.source_sha256,
                role=role,
                mechanism_ids=source_ids if role == "source" else (),
                measurement_sha256=step.measurement.measurement_sha256 if step else None,
                predicate_sha256=chosen.result_sha256 if chosen else None,
                episode_id=term.episode_id,
                support=quality,
                onset_bounds_ns=() if bounds is None else bounds,
                reason_codes=reasons,
            )
            rows.append(_seal(draft, "witness_id"))
    return tuple(sorted(rows, key=lambda r: r.witness_id))


def _path_value(
    source: RepairPropagationWitness,
    target: RepairPropagationWitness,
    components: tuple[int, ...],
    index: DependencyIndex,
    graph: GraphSnapshotPair,
    clock: FrozenClockResolution,
) -> RepairPropagationPath:
    reasons = []
    if source.candidate_service_key == target.candidate_service_key:
        reasons.append("same_candidate_is_not_cross_service_propagation")
    if not components:
        reasons.append("no_dependency_path")
    elif len(components) == 1:
        reasons.append("same_scc_no_internal_order")
    if len(components) - 1 > 2:
        reasons.append("hop_limit_exceeded")
    if graph.fault_snapshot.coverage.coverage_state != "sufficient":
        reasons.append("graph_coverage_insufficient_or_unknown")
    if clock.resolution_status != "resolved":
        reasons.append("clock_resolution_unresolved")
    left = cast(tuple[int, int] | None, source.onset_bounds_ns or None)
    right = cast(tuple[int, int] | None, target.onset_bounds_ns or None)
    truth, time_reasons = _compare_time_bounds(
        left, right, clock, (*source.reason_codes, *target.reason_codes)
    )
    if left is None or right is None:
        reasons.append("onset_unavailable")
    if truth != "true":
        reasons.extend(("temporal_not_strict", *time_reasons))
    hops = index.hops(components)
    selected, qualities = [], []
    for hop in hops:
        eligible = []
        for edge in hop:
            if (
                edge.coverage.coverage_state != "sufficient"
                or edge.coverage.coverage_fraction is None
            ):
                continue
            if (
                left is not None
                and right is not None
                and not (
                    edge.interval.start_unix_ns <= min(left[0], right[0])
                    and max(left[1], right[1]) < edge.interval.end_unix_ns
                )
            ):
                continue
            eligible.append(edge)
        if not eligible:
            reasons.append("insufficient_hop_coverage_or_interval")
            qualities.append(0.0)
        else:
            edge = min(
                eligible, key=lambda e: (-cast(float, e.coverage.coverage_fraction), e.edge_id)
            )
            selected.append(edge.edge_id)
            qualities.append(cast(float, edge.coverage.coverage_fraction))
    path_quality = min(qualities, default=0.0)
    contribution = (
        path_quality * min(source.support, target.support) / len(hops)
        if not reasons and hops
        else 0.0
    )
    return cast(
        RepairPropagationPath,
        _seal(
            RepairPropagationPath(
                path_id="0" * 64,
                source_service_key=source.candidate_service_key,
                target_service_key=target.candidate_service_key,
                source_term_id=source.term_id,
                target_term_id=target.term_id,
                source_witness_id=source.witness_id,
                target_witness_id=target.witness_id,
                source_mechanism_ids=source.mechanism_ids,
                component_members=tuple(index.components[i] for i in components),
                hop_edge_ids=tuple(tuple(e.edge_id for e in hop) for hop in hops),
                selected_edge_ids=tuple(selected),
                path_quality=path_quality,
                source_support=source.support,
                target_symptom_support=target.support,
                source_precedes_target=truth,
                contribution=contribution,
                reason_codes=tuple(sorted(set(reasons))) if reasons else ("credited",),
            ),
            "path_id",
        ),
    )


@_module_call("propagation")
def _compile_propagation_values(
    *,
    local: RepairLocalScoring,
    session: (
        RepairQuerySession | NormalQuerySession | NormalHybridQuerySession | NormalHybridToolSession
    ),
    graph: GraphSnapshotPair,
    clock: FrozenClockResolution,
) -> RepairPropagationEvidence:
    witnesses = _witnesses(local.strengths, session)
    sources = tuple(w for w in witnesses if w.role == "source")
    targets = tuple(w for w in witnesses if w.role == "symptom")
    index = DependencyIndex(graph)
    config = build_propagation_scoring_config()
    budget = _ComputationBudget(
        config.max_bundle_pairs, config.max_expanded_states, config.max_traversed_edges
    )
    paths = []
    for source in sources:
        for target in targets:
            budget.charge_bundle_pair()
            options = index.paths(
                source.candidate_service_key,
                target.candidate_service_key,
                budget=budget,
                maximum_hops=config.propagation_max_hops,
            )
            for component_path in options or ((),):
                paths.append(_path_value(source, target, component_path, index, graph, clock))
    ordered = tuple(sorted(paths, key=lambda p: p.path_id))
    candidates = []
    for key in local.candidate_service_keys:
        retained = tuple(p for p in ordered if p.source_service_key == key)
        best: dict[str, RepairPropagationPath] = {}
        for path in retained:
            current = best.get(path.target_service_key)
            if path.contribution > 0 and (
                current is None
                or (-path.contribution, path.path_id) < (-current.contribution, current.path_id)
            ):
                best[path.target_service_key] = path
        selected = sorted(
            best.values(), key=lambda p: (-p.contribution, p.target_service_key, p.path_id)
        )[:2]
        score = _top_two_f_gamma(tuple(p.contribution for p in selected), gamma=config.gamma)
        reasons = set(r for p in retained for r in p.reason_codes)
        if not any(s.candidate_service_key == key for s in sources):
            reasons.add("missing_source_mechanism_witness")
        if not targets:
            reasons.add("missing_paid_latency_symptom")
        candidates.append(
            RepairCandidatePropagation(
                candidate_service_key=key,
                selected_target_service_keys=tuple(p.target_service_key for p in selected),
                selected_path_ids=tuple(p.path_id for p in selected),
                retained_path_ids=tuple(p.path_id for p in retained),
                propagation_score=score,
                reason_codes=tuple(sorted(reasons)),
            )
        )
    return cast(
        RepairPropagationEvidence,
        _seal(
            RepairPropagationEvidence(
                semantics_sha256=REPAIR_PROPAGATION_SEMANTICS_SHA256,
                opaque_case_id=local.opaque_case_id,
                local_scoring=local,
                graph_pair=graph,
                clock_resolution=clock,
                candidate_service_keys=local.candidate_service_keys,
                witnesses=witnesses,
                paths=ordered,
                candidates=tuple(candidates),
                result_sha256="0" * 64,
            ),
            "result_sha256",
        ),
    )


@contextmanager
def _replayed_propagation_context(**source_arguments: Any) -> Iterator[RepairPropagationEvidence]:
    with ExitStack() as guards:
        mechanisms, session, summaries = guards.enter_context(
            _replayed_mechanism_context(**source_arguments)
        )
        args = source_arguments["runtime"].arguments
        if (
            type(args.get("graph_pair")) is not GraphSnapshotPair
            or type(args.get("clock_resolution")) is not FrozenClockResolution
        ):
            raise ValueError(
                "repair propagation requires complete independently pinned graph/clock context"
            )
        graph = GraphSnapshotPair.model_validate_json(canonical(args["graph_pair"], _ROOTS))
        clock = FrozenClockResolution.model_validate_json(
            canonical(args["clock_resolution"], _ROOTS)
        )
        guards.callback(require_unchanged, (graph, clock), _ROOTS, snapshot((graph, clock), _ROOTS))
        strengths = _compile_strength_values(
            mechanisms=mechanisms,
            session=session,
            sources=args["sources"],
            error_summaries=summaries,
        )
        guards.callback(require_unchanged, strengths, _ROOTS, snapshot(strengths, _ROOTS))
        local = _compile_local_values(strengths)
        guards.callback(require_unchanged, local, _ROOTS, snapshot(local, _ROOTS))
        result = _compile_propagation_values(local=local, session=session, graph=graph, clock=clock)
        canonical(result, _ROOTS)
        guards.callback(require_unchanged, result, _ROOTS, snapshot(result, _ROOTS))
        yield result


def build_repair_propagation(**source_arguments: Any) -> RepairPropagationEvidence:
    with _replayed_propagation_context(**source_arguments) as result:
        return result


def require_repair_propagation(
    *, evidence: RepairPropagationEvidence, expected_evidence_sha256: str, **source_arguments: Any
) -> RepairPropagationEvidence:
    if type(evidence) is not RepairPropagationEvidence:
        raise ValueError("exact repair propagation artifact required")
    before = snapshot(evidence, _ROOTS)
    try:
        require_pin(expected_evidence_sha256)
        rebuilt = build_repair_propagation(**source_arguments)
        if rebuilt.result_sha256 != expected_evidence_sha256 or canonical(
            rebuilt, _ROOTS
        ) != canonical(evidence, _ROOTS):
            raise ValueError("repair propagation differs from complete independent replay")
        return rebuilt
    finally:
        require_unchanged(evidence, _ROOTS, before)
