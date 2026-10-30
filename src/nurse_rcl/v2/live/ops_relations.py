"""Explicit graph direction and measured onset cards; no invented causal edges."""

from __future__ import annotations

from nurse_rcl.v2.dynamic_verification.predicate_evaluation import _compare_time_bounds
from nurse_rcl.v2.graph_snapshots import GraphSnapshotPair
from nurse_rcl.v2.live.normal_artifacts import NormalEvidence
from nurse_rcl.v2.live.ops_contracts import OpsNodeEvidence
from nurse_rcl.v2.live.ops_node_common import node_card, prepared_snapshot_id
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData
from nurse_rcl.v2.temporal_condensation import FrozenClockResolution


def compare_dependency_timing(
    *, snapshot: str, graph: GraphSnapshotPair, clock: FrozenClockResolution,
    source_candidate: str, target_candidate: str,
    source_bounds: tuple[int, int] | None, target_bounds: tuple[int, int] | None,
    source_evidence_id: str, target_evidence_id: str,
) -> OpsNodeEvidence:
    """Pure non-scoring diagnostic over tool-resolved, previously measured onsets.

    The agent adapter resolves evidence IDs itself; it must never accept onset
    numbers or graph objects from model arguments. A service-call observation
    supports only a reverse-dependency *symptom* check, not a confirmed failure.
    """
    current = graph.fault_snapshot
    if clock.opaque_case_id != graph.opaque_case_id or (
        clock.time_contract_sha256 != graph.time_contract_sha256
        or source_candidate not in current.node_service_keys
        or target_candidate not in current.node_service_keys
    ):
        raise ValueError("dependency timing case/candidate/clock binding differs")
    for bounds in (source_bounds, target_bounds):
        if bounds is not None and (
            len(bounds) != 2 or any(type(n) is not int or n < 0 for n in bounds)
            or bounds[0] > bounds[1]
        ):
            raise ValueError("invalid measured onset bounds")
    edges = tuple(edge for edge in current.edges if (
        edge.callee_service_key == source_candidate
        and edge.caller_service_key == target_candidate
        and edge.observation_state == "observed"
        and (edge.relation_type, edge.direction) in (
            ("service_call", "caller_to_callee"),
            ("received_dependency_failure", "callee_to_caller"),
        )
    ))
    reasons = []
    if source_candidate == target_candidate:
        reasons.append("same_candidate_not_dependency_order")
    if not edges:
        reasons.append("no_matching_dependency_edge")
    if current.coverage.coverage_state != "sufficient":
        reasons.append("graph_coverage_insufficient_or_unknown")
    if edges and not any(edge.coverage.coverage_state == "sufficient" for edge in edges):
        reasons.append("edge_coverage_insufficient_or_unknown")
    if source_bounds is None or target_bounds is None:
        reasons.append("onset_unavailable")
    elif edges and not any(
        edge.interval.start_unix_ns <= min(source_bounds[0], target_bounds[0])
        and max(source_bounds[1], target_bounds[1]) < edge.interval.end_unix_ns
        and edge.coverage.coverage_state == "sufficient" for edge in edges
    ):
        reasons.append("onsets_outside_edge_window")
    truth, limitations = _compare_time_bounds(source_bounds, target_bounds, clock, tuple(reasons))
    return node_card(
        snapshot=snapshot, node="timing",
        parameters=dict(source_evidence_id=source_evidence_id,
                        target_evidence_id=target_evidence_id),
        payload=dict(
            source_candidate=source_candidate, target_candidate=target_candidate,
            source_bounds_ns=source_bounds, target_bounds_ns=target_bounds,
            clock_uncertainty_ns=clock.clock_uncertainty_ns,
            edges=[e.model_dump(mode="json") for e in edges], score_authority=False,
            claim_kind="dependency_aligned_symptom_not_confirmed_failure",
        ),
        evidence_ids=(source_evidence_id, target_evidence_id),
        source_refs=(graph.pair_sha256, clock.resolution_sha256),
        interval=current.interval, unit="unix_nanoseconds",
        coverage=current.coverage.coverage_fraction, truth=truth,
        availability="partial" if truth == "unknown" else "available",
        limitations=(*limitations, "timing_and_dependency_do_not_confirm_causality"),
    )


def dependency_nodes(data: RepairPreparationData) -> tuple[OpsNodeEvidence, ...]:
    snapshot = prepared_snapshot_id(data)
    graph = data.graph.graph_pair.fault_snapshot
    cards = tuple(node_card(
        snapshot=snapshot, node="dependency", parameters=dict(edge_id=edge.edge_id),
        payload=dict(edge=edge.model_dump(mode="json"),
                     caller_symptom_is_not_callee_failure=True,
                     propagation_is_not_causality=True),
        evidence_ids=(edge.edge_id,), source_refs=(edge.edge_sha256, *edge.source_record_ids),
        interval=edge.interval, coverage=edge.coverage.coverage_fraction,
        truth="true" if edge.observation_state == "observed" else "unknown",
        availability="available" if edge.observation_state == "observed" else "partial",
        limitations=("edge_observation_not_causal_confirmation",
                     "caller_latency_not_callee_self_time", *data.graph.limitations),
    ) for edge in graph.edges)
    if cards:
        return cards
    return (node_card(
        snapshot=snapshot, node="dependency", parameters={}, payload=dict(edges=[]),
        evidence_ids=(graph.snapshot_sha256,), source_refs=(data.graph.result_sha256,),
        interval=graph.interval, coverage=graph.coverage.coverage_fraction,
        availability="unavailable", limitations=("no_observed_dependency_edges",),
    ),)


def timing_nodes(
    data: RepairPreparationData, evidence: NormalEvidence,
) -> tuple[OpsNodeEvidence, ...]:
    snapshot = prepared_snapshot_id(data)
    interval = data.diagnostics.legacy.sources.adapter.time_contract.fault_window
    cards = []
    for step in evidence.session.steps:
        for onset in step.feedback.onsets:
            cards.append(node_card(
                snapshot=snapshot, node="timing", parameters=dict(need_key=onset.need_key),
                payload=dict(onset=onset.model_dump(mode="json"),
                             clock=data.clock.model_dump(mode="json")),
                evidence_ids=(onset.predicate_sha256,),
                source_refs=(step.measurement.measurement_sha256,), interval=interval,
                unit="unix_nanoseconds", coverage=step.measurement.quality.coverage_quality,
                availability="available" if onset.bounds_ns is not None else "unavailable",
                limitations=(*onset.reason_codes, "onset_interval_not_causal_order"),
            ))
    for path in evidence.ranking.propagation.paths:
        cards.append(node_card(
            snapshot=snapshot, node="timing", parameters=dict(path_id=path.path_id),
            payload=dict(path=path.model_dump(mode="json")), evidence_ids=(path.path_id,),
            source_refs=(evidence.ranking.propagation.result_sha256,), interval=interval,
            coverage=path.path_quality, truth=path.source_precedes_target,
            limitations=(*path.reason_codes, "dependency_aligned_symptom_not_confirmed_failure"),
        ))
    if not cards:
        cards.append(node_card(
            snapshot=snapshot, node="timing", parameters={}, payload=dict(onsets=[]),
            evidence_ids=(evidence.session.session_sha256,),
            source_refs=(data.catalog.catalog_sha256,), interval=interval,
            availability="unavailable", limitations=("no_paid_onset_observations",),
        ))
    return tuple(cards)
