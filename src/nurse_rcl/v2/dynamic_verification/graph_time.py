"""Pre-query graph/event time domains and pure consumption of existing A4 results.

No source reads, roles or score permissions are granted here. Snapshot pins must
be supplied independently; session/guard still replay the real source blocks.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Literal

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.diagnostic_schemas import DiagnosticHypothesis
from nurse_rcl.v2.event_grouping import EventGroup
from nurse_rcl.v2.graph_snapshots import GraphSnapshot
from nurse_rcl.v2.hypothesis_guard import _PREDICATE_EVIDENCE_CONTRACTS
from nurse_rcl.v2.observation_schemas import CaseTimeContract, TimeInterval
from nurse_rcl.v2.propagation_scoring import (
    _MAX_BUNDLE_PAIRS,
    _MECHANISM_ROLE_RELATION_POLICY,
    PROPAGATION_RELATION_POLICY_SHA256,
)
from nurse_rcl.v2.rule_hypotheses import RuleHypothesisEnvelope
from nurse_rcl.v2.temporal_condensation import (
    FrozenClockResolution,
    require_frozen_clock_resolution,
)

from .contracts import (
    DynamicExpressionNode,
    DynamicFieldReference,
    DynamicGraphTimeBinding,
    DynamicGraphTimePair,
    DynamicTimeSide,
    DynamicVerificationPolicy,
    DynamicVerificationRequest,
    SignalEpisodeMeasurement,
    VerificationSourceCatalog,
)
from .predicate_evaluation import (
    _BOUND_ROOTS,
    _PERSISTENCE_FIELDS,
    _build_expression_leaf,
    _check_condition_sources,
    _compare_time_bounds,
    _time_side_bounds,
    _TimeSideBounds,
)
from .requests import build_dynamic_verification_policy, build_dynamic_verification_requests
from .safety import canonical, plain, require_pin, require_unchanged, snapshot

_ROOTS = (
    RuleHypothesisEnvelope,
    CaseTimeContract,
    VerificationSourceCatalog,
    DynamicVerificationPolicy,
    DynamicVerificationRequest,
    GraphSnapshot,
    DynamicGraphTimePair,
    SignalEpisodeMeasurement,
    FrozenClockResolution,
)
_DIRECTIONS = {
    "received_dependency_failure": "callee_to_caller",
    "caller_load_transmission": "caller_to_callee",
    "service_call": "caller_to_callee",
    "shared_resource_association": "undirected_association",
}


def _event_binding(
    *,
    seed: DiagnosticHypothesis,
    event: EventGroup,
    role: Literal["source_support", "symptom_support"],
    envelope: RuleHypothesisEnvelope,
    catalog: VerificationSourceCatalog,
    requests: tuple[DynamicVerificationRequest, ...],
) -> DynamicGraphTimeBinding:
    predicates = tuple(
        sorted(
            (
                row
                for row in envelope.rule_hypothesis_set.predicates
                if row.predicate_result_id in seed.necessary_predicate_result_ids
            ),
            key=lambda row: row.predicate_result_id,
        )
    )
    observations = {row.observation_id: row for row in envelope.event_grouping.observations}
    # Only approved directional necessary predicates may bind a propagation side.
    eligible = {
        row.observation_id: tuple(
            sorted(
                {
                    pred.predicate_kind
                    for pred in predicates
                    if _PERSISTENCE_FIELDS.get(pred.predicate_kind) == "persistent_up"
                    and row.observation_id in pred.observation_ids
                    and (
                        event.primary_signal_family
                        in _PREDICATE_EVIDENCE_CONTRACTS[
                            pred.predicate_kind
                        ].primary_signal_families
                        or observations[row.observation_id].signal_family
                        in _PREDICATE_EVIDENCE_CONTRACTS[pred.predicate_kind].source_signal_families
                    )
                }
            )
        )
        for row in catalog.rows
        if row.candidate_service_key == seed.candidate_service_key
        and row.observation_id in event.observation_ids
    }
    bound_ids = tuple(sorted(key for key, kinds in eligible.items() if kinds))
    series = tuple(
        sorted(
            {
                row.canonical_series_key
                for row in catalog.rows
                if row.observation_id in bound_ids and row.canonical_series_key is not None
            }
        )
    )
    domain = (
        DynamicTimeSide(
            candidate_service_key=seed.candidate_service_key,
            predicate_kinds=tuple(sorted({kind for key in bound_ids for kind in eligible[key]})),
            observation_ids=bound_ids,
            event_group_ids=(event.event_group_id,),
            canonical_series_keys=series,
            request_keys=tuple(
                sorted(
                    row.canonical_request_key
                    for row in requests
                    if row.canonical_series_key in series
                )
            ),
            direction="up",
        )
        if bound_ids
        else None
    )
    return DynamicGraphTimeBinding(
        candidate_service_key=seed.candidate_service_key,
        event_group_id=event.event_group_id,
        mechanism_key=seed.mechanism_key,
        role=role,
        seed=seed,
        event=event,
        necessary_predicates=predicates,
        domain=domain,
        reason_codes=("frozen_necessary_event_sources",)
        if domain is not None
        else ("no_supported_directional_necessary_source",),
    )


def _build_graph_time_pairs(
    *,
    envelope: RuleHypothesisEnvelope,
    time_contract: CaseTimeContract,
    catalog: VerificationSourceCatalog,
    requests: tuple[DynamicVerificationRequest, ...],
    graph_snapshots: tuple[GraphSnapshot, ...],
    expected_graph_snapshot_sha256s: tuple[str, ...],
) -> tuple[DynamicGraphTimePair, ...]:
    """Freeze every event binding without observing or selecting a measurement."""
    inputs = (
        envelope,
        time_contract,
        catalog,
        requests,
        graph_snapshots,
        expected_graph_snapshot_sha256s,
    )
    before = snapshot(inputs, _ROOTS)
    try:
        with ExitStack() as guards:
            if (
                type(envelope) is not RuleHypothesisEnvelope
                or type(time_contract) is not CaseTimeContract
                or type(catalog) is not VerificationSourceCatalog
                or type(requests) is not tuple
                or any(type(row) is not DynamicVerificationRequest for row in requests)
                or type(graph_snapshots) is not tuple
                or any(type(row) is not GraphSnapshot for row in graph_snapshots)
                or type(expected_graph_snapshot_sha256s) is not tuple
                or len(graph_snapshots) != 2
                or len(expected_graph_snapshot_sha256s) != 2
            ):
                raise ValueError("graph time pairs require exact complete frozen inputs")
            source = RuleHypothesisEnvelope.model_validate_json(canonical(envelope, _ROOTS))
            contract = CaseTimeContract.model_validate_json(canonical(time_contract, _ROOTS))
            sources = VerificationSourceCatalog.model_validate_json(canonical(catalog, _ROOTS))
            policy = build_dynamic_verification_policy()
            working = (source, contract, sources, policy)
            guards.callback(require_unchanged, working, _ROOTS, snapshot(working, _ROOTS))
            expected = build_dynamic_verification_requests(
                envelope=source,
                time_contract=contract,
                catalog=sources,
                expected_catalog_sha256=sources.catalog_sha256,
                policy=policy,
                expected_policy_sha256=policy.policy_sha256,
            )
            guards.callback(require_unchanged, expected, _ROOTS, snapshot(expected, _ROOTS))
            if canonical(requests, _ROOTS) != canonical(expected, _ROOTS):
                raise ValueError("graph time request directory differs from frozen time/sources")
            _check_condition_sources(source, sources, expected, ())
            graphs = _prepare_graph_time_snapshots(
                envelope=source,
                time_contract=contract,
                graph_snapshots=graph_snapshots,
                expected_graph_snapshot_sha256s=expected_graph_snapshot_sha256s,
            )
            guards.callback(require_unchanged, graphs, _ROOTS, snapshot(graphs, _ROOTS))
            return _graph_time_pair_values(
                envelope=source,
                time_contract=contract,
                catalog=sources,
                requests=expected,
                graph_snapshots=graphs,
            )
    finally:
        require_unchanged(inputs, _ROOTS, before)


def _prepare_graph_time_snapshots(
    *,
    envelope: RuleHypothesisEnvelope,
    time_contract: CaseTimeContract,
    graph_snapshots: tuple[GraphSnapshot, ...],
    expected_graph_snapshot_sha256s: tuple[str, ...],
) -> tuple[GraphSnapshot, ...]:
    """Authenticate the complete independently pinned graph domain for this call."""
    inputs = (envelope, time_contract, graph_snapshots, expected_graph_snapshot_sha256s)
    before = snapshot(inputs, _ROOTS)
    try:
        if (
            type(envelope) is not RuleHypothesisEnvelope
            or type(time_contract) is not CaseTimeContract
            or type(graph_snapshots) is not tuple
            or any(type(row) is not GraphSnapshot for row in graph_snapshots)
            or type(expected_graph_snapshot_sha256s) is not tuple
            or len(graph_snapshots) != 2
            or len(expected_graph_snapshot_sha256s) != 2
        ):
            raise ValueError("graph time pairs require exact complete frozen inputs")
        source, contract = envelope, time_contract
        graphs = tuple(
            GraphSnapshot.model_validate_json(canonical(row, _ROOTS)) for row in graph_snapshots
        )
        graph_before = snapshot(graphs, _ROOTS)
        try:
            if {row.window_kind for row in graphs} != {"normal", "fault"}:
                raise ValueError("graph time requires the normal and fault snapshot domain")
            for graph, pin in zip(graphs, expected_graph_snapshot_sha256s, strict=True):
                if graph.snapshot_sha256 != require_pin(pin) or not graph.verify():
                    raise ValueError("graph time snapshot independent pin/integrity mismatch")
                if len({edge.edge_id for edge in graph.edges}) != len(graph.edges):
                    raise ValueError("graph time snapshot contains duplicate edges")
                if (
                    graph.opaque_case_id != contract.opaque_case_id
                    or graph.candidate_freeze_sha256 != source.frozen_candidates.freeze_sha256
                    or graph.candidate_universe_sha256 != source.candidate_universe.universe_sha256
                    or graph.entity_registry_sha256 != source.entity_registry.registry_sha256
                    or graph.observation_contract_sha256
                    != source.candidate_universe.observation_contract_sha256
                    or graph.node_service_keys != source.frozen_candidates.candidate_service_keys
                    or graph.decision_time_unix_ns != contract.decision_time_unix_ns
                    or graph.interval
                    != (
                        contract.baseline_window
                        if graph.window_kind == "normal"
                        else contract.fault_window
                    )
                ):
                    raise ValueError("graph time snapshot differs from frozen case/C/time/registry")
                for edge in graph.edges:
                    if (
                        edge.direction != _DIRECTIONS[edge.relation_type]
                        or edge.interval.start_unix_ns < graph.interval.start_unix_ns
                        or edge.interval.end_unix_ns > graph.interval.end_unix_ns
                    ):
                        raise ValueError("graph time edge direction/window is invalid")
            return graphs
        finally:
            require_unchanged(graphs, _ROOTS, graph_before)
    finally:
        require_unchanged(inputs, _ROOTS, before)


def _graph_time_pair_values(
    *,
    envelope: RuleHypothesisEnvelope,
    time_contract: CaseTimeContract,
    catalog: VerificationSourceCatalog,
    requests: tuple[DynamicVerificationRequest, ...],
    graph_snapshots: tuple[GraphSnapshot, ...],
) -> tuple[DynamicGraphTimePair, ...]:
    """Enumerate unchanged graph/event values inside an owning protected operation."""
    source, contract, sources, expected = envelope, time_contract, catalog, requests
    graphs = graph_snapshots
    groups = {row.event_group_id: row for row in source.event_grouping.event_groups}
    pairs: list[DynamicGraphTimePair] = []
    for graph in graphs:
        for edge in graph.edges:
            if not edge.allows_directed_propagation:
                continue
            lower = max(edge.interval.start_unix_ns, contract.fault_window.start_unix_ns)
            upper = min(edge.interval.end_unix_ns, contract.fault_window.end_unix_ns)
            if lower >= upper:
                continue
            source_key, target_key = edge.caller_service_key, edge.callee_service_key
            if edge.direction == "callee_to_caller":
                source_key, target_key = target_key, source_key
            sides: dict[str, list[DynamicGraphTimeBinding]] = {
                "source_support": [],
                "symptom_support": [],
            }
            role_candidates: tuple[
                tuple[Literal["source_support", "symptom_support"], str], ...
            ] = (
                ("source_support", source_key),
                ("symptom_support", target_key),
            )
            for role, candidate in role_candidates:
                for seed in source.rule_hypothesis_set.hypotheses:
                    if seed.candidate_service_key != candidate or not any(
                        seed.mechanism_key == mechanism
                        and role == allowed_role
                        and edge.relation_type in relations
                        for mechanism, allowed_role, relations in (_MECHANISM_ROLE_RELATION_POLICY)
                    ):
                        continue
                    for event_id in seed.event_group_ids:
                        sides[role].append(
                            _event_binding(
                                seed=seed,
                                event=groups[event_id],
                                role=role,
                                envelope=source,
                                catalog=sources,
                                requests=expected,
                            )
                        )
            if (
                len(pairs) + len(sides["source_support"]) * len(sides["symptom_support"])
                > _MAX_BUNDLE_PAIRS
            ):
                raise ValueError("graph time binding pair resource limit exceeded; no truncation")
            for left in sides["source_support"]:
                for right in sides["symptom_support"]:
                    document = {
                        "condition_key": "condition_"
                        + sha256_hex(
                            canonical_json_bytes(
                                [
                                    edge.edge_sha256,
                                    left.seed.hypothesis_id,
                                    left.event_group_id,
                                    right.seed.hypothesis_id,
                                    right.event_group_id,
                                ]
                            )
                        ),
                        "opaque_case_id": sources.opaque_case_id,
                        "candidate_freeze_sha256": sources.candidate_freeze_sha256,
                        "time_contract_sha256": sources.time_contract_sha256,
                        "catalog_sha256": sources.catalog_sha256,
                        "source_rule_set_sha256": source.rule_hypothesis_set.result_sha256,
                        "source_kind": "graph_edge",
                        "graph_snapshot_sha256": graph.snapshot_sha256,
                        "relation_policy_sha256": PROPAGATION_RELATION_POLICY_SHA256,
                        "edge": plain(edge),
                        "source": plain(left),
                        "target": plain(right),
                        "relation_window": plain(
                            TimeInterval(start_unix_ns=lower, end_unix_ns=upper)
                        ),
                    }
                    document["time_pair_key"] = "timepair_" + sha256_hex(
                        canonical_json_bytes(document)
                    )
                    pairs.append(
                        DynamicGraphTimePair.model_validate_json(canonical_json_bytes(document))
                    )
    return tuple(sorted(pairs, key=lambda row: row.time_pair_key))


def _evaluate_graph_time_pair(
    *,
    pair: DynamicGraphTimePair,
    envelope: RuleHypothesisEnvelope,
    time_contract: CaseTimeContract,
    catalog: VerificationSourceCatalog,
    requests: tuple[DynamicVerificationRequest, ...],
    graph_snapshots: tuple[GraphSnapshot, ...],
    expected_graph_snapshot_sha256s: tuple[str, ...],
    measurements: tuple[SignalEpisodeMeasurement, ...],
    clock_resolution: FrozenClockResolution,
    expected_clock_resolution_sha256: str,
    pending_demand_keys: tuple[str, ...],
) -> DynamicExpressionNode:
    inputs = (
        pair,
        envelope,
        time_contract,
        catalog,
        requests,
        graph_snapshots,
        expected_graph_snapshot_sha256s,
        measurements,
        clock_resolution,
        expected_clock_resolution_sha256,
        pending_demand_keys,
    )
    before = snapshot(inputs, _ROOTS)
    try:
        with ExitStack() as guards:
            if (
                type(pair) is not DynamicGraphTimePair
                or type(measurements) is not tuple
                or any(type(row) is not SignalEpisodeMeasurement for row in measurements)
                or type(clock_resolution) is not FrozenClockResolution
                or type(pending_demand_keys) is not tuple
                or any(type(key) is not str for key in pending_demand_keys)
            ):
                raise ValueError("graph time evaluation requires exact frozen inputs")
            expected = _build_graph_time_pairs(
                envelope=envelope,
                time_contract=time_contract,
                catalog=catalog,
                requests=requests,
                graph_snapshots=graph_snapshots,
                expected_graph_snapshot_sha256s=expected_graph_snapshot_sha256s,
            )
            guards.callback(require_unchanged, expected, _ROOTS, snapshot(expected, _ROOTS))
            fresh = next(
                (row for row in expected if canonical(row, _ROOTS) == canonical(pair, _ROOTS)), None
            )
            if fresh is None:
                raise ValueError("graph time pair differs from frozen event/edge source")
            resolution = require_frozen_clock_resolution(
                clock_resolution=clock_resolution,
                time_contract=time_contract,
                expected_clock_resolution_sha256=expected_clock_resolution_sha256,
            )
            guards.callback(require_unchanged, resolution, _ROOTS, snapshot(resolution, _ROOTS))
            results = tuple(
                SignalEpisodeMeasurement.model_validate_json(canonical(row, _ROOTS))
                for row in measurements
            )
            guards.callback(require_unchanged, results, _ROOTS, snapshot(results, _ROOTS))
            _check_condition_sources(envelope, catalog, requests, results)
            sides: list[_TimeSideBounds] = []
            for binding in (fresh.source, fresh.target):
                bounds: _TimeSideBounds = (
                    _time_side_bounds(
                        side=binding.domain,
                        window=fresh.relation_window,
                        envelope=envelope,
                        catalog=catalog,
                        requests=requests,
                        measurements=results,
                    )
                    if binding.domain is not None
                    else (None, (), (), binding.reason_codes, False)
                )
                guards.callback(
                    require_unchanged, bounds, _BOUND_ROOTS, snapshot(bounds, _BOUND_ROOTS)
                )
                sides.append(bounds)
            left, right = sides
            return _graph_time_comparison_values(
                fresh=fresh,
                resolution=resolution,
                left=left,
                right=right,
                pending_demand_keys=pending_demand_keys,
            )
    finally:
        require_unchanged(inputs, _ROOTS, before)


def _graph_time_comparison_values(
    *,
    fresh: DynamicGraphTimePair,
    resolution: FrozenClockResolution,
    left: _TimeSideBounds,
    right: _TimeSideBounds,
    pending_demand_keys: tuple[str, ...],
) -> DynamicExpressionNode:
    """Pure event/edge comparison; the enclosing call owns source certification."""
    truth, reasons = _compare_time_bounds(left[0], right[0], resolution, left[3] + right[3])
    refs = (
        *left[1],
        *right[1],
        *(
            DynamicFieldReference(
                source_kind="time_pair", source_sha256=fresh.time_pair_sha256, field_path=field
            )
            for field in (
                "source",
                "target",
                "relation_window",
                "edge",
                "graph_snapshot_sha256",
                "relation_policy_sha256",
            )
        ),
        *(
            DynamicFieldReference(
                source_kind="clock_resolution",
                source_sha256=resolution.resolution_sha256,
                field_path=field,
            )
            for field in ("clock_uncertainty_ns", "resolution_status")
        ),
    )
    return _build_expression_leaf(
        kind="directional_time",
        truth_value=truth,
        field_references=refs,
        support_sources=(*left[2], *right[2]) if truth == "true" else (),
        pending_demand_keys=pending_demand_keys if left[4] or right[4] else (),
        reason_codes=tuple(reasons),
    )
