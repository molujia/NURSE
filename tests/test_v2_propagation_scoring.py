from __future__ import annotations

import importlib
import traceback
from itertools import pairwise
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.candidate_freeze import freeze_candidate_universe
from nurse_rcl.v2.candidate_universe import build_complete_candidate_universe
from nurse_rcl.v2.diagnostic_schemas import (
    MeasuredValue,
    PredicateThreshold,
    build_diagnostic_hypothesis,
    build_role_assertion,
)
from nurse_rcl.v2.event_grouping import build_event_grouping_result
from nurse_rcl.v2.graph_snapshots import (
    GRAPH_SNAPSHOT_SCHEMA_VERSION,
    GraphCoverageSummary,
    GraphMetricSummary,
    GraphSnapshot,
    GraphSnapshotEdge,
)
from nurse_rcl.v2.hypothesis_guard import execute_hypothesis_guard
from nurse_rcl.v2.mechanism_library import build_mechanism_library
from nurse_rcl.v2.observation_schemas import (
    CaseTimeContract,
    Observation,
    TimeInterval,
)
from nurse_rcl.v2.predicate_logic import evaluate_predicate
from nurse_rcl.v2.role_compiler import (
    build_role_binding_proposal,
    compile_event_roles,
)
from nurse_rcl.v2.temporal_condensation import (
    DirectedPropagationRelation,
    build_onset_interval,
    build_relation_scc_condensation,
    case_time_contract_sha256,
    evaluate_clock_aware_precedence,
    freeze_clock_resolution,
)
from nurse_rcl.v2.temporal_entity_registry import build_temporal_entity_registry
from tests.test_v2_candidate_universe import (
    DECISION_TIME,
    entity_input,
    metadata_record,
)
from tests.test_v2_event_grouping import claim, family_mapping, observation


def propagation_module() -> ModuleType:
    try:
        return importlib.import_module("nurse_rcl.v2.propagation_scoring")
    except ModuleNotFoundError:
        pytest.fail("the bounded V2 propagation scorer is not implemented", pytrace=False)


def _time_contract() -> CaseTimeContract:
    return CaseTimeContract(
        opaque_case_id="opq_" + "a" * 32,
        t0_unix_ns=50,
        t0_source="shared_detector",
        localization_scope="detector_triggered",
        baseline_window=TimeInterval(start_unix_ns=1, end_unix_ns=50),
        fault_window=TimeInterval(start_unix_ns=50, end_unix_ns=100),
        decision_time_unix_ns=100,
        timezone="UTC",
        expected_cadence_ns=10,
        clock_uncertainty_ns=0,
        missing_modalities=(),
        late_data_observed=False,
    )


def _edge(
    *,
    coverage_fraction: float,
    coverage_state: str,
    caller_service_key: str = "service-a",
    callee_service_key: str = "service-b",
    tag: str = "a-b",
    relation_type: DirectedPropagationRelation = "caller_load_transmission",
) -> GraphSnapshotEdge:
    coverage = GraphCoverageSummary(
        observation_state="observed",
        coverage_state=coverage_state,
        valid_sample_count=int(coverage_fraction * 4),
        expected_sample_count=4,
        coverage_fraction=coverage_fraction,
        minimum_coverage_fraction=0.5 if coverage_state == "sufficient" else 0.8,
        provenance_ids=(f"coverage:{tag}",),
    )
    unavailable = GraphMetricSummary(
        state="unknown",
        value=None,
        aggregation="not_computable",
        contributing_source_record_ids=(),
    )
    unsigned: dict[str, object] = {
        "caller_service_key": caller_service_key,
        "callee_service_key": callee_service_key,
        "relation_type": relation_type,
        "direction": (
            "callee_to_caller"
            if relation_type == "received_dependency_failure"
            else "caller_to_callee"
        ),
        "allows_directed_propagation": True,
        "interval": TimeInterval(start_unix_ns=10, end_unix_ns=100).model_dump(mode="json"),
        "observation_state": "observed",
        "coverage": coverage.model_dump(mode="json"),
        "traffic": unavailable.model_dump(mode="json"),
        "error_count": unavailable.model_dump(mode="json"),
        "error_rate": unavailable.model_dump(mode="json"),
        "latency_p95_seconds": unavailable.model_dump(mode="json"),
        "source_record_ids": (f"graph-source:{tag}",),
        "provenance_ids": (f"graph-provenance:{tag}",),
    }
    digest = sha256_hex(canonical_json_bytes(unsigned))
    return GraphSnapshotEdge.model_validate(
        {**unsigned, "edge_id": f"gedge_{digest[:24]}", "edge_sha256": digest}
    )


def _condensation(
    *,
    coverage_fraction: float = 0.75,
    coverage_state: str = "sufficient",
    edges: tuple[GraphSnapshotEdge, ...] | None = None,
    relation_type: DirectedPropagationRelation = "caller_load_transmission",
) -> Any:
    graph_edges = edges or (
        _edge(
            coverage_fraction=coverage_fraction,
            coverage_state=coverage_state,
            relation_type=relation_type,
        ),
    )
    graph_edges = tuple(
        sorted(
            graph_edges,
            key=lambda item: (
                item.relation_type,
                item.caller_service_key,
                item.callee_service_key,
            ),
        )
    )
    edge = graph_edges[0]
    unsigned: dict[str, object] = {
        "schema_version": GRAPH_SNAPSHOT_SCHEMA_VERSION,
        "opaque_case_id": "opq_" + "a" * 32,
        "window_kind": "fault",
        "interval": TimeInterval(start_unix_ns=10, end_unix_ns=100).model_dump(mode="json"),
        "decision_time_unix_ns": 100,
        "observation_contract_sha256": "1" * 64,
        "candidate_universe_sha256": "2" * 64,
        "candidate_freeze_sha256": "3" * 64,
        "entity_registry_sha256": "4" * 64,
        "node_service_keys": tuple(
            sorted(
                {
                    service_key
                    for graph_edge in graph_edges
                    for service_key in (
                        graph_edge.caller_service_key,
                        graph_edge.callee_service_key,
                    )
                }
                | {"service-a", "service-b", "service-c"}
            )
        ),
        "coverage": edge.coverage.model_dump(mode="json"),
        "edges": tuple(item.model_dump(mode="json") for item in graph_edges),
        "provenance_ids": tuple(
            sorted(
                {
                    provenance_id
                    for item in graph_edges
                    for provenance_id in (*item.provenance_ids, *item.coverage.provenance_ids)
                }
            )
        ),
    }
    snapshot = GraphSnapshot.model_validate(
        {**unsigned, "snapshot_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    return build_relation_scc_condensation(
        graph_snapshot=snapshot,
        relation_type=relation_type,
    )


def _inputs(*, coverage_fraction: float = 0.75, overlap: bool = False) -> dict[str, object]:
    return _scenario(
        edge_specs=(("service-a", "service-b", coverage_fraction),),
        target_specs=(("service-b", "2", 19 if overlap else 30, 0.6),),
        additional_service_keys=("service-c",),
    )


def _scenario(
    *,
    edge_specs: tuple[tuple[str, str, float], ...],
    target_specs: tuple[tuple[str, str, int, float], ...],
    relation_type: DirectedPropagationRelation = "received_dependency_failure",
    source_strength: float = 0.8,
    include_source_role: bool = True,
    include_target_roles: bool = True,
    source_mechanism_key: str | None = None,
    target_mechanism_key: str | None = None,
    additional_service_keys: tuple[str, ...] = (),
    source_role_interval: TimeInterval | None = None,
    target_role_interval: TimeInterval | None = None,
    include_source_peer_role: bool = False,
    include_target_peer_roles: bool = False,
    source_peer_role_interval: TimeInterval | None = None,
    target_peer_role_interval: TimeInterval | None = None,
) -> dict[str, object]:
    return _authoritative_scenario(
        edge_specs=edge_specs,
        target_specs=target_specs,
        relation_type=relation_type,
        source_strength=source_strength,
        include_source_role=include_source_role,
        include_target_roles=include_target_roles,
        source_mechanism_key=source_mechanism_key,
        target_mechanism_key=target_mechanism_key,
        additional_service_keys=additional_service_keys,
        source_role_interval=source_role_interval,
        target_role_interval=target_role_interval,
        include_source_peer_role=include_source_peer_role,
        include_target_peer_roles=include_target_peer_roles,
        source_peer_role_interval=source_peer_role_interval,
        target_peer_role_interval=target_peer_role_interval,
    )


def _resign_model(
    model: Any,
    *,
    prefix: str,
    id_field: str,
    sha_field: str,
    updates: dict[str, object],
) -> Any:
    unsigned = model.model_dump(mode="json", exclude={id_field, sha_field})
    unsigned.update(updates)
    digest = sha256_hex(canonical_json_bytes(unsigned))
    return type(model).model_validate(
        {**unsigned, id_field: f"{prefix}_{digest[:24]}", sha_field: digest}
    )


def _authoritative_scenario(
    *,
    edge_specs: tuple[tuple[str, str, float], ...] = (("service-a", "service-b", 0.75),),
    target_specs: tuple[tuple[str, str, int, float], ...] = (("service-b", "2", 30, 0.6),),
    relation_type: DirectedPropagationRelation = "received_dependency_failure",
    source_strength: float = 0.8,
    include_source_role: bool = True,
    include_target_roles: bool = True,
    source_mechanism_key: str | None = None,
    target_mechanism_key: str | None = None,
    additional_service_keys: tuple[str, ...] = (),
    source_role_interval: TimeInterval | None = None,
    target_role_interval: TimeInterval | None = None,
    include_source_peer_role: bool = False,
    include_target_peer_roles: bool = False,
    source_peer_role_interval: TimeInterval | None = None,
    target_peer_role_interval: TimeInterval | None = None,
) -> dict[str, object]:
    source_mechanism = source_mechanism_key or (
        "local_resource_or_execution_pressure"
        if relation_type == "received_dependency_failure"
        else "incoming_load_pressure"
    )
    target_mechanism = target_mechanism_key or (
        "received_dependency_failure"
        if relation_type == "received_dependency_failure"
        else "local_resource_or_execution_pressure"
    )
    mechanism_inputs = {
        "incoming_load_pressure": (
            "cpu_usage",
            (
                ("incoming_request_rate_increase", "mechanism:incoming-request-rate/1"),
                ("local_pressure_observed", "mechanism:local-pressure/1"),
            ),
        ),
        "local_resource_or_execution_pressure": (
            "cpu_usage",
            (("local_pressure_observed", "mechanism:local-pressure/1"),),
        ),
        "local_storage_or_network_failure": (
            "network_failure",
            (
                (
                    "local_storage_or_network_failure_observed",
                    "mechanism:local-storage-network/1",
                ),
            ),
        ),
        "received_dependency_failure": (
            "dependency_failure",
            (
                (
                    "outgoing_dependency_failure_observed",
                    "mechanism:outgoing-dependency-failure/1",
                ),
            ),
        ),
        "unclassified_local_anomaly": (
            "cpu_usage",
            (("local_anomaly_observed", "mechanism:local-anomaly/1"),),
        ),
        "workload_unavailability": (
            "workload_lifecycle",
            (
                (
                    "workload_unavailability_observed",
                    "mechanism:workload-unavailability/1",
                ),
            ),
        ),
    }

    def abnormal(
        *,
        suffix: str,
        start: int,
        strength: float,
        association_key: str,
        source_family: str = "cpu_usage",
    ) -> Observation:
        raw = observation(
            suffix=suffix,
            modality="metrics",
            source_family=source_family,
            start=start,
            end=start + 10,
            association_key=association_key,
        )
        return Observation.model_validate(
            {
                **raw.model_dump(mode="json"),
                "evidence_state": "abnormal",
                "anomaly_strength": 0.8,
                "coverage_quality": strength,
            }
        )

    source_family, _ = mechanism_inputs[source_mechanism]
    target_family, _ = mechanism_inputs[target_mechanism]
    source_relation_key = "export:authoritative-source"
    source = abnormal(
        suffix="1",
        start=10,
        strength=source_strength,
        association_key=source_relation_key,
        source_family=source_family,
    )
    source_peer = abnormal(
        suffix="f",
        start=10,
        strength=1.0,
        association_key=source_relation_key,
        source_family=source_family,
    )
    target_observations = tuple(
        abnormal(
            suffix=suffix,
            start=onset_time,
            strength=strength,
            association_key=f"export:authoritative-target-{index}",
            source_family=target_family,
        )
        for index, (_, suffix, onset_time, strength) in enumerate(target_specs)
    )
    target_peers = tuple(
        abnormal(
            suffix=f"{index + 10:x}",
            start=onset_time,
            strength=1.0,
            association_key=f"export:authoritative-target-{index}",
            source_family=target_family,
        )
        for index, (_, _, onset_time, _) in enumerate(target_specs)
    )
    service_keys = tuple(
        sorted(
            {
                "service-a",
                *additional_service_keys,
                *(service_key for service_key, _, _, _ in target_specs),
                *(service_key for edge in edge_specs for service_key in edge[:2]),
            }
        )
    )
    entity_by_service = {
        service_key: f"pod:{index}" for index, service_key in enumerate(service_keys)
    }
    registry = build_temporal_entity_registry(
        opaque_case_id=source.opaque_case_id,
        decision_time_unix_ns=DECISION_TIME,
        metadata_records=tuple(
            metadata_record(
                suffix=f"{index + 1:x}",
                entity=entity_by_service[service_key],
                service=service_key,
            )
            for index, service_key in enumerate(service_keys)
        ),
        observations=(
            entity_input(source, entity=entity_by_service["service-a"]),
            entity_input(source_peer, entity=entity_by_service["service-a"]),
            *(
                entity_input(item, entity=entity_by_service[service_key])
                for (service_key, _, _, _), item in zip(
                    target_specs, target_observations, strict=True
                )
            ),
            *(
                entity_input(item, entity=entity_by_service[service_key])
                for (service_key, _, _, _), item in zip(target_specs, target_peers, strict=True)
            ),
        ),
    )
    grouping_module = importlib.import_module("nurse_rcl.v2.event_grouping")
    source_relation = claim(grouping_module, source, source_peer, key=source_relation_key)
    target_relations = tuple(
        claim(
            grouping_module,
            item,
            peer,
            key=f"export:authoritative-target-{index}",
        )
        for index, (item, peer) in enumerate(zip(target_observations, target_peers, strict=True))
    )
    base_mapping = family_mapping(grouping_module)
    extra_mappings = tuple(
        grouping_module.build_signal_family_mapping_entry(
            adapter_semantics_sha256="f" * 64,
            source_signal_family=source_signal_family,
            primary_signal_family=primary_signal_family,
            provenance_ids=(f"adapter:{source_signal_family}/v1",),
        )
        for source_signal_family, primary_signal_family in (
            ("dependency_failure", "dependency_failure"),
            ("network_failure", "network"),
            ("workload_lifecycle", "lifecycle"),
        )
    )
    grouping = build_event_grouping_result(
        opaque_case_id=source.opaque_case_id,
        decision_time_unix_ns=DECISION_TIME,
        observations=tuple(item.observation for item in registry.resolutions),
        association_claims=(source_relation, *target_relations),
        family_mapping=grouping_module.build_frozen_signal_family_mapping(
            mapping_version="propagation-authority-map/1",
            entries=(*base_mapping.entries, *extra_mappings),
        ),
    )
    scope = TimeInterval(start_unix_ns=0, end_unix_ns=DECISION_TIME)
    universe = build_complete_candidate_universe(
        entity_registry=registry,
        event_grouping=grouping,
        scope_interval=scope,
    )
    frozen = freeze_candidate_universe(universe)
    library = build_mechanism_library()
    group_by_observation = {
        observation_id: group
        for group in grouping.event_groups
        for observation_id in group.observation_ids
    }
    predicates: list[Any] = []
    guards: list[Any] = []
    proposals: list[Any] = []
    role_inputs = []
    if include_source_role:
        role_inputs.append(
            (
                source,
                "service-a",
                "source_support",
                source_relation,
                source_mechanism,
                source_role_interval or source.interval,
            )
        )
    if include_source_peer_role:
        role_inputs.append(
            (
                source_peer,
                "service-a",
                "source_support",
                source_relation,
                source_mechanism,
                source_peer_role_interval or source_peer.interval,
            )
        )
    if include_target_roles:
        role_inputs.extend(
            (
                item,
                service_key,
                "symptom_support",
                relation,
                target_mechanism,
                target_role_interval or item.interval,
            )
            for (service_key, _, _, _), item, relation in zip(
                target_specs,
                target_observations,
                target_relations,
                strict=True,
            )
        )
    if include_target_peer_roles:
        role_inputs.extend(
            (
                item,
                service_key,
                "symptom_support",
                relation,
                target_mechanism,
                target_peer_role_interval or item.interval,
            )
            for (service_key, _, _, _), item, relation in zip(
                target_specs,
                target_peers,
                target_relations,
                strict=True,
            )
        )
    for role_index, (
        item,
        service_key,
        role,
        relation,
        mechanism_key,
        role_interval,
    ) in enumerate(role_inputs):
        mechanism = library.lookup(mechanism_key)
        _, predicate_kinds = mechanism_inputs[mechanism_key]
        role_predicates = tuple(
            evaluate_predicate(
                predicate_kind=predicate_kind,
                requirement="necessary",
                measured_values=(MeasuredValue(name="anomaly_strength", value=0.8, unit="ratio"),),
                threshold=PredicateThreshold(
                    comparator="gt",
                    value=0.5,
                    unit="ratio",
                    source="mechanism_constant",
                ),
                threshold_source=threshold_source,
                observation_ids=(item.observation_id,),
                verification_result_ids=(),
                provenance_ids=(f"predicate:{service_key}:{role_index}:{predicate_index}",),
            )
            for predicate_index, (predicate_kind, threshold_source) in enumerate(predicate_kinds)
        )
        event_group = group_by_observation[item.observation_id]
        hypothesis = build_diagnostic_hypothesis(
            opaque_case_id=universe.opaque_case_id,
            candidate_service_key=service_key,
            mechanism_key=mechanism.mechanism_key,
            mechanism_definition_sha256=mechanism.definition_sha256,
            origin="semantic_agent",
            origin_version="propagation-test/1",
            observation_ids=(item.observation_id,),
            relation_ids=(relation.claim_id,),
            event_group_ids=(event_group.event_group_id,),
            time_intervals=(role_interval,),
            necessary_predicate_result_ids=tuple(
                predicate.predicate_result_id for predicate in role_predicates
            ),
            optional_predicate_result_ids=(),
            refuting_predicate_result_ids=(),
            alternative_hypothesis_ids=(),
            verification_request_ids=(),
            unknown_reasons=(),
            provenance_ids=(f"semantic:{service_key}",),
            lifecycle_status="proposed",
        )
        guard = execute_hypothesis_guard(
            submitted_record=hypothesis.model_dump(mode="json"),
            predicates=role_predicates,
            candidate_universe=universe,
            frozen_candidates=frozen,
            entity_registry=registry,
            event_grouping=grouping,
            mechanism_library=library,
            scope_interval=scope,
        )
        assert guard.final_disposition == "accepted", guard.model_dump(mode="json")
        proposal = build_role_binding_proposal(
            guard_result_id=guard.guard_result_id,
            hypothesis_id=hypothesis.hypothesis_id,
            event_group_id=event_group.event_group_id,
            relation_id=relation.claim_id,
            interval=role_interval,
            role=role,
            provenance_ids=(f"role:{service_key}",),
        )
        predicates.extend(role_predicates)
        guards.append(guard)
        proposals.append(proposal)
    role_compilation = compile_event_roles(
        proposals=tuple(proposals),
        guard_results=tuple(guards),
        predicates=tuple(predicates),
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        event_grouping=grouping,
        mechanism_library=library,
        scope_interval=scope,
    )
    contract = _time_contract()
    source_onset = build_onset_interval(
        time_contract=contract,
        anomaly_id="authority-source",
        service_key="service-a",
        t_low_unix_ns=19,
        t_high_unix_ns=19,
        provenance_ids=(source.observation_id,),
    )
    target_onsets = tuple(
        build_onset_interval(
            time_contract=contract,
            anomaly_id=f"authority-target-{index}",
            service_key=service_key,
            t_low_unix_ns=onset_time,
            t_high_unix_ns=onset_time,
            provenance_ids=(item.observation_id,),
        )
        for index, ((service_key, _, onset_time, _), item) in enumerate(
            zip(target_specs, target_observations, strict=True)
        )
    )
    resolution = freeze_clock_resolution(
        time_contract=contract,
        expected_time_contract_sha256=case_time_contract_sha256(contract),
        resolution_status="resolved",
        provenance_ids=("clock:resolved",),
    )
    precedences = tuple(
        evaluate_clock_aware_precedence(
            source_onset=source_onset,
            target_onset=target_onset,
            time_contract=contract,
            clock_resolution=resolution,
            expected_clock_resolution_sha256=resolution.resolution_sha256,
        )
        for target_onset in target_onsets
    )
    edges = tuple(
        _edge(
            coverage_fraction=coverage,
            coverage_state="sufficient" if coverage >= 0.5 else "insufficient",
            caller_service_key=(
                target if relation_type == "received_dependency_failure" else source
            ),
            callee_service_key=(
                source if relation_type == "received_dependency_failure" else target
            ),
            tag=f"{source}-{target}-{index}",
            relation_type=relation_type,
        )
        for index, (source, target, coverage) in enumerate(edge_specs)
    )
    return {
        "frozen_candidates": frozen,
        "condensations": (_condensation(edges=edges, relation_type=relation_type),),
        "event_grouping": grouping,
        "onsets": (source_onset, *target_onsets),
        "precedences": precedences,
        "role_compilation": role_compilation,
        "role_proposals": tuple(proposals),
        "role_guard_results": tuple(guards),
        "role_predicates": tuple(predicates),
        "candidate_universe": universe,
        "entity_registry": registry,
        "mechanism_library": library,
        "scope_interval": scope,
    }


def _role_compilation_replays(inputs: dict[str, object]) -> bool:
    return inputs["role_compilation"].verify_inputs(
        proposals=inputs["role_proposals"],
        guard_results=inputs["role_guard_results"],
        predicates=inputs["role_predicates"],
        candidate_universe=inputs["candidate_universe"],
        frozen_candidates=inputs["frozen_candidates"],
        entity_registry=inputs["entity_registry"],
        event_grouping=inputs["event_grouping"],
        mechanism_library=inputs["mechanism_library"],
        scope_interval=inputs["scope_interval"],
    )


def test_scoring_requires_and_replays_authoritative_role_compilation() -> None:
    inputs = _authoritative_scenario()
    result = propagation_module().score_propagation_paths(**inputs)

    source_score = next(
        item for item in result.candidate_scores if item.candidate_service_key == "service-a"
    )
    assert source_score.propagation_score == pytest.approx(0.45)
    assert result.role_compilation_result_sha256 == inputs["role_compilation"].result_sha256
    assert result.relation_bindings
    assert {item.relation_type for item in result.relation_bindings} == {
        "received_dependency_failure"
    }


def test_authoritative_source_role_must_cover_the_closed_source_onset() -> None:
    inputs = _scenario(
        edge_specs=(("service-a", "service-b", 0.75),),
        target_specs=(("service-b", "2", 30, 0.6),),
        source_role_interval=TimeInterval(start_unix_ns=10, end_unix_ns=19),
    )
    assert _role_compilation_replays(inputs) is True
    source_assertion = next(
        item for item in inputs["role_compilation"].assertions if item.role == "source_support"
    )
    source_onset = next(item for item in inputs["onsets"] if item.service_key == "service-a")
    assert source_assertion.interval.end_unix_ns == source_onset.t_high_unix_ns

    result = propagation_module().score_propagation_paths(**inputs)

    path = result.path_audits[0]
    assert path.raw_contribution == 0.0
    assert "source_role_interval_mismatch" in path.reason_codes


def test_authoritative_target_role_must_cover_the_closed_target_onset() -> None:
    inputs = _scenario(
        edge_specs=(("service-a", "service-b", 0.75),),
        target_specs=(("service-b", "2", 30, 0.6),),
        target_role_interval=TimeInterval(start_unix_ns=31, end_unix_ns=40),
    )
    assert _role_compilation_replays(inputs) is True
    target_assertion = next(
        item for item in inputs["role_compilation"].assertions if item.role == "symptom_support"
    )
    target_onset = next(item for item in inputs["onsets"] if item.service_key == "service-b")
    assert target_assertion.interval.start_unix_ns > target_onset.t_low_unix_ns

    result = propagation_module().score_propagation_paths(**inputs)

    path = result.path_audits[0]
    assert path.raw_contribution == 0.0
    assert "target_role_interval_mismatch" in path.reason_codes


def test_source_contributor_maximum_is_selected_only_from_onset_covering_assertions() -> None:
    inputs = _scenario(
        edge_specs=(("service-a", "service-b", 0.75),),
        target_specs=(("service-b", "2", 30, 0.9),),
        source_strength=0.6,
        include_source_peer_role=True,
        source_peer_role_interval=TimeInterval(start_unix_ns=10, end_unix_ns=19),
    )
    assert _role_compilation_replays(inputs) is True
    source_assertions = tuple(
        item for item in inputs["role_compilation"].assertions if item.role == "source_support"
    )
    assert sorted(item.strength for item in source_assertions) == [0.6, 1.0]
    eligible = next(item for item in source_assertions if item.interval.end_unix_ns == 20)

    result = propagation_module().score_propagation_paths(**inputs)

    path = result.path_audits[0]
    assert path.source_assertion_ids == tuple(
        sorted(item.assertion_id for item in source_assertions)
    )
    assert path.source_contributing_assertion_id == eligible.assertion_id
    assert path.source_support == 0.6
    assert path.raw_contribution == pytest.approx(0.45)


def test_target_contributor_maximum_is_selected_only_from_onset_covering_assertions() -> None:
    inputs = _scenario(
        edge_specs=(("service-a", "service-b", 0.75),),
        target_specs=(("service-b", "2", 30, 0.6),),
        source_strength=0.9,
        include_target_peer_roles=True,
        target_peer_role_interval=TimeInterval(start_unix_ns=31, end_unix_ns=40),
    )
    assert _role_compilation_replays(inputs) is True
    target_assertions = tuple(
        item for item in inputs["role_compilation"].assertions if item.role == "symptom_support"
    )
    assert sorted(item.strength for item in target_assertions) == [0.6, 1.0]
    eligible = next(item for item in target_assertions if item.interval.start_unix_ns == 30)

    result = propagation_module().score_propagation_paths(**inputs)

    path = result.path_audits[0]
    assert path.target_assertion_ids == tuple(
        sorted(item.assertion_id for item in target_assertions)
    )
    assert path.target_contributing_assertion_id == eligible.assertion_id
    assert path.target_symptom_strength == 0.6
    assert path.raw_contribution == pytest.approx(0.45)


def test_bare_self_signed_accepted_role_assertion_is_rejected() -> None:
    inputs = _authoritative_scenario()
    compiled = inputs["role_compilation"].assertions[0]
    self_signed = build_role_assertion(
        hypothesis_id=compiled.hypothesis_id,
        candidate_service_key=compiled.candidate_service_key,
        event_group_id=compiled.event_group_id,
        relation_id=compiled.relation_id,
        interval=compiled.interval,
        role=compiled.role,
        strength=0.7,
        score_direction=compiled.score_direction,
        predicate_result_ids=compiled.predicate_result_ids,
        lifecycle_status="accepted",
        reason=compiled.reason,
        provenance_ids=("caller:self-signed",),
    )

    with pytest.raises(ValueError, match="propagation scoring input verification failed"):
        propagation_module().score_propagation_paths(
            frozen_candidates=inputs["frozen_candidates"],
            condensations=inputs["condensations"],
            event_grouping=inputs["event_grouping"],
            onsets=inputs["onsets"],
            precedences=inputs["precedences"],
            role_assertions=(self_signed,),
        )


def test_consistently_resigned_role_compilation_with_wrong_interval_is_rejected() -> None:
    inputs = _authoritative_scenario()
    compilation = inputs["role_compilation"]
    original = compilation.assertions[0]
    forged_assertion = _resign_model(
        original,
        prefix="rol",
        id_field="assertion_id",
        sha_field="assertion_sha256",
        updates={"interval": TimeInterval(start_unix_ns=0, end_unix_ns=5).model_dump(mode="json")},
    )
    forged_assertions = tuple(
        sorted(
            (forged_assertion, *compilation.assertions[1:]),
            key=lambda item: item.assertion_id,
        )
    )
    inputs["role_compilation"] = _resign_model(
        compilation,
        prefix="rcr",
        id_field="role_compilation_id",
        sha_field="result_sha256",
        updates={"assertions": tuple(item.model_dump(mode="json") for item in forged_assertions)},
    )

    with pytest.raises(ValueError, match="propagation scoring input verification failed"):
        propagation_module().score_propagation_paths(**inputs)


def test_missing_role_authority_replay_dependency_is_rejected() -> None:
    inputs = _authoritative_scenario()
    del inputs["role_guard_results"]

    with pytest.raises(ValueError, match="propagation scoring input verification failed"):
        propagation_module().score_propagation_paths(**inputs)


def test_one_hop_formula_uses_coverage_and_minimum_role_strength() -> None:
    result = propagation_module().score_propagation_paths(**_inputs())

    source = next(
        item for item in result.candidate_scores if item.candidate_service_key == "service-a"
    )
    selected = next(item for item in result.path_audits if item.selection_status == "selected")
    assert selected.path_length == 1
    assert selected.path_quality == 0.75
    assert selected.source_support == 0.8
    assert selected.target_symptom_strength == 0.6
    assert selected.raw_contribution == pytest.approx(0.45)
    assert source.propagation_score == pytest.approx(0.45)


def test_insufficient_coverage_retains_zero_credit_reason() -> None:
    result = propagation_module().score_propagation_paths(**_inputs(coverage_fraction=0.25))

    path = result.path_audits[0]
    assert path.raw_contribution == 0.0
    assert "insufficient_path_coverage" in path.reason_codes
    assert all(item.propagation_score == 0.0 for item in result.candidate_scores)


def test_unknown_precedence_retains_zero_credit_reason() -> None:
    result = propagation_module().score_propagation_paths(**_inputs(overlap=True))

    path = result.path_audits[0]
    assert path.raw_contribution == 0.0
    assert "temporal_equal_or_overlapping" in path.reason_codes
    assert all(item.propagation_score == 0.0 for item in result.candidate_scores)


def test_two_distinct_targets_use_reference_f_gamma() -> None:
    result = propagation_module().score_propagation_paths(
        **_scenario(
            edge_specs=(
                ("service-a", "service-b", 0.75),
                ("service-a", "service-c", 0.5),
            ),
            target_specs=(
                ("service-b", "2", 30, 0.6),
                ("service-c", "3", 40, 0.4),
            ),
        )
    )

    source_score = next(
        item for item in result.candidate_scores if item.candidate_service_key == "service-a"
    )
    assert source_score.top_target_contributions == pytest.approx((0.45, 0.2))
    assert source_score.propagation_score == pytest.approx(0.45 + 0.25 * (1.0 - 0.45) * 0.2)


def test_two_hop_path_is_divided_by_path_length() -> None:
    result = propagation_module().score_propagation_paths(
        **_scenario(
            edge_specs=(
                ("service-a", "service-x", 0.75),
                ("service-x", "service-b", 0.5),
            ),
            target_specs=(("service-b", "2", 30, 0.6),),
        )
    )

    selected = next(item for item in result.path_audits if item.selection_status == "selected")
    source_score = next(
        item for item in result.candidate_scores if item.candidate_service_key == "service-a"
    )
    assert selected.path_length == 2
    assert selected.path_quality == pytest.approx(0.5)
    assert selected.raw_contribution == pytest.approx(0.5 * min(0.8, 0.6) / 2)
    assert source_score.propagation_score == pytest.approx(0.15)


def test_path_over_hop_limit_is_retained_without_credit() -> None:
    module = propagation_module()
    inputs = _scenario(
        edge_specs=(
            ("service-a", "service-x", 0.75),
            ("service-x", "service-b", 0.5),
        ),
        target_specs=(("service-b", "2", 30, 0.6),),
    )
    inputs["config"] = module.build_propagation_scoring_config(propagation_max_hops=1)

    result = module.score_propagation_paths(**inputs)

    audit = next(item for item in result.path_audits if item.path_length == 2)
    source_score = next(
        item for item in result.candidate_scores if item.candidate_service_key == "service-a"
    )
    assert audit.selection_status == "rejected"
    assert audit.raw_contribution == 0.0
    assert "hop_limit_exceeded" in audit.reason_codes
    assert source_score.propagation_score == 0.0


def test_hop_limit_defaults_and_predeclared_values_are_enforced() -> None:
    module = propagation_module()

    assert module.build_propagation_scoring_config().propagation_max_hops == 2
    assert tuple(
        module.build_propagation_scoring_config(propagation_max_hops=value).propagation_max_hops
        for value in (1, 2, 3, 5)
    ) == (1, 2, 3, 5)
    for forbidden in (0, 4, 6, True, 2.0):
        with pytest.raises(ValueError, match="predeclared Dev value"):
            module.build_propagation_scoring_config(propagation_max_hops=forbidden)


def test_missing_onsets_keep_the_legal_path_and_zero_credit_reasons() -> None:
    module = propagation_module()
    inputs = _scenario(
        edge_specs=(("service-a", "service-b", 0.75),),
        target_specs=(("service-b", "2", 30, 0.6),),
    )
    inputs["onsets"] = ()
    inputs["precedences"] = ()

    result = module.score_propagation_paths(**inputs)

    source_score = next(
        item for item in result.candidate_scores if item.candidate_service_key == "service-a"
    )
    assert source_score.propagation_score == 0.0
    assert source_score.reason_code == "no_credited_path"
    assert len(result.path_audits) == 1
    audit = result.path_audits[0]
    assert audit.source_onset_id is None
    assert audit.target_onset_id is None
    assert "source_onset_binding_missing" in audit.reason_codes
    assert "target_onset_binding_missing" in audit.reason_codes
    assert "missing_temporal_precedence" in audit.reason_codes


def test_missing_graph_degrades_to_zero_for_every_frozen_candidate() -> None:
    module = propagation_module()
    inputs = _scenario(
        edge_specs=(("service-a", "service-b", 0.75),),
        target_specs=(("service-b", "2", 30, 0.6),),
    )
    frozen = inputs["frozen_candidates"]
    assert hasattr(frozen, "candidate_service_keys")
    inputs["condensations"] = ()

    result = module.score_propagation_paths(**inputs)

    assert tuple(item.candidate_service_key for item in result.candidate_scores) == (
        frozen.candidate_service_keys
    )
    assert not result.path_audits
    assert not result.target_scores or all(
        item.contribution == 0.0 for item in result.target_scores
    )
    assert all(item.propagation_score == 0.0 for item in result.candidate_scores)
    source_score = next(
        item for item in result.candidate_scores if item.candidate_service_key == "service-a"
    )
    assert source_score.reason_code == "no_legal_typed_path"


def test_same_target_uses_maximum_path_and_retains_unselected_path() -> None:
    result = propagation_module().score_propagation_paths(
        **_scenario(
            edge_specs=(
                ("service-a", "service-b", 0.5),
                ("service-a", "service-x", 0.75),
                ("service-x", "service-b", 0.75),
            ),
            target_specs=(("service-b", "2", 30, 0.6),),
        )
    )

    source_paths = tuple(
        item for item in result.path_audits if item.candidate_service_key == "service-a"
    )
    assert len(source_paths) == 2
    assert {item.selection_status for item in source_paths} == {
        "selected",
        "same_target_not_selected",
    }
    selected = next(item for item in source_paths if item.selection_status == "selected")
    not_selected = next(
        item for item in source_paths if item.selection_status == "same_target_not_selected"
    )
    assert selected.path_length == 1
    assert selected.raw_contribution == pytest.approx(0.3)
    assert not_selected.path_length == 2
    assert not_selected.raw_contribution == pytest.approx(0.225)
    target = next(
        item
        for item in result.target_scores
        if item.candidate_service_key == "service-a"
        and item.target_event_group_id == selected.target_event_group_id
    )
    assert len(target.retained_path_audit_ids) == 2
    assert target.selected_path_audit_id == selected.path_audit_id
    assert target.contribution == pytest.approx(0.3)


def test_fan_out_retains_all_targets_but_aggregates_only_top_two() -> None:
    result = propagation_module().score_propagation_paths(
        **_scenario(
            edge_specs=(
                ("service-a", "service-b", 0.75),
                ("service-a", "service-c", 0.5),
                ("service-a", "service-d", 0.75),
            ),
            target_specs=(
                ("service-b", "2", 30, 0.6),
                ("service-c", "3", 40, 0.4),
                ("service-d", "4", 50, 0.2),
            ),
        )
    )

    source_targets = tuple(
        item for item in result.target_scores if item.candidate_service_key == "service-a"
    )
    source_score = next(
        item for item in result.candidate_scores if item.candidate_service_key == "service-a"
    )
    assert sorted(item.contribution for item in source_targets) == pytest.approx([0.15, 0.2, 0.45])
    assert source_score.retained_target_score_ids == tuple(
        sorted(item.target_score_id for item in source_targets)
    )
    assert source_score.top_target_contributions == pytest.approx((0.45, 0.2))
    assert source_score.propagation_score == pytest.approx(0.4775)


def test_same_scc_records_cycle_without_inventing_internal_source_order() -> None:
    result = propagation_module().score_propagation_paths(
        **_scenario(
            edge_specs=(
                ("service-a", "service-b", 0.75),
                ("service-b", "service-a", 0.75),
            ),
            target_specs=(("service-b", "2", 30, 0.6),),
        )
    )

    audit = result.path_audits[0]
    source_score = next(
        item for item in result.candidate_scores if item.candidate_service_key == "service-a"
    )
    assert audit.path_length == 0
    assert audit.component_ids == (audit.component_ids[0],)
    assert "same_scc_no_internal_order" in audit.reason_codes
    assert audit.raw_contribution == 0.0
    assert source_score.propagation_score == 0.0


def test_edges_from_different_relation_condensations_do_not_form_a_path() -> None:
    inputs = _scenario(
        edge_specs=(("service-a", "service-x", 0.75),),
        target_specs=(("service-b", "2", 30, 0.6),),
    )
    load_edge = _edge(
        coverage_fraction=0.75,
        coverage_state="sufficient",
        caller_service_key="service-a",
        callee_service_key="service-x",
        tag="load-a-x",
        relation_type="caller_load_transmission",
    )
    dependency_edge = _edge(
        coverage_fraction=0.75,
        coverage_state="sufficient",
        caller_service_key="service-b",
        callee_service_key="service-x",
        tag="dependency-x-b",
        relation_type="received_dependency_failure",
    )
    inputs["condensations"] = (
        _condensation(edges=(load_edge,), relation_type="caller_load_transmission"),
        _condensation(
            edges=(dependency_edge,),
            relation_type="received_dependency_failure",
        ),
    )

    result = propagation_module().score_propagation_paths(**inputs)

    assert not result.path_audits
    source_score = next(
        item for item in result.candidate_scores if item.candidate_service_key == "service-a"
    )
    assert source_score.propagation_score == 0.0
    assert source_score.reason_code == "no_legal_typed_path"


def test_relation_type_mismatch_between_compiled_roles_and_graph_gets_no_credit() -> None:
    inputs = _scenario(
        edge_specs=(("service-a", "service-b", 0.75),),
        target_specs=(("service-b", "2", 30, 0.6),),
    )
    load_edge = _edge(
        coverage_fraction=0.75,
        coverage_state="sufficient",
        caller_service_key="service-a",
        callee_service_key="service-b",
        tag="mismatched-load-edge",
        relation_type="caller_load_transmission",
    )
    inputs["condensations"] = (
        _condensation(edges=(load_edge,), relation_type="caller_load_transmission"),
    )

    result = propagation_module().score_propagation_paths(**inputs)

    assert {item.relation_type for item in result.relation_bindings} == {
        "received_dependency_failure"
    }
    assert not result.path_audits
    source = next(
        item for item in result.candidate_scores if item.candidate_service_key == "service-a"
    )
    assert source.propagation_score == 0.0
    assert source.reason_code == "no_legal_typed_path"


def test_undeclared_mechanism_role_pair_produces_no_relation_binding_or_credit() -> None:
    inputs = _scenario(
        edge_specs=(("service-a", "service-b", 0.75),),
        target_specs=(("service-b", "2", 30, 0.6),),
        target_mechanism_key="unclassified_local_anomaly",
    )

    result = propagation_module().score_propagation_paths(**inputs)

    assert all(item.role != "symptom_support" for item in result.relation_bindings)
    assert not result.path_audits
    source = next(
        item for item in result.candidate_scores if item.candidate_service_key == "service-a"
    )
    assert source.propagation_score == 0.0
    assert source.reason_code == "missing_target_symptom_support"


@pytest.mark.parametrize(
    "source_mechanism_key",
    (
        "local_resource_or_execution_pressure",
        "local_storage_or_network_failure",
        "workload_unavailability",
    ),
)
def test_declared_received_failure_mechanism_pairs_are_compatible(
    source_mechanism_key: str,
) -> None:
    inputs = _scenario(
        edge_specs=(("service-a", "service-b", 0.75),),
        target_specs=(("service-b", "2", 30, 0.6),),
        source_mechanism_key=source_mechanism_key,
    )

    result = propagation_module().score_propagation_paths(**inputs)

    assert {item.mechanism_key for item in result.relation_bindings} == {
        source_mechanism_key,
        "received_dependency_failure",
    }
    source = next(
        item for item in result.candidate_scores if item.candidate_service_key == "service-a"
    )
    assert source.propagation_score == pytest.approx(0.45)


def test_missing_verified_roles_have_distinct_candidate_zero_reasons() -> None:
    module = propagation_module()
    without_source = _scenario(
        edge_specs=(("service-a", "service-b", 0.75),),
        target_specs=(("service-b", "2", 30, 0.6),),
        include_source_role=False,
    )
    source_result = module.score_propagation_paths(**without_source)
    source_score = next(
        item for item in source_result.candidate_scores if item.candidate_service_key == "service-a"
    )
    assert source_score.reason_code == "missing_source_support"

    without_target = _scenario(
        edge_specs=(("service-a", "service-b", 0.75),),
        target_specs=(("service-b", "2", 30, 0.6),),
        include_target_roles=False,
    )
    target_result = module.score_propagation_paths(**without_target)
    target_score = next(
        item for item in target_result.candidate_scores if item.candidate_service_key == "service-a"
    )
    assert target_score.reason_code == "missing_target_symptom_support"


def test_missing_precedence_keeps_path_with_explicit_zero_reason() -> None:
    inputs = _scenario(
        edge_specs=(("service-a", "service-b", 0.75),),
        target_specs=(("service-b", "2", 30, 0.6),),
    )
    inputs["precedences"] = ()

    result = propagation_module().score_propagation_paths(**inputs)

    assert len(result.path_audits) == 1
    audit = result.path_audits[0]
    assert audit.raw_contribution == 0.0
    assert audit.source_onset_id is not None
    assert audit.target_onset_id is not None
    assert "missing_temporal_precedence" in audit.reason_codes


def test_every_frozen_candidate_receives_a_propagation_score() -> None:
    inputs = _scenario(
        edge_specs=(
            ("service-a", "service-x", 0.75),
            ("service-x", "service-b", 0.5),
        ),
        target_specs=(("service-b", "2", 30, 0.6),),
    )
    frozen = inputs["frozen_candidates"]
    assert hasattr(frozen, "candidate_service_keys")

    result = propagation_module().score_propagation_paths(**inputs)

    assert tuple(item.candidate_service_key for item in result.candidate_scores) == (
        frozen.candidate_service_keys
    )
    intermediate = next(
        item for item in result.candidate_scores if item.candidate_service_key == "service-x"
    )
    assert intermediate.propagation_score == 0.0
    assert intermediate.reason_code == "missing_source_support"


@pytest.mark.parametrize(
    "field",
    (
        "onsets",
        "precedences",
        "role_proposals",
        "role_guard_results",
        "role_predicates",
    ),
)
def test_duplicate_replayed_artifact_is_rejected_at_the_input_boundary(field: str) -> None:
    inputs = _scenario(
        edge_specs=(("service-a", "service-b", 0.75),),
        target_specs=(("service-b", "2", 30, 0.6),),
    )
    values = inputs[field]
    assert isinstance(values, tuple)
    inputs[field] = (values[0], *values)

    with pytest.raises(ValueError, match="propagation scoring input verification failed"):
        propagation_module().score_propagation_paths(**inputs)


def test_input_order_has_exactly_deterministic_replay_and_content_hash() -> None:
    module = propagation_module()
    inputs = _scenario(
        edge_specs=(
            ("service-a", "service-b", 0.75),
            ("service-a", "service-c", 0.5),
        ),
        target_specs=(
            ("service-b", "2", 30, 0.6),
            ("service-c", "3", 40, 0.4),
        ),
    )

    result = module.score_propagation_paths(**inputs)
    reordered = {
        **inputs,
        "condensations": tuple(reversed(inputs["condensations"])),
        "onsets": tuple(reversed(inputs["onsets"])),
        "precedences": tuple(reversed(inputs["precedences"])),
        "role_proposals": tuple(reversed(inputs["role_proposals"])),
        "role_guard_results": tuple(reversed(inputs["role_guard_results"])),
        "role_predicates": tuple(reversed(inputs["role_predicates"])),
    }
    replayed = module.score_propagation_paths(**reordered)

    assert result == replayed
    assert result.result_sha256 == replayed.result_sha256
    assert result.verify()
    assert result.verify_inputs(
        expected_result_sha256=result.result_sha256,
        expected_authoritative_input_sha256=result.authoritative_input_sha256,
        **reordered,
    )
    tampered = result.model_copy(update={"result_sha256": "0" * 64})
    assert not tampered.verify()
    assert not tampered.verify_inputs(
        expected_result_sha256=result.result_sha256,
        expected_authoritative_input_sha256=result.authoritative_input_sha256,
        **inputs,
    )


def test_conflicting_precedence_for_one_onset_pair_is_rejected() -> None:
    inputs = _scenario(
        edge_specs=(("service-a", "service-b", 0.75),),
        target_specs=(("service-b", "2", 30, 0.6),),
    )
    onsets = inputs["onsets"]
    precedences = inputs["precedences"]
    assert isinstance(onsets, tuple)
    assert isinstance(precedences, tuple)
    contract = _time_contract()
    unresolved = freeze_clock_resolution(
        time_contract=contract,
        expected_time_contract_sha256=case_time_contract_sha256(contract),
        resolution_status="unresolved",
        provenance_ids=("clock:unresolved",),
    )
    conflicting = evaluate_clock_aware_precedence(
        source_onset=onsets[0],
        target_onset=onsets[1],
        time_contract=contract,
        clock_resolution=unresolved,
        expected_clock_resolution_sha256=unresolved.resolution_sha256,
    )
    inputs["precedences"] = (*precedences, conflicting)

    with pytest.raises(ValueError, match="propagation scoring input verification failed"):
        propagation_module().score_propagation_paths(**inputs)


@pytest.mark.parametrize(
    ("source_strength", "target_strength", "expected_reason"),
    (
        (0.0, 0.6, "missing_source_support"),
        (0.8, 0.0, "missing_target_symptom_support"),
    ),
)
def test_zero_strength_role_is_not_treated_as_verified_positive_support(
    source_strength: float,
    target_strength: float,
    expected_reason: str,
) -> None:
    result = propagation_module().score_propagation_paths(
        **_scenario(
            edge_specs=(("service-a", "service-b", 0.75),),
            target_specs=(("service-b", "2", 30, target_strength),),
            source_strength=source_strength,
        )
    )

    source_score = next(
        item for item in result.candidate_scores if item.candidate_service_key == "service-a"
    )
    assert source_score.propagation_score == 0.0
    assert source_score.reason_code == expected_reason
    assert not result.path_audits


def test_resigned_cross_layer_score_forgery_fails_verification() -> None:
    result = propagation_module().score_propagation_paths(**_inputs())
    original_path = next(
        item for item in result.path_audits if item.candidate_service_key == "service-a"
    )
    forged_path = _resign_model(
        original_path,
        prefix="ppath",
        id_field="path_audit_id",
        sha_field="path_audit_sha256",
        updates={"raw_contribution": 0.9},
    )
    original_target = next(
        item
        for item in result.target_scores
        if item.selected_path_audit_id == original_path.path_audit_id
    )
    forged_target = _resign_model(
        original_target,
        prefix="ptgt",
        id_field="target_score_id",
        sha_field="target_score_sha256",
        updates={
            "retained_path_audit_ids": (forged_path.path_audit_id,),
            "selected_path_audit_id": forged_path.path_audit_id,
            "contribution": 0.9,
        },
    )
    original_candidate = next(
        item for item in result.candidate_scores if item.candidate_service_key == "service-a"
    )
    forged_candidate = _resign_model(
        original_candidate,
        prefix="pcand",
        id_field="candidate_score_id",
        sha_field="candidate_score_sha256",
        updates={
            "retained_target_score_ids": (forged_target.target_score_id,),
            "top_target_contributions": (0.9,),
            "propagation_score": 0.9,
        },
    )
    path_audits = tuple(
        forged_path if item == original_path else item for item in result.path_audits
    )
    target_scores = tuple(
        forged_target if item == original_target else item for item in result.target_scores
    )
    candidate_scores = tuple(
        forged_candidate if item == original_candidate else item for item in result.candidate_scores
    )
    forged_result = _resign_model(
        result,
        prefix="pres",
        id_field="scoring_result_id",
        sha_field="result_sha256",
        updates={
            "path_audits": tuple(item.model_dump(mode="json") for item in path_audits),
            "target_scores": tuple(item.model_dump(mode="json") for item in target_scores),
            "candidate_scores": tuple(item.model_dump(mode="json") for item in candidate_scores),
        },
    )

    assert not forged_path.verify()
    assert not forged_result.verify()


def test_resigned_lower_same_target_path_cannot_replace_the_maximum() -> None:
    result = propagation_module().score_propagation_paths(
        **_scenario(
            edge_specs=(
                ("service-a", "service-b", 0.5),
                ("service-a", "service-x", 0.75),
                ("service-x", "service-b", 0.75),
            ),
            target_specs=(("service-b", "2", 30, 0.6),),
        )
    )
    original_selected = next(
        item for item in result.path_audits if item.selection_status == "selected"
    )
    original_lower = next(
        item for item in result.path_audits if item.selection_status == "same_target_not_selected"
    )
    forged_higher = _resign_model(
        original_selected,
        prefix="ppath",
        id_field="path_audit_id",
        sha_field="path_audit_sha256",
        updates={"selection_status": "same_target_not_selected"},
    )
    forged_lower = _resign_model(
        original_lower,
        prefix="ppath",
        id_field="path_audit_id",
        sha_field="path_audit_sha256",
        updates={"selection_status": "selected"},
    )
    original_target = next(
        item
        for item in result.target_scores
        if item.selected_path_audit_id == original_selected.path_audit_id
    )
    forged_target = _resign_model(
        original_target,
        prefix="ptgt",
        id_field="target_score_id",
        sha_field="target_score_sha256",
        updates={
            "retained_path_audit_ids": tuple(
                sorted((forged_higher.path_audit_id, forged_lower.path_audit_id))
            ),
            "selected_path_audit_id": forged_lower.path_audit_id,
            "contribution": forged_lower.raw_contribution,
        },
    )
    original_candidate = next(
        item for item in result.candidate_scores if item.candidate_service_key == "service-a"
    )
    forged_candidate = _resign_model(
        original_candidate,
        prefix="pcand",
        id_field="candidate_score_id",
        sha_field="candidate_score_sha256",
        updates={
            "retained_target_score_ids": (forged_target.target_score_id,),
            "top_target_contributions": (forged_lower.raw_contribution,),
            "propagation_score": forged_lower.raw_contribution,
        },
    )
    forged_result = _resign_model(
        result,
        prefix="pres",
        id_field="scoring_result_id",
        sha_field="result_sha256",
        updates={
            "path_audits": tuple(
                item.model_dump(mode="json")
                for item in sorted(
                    (forged_higher, forged_lower),
                    key=lambda item: item.path_audit_id,
                )
            ),
            "target_scores": tuple(
                item.model_dump(mode="json")
                for item in (
                    forged_target if target == original_target else target
                    for target in result.target_scores
                )
            ),
            "candidate_scores": tuple(
                item.model_dump(mode="json")
                for item in (
                    forged_candidate if candidate == original_candidate else candidate
                    for candidate in result.candidate_scores
                )
            ),
        },
    )

    assert forged_higher.verify()
    assert forged_lower.verify()
    assert not forged_result.verify()


@pytest.mark.parametrize("hop_limit", (3, 5))
def test_predeclared_longer_hop_limits_are_applied_to_path_traversal(hop_limit: int) -> None:
    module = propagation_module()
    nodes = ("service-a", *(f"service-x{index}" for index in range(1, hop_limit)), "service-b")
    inputs = _scenario(
        edge_specs=tuple((source, target, 0.75) for source, target in pairwise(nodes)),
        target_specs=(("service-b", "2", 30, 0.6),),
    )
    inputs["config"] = module.build_propagation_scoring_config(propagation_max_hops=hop_limit)

    result = module.score_propagation_paths(**inputs)

    selected = next(item for item in result.path_audits if item.selection_status == "selected")
    assert selected.path_length == hop_limit
    assert selected.raw_contribution == pytest.approx(0.75 * 0.6 / hop_limit)


def test_one_insufficient_hop_zeroes_the_whole_multihop_path() -> None:
    result = propagation_module().score_propagation_paths(
        **_scenario(
            edge_specs=(
                ("service-a", "service-x", 0.75),
                ("service-x", "service-b", 0.25),
            ),
            target_specs=(("service-b", "2", 30, 0.6),),
        )
    )

    assert len(result.path_audits) == 1
    audit = result.path_audits[0]
    assert tuple(item.coverage_status for item in audit.hops) == (
        "sufficient",
        "insufficient",
    )
    assert audit.path_quality == 0.0
    assert audit.limiting_hop_indices == (2,)
    assert "insufficient_path_coverage" in audit.reason_codes
    assert audit.raw_contribution == 0.0


def test_reachable_path_two_hops_over_default_limit_is_retained_for_audit() -> None:
    nodes = ("service-a", "service-x1", "service-x2", "service-x3", "service-b")
    result = propagation_module().score_propagation_paths(
        **_scenario(
            edge_specs=tuple((source, target, 0.75) for source, target in pairwise(nodes)),
            target_specs=(("service-b", "2", 30, 0.6),),
        )
    )

    source_path = next(
        item for item in result.path_audits if item.candidate_service_key == "service-a"
    )
    source_score = next(
        item for item in result.candidate_scores if item.candidate_service_key == "service-a"
    )
    assert source_path.path_length == 4
    assert source_path.reason_codes == ("hop_limit_exceeded",)
    assert source_path.raw_contribution == 0.0
    assert source_score.reason_code == "no_credited_path"


def test_reachable_path_beyond_the_old_six_hop_audit_boundary_is_retained() -> None:
    nodes = (
        "service-a",
        "service-x1",
        "service-x2",
        "service-x3",
        "service-x4",
        "service-x5",
        "service-x6",
        "service-b",
    )
    result = propagation_module().score_propagation_paths(
        **_scenario(
            edge_specs=tuple((source, target, 0.75) for source, target in pairwise(nodes)),
            target_specs=(("service-b", "2", 30, 0.6),),
        )
    )

    assert len(result.path_audits) == 1
    audit = result.path_audits[0]
    assert audit.path_length == 7
    assert audit.raw_contribution == 0.0
    assert "hop_limit_exceeded" in audit.reason_codes


def test_multiple_events_on_one_target_service_use_one_service_maximum() -> None:
    result = propagation_module().score_propagation_paths(
        **_scenario(
            edge_specs=(("service-a", "service-b", 0.75),),
            target_specs=(
                ("service-b", "2", 30, 0.6),
                ("service-b", "3", 40, 0.4),
            ),
        )
    )

    source_score = next(
        item for item in result.candidate_scores if item.candidate_service_key == "service-a"
    )
    assert source_score.top_target_service_keys == ("service-b",)
    assert source_score.top_target_contributions == pytest.approx((0.45,))
    assert source_score.propagation_score == pytest.approx(0.45)


def test_multiple_onsets_for_one_event_are_globally_ambiguous_and_never_credited() -> None:
    inputs = _inputs()
    onsets = inputs["onsets"]
    assert isinstance(onsets, tuple)
    source_onset = next(item for item in onsets if item.service_key == "service-a")
    target_onset = next(item for item in onsets if item.service_key == "service-b")
    second_target_onset = build_onset_interval(
        time_contract=_time_contract(),
        anomaly_id="derived-target-ambiguous",
        service_key="service-b",
        t_low_unix_ns=35,
        t_high_unix_ns=35,
        provenance_ids=target_onset.provenance_ids,
    )
    resolution = freeze_clock_resolution(
        time_contract=_time_contract(),
        expected_time_contract_sha256=case_time_contract_sha256(_time_contract()),
        resolution_status="resolved",
        provenance_ids=("clock:resolved",),
    )
    second_precedence = evaluate_clock_aware_precedence(
        source_onset=source_onset,
        target_onset=second_target_onset,
        time_contract=_time_contract(),
        clock_resolution=resolution,
        expected_clock_resolution_sha256=resolution.resolution_sha256,
    )
    inputs["onsets"] = (*onsets, second_target_onset)
    inputs["precedences"] = (*inputs["precedences"], second_precedence)

    result = propagation_module().score_propagation_paths(**inputs)

    assert len(result.path_audits) == 1
    audit = result.path_audits[0]
    assert audit.target_onset_id is None
    assert audit.target_onset_option_ids == tuple(
        sorted((target_onset.onset_id, second_target_onset.onset_id))
    )
    assert "target_onset_binding_ambiguous" in audit.reason_codes
    assert audit.raw_contribution == 0.0


def test_structural_verification_binds_every_frozen_candidate_key() -> None:
    result = propagation_module().score_propagation_paths(**_inputs())
    assert result.candidate_service_keys == (
        "service-a",
        "service-b",
        "service-c",
    )
    shortened = result.candidate_scores[:-1]
    forged = _resign_model(
        result,
        prefix="pres",
        id_field="scoring_result_id",
        sha_field="result_sha256",
        updates={
            "candidate_scores": tuple(item.model_dump(mode="json") for item in shortened),
        },
    )

    assert forged.verify_structure() is False
    assert forged.verify() is False


@pytest.mark.parametrize("kind", ("target", "candidate"))
def test_public_substructure_verification_rejects_unexpected_private_state(kind: str) -> None:
    result = propagation_module().score_propagation_paths(**_inputs())
    value = result.target_scores[0] if kind == "target" else result.candidate_scores[0]
    object.__setattr__(
        value,
        "__pydantic_private__",
        {"unexpected_state": "SECRET_SUBMITTED_VALUE"},
    )

    assert value.verify() is False


def test_public_substructure_verification_rejects_extra_instance_dictionary_state() -> None:
    result = propagation_module().score_propagation_paths(**_inputs())
    candidate = result.candidate_scores[0]
    object.__getattribute__(candidate, "__dict__")["unexpected_state"] = "SECRET_SUBMITTED_VALUE"

    assert candidate.verify() is False


def test_result_verification_rejects_a_model_subclass() -> None:
    module = propagation_module()
    result = module.score_propagation_paths(**_inputs())

    class ForgedResult(module.PropagationScoringResult):
        pass

    forged = ForgedResult.model_validate(result.model_dump(mode="json"))

    assert forged.verify_structure() is False


def test_result_verification_rejects_a_nested_model_subclass() -> None:
    module = propagation_module()
    result = module.score_propagation_paths(**_inputs())
    candidate = result.candidate_scores[0]

    class ForgedCandidate(module.CandidatePropagationScore):
        pass

    forged_candidate = ForgedCandidate.model_validate(candidate.model_dump(mode="json"))
    forged_result = result.model_copy(
        update={"candidate_scores": (forged_candidate, *result.candidate_scores[1:])}
    )

    assert forged_candidate.verify() is False
    assert forged_result.verify_structure() is False


def test_candidate_top_contributions_and_score_are_bounded_to_unit_interval() -> None:
    candidate = propagation_module().score_propagation_paths(**_inputs()).candidate_scores[0]

    with pytest.raises(ValidationError):
        _resign_model(
            candidate,
            prefix="pcand",
            id_field="candidate_score_id",
            sha_field="candidate_score_sha256",
            updates={
                "top_target_contributions": (1.1,),
                "propagation_score": 1.1,
            },
        )


def test_shared_bundle_pair_budget_rejects_before_combinatorial_expansion() -> None:
    module = propagation_module()
    inputs = _scenario(
        edge_specs=(
            ("service-a", "service-b", 0.75),
            ("service-a", "service-c", 0.75),
        ),
        target_specs=(
            ("service-b", "2", 30, 0.6),
            ("service-c", "3", 40, 0.6),
        ),
    )
    inputs["config"] = module.build_propagation_scoring_config(max_bundle_pairs=1)

    with pytest.raises(module.PropagationScoringBudgetExceeded) as captured:
        module.score_propagation_paths(**inputs)

    assert str(captured.value) == "propagation scoring budget exceeded"
    assert captured.value.reason_code == "bundle_pair_budget_exceeded"


@pytest.mark.parametrize(
    ("config_overrides", "reason_code"),
    (
        ({"max_expanded_states": 1}, "expanded_state_budget_exceeded"),
        ({"max_traversed_edges": 1}, "traversed_edge_budget_exceeded"),
    ),
)
def test_shared_graph_search_budgets_reject_before_over_limit_work(
    config_overrides: dict[str, int], reason_code: str
) -> None:
    module = propagation_module()
    inputs = _scenario(
        edge_specs=(
            ("service-a", "service-x", 0.75),
            ("service-x", "service-b", 0.75),
        ),
        target_specs=(("service-b", "2", 30, 0.6),),
    )
    inputs["config"] = module.build_propagation_scoring_config(**config_overrides)

    with pytest.raises(module.PropagationScoringBudgetExceeded) as captured:
        module.score_propagation_paths(**inputs)

    assert str(captured.value) == "propagation scoring budget exceeded"
    assert captured.value.reason_code == reason_code


def test_graph_search_budget_exact_boundary_is_permitted() -> None:
    module = propagation_module()
    inputs = _scenario(
        edge_specs=(
            ("service-a", "service-x", 0.75),
            ("service-x", "service-b", 0.75),
        ),
        target_specs=(("service-b", "2", 30, 0.6),),
    )
    inputs["config"] = module.build_propagation_scoring_config(
        max_expanded_states=2,
        max_traversed_edges=2,
    )

    result = module.score_propagation_paths(**inputs)

    assert result.path_audits[0].path_length == 2
    assert result.path_audits[0].raw_contribution == pytest.approx(0.225)


def test_condensation_structure_budget_is_checked_before_serialization_or_verification(
    monkeypatch: Any,
) -> None:
    module = propagation_module()
    inputs = _inputs()
    monkeypatch.setattr(module, "_MAX_CONDENSATION_COMPONENTS", 1)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("deep serialization or verification ran before the structure cap")

    monkeypatch.setattr(module, "_trusted_model", fail_if_called)
    condensation_type = type(inputs["condensations"][0])
    monkeypatch.setattr(condensation_type, "verify", fail_if_called)
    exact_tuple_field = module._exact_tuple_field

    def reject_nested_component_scan(
        value: object, *, field_name: str, label: str
    ) -> tuple[object, ...]:
        if label == "relation SCC component":
            raise AssertionError("component members were scanned after the component cap")
        return exact_tuple_field(value, field_name=field_name, label=label)

    monkeypatch.setattr(module, "_exact_tuple_field", reject_nested_component_scan)

    with pytest.raises(module.PropagationScoringBudgetExceeded) as captured:
        module.score_propagation_paths(**inputs)

    assert str(captured.value) == "propagation scoring budget exceeded"
    assert captured.value.reason_code == "condensation_structure_budget_exceeded"


def test_untrusted_strength_validation_error_is_fixed_and_secret_free() -> None:
    inputs = _inputs()
    compilation = inputs["role_compilation"]
    assertions = compilation.assertions
    submitted = assertions[0].model_dump(mode="python")
    submitted["strength"] = "SECRET_SUBMITTED_VALUE"
    malicious = type(assertions[0]).model_construct(**submitted)
    inputs["role_compilation"] = compilation.model_copy(
        update={"assertions": (malicious, *assertions[1:])}
    )

    with pytest.raises(ValueError) as captured:
        propagation_module().score_propagation_paths(**inputs)

    rendered = "".join(
        traceback.format_exception(
            type(captured.value),
            captured.value,
            captured.value.__traceback__,
        )
    )
    assert str(captured.value) == "propagation scoring input verification failed"
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "SECRET_SUBMITTED_VALUE" not in rendered


def test_trusted_verification_rejects_a_whole_tree_consistent_resign() -> None:
    result = propagation_module().score_propagation_paths(**_inputs())
    forged = _resign_model(
        result,
        prefix="pres",
        id_field="scoring_result_id",
        sha_field="result_sha256",
        updates={"candidate_freeze_sha256": "f" * 64},
    )

    assert forged.verify_structure() is True
    assert forged.verify() is True
    assert (
        forged.verify_trusted(
            expected_result_sha256=result.result_sha256,
            expected_authoritative_input_sha256=result.authoritative_input_sha256,
            **_inputs(),
        )
        is False
    )
