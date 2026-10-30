"""Source-replayed live onset comparison plus the existing graph/role constraints.

Independent graph manifest/pair pins come from the enclosing complete-case raw
graph replay, not the proposed result. A temporal gate does not authorize a
score: the full mechanism/role compiler must additionally verify its conditions.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Literal, cast

from pydantic import ValidationInfo, field_validator

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.dynamic_verification.contracts import SHA256, TruthValue
from nurse_rcl.v2.dynamic_verification.predicate_evaluation import _compare_time_bounds
from nurse_rcl.v2.dynamic_verification.safety import (
    plain,
    require_pin,
)
from nurse_rcl.v2.dynamic_verification.source_store import (
    SourceBlockStore,
    require_store_unchanged,
    store_snapshot,
)
from nurse_rcl.v2.graph_snapshots import (
    FrozenGraphInputManifest,
    GraphRelationType,
    GraphSnapshotPair,
)
from nurse_rcl.v2.live.repair_measurements import (
    REPAIR_MEASUREMENT_SEMANTICS_SHA256,
    RepairMeasuredPredicate,
    RepairSignalMeasurement,
    _measure_values,
    _predicate_values,
    _source_replay,
)
from nurse_rcl.v2.live.repair_safety import (
    REPAIR_CAPACITY_SHA256,
    RepairModel,
    canonical,
    require_unchanged,
    snapshot,
)
from nurse_rcl.v2.live.repair_sources import RepairMeasurementSource
from nurse_rcl.v2.observation_schemas import CandidateUniverse, CaseTimeContract, TimeInterval
from nurse_rcl.v2.propagation_scoring import (
    _MECHANISM_ROLE_RELATION_POLICY,
    PROPAGATION_RELATION_POLICY_SHA256,
)
from nurse_rcl.v2.temporal_condensation import (
    FrozenClockResolution,
    require_frozen_clock_resolution,
)
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

REPAIR_TEMPORAL_SCHEMA_VERSION: Literal["nurse.v2.live-repair-temporal/1"] = (
    "nurse.v2.live-repair-temporal/1"
)
REPAIR_TEMPORAL_SEMANTICS_SHA256 = sha256_hex(
    canonical_json_bytes(
        {
            "version": "nurse-v2-live-repair-temporal/1",
            "capacity": REPAIR_CAPACITY_SHA256,
            "measurements": REPAIR_MEASUREMENT_SEMANTICS_SHA256,
            "relation_policy": PROPAGATION_RELATION_POLICY_SHA256,
            "onset": "one_qualifying_bound_episode_no_gap_left_censor_or_ambiguous_binding",
            "lifecycle_onset": (
                "one_source_zero_event_positive_predecessor_raw_bracket_no_double_clock_padding"
            ),
            "clock": "reuse_strict_interval_comparison_with_frozen_clock_once",
            "graph": "complete_manifest_pair_replay_known_coverage_matching_directed_fault_edge",
            "relation_identity": "requested_registered_relation_retained_even_without_edge",
            "authority": "temporal_relation_gate_only_not_full_mechanism_or_score",
        }
    )
)


class RepairTemporalEvaluation(RepairModel):
    schema_version: Literal["nurse.v2.live-repair-temporal/1"] = REPAIR_TEMPORAL_SCHEMA_VERSION
    semantics_sha256: SHA256
    source_sha256s: tuple[str, str]
    measurement_sha256s: tuple[str, str]
    predicate_sha256s: tuple[str, str]
    graph_pair_sha256: SHA256
    graph_manifest_sha256: SHA256
    clock_resolution_sha256: SHA256
    candidate_service_keys: tuple[str, str]
    mechanism_keys: tuple[str, str]
    predicate_kinds: tuple[str, str]
    relation_type: GraphRelationType
    interval: TimeInterval
    source_bounds_ns: tuple[int, int] | None
    target_bounds_ns: tuple[int, int] | None
    source_precedes_target: TruthValue
    edge_sha256s: tuple[str, ...]
    temporal_relation_eligible: bool
    reason_codes: tuple[str, ...]
    score_authority: Literal[False] = False
    result_sha256: SHA256

    @field_validator("source_bounds_ns", "target_bounds_ns", mode="before")
    @classmethod
    def json_optional_bounds(cls, value: object, info: ValidationInfo) -> object:
        # Strict model preflight already verifies the exact JSON union shape.
        # The shared base restores tuple fields but not tuple | None unions.
        if info.mode == "json" and type(value) is list:
            return tuple(value)
        return value


_ROOTS = (
    RepairTemporalEvaluation,
    RepairMeasurementSource,
    RepairSignalMeasurement,
    RepairMeasuredPredicate,
    FrozenGraphInputManifest,
    GraphSnapshotPair,
    FrozenClockResolution,
    CandidateUniverse,
    FrozenCandidateUniverse,
    TemporalEntityRegistry,
    CaseTimeContract,
)


def _onset_bounds(
    measurement: RepairSignalMeasurement,
    predicate: RepairMeasuredPredicate,
) -> tuple[tuple[int, int] | None, tuple[str, ...]]:
    if predicate.truth_value != "true":
        return None, ("directional_predicate_not_true",)
    if predicate.measurement_field == "lifecycle.availability_support":
        lifecycle = measurement.lifecycle
        if lifecycle is None or not lifecycle.availability_support:
            return None, ("availability_onset_unavailable",)
        if len(lifecycle.events) != 1:
            return None, ("ambiguous_episode_binding",)
        event = lifecycle.events[0]
        if event.left_censored or event.onset_bounds_ns is None:
            return None, ("left_censored_onset",)
        index = next(
            i
            for i, point in enumerate(measurement.points)
            if point.timestamp_unix_ns == event.zero_timestamps_ns[0]
        )
        if index == 0:
            return None, ("left_censored_onset",)
        # Event bounds already include clock padding; use the original measured
        # points here so the shared comparator applies the frozen clock once.
        return (
            measurement.points[index - 1].timestamp_unix_ns,
            measurement.points[index].timestamp_unix_ns,
        ), ()
    if predicate.measurement_field not in {"persistent_up", "persistent_down"}:
        return None, ("directional_episode_binding_unavailable",)
    direction = "up" if predicate.measurement_field == "persistent_up" else "down"
    episodes = tuple(e for e in measurement.episodes if e.qualifying and e.direction == direction)
    if len(episodes) > 1:
        return None, ("ambiguous_episode_binding",)
    if not episodes:
        return None, ("no_qualifying_episode",)
    episode = episodes[0]
    low, high = episode.onset_lower_unix_ns, episode.onset_upper_unix_ns
    if episode.left_censored or low is None:
        return None, ("left_censored_onset",)
    if low < measurement.interval.start_unix_ns or high >= measurement.interval.end_unix_ns:
        return None, ("onset_outside_frozen_relation_window",)
    return (low, high), ()


def _temporal_values(
    sources: tuple[RepairMeasurementSource, ...],
    measurements: tuple[RepairSignalMeasurement, ...],
    predicates: tuple[RepairMeasuredPredicate, ...],
    graph: GraphSnapshotPair,
    clock: FrozenClockResolution,
    relation_type: GraphRelationType,
) -> RepairTemporalEvaluation:
    """Pure body inside source, graph and clock replay plus two-exit guards."""
    left, left_reasons = _onset_bounds(measurements[0], predicates[0])
    right, right_reasons = _onset_bounds(measurements[1], predicates[1])
    truth, time_reasons = _compare_time_bounds(left, right, clock, (*left_reasons, *right_reasons))
    keys = (predicates[0].candidate_service_key, predicates[1].candidate_service_key)
    reasons = list(time_reasons)
    fault = graph.fault_snapshot
    edges = tuple(
        edge
        for edge in fault.edges
        if edge.relation_type == relation_type
        and edge.allows_directed_propagation
        and (
            (edge.caller_service_key, edge.callee_service_key)
            if edge.direction == "caller_to_callee"
            else (edge.callee_service_key, edge.caller_service_key)
        )
        == keys
    )
    if not edges:
        reasons.append("directed_relation_unavailable")
    graph_permitted = bool(edges)
    if fault.coverage.coverage_state != "sufficient" or any(
        e.coverage.coverage_state != "sufficient" for e in edges
    ):
        reasons.append("graph_coverage_insufficient_or_unknown")
        graph_permitted = False
    for predicate, role in zip(predicates, ("source_support", "symptom_support"), strict=True):
        if not any(
            predicate.mechanism_key == mechanism
            and role == allowed_role
            and relation_type in relations
            for mechanism, allowed_role, relations in _MECHANISM_ROLE_RELATION_POLICY
        ):
            reasons.append("mechanism_role_relation_denied")
            graph_permitted = False
    if keys[0] == keys[1]:
        reasons.append("same_candidate_is_not_cross_service_propagation")
        graph_permitted = False
    for edge in edges:
        if edge.observation_state not in {"observed", "zero", "verified_normal"}:
            reasons.append("relation_observation_unavailable")
            graph_permitted = False
        if (
            left is not None
            and right is not None
            and not (
                edge.interval.start_unix_ns <= min(left[0], right[0])
                and max(left[1], right[1]) < edge.interval.end_unix_ns
            )
        ):
            reasons.append("onset_outside_frozen_relation_window")
            graph_permitted = False
    draft = RepairTemporalEvaluation(
        semantics_sha256=REPAIR_TEMPORAL_SEMANTICS_SHA256,
        source_sha256s=(sources[0].source_sha256, sources[1].source_sha256),
        measurement_sha256s=(
            measurements[0].measurement_sha256,
            measurements[1].measurement_sha256,
        ),
        predicate_sha256s=(predicates[0].result_sha256, predicates[1].result_sha256),
        graph_pair_sha256=graph.pair_sha256,
        graph_manifest_sha256=graph.graph_input_manifest_sha256,
        clock_resolution_sha256=clock.resolution_sha256,
        candidate_service_keys=keys,
        mechanism_keys=(predicates[0].mechanism_key, predicates[1].mechanism_key),
        predicate_kinds=(predicates[0].predicate_kind, predicates[1].predicate_kind),
        relation_type=relation_type,
        interval=fault.interval,
        source_bounds_ns=left,
        target_bounds_ns=right,
        source_precedes_target=truth,
        edge_sha256s=tuple(sorted(e.edge_sha256 for e in edges)),
        temporal_relation_eligible=truth == "true" and graph_permitted,
        reason_codes=tuple(sorted(set(reasons))),
        result_sha256="0" * 64,
    )
    unsigned = cast(dict[str, object], plain(draft))
    unsigned.pop("result_sha256")
    return draft.model_copy(update={"result_sha256": sha256_hex(canonical_json_bytes(unsigned))})


def evaluate_repair_temporal_pair(
    *,
    sources: tuple[RepairMeasurementSource, ...],
    expected_source_sha256s: tuple[str, ...],
    measurements: tuple[RepairSignalMeasurement, ...],
    expected_measurement_sha256s: tuple[str, ...],
    time_contract: CaseTimeContract,
    expected_time_contract_sha256: str,
    store: SourceBlockStore,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    graph_input_manifest: FrozenGraphInputManifest,
    expected_graph_input_manifest_sha256: str,
    graph_pair: GraphSnapshotPair,
    expected_graph_pair_sha256: str,
    clock_resolution: FrozenClockResolution,
    expected_clock_resolution_sha256: str,
    mechanism_keys: tuple[str, ...],
    predicate_kinds: tuple[str, ...],
    relation_type: GraphRelationType,
) -> RepairTemporalEvaluation:
    inputs = (
        sources,
        measurements,
        time_contract,
        candidate_universe,
        frozen_candidates,
        entity_registry,
        graph_input_manifest,
        graph_pair,
        clock_resolution,
        expected_source_sha256s,
        expected_measurement_sha256s,
        mechanism_keys,
        predicate_kinds,
        relation_type,
    )
    before = snapshot(inputs, _ROOTS)
    state = store_snapshot(store)
    try:
        if any(
            type(values) is not tuple or len(values) != 2
            for values in (
                sources,
                measurements,
                expected_source_sha256s,
                expected_measurement_sha256s,
                mechanism_keys,
                predicate_kinds,
            )
        ):
            raise ValueError("repair time requires an exact pair without truncation")
        for value, kind in (
            (time_contract, CaseTimeContract),
            (candidate_universe, CandidateUniverse),
            (frozen_candidates, FrozenCandidateUniverse),
            (entity_registry, TemporalEntityRegistry),
            (graph_input_manifest, FrozenGraphInputManifest),
            (graph_pair, GraphSnapshotPair),
            (clock_resolution, FrozenClockResolution),
        ):
            if type(value) is not kind:
                raise ValueError("repair time requires exact frozen graph/time inputs")
        if any(type(s) is not RepairMeasurementSource for s in sources) or any(
            type(m) is not RepairSignalMeasurement for m in measurements
        ):
            raise ValueError("repair time requires exact current source and measurement types")
        if type(relation_type) is not str:
            raise ValueError("repair time relation must be exact text")
        for pin in (
            *expected_source_sha256s,
            *expected_measurement_sha256s,
            expected_time_contract_sha256,
            expected_graph_input_manifest_sha256,
            expected_graph_pair_sha256,
            expected_clock_resolution_sha256,
        ):
            require_pin(pin)
        with ExitStack() as guards:
            universe = CandidateUniverse.model_validate_json(canonical(candidate_universe, _ROOTS))
            frozen = FrozenCandidateUniverse.model_validate_json(
                canonical(frozen_candidates, _ROOTS)
            )
            registry = TemporalEntityRegistry.model_validate_json(
                canonical(entity_registry, _ROOTS)
            )
            manifest = FrozenGraphInputManifest.model_validate_json(
                canonical(graph_input_manifest, _ROOTS)
            )
            graph = GraphSnapshotPair.model_validate_json(canonical(graph_pair, _ROOTS))
            contract = CaseTimeContract.model_validate_json(canonical(time_contract, _ROOTS))
            working = (universe, frozen, registry, manifest, graph, contract)
            guards.callback(require_unchanged, working, _ROOTS, snapshot(working, _ROOTS))
            if graph.pair_sha256 != expected_graph_pair_sha256 or not graph.verify_inputs(
                time_contract=contract,
                candidate_universe=universe,
                frozen_candidates=frozen,
                entity_registry=registry,
                graph_input_manifest=manifest,
                expected_graph_input_manifest_sha256=expected_graph_input_manifest_sha256,
                expected_minimum_coverage_fraction=0.8,
            ):
                raise ValueError(
                    "repair temporal graph cannot replay full independently pinned inputs"
                )
            resolution = require_frozen_clock_resolution(
                clock_resolution=clock_resolution,
                time_contract=contract,
                expected_clock_resolution_sha256=expected_clock_resolution_sha256,
            )
            guards.callback(require_unchanged, resolution, _ROOTS, snapshot(resolution, _ROOTS))
            rebuilt_sources, rebuilt_measurements, predicates = [], [], []
            for i in range(2):
                fresh = guards.enter_context(
                    _source_replay(
                        source=sources[i],
                        expected_source_sha256=expected_source_sha256s[i],
                        time_contract=contract,
                        expected_time_contract_sha256=expected_time_contract_sha256,
                        store=store,
                        extra=(measurements[i],),
                    )
                )
                if fresh.candidate_service_key not in frozen.candidate_service_keys or not any(
                    r.binding.status == "bound"
                    and r.binding.service_entity_key == fresh.candidate_service_key
                    and r.observed_entity_key == fresh.observed_entity_key
                    and fresh.observation.entity_binding_id is None
                    and canonical(
                        r.observation.model_copy(update={"entity_binding_id": None}), _ROOTS
                    )
                    == canonical(fresh.observation, _ROOTS)
                    for r in registry.resolutions
                ):
                    raise ValueError(
                        "repair time source is outside complete frozen "
                        "candidate/observation bindings"
                    )
                measured = _measure_values(fresh)
                guards.callback(require_unchanged, measured, _ROOTS, snapshot(measured, _ROOTS))
                if measured.measurement_sha256 != expected_measurement_sha256s[i] or (
                    canonical(measured, _ROOTS) != canonical(measurements[i], _ROOTS)
                ):
                    raise ValueError("repair time measurement differs from raw-source replay")
                predicate = _predicate_values(
                    fresh,
                    measured,
                    fresh.candidate_service_key,
                    mechanism_keys[i],
                    predicate_kinds[i],
                )
                guards.callback(require_unchanged, predicate, _ROOTS, snapshot(predicate, _ROOTS))
                rebuilt_sources.append(fresh)
                rebuilt_measurements.append(measured)
                predicates.append(predicate)
            return _temporal_values(
                tuple(rebuilt_sources),
                tuple(rebuilt_measurements),
                tuple(predicates),
                graph,
                resolution,
                relation_type,
            )
    finally:
        require_unchanged(inputs, _ROOTS, before)
        require_store_unchanged(store, state)
