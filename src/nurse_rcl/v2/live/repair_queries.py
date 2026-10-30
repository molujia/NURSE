"""Current-source projections for the existing feedback-first query policy.

No legacy source/catalog is relabelled. Selection uses only fixed summaries and
completed feedback. These projections grant neither execution nor score trust:
the session owns durable reservations and replays each completed measurement.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from functools import partial
from typing import Final, Literal, TypeVar, cast

from pydantic import Field, ValidationInfo, field_validator

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse, require_frozen_candidate_universe
from nurse_rcl.v2.dynamic_verification.contracts import SHA256, _SourceModel
from nurse_rcl.v2.dynamic_verification.requests import build_dynamic_verification_policy
from nurse_rcl.v2.dynamic_verification.safety import (
    plain,
    require_pin,
)
from nurse_rcl.v2.dynamic_verification.scheduler import RequestSelectionStatus
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
from nurse_rcl.v2.hypothesis_guard import _PREDICATE_EVIDENCE_CONTRACTS
from nurse_rcl.v2.live.repair_measurements import (
    _PRESSURE_METRICS,
    _UP_PREDICATES,
    REPAIR_MEASUREMENT_SEMANTICS_SHA256,
    RepairMeasuredPredicate,
    RepairSignalMeasurement,
    _measure_values,
    _numeric_identity,
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
from nurse_rcl.v2.live.repair_temporal import _onset_bounds
from nurse_rcl.v2.live.repair_timing import _preparation_scope
from nurse_rcl.v2.mechanism_library import build_mechanism_library
from nurse_rcl.v2.observation_schemas import CandidateUniverse, CaseTimeContract
from nurse_rcl.v2.propagation_scoring import _MECHANISM_ROLE_RELATION_POLICY
from nurse_rcl.v2.temporal_condensation import (
    FrozenClockResolution,
    require_frozen_clock_resolution,
)
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

REPAIR_QUERY_CATALOG_SCHEMA_VERSION: Final = "nurse.v2.live-repair-query-catalog/1"
REPAIR_QUERY_FEEDBACK_SCHEMA_VERSION: Final = "nurse.v2.live-repair-query-feedback/1"
REPAIR_QUERY_SELECTION_SCHEMA_VERSION: Final = "nurse.v2.live-repair-query-selection/1"


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


REPAIR_QUERY_SEMANTICS_SHA256 = _digest(
    {
        "version": "nurse-v2-live-repair-queries/1",
        "capacity": REPAIR_CAPACITY_SHA256,
        "measurements": REPAIR_MEASUREMENT_SEMANTICS_SHA256,
        "policy": build_dynamic_verification_policy().policy_sha256,
        "priority": "existing_tier_candidate_paid_distinct_need_count_stable_tie",
        "fixed": "extrema_only_no_episode_or_future_measurement",
        "time": "observed_counterpart_onset_promotes_only_pinned_eligible_relation_need",
        "authority": "selector_is_not_source_reservation_or_score_authority",
    }
)


class RepairQueryNeed(RepairModel):
    need_key: str
    condition_key: str
    mechanism_key: str
    predicate_kind: str
    requirement: Literal["necessary", "optional", "refuting"]
    queryable: bool
    initial_tier: int | None = Field(ge=1, le=2)
    reason_codes: tuple[str, ...]


class RepairQueryRequest(RepairModel):
    request_key: str
    source_sha256: SHA256
    signal_id: str
    observation_id: str
    canonical_series_key: str
    candidate_service_key: str | None
    needs: tuple[RepairQueryNeed, ...]


class RepairQueryTimeNeed(RepairModel):
    edge_sha256: SHA256
    relation_type: GraphRelationType
    source_need_keys: tuple[str, ...]
    target_need_keys: tuple[str, ...]


class RepairQueryCatalog(RepairModel):
    schema_version: Literal["nurse.v2.live-repair-query-catalog/1"] = (
        REPAIR_QUERY_CATALOG_SCHEMA_VERSION
    )
    semantics_sha256: SHA256
    opaque_case_id: str
    candidate_freeze_sha256: SHA256
    registry_sha256: SHA256
    time_contract_sha256: SHA256
    candidate_service_keys: tuple[str, ...]
    requests: tuple[RepairQueryRequest, ...]
    graph_pair_sha256: SHA256 | None
    clock_resolution_sha256: SHA256 | None
    time_needs: tuple[RepairQueryTimeNeed, ...]
    catalog_sha256: SHA256


class RepairQueryOnset(RepairModel):
    need_key: str
    predicate_sha256: SHA256
    bounds_ns: tuple[int, int] | None
    reason_codes: tuple[str, ...]

    @field_validator("bounds_ns", mode="before")
    @classmethod
    def json_optional_bounds(cls, value: object, info: ValidationInfo) -> object:
        if info.mode == "json" and type(value) is list:
            return tuple(value)
        return value


class RepairQueryFeedback(RepairModel):
    schema_version: Literal["nurse.v2.live-repair-query-feedback/1"] = (
        REPAIR_QUERY_FEEDBACK_SCHEMA_VERSION
    )
    semantics_sha256: SHA256
    catalog_sha256: SHA256
    request_key: str
    measurement_sha256: SHA256
    predicates: tuple[RepairMeasuredPredicate, ...]
    onsets: tuple[RepairQueryOnset, ...]
    result_sha256: SHA256


class RepairQuerySelection(RepairModel):
    schema_version: Literal["nurse.v2.live-repair-query-selection/1"] = (
        REPAIR_QUERY_SELECTION_SCHEMA_VERSION
    )
    semantics_sha256: SHA256
    catalog_sha256: SHA256
    feedback_sha256s: tuple[str, ...]
    submitted_request_keys: tuple[str, ...]
    requests: tuple[RequestSelectionStatus, ...]
    selected_request_key: str | None
    paid_count: int = Field(ge=0, le=8)
    remaining_budget: int = Field(ge=0, le=8)
    stop_reason: Literal["query_budget_exhausted", "no_admissible_progress"] | None
    result_sha256: SHA256


_ROOTS = (
    RepairQueryCatalog,
    RepairQueryFeedback,
    RepairQuerySelection,
    RepairQueryRequest,
    RepairMeasurementSource,
    RepairSignalMeasurement,
    RepairMeasuredPredicate,
    CaseTimeContract,
    CandidateUniverse,
    FrozenCandidateUniverse,
    TemporalEntityRegistry,
    GraphSnapshotPair,
    FrozenGraphInputManifest,
    FrozenClockResolution,
)
_T = TypeVar("_T", bound=_SourceModel)
_Q = TypeVar("_Q", RepairQueryCatalog, RepairQueryFeedback, RepairQuerySelection)


def _seal(draft: _T, field: str) -> _T:
    document = cast(dict[str, object], plain(draft))
    document.pop(field)
    return draft.model_copy(update={field: _digest(document)})


@contextmanager
def _checked(value: _Q, cls: type[_Q], pin: str, field: str) -> Iterator[_Q]:
    if type(value) is not cls:
        raise ValueError("repair query boundary requires exact current types")
    require_pin(pin)
    before = snapshot(value, _ROOTS)
    try:
        fresh = cls.model_validate_json(before[0])
        working = snapshot(fresh, _ROOTS)
        try:
            if (
                working[0] != before[0]
                or getattr(fresh, field) != pin
                or _seal(fresh, field) != fresh
            ):
                raise ValueError("repair query independent pin or content mismatch")
            if fresh.semantics_sha256 != REPAIR_QUERY_SEMANTICS_SHA256:
                raise ValueError("repair query semantics mismatch")
            yield fresh
        finally:
            require_unchanged(fresh, _ROOTS, working)
    finally:
        require_unchanged(value, _ROOTS, before)


def _need_values(source: RepairMeasurementSource) -> tuple[RepairQueryNeed, ...]:
    """Fixed scalar summaries only; lifecycle/error facts remain outside query costs."""
    if source.candidate_service_key is None:
        return ()
    obs, quality = source.observation, source.quality
    summary = quality.source_summary
    rows = []
    for definition in build_mechanism_library().definitions:
        for requirement in ("necessary", "optional", "refuting"):
            for kind in getattr(definition, requirement + "_predicate_kinds"):
                contract = _PREDICATE_EVIDENCE_CONTRACTS[kind]
                if obs.signal_family not in (
                    *contract.primary_signal_families,
                    *contract.source_signal_families,
                ):
                    continue
                reasons = []
                if requirement == "refuting" or kind not in (
                    *_UP_PREDICATES,
                    "local_anomaly_observed",
                ):
                    reasons.append("fixed_fact_or_distinct_semantics_not_dynamic_episode")
                if not quality.formal_scoring_eligible:
                    reasons.append("quality_ineligible")
                if obs.aggregation in {"window_count", "exposure_rate"}:
                    reasons.append("temporal_resolution_unavailable")
                if (
                    kind == "local_pressure_observed"
                    and source.metric_name not in _PRESSURE_METRICS[source.dataset]
                ):
                    reasons.append("pressure_metric_semantics_unavailable")
                if kind == "request_latency_increase" and (
                    obs.signal_type != "latency" or obs.unit != "ns"
                ):
                    reasons.append("latency_semantics_unavailable")
                if kind == "incoming_request_rate_increase" and (
                    obs.aggregation not in {"per_second_rate", "exposure_rate"}
                    or not obs.unit.endswith("/second")
                ):
                    reasons.append("rate_semantics_unavailable")
                if kind in {"request_failure_increase", "outgoing_dependency_failure_observed"}:
                    reasons.append("request_outcome_semantics_unavailable")
                median, scale = summary.baseline_summary.median, quality.baseline_scale
                low, high = summary.fault_summary.minimum, summary.fault_summary.maximum
                if median is None or scale is None or scale <= 0 or low is None or high is None:
                    reasons.append("summary_deviation_band_unavailable")
                else:
                    lo, hi = (low - median) / scale, (high - median) / scale
                    threshold = summary.configuration.abnormal_z_threshold
                    if not (math.isfinite(lo) and math.isfinite(hi)):
                        reasons.append("summary_deviation_band_unavailable")
                    elif (
                        max(abs(lo), abs(hi)) if kind == "local_anomaly_observed" else hi
                    ) < threshold:
                        # No observed point can form a qualifying episode. Missing
                        # values remain unknown; this grants no verified-normal fact.
                        reasons.append("summary_proves_no_qualifying_episode")
                condition = "condition_" + _digest(
                    [source.candidate_service_key, kind, requirement]
                )
                rows.append(
                    RepairQueryNeed(
                        need_key="need_"
                        + _digest([source.signal_id, definition.mechanism_key, kind, requirement]),
                        condition_key=condition,
                        mechanism_key=definition.mechanism_key,
                        predicate_kind=kind,
                        requirement=requirement,
                        queryable=not reasons,
                        initial_tier=None if reasons else (1 if requirement == "necessary" else 2),
                        reason_codes=tuple(sorted(set(reasons)))
                        if reasons
                        else ("unresolved_episode_condition",),
                    )
                )
    return tuple(sorted(rows, key=lambda row: row.need_key))


def _request_values(source: RepairMeasurementSource) -> RepairQueryRequest:
    identity = _numeric_identity(source)
    return RepairQueryRequest(
        request_key="request_" + identity.request_sha256,
        source_sha256=source.source_sha256,
        signal_id=source.signal_id,
        observation_id=source.observation.observation_id,
        canonical_series_key=identity.canonical_series_key,
        candidate_service_key=source.candidate_service_key,
        needs=_need_values(source),
    )


def _validate_catalog(directory: RepairQueryCatalog) -> None:
    keys = tuple(r.request_key for r in directory.requests)
    if keys != tuple(sorted(set(keys))) or len(
        {r.source_sha256 for r in directory.requests}
    ) != len(keys):
        raise ValueError("repair query catalog requests must be unique and ordered")
    if directory.candidate_service_keys != tuple(sorted(set(directory.candidate_service_keys))):
        raise ValueError("repair query catalog candidates must be unique and ordered")
    for row in directory.requests:
        if (
            row.candidate_service_key is not None
            and row.candidate_service_key not in directory.candidate_service_keys
        ):
            raise ValueError("repair request outside complete candidates")
        if tuple(n.need_key for n in row.needs) != tuple(sorted({n.need_key for n in row.needs})):
            raise ValueError("repair request needs must be unique and ordered")
        if any(n.queryable != (n.initial_tier is not None) for n in row.needs):
            raise ValueError("repair request queryability differs from priority")
    needs = {n.need_key for r in directory.requests for n in r.needs if n.queryable}
    if len(needs) != sum(n.queryable for r in directory.requests for n in r.needs):
        raise ValueError("repair catalog has repeated query need identities")
    if (directory.graph_pair_sha256 is None) != (directory.clock_resolution_sha256 is None):
        raise ValueError("repair catalog graph and clock must be bound together")
    if directory.time_needs and directory.graph_pair_sha256 is None:
        raise ValueError("repair time needs require graph identity")
    if tuple(n.edge_sha256 for n in directory.time_needs) != tuple(
        sorted({n.edge_sha256 for n in directory.time_needs})
    ):
        raise ValueError("repair time needs require unique ordered graph edges")
    for pair in directory.time_needs:
        for keys in (pair.source_need_keys, pair.target_need_keys):
            if not keys or keys != tuple(sorted(set(keys))) or not set(keys) <= needs:
                raise ValueError("repair time need is outside queryable conditions")


def _time_need_values(
    requests: tuple[RepairQueryRequest, ...], graph: GraphSnapshotPair, clock: FrozenClockResolution
) -> tuple[RepairQueryTimeNeed, ...]:
    if (
        clock.resolution_status != "resolved"
        or graph.fault_snapshot.coverage.coverage_state != "sufficient"
    ):
        return ()
    rows = []
    for edge in graph.fault_snapshot.edges:
        if (
            not edge.allows_directed_propagation
            or edge.coverage.coverage_state != "sufficient"
            or edge.observation_state != "observed"
        ):
            continue
        source, target = (
            (edge.caller_service_key, edge.callee_service_key)
            if edge.direction == "caller_to_callee"
            else (edge.callee_service_key, edge.caller_service_key)
        )
        if source == target:
            continue
        sides = []
        for candidate, role in ((source, "source_support"), (target, "symptom_support")):
            sides.append(
                tuple(
                    sorted(
                        n.need_key
                        for r in requests
                        if r.candidate_service_key == candidate
                        for n in r.needs
                        if n.queryable
                        and n.predicate_kind in _UP_PREDICATES
                        and any(
                            n.mechanism_key == mechanism
                            and role == allowed
                            and edge.relation_type in relations
                            for mechanism, allowed, relations in _MECHANISM_ROLE_RELATION_POLICY
                        )
                    )
                )
            )
        if all(sides):
            rows.append(
                RepairQueryTimeNeed(
                    edge_sha256=edge.edge_sha256,
                    relation_type=edge.relation_type,
                    source_need_keys=sides[0],
                    target_need_keys=sides[1],
                )
            )
    return tuple(sorted(rows, key=lambda row: row.edge_sha256))


class _CatalogChecks:
    """Checks owned by one live catalog scope, never a reusable validation token."""

    def __init__(self, full: Callable[[], None], decision: Callable[[], None]) -> None:
        self._full = full
        self._decision = decision
        self._active = True

    def __call__(self) -> None:
        if not self._active:
            raise ValueError("repair catalog owner is closed")
        self._full()

    def check_decision(self) -> None:
        if not self._active:
            raise ValueError("repair catalog owner is closed")
        self._decision()


@_preparation_scope("catalog_prepare", "catalog_exit")
@contextmanager
def _replayed_repair_query_catalog(
    *,
    sources: tuple[RepairMeasurementSource, ...],
    expected_source_sha256s: tuple[str, ...],
    time_contract: CaseTimeContract,
    expected_time_contract_sha256: str,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    store: SourceBlockStore,
    graph_input_manifest: FrozenGraphInputManifest | None = None,
    expected_graph_input_manifest_sha256: str | None = None,
    graph_pair: GraphSnapshotPair | None = None,
    expected_graph_pair_sha256: str | None = None,
    clock_resolution: FrozenClockResolution | None = None,
    expected_clock_resolution_sha256: str | None = None,
) -> Iterator[tuple[RepairQueryCatalog, _CatalogChecks]]:
    """Replay all source windows; never classify points or precompute episodes.

    Per-source guards retain the existing capacity per source without wrapping
    all raw samples in a new giant root. The outer case pins these inputs after
    its independent file/adapter replay; this boundary verifies complete binding.
    """
    context = (
        time_contract,
        candidate_universe,
        frozen_candidates,
        entity_registry,
        graph_input_manifest,
        graph_pair,
        clock_resolution,
    )
    before = snapshot(context, _ROOTS)
    state = store_snapshot(store)
    try:
        with ExitStack() as guards:
            callbacks: list[Callable[[], None]] = []
            decision_callbacks: list[Callable[[], None]] = []

            def protect(
                value: object, saved: tuple[bytes, str], *, source_payload: bool = False
            ) -> None:
                check = partial(require_unchanged, value, _ROOTS, saved)
                callbacks.append(check)
                if not source_payload:
                    decision_callbacks.append(check)
                guards.callback(check)

            def unchanged() -> None:
                # A failing check must not prevent any other established check.
                with ExitStack() as checks:
                    checks.callback(require_store_unchanged, store, state)
                    checks.callback(require_unchanged, context, _ROOTS, before)
                    for check in callbacks:
                        checks.callback(check)

            def decision_unchanged() -> None:
                with ExitStack() as checks:
                    checks.callback(require_store_unchanged, store, state)
                    checks.callback(require_unchanged, context, _ROOTS, before)
                    for check in decision_callbacks:
                        checks.callback(check)

            if (
                type(sources) is not tuple
                or type(expected_source_sha256s) is not tuple
                or len(sources) != len(expected_source_sha256s)
            ):
                raise ValueError("repair query catalog requires complete exact source tuples")
            source_before = tuple(snapshot(s, _ROOTS) for s in sources)
            for source, saved in zip(sources, source_before, strict=True):
                protect(source, saved, source_payload=True)
            contract = CaseTimeContract.model_validate_json(canonical(time_contract, _ROOTS))
            universe = CandidateUniverse.model_validate_json(canonical(candidate_universe, _ROOTS))
            frozen = FrozenCandidateUniverse.model_validate_json(
                canonical(frozen_candidates, _ROOTS)
            )
            registry = TemporalEntityRegistry.model_validate_json(
                canonical(entity_registry, _ROOTS)
            )
            working = (contract, universe, frozen, registry)
            protect(working, snapshot(working, _ROOTS))
            require_pin(expected_time_contract_sha256)
            if sha256_hex(canonical(contract, _ROOTS)) != expected_time_contract_sha256:
                raise ValueError("repair query time pin mismatch")
            require_frozen_candidate_universe(frozen=frozen, candidate_universe=universe)
            if (
                not registry.verify()
                or registry.opaque_case_id != contract.opaque_case_id
                or registry.decision_time_unix_ns != contract.decision_time_unix_ns
                or frozen.opaque_case_id != contract.opaque_case_id
            ):
                raise ValueError("repair query registry/case mismatch")
            ids = tuple(s.signal_id for s in sources)
            resolutions = {r.observation.observation_id: r for r in registry.resolutions}
            if (
                ids != tuple(sorted(set(ids)))
                or len(sources) != len(resolutions)
                or {s.observation.observation_id for s in sources} != set(resolutions)
            ):
                raise ValueError("repair query sources do not cover the complete case")
            rows = []
            for source, pin in zip(sources, expected_source_sha256s, strict=True):
                with _source_replay(
                    source=source,
                    expected_source_sha256=pin,
                    time_contract=contract,
                    expected_time_contract_sha256=expected_time_contract_sha256,
                    store=store,
                ) as fresh:
                    resolution = resolutions[fresh.observation.observation_id]
                    expected_candidate = (
                        resolution.binding.service_entity_key
                        if resolution.binding.status == "bound"
                        else None
                    )
                    if (
                        fresh.candidate_service_key != expected_candidate
                        or fresh.observed_entity_key != resolution.observed_entity_key
                        or fresh.observation.entity_binding_id is not None
                        or canonical(fresh.observation, _ROOTS)
                        != canonical(
                            resolution.observation.model_copy(update={"entity_binding_id": None}),
                            _ROOTS,
                        )
                    ):
                        raise ValueError("repair query source differs from actual entity binding")
                    rows.append(_request_values(fresh))
            ordered = tuple(sorted(rows, key=lambda r: r.request_key))
            protect(ordered, snapshot(ordered, _ROOTS))
            time_needs: tuple[RepairQueryTimeNeed, ...] = ()
            graph_values = (
                graph_input_manifest,
                expected_graph_input_manifest_sha256,
                graph_pair,
                expected_graph_pair_sha256,
                clock_resolution,
                expected_clock_resolution_sha256,
            )
            if any(value is not None for value in graph_values):
                if graph_input_manifest is None or graph_pair is None or clock_resolution is None:
                    raise ValueError("repair temporal query context must be complete")
                require_pin(expected_graph_input_manifest_sha256)
                require_pin(expected_graph_pair_sha256)
                require_pin(expected_clock_resolution_sha256)
                manifest = FrozenGraphInputManifest.model_validate_json(
                    canonical(graph_input_manifest, _ROOTS)
                )
                graph = GraphSnapshotPair.model_validate_json(canonical(graph_pair, _ROOTS))
                protect((manifest, graph), snapshot((manifest, graph), _ROOTS))
                if graph.pair_sha256 != expected_graph_pair_sha256 or not graph.verify_inputs(
                    time_contract=contract,
                    candidate_universe=universe,
                    frozen_candidates=frozen,
                    entity_registry=registry,
                    graph_input_manifest=manifest,
                    expected_graph_input_manifest_sha256=cast(
                        str, expected_graph_input_manifest_sha256
                    ),
                    expected_minimum_coverage_fraction=0.8,
                ):
                    raise ValueError("repair query graph cannot replay complete independent inputs")
                clock = require_frozen_clock_resolution(
                    clock_resolution=clock_resolution,
                    time_contract=contract,
                    expected_clock_resolution_sha256=cast(str, expected_clock_resolution_sha256),
                )
                protect(clock, snapshot(clock, _ROOTS))
                time_needs = _time_need_values(ordered, graph, clock)
            result = _seal(
                RepairQueryCatalog(
                    semantics_sha256=REPAIR_QUERY_SEMANTICS_SHA256,
                    opaque_case_id=contract.opaque_case_id,
                    candidate_freeze_sha256=frozen.freeze_sha256,
                    registry_sha256=registry.registry_sha256,
                    time_contract_sha256=expected_time_contract_sha256,
                    candidate_service_keys=frozen.candidate_service_keys,
                    requests=ordered,
                    graph_pair_sha256=expected_graph_pair_sha256,
                    clock_resolution_sha256=expected_clock_resolution_sha256,
                    time_needs=time_needs,
                    catalog_sha256="0" * 64,
                ),
                "catalog_sha256",
            )
            _validate_catalog(result)
            protect(result, snapshot(result, _ROOTS))
            owned_checks = _CatalogChecks(unchanged, decision_unchanged)
            try:
                yield result, owned_checks
            finally:
                owned_checks._active = False
    finally:
        try:
            require_unchanged(context, _ROOTS, before)
        finally:
            require_store_unchanged(store, state)


def build_repair_query_catalog(
    *,
    sources: tuple[RepairMeasurementSource, ...],
    expected_source_sha256s: tuple[str, ...],
    time_contract: CaseTimeContract,
    expected_time_contract_sha256: str,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    store: SourceBlockStore,
    graph_input_manifest: FrozenGraphInputManifest | None = None,
    expected_graph_input_manifest_sha256: str | None = None,
    graph_pair: GraphSnapshotPair | None = None,
    expected_graph_pair_sha256: str | None = None,
    clock_resolution: FrozenClockResolution | None = None,
    expected_clock_resolution_sha256: str | None = None,
) -> RepairQueryCatalog:
    """Standalone callers always reconstruct the complete catalog before return."""
    with _replayed_repair_query_catalog(
        sources=sources,
        expected_source_sha256s=expected_source_sha256s,
        time_contract=time_contract,
        expected_time_contract_sha256=expected_time_contract_sha256,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        store=store,
        graph_input_manifest=graph_input_manifest,
        expected_graph_input_manifest_sha256=expected_graph_input_manifest_sha256,
        graph_pair=graph_pair,
        expected_graph_pair_sha256=expected_graph_pair_sha256,
        clock_resolution=clock_resolution,
        expected_clock_resolution_sha256=expected_clock_resolution_sha256,
    ) as (result, _):
        return result


def build_repair_query_feedback(
    *,
    catalog: RepairQueryCatalog,
    expected_catalog_sha256: str,
    source: RepairMeasurementSource,
    expected_source_sha256: str,
    time_contract: CaseTimeContract,
    expected_time_contract_sha256: str,
    store: SourceBlockStore,
    measurement: RepairSignalMeasurement,
    expected_measurement_sha256: str,
) -> RepairQueryFeedback:
    with _checked(
        catalog, RepairQueryCatalog, expected_catalog_sha256, "catalog_sha256"
    ) as directory:
        _validate_catalog(directory)
        require_pin(expected_measurement_sha256)
        if directory.time_contract_sha256 != expected_time_contract_sha256:
            raise ValueError("repair query feedback time pin mismatch")
        with _source_replay(
            source=source,
            expected_source_sha256=expected_source_sha256,
            time_contract=time_contract,
            expected_time_contract_sha256=expected_time_contract_sha256,
            store=store,
            extra=(measurement,),
        ) as fresh:
            request = _request_values(fresh)
            if request not in directory.requests or request.candidate_service_key is None:
                raise ValueError("repair query feedback lacks a bound frozen request")
            rebuilt = _measure_values(fresh)
            before = snapshot(rebuilt, _ROOTS)
            predicates: tuple[RepairMeasuredPredicate, ...] = ()
            predicate_before: tuple[bytes, str] | None = None
            try:
                if rebuilt.measurement_sha256 != expected_measurement_sha256 or canonical(
                    rebuilt, _ROOTS
                ) != canonical(measurement, _ROOTS):
                    raise ValueError("repair query feedback measurement differs from raw sources")
                predicates = _feedback_predicates(fresh, rebuilt, request)
                predicate_before = snapshot(predicates, _ROOTS)
                return _feedback_from_predicates(directory, request, rebuilt, predicates)
            finally:
                try:
                    require_unchanged(rebuilt, _ROOTS, before)
                finally:
                    if predicate_before is not None:
                        require_unchanged(predicates, _ROOTS, predicate_before)


def _feedback_predicates(
    source: RepairMeasurementSource, measurement: RepairSignalMeasurement,
    request: RepairQueryRequest,
) -> tuple[RepairMeasuredPredicate, ...]:
    if request.candidate_service_key is None:
        raise ValueError("repair query feedback lacks a bound frozen request")
    return tuple(
        _predicate_values(source, measurement, request.candidate_service_key,
                          need.mechanism_key, need.predicate_kind)
        for need in request.needs
    )


def _feedback_from_predicates(
    catalog: RepairQueryCatalog, request: RepairQueryRequest,
    measurement: RepairSignalMeasurement, predicates: tuple[RepairMeasuredPredicate, ...],
) -> RepairQueryFeedback:
    onsets = []
    for need, predicate in zip(request.needs, predicates, strict=True):
        bounds, reasons = _onset_bounds(measurement, predicate)
        onsets.append(RepairQueryOnset(
            need_key=need.need_key, predicate_sha256=predicate.result_sha256,
            bounds_ns=bounds, reason_codes=reasons,
        ))
    return _seal(RepairQueryFeedback(
        semantics_sha256=REPAIR_QUERY_SEMANTICS_SHA256,
        catalog_sha256=catalog.catalog_sha256, request_key=request.request_key,
        measurement_sha256=measurement.measurement_sha256, predicates=predicates,
        onsets=tuple(onsets), result_sha256="0" * 64,
    ), "result_sha256")


def _feedback_values(
    catalog: RepairQueryCatalog, source: RepairMeasurementSource,
    measurement: RepairSignalMeasurement,
) -> RepairQueryFeedback:
    """One numerical construction from the owned measurement; no source replay."""
    request = _request_values(source)
    if request not in catalog.requests or request.candidate_service_key is None:
        raise ValueError("repair query feedback lacks a bound frozen request")
    if measurement.source_sha256 != source.source_sha256 or (
        measurement.time_contract_sha256 != catalog.time_contract_sha256
    ):
        raise ValueError("repair query feedback measurement binding differs")
    return _feedback_from_predicates(
        catalog, request, measurement, _feedback_predicates(source, measurement, request),
    )


def _select_values(
    directory: RepairQueryCatalog,
    feedback: tuple[RepairQueryFeedback, ...],
    submitted: tuple[str, ...],
) -> RepairQuerySelection:
    """No raw source, source store, labels, rank or future measurement is reachable."""
    policy = build_dynamic_verification_policy()
    by_request = {r.request_key: r for r in directory.requests}
    if submitted != tuple(sorted(set(submitted))) or not set(submitted) <= set(by_request):
        raise ValueError("repair query submitted prefix is inconsistent")
    completed = tuple(f.request_key for f in feedback)
    if (
        len(set(completed)) != len(completed)
        or not set(completed) <= set(submitted)
        or len(completed) > policy.case_query_budget
    ):
        raise ValueError("repair query feedback/submission prefix is inconsistent")
    true_conditions: set[str] = set()
    observed_onsets: set[str] = set()
    for item in feedback:
        request = by_request[item.request_key]
        if (
            item.catalog_sha256 != directory.catalog_sha256
            or len(item.predicates) != len(request.needs)
            or len(item.onsets) != len(request.needs)
        ):
            raise ValueError("repair query feedback catalog/needs differ")
        for predicate, need, onset in zip(item.predicates, request.needs, item.onsets, strict=True):
            if (
                predicate.source_sha256 != request.source_sha256
                or predicate.measurement_sha256 != item.measurement_sha256
                or predicate.candidate_service_key != request.candidate_service_key
                or predicate.observation_id != request.observation_id
                or predicate.mechanism_key != need.mechanism_key
                or predicate.predicate_kind != need.predicate_kind
                or predicate.semantics_sha256 != REPAIR_MEASUREMENT_SEMANTICS_SHA256
                or _seal(predicate, "result_sha256") != predicate
                or onset.need_key != need.need_key
                or onset.predicate_sha256 != predicate.result_sha256
            ):
                raise ValueError("repair query feedback predicate binding differs")
            if predicate.truth_value == "true":
                true_conditions.add(need.condition_key)
                if onset.bounds_ns is not None:
                    observed_onsets.add(need.need_key)
    promoted: set[str] = set()
    for pair in directory.time_needs:
        for current, other in (
            (pair.source_need_keys, pair.target_need_keys),
            (pair.target_need_keys, pair.source_need_keys),
        ):
            if set(other) & observed_onsets:
                promoted.update(current)
    paid = {
        c: sum(by_request[key].candidate_service_key == c for key in completed)
        for c in directory.candidate_service_keys
    }
    remaining = policy.case_query_budget - len(completed)
    rows = []
    for request in directory.requests:
        active = tuple(n for n in request.needs if n.queryable)
        tier = min(
            (
                2 if n.condition_key in true_conditions else cast(int, n.initial_tier)
                for n in active
            ),
            default=None,
        )
        if any(n.need_key in promoted for n in active):
            tier = 0
        reasons = []
        if request.request_key in submitted:
            reasons.append("already_submitted")
        if request.candidate_service_key is None:
            reasons.append("request_unavailable")
        if not active:
            reasons.append("no_admissible_demand")
        if remaining == 0:
            reasons.append("query_budget_exhausted")
        rows.append(
            RequestSelectionStatus(
                request_key=request.request_key,
                candidate_service_key=request.candidate_service_key,
                eligible=not reasons,
                demand_tier=tier,
                candidate_paid_count=paid.get(request.candidate_service_key or "", 0),
                unresolved_demand_keys=tuple(sorted({n.condition_key for n in active})),
                stable_tie_sha256=_digest(
                    [
                        policy.tie_seed,
                        request.candidate_service_key,
                        request.canonical_series_key,
                        policy.operator,
                    ]
                ),
                reason_codes=tuple(sorted(reasons))
                if reasons
                else ("observed_counterpart_onset",)
                if tier == 0
                else ("admissible_progress",),
            )
        )
    chosen = min(
        (r for r in rows if r.eligible),
        key=lambda r: cast(tuple[int, int, int, str, str], r.priority),
        default=None,
    )
    return _seal(
        RepairQuerySelection(
            semantics_sha256=REPAIR_QUERY_SEMANTICS_SHA256,
            catalog_sha256=directory.catalog_sha256,
            feedback_sha256s=tuple(f.result_sha256 for f in feedback),
            submitted_request_keys=submitted,
            requests=tuple(rows),
            selected_request_key=None if chosen is None else chosen.request_key,
            paid_count=len(completed),
            remaining_budget=remaining,
            stop_reason="query_budget_exhausted"
            if remaining == 0
            else "no_admissible_progress"
            if chosen is None
            else None,
            result_sha256="0" * 64,
        ),
        "result_sha256",
    )


def select_repair_query(
    *,
    catalog: RepairQueryCatalog,
    expected_catalog_sha256: str,
    feedback: tuple[RepairQueryFeedback, ...],
    expected_feedback_sha256s: tuple[str, ...],
    submitted_request_keys: tuple[str, ...],
) -> RepairQuerySelection:
    """Pure current-version adapter, reusing the existing deterministic priority."""
    originals = (catalog, feedback, expected_feedback_sha256s, submitted_request_keys)
    before = snapshot(originals, _ROOTS)
    try:
        with ExitStack() as guards:
            if (
                type(feedback) is not tuple
                or type(expected_feedback_sha256s) is not tuple
                or type(submitted_request_keys) is not tuple
                or len(feedback) != len(expected_feedback_sha256s)
            ):
                raise ValueError("repair query feedback prefix requires exact complete tuples")
            directory = guards.enter_context(
                _checked(catalog, RepairQueryCatalog, expected_catalog_sha256, "catalog_sha256")
            )
            _validate_catalog(directory)
            observed = tuple(
                guards.enter_context(_checked(row, RepairQueryFeedback, pin, "result_sha256"))
                for row, pin in zip(feedback, expected_feedback_sha256s, strict=True)
            )
            return _select_values(directory, observed, submitted_request_keys)
    finally:
        require_unchanged(originals, _ROOTS, before)
