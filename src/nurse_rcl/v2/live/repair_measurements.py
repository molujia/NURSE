"""Current live measurements and source-replayed, mechanism-bound scalar leaves.

The episode arithmetic is shared with the legacy verifier, not its source or
quality identity. A leaf is not a full hypothesis guard or scoring authority.
Case-wide completeness, query reservations and scoring remain enclosing duties.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Literal, NamedTuple, cast

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.contracts import (
    SHA256,
    SignalEpisode,
    SignalEpisodePoint,
    TruthValue,
    _SourceModel,
)
from nurse_rcl.v2.dynamic_verification.measurement import _classify_point, _episodes, _gap_reasons
from nurse_rcl.v2.dynamic_verification.requests import build_dynamic_verification_policy
from nurse_rcl.v2.dynamic_verification.safety import (
    canonical,
    plain,
    preflight,
    require_pin,
    require_unchanged,
    snapshot,
)
from nurse_rcl.v2.dynamic_verification.source_store import (
    SourceBlockStore,
)
from nurse_rcl.v2.hypothesis_guard import _PREDICATE_EVIDENCE_CONTRACTS
from nurse_rcl.v2.live.repair_lifecycle import _AVAILABILITY_MEANINGS, LifecycleObservation
from nurse_rcl.v2.live.repair_observations import (
    REPAIR_QUALITY_SEMANTICS_SHA256,
    RepairObservationQualityResult,
)
from nurse_rcl.v2.live.repair_sources import (
    RepairMeasurementSource,
    _replayed_repair_measurement_source,
)
from nurse_rcl.v2.mechanism_library import build_mechanism_library
from nurse_rcl.v2.observation_schemas import CaseTimeContract, TimeInterval

REPAIR_MEASUREMENT_SCHEMA_VERSION: Literal["nurse.v2.live-repair-measurement/1"] = (
    "nurse.v2.live-repair-measurement/1"
)
REPAIR_MEASURED_PREDICATE_SCHEMA_VERSION: Literal["nurse.v2.live-repair-measured-predicate/1"] = (
    "nurse.v2.live-repair-measured-predicate/1"
)
# Availability/capacity are deliberately absent: family membership is not a
# license to interpret every memory gauge's increase as resource pressure.
_PRESSURE_METRICS = {
    "rcabench": (
        "container.cpu.usage",
        "k8s.pod.cpu.usage",
        "jvm.cpu.recent_utilization",
        "jvm.system.cpu.utilization",
        "jvm.system.cpu.load_1m",
        "k8s.pod.cpu.node.utilization",
        "k8s.pod.cpu_limit_utilization",
        "container.memory.rss",
        "container.memory.usage",
        "container.memory.working_set",
        "k8s.pod.memory.rss",
        "k8s.pod.memory.usage",
        "k8s.pod.memory.working_set",
        "k8s.pod.memory.node.utilization",
        "k8s.pod.memory_limit_utilization",
    ),
    "aiops25": ("pod_cpu_usage", "pod_memory_working_set_bytes"),
}
_UP_PREDICATES = (
    "local_pressure_observed",
    "request_latency_increase",
    "local_error_increase",
    "request_failure_increase",
    "incoming_request_rate_increase",
    "outgoing_dependency_failure_observed",
)
_NORMAL_PREDICATES = (
    "local_signals_verified_normal",
    "local_resource_and_execution_verified_normal",
    "local_storage_and_network_verified_normal",
    "incoming_request_rate_verified_normal",
    "outgoing_dependency_calls_verified_normal",
)
REPAIR_MEASUREMENT_SEMANTICS_SHA256 = sha256_hex(
    canonical_json_bytes(
        {
            "version": "nurse-v2-live-repair-directional-consumption/1",
            "quality": REPAIR_QUALITY_SEMANTICS_SHA256,
            "numeric_policy": build_dynamic_verification_policy().policy_sha256,
            "pressure_metrics": _PRESSURE_METRICS,
            "up_predicates": _UP_PREDICATES,
            "normal_predicates": _NORMAL_PREDICATES,
            "general": "eligible_summary_or_persistent_deviation_or_eligible_lifecycle_event",
            "normal": "complete_observed_points_within_band_no_lifecycle_zero_or_invalid_state",
            "lifecycle": "registered_observed_zero_existence_not_continuous_pressure",
            "authorization": "source_replay_candidate_mechanism_family_unit_field_scope",
        }
    )
)


class RepairSignalMeasurement(_SourceModel):
    schema_version: Literal["nurse.v2.live-repair-measurement/1"] = (
        REPAIR_MEASUREMENT_SCHEMA_VERSION
    )
    semantics_sha256: SHA256
    source_sha256: SHA256
    time_contract_sha256: SHA256
    request_sha256: SHA256
    canonical_series_key: str
    interval: TimeInterval
    quality: RepairObservationQualityResult
    lifecycle: LifecycleObservation | None
    points: tuple[SignalEpisodePoint, ...]
    episodes: tuple[SignalEpisode, ...]
    persistent_up: TruthValue
    persistent_down: TruthValue
    persistent_deviation_present: TruthValue
    general_deviation_present: TruthValue
    verified_normal: TruthValue
    reason_codes: tuple[str, ...]
    measurement_sha256: SHA256


class RepairMeasuredPredicate(_SourceModel):
    schema_version: Literal["nurse.v2.live-repair-measured-predicate/1"] = (
        REPAIR_MEASURED_PREDICATE_SCHEMA_VERSION
    )
    semantics_sha256: SHA256
    source_sha256: SHA256
    measurement_sha256: SHA256
    candidate_service_key: str
    mechanism_key: str
    mechanism_definition_sha256: SHA256
    predicate_kind: str
    observation_id: str
    support_source_key: str
    measurement_field: str
    unit: str
    interval: TimeInterval
    truth_value: TruthValue
    disposition: Literal["usable", "unknown"]
    reason_codes: tuple[str, ...]
    score_authority: Literal[False] = False
    result_sha256: SHA256


_ROOTS = (
    RepairMeasurementSource,
    RepairSignalMeasurement,
    RepairMeasuredPredicate,
    CaseTimeContract,
)


@contextmanager
def _source_replay(
    *,
    source: RepairMeasurementSource,
    expected_source_sha256: str,
    time_contract: CaseTimeContract,
    expected_time_contract_sha256: str,
    store: SourceBlockStore,
    extra: tuple[RepairSignalMeasurement, ...] = (),
) -> Iterator[RepairMeasurementSource]:
    originals = (source, time_contract, extra)
    # Preserve the former combined capacity, without copying source/time twice.
    preflight(originals, _ROOTS)
    before = snapshot(extra, _ROOTS)
    with ExitStack() as guards:
        guards.callback(preflight, originals, _ROOTS)
        guards.callback(require_unchanged, extra, _ROOTS, before)
        fresh = guards.enter_context(
            _replayed_repair_measurement_source(
                source=source,
                expected_source_sha256=expected_source_sha256,
                time_contract=time_contract,
                expected_time_contract_sha256=expected_time_contract_sha256,
                store=store,
            )
        )
        yield fresh


class _NumericIdentity(NamedTuple):
    canonical_series_key: str
    request_sha256: str


def _numeric_identity(source: RepairMeasurementSource) -> _NumericIdentity:
    # A pure identity, not a legacy request/catalog masquerading as the new one.
    series = "series_" + sha256_hex(canonical_json_bytes([source.dataset, source.signal_id]))
    return _NumericIdentity(
        series,
        sha256_hex(
            canonical_json_bytes(
                {
                    "version": "nurse-v2-live-repair-measure/1",
                    "source_sha256": source.source_sha256,
                    "semantics_sha256": REPAIR_MEASUREMENT_SEMANTICS_SHA256,
                }
            )
        ),
    )


def _measure_values(source: RepairMeasurementSource) -> RepairSignalMeasurement:
    """Pure numeric body; caller must replay sources and protect both exits."""
    policy = build_dynamic_verification_policy()
    identity = _numeric_identity(source)
    quality = source.quality
    summary = quality.source_summary
    component = next(
        c for c in source.transform_result.windows[1].components if c.name == summary.component_name
    )
    cadence = source.time_contract.expected_cadence_ns
    blocked: set[str] = set()
    if component.aggregation in {"window_count", "exposure_rate"}:
        blocked.add("temporal_resolution_unavailable")
    if not quality.formal_scoring_eligible:
        blocked.add("quality_ineligible")
    if quality.baseline_scale is None or summary.baseline_summary.median is None:
        blocked.add("baseline_scale_unavailable")
    if summary.baseline_summary.coverage_fraction < summary.configuration.minimum_coverage_fraction:
        blocked.add("insufficient_baseline_coverage")
    points: list[SignalEpisodePoint] = []
    reasons = set(quality.quality_flags) | blocked
    for index, sample in enumerate(component.samples):
        z, state = (
            (None, "unknown")
            if blocked
            else _classify_point(
                sample.value,
                summary.baseline_summary.median,
                quality.baseline_scale,
                summary.configuration.abnormal_z_threshold,
            )
        )
        point_reasons = blocked.copy()
        if not blocked and state == "unknown":
            point_reasons.add("non_finite_point_deviation")
        point = SignalEpisodePoint(
            sample_index=index,
            timestamp_unix_ns=sample.timestamp_unix_ns,
            source_interval=sample.source_interval,
            value=sample.value,
            signed_z=z,
            state=state,
            reason_codes=tuple(sorted(point_reasons)),
        )
        if points and (gaps := _gap_reasons(points[-1], point, policy, cadence)):
            reasons.update(gaps)
            point_reasons.update(gap + "_before" for gap in gaps)
            point = point.model_copy(update={"reason_codes": tuple(sorted(point_reasons))})
        points.append(point)
        reasons.update(point_reasons)
    episodes = _episodes(points=tuple(points), request=identity, policy=policy, cadence=cadence)
    complete = (
        bool(points)
        and not blocked
        and all(p.state != "unknown" for p in points)
        and (
            summary.fault_summary_sufficient
            and summary.fault_summary.coverage_fraction
            >= summary.configuration.minimum_coverage_fraction
        )
        and not {"sampling_gap", "source_interval_gap"}.intersection(reasons)
    )
    default: TruthValue = "false" if complete else "unknown"
    up: TruthValue = (
        "true" if any(e.qualifying and e.direction == "up" for e in episodes) else default
    )
    down: TruthValue = (
        "true" if any(e.qualifying and e.direction == "down" for e in episodes) else default
    )
    deviation: TruthValue = "true" if "true" in (up, down) else default
    event = source.lifecycle
    summary_abnormal = (
        quality.formal_scoring_eligible
        and quality.robust_z is not None
        and (quality.robust_z >= summary.configuration.abnormal_z_threshold)
    )
    general: TruthValue = (
        "true"
        if (summary_abnormal or deviation == "true" or (event and event.formal_event_eligible))
        else default
    )
    normal: TruthValue = "unknown"
    if complete:
        normal = "true" if all(p.state == "within_band" for p in points) else "false"
    if event is not None and (event.observed_zero or event.invalid_state_timestamps_ns):
        normal = "false" if event.formal_event_eligible else "unknown"
    draft = RepairSignalMeasurement(
        semantics_sha256=REPAIR_MEASUREMENT_SEMANTICS_SHA256,
        source_sha256=source.source_sha256,
        time_contract_sha256=sha256_hex(canonical(source.time_contract, _ROOTS)),
        request_sha256=identity.request_sha256,
        canonical_series_key=identity.canonical_series_key,
        interval=source.query_interval,
        quality=quality,
        lifecycle=event,
        points=tuple(points),
        episodes=episodes,
        persistent_up=up,
        persistent_down=down,
        persistent_deviation_present=deviation,
        general_deviation_present=general,
        verified_normal=normal,
        reason_codes=tuple(sorted(reasons)),
        measurement_sha256="0" * 64,
    )
    unsigned = cast(dict[str, object], plain(draft))
    unsigned.pop("measurement_sha256")
    return draft.model_copy(
        update={"measurement_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )


def measure_repair_signal(
    *,
    source: RepairMeasurementSource,
    expected_source_sha256: str,
    time_contract: CaseTimeContract,
    expected_time_contract_sha256: str,
    store: SourceBlockStore,
) -> RepairSignalMeasurement:
    """One source measurement; enclosing session reserves it against the case budget."""
    with _source_replay(
        source=source,
        expected_source_sha256=expected_source_sha256,
        time_contract=time_contract,
        expected_time_contract_sha256=expected_time_contract_sha256,
        store=store,
    ) as fresh:
        return _measure_values(fresh)


def _predicate_values(
    source: RepairMeasurementSource,
    measurement: RepairSignalMeasurement,
    candidate_service_key: str,
    mechanism_key: str,
    predicate_kind: str,
) -> RepairMeasuredPredicate:
    if any(
        type(value) is not str for value in (candidate_service_key, mechanism_key, predicate_kind)
    ):
        raise ValueError("repair predicate binding requires exact text identities")
    if candidate_service_key != source.candidate_service_key:
        raise ValueError("repair measurement candidate binding mismatch")
    try:
        definition = build_mechanism_library().lookup(mechanism_key)
        contract = _PREDICATE_EVIDENCE_CONTRACTS[predicate_kind]
    except KeyError as error:
        raise ValueError("repair predicate mechanism is not registered") from error
    if predicate_kind not in (
        *definition.necessary_predicate_kinds,
        *definition.optional_predicate_kinds,
        *definition.refuting_predicate_kinds,
    ):
        raise ValueError("repair predicate is outside its registered mechanism")
    observation = source.observation
    if observation.signal_family not in (
        *contract.primary_signal_families,
        *contract.source_signal_families,
    ):
        raise ValueError("repair predicate signal family denied")
    truth: TruthValue = "unknown"
    field = "reason_codes"
    reasons = []
    event = measurement.lifecycle
    if predicate_kind in _UP_PREDICATES:
        field = "persistent_up"
        if (
            predicate_kind == "local_pressure_observed"
            and source.metric_name not in _PRESSURE_METRICS[source.dataset]
        ):
            reasons.append("pressure_metric_semantics_unavailable")
        elif predicate_kind == "request_latency_increase" and (
            observation.signal_type != "latency" or observation.unit != "ns"
        ):
            reasons.append("latency_semantics_unavailable")
        elif predicate_kind == "incoming_request_rate_increase" and (
            observation.aggregation not in {"per_second_rate", "exposure_rate"}
            or not observation.unit.endswith("/second")
        ):
            reasons.append("rate_semantics_unavailable")
        elif predicate_kind in {"request_failure_increase", "outgoing_dependency_failure_observed"}:
            reasons.append("request_outcome_semantics_unavailable")
        else:
            truth = measurement.persistent_up
    elif predicate_kind == "local_anomaly_observed":
        field, truth = "general_deviation_present", measurement.general_deviation_present
    elif predicate_kind in _NORMAL_PREDICATES:
        field, truth = "verified_normal", measurement.verified_normal
    elif predicate_kind in {
        "workload_unavailability_observed",
        "workload_availability_verified_normal",
    }:
        field = "lifecycle.availability_support"
        if event is None or event.state_meaning not in _AVAILABILITY_MEANINGS:
            reasons.append("availability_semantics_unavailable")
        elif predicate_kind == "workload_unavailability_observed":
            truth = (
                "true"
                if event.availability_support
                else ("false" if measurement.verified_normal == "true" else "unknown")
            )
        else:
            field = "verified_normal"
            truth = "false" if event.availability_support else measurement.verified_normal
    elif predicate_kind == "lifecycle_transition_observed":
        field = "lifecycle.events"
        if (
            event
            and event.formal_event_eligible
            and any(
                e.onset_bounds_ns is not None or e.recovery_bounds_ns is not None
                for e in event.events
            )
        ):
            truth = "true"
        else:
            reasons.append("observed_transition_bracket_unavailable")
    else:
        reasons.append("predicate_requires_distinct_measurement_semantics")
    reasons.extend(measurement.reason_codes)
    reasons.append("measured_predicate:" + truth)
    support = "support_" + sha256_hex(
        canonical_json_bytes(
            {
                "candidate_service_key": candidate_service_key,
                "source_fingerprint": observation.source_fingerprint,
                "adapter_semantics_sha256": observation.adapter_semantics_sha256,
            }
        )
    )
    draft = RepairMeasuredPredicate(
        semantics_sha256=REPAIR_MEASUREMENT_SEMANTICS_SHA256,
        source_sha256=source.source_sha256,
        measurement_sha256=measurement.measurement_sha256,
        candidate_service_key=candidate_service_key,
        mechanism_key=mechanism_key,
        mechanism_definition_sha256=definition.definition_sha256,
        predicate_kind=predicate_kind,
        observation_id=observation.observation_id,
        support_source_key=support,
        measurement_field=field,
        unit=observation.unit,
        interval=measurement.interval,
        truth_value=truth,
        disposition="unknown" if truth == "unknown" else "usable",
        reason_codes=tuple(sorted(set(reasons))),
        result_sha256="0" * 64,
    )
    unsigned = cast(dict[str, object], plain(draft))
    unsigned.pop("result_sha256")
    return draft.model_copy(update={"result_sha256": sha256_hex(canonical_json_bytes(unsigned))})


def consume_repair_measurement(
    *,
    source: RepairMeasurementSource,
    expected_source_sha256: str,
    time_contract: CaseTimeContract,
    expected_time_contract_sha256: str,
    store: SourceBlockStore,
    measurement: RepairSignalMeasurement,
    expected_measurement_sha256: str,
    candidate_service_key: str,
    mechanism_key: str,
    predicate_kind: str,
) -> RepairMeasuredPredicate:
    """Recompute actual B/F evidence before compiling one permitted predicate leaf.

    This does not authorize a role or score. The later complete rule compiler
    must preserve all applicable leaves, three-valued composition and guards.
    """
    if type(measurement) is not RepairSignalMeasurement:
        raise ValueError("repair consumption requires the exact current measurement")
    require_pin(expected_measurement_sha256)
    with _source_replay(
        source=source,
        expected_source_sha256=expected_source_sha256,
        time_contract=time_contract,
        expected_time_contract_sha256=expected_time_contract_sha256,
        store=store,
        extra=(measurement,),
    ) as fresh:
        rebuilt = _measure_values(fresh)
        before = snapshot(rebuilt, _ROOTS)
        try:
            if rebuilt.measurement_sha256 != expected_measurement_sha256 or (
                canonical(rebuilt, _ROOTS) != canonical(measurement, _ROOTS)
            ):
                raise ValueError("repair measurement differs from independently replayed sources")
            return _predicate_values(
                fresh, rebuilt, candidate_service_key, mechanism_key, predicate_kind
            )
        finally:
            require_unchanged(rebuilt, _ROOTS, before)
