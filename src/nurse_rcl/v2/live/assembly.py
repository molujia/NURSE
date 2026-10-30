"""Dataset-specific values → V2 transformations, quality, entities, full C and Q.

No labels, test fixtures, model proposals, inferred histogram buckets or
unverified counter resets participate. Undefined units/temporalities stay unknown.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pyarrow as pa  # type: ignore[import-untyped]

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse, freeze_candidate_universe
from nurse_rcl.v2.candidate_universe import build_complete_candidate_universe
from nurse_rcl.v2.event_grouping import (
    EventGroupingResult, PrimarySignalFamily, build_event_grouping_result, build_frozen_signal_family_mapping,
    build_signal_family_mapping_entry,
)
from nurse_rcl.v2.focus_selection import FocusSelectionResult, build_neutral_focus_selection
from nurse_rcl.v2.live.sources import RCABenchLiveSources, read_rcabench_live_sources
from nurse_rcl.v2.live.aiops25_sources import AIOps25LiveSources, read_aiops25_live_sources
from nurse_rcl.v2.observation_quality import (
    AbsoluteResolutionDeclaration, ObservationQualityContext, ObservationQualityResult,
    summarize_transformed_component,
)
from nurse_rcl.v2.observation_schemas import CandidateUniverse, Observation, SignalType, TelemetryModality, TimeInterval
from nurse_rcl.v2.signal_transforms import (
    ScalarSample, ScalarWindowInput, SignalTransformResult,
    transform_gauge, transform_latency, transform_unknown,
)
from nurse_rcl.v2.temporal_entity_registry import (
    ObservationEntityInput, TemporalEntityRegistry,
    build_entity_metadata_record, build_temporal_entity_registry,
)

VERSION = 'rcabench-live-observations/2'
# Exact published names with known instantaneous semantics. Unknown metric
# names (including all sum/histogram temporalities absent from the export) are
# retained as unknown. Native units are explicitly NOT a physical conversion.
_GAUGES = {
    'container.cpu.usage': ('cpu', 'native_value', None),
    'k8s.pod.cpu.usage': ('cpu', 'native_value', None),
    'jvm.cpu.recent_utilization': ('cpu', 'ratio', None),
    'jvm.system.cpu.utilization': ('cpu', 'ratio', None),
    'jvm.system.cpu.load_1m': ('cpu', 'native_value', None),
    'k8s.container.ready': ('lifecycle', 'count', 1.0),
    'k8s.deployment.available': ('lifecycle', 'count', 1.0),
    'k8s.deployment.desired': ('lifecycle', 'count', 1.0),
    'k8s.replicaset.available': ('lifecycle', 'count', 1.0),
    'k8s.replicaset.desired': ('lifecycle', 'count', 1.0),
    'k8s.statefulset.current_pods': ('lifecycle', 'count', 1.0),
    'k8s.statefulset.desired_pods': ('lifecycle', 'count', 1.0),
    'k8s.statefulset.ready_pods': ('lifecycle', 'count', 1.0),
    'k8s.statefulset.updated_pods': ('lifecycle', 'count', 1.0),
}
for _prefix in ('container', 'k8s.pod'):
    for _suffix in ('available', 'rss', 'usage', 'working_set'):
        _GAUGES[f'{_prefix}.memory.{_suffix}'] = ('memory', 'native_value', None)
    for _suffix in ('available', 'capacity', 'usage'):
        _GAUGES[f'{_prefix}.filesystem.{_suffix}'] = ('storage', 'native_value', None)
for _name in ('k8s.pod.cpu.node.utilization', 'k8s.pod.cpu_limit_utilization'):
    _GAUGES[_name] = ('cpu', 'ratio', None)
for _name in ('k8s.pod.memory.node.utilization', 'k8s.pod.memory_limit_utilization'):
    _GAUGES[_name] = ('memory', 'ratio', None)
SEMANTICS = dict(version=VERSION, gauges=_GAUGES, histogram='unknown_temporality_no_buckets',
                 sum='unknown_temporality', duplicate_conflict='unknown',
                 trace_duration_unit='ns', trace_aggregation='per_cadence_bin_p95_nearest_rank',
                 log_statistic='per_observed_cadence_bin_error_fraction',
                 missing_bin='no_record_not_zero', entity='explicit_service_only',
                 excluded_collectors=['kubernetes-endpoints'], raw_log_text='never_projected')
SEMANTICS_SHA256 = sha256_hex(canonical_json_bytes(SEMANTICS))

AIOPS25_VERSION = 'aiops25-live-observations/1'
# Exact exported instantaneous indicators only. Network/fs/count temporalities
# and rrt units are not specified by the preparer, so they remain unknown.
_AIOPS25_GAUGES = {
    'pod_cpu_usage': ('cpu', 'native_value', None),
    'pod_memory_working_set_bytes': ('memory', 'native_value', None),
    'pod_processes': ('lifecycle', 'count', 1.0),
    'error_ratio': ('error', 'ratio', None),
    'client_error_ratio': ('error', 'ratio', None),
    'server_error_ratio': ('error', 'ratio', None),
}
AIOPS25_SEMANTICS = dict(SEMANTICS, version=AIOPS25_VERSION, gauges=_AIOPS25_GAUGES,
    unregistered_metrics='unknown_unit_or_temporality',
    trace_duration_basis='mandated_preprocessor_microseconds_times_1000')
AIOPS25_SEMANTICS_SHA256 = sha256_hex(canonical_json_bytes(AIOPS25_SEMANTICS))


def _digest(value: Any) -> str:
    return sha256_hex(canonical_json_bytes(value))


def service_key(raw: str) -> str:
    return 'svc_' + _digest(['service', raw])[:24]


@dataclass(frozen=True)
class LiveSignal:
    signal_id: str
    service: str | None
    observed_entity_key: str
    modality: str
    signal_family: str
    component: str
    baseline: ScalarWindowInput
    fault: ScalarWindowInput
    transform: SignalTransformResult
    quality: ObservationQualityResult
    observation: Observation
    source_file_sha256s: tuple[str, ...]
    flags: tuple[str, ...]


@dataclass(frozen=True)
class LiveInputs:
    sources: RCABenchLiveSources | AIOps25LiveSources = field(repr=False)
    signals: tuple[LiveSignal, ...]
    entity_registry: TemporalEntityRegistry
    event_grouping: EventGroupingResult
    candidate_universe: CandidateUniverse
    frozen_candidates: FrozenCandidateUniverse
    focus: FocusSelectionResult
    scope: TimeInterval
    source_accounting: dict[str, int]
    q_base: int
    q_max: int

    def inference_descriptor(self) -> dict[str, Any]:
        aiops = type(self.sources) is AIOps25LiveSources
        source_descriptor = self.sources.descriptor()
        if aiops:
            # The prepared adapter contains a PRIVATE raw service directory.
            # Bind its full identity but never expose that directory as input.
            source_descriptor = dict(
                schema_version='nurse.v2.aiops25-live-source-projection/1',
                private_descriptor_sha256=source_descriptor['descriptor_sha256'],
                adapter_result_sha256=self.sources.adapter.result_sha256,
                time_contract=self.sources.adapter.time_contract.model_dump(mode='json'),
                tables=source_descriptor['tables'], rejections=source_descriptor['rejections'],
            )
        return dict(schema_version=('nurse.v2.aiops25-live-inputs/1' if aiops else 'nurse.v2.live-inputs/1'),
                    semantics_sha256=AIOPS25_SEMANTICS_SHA256 if aiops else SEMANTICS_SHA256,
                    source_descriptor=source_descriptor,
                    retained_signals_sha256=_digest([
                        dict(signal_id=s.signal_id, service=s.service, entity=s.observed_entity_key,
                             modality=s.modality, family=s.signal_family, component=s.component,
                             baseline=s.baseline.model_dump(mode='json'), fault=s.fault.model_dump(mode='json'),
                             transform=s.transform.model_dump(mode='json'), quality=s.quality.model_dump(mode='json'),
                             observation=s.observation.model_dump(mode='json'),
                             source_pins=s.source_file_sha256s, flags=s.flags) for s in self.signals]),
                    scope=self.scope.model_dump(mode='json'), q_base=self.q_base, q_max=self.q_max,
                    entity_registry=self.entity_registry.model_dump(mode='json'),
                    event_grouping=self.event_grouping.model_dump(mode='json'),
                    candidate_universe=self.candidate_universe.model_dump(mode='json'),
                    frozen_candidates=self.frozen_candidates.model_dump(mode='json'),
                    focus=self.focus.model_dump(mode='json'), source_accounting=self.source_accounting)

    def verify_sources(self, root: Path) -> bool:
        if not (
            (type(self) is LiveInputs and type(self.sources) is RCABenchLiveSources)
            or (type(self) is AIOps25LiveInputs and type(self.sources) is AIOps25LiveSources)
        ):
            return False
        try:
            rebuilt: LiveInputs
            if isinstance(self.sources, AIOps25LiveSources):
                aiops_sources = read_aiops25_live_sources(root,
                    source_case_id=self.sources.source_case_id,
                    opaque_case_id=self.sources.adapter.opaque_case_id,
                    authorized_queue=self.sources.authorized_queue,
                    configuration=self.sources.adapter.configuration)
                rebuilt = assemble_aiops25_live_inputs(aiops_sources, q_base=self.q_base, q_max=self.q_max)
            else:
                sources = read_rcabench_live_sources(root, opaque_case_id=self.sources.adapter.opaque_case_id,
                                                     configuration=self.sources.adapter.configuration)
                rebuilt = assemble_rcabench_live_inputs(sources, q_base=self.q_base, q_max=self.q_max)
            return rebuilt.inference_descriptor() == self.inference_descriptor()
        except (ValueError, OSError, pa.ArrowException):
            return False


@dataclass(frozen=True)
class AIOps25LiveInputs(LiveInputs):
    """Explicit new dataset result; legacy exact-type consumers still refuse it."""


def _identity(row: dict[str, Any]) -> tuple[str | None, str]:
    explicit = row.get('service_name') or row.get('attr.k8s.service.name')
    if type(explicit) is str and explicit.strip() and explicit != 'kubernetes-endpoints':
        key = service_key(explicit.strip())
        return key, key
    # Do not pretend node/collector/workload names are proven service bindings.
    fields = {k: v for k, v in row.items() if k.startswith('attr.k8s.') and type(v) is str and v}
    return None, 'unbound_' + _digest([explicit if type(explicit) is str else None, fields])[:24]


def assemble_rcabench_live_inputs(sources: RCABenchLiveSources, *, q_base: int = 8,
                                 q_max: int = 12) -> LiveInputs:
    if type(sources) is not RCABenchLiveSources or not sources.adapter.verify():
        raise ValueError('invalid source adapter')
    return _assemble_live_inputs(sources, q_base=q_base, q_max=q_max)


def assemble_aiops25_live_inputs(sources: AIOps25LiveSources, *, q_base: int = 8,
                               q_max: int = 12) -> AIOps25LiveInputs:
    if type(sources) is not AIOps25LiveSources or not sources.adapter.verify():
        raise ValueError('invalid source adapter')
    return cast(AIOps25LiveInputs, _assemble_live_inputs(sources, q_base=q_base, q_max=q_max))


def _assemble_live_inputs(sources: RCABenchLiveSources | AIOps25LiveSources, *,
                          q_base: int, q_max: int) -> LiveInputs:
    aiops = type(sources) is AIOps25LiveSources
    gauges = _AIOPS25_GAUGES if aiops else _GAUGES
    version = AIOPS25_VERSION if aiops else VERSION
    semantics_sha256 = AIOPS25_SEMANTICS_SHA256 if aiops else SEMANTICS_SHA256
    contract = sources.adapter.time_contract
    scope = TimeInterval(start_unix_ns=contract.baseline_window.start_unix_ns,
                         end_unix_ns=contract.decision_time_unix_ns)
    intervals = {'baseline': contract.baseline_window, 'fault': contract.fault_window}
    # Each complete series retains its exact entity/attribute identity. No top-N
    # series pruning, no target-based choice, and no train/test window splitting.
    groups: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    entity_sources: dict[tuple[str | None, str, TelemetryModality], set[str]] = defaultdict(set)

    def add(*, identity: Any, service: str | None, entity: str, modality: str, family: str,
            component: str, mode: str, unit: str, resolution: float | None,
            timestamp: int, value: Any, window: str, source: str, aggregation: str = 'instantaneous') -> None:
        sid = 'sig_' + _digest(identity)[:24]
        group = groups.setdefault(sid, dict(service=service, entity=entity, modality=modality,
            family=family, component=component, mode=mode, unit=unit, resolution=resolution,
            aggregation=aggregation, source=set(), flags=set(), values={'baseline': defaultdict(list), 'fault': defaultdict(list)}))
        group['source'].add(source)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            counts['invalid_numeric_values'] += 1
            group['flags'].add('invalid_numeric_values')
            return
        group['values'][window][timestamp].append(float(value))

    for item in sources.tables:
        window, kind = item.audit.window, item.audit.source_kind
        table = item.table.set_column(0, 'time', item.table.column('time').cast(pa.timestamp('ns', tz='UTC')).cast(pa.int64()))
        log_bins: dict[tuple[str | None, str, int], list[float]] = defaultdict(list)
        for row in table.to_pylist():
            counts['source_rows'] += 1
            t = row['time']
            service, entity = _identity(row)
            duration = row.get('duration')
            if kind == 'traces' and isinstance(duration, (int, float)) and not isinstance(duration, bool):
                if duration >= 0 and t + duration > contract.decision_time_unix_ns:
                    counts['trace_unfinished_at_decision'] += 1
                    continue
            entity_sources[(service, entity, item.audit.modality)].add(item.audit.source_sha256)
            if kind.startswith('metrics'):
                metric = row.get('metric')
                # Exact standard-name allowlist only; arbitrary telemetry names
                # cannot choose an inference family by embedding words like CPU.
                known = gauges.get(metric) if kind == 'metrics' and type(metric) is str else None
                family, unit, resolution = known or ('unclassified', 'native_value', None)
                attributes = {k: v for k, v in row.items() if k.startswith('attr.') and v is not None}
                for component in (('count', 'sum', 'min', 'max') if kind == 'metrics_histogram' else ('value',)):
                    add(identity=[kind, entity, metric, attributes, component], service=service, entity=entity,
                        modality='metrics', family=family, component=component,
                        mode='gauge' if known else 'unknown', unit=unit, resolution=resolution,
                        timestamp=t, value=row.get(component), window=window, source=item.audit.source_sha256)
            elif kind == 'traces':
                bucket = intervals[window].start_unix_ns + ((t - intervals[window].start_unix_ns)
                    // contract.expected_cadence_ns) * contract.expected_cadence_ns
                add(identity=['traces', entity, 'duration'], service=service, entity=entity,
                    modality='traces', family='latency', component='duration', mode='latency', unit='ns',
                    resolution=1.0, timestamp=bucket, value=duration, window=window,
                    source=item.audit.source_sha256, aggregation='p95')
            else:
                level = row.get('level')
                if type(level) is not str or level.upper() not in {'TRACE', 'DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR', 'FATAL', 'CRITICAL'}:
                    counts['unknown_log_severity'] += 1
                    continue
                bucket = intervals[window].start_unix_ns + ((t - intervals[window].start_unix_ns)
                    // contract.expected_cadence_ns) * contract.expected_cadence_ns
                log_bins[(service, entity, bucket)].append(float(level.upper() in {'ERROR', 'FATAL', 'CRITICAL'}))
        for (service, entity, t), values in log_bins.items():
            add(identity=['logs', entity, 'severity_fraction'], service=service, entity=entity,
                modality='logs', family='error', component='observed_log_error_fraction', mode='gauge',
                unit='ratio', resolution=None, timestamp=t, value=sum(values) / len(values),
                window=window, source=item.audit.source_sha256)

    # Identity-only records remain candidates even with no measurable signal.
    metadata = [build_entity_metadata_record(
        opaque_case_id=contract.opaque_case_id, metadata_source=modality,
        entity_kind='service' if service is not None else 'untyped', observed_entity_key=entity,
        relation='explicit_service_binding' if service is not None else 'collector_source',
        service_entity_key=service, valid_interval=scope, source_fingerprint=_digest(sorted(pins)),
        provenance_ids=tuple('source:' + pin for pin in sorted(pins)),
    ) for (service, entity, modality), pins in sorted(entity_sources.items(), key=lambda pair: (pair[0][1], pair[0][2]))]
    signals = []
    for sid, group in sorted(groups.items()):
        windows = {}
        for window_name, interval in intervals.items():
            samples = []
            for timestamp, values in sorted(group['values'][window_name].items()):
                if group['mode'] == 'latency':
                    # All observed spans in this cadence bin contribute. The
                    # nearest-rank p95 is an actual observed duration, not an
                    # invented histogram bucket or duplicate-gauge average.
                    value = sorted(values)[math.ceil(0.95 * len(values)) - 1]
                elif len(set(values)) != 1:
                    group['flags'].add('conflicting_timestamps')
                    counts['conflicting_timestamps'] += 1
                    continue
                else:
                    value = values[0]
                samples.append(ScalarSample(timestamp_unix_ns=timestamp, value=value))
            windows[window_name] = ScalarWindowInput(interval=interval, aggregation=group['aggregation'], samples=tuple(samples))
        mode: SignalType = group['mode'] if not group['flags'] else 'unknown'
        transform_args = dict(unit=group['unit'], baseline=windows['baseline'], fault=windows['fault'])
        transform = (transform_gauge(**transform_args) if mode == 'gauge' else
                     transform_latency(**transform_args) if mode == 'latency' else
                     transform_unknown(declared_signal_type='unresolved_export_semantics', **transform_args))
        component = 'latency' if mode == 'latency' else 'value'
        declaration = None
        if mode != 'unknown' and group['resolution'] is not None:
            declaration = AbsoluteResolutionDeclaration(signal_type=mode, component_name=component,
                unit=group['unit'], aggregation=group['aggregation'], value=group['resolution'],
                basis='unit_manifest', manifest_id=version, manifest_version='1',
                provenance_id='semantics:' + semantics_sha256)
        context = ObservationQualityContext(expected_cadence_ns=contract.expected_cadence_ns,
            binding_status='bound' if group['service'] else 'unbound', baseline_contamination='none',
            clock_uncertainty_ns=contract.clock_uncertainty_ns, late_data_observed=contract.late_data_observed,
            adapter_manifest_id=version, adapter_manifest_version='1', absolute_resolution=declaration)
        quality = summarize_transformed_component(transform_result=transform, component_name=component, context=context)
        source_pins = tuple(sorted(group['source']))
        fingerprint = _digest([sid, source_pins, transform.source_input_sha256])
        observation = Observation(observation_id='obs_' + _digest([sid, quality.result_sha256])[:24],
            opaque_case_id=contract.opaque_case_id, modality=group['modality'], signal_family=group['family'],
            signal_type=transform.signal_type, unit=quality.unit, interval=scope,
            decision_time_unix_ns=contract.decision_time_unix_ns, aggregation=quality.aggregation,
            source_fingerprint=fingerprint, baseline_summary=quality.baseline_summary,
            fault_summary=quality.fault_summary,
            evidence_state='unknown' if transform.signal_type == 'unknown' else quality.evidence_state,
            anomaly_strength=quality.anomaly_strength, coverage_quality=quality.coverage_quality,
            minimum_valid_points=quality.minimum_valid_points, transform=transform.transform,
            quality_flags=tuple(sorted(set(quality.quality_flags) | group['flags'])),
            provenance_ids=tuple(sorted(('source:' + pin for pin in source_pins))),
            formal_scoring_eligible=quality.formal_scoring_eligible, adapter_semantics_sha256=semantics_sha256)
        signals.append(LiveSignal(sid, group['service'], group['entity'], group['modality'], group['family'],
            group['component'], windows['baseline'], windows['fault'], transform, quality, observation,
            source_pins, tuple(sorted(group['flags']))))
    registry = build_temporal_entity_registry(opaque_case_id=contract.opaque_case_id,
        decision_time_unix_ns=contract.decision_time_unix_ns, metadata_records=metadata,
        observations=[ObservationEntityInput(observation=s.observation,
            entity_kind='service' if s.service else 'untyped', observed_entity_key=s.observed_entity_key) for s in signals])
    mapping = build_frozen_signal_family_mapping(mapping_version=version, entries=[
        build_signal_family_mapping_entry(adapter_semantics_sha256=semantics_sha256,
            source_signal_family=family, primary_signal_family=cast(PrimarySignalFamily, family),
            provenance_ids=('semantics:' + semantics_sha256,))
        for family in sorted({s.signal_family for s in signals})])
    grouping = build_event_grouping_result(opaque_case_id=contract.opaque_case_id,
        decision_time_unix_ns=contract.decision_time_unix_ns,
        observations=[r.observation for r in registry.resolutions], association_claims=(), family_mapping=mapping)
    candidates = build_complete_candidate_universe(entity_registry=registry, event_grouping=grouping, scope_interval=scope)
    frozen = freeze_candidate_universe(candidates)
    focus = build_neutral_focus_selection(candidate_universe=candidates, frozen_candidates=frozen,
        event_grouping=grouping, q_base=q_base, q_max=q_max)
    result_type = AIOps25LiveInputs if aiops else LiveInputs
    return result_type(sources, tuple(signals), registry, grouping, candidates, frozen, focus, scope,
                      dict(sorted(counts.items())), q_base, q_max)
