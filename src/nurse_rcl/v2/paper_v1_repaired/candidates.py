"""Candidate-path-only repairs for the A1/V1-repaired baseline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from nurse_rcl.methods.paper_v1.alarm import canonicalize_service_name
from nurse_rcl.methods.paper_v1.candidates import (
    CandidateSelection,
)
from nurse_rcl.methods.paper_v1.candidates import (
    build_candidates as build_frozen_candidates,
)
from nurse_rcl.methods.paper_v1.topology import NormalTopology, TopologyEdge
from nurse_rcl.models import EvidenceRecord
from nurse_rcl.schemas import DatasetName
from nurse_rcl.v2.observation_schemas import EntityBinding
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

_NANOSECONDS_PER_SECOND = 1_000_000_000


def build_candidates(
    *,
    case_id: str,
    dataset: DatasetName,
    alarm_service: str,
    topology: NormalTopology,
    evidence: Iterable[EvidenceRecord],
    entity_registry: TemporalEntityRegistry,
) -> CandidateSelection:
    """Preserve A0 unless its alarm-local check collapses usable evidence."""

    records = tuple(evidence)
    binding_map = _verified_binding_map(
        case_id=case_id,
        topology=topology,
        evidence=records,
        entity_registry=entity_registry,
    )
    bound_evidence, excluded_evidence = _bind_evidence(records, binding_map)
    bound_topology, excluded_topology = _bind_topology(topology, binding_map)
    binding_warnings: set[str] = set()
    if excluded_evidence:
        binding_warnings.add("unbound_metric_evidence_excluded")
    if excluded_topology:
        binding_warnings.add("unbound_topology_entities_excluded")

    raw_alarm = canonicalize_service_name(alarm_service)
    bound_alarm = binding_map.get(raw_alarm)
    if bound_alarm is None:
        binding_warnings.add("alarm_entity_unbound")
        bound_alarm = _select_bound_alarm(bound_evidence)
        binding_warnings.add("alarm_anchor_reselected_from_bound_metric_evidence")
    elif bound_alarm != raw_alarm:
        binding_warnings.add("alarm_entity_rebound")

    frozen = build_frozen_candidates(
        case_id=case_id,
        dataset=dataset,
        alarm_service=bound_alarm,
        topology=bound_topology,
        evidence=bound_evidence,
    )
    if frozen.stop_reason != "topology_missing":
        return _with_warnings(frozen, binding_warnings)

    alarm = frozen.alarm_service
    metric_candidates = tuple(
        service
        for service in sorted(
            {
                canonicalize_service_name(record.service)
                for record in bound_evidence
                if record.kind == "metric_anomaly"
            }
        )
        if service != alarm
    )
    topology_is_globally_usable = bool(bound_topology.edges)
    if not metric_candidates and not topology_is_globally_usable:
        return _with_warnings(frozen, binding_warnings)

    warnings = _coverage_warnings(bound_topology) | binding_warnings
    if metric_candidates:
        warnings.add("candidate_fallback_metric_evidence")
    if topology_is_globally_usable:
        warnings.add(
            "alarm_absent_from_normal_topology"
            if alarm not in bound_topology.nodes
            else "alarm_isolated_in_normal_topology"
        )
    else:
        warnings.update(("normal_topology_missing", "propagation_forced_zero"))

    return CandidateSelection(
        case_id=frozen.case_id,
        dataset=frozen.dataset,
        alarm_service=alarm,
        max_rounds=frozen.max_rounds,
        candidates=(alarm, *metric_candidates),
        rounds=(),
        stop_reason="saturated" if topology_is_globally_usable else "topology_missing",
        propagation_enabled=topology_is_globally_usable,
        warnings=tuple(sorted(warnings)),
    )


def _verified_binding_map(
    *,
    case_id: str,
    topology: NormalTopology,
    evidence: tuple[EvidenceRecord, ...],
    entity_registry: TemporalEntityRegistry,
) -> dict[str, str]:
    normalized_case_id = case_id.strip()
    if not normalized_case_id:
        raise ValueError("case_id must not be empty")
    if entity_registry.opaque_case_id != normalized_case_id:
        raise ValueError("entity registry belongs to a different candidate case")
    if not entity_registry.verify():
        raise ValueError("entity registry failed verification")

    scope_start = topology.window_start * _NANOSECONDS_PER_SECOND
    scope_end = (
        max((topology.window_end, *(record.window.end for record in evidence)))
        * _NANOSECONDS_PER_SECOND
    )
    grouped: dict[str, list[EntityBinding]] = {}
    for binding in entity_registry.timeline_bindings:
        key = canonicalize_service_name(binding.observed_entity_key)
        grouped.setdefault(key, []).append(binding)

    resolved: dict[str, str] = {}
    for key, bindings in grouped.items():
        bound_services = {
            canonicalize_service_name(binding.service_entity_key)
            for binding in bindings
            if binding.status == "bound" and binding.service_entity_key is not None
        }
        if len(bound_services) != 1:
            continue
        service = next(iter(bound_services))
        intervals = tuple(
            (
                binding.valid_interval.start_unix_ns,
                binding.valid_interval.end_unix_ns,
            )
            for binding in bindings
            if binding.status == "bound"
            and binding.service_entity_key is not None
            and canonicalize_service_name(binding.service_entity_key) == service
        )
        if _covers_scope(intervals, start=scope_start, end=scope_end):
            resolved[key] = service
    return resolved


def _covers_scope(
    intervals: tuple[tuple[int, int], ...],
    *,
    start: int,
    end: int,
) -> bool:
    cursor = start
    for interval_start, interval_end in sorted(intervals):
        if interval_end <= cursor:
            continue
        if interval_start > cursor:
            return False
        cursor = interval_end
        if cursor >= end:
            return True
    return cursor >= end


def _bind_evidence(
    records: tuple[EvidenceRecord, ...],
    binding_map: dict[str, str],
) -> tuple[tuple[EvidenceRecord, ...], bool]:
    bound: list[EvidenceRecord] = []
    excluded = False
    for record in records:
        service = binding_map.get(canonicalize_service_name(record.service))
        if service is None:
            excluded = True
            continue
        if service == record.service:
            bound.append(record)
        else:
            bound.append(record.model_copy(update={"service": service}))
    return tuple(bound), excluded


def _bind_topology(
    topology: NormalTopology,
    binding_map: dict[str, str],
) -> tuple[NormalTopology, bool]:
    mapped_nodes: set[str] = set()
    excluded = False
    for node in topology.nodes:
        service = binding_map.get(canonicalize_service_name(node))
        if service is None:
            excluded = True
        else:
            mapped_nodes.add(service)

    edge_counts: Counter[tuple[str, str]] = Counter()
    for edge in topology.edges:
        callee = binding_map.get(canonicalize_service_name(edge.callee))
        caller = binding_map.get(canonicalize_service_name(edge.caller))
        if callee is None or caller is None:
            excluded = True
            continue
        mapped_nodes.update((callee, caller))
        if callee != caller:
            edge_counts[(callee, caller)] += edge.link_count

    rebound = NormalTopology(
        case_id=topology.case_id,
        t0=topology.t0,
        window_start=topology.window_start,
        window_end=topology.window_end,
        nodes=tuple(sorted(mapped_nodes)),
        edges=tuple(
            TopologyEdge(callee=callee, caller=caller, link_count=link_count)
            for (callee, caller), link_count in sorted(edge_counts.items())
        ),
        coverage=topology.coverage,
    )
    return rebound, excluded


def _select_bound_alarm(records: tuple[EvidenceRecord, ...]) -> str:
    candidates = tuple(record for record in records if record.kind == "metric_anomaly")
    if not candidates:
        raise ValueError("unbound alarm has no bound metric evidence fallback")
    selected = min(
        candidates,
        key=lambda record: (
            record.first_anomalous_at is None,
            record.first_anomalous_at or 0,
            -record.weight,
            canonicalize_service_name(record.service),
            record.signal,
            record.evidence_id,
        ),
    )
    return canonicalize_service_name(selected.service)


def _with_warnings(
    selection: CandidateSelection,
    additions: set[str],
) -> CandidateSelection:
    if not additions:
        return selection
    return CandidateSelection.model_validate(
        {
            **selection.model_dump(mode="python"),
            "warnings": tuple(sorted(set(selection.warnings) | additions)),
        }
    )


def _coverage_warnings(topology: NormalTopology) -> set[str]:
    coverage = topology.coverage
    warnings: set[str] = set()
    if coverage.excluded_cross_case_span_count:
        warnings.add("normal_topology_cross_case_spans_excluded")
    if coverage.excluded_out_of_window_span_count:
        warnings.add("normal_topology_out_of_window_spans_excluded")
    if coverage.parent_reference_count and coverage.parent_link_coverage < 1.0:
        warnings.add("normal_topology_parent_links_incomplete")
    return warnings
