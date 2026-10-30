"""Evidence-gated candidate expansion for the fixed paper V1 method."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import Field

from nurse_rcl.methods.paper_v1.alarm import canonicalize_service_name
from nurse_rcl.methods.paper_v1.topology import NormalTopology
from nurse_rcl.models import EvidenceRecord
from nurse_rcl.schemas import DatasetName, StrictFrozenModel

MJ_BY_DATASET: dict[DatasetName, int] = {"rcabench": 7, "aiops25": 3}
StopReason = Literal["saturated", "max_rounds", "topology_missing"]


class CandidateExpansionRound(StrictFrozenModel):
    """One deterministic breadth-expansion decision and its exclusions."""

    round_number: int = Field(ge=1)
    frontier: tuple[str, ...]
    adjacent_unseen: tuple[str, ...]
    admitted: tuple[str, ...]
    excluded_without_metric_evidence: tuple[str, ...]


class CandidateSelection(StrictFrozenModel):
    """Frozen candidate order plus enough state to audit why expansion stopped."""

    case_id: str = Field(min_length=1)
    dataset: DatasetName
    alarm_service: str = Field(min_length=1)
    max_rounds: int = Field(ge=1)
    candidates: tuple[str, ...] = Field(min_length=1)
    rounds: tuple[CandidateExpansionRound, ...]
    stop_reason: StopReason
    propagation_enabled: bool
    warnings: tuple[str, ...]


def build_candidates(
    *,
    case_id: str,
    dataset: DatasetName,
    alarm_service: str,
    topology: NormalTopology,
    evidence: Iterable[EvidenceRecord],
) -> CandidateSelection:
    """Expand from the alarm through same-case topology and metric evidence only."""

    normalized_case_id = case_id.strip()
    if not normalized_case_id:
        raise ValueError("case_id must not be empty")
    if topology.case_id != normalized_case_id:
        raise ValueError("topology case identity does not match candidate case identity")
    if dataset not in MJ_BY_DATASET:
        raise ValueError(f"unsupported dataset: {dataset}")

    alarm = canonicalize_service_name(alarm_service)
    metric_services = {
        canonicalize_service_name(record.service)
        for record in evidence
        if record.kind == "metric_anomaly"
    }
    adjacency = _adjacency(topology)
    if not topology.edges or alarm not in topology.nodes or not adjacency.get(alarm):
        return CandidateSelection(
            case_id=normalized_case_id,
            dataset=dataset,
            alarm_service=alarm,
            max_rounds=MJ_BY_DATASET[dataset],
            candidates=(alarm,),
            rounds=(),
            stop_reason="topology_missing",
            propagation_enabled=False,
            warnings=(
                "normal_topology_missing",
                "candidate_coverage_alarm_only",
                "propagation_forced_zero",
            ),
        )

    selected = [alarm]
    selected_set = {alarm}
    frontier: tuple[str, ...] = (alarm,)
    rounds: list[CandidateExpansionRound] = []
    warnings = _coverage_warnings(topology)
    stop_reason: StopReason = "max_rounds"

    for round_number in range(1, MJ_BY_DATASET[dataset] + 1):
        adjacent_unseen = tuple(
            sorted(
                {
                    neighbor
                    for service in frontier
                    for neighbor in adjacency.get(service, ())
                    if neighbor not in selected_set
                }
            )
        )
        admitted = tuple(
            service for service in adjacent_unseen if service in metric_services
        )
        excluded = tuple(
            service for service in adjacent_unseen if service not in metric_services
        )
        rounds.append(
            CandidateExpansionRound(
                round_number=round_number,
                frontier=frontier,
                adjacent_unseen=adjacent_unseen,
                admitted=admitted,
                excluded_without_metric_evidence=excluded,
            )
        )
        if excluded:
            warnings.add("candidate_neighbors_without_metric_evidence")
        if not admitted:
            stop_reason = "saturated"
            break
        selected.extend(admitted)
        selected_set.update(admitted)
        frontier = admitted
    else:
        warnings.add("candidate_expansion_reached_mj")

    return CandidateSelection(
        case_id=normalized_case_id,
        dataset=dataset,
        alarm_service=alarm,
        max_rounds=MJ_BY_DATASET[dataset],
        candidates=tuple(selected),
        rounds=tuple(rounds),
        stop_reason=stop_reason,
        propagation_enabled=True,
        warnings=tuple(sorted(warnings)),
    )


def _adjacency(topology: NormalTopology) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {service: set() for service in topology.nodes}
    for edge in topology.edges:
        adjacency.setdefault(edge.callee, set()).add(edge.caller)
        adjacency.setdefault(edge.caller, set()).add(edge.callee)
    return adjacency


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
