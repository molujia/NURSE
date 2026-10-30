"""Fixed-evidence projection from preparation already admitted by the normal owner.

No query execution, episode calculation, B0 access, ranking or source replay.
This is not an outbound permit: the request boundary must check identity, scope
and its complete byte budget. Every retained Q source is projected, not top-k'd.
"""

from __future__ import annotations

from typing import Literal, cast

from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.live.normal_queries import _seal
from nurse_rcl.v2.live.repair_error_references import _referenced_error_values, _summary_values
from nurse_rcl.v2.live.repair_fixed_facts import RepairFixedFact, _signal_values
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData
from nurse_rcl.v2.live.repair_queries import RepairQueryRequest
from nurse_rcl.v2.observation_schemas import TimeInterval, WindowSummary


class SemanticSignal(StrictFrozenModel):
    reference_id: str
    source_sha256: SHA256
    candidate_service_key: str
    observation_id: str
    metric_name: str | None
    family: str
    modality: str
    unit: str | None
    baseline: WindowSummary
    fault: WindowSummary
    evidence_state: str
    quality_flags: tuple[str, ...]
    formal_scoring_eligible: bool
    lifecycle_meaning: str | None
    lifecycle_zero_count: int | None
    lifecycle_eligible: bool
    facts: tuple[RepairFixedFact, ...]


class SemanticError(StrictFrozenModel):
    reference_id: str
    candidate_service_key: str
    modality: str
    span_kind: str
    statistic: str
    severity_basis: str
    baseline_known_denominator: int
    fault_known_denominator: int
    baseline_fraction: float | None
    fault_fraction: float | None
    observed_error_count: int
    sampling_coverage: Literal["unknown"] = "unknown"
    facts: tuple[RepairFixedFact, ...]
    limitations: tuple[str, ...]


class SemanticEdge(StrictFrozenModel):
    reference_id: str
    window: Literal["baseline", "fault"]
    caller: str
    callee: str
    relation_type: str
    direction: str
    allows_directed_propagation: bool
    observation_state: str


class NormalSemanticContext(StrictFrozenModel):
    schema_version: Literal["nurse.v2.normal-semantic-context/1"] = (
        "nurse.v2.normal-semantic-context/1"
    )
    opaque_case_id: str
    catalog_sha256: SHA256
    candidate_service_keys: tuple[str, ...]
    focus_service_keys: tuple[str, ...]
    baseline_window: TimeInterval
    fault_window: TimeInterval
    scope: Literal["fixed_preparation_only"] = "fixed_preparation_only"
    signals: tuple[SemanticSignal, ...]
    errors: tuple[SemanticError, ...]
    edges: tuple[SemanticEdge, ...]
    graph_limitations: tuple[str, ...]
    requests: tuple[RepairQueryRequest, ...]
    score_authority: Literal[False] = False
    context_sha256: SHA256


def project_normal_semantic_context(data: RepairPreparationData) -> NormalSemanticContext:
    """Private normal owner entry, deliberately has no paired session parameter."""
    return _project_fixed_context(
        data, set(data.diagnostics.legacy.focus.focus_set.selected_service_keys),
    )


def _project_fixed_context(
    data: RepairPreparationData, selected: set[str],
) -> NormalSemanticContext:
    """Shared fixed-row projection; card owner controls and charges the selection.

    The original Q field stays unchanged. A card extracts only its own rows;
    this private helper must not replace the public first-round Q selection.
    """
    focus = data.diagnostics.legacy.focus.focus_set
    catalog = data.catalog
    contract = data.diagnostics.legacy.sources.adapter.time_contract
    if not set(focus.selected_service_keys) <= set(catalog.candidate_service_keys) or (
        focus.opaque_case_id != catalog.opaque_case_id
        or contract.opaque_case_id != catalog.opaque_case_id
        or data.candidate_freeze_sha256 != catalog.candidate_freeze_sha256
        or not selected <= set(catalog.candidate_service_keys)
    ):
        raise ValueError("normal semantic preparation C/Q/case differs")
    signals = []
    for source in data.sources:
        if source.candidate_service_key not in selected:
            continue
        if source.time_contract != contract:
            raise ValueError("normal semantic source time contract differs")
        assert source.candidate_service_key is not None
        fact, quality, life = _signal_values(source), source.quality, source.lifecycle
        signals.append(SemanticSignal(
            reference_id="fixed_" + fact.result_sha256, source_sha256=source.source_sha256,
            candidate_service_key=source.candidate_service_key,
            observation_id=source.observation.observation_id, metric_name=source.metric_name,
            family=source.observation.signal_family, modality=source.observation.modality,
            unit=source.observation.unit, baseline=quality.source_summary.baseline_summary,
            fault=quality.source_summary.fault_summary, evidence_state=quality.evidence_state,
            quality_flags=quality.quality_flags,
            formal_scoring_eligible=quality.formal_scoring_eligible,
            lifecycle_meaning=None if life is None else life.state_meaning,
            lifecycle_zero_count=None if life is None else len(life.events),
            lifecycle_eligible=life is not None and life.formal_event_eligible, facts=fact.facts,
        ))
    errors = _referenced_error_values(
        sources=data.error_sources,
        summaries=tuple(_summary_values(o) for o in data.diagnostics.errors),
        observation_sha256s=data.error_observation_sha256s,
        candidate_service_keys=catalog.candidate_service_keys,
    )
    error_rows = tuple(SemanticError(
        reference_id="fixed_" + f.result_sha256, candidate_service_key=f.candidate_service_key,
        modality=f.modality, span_kind=f.span_kind, statistic=f.statistic,
        severity_basis=f.severity_basis, baseline_known_denominator=f.baseline_known_denominator,
        fault_known_denominator=f.fault_known_denominator, baseline_fraction=f.baseline_fraction,
        fault_fraction=f.fault_fraction, observed_error_count=f.observed_error_count,
        facts=f.facts, limitations=f.limitations,
    ) for f in errors.facts if f.candidate_service_key is not None
        and f.candidate_service_key in selected)
    graph = data.graph.graph_pair
    edges = tuple(SemanticEdge(
        reference_id="edge_" + e.edge_sha256, window=cast(Literal["baseline", "fault"], window),
        caller=e.caller_service_key, callee=e.callee_service_key,
        relation_type=e.relation_type, direction=e.direction,
        allows_directed_propagation=e.allows_directed_propagation,
        observation_state=e.observation_state,
    ) for window, snapshot in (("baseline", graph.normal_snapshot), ("fault", graph.fault_snapshot))
        for e in snapshot.edges)
    return _seal(NormalSemanticContext(
        opaque_case_id=catalog.opaque_case_id, catalog_sha256=catalog.catalog_sha256,
        candidate_service_keys=catalog.candidate_service_keys,
        focus_service_keys=focus.selected_service_keys, baseline_window=contract.baseline_window,
        fault_window=contract.fault_window, signals=tuple(signals), errors=error_rows, edges=edges,
        graph_limitations=data.graph.limitations,
        requests=tuple(r for r in catalog.requests if r.candidate_service_key in selected),
        context_sha256="0" * 64,
    ), "context_sha256")
