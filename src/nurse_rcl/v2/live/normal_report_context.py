"""Small provenance records from already admitted sources; never re-prepare or score."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.live.normal_artifacts import NormalEvidence
from nurse_rcl.v2.live.repair_sources import RepairMeasurementSource
from nurse_rcl.v2.observation_schemas import TimeInterval

if TYPE_CHECKING:
    from nurse_rcl.v2.live.normal_b2_result import NormalB2Evidence


class NormalObservationContext(StrictFrozenModel):
    term_id: str
    term_sha256: str
    source_sha256: str
    observation_id: str
    opaque_case_id: str
    candidate_service_key: str | None
    metric_name: str | None
    unit: str
    baseline_window: TimeInterval
    fault_window: TimeInterval


def observation_contexts(
    evidence: NormalEvidence | NormalB2Evidence,
    sources: tuple[RepairMeasurementSource, ...],
) -> dict[str, NormalObservationContext]:
    """Private normal-worker inputs, with lightweight identity/window matching only."""
    by_pin = {s.source_sha256: s for s in sources}
    if len(by_pin) != len(sources):
        raise ValueError("report source context has duplicate identities")
    contexts = {}
    for term in evidence.ranking.propagation.local_scoring.strengths.terms:
        if term.candidate_service_key not in evidence.ranking.top_k_service_keys:
            continue
        source = by_pin.get(term.source_sha256)
        if source is None:
            continue
        observation, clock = source.observation, source.time_contract
        if (
            source.dataset != evidence.dataset
            or observation.opaque_case_id != evidence.opaque_case_id
            or clock.opaque_case_id != evidence.opaque_case_id
            or source.candidate_service_key != term.candidate_service_key
            or source.observed_entity_key != term.observed_entity_key
            or source.metric_name != term.metric_name
            or observation.modality != term.modality
            or observation.signal_family != term.primary_family
            or source.query_interval != term.interval
            or clock.fault_window != term.interval
            or observation.interval.start_unix_ns != clock.baseline_window.start_unix_ns
            or observation.interval.end_unix_ns != clock.fault_window.end_unix_ns
        ):
            raise ValueError("report source context case/candidate/metric/window differs")
        contexts[term.term_id] = NormalObservationContext(
            term_id=term.term_id,
            term_sha256=sha256_hex(canonical_json_bytes(term.model_dump(mode="json"))),
            source_sha256=source.source_sha256,
            observation_id=observation.observation_id,
            opaque_case_id=evidence.opaque_case_id,
            candidate_service_key=term.candidate_service_key,
            metric_name=term.metric_name,
            unit=observation.unit or "unavailable_in_retained_fact",
            baseline_window=clock.baseline_window,
            fault_window=clock.fault_window,
        )
    return contexts
