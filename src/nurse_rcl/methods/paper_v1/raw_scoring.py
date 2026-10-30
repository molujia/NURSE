"""Compute raw direct, propagation, and rule evidence with full traces."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.methods.paper_v1.evidence_initializer import PreliminaryGraphArtifact
from nurse_rcl.methods.paper_v1.guarded_graph import GuardedGraphArtifact
from nurse_rcl.models import EvidenceRecord, GuardedRelation
from nurse_rcl.schemas import StrictFrozenModel

RAW_EPR_VERSION: Literal["raw_epr_v1"] = "raw_epr_v1"
DirectSemanticRole = Literal["root_support", "source_support", "symptom_support"]
PropagationRole = Literal["source_consistency", "symptom_explanation"]


class DirectEvidenceContribution(StrictFrozenModel):
    evidence_id: str = Field(min_length=1)
    semantic_role: DirectSemanticRole
    relation_ids: tuple[str, ...] = Field(min_length=1)
    signed_value: float


class PropagationContribution(StrictFrozenModel):
    relation_id: str = Field(min_length=1)
    source_service: str = Field(min_length=1)
    symptom_service: str = Field(min_length=1)
    source_evidence_id: str = Field(min_length=1)
    symptom_evidence_id: str = Field(min_length=1)
    distance: int = Field(gt=0)
    fanout: int = Field(ge=0)
    role: PropagationRole
    factor: float = Field(ge=0.0)
    signed_value: float


class RuleEvidenceContribution(StrictFrozenModel):
    rule_id: str = Field(min_length=1)
    signed_value: float


class RawCandidateEPR(StrictFrozenModel):
    service: str = Field(min_length=1)
    e: float
    p: float
    r: float
    direct_contributions: tuple[DirectEvidenceContribution, ...]
    propagation_contributions: tuple[PropagationContribution, ...]
    rule_contributions: tuple[RuleEvidenceContribution, ...]


class RawEPRArtifact(StrictFrozenModel):
    scoring_version: Literal["raw_epr_v1"] = RAW_EPR_VERSION
    preliminary_graph_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    guarded_graph_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    guarded_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidates: tuple[RawCandidateEPR, ...] = Field(min_length=1)
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        """Verify the complete raw-score artifact content hash."""

        unsigned = self.model_dump(mode="json", exclude={"artifact_hash"})
        return self.artifact_hash == sha256_hex(canonical_json_bytes(unsigned))


@dataclass(frozen=True, slots=True)
class _PropagationLink:
    relation_id: str
    source_service: str
    symptom_service: str
    source_evidence_id: str
    symptom_evidence_id: str


def propagation_factor(
    *,
    source_weight: float,
    symptom_weight: float,
    source_time: int | None,
    symptom_time: int | None,
    distance: int,
    fanout: int,
) -> float:
    """Return fixed V1 g for one source-to-symptom relation."""

    if distance <= 0:
        raise ValueError("propagation distance must be positive")
    if fanout < 0:
        raise ValueError("propagation fanout cannot be negative")
    if not math.isfinite(source_weight) or not math.isfinite(symptom_weight):
        raise ValueError("propagation weights must be finite")
    if source_weight < 0.0 or symptom_weight < 0.0:
        raise ValueError("propagation weights cannot be negative")
    if source_time is None or symptom_time is None or source_time >= symptom_time:
        return 0.0
    return min(source_weight, symptom_weight) / (distance * max(1, fanout))


def compute_raw_epr(
    *,
    preliminary: PreliminaryGraphArtifact,
    guarded: GuardedGraphArtifact,
) -> RawEPRArtifact:
    """Compute signed E, symmetric relation-local P, and separate zero-default R."""

    _verify_input_chain(preliminary=preliminary, guarded=guarded)
    evidence = {record.evidence_id: record for record in preliminary.graph.evidence}
    services = {candidate.service for candidate in preliminary.graph.candidates}
    accepted_fact_hashes = {
        decision.fact_hash
        for decision in guarded.graph.decisions
        if decision.accepted
    }
    if any(
        relation.fact_hash is None or relation.fact_hash not in accepted_fact_hashes
        for relation in guarded.graph.relations
    ):
        raise ValueError("guarded relation is not backed by an accepted fact hash")

    direct_assignments: dict[
        tuple[str, str, DirectSemanticRole], set[str]
    ] = {}
    links: list[_PropagationLink] = []
    for relation in guarded.graph.relations:
        for service, evidence_id, role in _direct_roles(relation):
            if service not in services:
                raise ValueError("guarded relation service is outside preliminary candidates")
            record = _owned_evidence(evidence, evidence_id=evidence_id, service=service)
            key = (service, record.evidence_id, role)
            direct_assignments.setdefault(key, set()).add(relation.relation_id)
        link = _propagation_link(relation)
        if link is not None:
            links.append(link)

    direct_by_service: dict[str, list[DirectEvidenceContribution]] = {
        service: [] for service in services
    }
    for (service, evidence_id, role), relation_ids in direct_assignments.items():
        weight = evidence[evidence_id].weight
        signed_value = -weight if role == "symptom_support" else weight
        direct_by_service[service].append(
            DirectEvidenceContribution(
                evidence_id=evidence_id,
                semantic_role=role,
                relation_ids=tuple(sorted(relation_ids)),
                signed_value=signed_value,
            )
        )
    for contributions in direct_by_service.values():
        contributions.sort(
            key=lambda item: (item.evidence_id, item.semantic_role, item.relation_ids)
        )

    propagation_by_service = _propagation_contributions(
        services=services,
        links=links,
        evidence=evidence,
        preliminary=preliminary,
    )
    candidates = tuple(
        RawCandidateEPR(
            service=service,
            e=sum(item.signed_value for item in direct_by_service[service]),
            p=sum(item.signed_value for item in propagation_by_service[service]),
            r=0.0,
            direct_contributions=tuple(direct_by_service[service]),
            propagation_contributions=tuple(propagation_by_service[service]),
            rule_contributions=(),
        )
        for service in sorted(services)
    )
    unsigned = {
        "scoring_version": RAW_EPR_VERSION,
        "preliminary_graph_hash": preliminary.graph_hash,
        "guarded_graph_hash": guarded.graph_hash,
        "guarded_artifact_hash": guarded.artifact_hash,
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
    }
    result = RawEPRArtifact(
        preliminary_graph_hash=preliminary.graph_hash,
        guarded_graph_hash=guarded.graph_hash,
        guarded_artifact_hash=guarded.artifact_hash,
        candidates=candidates,
        artifact_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("raw E/P/R artifact verification failed")
    return result


def _verify_input_chain(
    *,
    preliminary: PreliminaryGraphArtifact,
    guarded: GuardedGraphArtifact,
) -> None:
    if not preliminary.verify():
        raise ValueError("preliminary graph hash verification failed")
    if not guarded.verify():
        raise ValueError("guarded graph artifact hash verification failed")
    if guarded.graph.preliminary_graph_hash != preliminary.graph_hash:
        raise ValueError("guarded graph does not match preliminary graph hash")


def _direct_roles(
    relation: GuardedRelation,
) -> tuple[tuple[str, str, DirectSemanticRole], ...]:
    if relation.relation_type == "supports_root":
        if len(relation.evidence_ids) != 1 or relation.target_service is not None:
            raise ValueError("supports_root relation shape is invalid")
        return ((relation.source_service, relation.evidence_ids[0], "root_support"),)
    if len(relation.evidence_ids) != 2 or relation.target_service is None:
        raise ValueError("propagation relation shape is invalid")
    if relation.relation_type == "propagates_to":
        return (
            (relation.source_service, relation.evidence_ids[0], "source_support"),
            (relation.target_service, relation.evidence_ids[1], "symptom_support"),
        )
    return (
        (relation.source_service, relation.evidence_ids[0], "symptom_support"),
        (relation.target_service, relation.evidence_ids[1], "source_support"),
    )


def _propagation_link(relation: GuardedRelation) -> _PropagationLink | None:
    if relation.relation_type == "supports_root":
        return None
    if len(relation.evidence_ids) != 2 or relation.target_service is None:
        raise ValueError("propagation relation shape is invalid")
    if relation.relation_type == "propagates_to":
        return _PropagationLink(
            relation_id=relation.relation_id,
            source_service=relation.source_service,
            symptom_service=relation.target_service,
            source_evidence_id=relation.evidence_ids[0],
            symptom_evidence_id=relation.evidence_ids[1],
        )
    return _PropagationLink(
        relation_id=relation.relation_id,
        source_service=relation.target_service,
        symptom_service=relation.source_service,
        source_evidence_id=relation.evidence_ids[1],
        symptom_evidence_id=relation.evidence_ids[0],
    )


def _propagation_contributions(
    *,
    services: set[str],
    links: list[_PropagationLink],
    evidence: Mapping[str, EvidenceRecord],
    preliminary: PreliminaryGraphArtifact,
) -> dict[str, list[PropagationContribution]]:
    contributions: dict[str, list[PropagationContribution]] = {
        service: [] for service in services
    }
    fanout_by_source = {service: 0 for service in services}
    for link in links:
        fanout_by_source[link.source_service] += 1
    adjacency = _undirected_adjacency(preliminary)
    for link in links:
        source_record = _owned_evidence(
            evidence,
            evidence_id=link.source_evidence_id,
            service=link.source_service,
        )
        symptom_record = _owned_evidence(
            evidence,
            evidence_id=link.symptom_evidence_id,
            service=link.symptom_service,
        )
        distance = _shortest_distance(
            adjacency,
            source=link.source_service,
            target=link.symptom_service,
        )
        fanout = fanout_by_source[link.source_service]
        factor = propagation_factor(
            source_weight=source_record.weight,
            symptom_weight=symptom_record.weight,
            source_time=source_record.first_anomalous_at,
            symptom_time=symptom_record.first_anomalous_at,
            distance=distance,
            fanout=fanout,
        )
        contributions[link.source_service].append(
            PropagationContribution(
                relation_id=link.relation_id,
                source_service=link.source_service,
                symptom_service=link.symptom_service,
                source_evidence_id=link.source_evidence_id,
                symptom_evidence_id=link.symptom_evidence_id,
                distance=distance,
                fanout=fanout,
                role="source_consistency",
                factor=factor,
                signed_value=factor,
            )
        )
        contributions[link.symptom_service].append(
            PropagationContribution(
                relation_id=link.relation_id,
                source_service=link.source_service,
                symptom_service=link.symptom_service,
                source_evidence_id=link.source_evidence_id,
                symptom_evidence_id=link.symptom_evidence_id,
                distance=distance,
                fanout=fanout,
                role="symptom_explanation",
                factor=factor,
                signed_value=-factor,
            )
        )
    for service_contributions in contributions.values():
        service_contributions.sort(key=lambda item: (item.relation_id, item.role))
    return contributions


def _owned_evidence(
    evidence: Mapping[str, EvidenceRecord],
    *,
    evidence_id: str,
    service: str,
) -> EvidenceRecord:
    record = evidence.get(evidence_id)
    if record is None:
        raise ValueError("guarded relation cites missing preliminary evidence")
    if record.service != service:
        raise ValueError("guarded relation cites evidence owned by another service")
    return record


def _undirected_adjacency(
    preliminary: PreliminaryGraphArtifact,
) -> dict[str, set[str]]:
    adjacency = {
        candidate.service: set[str]() for candidate in preliminary.graph.candidates
    }
    for edge in preliminary.graph.topology_edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
    return adjacency


def _shortest_distance(
    adjacency: Mapping[str, set[str]],
    *,
    source: str,
    target: str,
) -> int:
    pending = deque([(source, 0)])
    visited = {source}
    while pending:
        current, distance = pending.popleft()
        for neighbor in adjacency.get(current, set()):
            if neighbor == target:
                return distance + 1
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append((neighbor, distance + 1))
    raise ValueError("propagation relation has no preliminary topology path")
