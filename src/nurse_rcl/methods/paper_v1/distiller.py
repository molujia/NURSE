"""Deterministically distill a preliminary graph into the agent payload."""

from __future__ import annotations

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.methods.paper_v1.evidence_initializer import PreliminaryGraphArtifact
from nurse_rcl.models import SemanticPayload
from nurse_rcl.models.semantic import CandidateCard, SemanticFactType
from nurse_rcl.schemas import StrictFrozenModel

SEMANTIC_ROLE_POLICY: tuple[SemanticFactType, ...] = (
    "root_support",
    "source_support",
    "symptom_support",
)


class SemanticPayloadArtifact(StrictFrozenModel):
    """Hash-bearing payload passed to the closed Semantic Fact Agent interface."""

    payload: SemanticPayload
    payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        return self.payload_hash == _payload_hash(self.payload)


def distill_context(preliminary: PreliminaryGraphArtifact) -> SemanticPayloadArtifact:
    """Create stable cards and evidence summaries without assigning roles or ranks."""

    if not preliminary.verify():
        raise ValueError("preliminary graph hash verification failed")
    graph = preliminary.graph
    adjacency = {candidate.service: set[str]() for candidate in graph.candidates}
    for edge in graph.topology_edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)

    cards = tuple(
        CandidateCard(
            service=candidate.service,
            evidence_ids=tuple(sorted(candidate.evidence_ids)),
            neighbors=tuple(sorted(adjacency[candidate.service])),
        )
        for candidate in sorted(graph.candidates, key=lambda candidate: candidate.service)
    )
    evidence = tuple(
        sorted(
            graph.evidence,
            key=lambda record: (record.service, record.signal, record.evidence_id),
        )
    )
    payload = SemanticPayload(
        schema_version=1,
        preliminary_graph_hash=preliminary.graph_hash,
        candidates=cards,
        evidence=evidence,
        rules=tuple(sorted(graph.rules, key=lambda rule: rule.rule_id)),
        semantic_role_policy=SEMANTIC_ROLE_POLICY,
    )
    return SemanticPayloadArtifact(payload=payload, payload_hash=_payload_hash(payload))


def _payload_hash(payload: SemanticPayload) -> str:
    return sha256_hex(canonical_json_bytes(payload.model_dump(mode="json")))
