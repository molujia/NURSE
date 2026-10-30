"""Compile accepted semantic facts into a provenance-backed guarded graph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex, stable_identifier
from nurse_rcl.methods.paper_v1.distiller import SemanticPayloadArtifact, distill_context
from nurse_rcl.methods.paper_v1.evidence_initializer import PreliminaryGraphArtifact
from nurse_rcl.methods.paper_v1.semantic_guard import SemanticGuardArtifact
from nurse_rcl.models import (
    EvidenceRecord,
    GuardedEvidenceGraph,
    GuardedRelation,
    SemanticFact,
)
from nurse_rcl.models.semantic import RelationType
from nurse_rcl.schemas import StrictFrozenModel

GUARDED_GRAPH_VERSION: Literal["guarded_graph_v1"] = "guarded_graph_v1"


class GuardedGraphArtifact(StrictFrozenModel):
    """Content-addressed guarded graph with the complete semantic input chain."""

    graph_version: Literal["guarded_graph_v1"] = GUARDED_GRAPH_VERSION
    graph: GuardedEvidenceGraph
    payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    guard_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        """Verify the graph content hash and the enclosing chain artifact."""

        expected_graph_hash = _model_hash(self.graph)
        if self.graph_hash != expected_graph_hash:
            return False
        unsigned = self.model_dump(mode="json", exclude={"artifact_hash"})
        return self.artifact_hash == sha256_hex(canonical_json_bytes(unsigned))


def compile_guarded_graph(
    *,
    preliminary: PreliminaryGraphArtifact,
    payload: SemanticPayloadArtifact,
    guard: SemanticGuardArtifact,
) -> GuardedGraphArtifact:
    """Compile only accepted facts after verifying every preceding stage."""

    if not preliminary.verify():
        raise ValueError("preliminary graph hash verification failed")
    if not payload.verify():
        raise ValueError("semantic payload hash verification failed")
    if payload != distill_context(preliminary):
        raise ValueError("semantic payload does not match preliminary graph")
    if not guard.verify():
        raise ValueError("semantic guard artifact hash verification failed")
    if guard.payload_hash != payload.payload_hash:
        raise ValueError("semantic guard payload hash does not match")

    evidence = {record.evidence_id: record for record in preliminary.graph.evidence}
    relations: list[GuardedRelation] = []
    for fact, decision in zip(guard.parsed_facts, guard.decisions, strict=True):
        if not decision.accepted:
            continue
        if fact is None:  # pragma: no cover - protected by guard.verify()
            raise ValueError("accepted guard decision has no parsed fact")
        relations.append(
            _compile_relation(
                fact=fact,
                fact_hash=decision.fact_hash,
                evidence=evidence,
            )
        )

    graph = GuardedEvidenceGraph(
        preliminary_graph_hash=preliminary.graph_hash,
        relations=tuple(relations),
        decisions=guard.decisions,
    )
    graph_hash = _model_hash(graph)
    unsigned = {
        "graph_version": GUARDED_GRAPH_VERSION,
        "graph": graph.model_dump(mode="json"),
        "payload_hash": payload.payload_hash,
        "guard_artifact_hash": guard.artifact_hash,
        "graph_hash": graph_hash,
    }
    result = GuardedGraphArtifact(
        graph=graph,
        payload_hash=payload.payload_hash,
        guard_artifact_hash=guard.artifact_hash,
        graph_hash=graph_hash,
        artifact_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("guarded graph artifact verification failed")
    return result


def _compile_relation(
    *,
    fact: SemanticFact,
    fact_hash: str,
    evidence: Mapping[str, EvidenceRecord],
) -> GuardedRelation:
    relation_type: RelationType
    target_service: str | None
    evidence_ids: tuple[str, ...]
    if fact.fact_type == "root_support":
        relation_type = "supports_root"
        target_service = None
        evidence_ids = (fact.evidence_id,)
    else:
        if fact.related_service is None or fact.related_evidence_id is None:
            raise ValueError("accepted relational fact is incomplete")
        relation_type = (
            "propagates_to" if fact.fact_type == "source_support" else "symptom_of"
        )
        target_service = fact.related_service
        evidence_ids = (fact.evidence_id, fact.related_evidence_id)

    try:
        relation_weight = min(evidence[evidence_id].weight for evidence_id in evidence_ids)
    except KeyError as error:  # pragma: no cover - protected by the verified guard chain
        raise ValueError("accepted fact cites evidence outside the preliminary graph") from error
    relation_payload = {
        "relation_type": relation_type,
        "source_service": fact.service,
        "target_service": target_service,
        "evidence_ids": list(evidence_ids),
        "fact_hash": fact_hash,
    }
    return GuardedRelation(
        relation_id=stable_identifier("relation", relation_payload),
        relation_type=relation_type,
        source_service=fact.service,
        target_service=target_service,
        evidence_ids=evidence_ids,
        weight=relation_weight,
        fact_hash=fact_hash,
    )


def _model_hash(model: StrictFrozenModel) -> str:
    return sha256_hex(canonical_json_bytes(model.model_dump(mode="json")))
