"""Freeze ranking and audit traces into a hashed Verified RCL Context."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.methods.paper_v1.causal_history import HistoryScoreArtifact
from nurse_rcl.methods.paper_v1.final_ranking import FinalRankingArtifact
from nurse_rcl.methods.paper_v1.guarded_graph import GuardedGraphArtifact
from nurse_rcl.methods.paper_v1.scc_scoring import SCCScoringArtifact
from nurse_rcl.models import EvidencePath, SCCTrace, VerifiedRCLContext
from nurse_rcl.schemas import StrictFrozenModel

VERIFIED_CONTEXT_VERSION: Literal["verified_rcl_context_v1"] = (
    "verified_rcl_context_v1"
)


class VerifiedRCLContextArtifact(StrictFrozenModel):
    """Content-addressed frozen context and its complete ranking input chain."""

    context_version: Literal["verified_rcl_context_v1"] = VERIFIED_CONTEXT_VERSION
    context: VerifiedRCLContext
    context_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_ranking_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    guarded_graph_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scc_scoring_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    history_score_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        """Verify both the frozen context and its enclosing chain artifact."""

        if self.context_hash != _model_hash(self.context):
            return False
        unsigned = self.model_dump(mode="json", exclude={"artifact_hash"})
        return self.artifact_hash == sha256_hex(canonical_json_bytes(unsigned))


def freeze_verified_rcl_context(
    *,
    case_id: str,
    final: FinalRankingArtifact,
    guarded: GuardedGraphArtifact,
    scc: SCCScoringArtifact,
    history: HistoryScoreArtifact,
    warnings: tuple[str, ...],
) -> VerifiedRCLContextArtifact:
    """Freeze rank-bearing values and provenance without recomputing either."""

    _verify_input_chain(final=final, guarded=guarded, scc=scc, history=history)
    context = VerifiedRCLContext(
        case_id=case_id,
        top_k=final.ranking.top_k,
        evidence_paths=_evidence_paths(guarded),
        guard_decisions=guarded.graph.decisions,
        scc_trace=tuple(
            SCCTrace(
                component_id=component.component_id,
                members=component.members,
                removed_relation_id=component.removed_relation_id,
            )
            for component in sorted(scc.components, key=lambda item: item.component_id)
        ),
        history_trace=history.neighbors,
        warnings=tuple(sorted(set(warnings))),
    )
    context_hash = _model_hash(context)
    unsigned = {
        "context_version": VERIFIED_CONTEXT_VERSION,
        "context": context.model_dump(mode="json"),
        "context_hash": context_hash,
        "final_ranking_hash": final.artifact_hash,
        "guarded_graph_hash": guarded.artifact_hash,
        "scc_scoring_hash": scc.artifact_hash,
        "history_score_hash": history.artifact_hash,
    }
    result = VerifiedRCLContextArtifact(
        context=context,
        context_hash=context_hash,
        final_ranking_hash=final.artifact_hash,
        guarded_graph_hash=guarded.artifact_hash,
        scc_scoring_hash=scc.artifact_hash,
        history_score_hash=history.artifact_hash,
        artifact_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("Verified RCL Context artifact verification failed")
    return result


def _verify_input_chain(
    *,
    final: FinalRankingArtifact,
    guarded: GuardedGraphArtifact,
    scc: SCCScoringArtifact,
    history: HistoryScoreArtifact,
) -> None:
    if not final.verify():
        raise ValueError("final ranking artifact hash verification failed")
    if not guarded.verify():
        raise ValueError("guarded graph artifact hash verification failed")
    if not scc.verify():
        raise ValueError("SCC scoring artifact hash verification failed")
    if not history.verify():
        raise ValueError("history score artifact hash verification failed")
    if final.raw_epr_hash != scc.raw_epr_hash:
        raise ValueError("final ranking raw E/P/R hash does not match SCC scoring")
    if final.scc_scoring_hash != scc.artifact_hash:
        raise ValueError("final ranking SCC scoring hash does not match")
    if final.history_score_hash != history.artifact_hash:
        raise ValueError("final ranking history score hash does not match")
    if scc.original_guarded_graph_hash != guarded.graph_hash:
        raise ValueError("SCC scoring guarded graph hash does not match")
    if scc.original_guarded_artifact_hash != guarded.artifact_hash:
        raise ValueError("SCC scoring guarded artifact hash does not match")


def _evidence_paths(guarded: GuardedGraphArtifact) -> tuple[EvidencePath, ...]:
    relation_ids: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    evidence_ids: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for relation in guarded.graph.relations:
        if relation.relation_type == "supports_root":
            source_service = relation.source_service
            target_service = relation.source_service
        elif relation.relation_type == "propagates_to":
            if relation.target_service is None:
                raise ValueError("propagates_to relation requires a target service")
            source_service = relation.source_service
            target_service = relation.target_service
        else:
            if relation.target_service is None:
                raise ValueError("symptom_of relation requires a causal target service")
            source_service = relation.target_service
            target_service = relation.source_service
        key = (source_service, target_service)
        relation_ids[key].add(relation.relation_id)
        evidence_ids[key].update(relation.evidence_ids)
    return tuple(
        EvidencePath(
            source_service=source_service,
            target_service=target_service,
            relation_ids=tuple(sorted(relation_ids[(source_service, target_service)])),
            evidence_ids=tuple(sorted(evidence_ids[(source_service, target_service)])),
        )
        for source_service, target_service in sorted(relation_ids)
    )


def _model_hash(model: StrictFrozenModel) -> str:
    return sha256_hex(canonical_json_bytes(model.model_dump(mode="json")))
