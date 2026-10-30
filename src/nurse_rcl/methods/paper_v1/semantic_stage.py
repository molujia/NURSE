"""Finalize semantic facts with deterministic degraded continuation semantics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.methods.paper_v1.distiller import SemanticPayloadArtifact
from nurse_rcl.methods.paper_v1.evidence_initializer import PreliminaryGraphArtifact
from nurse_rcl.methods.paper_v1.guarded_graph import (
    GuardedGraphArtifact,
    compile_guarded_graph,
)
from nurse_rcl.methods.paper_v1.semantic_guard import (
    SemanticGuardArtifact,
    validate_semantic_fact_response,
)
from nurse_rcl.schemas import StrictFrozenModel

SEMANTIC_STAGE_VERSION: Literal["semantic_stage_v1"] = "semantic_stage_v1"
SemanticDegradationCode = Literal["semantic_agent_failed", "semantic_facts_empty"]


class SemanticStageArtifact(StrictFrozenModel):
    """Auditable semantic result that always leaves a graph for downstream ranking."""

    stage_version: Literal["semantic_stage_v1"] = SEMANTIC_STAGE_VERSION
    guard: SemanticGuardArtifact
    guarded_graph: GuardedGraphArtifact
    degraded: bool
    degradation_code: SemanticDegradationCode | None = None
    agent_failure_code: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_consistent_outcome(self) -> Self:
        if not _outcome_is_consistent(self):
            raise ValueError("semantic stage outcome is inconsistent")
        return self

    def verify(self) -> bool:
        """Verify both nested artifacts, their links, and this stage hash."""

        if not self.guard.verify() or not self.guarded_graph.verify():
            return False
        if not _outcome_is_consistent(self):
            return False
        unsigned = self.model_dump(mode="json", exclude={"artifact_hash"})
        return self.artifact_hash == sha256_hex(canonical_json_bytes(unsigned))


def complete_semantic_fact_stage(
    *,
    preliminary: PreliminaryGraphArtifact,
    payload: SemanticPayloadArtifact,
    response: Mapping[str, Any],
) -> SemanticStageArtifact:
    """Guard a response and degrade only when it has no accepted facts."""

    guard = validate_semantic_fact_response(artifact=payload, response=response)
    degradation_code: SemanticDegradationCode | None = (
        None if guard.accepted_facts else "semantic_facts_empty"
    )
    return _build_stage_artifact(
        preliminary=preliminary,
        payload=payload,
        guard=guard,
        degradation_code=degradation_code,
        agent_failure_code=None,
    )


def degrade_semantic_fact_stage(
    *,
    preliminary: PreliminaryGraphArtifact,
    payload: SemanticPayloadArtifact,
    failure_code: str,
) -> SemanticStageArtifact:
    """Continue deterministically after a sanitized case-local agent failure."""

    guard = validate_semantic_fact_response(
        artifact=payload,
        response={"payload_hash": payload.payload_hash, "facts": []},
    )
    return _build_stage_artifact(
        preliminary=preliminary,
        payload=payload,
        guard=guard,
        degradation_code="semantic_agent_failed",
        agent_failure_code=failure_code,
    )


def _build_stage_artifact(
    *,
    preliminary: PreliminaryGraphArtifact,
    payload: SemanticPayloadArtifact,
    guard: SemanticGuardArtifact,
    degradation_code: SemanticDegradationCode | None,
    agent_failure_code: str | None,
) -> SemanticStageArtifact:
    guarded_graph = compile_guarded_graph(
        preliminary=preliminary,
        payload=payload,
        guard=guard,
    )
    unsigned = {
        "stage_version": SEMANTIC_STAGE_VERSION,
        "guard": guard.model_dump(mode="json"),
        "guarded_graph": guarded_graph.model_dump(mode="json"),
        "degraded": degradation_code is not None,
        "degradation_code": degradation_code,
        "agent_failure_code": agent_failure_code,
    }
    result = SemanticStageArtifact(
        guard=guard,
        guarded_graph=guarded_graph,
        degraded=degradation_code is not None,
        degradation_code=degradation_code,
        agent_failure_code=agent_failure_code,
        artifact_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("semantic stage artifact verification failed")
    return result


def _outcome_is_consistent(artifact: SemanticStageArtifact) -> bool:
    if artifact.guarded_graph.guard_artifact_hash != artifact.guard.artifact_hash:
        return False
    if artifact.guarded_graph.payload_hash != artifact.guard.payload_hash:
        return False
    if artifact.guarded_graph.graph.decisions != artifact.guard.decisions:
        return False
    has_accepted_facts = bool(artifact.guard.accepted_facts)
    if has_accepted_facts:
        return (
            not artifact.degraded
            and artifact.degradation_code is None
            and artifact.agent_failure_code is None
        )
    if not artifact.degraded or artifact.degradation_code is None:
        return False
    if artifact.degradation_code == "semantic_agent_failed":
        return artifact.agent_failure_code is not None
    return artifact.agent_failure_code is None
