"""Tarjan SCC condensation and auditable propagation scoring view."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex, stable_identifier
from nurse_rcl.methods.paper_v1.guarded_graph import GuardedGraphArtifact
from nurse_rcl.methods.paper_v1.raw_scoring import RawCandidateEPR, RawEPRArtifact
from nurse_rcl.models import GuardedRelation
from nurse_rcl.schemas import StrictFrozenModel

SCC_SCORING_VERSION: Literal["scc_scoring_v1"] = "scc_scoring_v1"


class SCCPropagationComponent(StrictFrozenModel):
    component_id: str = Field(min_length=1)
    members: tuple[str, ...] = Field(min_length=1)
    member_order: tuple[str, ...] = Field(min_length=1)
    propagation_value: float
    removed_relation_id: str | None = None


class CondensationEdge(StrictFrozenModel):
    source_component_id: str = Field(min_length=1)
    target_component_id: str = Field(min_length=1)
    relation_ids: tuple[str, ...] = Field(min_length=1)


class SCCCandidatePropagation(StrictFrozenModel):
    service: str = Field(min_length=1)
    component_id: str = Field(min_length=1)
    local_e: float
    raw_p: float
    p: float
    member_order_index: int = Field(ge=0)


class SCCScoringArtifact(StrictFrozenModel):
    scoring_version: Literal["scc_scoring_v1"] = SCC_SCORING_VERSION
    raw_epr_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_guarded_graph_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_guarded_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    components: tuple[SCCPropagationComponent, ...] = Field(min_length=1)
    condensation_edges: tuple[CondensationEdge, ...]
    scoring_relation_ids: tuple[str, ...]
    candidates: tuple[SCCCandidatePropagation, ...] = Field(min_length=1)
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        """Verify the complete SCC scoring artifact content hash."""

        unsigned = self.model_dump(mode="json", exclude={"artifact_hash"})
        return self.artifact_hash == sha256_hex(canonical_json_bytes(unsigned))


@dataclass(frozen=True, slots=True)
class _DirectedRelation:
    relation_id: str
    source_service: str
    symptom_service: str
    factor: float


def score_scc_propagation(
    *,
    raw: RawEPRArtifact,
    guarded: GuardedGraphArtifact,
) -> SCCScoringArtifact:
    """Build a deterministic Tarjan condensation and share component-level raw P."""

    _verify_input_chain(raw=raw, guarded=guarded)
    raw_by_service = {candidate.service: candidate for candidate in raw.candidates}
    if len(raw_by_service) != len(raw.candidates):
        raise ValueError("raw E/P/R candidate services must be unique")
    factor_by_relation = _factor_by_relation(raw)
    directed_relations = tuple(
        _directed_relation(relation, factor_by_relation)
        for relation in guarded.graph.relations
        if relation.relation_type != "supports_root"
    )
    nodes = set(raw_by_service)
    if any(
        relation.source_service not in nodes or relation.symptom_service not in nodes
        for relation in directed_relations
    ):
        raise ValueError("propagation relation endpoint is outside raw candidates")
    adjacency = {node: set[str]() for node in nodes}
    for relation in directed_relations:
        adjacency[relation.source_service].add(relation.symptom_service)

    member_groups = _tarjan_components(nodes=nodes, adjacency=adjacency)
    component_id_by_members = {
        members: stable_identifier("scc", {"members": list(members)})
        for members in member_groups
    }
    component_by_service = {
        service: component_id_by_members[members]
        for members in member_groups
        for service in members
    }
    removed_by_component = _weakest_internal_relations(
        members=member_groups,
        component_ids=component_id_by_members,
        relations=directed_relations,
    )
    removed_relation_ids = {
        relation_id
        for relation_id in removed_by_component.values()
        if relation_id is not None
    }
    scoring_relation_ids = tuple(
        sorted(
            relation.relation_id
            for relation in directed_relations
            if relation.relation_id not in removed_relation_ids
        )
    )

    components = tuple(
        _component_trace(
            members=members,
            component_id=component_id_by_members[members],
            removed_relation_id=removed_by_component[component_id_by_members[members]],
            raw_by_service=raw_by_service,
        )
        for members in member_groups
    )
    component_trace_by_id = {component.component_id: component for component in components}
    candidates = tuple(
        SCCCandidatePropagation(
            service=service,
            component_id=component_by_service[service],
            local_e=raw_by_service[service].e,
            raw_p=raw_by_service[service].p,
            p=component_trace_by_id[component_by_service[service]].propagation_value,
            member_order_index=component_trace_by_id[
                component_by_service[service]
            ].member_order.index(service),
        )
        for service in sorted(nodes)
    )
    condensation_edges = _condensation_edges(
        relations=directed_relations,
        component_by_service=component_by_service,
    )
    unsigned = {
        "scoring_version": SCC_SCORING_VERSION,
        "raw_epr_hash": raw.artifact_hash,
        "original_guarded_graph_hash": guarded.graph_hash,
        "original_guarded_artifact_hash": guarded.artifact_hash,
        "components": [component.model_dump(mode="json") for component in components],
        "condensation_edges": [
            edge.model_dump(mode="json") for edge in condensation_edges
        ],
        "scoring_relation_ids": list(scoring_relation_ids),
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
    }
    result = SCCScoringArtifact(
        raw_epr_hash=raw.artifact_hash,
        original_guarded_graph_hash=guarded.graph_hash,
        original_guarded_artifact_hash=guarded.artifact_hash,
        components=components,
        condensation_edges=condensation_edges,
        scoring_relation_ids=scoring_relation_ids,
        candidates=candidates,
        artifact_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("SCC scoring artifact verification failed")
    return result


def _verify_input_chain(
    *,
    raw: RawEPRArtifact,
    guarded: GuardedGraphArtifact,
) -> None:
    if not raw.verify():
        raise ValueError("raw E/P/R artifact hash verification failed")
    if not guarded.verify():
        raise ValueError("guarded graph artifact hash verification failed")
    if raw.guarded_graph_hash != guarded.graph_hash:
        raise ValueError("raw E/P/R guarded graph hash does not match")
    if raw.guarded_artifact_hash != guarded.artifact_hash:
        raise ValueError("raw E/P/R guarded artifact hash does not match")
    if raw.preliminary_graph_hash != guarded.graph.preliminary_graph_hash:
        raise ValueError("raw E/P/R preliminary graph hash does not match")


def _factor_by_relation(raw: RawEPRArtifact) -> dict[str, float]:
    factors: dict[str, float] = {}
    roles: dict[str, set[str]] = {}
    for candidate in raw.candidates:
        for contribution in candidate.propagation_contributions:
            previous = factors.setdefault(contribution.relation_id, contribution.factor)
            if not math.isclose(previous, contribution.factor, rel_tol=0.0, abs_tol=0.0):
                raise ValueError("propagation relation has inconsistent factor traces")
            roles.setdefault(contribution.relation_id, set()).add(contribution.role)
    if any(
        relation_roles != {"source_consistency", "symptom_explanation"}
        for relation_roles in roles.values()
    ):
        raise ValueError("propagation relation is missing a symmetric contribution trace")
    return factors


def _directed_relation(
    relation: GuardedRelation,
    factor_by_relation: Mapping[str, float],
) -> _DirectedRelation:
    if relation.target_service is None:
        raise ValueError("propagation relation requires a target service")
    try:
        factor = factor_by_relation[relation.relation_id]
    except KeyError as error:
        raise ValueError("propagation relation is missing its raw factor trace") from error
    if relation.relation_type == "propagates_to":
        source_service = relation.source_service
        symptom_service = relation.target_service
    else:
        source_service = relation.target_service
        symptom_service = relation.source_service
    return _DirectedRelation(
        relation_id=relation.relation_id,
        source_service=source_service,
        symptom_service=symptom_service,
        factor=factor,
    )


def _tarjan_components(
    *,
    nodes: set[str],
    adjacency: Mapping[str, set[str]],
) -> tuple[tuple[str, ...], ...]:
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in sorted(adjacency.get(node, set())):
            if neighbor not in indices:
                visit(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])

        if lowlinks[node] != indices[node]:
            return
        members: list[str] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            members.append(member)
            if member == node:
                break
        components.append(tuple(sorted(members)))

    for node in sorted(nodes):
        if node not in indices:
            visit(node)
    return tuple(sorted(components))


def _weakest_internal_relations(
    *,
    members: tuple[tuple[str, ...], ...],
    component_ids: Mapping[tuple[str, ...], str],
    relations: tuple[_DirectedRelation, ...],
) -> dict[str, str | None]:
    removed: dict[str, str | None] = {}
    for component_members in members:
        component_id = component_ids[component_members]
        if len(component_members) == 1:
            removed[component_id] = None
            continue
        member_set = set(component_members)
        internal = [
            relation
            for relation in relations
            if relation.source_service in member_set
            and relation.symptom_service in member_set
        ]
        if not internal:  # pragma: no cover - impossible for a multi-node SCC
            raise ValueError("multi-service SCC has no internal propagation relation")
        weakest = min(internal, key=lambda relation: (relation.factor, relation.relation_id))
        removed[component_id] = weakest.relation_id
    return removed


def _component_trace(
    *,
    members: tuple[str, ...],
    component_id: str,
    removed_relation_id: str | None,
    raw_by_service: Mapping[str, RawCandidateEPR],
) -> SCCPropagationComponent:
    member_order = tuple(
        sorted(members, key=lambda service: (-raw_by_service[service].e, service))
    )
    propagation_value = math.fsum(raw_by_service[service].p for service in members)
    return SCCPropagationComponent(
        component_id=component_id,
        members=members,
        member_order=member_order,
        propagation_value=propagation_value,
        removed_relation_id=removed_relation_id,
    )


def _condensation_edges(
    *,
    relations: tuple[_DirectedRelation, ...],
    component_by_service: Mapping[str, str],
) -> tuple[CondensationEdge, ...]:
    relation_ids_by_edge: dict[tuple[str, str], list[str]] = {}
    for relation in relations:
        source_component = component_by_service[relation.source_service]
        target_component = component_by_service[relation.symptom_service]
        if source_component == target_component:
            continue
        relation_ids_by_edge.setdefault((source_component, target_component), []).append(
            relation.relation_id
        )
    return tuple(
        CondensationEdge(
            source_component_id=source_component,
            target_component_id=target_component,
            relation_ids=tuple(sorted(relation_ids)),
        )
        for (source_component, target_component), relation_ids in sorted(
            relation_ids_by_edge.items()
        )
    )
