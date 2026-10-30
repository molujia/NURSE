"""Pure bounded reverse-dependency paths; original graph records stay unchanged."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from itertools import pairwise

from nurse_rcl.v2.graph_snapshots import GraphSnapshotEdge, GraphSnapshotPair
from nurse_rcl.v2.propagation_scoring import (
    _MAX_CONDENSATION_COMPONENTS,
    _MAX_CONDENSATION_EDGES,
    _MAX_ENUMERATED_PATHS,
    _MAX_SELECTED_ORIGINAL_EDGES,
    PropagationScoringBudgetExceeded,
    _ComputationBudget,
)
from nurse_rcl.v2.temporal_condensation import _scc_from_endpoints


def dependency_edges(graph: GraphSnapshotPair) -> tuple[GraphSnapshotEdge, ...]:
    return tuple(
        e
        for e in graph.fault_snapshot.edges
        if e.observation_state == "observed"
        and (
            (e.relation_type == "service_call" and e.direction == "caller_to_callee")
            or (
                e.relation_type == "received_dependency_failure"
                and e.direction == "callee_to_caller"
            )
        )
    )


class DependencyIndex:
    """Invocation-private index after complete graph/pin replay, not a receipt."""

    def __init__(self, graph: GraphSnapshotPair):
        edges = dependency_edges(graph)
        nodes = graph.fault_snapshot.node_service_keys
        if len(nodes) > _MAX_CONDENSATION_COMPONENTS or len(edges) > _MAX_SELECTED_ORIGINAL_EDGES:
            raise PropagationScoringBudgetExceeded("condensation_structure_budget_exceeded")
        self.components = _scc_from_endpoints(
            nodes=nodes,
            endpoints=tuple((e.callee_service_key, e.caller_service_key) for e in edges),
        )
        self.by_service = {s: i for i, members in enumerate(self.components) for s in members}
        grouped: dict[tuple[int, int], list[GraphSnapshotEdge]] = defaultdict(list)
        for e in edges:
            pair = (self.by_service[e.callee_service_key], self.by_service[e.caller_service_key])
            if pair[0] != pair[1]:
                grouped[pair].append(e)
        if len(grouped) > _MAX_CONDENSATION_EDGES:
            raise PropagationScoringBudgetExceeded("condensation_structure_budget_exceeded")
        self.edges = {
            pair: tuple(sorted(values, key=lambda e: e.edge_id)) for pair, values in grouped.items()
        }
        self.adjacency: dict[int, tuple[int, ...]] = {
            i: tuple(sorted(b for a, b in grouped if a == i)) for i in range(len(self.components))
        }

    def paths(
        self, source: str, target: str, *, budget: _ComputationBudget, maximum_hops: int
    ) -> tuple[tuple[int, ...], ...]:
        left, right = self.by_service[source], self.by_service[target]
        if left == right:
            return ((left,),)
        found = []
        stack: list[tuple[int, ...]] = [(left,)]
        budget.charge_state()
        while stack:
            path = stack.pop()
            if len(path) - 1 >= maximum_hops:
                continue
            for node in reversed(self.adjacency[path[-1]]):
                budget.charge_edge()
                if node in path:
                    continue
                current = (*path, node)
                if node == right:
                    found.append(current)
                    if len(found) > _MAX_ENUMERATED_PATHS:
                        raise PropagationScoringBudgetExceeded("path_count_budget_exceeded")
                else:
                    budget.charge_state()
                    stack.append(current)
        if not found:
            queue: deque[tuple[int, ...]] = deque([(left,)])
            visited = {left}
            budget.charge_state()
            while queue:
                path = queue.popleft()
                for node in self.adjacency[path[-1]]:
                    budget.charge_edge()
                    if node in visited:
                        continue
                    current = (*path, node)
                    if node == right:
                        return (current,)  # Retain an over-hop rejection, never score it.
                    visited.add(node)
                    budget.charge_state()
                    queue.append(current)
        return tuple(sorted(found))

    def hops(self, path: Sequence[int]) -> tuple[tuple[GraphSnapshotEdge, ...], ...]:
        return tuple(self.edges[(a, b)] for a, b in pairwise(path))
