"""Temporal propagation inference for Nurse evidence closure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from .evidence_program import EvidenceFact
from .matching import canonical_entity


@dataclass(frozen=True)
class PropagationEdge:
    source: str
    target: str
    confidence: float
    reason: str
    lag_seconds: float | None = None


def _first_seen(fact: EvidenceFact) -> float | None:
    value = fact.attrs.get("first_seen")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _role(fact: EvidenceFact) -> str:
    return str(fact.attrs.get("role") or "")


def infer_temporal_edges(
    facts: Iterable[EvidenceFact],
    topology_edges: Iterable[Tuple[str, str]],
    *,
    max_lag_seconds: float = 120.0,
) -> List[PropagationEdge]:
    source_facts = [
        fact
        for fact in facts
        if fact.predicate == "anomaly" and _role(fact) in {"source_metric", "rule_supports_root"}
    ]
    symptom_facts = [
        fact
        for fact in facts
        if fact.predicate == "anomaly" and _role(fact) in {"symptom_metric", "downstream_symptom"}
    ]
    dependency_edges = {
        (canonical_entity(callee), canonical_entity(caller))
        for caller, callee in topology_edges
        if canonical_entity(caller) and canonical_entity(callee)
    }
    edges: List[PropagationEdge] = []
    seen = set()

    for src_fact in source_facts:
        src = canonical_entity(src_fact.subject)
        src_time = _first_seen(src_fact)
        for dst_fact in symptom_facts:
            dst = canonical_entity(dst_fact.subject)
            if not src or not dst or src == dst or (src, dst) not in dependency_edges:
                continue
            dst_time = _first_seen(dst_fact)
            if src_time is None or dst_time is None:
                lag = None
                confidence = 0.25
            else:
                lag = dst_time - src_time
                if lag < 0 or lag > max_lag_seconds:
                    continue
                confidence = max(0.25, 1.0 - (lag / max(max_lag_seconds, 1.0)) * 0.5)
            key = (src, dst)
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                PropagationEdge(
                    source=src,
                    target=dst,
                    confidence=float(confidence),
                    reason="topology_time_lag",
                    lag_seconds=lag,
                )
            )
    return edges
