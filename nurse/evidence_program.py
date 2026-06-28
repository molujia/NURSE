"""Probabilistic evidence facts and root-cause queries for Nurse."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class EvidenceFact:
    predicate: str
    subject: str
    object: str
    weight: float
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvidenceFact":
        return cls(
            predicate=str(payload.get("predicate") or ""),
            subject=str(payload.get("subject") or ""),
            object=str(payload.get("object") or ""),
            weight=float(payload.get("weight", 0.0) or 0.0),
            attrs=dict(payload.get("attrs") or {}),
        )


@dataclass
class EvidenceQueryResult:
    ranking: List[str]
    scores: Dict[str, float]
    explanations: Dict[str, List[str]]


class EvidenceProgram:
    """Small weighted logic layer for root-cause evidence.

    This is deliberately deterministic: neural/LLM components may propose
    facts, but the program only scores facts that survived verification.
    """

    SOURCE_ROLES = {"source_metric", "rule_supports_root", "counterfactual_drop"}
    SYMPTOM_ROLES = {"symptom_metric", "downstream_symptom"}
    DOWNSTREAM_PENALTY_ROLES = {"downstream_symptom"}
    CONTRADICTION_ROLES = {"unsupported_proposal", "contradiction"}

    def query_root_probabilities(self, facts: Iterable[EvidenceFact]) -> EvidenceQueryResult:
        scores: Dict[str, float] = defaultdict(float)
        explanations: Dict[str, List[str]] = defaultdict(list)
        propagation: List[EvidenceFact] = []

        for fact in facts:
            if not fact.subject:
                continue
            if fact.predicate == "propagates_to":
                propagation.append(fact)
                continue
            role = str(fact.attrs.get("role") or "generic")
            weight = float(fact.weight or 0.0)
            if not math.isfinite(weight):
                weight = 0.0
            if role in self.SOURCE_ROLES:
                multiplier = 1.4
            elif role in self.DOWNSTREAM_PENALTY_ROLES:
                multiplier = -0.35
            elif role in self.SYMPTOM_ROLES:
                multiplier = 0.45
            elif role in self.CONTRADICTION_ROLES or fact.predicate == "contradiction":
                multiplier = -1.0
            else:
                multiplier = 0.8
            scores[fact.subject] += weight * multiplier
            explanations[fact.subject].append(role)

        for fact in propagation:
            src = fact.subject
            dst = fact.object
            if not src or not dst:
                continue
            confidence = float(fact.weight or fact.attrs.get("confidence", 1.0) or 1.0)
            if dst in scores:
                scores[src] += 0.6 * confidence
                scores[dst] -= 0.3 * confidence
                explanations[src].append(f"explains:{dst}")
                explanations[dst].append(f"symptom_of:{src}")

        ranked = sorted(scores, key=lambda service: (-scores[service], service))
        return EvidenceQueryResult(
            ranking=ranked,
            scores={service: float(scores[service]) for service in ranked},
            explanations={service: list(explanations[service]) for service in ranked},
        )
