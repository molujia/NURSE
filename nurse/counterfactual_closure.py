"""Counterfactual evidence drop scoring for NURSE root ranking."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Set

from .closure import classify_signal
from .evidence_program import EvidenceFact
from .matching import canonical_entity
from .schemas import EvidenceGraph


SOURCE_ROLES = {"source_metric", "rule_supports_root", "counterfactual_drop"}
SYMPTOM_ROLES = {"symptom_metric", "downstream_symptom"}
CONTRADICTION_ROLES = {"unsupported_proposal", "contradiction"}


def _role(fact: EvidenceFact) -> str:
    return str(fact.attrs.get("role") or "")


def _services(facts: Iterable[EvidenceFact]) -> Set[str]:
    services: Set[str] = set()
    for fact in facts:
        if fact.subject:
            services.add(fact.subject)
        if fact.predicate == "propagates_to" and fact.object:
            services.add(fact.object)
    return services


def _explainability(facts: List[EvidenceFact]) -> float:
    source_services = {fact.subject for fact in facts if _role(fact) in SOURCE_ROLES}
    symptom_services = {fact.subject for fact in facts if _role(fact) in SYMPTOM_ROLES}
    score = 0.0
    for fact in facts:
        role = _role(fact)
        weight = float(fact.weight or 0.0)
        if not math.isfinite(weight):
            weight = 0.0
        weight = max(weight, 0.0)
        if fact.predicate == "contradiction" or role in CONTRADICTION_ROLES:
            score -= weight
        elif role in SOURCE_ROLES:
            score += weight * 1.2
        elif role in SYMPTOM_ROLES:
            score += weight * 0.4
        elif fact.predicate != "propagates_to":
            score += weight * 0.6

    for fact in facts:
        if fact.predicate != "propagates_to":
            continue
        if fact.subject in source_services and fact.object in symptom_services:
            score += max(float(fact.weight or 1.0), 0.0) * 2.0
    return score


def compute_counterfactual_drops(facts: Iterable[EvidenceFact]) -> Dict[str, float]:
    fact_list = list(facts)
    original = _explainability(fact_list)
    drops: Dict[str, float] = {}
    for service in _services(fact_list):
        ablated = [
            fact
            for fact in fact_list
            if not (
                fact.subject == service
                or (fact.predicate == "propagates_to" and fact.subject == service)
            )
        ]
        drops[service] = max(original - _explainability(ablated), 0.0)
    return drops


def _role_from_signal(signal: str) -> str:
    klass = classify_signal(signal)
    if klass == "source_like":
        return "source_metric"
    if klass == "symptom_like":
        return "symptom_metric"
    return "generic_metric"


def facts_from_graph(graph: EvidenceGraph) -> List[EvidenceFact]:
    facts: List[EvidenceFact] = []
    for evidence in graph.evidences:
        if not evidence.verified or evidence.kind == "topology":
            continue
        entity = canonical_entity(evidence.entity)
        if not entity:
            continue
        role = str(evidence.claim.get("metric_role") or _role_from_signal(evidence.signal))
        try:
            weight = float(
                evidence.claim.get("evidence_program_weight")
                or evidence.claim.get("anomaly_score")
                or evidence.observed.get("matching_points")
                or 1.0
            )
        except (TypeError, ValueError):
            weight = 1.0
        if not math.isfinite(weight):
            weight = 0.0
        facts.append(
            EvidenceFact(
                "anomaly",
                entity,
                evidence.signal,
                max(weight, 0.0),
                {
                    "role": role,
                    "evidence_id": evidence.evidence_id,
                    "first_seen": evidence.claim.get("first_seen"),
                    "verification_rule": evidence.verification_rule,
                },
            )
        )

    for relation in graph.relations:
        if relation.get("type") == "propagates_to":
            src = canonical_entity(str(relation.get("src") or ""))
            dst = canonical_entity(str(relation.get("dst") or ""))
            if not src or not dst:
                continue
            facts.append(
                EvidenceFact(
                    "propagates_to",
                    src,
                    dst,
                    float(relation.get("confidence") or 1.0),
                    {"role": "propagation", "via": relation.get("via", "")},
                )
            )
        elif relation.get("type") == "verified_by_sop":
            src = canonical_entity(str(relation.get("src") or ""))
            if not src:
                continue
            facts.append(
                EvidenceFact(
                    "rule_support",
                    src,
                    str(relation.get("signal") or "sop"),
                    1.0,
                    {"role": "rule_supports_root"},
                )
            )

    unsupported_counts: Dict[str, int] = defaultdict(int)
    for evidence in graph.evidences:
        if not evidence.verified and evidence.provenance.get("source") == "llm_proposal":
            entity = canonical_entity(evidence.entity)
            if entity:
                unsupported_counts[entity] += 1
    for service, count in unsupported_counts.items():
        facts.append(
            EvidenceFact(
                "contradiction",
                service,
                "unsupported_root",
                float(count),
                {"role": "unsupported_proposal"},
            )
        )
    return facts
