"""Convert LLM hypotheses/rules into bounded, verifiable Nurse facts."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set

from .evidence_program import EvidenceFact
from .matching import canonical_entity


def _candidate_lookup(service_candidates: Iterable[str]) -> Dict[str, str]:
    lookup = {}
    for service in service_candidates:
        canonical = canonical_entity(str(service))
        if canonical:
            lookup[canonical] = canonical
    return lookup


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.5


def proposal_to_facts(
    proposal: Dict[str, Any],
    *,
    service_candidates: Iterable[str],
    verified_evidence_ids: Iterable[str],
    confidence_scale: float = 0.25,
    max_weight: float = 0.25,
) -> List[EvidenceFact]:
    allowed = _candidate_lookup(service_candidates)
    verified_ids: Set[str] = {str(item) for item in verified_evidence_ids}
    facts: List[EvidenceFact] = []

    for item in proposal.get("hypotheses", []) or []:
        if not isinstance(item, dict):
            continue
        service = canonical_entity(str(item.get("service") or item.get("entity") or ""))
        if service not in allowed:
            continue
        support_ids = [str(value) for value in item.get("supporting_evidence_ids", []) or []]
        if not support_ids:
            continue
        if any(evidence_id not in verified_ids for evidence_id in support_ids):
            continue
        bounded_weight = min(_confidence(item.get("confidence", 0.5)) * confidence_scale, max_weight)
        facts.append(
            EvidenceFact(
                "rule_support",
                service,
                "llm_hypothesis",
                bounded_weight,
                {
                    "role": "rule_supports_root",
                    "source": "llm_rule_induction",
                    "supporting_evidence_ids": support_ids,
                    "reason": str(item.get("reason") or item.get("rationale") or "")[:500],
                },
            )
        )

    return facts
