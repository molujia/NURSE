"""Guarded semantic facts produced by the LLM before ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .llm_payload import semantic_rank_role_policy_for_dataset
from .matching import canonical_entity
from .llm_payload import semantic_role_for_service


ALLOWED_FACT_TYPES = {
    "root_support",
    "source_support",
    "symptom_support",
}


@dataclass(frozen=True)
class GuardedSemanticFact:
    service: str
    fact_type: str
    supporting_evidence_ids: tuple[str, ...]
    confidence: float
    source: str


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: List[str] = []
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _raw_items(proposal: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
    raw = proposal.get("semantic_facts")
    if isinstance(raw, list) and raw:
        return [item for item in raw if isinstance(item, dict)]
    fallback = proposal.get("hypotheses")
    if isinstance(fallback, list):
        return [item for item in fallback if isinstance(item, dict)]
    return []


def _normalize_fact_type(value: Any) -> str:
    fact_type = str(value or "").strip().lower()
    if not fact_type:
        return ""
    if fact_type in ALLOWED_FACT_TYPES:
        return fact_type
    if "symptom" in fact_type or "downstream" in fact_type:
        return "symptom_support"
    if "topology" in fact_type or "propagat" in fact_type or "dependency" in fact_type:
        return "source_support"
    if "root" in fact_type or "cause" in fact_type:
        return "root_support"
    if any(
        keyword in fact_type
        for keyword in (
            "anomal",
            "deviation",
            "metric",
            "resource",
            "usage",
            "utilization",
            "performance",
            "elevat",
            "abnormal",
            "cpu",
            "memory",
            "disk",
            "fault_window",
        )
    ):
        return "source_support"
    return ""


def _source_support_for(service: str, score_breakdown: Dict[str, Dict[str, float]]) -> float:
    parts = score_breakdown.get(service)
    if not isinstance(parts, dict):
        return 0.0
    try:
        return float(parts.get("source_evidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _discouraged_fact_types_for_dataset(dataset: str) -> set[str]:
    normalized = str(dataset or "").strip().lower()
    if normalized == "aiops25-retained-nonnode":
        return {
            "root_support",
            "source_support",
        }
    if normalized == "rcaeval-tt":
        return {
            "root_support",
            "source_support",
        }
    return set()


def normalize_and_guard_semantic_facts(
    *,
    proposal: Dict[str, Any],
    service_candidates: Iterable[str],
    verified_evidence_ids: Iterable[str],
    score_breakdown: Dict[str, Dict[str, float]] | None = None,
    dataset: str = "",
) -> Tuple[List[GuardedSemanticFact], List[Dict[str, Any]]]:
    allowed_services = {
        canonical_entity(str(service))
        for service in service_candidates
        if canonical_entity(str(service))
    }
    verified_ids = {str(item) for item in verified_evidence_ids}
    score_breakdown = score_breakdown or {}
    role_policy = semantic_rank_role_policy_for_dataset(dataset)
    discouraged_fact_types = _discouraged_fact_types_for_dataset(dataset)

    accepted: List[GuardedSemanticFact] = []
    rejected: List[Dict[str, Any]] = []

    for raw in _raw_items(proposal or {}):
        service = canonical_entity(
            str(
                raw.get("service")
                or raw.get("entity")
                or raw.get("subject")
                or ""
            )
        )
        raw_fact_type = str(raw.get("fact_type") or raw.get("type") or raw.get("role") or "root_support").strip().lower()
        fact_type = _normalize_fact_type(raw_fact_type)
        evidence_ids = _ordered_unique(
            raw.get("supporting_evidence_ids")
            or raw.get("evidence_ids")
            or raw.get("support_ids")
            or []
        )
        confidence = _confidence(raw.get("confidence", 0.5))
        source = str(raw.get("source") or raw.get("provenance") or raw.get("reason") or "llm_semantic_rank").strip()

        if not service:
            rejected.append({"reason": "missing_service", "raw": dict(raw)})
            continue
        if allowed_services and service not in allowed_services:
            rejected.append({"service": service, "reason": "outside_candidate_set"})
            continue
        service_role = semantic_role_for_service(service, dataset)
        if service_role in role_policy["discouraged_target_roles"] and fact_type in discouraged_fact_types:
            rejected.append(
                {
                    "service": service,
                    "reason": "discouraged_target_role",
                    "semantic_role": service_role,
                    "fact_type": fact_type,
                }
            )
            continue
        if fact_type not in ALLOWED_FACT_TYPES:
            rejected.append({"service": service, "reason": "unsupported_fact_type", "fact_type": raw_fact_type})
            continue
        if not evidence_ids:
            rejected.append({"service": service, "reason": "missing_supporting_evidence", "fact_type": fact_type})
            continue
        if any(evidence_id not in verified_ids for evidence_id in evidence_ids):
            rejected.append(
                {
                    "service": service,
                    "reason": "unverified_evidence",
                    "supporting_evidence_ids": list(evidence_ids),
                }
            )
            continue
        if confidence < 0.5:
            rejected.append({"service": service, "reason": "low_confidence", "confidence": confidence})
            continue
        score_gap = raw.get("score_gap_from_top")
        if score_gap is not None:
            try:
                if float(score_gap) > 3.0:
                    rejected.append(
                        {
                            "service": service,
                            "reason": "score_gap_too_large",
                            "score_gap_from_top": float(score_gap),
                        }
                    )
                    continue
            except (TypeError, ValueError):
                pass
        if fact_type == "symptom_support" and _source_support_for(service, score_breakdown) <= 0.0:
            rejected.append({"service": service, "reason": "symptom_only_without_source_support"})
            continue
        if len(accepted) >= 3:
            rejected.append({"service": service, "reason": "too_many_semantic_facts"})
            continue

        accepted.append(
            GuardedSemanticFact(
                service=service,
                fact_type=fact_type,
                supporting_evidence_ids=evidence_ids,
                confidence=confidence,
                source=source,
            )
        )

    return accepted, rejected


def semantic_fact_score(fact: GuardedSemanticFact, *, prior_weight: float = 1.0) -> float:
    fact_type_scale = {
        "root_support": 1.0,
        "source_support": 0.85,
        "symptom_support": 0.65,
    }.get(fact.fact_type, 1.0)
    evidence_bonus = 1.0 + min(max(len(fact.supporting_evidence_ids) - 1, 0), 2) * 0.05
    return max(float(prior_weight) * fact.confidence * fact_type_scale * evidence_bonus, 0.0)
