"""Verified neural rank repair for Nurse.

The LLM can propose a bounded reorder, but this module is the symbolic guard:
only candidates already present in the no-LLM ranking and backed by verified
evidence IDs may move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Set

from .matching import canonical_entity




@dataclass
class SlotRescueResult:
    ranking: List[str]
    applied: bool = False
    promoted_service: str = ""
    target_rank: int = 0
    reason: str = ""

@dataclass
class RankRepairResult:
    ranking: List[str]
    verified_repairs: List[Dict[str, Any]] = field(default_factory=list)
    rejected_repairs: List[Dict[str, Any]] = field(default_factory=list)


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(number, 1.0))


def _proposal_service(proposal: Dict[str, Any]) -> str:
    return canonical_entity(
        str(
            proposal.get("promote")
            or proposal.get("service")
            or proposal.get("entity")
            or ""
        )
    )


def _proposal_evidence_ids(proposal: Dict[str, Any]) -> List[str]:
    raw = (
        proposal.get("supporting_evidence_ids")
        or proposal.get("evidence_ids")
        or proposal.get("support_ids")
        or []
    )
    return [str(item) for item in raw if str(item)]


def _target_index(proposal: Dict[str, Any]) -> int:
    try:
        requested = int(proposal.get("to_rank", 1))
    except (TypeError, ValueError):
        requested = 1
    return max(0, requested - 1)


def _reject(proposal: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {**dict(proposal), "reason": reason}



def apply_primary_service_slot_rescue(
    ranking: Sequence[str],
    *,
    primary_services: Iterable[str],
    top_k: int = 5,
    search_k: int = 10,
) -> SlotRescueResult:
    """Move the highest-ranked primary service into the last Top-K slot.

    This is intentionally conservative: it never changes Top-1, and it only
    fires when the current Top-K contains no primary service at all.
    """

    normalized_ranking = [canonical_entity(str(item)) for item in ranking if canonical_entity(str(item))]
    primary = {canonical_entity(str(item)) for item in primary_services if canonical_entity(str(item))}
    top_limit = max(1, int(top_k))
    search_limit = max(top_limit, int(search_k))
    if not normalized_ranking or not primary:
        return SlotRescueResult(ranking=normalized_ranking)
    if any(service in primary for service in normalized_ranking[:top_limit]):
        return SlotRescueResult(ranking=normalized_ranking)
    for index, service in enumerate(normalized_ranking[top_limit:search_limit], start=top_limit):
        if service not in primary:
            continue
        repaired = list(normalized_ranking)
        repaired.pop(index)
        repaired.insert(top_limit - 1, service)
        return SlotRescueResult(
            ranking=repaired,
            applied=True,
            promoted_service=service,
            target_rank=top_limit,
            reason="primary_service_absent_from_top_k",
        )
    return SlotRescueResult(ranking=normalized_ranking)


def apply_symptom_family_top3_rescue(
    ranking: Sequence[str],
    *,
    symptom_family: Iterable[str],
    blocked_prefixes: Iterable[str] | None = None,
    top_k: int = 3,
) -> SlotRescueResult:
    """Promote a nearby application candidate above a symptom-family Top-1.

    The guard is intentionally narrow: it only fires when the current Top-1 is
    a known symptom-prone service and the replacement is already visible in the
    first ``top_k`` ranked candidates. It never invents candidates and never
    searches deeper than the bounded neural repair context.
    """

    normalized_ranking = [canonical_entity(str(item)) for item in ranking if canonical_entity(str(item))]
    symptom_services = {
        canonical_entity(str(item))
        for item in symptom_family
        if canonical_entity(str(item))
    }
    blocked = tuple(str(prefix) for prefix in (blocked_prefixes or []) if str(prefix))
    search_limit = max(1, int(top_k))
    if not normalized_ranking or normalized_ranking[0] not in symptom_services:
        return SlotRescueResult(ranking=normalized_ranking)

    for index, service in enumerate(normalized_ranking[1:search_limit], start=1):
        if service in symptom_services:
            continue
        if blocked and service.startswith(blocked):
            continue
        repaired = list(normalized_ranking)
        repaired.pop(index)
        repaired.insert(0, service)
        return SlotRescueResult(
            ranking=repaired,
            applied=True,
            promoted_service=service,
            target_rank=1,
            reason="symptom_family_top1_with_top3_application_candidate",
        )
    return SlotRescueResult(ranking=normalized_ranking)
def apply_verified_rank_repair(
    *,
    original_ranking: Sequence[str],
    proposals: Iterable[Dict[str, Any]],
    verified_evidence_ids: Iterable[str],
    repair_top_k: int = 20,
    min_confidence: float = 0.55,
    max_repairs: int = 2,
    blocked_prefixes: Iterable[str] | None = None,
    blocked_services: Iterable[str] | None = None,
    scores: Dict[str, float] | None = None,
    score_breakdown: Dict[str, Dict[str, float]] | None = None,
    max_score_gap_from_top: float | None = None,
    score_gap_bypass_services: Iterable[str] | None = None,
    score_gap_bypass_protected_top1_services: Iterable[str] | None = None,
    score_gap_bypass_min_target_rank: int = 2,
    min_source_evidence_for_top1_repair: float | None = None,
    reject_symptom_penalty_at_or_below: float | None = None,
) -> RankRepairResult:
    """Apply bounded LLM reorder proposals under symbolic verification.

    The repair candidate set is the current no-LLM top-K ranking. This prevents
    free-form LLM generation from injecting unseen services while still letting
    the neural layer correct symptom-vs-root ordering mistakes.
    """

    ranking = [canonical_entity(str(item)) for item in original_ranking if canonical_entity(str(item))]
    allowed = set(ranking[: max(1, int(repair_top_k))])
    verified: Set[str] = {str(item) for item in verified_evidence_ids}
    blocked = tuple(str(prefix) for prefix in (blocked_prefixes or []) if str(prefix))
    blocked_service_set = {canonical_entity(str(item)) for item in (blocked_services or []) if canonical_entity(str(item))}
    score_map = {canonical_entity(str(key)): float(value) for key, value in (scores or {}).items()}
    gap_bypass_services = {canonical_entity(str(item)) for item in (score_gap_bypass_services or []) if canonical_entity(str(item))}
    gap_bypass_protected_top1_services = {
        canonical_entity(str(item))
        for item in (score_gap_bypass_protected_top1_services or [])
        if canonical_entity(str(item))
    }
    gap_bypass_min_rank = max(1, int(score_gap_bypass_min_target_rank or 1))
    breakdown_map: Dict[str, Dict[str, float]] = {}
    for key, parts in (score_breakdown or {}).items():
        if not isinstance(parts, dict):
            continue
        service_key = canonical_entity(str(key))
        if not service_key:
            continue
        breakdown_map[service_key] = {
            str(part): float(value)
            for part, value in parts.items()
            if isinstance(value, (int, float))
        }
    top_score = score_map.get(ranking[0]) if ranking else None
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for raw_proposal in proposals or []:
        if not isinstance(raw_proposal, dict):
            continue
        proposal = dict(raw_proposal)
        service = _proposal_service(proposal)
        if not service:
            rejected.append(_reject(proposal, "missing_service"))
            continue
        if service not in allowed:
            rejected.append(_reject(proposal, "outside_repair_candidate_set"))
            continue
        if service in blocked_service_set or (blocked and service.startswith(blocked)):
            rejected.append(_reject(proposal, "blocked_service"))
            continue
        if _confidence(proposal.get("confidence", 0.0)) < min_confidence:
            rejected.append(_reject(proposal, "low_confidence"))
            continue
        evidence_ids = _proposal_evidence_ids(proposal)
        if not evidence_ids:
            rejected.append(_reject(proposal, "missing_supporting_evidence"))
            continue
        if any(evidence_id not in verified for evidence_id in evidence_ids):
            rejected.append(_reject(proposal, "unverified_evidence"))
            continue
        if max_score_gap_from_top is not None and top_score is not None:
            service_score = score_map.get(service)
            target_rank = _target_index(proposal) + 1
            top1_service = ranking[0] if ranking else ""
            bypass_score_gap = (
                service in gap_bypass_services
                and target_rank >= gap_bypass_min_rank
                and top1_service not in gap_bypass_protected_top1_services
            )
            if (
                service_score is not None
                and service_score < top_score - float(max_score_gap_from_top)
                and not bypass_score_gap
            ):
                rejected.append(_reject(proposal, "score_gap_too_large"))
                continue
        if (
            min_source_evidence_for_top1_repair is not None
            and reject_symptom_penalty_at_or_below is not None
            and ranking
            and _target_index(proposal) == 0
            and service != ranking[0]
        ):
            service_score = score_map.get(service)
            promoted_breakdown = breakdown_map.get(service, {})
            promoted_source = float(promoted_breakdown.get("source_evidence", 0.0))
            promoted_symptom_penalty = float(promoted_breakdown.get("symptom_penalty", 0.0))
            if (
                (top_score is None or service_score is None or service_score <= top_score)
                and promoted_source <= float(min_source_evidence_for_top1_repair)
                and promoted_symptom_penalty <= float(reject_symptom_penalty_at_or_below)
            ):
                rejected.append(_reject(proposal, "weak_source_symptom_candidate"))
                continue
        if service not in ranking:
            rejected.append(_reject(proposal, "outside_ranking"))
            continue

        ranking.remove(service)
        ranking.insert(min(_target_index(proposal), len(ranking)), service)
        accepted.append(
            {
                **proposal,
                "promote": service,
                "supporting_evidence_ids": evidence_ids,
                "target_rank": _target_index(proposal) + 1,
            }
        )
        if len(accepted) >= max_repairs:
            break

    return RankRepairResult(
        ranking=ranking,
        verified_repairs=accepted,
        rejected_repairs=rejected,
    )
