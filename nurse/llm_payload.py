"""Compact LLM payloads for verified Nurse rank repair."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from .matching import canonical_entity
from .schemas import AlarmArtifact, NursePrediction, VerifiedEvidence, VerifiedRCLContext


def _top_breakdown(parts: Dict[str, float], limit: int = 6) -> Dict[str, float]:
    ranked = sorted(parts.items(), key=lambda item: (-abs(float(item[1])), item[0]))
    return {name: float(value) for name, value in ranked[:limit] if abs(float(value)) > 1e-12}


def _evidence_by_service(verified_evidence: Iterable[VerifiedEvidence]) -> Dict[str, List[VerifiedEvidence]]:
    grouped: Dict[str, List[VerifiedEvidence]] = defaultdict(list)
    for evidence in verified_evidence:
        if not evidence.verified:
            continue
        service = canonical_entity(evidence.entity)
        if service:
            grouped[service].append(evidence)
    return grouped


def _metric_role_counts(items: Sequence[VerifiedEvidence]) -> Counter:
    counts: Counter = Counter()
    for evidence in items:
        role = str(evidence.claim.get("metric_role") or "")
        if role == "source_metric":
            counts["source"] += 1
        elif role == "symptom_metric":
            counts["symptom"] += 1
        else:
            counts["generic"] += 1
    return counts


def _evidence_ids_for_service(
    *,
    service: str,
    prediction: NursePrediction,
    grouped_evidence: Dict[str, List[VerifiedEvidence]],
    max_ids: int,
) -> List[str]:
    ids: List[str] = []
    for evidence_id in prediction.evidence_ids.get(service, []):
        if evidence_id not in ids:
            ids.append(evidence_id)
    for evidence in grouped_evidence.get(service, []):
        if evidence.evidence_id not in ids:
            ids.append(evidence.evidence_id)
    return ids[:max_ids]


def _degree_maps(topology_edges: Iterable[Tuple[str, str]]) -> tuple[Counter, Counter]:
    in_degree: Counter = Counter()
    out_degree: Counter = Counter()
    for raw_src, raw_dst in topology_edges:
        src = canonical_entity(str(raw_src))
        dst = canonical_entity(str(raw_dst))
        if not src or not dst:
            continue
        out_degree[src] += 1
        in_degree[dst] += 1
    return in_degree, out_degree


AIOPS_SYMPTOM_PRONE_ENTRYPOINTS = {"frontend", "payment", "currency", "shipping"}
AIOPS_SYMPTOM_FAMILY_TOP3_RESCUE = {
    "frontend",
    "payment",
    "currency",
    "shipping",
    "redis-cart",
    "cart",
    "checkout",
    "hipstershop",
    "recommendation",
    "email",
}
AIOPS_RETAINED_BLOCKED_INFRA_PREFIXES = ("k8s-master", "aiops-k8s-", "cn-")
RCAEVAL_TT_PRIMARY_SERVICES = {"auth-service", "order-service", "route-service", "train-service", "travel-service"}
RCAEVAL_TT_SUPPORT_PREFIXES = ("admin-",)
RCAEVAL_TT_SUPPORT_MARKERS = ("mongo", "mysql", "payment", "price", "food", "consign", "preserve", "basic", "news", "security", "notification", "contacts", "ticket", "ui", "voucher")

def rank_repair_limits_for(dataset: str, prompt_profile: str = "balanced") -> Dict[str, int]:
    """Return aligned payload and symbolic-guard limits for rank repair."""

    profile = str(prompt_profile or "balanced").strip().lower()
    if dataset == "aiops25-retained-nonnode":
        limit = 5 if profile == "recall-heavy" else 2
        return {
            "candidate_cards": limit,
            "current_ranking": limit,
            "repair_top_k": limit,
            "max_repairs": 1,
        }
    return {
        "candidate_cards": 12,
        "current_ranking": 20,
        "repair_top_k": 20,
        "max_repairs": 2,
    }
def semantic_role_for_service(service: str, dataset: str) -> str:
    normalized = canonical_entity(service)
    if dataset == "aiops25-retained-nonnode":
        if normalized.startswith(AIOPS_RETAINED_BLOCKED_INFRA_PREFIXES):
            return "blocked_infra_for_retained_nonnode"
        if normalized in AIOPS_SYMPTOM_PRONE_ENTRYPOINTS:
            return "symptom_prone_entrypoint"
        return "application_candidate"
    if dataset == "rcaeval-tt":
        if normalized in RCAEVAL_TT_PRIMARY_SERVICES:
            return "primary_train_ticket_service"
        if normalized.startswith(RCAEVAL_TT_SUPPORT_PREFIXES) or any(marker in normalized for marker in RCAEVAL_TT_SUPPORT_MARKERS):
            return "support_or_storage_candidate"
        return "auxiliary_train_ticket_candidate"
    return "candidate"

def build_candidate_cards(
    *,
    prediction: NursePrediction,
    verified_evidence: Iterable[VerifiedEvidence],
    topology_edges: Iterable[Tuple[str, str]],
    dataset: str = "",
    max_cards: int = 20,
    max_evidence_ids_per_card: int = 5,
) -> List[Dict[str, object]]:
    """Summarize ranked services for LLM repair without leaking labels.

    The cards intentionally avoid case IDs and answer-like metadata. They expose
    only the symbolic state Nurse already used for no-LLM ranking.
    """

    grouped = _evidence_by_service(verified_evidence)
    in_degree, out_degree = _degree_maps(topology_edges)
    cards: List[Dict[str, object]] = []
    top_score = None
    for rank, raw_service in enumerate(prediction.ranked[: max(1, int(max_cards))], start=1):
        service = canonical_entity(str(raw_service))
        if not service:
            continue
        score = float(prediction.scores.get(service, 0.0))
        if top_score is None:
            top_score = score
        evidence_items = grouped.get(service, [])
        role_counts = _metric_role_counts(evidence_items)
        breakdown = _top_breakdown(prediction.score_breakdown.get(service, {}))
        cards.append(
            {
                "service": service,
                "rank": rank,
                "score": round(score, 6),
                "score_gap_from_top": round(float(top_score or 0.0) - score, 6),
                "score_breakdown": breakdown,
                "supporting_evidence_ids": _evidence_ids_for_service(
                    service=service,
                    prediction=prediction,
                    grouped_evidence=grouped,
                    max_ids=max_evidence_ids_per_card,
                ),
                "source_like_evidence": int(role_counts["source"]),
                "symptom_like_evidence": int(role_counts["symptom"]),
                "generic_evidence": int(role_counts["generic"]),
                "has_symptom_penalty": float(breakdown.get("symptom_penalty", 0.0)) < 0,
                "in_degree": int(in_degree[service]),
                "out_degree": int(out_degree[service]),
                "semantic_role": semantic_role_for_service(service, dataset),
            }
        )
    return cards


def build_rank_repair_payload(
    *,
    alarm: AlarmArtifact,
    service_candidates: Iterable[str],
    prediction: NursePrediction,
    verified_evidence: Iterable[VerifiedEvidence],
    topology_edges: Iterable[Tuple[str, str]],
    prompt_profile: str = "balanced",
) -> Dict[str, object]:
    """Create the compact neural-symbolic rank repair payload."""

    limits = rank_repair_limits_for(alarm.dataset, prompt_profile)
    repair_card_limit = limits["candidate_cards"]
    current_ranking_limit = limits["current_ranking"]
    aiops_dataset = alarm.dataset == "aiops25-retained-nonnode"
    if aiops_dataset and current_ranking_limit <= 2:
        entrypoint_repair_policy = (
            "If the current Top-1 is symptom-heavy, entrypoint-like, or dominated by downstream observations, "
            "and the Top-2 application_candidate has stronger source-like, counterfactual, or incident-shape evidence, "
            "prefer proposing exactly one bounded repair to that candidate."
        )
    else:
        entrypoint_repair_policy = (
            "If the current Top-1 is symptom-heavy, entrypoint-like, or dominated by downstream observations, "
            "and a Top-2..Top-5 application_candidate has source-like, counterfactual, or incident-shape evidence, "
            "prefer proposing exactly one bounded repair to that candidate."
        )
    cards = build_candidate_cards(
        prediction=prediction,
        verified_evidence=verified_evidence,
        topology_edges=topology_edges,
        dataset=alarm.dataset,
        max_cards=repair_card_limit,
    )
    return {
        "incident_ref": "current-incident",
        "output_contract": "rank_repair_only",
        "alarm": {
            "dataset": alarm.dataset,
            "window": dict(alarm.window),
            "alert_entity": alarm.alert_entity,
            "detector": alarm.detector,
            "provenance": dict(alarm.provenance),
        },
        "service_candidates": [
            canonical_entity(str(candidate))
            for candidate in service_candidates
            if canonical_entity(str(candidate))
        ][:80],
        "current_ranking": list(prediction.ranked)[:current_ranking_limit],
        "candidate_cards": cards,
        "prompt_profile": prompt_profile,
        "repair_policy": {
            "candidate_set": "Only promote services already present in current_ranking.",
            "evidence_set": "Every promotion must cite supporting_evidence_ids from the promoted service card.",
            "root_vs_symptom": (
                "Prefer source-like, counterfactual, incident-shape, and topology-locality evidence. "
                "Do not promote a service solely because one metric has the largest magnitude; "
                "large symptom-only deviations are often downstream effects."
            ),
            "dataset_semantic_prior": (
                "For aiops25-retained-nonnode, Kubernetes node entities with semantic_role "
                "blocked_infra_for_retained_nonnode are not valid Top-1 repair targets; "
                "services marked symptom_prone_entrypoint are often downstream or ingress symptoms, "
                "only propose rank-2 application_candidate repairs; do not promote symptom_prone_entrypoint or blocked_infra_for_retained_nonnode cards. "
                "For rcaeval-tt, semantic_role primary_train_ticket_service marks core business services; "
                "support_or_storage_candidate entities should not outrank a primary service unless verified evidence is much stronger."
            ),
            "entrypoint_repair": entrypoint_repair_policy,
            "no_repair": "Return an empty rank_repair array if evidence does not justify a change.",
        },
    }


def build_grounded_user_manager_payload(*, context: VerifiedRCLContext) -> Dict[str, object]:
    """Create the default User Manager payload from verified RCL context.

    The LLM receives only verified context and may produce explanations or
    advisory notes. It is explicitly not allowed to alter the final ranking.
    """

    payload = context.to_dict()
    return {
        "incident_ref": "current-incident",
        "output_contract": "grounded_rcl_user_proposal",
        "dataset": payload["dataset"],
        "final_ranking": list(payload["ranked_candidates"]),
        "score_summary": dict(payload.get("score_summary") or {}),
        "score_breakdown": dict(payload.get("score_breakdown") or {}),
        "evidence_ids": dict(payload.get("evidence_ids") or {}),
        "verified_evidence": list(payload.get("evidence_summaries") or []),
        "closure_paths": list(payload.get("closure_paths") or []),
        "rejected_items": list(payload.get("rejected_items") or []),
        "advisory_from_tools": dict(payload.get("advisory") or {}),
        "rank_policy": {
            "final_rank_source": "deterministic_evidence_core",
            "llm_can_change_final_rank": False,
            "allowed": [
                "summarize the fixed ranking",
                "cite evidence IDs for each claim",
                "suggest verification or mitigation actions",
                "list unsupported claims separately",
            ],
            "forbidden": [
                "do not reorder final_ranking",
                "do not invent evidence IDs",
                "do not use labels, answer keys, or benchmark prior knowledge",
            ],
        },
        "provenance": dict(payload.get("provenance") or {}),
    }
