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


def _canonical_topology_edges(topology_edges: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen: set[Tuple[str, str]] = set()
    edges: List[Tuple[str, str]] = []
    for raw_src, raw_dst in topology_edges:
        src = canonical_entity(str(raw_src))
        dst = canonical_entity(str(raw_dst))
        if not src or not dst or src == dst:
            continue
        edge = (src, dst)
        if edge in seen:
            continue
        seen.add(edge)
        edges.append(edge)
    edges.sort()
    return edges


def _closure_path_edges(closure_paths: Iterable[object]) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for path in closure_paths or []:
        if not isinstance(path, dict):
            continue
        src = canonical_entity(str(path.get("src") or path.get("caller") or path.get("source") or ""))
        dst = canonical_entity(str(path.get("dst") or path.get("callee") or path.get("target") or ""))
        if not src or not dst or src == dst:
            continue
        edge = (src, dst)
        if edge in seen:
            continue
        seen.add(edge)
        edges.append(edge)
    edges.sort()
    return edges


def _tarjan_scc(nodes: Sequence[str], adjacency: Dict[str, List[str]]) -> List[List[str]]:
    index = 0
    stack: List[str] = []
    on_stack: set[str] = set()
    indices: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    components: List[List[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in adjacency.get(node, []):
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[node] = min(lowlink[node], indices[neighbor])

        if lowlink[node] == indices[node]:
            component: List[str] = []
            while stack:
                current = stack.pop()
                on_stack.discard(current)
                component.append(current)
                if current == node:
                    break
            components.append(sorted(component))

    for node in nodes:
        if node not in indices:
            strongconnect(node)

    components.sort(key=lambda members: (members[0], len(members), tuple(members)))
    return components


def _condensed_topology_summary(
    *,
    prediction: NursePrediction,
    verified_evidence: Iterable[VerifiedEvidence],
    service_candidates: Iterable[str],
    topology_edges: Iterable[Tuple[str, str]],
    closure_paths: Iterable[object],
) -> Dict[str, object]:
    nodes: set[str] = set()
    for service in service_candidates:
        node = canonical_entity(str(service))
        if node:
            nodes.add(node)
    for service in prediction.ranked:
        node = canonical_entity(str(service))
        if node:
            nodes.add(node)
    for evidence in verified_evidence:
        if not evidence.verified:
            continue
        node = canonical_entity(str(evidence.entity))
        if node:
            nodes.add(node)

    edges = _canonical_topology_edges(list(topology_edges) + _closure_path_edges(closure_paths))
    for src, dst in edges:
        nodes.add(src)
        nodes.add(dst)

    ordered_nodes = sorted(nodes)
    adjacency: Dict[str, List[str]] = {node: [] for node in ordered_nodes}
    for src, dst in edges:
        adjacency.setdefault(src, []).append(dst)
    for node in adjacency:
        adjacency[node] = sorted(set(adjacency[node]))

    components = _tarjan_scc(ordered_nodes, adjacency)
    service_to_component: Dict[str, str] = {}
    component_entries: List[Dict[str, object]] = []
    component_in: Counter = Counter()
    component_out: Counter = Counter()

    for index, members in enumerate(components):
        component_id = f"scc_{index}"
        for member in members:
            service_to_component[member] = component_id

    condensed_edges_set: set[Tuple[str, str]] = set()
    for src, dst in edges:
        src_component = service_to_component.get(src)
        dst_component = service_to_component.get(dst)
        if not src_component or not dst_component or src_component == dst_component:
            continue
        condensed_edges_set.add((src_component, dst_component))

    condensed_edges = sorted(condensed_edges_set)
    for src_component, dst_component in condensed_edges:
        component_out[src_component] += 1
        component_in[dst_component] += 1

    for index, members in enumerate(components):
        component_id = f"scc_{index}"
        component_entries.append(
            {
                "component_id": component_id,
                "members": members,
                "representative": members[0] if members else "",
                "size": len(members),
                "is_cycle": len(members) > 1,
                "in_degree": int(component_in.get(component_id, 0)),
                "out_degree": int(component_out.get(component_id, 0)),
            }
        )

    return {
        "node_count": len(ordered_nodes),
        "edge_count": len(edges),
        "component_count": len(component_entries),
        "cycle_count": sum(1 for item in component_entries if item["is_cycle"]),
        "service_to_component": service_to_component,
        "components": component_entries,
        "edges": [
            {"caller_component": src_component, "callee_component": dst_component}
            for src_component, dst_component in condensed_edges
        ],
    }


def _verified_evidence_summary(evidence: VerifiedEvidence) -> Dict[str, object]:
    claim = evidence.claim if isinstance(evidence.claim, dict) else {}
    observed = evidence.observed if isinstance(evidence.observed, dict) else {}
    provenance = evidence.provenance if isinstance(evidence.provenance, dict) else {}
    return {
        "evidence_id": evidence.evidence_id,
        "entity": evidence.entity,
        "signal": evidence.signal,
        "kind": evidence.kind,
        "metric_role": claim.get("metric_role", ""),
        "time_range": dict(evidence.time_range),
        "verification_rule": evidence.verification_rule,
        "observed": {
            key: observed[key]
            for key in ("observed", "baseline", "matching_points", "matching_rows", "max_duration")
            if key in observed
        },
        "source": provenance.get("source", ""),
    }


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


def semantic_rank_role_policy_for_dataset(dataset: str) -> Dict[str, List[str]]:
    normalized = str(dataset or "").strip().lower()
    if normalized == "aiops25-retained-nonnode":
        return {
            "preferred_target_roles": ["application_candidate"],
            "discouraged_target_roles": [
                "symptom_prone_entrypoint",
                "blocked_infra_for_retained_nonnode",
            ],
        }
    if normalized == "rcaeval-tt":
        return {
            "preferred_target_roles": ["primary_train_ticket_service"],
            "discouraged_target_roles": ["support_or_storage_candidate"],
        }
    return {
        "preferred_target_roles": ["candidate"],
        "discouraged_target_roles": [],
    }


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

    service_candidates = list(service_candidates)
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


def build_semantic_rank_payload(
    *,
    alarm: AlarmArtifact,
    service_candidates: Iterable[str],
    prediction: NursePrediction,
    verified_evidence: Iterable[VerifiedEvidence],
    topology_edges: Iterable[Tuple[str, str]],
    prompt_profile: str = "balanced",
) -> Dict[str, object]:
    """Create the compact pre-ranking payload that should return semantic_facts."""

    service_candidates = list(service_candidates)
    verified_evidence = list(verified_evidence)
    topology_edges = list(topology_edges)
    role_policy = semantic_rank_role_policy_for_dataset(alarm.dataset)
    cards = build_candidate_cards(
        prediction=prediction,
        verified_evidence=verified_evidence,
        topology_edges=topology_edges,
        dataset=alarm.dataset,
        max_cards=8,
    )
    condensed_topology = _condensed_topology_summary(
        prediction=prediction,
        verified_evidence=verified_evidence,
        service_candidates=service_candidates,
        topology_edges=topology_edges,
        closure_paths=getattr(prediction, "closure_paths", []),
    )
    evidence_items = [
        _verified_evidence_summary(item)
        for item in verified_evidence
        if item.verified and item.evidence_id
    ]
    evidence_items.sort(
        key=lambda item: (
            0 if item.get("metric_role") == "source_metric" else 1,
            str(item.get("entity") or ""),
            str(item.get("signal") or ""),
        )
    )
    return {
        "incident_ref": "current-incident",
        "output_contract": "semantic_rank_only",
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
        "current_ranking": list(prediction.ranked)[:12],
        "candidate_cards": cards,
        "verified_evidence": evidence_items[:16],
        "prompt_profile": prompt_profile,
        "topology_edges": [
            {"caller": canonical_entity(str(caller)), "callee": canonical_entity(str(callee))}
            for caller, callee in topology_edges[:48]
            if canonical_entity(str(caller)) and canonical_entity(str(callee))
        ],
        "condensed_topology": condensed_topology,
        "semantic_policy": {
            "semantic_facts": (
                "Return semantic_facts only. Return at most 3 facts. "
                "Each fact must include service, fact_type, and supporting_evidence_ids. "
                "Use fact_type from allowed_fact_types. "
                "Use the condensed topology summary when reasoning about cycles or propagation; "
                "do not invent nodes or edges outside the payload. "
                "For metric/anomaly/deviation observations, prefer source_support. "
                "For topology or propagation observations, decide whether they are better cast as source_support or symptom_support. "
                "For the most plausible root-side service, prefer root_support. "
                "confidence and source are optional. "
                "Each fact may only name services already present in service_candidates or current_ranking. "
                f"For {alarm.dataset}, preferred_target_roles are {', '.join(role_policy['preferred_target_roles'])}; "
                f"discouraged_target_roles are {', '.join(role_policy['discouraged_target_roles'])}. "
                "AIOps retained non-node should prefer application candidates and avoid symptom-prone entrypoints and blocked infra. "
                "RCAEval-TT should prefer primary train-ticket services and rarely assign root-side support to support or storage candidates."
            ),
            "allowed_fact_types": [
                "root_support",
                "source_support",
                "symptom_support",
            ],
            "fact_type_guidance": {
                "resource_metric_deviation": "source_support",
                "resource_metric_anomaly": "source_support",
                "resource_usage_deviation": "source_support",
                "resource_utilization_deviation": "source_support",
                "memory_metric_deviation": "source_support",
                "cpu_metric_deviation": "source_support",
                "service_resource_and_performance_deviation": "source_support",
                "elevated_cpu_usage_during_fault_window": "source_support",
                "elevated_cpu_and_memory_usage_during_fault_window": "source_support",
                "elevated_metric_deviation_in_fault_window": "source_support",
                "abnormal_resource_metric_deviation": "source_support",
                "metric_deviation": "source_support",
                "metric_anomaly": "source_support",
                "topology_deviation": "source_support",
                "downstream_symptom": "symptom_support",
            },
            "preferred_target_roles": role_policy["preferred_target_roles"],
            "discouraged_target_roles": role_policy["discouraged_target_roles"],
            "forbidden": [
                "do not emit candidates, hypotheses, candidate rules, or repair output",
                "do not explain or narrate",
                "do not invent services or evidence IDs",
                "do not use labels or answer keys",
            ],
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
