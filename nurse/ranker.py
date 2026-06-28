"""Nurse ranking from verified evidence closure."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List

from .closure import classify_signal
from .counterfactual_closure import compute_counterfactual_drops, facts_from_graph
from .evidence_program import EvidenceProgram, EvidenceQueryResult
from .matching import canonical_entity
from .rule_bank import apply_default_rules
from .semantic_rank_facts import normalize_and_guard_semantic_facts, semantic_fact_score
from .schemas import EvidenceGraph, NursePrediction


RCAEVAL_INFRA_SERVICES = {
    "front-end",
    "frontend-external",
    "main",
    "passthroughcluster",
    "queue-master",
    "rabbitmq",
    "rabbitmq-exporter",
    "redis",
    "ui-dashboard",
}

class NurseRanker:
    def __init__(
        self,
        *,
        source_weight: float = 1.4,
        generic_weight: float = 0.6,
        symptom_weight: float = 0.35,
        support_bonus: float = 0.8,
        propagation_bonus: float = 1.0,
        symptom_penalty: float = -0.55,
        evidence_program_weight: float = 2.0,
        counterfactual_weight: float = 1.6,
        metric_profile_enabled: bool = True,
        metric_tier_size: int = 6,
        metric_closure_prior_weight: float = 0.2,
        metric_deviation_consensus_weight: float = 1.5,
        metric_deviation_magnitude_weight: float = 0.1,
        metric_rank_prior_weight: float = 0.05,
        service_role_prior_weight: float = 1.0,
        llm_semantic_prior_weight: float = 1.0,
        llm_hypothesis_confidence_scale: float = 0.25,
        llm_hypothesis_max_weight: float = 0.25,
        allow_llm_scoring: bool = False,
        symbolic_score_enabled: bool = True,
        incident_shape_prior_weight: float = 2.5,
        incident_shape_counterfactual_weight: float = 1.8,
        incident_shape_topology_weight: float = 0.8,
        aiops_infra_projection_weight: float = 1.4,
        aiops_infra_penalty_weight: float = 12.0,
    ) -> None:
        self.source_weight = source_weight
        self.generic_weight = generic_weight
        self.symptom_weight = symptom_weight
        self.support_bonus = support_bonus
        self.propagation_bonus = propagation_bonus
        self.symptom_penalty = symptom_penalty
        self.evidence_program_weight = evidence_program_weight
        self.counterfactual_weight = counterfactual_weight
        self.metric_profile_enabled = metric_profile_enabled
        self.metric_tier_size = max(int(metric_tier_size), 0)
        self.metric_closure_prior_weight = metric_closure_prior_weight
        self.metric_deviation_consensus_weight = metric_deviation_consensus_weight
        self.metric_deviation_magnitude_weight = metric_deviation_magnitude_weight
        self.metric_rank_prior_weight = metric_rank_prior_weight
        self.service_role_prior_weight = service_role_prior_weight
        self.llm_semantic_prior_weight = llm_semantic_prior_weight
        self.llm_hypothesis_confidence_scale = llm_hypothesis_confidence_scale
        self.llm_hypothesis_max_weight = llm_hypothesis_max_weight
        self.allow_llm_scoring = allow_llm_scoring
        self.symbolic_score_enabled = symbolic_score_enabled
        self.incident_shape_prior_weight = incident_shape_prior_weight
        self.incident_shape_counterfactual_weight = incident_shape_counterfactual_weight
        self.incident_shape_topology_weight = incident_shape_topology_weight
        self.aiops_infra_projection_weight = aiops_infra_projection_weight
        self.aiops_infra_penalty_weight = aiops_infra_penalty_weight

    @staticmethod
    def _metric_deviation_scores(graph: EvidenceGraph) -> Dict[str, float]:
        if not isinstance(graph.metadata, dict):
            return {}
        raw_scores = graph.metadata.get("metric_deviation_scores")
        if not isinstance(raw_scores, dict):
            return {}
        scores: Dict[str, float] = {}
        for raw_service, raw_score in raw_scores.items():
            service = canonical_entity(str(raw_service))
            if not service:
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if math.isfinite(score):
                scores[service] = max(scores.get(service, 0.0), score)
        return scores

    @staticmethod
    def _use_metric_deviation_profile(graph: EvidenceGraph, metric_scores: Dict[str, float]) -> bool:
        if len(metric_scores) < 2 or not isinstance(graph.metadata, dict):
            return False
        profile = str(graph.metadata.get("ranking_profile") or "")
        dataset = str(graph.metadata.get("dataset") or "")
        return profile == "metric_deviation_closure_v3" or dataset.startswith("rcaeval")

    @staticmethod
    def _service_role_prior(dataset: str, service: str) -> float:
        if not dataset.startswith("rcaeval"):
            return 0.0
        prior = 0.0
        if (
            service in RCAEVAL_INFRA_SERVICES
            or service.endswith("-db")
            or service.endswith("-mongo")
        ):
            prior -= 1.0
        if dataset == "rcaeval-tt" and service.endswith("-service") and not service.startswith("admin-"):
            prior += 0.08
        return prior

    @staticmethod
    def _normalise(values: Dict[str, float]) -> Dict[str, float]:
        if not values:
            return {}
        finite = {}
        for key, value in values.items():
            number = float(value)
            finite[key] = number if math.isfinite(number) else 0.0
        lo = min(finite.values())
        hi = max(finite.values())
        if hi - lo <= 1e-12:
            return {key: (1.0 if value > 0 else 0.0) for key, value in finite.items()}
        return {key: (value - lo) / (hi - lo) for key, value in finite.items()}

    @staticmethod
    def _has_infra_candidates(service_candidates: object) -> bool:
        if not isinstance(service_candidates, list):
            return False
        for candidate in service_candidates:
            service = canonical_entity(str(candidate))
            if service.startswith("cn-") and "." in service:
                return True
        return False

    @staticmethod
    def _is_aiops_infra_entity(service: str) -> bool:
        text = service.lower()
        return (
            text.startswith("aiops-k8s-")
            or text.startswith("k8s-master")
            or (text.startswith("cn-") and "." in text)
            or text.startswith("host")
        )

    @staticmethod
    def _use_aiops_nonnode_projection(graph: EvidenceGraph) -> bool:
        if not isinstance(graph.metadata, dict):
            return False
        return str(graph.metadata.get("dataset") or "") == "aiops25-retained-nonnode"

    @staticmethod
    def _incident_shape_parts(graph: EvidenceGraph) -> Dict[str, Dict[str, float]]:
        if not isinstance(graph.metadata, dict):
            return {}
        raw_scores = graph.metadata.get("incident_shape_prior_scores")
        if not isinstance(raw_scores, dict):
            return {}
        parts: Dict[str, Dict[str, float]] = {}
        for raw_service, raw_value in raw_scores.items():
            service = canonical_entity(str(raw_service))
            if not service:
                continue
            if isinstance(raw_value, dict):
                total = raw_value.get("total", 0.0)
                counterfactual = raw_value.get("counterfactual", 0.0)
                topology = raw_value.get("topology_locality", 0.0)
            else:
                total = raw_value
                counterfactual = 0.0
                topology = 0.0
            try:
                total_value = float(total)
                counterfactual_value = float(counterfactual)
                topology_value = float(topology)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(total_value):
                total_value = 0.0
            if not math.isfinite(counterfactual_value):
                counterfactual_value = 0.0
            if not math.isfinite(topology_value):
                topology_value = 0.0
            parts[service] = {
                "total": max(total_value, 0.0),
                "counterfactual": max(counterfactual_value, 0.0),
                "topology_locality": max(topology_value, 0.0),
            }
        return parts

    def _apply_incident_shape_prior(
        self,
        graph: EvidenceGraph,
        scores: Dict[str, float],
        breakdown: Dict[str, Dict[str, float]],
    ) -> List[Dict[str, float]]:
        if self.incident_shape_prior_weight <= 0:
            return []
        raw_parts = self._incident_shape_parts(graph)
        if not raw_parts:
            return []
        candidate_pool = graph.metadata.get("service_candidates", []) if isinstance(graph.metadata, dict) else []
        allowed = {
            canonical_entity(str(candidate))
            for candidate in candidate_pool
            if canonical_entity(str(candidate))
        }
        if not allowed:
            allowed = set(scores) | set(raw_parts)
        total_values: Dict[str, float] = {}
        counterfactual_values: Dict[str, float] = {}
        topology_values: Dict[str, float] = {}
        for service, parts in raw_parts.items():
            if service not in allowed and service not in scores:
                continue
            total_values[service] = parts.get("total", 0.0)
            counterfactual_values[service] = parts.get("counterfactual", 0.0)
            topology_values[service] = parts.get("topology_locality", 0.0)
        if not total_values:
            return []
        total_norm = self._normalise(total_values)
        counterfactual_norm = self._normalise(counterfactual_values)
        topology_norm = self._normalise(topology_values)
        prior_rows: List[Dict[str, float]] = []
        for service in sorted(set(total_norm) | set(counterfactual_norm) | set(topology_norm)):
            prior_part = self.incident_shape_prior_weight * total_norm.get(service, 0.0)
            counterfactual_part = self.incident_shape_counterfactual_weight * counterfactual_norm.get(service, 0.0)
            topology_part = self.incident_shape_topology_weight * topology_norm.get(service, 0.0)
            total_part = prior_part + counterfactual_part + topology_part
            if abs(total_part) <= 1e-12:
                continue
            scores[service] += total_part
            breakdown[service]["incident_shape_prior"] += prior_part
            breakdown[service]["incident_shape_counterfactual"] += counterfactual_part
            breakdown[service]["incident_shape_topology_locality"] += topology_part
            prior_rows.append(
                {
                    "service": service,
                    "incident_shape_prior": float(prior_part),
                    "incident_shape_counterfactual": float(counterfactual_part),
                    "incident_shape_topology_locality": float(topology_part),
                    "total": float(total_part),
                }
            )
        prior_rows.sort(key=lambda item: (-item["total"], item["service"]))
        return prior_rows[:10]

    def rank(self, graph: EvidenceGraph) -> NursePrediction:
        scores: Dict[str, float] = defaultdict(float)
        breakdown: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        evidence_ids: Dict[str, List[str]] = defaultdict(list)
        aiops_nonnode_projection = self._use_aiops_nonnode_projection(graph)

        base_facts = facts_from_graph(graph)
        service_candidates = graph.metadata.get("service_candidates") if isinstance(graph.metadata, dict) else None
        llm_proposal = graph.metadata.get("llm_proposal") if isinstance(graph.metadata, dict) else None
        semantic_rank_facts = []
        semantic_rank_rejections = []
        if self.allow_llm_scoring and isinstance(llm_proposal, dict):
            verified_ids = [
                evidence.evidence_id
                for evidence in graph.evidences
                if evidence.verified and evidence.evidence_id
            ]
            semantic_rank_facts, semantic_rank_rejections = normalize_and_guard_semantic_facts(
                proposal=llm_proposal,
                service_candidates=service_candidates or [],
                verified_evidence_ids=verified_ids,
                score_breakdown=graph.metadata.get("score_breakdown", {}) if isinstance(graph.metadata, dict) else {},
            )
        symbolic_facts = list(base_facts)
        if self.symbolic_score_enabled:
            for fact in semantic_rank_facts:
                service = canonical_entity(fact.service)
                if not service:
                    continue
                semantic_score = semantic_fact_score(fact, prior_weight=self.llm_semantic_prior_weight)
                scores[service] += semantic_score
                breakdown[service]["symbolic_semantic_facts"] += semantic_score
                breakdown[service]["S"] += semantic_score
            symbolic_facts = symbolic_facts + apply_default_rules(base_facts, service_candidates=service_candidates)
            program_result = EvidenceProgram().query_root_probabilities(symbolic_facts)
            program_scores = self._normalise(program_result.scores)
            counterfactual_scores = self._normalise(compute_counterfactual_drops(symbolic_facts))
        else:
            program_result = EvidenceQueryResult(ranking=[], scores={}, explanations={})
            program_scores = {}
            counterfactual_scores = self._normalise(compute_counterfactual_drops(symbolic_facts))
        for service, value in program_scores.items():
            part = self.evidence_program_weight * value
            scores[service] += part
            breakdown[service]["evidence_program"] += part
        for service, value in counterfactual_scores.items():
            part = self.counterfactual_weight * value
            scores[service] += part
            breakdown[service]["counterfactual_drop"] += part

        for evidence in graph.evidences:
            if evidence.kind == "topology":
                continue
            entity = canonical_entity(evidence.entity)
            if not entity:
                continue
            role = str(evidence.claim.get("metric_role") or "")
            if evidence.provenance.get("candidate_source") == "llm_proposal" or role == "llm_verified_check":
                signal_class = "llm_check"
            elif role == "source_metric":
                signal_class = "source_like"
            elif role == "symptom_metric":
                signal_class = "symptom_like"
            else:
                signal_class = classify_signal(evidence.signal)
            if signal_class == "source_like":
                value = self.source_weight * 0.5
                part = "source_evidence"
            elif signal_class == "symptom_like":
                value = self.symptom_weight * 0.5
                part = "symptom_evidence"
                scores[entity] += self.symptom_penalty
                breakdown[entity]["symptom_penalty"] += self.symptom_penalty
            else:
                if signal_class == "llm_check":
                    value = 0.05
                    part = "llm_verified_check"
                else:
                    value = self.generic_weight * 0.5
                    part = "generic_evidence"
            scores[entity] += value
            breakdown[entity][part] += value
            try:
                rank_bonus = float(evidence.claim.get("rank_bonus") or 0.0)
            except (TypeError, ValueError):
                rank_bonus = 0.0
            if rank_bonus > 0 and evidence.kind == "event":
                scores[entity] += rank_bonus
                breakdown[entity]["verified_infra_event"] += rank_bonus
            evidence_ids[entity].append(evidence.evidence_id)

        propagation_fanout: Dict[str, set[str]] = defaultdict(set)
        for relation in graph.relations:
            if relation.get("type") != "propagates_to":
                continue
            src = canonical_entity(str(relation.get("src") or ""))
            dst = canonical_entity(str(relation.get("dst") or ""))
            if src and dst and self._is_aiops_infra_entity(src):
                propagation_fanout[src].add(dst)

        closure_paths = list(graph.relations)
        for relation in graph.relations:
            src = canonical_entity(str(relation.get("src") or ""))
            dst = canonical_entity(str(relation.get("dst") or ""))
            rtype = relation.get("type")
            if rtype == "supports_root" and src:
                if aiops_nonnode_projection and self._is_aiops_infra_entity(src):
                    breakdown[src]["aiops_infra_support_suppressed"] += -self.support_bonus * 0.5
                    continue
                bonus = self.support_bonus * 0.5
                scores[src] += bonus
                breakdown[src]["closure_support"] += bonus
            elif rtype == "propagates_to" and src:
                confidence = float(relation.get("confidence") or 1.0)
                fanout = max(len(propagation_fanout.get(src, set())), 1)
                if aiops_nonnode_projection and self._is_aiops_infra_entity(src):
                    if dst and not self._is_aiops_infra_entity(dst):
                        projection = self.aiops_infra_projection_weight * confidence / math.sqrt(fanout)
                        scores[dst] += projection
                        breakdown[dst]["aiops_infra_projection"] += projection
                    penalty = -self.aiops_infra_penalty_weight / math.sqrt(fanout)
                    scores[src] += penalty
                    breakdown[src]["aiops_infra_penalty"] += penalty
                    continue
                raw_bonus = self.propagation_bonus * 0.5 * confidence
                if fanout >= 3:
                    bonus = raw_bonus / math.sqrt(fanout)
                    breakdown[src]["broad_infra_fanout_attenuation"] += bonus - raw_bonus
                else:
                    bonus = raw_bonus
                scores[src] += bonus
                breakdown[src]["propagation_consistency"] += bonus
                if dst:
                    scores[dst] += 0.05
                    breakdown[dst]["downstream_observation"] += 0.05

        candidate_pool = graph.metadata.get("service_candidates", []) if isinstance(graph.metadata, dict) else []
        for index, candidate in enumerate(candidate_pool):
            service = canonical_entity(str(candidate))
            if service and service not in scores:
                scores[service] = -1e-6 - (index * 1e-9)
                breakdown[service]["candidate_pool"] += 0.0

        ranker_name = "evidence_program_v2"
        metric_scores = self._metric_deviation_scores(graph)
        metric_tier: List[str] = []
        incident_shape_top: List[Dict[str, float]] = []
        if self.metric_profile_enabled and self._use_metric_deviation_profile(graph, metric_scores):
            ranker_name = "metric_deviation_closure_v3"
            dataset = str(graph.metadata.get("dataset") or "") if isinstance(graph.metadata, dict) else ""
            base_norm = self._normalise(dict(scores))
            deviation_norm = self._normalise(metric_scores)
            services = set(scores) | set(metric_scores)
            metric_tier = sorted(metric_scores, key=lambda service: (-metric_scores[service], service))[
                : self.metric_tier_size
            ]
            for service in services:
                closure_prior = self.metric_closure_prior_weight * base_norm.get(service, 0.0)
                deviation_consensus = (
                    self.metric_deviation_consensus_weight * deviation_norm.get(service, 0.0)
                )
                deviation_magnitude = self.metric_deviation_magnitude_weight * math.log1p(
                    max(metric_scores.get(service, 0.0), 0.0)
                )
                service_role_prior = (
                    self.service_role_prior_weight * self._service_role_prior(dataset, service)
                )
                rank_prior = (
                    self.metric_rank_prior_weight / (metric_tier.index(service) + 1)
                    if service in metric_tier
                    else 0.0
                )
                scores[service] = (
                    closure_prior
                    + deviation_consensus
                    + deviation_magnitude
                    + service_role_prior
                    + rank_prior
                )
                breakdown[service]["base_closure_prior"] += closure_prior
                breakdown[service]["metric_deviation_consensus"] += deviation_consensus
                breakdown[service]["metric_deviation_magnitude"] += deviation_magnitude
                breakdown[service]["service_role_prior"] += service_role_prior
                breakdown[service]["metric_rank_prior"] += rank_prior
            incident_shape_top = self._apply_incident_shape_prior(graph, scores, breakdown)
            metric_tier_set = set(metric_tier)
            ranked = sorted(
                scores,
                key=lambda service: (0 if service in metric_tier_set else 1, -scores[service], service),
            )
        else:
            incident_shape_top = self._apply_incident_shape_prior(graph, scores, breakdown)
            ranked = sorted(scores, key=lambda service: (-scores[service], service))
        return NursePrediction(
            case_id=graph.case_id,
            ranked=ranked,
            scores={service: float(scores[service]) for service in ranked},
            score_breakdown={
                service: {part: float(value) for part, value in parts.items()}
                for service, parts in breakdown.items()
            },
            evidence_ids={service: ids for service, ids in evidence_ids.items()},
            closure_paths=closure_paths,
            metadata={
                "ranker": ranker_name,
                "evidence_program_scores": program_result.scores,
                "evidence_program_explanations": program_result.explanations,
                "metric_deviation_top_tier": metric_tier,
                "llm_rule_facts": len(semantic_rank_facts),
                "llm_semantic_prior_facts": len(semantic_rank_facts),
                "semantic_rank_facts_used": len(semantic_rank_facts) if self.symbolic_score_enabled else 0,
                "semantic_rank_facts_rejected": len(semantic_rank_rejections),
                "llm_scoring_allowed": bool(self.allow_llm_scoring),
                "incident_shape_prior_enabled": bool(incident_shape_top),
                "incident_shape_prior_top": incident_shape_top,
            },
        )




