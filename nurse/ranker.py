"""NURSE Evidence Ranker.

The released score follows the current methodology:

    Score(s) = lambda_E E(s) + lambda_P P(s) + lambda_H H(s) + lambda_R R(s)

E(s): direct evidence contribution
P(s): topology propagation contribution
H(s): historical semantic-payload prior
R(s): explicit SRE rule contribution
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List

from .closure import classify_signal
from .counterfactual_closure import compute_counterfactual_drops, facts_from_graph
from .evidence_program import EvidenceProgram, EvidenceQueryResult
from .history_prior import (
    DEFAULT_HISTORY_TOP_K,
    DEFAULT_HISTORY_WEIGHT,
    compute_history_prior,
    get_default_encoder,
)
from .llm_payload import RCAEVAL_TT_PRIMARY_SERVICES
from .matching import canonical_entity
from .rule_bank import apply_default_rules
from .semantic_rank_facts import normalize_and_guard_semantic_facts, semantic_fact_score
from .schemas import EvidenceGraph, NursePrediction, VerifiedEvidence


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

DIRECT_EVIDENCE_KEYS = {
    "root_evidence",
    "source_evidence",
    "generic_evidence",
    "symptom_evidence",
    "symptom_penalty",
    "llm_verified_check",
    "verified_infra_event",
    "closure_support",
    "semantic_fact_evidence",
    "evidence_program",
    "counterfactual_drop",
    "base_closure_prior",
    "metric_deviation_consensus",
    "metric_deviation_magnitude",
    "service_role_prior",
    "metric_rank_prior",
    "incident_shape_prior",
    "incident_shape_counterfactual",
    "incident_shape_topology_locality",
}
PROPAGATION_KEYS = {
    "propagation_consistency",
    "downstream_observation",
    "upstream_source_penalty",
    "aiops_infra_projection",
    "aiops_infra_penalty",
    "broad_infra_fanout_attenuation",
}
HISTORY_KEYS = {"history_prior"}
RULE_KEYS = {"sre_rule"}
IGNORED_COMPONENT_KEYS = {
    "candidate_pool",
    "aiops_infra_support_suppressed",
}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _dedupe_ids(values: List[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


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
        history_prior_weight: float = DEFAULT_HISTORY_WEIGHT,
        history_prior_top_k: int = DEFAULT_HISTORY_TOP_K,
        propagation_observation_weight: float = 0.05,
        rule_score_weight: float = 1.0,
        allow_llm_scoring: bool = False,
        symbolic_score_enabled: bool = True,
        incident_shape_prior_weight: float = 2.5,
        incident_shape_counterfactual_weight: float = 1.8,
        incident_shape_topology_weight: float = 0.8,
        aiops_infra_projection_weight: float = 1.4,
        aiops_infra_penalty_weight: float = 12.0,
        lambda_e: float = 1.0,
        lambda_p: float = 1.0,
        lambda_h: float = 1.0,
        lambda_r: float = 1.0,
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
        self.history_prior_weight = history_prior_weight
        self.history_prior_top_k = max(int(history_prior_top_k), 0)
        self.propagation_observation_weight = propagation_observation_weight
        self.rule_score_weight = rule_score_weight
        self.allow_llm_scoring = allow_llm_scoring
        self.symbolic_score_enabled = symbolic_score_enabled
        self.incident_shape_prior_weight = incident_shape_prior_weight
        self.incident_shape_counterfactual_weight = incident_shape_counterfactual_weight
        self.incident_shape_topology_weight = incident_shape_topology_weight
        self.aiops_infra_projection_weight = aiops_infra_projection_weight
        self.aiops_infra_penalty_weight = aiops_infra_penalty_weight
        self.lambda_e = lambda_e
        self.lambda_p = lambda_p
        self.lambda_h = lambda_h
        self.lambda_r = lambda_r

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
            score = _safe_float(raw_score)
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
        if dataset == "rcaeval-tt" and service in RCAEVAL_TT_PRIMARY_SERVICES:
            prior += 0.7
        return prior

    @staticmethod
    def _normalise(values: Dict[str, float]) -> Dict[str, float]:
        if not values:
            return {}
        finite = {}
        for key, value in values.items():
            finite[key] = _safe_float(value)
        lo = min(finite.values())
        hi = max(finite.values())
        if hi - lo <= 1e-12:
            return {key: (1.0 if value > 0 else 0.0) for key, value in finite.items()}
        return {key: (value - lo) / (hi - lo) for key, value in finite.items()}

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
            total_value = max(_safe_float(total), 0.0)
            counterfactual_value = max(_safe_float(counterfactual), 0.0)
            topology_value = max(_safe_float(topology), 0.0)
            parts[service] = {
                "total": total_value,
                "counterfactual": counterfactual_value,
                "topology_locality": topology_value,
            }
        return parts

    @staticmethod
    def _component_totals(parts: Dict[str, float]) -> Dict[str, float]:
        buckets = {"E": 0.0, "P": 0.0, "H": 0.0, "R": 0.0}
        for key, raw_value in dict(parts or {}).items():
            if key in buckets or key in IGNORED_COMPONENT_KEYS:
                continue
            value = _safe_float(raw_value)
            if key in DIRECT_EVIDENCE_KEYS:
                buckets["E"] += value
            elif key in PROPAGATION_KEYS:
                buckets["P"] += value
            elif key in HISTORY_KEYS:
                buckets["H"] += value
            elif key in RULE_KEYS:
                buckets["R"] += value
            else:
                buckets["E"] += value
        return buckets

    def _apply_incident_shape_prior(
        self,
        graph: EvidenceGraph,
        scores: Dict[str, float],
        breakdown: Dict[str, Dict[str, float]],
    ) -> List[Dict[str, float]]:
        if self.incident_shape_prior_weight <= 0 or self.lambda_e == 0.0:
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
            prior_part = self.lambda_e * self.incident_shape_prior_weight * total_norm.get(service, 0.0)
            counterfactual_part = (
                self.lambda_e
                * self.incident_shape_counterfactual_weight
                * counterfactual_norm.get(service, 0.0)
            )
            topology_part = self.lambda_e * self.incident_shape_topology_weight * topology_norm.get(service, 0.0)
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

    def _apply_history_prior(
        self,
        graph: EvidenceGraph,
        scores: Dict[str, float],
        breakdown: Dict[str, Dict[str, float]],
    ) -> List[Dict[str, float]]:
        if self.history_prior_weight <= 0 or self.lambda_h == 0.0:
            return []
        if not isinstance(graph.metadata, dict):
            return []
        semantic_payload = graph.metadata.get("semantic_rank_payload")
        if not isinstance(semantic_payload, dict):
            semantic_payload = graph.metadata.get("semantic_payload")
        if not isinstance(semantic_payload, dict):
            return []
        history_index = graph.metadata.get("approved_history_index")
        if history_index is None:
            history_index = graph.metadata.get("history_index")
        if history_index is not None and not isinstance(history_index, list):
            try:
                history_index = list(history_index)
            except TypeError:
                history_index = []
        history_prior_result = compute_history_prior(
            case_id=graph.case_id,
            dataset=str(graph.metadata.get("dataset") or ""),
            current_payload=semantic_payload,
            history_index=history_index,
            service_candidates=graph.metadata.get("service_candidates"),
            top_k=self.history_prior_top_k,
            history_weight=self.history_prior_weight,
            encoder=get_default_encoder(),
        )
        if not history_prior_result.active:
            return []
        for service, value in history_prior_result.score_contribution.items():
            weighted = self.lambda_h * float(value)
            scores[service] += weighted
            breakdown[service]["history_prior"] += weighted
        return history_prior_result.matched_history

    def _apply_sre_rules(
        self,
        graph: EvidenceGraph,
        scores: Dict[str, float],
        breakdown: Dict[str, Dict[str, float]],
    ) -> List[Dict[str, float]]:
        if self.rule_score_weight <= 0 or self.lambda_r == 0.0:
            return []
        if not isinstance(graph.metadata, dict):
            return []
        raw_scores = graph.metadata.get("sre_rule_scores")
        if not isinstance(raw_scores, dict):
            return []
        rows: List[Dict[str, float]] = []
        for raw_service, raw_value in raw_scores.items():
            service = canonical_entity(str(raw_service))
            if not service:
                continue
            value = self.lambda_r * self.rule_score_weight * _safe_float(raw_value)
            if abs(value) <= 1e-12:
                continue
            scores[service] += value
            breakdown[service]["sre_rule"] += value
            rows.append({"service": service, "rule_score": float(value)})
        rows.sort(key=lambda item: (-abs(item["rule_score"]), item["service"]))
        return rows

    def rank(self, graph: EvidenceGraph) -> NursePrediction:
        scores: Dict[str, float] = defaultdict(float)
        breakdown: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        evidence_ids: Dict[str, List[str]] = defaultdict(list)
        dataset = str(graph.metadata.get("dataset") or "") if isinstance(graph.metadata, dict) else ""
        service_candidates = graph.metadata.get("service_candidates", []) if isinstance(graph.metadata, dict) else []
        aiops_nonnode_projection = self._use_aiops_nonnode_projection(graph)

        base_facts = facts_from_graph(graph)
        llm_proposal = graph.metadata.get("llm_proposal") if isinstance(graph.metadata, dict) else None
        semantic_rank_facts = []
        semantic_rank_rejections = []
        if self.allow_llm_scoring and self.symbolic_score_enabled and isinstance(llm_proposal, dict):
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
                dataset=dataset,
            )

        symbolic_facts = list(base_facts)
        if self.symbolic_score_enabled:
            for fact in semantic_rank_facts:
                service = canonical_entity(fact.service)
                if not service:
                    continue
                magnitude = self.lambda_e * semantic_fact_score(
                    fact,
                    prior_weight=self.llm_semantic_prior_weight,
                )
                contribution = -magnitude if fact.fact_type == "symptom_support" else magnitude
                scores[service] += contribution
                breakdown[service]["semantic_fact_evidence"] += contribution
            symbolic_facts = symbolic_facts + apply_default_rules(base_facts, service_candidates=service_candidates)
            program_result = EvidenceProgram().query_root_probabilities(symbolic_facts)
            program_scores = self._normalise(program_result.scores)
            counterfactual_scores = self._normalise(compute_counterfactual_drops(symbolic_facts))
        else:
            program_result = EvidenceQueryResult(ranking=[], scores={}, explanations={})
            program_scores = {}
            counterfactual_scores = self._normalise(compute_counterfactual_drops(symbolic_facts))

        if self.lambda_e != 0.0:
            for service, value in program_scores.items():
                part = self.lambda_e * self.evidence_program_weight * value
                scores[service] += part
                breakdown[service]["evidence_program"] += part
            for service, value in counterfactual_scores.items():
                part = self.lambda_e * self.counterfactual_weight * value
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
                value = self.lambda_e * self.source_weight * 0.5
                part = "source_evidence"
            elif signal_class == "symptom_like":
                value = self.lambda_e * self.symptom_weight * 0.5
                part = "symptom_evidence"
                penalty = self.lambda_e * self.symptom_penalty
                scores[entity] += penalty
                breakdown[entity]["symptom_penalty"] += penalty
            else:
                if signal_class == "llm_check":
                    value = self.lambda_e * 0.05
                    part = "llm_verified_check"
                else:
                    value = self.lambda_e * self.generic_weight * 0.5
                    part = "generic_evidence"

            scores[entity] += value
            breakdown[entity][part] += value

            rank_bonus = _safe_float(evidence.claim.get("rank_bonus"), 0.0)
            if rank_bonus > 0 and evidence.kind == "event":
                weighted_bonus = self.lambda_e * rank_bonus
                scores[entity] += weighted_bonus
                breakdown[entity]["verified_infra_event"] += weighted_bonus
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
                    breakdown[src]["aiops_infra_support_suppressed"] += -self.lambda_e * self.support_bonus * 0.5
                    continue
                bonus = self.lambda_e * self.support_bonus * 0.5
                scores[src] += bonus
                breakdown[src]["closure_support"] += bonus
            elif rtype == "propagates_to" and src and self.lambda_p != 0.0:
                confidence = max(_safe_float(relation.get("confidence"), 1.0), 0.0)
                fanout = max(len(propagation_fanout.get(src, set())), 1)
                if aiops_nonnode_projection and self._is_aiops_infra_entity(src):
                    if dst and not self._is_aiops_infra_entity(dst):
                        projection = self.lambda_p * self.aiops_infra_projection_weight * confidence / math.sqrt(fanout)
                        scores[dst] += projection
                        breakdown[dst]["aiops_infra_projection"] += projection
                    penalty = -self.lambda_p * self.aiops_infra_penalty_weight / math.sqrt(fanout)
                    scores[src] += penalty
                    breakdown[src]["aiops_infra_penalty"] += penalty
                    continue
                raw_bonus = self.lambda_p * self.propagation_bonus * 0.5 * confidence
                if fanout >= 3:
                    bonus = raw_bonus / math.sqrt(fanout)
                    breakdown[src]["broad_infra_fanout_attenuation"] += bonus - raw_bonus
                else:
                    bonus = raw_bonus
                scores[src] += bonus
                breakdown[src]["propagation_consistency"] += bonus
                if dst:
                    downstream = self.lambda_p * self.propagation_observation_weight * confidence
                    scores[dst] += downstream
                    breakdown[dst]["downstream_observation"] += downstream

        candidate_pool = list(service_candidates) if isinstance(service_candidates, list) else []
        for index, candidate in enumerate(candidate_pool):
            service = canonical_entity(str(candidate))
            if service and service not in scores:
                scores[service] = -1e-6 - (index * 1e-9)
                breakdown[service]["candidate_pool"] += 0.0

        ranker_name = "evidence_program_v2"
        metric_scores = self._metric_deviation_scores(graph)
        metric_tier: List[str] = []
        if self.metric_profile_enabled and self._use_metric_deviation_profile(graph, metric_scores):
            ranker_name = "metric_deviation_closure_v3"
            base_norm = self._normalise(dict(scores))
            deviation_norm = self._normalise(metric_scores)
            services = set(scores) | set(metric_scores)
            metric_tier = sorted(metric_scores, key=lambda service: (-metric_scores[service], service))[
                : self.metric_tier_size
            ]
            for service in services:
                closure_prior = self.lambda_e * self.metric_closure_prior_weight * base_norm.get(service, 0.0)
                deviation_consensus = (
                    self.lambda_e * self.metric_deviation_consensus_weight * deviation_norm.get(service, 0.0)
                )
                deviation_magnitude = self.lambda_e * self.metric_deviation_magnitude_weight * math.log1p(
                    max(metric_scores.get(service, 0.0), 0.0)
                )
                service_role_prior = (
                    self.lambda_e * self.service_role_prior_weight * self._service_role_prior(dataset, service)
                )
                rank_prior = (
                    self.lambda_e * self.metric_rank_prior_weight / (metric_tier.index(service) + 1)
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
        history_top = self._apply_history_prior(graph, scores, breakdown)
        sre_rule_top = self._apply_sre_rules(graph, scores, breakdown)

        for service in list(scores):
            evidence_ids[service] = _dedupe_ids(evidence_ids.get(service, []))

        if metric_tier:
            metric_tier_set = set(metric_tier)
            ranked = sorted(
                scores,
                key=lambda service: (0 if service in metric_tier_set else 1, -scores[service], service),
            )
        else:
            ranked = sorted(scores, key=lambda service: (-scores[service], service))

        final_breakdown = {}
        for service, parts in breakdown.items():
            materialized = {part: float(value) for part, value in parts.items()}
            materialized.update(
                {key: float(value) for key, value in self._component_totals(materialized).items()}
            )
            final_breakdown[service] = materialized

        return NursePrediction(
            case_id=graph.case_id,
            ranked=ranked,
            scores={service: float(scores[service]) for service in ranked},
            score_breakdown=final_breakdown,
            evidence_ids={service: evidence_ids.get(service, []) for service in ranked},
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
                "llm_scoring_allowed": bool(self.allow_llm_scoring and self.symbolic_score_enabled),
                "lambda_e": float(self.lambda_e),
                "lambda_p": float(self.lambda_p),
                "lambda_h": float(self.lambda_h),
                "lambda_r": float(self.lambda_r),
                "history_prior_enabled": bool(history_top),
                "history_prior_top": history_top,
                "history_prior_weight": float(self.history_prior_weight),
                "history_prior_top_k": int(self.history_prior_top_k),
                "sre_rule_enabled": bool(sre_rule_top),
                "sre_rule_top": sre_rule_top,
                "incident_shape_prior_enabled": bool(incident_shape_top),
                "incident_shape_prior_top": incident_shape_top,
            },
        )
