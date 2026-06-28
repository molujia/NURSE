"""Label-free incident-shape prior for Nurse ranking.

This module keeps the no-LLM ranking path neuro-symbolic: raw metric
signals are mapped into auditable symbolic families, then deterministic
counterfactual and topology rules produce candidate priors.  It intentionally
accepts only inference-safe inputs: metric deviations, topology edges,
candidate entities, and dataset name.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

from .matching import canonical_entity


METRIC_FAMILIES: dict[str, tuple[str, ...]] = {
    "cpu": ("cpu", "processor"),
    "memory": ("mem", "memory", "working_set", "working-set"),
    "disk": ("disk", "io", "iops", "fs", "filesystem"),
    "latency": ("rrt", "latency", "duration", "timeout", "delay"),
    "error": ("error", "fail", "failure", "exception", "5xx", "4xx"),
    "network": ("network", "transmit", "receive", "packet", "bytes", "loss"),
    "traffic": ("request", "response", "qps", "throughput", "traffic"),
}

FAMILY_WEIGHTS: dict[str, float] = {
    "cpu": 4.0,
    "memory": 3.5,
    "disk": 3.2,
    "latency": 2.6,
    "error": 2.4,
    "network": 1.7,
    "traffic": 0.35,
    "generic": 0.8,
}

SOURCE_FAMILIES = {"cpu", "memory", "disk"}
MIXED_FAMILIES = {"latency", "error", "network"}
SYMPTOM_FAMILIES = {"traffic"}


LabelFreeMetricScores = Mapping[str, Mapping[str, float] | float | int]


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _normalise(values: Mapping[str, float]) -> dict[str, float]:
    finite = {key: _safe_float(value) for key, value in values.items()}
    if not finite:
        return {}
    lo = min(finite.values())
    hi = max(finite.values())
    if hi - lo <= 1e-12:
        return {key: (1.0 if value > 0 else 0.0) for key, value in finite.items()}
    return {key: (value - lo) / (hi - lo) for key, value in finite.items()}


def classify_metric_family(metric_name: str) -> str:
    """Map a raw metric name to an auditable symbolic signal family."""

    text = str(metric_name or "").lower().replace("-", "_")
    for family, tokens in METRIC_FAMILIES.items():
        if any(token in text for token in tokens):
            return family
    return "generic"


def _iter_metric_scores(metric_scores: LabelFreeMetricScores) -> Iterable[tuple[str, str, float]]:
    for raw_entity, raw_metrics in (metric_scores or {}).items():
        entity = canonical_entity(str(raw_entity))
        if not entity:
            continue
        if isinstance(raw_metrics, Mapping):
            for raw_metric, raw_score in raw_metrics.items():
                score = max(_safe_float(raw_score), 0.0)
                if score > 0:
                    yield entity, str(raw_metric), score
        else:
            score = max(_safe_float(raw_metrics), 0.0)
            if score > 0:
                yield entity, "value", score


def _canonical_candidates(service_candidates: Iterable[str] | None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in service_candidates or []:
        entity = canonical_entity(str(raw))
        if entity and entity not in seen:
            seen.add(entity)
            ordered.append(entity)
    return ordered


def _canonical_edges(topology_edges: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_src, raw_dst in topology_edges or []:
        src = canonical_entity(str(raw_src))
        dst = canonical_entity(str(raw_dst))
        if not src or not dst or src == dst:
            continue
        edge = (src, dst)
        if edge not in seen:
            seen.add(edge)
            edges.append(edge)
    return edges


def _entity_kind(dataset: str, entity: str) -> str:
    text = entity.lower()
    if dataset == "aiops25" and (text.startswith("cn-") or "node" in text or text.startswith("host")):
        return "node"
    if dataset == "aiops25" and ("->" in text or ":" in text):
        return "network_endpoint"
    return "service"


def build_incident_shape_features(
    metric_scores: LabelFreeMetricScores,
    topology_edges: Iterable[tuple[str, str]],
    service_candidates: Iterable[str] | None,
    dataset: str,
) -> dict[str, Any]:
    """Build symbolic incident-shape features without labels or fault metadata."""

    dataset_name = str(dataset or "")
    candidates = _canonical_candidates(service_candidates)
    candidate_set = set(candidates)
    entities: dict[str, dict[str, Any]] = {}

    for candidate in candidates:
        entities[candidate] = {
            "families": {},
            "metrics": {},
            "kind": _entity_kind(dataset_name, candidate),
            "candidate": True,
        }

    for entity, metric, score in _iter_metric_scores(metric_scores):
        if candidate_set and entity not in candidate_set:
            continue
        family = classify_metric_family(metric)
        bucket = entities.setdefault(
            entity,
            {
                "families": {},
                "metrics": {},
                "kind": _entity_kind(dataset_name, entity),
                "candidate": entity in candidate_set if candidate_set else True,
            },
        )
        bucket["families"][family] = float(bucket["families"].get(family, 0.0) + score)
        bucket["metrics"][metric] = max(float(bucket["metrics"].get(metric, 0.0)), float(score))

    edges = _canonical_edges(topology_edges)
    return {
        "dataset": dataset_name,
        "candidates": candidates,
        "topology_edges": edges,
        "entities": entities,
    }


def _family_vector(entity_features: Mapping[str, Any]) -> list[float]:
    families = dict(entity_features.get("families") or {})
    ordered = ["cpu", "memory", "disk", "latency", "error", "network", "traffic", "generic"]
    return [
        math.log1p(max(_safe_float(families.get(family)), 0.0)) * FAMILY_WEIGHTS[family]
        for family in ordered
    ]


def compute_counterfactual_shape_drop(features: Mapping[str, Any]) -> dict[str, float]:
    """Score each candidate by the anomaly magnitude removed by ablating it."""

    entities = dict(features.get("entities") or {})
    vectors = {entity: _family_vector(data) for entity, data in entities.items()}
    total_sq = sum(value * value for vector in vectors.values() for value in vector)
    total_norm = math.sqrt(total_sq)
    drops: dict[str, float] = {}
    for entity, vector in vectors.items():
        without_sq = max(total_sq - sum(value * value for value in vector), 0.0)
        drops[entity] = max(total_norm - math.sqrt(without_sq), 0.0)
    return drops


def _source_strength(families: Mapping[str, float]) -> float:
    return sum(
        math.log1p(max(_safe_float(families.get(family)), 0.0)) * FAMILY_WEIGHTS[family]
        for family in SOURCE_FAMILIES
    )


def _mixed_strength(families: Mapping[str, float], family_counts: Mapping[str, int]) -> float:
    score = 0.0
    for family in MIXED_FAMILIES:
        count = max(int(family_counts.get(family, 0)), 1)
        raw = max(_safe_float(families.get(family)), 0.0)
        score += math.log1p(raw) * FAMILY_WEIGHTS[family] / math.sqrt(count)
    return score


def _symptom_strength(families: Mapping[str, float]) -> float:
    return sum(
        math.log1p(max(_safe_float(families.get(family)), 0.0)) * FAMILY_WEIGHTS[family]
        for family in SYMPTOM_FAMILIES
    )


def _entity_type_bonus(dataset: str, entity_data: Mapping[str, Any]) -> float:
    kind = str(entity_data.get("kind") or "service")
    if dataset == "aiops25" and kind in {"node", "network_endpoint"}:
        return 0.35
    return 0.0


def _topology_locality(
    entities: Mapping[str, Mapping[str, Any]],
    topology_edges: Iterable[tuple[str, str]],
    family_counts: Mapping[str, int],
) -> dict[str, float]:
    locality: dict[str, float] = defaultdict(float)
    for src, dst in topology_edges:
        src_data = entities.get(src)
        dst_data = entities.get(dst)
        if not src_data or not dst_data:
            continue
        src_families = dict(src_data.get("families") or {})
        dst_families = dict(dst_data.get("families") or {})
        src_source = _source_strength(src_families)
        dst_symptom = _mixed_strength(dst_families, family_counts) + _symptom_strength(dst_families)
        if src_source > 0 and dst_symptom > 0:
            locality[src] += min(src_source, dst_symptom) * 0.5
    return dict(locality)


def score_incident_shape_prior(features: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Return auditable candidate prior parts derived from incident-shape evidence."""

    entities = dict(features.get("entities") or {})
    dataset = str(features.get("dataset") or "")
    family_counts: dict[str, int] = defaultdict(int)
    for data in entities.values():
        for family, value in dict(data.get("families") or {}).items():
            if _safe_float(value) > 0:
                family_counts[family] += 1

    drops = compute_counterfactual_shape_drop(features)
    drop_norm = _normalise(drops)
    topology_parts = _topology_locality(
        entities,
        features.get("topology_edges") or [],
        family_counts,
    )

    scores: dict[str, dict[str, float]] = {}
    for entity, data in entities.items():
        families = dict(data.get("families") or {})
        source = _source_strength(families)
        mixed = _mixed_strength(families, family_counts)
        symptom = _symptom_strength(families)
        symptom_penalty = -0.20 * symptom if source <= 0 and mixed <= 0 else 0.0
        counterfactual = drop_norm.get(entity, 0.0)
        topology = topology_parts.get(entity, 0.0)
        entity_type = _entity_type_bonus(dataset, data)
        total = source + mixed + symptom_penalty + counterfactual + topology + entity_type
        scores[entity] = {
            "source_signal": float(source),
            "mixed_signal": float(mixed),
            "symptom_penalty": float(symptom_penalty),
            "counterfactual": float(counterfactual),
            "topology_locality": float(topology),
            "entity_type_prior": float(entity_type),
            "total": float(total),
        }
    return scores
