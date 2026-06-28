"""Neuro-symbolic evidence closure for Nurse."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from .causal_propagation import infer_temporal_edges
from .evidence_program import EvidenceFact
from .schemas import EvidenceGraph, VerifiedEvidence


ROOT_LIKE_SIGNALS = {
    "cpu",
    "mem",
    "memory",
    "disk",
    "server_error",
    "server_error_ratio",
    "error_rate",
    "error",
    "timeout",
    "pod_cpu_usage",
    "pod_memory_working_set_bytes",
}

SYMPTOM_SIGNALS = {
    "latency",
    "latency-50",
    "latency-90",
    "rrt",
    "rrt_max",
    "client_error",
    "client_error_ratio",
    "request",
    "response",
}


def _signal_class(signal: str) -> str:
    s = signal.lower()
    if s in ROOT_LIKE_SIGNALS or any(token in s for token in ("cpu", "mem", "server_error")):
        return "source_like"
    if s in SYMPTOM_SIGNALS or any(token in s for token in ("latency", "rrt", "request")):
        return "symptom_like"
    return "generic"


def build_evidence_graph(
    *,
    case_id: str,
    evidences: Iterable[VerifiedEvidence],
    topology_edges: Iterable[Tuple[str, str]],
    alert_entity: str = "",
) -> EvidenceGraph:
    verified = [e for e in evidences if e.verified]
    relations: List[Dict] = []
    by_entity: Dict[str, List[VerifiedEvidence]] = {}
    for evidence in verified:
        if evidence.kind == "topology":
            continue
        by_entity.setdefault(evidence.entity, []).append(evidence)
        if "sop" in evidence.verification_rule.lower() or evidence.provenance.get("source") == "sop_check":
            relations.append(
                {
                    "type": "verified_by_sop",
                    "src": evidence.entity,
                    "dst": evidence.entity,
                    "evidence_id": evidence.evidence_id,
                    "signal": evidence.signal,
                }
            )
        role = str(evidence.claim.get("metric_role") or "")
        if evidence.provenance.get("candidate_source") == "llm_proposal" or role == "llm_verified_check":
            klass = "generic"
        elif role == "source_metric":
            klass = "source_like"
        elif role == "symptom_metric":
            klass = "symptom_like"
        else:
            klass = _signal_class(evidence.signal)
        if klass == "source_like":
            relations.append(
                {
                    "type": "supports_root",
                    "src": evidence.entity,
                    "dst": evidence.entity,
                    "evidence_id": evidence.evidence_id,
                    "signal": evidence.signal,
                }
            )
        elif klass == "symptom_like":
            relations.append(
                {
                    "type": "symptom_of",
                    "src": evidence.entity,
                    "dst": "unknown_root",
                    "evidence_id": evidence.evidence_id,
                    "signal": evidence.signal,
                }
            )

    topology = list(topology_edges)
    temporal_facts: List[EvidenceFact] = []
    for evidence in verified:
        if evidence.kind == "topology":
            continue
        role = str(evidence.claim.get("metric_role") or "")
        if evidence.provenance.get("candidate_source") == "llm_proposal":
            role = "llm_verified_check"
        elif not role:
            klass = _signal_class(evidence.signal)
            if klass == "source_like":
                role = "source_metric"
            elif klass == "symptom_like":
                role = "symptom_metric"
            else:
                role = "generic_metric"
        try:
            fact_weight = float(
                evidence.claim.get("evidence_program_weight")
                or evidence.claim.get("anomaly_score")
                or 1.0
            )
        except (TypeError, ValueError):
            fact_weight = 1.0
        temporal_facts.append(
            EvidenceFact(
                "anomaly",
                evidence.entity,
                evidence.signal,
                fact_weight,
                {
                    "role": role,
                    "first_seen": evidence.claim.get("first_seen"),
                    "evidence_id": evidence.evidence_id,
                },
            )
        )
    for edge in infer_temporal_edges(temporal_facts, topology):
        relations.append(
            {
                "type": "propagates_to",
                "src": edge.source,
                "dst": edge.target,
                "confidence": edge.confidence,
                "via": edge.reason,
                "lag_seconds": edge.lag_seconds,
            }
        )

    for caller, callee in topology:
        if caller == callee:
            continue
        if callee in by_entity and (caller in by_entity or caller == alert_entity):
            for evidence in by_entity[callee]:
                relations.append(
                    {
                        "type": "propagates_to",
                        "src": callee,
                        "dst": caller,
                        "via": "calls",
                        "evidence_id": evidence.evidence_id,
                    }
                )
        if caller in by_entity and callee in by_entity:
            relations.append(
                {
                    "type": "topology_neighbor",
                    "src": caller,
                    "dst": callee,
                    "via": "calls",
                }
            )

    return EvidenceGraph(
        case_id=case_id,
        evidences=verified,
        relations=relations,
        metadata={"alert_entity": alert_entity, "topology_edges": len(topology)},
    )


def classify_signal(signal: str) -> str:
    return _signal_class(signal)
