"""Evidence Graph Builder tool."""

from __future__ import annotations

from typing import Any, Dict

from nurse.evidence_graph_builder import build_verified_evidence_graph
from nurse.schemas import VerifiedEvidence


def run(payload: Dict[str, Any]) -> Dict[str, Any]:
    verified = [VerifiedEvidence.from_dict(item) for item in payload.get("verified_evidence", [])]
    topology_edges = [tuple(edge) for edge in payload.get("topology_edges", [])]
    alarm = dict(payload.get("alarm") or {})
    graph = build_verified_evidence_graph(
        case_id=str(payload["case_id"]),
        evidences=verified,
        topology_edges=topology_edges,
        alert_entity=str(alarm.get("alert_entity") or ""),
    )
    graph.metadata["dataset"] = str(payload["dataset"])
    graph.metadata["service_candidates"] = list(payload.get("service_candidates", []))
    return {
        "dataset": str(payload["dataset"]),
        "case_id": str(payload["case_id"]),
        "graph": graph.to_dict(),
        "graph_nodes": len({item.entity for item in verified if item.verified}),
        "graph_relations": len(graph.relations),
    }
