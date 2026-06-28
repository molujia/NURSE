"""Paper-facing Evidence Graph Builder wrapper for NURSE."""

from __future__ import annotations

from typing import Iterable, Tuple

from .closure import build_evidence_graph
from .schemas import EvidenceGraph, VerifiedEvidence


def build_verified_evidence_graph(
    *,
    case_id: str,
    evidences: Iterable[VerifiedEvidence],
    topology_edges: Iterable[Tuple[str, str]],
    alert_entity: str = "",
) -> EvidenceGraph:
    """Build the typed Evidence Graph from admitted evidence."""

    return build_evidence_graph(
        case_id=case_id,
        evidences=list(evidences),
        topology_edges=list(topology_edges),
        alert_entity=alert_entity,
    )
