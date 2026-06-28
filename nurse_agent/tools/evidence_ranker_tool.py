"""Evidence Ranker tool."""

from __future__ import annotations

from typing import Any, Dict

from nurse.evidence_ranker import EvidenceRanker
from nurse.schemas import EvidenceGraph


def run(payload: Dict[str, Any]) -> Dict[str, Any]:
    graph = EvidenceGraph.from_dict(dict(payload["graph"]))
    prediction = EvidenceRanker().rank(graph)
    return {
        "dataset": str(payload["dataset"]),
        "case_id": graph.case_id,
        "prediction": prediction.to_dict(),
        "top1": prediction.ranked[0] if prediction.ranked else "",
        "ranked_count": len(prediction.ranked),
    }
