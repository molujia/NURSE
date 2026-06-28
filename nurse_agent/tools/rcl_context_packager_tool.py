"""Verified RCL Context Packager tool."""

from __future__ import annotations

from typing import Any, Dict

from nurse.context_packager import package_verified_rcl_context
from nurse.schemas import EvidenceGraph, NursePrediction


def run(payload: Dict[str, Any]) -> Dict[str, Any]:
    graph = EvidenceGraph.from_dict(dict(payload["graph"]))
    prediction = NursePrediction.from_dict(dict(payload["prediction"]))
    context = package_verified_rcl_context(
        graph=graph,
        prediction=prediction,
        dataset=str(payload["dataset"]),
        llm_advisory=payload.get("llm_advisory") or {},
        rejected_items=payload.get("rejected_items") or [],
    )
    return {
        "dataset": str(payload["dataset"]),
        "case_id": prediction.case_id,
        "context": context.to_dict(),
        "ranked_count": len(context.ranked_candidates),
        "evidence_summary_count": len(context.evidence_summaries),
    }
