"""Package verified Nurse core output for the LLM User Manager.

The packager is intentionally non-mutating: it copies the Evidence Core ranking
into a context object and records LLM suggestions as advisory metadata only.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .schemas import EvidenceGraph, NursePrediction, VerifiedEvidence, VerifiedRCLContext


def _evidence_summary(evidence: VerifiedEvidence) -> Dict[str, Any]:
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


def package_verified_rcl_context(
    *,
    graph: EvidenceGraph,
    prediction: NursePrediction,
    dataset: str,
    llm_advisory: Dict[str, Any] | None = None,
    rejected_items: List[Dict[str, Any]] | None = None,
) -> VerifiedRCLContext:
    """Create a verified, label-free context for user-facing LLM explanation."""

    verified_summaries = [
        _evidence_summary(evidence)
        for evidence in graph.evidences
        if evidence.verified and evidence.evidence_id
    ]
    return VerifiedRCLContext(
        case_id=prediction.case_id,
        dataset=str(dataset),
        ranked_candidates=list(prediction.ranked),
        score_summary={service: float(score) for service, score in prediction.scores.items()},
        score_breakdown={
            service: {part: float(value) for part, value in parts.items()}
            for service, parts in prediction.score_breakdown.items()
        },
        evidence_ids={
            service: list(ids) for service, ids in prediction.evidence_ids.items()
        },
        evidence_summaries=verified_summaries,
        closure_paths=list(prediction.closure_paths),
        rejected_items=list(rejected_items or []),
        advisory=dict(llm_advisory or {}),
        provenance={
            "rank_path": "evidence_core_v2",
            "ranker": prediction.metadata.get("ranker", ""),
            "llm_rank_mutation": False,
        },
    )
