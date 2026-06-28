"""Incident Prior contracts for NURSE.

The MVP keeps the prior inactive unless an approved history index is supplied.
This gives the agentic architecture a stable, auditable boundary without
claiming hidden historical knowledge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .schemas import _assert_no_forbidden


@dataclass(frozen=True)
class IncidentPriorRecord:
    case_id: str
    dataset: str
    active: bool
    method: str
    matched_history: List[Dict[str, Any]] = field(default_factory=list)
    score_contribution: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""

    def __post_init__(self) -> None:
        _assert_no_forbidden(asdict(self))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        _assert_no_forbidden(payload)
        return payload


def compute_incident_prior(
    *,
    case_id: str,
    dataset: str,
    service_candidates: Iterable[str],
    history_index: Optional[Iterable[Mapping[str, Any]]] = None,
    top_k: int = 5,
) -> IncidentPriorRecord:
    """Compute a bounded Incident Prior contribution.

    Without an approved history index, the prior is inactive and contributes no
    score. The provided history records are still contract-checked so label-like
    fields cannot slip into inference.
    """

    checked_history = [dict(item) for item in (history_index or [])]
    _assert_no_forbidden(
        {
            "case_id": case_id,
            "dataset": dataset,
            "service_candidates": list(service_candidates),
            "history_index": checked_history,
            "top_k": top_k,
        }
    )
    if not checked_history:
        return IncidentPriorRecord(
            case_id=case_id,
            dataset=dataset,
            active=False,
            method="none",
            matched_history=[],
            score_contribution={},
            rationale="No approved historical incident index was supplied.",
        )
    matched = checked_history[: max(int(top_k), 0)]
    contributions: Dict[str, float] = {}
    for item in matched:
        service = str(item.get("candidate") or item.get("service") or "").strip()
        if not service:
            continue
        try:
            similarity = float(item.get("similarity", 0.0) or 0.0)
            historical_score = float(item.get("historical_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        contributions[service] = max(contributions.get(service, 0.0), similarity * historical_score)
    return IncidentPriorRecord(
        case_id=case_id,
        dataset=dataset,
        active=bool(contributions),
        method="approved_history_similarity",
        matched_history=matched,
        score_contribution=contributions,
        rationale="Incident Prior was computed from an approved history index.",
    )
