"""Incident Prior tool."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

from nurse.incident_prior import compute_incident_prior
from nurse_agent.llm_client import OpenAICompatibleClient


def _history_prompt(*, case_id: str, dataset: str, service_candidates: Iterable[str], history_index: Iterable[Mapping[str, Any]]) -> str:
    lines = [
        "You are the NURSE Incident Prior comparator.",
        "Compare the current incident only against the approved historical incidents listed below.",
        "Cite historical incident IDs exactly as provided. Do not infer or mention benchmark labels.",
        f"Current case: {case_id}",
        f"Dataset: {dataset}",
        f"Candidate entities: {', '.join(str(x) for x in service_candidates)}",
        "Approved history:",
    ]
    for item in history_index:
        lines.append(
            f"- {item.get('incident_id') or item.get('ticket_id')}: service={item.get('service') or item.get('candidate')} similarity={item.get('similarity', '')} summary={item.get('summary', '')}"
        )
    lines.append("Return one short rationale that cites the closest approved incident ID.")
    return "\n".join(lines)


def run(payload: Dict[str, Any]) -> Dict[str, Any]:
    history_index = [dict(item) for item in (payload.get("history_index") or [])]
    service_candidates = payload.get("service_candidates") or payload.get("ranked") or []
    prior = compute_incident_prior(
        case_id=str(payload["case_id"]),
        dataset=str(payload["dataset"]),
        service_candidates=service_candidates,
        history_index=history_index,
        top_k=int(payload.get("incident_prior_top_k", 5)),
    ).to_dict()
    if history_index and bool(payload.get("use_llm", False)):
        client = OpenAICompatibleClient(model=str(payload.get("model") or "gpt-5-nano"))
        rationale = client.complete_text(
            _history_prompt(
                case_id=str(payload["case_id"]),
                dataset=str(payload["dataset"]),
                service_candidates=service_candidates,
                history_index=history_index[: int(payload.get("incident_prior_top_k", 5))],
            ),
            max_output_tokens=int(payload.get("max_output_tokens", 512)),
        ).strip()
        if prior.get("active"):
            prior["method"] = "retrieval_then_llm_similarity"
        prior["rationale"] = rationale or prior.get("rationale", "")
    return {
        "dataset": str(payload["dataset"]),
        "case_id": str(payload["case_id"]),
        "incident_prior": prior,
    }
