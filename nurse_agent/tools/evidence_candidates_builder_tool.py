"""Evidence Candidates Builder tool."""

from __future__ import annotations

from typing import Any, Dict

from nurse.evidence_extractor import extract_candidates
from nurse.raw_adapters import load_raw_incident
from nurse.raw_evidence_compiler import service_candidates_from_incident
from nurse.schemas import AlarmArtifact

from .common import dataclass_dicts


def run(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset = str(payload["dataset"])
    incident = load_raw_incident(payload["case_dir"], dataset=dataset)
    alarm = AlarmArtifact.from_dict(dict(payload["alarm"]))
    backend = str(payload.get("metric_score_backend") or "evidence_robust")
    candidates = extract_candidates(incident, alarm, metric_score_backend=backend)
    service_candidates = service_candidates_from_incident(incident, alarm)
    return {
        "dataset": dataset,
        "case_id": incident.case_id,
        "case_dir": str(payload["case_dir"]),
        "alarm": alarm.to_dict(),
        "metric_score_backend": backend,
        "service_candidates": list(service_candidates),
        "candidates": dataclass_dicts(candidates),
        "candidate_count": len(candidates),
    }
