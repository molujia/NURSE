"""Evidence Admission tool."""

from __future__ import annotations

from typing import Any, Dict

from nurse.evidence_admission import admit_evidence_candidates
from nurse.raw_adapters import load_raw_incident
from nurse.schemas import EvidenceCandidate

from .common import dataclass_dicts


def run(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset = str(payload["dataset"])
    incident = load_raw_incident(payload["case_dir"], dataset=dataset)
    candidates = [EvidenceCandidate.from_dict(item) for item in payload.get("candidates", [])]
    verified = admit_evidence_candidates(incident, candidates)
    return {
        "dataset": dataset,
        "case_id": incident.case_id,
        "case_dir": str(payload["case_dir"]),
        "alarm": dict(payload["alarm"]),
        "service_candidates": list(payload.get("service_candidates", [])),
        "verified_evidence": dataclass_dicts(verified),
        "verified_count": sum(1 for item in verified if item.verified),
        "rejected_count": sum(1 for item in verified if not item.verified),
    }
