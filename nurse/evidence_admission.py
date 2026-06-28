"""Paper-facing Evidence Admission wrapper for NURSE."""

from __future__ import annotations

from typing import Iterable, List

from .raw_adapters import RawIncident
from .schemas import EvidenceCandidate, VerifiedEvidence
from .verifier import verify_candidates


def admit_evidence_candidates(
    incident: RawIncident,
    candidates: Iterable[EvidenceCandidate],
) -> List[VerifiedEvidence]:
    """Verify candidate evidence before it can enter the Evidence Graph."""

    return verify_candidates(incident, list(candidates))
