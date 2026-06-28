"""Shared Nurse data contracts.

The inference-facing contracts intentionally reject label-like fields so raw
dataset adapters cannot accidentally leak ground truth into Nurse.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


FORBIDDEN_INFERENCE_FIELDS = {"root_cause", "answer_key", "inject_service"}


def _assert_no_forbidden(value: Any, path: str = "payload") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_INFERENCE_FIELDS:
                raise ValueError(f"forbidden inference field `{key}` at {path}")
            _assert_no_forbidden(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden(child, f"{path}[{index}]")


def _dict(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


@dataclass
class AlarmArtifact:
    case_id: str
    dataset: str
    trigger_time: int | float | str
    window: Dict[str, Any]
    alert_entity: str
    signals: List[Dict[str, Any]]
    provenance: Dict[str, Any]
    detector: str = "robust_mad"

    def __post_init__(self) -> None:
        _assert_no_forbidden(asdict(self))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AlarmArtifact":
        _assert_no_forbidden(payload)
        return cls(
            case_id=str(payload["case_id"]),
            dataset=str(payload["dataset"]),
            trigger_time=payload["trigger_time"],
            window=_dict(payload.get("window")),
            alert_entity=str(payload.get("alert_entity") or ""),
            signals=list(payload.get("signals") or []),
            provenance=_dict(payload.get("provenance")),
            detector=str(payload.get("detector") or "robust_mad"),
        )

@dataclass
class EvidenceScope:
    case_id: str
    dataset: str
    alarm_window: Dict[str, Any]
    baseline_window: Dict[str, Any]
    candidate_services: List[str]
    telemetry_sources: List[str]
    selectors: Dict[str, Any]
    topology_scope: Dict[str, Any]
    provenance: Dict[str, Any]

    def __post_init__(self) -> None:
        _assert_no_forbidden(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "dataset": self.dataset,
            "alarm_window": dict(self.alarm_window),
            "baseline_window": dict(self.baseline_window),
            "candidate_services": list(self.candidate_services),
            "telemetry_sources": list(self.telemetry_sources),
            "selectors": dict(self.selectors),
            "topology_scope": dict(self.topology_scope),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvidenceScope":
        _assert_no_forbidden(payload)
        return cls(
            case_id=str(payload["case_id"]),
            dataset=str(payload["dataset"]),
            alarm_window=_dict(payload.get("alarm_window")),
            baseline_window=_dict(payload.get("baseline_window")),
            candidate_services=[str(x) for x in payload.get("candidate_services", [])],
            telemetry_sources=[str(x) for x in payload.get("telemetry_sources", [])],
            selectors=_dict(payload.get("selectors")),
            topology_scope=_dict(payload.get("topology_scope")),
            provenance=_dict(payload.get("provenance")),
        )

@dataclass
class EvidenceCandidate:
    evidence_id: str
    case_id: str
    entity: str
    signal: str
    kind: str
    time_range: Dict[str, Any]
    claim: Dict[str, Any]
    provenance: Dict[str, Any]
    confidence: float = 1.0

    def __post_init__(self) -> None:
        _assert_no_forbidden(asdict(self))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvidenceCandidate":
        _assert_no_forbidden(payload)
        return cls(
            evidence_id=str(payload["evidence_id"]),
            case_id=str(payload["case_id"]),
            entity=str(payload["entity"]),
            signal=str(payload["signal"]),
            kind=str(payload["kind"]),
            time_range=_dict(payload.get("time_range")),
            claim=_dict(payload.get("claim")),
            provenance=_dict(payload.get("provenance")),
            confidence=float(payload.get("confidence", 1.0)),
        )


@dataclass
class VerifiedEvidence:
    evidence_id: str
    case_id: str
    entity: str
    signal: str
    kind: str
    time_range: Dict[str, Any]
    claim: Dict[str, Any]
    provenance: Dict[str, Any]
    observed: Dict[str, Any]
    verification_rule: str
    verified: bool = True
    rejection_reason: str = ""

    def __post_init__(self) -> None:
        _assert_no_forbidden(asdict(self))

    @classmethod
    def from_candidate(
        cls,
        candidate: EvidenceCandidate,
        *,
        observed: Dict[str, Any],
        verification_rule: str,
        verified: bool = True,
        rejection_reason: str = "",
    ) -> "VerifiedEvidence":
        return cls(
            evidence_id=candidate.evidence_id,
            case_id=candidate.case_id,
            entity=candidate.entity,
            signal=candidate.signal,
            kind=candidate.kind,
            time_range=dict(candidate.time_range),
            claim=dict(candidate.claim),
            provenance=dict(candidate.provenance),
            observed=dict(observed),
            verification_rule=verification_rule,
            verified=verified,
            rejection_reason=rejection_reason,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VerifiedEvidence":
        _assert_no_forbidden(payload)
        return cls(
            evidence_id=str(payload["evidence_id"]),
            case_id=str(payload["case_id"]),
            entity=str(payload["entity"]),
            signal=str(payload["signal"]),
            kind=str(payload["kind"]),
            time_range=_dict(payload.get("time_range")),
            claim=_dict(payload.get("claim")),
            provenance=_dict(payload.get("provenance")),
            observed=_dict(payload.get("observed")),
            verification_rule=str(payload.get("verification_rule") or ""),
            verified=bool(payload.get("verified", True)),
            rejection_reason=str(payload.get("rejection_reason") or ""),
        )


@dataclass
class EvidenceGraph:
    case_id: str
    evidences: List[VerifiedEvidence] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_no_forbidden(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "evidences": [e.to_dict() for e in self.evidences],
            "relations": list(self.relations),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvidenceGraph":
        _assert_no_forbidden(payload)
        return cls(
            case_id=str(payload["case_id"]),
            evidences=[VerifiedEvidence.from_dict(e) for e in payload.get("evidences", [])],
            relations=list(payload.get("relations") or []),
            metadata=_dict(payload.get("metadata")),
        )

@dataclass
class VerifiedRCLContext:
    case_id: str
    dataset: str
    ranked_candidates: List[str]
    score_breakdown: Dict[str, Dict[str, float]]
    evidence_ids: Dict[str, List[str]]
    rejected_items: List[Dict[str, Any]]
    provenance: Dict[str, Any]
    score_summary: Dict[str, float] = field(default_factory=dict)
    evidence_summaries: List[Dict[str, Any]] = field(default_factory=list)
    closure_paths: List[Dict[str, Any]] = field(default_factory=list)
    advisory: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_no_forbidden(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "dataset": self.dataset,
            "ranked_candidates": list(self.ranked_candidates),
            "score_summary": dict(self.score_summary),
            "score_breakdown": {
                service: dict(parts) for service, parts in self.score_breakdown.items()
            },
            "evidence_ids": {
                service: list(ids) for service, ids in self.evidence_ids.items()
            },
            "evidence_summaries": list(self.evidence_summaries),
            "closure_paths": list(self.closure_paths),
            "rejected_items": list(self.rejected_items),
            "advisory": dict(self.advisory),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VerifiedRCLContext":
        _assert_no_forbidden(payload)
        return cls(
            case_id=str(payload["case_id"]),
            dataset=str(payload["dataset"]),
            ranked_candidates=[str(x) for x in payload.get("ranked_candidates", [])],
            score_summary={str(k): float(v) for k, v in (payload.get("score_summary") or {}).items()},
            score_breakdown={
                str(k): {str(pk): float(pv) for pk, pv in dict(v).items()}
                for k, v in (payload.get("score_breakdown") or {}).items()
            },
            evidence_ids={
                str(k): [str(x) for x in v] for k, v in (payload.get("evidence_ids") or {}).items()
            },
            evidence_summaries=list(payload.get("evidence_summaries") or []),
            closure_paths=list(payload.get("closure_paths") or []),
            rejected_items=list(payload.get("rejected_items") or []),
            advisory=_dict(payload.get("advisory")),
            provenance=_dict(payload.get("provenance")),
        )

@dataclass
class NursePrediction:
    case_id: str
    ranked: List[str]
    scores: Dict[str, float]
    score_breakdown: Dict[str, Dict[str, float]]
    evidence_ids: Dict[str, List[str]]
    closure_paths: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_no_forbidden(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "ranked": list(self.ranked),
            "scores": dict(self.scores),
            "score_breakdown": {
                service: dict(parts) for service, parts in self.score_breakdown.items()
            },
            "evidence_ids": {service: list(ids) for service, ids in self.evidence_ids.items()},
            "closure_paths": list(self.closure_paths),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NursePrediction":
        _assert_no_forbidden(payload)
        return cls(
            case_id=str(payload["case_id"]),
            ranked=[str(x) for x in payload.get("ranked", [])],
            scores={str(k): float(v) for k, v in (payload.get("scores") or {}).items()},
            score_breakdown={
                str(k): {str(pk): float(pv) for pk, pv in dict(v).items()}
                for k, v in (payload.get("score_breakdown") or {}).items()
            },
            evidence_ids={
                str(k): [str(x) for x in v] for k, v in (payload.get("evidence_ids") or {}).items()
            },
            closure_paths=list(payload.get("closure_paths") or []),
            metadata=_dict(payload.get("metadata")),
        )

