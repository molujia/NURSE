"""Evidence and preliminary graph contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.schemas import StrictFrozenModel

EvidenceKind = Literal["metric_anomaly", "log_audit", "sre_rule"]
RuleKind = Literal["metric_verification", "normal_topology"]
ProvenanceObjectKind = Literal["candidate", "evidence", "topology_edge", "rule"]


class TimeWindow(StrictFrozenModel):
    start: int
    end: int

    @model_validator(mode="after")
    def require_positive_duration(self) -> Self:
        if self.end <= self.start:
            raise ValueError("time window must end after start")
        return self


class EvidenceRecord(StrictFrozenModel):
    evidence_id: str = Field(min_length=1)
    service: str = Field(min_length=1)
    signal: str = Field(min_length=1)
    kind: EvidenceKind
    window: TimeWindow
    weight: float = Field(ge=0.0, le=100.0)
    verification_rule: str = Field(min_length=1)
    observed: Any = None
    baseline: Any = None
    first_anomalous_at: int | None = None


class CandidateNode(StrictFrozenModel):
    service: str = Field(min_length=1)
    evidence_ids: tuple[str, ...]


class TelemetryEdge(StrictFrozenModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_distinct_services(self) -> Self:
        if self.source == self.target:
            raise ValueError("telemetry edge requires distinct services")
        return self


class EvidenceRule(StrictFrozenModel):
    rule_id: str = Field(min_length=1)
    kind: RuleKind
    statement: str = Field(min_length=1)


class ProvenanceRecord(StrictFrozenModel):
    provenance_id: str = Field(min_length=1)
    object_kind: ProvenanceObjectKind
    object_id: str = Field(min_length=1)
    source_ids: tuple[str, ...]


class PreliminaryEvidenceGraph(StrictFrozenModel):
    candidates: tuple[CandidateNode, ...] = Field(min_length=1)
    evidence: tuple[EvidenceRecord, ...]
    topology_edges: tuple[TelemetryEdge, ...]
    rules: tuple[EvidenceRule, ...] = ()
    provenance: tuple[ProvenanceRecord, ...] = ()

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        services = [candidate.service for candidate in self.candidates]
        if len(services) != len(set(services)):
            raise ValueError("candidate services must be unique")

        by_id = {record.evidence_id: record for record in self.evidence}
        if len(by_id) != len(self.evidence):
            raise ValueError("evidence identifiers must be unique")

        rule_ids = {rule.rule_id for rule in self.rules}
        if len(rule_ids) != len(self.rules):
            raise ValueError("rule identifiers must be unique")
        provenance_ids = {record.provenance_id for record in self.provenance}
        if len(provenance_ids) != len(self.provenance):
            raise ValueError("provenance identifiers must be unique")

        for candidate in self.candidates:
            for evidence_id in candidate.evidence_ids:
                record = by_id.get(evidence_id)
                if record is None:
                    raise ValueError(f"candidate cites unknown evidence: {evidence_id}")
                if record.service != candidate.service:
                    raise ValueError(f"candidate does not own evidence: {evidence_id}")
        if self.rules:
            for record in self.evidence:
                if record.verification_rule not in rule_ids:
                    raise ValueError(
                        f"evidence cites unknown verification rule: {record.verification_rule}"
                    )
        for edge in self.topology_edges:
            if edge.source not in services or edge.target not in services:
                raise ValueError("topology edge endpoint is outside candidate scope")
        return self
