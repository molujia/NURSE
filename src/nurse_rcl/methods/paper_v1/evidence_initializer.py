"""Create an immutable, provenance-complete preliminary evidence graph."""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex, stable_identifier
from nurse_rcl.methods.paper_v1.alarm import canonicalize_service_name
from nurse_rcl.methods.paper_v1.candidates import CandidateSelection
from nurse_rcl.methods.paper_v1.topology import NormalTopology
from nurse_rcl.models import (
    CandidateNode,
    EvidenceRecord,
    EvidenceRule,
    PreliminaryEvidenceGraph,
    ProvenanceRecord,
    TelemetryEdge,
)
from nurse_rcl.schemas import StrictFrozenModel

NORMAL_TOPOLOGY_RULE_ID = "rule-paper-v1-normal-topology"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class PreliminaryGraphArtifact(StrictFrozenModel):
    """Preliminary graph plus its canonical content hash."""

    graph: PreliminaryEvidenceGraph
    graph_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        return self.graph_hash == _model_hash(self.graph)


def initialize_preliminary_graph(
    *,
    selection: CandidateSelection,
    evidence: Iterable[EvidenceRecord],
    topology: NormalTopology,
    rank_input_fingerprint: str,
) -> PreliminaryGraphArtifact:
    """Freeze selected metric evidence, candidate topology, rules, and provenance."""

    if topology.case_id != selection.case_id:
        raise ValueError("topology case identity does not match candidate case identity")
    if not _SHA256_PATTERN.fullmatch(rank_input_fingerprint):
        raise ValueError("rank_input_fingerprint must be a lowercase SHA-256 digest")

    selected_services = tuple(selection.candidates)
    selected_set = set(selected_services)
    records, metric_rules = _selected_metric_evidence(
        evidence=evidence,
        selected_services=selected_set,
    )
    topology_rule = EvidenceRule(
        rule_id=NORMAL_TOPOLOGY_RULE_ID,
        kind="normal_topology",
        statement="only same-case spans in [t0-480s,t0) form callee-to-caller edges",
    )
    rules = tuple(sorted((*metric_rules, topology_rule), key=lambda rule: rule.rule_id))

    evidence_by_service: dict[str, list[str]] = {
        service: [] for service in selected_services
    }
    for record in records:
        evidence_by_service[record.service].append(record.evidence_id)
    candidates = tuple(
        CandidateNode(
            service=service,
            evidence_ids=tuple(sorted(evidence_by_service[service])),
        )
        for service in selected_services
    )
    edges = tuple(
        sorted(
            (
                TelemetryEdge(source=edge.callee, target=edge.caller)
                for edge in topology.edges
                if edge.callee in selected_set and edge.caller in selected_set
            ),
            key=lambda edge: (edge.source, edge.target),
        )
    )

    selection_hash = _model_hash(selection)
    topology_hash = _model_hash(topology)
    provenance = _provenance(
        candidates=candidates,
        evidence=records,
        edges=edges,
        rules=rules,
        selection_hash=selection_hash,
        topology_hash=topology_hash,
        rank_input_fingerprint=rank_input_fingerprint,
    )
    graph = PreliminaryEvidenceGraph(
        candidates=candidates,
        evidence=records,
        topology_edges=edges,
        rules=rules,
        provenance=provenance,
    )
    return PreliminaryGraphArtifact(graph=graph, graph_hash=_model_hash(graph))


def _selected_metric_evidence(
    *,
    evidence: Iterable[EvidenceRecord],
    selected_services: set[str],
) -> tuple[tuple[EvidenceRecord, ...], tuple[EvidenceRule, ...]]:
    records: list[EvidenceRecord] = []
    rules_by_id: dict[str, EvidenceRule] = {}
    for record in evidence:
        service = canonicalize_service_name(record.service)
        if record.kind != "metric_anomaly" or service not in selected_services:
            continue
        rule_payload = {
            "kind": "metric_verification",
            "statement": record.verification_rule,
        }
        rule_id = stable_identifier("rule", rule_payload)
        rules_by_id[rule_id] = EvidenceRule(
            rule_id=rule_id,
            kind="metric_verification",
            statement=record.verification_rule,
        )
        records.append(
            record.model_copy(
                update={"service": service, "verification_rule": rule_id},
                deep=True,
            )
        )
    return (
        tuple(sorted(records, key=lambda record: (record.evidence_id, record.service))),
        tuple(sorted(rules_by_id.values(), key=lambda rule: rule.rule_id)),
    )


def _provenance(
    *,
    candidates: tuple[CandidateNode, ...],
    evidence: tuple[EvidenceRecord, ...],
    edges: tuple[TelemetryEdge, ...],
    rules: tuple[EvidenceRule, ...],
    selection_hash: str,
    topology_hash: str,
    rank_input_fingerprint: str,
) -> tuple[ProvenanceRecord, ...]:
    records: list[ProvenanceRecord] = []
    for candidate in candidates:
        records.append(
            _provenance_record(
                object_kind="candidate",
                object_id=candidate.service,
                source_ids=(selection_hash, *candidate.evidence_ids),
            )
        )
    for record in evidence:
        records.append(
            _provenance_record(
                object_kind="evidence",
                object_id=record.evidence_id,
                source_ids=(rank_input_fingerprint, record.verification_rule),
            )
        )
    for edge in edges:
        edge_id = stable_identifier(
            "edge",
            {"source": edge.source, "target": edge.target},
        )
        records.append(
            _provenance_record(
                object_kind="topology_edge",
                object_id=edge_id,
                source_ids=(topology_hash, NORMAL_TOPOLOGY_RULE_ID),
            )
        )
    for rule in rules:
        source_hash = topology_hash if rule.kind == "normal_topology" else rank_input_fingerprint
        records.append(
            _provenance_record(
                object_kind="rule",
                object_id=rule.rule_id,
                source_ids=(source_hash,),
            )
        )
    return tuple(
        sorted(
            records,
            key=lambda record: (
                record.object_kind,
                record.object_id,
                record.provenance_id,
            ),
        )
    )


def _provenance_record(
    *,
    object_kind: str,
    object_id: str,
    source_ids: tuple[str, ...],
) -> ProvenanceRecord:
    payload = {
        "object_kind": object_kind,
        "object_id": object_id,
        "source_ids": list(source_ids),
    }
    return ProvenanceRecord.model_validate(
        {
            "provenance_id": stable_identifier("prov", payload),
            **payload,
        }
    )


def _model_hash(model: StrictFrozenModel) -> str:
    return sha256_hex(canonical_json_bytes(model.model_dump(mode="json")))
