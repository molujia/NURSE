"""Versioned graph, ranking, tie, history, and frozen-report contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import DatasetName, StrictFrozenModel
from nurse_rcl.v2.observation_schemas import TimeInterval

GRAPH_SNAPSHOT_SCHEMA_VERSION: Literal["nurse.v2.graph-snapshot/1"] = "nurse.v2.graph-snapshot/1"
SCC_PROVENANCE_SCHEMA_VERSION: Literal["nurse.v2.scc-provenance/1"] = "nurse.v2.scc-provenance/1"
SCORE_DECOMPOSITION_SCHEMA_VERSION: Literal["nurse.v2.score-decomposition/1"] = (
    "nurse.v2.score-decomposition/1"
)
TIE_GROUP_SCHEMA_VERSION: Literal["nurse.v2.tie-group/1"] = "nurse.v2.tie-group/1"
LEGACY_TOP_K_SCHEMA_VERSION: Literal["nurse.v2.top-k/1"] = "nurse.v2.top-k/1"
TOP_K_SCHEMA_VERSION: Literal["nurse.v2.top-k/2"] = "nurse.v2.top-k/2"
SCORING_RULE_SCHEMA_VERSION: Literal["nurse.v2.scoring-rule/1"] = "nurse.v2.scoring-rule/1"
RULE_SET_SCHEMA_VERSION: Literal["nurse.v2.rule-set/1"] = "nurse.v2.rule-set/1"
HISTORY_CONTEXT_SCHEMA_VERSION: Literal["nurse.v2.history-context/1"] = "nurse.v2.history-context/1"
LEGACY_FROZEN_REPORT_CONTEXT_SCHEMA_VERSION: Literal["nurse.v2.frozen-report-context/1"] = (
    "nurse.v2.frozen-report-context/1"
)
FROZEN_REPORT_CONTEXT_SCHEMA_VERSION: Literal["nurse.v2.frozen-report-context/2"] = (
    "nurse.v2.frozen-report-context/2"
)

GraphRelationType = Literal[
    "service_call",
    "received_dependency_failure",
    "caller_load_transmission",
    "shared_resource_association",
]
HistoryMode = Literal["off", "hint", "score"]
_QUANTUM = Decimal("0.000000000001")


class GraphEdge(StrictFrozenModel):
    """Original caller/callee observation or explicitly undirected association."""

    edge_id: str = Field(min_length=1)
    caller_service_key: str = Field(min_length=1)
    callee_service_key: str = Field(min_length=1)
    relation_type: GraphRelationType
    direction: Literal["caller_to_callee", "undirected_association"]
    interval: TimeInterval
    coverage_quality: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    request_count: int | None = Field(default=None, ge=0)
    error_count: int | None = Field(default=None, ge=0)
    latency_p95: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    change_state: Literal[
        "stable",
        "new_edge",
        "missing_edge",
        "increased_traffic",
        "increased_error",
        "increased_latency",
        "unknown",
    ]
    provenance_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_honest_direction(self) -> Self:
        if self.caller_service_key == self.callee_service_key:
            raise ValueError("graph edge endpoints must be distinct")
        if (
            self.relation_type == "shared_resource_association"
            and self.direction != "undirected_association"
        ):
            raise ValueError("shared resource association is undirected")
        if (
            self.relation_type != "shared_resource_association"
            and self.direction != "caller_to_callee"
        ):
            raise ValueError("service relations retain caller-to-callee direction")
        if self.provenance_ids != tuple(sorted(set(self.provenance_ids))):
            raise ValueError("graph provenance IDs must be sorted and unique")
        return self


class GraphSnapshot(StrictFrozenModel):
    """Content-addressed normal or fault graph with original edges retained."""

    schema_version: Literal["nurse.v2.graph-snapshot/1"] = GRAPH_SNAPSHOT_SCHEMA_VERSION
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    window_kind: Literal["normal", "fault"]
    observation_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    node_service_keys: tuple[str, ...] = Field(min_length=1)
    edges: tuple[GraphEdge, ...]
    graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if self.node_service_keys != tuple(sorted(set(self.node_service_keys))):
            return False
        if self.edges != tuple(sorted(self.edges, key=lambda item: item.edge_id)):
            return False
        if len({item.edge_id for item in self.edges}) != len(self.edges):
            return False
        nodes = set(self.node_service_keys)
        if any(
            edge.caller_service_key not in nodes or edge.callee_service_key not in nodes
            for edge in self.edges
        ):
            return False
        unsigned = self.model_dump(mode="json", exclude={"graph_sha256"})
        return self.graph_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_graph_snapshot(
    *,
    opaque_case_id: str,
    window_kind: Literal["normal", "fault"],
    observation_contract_sha256: str,
    nodes: Sequence[str],
    edges: Sequence[GraphEdge],
) -> GraphSnapshot:
    ordered_nodes = tuple(sorted(set(nodes)))
    ordered_edges = tuple(sorted(edges, key=lambda item: item.edge_id))
    if len(ordered_nodes) != len(tuple(nodes)):
        raise ValueError("graph nodes must be unique")
    if len({item.edge_id for item in ordered_edges}) != len(ordered_edges):
        raise ValueError("graph edge IDs must be unique")
    unsigned: dict[str, object] = {
        "schema_version": GRAPH_SNAPSHOT_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "window_kind": window_kind,
        "observation_contract_sha256": observation_contract_sha256,
        "node_service_keys": ordered_nodes,
        "edges": [item.model_dump(mode="json") for item in ordered_edges],
    }
    snapshot = GraphSnapshot.model_validate(
        {**unsigned, "graph_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    if not snapshot.verify():
        raise ValueError("graph snapshot contains an edge outside its node set")
    return snapshot


class SCCComponent(StrictFrozenModel):
    component_id: str = Field(pattern=r"^scc_[0-9a-f]{24}$")
    member_service_keys: tuple[str, ...] = Field(min_length=1)
    internal_edge_ids: tuple[str, ...]
    internal_source_service_key: None = None

    @model_validator(mode="after")
    def require_canonical_cycle(self) -> Self:
        if self.member_service_keys != tuple(sorted(set(self.member_service_keys))):
            raise ValueError("SCC members must be sorted and unique")
        if self.internal_edge_ids != tuple(sorted(set(self.internal_edge_ids))):
            raise ValueError("SCC internal edges must be sorted and unique")
        return self


class CondensedEdge(StrictFrozenModel):
    source_component_id: str = Field(pattern=r"^scc_[0-9a-f]{24}$")
    target_component_id: str = Field(pattern=r"^scc_[0-9a-f]{24}$")
    original_edge_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_distinct_components(self) -> Self:
        if self.source_component_id == self.target_component_id:
            raise ValueError("condensed edge cannot be internal to one component")
        if self.original_edge_ids != tuple(sorted(set(self.original_edge_ids))):
            raise ValueError("condensed original edge IDs must be sorted and unique")
        return self


class SCCProvenance(StrictFrozenModel):
    """Condensation result that proves every selected original edge was retained."""

    schema_version: Literal["nurse.v2.scc-provenance/1"] = SCC_PROVENANCE_SCHEMA_VERSION
    graph_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    relation_type: GraphRelationType
    components: tuple[SCCComponent, ...] = Field(min_length=1)
    condensed_edges: tuple[CondensedEdge, ...]
    retained_original_edge_ids: tuple[str, ...]
    arbitrary_edge_deletion: Literal[False] = False
    provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if self.components != tuple(sorted(self.components, key=lambda item: item.component_id)):
            return False
        if self.condensed_edges != tuple(
            sorted(
                self.condensed_edges,
                key=lambda item: (
                    item.source_component_id,
                    item.target_component_id,
                ),
            )
        ):
            return False
        component_ids = {item.component_id for item in self.components}
        if len(component_ids) != len(self.components):
            return False
        if any(
            edge.source_component_id not in component_ids
            or edge.target_component_id not in component_ids
            for edge in self.condensed_edges
        ):
            return False
        retained = tuple(
            sorted(
                {
                    edge_id
                    for component in self.components
                    for edge_id in component.internal_edge_ids
                }
                | {edge_id for edge in self.condensed_edges for edge_id in edge.original_edge_ids}
            )
        )
        if retained != self.retained_original_edge_ids:
            return False
        unsigned = self.model_dump(mode="json", exclude={"provenance_sha256"})
        return self.provenance_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_scc_provenance(
    *,
    graph_snapshot: GraphSnapshot,
    relation_type: GraphRelationType,
    components: Sequence[SCCComponent],
    condensed_edges: Sequence[CondensedEdge],
) -> SCCProvenance:
    if not graph_snapshot.verify():
        raise ValueError("graph snapshot verification failed")
    selected_edge_ids = {
        edge.edge_id for edge in graph_snapshot.edges if edge.relation_type == relation_type
    }
    retained = {edge_id for component in components for edge_id in component.internal_edge_ids} | {
        edge_id for edge in condensed_edges for edge_id in edge.original_edge_ids
    }
    if retained != selected_edge_ids:
        raise ValueError("SCC provenance must retain every selected original edge")
    ordered_components = tuple(sorted(components, key=lambda item: item.component_id))
    ordered_edges = tuple(
        sorted(
            condensed_edges,
            key=lambda item: (item.source_component_id, item.target_component_id),
        )
    )
    members = tuple(
        service for component in ordered_components for service in component.member_service_keys
    )
    if len(members) != len(set(members)) or set(members) != set(graph_snapshot.node_service_keys):
        raise ValueError("SCC components must partition all graph nodes")
    unsigned: dict[str, object] = {
        "schema_version": SCC_PROVENANCE_SCHEMA_VERSION,
        "graph_snapshot_sha256": graph_snapshot.graph_sha256,
        "relation_type": relation_type,
        "components": [item.model_dump(mode="json") for item in ordered_components],
        "condensed_edges": [item.model_dump(mode="json") for item in ordered_edges],
        "retained_original_edge_ids": tuple(sorted(retained)),
        "arbitrary_edge_deletion": False,
    }
    provenance = SCCProvenance.model_validate(
        {
            **unsigned,
            "provenance_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not provenance.verify():  # pragma: no cover - construction invariant
        raise AssertionError("SCC provenance verification failed")
    return provenance


class ScoreDecomposition(StrictFrozenModel):
    """All transparent terms required to recompute one candidate rank score."""

    schema_version: Literal["nurse.v2.score-decomposition/1"] = SCORE_DECOMPOSITION_SCHEMA_VERSION
    candidate_service_key: str = Field(min_length=1)
    event_evidence_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    propagation_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    rule_score: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    history_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    lambda_p: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    eta_r: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    eta_h: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    total_score: float = Field(allow_inf_nan=False)
    quantized_total_score: str = Field(pattern=r"^-?\d+\.\d{12}$")
    quantized_event_evidence_score: str = Field(pattern=r"^\d+\.\d{12}$")
    quantized_propagation_score: str = Field(pattern=r"^\d+\.\d{12}$")
    event_group_ids: tuple[str, ...]
    propagation_path_ids: tuple[str, ...]
    rule_ids: tuple[str, ...]
    history_neighbor_ids: tuple[str, ...]
    scoring_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    is_root_cause_probability: Literal[False] = False
    score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_inputs(self) -> Self:
        for values, name in (
            (self.event_group_ids, "event group IDs"),
            (self.propagation_path_ids, "propagation path IDs"),
            (self.rule_ids, "rule IDs"),
            (self.history_neighbor_ids, "history neighbor IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        return self

    def verify(self) -> bool:
        expected_total = (
            (1.0 - self.lambda_p) * self.event_evidence_score
            + self.lambda_p * self.propagation_score
            + self.eta_r * self.rule_score
            + self.eta_h * self.history_score
        )
        if not math.isclose(
            self.total_score,
            expected_total,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            return False
        if _quantize(self.total_score) != _quantize(expected_total):
            return False
        if self.quantized_total_score != _quantize(self.total_score):
            return False
        if self.quantized_event_evidence_score != _quantize(self.event_evidence_score):
            return False
        if self.quantized_propagation_score != _quantize(self.propagation_score):
            return False
        unsigned = self.model_dump(mode="json", exclude={"score_sha256"})
        return self.score_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_score_decomposition(
    *,
    candidate_service_key: str,
    event_evidence_score: float,
    propagation_score: float,
    rule_score: float,
    history_score: float,
    lambda_p: float,
    eta_r: float,
    eta_h: float,
    total_score: float,
    event_group_ids: Sequence[str],
    propagation_path_ids: Sequence[str],
    rule_ids: Sequence[str],
    history_neighbor_ids: Sequence[str],
    scoring_configuration_sha256: str,
) -> ScoreDecomposition:
    expected_total = (
        (1.0 - lambda_p) * event_evidence_score
        + lambda_p * propagation_score
        + eta_r * rule_score
        + eta_h * history_score
    )
    if not math.isclose(
        total_score,
        expected_total,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise ValueError("total score does not match decomposition")
    if _quantize(total_score) != _quantize(expected_total):
        raise ValueError("quantized total differs from recomputed decomposition")
    unsigned: dict[str, object] = {
        "schema_version": SCORE_DECOMPOSITION_SCHEMA_VERSION,
        "candidate_service_key": candidate_service_key,
        "event_evidence_score": event_evidence_score,
        "propagation_score": propagation_score,
        "rule_score": rule_score,
        "history_score": history_score,
        "lambda_p": lambda_p,
        "eta_r": eta_r,
        "eta_h": eta_h,
        "total_score": total_score,
        "quantized_total_score": _quantize(total_score),
        "quantized_event_evidence_score": _quantize(event_evidence_score),
        "quantized_propagation_score": _quantize(propagation_score),
        "event_group_ids": tuple(sorted(set(event_group_ids))),
        "propagation_path_ids": tuple(sorted(set(propagation_path_ids))),
        "rule_ids": tuple(sorted(set(rule_ids))),
        "history_neighbor_ids": tuple(sorted(set(history_neighbor_ids))),
        "scoring_configuration_sha256": scoring_configuration_sha256,
        "is_root_cause_probability": False,
    }
    decomposition = ScoreDecomposition.model_validate(
        {**unsigned, "score_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    if not decomposition.verify():  # pragma: no cover - construction invariant
        raise AssertionError("score decomposition verification failed")
    return decomposition


class EvidenceTieGroup(StrictFrozenModel):
    """Candidates equal on quantized Score, E, and P before hash ordering."""

    schema_version: Literal["nurse.v2.tie-group/1"] = TIE_GROUP_SCHEMA_VERSION
    tie_group_id: str = Field(pattern=r"^tie_[0-9a-f]{24}$")
    quantized_total_score: str = Field(pattern=r"^-?\d+\.\d{12}$")
    quantized_event_evidence_score: str = Field(pattern=r"^\d+\.\d{12}$")
    quantized_propagation_score: str = Field(pattern=r"^\d+\.\d{12}$")
    member_service_keys: tuple[str, ...] = Field(min_length=1)
    member_score_sha256: tuple[str, ...] = Field(min_length=1)
    tie_group_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if self.member_service_keys != tuple(sorted(set(self.member_service_keys))):
            return False
        if len(self.member_score_sha256) != len(self.member_service_keys):
            return False
        if len(set(self.member_score_sha256)) != len(self.member_score_sha256):
            return False
        if any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.member_score_sha256
        ):
            return False
        unsigned = self.model_dump(
            mode="json",
            exclude={"tie_group_id", "tie_group_sha256"},
        )
        digest = sha256_hex(canonical_json_bytes(unsigned))
        return self.tie_group_sha256 == digest and self.tie_group_id == (f"tie_{digest[:24]}")


def build_tie_group(*, scores: Sequence[ScoreDecomposition]) -> EvidenceTieGroup:
    ordered = tuple(sorted(scores, key=lambda item: item.candidate_service_key))
    if not ordered or any(not item.verify() for item in ordered):
        raise ValueError("tie group scores must verify")
    if len({item.candidate_service_key for item in ordered}) != len(ordered):
        raise ValueError("tie group candidates must be unique")
    tie_keys = {
        (
            item.quantized_total_score,
            item.quantized_event_evidence_score,
            item.quantized_propagation_score,
        )
        for item in ordered
    }
    if len(tie_keys) != 1:
        raise ValueError("tie group members do not share evidence score terms")
    total, event, propagation = next(iter(tie_keys))
    unsigned: dict[str, object] = {
        "schema_version": TIE_GROUP_SCHEMA_VERSION,
        "quantized_total_score": total,
        "quantized_event_evidence_score": event,
        "quantized_propagation_score": propagation,
        "member_service_keys": tuple(item.candidate_service_key for item in ordered),
        "member_score_sha256": tuple(item.score_sha256 for item in ordered),
    }
    digest = sha256_hex(canonical_json_bytes(unsigned))
    group = EvidenceTieGroup.model_validate(
        {
            **unsigned,
            "tie_group_id": f"tie_{digest[:24]}",
            "tie_group_sha256": digest,
        }
    )
    if not group.verify():  # pragma: no cover - construction invariant
        raise AssertionError("tie group verification failed")
    return group


class TopKEntry(StrictFrozenModel):
    rank: int = Field(ge=1)
    service_entity_key: str = Field(min_length=1)
    score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_tie_group_id: str = Field(pattern=r"^tie_[0-9a-f]{24}$")
    stable_tie_break_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tie_break_is_evidence: Literal[False] = False


class TopKArtifactV1(StrictFrozenModel):
    """Recognizable legacy Top-K contract without case or candidate scope."""

    schema_version: Literal["nurse.v2.top-k/1"] = LEGACY_TOP_K_SCHEMA_VERSION
    k: int = Field(ge=1)
    total_candidate_count: int = Field(ge=1)
    tie_break_seed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entries: tuple[TopKEntry, ...] = Field(min_length=1)
    evidence_tie_groups: tuple[EvidenceTieGroup, ...] = Field(min_length=1)
    ranking_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if tuple(item.rank for item in self.entries) != tuple(range(1, len(self.entries) + 1)):
            return False
        if len(self.entries) != min(self.k, self.total_candidate_count):
            return False
        if len({item.service_entity_key for item in self.entries}) != len(self.entries):
            return False
        groups = {item.tie_group_id: item for item in self.evidence_tie_groups}
        if len(groups) != len(self.evidence_tie_groups):
            return False
        if any(item.evidence_tie_group_id not in groups for item in self.entries):
            return False
        if any(not item.verify() for item in self.evidence_tie_groups):
            return False
        unsigned = self.model_dump(mode="json", exclude={"ranking_sha256"})
        return self.ranking_sha256 == sha256_hex(canonical_json_bytes(unsigned))


class TopKArtifact(StrictFrozenModel):
    """Case- and candidate-bound ranking projection for evaluator attribution."""

    schema_version: Literal["nurse.v2.top-k/2"] = TOP_K_SCHEMA_VERSION
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_universe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    k: int = Field(ge=1)
    total_candidate_count: int = Field(ge=1)
    tie_break_seed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entries: tuple[TopKEntry, ...] = Field(min_length=1)
    evidence_tie_groups: tuple[EvidenceTieGroup, ...] = Field(min_length=1)
    ranking_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if tuple(item.rank for item in self.entries) != tuple(range(1, len(self.entries) + 1)):
            return False
        if len(self.entries) != min(self.k, self.total_candidate_count):
            return False
        if len({item.service_entity_key for item in self.entries}) != len(self.entries):
            return False
        groups = {item.tie_group_id: item for item in self.evidence_tie_groups}
        if len(groups) != len(self.evidence_tie_groups):
            return False
        if self.evidence_tie_groups != tuple(
            sorted(self.evidence_tie_groups, key=lambda item: item.tie_group_id)
        ):
            return False
        if any(item.evidence_tie_group_id not in groups for item in self.entries):
            return False
        if any(not item.verify() for item in self.evidence_tie_groups):
            return False
        evidence_by_service: dict[str, tuple[str, str, str, str, str]] = {}
        score_digests: set[str] = set()
        for group in self.evidence_tie_groups:
            for service, score_digest in zip(
                group.member_service_keys,
                group.member_score_sha256,
                strict=True,
            ):
                if service in evidence_by_service or score_digest in score_digests:
                    return False
                score_digests.add(score_digest)
                evidence_by_service[service] = (
                    group.quantized_total_score,
                    group.quantized_event_evidence_score,
                    group.quantized_propagation_score,
                    score_digest,
                    group.tie_group_id,
                )
        if len(evidence_by_service) != self.total_candidate_count:
            return False
        expected_order = tuple(
            sorted(
                evidence_by_service,
                key=lambda service: (
                    -Decimal(evidence_by_service[service][0]),
                    -Decimal(evidence_by_service[service][1]),
                    -Decimal(evidence_by_service[service][2]),
                    _stable_tie_break_sha256(
                        tie_break_seed_sha256=self.tie_break_seed_sha256,
                        candidate_service_key=service,
                    ),
                ),
            )
        )
        if (
            tuple(item.service_entity_key for item in self.entries)
            != expected_order[: len(self.entries)]
        ):
            return False
        for item in self.entries:
            evidence = evidence_by_service.get(item.service_entity_key)
            if evidence is None:
                return False
            if (
                item.score_sha256 != evidence[3]
                or item.evidence_tie_group_id != evidence[4]
                or item.stable_tie_break_sha256
                != _stable_tie_break_sha256(
                    tie_break_seed_sha256=self.tie_break_seed_sha256,
                    candidate_service_key=item.service_entity_key,
                )
            ):
                return False
        unsigned = self.model_dump(mode="json", exclude={"ranking_sha256"})
        return self.ranking_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_top_k(
    *,
    opaque_case_id: str,
    candidate_universe_sha256: str,
    scores: Sequence[ScoreDecomposition],
    tie_groups: Sequence[EvidenceTieGroup],
    ordered_service_keys: Sequence[str],
    k: int,
    tie_break_seed_sha256: str,
) -> TopKArtifact:
    score_by_service = {item.candidate_service_key: item for item in scores}
    if len(score_by_service) != len(scores) or any(not item.verify() for item in scores):
        raise ValueError("Top-K scores must be unique and verified")
    order = tuple(ordered_service_keys)
    if len(order) != len(set(order)) or set(order) != set(score_by_service):
        raise ValueError("ordered services must exactly cover score candidates")
    expected_order = _deterministic_score_order(
        scores=scores,
        tie_break_seed_sha256=tie_break_seed_sha256,
    )
    if order != expected_order:
        raise ValueError("ordered services do not match declared deterministic ordering")
    group_by_service: dict[str, EvidenceTieGroup] = {}
    for group in tie_groups:
        if not group.verify():
            raise ValueError("Top-K tie groups must verify")
        for service in group.member_service_keys:
            if service in group_by_service:
                raise ValueError("candidate appears in more than one evidence tie group")
            group_by_service[service] = group
    if set(group_by_service) != set(score_by_service):
        raise ValueError("tie groups must exactly cover score candidates")
    entries = tuple(
        TopKEntry(
            rank=index,
            service_entity_key=service,
            score_sha256=score_by_service[service].score_sha256,
            evidence_tie_group_id=group_by_service[service].tie_group_id,
            stable_tie_break_sha256=sha256_hex(
                canonical_json_bytes(
                    {
                        "seed_sha256": tie_break_seed_sha256,
                        "service_entity_key": service,
                    }
                )
            ),
            tie_break_is_evidence=False,
        )
        for index, service in enumerate(order[:k], start=1)
    )
    ordered_groups = tuple(sorted(tie_groups, key=lambda item: item.tie_group_id))
    unsigned: dict[str, object] = {
        "schema_version": TOP_K_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "candidate_universe_sha256": candidate_universe_sha256,
        "k": k,
        "total_candidate_count": len(scores),
        "tie_break_seed_sha256": tie_break_seed_sha256,
        "entries": [item.model_dump(mode="json") for item in entries],
        "evidence_tie_groups": [item.model_dump(mode="json") for item in ordered_groups],
    }
    artifact = TopKArtifact.model_validate(
        {**unsigned, "ranking_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    if not artifact.verify():  # pragma: no cover - construction invariant
        raise AssertionError("Top-K artifact verification failed")
    return artifact


class ScoringRule(StrictFrozenModel):
    schema_version: Literal["nurse.v2.scoring-rule/1"] = SCORING_RULE_SCHEMA_VERSION
    rule_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    rule_version: str = Field(min_length=1)
    direction: Literal["promote", "demote"]
    strength: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    predicate_result_ids: tuple[str, ...] = Field(min_length=1)
    enabled: bool
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    rule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if self.predicate_result_ids != tuple(sorted(set(self.predicate_result_ids))):
            return False
        if self.provenance_ids != tuple(sorted(set(self.provenance_ids))):
            return False
        unsigned = self.model_dump(mode="json", exclude={"rule_sha256"})
        return self.rule_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_scoring_rule(
    *,
    rule_key: str,
    rule_version: str,
    direction: Literal["promote", "demote"],
    strength: float,
    predicate_result_ids: Sequence[str],
    enabled: bool,
    provenance_ids: Sequence[str],
) -> ScoringRule:
    unsigned: dict[str, object] = {
        "schema_version": SCORING_RULE_SCHEMA_VERSION,
        "rule_key": rule_key,
        "rule_version": rule_version,
        "direction": direction,
        "strength": strength,
        "predicate_result_ids": tuple(sorted(set(predicate_result_ids))),
        "enabled": enabled,
        "provenance_ids": tuple(sorted(set(provenance_ids))),
    }
    rule = ScoringRule.model_validate(
        {**unsigned, "rule_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    if not rule.verify():  # pragma: no cover - construction invariant
        raise AssertionError("scoring rule verification failed")
    return rule


class RuleSet(StrictFrozenModel):
    schema_version: Literal["nurse.v2.rule-set/1"] = RULE_SET_SCHEMA_VERSION
    default_directional_rules_enabled: Literal[False] = False
    rules: tuple[ScoringRule, ...]
    rule_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if self.rules != tuple(sorted(self.rules, key=lambda item: item.rule_key)):
            return False
        if len({item.rule_key for item in self.rules}) != len(self.rules):
            return False
        if any(not item.verify() for item in self.rules):
            return False
        unsigned = self.model_dump(mode="json", exclude={"rule_set_sha256"})
        return self.rule_set_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_rule_set(*, rules: Sequence[ScoringRule]) -> RuleSet:
    ordered = tuple(sorted(rules, key=lambda item: item.rule_key))
    unsigned: dict[str, object] = {
        "schema_version": RULE_SET_SCHEMA_VERSION,
        "default_directional_rules_enabled": False,
        "rules": [item.model_dump(mode="json") for item in ordered],
    }
    rule_set = RuleSet.model_validate(
        {**unsigned, "rule_set_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    if not rule_set.verify():
        raise ValueError("rule set must contain unique verified explicit rules")
    return rule_set


class HistoryNeighbor(StrictFrozenModel):
    neighbor_id: str = Field(pattern=r"^nbr_[0-9a-f]{24}$")
    source_opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    source_group_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_history_mode: Literal["off"] = "off"
    similarity: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    base_score_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hint_reference: str | None = Field(default=None, min_length=1)
    score_contribution: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def forbid_double_injection(self) -> Self:
        if self.hint_reference is not None and self.score_contribution != 0.0:
            raise ValueError("hint mode cannot inject score; one neighbor cannot inject both")
        return self


class HistoryContext(StrictFrozenModel):
    """Dataset-specific, read-only, nonrecursive history mode for one case."""

    schema_version: Literal["nurse.v2.history-context/1"] = HISTORY_CONTEXT_SCHEMA_VERSION
    dataset: DatasetName
    current_opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    current_group_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: HistoryMode
    library_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    read_only: Literal[True] = True
    neighbors: tuple[HistoryNeighbor, ...]
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_isolated_mode(self) -> Self:
        if self.neighbors != tuple(sorted(self.neighbors, key=lambda item: item.neighbor_id)):
            raise ValueError("history neighbors must be canonically ordered")
        if len({item.neighbor_id for item in self.neighbors}) != len(self.neighbors):
            raise ValueError("history neighbor IDs must be unique")
        if any(
            item.source_opaque_case_id == self.current_opaque_case_id
            or item.source_group_fingerprint == self.current_group_fingerprint
            for item in self.neighbors
        ):
            raise ValueError("history cannot use current case or incident group")
        if self.mode == "off":
            if self.library_sha256 is not None or self.neighbors:
                raise ValueError("history-off cannot load a library or neighbors")
        elif self.library_sha256 is None or not self.neighbors:
            raise ValueError("enabled history requires a library and neighbors")
        elif self.mode == "hint":
            if any(
                item.hint_reference is None or item.score_contribution != 0.0
                for item in self.neighbors
            ):
                raise ValueError("hint mode cannot inject score")
        elif any(
            item.hint_reference is not None or item.score_contribution <= 0.0
            for item in self.neighbors
        ):
            raise ValueError("score mode cannot expose hints")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"context_sha256"})
        return self.context_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_history_context(
    *,
    dataset: DatasetName,
    current_opaque_case_id: str,
    current_group_fingerprint: str,
    mode: HistoryMode,
    library_sha256: str | None,
    neighbors: Sequence[HistoryNeighbor],
) -> HistoryContext:
    ordered = tuple(sorted(neighbors, key=lambda item: item.neighbor_id))
    if any(
        item.source_opaque_case_id == current_opaque_case_id
        or item.source_group_fingerprint == current_group_fingerprint
        for item in ordered
    ):
        raise ValueError("history cannot use current case or incident group")
    unsigned: dict[str, object] = {
        "schema_version": HISTORY_CONTEXT_SCHEMA_VERSION,
        "dataset": dataset,
        "current_opaque_case_id": current_opaque_case_id,
        "current_group_fingerprint": current_group_fingerprint,
        "mode": mode,
        "library_sha256": library_sha256,
        "read_only": True,
        "neighbors": [item.model_dump(mode="json") for item in ordered],
    }
    context = HistoryContext.model_validate(
        {**unsigned, "context_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    if not context.verify():  # pragma: no cover - construction invariant
        raise AssertionError("history context verification failed")
    return context


class FrozenReportContextV1(StrictFrozenModel):
    """Recognizable legacy report context containing an unscoped Top-K."""

    schema_version: Literal["nurse.v2.frozen-report-context/1"] = (
        LEGACY_FROZEN_REPORT_CONTEXT_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_universe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    focus_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entity_binding_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_group_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hypothesis_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_snapshot_sha256_by_window: dict[Literal["normal", "fault"], str]
    scc_provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scores: tuple[ScoreDecomposition, ...] = Field(min_length=1)
    tie_groups: tuple[EvidenceTieGroup, ...] = Field(min_length=1)
    top_k: TopKArtifactV1
    rule_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    history_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_at_utc: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    ranking_mutable: Literal[False] = False
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if set(self.graph_snapshot_sha256_by_window) != {"normal", "fault"}:
            return False
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in self.graph_snapshot_sha256_by_window.values()
        ):
            return False
        if self.scores != tuple(
            sorted(self.scores, key=lambda item: item.candidate_service_key)
        ) or any(not item.verify() for item in self.scores):
            return False
        if self.tie_groups != tuple(
            sorted(self.tie_groups, key=lambda item: item.tie_group_id)
        ) or any(not item.verify() for item in self.tie_groups):
            return False
        if not self.top_k.verify():
            return False
        unsigned = self.model_dump(mode="json", exclude={"context_sha256"})
        return self.context_sha256 == sha256_hex(canonical_json_bytes(unsigned))


class FrozenReportContext(StrictFrozenModel):
    """Post-ranking immutable boundary available to report generation only."""

    schema_version: Literal["nurse.v2.frozen-report-context/2"] = (
        FROZEN_REPORT_CONTEXT_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_universe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    focus_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entity_binding_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_group_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hypothesis_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_snapshot_sha256_by_window: dict[Literal["normal", "fault"], str]
    scc_provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scores: tuple[ScoreDecomposition, ...] = Field(min_length=1)
    tie_groups: tuple[EvidenceTieGroup, ...] = Field(min_length=1)
    top_k: TopKArtifact
    rule_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    history_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_at_utc: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    ranking_mutable: Literal[False] = False
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if set(self.graph_snapshot_sha256_by_window) != {"normal", "fault"}:
            return False
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in self.graph_snapshot_sha256_by_window.values()
        ):
            return False
        if self.scores != tuple(
            sorted(self.scores, key=lambda item: item.candidate_service_key)
        ) or any(not item.verify() for item in self.scores):
            return False
        if self.tie_groups != tuple(
            sorted(self.tie_groups, key=lambda item: item.tie_group_id)
        ) or any(not item.verify() for item in self.tie_groups):
            return False
        if not self.top_k.verify():
            return False
        if (
            self.top_k.opaque_case_id != self.opaque_case_id
            or self.top_k.candidate_universe_sha256 != self.candidate_universe_sha256
        ):
            return False
        unsigned = self.model_dump(mode="json", exclude={"context_sha256"})
        return self.context_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def build_frozen_report_context(
    *,
    opaque_case_id: str,
    candidate_universe_sha256: str,
    focus_set_sha256: str,
    observation_set_sha256: str,
    entity_binding_set_sha256: str,
    event_group_set_sha256: str,
    hypothesis_set_sha256: str,
    graph_snapshot_sha256_by_window: Mapping[Literal["normal", "fault"], str],
    scc_provenance_sha256: str,
    scores: Sequence[ScoreDecomposition],
    tie_groups: Sequence[EvidenceTieGroup],
    top_k: TopKArtifact,
    rule_set_sha256: str,
    history_context_sha256: str,
    frozen_at_utc: str,
) -> FrozenReportContext:
    ordered_scores = tuple(sorted(scores, key=lambda item: item.candidate_service_key))
    ordered_ties = tuple(sorted(tie_groups, key=lambda item: item.tie_group_id))
    if set(graph_snapshot_sha256_by_window) != {"normal", "fault"}:
        raise ValueError("frozen report context requires normal and fault graphs")
    if (
        top_k.opaque_case_id != opaque_case_id
        or top_k.candidate_universe_sha256 != candidate_universe_sha256
    ):
        raise ValueError("Top-K belongs to another frozen report scope")
    unsigned: dict[str, object] = {
        "schema_version": FROZEN_REPORT_CONTEXT_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "candidate_universe_sha256": candidate_universe_sha256,
        "focus_set_sha256": focus_set_sha256,
        "observation_set_sha256": observation_set_sha256,
        "entity_binding_set_sha256": entity_binding_set_sha256,
        "event_group_set_sha256": event_group_set_sha256,
        "hypothesis_set_sha256": hypothesis_set_sha256,
        "graph_snapshot_sha256_by_window": dict(sorted(graph_snapshot_sha256_by_window.items())),
        "scc_provenance_sha256": scc_provenance_sha256,
        "scores": [item.model_dump(mode="json") for item in ordered_scores],
        "tie_groups": [item.model_dump(mode="json") for item in ordered_ties],
        "top_k": top_k.model_dump(mode="json"),
        "rule_set_sha256": rule_set_sha256,
        "history_context_sha256": history_context_sha256,
        "frozen_at_utc": frozen_at_utc,
        "ranking_mutable": False,
    }
    context = FrozenReportContext.model_validate(
        {**unsigned, "context_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    if not context.verify():  # pragma: no cover - construction invariant
        raise AssertionError("frozen report context verification failed")
    return context


def _quantize(value: float) -> str:
    return format(Decimal(str(value)).quantize(_QUANTUM, rounding=ROUND_HALF_EVEN), "f")


def _stable_tie_break_sha256(
    *,
    tie_break_seed_sha256: str,
    candidate_service_key: str,
) -> str:
    return sha256_hex(
        canonical_json_bytes(
            {
                "seed_sha256": tie_break_seed_sha256,
                "service_entity_key": candidate_service_key,
            }
        )
    )


def _deterministic_score_order(
    *,
    scores: Sequence[ScoreDecomposition],
    tie_break_seed_sha256: str,
) -> tuple[str, ...]:
    return tuple(
        item.candidate_service_key
        for item in sorted(
            scores,
            key=lambda item: (
                -Decimal(item.quantized_total_score),
                -Decimal(item.quantized_event_evidence_score),
                -Decimal(item.quantized_propagation_score),
                _stable_tie_break_sha256(
                    tie_break_seed_sha256=tie_break_seed_sha256,
                    candidate_service_key=item.candidate_service_key,
                ),
            ),
        )
    )
